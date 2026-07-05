"""
argos.evaluation
================

Evaluate a trained argos model against pre-extracted scenes.

:func:`evaluate_model` runs a model over the scenes of an :class:`ArgosDataset`
and accumulates precipitation-quantification metrics from
:mod:`satrain.metrics`: MAE, MSE, the linear correlation coefficient and the
(relative) bias.

In addition to the overall scores, the accuracy is broken down jointly by the
microwave input, using two per-pixel diagnostic fields derived for every scene
(see :func:`_mw_fields`):

* ``step``: the temporal offset, in time steps, between the most recent
  microwave observation covering a pixel and the target time (``0`` = concurrent
  with the target; larger = older). For single-time-step scenes this is always
  ``0`` where microwave data is present.
* ``sensor``: the sensor that provided that most recent observation.

Every covered pixel has exactly one ``(step, sensor)`` pair, so the metrics are
conditioned on both simultaneously and returned on a 2-D ``(step, sensor)`` grid
(``*_by_input``) alongside the overall scores.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr

from satrain.metrics import MAE, MSE, CorrelationCoef, Bias

from argos.data.dataset import ArgosDataset

__all__ = ["evaluate_model"]


# The quantification metrics tracked for every conditioning group. Each entry
# maps an output variable name to a factory creating a fresh metric instance.
_METRIC_FACTORIES: Dict[str, Callable[[], object]] = {
    "mae": MAE,
    "mse": MSE,
    "correlation_coef": CorrelationCoef,
    "bias": lambda: Bias(relative=True),
}


def _new_metric_set() -> Dict[str, object]:
    """Create a fresh set of metrics (one instance per tracked metric)."""
    return {name: factory() for name, factory in _METRIC_FACTORIES.items()}


def _point_estimate(prediction: object) -> torch.Tensor:
    """
    Reduce a model prediction to a point estimate on the CPU.

    Quantile predictions (:class:`pytorch_retrieve.tensors.QuantileTensor`, or
    any object exposing ``expected_value``) are reduced to their posterior mean;
    deterministic tensors are used as they are.
    """
    if hasattr(prediction, "expected_value"):
        prediction = prediction.expected_value()
    return torch.as_tensor(prediction).detach().to("cpu", torch.float32)


def _mw_fields(
    mw: np.ndarray,
    mw_sensor: np.ndarray,
    temporal: bool,
    frames: int,
) -> tuple:
    """
    Two per-pixel diagnostic fields describing a scene's microwave input.

    Args:
        mw: The microwave input of a single scene, either ``(channel, y, x)``
            (single time step) or ``(channel, step, y, x)`` (temporal).
        mw_sensor: The sensor index of the microwave observation, either a scalar
            (single time step) or one value per step ``(step,)``. ``-1`` marks a
            step/scene without a microwave observation.
        temporal: Whether ``mw`` carries a time-step dimension.
        frames: The number of time steps (``1`` for single-time-step scenes).

    Returns:
        A tuple ``(step_field, sensor_field)`` of ``(y, x)`` integer arrays. In
        ``step_field`` each pixel holds the offset, in time steps, between the
        most recent microwave observation covering it and the target time (``0``
        = concurrent). In ``sensor_field`` each pixel holds the sensor index of
        that observation. Both are ``-1`` where no microwave data is available.
    """
    if temporal:
        # A pixel is covered at a step if any microwave channel is finite there.
        covered = np.isfinite(mw).any(axis=0)  # (step, y, x)
        height, width = covered.shape[-2:]
        step_field = np.full((height, width), -1, dtype=np.int64)
        sensor_field = np.full((height, width), -1, dtype=np.int64)
        # Iterate steps oldest-to-newest so the most recent observation wins.
        for step in range(frames):
            here = covered[step]
            step_field[here] = (frames - 1) - step
            sensor_field[here] = mw_sensor[step]
        return step_field, sensor_field

    covered = np.isfinite(mw).any(axis=0)  # (y, x)
    step_field = np.where(covered, 0, -1).astype(np.int64)
    sensor_field = np.where(covered, np.int64(mw_sensor), -1).astype(np.int64)
    return step_field, sensor_field


def _merge_metric_set(metric_set: Dict[str, object]) -> xr.Dataset:
    """Compute all metrics in a set and merge them into a single dataset."""
    return xr.merge([metric.compute() for metric in metric_set.values()])


def _combine(
    overall: Dict[str, object],
    by_input: Dict[Tuple[int, int], Dict[str, object]],
    sensors: List[str],
) -> xr.Dataset:
    """Combine the accumulated metrics into a single result dataset."""
    result = _merge_metric_set(overall)
    if not by_input:
        return result

    steps = sorted({step for (step, _) in by_input})
    sensor_ids = sorted({sensor for (_, sensor) in by_input})
    step_pos = {step: i for i, step in enumerate(steps)}
    sensor_pos = {sensor: j for j, sensor in enumerate(sensor_ids)}

    # Lay the (sparse) (step, sensor) pairs onto the full 2-D grid, filling
    # combinations that never occur with NaN.
    var_names = list(_METRIC_FACTORIES)
    grids = {
        name: np.full((len(steps), len(sensor_ids)), np.nan) for name in var_names
    }
    attrs: Dict[str, dict] = {}
    for (step, sensor), metric_set in by_input.items():
        metrics = _merge_metric_set(metric_set)
        i, j = step_pos[step], sensor_pos[sensor]
        for name in var_names:
            grids[name][i, j] = float(metrics[name].values)
            attrs.setdefault(name, dict(metrics[name].attrs))

    names = [sensors[k] if 0 <= k < len(sensors) else str(k) for k in sensor_ids]
    joint = xr.Dataset(
        {name: (("step", "sensor"), grids[name]) for name in var_names},
        coords={
            "step": ("step", np.array(steps, np.int64)),
            "sensor": ("sensor", names),
        },
    )
    for name in var_names:
        joint[name].attrs = attrs.get(name, {})
    joint = joint.rename({name: f"{name}_by_input" for name in var_names})
    return xr.merge([result, joint])


def evaluate_model(
    model: torch.nn.Module,
    dataset: ArgosDataset,
    batch_size: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    target_key: str = "surface_precip",
) -> xr.Dataset:
    """
    Evaluate a trained model against the scenes of an :class:`ArgosDataset`.

    The scenes are read directly from the dataset's store in order (no shuffling,
    no augmentation, and no empty-target fallback), so every scene is evaluated
    exactly once against its true microwave metadata. For each scene the model's
    point prediction is compared against the ``surface_precip`` target, updating
    the overall metrics as well as the metrics conditioned jointly on the
    microwave input time step and sensor (see the module docstring and
    :func:`_mw_fields`).

    Args:
        model: The trained model. Must accept the ``{"geo", "mw"}`` input dict
            (as produced for training) and return a mapping with ``target_key``.
        dataset: The dataset of pre-extracted scenes to evaluate against.
        batch_size: Number of scenes to run through the model at once.
        device: Device to run the model on. Defaults to the model's own device.
        target_key: Key of the target/prediction variable.

    Returns:
        An :class:`xarray.Dataset` with the overall metrics (``mae``, ``mse``,
        ``correlation_coef``, ``bias``) and their joint breakdown
        (``*_by_input``) on a 2-D grid with a ``step`` dimension and a ``sensor``
        dimension of sensor names.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model = model.to(device).eval()

    store = dataset.store
    sensors = dataset.sensors
    n_scenes = store[target_key].shape[0]
    temporal = store["mw"].ndim == 5
    frames = int(store["mw"].shape[2]) if temporal else 1

    overall = _new_metric_set()
    by_input: Dict[Tuple[int, int], Dict[str, object]] = {}
    # Track every set so the shared memory can be released at the end.
    all_sets: List[Dict[str, object]] = [overall]

    def _get_set(key: Tuple[int, int]) -> Dict[str, object]:
        metric_set = by_input.get(key)
        if metric_set is None:
            metric_set = by_input[key] = _new_metric_set()
            all_sets.append(metric_set)
        return metric_set

    try:
        for start in range(0, n_scenes, batch_size):
            stop = min(start + batch_size, n_scenes)
            geo = np.asarray(store["geo"][start:stop])
            mw = np.asarray(store["mw"][start:stop])
            target = np.asarray(store[target_key][start:stop])
            mw_sensor = np.asarray(store["mw_sensor"][start:stop])

            inpt = {
                "geo": torch.from_numpy(geo).to(device),
                "mw": torch.from_numpy(mw).to(device),
            }
            with torch.no_grad():
                prediction = model(inpt)[target_key]
            pred = _point_estimate(prediction).numpy()

            for b in range(stop - start):
                p = pred[b]
                t = target[b]
                for metric in overall.values():
                    metric.update(p, t)

                step_field, sensor_field = _mw_fields(
                    mw[b], mw_sensor[b], temporal, frames
                )
                covered = step_field >= 0
                if not covered.any():
                    continue
                # Each covered pixel maps to one (step, sensor) pair; group them.
                pairs = np.unique(
                    np.stack([step_field[covered], sensor_field[covered]], axis=1),
                    axis=0,
                )
                for step_value, sensor_value in pairs:
                    mask = (step_field == step_value) & (sensor_field == sensor_value)
                    metric_set = _get_set((int(step_value), int(sensor_value)))
                    for metric in metric_set.values():
                        metric.update(p[mask], t[mask])

        result = _combine(overall, by_input, sensors)
    finally:
        for metric_set in all_sets:
            for metric in metric_set.values():
                metric.cleanup()

    return result
