"""
argos.evaluation
================

Evaluate a trained argos model against pre-extracted scenes.

:func:`evaluate_model` runs a model over the scenes of an :class:`ArgosDataset`
and accumulates precipitation-quantification metrics from
:mod:`pytorch_retrieve.metrics`: MAE, MSE, the linear correlation coefficient and
the relative bias.

These metrics provide a built-in ``conditional`` option that bins each pixel's
contribution by one or more per-pixel coordinate fields (via
``torch.histogramdd``), so the conditioned accuracy is obtained by simply passing
the coordinate fields to ``update`` -- no manual grouping required. Two per-pixel
fields are derived for every scene (see :func:`_mw_fields`):

* ``step``: the temporal offset, in time steps, between the most recent
  microwave observation covering a pixel and the target time (``0`` = concurrent
  with the target; larger = older). For single-time-step scenes this is always
  ``0`` where microwave data is present.
* ``sensor``: the sensor that provided that most recent observation.

Metrics are returned overall, conditioned on ``step`` alone (``*_by_step``), and
conditioned jointly on ``(step, sensor)`` (``*_by_input``). Pixels without a
microwave observation carry ``-1`` in both fields and fall outside the histogram
bins, so they are naturally excluded from the conditioned metrics.
"""
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from pytorch_retrieve.metrics import MAE, MSE, CorrelationCoef, RelativeBias
from pytorch_retrieve.tensors import MaskedTensor

from argos.data.dataset import ArgosDataset

__all__ = ["evaluate_model"]


# The quantification metrics tracked, mapping an output variable name to its
# metric class and to the attributes attached to the resulting dataset variable.
_METRIC_FACTORIES = {
    "mae": MAE,
    "mse": MSE,
    "correlation_coef": CorrelationCoef,
    "bias": RelativeBias,
}
_METRIC_ATTRS = {
    "mae": {"full_name": "MAE", "unit": "mm h^{-1}"},
    "mse": {"full_name": "MSE", "unit": "(mm h^{-1})^2"},
    "correlation_coef": {"full_name": "Correlation coeff.", "unit": ""},
    "bias": {"full_name": "Bias", "unit": "%"},
}


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


def _make_metrics(conditional: Optional[Dict[str, int]]) -> Dict[str, object]:
    """Create one metric instance per tracked metric with the given binning."""
    if conditional is None:
        return {name: cls() for name, cls in _METRIC_FACTORIES.items()}
    return {name: cls(conditional=conditional) for name, cls in _METRIC_FACTORIES.items()}


def evaluate_model(
    model: torch.nn.Module,
    dataset: ArgosDataset,
    batch_size: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    target_key: str = "surface_precip",
    geo_only: bool = False,
    mw_sensors: Optional[List[int]] = None,
) -> xr.Dataset:
    """
    Evaluate a trained model against the scenes of an :class:`ArgosDataset`.

    The scenes are read directly from the dataset's store in order (no shuffling,
    no augmentation, and no empty-target fallback), so every scene is evaluated
    exactly once against its true microwave metadata. For each batch the model's
    point prediction is compared against the ``surface_precip`` target, updating
    the overall metrics as well as the metrics conditioned on the microwave input
    time step alone and jointly on time step and sensor. The conditioning is
    handled by the metrics themselves through their ``conditional`` option (see
    the module docstring and :func:`_mw_fields`).

    Args:
        model: The trained model. Must accept the ``{"geo", "mw"}`` input dict
            (as produced for training) and return a mapping with ``target_key``.
        dataset: The dataset of pre-extracted scenes to evaluate against.
        batch_size: Number of scenes to run through the model at once.
        device: Device to run the model on. Defaults to the model's own device.
        target_key: Key of the target/prediction variable.
        geo_only: If ``True``, set all microwave observations to ``NaN`` so the
            model is evaluated on the geostationary input alone. The conditioned
            breakdowns are then empty (no microwave observations are available).
        mw_sensors: Optional list of sensor indices to keep. Microwave
            observations from any other sensor are set to ``NaN``. Mutually
            exclusive with ``geo_only``.

    Returns:
        An :class:`xarray.Dataset` with the overall metrics (``mae``, ``mse``,
        ``correlation_coef``, ``bias``), their step-only breakdown (``*_by_step``,
        along a ``step`` dimension) and their joint breakdown (``*_by_input``) on
        a 2-D grid with a ``step`` dimension and a ``sensor`` dimension of sensor
        names.
    """
    if geo_only and mw_sensors is not None:
        raise ValueError("Pass at most one of 'geo_only' and 'mw_sensors'.")
    allowed_sensors = (
        None
        if mw_sensors is None
        else np.array(sorted({int(s) for s in mw_sensors}), dtype=np.int64)
    )

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model = model.to(device).eval()

    store = dataset.store
    sensors = dataset.sensors
    n_sensors = len(sensors)
    n_scenes = store[target_key].shape[0]
    temporal = store["mw"].ndim == 5
    frames = int(store["mw"].shape[2]) if temporal else 1

    # ``step`` has ``frames`` possible values (0 .. frames - 1); ``sensor`` has
    # ``n_sensors``. The metrics bin the per-pixel coordinates internally.
    overall = _make_metrics(None)
    by_step = _make_metrics({"step": frames})
    by_input = (
        _make_metrics({"step": frames, "sensor": n_sensors}) if n_sensors else None
    )

    for start in tqdm(
        range(0, n_scenes, batch_size),
        desc="Evaluating scenes",
        unit="batch",
        total=(n_scenes + batch_size - 1) // batch_size,
    ):
        stop = min(start + batch_size, n_scenes)
        geo = np.asarray(store["geo"][start:stop])
        mw = np.array(store["mw"][start:stop])  # writable: nulled in place below
        mw_orig = mw.copy()
        target_np = np.asarray(store[target_key][start:stop])
        mw_sensor = np.asarray(store["mw_sensor"][start:stop])

        # Optionally drop microwave observations before they reach the model and
        # the conditioning fields (both derived from ``mw`` below).
        if geo_only:
            mw[:] = np.nan
        elif allowed_sensors is not None:
            drop = ~np.isin(mw_sensor, allowed_sensors)  # (b,) or (b, step)
            if temporal:
                mw[np.broadcast_to(drop[:, None, :, None, None], mw.shape)] = np.nan
            else:
                mw[drop] = np.nan

        inpt = {
            "geo": torch.from_numpy(geo).to(device),
            "mw": torch.from_numpy(mw).to(device),
        }
        with torch.no_grad():
            prediction = model(inpt)[target_key]
        pred = _point_estimate(prediction)

        target = torch.from_numpy(target_np).to(torch.float32)
        invalid = ~torch.isfinite(target) | ~torch.isfinite(pred)
        masked_target = MaskedTensor(target, mask=invalid)

        # Per-pixel (step, sensor) coordinate fields for the whole batch.
        step_coords = np.empty(target_np.shape, dtype=np.float32)
        sensor_coords = np.empty(target_np.shape, dtype=np.float32)
        for b in range(stop - start):
            step_field, sensor_field = _mw_fields(
                mw_orig[b], mw_sensor[b], temporal, frames
            )
            step_coords[b] = step_field
            sensor_coords[b] = sensor_field
        step_coords_t = torch.from_numpy(step_coords)
        sensor_coords_t = torch.from_numpy(sensor_coords)

        for name in _METRIC_FACTORIES:
            overall[name].update(pred, masked_target)
            by_step[name].update(
                pred, masked_target, conditional={"step": step_coords_t}
            )
            if by_input is not None:
                by_input[name].update(
                    pred,
                    masked_target,
                    conditional={"step": step_coords_t, "sensor": sensor_coords_t},
                )

    return _assemble(overall, by_step, by_input, sensors)


def _assemble(
    overall: Dict[str, object],
    by_step: Dict[str, object],
    by_input: Optional[Dict[str, object]],
    sensors: List[str],
) -> xr.Dataset:
    """Collect the computed metrics into a single result dataset."""
    names = list(_METRIC_FACTORIES)
    data_vars: Dict[str, object] = {}
    coords: Dict[str, object] = {}

    for name in names:
        data_vars[name] = ((), float(overall[name].compute()))

    # Restrict the breakdowns to the steps/sensors that actually occur (non-zero
    # histogram counts), using the counts tracked by the metrics.
    # ``compute()`` squeezes size-1 dimensions, so reshape it back to the
    # metric's declared bin shape before indexing.
    step_shape = tuple(by_step["mae"].shape)
    step_counts = np.asarray(by_step["mae"].counts).reshape(step_shape)
    observed_steps = np.flatnonzero(step_counts > 0)
    if observed_steps.size:
        coords["step"] = ("step", observed_steps.astype(np.int64))
        for name in names:
            values = np.asarray(by_step[name].compute()).reshape(step_shape)
            data_vars[f"{name}_by_step"] = (("step",), values[observed_steps])

    if by_input is not None:
        joint_shape = tuple(by_input["mae"].shape)
        joint_counts = np.asarray(by_input["mae"].counts).reshape(joint_shape)
        observed_sensors = np.flatnonzero(joint_counts.sum(axis=0) > 0)
        if observed_steps.size and observed_sensors.size:
            names_sel = [
                sensors[k] if 0 <= k < len(sensors) else str(k)
                for k in observed_sensors
            ]
            coords["sensor"] = ("sensor", names_sel)
            grid_index = np.ix_(observed_steps, observed_sensors)
            for name in names:
                grid = np.asarray(by_input[name].compute()).reshape(joint_shape)
                data_vars[f"{name}_by_input"] = (("step", "sensor"), grid[grid_index])

    result = xr.Dataset(data_vars, coords=coords)
    for name in names:
        for suffix in ("", "_by_step", "_by_input"):
            var = f"{name}{suffix}"
            if var in result:
                result[var].attrs.update(_METRIC_ATTRS[name])
    return result
