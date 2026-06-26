"""
argos.data.dataset
==================

A PyTorch dataset for loading co-located geostationary observations and GPM
reference data extracted into per-time-step ``.zarr`` stores.

The training data is organized into one sub-directory per sensor, for example::

    training_data/
        goes16/      goes16_<timestamp>.zarr
        goes18/      goes18_<timestamp>.zarr
        seviri/      seviri_<timestamp>.zarr
        seviri_io/   seviri_io_<timestamp>.zarr
        gprof_gmi/   gprof_gmi_<timestamp>.zarr

Each input (geostationary) store holds an ``obs`` array of shape
``(channel, 7200, 14400)`` on a global 0.025-degree grid, while the reference
(``gprof_gmi``) store holds a ``surface_precip`` array of shape ``(3600, 7200)``
on a global 0.05-degree grid -- i.e. the geostationary observations have twice
the resolution of the reference data.

Both kinds of store additionally contain:

* an ``availability`` field of shape ``(90, 180)`` -- the data coverage on a
  grid coarsened by a factor of 80 (relative to the 0.025-degree grid), so each
  cell spans 2 degrees and the cells of the input and reference grids are
  spatially aligned.
* a ``time`` field: a scalar acquisition time for the (instantaneous)
  geostationary observations and a ``(90, 180)`` field of per-cell scan times
  for the GPM swath reference data.

The availability and time fields of every file are read up front so that
suitable, temporally-matched samples can be enumerated without touching the
(large) observation arrays.
"""
from functools import cached_property
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import xarray as xr
import zarr


LOGGER = logging.getLogger(__name__)


# Geometry of the global grids. The availability/time fields are stored on a
# grid coarsened by ``COARSENING`` relative to the 0.025-degree input grid, so
# each availability cell covers ``OBS_CELL`` input pixels and ``REF_CELL``
# reference pixels.
N_CELLS_LAT = 90
N_CELLS_LON = 180
OBS_CELL = 80  # Input (0.025-degree) pixels per availability cell.
REF_CELL = 40  # Reference (0.05-degree) pixels per availability cell.
RESOLUTION_RATIO = OBS_CELL // REF_CELL  # Input pixels per reference pixel.

DEFAULT_INPUT_SENSORS = ("goes16", "goes18", "seviri", "seviri_io")
DEFAULT_MICROWAVE_SENSORS = (
    "noaa20_atms", "npp_atms",
    "f16_ssmis", "f17_ssmis", "f18_ssmis", "f19_ssmis",
    "noaa18_mhs", "noaa19_mhs", "metopa_mhs", "metopb_mhs", "metopc_mhs",
    "tropics03_tms", "tropics05_tms", "tropics06_tms",
)
DEFAULT_REFERENCE = "gprof_gmi"

# ----------------------------------------------------------------------
# Channel slotting
# ----------------------------------------------------------------------
# Different geostationary imagers carry different numbers of channels at
# slightly different wavelengths. To present them to a model through a single,
# fixed tensor we map every sensor's channels onto a common set of ``N_SLOTS``
# spectral slots. Slots 0-15 are the GOES ABI band set (the most complete
# 16-channel imager here); each sensor channel is assigned to the ABI slot with
# the nearest central wavelength. The final slot is a catch-all for visible
# bands without an ABI counterpart (the Himawari green and the SEVIRI HRV band);
# no sensor carries both, so they share it. Slots a sensor lacks are left empty
# (NaN).
N_SLOTS = 17
EXTRA_SLOT = N_SLOTS - 1  # Catch-all slot for bands without an ABI counterpart.

# Canonical slot center wavelengths in micrometers (GOES ABI bands 1-16, then the
# extra visible slot which has no fixed wavelength) and a short label for each.
SLOT_WAVELENGTHS = (
    0.47, 0.64, 0.86, 1.37, 1.6, 2.2, 3.9, 6.2,
    6.9, 7.3, 8.4, 9.6, 10.3, 11.2, 12.3, 13.3, np.nan,
)
SLOT_NAMES = (
    "blue", "red", "veggie", "cirrus", "snow/ice", "cloud particle",
    "shortwave IR", "upper WV", "mid WV", "lower WV", "cloud phase", "ozone",
    "clean IR window", "IR window", "dirty IR window", "CO2",
    "extra visible (green/HRV)",
)

# Nominal channel center wavelengths (micrometers) in the order the channels are
# stored in each sensor's ``obs`` array.
SENSOR_WAVELENGTHS = {
    "goes16": (
        0.47, 0.64, 0.86, 1.37, 1.6, 2.2, 3.9, 6.2,
        6.9, 7.3, 8.4, 9.6, 10.3, 11.2, 12.3, 13.3,
    ),
    "goes18": (
        0.47, 0.64, 0.86, 1.37, 1.6, 2.2, 3.9, 6.2,
        6.9, 7.3, 8.4, 9.6, 10.3, 11.2, 12.3, 13.3,
    ),
    "himawari9": (
        0.455, 0.51, 0.645, 0.86, 1.61, 2.26, 3.85, 6.25,
        6.95, 7.3, 8.6, 9.63, 10.45, 11.2, 12.3, 13.3,
    ),
    "seviri": (
        0.75, 0.63, 0.81, 1.63, 3.9, 6.25, 7.35, 8.7, 9.66, 10.8, 12.0, 13.4,
    ),
    "seviri_io": (
        0.75, 0.63, 0.81, 1.63, 3.9, 6.25, 7.35, 8.7, 9.66, 10.8, 12.0, 13.4,
    ),
}


def _build_slot_map(wavelengths: Sequence[float]) -> List[int]:
    """
    Assign each channel to a slot, returning a slot index per channel.

    Channels are matched to the nearest ABI slot (slots with a defined
    wavelength) by central wavelength. If several channels of a sensor fall onto
    the same ABI slot only the closest is kept. Channels left without an ABI slot
    (e.g. the Himawari green and the SEVIRI HRV bands) are placed in the
    catch-all :data:`EXTRA_SLOT`.
    """
    abi_slots = [(i, wl) for i, wl in enumerate(SLOT_WAVELENGTHS) if np.isfinite(wl)]
    result = [-1] * len(wavelengths)
    assigned: Dict[int, Tuple[int, float]] = {}
    for channel, wavelength in enumerate(wavelengths):
        slot, distance = min(
            ((i, abs(wavelength - wl)) for i, wl in abi_slots), key=lambda t: t[1]
        )
        if slot in assigned:
            if distance < assigned[slot][1]:
                result[assigned[slot][0]] = -1
                assigned[slot] = (channel, distance)
                result[channel] = slot
        else:
            assigned[slot] = (channel, distance)
            result[channel] = slot
    # Channels without an ABI counterpart go to the catch-all extra slot.
    for channel in range(len(wavelengths)):
        if result[channel] == -1:
            result[channel] = EXTRA_SLOT
    return result


# Mapping of sensor name to a list (one entry per stored channel) giving the
# target slot index, or ``-1`` for channels without a slot.
CHANNEL_SLOTS = {
    sensor: _build_slot_map(wavelengths)
    for sensor, wavelengths in SENSOR_WAVELENGTHS.items()
}


# ----------------------------------------------------------------------
# Microwave channel slotting (ATMS + SSMIS)
# ----------------------------------------------------------------------
# Microwave channels are slotted by ``(frequency, 183-GHz water-vapor offset,
# polarization)``. V/QV and H/QH share a polarization class. Channels are
# assigned to the nearest slot of the same polarization class.
N_MW_SLOTS = 16

# Canonical microwave slots: (frequency [GHz], 183-GHz offset [GHz], pol class).
# Slot 14 is the V-polarized 157 GHz MHS channel, which has no counterpart among
# the conically-scanning imagers (their 157 GHz is H-polarized); the MHS 190.31
# GHz channel equals 183.31 + 7 GHz and is matched to the 183±7 slot. Slot 15 is
# the cross-track TROPICS 91.655 GHz window channel, whose polarization is neither
# H nor V ("TMS"). The TROPICS 184/186/190 GHz water-vapor channels share the
# 183±1/±3/±7 slots and the 204.8 GHz window channel the 157/165 GHz slot.
MW_SLOTS = (
    (19.35, 0.0, "V"), (19.35, 0.0, "H"),
    (22.5, 0.0, "V"), (31.4, 0.0, "V"),
    (37.0, 0.0, "V"), (37.0, 0.0, "H"),
    (89.5, 0.0, "V"), (91.65, 0.0, "H"),
    (157.0, 0.0, "H"),
    (183.31, 1.0, "H"), (183.31, 1.8, "H"), (183.31, 3.0, "H"),
    (183.31, 4.5, "H"), (183.31, 6.8, "H"),
    (157.0, 0.0, "V"),
    (91.655, 0.0, "TMS"),
)
MW_SLOT_NAMES = (
    "19V", "19H", "23V", "31V", "37V", "37H", "89V", "89H", "157H",
    "183±1", "183±1.8", "183±3", "183±4.5", "183±7",
    "157V",
    "91 (TMS)",
)

# Per-sensor channels as (frequency [GHz], offset [GHz], polarization) in the
# order the channels are stored in each sensor's ``obs`` array.
_ATMS_CHANNELS = (
    (23.8, 0.0, "V"), (31.4, 0.0, "V"), (88.2, 0.0, "V"), (165.5, 0.0, "H"),
    (183.31, 7.0, "H"), (183.31, 4.5, "H"), (183.31, 3.0, "H"),
    (183.31, 1.8, "H"), (183.31, 1.0, "H"),
)
_SSMIS_CHANNELS = (
    (19.35, 0.0, "V"), (19.35, 0.0, "H"), (22.235, 0.0, "V"),
    (37.0, 0.0, "V"), (37.0, 0.0, "H"), (150.0, 0.0, "H"),
    (183.31, 6.6, "H"), (183.31, 3.0, "H"), (183.31, 1.0, "H"),
    (91.65, 0.0, "V"), (91.65, 0.0, "H"),
)
_MHS_CHANNELS = (
    (89.0, 0.0, "V"), (157.0, 0.0, "V"),
    (183.31, 1.0, "H"), (183.31, 3.0, "H"),
    # 190.31 GHz = 183.31 + 7, matched to the 183±7 slot.
    (183.31, 7.0, "H"),
)
# TROPICS Microwave Sounder, in the stored swath/channel order. Its polarization
# is neither H nor V, so the 91.655 GHz window uses a dedicated "TMS" slot, while
# the 184/186/190 GHz water-vapor channels share the 183±1/±3/±7 slots and the
# 204.80 GHz window channel the 157/165 GHz slot (explicit slot indices).
_TMS_CHANNELS = (
    (91.655, 0.0, "TMS"),
    9,   # 184.41 GHz (183.31 + 1.1) -> 183±1
    11,  # 186.51 GHz (183.31 + 3.2) -> 183±3
    13,  # 190.31 GHz (183.31 + 7.0) -> 183±7
    8,   # 204.80 GHz window -> 157/165
)
MW_SENSOR_CHANNELS = {
    "noaa20_atms": _ATMS_CHANNELS,
    "npp_atms": _ATMS_CHANNELS,
    "f16_ssmis": _SSMIS_CHANNELS,
    "f17_ssmis": _SSMIS_CHANNELS,
    "f18_ssmis": _SSMIS_CHANNELS,
    "f19_ssmis": _SSMIS_CHANNELS,
    "noaa18_mhs": _MHS_CHANNELS,
    "noaa19_mhs": _MHS_CHANNELS,
    "metopa_mhs": _MHS_CHANNELS,
    "metopb_mhs": _MHS_CHANNELS,
    "metopc_mhs": _MHS_CHANNELS,
    "tropics03_tms": _TMS_CHANNELS,
    "tropics05_tms": _TMS_CHANNELS,
    "tropics06_tms": _TMS_CHANNELS,
}


def _pol_class(pol: str) -> str:
    """
    Map a polarization to its class (V/QV -> 'V', H/QH -> 'H').

    Other labels (e.g. the cross-track TROPICS channels, whose polarization is
    neither H nor V) are returned unchanged and so form their own class.
    """
    pol = pol.upper()
    if pol in ("V", "QV"):
        return "V"
    if pol in ("H", "QH"):
        return "H"
    return pol


def _build_mw_slot_map(
    channels: Sequence[Union[int, Tuple[float, float, str]]]
) -> List[int]:
    """
    Assign each microwave channel to a slot, returning a slot index per channel.

    A channel given as ``(frequency, offset, polarization)`` is matched to the
    nearest slot of the same polarization class; if several channels of a sensor
    fall onto the same slot only the closest is kept. A channel given as a plain
    ``int`` is assigned that slot index explicitly.
    """
    result = [-1] * len(channels)
    assigned: Dict[int, Tuple[int, float]] = {}
    for channel, spec in enumerate(channels):
        if isinstance(spec, int):
            result[channel] = spec
            continue
        freq, offset, pol = spec
        pc = _pol_class(pol)
        candidates = [
            (i, (freq - sf) ** 2 + (offset - so) ** 2)
            for i, (sf, so, sp) in enumerate(MW_SLOTS)
            if sp == pc
        ]
        if not candidates:
            continue
        slot, distance = min(candidates, key=lambda t: t[1])
        if slot in assigned:
            if distance < assigned[slot][1]:
                result[assigned[slot][0]] = -1
                assigned[slot] = (channel, distance)
                result[channel] = slot
        else:
            assigned[slot] = (channel, distance)
            result[channel] = slot
    return result


MW_CHANNEL_SLOTS = {
    sensor: _build_mw_slot_map(channels)
    for sensor, channels in MW_SENSOR_CHANNELS.items()
}


def slot_observations(
    obs: np.ndarray, sensor: str, fill: float = np.nan
) -> np.ndarray:
    """
    Map a sensor's observations onto its common set of channel slots.

    Geostationary sensors are mapped onto the :data:`N_SLOTS` spectral slots and
    microwave sensors onto the :data:`N_MW_SLOTS` frequency slots, depending on
    which slotting scheme the sensor belongs to.

    Args:
        obs: An observation array of shape ``(channels, ...)`` in the sensor's
            stored channel order.
        sensor: The sensor name.
        fill: Value used for slots the sensor does not provide.

    Returns:
        An array of shape ``(n_slots, ...)`` with each channel placed in its
        slot and missing slots set to ``fill``.
    """
    if sensor in CHANNEL_SLOTS:
        slots, n_slots = CHANNEL_SLOTS[sensor], N_SLOTS
    elif sensor in MW_CHANNEL_SLOTS:
        slots, n_slots = MW_CHANNEL_SLOTS[sensor], N_MW_SLOTS
    else:
        raise KeyError(f"No channel slotting defined for sensor '{sensor}'.")
    if obs.shape[0] != len(slots):
        raise ValueError(
            f"'{sensor}' observations have {obs.shape[0]} channels but the "
            f"slotting scheme expects {len(slots)}."
        )
    out = np.full((n_slots,) + obs.shape[1:], fill, dtype=np.float32)
    for channel, slot in enumerate(slots):
        if slot >= 0:
            out[slot] = obs[channel]
    return out


_TIMESTAMP_REGEXP = re.compile(r"(\d{14})")


def _parse_timestamp(path: Path) -> Optional[np.datetime64]:
    """Parse the ``YYYYmmddHHMMSS`` timestamp from a store's file name."""
    match = _TIMESTAMP_REGEXP.search(path.stem)
    if match is None:
        return None
    return np.datetime64(
        f"{match[1][:4]}-{match[1][4:6]}-{match[1][6:8]}T"
        f"{match[1][8:10]}:{match[1][10:12]}:{match[1][12:14]}",
        "ns",
    )


def worker_init_fn(worker_id: int) -> None:
    """
    Seed NumPy's global RNG for a ``DataLoader`` worker.

    The dataset uses ``numpy.random`` for the random center jitter and sensor
    selection in :meth:`ArgosTrainingData.__getitem__`. Forked workers would
    otherwise inherit the same RNG state and produce identical random draws, so
    pass this as the ``worker_init_fn`` of a ``torch.utils.data.DataLoader``::

        DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)

    Each worker is seeded from its (already worker-distinct) torch seed, so draws
    differ across workers and across epochs.
    """
    np.random.seed(torch.initial_seed() % (2**32))


class ArgosTrainingData(Dataset):
    """
    Dataset pairing geostationary observations with GPM surface precipitation.

    Each sample is a spatially- and temporally-matched tile consisting of a
    geostationary observation patch (the input) and the co-located
    ``surface_precip`` patch from the GPM reference data (the target). Tiles are
    aligned to the 2-degree availability grid and the input patch is
    ``RESOLUTION_RATIO`` times larger (in pixels) than the reference patch.
    """

    def __init__(
        self,
        path: Union[str, Path],
        input_sensors: Sequence[str] = DEFAULT_INPUT_SENSORS,
        microwave_sensors: Sequence[str] = DEFAULT_MICROWAVE_SENSORS,
        reference_name: str = DEFAULT_REFERENCE,
        tile_size: int = 128,
        time_window: np.timedelta64 = np.timedelta64(10, "m"),
        position_jitter: int = REF_CELL // 2,
        slot_channels: bool = True,
        normalize: bool = True,
        require_both_inputs: bool = False,
    ):
        """
        Args:
            path: Root directory containing the per-sensor sub-directories.
            input_sensors: Names of the geostationary sensor sub-directories to
                use as the high-resolution ``"geo"`` input.
            microwave_sensors: Names of the microwave sensor sub-directories
                (ATMS, SSMIS, ...) to use as the ``"mw"`` input. These are on the
                lower (reference) resolution grid and, like the reference, have a
                per-cell scan time.
            reference_name: Name of the reference sub-directory.
            tile_size: Side length of the (square) reference tile in reference
                pixels. The input tile has side length
                ``tile_size * RESOLUTION_RATIO``.
            time_window: Maximum allowed difference between an input
                acquisition time and a reference cell's scan time.
            position_jitter: Maximum random shift of the tile center applied when
                loading a sample, in reference (0.05-degree) pixels (and
                ``RESOLUTION_RATIO`` times as many input pixels). Defaults to half
                an availability cell (20 reference / 40 input pixels). Set to 0 to
                disable jittering.
            slot_channels: If ``True`` (the default), the loaded sensor's
                observations are mapped onto the common spectral slots (see
                :func:`slot_observations`) and returned as the ``"geo"`` and
                ``"mw"`` inputs of shape ``(n_slots, H, W)`` (absent bands set to
                ``NaN``). If ``False``, the raw observation tensor of the loaded
                sensor is returned under the sensor's name.
            normalize: If ``True`` (the default), the loaded observations are
                scaled to ``[0, 1]`` per channel using the sensor-wise
                :attr:`normalization_stats` (min/max across all of the sensor's
                stores).
            require_both_inputs: If ``True``, only keep samples that have both a
                geostationary and a microwave observation (by default a sample is
                kept if it has either).
        """
        super().__init__()
        self.path = Path(path)
        self.input_sensors = tuple(input_sensors)
        self.microwave_sensors = tuple(microwave_sensors)
        self.reference_name = reference_name
        self.tile_size = int(tile_size)
        self.time_window = np.timedelta64(time_window, "ns")
        self.position_jitter = int(position_jitter)
        self.slot_channels = bool(slot_channels)
        self.normalize = bool(normalize)
        self.require_both_inputs = bool(require_both_inputs)

        if self.tile_size > REF_CELL * min(N_CELLS_LAT, N_CELLS_LON):
            raise ValueError(
                f"A tile_size of {self.tile_size} exceeds the global grid."
            )

        # Group each input sensor as high-resolution geo ("geo") or
        # reference-resolution microwave ("mw").
        self._sensor_group = {
            **{sensor: "geo" for sensor in self.input_sensors},
            **{sensor: "mw" for sensor in self.microwave_sensors},
        }

    # ``DataLoader(worker_init_fn=...)`` helper to seed NumPy per worker.
    worker_init_fn = staticmethod(worker_init_fn)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    @cached_property
    def geo_files(self) -> Dict[str, List[Path]]:
        """Mapping of each input sensor name to its sorted list of '.zarr' stores."""
        return {
            sensor: sorted((self.path / sensor).glob("*.zarr"))
            for sensor in self.input_sensors
        }

    @cached_property
    def mw_files(self) -> Dict[str, List[Path]]:
        """Mapping of each microwave sensor name to its sorted '.zarr' stores."""
        return {
            sensor: sorted((self.path / sensor).glob("*.zarr"))
            for sensor in self.microwave_sensors
        }

    @cached_property
    def reference_files(self) -> Dict[str, List[Path]]:
        """Mapping of the reference name to its sorted list of '.zarr' stores."""
        files = sorted((self.path / self.reference_name).glob("*.zarr"))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No reference '.zarr' stores found in "
                f"'{self.path / self.reference_name}'."
            )
        return {self.reference_name: files}

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------
    def _load_indices(self, files_by_name, compute) -> Dict[str, xr.Dataset]:
        """
        Load (or compute and cache) the per-sensor index of each named dataset.

        Returns a mapping of name to its index dataset, skipping names with no
        stores. A cached index that predates the per-channel ``obs_min``/
        ``obs_max`` statistics is recomputed so the normalization stats stay
        available.
        """
        indices = {}
        for name, files in files_by_name.items():
            meta = self._read_index(name)
            if meta is None or "obs_min" not in meta:
                meta = compute(name, files)
                self._write_index(name, meta)
            if meta.sizes["samples"] > 0:
                indices[name] = meta
        return indices

    @cached_property
    def _geo_indices(self) -> Dict[str, xr.Dataset]:
        """Per-sensor index datasets of the geostationary inputs."""
        return self._load_indices(self.geo_files, self._compute_geo_meta)

    @cached_property
    def _mw_indices(self) -> Dict[str, xr.Dataset]:
        """Per-sensor index datasets of the microwave inputs."""
        return self._load_indices(self.mw_files, self._compute_reference_meta)

    @cached_property
    def geo_meta(self) -> xr.Dataset:
        """
        Availability and acquisition time of all input stores.

        A single :class:`xarray.Dataset` with the stores of every sensor
        concatenated along a ``samples`` dimension. It holds ``availability``
        (``(samples, lat_cell, lon_cell)`` boolean) and ``time`` (``(samples,)``
        ``datetime64[ns]``), with ``name`` (sensor) and ``path`` coordinates
        along ``samples`` providing the mapping from sample index to file. The
        per-sensor metadata is cached to ``index_<sensor>.nc`` and loaded from
        there on subsequent runs (see :meth:`recompute_indices` to refresh). The
        per-channel ``obs_min``/``obs_max`` statistics are dropped here (their
        channel dimension differs between sensors) and exposed instead through
        :attr:`normalization_stats`.
        """
        datasets = [
            self._attach_paths(idx, sensor).drop_vars(
                ["obs_min", "obs_max"], errors="ignore"
            )
            for sensor, idx in self._geo_indices.items()
        ]
        if not datasets:
            raise FileNotFoundError("No input '.zarr' stores found.")
        return xr.concat(datasets, dim="samples")

    def _compute_geo_meta(self, sensor: str, files: List[Path]) -> xr.Dataset:
        """Read the availability and acquisition time of one sensor's stores."""
        LOGGER.info(
            "Loading metadata for input sensor '%s' (%d files).", sensor, len(files)
        )
        names: List[str] = []
        avail: List[np.ndarray] = []
        time: List[np.datetime64] = []
        obs_min: List[np.ndarray] = []
        obs_max: List[np.ndarray] = []
        for rec in tqdm(files, desc=f"Loading {sensor} metadata", unit="file"):
            try:
                store = zarr.open_group(str(rec), mode="r")
                availability = np.asarray(store["availability"][:]) > 0
                # Prefer the stored acquisition time, fall back to the name.
                stored = np.asarray(store["time"][...])
                acq = (
                    stored.astype("datetime64[ns]")
                    if stored.size == 1
                    else _parse_timestamp(rec)
                )
            except Exception as exc:  # noqa: BLE001 - skip unreadable stores.
                LOGGER.warning("Skipping input store '%s': %s", rec, exc)
                continue
            if acq is None:
                LOGGER.warning("No timestamp for input store '%s'.", rec)
                continue
            names.append(rec.name)
            avail.append(availability)
            time.append(np.datetime64(acq, "ns"))
            if "obs_min" in store:
                obs_min.append(np.asarray(store["obs_min"][:], dtype=np.float32))
                obs_max.append(np.asarray(store["obs_max"][:], dtype=np.float32))
        LOGGER.info(
            "Loaded metadata for %d/%d '%s' stores.", len(names), len(files), sensor
        )
        availability = (
            np.stack(avail)
            if avail
            else np.empty((0, N_CELLS_LAT, N_CELLS_LON), dtype=bool)
        )
        data_vars = {
            "availability": (
                ("samples", "lat_cell", "lon_cell"),
                availability.astype("int8"),
            ),
            "time": ("samples", np.array(time, dtype="datetime64[ns]")),
        }
        if len(obs_min) == len(names) and obs_min:
            data_vars["obs_min"] = (("samples", "channel"), np.stack(obs_min))
            data_vars["obs_max"] = (("samples", "channel"), np.stack(obs_max))
        return xr.Dataset(
            data_vars,
            coords={"filename": ("samples", np.array(names, dtype=object))},
        )

    @cached_property
    def reference_meta(self) -> xr.Dataset:
        """
        Availability and per-cell scan time of all reference stores.

        A single :class:`xarray.Dataset` with the reference stores concatenated
        along a ``samples`` dimension. It holds ``availability``
        (``(samples, lat_cell, lon_cell)`` boolean) and ``time``
        (``(samples, lat_cell, lon_cell)`` ``datetime64[ns]``, ``NaT`` where no
        data was observed), with ``name`` and ``path`` coordinates along
        ``samples`` providing the mapping from sample index to file. The metadata
        is cached to ``index_<reference_name>.nc`` and loaded from there on
        subsequent runs (see :meth:`recompute_indices` to refresh).
        """
        datasets = []
        for name, files in self.reference_files.items():
            meta = self._read_index(name)
            if meta is None:
                meta = self._compute_reference_meta(name, files)
                self._write_index(name, meta)
            datasets.append(self._attach_paths(meta, name))
        datasets = [ds for ds in datasets if ds.sizes["samples"] > 0]
        if not datasets:
            raise FileNotFoundError("No reference '.zarr' stores found.")
        return xr.concat(datasets, dim="samples")

    @cached_property
    def microwave_meta(self) -> xr.Dataset:
        """
        Availability and per-cell scan time of all microwave stores.

        Like :attr:`reference_meta` (the microwave data is on the reference grid
        with a per-cell scan time), but with the stores of every microwave sensor
        concatenated along ``samples`` and ``name``/``path`` coordinates for the
        sample-index-to-file mapping. Returns an empty dataset if there are no
        microwave stores. Cached per sensor to ``index_<sensor>.nc``. Per-channel
        ``obs_min``/``obs_max`` statistics are exposed via
        :attr:`normalization_stats`.
        """
        datasets = [
            self._attach_paths(idx, sensor).drop_vars(
                ["obs_min", "obs_max"], errors="ignore"
            )
            for sensor, idx in self._mw_indices.items()
        ]
        if not datasets:
            return self._attach_paths(self._compute_reference_meta("", []), "")
        return xr.concat(datasets, dim="samples")

    @cached_property
    def normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Per-sensor input normalization statistics.

        A mapping of input sensor name to ``{"min": ..., "max": ...}``, each a
        per-channel array giving the minimum and maximum observed value across
        all of the sensor's stores (in the sensor's stored channel order). Used
        by :meth:`__getitem__` to scale the inputs to ``[0, 1]`` when
        ``normalize`` is enabled.
        """
        stats = {}
        for sensor, idx in {**self._geo_indices, **self._mw_indices}.items():
            if "obs_min" not in idx:
                continue
            # A channel that was never observed yields an all-NaN slice (and a
            # NaN stat); :meth:`_normalize` passes such channels through.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats[sensor] = {
                    "min": np.nanmin(idx["obs_min"].values, axis=0),
                    "max": np.nanmax(idx["obs_max"].values, axis=0),
                }
        return stats

    def _compute_reference_meta(self, name: str, files: List[Path]) -> xr.Dataset:
        """Read the availability and per-cell scan time of the reference stores."""
        LOGGER.info(
            "Loading metadata for reference '%s' (%d files).", name, len(files)
        )
        names: List[str] = []
        avail: List[np.ndarray] = []
        time: List[np.ndarray] = []
        obs_min: List[np.ndarray] = []
        obs_max: List[np.ndarray] = []
        for rec in tqdm(files, desc=f"Loading {name} metadata", unit="file"):
            try:
                store = zarr.open_group(str(rec), mode="r")
                availability = np.asarray(store["availability"][:]) > 0
                raw = np.asarray(store["time"][:]).astype("int64")
                times = raw.astype("datetime64[ns]")
                # A scan time of 0 (the epoch) marks cells without data.
                times[raw == 0] = np.datetime64("NaT")
            except Exception as exc:  # noqa: BLE001 - skip unreadable stores.
                LOGGER.warning("Skipping reference store '%s': %s", rec, exc)
                continue
            names.append(rec.name)
            avail.append(availability)
            time.append(times)
            # Microwave stores carry per-channel obs statistics; the reference
            # (surface precipitation) store does not.
            if "obs_min" in store:
                obs_min.append(np.asarray(store["obs_min"][:], dtype=np.float32))
                obs_max.append(np.asarray(store["obs_max"][:], dtype=np.float32))
        LOGGER.info(
            "Loaded metadata for %d/%d '%s' stores.", len(names), len(files), name
        )
        availability = (
            np.stack(avail)
            if avail
            else np.empty((0, N_CELLS_LAT, N_CELLS_LON), dtype=bool)
        )
        times = (
            np.stack(time)
            if time
            else np.empty((0, N_CELLS_LAT, N_CELLS_LON), dtype="datetime64[ns]")
        )
        data_vars = {
            "availability": (
                ("samples", "lat_cell", "lon_cell"),
                availability.astype("int8"),
            ),
            "time": (("samples", "lat_cell", "lon_cell"), times),
        }
        if len(obs_min) == len(names) and obs_min:
            data_vars["obs_min"] = (("samples", "channel"), np.stack(obs_min))
            data_vars["obs_max"] = (("samples", "channel"), np.stack(obs_max))
        return xr.Dataset(
            data_vars,
            coords={"filename": ("samples", np.array(names, dtype=object))},
        )

    def _attach_paths(self, meta: xr.Dataset, name: str) -> xr.Dataset:
        """Add ``name`` and full ``path`` coordinates to a per-name index."""
        n = meta.sizes["samples"]
        paths = [str(self.path / name / str(fn)) for fn in meta["filename"].values]
        return meta.assign_coords(
            name=("samples", np.full(n, name, dtype=object)),
            path=("samples", np.array(paths, dtype=object)),
        )

    # ------------------------------------------------------------------
    # Index caching
    # ------------------------------------------------------------------
    def _index_path(self, name: str) -> Path:
        """Path of the cached metadata index for a given dataset name."""
        return self.path / f"index_{name}.nc"

    def _read_index(self, name: str) -> Optional[xr.Dataset]:
        """Load a cached metadata index, or return ``None`` if it is absent."""
        path = self._index_path(name)
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            meta = ds.load()
        LOGGER.info(
            "Loaded cached index for '%s' from '%s' (%d stores).",
            name,
            path,
            meta.sizes["samples"],
        )
        return meta

    def _write_index(self, name: str, meta: xr.Dataset) -> None:
        """Cache a metadata index to ``index_<name>.nc``."""
        if meta.sizes["samples"] == 0:
            # Nothing to cache; allow data that appears later to be picked up.
            return
        path = self._index_path(name)
        meta.to_netcdf(path)
        LOGGER.info("Wrote metadata index for '%s' to '%s'.", name, path)

    def recompute_indices(self) -> None:
        """
        Delete any cached metadata indices and recompute them from the stores.

        This discards the ``index_<name>.nc`` files and the cached ``geo_meta``,
        ``microwave_meta``, ``reference_meta`` and ``samples`` so that the next
        access re-reads the stores and writes fresh indices.
        """
        names = (
            *self.input_sensors,
            *self.microwave_sensors,
            self.reference_name,
        )
        for name in names:
            path = self._index_path(name)
            if path.exists():
                path.unlink()
                LOGGER.info("Removed cached index '%s'.", path)
        for attr in (
            "_geo_indices", "_mw_indices", "geo_meta", "microwave_meta",
            "reference_meta", "normalization_stats", "samples",
        ):
            self.__dict__.pop(attr, None)
        # Trigger recomputation (and re-caching) of the metadata.
        self.geo_meta
        self.microwave_meta
        self.reference_meta

    @cached_property
    def _reference_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """The (latitude, longitude) arrays of the global reference grid."""
        first = self.reference_files[self.reference_name][0]
        store = zarr.open_group(str(first), mode="r")
        return (
            np.asarray(store["latitude"][:]).astype(np.float32),
            np.asarray(store["longitude"][:]).astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Sample enumeration
    # ------------------------------------------------------------------
    @cached_property
    def samples(self) -> List[Dict[str, object]]:
        """
        Produces list of samples specification.

        One sample per reduced-resolution reference cell with any valid input.

        A sample is created for every reference availability cell that has valid
        reference data and at least one input (geostationary or microwave) within
        the time window -- or both inputs if ``require_both_inputs`` is set. Each
        sample is a dictionary with:

        * ``"coordinates"``: the ``(row, column)`` index of the cell center with
          respect to the full-resolution (0.025-degree) input grid.
        * ``"geo"`` / ``"mw"``: a mapping of sensor name to the ``.zarr`` store
          available at the cell, for the geostationary and microwave inputs
          respectively (the sensor is chosen at random on loading).
        * ``"reference"``: the reference ``.zarr`` store.
        """
        window = self.time_window

        # Per-sensor input arrays (geo and microwave), with a per-file time span
        # used to pre-filter candidates.
        inputs = {
            **self._input_arrays(self.geo_meta),
            **self._input_arrays(self.microwave_meta),
        }

        ref = self.reference_meta
        ref_avail = ref["availability"].values.astype(bool)
        ref_time = ref["time"].values
        ref_path = ref["path"].values

        samples: List[Dict[str, object]] = []
        for ref_idx in tqdm(
            range(ref_avail.shape[0]), desc="Building samples", unit="granule"
        ):
            r_avail = ref_avail[ref_idx]
            r_time = ref_time[ref_idx]
            valid = r_time[r_avail & ~np.isnat(r_time)]
            if valid.size == 0:
                continue
            tmin, tmax = valid.min(), valid.max()

            # For each sensor, pick the file that best matches this reference
            # granule and keep its per-cell joint availability, i.e. cells that
            # are valid in both the reference and the input and within the time
            # window.
            matched_by_sensor: Dict[str, Tuple[np.ndarray, str]] = {}
            for sensor, info in inputs.items():
                candidates = np.where(
                    (info["tmax"] >= tmin - window) & (info["tmin"] <= tmax + window)
                )[0]
                best_matched, best_path, best_score = None, None, 0
                for k in candidates:
                    matched = (
                        r_avail
                        & info["availability"][k]
                        & (np.abs(r_time - info["time"][k]) <= window)
                    )
                    score = matched.sum()
                    if score > best_score:
                        best_matched, best_path, best_score = matched, info["path"][k], score
                if best_path is not None:
                    matched_by_sensor[sensor] = (best_matched, best_path)
            if not matched_by_sensor:
                continue

            # Every reference cell with at least one matching input is a sample.
            any_matched = np.zeros((N_CELLS_LAT, N_CELLS_LON), dtype=bool)
            for matched, _ in matched_by_sensor.values():
                any_matched |= matched
            for ci, cj in np.argwhere(any_matched):
                geo = {
                    sensor: Path(str(path))
                    for sensor, (matched, path) in matched_by_sensor.items()
                    if self._sensor_group[sensor] == "geo" and matched[ci, cj]
                }
                mw = {
                    sensor: Path(str(path))
                    for sensor, (matched, path) in matched_by_sensor.items()
                    if self._sensor_group[sensor] == "mw" and matched[ci, cj]
                }
                if self.require_both_inputs and not (geo and mw):
                    continue
                samples.append(
                    {
                        "coordinates": (
                            int(ci) * OBS_CELL + OBS_CELL // 2,
                            int(cj) * OBS_CELL + OBS_CELL // 2,
                        ),
                        "geo": geo,
                        "mw": mw,
                        "reference": Path(str(ref_path[ref_idx])),
                    }
                )

        LOGGER.info(
            "Built %d samples from %d reference and %d input stores.",
            len(samples),
            ref_avail.shape[0],
            sum(info["path"].shape[0] for info in inputs.values()),
        )
        return samples

    def _input_arrays(self, meta: xr.Dataset) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Group a metadata dataset into per-sensor matching arrays.

        Returns, per sensor, the ``availability`` ``(n, 90, 180)`` and ``time``
        (``(n,)`` for geo or ``(n, 90, 180)`` for microwave) arrays, the file
        ``path`` array, and the per-file time span (``tmin``/``tmax``) used to
        pre-filter candidates.
        """
        if meta.sizes["samples"] == 0:
            return {}
        availability = meta["availability"].values.astype(bool)
        time = meta["time"].values
        path = meta["path"].values
        name = meta["name"].values

        result: Dict[str, Dict[str, np.ndarray]] = {}
        for sensor in dict.fromkeys(name.tolist()):
            sel = np.where(name == sensor)[0]
            s_time = time[sel]
            if s_time.ndim == 1:
                # Geostationary: a single acquisition time per file.
                tmin = tmax = s_time
            else:
                # Microwave/reference: a per-cell scan time, reduced per file.
                s_avail = availability[sel]
                tmin = np.full(len(sel), np.datetime64("NaT"), dtype="datetime64[ns]")
                tmax = tmin.copy()
                for i in range(len(sel)):
                    valid = s_time[i][s_avail[i] & ~np.isnat(s_time[i])]
                    if valid.size:
                        tmin[i], tmax[i] = valid.min(), valid.max()
            result[str(sensor)] = {
                "availability": availability[sel],
                "time": s_time,
                "path": path[sel],
                "tmin": tmin,
                "tmax": tmax,
            }
        return result

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_obs(path: Path, r0: int, c0: int, size: int) -> np.ndarray:
        """Load and decode an observation patch (``255`` -> ``NaN``)."""
        store = zarr.open_group(str(path), mode="r")
        obs = np.asarray(store["obs"][:, r0 : r0 + size, c0 : c0 + size])
        offsets = np.asarray(
            store.attrs.get("offsets", np.zeros(obs.shape[0])), dtype=np.float32
        )
        fill = store.attrs.get("fill_value", 255)
        obs_f = obs.astype(np.float32) + offsets[:, None, None]
        obs_f[obs == fill] = np.nan
        return obs_f

    @staticmethod
    def _load_mw_obs(path: Path, r0: int, c0: int, size: int) -> np.ndarray:
        """Load and decode a microwave observation patch (fill -> ``NaN``)."""
        store = zarr.open_group(str(path), mode="r")
        obs = np.asarray(store["obs"][:, r0 : r0 + size, c0 : c0 + size])
        scale = float(store.attrs.get("scale_factor", 0.1))
        fill = store.attrs.get("fill_value", -1)
        obs_f = obs.astype(np.float32) * scale
        obs_f[obs == fill] = np.nan
        return obs_f

    @staticmethod
    def _load_reference(path: Path, r0: int, c0: int, size: int) -> np.ndarray:
        """Load and decode a surface-precipitation patch (fill -> ``NaN``)."""
        store = zarr.open_group(str(path), mode="r")
        sp = np.asarray(
            store["surface_precip"][r0 : r0 + size, c0 : c0 + size]
        )
        scale = float(store.attrs.get("scale_factor", 0.02))
        fill = store.attrs.get("fill_value", -1)
        sp_f = sp.astype(np.float32) * scale
        sp_f[sp == fill] = np.nan
        return sp_f

    def _normalize(self, array: np.ndarray, sensor: str) -> np.ndarray:
        """Scale a sensor's ``(channel, ...)`` observations to ``[0, 1]``."""
        stats = self.normalization_stats.get(sensor)
        if stats is None:
            return array
        lo = stats["min"][:, None, None]
        span = stats["max"][:, None, None] - lo
        # Leave channels without valid statistics (e.g. never-observed) as is.
        valid = np.isfinite(lo) & np.isfinite(span) & (span > 0)
        lo = np.where(valid, lo, 0.0)
        span = np.where(valid, span, 1.0)
        return ((array - lo) / span).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, object], torch.Tensor]:
        sample = self.samples[index]
        row_c, col_c = sample["coordinates"]
        latitude, longitude = self._reference_grid

        # Randomly jitter the tile center (kept aligned between the input and
        # reference grids), then derive the crop windows.
        obs_size = self.tile_size * RESOLUTION_RATIO
        obs_rows, obs_cols = N_CELLS_LAT * OBS_CELL, N_CELLS_LON * OBS_CELL
        if self.position_jitter > 0:
            row_c = int(row_c) + RESOLUTION_RATIO * np.random.randint(
                -self.position_jitter, self.position_jitter + 1
            )
            col_c = int(col_c) + RESOLUTION_RATIO * np.random.randint(
                -self.position_jitter, self.position_jitter + 1
            )
        obs_r0 = min(max(int(row_c) - obs_size // 2, 0), obs_rows - obs_size)
        obs_c0 = min(max(int(col_c) - obs_size // 2, 0), obs_cols - obs_size)
        ref_r0, ref_c0 = obs_r0 // RESOLUTION_RATIO, obs_c0 // RESOLUTION_RATIO

        # For each input group, randomly choose one of the available sensors and
        # load its observations. ``"geo"`` is high resolution, ``"mw"`` is on the
        # reference grid. When slotting, the observations are mapped onto the
        # group's common slots; otherwise the raw tensor is returned under the
        # sensor name.
        obs = {}
        for group, sensors, loader, (r0, c0, size) in (
            ("geo", sample["geo"], self._load_obs, (obs_r0, obs_c0, obs_size)),
            ("mw", sample["mw"], self._load_mw_obs, (ref_r0, ref_c0, self.tile_size)),
        ):
            if not sensors:
                continue
            sensor = str(np.random.choice(sorted(sensors)))
            array = loader(sensors[sensor], r0, c0, size)
            if self.normalize:
                array = self._normalize(array, sensor)
            if self.slot_channels:
                obs[group] = torch.from_numpy(slot_observations(array, sensor))
            else:
                obs[sensor] = torch.from_numpy(array)

        # Reference surface precipitation (the target).
        surface_precip = self._load_reference(
            sample["reference"], ref_r0, ref_c0, self.tile_size
        )

        inputs = {
            "obs": obs,
            "coordinates": (int(row_c), int(col_c)),
            "latitude": torch.from_numpy(
                latitude[ref_r0 : ref_r0 + self.tile_size].copy()
            ),
            "longitude": torch.from_numpy(
                longitude[ref_c0 : ref_c0 + self.tile_size].copy()
            ),
        }
        return inputs, torch.from_numpy(surface_precip)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _file_days(meta: xr.Dataset) -> np.ndarray:
        """Per-file day (``datetime64[D]``) for an availability/time index."""
        times = meta["time"].values
        if times.ndim > 1:
            # Reference: reduce the per-cell scan times to the earliest valid one.
            flat = times.reshape(times.shape[0], -1)
            reduced = []
            for row in flat:
                valid = row[~np.isnat(row)]
                reduced.append(valid.min() if valid.size else np.datetime64("NaT"))
            times = np.array(reduced, dtype="datetime64[ns]")
        return times.astype("datetime64[D]")

    def plot_file_availability(self, ax=None):
        """
        Plot the number of available files per day for every dataset.

        One line per input sensor (geostationary and microwave) and one for the
        reference dataset, showing how many files are available on each day
        across the full date range.

        Args:
            ax: An optional ``matplotlib`` axis to draw into.

        Returns:
            The created or parent ``matplotlib`` figure.
        """
        import matplotlib.pyplot as plt

        # Per-dataset arrays of per-file days, for every input group and the
        # reference.
        days_per_dataset: Dict[str, np.ndarray] = {}
        for meta, sensors in (
            (self.geo_meta, self.input_sensors),
            (self.microwave_meta, self.microwave_sensors),
        ):
            if meta.sizes["samples"] == 0:
                continue
            names = meta["name"].values
            days = self._file_days(meta)
            for sensor in sensors:
                mask = names == sensor
                if mask.any():
                    days_per_dataset[sensor] = days[mask]
        days_per_dataset[self.reference_name] = self._file_days(self.reference_meta)

        # Common, contiguous range of days.
        valid = [d[~np.isnat(d)] for d in days_per_dataset.values()]
        valid = [d for d in valid if d.size]
        if not valid:
            raise ValueError("No dated files available to plot.")
        all_days = np.concatenate(valid)
        day_range = np.arange(
            all_days.min(),
            all_days.max() + np.timedelta64(1, "D"),
            dtype="datetime64[D]",
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8.0, 4.0))
        else:
            fig = ax.figure

        for name, days in days_per_dataset.items():
            days = days[~np.isnat(days)]
            counts = np.zeros(len(day_range), dtype=int)
            if days.size:
                unique, n = np.unique(days, return_counts=True)
                counts[np.searchsorted(day_range, unique)] = n
            ax.plot(day_range, counts, marker="o", markersize=3, label=name)

        ax.set_xlabel("day")
        ax.set_ylabel("number of files")
        ax.set_title("File availability by day")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig


# Reference surface-precipitation contour levels and their line styles.
PRECIP_LEVELS = (0.1, 1.0, 10.0)
PRECIP_LINESTYLES = (":", "--", "-")


def plot_sample(sample: Tuple[Dict[str, object], object], n_channels: int = 3, rng=None):
    """
    Plot a sample with one row of input channels per sensor.

    For every input sensor ``n_channels`` randomly chosen channels are shown,
    each overlaid with grey contours of the reference surface precipitation at
    ``PRECIP_LEVELS`` (0.1, 1.0 and 10.0), drawn dotted, dashed and solid
    respectively.

    Args:
        sample: An ``(inputs, target)`` tuple as returned by
            ``ArgosTrainingData.__getitem__`` -- ``inputs`` holding an ``"obs"``
            mapping of sensor name to observation tensor and ``"latitude"`` /
            ``"longitude"`` coordinate tensors, and ``target`` the surface
            precipitation tensor.
        n_channels: The number of channels to show per input sensor.
        rng: Optional seed or ``numpy.random.Generator`` controlling the random
            channel selection.

    Returns:
        The created ``matplotlib`` figure.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(rng)

    inputs, target = sample
    obs = inputs["obs"]
    sensors = list(obs)
    precip = np.asarray(target)
    lat = np.asarray(inputs["latitude"])
    lon = np.asarray(inputs["longitude"])
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Only contour the levels that fall within the available precip range.
    levels, linestyles = [], []
    if np.isfinite(precip).any():
        pmin, pmax = np.nanmin(precip), np.nanmax(precip)
        for level, style in zip(PRECIP_LEVELS, PRECIP_LINESTYLES):
            if pmin <= level <= pmax:
                levels.append(level)
                linestyles.append(style)

    fig, axs = plt.subplots(
        len(sensors),
        n_channels,
        figsize=(4 * n_channels, 4 * len(sensors)),
        squeeze=False,
    )

    for row, sensor in enumerate(sensors):
        data = np.asarray(obs[sensor])
        channels = rng.choice(
            data.shape[0], size=min(n_channels, data.shape[0]), replace=False
        )
        for col in range(n_channels):
            ax = axs[row, col]
            if col >= len(channels):
                ax.set_axis_off()
                continue
            channel = int(channels[col])
            image = data[channel]
            if np.isfinite(image).any():
                vmin, vmax = np.nanpercentile(image, [2, 98])
            else:
                vmin, vmax = None, None
            ax.imshow(
                image,
                extent=extent,
                origin="upper",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            if levels:
                ax.contour(
                    lon_grid,
                    lat_grid,
                    precip,
                    levels=levels,
                    colors="grey",
                    linestyles=linestyles,
                    linewidths=2.0,
                )
            ax.set_title(f"{sensor} — channel {channel}")
            if col == 0:
                ax.set_ylabel("latitude")
            if row == len(sensors) - 1:
                ax.set_xlabel("longitude")

    fig.tight_layout()
    return fig
