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
import hashlib
import html
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional
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


class _SampleIndex:
    """
    Compact, fork-safe table of training samples.

    Each sample is one reduced-resolution reference cell with at least one valid
    input. The data is held in a few contiguous numpy arrays rather than a list
    of dicts, so forked ``DataLoader`` workers share it via copy-on-write without
    the memory blow-up (and refcount churn) of a large Python object graph. It is
    also small enough to cache to disk and reload directly, avoiding a re-read of
    the large per-sensor metadata indices in every worker.

    Arrays (``n`` samples, ``S`` input sensors, ``R`` reference files):

    * ``coords`` ``(n, 2)`` int32 -- full-resolution ``(row, column)`` cell center
    * ``ref_file`` ``(n,)`` int32 -- index into ``ref_filenames``
    * ``ref_time`` ``(n,)`` datetime64 -- the reference cell's scan time (from the
      per-cell time field, *not* the granule filename, which can be ~90 min off)
    * ``sensor_file`` ``(n, S)`` int32 -- per-sensor file index, ``-1`` if the
      sensor has no matching data at the sample
    * ``sensors`` ``(S,)`` / ``groups`` ``(S,)`` -- column sensor names and their
      ``"geo"``/``"mw"`` group
    * ``filenames`` -- sensor name -> its ``str`` filename array
    * ``ref_filenames`` ``(R,)`` -- reference filenames
    * ``norm_stats`` -- per-sensor normalization statistics
    """

    def __init__(
        self,
        coords: np.ndarray,
        ref_file: np.ndarray,
        ref_time: np.ndarray,
        sensor_file: np.ndarray,
        sensors: np.ndarray,
        groups: np.ndarray,
        filenames: Dict[str, np.ndarray],
        ref_filenames: np.ndarray,
        norm_stats: Dict[str, Dict[str, np.ndarray]],
    ):
        self.coords = coords
        self.ref_file = ref_file
        self.ref_time = ref_time
        self.sensor_file = sensor_file
        self.sensors = sensors
        self.groups = groups
        self.filenames = filenames
        self.ref_filenames = ref_filenames
        self.norm_stats = norm_stats
        self.geo_cols = np.where(groups == "geo")[0]
        self.mw_cols = np.where(groups == "mw")[0]

    def __len__(self) -> int:
        return self.coords.shape[0]

    def save(self, path: Path) -> None:
        arrays = {
            "coords": self.coords,
            "ref_file": self.ref_file,
            "ref_time": self.ref_time,
            "sensor_file": self.sensor_file,
            "sensors": self.sensors,
            "groups": self.groups,
            "ref_filenames": self.ref_filenames,
        }
        for sensor in self.sensors.tolist():
            arrays[f"fn__{sensor}"] = self.filenames[sensor]
            stats = self.norm_stats.get(sensor)
            if stats is not None:
                arrays[f"nmin__{sensor}"] = stats["min"]
                arrays[f"nmax__{sensor}"] = stats["max"]
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path: Path) -> "_SampleIndex":
        with np.load(str(path), allow_pickle=False) as cache:
            sensors = cache["sensors"]
            filenames, norm_stats = {}, {}
            for sensor in sensors.tolist():
                filenames[sensor] = cache[f"fn__{sensor}"]
                if f"nmin__{sensor}" in cache:
                    norm_stats[sensor] = {
                        "min": cache[f"nmin__{sensor}"],
                        "max": cache[f"nmax__{sensor}"],
                    }
            return cls(
                coords=cache["coords"],
                ref_file=cache["ref_file"],
                ref_time=cache["ref_time"],
                sensor_file=cache["sensor_file"],
                sensors=sensors,
                groups=cache["groups"],
                filenames=filenames,
                ref_filenames=cache["ref_filenames"],
                norm_stats=norm_stats,
            )


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
        position_jitter: int = REF_CELL,
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
    def _list_stores(self, name: str) -> List[Path]:
        """
        List a dataset's ``.zarr`` stores by globbing its sub-directory.

        This is only used when an index has to be (re)computed; on the cached
        path the store filenames are taken from ``index_<name>.nc`` instead,
        avoiding the slow directory listing over the network filesystem.
        """
        return sorted((self.path / name).glob("*.zarr"))

    def _load_indices(self, sensors, compute) -> Dict[str, xr.Dataset]:
        """
        Load (or compute and cache) the per-sensor index of each named dataset.

        Returns a mapping of name to its index dataset, skipping names with no
        stores. The cached ``index_<name>.nc`` already records the store
        filenames, so the (slow) directory listing is only performed for a sensor
        whose index is missing or predates the per-channel ``obs_min``/
        ``obs_max`` statistics.
        """
        indices = {}
        for name in sensors:
            meta = self._read_index(name)
            if meta is None or "obs_min" not in meta:
                meta = compute(name, self._list_stores(name))
                self._write_index(name, meta)
            if meta.sizes["samples"] > 0:
                indices[name] = meta
        return indices

    @cached_property
    def geo_meta(self) -> Dict[str, xr.Dataset]:
        """
        Per-sensor availability and acquisition time of the geostationary inputs.

        A mapping of sensor name to its own :class:`xarray.Dataset` (kept
        separate, not concatenated across sensors). Each dataset holds
        ``availability`` (``(samples, lat_cell, lon_cell)`` boolean), the scalar
        ``time`` (``(samples,)`` ``datetime64[ns]``), the per-channel
        ``obs_min``/``obs_max`` statistics and a ``filename`` coordinate (the
        full store path is built on the fly with :meth:`_store_path`). The
        per-sensor metadata is cached to ``index_<sensor>.nc`` and loaded from
        there on subsequent runs (see :meth:`recompute_indices` to refresh).
        """
        return self._load_indices(self.input_sensors, self._compute_geo_meta)

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
        Availability and per-cell scan time of the reference stores.

        A single :class:`xarray.Dataset` (the reference is one dataset) holding
        ``availability`` (``(samples, lat_cell, lon_cell)`` boolean), ``time``
        (``(samples, lat_cell, lon_cell)`` ``datetime64[ns]``, ``NaT`` where no
        data was observed) and a ``filename`` coordinate. Cached to
        ``index_<reference_name>.nc`` (see :meth:`recompute_indices`).
        """
        name = self.reference_name
        meta = self._read_index(name)
        if meta is None:
            files = self._list_stores(name)
            if not files:
                raise FileNotFoundError(
                    f"No reference '.zarr' stores found in '{self.path / name}'."
                )
            meta = self._compute_reference_meta(name, files)
            self._write_index(name, meta)
        if meta.sizes["samples"] == 0:
            raise FileNotFoundError("No reference '.zarr' stores found.")
        return meta

    @cached_property
    def microwave_meta(self) -> Dict[str, xr.Dataset]:
        """
        Per-sensor availability and per-cell scan time of the microwave inputs.

        Like :attr:`geo_meta`, but each per-sensor dataset has a per-cell ``time``
        (``(samples, lat_cell, lon_cell)``, ``NaT`` where no data was observed)
        on the reference grid. Empty if there are no microwave stores.
        """
        return self._load_indices(
            self.microwave_sensors, self._compute_reference_meta
        )

    @property
    def normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Per-sensor input normalization statistics.

        A mapping of input sensor name to ``{"min": ..., "max": ...}``, each a
        per-channel array giving the minimum and maximum observed value across
        all of the sensor's stores (in the sensor's stored channel order). Used
        by :meth:`__getitem__` to scale the inputs to ``[0, 1]`` when
        ``normalize`` is enabled. Computed once with the sample index and stored
        in its cache.
        """
        return self.samples.norm_stats

    def _compute_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Per-channel min/max across each sensor's stores (built with samples)."""
        stats = {}
        for sensor, idx in {**self.geo_meta, **self.microwave_meta}.items():
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

    @staticmethod
    def _as_str(value) -> str:
        """Coerce a coordinate value to a plain ``str`` (decoding bytes)."""
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    def _store_path(self, name, filename) -> Path:
        """Reconstruct a store path from its sensor name and filename."""
        return self.path / self._as_str(name) / self._as_str(filename)

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

        This discards the ``index_<name>.nc`` files and the cached sample index so
        that the next access re-reads the stores and rebuilds them.
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
        sample_cache = self._samples_cache_path
        if sample_cache.exists():
            sample_cache.unlink()
            LOGGER.info("Removed cached sample index '%s'.", sample_cache)
        for attr in (
            "geo_meta", "microwave_meta", "reference_meta",
            "samples", "_reference_grid",
        ):
            self.__dict__.pop(attr, None)
        # Trigger recomputation (and re-caching) of the sample index.
        self.samples

    @cached_property
    def _reference_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """The (latitude, longitude) arrays of the global reference grid."""
        first = self._store_path(
            self.reference_name, self.samples.ref_filenames[0]
        )
        store = zarr.open_group(str(first), mode="r")
        return (
            np.asarray(store["latitude"][:]).astype(np.float32),
            np.asarray(store["longitude"][:]).astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Sample enumeration
    # ------------------------------------------------------------------
    @cached_property
    def _samples_cache_path(self) -> Path:
        """Path of the on-disk sample-index cache (keyed by the relevant config)."""
        key = "|".join(
            [
                "v2",  # cache format version (bump to invalidate old caches)
                ",".join(self.input_sensors),
                ",".join(self.microwave_sensors),
                self.reference_name,
                str(int(self.time_window.astype("int64"))),
                str(self.require_both_inputs),
            ]
        )
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.path / f"samples_{digest}.npz"

    @cached_property
    def samples(self) -> _SampleIndex:
        """
        Compact index of all training samples (see :class:`_SampleIndex`).

        One sample per reduced-resolution reference cell that has valid reference
        data and at least one matching input (or both if ``require_both_inputs``).
        Built once from the metadata indices and cached to ``samples_<hash>.npz``;
        subsequent constructions and ``DataLoader`` workers load that compact
        cache directly, without re-reading the large per-sensor index files.
        """
        path = self._samples_cache_path
        if path.exists():
            LOGGER.info("Loading cached sample index from '%s'.", path)
            return _SampleIndex.load(path)
        samples = self._build_samples()
        samples.save(path)
        LOGGER.info("Wrote sample index cache to '%s'.", path)
        return samples

    def _build_samples(self) -> _SampleIndex:
        """Enumerate samples from the metadata indices (the slow, one-off path)."""
        window = self.time_window

        # Per-sensor input arrays (geo and microwave), with a per-file time span
        # used to pre-filter candidates.
        inputs = {
            sensor: self._sensor_arrays(meta)
            for sensor, meta in {**self.geo_meta, **self.microwave_meta}.items()
        }
        sensors = list(inputs)
        n_sensors = len(sensors)
        groups = np.array([self._sensor_group[s] for s in sensors])
        geo_cols = list(np.where(groups == "geo")[0])
        mw_cols = list(np.where(groups == "mw")[0])

        ref = self.reference_meta
        ref_avail = ref["availability"].values.astype(bool)
        ref_time = ref["time"].values

        coords_blocks, ref_blocks, file_blocks, time_blocks = [], [], [], []
        for ref_idx in tqdm(
            range(ref_avail.shape[0]), desc="Building samples", unit="granule"
        ):
            r_avail = ref_avail[ref_idx]
            r_time = ref_time[ref_idx]
            valid = r_time[r_avail & ~np.isnat(r_time)]
            if valid.size == 0:
                continue
            tmin, tmax = valid.min(), valid.max()

            # For each sensor, pick the file that best matches this granule and
            # keep its per-cell joint availability (valid in both, in time).
            masks: Dict[int, np.ndarray] = {}
            best_k: Dict[int, int] = {}
            for col, sensor in enumerate(sensors):
                info = inputs[sensor]
                candidates = np.where(
                    (info["tmax"] >= tmin - window) & (info["tmin"] <= tmax + window)
                )[0]
                best_matched, best_idx, best_score = None, -1, 0
                for k in candidates:
                    matched = (
                        r_avail
                        & info["availability"][k]
                        & (np.abs(r_time - info["time"][k]) <= window)
                    )
                    score = matched.sum()
                    if score > best_score:
                        best_matched, best_idx, best_score = matched, k, score
                if best_matched is not None:
                    masks[col] = best_matched
                    best_k[col] = best_idx
            if not masks:
                continue

            # Every reference cell with at least one matching input is a sample.
            any_matched = np.zeros((N_CELLS_LAT, N_CELLS_LON), dtype=bool)
            for matched in masks.values():
                any_matched |= matched
            cells = np.argwhere(any_matched)
            if cells.size == 0:
                continue
            rows, cols = cells[:, 0], cells[:, 1]
            sensor_file = np.full((cells.shape[0], n_sensors), -1, dtype=np.int32)
            for col, matched in masks.items():
                sensor_file[matched[rows, cols], col] = best_k[col]

            if self.require_both_inputs:
                has_geo = (sensor_file[:, geo_cols] >= 0).any(axis=1)
                has_mw = (sensor_file[:, mw_cols] >= 0).any(axis=1)
                keep = has_geo & has_mw
                if not keep.any():
                    continue
                cells, sensor_file = cells[keep], sensor_file[keep]

            coords_blocks.append(
                (cells * OBS_CELL + OBS_CELL // 2).astype(np.int32)
            )
            ref_blocks.append(np.full(cells.shape[0], ref_idx, dtype=np.int32))
            file_blocks.append(sensor_file)
            # The reference cell's actual scan time (not the granule filename).
            time_blocks.append(r_time[cells[:, 0], cells[:, 1]])

        if coords_blocks:
            coords = np.concatenate(coords_blocks)
            ref_file = np.concatenate(ref_blocks)
            sensor_file = np.concatenate(file_blocks)
            ref_time = np.concatenate(time_blocks)
        else:
            coords = np.empty((0, 2), dtype=np.int32)
            ref_file = np.empty((0,), dtype=np.int32)
            sensor_file = np.empty((0, n_sensors), dtype=np.int32)
            ref_time = np.empty((0,), dtype="datetime64[ns]")

        samples = _SampleIndex(
            coords=coords,
            ref_file=ref_file,
            ref_time=ref_time,
            sensor_file=sensor_file,
            sensors=np.array(sensors),
            groups=groups,
            filenames={
                s: np.asarray(inputs[s]["filename"]).astype(str) for s in sensors
            },
            ref_filenames=np.asarray(ref["filename"].values).astype(str),
            norm_stats=self._compute_normalization_stats(),
        )
        LOGGER.info(
            "Built %d samples from %d reference and %d input stores.",
            len(samples),
            ref_avail.shape[0],
            sum(inputs[s]["filename"].shape[0] for s in sensors),
        )
        # The heavy metadata is no longer needed once the compact index is built.
        for attr in ("geo_meta", "microwave_meta", "reference_meta"):
            self.__dict__.pop(attr, None)
        return samples

    def _sensor_arrays(self, meta: xr.Dataset) -> Dict[str, np.ndarray]:
        """
        Convert one sensor's metadata into matching arrays.

        Returns the ``availability`` ``(n, 90, 180)`` and ``time`` (``(n,)`` for
        geo or ``(n, 90, 180)`` for microwave) arrays, the ``filename`` array
        (``str``; the full path is built on the fly via :meth:`_store_path`), and
        the per-file time span (``tmin``/``tmax``) used to pre-filter candidates.
        """
        availability = meta["availability"].values.astype(bool)
        time = meta["time"].values
        filename = meta["filename"].values
        if time.ndim == 1:
            # Geostationary: a single acquisition time per file.
            tmin = tmax = time
        else:
            # Microwave/reference: a per-cell scan time, reduced per file.
            tmin = np.full(len(time), np.datetime64("NaT"), dtype="datetime64[ns]")
            tmax = tmin.copy()
            for i in range(len(time)):
                valid = time[i][availability[i] & ~np.isnat(time[i])]
                if valid.size:
                    tmin[i], tmax[i] = valid.min(), valid.max()
        return {
            "availability": availability,
            "time": time,
            "filename": filename,
            "tmin": tmin,
            "tmax": tmax,
        }

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

    @staticmethod
    def _scale_to_unit(array: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Scale ``(channel, ...)`` observations to ``[-1, 1]`` using min/max stats."""
        lower = stats["min"][:, None, None]
        rng = stats["max"][:, None, None] - lower
        # Leave channels without valid statistics (e.g. never-observed) as is.
        valid = np.isfinite(lower) & np.isfinite(rng) & (rng > 0)
        lower = np.where(valid, lower, 0.0)
        rng = np.where(valid, rng, 1.0)
        return -1.0 + 2.0 * ((array - lower) / rng).astype(np.float32)

    def _normalize(self, array: np.ndarray, sensor: str) -> np.ndarray:
        """Scale a sensor's ``(channel, ...)`` observations to ``[-1, 1]``."""
        stats = self.normalization_stats.get(sensor)
        if stats is None:
            return array
        return self._scale_to_unit(array, stats)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, object], torch.Tensor]:
        si = self.samples
        row_c, col_c = int(si.coords[index, 0]), int(si.coords[index, 1])
        latitude, longitude = self._reference_grid

        # Randomly jitter the tile center (kept aligned between the input and
        # reference grids), then derive the crop windows.
        obs_size = self.tile_size * RESOLUTION_RATIO
        obs_rows, obs_cols = N_CELLS_LAT * OBS_CELL, N_CELLS_LON * OBS_CELL
        if self.position_jitter > 0:
            row_c += RESOLUTION_RATIO * np.random.randint(
                -self.position_jitter, self.position_jitter + 1
            )
            col_c += RESOLUTION_RATIO * np.random.randint(
                -self.position_jitter, self.position_jitter + 1
            )
        obs_r0 = min(max(row_c - obs_size // 2, 0), obs_rows - obs_size)
        obs_c0 = min(max(col_c - obs_size // 2, 0), obs_cols - obs_size)
        ref_r0, ref_c0 = obs_r0 // RESOLUTION_RATIO, obs_c0 // RESOLUTION_RATIO

        # Reference surface precipitation (the target). The jittered crop can miss
        # the (sparse) valid reference data, so fall back to a random sample if
        # the target has no valid pixel.
        reference = self._store_path(
            self.reference_name, si.ref_filenames[si.ref_file[index]]
        )
        surface_precip = self._load_reference(
            reference, ref_r0, ref_c0, self.tile_size
        )
        if not np.isfinite(surface_precip).any():
            return self[np.random.randint(len(self))]

        # For each input group, randomly choose one of the available sensors and
        # load its observations. ``"geo"`` is high resolution, ``"mw"`` is on the
        # reference grid. When slotting, the observations are mapped onto the
        # group's common slots; otherwise the raw tensor is returned under the
        # sensor name.
        sensor_file = si.sensor_file[index]
        obs = {}
        for group, columns, loader, (r0, c0, size) in (
            ("geo", si.geo_cols, self._load_obs, (obs_r0, obs_c0, obs_size)),
            ("mw", si.mw_cols, self._load_mw_obs, (ref_r0, ref_c0, self.tile_size)),
        ):
            present = columns[sensor_file[columns] >= 0]
            if present.size == 0:
                continue
            col = int(np.random.choice(present))
            sensor = str(si.sensors[col])
            filename = si.filenames[sensor][sensor_file[col]]
            array = loader(self._store_path(sensor, filename), r0, c0, size)
            if self.normalize:
                array = self._normalize(array, sensor)
            if self.slot_channels:
                obs[group] = torch.from_numpy(slot_observations(array, sensor))
            else:
                obs[sensor] = torch.from_numpy(array)

        inputs = {
            **obs,
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
    # Extraction
    # ------------------------------------------------------------------
    def _load_slotted(
        self, si, index, col, loader, r0: int, c0: int, size: int
    ) -> np.ndarray:
        """Load, optionally normalize and slot one sensor's observation crop."""
        sensor = str(si.sensors[col])
        filename = si.filenames[sensor][si.sensor_file[index, col]]
        array = loader(self._store_path(sensor, filename), r0, c0, size)
        if self.normalize:
            array = self._normalize(array, sensor)
        return slot_observations(array, sensor)

    @staticmethod
    def _filename_time(filename) -> np.datetime64:
        """Timestamp parsed from a store filename (``NaT`` if absent)."""
        stamp = _parse_timestamp(Path(str(filename)))
        return stamp if stamp is not None else np.datetime64("NaT")

    def _scene_windows(self, si, index):
        """Crop windows for a sample, with the random ``position_jitter`` applied."""
        obs_size = self.tile_size * RESOLUTION_RATIO
        obs_rows, obs_cols = N_CELLS_LAT * OBS_CELL, N_CELLS_LON * OBS_CELL
        row_c, col_c = int(si.coords[index, 0]), int(si.coords[index, 1])
        if self.position_jitter > 0:
            row_c += RESOLUTION_RATIO * np.random.randint(
                -self.position_jitter, self.position_jitter + 1
            )
            col_c += RESOLUTION_RATIO * np.random.randint(
                -self.position_jitter, self.position_jitter + 1
            )
        obs_r0 = min(max(row_c - obs_size // 2, 0), obs_rows - obs_size)
        obs_c0 = min(max(col_c - obs_size // 2, 0), obs_cols - obs_size)
        ref_r0, ref_c0 = obs_r0 // RESOLUTION_RATIO, obs_c0 // RESOLUTION_RATIO
        return row_c, col_c, obs_r0, obs_c0, ref_r0, ref_c0

    @staticmethod
    def _subsample(scenes, sampling_rate):
        """Randomly keep a fraction ``sampling_rate`` of the enumerated scenes."""
        if sampling_rate >= 1.0:
            return scenes
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1].")
        keep = np.random.random(len(scenes)) < sampling_rate
        return [scene for scene, take in zip(scenes, keep) if take]

    def _enumerate_scenes(self, start_time, end_time):
        """
        Select samples within ``[start_time, end_time]`` and expand them into
        ``(sample index, geo column, mw column)`` scenes -- one per matching
        ``(geo, mw)`` sensor combination (``-1`` where a group is absent). Returns
        the scene list and the per-sample reference times.
        """
        si = self.samples
        # The reference cell's actual scan time (per-cell, not the granule
        # filename time which can be ~90 min off within an orbit).
        sample_times = si.ref_time
        keep = np.ones(len(si), dtype=bool)
        if start_time is not None:
            keep &= sample_times >= np.datetime64(start_time)
        if end_time is not None:
            keep &= sample_times <= np.datetime64(end_time)

        scenes = []
        for index in np.where(keep)[0]:
            sensor_file = si.sensor_file[index]
            geo = [c for c in si.geo_cols if sensor_file[c] >= 0] or [-1]
            mw = [c for c in si.mw_cols if sensor_file[c] >= 0] or [-1]
            for geo_col in geo:
                for mw_col in mw:
                    if geo_col < 0 and mw_col < 0:
                        continue
                    scenes.append((int(index), int(geo_col), int(mw_col)))
        return scenes, sample_times

    def _mw_observations_by_cell(self, cells, sensor_column):
        """
        Microwave observations (from any sensor) covering each requested cell.

        Returns ``{(ci, cj): [(scan_time, sensor_column, filename), ...]}``, built
        from the microwave metadata indices -- the per-cell scan time is needed to
        place each observation in the right temporal slot (the granule's filename
        time is not the cell's observation time for a swath). This re-reads the
        (large) microwave index files, so it is only used by the extraction.
        """
        by_cell = {cell: [] for cell in cells}
        for sensor, meta in self.microwave_meta.items():
            column = sensor_column.get(sensor)
            if column is None:
                continue
            availability = meta["availability"].values.astype(bool)
            scan_time = meta["time"].values
            filenames = np.asarray(meta["filename"].values).astype(str)
            for ci, cj in cells:
                times = scan_time[:, ci, cj]
                valid = availability[:, ci, cj] & ~np.isnat(times)
                for f in np.where(valid)[0]:
                    by_cell[(ci, cj)].append((times[f], column, filenames[f]))
        return by_cell

    def _write_scenes(self, output_path, scenes, sample_times) -> Path:
        """Write single-observation scenes to a standalone ``.zarr`` store."""
        from numcodecs.zarr3 import Blosc

        output_path = Path(output_path)
        si = self.samples
        obs_size = self.tile_size * RESOLUTION_RATIO
        n_scenes = len(scenes)
        LOGGER.info("Extracting %d scenes to '%s'.", n_scenes, output_path)

        compressor = Blosc(cname="zstd", clevel=4)
        store = zarr.open_group(str(output_path), mode="w")
        store.attrs["sensors"] = [str(s) for s in si.sensors]
        store.attrs["groups"] = [str(g) for g in si.groups]
        store.attrs["tile_size"] = self.tile_size
        store.attrs["resolution_ratio"] = RESOLUTION_RATIO
        store.attrs["normalized"] = self.normalize
        store.attrs["time_units"] = "nanoseconds since 1970-01-01"

        store.create_array(
            "geo", shape=(n_scenes, N_SLOTS, obs_size, obs_size),
            chunks=(1, N_SLOTS, obs_size, obs_size), dtype=np.float32,
            fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "geo_channel", "geo_y", "geo_x"),
        )
        store.create_array(
            "mw", shape=(n_scenes, N_MW_SLOTS, self.tile_size, self.tile_size),
            chunks=(1, N_MW_SLOTS, self.tile_size, self.tile_size), dtype=np.float32,
            fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "mw_channel", "y", "x"),
        )
        store.create_array(
            "surface_precip", shape=(n_scenes, self.tile_size, self.tile_size),
            chunks=(1, self.tile_size, self.tile_size), dtype=np.float32,
            fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "y", "x"),
        )
        store.create_array(
            "coordinates", shape=(n_scenes, 2), dtype=np.int32,
            dimension_names=("scene", "row_col"),
        )
        store.create_array(
            "time", shape=(n_scenes,), dtype=np.int64, dimension_names=("scene",),
        )
        for name in ("geo_sensor", "mw_sensor"):
            store.create_array(
                name, shape=(n_scenes,), dtype=np.int16, fill_value=-1,
                dimension_names=("scene",),
            )

        for i, (index, geo_col, mw_col) in enumerate(
            tqdm(scenes, desc="Extracting scenes", unit="scene")
        ):
            row_c, col_c, obs_r0, obs_c0, ref_r0, ref_c0 = self._scene_windows(
                si, index
            )
            store["surface_precip"][i] = self._load_reference(
                self._store_path(
                    self.reference_name, si.ref_filenames[si.ref_file[index]]
                ),
                ref_r0, ref_c0, self.tile_size,
            )
            if geo_col >= 0:
                store["geo"][i] = self._load_slotted(
                    si, index, geo_col, self._load_obs, obs_r0, obs_c0, obs_size
                )
                store["geo_sensor"][i] = geo_col
            if mw_col >= 0:
                store["mw"][i] = self._load_slotted(
                    si, index, mw_col, self._load_mw_obs,
                    ref_r0, ref_c0, self.tile_size,
                )
                store["mw_sensor"][i] = mw_col
            store["coordinates"][i] = (row_c, col_c)
            store["time"][i] = sample_times[index].astype("int64")

        LOGGER.info("Wrote %d scenes to '%s'.", n_scenes, output_path)
        return output_path

    def extract_samples(
        self,
        output_path: Union[str, Path],
        sampling_rate: float = 1.0,
        start_time: Optional[Union[str, np.datetime64]] = None,
        end_time: Optional[Union[str, np.datetime64]] = None,
    ) -> Path:
        """
        Extract fixed-size slotted scenes into a standalone ``.zarr`` store.

        For every sample whose reference time falls within
        ``[start_time, end_time]`` (both optional), one scene is written per
        matching input observation: if a sample has several matching
        geostationary and/or microwave sensors, every combination is written as
        its own scene. Each scene holds the slotted ``geo`` and ``mw`` inputs and
        the ``surface_precip`` target at a fixed size, so the resulting store is a
        self-contained training set that no longer needs the source stores or the
        metadata indices. Observations are normalized according to ``normalize``
        and the tile center is randomly jittered by ``position_jitter`` (the
        stored ``coordinates`` are the jittered center).

        Args:
            output_path: Path of the output ``.zarr`` store.
            sampling_rate: Fraction in ``(0, 1]`` of the enumerated scenes to
                extract, each kept independently at random (default 1.0 = all).
            start_time: Optional lower bound on the sample's reference time
                (anything ``numpy.datetime64`` accepts).
            end_time: Optional upper bound on the sample's reference time.

        Returns:
            The output path.
        """
        scenes, sample_times = self._enumerate_scenes(start_time, end_time)
        scenes = self._subsample(scenes, sampling_rate)
        return self._write_scenes(output_path, scenes, sample_times)

    def extract_temporal_samples(
        self,
        output_path: Union[str, Path],
        n_steps: int,
        step: np.timedelta64 = np.timedelta64(20, "m"),
        tolerance: Optional[np.timedelta64] = None,
        sampling_rate: float = 1.0,
        require_microwave: bool = True,
        start_time: Optional[Union[str, np.datetime64]] = None,
        end_time: Optional[Union[str, np.datetime64]] = None,
    ) -> Path:
        """
        Extract scenes with a temporal sequence of inputs.

        Each scene is built on a regular time grid of ``n_steps + 1`` slots,
        ``step`` apart (the geostationary cadence, 20 minutes by default), ending
        at the matched geostationary observation time. One scene is written per
        geostationary sensor available at a sample.

        For every slot:

        * ``geo`` is the chosen geostationary sensor's store closest to the slot
          time (within ``tolerance``);
        * ``mw`` is the closest microwave observation *from any sensor* that
          covers the location at the slot time (within ``tolerance``) -- so the
          microwave slots are filled across the whole constellation rather than
          from a single (polar-orbiting) sensor.

        Frames are stacked along axis 1, oldest first with the matched frame last;
        slots with no observation are left as ``NaN``. ``geo`` has shape
        ``(n_scenes, N_SLOTS, n_steps + 1, H, W)`` and ``mw``
        ``(n_scenes, N_MW_SLOTS, n_steps + 1, h, w)``; ``mw_sensor`` is
        ``(n_scenes, n_steps + 1)`` (the sensor used in each slot).

        Args:
            output_path: Path of the output ``.zarr`` store.
            n_steps: Number of previous time steps (the grid has ``n_steps + 1``
                slots).
            step: Time between slots (``numpy.timedelta64``; default 20 minutes).
            tolerance: Maximum offset between an observation and a slot time for it
                to fill the slot (default ``step / 2``).
            sampling_rate: Fraction in ``(0, 1]`` of the enumerated scenes to
                extract, each kept independently at random (default 1.0 = all).
            require_microwave: If ``True`` (the default), only keep scenes that
                have a microwave observation in at least one slot.
            start_time: Optional lower bound on the sample's reference time.
            end_time: Optional upper bound on the sample's reference time.

        Returns:
            The output path.
        """
        from numcodecs.zarr3 import Blosc

        output_path = Path(output_path)
        n_steps = int(n_steps)
        frames = n_steps + 1
        step = np.timedelta64(step)
        tolerance = np.timedelta64(step / 2 if tolerance is None else tolerance)

        si = self.samples
        obs_size = self.tile_size * RESOLUTION_RATIO
        sensor_column = {str(s): i for i, s in enumerate(si.sensors)}

        # Filter by reference time and enumerate one scene per available geo sensor.
        # ``ref_time`` is the reference cell's actual (per-cell) scan time.
        sample_times = si.ref_time
        keep = np.ones(len(si), dtype=bool)
        if start_time is not None:
            keep &= sample_times >= np.datetime64(start_time)
        if end_time is not None:
            keep &= sample_times <= np.datetime64(end_time)
        scenes = [
            (int(index), int(col))
            for index in np.where(keep)[0]
            for col in si.geo_cols
            if si.sensor_file[index, col] >= 0
        ]
        scenes = self._subsample(scenes, sampling_rate)

        # Acquisition time of every geo store (geostationary obs are instantaneous,
        # so the filename timestamp is the frame time).
        geo_times = {
            str(si.sensors[col]): np.array(
                [self._filename_time(fn) for fn in si.filenames[str(si.sensors[col])]],
                dtype="datetime64[ns]",
            )
            for col in si.geo_cols
        }
        # Per-cell microwave observations from any sensor, to fill the mw slots.
        cells = {
            (int(si.coords[idx, 0]) // OBS_CELL, int(si.coords[idx, 1]) // OBS_CELL)
            for idx, _ in scenes
        }
        mw_by_cell = self._mw_observations_by_cell(cells, sensor_column)

        # For each scene compute its time grid (anchored at the matched geo
        # observation) and the microwave observation assigned to each slot (the
        # nearest from any sensor within ``tolerance``). Scenes with no microwave
        # observation in any slot are dropped when ``require_microwave`` is set.
        records = []  # (index, geo_col, slot_times, mw_assignment)
        for index, geo_col in scenes:
            geo_sensor = str(si.sensors[geo_col])
            anchor = geo_times[geo_sensor][int(si.sensor_file[index, geo_col])]
            slot_times = [anchor - (n_steps - s) * step for s in range(frames)]
            cell = (
                int(si.coords[index, 0]) // OBS_CELL,
                int(si.coords[index, 1]) // OBS_CELL,
            )
            candidates = mw_by_cell.get(cell, [])
            mw_assignment = []
            for slot in slot_times:
                best_column, best_filename, best_offset = -1, None, tolerance
                for scan_time, column, filename in candidates:
                    offset = abs(scan_time - slot)
                    if offset <= best_offset:
                        best_column, best_filename, best_offset = (
                            column, filename, offset
                        )
                mw_assignment.append((best_column, best_filename))
            if require_microwave and not any(
                filename is not None for _, filename in mw_assignment
            ):
                continue
            records.append((index, geo_col, slot_times, mw_assignment))
        n_scenes = len(records)

        LOGGER.info(
            "Extracting %d temporal scenes (%d steps) to '%s'.",
            n_scenes, n_steps, output_path,
        )
        compressor = Blosc(cname="zstd", clevel=4)
        store = zarr.open_group(str(output_path), mode="w")
        store.attrs["sensors"] = [str(s) for s in si.sensors]
        store.attrs["groups"] = [str(g) for g in si.groups]
        store.attrs["tile_size"] = self.tile_size
        store.attrs["resolution_ratio"] = RESOLUTION_RATIO
        store.attrs["normalized"] = self.normalize
        store.attrs["time_units"] = "nanoseconds since 1970-01-01"
        store.attrs["n_steps"] = n_steps
        store.attrs["step_minutes"] = float(step / np.timedelta64(1, "m"))

        store.create_array(
            "geo", shape=(n_scenes, N_SLOTS, frames, obs_size, obs_size),
            chunks=(1, N_SLOTS, frames, obs_size, obs_size), dtype=np.float32,
            fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "geo_channel", "step", "geo_y", "geo_x"),
        )
        store.create_array(
            "mw", shape=(n_scenes, N_MW_SLOTS, frames, self.tile_size, self.tile_size),
            chunks=(1, N_MW_SLOTS, frames, self.tile_size, self.tile_size),
            dtype=np.float32, fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "mw_channel", "step", "y", "x"),
        )
        store.create_array(
            "surface_precip", shape=(n_scenes, self.tile_size, self.tile_size),
            chunks=(1, self.tile_size, self.tile_size), dtype=np.float32,
            fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "y", "x"),
        )
        store.create_array(
            "coordinates", shape=(n_scenes, 2), dtype=np.int32,
            dimension_names=("scene", "row_col"),
        )
        store.create_array(
            "time", shape=(n_scenes,), dtype=np.int64, dimension_names=("scene",),
        )
        store.create_array(
            "step_time", shape=(n_scenes, frames), dtype=np.int64,
            dimension_names=("scene", "step"),
        )
        store.create_array(
            "geo_sensor", shape=(n_scenes,), dtype=np.int16, fill_value=-1,
            dimension_names=("scene",),
        )
        store.create_array(
            "mw_sensor", shape=(n_scenes, frames), dtype=np.int16, fill_value=-1,
            dimension_names=("scene", "step"),
        )

        for i, (index, geo_col, slot_times, mw_assignment) in enumerate(
            tqdm(records, desc="Extracting scenes", unit="scene")
        ):
            row_c, col_c, obs_r0, obs_c0, ref_r0, ref_c0 = self._scene_windows(
                si, index
            )
            store["surface_precip"][i] = self._load_reference(
                self._store_path(
                    self.reference_name, si.ref_filenames[si.ref_file[index]]
                ),
                ref_r0, ref_c0, self.tile_size,
            )
            store["coordinates"][i] = (row_c, col_c)
            store["time"][i] = sample_times[index].astype("int64")
            store["geo_sensor"][i] = geo_col
            store["step_time"][i] = np.array(
                [t.astype("int64") for t in slot_times], dtype=np.int64
            )

            # geo: nearest store of the chosen sensor to each slot time.
            geo_sensor = str(si.sensors[geo_col])
            gt = geo_times[geo_sensor]
            geo_scene = np.full(
                (N_SLOTS, frames, obs_size, obs_size), np.nan, dtype=np.float32
            )
            for s, slot in enumerate(slot_times):
                j = int(np.argmin(np.abs(gt - slot)))
                if np.abs(gt[j] - slot) <= tolerance:
                    array = self._load_obs(
                        self._store_path(geo_sensor, si.filenames[geo_sensor][j]),
                        obs_r0, obs_c0, obs_size,
                    )
                    if self.normalize:
                        array = self._normalize(array, geo_sensor)
                    geo_scene[:, s] = slot_observations(array, geo_sensor)
            store["geo"][i] = geo_scene

            # mw: the pre-assigned microwave observation (any sensor) per slot.
            mw_scene = np.full(
                (N_MW_SLOTS, frames, self.tile_size, self.tile_size),
                np.nan, dtype=np.float32,
            )
            mw_sensor = np.full(frames, -1, dtype=np.int16)
            for s, (column, filename) in enumerate(mw_assignment):
                if filename is not None:
                    sensor = str(si.sensors[column])
                    array = self._load_mw_obs(
                        self._store_path(sensor, filename),
                        ref_r0, ref_c0, self.tile_size,
                    )
                    if self.normalize:
                        array = self._normalize(array, sensor)
                    mw_scene[:, s] = slot_observations(array, sensor)
                    mw_sensor[s] = column
            store["mw"][i] = mw_scene
            store["mw_sensor"][i] = mw_sensor

        LOGGER.info("Wrote %d temporal scenes to '%s'.", n_scenes, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @cached_property
    def _grid_latlon(self) -> Tuple[np.ndarray, np.ndarray]:
        """The (latitude, longitude) arrays of the reference grid (from a store)."""
        first = self.reference_files[self.reference_name][0]
        store = zarr.open_group(str(first), mode="r")
        return (
            np.asarray(store["latitude"][:]).astype(np.float32),
            np.asarray(store["longitude"][:]).astype(np.float32),
        )

    @staticmethod
    def _coord_window(axis: np.ndarray, lo: float, hi: float) -> Tuple[int, int]:
        """Half-open index range ``[start, end)`` of ``axis`` covering ``[lo, hi]``."""
        within = np.where((axis >= lo) & (axis <= hi))[0]
        if within.size == 0:
            return 0, 0
        return int(within.min()), int(within.max()) + 1

    @staticmethod
    def _tile_starts(begin: int, end: int, tile: int, stride: int, limit: int) -> List[int]:
        """Tile start indices covering ``[begin, end)`` with windows inside ``limit``."""
        if tile >= limit:
            return [0]
        starts = list(range(begin, end, stride)) or [begin]
        if starts[-1] + tile < end:  # make sure the far edge is covered
            starts.append(end - tile)
        return sorted({min(max(s, 0), limit - tile) for s in starts})

    @staticmethod
    def _accumulate_tile(
        acc, wgt, pred, window, tr0, tc0, tile, r_start, c_start, r_end, c_end
    ) -> None:
        """Blend a tile's prediction into the region accumulators via ``window``."""
        rr0, rr1 = max(tr0, r_start), min(tr0 + tile, r_end)
        cc0, cc1 = max(tc0, c_start), min(tc0 + tile, c_end)
        if rr1 <= rr0 or cc1 <= cc0:
            return
        tile_rows, tile_cols = slice(rr0 - tr0, rr1 - tr0), slice(cc0 - tc0, cc1 - tc0)
        acc_rows = slice(rr0 - r_start, rr1 - r_start)
        acc_cols = slice(cc0 - c_start, cc1 - c_start)
        p = pred[tile_rows, tile_cols]
        w = window[tile_rows, tile_cols]
        valid = np.isfinite(p)
        acc[acc_rows, acc_cols] += np.where(valid, p * w, 0.0)
        wgt[acc_rows, acc_cols] += np.where(valid, w, 0.0)

    @staticmethod
    def _merge_aux_tile(
        age_acc, sensor_acc, tile_age, tile_sensor,
        tr0, tc0, tile, r_start, c_start, r_end, c_end,
    ) -> None:
        """
        Merge a tile's per-pixel age/sensor fields into the region accumulators.

        Unlike the (continuous) prediction, these fields are categorical, so
        overlapping tiles are resolved by keeping the most recent observation
        (smallest age) per pixel rather than by blending.
        """
        rr0, rr1 = max(tr0, r_start), min(tr0 + tile, r_end)
        cc0, cc1 = max(tc0, c_start), min(tc0 + tile, c_end)
        if rr1 <= rr0 or cc1 <= cc0:
            return
        tile_rows, tile_cols = slice(rr0 - tr0, rr1 - tr0), slice(cc0 - tc0, cc1 - tc0)
        acc_rows = slice(rr0 - r_start, rr1 - r_start)
        acc_cols = slice(cc0 - c_start, cc1 - c_start)
        sub_age = tile_age[tile_rows, tile_cols]
        sub_sensor = tile_sensor[tile_rows, tile_cols]
        current = age_acc[acc_rows, acc_cols]
        better = np.isfinite(sub_age) & (~np.isfinite(current) | (sub_age < current))
        age_acc[acc_rows, acc_cols][better] = sub_age[better]
        sensor_acc[acc_rows, acc_cols][better] = sub_sensor[better]

    def infer(
        self,
        model: "torch.nn.Module",
        target_time: Union[str, np.datetime64],
        bounds: Tuple[float, float, float, float],
        tile_size: Optional[int] = None,
        overlap: int = 0,
        n_steps: int = 1,
        step: np.timedelta64 = np.timedelta64(20, "m"),
        tolerance: Optional[np.timedelta64] = None,
        geo_only: bool = False,
        device: Optional[Union[str, "torch.device"]] = None,
    ) -> xr.Dataset:
        """
        Run tiled inference for a target time over a lon/lat bounding box.

        The bounding box (on the reference 0.05-degree grid) is covered with
        overlapping ``tile_size`` tiles, stepping by ``tile_size - overlap``. For
        every tile the geostationary and microwave observations available around
        ``target_time`` are loaded -- as a temporal stack of ``n_steps`` frames,
        ``step`` apart and ending at ``target_time`` -- slotted exactly as for
        training, and passed through ``model``. Per-tile point predictions (the
        posterior mean of quantile outputs) are blended back together with a Hann
        window and returned as a gridded field.

        Args:
            model: The trained model, taking the ``{"geo", "mw"}`` input dict.
            target_time: The time to run the retrieval for (the last frame).
            bounds: The bounding box as ``(lon_ll, lat_ll, lon_ur, lat_ur)`` --
                the lower-left and upper-right longitude/latitude corners.
            tile_size: Side length of the (square) reference tile in reference
                pixels. Defaults to the dataset's ``tile_size``.
            overlap: Overlap between neighbouring tiles in reference pixels.
            n_steps: Number of time steps (frames) to load. The last frame is at
                ``target_time`` and earlier frames are ``step`` apart before it;
                ``n_steps == 1`` gives a single (non-temporal) input.
            step: Time between frames (``numpy.timedelta64``; default 20 minutes).
            tolerance: Maximum offset between an observation and a frame time for
                it to fill the frame (default ``step / 2``).
            geo_only: If ``True``, run the retrieval on the geostationary
                observations alone: no microwave observations are loaded and the
                ``mw`` input stays all-``NaN`` (which the model treats as
                missing).
            device: Device to run the model on. Defaults to the model's device.

        Returns:
            An :class:`xarray.Dataset` with the retrieved ``surface_precip`` on the
            reference grid, with ``latitude``/``longitude`` coordinates spanning
            the bounding box and a scalar ``time`` coordinate. Two auxiliary
            per-pixel fields describe the microwave input: ``mw_age``, the age (in
            minutes before ``target_time``, based on the frame times) of the most
            recent microwave observation covering the pixel (``NaN`` where none),
            and ``mw_sensor``, the index into ``microwave_sensors`` (listed in the
            variable's ``sensors`` attribute) of the sensor that provided it
            (``-1`` where none). Where tiles overlap, the most recent observation
            wins.
        """
        tile = int(self.tile_size if tile_size is None else tile_size)
        overlap = int(overlap)
        n_steps = int(n_steps)
        if not 0 <= overlap < tile:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size.")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        step = np.timedelta64(step)
        tolerance = np.timedelta64(step / 2 if tolerance is None else tolerance)
        target = np.datetime64(target_time, "ns")
        temporal = n_steps > 1
        # Frame times: ``n_steps`` slots ``step`` apart, ending at ``target``.
        slot_times = [target - (n_steps - 1 - s) * step for s in range(n_steps)]

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        device = torch.device(device)
        model = model.to(device).eval()

        # Map the bounding box to a reference-grid window and tile it.
        latitude, longitude = self._grid_latlon
        lon_ll, lat_ll, lon_ur, lat_ur = bounds
        r_start, r_end = self._coord_window(latitude, lat_ll, lat_ur)
        c_start, c_end = self._coord_window(longitude, lon_ll, lon_ur)
        if r_end <= r_start or c_end <= c_start:
            raise ValueError("The bounding box does not intersect the reference grid.")
        ref_rows, ref_cols = latitude.shape[0], longitude.shape[0]
        obs_size = tile * RESOLUTION_RATIO
        stride = tile - overlap
        row_starts = self._tile_starts(r_start, r_end, tile, stride, ref_rows)
        col_starts = self._tile_starts(c_start, c_end, tile, stride, ref_cols)

        # Restrict the metadata to the stores relevant for the target time window,
        # so the per-tile search stays cheap.
        window_lo = min(slot_times) - tolerance
        window_hi = max(slot_times) + tolerance
        geo_pool = {}
        for sensor, meta in self.geo_meta.items():
            times = meta["time"].values
            sel = (times >= window_lo) & (times <= window_hi)
            if sel.any():
                geo_pool[sensor] = (
                    times[sel],
                    meta["filename"].values[sel],
                    meta["availability"].values.astype(bool)[sel],
                )
        # With ``geo_only`` the microwave pool stays empty, so no microwave
        # observations are loaded and the ``mw`` input remains all-NaN.
        mw_pool = {}
        if not geo_only:
            for sensor, meta in self.microwave_meta.items():
                times = meta["time"].values
                avail = meta["availability"].values.astype(bool)
                in_win = (
                    avail
                    & ~np.isnat(times)
                    & (times >= window_lo)
                    & (times <= window_hi)
                )
                sel = in_win.reshape(times.shape[0], -1).any(axis=1)
                if sel.any():
                    mw_pool[sensor] = (
                        times[sel], meta["filename"].values[sel], avail[sel]
                    )

        # Normalization statistics, computed from the metadata indices so that
        # inference does not need the (reference-granule) sample index.
        norm_stats = self._compute_normalization_stats() if self.normalize else {}

        # Hann blending window (with a floor so lone-tile edges keep full weight).
        taper = np.hanning(tile + 2)[1:-1].astype(np.float64)
        window = np.outer(taper, taper) + 1e-3

        acc = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float64)
        wgt = np.zeros_like(acc)
        # Auxiliary per-pixel fields: age (minutes before ``target_time``) and
        # sensor index of the most recent microwave observation covering a pixel.
        step_minutes = float(step / np.timedelta64(1, "m"))
        sensor_index = {s: i for i, s in enumerate(self.microwave_sensors)}
        age_acc = np.full(acc.shape, np.nan, dtype=np.float32)
        sensor_acc = np.full(acc.shape, -1, dtype=np.int16)

        for tr0 in tqdm(row_starts, desc="Inferring tiles", unit="row"):
            for tc0 in col_starts:
                geo_scene, mw_scene, mw_frame_sensors, has_valid = (
                    self._load_inference_tile(
                        tr0, tc0, tile, obs_size, slot_times, tolerance,
                        geo_pool, mw_pool, norm_stats,
                    )
                )
                if not has_valid:
                    continue

                # Per-pixel microwave age/sensor fields for this tile. Iterating
                # frames oldest-to-newest makes the most recent observation win.
                tile_age = np.full((tile, tile), np.nan, dtype=np.float32)
                tile_sensor = np.full((tile, tile), -1, dtype=np.int16)
                covered = np.isfinite(mw_scene).any(axis=0)  # (n_steps, tile, tile)
                for s in range(n_steps):
                    if mw_frame_sensors[s] is None:
                        continue
                    here = covered[s]
                    tile_age[here] = (n_steps - 1 - s) * step_minutes
                    tile_sensor[here] = sensor_index[mw_frame_sensors[s]]
                self._merge_aux_tile(
                    age_acc, sensor_acc, tile_age, tile_sensor,
                    tr0, tc0, tile, r_start, c_start, r_end, c_end,
                )
                if temporal:
                    geo = torch.from_numpy(geo_scene[None])
                    mw = torch.from_numpy(mw_scene[None])
                else:
                    geo = torch.from_numpy(geo_scene[:, 0][None])
                    mw = torch.from_numpy(mw_scene[:, 0][None])
                inpt = {"geo": geo.to(device), "mw": mw.to(device)}
                with torch.no_grad():
                    out = model(inpt)
                sp = out["surface_precip"] if isinstance(out, dict) else out
                if hasattr(sp, "expected_value"):
                    sp = sp.expected_value()
                pred = (
                    torch.as_tensor(sp).detach().to("cpu", torch.float32).numpy()
                ).reshape(tile, tile)
                self._accumulate_tile(
                    acc, wgt, pred, window, tr0, tc0, tile,
                    r_start, c_start, r_end, c_end,
                )

        with np.errstate(invalid="ignore"):
            result = np.where(wgt > 0, acc / wgt, np.nan).astype(np.float32)

        results = xr.Dataset(
            {
                "surface_precip": (("latitude", "longitude"), result),
                "mw_age": (("latitude", "longitude"), age_acc),
                "mw_sensor": (("latitude", "longitude"), sensor_acc),
            },
            coords={
                "latitude": ("latitude", latitude[r_start:r_end]),
                "longitude": ("longitude", longitude[c_start:c_end]),
                "time": target,
            },
        )
        results.mw_age.attrs.update(
            full_name="Age of the most recent microwave observation",
            unit="minutes before target time",
        )
        results.mw_sensor.attrs.update(
            full_name="Sensor of the most recent microwave observation",
            sensors=list(self.microwave_sensors),
            fill_value=-1,
        )
        return results

    def _load_inference_tile(
        self, tr0, tc0, tile, obs_size, slot_times, tolerance,
        geo_pool, mw_pool, norm_stats,
    ):
        """
        Load the slotted geo/microwave frames for one inference tile.

        Returns ``(geo_scene, mw_scene, mw_frame_sensors, has_valid)`` where
        ``geo_scene`` has shape ``(N_SLOTS, n_steps, obs_size, obs_size)`` and
        ``mw_scene`` ``(N_MW_SLOTS, n_steps, tile, tile)`` (missing frames left
        ``NaN``), ``mw_frame_sensors`` names the microwave sensor filling each
        frame (``None`` for empty frames), and ``has_valid`` indicates whether
        any observation was loaded at all.
        """
        n_steps = len(slot_times)
        obs_r0, obs_c0 = tr0 * RESOLUTION_RATIO, tc0 * RESOLUTION_RATIO
        cr0, cr1 = tr0 // REF_CELL, (tr0 + tile - 1) // REF_CELL + 1
        cc0, cc1 = tc0 // REF_CELL, (tc0 + tile - 1) // REF_CELL + 1

        geo_scene = np.full((N_SLOTS, n_steps, obs_size, obs_size), np.nan, np.float32)
        mw_scene = np.full((N_MW_SLOTS, n_steps, tile, tile), np.nan, np.float32)
        mw_frame_sensors: List[Optional[str]] = [None] * n_steps
        has_valid = False

        # Geo: the sensor best covering the tile, nearest store to each frame.
        geo_sensor, best_cov = None, 0
        for sensor, (_, files, avail) in geo_pool.items():
            cov = avail[:, cr0:cr1, cc0:cc1].reshape(len(files), -1).sum(axis=1)
            top = int(cov.max()) if cov.size else 0
            if top > best_cov:
                geo_sensor, best_cov = sensor, top
        if geo_sensor is not None:
            gtimes, gfiles, gavail = geo_pool[geo_sensor]
            covers = gavail[:, cr0:cr1, cc0:cc1].reshape(len(gfiles), -1).any(axis=1)
            for s, slot in enumerate(slot_times):
                offsets = np.abs(gtimes - slot)
                candidates = np.where(covers & (offsets <= tolerance))[0]
                if candidates.size == 0:
                    continue
                k = int(candidates[np.argmin(offsets[candidates])])
                array = self._load_obs(
                    self._store_path(geo_sensor, gfiles[k]), obs_r0, obs_c0, obs_size
                )
                stats = norm_stats.get(geo_sensor)
                if stats is not None:
                    array = self._scale_to_unit(array, stats)
                geo_scene[:, s] = slot_observations(array, geo_sensor)
                has_valid = True

        # Microwave: nearest observation from any sensor covering the tile.
        mw_candidates = []  # (representative scan time, sensor, filename)
        for sensor, (mtimes, mfiles, mavail) in mw_pool.items():
            sub_avail = mavail[:, cr0:cr1, cc0:cc1]
            sub_time = mtimes[:, cr0:cr1, cc0:cc1]
            covered = sub_avail & ~np.isnat(sub_time)
            for k in np.where(covered.reshape(len(mfiles), -1).any(axis=1))[0]:
                times_k = np.sort(sub_time[k][covered[k]])
                mw_candidates.append((times_k[times_k.size // 2], sensor, mfiles[k]))
        for s, slot in enumerate(slot_times):
            best = None  # (offset, sensor, filename)
            for scan_time, sensor, filename in mw_candidates:
                offset = abs(scan_time - slot)
                if offset <= tolerance and (best is None or offset < best[0]):
                    best = (offset, sensor, filename)
            if best is None:
                continue
            _, sensor, filename = best
            array = self._load_mw_obs(
                self._store_path(sensor, filename), tr0, tc0, tile
            )
            stats = norm_stats.get(sensor)
            if stats is not None:
                array = self._scale_to_unit(array, stats)
            mw_scene[:, s] = slot_observations(array, sensor)
            mw_frame_sensors[s] = sensor
            has_valid = True

        return geo_scene, mw_scene, mw_frame_sensors, has_valid

    # ------------------------------------------------------------------
    # Slot assignment tables
    # ------------------------------------------------------------------
    @staticmethod
    def _slot_table_html(caption, slot_columns, sensors, channel_cells) -> str:
        """
        Render a slot-assignment table as an HTML string.

        Args:
            caption: The table caption.
            slot_columns: A list of ``(header, values)`` pairs describing the
                per-slot leading columns (e.g. slot index, band name), where
                ``values`` has one entry per slot.
            sensors: The sensor names forming the remaining columns.
            channel_cells: A ``{sensor: [cell, ...]}`` mapping giving, for each
                slot, the HTML cell content of the channel(s) mapped to it (empty
                where the sensor has no channel in that slot).
        """
        n_slots = len(slot_columns[0][1])
        style = (
            "border-collapse:collapse;font-family:sans-serif;font-size:13px;"
            "text-align:center"
        )
        cell = "border:1px solid #ccc;padding:4px 8px"
        head = f"{cell};background:#f0f0f0"

        parts = [f'<table style="{style}">']
        parts.append(f"<caption style='padding:6px;font-weight:bold'>{html.escape(caption)}</caption>")
        headers = [h for h, _ in slot_columns] + list(sensors)
        parts.append("<tr>" + "".join(
            f'<th style="{head}">{html.escape(str(h))}</th>' for h in headers
        ) + "</tr>")

        for slot in range(n_slots):
            row = [
                f'<td style="{head}">{values[slot]}</td>' for _, values in slot_columns
            ]
            for sensor in sensors:
                content = channel_cells[sensor][slot]
                bg = "" if content else ";background:#fafafa;color:#bbb"
                row.append(f'<td style="{cell}{bg}">{content or "&mdash;"}</td>')
            parts.append("<tr>" + "".join(row) + "</tr>")
        parts.append("</table>")
        return "".join(parts)

    def geo_slot_table_html(self) -> str:
        """
        Render the geostationary channel-to-slot assignment as an HTML table.

        Rows are the :data:`N_SLOTS` spectral slots (with the slot's band name and
        canonical wavelength) and columns are the dataset's input sensors; each
        cell shows the sensor channel mapped to that slot (its stored index and
        central wavelength), or a dash where the sensor has no such channel.

        Returns:
            The table as an HTML string (renders in a Jupyter notebook).
        """
        sensors = [s for s in self.input_sensors if s in CHANNEL_SLOTS]
        cells = {sensor: ["" for _ in range(N_SLOTS)] for sensor in sensors}
        for sensor in sensors:
            wavelengths = SENSOR_WAVELENGTHS[sensor]
            for channel, slot in enumerate(CHANNEL_SLOTS[sensor]):
                if slot < 0:
                    continue
                entry = f"ch {channel} ({wavelengths[channel]:g} &micro;m)"
                cells[sensor][slot] = (
                    f"{cells[sensor][slot]}<br>{entry}"
                    if cells[sensor][slot]
                    else entry
                )
        slot_columns = [
            ("Slot", [str(s) for s in range(N_SLOTS)]),
            ("Band", [html.escape(name) for name in SLOT_NAMES]),
            (
                "&lambda; (&micro;m)",
                [
                    f"{wl:g}" if np.isfinite(wl) else "&mdash;"
                    for wl in SLOT_WAVELENGTHS
                ],
            ),
        ]
        return self._slot_table_html(
            "Geostationary channel slots", slot_columns, sensors, cells
        )

    def mw_slot_table_html(self) -> str:
        """
        Render the microwave channel-to-slot assignment as an HTML table.

        Rows are the :data:`N_MW_SLOTS` frequency slots (with the slot's band
        name, frequency and polarization) and columns are the dataset's microwave
        sensors; each cell shows the sensor channel mapped to that slot (its
        stored index and, where available, frequency/polarization), or a dash.

        Returns:
            The table as an HTML string (renders in a Jupyter notebook).
        """
        sensors = [s for s in self.microwave_sensors if s in MW_CHANNEL_SLOTS]
        cells = {sensor: ["" for _ in range(N_MW_SLOTS)] for sensor in sensors}
        for sensor in sensors:
            specs = MW_SENSOR_CHANNELS[sensor]
            for channel, slot in enumerate(MW_CHANNEL_SLOTS[sensor]):
                if slot < 0:
                    continue
                spec = specs[channel]
                if isinstance(spec, tuple):
                    freq, offset, pol = spec
                    label = f"{freq:g}&plusmn;{offset:g}" if offset else f"{freq:g}"
                    entry = f"ch {channel} ({label} GHz {html.escape(pol)})"
                else:
                    entry = f"ch {channel}"
                cells[sensor][slot] = (
                    f"{cells[sensor][slot]}<br>{entry}"
                    if cells[sensor][slot]
                    else entry
                )
        slot_columns = [
            ("Slot", [str(s) for s in range(N_MW_SLOTS)]),
            ("Band", [html.escape(name) for name in MW_SLOT_NAMES]),
            ("Freq (GHz)", [f"{freq:g}" for freq, _, _ in MW_SLOTS]),
            ("Pol", [html.escape(pol) for _, _, pol in MW_SLOTS]),
        ]
        return self._slot_table_html(
            "Microwave channel slots", slot_columns, sensors, cells
        )

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

        # Per-dataset arrays of per-file days, for every input sensor and the
        # reference.
        days_per_dataset: Dict[str, np.ndarray] = {}
        for meta_by_sensor in (self.geo_meta, self.microwave_meta):
            for sensor, meta in meta_by_sensor.items():
                days_per_dataset[sensor] = self._file_days(meta)
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


class ArgosDataset(Dataset):
    """
    Dataset of pre-extracted training scenes.

    Loads the fixed-size, slotted scenes written by
    :meth:`ArgosTrainingData.extract_samples`. Each item is the same
    ``(inputs, target)`` tuple a model consumes -- ``inputs`` holding the ``geo``
    and ``mw`` tensors (and the tile ``coordinates``) and ``target`` the
    ``surface_precip`` tensor. Absent inputs are kept as ``NaN`` so the keys are
    always present (and batch cleanly); the model treats them as zeros.

    With ``augment=True`` a random affine transform (rotation, isotropic scaling
    and shear) is drawn per item and applied identically to ``geo``, ``mw`` and
    the target, so they stay co-registered. Resampling is NaN-aware: missing
    pixels do not bleed into valid ones and stay ``NaN``.
    """

    def __init__(
        self,
        path: Union[str, Path],
        augment: bool = False,
        rotation: Tuple[float, float] = (-180.0, 180.0),
        scale: Tuple[float, float] = (0.8, 1.2),
        shear: Tuple[float, float] = (-15.0, 15.0),
    ):
        """
        Args:
            path: Path of a ``.zarr`` store written by ``extract_samples``.
            augment: If ``True``, apply a random affine augmentation to each item.
            rotation: Range (degrees) for the random rotation.
            scale: Range for the random isotropic scale factor.
            shear: Range (degrees) for the random x/y shear.
        """
        super().__init__()
        self.path = Path(path)
        self.augment = bool(augment)
        self.rotation = rotation
        self.scale = scale
        self.shear = shear

    # ``DataLoader(worker_init_fn=...)`` helper to seed NumPy per worker (the
    # augmentation uses ``numpy.random``).
    worker_init_fn = staticmethod(worker_init_fn)

    @cached_property
    def store(self) -> "zarr.Group":
        """The (lazily opened) zarr store, opened per process for fork safety."""
        return zarr.open_group(str(self.path), mode="r")

    @property
    def sensors(self) -> List[str]:
        """The sensor names indexed by ``geo_sensor``/``mw_sensor``."""
        return list(self.store.attrs.get("sensors", []))

    def __len__(self) -> int:
        return self.store["surface_precip"].shape[0]

    def _sample_affine_params(self) -> Dict[str, object]:
        """Draw random affine parameters (rotation, scale, x/y shear)."""
        return {
            "angle": float(np.random.uniform(*self.rotation)),
            "scale": float(np.random.uniform(*self.scale)),
            "shear": [
                float(np.random.uniform(*self.shear)),
                float(np.random.uniform(*self.shear)),
            ],
        }

    @staticmethod
    def _apply_affine(tensor: torch.Tensor, params: Dict[str, object]) -> torch.Tensor:
        """
        Apply an affine transform to a spatial tensor about its center.

        Uses nearest-neighbor resampling, so values are copied rather than
        blended: ``NaN`` pixels stay ``NaN`` and never contaminate valid
        neighbours. Samples that fall outside the input (e.g. the corners exposed
        by a rotation) are filled with ``NaN`` via ``fill``, so no separate
        validity mask is needed. Leading channel/step dimensions are flattened;
        the transform acts on the trailing ``(H, W)`` axes.
        """
        shape = tensor.shape
        height, width = shape[-2], shape[-1]
        x = tensor.reshape(1, -1, height, width).float()
        out = tv_functional.affine(
            x,
            angle=params["angle"],
            translate=[0, 0],
            scale=params["scale"],
            shear=params["shear"],
            interpolation=InterpolationMode.NEAREST,
            fill=float("nan"),
        )
        return out.reshape(shape)

    def __getitem__(self, index: int) -> Tuple[Dict[str, object], torch.Tensor]:
        store = self.store
        coords = np.asarray(store["coordinates"][index])
        geo = torch.from_numpy(np.asarray(store["geo"][index]))
        mw = torch.from_numpy(np.asarray(store["mw"][index]))
        target = torch.from_numpy(np.asarray(store["surface_precip"][index]))

        # Skip samples without any valid reference data by falling back to a
        # randomly drawn sample.
        if not bool(torch.isfinite(target).any()):
            LOGGER.warning(
                "Sample %d has no valid surface_precip values; drawing a "
                "random replacement sample.",
                index,
            )
            return self[np.random.randint(len(self))]

        if self.augment:
            # One transform per item, applied to every field so they stay aligned.
            params = self._sample_affine_params()
            geo = self._apply_affine(geo, params)
            mw = self._apply_affine(mw, params)
            target = self._apply_affine(target, params)

        inputs = {
            "geo": geo,
            "mw": mw,
            "coordinates": (int(coords[0]), int(coords[1])),
        }
        return inputs, target


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
            ``ArgosTrainingData.__getitem__`` -- ``inputs`` holding the
            observation tensors keyed by group/sensor at the top level alongside
            ``"latitude"`` / ``"longitude"`` coordinate tensors, and ``target``
            the surface precipitation tensor.
        n_channels: The number of channels to show per input sensor.
        rng: Optional seed or ``numpy.random.Generator`` controlling the random
            channel selection.

    Returns:
        The created ``matplotlib`` figure.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(rng)

    inputs, target = sample
    obs = {
        key: value
        for key, value in inputs.items()
        if key not in ("coordinates", "latitude", "longitude")
    }
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
