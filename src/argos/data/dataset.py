"""
argos.data.dataset
==================

An interface to extract training data for the Argos multi-satellite precipitation retrieval
system. The interface combines preprocessed observations from geostationary and polar-orbiting
satellites and combines them with reference precipitation estimates from GPROF V08.


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
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
import hashlib
import html
import logging
import multiprocessing
from pathlib import Path
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
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

DEFAULT_INPUT_SATELLITES = ("goes16", "goes18", "goes19", "seviri", "seviri_io")
DEFAULT_MICROWAVE_SATELLITES = (
    "noaa20_atms", "npp_atms",
    "f16_ssmis", "f17_ssmis", "f18_ssmis", "f19_ssmis",
    "noaa18_mhs", "noaa19_mhs", "metopa_mhs", "metopb_mhs", "metopc_mhs",
    "tropics03_tms", "tropics05_tms", "tropics06_tms",
)
DEFAULT_REFERENCE = "gprof_gmi"

from argos.data.satellite import (  # noqa: E402
    N_SLOTS, N_MW_SLOTS, EXTRA_SLOT,
    SLOT_WAVELENGTHS, SLOT_NAMES, MW_SLOTS, MW_SLOT_NAMES,
    get_satellite,
)


def _match_cell(
    dataset: "TrainingDataset",
    cell_scan_time: Tuple[Tuple[int, int], "np.datetime64"],
    cell_rng: "np.random.Generator",
    ref_times: "np.ndarray",
    ref_filenames: "np.ndarray",
    n_steps: int,
    frames: int,
    step: "np.timedelta64",
    scene_size: int,
    require_microwave: bool,
) -> List[Tuple]:
    (row, col), scan_time = cell_scan_time
    matches = np.where(
        ref_times[:, row, col] == np.datetime64(scan_time, "ns")
    )[0]
    if matches.size == 0:
        return []
    ref_path = dataset._store_path(
        dataset.reference_name, str(ref_filenames[matches[0]])
    )

    slot_obs = [
        dataset.find_slot_observations(
            (row, col), scan_time,
            slot=n_steps - s, step=step, super_cell=scene_size,
        )
        for s in range(frames)
    ]

    mw_files: List[Optional[Path]] = [
        obs["mw"][int(cell_rng.integers(len(obs["mw"])))] if obs["mw"] else None
        for obs in slot_obs
    ]
    if require_microwave and all(f is None for f in mw_files):
        return []

    seen_geo: set = set()
    geo_sat_names: List[str] = []
    for obs in slot_obs:
        for p in obs["geo"]:
            n = p.parent.name
            if n not in seen_geo:
                seen_geo.add(n)
                geo_sat_names.append(n)

    result = []
    for geo_sat in geo_sat_names:
        geo_files: List[Optional[Path]] = [
            next((p for p in obs["geo"] if p.parent.name == geo_sat), None)
            for obs in slot_obs
        ]
        result.append(
            ((row, col), scan_time, ref_path, geo_sat, geo_files, mw_files)
        )
    return result


def slot_observations(
    obs: np.ndarray, satellite: str, fill: float = np.nan
) -> np.ndarray:
    """
    Map a satellite's observations onto its common set of channel slots.

    Geostationary satellites are mapped onto the :data:`N_SLOTS` spectral slots
    and microwave satellites onto the :data:`N_MW_SLOTS` frequency slots.

    Args:
        obs: An observation array of shape ``(channels, ...)`` in the
            satellite's stored channel order.
        satellite: The satellite name (must match a TOML definition).
        fill: Value used for slots the satellite does not provide.

    Returns:
        An array of shape ``(n_slots, ...)`` with each channel placed in its
        slot and missing slots set to ``fill``.
    """
    sat = get_satellite(satellite)
    slots, n_slots = sat.slots, sat.n_slots
    if obs.shape[0] != len(slots):
        raise ValueError(
            f"'{satellite}' observations have {obs.shape[0]} channels but the "
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
        if self.norm_stats:
            arrays["norm_stat_types"] = np.array(sorted(self.norm_stats.keys()))
            for stype, stats in self.norm_stats.items():
                arrays[f"nmin__{stype}"] = stats["min"]
                arrays[f"nmax__{stype}"] = stats["max"]
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path: Path) -> "_SampleIndex":
        with np.load(str(path), allow_pickle=False) as cache:
            sensors = cache["sensors"]
            filenames, norm_stats = {}, {}
            for sensor in sensors.tolist():
                filenames[sensor] = cache[f"fn__{sensor}"]
            if "norm_stat_types" in cache:
                for stype in cache["norm_stat_types"].tolist():
                    norm_stats[stype] = {
                        "min": cache[f"nmin__{stype}"],
                        "max": cache[f"nmax__{stype}"],
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


class ArgosData(Dataset):
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
        input_satellites: Sequence[str] = DEFAULT_INPUT_SATELLITES,
        microwave_satellites: Sequence[str] = DEFAULT_MICROWAVE_SATELLITES,
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
            path: Root directory containing the per-satellite sub-directories.
            input_satellites: Names of the geostationary satellite
                sub-directories to use as the high-resolution ``"geo"`` input.
            microwave_satellites: Names of the microwave satellite
                sub-directories (ATMS, SSMIS, ...) to use as the ``"mw"``
                input. These are on the lower (reference) resolution grid and,
                like the reference, have a per-cell scan time.
            reference_name: Name of the reference sub-directory.
            tile_size: Side length of the (square) reference tile in reference
                pixels. The input tile has side length
                ``tile_size * RESOLUTION_RATIO``.
            time_window: Maximum allowed difference between an input
                acquisition time and a reference cell's scan time.
            position_jitter: Maximum random shift of the tile center applied
                when loading a sample, in reference (0.05-degree) pixels (and
                ``RESOLUTION_RATIO`` times as many input pixels). Defaults to
                half an availability cell (20 reference / 40 input pixels). Set
                to 0 to disable jittering.
            slot_channels: If ``True`` (the default), the loaded satellite's
                observations are mapped onto the common spectral slots (see
                :func:`slot_observations`) and returned as the ``"geo"`` and
                ``"mw"`` inputs of shape ``(n_slots, H, W)`` (absent bands set
                to ``NaN``). If ``False``, the raw observation tensor of the
                loaded satellite is returned under the satellite's name.
            normalize: If ``True`` (the default), the loaded observations are
                scaled to ``[0, 1]`` per channel using the satellite-wise
                :attr:`normalization_stats` (min/max across all of the
                satellite's stores).
            require_both_inputs: If ``True``, only keep samples that have both
                a geostationary and a microwave observation (by default a sample
                is kept if it has either).
        """
        super().__init__()
        self.path = Path(path)
        self.input_satellites = tuple(input_satellites)
        self.microwave_satellites = tuple(microwave_satellites)
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

        # Map each satellite name to its type ("geo" or "mw") from the registry.
        self._satellite_group = {
            **{sat: "geo" for sat in self.input_satellites},
            **{sat: "mw" for sat in self.microwave_satellites},
        }

    # ``DataLoader(worker_init_fn=...)`` helper to seed NumPy per worker.
    worker_init_fn = staticmethod(worker_init_fn)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    @cached_property
    def geo_files(self) -> Dict[str, List[Path]]:
        """Mapping of each input satellite name to its sorted list of '.zarr' stores."""
        return {
            sat: sorted((self.path / sat).glob("*.zarr"))
            for sat in self.input_satellites
        }

    @cached_property
    def mw_files(self) -> Dict[str, List[Path]]:
        """Mapping of each microwave satellite name to its sorted '.zarr' stores."""
        return {
            sat: sorted((self.path / sat).glob("*.zarr"))
            for sat in self.microwave_satellites
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

    def _load_indices(self, satellites, compute) -> Dict[str, xr.Dataset]:
        """
        Load (or compute and cache) the per-satellite index of each named dataset.

        Returns a mapping of name to its index dataset (availability and time
        only; normalization stats are kept in separate files under
        ``.argos/stats/``). Both the index and the stats are (re)computed if
        either is missing.
        """
        indices = {}
        for name in satellites:
            index_ok = self._index_path(name).exists()
            stats_ok = self._stats_path(name).exists()
            if not index_ok or not stats_ok:
                meta = compute(name, self._list_stores(name))
                self._write_index(name, meta)
                self._write_stats(name, meta)
            else:
                meta = self._read_index(name)
            if meta is not None and meta.sizes["samples"] > 0:
                indices[name] = meta.drop_vars(
                    ["obs_min", "obs_max"], errors="ignore"
                )
        return indices

    @cached_property
    def geo_meta(self) -> Dict[str, xr.Dataset]:
        """
        Per-satellite availability and acquisition time of the geostationary inputs.

        A mapping of satellite name to its own :class:`xarray.Dataset` (kept
        separate, not concatenated across satellites). Each dataset holds
        ``availability`` (``(samples, lat_cell, lon_cell)`` boolean), the scalar
        ``time`` (``(samples,)`` ``datetime64[ns]``), the per-channel
        ``obs_min``/``obs_max`` statistics and a ``filename`` coordinate (the
        full store path is built on the fly with :meth:`_store_path`). The
        per-satellite metadata is cached to ``index_<satellite>.nc`` and loaded
        from there on subsequent runs (see :meth:`recompute_indices` to refresh).
        """
        return self._load_indices(self.input_satellites, self._compute_geo_meta)

    def _compute_geo_meta(self, satellite: str, files: List[Path]) -> xr.Dataset:
        """Read the availability and acquisition time of one satellite's stores."""
        sensor = satellite  # local alias for log messages / tqdm labels
        LOGGER.info(
            "Loading metadata for input satellite '%s' (%d files).", sensor, len(files)
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
            self._write_stats(name, meta)
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
            self.microwave_satellites, self._compute_reference_meta
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

    @staticmethod
    def _sensor_type(name: str) -> str:
        """Sensor instrument name for a satellite instance (e.g. ``'ABI'`` for ``'goes16'``)."""
        return get_satellite(name).sensor

    def _compute_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Per-channel min/max keyed by sensor type, aggregated across all instances."""
        type_mins: Dict[str, List[np.ndarray]] = {}
        type_maxs: Dict[str, List[np.ndarray]] = {}
        for sensor in (*self.input_satellites, *self.microwave_satellites):
            ds = self._read_stats(sensor)
            if ds is None or "obs_min" not in ds:
                continue
            stype = self._sensor_type(sensor)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                type_mins.setdefault(stype, []).append(
                    np.nanmin(ds["obs_min"].values, axis=0)
                )
                type_maxs.setdefault(stype, []).append(
                    np.nanmax(ds["obs_max"].values, axis=0)
                )
        stats = {}
        for stype in type_mins:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats[stype] = {
                    "min": np.nanmin(type_mins[stype], axis=0),
                    "max": np.nanmax(type_maxs[stype], axis=0),
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
    @property
    def _indices_dir(self) -> Path:
        """Directory for compressed availability / scan-time index files."""
        return self.path / ".argos" / "indices"

    @property
    def _stats_dir(self) -> Path:
        """Directory for per-satellite normalization statistics files."""
        return self.path / ".argos" / "stats"

    def _index_path(self, name: str) -> Path:
        return self._indices_dir / f"{name}.nc"

    def _stats_path(self, name: str) -> Path:
        return self._stats_dir / f"{name}.nc"

    def _read_index(self, name: str) -> Optional[xr.Dataset]:
        """Load a cached availability/time index, or ``None`` if absent."""
        path = self._index_path(name)
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            meta = ds.load()
        LOGGER.info(
            "Loaded cached index for '%s' from '%s' (%d stores).",
            name, path, meta.sizes["samples"],
        )
        return meta

    def _write_index(self, name: str, meta: xr.Dataset) -> None:
        """Write the availability/time index to ``.argos/indices/`` with compression."""
        if meta.sizes["samples"] == 0:
            return
        path = self._index_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        index_ds = meta.drop_vars(["obs_min", "obs_max"], errors="ignore")
        encoding = {
            v: {"zlib": True, "complevel": 4}
            for v in index_ds.data_vars
        }
        index_ds.to_netcdf(path, encoding=encoding)
        LOGGER.info("Wrote index for '%s' to '%s'.", name, path)

    def _read_stats(self, name: str) -> Optional[xr.Dataset]:
        """Load a cached normalization-stats dataset, or ``None`` if absent."""
        path = self._stats_path(name)
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            return ds.load()

    def _write_stats(self, name: str, meta: xr.Dataset) -> None:
        """Write per-file obs_min/obs_max to ``.argos/stats/``."""
        if "obs_min" not in meta:
            return
        path = self._stats_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats_ds = meta[["obs_min", "obs_max"]]
        stats_ds.to_netcdf(path)
        LOGGER.info("Wrote normalization stats for '%s' to '%s'.", name, path)

    def recompute_indices(self) -> None:
        """
        Delete cached indices and stats and recompute them from the stores.

        This discards the files under ``.argos/indices/`` and ``.argos/stats/``
        as well as the sample-index cache, so the next access re-reads all
        stores and rebuilds everything.
        """
        names = (
            *self.input_satellites,
            *self.microwave_satellites,
            self.reference_name,
        )
        for name in names:
            for path in (self._index_path(name), self._stats_path(name)):
                if path.exists():
                    path.unlink()
                    LOGGER.info("Removed cached file '%s'.", path)
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
        """Path of the on-disk sample-index cache under ``.argos/``."""
        key = "|".join(
            [
                "v3",  # cache format version (bump to invalidate old caches)
                ",".join(self.input_satellites),
                ",".join(self.microwave_satellites),
                self.reference_name,
                str(int(self.time_window.astype("int64"))),
                str(self.require_both_inputs),
            ]
        )
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.path / ".argos" / f"samples_{digest}.npz"

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
        path.parent.mkdir(parents=True, exist_ok=True)
        samples.save(path)
        LOGGER.info("Wrote sample index cache to '%s'.", path)
        return samples

    def _build_samples(self) -> _SampleIndex:
        """Enumerate samples from the metadata indices (the slow, one-off path)."""
        window = self.time_window

        # Per-satellite input arrays (geo and microwave), with a per-file time
        # span used to pre-filter candidates.
        inputs = {
            sat: self._sensor_arrays(meta)
            for sat, meta in {**self.geo_meta, **self.microwave_meta}.items()
        }
        sensors = list(inputs)
        n_sensors = len(sensors)
        groups = np.array([self._satellite_group[s] for s in sensors])
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
    # Super-cell iteration
    # ------------------------------------------------------------------
    def iter_reference_super_cells(
        self,
        *,
        subsample: int = 16,
        rng: Optional[Union[int, "np.random.Generator"]] = None,
        start_time: Optional[Union[str, "np.datetime64"]] = None,
        end_time: Optional[Union[str, "np.datetime64"]] = None,
    ) -> Iterator[Tuple[Tuple[int, int], "np.datetime64"]]:
        """
        Iterate over availability cells that anchor a valid reference estimate.

        For each reference granule the function finds every availability cell
        (on the 90×180 grid) that has both a ``True`` availability flag and a
        non-NaT per-cell scan time.  Each such cell is a candidate anchor for
        a 4×4 super-cell sample window (160×160 reference pixels, 320×320 geo
        pixels) centred on that cell.  A random fraction ``1/subsample`` of
        the candidates is kept; the default of 16 gives roughly one anchor per
        4×4 block, i.e. approximately non-overlapping windows.

        Args:
            subsample: Keep each valid cell independently with probability
                ``1/subsample``.  Default 16.
            rng: :class:`numpy.random.Generator` or integer seed.  ``None``
                uses :func:`numpy.random.default_rng`.
            start_time: Optional lower bound on the per-cell scan time
                (inclusive).  Anything accepted by ``numpy.datetime64``.
            end_time: Optional upper bound on the per-cell scan time
                (inclusive).

        Yields:
            ``((row, col), scan_time)`` — the availability-cell coordinate on
            the 90×180 grid and the per-cell reference scan time
            (``datetime64[ns]``).
        """
        rng = np.random.default_rng(rng)
        t_start = np.datetime64(start_time, "ns") if start_time is not None else None
        t_end = np.datetime64(end_time, "ns") if end_time is not None else None

        ref = self.reference_meta
        avail = ref["availability"].values.astype(bool)  # (n_files, 90, 180)
        times = ref["time"].values                        # (n_files, 90, 180)

        valid = avail & ~np.isnat(times)
        if t_start is not None:
            valid &= times >= t_start
        if t_end is not None:
            valid &= times <= t_end

        fs, rows, cols = np.where(valid)
        if fs.size == 0:
            return
        if subsample > 1:
            keep = rng.random(fs.size) < (1.0 / subsample)
            fs, rows, cols = fs[keep], rows[keep], cols[keep]

        for f, r, c in tqdm(
            zip(fs.tolist(), rows.tolist(), cols.tolist()),
            desc="Iterating reference super-cells",
            total=fs.size,
            unit="cell",
        ):
            yield (r, c), times[f, r, c]

    def find_slot_observations(
        self,
        cell: Tuple[int, int],
        scan_time: "np.datetime64",
        slot: int,
        step: np.timedelta64 = np.timedelta64(20, "m"),
        super_cell: int = 1,
    ) -> Dict[str, List[Path]]:
        """
        Find available geo and MW observation files for a cell and temporal slot.

        Slot 0 is the coincident slot — files whose acquisition time (geo) or
        per-cell scan time (MW) falls within ``time_window`` of ``scan_time``.
        Positive slot indices step back in time by ``step`` each: slot *s*
        targets time ``scan_time - s * step``.

        Args:
            cell: Availability-cell coordinate ``(row, col)`` on the 90×180 grid.
            scan_time: Reference per-cell scan time (from the iterator) for slot 0.
            slot: Non-negative temporal slot index.
            step: Time between consecutive slots (default 20 minutes).
            super_cell: Side length in availability cells of the spatial search
                window centred on ``cell``.  ``1`` checks only the anchor cell;
                odd values like ``5`` or ``7`` extend the search so that files
                covering any part of the super-cell are included.

        Returns:
            ``{"geo": [...], "mw": [...]}`` — for each group a list of
            :class:`~pathlib.Path` objects pointing to matching ``.zarr`` stores,
            ordered from highest to lowest satellite priority.  Files from the
            same satellite are further sorted by the minimum temporal offset
            between the slot target and any valid cell in the search window.
        """
        row, col = int(cell[0]), int(cell[1])
        target = np.datetime64(scan_time, "ns") - slot * np.timedelta64(step, "ns")
        window = self.time_window

        half = super_cell // 2
        r0 = max(0, row - half)
        r1 = min(N_CELLS_LAT, row + half + 1)
        c0 = max(0, col - half)
        c1 = min(N_CELLS_LON, col + half + 1)

        def _ns(delta: np.timedelta64) -> int:
            return int(delta / np.timedelta64(1, "ns"))

        geo_candidates: List[Tuple[int, int, Path]] = []
        for sat_name, meta in self.geo_meta.items():
            priority = get_satellite(sat_name).priority
            file_times = meta["time"].values                                           # (n_files,)
            region_avail = meta["availability"].values[:, r0:r1, c0:c1].astype(bool).any(axis=(1, 2))
            deltas = np.abs(file_times - target)
            for idx in np.where(region_avail & (deltas <= window))[0]:
                path = self._store_path(sat_name, meta["filename"].values[idx])
                geo_candidates.append((-priority, _ns(deltas[idx]), path))
        geo_candidates.sort(key=lambda t: (t[0], t[1]))

        mw_candidates: List[Tuple[int, int, Path]] = []
        for sat_name, meta in self.microwave_meta.items():
            priority = get_satellite(sat_name).priority
            region_avail = meta["availability"].values[:, r0:r1, c0:c1].astype(bool)  # (n, h, w)
            region_times = meta["time"].values[:, r0:r1, c0:c1]                        # (n, h, w)
            valid = region_avail & ~np.isnat(region_times)
            for idx in np.where(valid.any(axis=(1, 2)))[0]:
                cell_deltas = np.abs(region_times[idx][valid[idx]] - target)
                if (cell_deltas <= window).any():
                    path = self._store_path(sat_name, meta["filename"].values[idx])
                    mw_candidates.append((-priority, _ns(cell_deltas.min()), path))
        mw_candidates.sort(key=lambda t: (t[0], t[1]))

        return {
            "geo": [p for _, _, p in geo_candidates],
            "mw": [p for _, _, p in mw_candidates],
        }

    def plot_sample(
        self,
        cell: Tuple[int, int],
        scan_time: "np.datetime64",
        slot: int = 0,
        geo_channels: Tuple[int, ...] = (1, 6, 12, 7),
        mw_channels: Tuple[int, ...] = (0, 2, 6, 9),
        layer: int = 0,
        step: np.timedelta64 = np.timedelta64(20, "m"),
    ) -> "matplotlib.figure.Figure":
        """
        Plot geo channels, MW channels and reference precipitation for a sample.

        Three-row figure:

        * **Row 0** — four geostationary images (``geo_channels`` slot indices)
          with satellite name and filename to the left.
        * **Row 1** — four microwave images (``mw_channels`` slot indices) with
          satellite name, filename and per-cell scan time to the left.
        * **Row 2** — surface precipitation from the reference dataset.

        The suptitle shows the reference scan time and, when ``slot > 0``, the
        target time for the slot.  ``layer`` selects which file to display when
        multiple files match; it is clamped to the last available file.

        Args:
            cell: Availability-cell coordinate ``(row, col)`` on the 90×180 grid.
            scan_time: Reference per-cell scan time (slot-0 anchor, from the
                iterator).
            slot: Temporal slot index (0 = coincident; higher = further back).
            geo_channels: Four geo slot indices to display (default: red,
                shortwave IR, clean IR window, upper WV).
            mw_channels: Four MW slot indices to display (default: 19V, 23V,
                89V, 183±1).
            layer: Index into the priority-sorted file list returned by
                :meth:`find_slot_observations` (clamped to last element).
            step: Time between consecutive slots (default 20 minutes).

        Returns:
            The :class:`matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        row, col = int(cell[0]), int(cell[1])

        # 4×4 super-cell pixel windows, clamped to global grid bounds.
        obs_size = 4 * OBS_CELL   # 320 geo pixels
        ref_size = 4 * REF_CELL   # 160 reference pixels
        obs_r0 = min(max((row - 2) * OBS_CELL, 0), N_CELLS_LAT * OBS_CELL - obs_size)
        obs_c0 = min(max((col - 2) * OBS_CELL, 0), N_CELLS_LON * OBS_CELL - obs_size)
        ref_r0 = obs_r0 // RESOLUTION_RATIO
        ref_c0 = obs_c0 // RESOLUTION_RATIO

        # Resolve geo / MW files for this slot.
        obs = self.find_slot_observations(cell, scan_time, slot, step=step)
        geo_list, mw_list = obs["geo"], obs["mw"]
        geo_path = geo_list[min(layer, len(geo_list) - 1)] if geo_list else None
        mw_path = mw_list[min(layer, len(mw_list) - 1)] if mw_list else None

        # Resolve the reference file from scan_time.
        ref_meta = self.reference_meta
        cell_times = ref_meta["time"].values[:, row, col]
        matches = np.where(cell_times == np.datetime64(scan_time, "ns"))[0]
        ref_path = (
            self._store_path(
                self.reference_name,
                str(ref_meta["filename"].values[matches[0]]),
            )
            if matches.size > 0 else None
        )

        # Layout: narrow text column (0) + 4 image columns (1–4).
        fig = plt.figure(figsize=(17, 11))
        gs = gridspec.GridSpec(
            3, 5, figure=fig,
            width_ratios=[1.4, 3, 3, 3, 3],
            hspace=0.45, wspace=0.25,
        )
        title = f"Reference time: {str(np.datetime64(scan_time, 's'))}"
        if slot > 0:
            target = np.datetime64(scan_time, "ns") - slot * np.timedelta64(step, "ns")
            title += f"   |   Slot {slot} → {str(np.datetime64(target, 's'))}"
        fig.suptitle(title, fontsize=11)

        # ---- Row 0: geostationary channels ----
        ax_lbl = fig.add_subplot(gs[0, 0])
        ax_lbl.axis("off")
        if geo_path is not None:
            sat_name = geo_path.parent.name
            raw = self._load_obs(geo_path, obs_r0, obs_c0, obs_size)
            slotted = slot_observations(raw, sat_name)
            ax_lbl.text(
                0.5, 0.5,
                f"{sat_name}\n{geo_path.name}",
                ha="center", va="center", fontsize=7, rotation=90,
                transform=ax_lbl.transAxes,
            )
            for i, ch in enumerate(geo_channels):
                ax = fig.add_subplot(gs[0, i + 1])
                data = (
                    slotted[ch]
                    if ch < slotted.shape[0]
                    else np.full(slotted.shape[1:], np.nan)
                )
                ax.imshow(data, cmap="gray", origin="upper", aspect="auto",
                          interpolation="nearest")
                lbl = SLOT_NAMES[ch] if ch < len(SLOT_NAMES) else f"slot {ch}"
                ax.set_title(f"{ch}: {lbl}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax_lbl.text(0.5, 0.5, "No geo\ndata", ha="center", va="center",
                        fontsize=9, transform=ax_lbl.transAxes)
            for i in range(4):
                fig.add_subplot(gs[0, i + 1]).axis("off")

        # ---- Row 1: microwave channels ----
        ax_lbl = fig.add_subplot(gs[1, 0])
        ax_lbl.axis("off")
        if mw_path is not None:
            sat_name = mw_path.parent.name
            label_parts = [sat_name, mw_path.name]
            mw_meta = self.microwave_meta.get(sat_name)
            if mw_meta is not None:
                filenames = [self._as_str(f) for f in mw_meta["filename"].values]
                try:
                    fidx = filenames.index(mw_path.name)
                    t = mw_meta["time"].values[fidx, row, col]
                    if not np.isnat(t):
                        label_parts.append(str(np.datetime64(t, "s")))
                except ValueError:
                    pass
            ax_lbl.text(
                0.5, 0.5,
                "\n".join(label_parts),
                ha="center", va="center", fontsize=7, rotation=90,
                transform=ax_lbl.transAxes,
            )
            raw = self._load_mw_obs(mw_path, ref_r0, ref_c0, ref_size)
            slotted = slot_observations(raw, sat_name)
            for i, ch in enumerate(mw_channels):
                ax = fig.add_subplot(gs[1, i + 1])
                data = (
                    slotted[ch]
                    if ch < slotted.shape[0]
                    else np.full(slotted.shape[1:], np.nan)
                )
                ax.imshow(data, cmap="viridis", origin="upper", aspect="auto",
                          interpolation="nearest")
                lbl = MW_SLOT_NAMES[ch] if ch < len(MW_SLOT_NAMES) else f"slot {ch}"
                ax.set_title(f"{ch}: {lbl}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax_lbl.text(0.5, 0.5, "No MW\ndata", ha="center", va="center",
                        fontsize=9, transform=ax_lbl.transAxes)
            for i in range(4):
                fig.add_subplot(gs[1, i + 1]).axis("off")

        # ---- Row 2: reference precipitation ----
        for i in range(1, 5):
            fig.add_subplot(gs[2, i]).axis("off")
        ax_ref = fig.add_subplot(gs[2, 1])
        if ref_path is not None:
            sp = self._load_reference(ref_path, ref_r0, ref_c0, ref_size)
            im = ax_ref.imshow(sp, cmap="Blues", origin="upper", aspect="auto",
                               vmin=0, interpolation="nearest")
            plt.colorbar(im, ax=ax_ref, label="mm/h", fraction=0.015, pad=0.02)
            ax_ref.set_title(
                f"Surface precipitation ({self.reference_name})", fontsize=9
            )
        else:
            ax_ref.text(0.5, 0.5, "No reference data", ha="center", va="center",
                        fontsize=10, transform=ax_ref.transAxes)
        ax_ref.set_xticks([])
        ax_ref.set_yticks([])

        return fig

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
        stats = self.normalization_stats.get(self._sensor_type(sensor))
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

        # For each input group, choose one of the available satellites weighted
        # by priority and load its observations. ``"geo"`` is high resolution,
        # ``"mw"`` is on the reference grid. When slotting, the observations
        # are mapped onto the group's common slots; otherwise the raw tensor is
        # returned under the satellite's name.
        sensor_file = si.sensor_file[index]
        obs = {}
        for group, columns, loader, (r0, c0, size) in (
            ("geo", si.geo_cols, self._load_obs, (obs_r0, obs_c0, obs_size)),
            ("mw", si.mw_cols, self._load_mw_obs, (ref_r0, ref_c0, self.tile_size)),
        ):
            present = columns[sensor_file[columns] >= 0]
            if present.size == 0:
                continue
            priorities = np.array(
                [get_satellite(str(si.sensors[c])).priority for c in present],
                dtype=float,
            )
            weights = priorities / priorities.sum()
            col = int(np.random.choice(present, p=weights))
            satellite = str(si.sensors[col])
            filename = si.filenames[satellite][sensor_file[col]]
            array = loader(self._store_path(satellite, filename), r0, c0, size)
            if self.normalize:
                array = self._normalize(array, satellite)
            if self.slot_channels:
                obs[group] = torch.from_numpy(slot_observations(array, satellite))
            else:
                obs[satellite] = torch.from_numpy(array)

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
            print("TIMES :: ", slot_time, sample_times[index].astype("int64"))
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

    def extract_super_cell_samples(
        self,
        output_path: Union[str, Path],
        n_steps: int = 0,
        scene_size: int = 5,
        step: np.timedelta64 = np.timedelta64(20, "m"),
        subsample: int = 16,
        rng: Optional[Union[int, "np.random.Generator"]] = None,
        require_microwave: bool = True,
        start_time: Optional[Union[str, "np.datetime64"]] = None,
        end_time: Optional[Union[str, "np.datetime64"]] = None,
        workers: int = 1,
        position_jitter: int = REF_CELL,
    ) -> Path:
        """
        Extract super-cell training samples anchored on per-cell reference times.

        Each sample is centred on an availability cell from
        :meth:`iter_reference_super_cells`.  Geostationary and microwave
        observations are matched independently for every reference cell via
        :meth:`find_slot_observations` using the cell's actual scan time as the
        temporal anchor, avoiding per-granule approximations.

        The scene covers ``scene_size × scene_size`` availability cells,
        i.e. ``scene_size * REF_CELL`` reference pixels and
        ``scene_size * OBS_CELL`` geo pixels per side.
        Temporal depth is ``n_steps + 1`` slots separated by ``step``;
        slot 0 is coincident with the reference scan time and higher slot
        indices reach further back in time.  Frames are stored oldest-first.
        """
        from numcodecs.zarr3 import Blosc

        output_path = Path(output_path)
        rng = np.random.default_rng(rng)
        frames = n_steps + 1
        step = np.timedelta64(step)
        ref_size = scene_size * REF_CELL
        obs_size = scene_size * OBS_CELL

        geo_satellites = list(self.geo_meta.keys())
        mw_satellites = list(self.microwave_meta.keys())
        geo_sensor_idx: Dict[str, int] = {s: i for i, s in enumerate(geo_satellites)}
        mw_sensor_idx: Dict[str, int] = {s: i for i, s in enumerate(mw_satellites)}

        ref_meta = self.reference_meta
        ref_times = ref_meta["time"].values         # (n_ref, 90, 180)
        ref_filenames = ref_meta["filename"].values

        # ---- Pass 1: resolve observations for every valid reference cell ----
        # Each reference cell may match multiple geo satellites; one scene is
        # created per satellite so that overlapping coverage is fully utilised.
        # scene tuple: (cell, scan_time, ref_path, geo_sat, geo_files, mw_files)
        # geo_files[s] / mw_files[s]: Optional[Path], oldest frame first.

        all_cells = list(self.iter_reference_super_cells(
            subsample=subsample, rng=rng,
            start_time=start_time, end_time=end_time,
        ))
        # Pre-spawn one child RNG per cell so workers are independent.
        child_rngs = rng.spawn(len(all_cells))

        scenes: List[Tuple] = []
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = [
                executor.submit(
                    _match_cell,
                    self, cell_scan_time, child_rng,
                    ref_times, ref_filenames,
                    n_steps, frames, step, scene_size, require_microwave,
                )
                for cell_scan_time, child_rng in zip(all_cells, child_rngs)
            ]
            for future in tqdm(futures, desc="Matching observations", unit="cell"):
                scenes.extend(future.result())

        n_scenes = len(scenes)
        LOGGER.info(
            "Extracting %d super-cell scenes (%d step(s)) to '%s'.",
            n_scenes, n_steps, output_path,
        )
        if n_scenes == 0:
            LOGGER.warning("No scenes to extract; nothing written.")
            return output_path

        # ---- Pass 2: allocate zarr store ----
        compressor = Blosc(cname="zstd", clevel=4)
        store = zarr.open_group(str(output_path), mode="w")
        store.attrs.update({
            "geo_satellites": geo_satellites,
            "mw_satellites": mw_satellites,
            "scene_size": scene_size,
            "n_steps": n_steps,
            "step_minutes": float(step / np.timedelta64(1, "m")),
            "ref_size": ref_size,
            "obs_size": obs_size,
            "resolution_ratio": RESOLUTION_RATIO,
            "normalized": self.normalize,
            "time_units": "nanoseconds since 1970-01-01",
        })
        store.create_array(
            "geo",
            shape=(n_scenes, N_SLOTS, frames, obs_size, obs_size),
            chunks=(1, N_SLOTS, frames, obs_size, obs_size),
            dtype=np.float32, fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "geo_channel", "step", "geo_y", "geo_x"),
        )
        store.create_array(
            "mw",
            shape=(n_scenes, N_MW_SLOTS, frames, ref_size, ref_size),
            chunks=(1, N_MW_SLOTS, frames, ref_size, ref_size),
            dtype=np.float32, fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "mw_channel", "step", "y", "x"),
        )
        store.create_array(
            "surface_precip",
            shape=(n_scenes, ref_size, ref_size),
            chunks=(1, ref_size, ref_size),
            dtype=np.float32, fill_value=np.nan, compressors=compressor,
            dimension_names=("scene", "y", "x"),
        )
        store.create_array(
            "coordinates", shape=(n_scenes, 2), dtype=np.int32,
            dimension_names=("scene", "row_col"),
        )
        store.create_array(
            "time", shape=(n_scenes,), dtype=np.int64,
            dimension_names=("scene",),
        )
        store.create_array(
            "step_time", shape=(n_scenes, frames), dtype=np.int64,
            dimension_names=("scene", "step"),
        )
        store.create_array(
            "geo_sensor", shape=(n_scenes, frames), dtype=np.int16, fill_value=-1,
            dimension_names=("scene", "step"),
        )
        store.create_array(
            "mw_sensor", shape=(n_scenes, frames), dtype=np.int16, fill_value=-1,
            dimension_names=("scene", "step"),
        )

        # ---- Pass 3: load and write each scene ----
        for i, ((row, col), scan_time, ref_path, geo_sat, geo_files, mw_files) in enumerate(
            tqdm(scenes, desc="Extracting scenes", unit="scene")
        ):
            dr = int(rng.integers(-position_jitter, position_jitter + 1)) if position_jitter > 0 else 0
            dc = int(rng.integers(-position_jitter, position_jitter + 1)) if position_jitter > 0 else 0
            r0_ref = min(
                max((row - scene_size // 2) * REF_CELL + dr, 0),
                N_CELLS_LAT * REF_CELL - ref_size,
            )
            c0_ref = min(
                max((col - scene_size // 2) * REF_CELL + dc, 0),
                N_CELLS_LON * REF_CELL - ref_size,
            )
            r0_obs = r0_ref * RESOLUTION_RATIO
            c0_obs = c0_ref * RESOLUTION_RATIO

            store["surface_precip"][i] = self._load_reference(
                ref_path, r0_ref, c0_ref, ref_size
            )
            store["coordinates"][i] = (row, col)
            store["time"][i] = np.datetime64(scan_time, "ns").astype("int64")
            store["step_time"][i] = np.array(
                [
                    (np.datetime64(scan_time, "ns") - (n_steps - s) * step).astype("int64")
                    for s in range(frames)
                ],
                dtype=np.int64,
            )

            geo_scene = np.full(
                (N_SLOTS, frames, obs_size, obs_size), np.nan, dtype=np.float32
            )
            mw_scene = np.full(
                (N_MW_SLOTS, frames, ref_size, ref_size), np.nan, dtype=np.float32
            )
            geo_sensor_arr = np.full(frames, -1, dtype=np.int16)
            mw_sensor_arr = np.full(frames, -1, dtype=np.int16)

            for s, geo_path in enumerate(geo_files):
                if geo_path is not None:
                    array = self._load_obs(geo_path, r0_obs, c0_obs, obs_size)
                    if self.normalize:
                        array = self._normalize(array, geo_sat)
                    geo_scene[:, s] = slot_observations(array, geo_sat)
                    geo_sensor_arr[s] = geo_sensor_idx.get(geo_sat, -1)

            for s, mw_path in enumerate(mw_files):
                if mw_path is not None:
                    sat_name = mw_path.parent.name
                    array = self._load_mw_obs(mw_path, r0_ref, c0_ref, ref_size)
                    if self.normalize:
                        array = self._normalize(array, sat_name)
                    mw_scene[:, s] = slot_observations(array, sat_name)
                    mw_sensor_arr[s] = mw_sensor_idx.get(sat_name, -1)

            store["geo"][i] = geo_scene
            store["mw"][i] = mw_scene
            store["geo_sensor"][i] = geo_sensor_arr
            store["mw_sensor"][i] = mw_sensor_arr

        LOGGER.info("Wrote %d super-cell scenes to '%s'.", n_scenes, output_path)
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
            and ``mw_sensor``, the index into ``microwave_satellites`` (listed in the
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
        sensor_index = {s: i for i, s in enumerate(self.microwave_satellites)}
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
            sensors=list(self.microwave_satellites),
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

        # Geo: satellite with the best tile coverage; priority breaks ties.
        geo_sensor, best_cov, best_prio = None, 0, -1
        for sensor, (_, files, avail) in geo_pool.items():
            cov = avail[:, cr0:cr1, cc0:cc1].reshape(len(files), -1).sum(axis=1)
            top = int(cov.max()) if cov.size else 0
            prio = get_satellite(sensor).priority
            if top > best_cov or (top == best_cov and prio > best_prio):
                geo_sensor, best_cov, best_prio = sensor, top, prio
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
                stats = norm_stats.get(self._sensor_type(geo_sensor))
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
            stats = norm_stats.get(self._sensor_type(sensor))
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

        Rows are the :data:`N_SLOTS` spectral slots (with the slot's band name
        and canonical wavelength) and columns are the dataset's input satellites;
        each cell shows the satellite channel mapped to that slot (its stored
        index and central wavelength), or a dash where the satellite has no such
        channel.

        Returns:
            The table as an HTML string (renders in a Jupyter notebook).
        """
        satellites = [s for s in self.input_satellites if get_satellite(s).is_geo]
        cells = {sat: ["" for _ in range(N_SLOTS)] for sat in satellites}
        for sat_name in satellites:
            sat = get_satellite(sat_name)
            for channel, slot in enumerate(sat.slots):
                if slot < 0:
                    continue
                wl = sat.channels[channel]
                entry = f"ch {channel} ({wl:g} &micro;m)"
                cells[sat_name][slot] = (
                    f"{cells[sat_name][slot]}<br>{entry}"
                    if cells[sat_name][slot]
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
            "Geostationary channel slots", slot_columns, satellites, cells
        )

    def mw_slot_table_html(self) -> str:
        """
        Render the microwave channel-to-slot assignment as an HTML table.

        Rows are the :data:`N_MW_SLOTS` frequency slots (with the slot's band
        name, frequency and polarization) and columns are the dataset's
        microwave satellites; each cell shows the satellite channel mapped to
        that slot (its stored index and frequency/polarization), or a dash.

        Returns:
            The table as an HTML string (renders in a Jupyter notebook).
        """
        satellites = [
            s for s in self.microwave_satellites if get_satellite(s).is_mw
        ]
        cells = {sat: ["" for _ in range(N_MW_SLOTS)] for sat in satellites}
        for sat_name in satellites:
            sat = get_satellite(sat_name)
            for channel, slot in enumerate(sat.slots):
                if slot < 0:
                    continue
                ch = sat.channels[channel]
                freq, offset, pol = ch
                label = f"{freq:g}&plusmn;{offset:g}" if offset else f"{freq:g}"
                entry = f"ch {channel} ({label} GHz {html.escape(pol)})"
                cells[sat_name][slot] = (
                    f"{cells[sat_name][slot]}<br>{entry}"
                    if cells[sat_name][slot]
                    else entry
                )
        slot_columns = [
            ("Slot", [str(s) for s in range(N_MW_SLOTS)]),
            ("Band", [html.escape(name) for name in MW_SLOT_NAMES]),
            ("Freq (GHz)", [f"{freq:g}" for freq, _, _ in MW_SLOTS]),
            ("Pol", [html.escape(pol) for _, _, pol in MW_SLOTS]),
        ]
        return self._slot_table_html(
            "Microwave channel slots", slot_columns, satellites, cells
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
    :meth:`ArgosData.extract_samples` or :meth:`ArgosData.extract_super_cell_samples`.
    Accepts either a path to a single ``.zarr`` store or a directory containing
    multiple ``.zarr`` stores; in the latter case all stores are concatenated into
    one logical dataset. Each item is the same ``(inputs, target)`` tuple a model
    consumes -- ``inputs`` holding the ``geo`` and ``mw`` tensors (and the tile
    ``coordinates``) and ``target`` the ``surface_precip`` tensor. Absent inputs
    are kept as ``NaN`` so the keys are always present (and batch cleanly); the
    model treats them as zeros.

    When a temporal step dimension is present its size-1 case is collapsed: if
    ``geo`` or ``mw`` are 4-D with a step axis of length 1, that axis is squeezed
    so the model sees the same 3-D shape as scenes without temporal context.

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
            path: Path of a ``.zarr`` store written by ``extract_samples``, or a
                directory containing multiple such stores (all ``.zarr`` entries
                found by glob are loaded and concatenated).
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

        # Discover store paths: single store or all *.zarr in a directory.
        p = self.path
        if p.suffix == ".zarr" or (p / ".zgroup").exists():
            self._store_paths: List[Path] = [p]
        else:
            self._store_paths = sorted(p.glob("*.zarr"))
            if not self._store_paths:
                raise FileNotFoundError(f"No .zarr stores found in '{p}'.")

        # Per-file sample counts and cumulative offsets for global index mapping.
        # Open briefly (just for shape metadata) before workers are forked.
        lengths = [
            zarr.open_group(str(sp), mode="r")["surface_precip"].shape[0]
            for sp in self._store_paths
        ]
        self._file_lengths: List[int] = lengths
        self._cumlen: np.ndarray = np.concatenate([[0], np.cumsum(lengths)])

    # ``DataLoader(worker_init_fn=...)`` helper to seed NumPy per worker (the
    # augmentation uses ``numpy.random``).
    worker_init_fn = staticmethod(worker_init_fn)

    @cached_property
    def _stores(self) -> List["zarr.Group"]:
        """Lazily opened zarr stores, one per file (opened per process for fork safety)."""
        return [zarr.open_group(str(p), mode="r") for p in self._store_paths]

    @property
    def store(self) -> "zarr.Group":
        """The first zarr store (backward-compatible single-store accessor)."""
        return self._stores[0]

    @property
    def sensors(self) -> List[str]:
        """The sensor names indexed by ``geo_sensor``/``mw_sensor``."""
        return list(self._stores[0].attrs.get("sensors", []))

    def __len__(self) -> int:
        return int(self._cumlen[-1])

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
        file_idx = int(np.searchsorted(self._cumlen[1:], index, side="right"))
        local_idx = index - int(self._cumlen[file_idx])
        store = self._stores[file_idx]

        coords = np.asarray(store["coordinates"][local_idx])
        geo = torch.from_numpy(np.asarray(store["geo"][local_idx]))
        mw = torch.from_numpy(np.asarray(store["mw"][local_idx]))
        target = torch.from_numpy(np.asarray(store["surface_precip"][local_idx]))

        # Collapse a size-1 temporal step axis so that single-frame stores produce
        # the same (channel, H, W) shape as non-temporal ones.
        if geo.ndim == 4 and geo.shape[1] == 1:
            geo = geo.squeeze(1)
        if mw.ndim == 4 and mw.shape[1] == 1:
            mw = mw.squeeze(1)

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
