"""
argos.data.mrms
===============

Functionality to download and regrid NOAA MRMS (Multi-Radar Multi-Sensor)
1-hour gauge-corrected precipitation accumulations.

Product used: MultiSensor_QPE_01H_Pass2 (``pansat.products.ground_based.mrms.precip_1h_ms``)

Files are grib2 (optionally gzip-compressed), one per hour, at 0.01°
resolution covering the CONUS domain (lon 230°–300°E / -130°–-60°W,
lat 20°–55°N).  This module uses pansat to find and download files, opens
them directly via cfgrib (working around a read-only-array bug in pansat's
``open()``), aggregates each granule to the argos 0.05° grid by computing
per-cell means with ``scipy.stats.binned_statistic_2d``, and writes one zarr
file per hour.
"""
from datetime import datetime, timedelta
import gzip
import logging
from pathlib import Path
import tempfile
from typing import List

import pandas as pd

import numpy as np
from pansat.products.ground_based.mrms import precip_1h_ms
from pansat.time import TimeRange
from scipy.stats import binned_statistic_2d
import xarray as xr
import zarr
from numcodecs.zarr3 import Blosc

from argos.grids import get_default_grid

LOGGER = logging.getLogger(__name__)

# ── Output scale / encoding ────────────────────────────────────────────────
OUT_SCALE_FACTOR = 0.1   # stored value × 0.1 = mm
OUT_FILL_VALUE = np.int16(-1)

# 0.05° target grid (every other point of the default 0.025° global grid)
_GRID = get_default_grid()[::2, ::2]
_TGT_LONS_2D, _TGT_LATS_2D = _GRID.get_lonlats()
_TGT_LAT_1D: np.ndarray = _TGT_LATS_2D[:, 0]   # shape (3600,), descending
_TGT_LON_1D: np.ndarray = _TGT_LONS_2D[0, :]   # shape (7200,), ascending
_N_TGT_LAT, _N_TGT_LON = _GRID.shape

# ── Bin edges for binned_statistic_2d ─────────────────────────────────────
# lat is stored north-to-south; flip to ascending for bin computation, then
# flip the result back in _regrid().
_tgt_lat_asc = _TGT_LAT_1D[::-1]
_dlat = abs(_tgt_lat_asc[1] - _tgt_lat_asc[0])
_dlon = abs(_TGT_LON_1D[1] - _TGT_LON_1D[0])

_lat_edges = np.empty(_N_TGT_LAT + 1)
_lat_edges[0] = _tgt_lat_asc[0] - _dlat / 2
_lat_edges[1:] = _tgt_lat_asc + _dlat / 2

_lon_edges = np.empty(_N_TGT_LON + 1)
_lon_edges[0] = _TGT_LON_1D[0] - _dlon / 2
_lon_edges[1:] = _TGT_LON_1D + _dlon / 2


def _open_mrms(rec) -> xr.Dataset:
    """
    Open a MRMS grib2 (or gzip-compressed grib2) file record as an xarray
    Dataset.

    Works around a read-only array bug in ``pansat``'s ``MRMSProduct.open()``
    by decompressing and loading the file directly via cfgrib, then converting
    longitudes from 0–360° to –180–180° with an explicit copy.

    Args:
        rec: A pansat ``FileRecord`` for a ``precip_1h_ms`` granule.

    Returns:
        xarray Dataset with coordinates ``latitude`` and ``longitude`` (the
        latter in –180–180° range) and a ``precip_1h_ms`` data variable
        containing 1-hour accumulated precipitation in mm.
    """
    path = rec.local_path

    if path.suffix == ".gz":
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "data.grib2"
            with open(path, "rb") as fh:
                dest.write_bytes(gzip.decompress(fh.read()))
            ds = xr.load_dataset(str(dest), engine="cfgrib")
    else:
        ds = xr.load_dataset(str(path), engine="cfgrib")

    # Convert longitudes from 0–360 to –180–180.  Use .copy() because cfgrib
    # returns read-only coordinate arrays.
    lons = ds.longitude.data.copy()
    lons[lons > 180] -= 360.0
    ds = ds.assign_coords(longitude=lons)

    return ds.rename({"unknown": "precip_1h_ms"})


def _regrid(ds: xr.Dataset) -> np.ndarray:
    """
    Aggregate MRMS 0.01° data to the argos 0.05° target grid using
    ``scipy.stats.binned_statistic_2d`` (mean within each 0.05° cell).

    The MRMS domain covers approximately lon –130°––60°W, lat 20°–55°N;
    all target cells outside this domain remain ``nan``.

    Args:
        ds: xarray Dataset as returned by :func:`_open_mrms`.

    Returns:
        ``float32 (3600, 7200)`` on the argos 0.05° grid; ``nan`` for no data.
    """
    precip = ds["precip_1h_ms"].data  # (n_lat, n_lon), float32

    lat_1d = ds.latitude.data    # descending
    lon_1d = ds.longitude.data   # ascending, already –180–180

    # Build coordinate arrays for all pixels without a full meshgrid
    n_lat, n_lon = precip.shape
    lat_vals = np.repeat(lat_1d, n_lon)
    lon_vals = np.tile(lon_1d, n_lat)
    precip_vals = precip.ravel()

    # Mask missing values (GRIB missing value is a large sentinel float)
    valid = np.isfinite(precip_vals) & (precip_vals >= 0)
    if not valid.any():
        return np.full((_N_TGT_LAT, _N_TGT_LON), np.nan, dtype=np.float32)

    result_asc, _, _, _ = binned_statistic_2d(
        lat_vals[valid],
        lon_vals[valid],
        precip_vals[valid],
        statistic="mean",
        bins=[_lat_edges, _lon_edges],
    )
    # result_asc has lat in ascending order; flip to match north-to-south storage.
    return result_asc[::-1].astype(np.float32)


def _save_zarr(
    precip_grid: np.ndarray,
    end_time: datetime,
    output_file: Path,
) -> None:
    """
    Write a regridded MRMS granule to a zarr store.

    Args:
        precip_grid: ``float32 (3600, 7200)`` — 1-hour accumulated
            precipitation in mm on the argos 0.05° grid.
        end_time: End time of the accumulation period (UTC).
        output_file: Destination ``.zarr`` path.
    """
    valid = np.isfinite(precip_grid) & (precip_grid >= 0)
    scaled = np.full(precip_grid.shape, OUT_FILL_VALUE, dtype=np.int16)
    scaled[valid] = np.round(precip_grid[valid] / OUT_SCALE_FACTOR).astype(np.int16)

    store = zarr.open_group(str(output_file), mode="w")
    store.attrs["scale_factor"] = float(OUT_SCALE_FACTOR)
    store.attrs["fill_value"] = int(OUT_FILL_VALUE)
    store.attrs["units"] = "mm"
    store.attrs["long_name"] = "MRMS MultiSensor QPE 1-hour gauge-corrected precipitation"
    store.attrs["time_coverage_end"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    store.create_array(
        "latitude",
        data=_TGT_LAT_1D.astype(np.float32),
        dimension_names=("latitude",),
    )
    store.create_array(
        "longitude",
        data=_TGT_LON_1D.astype(np.float32),
        dimension_names=("longitude",),
    )
    store.create_array(
        "precip_1h",
        data=scaled,
        chunks=(512, 512),
        fill_value=int(OUT_FILL_VALUE),
        compressors=Blosc(cname="zstd", clevel=4),
        dimension_names=("latitude", "longitude"),
    )

    # Store end time as nanoseconds since Unix epoch
    time_ns = np.datetime64(end_time, "ns").astype(np.uint64)
    store.create_array("time", data=time_ns, dimension_names=())

    LOGGER.info("Wrote %s", output_file.name)


def extract_data(
    year: int,
    month: int,
    day: int,
    output_path: Path,
) -> None:
    """
    Download all MRMS ``precip_1h_ms`` granules for one calendar day,
    regrid them to the argos 0.05° grid, and write one zarr file per hour.

    Pansat is used to locate and download files; output is written to
    ``output_path/mrms/``.

    Args:
        year: Four-digit year.
        month: Month (1–12).
        day: Day of month.
        output_path: Root output directory.
    """
    out_dir = Path(output_path) / "mrms"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime(year, month, day)
    end_time = start_time + timedelta(days=1)

    LOGGER.info(
        "Fetching MRMS precip_1h_ms records for %04d-%02d-%02d …",
        year, month, day,
    )
    recs = list(precip_1h_ms.get(TimeRange(start_time, end_time)))
    if not recs:
        LOGGER.warning(
            "No MRMS precip_1h_ms records found for %04d-%02d-%02d.", year, month, day
        )
        return

    LOGGER.info("Found %d granules.", len(recs))

    for rec in recs:
        cov = precip_1h_ms.get_temporal_coverage(rec)
        granule_end_dt = pd.Timestamp(cov.end).to_pydatetime().replace(tzinfo=None)

        out_stem = granule_end_dt.strftime("mrms_%Y%m%d%H%M%S")
        output_file = out_dir / f"{out_stem}.zarr"

        if output_file.exists():
            LOGGER.debug("Skipping %s (output exists).", out_stem)
            continue

        try:
            LOGGER.info("Processing %s …", rec.filename)
            ds = _open_mrms(rec)
            grid_data = _regrid(ds)
            _save_zarr(grid_data, granule_end_dt, output_file)
        except Exception:
            LOGGER.exception("Error processing %s.", rec.filename)
            continue
