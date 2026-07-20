"""
argos.data.scampr
=================

Functionality to download and regrid NOAA Enterprise Precipitation Estimation
rain rate data (the product previously known as ScamPR / RRQPE).

Data source: NOAA Open Data Dissemination (NODD) S3 bucket
  s3://noaa-enterprise-rainrate-pds/BLEND/RainRate-Blend-INST/

Files are netCDF4, one per 10-minute interval, at 0.02° resolution covering
latitudes -60° to 70°N and all longitudes. This module downloads all GLB-2
(0.02°) files for a requested day, aggregates them to the argos 0.05° grid by
computing per-cell means, and writes one zarr file per 10-minute granule.
"""
from datetime import datetime
import logging
from pathlib import Path
import re
import tempfile
import urllib.request
from typing import List, Optional, Tuple

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import netCDF4 as nc
import numpy as np
from scipy.stats import binned_statistic_2d
import zarr
from numcodecs.zarr3 import Blosc

from argos.grids import get_default_grid

LOGGER = logging.getLogger(__name__)

BUCKET_NAME = "noaa-enterprise-rainrate-pds"
PRODUCT_PREFIX = "BLEND/RainRate-Blend-INST"

# ── Source grid metadata (read from file global attributes) ────────────────
# Row 0 = northernmost latitude (70°N), last row = -60°S. Orientation
# confirmed from DQF: all pixels at row 0 carry the "LZA-degraded" flag,
# consistent with 70°N being at the product's northern limit.
SRC_LAT_NORTH = 70.0
SRC_LAT_SOUTH = -60.0
SRC_LON_WEST = -180.0
SRC_RES = 0.02          # degrees
SRC_N_ROWS = 6501       # rows (lat)
SRC_N_COLS = 18000      # cols (lon)
SRC_SCALE_FACTOR = 0.1  # int16 → mm/hr
SRC_FILL_VALUE = np.int16(-9990)

# ── Output scale / encoding ────────────────────────────────────────────────
OUT_SCALE_FACTOR = 0.1   # stored value × 0.1 = mm/hr
OUT_FILL_VALUE = np.int16(-1)

# 0.05° target grid (every other point of the default 0.025° global grid)
_GRID = get_default_grid()[::2, ::2]

# Pre-compute regridding look-up tables once at module load time.
# The target grid stores rows north-to-south (lat[0] ≈ +90°, lat[-1] ≈ -90°).
_TGT_LONS_2D, _TGT_LATS_2D = _GRID.get_lonlats()
_TGT_LAT_1D: np.ndarray = _TGT_LATS_2D[:, 0]   # shape (3600,), descending
_TGT_LON_1D: np.ndarray = _TGT_LONS_2D[0, :]   # shape (7200,), ascending
_N_TGT_LAT, _N_TGT_LON = _GRID.shape

# Source 1-D coordinate arrays (fixed geometry)
_SRC_LAT = SRC_LAT_NORTH - np.arange(SRC_N_ROWS) * SRC_RES   # descending
_SRC_LON = SRC_LON_WEST + np.arange(SRC_N_COLS) * SRC_RES    # ascending

# ── Bin edges for binned_statistic_2d ─────────────────────────────────────
# binned_statistic_2d requires ascending bin edges.  The target lat array is
# stored north-to-south (descending), so we flip it first and flip the result
# back afterwards inside _regrid().
_tgt_lat_asc = _TGT_LAT_1D[::-1]   # ascending copy for bin computation
_dlat = abs(_tgt_lat_asc[1] - _tgt_lat_asc[0])
_dlon = abs(_TGT_LON_1D[1] - _TGT_LON_1D[0])

_lat_edges = np.empty(_N_TGT_LAT + 1)
_lat_edges[0] = _tgt_lat_asc[0] - _dlat / 2
_lat_edges[1:] = _tgt_lat_asc + _dlat / 2

_lon_edges = np.empty(_N_TGT_LON + 1)
_lon_edges[0] = _TGT_LON_1D[0] - _dlon / 2
_lon_edges[1:] = _TGT_LON_1D + _dlon / 2


# ── Timestamp regex ────────────────────────────────────────────────────────
_TS_RE = re.compile(r"_s(\d{14})\d?_")


def _parse_start_time(filename: str) -> Optional[datetime]:
    """Return the granule start time parsed from a RRQPE filename."""
    m = _TS_RE.search(filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return None


def list_files(year: int, month: int, day: int) -> List[str]:
    """
    Return a sorted list of S3 keys for all GLB-2 (0.02°) RRQPE granules on
    the requested calendar day.

    Args:
        year: Four-digit year.
        month: Month (1–12).
        day: Day of month.

    Returns:
        List of S3 object keys, e.g.
        ``"BLEND/RainRate-Blend-INST/2025/02/10/15/RRQPE-INST-GLB-2_…nc"``.
    """
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = f"{PRODUCT_PREFIX}/{year}/{month:02d}/{day:02d}/"
    paginator = client.get_paginator("list_objects_v2")

    keys: List[str] = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            if "GLB-2" in key and key.endswith(".nc"):
                keys.append(key)

    return sorted(keys)


def _download_key(key: str, dest_dir: Path) -> Path:
    """Download one S3 object (anonymous, public bucket) to *dest_dir*."""
    url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"
    filename = Path(key).name
    local_path = dest_dir / filename
    if local_path.exists():
        LOGGER.debug("Already cached: %s", filename)
        return local_path
    LOGGER.info("Downloading %s …", filename)
    urllib.request.urlretrieve(url, local_path)
    return local_path


def _read_rain_rate(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open a RRQPE NetCDF4 file and return ``(rain_rate, valid)`` arrays.

    Args:
        path: Local path to the ``.nc`` file.

    Returns:
        rain_rate: ``float32`` array of shape ``(6501, 18000)`` in mm/hr.
            Invalid/missing pixels are ``nan``.
        valid: ``bool`` array of the same shape; ``True`` where data exist.
    """
    with nc.Dataset(str(path)) as ds:
        raw: np.ndarray = ds.variables["RRQPE"][:]  # int16 masked array
        scale: float = float(ds.variables["RRQPE"].scale_factor)
        fill = SRC_FILL_VALUE

    raw_data: np.ndarray = np.array(raw, dtype=np.int16)
    valid: np.ndarray = raw_data != fill

    rain_rate = np.where(valid, raw_data.astype(np.float32) * scale, np.nan).astype(
        np.float32
    )
    return rain_rate, valid


def _regrid(rain_rate: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Aggregate the 0.02° source array to the argos 0.05° target grid using
    ``scipy.stats.binned_statistic_2d`` (mean within each 0.05° cell).

    Only valid pixels are passed to the function; cells with no valid source
    pixels remain ``nan``.

    Args:
        rain_rate: ``float32 (6501, 18000)`` in mm/hr.
        valid: ``bool (6501, 18000)``.

    Returns:
        ``float32 (3600, 7200)`` on the argos 0.05° grid; ``nan`` for no data.
    """
    # Extract coordinates and values for valid pixels only, avoiding a full
    # (6501 × 18000) meshgrid.
    row_idx, col_idx = np.where(valid)
    lat_vals = _SRC_LAT[row_idx]
    lon_vals = _SRC_LON[col_idx]
    rain_vals = rain_rate[row_idx, col_idx]

    result_asc, _, _, _ = binned_statistic_2d(
        lat_vals,
        lon_vals,
        rain_vals,
        statistic="mean",
        bins=[_lat_edges, _lon_edges],
    )
    # result_asc is (n_tgt_lat, n_tgt_lon) with lat in ascending order.
    # Flip rows to match the north-to-south storage convention.
    return result_asc[::-1].astype(np.float32)


def _save_zarr(rain_rate_grid: np.ndarray, start_time: datetime, output_file: Path) -> None:
    """
    Write a regridded granule to a zarr store.

    Args:
        rain_rate_grid: ``float32 (3600, 7200)`` on the argos 0.05° grid.
        start_time: Granule start time (UTC).
        output_file: Destination ``.zarr`` path (created, not appended).
    """
    # Encode as int16 with scale factor 0.1; fill = OUT_FILL_VALUE
    valid = np.isfinite(rain_rate_grid) & (rain_rate_grid >= 0)
    scaled = np.full(rain_rate_grid.shape, OUT_FILL_VALUE, dtype=np.int16)
    scaled[valid] = np.round(rain_rate_grid[valid] / OUT_SCALE_FACTOR).astype(np.int16)

    store = zarr.open_group(str(output_file), mode="w")
    store.attrs["scale_factor"] = float(OUT_SCALE_FACTOR)
    store.attrs["fill_value"] = int(OUT_FILL_VALUE)
    store.attrs["units"] = "mm/h"
    store.attrs["long_name"] = "NOAA Enterprise Rain Rate QPE (ScamPR blend)"
    store.attrs["time_coverage_start"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

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
        "surface_precip",
        data=scaled,
        chunks=(512, 512),
        fill_value=int(OUT_FILL_VALUE),
        compressors=Blosc(cname="zstd", clevel=4),
        dimension_names=("latitude", "longitude"),
    )

    # Store time as nanoseconds since Unix epoch (consistent with other modules)
    time_ns = np.datetime64(start_time, "ns").astype(np.uint64)
    store.create_array("time", data=time_ns, dimension_names=())

    LOGGER.info("Wrote %s", output_file.name)


def extract_data(
    year: int,
    month: int,
    day: int,
    output_path: Path,
    keep_raw: bool = False,
) -> None:
    """
    Download all RRQPE GLB-2 granules for one calendar day, regrid them to
    the argos 0.05° grid, and write one zarr file per 10-minute granule.

    Args:
        year: Four-digit year.
        month: Month (1–12).
        day: Day of month.
        output_path: Root output directory.  Results are written to
            ``output_path/scampr/``.
        keep_raw: If ``True``, retain the downloaded NetCDF4 files inside
            ``output_path/scampr/raw/``.  Otherwise they are deleted after
            processing.
    """
    out_dir = Path(output_path) / "scampr"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list_files(year, month, day)
    if not keys:
        LOGGER.warning(
            "No RRQPE GLB-2 files found for %04d-%02d-%02d.", year, month, day
        )
        return

    LOGGER.info(
        "Found %d GLB-2 granules for %04d-%02d-%02d.", len(keys), year, month, day
    )

    if keep_raw:
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory()
        raw_dir = Path(tmp_ctx.name)

    try:
        for key in keys:
            filename = Path(key).name
            start_time = _parse_start_time(filename)
            if start_time is None:
                LOGGER.warning("Could not parse timestamp from %s – skipping.", filename)
                continue

            out_stem = start_time.strftime("scampr_%Y%m%d%H%M%S")
            output_file = out_dir / f"{out_stem}.zarr"

            if output_file.exists():
                LOGGER.debug("Skipping %s (output exists).", out_stem)
                continue

            try:
                local_nc = _download_key(key, raw_dir)
                rain_rate, valid = _read_rain_rate(local_nc)
                grid_data = _regrid(rain_rate, valid)
                _save_zarr(grid_data, start_time, output_file)

                if not keep_raw:
                    local_nc.unlink(missing_ok=True)

            except Exception:
                LOGGER.exception("Error processing %s.", filename)
                continue
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
