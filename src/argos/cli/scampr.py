"""
argos.cli.scampr
================

Command line interface for downloading and regridding NOAA Enterprise
Precipitation Estimation (ScamPR) rain rate data.
"""
import calendar
import logging
import sys
from pathlib import Path

import click

from argos.data.scampr import extract_data

LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("year", type=int)
@click.argument("month", type=click.IntRange(1, 12))
@click.argument("day", type=int)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("."),
    show_default=True,
    help="Root output directory.  Results are written to OUTPUT_PATH/scampr/.",
)
@click.option(
    "--keep-raw",
    is_flag=True,
    default=False,
    help=(
        "Retain the downloaded NetCDF4 source files in OUTPUT_PATH/scampr/raw/ "
        "instead of deleting them after processing."
    ),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose (DEBUG) logging.",
)
def scampr(
    year: int,
    month: int,
    day: int,
    output_path: Path,
    keep_raw: bool,
    verbose: bool,
) -> None:
    """Download and regrid NOAA Enterprise Precipitation (ScamPR) data.

    Downloads all 10-minute RRQPE GLB-2 (0.02°) granules for the requested
    calendar day from the public NOAA S3 bucket, aggregates each granule to
    the argos 0.05° global grid by computing per-cell means, and writes one
    zarr file per granule to OUTPUT_PATH/scampr/.

    \b
    Arguments:
      YEAR   Four-digit year (e.g. 2025)
      MONTH  Month number 1–12
      DAY    Day of month 1–31

    \b
    Examples:
      argos scampr 2025 2 10
      argos scampr 2025 2 10 --output-path /data/scampr --keep-raw --verbose
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate day for the given month/year
    _, last_day = calendar.monthrange(year, month)
    if not (1 <= day <= last_day):
        click.echo(
            f"Error: day {day} is out of range for {year}-{month:02d} "
            f"(valid: 1–{last_day}).",
            err=True,
        )
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Downloading ScamPR data for %04d-%02d-%02d → %s",
        year, month, day, output_path / "scampr",
    )

    try:
        extract_data(
            year=year,
            month=month,
            day=day,
            output_path=output_path,
            keep_raw=keep_raw,
        )
        LOGGER.info("ScamPR extraction completed successfully.")
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        LOGGER.error("Extraction failed: %s", exc)
        sys.exit(1)
