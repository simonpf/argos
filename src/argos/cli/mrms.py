"""
argos.cli.mrms
==============

Command line interface for downloading and regridding MRMS 1-hour
gauge-corrected precipitation accumulations.
"""
import calendar
import logging
import sys
from pathlib import Path

import click

from argos.data.mrms import extract_data

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
    help="Root output directory.  Results are written to OUTPUT_PATH/mrms/.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose (DEBUG) logging.",
)
def mrms(
    year: int,
    month: int,
    day: int,
    output_path: Path,
    verbose: bool,
) -> None:
    """Download and regrid MRMS 1-hour gauge-corrected precipitation data.

    Uses pansat to locate and download MultiSensor_QPE_01H_Pass2 granules for
    the requested calendar day, aggregates each hourly file to the argos 0.05°
    global grid by computing per-cell means, and writes one zarr file per hour
    to OUTPUT_PATH/mrms/.

    The output variable ``precip_1h`` stores 1-hour accumulated precipitation
    in mm, encoded as int16 with scale_factor=0.1.  Data outside the MRMS
    CONUS domain (approximately lon –130°––60°W, lat 20°–55°N) are NaN.

    \b
    Arguments:
      YEAR   Four-digit year (e.g. 2021)
      MONTH  Month number 1–12
      DAY    Day of month 1–31

    \b
    Examples:
      argos mrms 2021 6 15
      argos mrms 2021 6 15 --output-path /data --verbose
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _, last_day = calendar.monthrange(year, month)
    if not (1 <= day <= last_day):
        click.echo(
            f"Error: day {day} is out of range for {year}-{month:02d} "
            f"(valid: 1–{last_day}).",
            err=True,
        )
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        extract_data(year=year, month=month, day=day, output_path=output_path)
        LOGGER.info("MRMS extraction completed successfully.")
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        LOGGER.error("Extraction failed: %s", exc)
        sys.exit(1)
