"""
argos.cli.gpm
=============

Command line interface for GPM data extraction.
"""
import calendar
import logging
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import Optional, List

import click
import numpy as np

from argos.data.gpm import (
    gpm_gmi_obs,
    gpm_gmi_ref,
    gpm_amsr2_ref,
    f16_ssmis_obs,
    f17_ssmis_obs,
    f18_ssmis_obs,
    f19_ssmis_obs,
    npp_atms_obs,
    noaa18_mhs_obs,
    noaa19_mhs_obs,
    metopa_mhs_obs,
    metopb_mhs_obs,
    metopc_mhs_obs,
    noaa20_atms_obs,
)


LOGGER = logging.getLogger(__name__)


SENSORS = {
    'gpm_gmi': gpm_gmi_obs,
    'f16_ssmis': f16_ssmis_obs,
    'f17_ssmis': f17_ssmis_obs,
    'f18_ssmis': f18_ssmis_obs,
    'f19_ssmis': f19_ssmis_obs,
    'npp_atms': npp_atms_obs,
    'noaa18_mhs': noaa18_mhs_obs,
    'noaa19_mhs': noaa19_mhs_obs,
    'metopa_mhs_obs': metopa_mhs_obs,
    'metopb_mhs_obs': metopb_mhs_obs,
    'metopc_mhs_obs': metopc_mhs_obs,
    'noaa20_atms': noaa20_atms_obs,
    'gprof_gmi': gpm_gmi_ref,
    'gprof_amsr2': gpm_amsr2_ref,
}


def extract_day_data(args):
    """
    Extract data for a single day. Used for multiprocessing.

    Args:
        args: Tuple of (sensor, year, month, day, step, output_path)
    """
    sensor, year, month, day, step, output_path = args
    try:
        LOGGER.info(f"Processing {sensor.name} data for {year}-{month:02d}-{day:02d}")
        sensor.extract_data(year, month, day, step, output_path)
        LOGGER.info(f"Completed {sensor.name} data for {year}-{month:02d}-{day:02d}")
    except Exception as e:
        LOGGER.exception(f"Failed to process {sensor.name} data for {year}-{month:02d}-{day:02d}: {e}")


def extract_gpm_data(
    sensor: str,
    year: int,
    month: int,
    days: Optional[List[int]] = None,
    step: int = 20,
    output_path: Path = Path("."),
    num_processes: int = 1
) -> None:
    """
    Extract GPM data for specified time period using multiprocessing.

    Args:
        sensor: Sensor name ('gpm_gmi' or 'gprof_gmi')
        year: Year to extract data for
        month: Month to extract data for
        days: Specific days to extract (if None, extract all days in month)
        step: Time step for data extraction in minutes.
        output_path: Directory to write output files
        num_processes: Number of processes to use
    """
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor: {sensor}. Available: {list(SENSORS.keys())}")

    sensor_obj = SENSORS[sensor]

    # Convert step to numpy timedelta64
    step_td = np.timedelta64(step, "m")

    # Determine days to process
    if days is None:
        _, last_day = calendar.monthrange(year, month)
        days = list(range(1, last_day + 1))

    # Prepare arguments for multiprocessing
    process_args = [
        (sensor_obj, year, month, day, step_td, output_path)
        for day in days
    ]

    if num_processes == 1:
        # Sequential processing
        for args in process_args:
            extract_day_data(args)
    else:
        # Parallel processing
        LOGGER.info(f"Processing {len(process_args)} days using {num_processes} processes")
        with Pool(processes=num_processes) as pool:
            pool.map(extract_day_data, process_args)


@click.command()
@click.argument("sensor", type=click.Choice(list(SENSORS.keys())))
@click.argument("year", type=int)
@click.argument("month", type=click.IntRange(1, 12))
@click.option(
    "--days",
    type=int,
    multiple=True,
    help="Specific days to extract (default: all days in month). Can be used multiple times."
)
@click.option(
    "--step",
    type=int,
    default=20,
    show_default=True,
    help="Time step for data extraction in minutes."
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("."),
    show_default=True,
    help="Output directory for extracted data."
)
@click.option(
    "--processes",
    type=int,
    default=1,
    show_default=True,
    help="Number of processes to use for parallel extraction."
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging."
)
def gpm(
    sensor: str,
    year: int,
    month: int,
    days: tuple[int, ...],
    step: int,
    output_path: Path,
    processes: int,
    verbose: bool
):
    """Extract GPM satellite observation data.

    SENSOR: Sensor to extract data from (gpm_gmi, gprof_gmi)
    YEAR: Year to extract data for
    MONTH: Month to extract data for (1-12)
    """
    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Validate days if specified
    if days:
        _, last_day = calendar.monthrange(year, month)
        for day in days:
            if not (1 <= day <= last_day):
                click.echo(f"Error: Invalid day {day} for {year}-{month:02d}", err=True)
                sys.exit(1)

    # Convert days tuple to list (None if empty)
    days_list = list(days) if days else None

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        extract_gpm_data(
            sensor=sensor,
            year=year,
            month=month,
            days=days_list,
            step=step,
            output_path=output_path,
            num_processes=processes
        )
        LOGGER.info("GPM data extraction completed successfully")
    except KeyboardInterrupt:
        LOGGER.info("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Error during extraction: {e}")
        sys.exit(1)


def main():
    """Legacy main function for backward compatibility."""
    gpm()


if __name__ == "__main__":
    main()
