"""
argos.cli.goes
==============

Command line interface for GOES data extraction.
"""
import calendar
from datetime import datetime
import logging
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import Optional, List

import click
import numpy as np

from argos.data.goes import goes_16_obs, goes_18_obs, goes_19_obs


LOGGER = logging.getLogger(__name__)


SATELLITES = {
    'goes16': goes_16_obs,
    'goes18': goes_18_obs,
    'goes19': goes_19_obs
}


def extract_day_data(args):
    """
    Extract data for a single day. Used for multiprocessing.
    
    Args:
        args: Tuple of (satellite, year, month, day, step, output_path)
    """
    satellite, year, month, day, step, output_path = args
    try:
        LOGGER.info(f"Processing {satellite.name} data for {year}-{month:02d}-{day:02d}")
        satellite.extract_data(year, month, day, step, output_path)
        LOGGER.info(f"Completed {satellite.name} data for {year}-{month:02d}-{day:02d}")
    except Exception as e:
        LOGGER.exception(f"Failed to process {satellite.name} data for {year}-{month:02d}-{day:02d}: {e}")


def extract_goes_data(
    satellite: str,
    year: int,
    month: int,
    days: Optional[List[int]] = None,
    step: int = 20,
    output_path: Path = Path("."),
    num_processes: int = 1
) -> None:
    """
    Extract GOES data for specified time period using multiprocessing.
    
    Args:
        satellite: Satellite name ('goes16', 'goes18', or 'goes19')
        year: Year to extract data for
        month: Month to extract data for
        days: Specific days to extract (if None, extract all days in month)
        step: Time step for data extraction in minutes.
        output_path: Directory to write output files
        num_processes: Number of processes to use
    """
    if satellite not in SATELLITES:
        raise ValueError(f"Unknown satellite: {satellite}. Available: {list(SATELLITES.keys())}")
    
    satellite_obj = SATELLITES[satellite]
    
    # Convert step to numpy timedelta64
    step_td = np.timedelta64(step, "m")
    
    # Determine days to process
    if days is None:
        _, last_day = calendar.monthrange(year, month)
        days = list(range(1, last_day + 1))
    
    # Prepare arguments for multiprocessing
    process_args = [
        (satellite_obj, year, month, day, step_td, output_path)
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
@click.argument("satellite", type=click.Choice(list(SATELLITES.keys())))
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
def goes(
    satellite: str,
    year: int,
    month: int,
    days: tuple[int, ...],
    step: int,
    output_path: Path,
    processes: int,
    verbose: bool
):
    """Extract GOES satellite observation data.
    
    SATELLITE: Satellite to extract data from (goes16, goes18, goes19)
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
        extract_goes_data(
            satellite=satellite,
            year=year,
            month=month,
            days=days_list,
            step=step,
            output_path=output_path,
            num_processes=processes
        )
        LOGGER.info("GOES data extraction completed successfully")
    except KeyboardInterrupt:
        LOGGER.info("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Error during extraction: {e}")
        sys.exit(1)


def main():
    """Legacy main function for backward compatibility."""
    goes()


if __name__ == "__main__":
    main()
