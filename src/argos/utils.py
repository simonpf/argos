"""
argus.utils
===========

Module providing shared utility functions.
"""
from pathlib import Path
from datetime import datetime

import numpy as np
from pansat.time import to_datetime


def get_filename(prefix: str, time: np.datetime64) -> str:
    """
    Return filename for a given time stamp.

    Args:
        prefix: The prefix to which to append the filename

    Return:
        A string containing the filename stem.

    """
    date = to_datetime(time)
    return date.strftime(f"{prefix}_%Y%m%d%H%M%S")


def get_file_date(path: Path) -> datetime:
    """
    Extract date from filename.

    Args:
        path: A Path object pointing to a file adhering to argus filenaming conventions.

    Return:
        A datetime object representing the time.
    """
    return datetime.strptime(Path(path).stem.split()[-1], "%Y%m%d%H%M%S")
