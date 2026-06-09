"""
argos.channel_properties
=======================

Defines the ChanneProperties dataclass used to represent observation properties.
"""
from dataclasses import dataclass

import numpy as np


def pol_to_num(polarization: str) -> int:
    """
    Convert polarization to numeric representation.

    Args:
        polarization: The polarization 'none', h', 'v', 'qh', 'qv', 'other'

    Return:
        An integer representing the polarization class.
    """
    polarization = polarization.lower()
    if polarization == "none":
        return 0
    elif polarization == "h":
        return 1
    elif polarization == "v":
        return 2
    elif polarization == "qh":
        return 3
    elif polarization == "qv":
        return 4
    elif polarization == "other":
        return 5
    raise ValueError(
        f"Ecountered unsupported polarization '{polarization}'",
        polarization
    )


@dataclass
class ChannelProperties:
    wavelength: float
    offset: float
    polarization: str

    @property
    def properties(self) -> np.ndarray:
        """
        The numerical representation of the channel properties.
        """
        return np.array([
            self.wavelength,
            self.offset,
            pol_to_num(self.polarization)
        ])
