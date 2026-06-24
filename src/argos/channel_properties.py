"""
argos.channel_properties
=======================

Defines the ChanneProperties dataclass used to represent observation properties.
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import constants


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


def freq_to_wavelength(freq: Union[float, np.ndarray]):
    """
    Convert frequency to wavelength.
    """
    return constants.c / freq


@dataclass
class ChannelProperties:
    wavelength: float # Wavelength in m
    bandwidth: float # Bandwidth in m
    offset: float # The channel offest
    polarization: str # A string representing the polarization

    @property
    def properties(self) -> np.ndarray:
        """
        The numerical representation of the channel properties.
        """
        return np.array([
            self.wavelength,
            self.bandwidth,
            self.offset,
            pol_to_num(self.polarization)
        ])


@dataclass
class MWChannelProperties:
    frequency: float # Wavelength in GHz
    bandwidth: float # Bandwidth in MHz
    offset: float # The channel offest in GHz
    polarization: str # A string representing the polarization

    @property
    def properties(self) -> np.ndarray:
        """
        The numerical representation of the channel properties.
        """
        wavelength = freq_to_wavelength(self.frequency * 1e9)
        lower = freq_to_wavelength((self.frequency - self.bandwidth * 1e-3) * 1e9)
        upper = freq_to_wavelength((self.frequency + self.bandwidth * 1e-3) * 1e9)
        bandwidth = upper - lower

        lower = freq_to_wavelength((self.frequency - self.offset) * 1e9)
        upper = freq_to_wavelength((self.frequency + self.offset) * 1e9)
        offset = (upper - lower) / 2.0

        return np.array([
            wavelength,
            bandwidth,
            offset,
            pol_to_num(self.polarization)
        ])
