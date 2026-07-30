"""
argos.data.satellite
====================

TOML-based satellite definitions with channel-to-slot mapping.

Each satellite (a platform/sensor pair, e.g. ATMS on NOAA-20) is described
by a ``.toml`` file in the ``satellites/`` sub-directory next to this module.
The file stem is the satellite's identifier, which is also the name of the
per-satellite sub-directory used in the training-data tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


SATELLITES_DIR = Path(__file__).parent / "satellites"

# ---------------------------------------------------------------------------
# Canonical geostationary slot definitions
# ---------------------------------------------------------------------------
N_SLOTS = 17
EXTRA_SLOT = N_SLOTS - 1

# Canonical slot center wavelengths in micrometers (GOES ABI bands 1-16, then
# the catch-all extra visible slot which has no fixed wavelength).
SLOT_WAVELENGTHS: Tuple[float, ...] = (
    0.47, 0.64, 0.86, 1.37, 1.6, 2.2, 3.9, 6.2,
    6.9, 7.3, 8.4, 9.6, 10.3, 11.2, 12.3, 13.3, np.nan,
)
SLOT_NAMES: Tuple[str, ...] = (
    "blue", "red", "veggie", "cirrus", "snow/ice", "cloud particle",
    "shortwave IR", "upper WV", "mid WV", "lower WV", "cloud phase", "ozone",
    "clean IR window", "IR window", "dirty IR window", "CO2",
    "extra visible (green/HRV)",
)

# ---------------------------------------------------------------------------
# Canonical microwave slot definitions
# ---------------------------------------------------------------------------
N_MW_SLOTS = 16

# Canonical microwave slots: (frequency [GHz], 183-GHz offset [GHz], pol class).
MW_SLOTS: Tuple[Tuple[float, float, str], ...] = (
    (19.35, 0.0, "V"), (19.35, 0.0, "H"),
    (22.5, 0.0, "V"), (31.4, 0.0, "V"),
    (37.0, 0.0, "V"), (37.0, 0.0, "H"),
    (89.5, 0.0, "V"), (91.65, 0.0, "H"),
    (157.0, 0.0, "H"),
    (183.31, 1.0, "H"), (183.31, 1.8, "H"), (183.31, 3.0, "H"),
    (183.31, 4.5, "H"), (183.31, 6.8, "H"),
    (157.0, 0.0, "V"),
    (91.655, 0.0, "TMS"),
)
MW_SLOT_NAMES: Tuple[str, ...] = (
    "19V", "19H", "23V", "31V", "37V", "37H", "89V", "89H", "157H",
    "183±1", "183±1.8", "183±3", "183±4.5", "183±7",
    "157V",
    "91 (TMS)",
)

# ---------------------------------------------------------------------------
# Slot-building helpers
# ---------------------------------------------------------------------------

def _pol_class(pol: str) -> str:
    """Map a polarization label to its class (V/QV -> 'V', H/QH -> 'H')."""
    pol = pol.upper()
    if pol in ("V", "QV"):
        return "V"
    if pol in ("H", "QH"):
        return "H"
    return pol


def _build_geo_slots(
    wavelengths: Sequence[float],
    explicit: Optional[Sequence[Optional[int]]] = None,
) -> List[int]:
    """
    Assign geo channels to slots by nearest wavelength.

    ``explicit[i]``, when not ``None``, overrides auto-computation for channel
    ``i``. Channels that cannot be matched to any ABI slot are placed in
    :data:`EXTRA_SLOT`.
    """
    abi_slots = [(i, wl) for i, wl in enumerate(SLOT_WAVELENGTHS) if np.isfinite(wl)]
    result = [-1] * len(wavelengths)
    assigned: Dict[int, Tuple[int, float]] = {}
    for channel, wavelength in enumerate(wavelengths):
        if explicit is not None and explicit[channel] is not None:
            result[channel] = int(explicit[channel])
            continue
        slot, distance = min(
            ((i, abs(wavelength - wl)) for i, wl in abi_slots), key=lambda t: t[1]
        )
        if slot in assigned:
            if distance < assigned[slot][1]:
                result[assigned[slot][0]] = -1
                assigned[slot] = (channel, distance)
                result[channel] = slot
        else:
            assigned[slot] = (channel, distance)
            result[channel] = slot
    for channel in range(len(wavelengths)):
        if result[channel] == -1:
            result[channel] = EXTRA_SLOT
    return result


def _build_mw_slots(
    channels: Sequence[Union[int, Tuple[float, float, str]]],
    explicit: Optional[Sequence[Optional[int]]] = None,
) -> List[int]:
    """
    Assign microwave channels to slots by nearest (frequency, offset, pol).

    A channel given as a plain ``int`` is assigned that slot directly.
    ``explicit[i]``, when not ``None``, overrides auto-computation for channel
    ``i`` (useful when a channel has frequency info but a prescribed slot).
    """
    result = [-1] * len(channels)
    assigned: Dict[int, Tuple[int, float]] = {}
    for channel, spec in enumerate(channels):
        if explicit is not None and explicit[channel] is not None:
            result[channel] = int(explicit[channel])
            continue
        if isinstance(spec, int):
            result[channel] = spec
            continue
        freq, offset, pol = spec
        pc = _pol_class(pol)
        candidates = [
            (i, (freq - sf) ** 2 + (offset - so) ** 2)
            for i, (sf, so, sp) in enumerate(MW_SLOTS)
            if _pol_class(sp) == pc
        ]
        if not candidates:
            continue
        slot, distance = min(candidates, key=lambda t: t[1])
        if slot in assigned:
            if distance < assigned[slot][1]:
                result[assigned[slot][0]] = -1
                assigned[slot] = (channel, distance)
                result[channel] = slot
        else:
            assigned[slot] = (channel, distance)
            result[channel] = slot
    return result


# ---------------------------------------------------------------------------
# Satellite dataclass
# ---------------------------------------------------------------------------

GeoChannel = float  # center wavelength in micrometers
MWChannel = Tuple[float, float, str]  # (frequency GHz, offset GHz, polarization)
AnyChannel = Union[GeoChannel, MWChannel]


@dataclass
class Satellite:
    """
    A satellite instrument (platform + sensor) with channel-to-slot mapping.

    Attributes:
        name:     Identifier matching the TOML filename stem and the data
                  sub-directory name (e.g. ``"noaa20_atms"``).
        platform: Human-readable platform name (e.g. ``"NOAA-20"``).
        sensor:   Instrument name (e.g. ``"ATMS"``).
        type:     ``"geo"`` for geostationary imagers, ``"mw"`` for microwave
                  cross-track / conical scanners.
        priority: Integer preference when multiple satellites have overlapping
                  observations; higher wins.
        channels: Per-channel specifications -- a list of wavelengths (μm) for
                  geo satellites, or ``(frequency GHz, offset GHz, polarization)``
                  tuples for microwave satellites.
        slots:    Per-channel slot index into the common geo or MW slot set.
    """

    name: str
    platform: str
    sensor: str
    type: str
    priority: int
    channels: List[AnyChannel]
    slots: List[int]

    @property
    def is_geo(self) -> bool:
        return self.type == "geo"

    @property
    def is_mw(self) -> bool:
        return self.type == "mw"

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_slots(self) -> int:
        return N_SLOTS if self.is_geo else N_MW_SLOTS

    @classmethod
    def from_toml(cls, path: Path) -> "Satellite":
        """Load a satellite definition from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        name = path.stem
        sat_type = data["type"]
        if sat_type not in ("geo", "mw"):
            raise ValueError(
                f"Satellite '{name}': type must be 'geo' or 'mw', got '{sat_type}'."
            )

        raw_channels = data.get("channels", [])

        if sat_type == "geo":
            channels: List[AnyChannel] = [
                float(ch["wavelength"]) for ch in raw_channels
            ]
            explicit_slots = [ch.get("slot") for ch in raw_channels]
            slots = _build_geo_slots(channels, explicit_slots)  # type: ignore[arg-type]
        else:
            channels = []
            explicit_slots = []
            for ch in raw_channels:
                channels.append((
                    float(ch["frequency"]),
                    float(ch.get("offset", 0.0)),
                    str(ch["polarization"]),
                ))
                explicit_slots.append(ch.get("slot"))
            slots = _build_mw_slots(channels, explicit_slots)  # type: ignore[arg-type]

        return cls(
            name=name,
            platform=str(data["platform"]),
            sensor=str(data["sensor"]),
            type=sat_type,
            priority=int(data.get("priority", 0)),
            channels=channels,
            slots=slots,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Optional[Dict[str, Satellite]] = None


def load_all_satellites(
    satellites_dir: Optional[Path] = None,
) -> Dict[str, Satellite]:
    """Load all satellite definitions from the ``satellites/`` directory."""
    if satellites_dir is None:
        satellites_dir = SATELLITES_DIR
    return {
        path.stem: Satellite.from_toml(path)
        for path in sorted(satellites_dir.glob("*.toml"))
    }


def get_satellite(name: str) -> Satellite:
    """Return the named satellite, loading the registry on first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_all_satellites()
    if name not in _REGISTRY:
        raise KeyError(
            f"No satellite definition found for '{name}'. "
            f"Available: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]
