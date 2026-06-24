"""
argos.data.gpm
==============

Functionality to extract GPM L1C observations.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import pansat
from pansat.time import TimeRange
from pansat.utils import get_resample_info, resample_data
from pansat.products.satellite.gpm import (
    l1c_r_xcal2016c_gpm_gmi_v07a,
    l1c_r_xcal2016c_gpm_gmi_v07b,
    l1c_xcal2016v_gcomw1_amsr2_v07a,
    l1c_xcal2019v_noaa20_atms_v07a,
    l1c_xcal2019v_npp_atms_v07a,
    l1c_xcal2016v_noaa18_mhs_v07a,
    l1c_xcal2016v_noaa19_mhs_v07a,
    l1c_xcal2021v_metopa_mhs_v07a,
    l1c_xcal2016v_metopb_mhs_v07a,
    l1c_xcal2019v_metopc_mhs_v07a,
    l1c_xcal2021v_f16_ssmis_v07a,
    l1c_xcal2021v_f16_ssmis_v07b,
    l1c_xcal2021v_f17_ssmis_v07a,
    l1c_xcal2021v_f17_ssmis_v07b,
    l1c_xcal2021v_f18_ssmis_v07a,
    l1c_xcal2021v_f18_ssmis_v07b,
    l1c_xcal2021v_f19_ssmis_v07a,
    l2a_gprof_gpm_gmi_v08a,
    l2a_gprof_gcomw1_amsr2_v08a
)
import xarray as xr
import zarr
from numcodecs.zarr3 import Blosc

from argos.grids import get_default_grid
from argos.utils import get_filename
from argos.channel_properties import MWChannelProperties


grid = get_default_grid()[::2, ::2]


LOGGER = logging.getLogger(__file__)

_CHANNEL_REGEXP = re.compile(r"([\d\.]+)\s*(?:GHz)?(?:\+-)?s*(?:\+\/-)?s*([\d\.]*)\s*(?:GHz)?\s*(\w+)-Pol")
_CHANNEL_REGEXP_NO_POL = re.compile(r"([\d\.]+)\s*(?:\+-)?s*(?:\+\/-)?s*([\d\.]*)\s*(?:GHz)")


class GPMObs:
    """
    Class for extracting and loading GPM L1C observations.
    """
    
    def __init__(
            self,
            name: str,
            products: List[pansat.products.Product],
            radius_of_influence: float,
            channel_properties: Dict[Tuple[int, int], MWChannelProperties]
    ):
        self.name = name
        self.products = products
        self.radius_of_influence = radius_of_influence
        self.channel_properties = channel_properties


    @property
    def n_chans(self) -> int:
        """
        The number of channels of the GPM sensor.
        """
        return len(self.channel_properties)


    def extract_data(
            self,
            year: int,
            month: int,
            day: int,
            step: np.timedelta64,
            output_path: Path
    ) -> None:
        """
        Extract GPM training data for a given day.

        Args:
            year: The year.
            month: The month.
            day: The day.
            step: The timestep at which to extract observations.
            output_path: The path to which to write the results.
        """
        output_path = Path(output_path) / self.name
        output_path.mkdir(exist_ok=True)

        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(days=1)
        time_range = TimeRange(start_time, end_time)

        for product in self.products:
            recs = product.get(TimeRange(start_time, end_time))
            for rec in recs:
                try:
                    l1c_obs = product.open(rec)

                    if "latitude" in l1c_obs:
                        vars = ["latitude", "longitude", "tbs", "channels"]
                        l1c_obs = l1c_obs.rename(dict([(name, name + "_s1") for name in vars]))

                    swath_ind = 1
                    chan_ind_tot = 0

                    while f"latitude_s{swath_ind}" in l1c_obs:

                        freqs = []
                        offsets = []
                        pols = []

                        # Parse channel properties from metadata
                        for match in _CHANNEL_REGEXP.findall(l1c_obs[f"tbs_s{swath_ind}"].attrs["LongName"]):
                            freq, offs, pol = match
                            freqs.append(float(freq))
                            offsets.append(float(offs) if offs else 0.0)
                            pols.append(pol)

                        # Special case for AMSU-B
                        if len(freqs) == 0:
                            for match in _CHANNEL_REGEXP_NO_POL.findall(l1c_obs[f"tbs_s{swath_ind}"].attrs["LongName"]):
                                freq, offs = match
                                freqs.append(float(freq))
                                offsets.append(float(offs) if offs else 0.0)
                                pols.append("qv")

                        if len(freqs) == 0:
                            swath_ind += 1
                            continue

                        # Extract swath data
                        swath_data = l1c_obs[[
                            f"longitude_s{swath_ind}",
                            f"latitude_s{swath_ind}",
                            f"tbs_s{swath_ind}",
                            f"channels_s{swath_ind}",
                            "scan_time"
                        ]].reset_coords("scan_time")

                        swath_data = swath_data.rename({
                            f"longitude_s{swath_ind}": "longitude",
                            f"latitude_s{swath_ind}": "latitude",
                            f"tbs_s{swath_ind}": "obs"
                        })

                        swath_data["obs"] = swath_data["obs"].astype(np.float32)

                        info = None
                        for chan_ind, (freq, offset, pol) in enumerate(zip(freqs, offsets, pols)):

                            if info is None:
                                info = get_resample_info(
                                    swath_data,
                                    grid,
                                    radius_of_influence=self.radius_of_influence
                                )

                            # Resample to grid
                            try:
                                data_r = resample_data(
                                    swath_data[{f"channels_s{swath_ind}": chan_ind}],
                                    grid,
                                    info=info,
                                    radius_of_influence=self.radius_of_influence
                                )
                            except ValueError:
                                swath_ind += 1
                                continue

                            time_range = rec.temporal_coverage
                            output_file = output_path / get_filename(self.name, time_range.start)
                            output_file = output_file.with_suffix(".zarr")

                            LOGGER.info(
                                f"Processing channel {chan_ind_tot + 1} / {self.n_chans} from {rec.local_path.name}."
                            )

                            ch_obs = data_r[f"obs"]
                            ch_obs = ch_obs.astype(np.float32)

                            # Quality control
                            valid = (ch_obs > 0) & (ch_obs < 400) & np.isfinite(ch_obs)
                            availability = valid.coarsen(longitude=40, latitude=40).sum()

                            ch_obs = ch_obs.data
                            ch_obs[~valid] = np.nan

                            ch_obs_scaled = (ch_obs / 0.1).astype(np.int16)
                            ch_obs_scaled[~valid] = -1

                            if output_file.exists():
                                # Open existing zarr store and update the channel
                                store = zarr.open_group(str(output_file), mode='r+')

                                # Update observations for this channel
                                store['obs'][chan_ind_tot, :, :] = ch_obs_scaled
                                props = self.channel_properties[(swath_ind, chan_ind)].properties
                                store['channel_properties'][chan_ind_tot, :] = props

                                # Update statistics for this channel (use original float values)
                                ch_obs_valid = ch_obs[valid]
                                if len(ch_obs_valid) > 0:
                                    store['obs_min'][chan_ind_tot] = ch_obs_valid.min()
                                    store['obs_max'][chan_ind_tot] = ch_obs_valid.max()
                                    store['obs_sum'][chan_ind_tot] = ch_obs_valid.sum()
                                    store['obs_cts'][chan_ind_tot] = valid.sum()

                                # Update availability (combine with existing)
                                existing_availability = store['availability'][:]
                                store['availability'][:] = 0 < (existing_availability + availability.data)

                                LOGGER.info(
                                    f"Updated {chan_ind_tot + 1} / {self.n_chans} in zarr file '{output_file}'."
                                )

                            else:
                                # Create new zarr store using low-level operations
                                lons, lats = grid.get_lonlats()
                                lons = lons[0]
                                lats = lats[:, 0]

                                # Get shape from current channel data
                                channel_shape = data_r.obs.data.shape
                                obs_shape = (self.n_chans,) + channel_shape

                                # Create zarr group
                                store = zarr.open_group(str(output_file), mode='w')
                                store.attrs["scale_factor"] = 0.1
                                store.attrs["fill_value"] = -1

                                # Store coordinate arrays
                                store.create_array(
                                    'longitude',
                                    data=lons.astype(np.float32),
                                    dimension_names=("longitude",),
                                )
                                store.create_array(
                                    'latitude',
                                    data=lats.astype(np.float32),
                                    dimension_names=("latitude",),
                                )

                                # Create observation array without initializing data (lazy allocation)
                                store.create_array(
                                    'obs',
                                    shape=obs_shape,
                                    chunks=(1, min(512, channel_shape[0]), min(512, channel_shape[1])),
                                    dtype=np.int16,
                                    fill_value=-1,
                                    compressors=Blosc(cname='zstd', clevel=4),
                                    dimension_names=("channel", "latitude", "longitude"),
                                )

                                # Create statistics arrays
                                dims = ("channel",)
                                store.create_array(
                                    'obs_min',
                                    shape=(self.n_chans,),
                                    dtype=np.float32,
                                    fill_value=np.nan,
                                    dimension_names=dims
                                )
                                store.create_array(
                                    'obs_max',
                                    shape=(self.n_chans,),
                                    dtype=np.float32,
                                    fill_value=np.nan,
                                    dimension_names=dims
                                )
                                store.create_array(
                                    'obs_sum',
                                    shape=(self.n_chans,),
                                    dtype=np.float32,
                                    fill_value=np.nan,
                                    dimension_names=dims
                                )
                                store.create_array(
                                    'obs_cts',
                                    shape=(self.n_chans,),
                                    dtype=np.int32,
                                    fill_value=0,
                                    dimension_names=dims
                                )

                                # Store availability
                                store.create_array(
                                    'availability',
                                    data=(0 < availability.data).astype(np.int8),
                                    compressor=Blosc(cname='zstd', clevel=4),
                                    dimension_names=("latitude_80", "longitude_80")
                                )

                                # Store channel properties
                                store.create_array(
                                    'channel_properties',
                                    shape=(self.n_chans, 4),
                                    dtype=np.float32,
                                    dimension_names=("channel", "feature")
                                )

                                # Now write the current channel data
                                store['obs'][chan_ind, :, :] = ch_obs_scaled
                                props = self.channel_properties[(swath_ind, chan_ind)].properties
                                store['channel_properties'][chan_ind_tot, :] = props

                                # Update statistics for this channel (use original float values)
                                ch_obs_valid = ch_obs[valid]
                                if len(ch_obs_valid) > 0:
                                    store['obs_min'][chan_ind_tot] = ch_obs_valid.min()
                                    store['obs_max'][chan_ind_tot] = ch_obs_valid.max()
                                    store['obs_sum'][chan_ind_tot] = ch_obs_valid.sum()
                                    store['obs_cts'][chan_ind_tot] = valid.sum()

                                # Add scan time
                                l1c_data = product.open(rec)
                                if "latitude" in l1c_data:
                                    vars = ["latitude", "longitude"]
                                    l1c_data = l1c_data.rename(dict([(name, name + "_s1") for name in vars]))
                                l1c_data = l1c_data[["latitude_s1", "longitude_s1", "scan_time"]]
                                l1c_data = l1c_data.rename(
                                    latitude_s1="latitude",
                                    longitude_s1="longitude",
                                )
                                scan_time, _ = xr.broadcast(l1c_data.scan_time, l1c_data.latitude)
                                l1c_data["scan_time"] = scan_time
                                data_r = resample_data(
                                    l1c_data.reset_coords(names="scan_time"),
                                    grid[::40, ::40]
                                )
                                # Store channel properties
                                store.create_array(
                                    'time',
                                    data=data_r.scan_time.data.astype(np.uint64),
                                    dimension_names=("latitude_80", "longitude_80")
                                )
                                LOGGER.info(
                                    f"Created new zarr file '{output_file}' with channel {chan_ind_tot} of {self.n_chans}."
                                )


                            chan_ind_tot += 1
                        swath_ind += 1

                except Exception as e:
                    LOGGER.exception(f"Error processing {rec.local_path}: {e}")
                    continue


# GPM sensor instances
gpm_gmi_obs = GPMObs(
    name="gpm_gmi",
    products=[l1c_r_xcal2016c_gpm_gmi_v07a, l1c_r_xcal2016c_gpm_gmi_v07b],
    radius_of_influence=15e3,
    channel_properties={
        (1, 0): MWChannelProperties(10.65, 96.5, 0.0, "V"),
        (1, 1): MWChannelProperties(10.65, 94.7, 0.0, "H"),
        (1, 2): MWChannelProperties(18.70, 193.0, 0.0, "V"),
        (1, 3): MWChannelProperties(18.70, 194.0, 0.0, "H"),
        (1, 4): MWChannelProperties(23.80, 367.0, 0.0, "V"),
        (1, 5): MWChannelProperties(36.64, 697.0, 0.0, "V"),
        (1, 6): MWChannelProperties(36.64, 707.0, 0.0, "H"),
        (1, 7): MWChannelProperties(89.00, 5865.0, 0.0, "V"),
        (1, 8): MWChannelProperties(89.00, 5918.0, 0.0, "H"),
        (2, 0): MWChannelProperties(166.00, 3850.0, 0.0, "V"),
        (2, 1): MWChannelProperties(166.00, 3893.0, 0.0, "H"),
        (2, 2): MWChannelProperties(183.31, 1358.0, 3.0, "V"),
        (2, 3): MWChannelProperties(183.31, 1874.0, 7.0, "V"),
    }
)

# Channel properties are keyed by (swath_index, channel_index) following the GPM
# L1C swath layout. Values (center frequency [GHz], bandwidth [MHz], offset [GHz]
# and polarization) are taken from the sensor spec sheets in `specs/`. Cross-track
# sounders (MHS, ATMS) use quasi-polarization (QV/QH); conically scanning imagers
# (AMSR2, SSMIS) use V/H.

# MHS: single cross-track swath with 5 channels.
_MHS_CHANNEL_PROPERTIES = {
    (1, 0): MWChannelProperties(89.00, 2800.0, 0.0, "QV"),
    (1, 1): MWChannelProperties(157.00, 2800.0, 0.0, "QV"),
    (1, 2): MWChannelProperties(183.31, 1000.0, 1.0, "QH"),
    (1, 3): MWChannelProperties(183.31, 2000.0, 3.0, "QH"),
    (1, 4): MWChannelProperties(190.31, 2200.0, 0.0, "QV"),
}

noaa18_mhs_obs = GPMObs(
    name="noaa18_mhs",
    products=[l1c_xcal2016v_noaa18_mhs_v07a],
    radius_of_influence=30e3,
    channel_properties=_MHS_CHANNEL_PROPERTIES,
)

noaa19_mhs_obs = GPMObs(
    name="noaa19_mhs",
    products=[l1c_xcal2016v_noaa19_mhs_v07a],
    radius_of_influence=30e3,
    channel_properties=_MHS_CHANNEL_PROPERTIES,
)

metopa_mhs_obs = GPMObs(
    name="metopa_mhs",
    products=[l1c_xcal2021v_metopa_mhs_v07a],
    radius_of_influence=30e3,
    channel_properties=_MHS_CHANNEL_PROPERTIES,
)

metopb_mhs_obs = GPMObs(
    name="metopb_mhs",
    products=[l1c_xcal2016v_metopb_mhs_v07a],
    radius_of_influence=30e3,
    channel_properties=_MHS_CHANNEL_PROPERTIES,
)

metopc_mhs_obs = GPMObs(
    name="metopc_mhs",
    products=[l1c_xcal2019v_metopc_mhs_v07a],
    radius_of_influence=30e3,
    channel_properties=_MHS_CHANNEL_PROPERTIES,
)

# ATMS: window channels (S1), 88.2 GHz (S2) and the 165/183 GHz sounding
# channels (S3).
_ATMS_CHANNEL_PROPERTIES = {
    (1, 0): MWChannelProperties(23.80, 270.0, 0.0, "QV"),
    (2, 1): MWChannelProperties(31.40, 180.0, 0.0, "QV"),
    (3, 0): MWChannelProperties(88.20, 2000.0, 0.0, "QV"),
    (4, 0): MWChannelProperties(165.50, 3000.0, 0.0, "QH"),
    (4, 1): MWChannelProperties(183.31, 2000.0, 7.0, "QH"),
    (4, 2): MWChannelProperties(183.31, 2000.0, 4.5, "QH"),
    (4, 3): MWChannelProperties(183.31, 1000.0, 3.0, "QH"),
    (4, 4): MWChannelProperties(183.31, 1000.0, 1.8, "QH"),
    (4, 5): MWChannelProperties(183.31, 500.0, 1.0, "QH"),
}

noaa20_atms_obs = GPMObs(
    name="noaa20_atms",
    products=[l1c_xcal2019v_noaa20_atms_v07a],
    radius_of_influence=30e3,
    channel_properties=_ATMS_CHANNEL_PROPERTIES,
)

npp_atms_obs = GPMObs(
    name="npp_atms",
    products=[l1c_xcal2019v_npp_atms_v07a],
    radius_of_influence=30e3,
    channel_properties=_ATMS_CHANNEL_PROPERTIES,
)

# AMSR2: one swath per frequency (V/H), with the 89 GHz A- and B-scans as
# separate swaths.
_AMSR2_CHANNEL_PROPERTIES = {
    (1, 0): MWChannelProperties(10.65, 100.0, 0.0, "V"),
    (1, 1): MWChannelProperties(10.65, 100.0, 0.0, "H"),
    (2, 0): MWChannelProperties(18.70, 200.0, 0.0, "V"),
    (2, 1): MWChannelProperties(18.70, 200.0, 0.0, "H"),
    (3, 0): MWChannelProperties(23.80, 400.0, 0.0, "V"),
    (3, 1): MWChannelProperties(23.80, 400.0, 0.0, "H"),
    (4, 0): MWChannelProperties(36.50, 1000.0, 0.0, "V"),
    (4, 1): MWChannelProperties(36.50, 1000.0, 0.0, "H"),
    (5, 0): MWChannelProperties(89.00, 3000.0, 0.0, "V"),
    (5, 1): MWChannelProperties(89.00, 3000.0, 0.0, "H"),
    (6, 0): MWChannelProperties(89.00, 3000.0, 0.0, "V"),
    (6, 1): MWChannelProperties(89.00, 3000.0, 0.0, "H"),
}

gcomw1_amsr2_obs = GPMObs(
    name="gcomw1_amsr2",
    products=[l1c_xcal2016v_gcomw1_amsr2_v07a],
    radius_of_influence=15e3,
    channel_properties=_AMSR2_CHANNEL_PROPERTIES,
)

# SSMIS: low-frequency imaging channels (S1), 91 GHz (S2) and the 150/183 GHz
# sounding channels (S3).
_SSMIS_CHANNEL_PROPERTIES = {
    (1, 0): MWChannelProperties(19.35, 400.0, 0.0, "V"),
    (1, 1): MWChannelProperties(19.35, 400.0, 0.0, "H"),
    (1, 2): MWChannelProperties(22.235, 450.0, 0.0, "V"),
    (2, 0): MWChannelProperties(37.00, 1500.0, 0.0, "V"),
    (2, 1): MWChannelProperties(37.00, 1500.0, 0.0, "H"),
    (3, 0): MWChannelProperties(150.00, 1500.0, 0.0, "H"),
    (3, 1): MWChannelProperties(183.31, 1500.0, 6.6, "H"),
    (3, 2): MWChannelProperties(183.31, 1000.0, 3.0, "H"),
    (3, 3): MWChannelProperties(183.31, 500.0, 1.0, "H"),
    (4, 0): MWChannelProperties(91.655, 1500.0, 0.0, "V"),
    (4, 1): MWChannelProperties(91.655, 1500.0, 0.0, "H"),
}

f16_ssmis_obs = GPMObs(
    name="f16_ssmis",
    products=[
        l1c_xcal2021v_f16_ssmis_v07a,
        l1c_xcal2021v_f16_ssmis_v07b,
    ],
    radius_of_influence=30e3,
    channel_properties=_SSMIS_CHANNEL_PROPERTIES,
)

f17_ssmis_obs = GPMObs(
    name="f17_ssmis",
    products=[
        l1c_xcal2021v_f17_ssmis_v07a,
        l1c_xcal2021v_f17_ssmis_v07b,
    ],
    radius_of_influence=30e3,
    channel_properties=_SSMIS_CHANNEL_PROPERTIES,
)

f18_ssmis_obs = GPMObs(
    name="f18_ssmis",
    products=[
        l1c_xcal2021v_f17_ssmis_v07a,
        l1c_xcal2021v_f17_ssmis_v07b,
    ],
    radius_of_influence=30e3,
    channel_properties=_SSMIS_CHANNEL_PROPERTIES,
)

f19_ssmis_obs = GPMObs(
    name="f19_ssmis",
    products=[
        l1c_xcal2021v_f19_ssmis_v07a,
    ],
    radius_of_influence=30e3,
    channel_properties=_SSMIS_CHANNEL_PROPERTIES,
)




class GPMRefData:
    """
    Class for extracting and loading GPM L1C observations.
    """

    def __init__(
            self,
            name: str,
            products: List[pansat.products.Product],
            radius_of_influence: float,
    ):
        self.name = name
        self.products = products
        self.radius_of_influence = radius_of_influence


    def extract_data(
            self,
            year: int,
            month: int,
            day: int,
            step: np.timedelta64,
            output_path: Path
    ) -> None:
        """
        Extract GPM training data for a given day.

        Args:
            year: The year.
            month: The month.
            day: The day.
            step: The timestep at which to extract observations.
            output_path: The path to which to write the results.
        """
        output_path = Path(output_path) / self.name
        output_path.mkdir(exist_ok=True)

        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(days=1)
        time_range = TimeRange(start_time, end_time)

        for product in self.products:
            recs = product.get(TimeRange(start_time, end_time))
            for rec in recs:
                try:
                    ref_data = product.open(rec)

                    # Extract swath data
                    swath_data = ref_data[[
                        f"longitude",
                        f"latitude",
                        f"surface_precipitation",
                        "scan_time"
                    ]].reset_coords("scan_time")
                    swath_data = swath_data.rename({
                        f"surface_precipitation": "surface_precip"
                    })

                    swath_data["surface_precip"] = swath_data["surface_precip"].astype(np.float32)

                    data_r = resample_data(
                        swath_data,
                        grid,
                        radius_of_influence=self.radius_of_influence
                    )

                    time_range = rec.temporal_coverage
                    output_file = output_path / get_filename(self.name, time_range.start)
                    output_file = output_file.with_suffix(".zarr")

                    surface_precip = data_r[f"surface_precip"].astype(np.float32)
                    valid = 0 <= surface_precip
                    availability = valid.coarsen(longitude=40, latitude=40).sum()

                    surface_precip = surface_precip.data
                    surface_precip[~valid] = np.nan

                    surface_precip_scaled = (surface_precip / 0.02).astype(np.int16)
                    surface_precip_scaled[~valid] = -1

                    # Create new zarr store using low-level operations
                    lons, lats = grid.get_lonlats()
                    lons = lons[0]
                    lats = lats[:, 0]

                    # Get shape from current channel data
                    channel_shape = data_r.surface_precip.data.shape
                    surface_precip_shape = channel_shape

                    # Create zarr group
                    store = zarr.open_group(str(output_file), mode='w')
                    store.attrs["scale_factor"] = 0.02
                    store.attrs["fill_value"] = -1

                    # Store coordinate arrays
                    store.create_array(
                        'longitude',
                        data=lons.astype(np.float32),
                        dimension_names=("longitude",),
                    )
                    store.create_array(
                        'latitude',
                        data=lats.astype(np.float32),
                        dimension_names=("latitude",),
                    )

                    # Create observation array without initializing data (lazy allocation)
                    store.create_array(
                        'surface_precip',
                        data=surface_precip_scaled,
                        chunks=(512, 512),
                        fill_value=-1,
                        compressors=Blosc(cname='zstd', clevel=4),
                        dimension_names=("latitude", "longitude"),
                    )

                    # Create statistics arrays
                    store.create_array(
                        'surface_precip_min',
                        shape=(1,),
                        dtype=np.float32,
                        fill_value=np.nan,
                        dimension_names=None
                    )
                    store.create_array(
                        'surface_precip_max',
                        shape=(1,),
                        dtype=np.float32,
                        fill_value=np.nan,
                        dimension_names=None
                    )
                    store.create_array(
                        'surface_precip_sum',
                        shape=(1,),
                        dtype=np.float32,
                        fill_value=np.nan,
                        dimension_names=None
                    )
                    store.create_array(
                        'surface_precip_cts',
                        shape=(1,),
                        dtype=np.int32,
                        fill_value=0,
                        dimension_names=None
                    )

                    # Store availability
                    store.create_array(
                        'availability',
                        data=(0 < availability.data).astype(np.int8),
                        compressor=Blosc(cname='zstd', clevel=4),
                        dimension_names=("latitude_80", "longitude_80")
                    )

                    # Update statistics for this channel (use original float values)
                    surface_precip_valid = surface_precip[valid]
                    if len(surface_precip_valid) > 0:
                        store['surface_precip_min'][0] = surface_precip_valid.min()
                        store['surface_precip_max'][0] = surface_precip_valid.max()
                        store['surface_precip_sum'][0] = surface_precip_valid.sum()
                        store['surface_precip_cts'][0] = valid.sum()

                    # Add scan time
                    scan_time, _ = xr.broadcast(ref_data.scan_time, ref_data.latitude)
                    ref_data["scan_time"] = scan_time
                    data_r = resample_data(
                        ref_data.reset_coords(names="scan_time"),
                        grid[::40, ::40]
                    )
                    # Store channel properties
                    store.create_array(
                        'time',
                        data=data_r.scan_time.data.astype(np.uint64),
                        dimension_names=("latitude_80", "longitude_80")
                    )
                    LOGGER.info(
                        f"Created new zarr file '{output_file}'."
                    )

                except Exception as e:
                    LOGGER.exception(f"Error processing {rec.local_path}: {e}")
                    continue

# GPM sensor instances
gpm_gmi_ref = GPMRefData(
    name="gprof_gmi",
    products=[l2a_gprof_gpm_gmi_v08a],
    radius_of_influence=15e3,
)

gpm_amsr2_ref = GPMRefData(
    name="gprof_amsr2",
    products=[l2a_gprof_gcomw1_amsr2_v08a],
    radius_of_influence=15e3,
)
