"""
argos.data.seviri
=================

Functionality to extract SEVIRI input observations.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List

import numpy as np
import pansat
from pansat.time import TimeRange
from pansat.utils import resample_data, get_resample_info
from pansat.products.satellite.meteosat import (
    l1b_msg_seviri,
    l1b_msg_seviri_io
)
from satpy import Scene
import xarray as xr
import zarr
from numcodecs.zarr3 import Blosc

from argos.grids import get_default_grid
from argos.utils import get_filename
from argos.channel_properties import ChannelProperties

grid = get_default_grid()


LOGGER = logging.getLogger(__file__)


class SEVIRIObs:
    """
    Class for extracting and loading Meteosat 2nd generation SEVIRI observations.
    """

    channel_properties = {
        "HRV": ChannelProperties(0.75e-6, 0.3e-6, 0.0, "none"),
        "VIS006": ChannelProperties(0.63e-6, 0.15e-6, 0.0, "none"),
        "VIS008": ChannelProperties(0.81e-6, 0.14e-6, 0.0, "none"),
        "IR_016": ChannelProperties(1.63e-6, 0.28e-6, 0.0, "none"),
        "IR_039": ChannelProperties(4.0e-6, 0.98e-6, 0.0, "none"),
        "WV_062": ChannelProperties(6.25e-6, 1.8e-6, 0.0, "none"),
        "WV_073": ChannelProperties(7.35e-6, 1.0e-6, 0.0, "none"),
        "IR_087": ChannelProperties(8.7e-6, 0.8e-6, 0.0, "none"),
        "IR_097": ChannelProperties(9.66e-6, 0.56e-6, 0.0, "none"),
        "IR_108": ChannelProperties(10.8e-6, 2e-6, 0.0, "none"),
        "IR_120": ChannelProperties(12.0e-6, 2e-6, 0.0, "none"),
        "IR_134": ChannelProperties(13.4e-6, 2e-6, 0.0, "none"),
    }

    mins = np.array([0.0] * 3  + [150.0] * 9)

    def __init__(
            self,
            name: str,
            product: pansat.products.Product
    ):
        self.name = name
        self.product = product


    def extract_data(
            self,
            year: int,
            month: int,
            day: int,
            step: np.timedelta64,
            output_path: Path
    ) -> None:
        """
        Extract SEVIR training data for a given day.

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

        time = start_time
        while time < end_time:
            recs = self.product.get(TimeRange(time, time + np.timedelta64(5, "m")))
            for rec in recs:
                LOGGER.info(
                    f"Loading SEVIRI observations."
                )
                seviri_obs = self.product.open(rec)
                channels = list(self.channel_properties.keys())
                
                info = None

                seviri_obs = seviri_obs.drop_vars(("latitude_1", "longitude_1"))
                seviri_obs["HRV"] = seviri_obs["HRV"][{
                    "latitude_1": slice(0, None, 2),
                    "longitude_1": slice(0, None, 2)
                }]
                seviri_obs = seviri_obs.rename(
                    latitude_0="latitude",
                    longitude_0="longitude"
                )
                
                for chan_ind, chan in enumerate(channels):

                    data = seviri_obs[[chan, "latitude", "longitude"]]

                    if info is None:
                        info = get_resample_info(data, grid, radius_of_influence=10e3)
                    data_r = resample_data(data, grid, info=info)

                    time_range = rec.temporal_coverage
                    output_file = output_path / get_filename(self.name, time_range.start)
                    output_file = output_file.with_suffix(".zarr")

                    ch_obs = data_r[chan]

                    valid = np.isfinite(ch_obs)
                    availability = valid.coarsen(longitude=80, latitude=80).sum()

                    ch_obs = ch_obs.data
                    valid = valid.data
                    availability = availability.data
                    ch_obs[~valid] = np.nan

                    # Apply scaling using mins array and convert to uint8
                    ch_obs_scaled = ch_obs - self.mins[chan_ind]
                    ch_obs_scaled = np.maximum(ch_obs_scaled, 0.0)  # Ensure non-negative
                    ch_obs_uint8 = np.minimum(ch_obs_scaled, 254.0).astype(np.uint8)
                    ch_obs_uint8[~valid] = 255  # Use 255 as NaN value

                    if output_file.exists():
                        # Open existing zarr store and update the channel
                        store = zarr.open_group(str(output_file), mode='r+')
                        
                        # Update observations for this channel
                        store['obs'][chan_ind, :, :] = ch_obs_uint8
                        
                        # Update statistics for this channel (use original float values)
                        ch_obs_valid = ch_obs[valid]
                        if len(ch_obs_valid) > 0:
                            store['obs_min'][chan_ind] = ch_obs_valid.min()
                            store['obs_max'][chan_ind] = ch_obs_valid.max()
                            store['obs_sum'][chan_ind] = ch_obs_valid.sum()
                            store['obs_cts'][chan_ind] = valid.sum()
                        
                        # Update availability (combine with existing)
                        existing_availability = store['availability'][:]
                        store['availability'][:] = 0 < (existing_availability + availability)
                        
                        LOGGER.info(
                            f"Updated {chan} in zarr file '{output_file}'."
                        )

                    else:
                        # Create new zarr store using low-level operations
                        lons, lats = grid.get_lonlats()
                        lons = lons[0]
                        lats = lats[:, 0]

                        # Get shape from current channel data
                        channel_shape = data_r[chan].data.shape
                        obs_shape = (12,) + channel_shape  # 12 channels for SEVIRI
                        
                        # Create zarr group
                        store = zarr.open_group(str(output_file), mode='w')
                        store.attrs["fill_value"] = 255
                        store.attrs["offsets"] = list(self.mins)
                        
                        # Store coordinate arrays
                        store.create_array(
                            'time',
                            data=np.datetime64(time_range.start),
                            dimension_names=None
                        )
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
                            dtype=np.uint8,
                            fill_value=255,
                            compressors=Blosc(cname='zstd', clevel=4),
                            dimension_names=("channel", "latitude", "longitude"),
                        )
                        
                        # Create statistics arrays
                        dims = ("channel",)
                        store.create_array(
                            'obs_min',
                            shape=(12,),
                            dtype=np.float32,
                            fill_value=np.nan,
                            dimension_names=dims
                        )
                        store.create_array(
                            'obs_max',
                            shape=(12,),
                            dtype=np.float32,
                            fill_value=np.nan,
                            dimension_names=dims
                        )
                        store.create_array(
                            'obs_sum',
                            shape=(12,),
                            dtype=np.float32,
                            fill_value=np.nan,
                            dimension_names=dims
                        )
                        store.create_array(
                            'obs_cts',
                            shape=(12,),
                            dtype=np.int32,
                            fill_value=0,
                            dimension_names=dims
                        )
                        
                        # Store availability
                        store.create_array(
                            'availability',
                            data=(0 < availability).astype(np.int8),
                            compressor=Blosc(cname='zstd', clevel=4),
                            dimension_names=("latitude_80", "longitude_80")
                        )
                        
                        # Store channel properties
                        channel_props = np.stack([
                            self.channel_properties[chan].properties for chan in channels
                        ])
                        store.create_array('channel_properties', data=channel_props, dimension_names=("channel", "feature"))
                        
                        # Now write the current channel data
                        store['obs'][chan_ind, :, :] = ch_obs_uint8
                        
                        # Update statistics for this channel (use original float values)
                        ch_obs_valid = ch_obs[valid]
                        if len(ch_obs_valid) > 0:
                            store['obs_min'][chan_ind] = ch_obs_valid.min()
                            store['obs_max'][chan_ind] = ch_obs_valid.max()
                            store['obs_sum'][chan_ind] = ch_obs_valid.sum()
                            store['obs_cts'][chan_ind] = valid.sum()
                        
                        LOGGER.info(
                            f"Created new zarr file '{output_file}' with {chan}."
                        )

            time = time + step


seviri_obs = SEVIRIObs(
    name="seviri",
    product=l1b_msg_seviri,
)
seviri_io_obs = SEVIRIObs(
    name="seviri_io",
    product=l1b_msg_seviri,
)
