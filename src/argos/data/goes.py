"""
argos.data.goes
===============

Functionality to extract GOES input observations.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List

import numpy as np
import pansat
from pansat.time import TimeRange
from pansat.utils import resample_data, get_resample_info
from pansat.products.satellite.goes import (
    GOES16L1BRadiances,
    GOES18L1BRadiances,
    GOES19L1BRadiances
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


class GOESObs:
    """
    Class for extracting and loading GOES ABI observations.
    """

    channel_properties = {
        1: ChannelProperties(47e-6, 0.04e-6, 0.0, "none"),
        2: ChannelProperties(64e-6, 0.10e-6, 0.0, "none"),
        3: ChannelProperties(86e-6, 0.04e-6, 0.0, "none"),
        4: ChannelProperties(1.37e-6, 0.03e-6, 0.0, "none"),
        5: ChannelProperties(1.6e-6, 0.06e-6, 0.0, "none"),
        6: ChannelProperties(2.2e-6, 0.05e-6, 0.0, "none"),
        7: ChannelProperties(3.9e-6, 0.2e-6, 0.0, "none"),
        8: ChannelProperties(6.2e-6, 0.9e-6, 0.0, "none"),
        9: ChannelProperties(6.9e-6, 0.4e-6, 0.0, "none"),
        10: ChannelProperties(7.3e-6, 0.2e-6, 0.0, "none"),
        11: ChannelProperties(8.4e-6, 0.4e-6, 0.0, "none"),
        12: ChannelProperties(9.6e-6, 0.2e-6, 0.0, "none"),
        13: ChannelProperties(10.3e-6, 0.5e-6, 0.0, "none"),
        14: ChannelProperties(11.2e-6, 0.8e-6, 0.0, "none"),
        15: ChannelProperties(12.3e-6, 1.0e-6, 0.0, "none"),
        16: ChannelProperties(13.4e-6, 0.6e-6, 0.0, "none"),
    }
    mins = np.array([0.0] * 6  + [150.0] * 10)

    def __init__(
            self,
            name: str,
            products: List[pansat.products.Product]
    ):
        self.name = name
        self.products = products


    def extract_data(
            self,
            year: int,
            month: int,
            day: int,
            step: np.timedelta64,
            output_path: Path
    ) -> None:
        """
        Extract GOES training data for a given day.

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
            try:
                recs = [
                    prod.get(TimeRange(time, time + np.timedelta64(5, "m")))[0] for prod in self.products
                ]
                scene = Scene([rec.local_path for rec in recs], reader="abi_l1b")
                datasets = scene.available_dataset_names()
                scene.load(datasets)

                for area, datasets in scene.iter_by_area():

                    info = None
                    lons, lats = area.get_lonlats()

                    for dataset in datasets:
                        channel = dataset.get("name")
                        chan_ind = int(channel[1:]) - 1

                        LOGGER.info(
                            f"Loading observations for channel {channel}."
                        )
                        data = xr.Dataset({
                            channel: scene[channel].compute()
                        })

                        if channel in ["C01", "C03", "C05"]:
                            data = data[{"x": slice(0, None, 2), "y": slice(0, None, 2)}]
                            data["longitude"] = (("y", "x"), lons[::2, ::2].astype(np.float32))
                            data["latitude"] = (("y", "x"), lats[::2, ::2].astype(np.float32))
                        elif channel == "C02":
                            data = data[{"x": slice(0, None, 4), "y": slice(0, None, 4)}]
                            data["longitude"] = (("y", "x"), lons[::4, ::4].astype(np.float32))
                            data["latitude"] = (("y", "x"), lats[::4, ::4].astype(np.float32))
                        else:
                            data["longitude"] = (("y", "x"), lons.astype(np.float32))
                            data["latitude"] = (("y", "x"), lats.astype(np.float32))

                        LOGGER.info(
                            f"Loaded observations for channel {channel}."
                        )

                        if info is None:
                            info = get_resample_info(data, grid)
                        data_r = resample_data(data, grid, info=info)

                        LOGGER.info(
                            f"Resampling"
                        )

                        time_range = recs[0].temporal_coverage
                        output_file = output_path / get_filename(self.name, time_range.start)
                        output_file = output_file.with_suffix(".zarr")

                        ch_obs = data_r[channel]

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
                                f"Updated {channel} in zarr file '{output_file}'."
                            )

                        else:
                            # Create new zarr store using low-level operations
                            lons_g, lats_g = grid.get_lonlats()
                            lons_g = lons[0]
                            lats_g = lats[:, 0]

                            # Get shape from current channel data
                            channel_shape = data_r[channel].data.shape
                            obs_shape = (16,) + channel_shape
                            
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
                                data=lons_g.astype(np.float32),
                                dimension_names=("longitude",),
                            )
                            store.create_array(
                                'latitude',
                                data=lats_g.astype(np.float32),
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
                                shape=(16,),
                                dtype=np.float32,
                                fill_value=np.nan,
                                dimension_names=dims
                            )
                            store.create_array(
                                'obs_max',
                                shape=(16,),
                                dtype=np.float32,
                                fill_value=np.nan,
                                dimension_names=dims
                            )
                            store.create_array(
                                'obs_sum',
                                shape=(16,),
                                dtype=np.float32,
                                fill_value=np.nan,
                                dimension_names=dims
                            )
                            store.create_array(
                                'obs_cts',
                                shape=(16,),
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
                                self.channel_properties[ind].properties for ind in range(1, 17)
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
                                f"Created new zarr file '{output_file}' with {channel}."
                            )
            except Exception as e:
                LOGGER.error(f"Failed to extract data for {time}: {e}")
            
            time = time + step


goes_16_obs = GOESObs(
    name="goes16",
    products=[GOES16L1BRadiances("F", channel=ind) for ind in range(1, 17)]
)
goes_18_obs = GOESObs(
    name="goes18",
    products=[GOES18L1BRadiances("F", channel=ind) for ind in range(1, 17)]
)
goes_19_obs = GOESObs(
    name="goes19",
    products=[GOES19L1BRadiances("F", channel=ind) for ind in range(1, 17)]
)
