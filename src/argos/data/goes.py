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
from pansat.utils import resample_data
from pansat.products.satellite.goes import (
    GOES16L1BRadiances,
    GOES18L1BRadiances,
    GOES19L1BRadiances
)
from satpy import Scene
import xarray as xr

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
        1: ChannelProperties(47e-6, 0.0, "none"),
        2: ChannelProperties(64e-6, 0.0, "none"),
        3: ChannelProperties(86e-6, 0.0, "none"),
        4: ChannelProperties(1.37e-6, 0.0, "none"),
        5: ChannelProperties(1.6e-6, 0.0, "none"),
        6: ChannelProperties(2.2e-6, 0.0, "none"),
        7: ChannelProperties(3.9e-6, 0.0, "none"),
        8: ChannelProperties(6.2e-6, 0.0, "none"),
        9: ChannelProperties(6.9e-6, 0.0, "none"),
        10: ChannelProperties(7.3e-6, 0.0, "none"),
        11: ChannelProperties(8.4e-6, 0.0, "none"),
        12: ChannelProperties(9.6e-6, 0.0, "none"),
        13: ChannelProperties(10.3e-6, 0.0, "none"),
        14: ChannelProperties(12.2e-6, 0.0, "none"),
        15: ChannelProperties(12.3e-6, 0.0, "none"),
        16: ChannelProperties(13.4e-6, 0.0, "none"),
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
            for product in self.products:
                recs = product.get(TimeRange(time, time + np.timedelta64(5, "m")))
                for rec in recs:
                    scene = Scene([rec.local_path], reader="abi_l1b")
                    channel = next(iter(scene.available_dataset_names()))
                    chan_ind = int(channel[1:]) - 1

                    LOGGER.info(
                        f"Loading observations for channel {channel}."
                    )
                    scene.load([channel])
                    lons, lats = scene.coarsest_area().get_lonlats()
                    data = scene.to_xarray_dataset().compute()
                    data[channel] = data[channel].astype(np.float32)
                    data["longitude"] = (("y", "x"), lons.astype(np.float32))
                    data["latitude"] = (("y", "x"), lats.astype(np.float32))
                    data_r = resample_data(data, grid)

                    time_range = rec.temporal_coverage
                    output_file = output_path / get_filename(self.name, time_range.start)
                    output_file = output_file.with_suffix(".nc")

                    ch_obs = data_r[channel] - self.mins[chan_ind]
                    ch_obs = np.minimum(np.maximum(ch_obs, 0.0), self.mins[chan_ind] + 254.0)

                    valid = np.isfinite(ch_obs)
                    availability = valid.coarsen(longitude=80, latitude=80).sum()

                    ch_obs = ch_obs.data
                    valid = valid.data
                    availability = availability.data
                    ch_obs[~valid] = np.nan

                    if output_file.exists():
                        data = xr.load_dataset(output_file)
                        data.obs.data[chan_ind, :, :] = ch_obs

                        ch_obs = ch_obs[valid]
                        data.obs_min.data[chan_ind] = ch_obs.min()
                        data.obs_max.data[chan_ind] = ch_obs.max()
                        data.obs_sum.data[chan_ind] = ch_obs.sum()
                        data.obs_cts.data[chan_ind] = valid.sum()
                        data.availability.data[:] = 0 < (data.availability.data + availability)
                        data.to_netcdf(output_file)

                        LOGGER.info(
                            f"Writing {channel} to output file '{output_file}'."
                        )

                    else:
                        lons, lats = grid.get_lonlats()
                        lons = lons[0]
                        lats = lats[:, 0]

                        obs = np.zeros((16,) + data_r[channel].data.shape)
                        obs[chan_ind, :, :] = ch_obs

                        obs_min = np.zeros(16,)
                        obs_max = np.zeros(16,)
                        obs_sum = np.zeros(16,)
                        obs_cts = np.zeros(16,)

                        ch_obs = ch_obs[valid]
                        obs_min[chan_ind] = ch_obs.min()
                        obs_max[chan_ind] = ch_obs.max()
                        obs_sum[chan_ind] = ch_obs.sum()
                        obs_cts[chan_ind] = valid.sum()

                        channel_props = np.stack([
                            self.channel_properties[ind].properties for ind in range(1, 17)
                        ])

                        data_r = xr.Dataset({
                            "time": np.datetime64(time_range.start),
                            "longitude": (("longitude",), lons),
                            "latitude": (("latitude",), lats),
                            "obs": (("channel", "latitude", "longitude"), obs),
                            "obs_min": (("channel",), obs_min),
                            "obs_max": (("channel",), obs_max),
                            "obs_sum": (("channel",), obs_sum),
                            "obs_cts": (("channel",), obs_cts),
                            "availability": (("latitude_80", "longitude_80"), 0 < availability),
                            "channel_properties": (("channel", "channel_features"), channel_props)
                        })
                        data_r.obs.encoding = {
                            "dtype": "uint8",
                            "_FillValue": 255,
                            "zlib": True,
                            "complevel": 6,
                            "scale_factor": 1.0,
                            "chunksizes": (16, 512, 512)
                        }
                        data_r.availability.encoding = {
                            "dtype": "int8",
                            "zlib": True,
                            "shuffle":  True
                        }
                        LOGGER.info(
                            f"Writing {channel} to output file '{output_file}'."
                        )
                        data_r.to_netcdf(output_file)

            time = time + step
            break


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
