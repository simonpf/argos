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
from pansat.utils import resample_data
from pansat.products.satellite.meteosat import (
    l1b_msg_seviri,
    l1b_msg_seviri_io
)
from satpy import Scene

import xarray as xr

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
                for chan_ind, chan in enumerate(channels):

                    if chan == "HRV":
                        data_r = resample_data(
                            seviri_obs[[chan, "latitude_1", "longitude_1"]].rename(
                                latitude_1="latitude",
                                longitude_1="longitude",
                            ),
                            grid
                        )
                    else:
                        data_r = resample_data(
                            seviri_obs[[chan, "latitude_0", "longitude_0"]].rename(
                                latitude_0="latitude",
                                longitude_0="longitude",
                            ),
                            grid
                        )

                    time_range = rec.temporal_coverage
                    output_file = output_path / get_filename(self.name, time_range.start)
                    output_file = output_file.with_suffix(".nc")

                    ch_obs = data_r[chan] - self.mins[chan_ind]
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
                            f"Writing {chan} channel to output file '{output_file}'."
                        )

                    else:
                        lons, lats = grid.get_lonlats()
                        lons = lons[0]
                        lats = lats[:, 0]

                        obs = np.zeros((12,) + data_r[chan].data.shape)
                        obs[chan_ind, :, :] = ch_obs

                        obs_min = np.zeros(12,)
                        obs_max = np.zeros(12,)
                        obs_sum = np.zeros(12,)
                        obs_cts = np.zeros(12,)

                        ch_obs = ch_obs[valid]
                        obs_min[chan_ind] = ch_obs.min()
                        obs_max[chan_ind] = ch_obs.max()
                        obs_sum[chan_ind] = ch_obs.sum()
                        obs_cts[chan_ind] = valid.sum()

                        channel_props = np.stack([
                            self.channel_properties[chan].properties for chan in channels
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
                            "chunksizes": (12, 512, 512)
                        }
                        data_r.availability.encoding = {
                            "dtype": "int8",
                            "zlib": True,
                            "shuffle":  True
                        }
                        LOGGER.info(
                            f"Writing {chan} to output file '{output_file}'."
                        )
                        data_r.to_netcdf(output_file)

                time = time + step


seviri_obs = SEVIRIObs(
    name="seviri",
    product=l1b_msg_seviri,
)
seviri_io_obs = SEVIRIObs(
    name="seviri",
    product=l1b_msg_seviri,
)
