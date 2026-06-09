"""
argos.data.himawari
===================

Functionality to extract Himawari AHI input observations.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List

import numpy as np
import pansat
from pansat.time import TimeRange
from pansat.utils import resample_data
from pansat.products.satellite.himawari import (
    l1b_himawari8_all,
    l1b_himawari9_all
)
from satpy import Scene

import xarray as xr

from argos.grids import get_default_grid
from argos.utils import get_filename
from argos.channel_properties import ChannelProperties

grid = get_default_grid()


LOGGER = logging.getLogger(__file__)


class HimawariObs:
    """
    Class for extracting and loading Meteosat 2nd generation Himawari observations.
    """

    channel_properties = {
        "C01": ChannelProperties(0.455e-6, 0.05e-6, 0.0, "none"),
        "C02": ChannelProperties(0.51e-6, 0.02e-6, 0.0, "none"),
        "C03": ChannelProperties(0.645e-6, 0.03e-6, 0.0, "none"),
        "C04": ChannelProperties(0.86e-6, 0.02e-6, 0.0, "none"),
        "C05": ChannelProperties(1.61e-6, 0.02e-6, 0.0, "none"),
        "C06": ChannelProperties(2.26e-6, 0.02e-6, 0.0, "none"),
        "C07": ChannelProperties(3.85e-6, 0.22e-6, 0.0, "none"),
        "C08": ChannelProperties(6.25e-6, 0.37e-6, 0.0, "none"),
        "C09": ChannelProperties(6.95e-6, 0.12e-6, 0.0, "none"),
        "C10": ChannelProperties(7.3e-6, 0.17e-6, 0.0, "none"),
        "C11": ChannelProperties(8.6e-6, 0.32e-6, 0.0, "none"),
        "C12": ChannelProperties(9.63e-6, 0.18e-6, 0.0, "none"),
        "C13": ChannelProperties(10.45e-6, 0.2e-6, 0.0, "none"),
        "C14": ChannelProperties(11.2e-6, 0.2e-6, 0.0, "none"),
        "C15": ChannelProperties(12.3e-6, 0.3e-6, 0.0, "none"),
        "C16": ChannelProperties(13.3e-6, 0.2e-6, 0.0, "none"),
    }
    mins = np.array([0.0] * 6  + [150.0] * 10)

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
        Extract Himawari training data for a given day.

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
            recs = self.product.get(TimeRange(time + np.timedelta64(5, "m")))
            for band in range(1, 17):
                band_recs = [rec for rec in recs if f"B{band:02}" in rec.filename]
                scene = Scene([str(rec.local_path) for rec in band_recs])
                scene.load([f"B{band:02}"])
                channel = next(iter(scene.available_dataset_names()))
                chan_ind = band - 1

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

                time_range = band_recs[0].temporal_coverage
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
                        self.channel_properties[f"C{ind:02}"].properties for ind in range(1, 17)
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


himawari8_obs = HimawariObs(
    name="himawari8",
    product=l1b_himawari8_all,
)
himawari9_obs = HimawariObs(
    name="himawari9",
    product=l1b_himawari9_all,
)
