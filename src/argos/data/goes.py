"""
argos.data.goes
===============

Functionality to extract GOES input observations.
"""
from datetime import datetime, timedelta
from functools import cache
import logging
from pathlib import Path
from typing import List

import numpy as np
import pansat
from pansat.time import TimeRange
from pansat.utils import resample_data, get_resample_info
from pansat.products.satellite.goes import (
    l2b_goes_16_mcmip_f,
    l2b_goes_18_mcmip_f,
    l2b_goes_19_mcmip_f,
)
from pyproj import CRS, Transformer
from satpy import Scene
import xarray as xr
import zarr
from numcodecs.zarr3 import Blosc

from argos.grids import get_default_grid
from argos.utils import get_filename
from argos.channel_properties import ChannelProperties


grid = get_default_grid()

LOGGER = logging.getLogger(__file__)


@cache
def get_goes_lonlats(
        x_0: float,
        x_1: float,
        n_x: int,
        y_0: float,
        y_1: float,
        n_y: int,
        perspective_point_height: float,
        longitude_of_projection_origin: float,
        semi_major_axis: float,
        semi_minor_axis: float,
        sweep_angle_axis: str,
):
    """
    Calculate longitude and latitude coordinates for a GOES ABI dataset.

    Args:
        x_0: Start value of the x coordinates.
        x_1: End value of the x coordinates.
        n_x: The number of values along the x dimension.
        y_0: Start value of the y coordinates.
        y_1: End value of the y coordinates.
        n_y: The number of values along the y dimension.
        perspective_point_height: Perspective point height from projection attributes.
        longitude_of_projection_origin: Longitude of projection origin from projection attributes.
        semi_major_axis: The length of the semi-major axis from the projection attributes
        semi_minor_axis: The length of the semi-minor axis from the projection attributes
        sweep_angle_axis: The sweep-angle axis from the projection attributes.

    Returns:
        lon, lat: 2D arrays of longitude and latitude.
    """
    # Define GOES geostationary projection.
    crs = CRS.from_proj4(
        f"+proj=geos "
        f"+h={perspective_point_height} "
        f"+lon_0={longitude_of_projection_origin} "
        f"+sweep={sweep_angle_axis} "
        f"+a={semi_major_axis} "
        f"+b={semi_minor_axis}"
    )

    transformer = Transformer.from_crs(
        crs, CRS.from_epsg(4326), always_xy=True
    )

    x = np.linspace(x_0, x_1, n_x) * perspective_point_height
    y = np.linspace(y_0, y_1, n_y) * perspective_point_height

    xx, yy = np.meshgrid(x, y)

    lon, lat = transformer.transform(xx, yy)

    # Mask pixels outside the Earth disk.
    lon[~np.isfinite(lon)] = np.nan
    lat[~np.isfinite(lat)] = np.nan

    return lon.copy(), lat.copy()


_RESAMPLE_INFO = {}


def get_resample_info_cached(dataset: xr.Dataset, domain):
    """
    Cached calculation of resampling info.

    Args:
        dataset: An xarray.Dataset containing the GOES observations with latitude
            and longitude coordinates.
        domain: A pyresample area definition defining the target grid.

    Return:
        Nearest-neighbor resampling information that can be used with the
        pansat.utils.resample_data function.
    """
    global _RESAMPLE_INFO
    lons = dataset.longitude.data
    lon_min, lon_max, n_lons = np.nanmin(lons), np.nanmax(lons), lons.size
    lats = dataset.latitude.data
    lat_min, lat_max, n_lats = np.nanmin(lats), np.nanmax(lats), lats.size
    key = (lon_min, lon_max, n_lons, lat_min, lat_max, n_lats)

    if not key in _RESAMPLE_INFO:
        _RESAMPLE_INFO[key] = get_resample_info(
            dataset,
            domain,
            radius_of_influence=5e3
        )
    return _RESAMPLE_INFO[key]


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

        time = np.datetime64(start_time)
        while time < end_time:
            try:
                time_range = TimeRange(time - np.timedelta64(5, "m"), time + np.timedelta64(5, "m"))
                recs = self.product.find_files(time_range)
                deltas = np.array([
                    (rec.central_time.start - time).total_seconds() for rec in recs
                ])
                ind = np.argmin(deltas)
                rec = recs[ind].get()
                data = xr.open_dataset(rec.local_path)

                # Get projection properties.
                proj = data["goes_imager_projection"].attrs
                x_0 = data.x.data[0]
                x_1 = data.x.data[-1]
                n_x = data.x.size
                y_0 = data.y.data[0]
                y_1 = data.y.data[-1]
                n_y = data.y.size
                lons, lats = get_goes_lonlats(
                    x_0, x_1, n_x,
                    y_0, y_1, n_y,
                    perspective_point_height=proj["perspective_point_height"],
                    longitude_of_projection_origin=proj["longitude_of_projection_origin"],
                    semi_major_axis=proj["semi_major_axis"],
                    semi_minor_axis=proj["semi_minor_axis"],
                    sweep_angle_axis=proj["sweep_angle_axis"]
                )

                data["latitude"] = (("y", "x"), lats)
                data["longitude"] = (("y", "x"), lons)

                resample_info = get_resample_info_cached(data, grid)

                for chan in range(1, 17):

                    chan_ind = chan - 1

                    obs_var = f"CMI_C{chan:02}"
                    obs = data[[obs_var, "longitude", "latitude"]].compute()
                    obs_r = resample_data(obs, grid, info=resample_info, radius_of_influence=15e3)
                    obs_vals = np.maximum(obs_r[obs_var].data, 0.0)

                    valid = np.isfinite(obs_vals)
                    availability = np.isfinite(obs_r[obs_var]).coarsen(
                        longitude=80, latitude=80
                    )
                    availability = availability.sum().data

                    scale_fac = 0.01
                    fill_val = 2 ** 16 - 1
                    obs_vals = (obs_vals / scale_fac).astype("uint16")
                    obs_vals[~valid] = fill_val

                    output_file = output_path / get_filename(self.name, time_range.start)
                    output_file = output_file.with_suffix(".zarr")

                    if output_file.exists():
                        # Open existing zarr store and update the channel
                        store = zarr.open_group(str(output_file), mode='r+')

                        # Update observations for this channel
                        store['obs'][chan_ind, :, :] = obs_vals

                        # Update statistics for this channel (use original float values)
                        valid = np.isfinite(obs_vals)
                        obs_valid = obs_vals[valid]
                        if len(obs_valid) > 0:
                            store['obs_min'][chan_ind] = obs_valid.min()
                            store['obs_max'][chan_ind] = obs_valid.max()
                            store['obs_sum'][chan_ind] = obs_valid.sum()
                            store['obs_cts'][chan_ind] = valid.sum()

                        # Update availability (combine with existing)
                        existing_availability = store['availability'][:]
                        store['availability'][:] = 0 < (existing_availability + availability)

                        LOGGER.info(
                            f"Updated channel {chan} in zarr file '{output_file}'."
                        )

                    else:
                        # Create new zarr store using low-level operations
                        lons_g, lats_g = grid.get_lonlats()
                        lons_g = lons_g[0]
                        lats_g = lats_g[:, 0]

                        # Get shape from current channel data
                        channel_shape = obs_vals.shape
                        obs_shape = (16,) + channel_shape

                        # Create zarr group
                        store = zarr.open_group(str(output_file), mode='w')
                        store.attrs["fill_value"] = fill_val
                        store.attrs["scale_factor"] = scale_fac

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
                            dtype=np.uint16,
                            fill_value=2 ** 16 - 1,
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
                        store['obs'][chan_ind, :, :] = obs_vals

                        # Update statistics for this channel (use original float values)
                        obs_valid = obs_vals[valid]
                        if len(obs_valid) > 0:
                            store['obs_min'][chan_ind] = obs_valid.min()
                            store['obs_max'][chan_ind] = obs_valid.max()
                            store['obs_sum'][chan_ind] = obs_valid.sum()
                            store['obs_cts'][chan_ind] = valid.sum()

                        LOGGER.info(
                            f"Created new zarr file '{output_file}' with channel {chan}."
                        )
            except Exception as e:
                raise e
                LOGGER.error(f"Failed to extract data for {time}: {e}")
            
            time = time + step


goes_16_obs = GOESObs(
    name="goes16",
    product=l2b_goes_16_mcmip_f,
)
goes_18_obs = GOESObs(
    name="goes18",
    product=l2b_goes_18_mcmip_f,
)
goes_19_obs = GOESObs(
    name="goes19",
    product=l2b_goes_19_mcmip_f,
)
