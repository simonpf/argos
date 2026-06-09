"""
argos.data.gpm
==============

Functionality to extract GPM L1C observations.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
from typing import List

import numpy as np
import pansat
from pansat.time import TimeRange
from pansat.utils import resample_data
from pansat.products.satellite.gpm import (
    l1c_r_xcal2016c_gpm_gmi_v07a,
    l1c_r_xcal2016c_gpm_gmi_v07b,
    l1c_xcal2016v_gcomw1_amsr2_v07a,
    l1c_xcal2019v_noaa20_atms_v07a,
    l1c_xcal2019v_npp_atms_v07a
)
import xarray as xr

from argos.grids import get_default_grid
from argos.utils import get_filename
from argos.channel_properties import ChannelProperties

grid = get_default_grid()


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

        time = start_time
        while time < end_time:
            for product in self.products:
                recs = product.get(TimeRange(time, time + np.timedelta64(5, "m")))
                for rec in recs:
                    try:
                        l1c_obs = product.open(rec)
                        
                        if "latitude" in l1c_obs:
                            vars = ["latitude", "longitude", "tbs", "channels"]
                            l1c_obs = l1c_obs.rename(dict([(name, name + "_s1") for name in vars]))

                        swath_ind = 1
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
                                f"latitude_s{swath_ind}": "latitude"
                            })

                            # Resample to grid
                            try:
                                data_r = resample_data(swath_data, grid)
                            except ValueError:
                                swath_ind += 1
                                continue

                            time_range = rec.temporal_coverage
                            output_file = output_path / get_filename(self.name, time_range.start)
                            output_file = output_file.with_suffix(".nc")

                            LOGGER.info(
                                f"Processing {len(freqs)} channels from {rec.local_path.name}."
                            )

                            for chan_ind, (freq, offset, pol) in enumerate(zip(freqs, offsets, pols)):
                                
                                ch_obs = data_r[f"tbs_s{swath_ind}"].data[..., chan_ind]
                                ch_obs = ch_obs.astype(np.float32)
                                
                                # Quality control
                                valid = (ch_obs > 0) & (ch_obs < 400) & np.isfinite(ch_obs)
                                availability = valid.coarsen(longitude=80, latitude=80).sum()

                                ch_obs[~valid] = np.nan

                                if output_file.exists():
                                    data = xr.load_dataset(output_file)
                                    data.obs.data[chan_ind, :, :] = ch_obs

                                    ch_obs_valid = ch_obs[valid]
                                    if len(ch_obs_valid) > 0:
                                        data.obs_min.data[chan_ind] = ch_obs_valid.min()
                                        data.obs_max.data[chan_ind] = ch_obs_valid.max()
                                        data.obs_sum.data[chan_ind] = ch_obs_valid.sum()
                                        data.obs_cts.data[chan_ind] = valid.sum()
                                    data.availability.data[:] = 0 < (data.availability.data + availability.data)
                                    data.to_netcdf(output_file)

                                    LOGGER.info(
                                        f"Updated channel {chan_ind+1} in output file '{output_file}'."
                                    )

                                else:
                                    lons, lats = grid.get_lonlats()
                                    lons = lons[0]
                                    lats = lats[:, 0]

                                    n_channels = len(freqs)
                                    obs = np.full((n_channels,) + ch_obs.shape, np.nan, dtype=np.float32)
                                    obs[chan_ind, :, :] = ch_obs

                                    obs_min = np.zeros(n_channels)
                                    obs_max = np.zeros(n_channels) 
                                    obs_sum = np.zeros(n_channels)
                                    obs_cts = np.zeros(n_channels)

                                    ch_obs_valid = ch_obs[valid]
                                    if len(ch_obs_valid) > 0:
                                        obs_min[chan_ind] = ch_obs_valid.min()
                                        obs_max[chan_ind] = ch_obs_valid.max()
                                        obs_sum[chan_ind] = ch_obs_valid.sum()
                                        obs_cts[chan_ind] = valid.sum()

                                    # Create channel properties based on frequency and polarization
                                    channel_props = np.zeros((n_channels, 3))
                                    for i, (f, o, p) in enumerate(zip(freqs, offsets, pols)):
                                        channel_props[i] = ChannelProperties(
                                            wavelength=3e8 / (f * 1e9) * 1e6,  # Convert to micrometers
                                            offset=o,
                                            polarization=p.lower()
                                        ).properties

                                    data_r = xr.Dataset({
                                        "time": np.datetime64(time_range.start),
                                        "longitude": (("longitude",), lons),
                                        "latitude": (("latitude",), lats),
                                        "obs": (("channel", "latitude", "longitude"), obs),
                                        "obs_min": (("channel",), obs_min),
                                        "obs_max": (("channel",), obs_max),
                                        "obs_sum": (("channel",), obs_sum),
                                        "obs_cts": (("channel",), obs_cts),
                                        "availability": (("latitude_80", "longitude_80"), 0 < availability.data),
                                        "channel_properties": (("channel", "channel_features"), channel_props)
                                    })
                                    data_r.obs.encoding = {
                                        "dtype": "float32",
                                        "_FillValue": np.nan,
                                        "zlib": True,
                                        "complevel": 6,
                                        "chunksizes": (n_channels, 512, 512)
                                    }
                                    data_r.availability.encoding = {
                                        "dtype": "int8",
                                        "zlib": True,
                                        "shuffle": True
                                    }
                                    LOGGER.info(
                                        f"Writing {n_channels} channels to output file '{output_file}'."
                                    )
                                    data_r.to_netcdf(output_file)
                                    break  # Only create file once

                            swath_ind += 1

                    except Exception as e:
                        LOGGER.exception(f"Error processing {rec.local_path}: {e}")
                        continue

            time = time + step


# GPM sensor instances
gpm_gmi_obs = GPMObs(
    name="gpm_gmi",
    products=[l1c_r_xcal2016c_gpm_gmi_v07a, l1c_r_xcal2016c_gpm_gmi_v07b]
)

gcomw1_amsr2_obs = GPMObs(
    name="gcomw1_amsr2", 
    products=[l1c_xcal2016v_gcomw1_amsr2_v07a]
)

noaa20_atms_obs = GPMObs(
    name="noaa20_atms",
    products=[l1c_xcal2019v_noaa20_atms_v07a]
)

npp_atms_obs = GPMObs(
    name="npp_atms",
    products=[l1c_xcal2019v_npp_atms_v07a]
)
