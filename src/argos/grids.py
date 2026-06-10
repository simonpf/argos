"""
Grid definitions for precipitation retrievals.

This module provides standard grid definitions using pyresample.
"""

import pyresample


def get_regular_latlon_grid(
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    resolution: float = 0.025,
) -> pyresample.geometry.AreaDefinition:
    """Create a regular latitude-longitude grid.
    
    Args:
        lat_min: Minimum latitude in degrees. Defaults to -90.0.
        lat_max: Maximum latitude in degrees. Defaults to 90.0.
        lon_min: Minimum longitude in degrees. Defaults to -180.0.
        lon_max: Maximum longitude in degrees. Defaults to 180.0.
        resolution: Grid resolution in degrees. Defaults to 0.025.
        
    Returns:
        Grid area definition.
    """
    # Calculate grid dimensions
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)
    
    # Create area definition
    area_def = pyresample.geometry.AreaDefinition(
        area_id='regular_latlon',
        description='Regular latitude-longitude grid',
        proj_id='latlon',
        projection='+proj=longlat +datum=WGS84',
        width=width,
        height=height,
        area_extent=[lon_min, lat_min, lon_max, lat_max]
    )
    
    return area_def


def get_default_grid() -> pyresample.geometry.AreaDefinition:
    """Get the default 0.025-degree global grid.
    
    Returns:
        Default grid area definition.
    """
    return get_regular_latlon_grid()
