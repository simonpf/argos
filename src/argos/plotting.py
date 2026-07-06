"""
argos.plotting
==============

Plotting utilities for argos surface-precipitation retrievals.

:func:`animate_results` renders one or more retrieval datasets (each holding a
``surface_precip`` field with a ``time`` coordinate) as a time animation on a
cartopy map with coastlines. The datasets need not share the same time steps:
every dataset is interpolated in time onto a common animation axis, so they
advance together.
"""
from typing import List, Optional, Sequence, Union

import numpy as np
import xarray as xr

__all__ = ["animate_results"]


def _dataset_times(ds: xr.Dataset) -> Optional[np.ndarray]:
    """The dataset's time coordinate as a 1-D array, or ``None`` if absent."""
    if "time" not in ds.coords and "time" not in ds.variables:
        return None
    return np.atleast_1d(np.asarray(ds["time"].values)).astype("datetime64[ns]")


def _common_times(
    datasets: Sequence[xr.Dataset],
    times: Optional[np.ndarray],
    n_frames: Optional[int],
) -> np.ndarray:
    """Build the common animation time axis for the datasets."""
    if times is not None:
        return np.asarray(times, dtype="datetime64[ns]")

    per_dataset = [t for t in (_dataset_times(ds) for ds in datasets) if t is not None]
    if not per_dataset:
        raise ValueError("None of the datasets carries a 'time' coordinate.")
    union = np.unique(np.concatenate(per_dataset))

    if n_frames is None or union.size <= 1:
        return union
    # A regular grid of ``n_frames`` times spanning the overall time range.
    span = union[-1] - union[0]
    fracs = np.linspace(0.0, 1.0, int(n_frames))
    return (union[0] + fracs * span).astype("datetime64[ns]")


def _interp_to_times(ds: xr.Dataset, times: np.ndarray) -> xr.DataArray:
    """
    Interpolate a dataset's ``surface_precip`` onto ``times``.

    Datasets with several time steps are linearly interpolated (frames outside
    their own time range become ``NaN``); a single-time dataset is held constant,
    and a dataset without a time dimension is broadcast across all frames.
    """
    da = ds["surface_precip"]
    if "time" not in da.dims:
        return da.expand_dims(time=times).transpose("time", ...)
    if da.sizes["time"] == 1:
        return da.reindex(time=times, method="nearest")
    return da.interp(time=times)


def animate_results(
    datasets: Union[xr.Dataset, Sequence[xr.Dataset]],
    times: Optional[Sequence[np.datetime64]] = None,
    n_frames: Optional[int] = None,
    titles: Optional[Sequence[str]] = None,
    vmin: float = 0.1,
    vmax: float = 50.0,
    cmap: str = "turbo",
    norm=None,
    figsize: Optional[tuple] = None,
    interval: int = 200,
    coastline_resolution: str = "50m",
    save_path: Optional[str] = None,
    fps: int = 5,
):
    """
    Animate surface-precipitation retrievals over time.

    Each dataset is drawn in its own panel on a cartopy map (with coastlines) and
    animated over a shared time axis. Because the datasets may be sampled at
    different times, every dataset is interpolated in time onto the common axis
    (see :func:`_interp_to_times`).

    Args:
        datasets: A dataset or list of datasets, each with a ``surface_precip``
            field (dimensions ``(time, latitude, longitude)``, or
            ``(latitude, longitude)`` for a constant field) and
            ``latitude``/``longitude`` coordinates.
        times: Explicit animation times. If omitted, the union of the datasets'
            times is used (optionally resampled to ``n_frames``).
        n_frames: If given (and ``times`` is not), resample the overall time range
            to this many equally spaced frames.
        titles: Optional per-panel titles.
        vmin: Lower bound of the (logarithmic) color scale, in mm/h.
        vmax: Upper bound of the color scale, in mm/h.
        cmap: Name of the colormap.
        norm: Optional explicit :class:`matplotlib.colors.Normalize`. Defaults to
            a :class:`~matplotlib.colors.LogNorm` over ``[vmin, vmax]``.
        figsize: Figure size; defaults to ``(4.5 * n_panels, 4.5)``.
        interval: Delay between frames in milliseconds.
        coastline_resolution: Resolution of the cartopy coastlines
            (``"110m"``, ``"50m"`` or ``"10m"``).
        save_path: If given, save the animation to this path (e.g. ``.gif`` or
            ``.mp4``) using ``fps``.
        fps: Frames per second used when saving.

    Returns:
        The :class:`matplotlib.animation.FuncAnimation` instance.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib import animation, colors

    if isinstance(datasets, xr.Dataset):
        datasets = [datasets]
    datasets = list(datasets)
    if not datasets:
        raise ValueError("Provide at least one dataset.")

    frame_times = _common_times(datasets, times, n_frames)
    fields = [_interp_to_times(ds, frame_times) for ds in datasets]

    n = len(datasets)
    if norm is None:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.9")  # grey for missing/zero precipitation

    if figsize is None:
        figsize = (4.5 * n, 4.5)
    # Data are on a regular lon/lat grid, so use a plate-carrée map projection.
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, n, figsize=figsize, squeeze=False,
        subplot_kw={"projection": data_crs},
    )
    axes = axes.ravel()

    meshes = []
    for i, (ax, ds, da) in enumerate(zip(axes, datasets, fields)):
        lon = np.asarray(ds["longitude"].values)
        lat = np.asarray(ds["latitude"].values)
        first = np.ma.masked_invalid(da.isel(time=0).values)
        mesh = ax.pcolormesh(
            lon, lat, first, norm=norm, cmap=cmap_obj, shading="nearest",
            transform=data_crs,
        )
        ax.coastlines(resolution=coastline_resolution, linewidth=0.8)
        ax.set_extent(
            [lon.min(), lon.max(), lat.min(), lat.max()], crs=data_crs
        )
        gridlines = ax.gridlines(
            draw_labels=True, linewidth=0.4, color="0.6", alpha=0.5
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        ax.set_title(titles[i] if titles is not None else f"Dataset {i + 1}")
        meshes.append(mesh)

    fig.colorbar(
        meshes[-1], ax=list(axes), shrink=0.85,
        label=r"Surface precip. (mm h$^{-1}$)",
    )
    suptitle = fig.suptitle(np.datetime_as_string(frame_times[0], unit="m"))

    def update(frame: int):
        for mesh, da in zip(meshes, fields):
            data = np.ma.masked_invalid(da.isel(time=frame).values)
            mesh.set_array(data.ravel())
        suptitle.set_text(np.datetime_as_string(frame_times[frame], unit="m"))
        return [*meshes, suptitle]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_times), interval=interval, blit=False
    )
    if save_path is not None:
        ani.save(save_path, fps=fps)
    return ani
