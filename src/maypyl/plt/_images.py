"""Functions to plot 2D datasets as images."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from maypyl._types import NDArray


def histogram2d(
    fig: Figure,
    axes: Axes,
    histogram_2d: NDArray[tuple[int, int], np.floating[Any]],
    ranges: Sequence[float] | NDArray[tuple[int], np.floating[Any]],
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    value_range: Sequence[float] | None = None,
    colormap: str | Colormap = "inferno",
    cbar_label: str = "Count",
    cbar_ticks: NDArray[tuple[int], np.floating[Any]] | None = None,
    cbar_limits: Sequence[float | None] | None = None,
    scale: Literal["linear", "log"] = "linear",
    labelsize: int = 12,
    suppress_colorbar: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot the given 2D histogram.

    Plots a histogram given as a 2D array. The data must already exist
    as histogram data, for example created by ``np.histogram2d``. The
    plot is augmented by the optional given parameters. These can be
    used to set the axis labels, figure title, the colorbar scale and
    ticks, as well as the size of the labels.

    Function returns figure and axis objects, and does NOT save the plot
    to file or displays it. Returned figure must be saved separately.

    .. attention:: The expected histogram array is ordered ``(y, x)``,
        as opposed to the order of the return value of many histogram-
        generating functions such as ``numpy.histogram2d``, which will
        give an array of shape (x, y). If you simply feed the return
        value of ``numpy.histogram2d`` into this function, the histogram
        will appear rotated by 90 degrees. Use ``histogram.transpose()``
        to transform the histogram to the expected order first!

    :param fig: The figure object onto whose axes the histogram will be
        plotted.
    :param axes: The axes onto which to plot the histogram and the
        colorbar.
    :param histogram_2d: Array of 2D histograms of shape (Y, R) where
        R is the number of radial bins of the histogram and Y the number
        of y-bins, for example temperature bins.
    :param ranges: The min and max value for every axis in the format
        [xmin, xmax, ymin, ymax].
    :param xlabel: The label for the x-axis; can be a raw string. Can be
        set to None, to not set an axes label.
    :param ylabel: The label for the y-axis; can be a raw string. Can be
        set to None, to not set an axes label.
    :param title: Title of the figure. Set to None to leave the figure
        without a title. Can be a raw string to use formulas.
    :param value_range: The range of values for the histogram. If given,
        all values are limited to this range. Must be of the form
        [vmin, vmax].
    :param colormap: A matplotlib colormap for the plot. Defaults to
        "inferno".
    :param cbar_label: The label for the colorbar data. Defaults to
        "Count". The label is not automatically updated for logarithmic
        scales, so when using ``scale="log"``, the label must be chosen
        appropriately.
    :param cbar_ticks: Sequence or array of the tick markers for the
        colorbar. Optional, defaults to None (automatically chosen ticks).
    :param cbar_limits: The lower and upper limit of the colorbars as a
        sequence [lower, upper]. All values above and below will be
        clipped. Setting these values will assume that the colorbar is
        not showing the full range of values, so the ends of the colorbar
        will be turned into open ends (with an arrow-end instead of a
        flat cap). To only limit the colorbar in one direction, set the
        other to None: ``cbar_limits=(-1, None)``. Set to None to show
        the full range of values in the colorbar. If ``log`` is set to
        True, the limits must be given in logarithmic values.
    :param scale: If the histogram data is not already given in log
        scale, this parameter can be set to "log" to plot the log10 of
        the given histogram data.
    :param labelsize: Size of the axes labels in points. Optional,
        defaults to 12 pt.
    :param suppress_colorbar: When set to True, no colorbar is added to
        the figure.
    :return: The figure and axes objects as tuple with the histogram
        added to them; returned for convenience, axes object is altered
        in place.
    """
    # axes config
    if title:
        axes.set_title(title)
    if xlabel:
        axes.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        axes.set_ylabel(ylabel, fontsize=labelsize)

    # scaling
    if scale == "log":
        with np.errstate(divide="ignore"):
            histogram_2d = np.log10(histogram_2d)

    # clipping (clip values and determine open ends of colorbar)
    if cbar_limits is not None:
        if len(cbar_limits) != 2:
            warnings.warn(
                "The sequence of limits for the colorbar does not have length "
                "2. Only first two values will be used for limits. This might "
                "cause unexpected behavior!",
                RuntimeWarning,
                stacklevel=2,
            )
        lower_limit, upper_limit = -np.inf, np.inf
        cbar_extend = "neither"
        if cbar_limits[0] is not None:
            lower_limit = cbar_limits[0]
            cbar_extend = "min"
        if cbar_limits[1] is not None:
            upper_limit = cbar_limits[1]
            cbar_extend = "max"
        # clip histogram
        histogram_2d = np.clip(histogram_2d, lower_limit, upper_limit)
        # determine correct colorbar extent
        if all(cbar_limits):
            cbar_extend = "both"

    # plot the 2D hist
    hist_config: dict[str, Any] = {
        "cmap": colormap,
        "interpolation": "nearest",
        "origin": "lower",
        "aspect": "auto",
        "extent": ranges,
    }
    if value_range is not None:
        hist_config.update({"vmin": value_range[0], "vmax": value_range[1]})
    profile = axes.imshow(histogram_2d, **hist_config)

    # add colorbar
    if not suppress_colorbar:
        cbar_config: dict[str, Any] = {
            "location": "right",
            "label": cbar_label,
        }
        if cbar_ticks is not None:
            cbar_config.update({"ticks": cbar_ticks})
        if cbar_limits is not None:
            cbar_config.update({"extend": cbar_extend})
        fig.colorbar(profile, ax=axes, **cbar_config)

    return fig, axes
