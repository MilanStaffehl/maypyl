"""Functions to plot 2D datasets as images."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import (  # type: ignore[import-untyped]
    make_axes_locatable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from maypyl._types import NDArray


@overload
def histogram2d(
    fig: Figure,
    axes: Axes,
    histogram_2d: NDArray[tuple[int, int], np.floating[Any]],
    ranges: Sequence[float] | NDArray[tuple[int], np.floating[Any]],
    xlabel: str | None = ...,
    ylabel: str | None = ...,
    title: str | None = ...,
    value_range: tuple[float | None, float | None] | None = ...,
    colormap: str | Colormap = ...,
    cbar_label: str = ...,
    cbar_ticks: NDArray[tuple[int], np.floating[Any]] | None = ...,
    scale: Literal["linear", "log"] = ...,
    labelsize: int = ...,
    suppress_colorbar: bool = ...,
    return_scalar_mappable: Literal[False] = ...,
) -> tuple[Figure, Axes]: ...


@overload
def histogram2d(
    fig: Figure,
    axes: Axes,
    histogram_2d: NDArray[tuple[int, int], np.floating[Any]],
    ranges: Sequence[float] | NDArray[tuple[int], np.floating[Any]],
    xlabel: str | None = ...,
    ylabel: str | None = ...,
    title: str | None = ...,
    value_range: tuple[float | None, float | None] | None = ...,
    colormap: str | Colormap = ...,
    cbar_label: str = ...,
    cbar_ticks: NDArray[tuple[int], np.floating[Any]] | None = ...,
    scale: Literal["linear", "log"] = ...,
    labelsize: int = ...,
    suppress_colorbar: bool = ...,
    return_scalar_mappable: Literal[True] = ...,
) -> tuple[Figure, Axes, ScalarMappable]: ...


def histogram2d(  # noqa: C901
    fig: Figure,
    axes: Axes,
    histogram_2d: NDArray[tuple[int, int], np.floating[Any]],
    ranges: Sequence[float] | NDArray[tuple[int], np.floating[Any]],
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    value_range: tuple[float | None, float | None] | None = None,
    colormap: str | Colormap = "inferno",
    cbar_label: str = "Count",
    cbar_ticks: NDArray[tuple[int], np.floating[Any]] | None = None,
    scale: Literal["linear", "log"] = "linear",
    labelsize: int = 12,
    suppress_colorbar: bool = False,
    return_scalar_mappable: bool = False,
) -> tuple[Figure, Axes] | tuple[Figure, Axes, ScalarMappable]:
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
        [vmin, vmax], where both can be either a float giving the limiting
        value, or ``None`` to set a limit automatically. This allows
        limiting values only in one direction, but not the other. The
        colorbar will automatically receive caps, based on whether any
        values exceed the corresponding limit. When values are given,
        they must be in the same scale and units as the histogram data.
        Defaults to None, which means the full range of values will be
        considered.
    :param colormap: A matplotlib colormap for the plot. Defaults to
        "inferno".
    :param cbar_label: The label for the colorbar data. Defaults to
        "Count". The label is not automatically updated for logarithmic
        scales, so when using ``scale="log"``, the label must be chosen
        appropriately.
    :param cbar_ticks: Sequence or array of the tick markers for the
        colorbar. Optional, defaults to None (automatically chosen ticks).
    :param scale: If the histogram data is not already given in log
        scale, this parameter can be set to "log" to plot the log10 of
        the given histogram data.
    :param labelsize: Size of the axes labels in points. Optional,
        defaults to 12 pt.
    :param suppress_colorbar: When set to True, no colorbar is added to
        the figure.
    :param return_scalar_mappable: When set to True, the function will
        return the ``ScalarMappable`` instance returned by ``imshow``
        alongside the figure and axes instances. This is useful for
        adding a colorbar manually later, for example in multi-axes
        plots that will share a common colorbar.
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
    norm_class: type[matplotlib.colors.Normalize] = matplotlib.colors.Normalize
    if scale == "log":
        norm_class = matplotlib.colors.LogNorm
    elif scale != "linear":
        raise ValueError("Scale must be either 'log' or 'linear'.")

    # limit range (clip values and determine open ends of colorbar)
    min_value = np.min(histogram_2d[np.isfinite(histogram_2d)])
    max_value = np.max(histogram_2d[np.isfinite(histogram_2d)])
    cbar_extend = "neither"
    if value_range is not None:
        if len(value_range) != 2:
            warnings.warn(
                "The sequence of limits for the colorbar does not have length "
                "2. Only first two values will be used for limits. This might "
                "cause unexpected behavior!",
                RuntimeWarning,
                stacklevel=2,
            )
        # determine value range
        if value_range[0] is not None:
            min_value = value_range[0]
        if value_range[1] is not None:
            max_value = value_range[1]
        # determine correct colorbar extent
        upper_clipped = np.count_nonzero(histogram_2d > max_value)
        lower_clipped = np.count_nonzero(histogram_2d < min_value)
        if upper_clipped > 0:
            cbar_extend = "max"
        if lower_clipped > 0:
            cbar_extend = "min"
        if upper_clipped > 0 and lower_clipped > 0:
            cbar_extend = "both"

    # check validity of values in log scale
    if scale == "log" and min_value <= 0:
        raise ValueError(
            "Histogram contains negative values or zero, cannot use "
            "log scale. Set appropriate value ranges to clip values."
        )

    # plot the 2D hist
    hist_config: dict[str, Any] = {
        "cmap": colormap,
        "interpolation": "nearest",
        "origin": "lower",
        "aspect": "auto",
        "extent": ranges,
    }
    if value_range is not None:
        norm = norm_class(vmin=min_value, vmax=max_value, clip=True)
        hist_config.update({"norm": norm})
    profile = axes.imshow(histogram_2d, **hist_config)

    # add colorbar
    if not suppress_colorbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_config: dict[str, Any] = {
            "location": "right",
            "label": cbar_label,
            "extend": cbar_extend,
        }
        if cbar_ticks is not None:
            cbar_config.update({"ticks": cbar_ticks})
        fig.colorbar(profile, cax=cax, **cbar_config)

    if return_scalar_mappable:
        return fig, axes, profile
    return fig, axes
