# Copyright (c) 2025 Milan Staffehl - subject to the MIT license.
"""Functions for statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._types import NDArray


__all__ = ["axis_aligned_cell_projection"]


type Array1D = NDArray[tuple[int], np.floating[Any]]
type Array1DCollection = (
    NDArray[tuple[int, int], np.floating[Any]] | Sequence[Array1D]
)
type Image = NDArray[tuple[int, int], np.float64]
type Edges = NDArray[tuple[int, int], np.float64]


def axis_aligned_cell_projection(
    cell_edges: Array1DCollection,
    values: Array1D,
    weights: Array1D | None = None,
    extent: Array1D | Sequence[float | np.floating[Any]] | None = None,
    nx_bins: int = 50,
    ny_bins: int = 50,
    projection_axis: Literal["x", "y", "z"] = "z",
    mode: Literal["sum", "mean"] = "sum",
) -> tuple[Image, Edges]:
    # NOTE: integrate:
    if mode not in ["sum", "mean"]:
        raise ValueError("`mode` must be 'integrate' or 'mean")
    # verify cell edges and unpack their values
    if isinstance(cell_edges, np.ndarray):
        if not cell_edges.ndim == 2 and cell_edges.shape[0] == 6:
            raise ValueError("`cell_edges` must have shape (6, N)")
    elif len(cell_edges) != 6:
        raise ValueError(
            f"Received {len(cell_edges)} 1D arrays for the cell edges, "
            f"expected 6."
        )
    elif not all([isinstance(x, np.ndarray) for x in cell_edges]):
        cell_edges = [np.array(x) for x in cell_edges]
    x_low, y_low, z_low, x_upp, y_upp, z_upp = cell_edges

    # Depending on projection axis, we define our (u, v) image plane and
    # the corresponding depth per cell:
    if projection_axis == "x":
        u_low, u_upp, v_low, v_upp = y_low, y_upp, z_low, z_upp
        cell_depths = x_upp - x_low
    elif projection_axis == "y":
        u_low, u_upp, v_low, v_upp = x_low, x_upp, z_low, z_upp
        cell_depths = y_upp - y_low
    elif projection_axis == "z":
        u_low, u_upp, v_low, v_upp = x_low, x_upp, y_low, y_upp
        cell_depths = z_upp - z_low
    else:
        raise ValueError(f"Unrecognized projection axis: {projection_axis}")

    # if no weights are set, we set them to unity (equivalent to no weights)
    if weights is None:
        weights = np.ones_like(values)

    # set the limits of the image in units of the cells
    if extent is None:
        extent = (np.min(u_low), np.max(u_upp), np.min(v_low), np.max(v_upp))
    u_min, u_max, v_min, v_max = extent

    # Mask all cells that fall outside the image extent; note that we
    # compare the upper cell edge with the lower image edge, to ensure
    # that also partially covering cells are still considered. Same for
    # comparing the lower cell edge with the upper image edge.
    mask_u = np.logical_and(u_upp >= u_min, u_low <= u_max)
    mask_v = np.logical_and(v_upp >= v_min, v_low <= v_max)
    in_image_mask = np.logical_and(mask_u, mask_v)
    u_low, u_upp = u_low[in_image_mask], u_upp[in_image_mask]
    v_low, v_upp = v_low[in_image_mask], v_upp[in_image_mask]
    cell_depths = cell_depths[in_image_mask]
    values = values[in_image_mask]
    weights = weights[in_image_mask]

    # create a pixel image grid
    n_pixels = nx_bins * ny_bins
    px_du = (u_max - u_min) / nx_bins  # width of a pixel in u-direction
    px_dv = (v_max - v_min) / ny_bins  # width of a pixel in v-direction
    pixel_area = px_du * px_dv
    pixel_u_low = u_min + np.arange(nx_bins + 1) * px_du
    pixel_v_low = v_min + np.arange(ny_bins + 1) * px_dv
    # allocate memory for the integrated value and weights
    pixel_value = np.zeros(n_pixels, dtype=np.float64)

    # loop over the pixels and sum/integrate along their normal
    for i in range(n_pixels):
        # shorthands for current values
        row_idx = int(np.floor(i / nx_bins))
        col_idx = int(i - row_idx * nx_bins)
        px_u_low, px_u_upp = pixel_u_low[col_idx], pixel_u_low[col_idx + 1]
        px_v_low, px_v_upp = pixel_v_low[row_idx], pixel_v_low[row_idx + 1]

        # mask all cells that lie outside the current pixel for sure
        mask_u = np.logical_and(u_upp >= px_u_low, u_low <= px_u_upp)
        mask_v = np.logical_and(v_upp >= px_v_low, v_low <= px_v_upp)
        in_pixel_mask = np.logical_and(mask_u, mask_v)
        current_values = values[in_pixel_mask]
        current_weights = weights[in_pixel_mask]
        current_depths = cell_depths[in_pixel_mask]
        u_low_curr, u_upp_curr = u_low[in_pixel_mask], u_upp[in_pixel_mask]
        v_low_curr, v_upp_curr = v_low[in_pixel_mask], v_upp[in_pixel_mask]

        # for these selected pixels, calculate the covering fraction
        du_overlap = np.maximum(
            0,
            np.minimum(u_upp_curr, px_u_upp)
            - np.maximum(u_low_curr, px_u_low),
        )
        dv_overlap = np.maximum(
            0,
            np.minimum(v_upp_curr, px_v_upp)
            - np.maximum(v_low_curr, px_v_low),
        )
        overlap_area = dv_overlap * du_overlap
        cf = np.minimum(1, overlap_area / pixel_area)

        # for the current pixel, add the values of all cells, weighted by the
        # covering fraction, the weight, and potentially the cell depth
        px_value = np.sum(
            current_values * current_weights * current_depths * cf
        )
        if mode == "mean":
            px_weights = np.sum(current_weights * current_depths * cf)
            pixel_value[i] = px_value / px_weights
        else:
            pixel_value[i] = px_value

    # reshape the image into the proper shape. Shape must be (Y, X) to
    # preserve correct pixel order (pixels were stored in row-column order)
    image = pixel_value.reshape((ny_bins, nx_bins))
    # create an array of edge values
    edges = np.array([pixel_u_low, pixel_v_low])
    return image, edges
