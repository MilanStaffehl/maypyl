# Copyright (c) 2025 Milan Staffehl - subject to the MIT license.
"""Functions for statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._types import AnyFloat, NDArray


__all__ = ["axis_aligned_cell_projection"]


type Array1D = NDArray[tuple[int], np.floating[Any]]
type Array1DCollection = (
    NDArray[tuple[int, int], np.floating[Any]] | Sequence[Array1D]
)
type Image = NDArray[tuple[int, int], np.float64]
type Edges = tuple[
    NDArray[tuple[int], np.float64], NDArray[tuple[int], np.float64]
]


def axis_aligned_cell_projection(
    cell_edges: Array1DCollection,
    values: Array1D,
    weights: Array1D | None = None,
    extent: Array1D | Sequence[AnyFloat] | None = None,
    nx_bins: int = 50,
    ny_bins: int = 50,
    projection_axis: Literal["x", "y", "z"] = "z",
    projection_range: tuple[AnyFloat, AnyFloat] | None = None,
    mode: Literal["sum", "mean"] = "sum",
) -> tuple[Image, Edges]:
    """
    Create an axis-aligned projection image of a given quantity.

    This function allows to create a projection of any quantity given
    by ``value`` on an arbitrary grid structure defined by the grids
    cell edges, provided these cells are aligned along the three cardinal
    directions x, y, and z. Examples could be AMR grids or oct-tree
    grids. The function projects the given value along one of the three
    major axes, creating an image parallel to the plane of the other
    two axes.

    .. hint:: This function iterates over the pixels in the image to
        determine cells that at least partially cover this pixel. This
        is very efficient in cases where cells are so large that they
        cover many pixels, but inefficient in the inverse case of any
        one pixel covering many cells. It is therefore recommended to
        only use this function when the projected size of cells is
        larger or comparable to the size of cells. In the case of large
        pixels and small cells, a simple 2D histogram of values at the
        cell center approximates the result of this function sufficiently
        well and is recommended instead.

    The information on cell geometry must be given as a 6-tuple of
    numpy arrays, or alternatively a numpy array of shape (6, N), where
    N is the number of cells. The six entries must be, in that order:

    .. code:: (x_lower, y_lower, z_lower, x_upper, y_upper, z_upper)

    That is, they must give the position of the lower and upper edge of
    all cells in all three coordinates x, y, and z. The order of these
    cells within each of the six arrays does not matter, **as long as**
    it is kept consistent between all six entries of ``cell_edges``, as
    well as ``values`` and ``weights`` (i.e. the value at index ``i`` in
    ``values`` must belong to the cell with cell edges ``cell_edges[:, i]``).

    The arrays ``values`` and ``weights`` specify the values to project
    as well as the optional weights by which to weigh them. They must
    be arrays of shape (N, ) when given, and must have the same cell
    order as ``cell_edges``.

    The projection can be done in one of two ways: as an integration
    along the line of sight defined by each pixel, or as a mean over
    this column. This is determined by the choice of ``mode``:

    - ``sum``: The value given is simply summed up along the line of
      sight, weighted by the depth of each cell, and optionally the
      provided weights. Mathematically, this evaluates to:

      .. math::

          Q_p(x, y) = \\sum_i q_i(x, y) w_i(x, y) \text{cf}_i \\Delta z_i

      Where :math:`i` iterates over all cells that overlap with current
      pixel :math:`p` when projected to the image plane, :math:`q_i` is
      the value of the quantity in each cell, :math:`w_i` is the weight
      assigned to each cell (which is 1 if no weights are given), and
      :math:`\\Delta z_i` is the depth of the cell along the line of sight.
      :math:`\text{cf}_i` is the "covering fraction", i.e. the projected
      area of the pixel that cell :math:`i` occupies:

      .. math::

          \text{cf}_i = \\dfrac{A_{\text{overlap} \\ i,p}}{A_\text{pixel}}

      This is mode useful to create column density plots when given a
      density per cell, or a surface brightness when given an emissivity.
    - ``mean``: The function finds the (weighted) mean along each line
      of sight. Mathematically, this means the value in each pixel is
      given by

      .. math ::

          Q_p(x, y) = V_p(x, y) / W_p(x, y)

      where the summed value :math:`V_p(x, y)` and the sum over the weights
      :math:`W_p(x, y)` is given by:

      .. math::

          V_p(x, y) = \\sum_i q_i(x, y) w_i(x, y) \text{cf}_i \\Delta z_i \\
          W_p(x, y) = \\sum_i w_i(x, y) \text{cf}_i \\Delta z_i

      This mode is particularly useful to create density-weighted means
      of quantities such as temperature or mass-weighted velocity.

    In both modes, weights are optional. If no weights are desired, set
    ``weights`` to None. In this case, all weights will be automatically
    set to unity.

    Depending on the chosen projection axis, the image axes ``u`` and
    ``v`` (horizontal and vertical image axes respectively) will align
    with two of the principal coordinates of the cell coordinate system:

    - Projection along x: image coordinate ``u`` is equal to the ``y``
      coordinate; image coordinate ``v`` is equivalent to the ``z``
      coordinate.
    - Projection along y: image coordinate ``u`` is equal to the ``x``
      coordinate; image coordinate ``v`` is equivalent to the ``z``
      coordinate.
    - Projection along z: image coordinate ``u`` is equal to the ``x``
      coordinate; image coordinate ``v`` is equivalent to the ``y``
      coordinate.

    The projection can be limited to a specific range of values along
    the projection axis, using ``projection_range``. If this value is
    not specified, all cells along the projection axis will be considered.

    .. warning:: This function does not currently take into account the
        length of the column each pixel projects along. This means that
        if your cells are not starting and ending at the same depth
        along the projection axis, columns will have a different length.
        In such a case, all resulting projections in mode ``sum`` will
        be wrong as they sum over columns of different length. For
        protrusions that are small compared to the average column length,
        this error will be negligible, but it can become significant for
        vastly different column lengths.

    :param cell_edges: Either a 6-tuple of numpy arrays or an array of
        shape (6, N). The six arrays must contain the lower and upper
        edges of all cells, in the order ``(x_lower, y_lower, z_lower,
        x_upper, y_upper, z_upper)``. Any space not covered by cells
        will be considered empty.
    :param values: An array of values in each cell to project. This must
        be an array of shape (N, ) where N is the number of cells.
    :param weights: An optional weight for each cell. When given, this
        must be an array of shape (N, ) where N is the number of cells.
        Each value in the cell will be weighted by this weight. Optional,
        defaults to None which is equivalent to setting all weights to
        unity (i.e. unweighted projection).
    :param extent: The extent of the final image in coordinates of the
        cells in the form ``(u_min, u_max, v_min, v_max)``. All cells
        that lie entirely outside of this extent are ignored. Note that
        u and v are image coordinates, which will be equal to two of the
        principal coordinates of the cells, depending on the choice of
        projection axis. See above for details. When not specified, the
        minimum and maximum along both coordinates will be used. Optional,
        defaults to None, which means min and max coordinates will be
        used.
    :param nx_bins: The number of bins (i.e. pixels) along the horizontal
        axis of the final image. Optional, defaults to 50.
    :param ny_bins: The number of bins (i.e. pixels) along the vertical
        axis of the final image. Optional, defaults to 50.
    :param projection_axis: The axis along which to project the image.
        Since this function only allows axis-aligned projections, this
        must be one of the three principal axes. This parameter must be
        one of the following strings: ``"x"``, ``"y"``, or ``"z"``.
        Defaults to ``"z"``.
    :param projection_range: A tuple of floats in the same units as the
        ``cell_edges``, giving the minimum and maximum coordinate along
        the projection axis within which cells shall be considered. All
        cells that lie entirely outside of this range along the projection
        axis will be discarded and not considered for the projection.
        Optional, defaults to ``None``, which means all cells will be
        used, irrespective of their position along the projection axis.
    :param mode: The projection mode to use. This must either be ``"sum"``
        or ``"mean"`. Mode ``sum`` means that all values along a pixel
        line of sight are summed (weighted by the respective cell depth
        and possibly weights). Mode ``mean`` takes the (weighted) mean
        along this line of sight. See above for details. Defaults to
        ``"sum"``.
    :return (image, (x_edges, y_edges)): A tuple of two entries. The
        first entry is the image as an array of shape (``ny_bins``,
        ``nx_bins``), with the value in each pixel being the projection
        along that pixels line of sight along the chosen projection axis.
        The second entry is a tuple of two arrays, giving the lower edges
        of the pixels along the image's horizontal and vertical axis,
        respectively.
    """
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
        w_low, w_upp = x_low, x_upp
    elif projection_axis == "y":
        u_low, u_upp, v_low, v_upp = x_low, x_upp, z_low, z_upp
        w_low, w_upp = y_low, y_upp
    elif projection_axis == "z":
        u_low, u_upp, v_low, v_upp = x_low, x_upp, y_low, y_upp
        w_low, w_upp = z_low, z_upp
    else:
        raise ValueError(f"Unrecognized projection axis: {projection_axis}")

    # determine depth of each cell along projection axis
    cell_depths = w_upp - w_low

    # if no weights are set, we set them to unity (equivalent to no weights)
    if weights is None:
        weights = np.ones_like(values)

    # set the limits of the image in units of the cells
    if extent is None:
        extent = (np.min(u_low), np.max(u_upp), np.min(v_low), np.max(v_upp))
    u_min, u_max, v_min, v_max = extent

    # similarly, limit the depths of cells considered
    if projection_range is None:
        projection_range = (np.min(w_low), np.max(w_upp))

    # Mask all cells that fall outside the image extent; note that we
    # compare the upper cell edge with the lower image edge, to ensure
    # that also partially covering cells are still considered. Same for
    # comparing the lower cell edge with the upper image edge.
    mask_u = np.logical_and(u_upp >= u_min, u_low <= u_max)
    mask_v = np.logical_and(v_upp >= v_min, v_low <= v_max)
    mask_w = np.logical_and(
        w_upp >= projection_range[0], w_low <= projection_range[1]
    )
    in_image_mask = np.logical_and(mask_u, mask_v)
    in_image_mask = np.logical_and(in_image_mask, mask_w)
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
    pixel_u_low = u_min + np.arange(nx_bins + 1, dtype=np.float64) * px_du
    pixel_v_low = v_min + np.arange(ny_bins + 1, dtype=np.float64) * px_dv
    # allocate memory for the integrated value and weights
    pixel_value = np.zeros(n_pixels, dtype=np.float64)

    # loop over the pixels and sum/find the mean along their normal
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

        # for the current pixel, add the values of all cells, weighted by the
        # covering fraction, the weight, and potentially the cell depth
        px_value = np.sum(
            current_values * current_weights * current_depths * overlap_area
        )
        if mode == "mean":
            px_weights = np.sum(
                current_weights * current_depths * overlap_area
            )
            # area of pixel cancels out, we divide only by sum of weights
            pixel_value[i] = px_value / px_weights
        else:
            pixel_value[i] = px_value / pixel_area

    # reshape the image into the proper shape. Shape must be (Y, X) to
    # preserve correct pixel order (pixels were stored in row-column order)
    image = pixel_value.reshape((ny_bins, nx_bins))
    # create an array of edge values
    edges: Edges = pixel_u_low, pixel_v_low
    return image, edges
