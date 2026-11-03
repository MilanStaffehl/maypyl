# Copyright (c) 2025-2026 Milan Staffehl - subject to the MIT license.
"""Projections of cuboidal (AMR/octree) cell data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._types import AnyFloat, NDArray


__all__ = ["axis_aligned_cell_projection"]


type Array1D = NDArray[tuple[int], np.floating[Any]]
type Array1DCollection = (
    NDArray[tuple[int, int], np.floating[Any]] | Sequence[Array1D]
)
type CellEdgesTuple = tuple[
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
]
type UVCellEdgesTuple = tuple[
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
    NDArray[tuple[int], np.floating[Any]],
]
type PixelSpans = tuple[
    NDArray[tuple[int], np.intp],
    NDArray[tuple[int], np.intp],
    NDArray[tuple[int], np.intp],
    NDArray[tuple[int], np.intp],
]
type ImageArray = NDArray[tuple[int, int], np.float64]
type EdgesTuple = tuple[
    NDArray[tuple[int], np.float64],
    NDArray[tuple[int], np.float64],
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
    mode: Literal["sum", "mean", "max", "min"] = "sum",
    method: Literal["area average", "ray"] = "area average",
) -> tuple[ImageArray, EdgesTuple]:
    r"""
    Create an axis-aligned projection image of a given quantity.

    This function allows to create a projection of any quantity given
    by ``value`` on an arbitrary cartesian grid structure defined by the
    grids cell edges, provided these cells are aligned along the three
    cardinal directions x, y, and z. Examples could be AMR grids or
    oct-tree grids. The function projects the given value along one of
    the three major axes, creating an image parallel to the plane of the
    other two axes.

    What the resulting image shows depends on the choice of ``method``:

    - ``area average``: The default. Image shows, in every pixel, the
      pixel-area-averaged sum or mean along the LOS column of that pixel,
      potentially weighted by the given weights. This is equivalent to
      evaluating the sum or mean along an infinite number of rays
      uniformly covering the entire pixel. As such, it differs from the
      ``ray`` method below in washing out finer details from cells
      much smaller than the pixel. Prefer this method when your cells
      are typically larger or much larger than your pixels, or you can
      afford high pixel resolutions. If instead you require high detail
      fidelity, even at low pixel resolutions, prefer the ``ray`` method
      instead. This method is roughly equivalent to what an instrument
      with extended collection pixels (i.e. a CCD-chip) would see of the
      quantity.
    - ``ray``: The image shows the sum or mean along a single ray,
      centered on each pixel, weighted by the length of each cell and
      potentially the weights. This is the traditional method, used in
      tools such as ``yt``. It preserves high-value peaks instead of
      washing them out by area-averaging, but it can miss high-value
      cells if the cells are very small compared to the pixel size. This
      method does not use supersampling of pixels; it sends only one ray
      per pixel. Prefer this method when your grid cells are aligned
      with the pixel grid, as it then accurately represents the structure
      of your grid at slightly lower computational cost. This method is
      a visualization of projections along a number of ``nx_bins`` x
      ``ny_bins`` of pixels, completely neglecting all cells not traversed
      by these rays. It is therefore often the more theoretically correct
      method (for example when projecting densities to get column
      densities, this method directly follows the theoretical definition
      of a column density), but very high or very low value cells can
      easily be missed if cells are small. This may lead to
      misinterpretations of results when not well understood.

    For cells that are large, or sufficiently high pixel resolution,
    both methods produce similar or identical images.

    .. note:: The function is driven by the cells, not the pixels: every
        cell is mapped to the range of pixels it covers, and its
        contribution is accumulated into exactly those pixels. The cost
        therefore scales with the number of (cell, pixel) pairs that
        actually overlap, i.e. roughly the number of cells plus the number
        of pixels times the typical number of cells along a line of sight,
        instead of the number of pixels times the number of cells. Expanded
        cell--pixel pair arrays are processed in bounded chunks, so their
        temporary memory does not scale with the total number of overlaps.

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

    Every cell must have positive width in all three dimensions. For AMR
    data, pass only active (leaf) cells: refined parent cells are not
    automatically masked and would otherwise be counted alongside their
    children.

    The arrays ``values`` and ``weights`` specify the values to project
    as well as the optional weights by which to weigh them. They must
    be arrays of shape (N, ) when given, and must have the same cell
    order as ``cell_edges``.

    The projection can be done in one of four ways: as an integration
    along the line of sight defined by each pixel, as a mean over
    this column, or as a simple max/min along the line of sight. This is
    determined by the choice of ``mode``:

    - ``sum``: The value given is simply summed up along the line of
      sight, weighted by the depth of each cell, and optionally the
      provided weights. When using the method ``area average``, this
      mathematically evaluates to:

      .. math::

          Q_p(x, y) = \dfrac{1}{A_\text{px}}
          \sum_i q_i w_i A_{\text{overlap},i} \Delta z_i

      Where :math:`i` iterates over all cells that overlap with current
      pixel :math:`p` when projected to the image plane, :math:`q_i` is
      the value of the quantity in each cell, :math:`w_i` is the weight
      assigned to each cell (which is 1 if no weights are given), and
      :math:`\Delta z_i` is the depth of the cell along the line of sight
      (restricted to ``projection_range`` if given).
      :math:`A_{\text{overlap},i}` is the "overlap area" of the i-th
      cell with the current pixel. Normalization with the pixel area
      :math:`A_\text{px}` then finalizes the projection and corrects
      the units.
      When using the method ``ray``, this instead evaluates to:

      .. math::

            Q_p(x, y) = \sum_j q_j w_j \Delta z_j

      where :math:`j` iterates over all cells that the ray in the center
      of the pixel at coordinates :math:`(x_c, y_c)` traverses,
      :math:`q_j` is the value of the quantity in that cell, :math:`w_j`
      is the weight assigned to the cell (which is 1 if no weights are
      given), and :math:`\Delta z_j` is the depth of the cell along the
      ray.
      This mode is useful to create column density plots when given a
      density per cell, or a surface brightness when given an emissivity.
      Pixels not covered by any cell are 0.
    - ``mean``: The function finds the (weighted) mean along each line
      of sight. When using the ``area average`` method, this means that,
      mathematically, the value in each pixel is given by

      .. math ::

          Q_p(x, y) = V_p(x, y) / W_p(x, y)

      where we use the same indices as above, and where the summed value
      :math:`V_p(x, y)` and the sum over the weights :math:`W_p(x, y)`
      are given by:

      .. math::

          V_p(x, y) = \sum_i q_i w_i A_{\text{overlap},i}
            \Delta z_i; \\
          W_p(x, y) = \sum_i w_i A_{\text{overlap},i} \Delta z_i

      In contrast, using method ``ray``, the summed value and weights
      are determined by

      .. math::

          V_p(x, y) = \sum_j q_j w_j \Delta z_j; \\
          W_p(x, y) = \sum_j w_j \Delta z_j

      This mode is particularly useful to create density-weighted means
      of quantities such as temperature or mass-weighted velocity.
      Pixels not covered by any cell, or whose total weight is zero, are
      NaN.
    - ``max``: The function finds the (weighted) maximum along each line
      of sight. When using the ``area average`` method, this means that
      all cells that overlap with the current pixel are taken into account,
      even if their overlap with the pixel is minimal. This can mean that
      pixels with very small overlap can dominate the maximum of many
      adjacent pixels. (Cells that merely touch the pixel edge without
      overlapping it are *not* taken into account.) Mathematically, this
      translates to

      .. math::

          Q_p(x, y) = \max_i (q_i w_i)

      where :math:`i` iterates over all cells overlapping the pixel (for
      method ``ray``: all cells traversed by the central ray). When weights
      are provided, the values of each cell are multiplied by these
      weights, and the maximum of the product is found for each pixel. The
      depth of the cell and the overlap area play no role. NaNs are ignored,
      but infinite values are treated as valid results. Pixels not covered
      by any cell (or covered only by NaNs) are NaN.
    - ``min``: The same as ``max``, except it finds the lowest non-NaN
      value in each line of sight.

    In all modes, weights are optional. If no weights are desired, set
    ``weights`` to None. In this case, all weights will be automatically
    set to unity.

    The modes ``sum`` and ``mean`` propagate NaNs in ``values`` or
    ``weights`` into the affected pixels.

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
    Cells that straddle the boundary of the range contribute only the part
    of their depth that lies inside the range (relevant for ``sum`` and
    ``mean``).

    .. warning:: The function integrates only over the volume represented
        by the supplied cells; it does not infer or fill missing geometry.
        If gaps in the cell data represent missing data rather than
        empty space, ``sum`` projections can underestimate the intended
        full-domain column integral. This is distinct from a physically
        varying path length through an incomplete/irregular region.

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
        axis will be discarded and not considered for the projection;
        cells partially inside the range are cut off at the range
        boundary. Optional, defaults to ``None``, which means all cells
        will be used, irrespective of their position along the projection
        axis.
    :param mode: The projection mode to use. This must either be ``"sum"``,
        ``"mean"``, ``"max"``, or ``"min"``. Mode ``sum`` means that all
        values along a pixel line of sight are summed (weighted by the
        respective cell depth and possibly weights). Mode ``mean`` takes
        the (weighted) mean along this line of sight. ``max`` and ``min``
        Find the maximum or minimum weighted value along each line of
        sight. See above for details. Defaults to ``"sum"``.
    :param method: What method to use to calculate projections. Can be
        either ``area average`` for a pixel-area averaged value, or
        ``ray`` for a value integrated along a single ray from the
        center of each pixel. See above for details. Defaults to ``area
        average``.
    :return (image, (u_edges, v_edges)): A tuple of two entries. The
        first entry is the image as an array of shape (``ny_bins``,
        ``nx_bins``), with the value in each pixel being the projection
        along that pixels line of sight along the chosen projection axis.
        The second entry is a tuple of two arrays, giving the pixel edges
        along the image's horizontal and vertical axis, respectively.
        These have ``nx_bins + 1`` and ``ny_bins + 1`` entries, i.e. they
        include the upper edge of the last pixel, and can be passed
        directly to e.g. ``pcolormesh``.
    """
    # For each projection axis: indices of the (u, v, w) coordinates
    axis_indices = {"x": (1, 2, 0), "y": (0, 2, 1), "z": (0, 1, 2)}

    # validate arguments
    valid_modes = ("sum", "mean", "max", "min")
    if mode not in valid_modes:
        raise ValueError(f"`mode` must be one of {valid_modes}, got {mode!r}")
    valid_methods = ("area average", "ray")
    if method not in valid_methods:
        raise ValueError(
            f"`method` must be one of {valid_methods}, got {method!r}"
        )
    if projection_axis not in axis_indices:
        raise ValueError(
            f"`projection_axis` must be 'x', 'y', or 'z', "
            f"got {projection_axis!r}"
        )
    nx_bins = int(nx_bins)
    ny_bins = int(ny_bins)
    if nx_bins <= 0 or ny_bins <= 0:
        raise ValueError("`nx_bins` and `ny_bins` must be positive")

    # check the arrays for cell edges, values, and weights are ok:
    cell_arrays = _unpack_cell_edges(cell_edges)
    n_cells = cell_arrays[0].shape[0]
    if values.shape != (n_cells,):
        raise ValueError(
            f"`values` must have shape ({n_cells},) to match cells, "
            f"got {values.shape}"
        )
    if weights is not None and weights.shape != (n_cells,):
        raise ValueError(
            f"`weights` must have shape ({n_cells},) to match cells, "
            f"got {values.shape}"
        )

    # Depending on projection axis, we define our (u, v) image plane and
    # the corresponding depth (w) per cell:
    iu, iv, iw = axis_indices[projection_axis]
    u_low, u_upp = cell_arrays[iu], cell_arrays[iu + 3]
    v_low, v_upp = cell_arrays[iv], cell_arrays[iv + 3]
    w_low, w_upp = cell_arrays[iw], cell_arrays[iw + 3]

    # set the limits of the image in units of the cells
    if extent is None:
        extent = (np.min(u_low), np.max(u_upp), np.min(v_low), np.max(v_upp))
    try:
        u_min, u_max, v_min, v_max = extent
    except (ValueError, TypeError) as exc:
        raise ValueError("`extent` must be an iterable of 4 values") from exc
    if not (u_max > u_min and v_max > v_min):
        raise ValueError(
            "`extent` must have u_max > u_min and v_max > v_min, "
            f"got {(u_min, u_max, v_min, v_max)}"
        )

    # discard cells that cannot contribute
    keep = (
        (u_upp > u_min)
        & (u_low < u_max)
        & (v_upp > v_min)
        & (v_low < v_max)
        & (w_upp > w_low)
    )

    # similarly, limit the depths of cells considered
    if projection_range is None:
        projection_range = (np.min(w_low), np.max(w_upp))
    try:
        w_min, w_max = projection_range
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "`projection_range` must contain two finite values (lower, upper)"
        ) from exc
    if w_max <= w_min:
        raise ValueError("`projection_range` must satisfy lower < upper")
    keep &= (w_upp > w_min) & (w_low < w_max)

    # mask arrays of cell edges, values, and weights
    u_low, u_upp, v_low, v_upp, w_low, w_upp = (
        a[keep] for a in (u_low, u_upp, v_low, v_upp, w_low, w_upp)
    )
    q = values[keep]
    wghts = None if weights is None else weights[keep]

    # cut off cells at the projection depth should they extend beyond it
    w_low, w_upp = np.maximum(w_low, w_min), np.minimum(w_upp, w_max)
    depth = w_upp - w_low

    # create a pixel grid
    u_px_edges = np.linspace(u_min, u_max, nx_bins + 1, dtype=np.float64)
    v_px_edges = np.linspace(v_min, v_max, ny_bins + 1, dtype=np.float64)

    # find the range of pixels covered by every cell (f being first, and
    # l being last pixel):
    if method == "area average":
        u_idx_f, u_idx_l = _pixel_span_overlap(u_low, u_upp, u_px_edges)
        v_idx_f, v_idx_l = _pixel_span_overlap(v_low, v_upp, v_px_edges)
    else:
        u_idx_f, u_idx_l = _pixel_span_centers(u_low, u_upp, u_px_edges)
        v_idx_f, v_idx_l = _pixel_span_centers(v_low, v_upp, v_px_edges)
    # create a mask selecting only cells that cover at least one pixel
    valid = (u_idx_l >= u_idx_f) & (v_idx_l >= v_idx_f)

    # prepare quantities for projection
    coef: NDArray[tuple[int], np.floating[Any]]
    coef_w: NDArray[tuple[int], np.floating[Any]] | None
    if mode in ("sum", "mean"):
        weight_depth = depth if wghts is None else wghts * depth
        coef = q * weight_depth
        coef_w = weight_depth
    else:
        coef = q if wghts is None else q * wghts
        coef_w = None

    cell_bounds = tuple(a[valid] for a in (u_low, u_upp, v_low, v_upp))
    pixel_spans = tuple(a[valid] for a in (u_idx_f, u_idx_l, v_idx_f, v_idx_l))
    image = _rasterize(
        mode=mode,
        method=method,
        shape=(ny_bins, nx_bins),
        edges=(u_px_edges, v_px_edges),
        # all per-cell arrays, restricted to cells covering at least 1 pixel
        cell_bounds=cast(UVCellEdgesTuple, cell_bounds),
        pixel_spans=cast(PixelSpans, pixel_spans),
        coef=coef[valid],
        coef_w=None if coef_w is None else coef_w[valid],
    )
    return image.astype(np.float64), (u_px_edges, v_px_edges)


def _unpack_cell_edges(cell_edges: Array1DCollection) -> CellEdgesTuple:
    """
    Unpack and validate input cell edges sequence.

    The function unpacks the given sequence of cell edges into a common
    type and verifies that it has the correct dimensionality and shape.
    The output of this function is then a tuple of 6 arrays of dtype
    float, giving the cell edges in the order ``[xmin, ymin, zmin, xmax,
    ymax, zmax]``.

    :param cell_edges: The input cell edges sequence.
    :raises ValueError: If the input has the wrong shape or dimensionality.
    :return: A tuple of 6 one-dimensional arrays of dtype float, giving
        the cell edges in the order ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
    """
    if isinstance(cell_edges, np.ndarray):
        if cell_edges.ndim != 2 or cell_edges.shape[0] != 6:
            raise ValueError(
                f"`cell_edges` must have shape (6, N), got {cell_edges.shape}"
            )
        arrays = tuple(cell_edges)
    else:
        try:
            raw_arrays = tuple(cell_edges)
        except TypeError as exc:
            raise ValueError(
                "`cell_edges` must be a (6, N) array or a sequence of six arrays"
            ) from exc
        if len(raw_arrays) != 6:
            raise ValueError(
                f"Received {len(raw_arrays)} arrays for `cell_edges`; expected 6"
            )
        arrays = tuple(np.asarray(a) for a in raw_arrays)

    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError(
            "All six `cell_edges` arrays must be 1D and have equal length"
        )
    return cast(CellEdgesTuple, arrays)  # too complex for type checkers...


def _pixel_span_overlap(
    low: NDArray[tuple[int], np.floating[Any]],
    upp: NDArray[tuple[int], np.floating[Any]],
    px_edges: NDArray[tuple[int], np.floating[Any]],
) -> tuple[NDArray[tuple[int], np.intp], NDArray[tuple[int], np.intp]]:
    """
    Find first and last pixel covered by each cell.

    Given an array of cell lower edges along either the u or v axis, and
    an array of the upper edges along the same axis, this function finds
    the indices of the first and last pixel along that axis the cell can
    cover. That means that for each cell, the function returns an index
    for the left-most/lower-most pixel and an index for the right-most/
    upper-most pixel.

    Additionally, the function takes into account cell edges that align
    with pixel edges exactly, correctly ignoring the pixel adjacent to
    the clipped pixel. To avoid floating point imprecision from placing
    a cell inside of a pixel with which it should not have an overlap,
    the function also rejects overlaps below a certain threshold. This
    ensures that cells do not bleed into adjacent pixels due to a tiny
    spurious overlap in modes ``max`` and ``min``.

    This function is used with method ``area average`` as it includes
    contributions from *all* cells overlapping a pixel, not just those
    that would be hit by a ray sent from the center of the pixel.

    :param low: An array of shape (N, ), giving the lower edges of all
        cells along the chosen axis (u- or v-axis of the image).
    :param upp: An array of shape (N, ), giving the upper edges of all
        cells along the chosen axis (u- or v-axis of the image).
    :param px_edges: An array giving all the edges of the pixels along
        the chosen axis (u- or v-axis of the image). This must include
        the upper edge of the last pixel as well (i.e. must include the
        end-point).
    :return: A tuple of two arrays of indices, both of shape (N, ). The
        first array gives, for every cell, the index of the first pixel
        that the cell has any meaningful overlap with. The second array
        gives, for every cell, the index of the last pixel that the cell
        has any meaningful overlap with. See above for details on what
        constitutes a "meaningful" overlap.
    """
    # A cell only counts as overlapping a pixel if the overlap exceeds this
    # fraction of min(cell width, pixel width), along both image axes. This
    # discards floating point slivers when cell faces coincide with pixel
    # edges (typical for AMR grids), which would otherwise let neighbouring
    # cells leak into max/min images.
    overlap_rtol = 1e-9

    n_px = px_edges.size - 1
    width = (px_edges[-1] - px_edges[0]) / n_px

    # pixel i spans [px_edges[i], px_edges[i+1]]; we find, for each cell,
    # the left-most (first) pixel which is covered by a cell. We do this
    # by finding out where the lower edge would be sorted into the array
    # of pixel edges. We use "right" to avoid including entire superfluous
    # pixels should a cell edge line up perfectly with a pixel edge. To
    # get pixel indices, we subtract one, and clip to the range [0,
    # n_px - 1] for cases where searchsorted places us outside the valid
    # range of indices. We do the same for the right-most (last) pixel:
    first_pixel = np.clip(
        np.searchsorted(px_edges, low, side="right") - 1, 0, n_px - 1
    )
    last_pixel = np.clip(
        np.searchsorted(px_edges, upp, side="left") - 1, 0, n_px - 1
    )

    # Calculate overlap of first and last pixel with cell, to correct
    # for spurious overlaps due to floating point imprecision. Only the
    # first and the last pixel can be covered partially by the cell:
    tol = overlap_rtol * np.minimum(upp - low, width)
    overlap_first = np.minimum(upp, px_edges[first_pixel + 1]) - np.maximum(
        low, px_edges[first_pixel]
    )
    overlap_last = np.minimum(upp, px_edges[last_pixel + 1]) - np.maximum(
        low, px_edges[last_pixel]
    )

    # to add/subtract 1, we simply turn a boolean array into an int:
    first_pixel = first_pixel + (overlap_first <= tol).astype(np.intp)
    last_pixel = last_pixel - (overlap_last <= tol).astype(np.intp)

    return first_pixel, last_pixel


def _pixel_span_centers(
    low: NDArray[tuple[int], np.floating[Any]],
    upp: NDArray[tuple[int], np.floating[Any]],
    px_edges: NDArray[tuple[int], np.floating[Any]],
) -> tuple[NDArray[tuple[int], np.intp], NDArray[tuple[int], np.intp]]:
    """
    Find first and last pixel covered by each cell center.

    Given an array of cell lower edges along either the u or v axis, and
    an array of the upper edges along the same axis, this function finds
    the indices of the first and last pixel along that axis whose center
    will overlap with the cell. That means that for each cell, the
    function returns an index for the left-most/lower-most pixel and an
    index for the right-most/upper-most pixel. They are found by
    projecting a ray from the center of each pixel and checking if it
    intersects with the cell.

    This function is used with method `ray`` as it includes contributions
    only from cells overlapping with the *center* of a pixel, i.e. those
    that would be hit by a ray sent from the center of the pixel. It does
    *not* count cells that overlap with the pixel but does not reach its
    center.

    :param low: An array of shape (N, ), giving the lower edges of all
        cells along the chosen axis (u- or v-axis of the image).
    :param upp: An array of shape (N, ), giving the upper edges of all
        cells along the chosen axis (u- or v-axis of the image).
    :param px_edges: An array giving all the edges of the pixels along
        the chosen axis (u- or v-axis of the image). This must include
        the upper edge of the last pixel as well (i.e. must include the
        end-point).
    :return: A tuple of two arrays of indices, both of shape (N, ). The
        first array gives, for every cell, the index of the first pixel
        whose center the cell overlaps with. The second array gives, for
        every cell, the index of the last pixel whose center the cell
        overlaps with.
    """
    centers = 0.5 * (px_edges[:-1] + px_edges[1:])
    first = np.searchsorted(centers, low, side="left")
    last = np.searchsorted(centers, upp, side="left") - 1
    return first, last


def _expand_pairs(
    u_idx_f: NDArray[tuple[int], np.intp],
    v_idx_f: NDArray[tuple[int], np.intp],
    n_u: NDArray[tuple[int], np.intp],
    n_covered_pixels: NDArray[tuple[int], np.intp],
) -> tuple[
    NDArray[tuple[int], np.intp],
    NDArray[tuple[int], np.intp],
    NDArray[tuple[int], np.intp],
]:
    """
    Construct the array of cell indices and corresponding pixel indices.

    This helper function finds for each pixel in a chunk the index to
    the corresponding cell. In addition, the pixel indices are also
    constructed and returned. The idea here is that, given the indices
    of the first pixel covered by each cell in the chunk along both axes
    (represented by the ``u_idx_f`` and ``v_idx_f`` arrays) and the
    number of pixels each of the cells in the chunk covers (given by the
    ``n_covered_pixels`` array), the function finds the index into the
    array of cells in the chunk that belongs to each pixel. Additionally,
    the pixel coordinate in the image is computed and returned.

    :param u_idx_f: Array of the indices of the first pixel along the
        u-axis of the image that each cell covers.
    :param v_idx_f: Array of the indices of the first pixel along the
        v-axis of the image that each cell covers.
    :param n_u: Array of the number of pixels each cell covers along the
        u-axis of the image.
    :param n_covered_pixels: The total number of pixel each cell covers,
        along both axes of the image.
    :return: A tuple of three values: (1) The indices into the array of
        cells in the current chunk for each pixel, of the same length as
        the number of cell-pixel pairs in the chunk, (2) the column index
        of the pixel, and (3) the row index of the pixel.
    """
    n_cells = n_covered_pixels.size
    total = int(n_covered_pixels.sum())
    if total == n_cells:  # common case: every cell sits in a single pixel
        return np.arange(n_cells), u_idx_f, v_idx_f
    cell = np.repeat(np.arange(n_cells), n_covered_pixels)
    offset = np.arange(total)
    offset -= np.repeat(
        np.cumsum(n_covered_pixels) - n_covered_pixels, n_covered_pixels
    )
    row, col = np.divmod(offset, n_u[cell])
    return cell, u_idx_f[cell] + col, v_idx_f[cell] + row


def _rasterize(
    *,
    mode: Literal["mean", "sum", "max", "min"],
    method: Literal["area average", "ray"],
    shape: tuple[int, int],
    edges: EdgesTuple,
    cell_bounds: UVCellEdgesTuple,
    pixel_spans: PixelSpans,
    coef: NDArray[tuple[int], np.floating[Any]],
    coef_w: NDArray[tuple[int], np.floating[Any]] | None,
) -> ImageArray:
    """
    Create a projection map by rasterizing cell contributions.

    Given a target image shape ``shape`` and pixel edges ``edges`` in
    u-v-space, this function accumulates the contributions of each cell
    to every pixel according to both the ``method`` and ``mode`` provided.
    The result is a 2D array of shape ``(Y, X)`` (row-major order).

    :param mode: The projection mode to use. This must either be ``"sum"``,
        ``"mean"``, ``"max"``, or ``"min"``.
    :param method: The method that was employed to get the pixel spans.
        This must either be ``"area average"`` or ``"ray"``.
    :param shape: The target shape ``(Y, X)`` of the image, i.e. the
        number of pixels on each side of the final image. Note that the
        y-shape must be specified first.
    :param edges: A tuple of cell edges for both axes, matching ``shape``.
        This must be in the same units and coordinates as the cell data.
    :param cell_bounds: The boundaries of the cells to project. This must
        be four arrays, giving the lower and upper cell edges in the
        u-v-plane of the image as ``[u_low, u_upp, v_low, v_upp]``.
    :param pixel_spans: A tuple of four arrays, giving for every cell
        the index of the first and last pixel along each axis that the
        cell covers according to ``method``. Must be given in the order
        ``[u_first, u_last, v_first, v_last]``.
    :param coef: An array giving, for every cell, the contribution of
        that cell to the image. This must already include all weights
        and width weighting, but it must not yet include the overlap
        area weighting.
    :param coef_w: If ``mode`` is ``"mean"``, this must be an array of
        the same shape as ``coef``, giving the weights of every cell,
        i.e. the product of weights and depth weight. Otherwise, this
        can be None.
    :return: The projected image for the given input data, as a row-major
        ordered array of shape ``(Y, X)``. This can be fed directly into
        ``matplotlib``'s ``pcolormesh`` without transposing it first.
    """
    # Maximum number of expanded (cell, pixel) pairs processed at once
    max_pairs = 2_000_000

    ny, nx = shape
    u_px_edges, v_px_edges = edges
    u_low, u_upp, v_low, v_upp = cell_bounds
    u_idx_f, u_idx_l, v_idx_f, v_idx_l = pixel_spans
    n_pixels = nx * ny
    px_du = (u_px_edges[-1] - u_px_edges[0]) / nx
    px_dv = (v_px_edges[-1] - v_px_edges[0]) / ny

    # allocate accumulation arrays
    if mode in ("sum", "mean"):
        acc = np.zeros(n_pixels)
    else:
        acc = np.full(n_pixels, np.nan)
    acc_w = np.zeros(n_pixels) if mode == "mean" else None

    # number of pixel in u-direction covered by each cell
    n_u = u_idx_l - u_idx_f + 1
    n_covered_pixels = n_u * (v_idx_l - v_idx_f + 1)
    ends = np.cumsum(n_covered_pixels)  # used for determining chunks

    def accumulate_pairs(
        cell_slice: slice,
        cell_idx: NDArray[tuple[int], np.intp],
        i: NDArray[tuple[int], np.intp],
        j: NDArray[tuple[int], np.intp],
    ) -> None:
        """
        Accumulate contributions from a chunk of cell-pixel pairs.

        Given a chunk of pixels associated with the cells that cover
        them, this function finds the contribution of each of these pairs
        to the pixel value and updates the previously allocated
        accumulation array ``acc`` with the contribution. What this
        contribution is depends on the ``mode`` of the function call.
        If mode is ``"mean"``, the accumulation array for the weights is
        also updated.

        For modes ``"min"`` and ``"max"`` the accumulation is simple:
        The function merely checks if any cell associated with a pixel
        updates the currently set min or max value. NaNs are always
        overwritten by the newly found min/max.

        For the sum, the function adds the contribution of each cell to
        the corresponding pixel, and for ``"mean"``, it does the same.
        Both additions take into account potentially provided weights.
        The summation will be done according to the method chosen, i.e.
        with method ``area average``, the fraction of the area overlap
        of each cell with the pixel is considered.

        :param cell_slice: A slice that, when applied to the array of
            cell edges, gives only those cells that are in the current
            chunk.
        :param cell_idx: For each pixel-cell pair derived from the pixel
            spans of each cell, this array gives an index relative to
            ``cell_slice`` to the corresponding original cell. In other
            words: ``u_upp[cell_slice][cell_idx]`` gives exactly the
            upper edge of the cell associated with each pixel.
        :param i: The column index of the pixel.
        :param j: The row index of the pixel.
        :return: None, the accumulation array is altered in place.
        """
        flat_idx = j * nx + i  # index into flattened acc array
        if mode == "max":
            np.fmax.at(acc, flat_idx, coef[cell_slice][cell_idx])
            return
        if mode == "min":
            np.fmin.at(acc, flat_idx, coef[cell_slice][cell_idx])
            return

        contrib = coef[cell_slice][cell_idx]
        if acc_w is None:
            contrib_w = None
        else:
            assert coef_w is not None  # satisfy type checkers
            contrib_w = coef_w[cell_slice][cell_idx]
        if method == "area average":
            # Update for coverage fraction: find the overlap area of
            # each pixel with the corresponding cell:
            u_low_current = u_low[cell_slice][cell_idx]
            u_upp_current = u_upp[cell_slice][cell_idx]
            v_low_current = v_low[cell_slice][cell_idx]
            v_upp_current = v_upp[cell_slice][cell_idx]
            overlap_u = np.minimum(
                u_upp_current, u_px_edges[i + 1]
            ) - np.maximum(u_low_current, u_px_edges[i])
            overlap_v = np.minimum(
                v_upp_current, v_px_edges[j + 1]
            ) - np.maximum(v_low_current, v_px_edges[j])
            frac = (overlap_u / px_du) * (overlap_v / px_dv)
            contrib = contrib * frac
            if contrib_w is not None:
                contrib_w = contrib_w * frac

        # Omitting `minlength` avoids allocating a full-image temporary
        # when this batch only touches a low-index subset of the pixels:
        chunk_sum = np.bincount(flat_idx, weights=contrib)
        acc[: chunk_sum.size] += chunk_sum
        if acc_w is not None:
            chunk_weight = np.bincount(flat_idx, weights=contrib_w)
            acc_w[: chunk_weight.size] += chunk_weight

    # Process cells in chunks
    start = 0  # starting index of the pixel-cell-pair in chunk
    while start < n_covered_pixels.size:
        if n_covered_pixels[start] > max_pairs:
            # single cell already covers more pixels than we allow per
            # chunk, so we chunk THAT cell, too:
            cell_slice_ = slice(start, start + 1)
            for pair_start in range(
                0, int(n_covered_pixels[start]), max_pairs
            ):
                pair_stop = min(
                    pair_start + max_pairs, int(n_covered_pixels[start])
                )
                offset = np.arange(pair_start, pair_stop, dtype=np.intp)
                row, col = np.divmod(offset, n_u[start])
                cell_ = np.zeros(offset.size, dtype=np.intp)  # only one cell
                accumulate_pairs(
                    cell_slice_,
                    cell_,
                    u_idx_f[start] + col,
                    v_idx_f[start] + row,
                )
            start += 1  # advance to next cell
            continue  # retry again with the new starting index

        base = ends[start - 1] if start else 0
        stop = int(np.searchsorted(ends, base + max_pairs, side="right"))
        stop = max(stop, start + 1)
        cell_slice_ = slice(start, stop)
        cell_, i_, j_ = _expand_pairs(
            u_idx_f[cell_slice_],
            v_idx_f[cell_slice_],
            n_u[cell_slice_],
            n_covered_pixels[cell_slice_],
        )
        accumulate_pairs(cell_slice_, cell_, i_, j_)
        start = stop

    if acc_w is not None:  # mean: normalize by the summed weights
        acc = np.divide(
            acc, acc_w, out=np.full(n_pixels, np.nan), where=acc_w != 0
        )
    # shape is (Y, X): pixels were stored in row-major order
    return acc.reshape((ny, nx))
