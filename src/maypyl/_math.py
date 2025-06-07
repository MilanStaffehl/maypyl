# Copyright (c) 2025 Milan Staffehl - subject to the MIT license.
"""Simple math functions and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._types import NDArray

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def nanlog10(
    x: ArrayLike,
    sentinel: float | int | None = None,
    *,
    relative: bool = False,
) -> NDArray[tuple[int, ...], np.generic]:
    """
    Decadic logarithm with replacement for invalid results.

    Function takes an array or scalar and computes the logarithm to base
    10, suppressing all warnings caused by zeros and negative values.
    Zero values resulting in ``-inf`` and negative values resulting in
    ``NaN`` are then replaced by a sentinel value. The sentinel can
    either be chosen explicitly with ``sentinel`` or it can be computed
    as a fraction of the smallest value in the array by specifying this
    fraction as ``sentinel`` and setting ``relative`` to True.

    Example:

    >>>arr = np.array([0, 10, -1])
    >>>nanlog10(arr)
    array([-inf,   1., nan])
    >>>nanlog10(arr, np.nan)
    array([nan,   1., nan])
    >>>nanlog10(arr, 0.1, relative=True)
    array([0.1, 1., 0.1])

    :param x: Array or scalar to take the logarithm of.
    :param sentinel: Explicit sentinel value. When given, this value
        will replace all ``-inf`` in the result arising from zeros and
        all ``NaN`` in the result arising from negative values in ``x``.
        Optional, defaults to None which means no replacement of ``-inf``.
    :param relative: Whether to use a relative sentinel. When set to True,
        the sentinel value will be determined as the smallest value in
        ``x``, multiplied by ``sentinel``. In this case, ``sentinel``
        should be a number between 0 and 1. Optional, defaults to False.
        Argument is keyword-only.
    :return: The logarithm to base 10 of ``x``, optionally with ``-inf``
        entries replaced by the specified sentinel value.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log10(x)
    # replace NaNs with sentinel value
    if sentinel is not None:
        if relative:
            sentinel = sentinel * np.log10(np.nanmin(x[x > 0]))
        result[x <= 0] = sentinel
    return result
