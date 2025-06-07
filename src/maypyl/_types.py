# Copyright (c) 2025 Milan Staffehl - subject to the MIT license.
"""Custom types for general use across the library"""

from __future__ import annotations

from typing import Any

import numpy as np

type NDArray[S: tuple[int, ...], T: np.generic] = np.ndarray[S, np.dtype[T]]
type ArrayOrScalar = NDArray[tuple[int, ...], Any] | np.generic | float | int
type ArrayOrPureScalar = NDArray[tuple[int, ...], Any] | np.generic
