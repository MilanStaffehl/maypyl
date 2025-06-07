# Copyright (c) 2025 Milan Staffehl - subject to the MIT license.
"""Custom types for general use across the library"""

from __future__ import annotations

import numpy as np

type NDArray[S: tuple[int, ...], T: np.generic] = np.ndarray[S, np.dtype[T]]
