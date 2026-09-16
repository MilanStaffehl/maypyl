"""Plotting utilities for astrophysics."""

from ._images import histogram2d
from ._util import frexp10, silence_matplotlib

__all__ = ["histogram2d", "silence_matplotlib", "frexp10"]
