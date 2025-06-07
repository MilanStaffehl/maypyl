# Copyright (c) 2025 Milan Staffehl - subject to the MIT license.
"""Tests for the :py:mod:`~maypyl._math` module."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

import maypyl

if TYPE_CHECKING:
    from pytest_subtests import SubTests


def test_nanlog10_basic() -> None:
    """Test that the function can perform normal log operations."""
    x = np.array([10, 100, 1000])
    result = maypyl.nanlog10(x)
    np.testing.assert_array_equal(result, np.log10(x))


def test_nanlog10_zeros() -> None:
    """Test that function can handle zeros and negatives without warning."""
    x = np.array([0, 10, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = maypyl.nanlog10(x)
    np.testing.assert_array_equal(result, np.log10(x))


def test_nanlog10_explicit_sentinel() -> None:
    """Test that function can handle explicit sentinels."""
    x = np.array([0, 10, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = maypyl.nanlog10(x, sentinel=2)
    np.testing.assert_array_equal(result, np.array([2, 1, 2]))


def test_nanlog10_nan_sentinel() -> None:
    """Test that function allows NaN as sentinel value."""
    x = np.array([0, 10, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = maypyl.nanlog10(x, sentinel=np.nan)
    np.testing.assert_array_equal(result, np.array([np.nan, 1, np.nan]))


def test_nanlog10_relative_sentinel() -> None:
    """Test that function can handle relative sentinels."""
    x = np.array([0, 10, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = maypyl.nanlog10(x, 0.1, relative=True)
    np.testing.assert_almost_equal(result, np.array([0.1, 1, 0.1]))


def test_nanlog10_scalars(subtests: SubTests) -> None:
    """Test that scalars can be passed to the function."""
    mapping = {
        10: 1,
        0: -np.inf,
        -1: np.nan,
    }
    for scalar, expected in mapping.items():
        with subtests.test(scalar=scalar):
            result = maypyl.nanlog10(scalar)
            assert isinstance(result, np.floating)
            np.testing.assert_array_equal(result, expected)
