# Copyright (c) 2025 Milan Staffehl - subject to the MIT license.
"""Tests for the :py:mod:`~maypyl._statistics` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import maypyl

if TYPE_CHECKING:
    import numpy.typing as npt


type CellSetupType = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


@pytest.fixture
def simple_cell_setup() -> CellSetupType:
    """Returns a single 1x2 cell setup."""
    cell_edges = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    values = np.array([0.1, 0.2])
    return cell_edges, values


@pytest.fixture
def base_cell_setup() -> CellSetupType:
    """Returns cell edges for a simple 4x4 cube setup."""
    cell_edges = np.array(
        [
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],  # xmin
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],  # ymin
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],  # zmin
            [0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],  # xmax
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5],  # ymax
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],  # zmax
        ]
    )
    values = np.array([1, 2, 3, 1, 2, 1, 1, 3], dtype=np.float64)
    return cell_edges, values


def assert_cell_setup(
    image: npt.NDArray[np.float64],
    im_edges: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    expected_image: npt.NDArray[np.float64],
    expected_edges: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    """Helper function to assert the outcome of a simple test scenario."""
    assert isinstance(image, np.ndarray)
    assert isinstance(im_edges, tuple)
    assert len(im_edges) == 2
    assert isinstance(im_edges[0], np.ndarray)
    assert isinstance(im_edges[1], np.ndarray)
    assert image.shape == expected_image.shape
    assert im_edges[0].shape == expected_edges[0].shape
    assert im_edges[1].shape == expected_edges[1].shape
    np.testing.assert_almost_equal(image, expected_image)
    np.testing.assert_almost_equal(im_edges[0], expected_edges[0])
    np.testing.assert_almost_equal(im_edges[1], expected_edges[1])


def test_axis_aligned_projection_single_sum_no_weight(
    simple_cell_setup: CellSetupType,
) -> None:
    """Test function for no weights, mode sum, simplest set-up."""
    cell_edges, values = simple_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.0, 1.0, 0.0, 1.0),
        nx_bins=1,
        ny_bins=1,
        mode="sum",
    )
    expected_img = np.array([[0.3]])
    expected_edges = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_single_sum_with_weight(
    simple_cell_setup: CellSetupType,
) -> None:
    """Test function with weights, mode sum, simplest set-up."""
    cell_edges, values = simple_cell_setup
    weights = np.array([5.0, 1.0], dtype=np.float64)
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.0, 1.0, 0.0, 1.0),
        nx_bins=1,
        ny_bins=1,
        mode="sum",
    )
    expected_img = np.array([[0.7]])
    expected_edges = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_single_mean_no_weight(
    simple_cell_setup: CellSetupType,
) -> None:
    """Test function for no weights, mode mean, simplest set-up."""
    cell_edges, values = simple_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.0, 1.0, 0.0, 1.0),
        nx_bins=1,
        ny_bins=1,
        mode="mean",
    )
    expected_img = np.array([[0.15]])
    expected_edges = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_single_mean_with_weight(
    simple_cell_setup: CellSetupType,
) -> None:
    """Test function with weights, mode mean, simplest set-up."""
    cell_edges, values = simple_cell_setup
    weights = np.array([5.0, 1.0], dtype=np.float64)
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.0, 1.0, 0.0, 1.0),
        nx_bins=1,
        ny_bins=1,
        mode="mean",
    )
    expected_img = np.array([[0.7 / 6.0]])
    expected_edges = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_base_sum_no_weight(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function for no weights, mode sum, basic set-up."""
    cell_edges, values = base_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        mode="sum",
    )
    expected_img = np.array([[1.75]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_base_sum_with_weight(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function with weights, mode sum, basic set-up."""
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        mode="sum",
    )
    expected_img = np.array([[1.9]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_base_mean_no_weight(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function for no weights, mode mean, basic set-up."""
    cell_edges, values = base_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        mode="mean",
    )
    # Unfortunately, in this set-up, the result turns out to be identical
    # to the unweighted sum scenario as the sum of the weights turns out
    # to be unity - simply a byproduct of the geometry.
    expected_img = np.array([[1.75]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_base_mean_with_weight(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function with weights, mode mean, basic set-up."""
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        mode="mean",
    )
    expected_img = np.array([[15.2 / 8.2]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_offset_sum_no_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function with the pixel being offset, mode sum, no weights."""
    # off-center the pixel so that it mostly covers the lower left cell:
    cell_edges, values = base_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.1, 0.6, 0.1, 0.6),
        nx_bins=1,
        ny_bins=1,
        mode="sum",
    )
    expected_img = np.array([[1.9]])
    expected_edges = np.array([0.1, 0.6]), np.array([0.1, 0.6])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_offset_sum_with_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function with the pixel being offset, mode sum, with weights."""
    # off-center the pixel so that it mostly covers the lower left cell:
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.1, 0.6, 0.1, 0.6),
        nx_bins=1,
        ny_bins=1,
        mode="sum",
    )
    expected_img = np.array([[1.936]])
    expected_edges = np.array([0.1, 0.6]), np.array([0.1, 0.6])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_offset_mean_no_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function with the pixel being offset, mode mean, no weights."""
    # off-center the pixel so that it mostly covers the lower left cell:
    cell_edges, values = base_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.1, 0.6, 0.1, 0.6),
        nx_bins=1,
        ny_bins=1,
        mode="mean",
    )
    # again, the geometry causes the sum of the weights to evaluate to 1.
    expected_img = np.array([[1.9]])
    expected_edges = np.array([0.1, 0.6]), np.array([0.1, 0.6])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_offset_mean_with_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test function with the pixel being offset, mode mean, with weights."""
    # off-center the pixel so that it mostly covers the lower left cell:
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.1, 0.6, 0.1, 0.6),
        nx_bins=1,
        ny_bins=1,
        mode="mean",
    )
    expected_img = np.array([[1.936 / 0.956]])
    expected_edges = np.array([0.1, 0.6]), np.array([0.1, 0.6])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_multipixel_sum_no_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test the function for multiple pixels, mode sum, no weights."""
    cell_edges, values = base_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.1, 0.9, 0.1, 0.9),
        nx_bins=3,
        ny_bins=3,
        mode="sum",
    )
    expected_img = np.array(
        [[2.0, 2.0, 2.0], [1.75, 1.75, 1.75], [1.5, 1.5, 1.5]]
    )
    expected_edges = (
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
    )
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_multipixel_sum_with_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test the function for multiple pixels, mode sum, with weights."""
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.1, 0.9, 0.1, 0.9),
        nx_bins=3,
        ny_bins=3,
        mode="sum",
    )
    expected_img = np.array(
        [[2.0, 1.775, 1.55], [2.025, 1.9, 1.775], [2.05, 2.025, 2.0]]
    )
    expected_edges = (
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
    )
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_multipixel_mean_no_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test the function for multiple pixels, mode mean, no weights."""
    cell_edges, values = base_cell_setup
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.1, 0.9, 0.1, 0.9),
        nx_bins=3,
        ny_bins=3,
        mode="mean",
    )
    # Again: tough luck, evaluates to just the sum
    expected_img = np.array(
        [[2.0, 2.0, 2.0], [1.75, 1.75, 1.75], [1.5, 1.5, 1.5]]
    )
    expected_edges = (
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
    )
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_multipixel_mean_with_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test the function for multiple pixels, mode mean, with weights."""
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.1, 0.9, 0.1, 0.9),
        nx_bins=3,
        ny_bins=3,
        mode="mean",
    )
    expected_img = np.array(
        [
            [2.0, 1.775 / 0.775, 1.55 / 0.55],
            [2.025 / 1.025, 1.9 / 1.025, 1.775 / 1.025],
            [2.05 / 1.05, 2.025 / 1.275, 2.0 / 1.5],
        ]
    )
    expected_edges = (
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
        np.array([0.1, 0.36666667, 0.63333333, 0.9]),
    )
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_limited_depth_sum_no_weights(
    base_cell_setup: CellSetupType,
) -> None:
    """Test limiting the depth of the projection, mode sum, no weights."""
    cell_edges, values = base_cell_setup
    projection_depth = (0.01, 0.49)
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        projection_range=projection_depth,
        mode="sum",
    )
    expected_img = np.array([[0.875]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_limited_depth_sum_with_weight(
    base_cell_setup: CellSetupType,
) -> None:
    """Test limiting the depth of the projection, mode sum, basic set-up."""
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    projection_depth = (0.01, 0.49)
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        projection_range=projection_depth,
        mode="sum",
    )
    expected_img = np.array([[0.65]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_limited_depth_mean_no_weight(
    base_cell_setup: CellSetupType,
) -> None:
    """Test limiting the depth of the projection, mode mean, basic set-up."""
    cell_edges, values = base_cell_setup
    projection_depth = (0.01, 0.49)
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        projection_range=projection_depth,
        mode="mean",
    )
    expected_img = np.array([[1.75]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)


def test_axis_aligned_projection_limited_depth_mean_with_weight(
    base_cell_setup: CellSetupType,
) -> None:
    """Test limiting the depth of the projection, mode mean, basic set-up."""
    cell_edges, values = base_cell_setup
    weights = np.array([0.1, 1.0, 1.0, 0.1, 2.0, 2.0, 1.0, 1.0])
    projection_depth = (0.01, 0.49)
    image, im_edges = maypyl.axis_aligned_cell_projection(
        cell_edges,
        values,
        weights=weights,
        extent=(0.25, 0.75, 0.25, 0.75),
        nx_bins=1,
        ny_bins=1,
        projection_range=projection_depth,
        mode="mean",
    )
    expected_img = np.array([[0.65 / 0.275]])
    expected_edges = np.array([0.25, 0.75]), np.array([0.25, 0.75])
    assert_cell_setup(image, im_edges, expected_img, expected_edges)
