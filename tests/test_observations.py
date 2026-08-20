# Copyright (c) 2026 Milan Staffehl - subject to the MIT license.
"""Tests for the :py:mod:`~maypyl._observations` module."""

import pytest

from maypyl import velocity_to_wavelength, wavelength_to_velocity


def test_wavelength_to_velocity() -> None:
    """Test the function returns proper LOS velocity."""
    lambda_0 = 656.46  # H-alpha, in nm

    # blueshift
    lambda_obs = 654.32
    expected = -977.296
    output = wavelength_to_velocity(lambda_obs, lambda_0)
    assert output == pytest.approx(expected)

    # redshift
    lambda_obs = 657.89
    expected = 653.053
    output = wavelength_to_velocity(lambda_obs, lambda_0)
    assert output == pytest.approx(expected)


def test_velocity_to_wavelength() -> None:
    """Test the function returns proper shifted wavelengths."""
    lambda_0 = 1215.67  # Ly-alpha, in Å

    # blueshift
    los_velocity = -220.0  # in km/s
    expected = 1214.777892
    output = velocity_to_wavelength(los_velocity, lambda_0)
    assert output == pytest.approx(expected)

    # redshift
    los_velocity = +354.2  # in km/s
    expected = 1217.106295
    output = velocity_to_wavelength(los_velocity, lambda_0)
    assert output == pytest.approx(expected)


def test_velocity_wavelength_roundtrip() -> None:
    """Test that the two functions are exactly inverse."""
    lambda_0 = 2799.117  # Mg II, in Å

    # one direction
    lambda_obs = 2800.51
    output = velocity_to_wavelength(
        wavelength_to_velocity(lambda_obs, lambda_0), lambda_0
    )
    assert output == pytest.approx(lambda_obs)

    # and the other
    los_velocity = +135.67
    output = wavelength_to_velocity(
        velocity_to_wavelength(los_velocity, lambda_0), lambda_0
    )
    assert output == pytest.approx(los_velocity)
