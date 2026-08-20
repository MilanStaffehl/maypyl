# Copyright (c) 2026 Milan Staffehl - subject to the MIT license.
"""Functions for working with observations."""

from typing import Any, Final, cast

import numpy as np

from ._types import NDArray

_SPEED_OF_LIGHT: Final[float] = 299792.458  # in km/s


def wavelength_to_velocity[
    T: float | NDArray[tuple[int, ...], np.floating[Any]]
](lambda_observed: T, lambda_0: float) -> T:
    """
    Transform a wavelength difference to a LOS velocity.

    This function takes an emission line rest-frame wavelength ``lambda_0``
    and an observed emission line wavelength ``lambda_observed`` which is
    Doppler-shifted with respect to the original wavelength, and computes
    the relative velocity along the line of sight required to account for
    the shift in wavelength. Note that this function only uses the non-
    relativistic Doppler effect for its calculations.

    Note that the inverse function also exists: :func:`velocity_to_wavelength`.

    :param lambda_observed: The wavelength which to turn into a velocity.
    :param lambda_0: The rest frame wavelength of ``lambda_observed``. Must
        be in the same unit as ``lambda_observed``.
    :return: The line-of-sight velocity required to shift ``lambda_0``
        to ``lambda_observed``. In units of km/s. Positive velocity means
        receding velocity, i.e. redshift, while negative velocity means
        approaching velocity, i.e. blueshift.
    """
    return cast(T, _SPEED_OF_LIGHT * (lambda_observed / lambda_0 - 1))


def velocity_to_wavelength[
    T: float | NDArray[tuple[int, ...], np.floating[Any]]
](los_velocity: T, lambda_0: float) -> T:
    """
    Transform a LOS velocity to a shifted wavelength.

    This function takes a wavelength ``lambda_0`` and a line-of-sight
    velocity ``los_velocity`` and Doppler-shifts the rest-frame wavelength
    ``lambda_0`` of an emission line according to that velocity. Positive
    velocities are interpreted as receding, causing redshift, while
    negative velocities are interpreted as approaching, causing blueshift.
    Note that this function assumes non-relativistic Doppler effects.

    Note that the inverse function also exists: :func:`wavelength_to_velocity`.

    :param los_velocity: The velocity along the line of sight which to
        turn into a wavelength, in km/s. Negative velocities are
        interpreted as approaching, causing a blueshift, while positive
        velocities are interpreted as receding, causing a redshift.
    :param lambda_0: The rest frame wavelength of ``lambda_observed``.
    :return: The wavelength to which the ``velocity`` shifts the rest
        frame wavelength ``lambda_0``. In the same units as ``lambda_0``.
    """
    return cast(T, (1 + los_velocity / _SPEED_OF_LIGHT) * lambda_0)
