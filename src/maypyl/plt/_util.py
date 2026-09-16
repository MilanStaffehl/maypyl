# Copyright (c) 2026 Milan Staffehl - subject to the MIT license.
"""Utilities for working with matplotlib."""

from __future__ import annotations

import functools
import logging
import math
from typing import Callable


def silence_matplotlib[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to mute matplotlib logging during function execution.

    Function acts as a decorator which turns logging level during the
    execution of the function to level WARNING, avoiding clutter due to
    the many DEBUG level logs that matplotlib tends to emit. The log
    level is set globally within the matplotlib root logger before the
    body of the function is entered. It is then reset to its original
    value afterward, effectively silencing matplotlib only during the
    execution of the function. The matplotlib root logger thus returns
    to the previous state after execution ends.

    :param func: Any callable that invokes matplotlib and which should
        execute all matplotlib code silently, except warnings and errors.
    :return: The decorated function, with matplotlib silenced.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        mpl_logger = logging.getLogger("matplotlib")
        mpl_log_level = mpl_logger.level
        mpl_logger.setLevel(logging.WARNING)
        result = func(*args, **kwargs)
        mpl_logger.setLevel(mpl_log_level)
        return result

    return wrapper


def frexp10(number: float) -> tuple[float, int]:
    """
    Decompose a number into mantissa and exponent in base 10.

    Given any floating-point number, this function decomposes the number
    into mantissa and exponent in base 10. This is useful for labels in
    plots that require pretty formatting of number values::

        number = mantissa * 10^exponent

    .. warning::

        The conversion is not exact! It will introduce rounding and
        floating-point errors. For example, passing ``9.87e-6`` will
        result in a mantissa of ``9.870000000000001``.

        This function is located in the *plotting* module for a reason:
        It is meant primarily as a way to conveniently format labels of
        large numbers. Do not use this function where numerical precision
        is required!

    :param number: Any floating-point number that cannot be easily
        displayed as plain float.
    :return: A tuple of a float and an integer: the mantissa (first, type
        float) and exponent (second, type int) of the given number in
        base 10.
    """
    if number == 0:
        exponent = 0
    else:
        exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / (10**exponent)
    return mantissa, exponent
