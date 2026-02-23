# Copyright (c) 2026 Milan Staffehl - subject to the MIT license.
"""Utilities for working with matplotlib."""

from __future__ import annotations

import functools
import logging
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
