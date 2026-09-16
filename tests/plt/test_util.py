# Copyright (c) 2026 Milan Staffehl - subject to the MIT license.
"""Tests for the :py:mod:`~maypyl.plt._util` module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import maypyl.plt

if TYPE_CHECKING:
    from pytest import LogCaptureFixture


def working_function(log_msg: str, log_level: int) -> int:
    """
    Test function, logs a message to root logger.

    The function additionally invokes several matplotlib functions which
    tend to log many DEBUG messages. These should not appear if the
    function is run through the ``silence_matplotlib`` decorator.

    :param log_msg: An arbitrary test message.
    :param log_level: The log level for the test message.
    :return: The log level as given to the function.
    """
    fig, axes = plt.subplots()
    axes.text(0.5, 1.0, r"Test string with LaTeX: $\log_{10} \text{T} = 4.5$")

    # for good measure: force matplotlib logger to emit message manually
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.debug("Manual DEBUG message for testing.")

    root_logger = logging.getLogger()
    root_logger.log(log_level, log_msg)
    return log_level


def test_silence_matplotlib_decorator(caplog: LogCaptureFixture) -> None:
    """Test that the decorator silences ONLY matplotlib messages."""
    caplog.set_level(logging.DEBUG)

    # Part One: without decorator
    out = working_function("Test message", logging.DEBUG)
    assert out == logging.DEBUG
    assert len(caplog.records) > 2
    assert "Manual DEBUG message for testing." in caplog.text
    assert "Test message" in caplog.text

    # reset caplog fixture
    caplog.clear()

    # Part Two: with decorator
    out = maypyl.plt.silence_matplotlib(working_function)(
        "Test message 2", logging.DEBUG
    )
    assert out == logging.DEBUG
    assert len(caplog.records) == 1
    log_record = caplog.records[0]
    assert log_record.levelno == logging.DEBUG
    assert log_record.message == "Test message 2"
    assert "Manual DEBUG message for testing." not in caplog.text

    # Part Three: test that the logger is reset again
    caplog.clear()
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.debug("I should show up.")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert caplog.records[0].message == "I should show up."


def test_frexp10_base() -> None:
    """Test the function to decompose in base 10."""
    number = 1.23e4
    output = maypyl.plt.frexp10(number)
    assert output == (1.23, 4)

    number = 9.87e6
    output = maypyl.plt.frexp10(number)
    assert output == (9.87, 6)


def test_frexp10_negative_number() -> None:
    """Test decomposing a negative number into mantissa and exponent."""
    number = -1.23e4
    output = maypyl.plt.frexp10(number)
    assert output == (-1.23, 4)

    number = -9.87e6
    output = maypyl.plt.frexp10(number)
    assert output == (-9.87, 6)


def test_frexp10_negative_exponent() -> None:
    """Test the function for a negative exponent."""
    number = 1.23e-4
    output = maypyl.plt.frexp10(number)
    assert output == (1.23, -4)

    # here is an example for an inexact conversion:
    number = 9.87e-6
    output = maypyl.plt.frexp10(number)
    assert output == (9.870000000000001, -6)


def test_frexp10_zero_exponent() -> None:
    """Test the function with a zero exponent."""
    number = 1.23e00
    output = maypyl.plt.frexp10(number)
    assert output == (1.23, 0)


def test_frexp10_zero() -> None:
    """Test the function when the number is zero."""
    number = 0.0
    output = maypyl.plt.frexp10(number)
    assert output == (0.0, 0)

    number = 0e2
    output = maypyl.plt.frexp10(number)
    assert output == (0.0, 0)

    number = 0e-3
    output = maypyl.plt.frexp10(number)
    assert output == (0.0, 0)

    number = -0.0
    output = maypyl.plt.frexp10(number)
    assert output == (0.0, 0)
