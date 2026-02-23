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
