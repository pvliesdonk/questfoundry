"""Tests for logging configuration."""

import logging

from questfoundry.logging_config import (
    TRACE,
    TraceFormatter,
    get_logger,
    setup_logging,
)


def test_trace_level_defined():
    """Test TRACE level is defined."""
    assert TRACE == 5
    assert logging.getLevelName(TRACE) == "TRACE"


def test_trace_formatter_default_format():
    """Test TraceFormatter with default format."""
    formatter = TraceFormatter()
    assert formatter is not None
    assert isinstance(formatter, logging.Formatter)


def test_trace_formatter_without_module():
    """Test TraceFormatter without module information."""
    formatter = TraceFormatter(include_module=False)
    assert formatter is not None


def test_trace_formatter_custom_format():
    """Test TraceFormatter with custom format."""
    formatter = TraceFormatter(fmt="%(levelname)s - %(message)s")
    assert formatter is not None


def test_trace_formatter_format_record():
    """Test TraceFormatter can format a log record."""
    formatter = TraceFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "Test message" in formatted
    assert "INFO" in formatted


def test_setup_logging_default():
    """Test setup_logging with defaults."""
    setup_logging()
    logger = logging.getLogger("questfoundry")
    assert logger.level == logging.INFO


def test_setup_logging_debug():
    """Test setup_logging with DEBUG level."""
    setup_logging(level="DEBUG")
    logger = logging.getLogger("questfoundry")
    assert logger.level == logging.DEBUG


def test_setup_logging_trace():
    """Test setup_logging with TRACE level."""
    setup_logging(level="TRACE")
    logger = logging.getLogger("questfoundry")
    assert logger.level == TRACE


def test_setup_logging_custom_format(capsys):
    """Test setup_logging with custom format."""
    setup_logging(level="INFO", format_string="%(levelname)s - %(message)s")
    logger = get_logger("questfoundry.test")
    logger.info("test message")
    captured = capsys.readouterr()
    assert "INFO - test message" in captured.err


def test_setup_logging_without_module(capsys):
    """Test setup_logging without module info."""
    setup_logging(level="INFO", include_module=False)
    logger = get_logger("questfoundry.test")
    logger.info("test message")
    captured = capsys.readouterr()
    assert "test_logging_config.py" not in captured.err
    assert "test message" in captured.err


def test_get_logger():
    """Test get_logger returns a logger."""
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_logger_trace_method_exists():
    """Test that logger has trace method."""
    logger = get_logger("test")
    assert hasattr(logger, "trace")
    assert callable(logger.trace)


def test_logger_trace_method_works(capsys):
    """Test that trace method can be called and logs a message."""
    setup_logging(level="TRACE")
    logger = get_logger("questfoundry.test_trace")
    logger.trace("Test trace message")  # type: ignore
    captured = capsys.readouterr()
    assert "TRACE" in captured.err
    assert "Test trace message" in captured.err
