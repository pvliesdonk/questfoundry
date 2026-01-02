"""Tests for structured logging configuration."""

from __future__ import annotations

import logging

from questfoundry.observability import configure_logging, get_logger


def test_configure_logging_sets_level_warning() -> None:
    """Default verbosity sets WARNING level."""
    configure_logging(0)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.WARNING


def test_configure_logging_sets_level_info() -> None:
    """Verbosity 1 sets INFO level."""
    configure_logging(1)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO


def test_configure_logging_sets_level_debug() -> None:
    """Verbosity 2+ sets DEBUG level."""
    configure_logging(2)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_get_logger_returns_bound_logger() -> None:
    """get_logger returns a structlog logger with expected methods."""
    logger = get_logger(__name__)

    # Structlog uses a lazy proxy, verify it has expected logging methods
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "error")
    assert hasattr(logger, "warning")


def test_get_logger_auto_configures() -> None:
    """get_logger configures logging if not already done."""
    # Reset configuration state
    import questfoundry.observability.logging as log_module

    log_module._configured = False

    logger = get_logger("test")

    assert log_module._configured is True
    assert logger is not None


def test_get_logger_with_name() -> None:
    """get_logger accepts optional name."""
    logger = get_logger("my.module.name")

    # Should not raise
    assert logger is not None


def test_configure_logging_suppresses_httpx() -> None:
    """Logging configuration suppresses noisy httpx logger."""
    configure_logging(2)  # DEBUG level

    httpx_logger = logging.getLogger("httpx")
    assert httpx_logger.level == logging.WARNING
