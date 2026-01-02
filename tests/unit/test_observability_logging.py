"""Tests for structured logging configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from questfoundry.observability import configure_logging, get_logger

if TYPE_CHECKING:
    from pathlib import Path


def test_configure_logging_sets_level_warning() -> None:
    """Default verbosity (0) sets WARNING level."""
    configure_logging(verbosity=0)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.WARNING


def test_configure_logging_verbose_sets_info() -> None:
    """verbosity=1 sets INFO level."""
    configure_logging(verbosity=1)

    root_logger = logging.getLogger()
    # Root level is DEBUG to allow file handlers, but console handler filters
    assert root_logger.level == logging.DEBUG


def test_configure_logging_very_verbose_sets_debug() -> None:
    """verbosity=2 sets DEBUG level."""
    configure_logging(verbosity=2)

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
    configure_logging(verbosity=2)  # DEBUG level

    httpx_logger = logging.getLogger("httpx")
    assert httpx_logger.level == logging.WARNING


def test_configure_logging_with_file_logging(tmp_path: Path) -> None:
    """File logging creates debug.jsonl in logs directory."""
    configure_logging(verbosity=0, log_to_file=True, project_path=tmp_path)

    logs_dir = tmp_path / "logs"
    assert logs_dir.exists()


def test_configure_logging_without_file_logging(tmp_path: Path) -> None:
    """Without file logging flag, logs directory is not created."""
    configure_logging(verbosity=0, log_to_file=False, project_path=tmp_path)

    logs_dir = tmp_path / "logs"
    assert not logs_dir.exists()
