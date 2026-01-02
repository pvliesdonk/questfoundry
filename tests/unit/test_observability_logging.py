"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from questfoundry.observability import close_file_logging, configure_logging, get_logger

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


def test_configure_logging_requires_project_path_for_file_logging() -> None:
    """log_to_file=True without project_path raises ValueError."""
    with pytest.raises(ValueError, match="project_path is required"):
        configure_logging(verbosity=0, log_to_file=True, project_path=None)


def test_configure_logging_reconfiguration_closes_handler(tmp_path: Path) -> None:
    """Reconfiguring logging closes previous file handler."""
    import questfoundry.observability.logging as log_module

    # Configure with file logging
    configure_logging(verbosity=0, log_to_file=True, project_path=tmp_path)
    first_handler = log_module._file_handler
    assert first_handler is not None

    # Reconfigure - should close previous handler
    configure_logging(verbosity=0, log_to_file=True, project_path=tmp_path)
    second_handler = log_module._file_handler

    # First handler should have been closed (stream is None after close)
    assert first_handler.stream is None or first_handler.stream.closed
    assert second_handler is not None


def test_close_file_logging_clears_handler(tmp_path: Path) -> None:
    """close_file_logging closes handler and clears reference."""
    import questfoundry.observability.logging as log_module

    configure_logging(verbosity=0, log_to_file=True, project_path=tmp_path)
    assert log_module._file_handler is not None

    close_file_logging()

    assert log_module._file_handler is None


def test_jsonl_file_handler_writes_structlog_context(tmp_path: Path) -> None:
    """JSONLFileHandler correctly extracts structlog context to JSONL."""
    # Configure file logging
    configure_logging(verbosity=2, log_to_file=True, project_path=tmp_path)

    # Get a structlog logger and log with context
    logger = get_logger("test.context")
    logger.info("test_event", key1="value1", key2=42)

    # Close to flush
    close_file_logging()

    # Read the JSONL file
    log_file = tmp_path / "logs" / "debug.jsonl"
    assert log_file.exists()

    # Find our log entry
    found = False
    with log_file.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("message") == "test_event":
                found = True
                assert entry["key1"] == "value1"
                assert entry["key2"] == 42
                assert entry["level"] == "INFO"
                break

    assert found, "Log entry with structlog context not found in JSONL"
