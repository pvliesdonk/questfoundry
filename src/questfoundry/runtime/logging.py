"""Logging configuration for QuestFoundry CLI.

Provides:
- Countable -v flags for console verbosity (WARNING → INFO → DEBUG)
- Structured JSONL logging to file when --log-dir specified
- Rich console formatting for readable output

Verbosity Levels:
- No -v: WARNING only (quiet)
- -v: INFO (normal)
- -vv: DEBUG (verbose)
- -vvv: DEBUG + show_path (trace)

Usage:
    from questfoundry.runtime.logging import setup_logging, VerbosityLevel

    # In CLI callback
    setup_logging(verbosity=2, log_dir=Path("./logs"))
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

if TYPE_CHECKING:
    pass

__all__ = [
    "VerbosityLevel",
    "setup_logging",
    "get_logger",
    "configure_structured_logging",
]


class VerbosityLevel(IntEnum):
    """Verbosity levels for CLI."""

    QUIET = 0  # WARNING only
    NORMAL = 1  # INFO
    VERBOSE = 2  # DEBUG
    TRACE = 3  # DEBUG + paths


# QuestFoundry console theme
QF_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold blue",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "log.time": "dim",
        "log.message": "white",
        "log.path": "dim cyan",
    }
)

# Module state
_log_dir: Path | None = None
_structured_configured: bool = False
_file_handlers: list[logging.FileHandler] = []


class _JsonlFormatter(logging.Formatter):
    """Format log records as JSONL for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        # Include extra fields if present
        for key in ["tool", "args", "result", "role", "artifact", "duration_ms"]:
            if hasattr(record, key):
                data[key] = getattr(record, key)
        return json.dumps(data, default=str)


def setup_logging(
    verbosity: int = 1,
    log_dir: Path | None = None,
    force: bool = True,
) -> None:
    """Configure logging with verbosity levels and optional file output.

    Parameters
    ----------
    verbosity : int
        Verbosity level:
        - 0: WARNING only (quiet)
        - 1: INFO (normal, default)
        - 2: DEBUG (verbose)
        - 3+: DEBUG with file paths (trace)
    log_dir : Path | None
        Directory for structured JSONL logs. If provided, writes:
        - debug.jsonl: All DEBUG+ messages
        - tools.jsonl: Tool invocations
        - state.jsonl: Hot/cold store operations
    force : bool
        Override existing logging configuration.
    """
    # Map verbosity to log level
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    show_path = verbosity >= 3

    # Handle Windows UTF-8
    if sys.platform == "win32":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Create Rich console
    console = Console(theme=QF_THEME, stderr=True)

    # Create Rich handler
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
        log_time_format="[%X]",
    )

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=force,
    )

    # Reduce noise from verbose libraries
    for noisy in ["httpx", "httpcore", "openai", "anthropic", "urllib3", "langchain"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Set up structured file logging if requested
    if log_dir:
        configure_structured_logging(log_dir)


def configure_structured_logging(log_dir: Path) -> None:
    """Configure structured JSONL logging to files.

    Creates JSONL log files in the specified directory:
    - debug.jsonl: All DEBUG+ messages from questfoundry loggers
    - tools.jsonl: Tool invocations (qf.tools logger)
    - state.jsonl: Hot/cold state operations (qf.state logger)
    - delegations.jsonl: Role delegations (qf.delegations logger)

    Parameters
    ----------
    log_dir : Path
        Directory for log files (created if needed).
    """
    global _log_dir, _structured_configured, _file_handlers

    log_dir.mkdir(parents=True, exist_ok=True)
    _log_dir = log_dir

    # Domain-specific log files
    domains = {
        "qf.tools": "tools.jsonl",
        "qf.state": "state.jsonl",
        "qf.delegations": "delegations.jsonl",
    }

    for name, filename in domains.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        handler.setFormatter(_JsonlFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        _file_handlers.append(handler)

    # Catch-all debug.jsonl for all questfoundry messages
    debug_handler = logging.FileHandler(log_dir / "debug.jsonl", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(_JsonlFormatter())

    qf_logger = logging.getLogger("questfoundry")
    qf_logger.addHandler(debug_handler)
    _file_handlers.append(debug_handler)

    _structured_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)


def is_structured_logging_configured() -> bool:
    """Check if structured logging is configured."""
    return _structured_configured


def reset_logging() -> None:
    """Reset logging configuration (for testing)."""
    global _log_dir, _structured_configured, _file_handlers

    for handler in _file_handlers:
        handler.close()
    _file_handlers.clear()

    _log_dir = None
    _structured_configured = False
