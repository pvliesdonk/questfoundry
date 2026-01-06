"""Structured logging configuration for QuestFoundry.

Provides two logging modes:
- Console logging: Controlled by -v flag (DEBUG to stderr)
- File logging: Controlled by --log flag (all events to {projectdir}/logs/)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - Used at runtime for path operations
from typing import TYPE_CHECKING

import structlog
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from structlog.typing import Processor

# Module-level state
_configured = False
_file_handler: logging.FileHandler | None = None
_logs_dir: Path | None = None


class JSONLFileHandler(logging.FileHandler):
    """File handler that writes JSONL format."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write log record as JSON line."""
        try:
            # Build log entry
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
            }

            # Add structlog context if available, extracting event as message
            # structlog passes event dict via record.msg when using wrap_for_formatter
            if isinstance(record.msg, dict):
                event_dict = record.msg.copy()
                # Remove structlog internal fields we already capture
                event_dict.pop("level", None)
                event_dict.pop("timestamp", None)
                entry["message"] = event_dict.pop("event", str(record.msg))
                entry.update(event_dict)
            else:
                entry["message"] = record.getMessage()

            # Write as JSON line
            line = json.dumps(entry) + "\n"
            if self.stream:
                self.stream.write(line)
                self.stream.flush()
        except Exception:
            self.handleError(record)


def configure_logging(
    verbosity: int = 0,
    log_to_file: bool = False,
    project_path: Path | None = None,
) -> None:
    """Configure logging for QuestFoundry.

    Args:
        verbosity: 0=WARNING (default), 1=INFO, 2+=DEBUG
        log_to_file: If True, enable file logging to {project_path}/logs/.
        project_path: Project directory for file logging. Required if log_to_file=True.

    Raises:
        ValueError: If log_to_file=True but project_path is not provided.
    """
    global _configured, _file_handler, _logs_dir

    # Validate arguments
    if log_to_file and project_path is None:
        raise ValueError("project_path is required when log_to_file=True")

    # Close existing file handler if reconfiguring
    if _file_handler is not None:
        _file_handler.close()
        _file_handler = None

    # Map verbosity to log level
    levels = {0: logging.WARNING, 1: logging.INFO}
    console_level = levels.get(verbosity, logging.DEBUG)

    # Configure Rich handler for console output
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=verbosity >= 2,
        show_time=verbosity >= 1,
        show_path=verbosity >= 2,
        markup=True,
        level=console_level,
    )

    handlers: list[logging.Handler] = [console_handler]

    # Configure file handler if requested
    if log_to_file and project_path:
        _logs_dir = project_path / "logs"
        _logs_dir.mkdir(parents=True, exist_ok=True)

        debug_log_path = _logs_dir / "debug.jsonl"
        _file_handler = JSONLFileHandler(str(debug_log_path), mode="a")
        _file_handler.setLevel(logging.DEBUG)  # Capture everything
        handlers.append(_file_handler)

    # Set up root logger
    root_level = logging.DEBUG if (verbosity > 0 or log_to_file) else logging.WARNING
    logging.basicConfig(
        level=root_level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )

    # Suppress noisy loggers from dependencies
    # These libraries produce excessive DEBUG output that drowns out useful logs
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "langchain",
        "langchain_core",
        "langsmith",
        "asyncio",  # EpollSelector messages
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Configure structlog processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(root_level),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.typing.FilteringBoundLogger:
    """Get a structured logger instance.

    Automatically configures logging if not already done.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Bound logger instance.
    """
    if not _configured:
        configure_logging()

    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger


def get_logs_dir() -> Path | None:
    """Get the configured logs directory.

    Returns:
        Path to logs directory if file logging is enabled, None otherwise.
    """
    return _logs_dir


def close_file_logging() -> None:
    """Close file logging handler."""
    global _file_handler
    if _file_handler:
        _file_handler.close()
        _file_handler = None
