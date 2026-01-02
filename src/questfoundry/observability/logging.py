"""Structured logging configuration for QuestFoundry.

Uses structlog with Rich integration for beautiful console output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import structlog
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from structlog.typing import Processor

# Module-level logger cache
_configured = False


def configure_logging(verbosity: int = 0) -> None:
    """Configure logging with Rich handler.

    Args:
        verbosity: 0=WARNING (default), 1=INFO, 2+=DEBUG
    """
    global _configured

    # Map verbosity to log level
    levels = {
        0: logging.WARNING,
        1: logging.INFO,
    }
    level = levels.get(verbosity, logging.DEBUG)

    # Configure Rich handler for beautiful output
    console = Console(stderr=True)
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=verbosity >= 2,
        show_time=verbosity >= 1,
        show_path=verbosity >= 2,
        markup=True,
    )

    # Set up root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
        force=True,  # Override any existing config
    )

    # Suppress noisy loggers from dependencies
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langchain_core").setLevel(logging.WARNING)

    # Configure structlog processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Configure structlog to integrate with standard logging
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
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
