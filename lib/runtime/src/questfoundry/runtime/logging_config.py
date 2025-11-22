"""Logging configuration with Rich console formatting.

Provides beautiful, readable logging output using Rich console formatting.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for QuestFoundry
QUESTFOUNDRY_THEME = Theme({
    "logging.level.debug": "dim cyan",
    "logging.level.info": "bold blue",
    "logging.level.warning": "bold yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
    "log.time": "dim",
    "log.message": "white",
    "log.path": "dim cyan",
})


def setup_logging(
    level: str = "INFO",
    show_time: bool = True,
    show_path: bool = False,
    rich_tracebacks: bool = True,
) -> None:
    """
    Configure logging with Rich console formatting.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        show_time: Show timestamp in log messages
        show_path: Show file path and line number
        rich_tracebacks: Use Rich's beautiful traceback formatting

    Examples:
        >>> setup_logging(level="DEBUG", show_path=True)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Starting QuestFoundry runtime")
    """
    # Create Rich console with custom theme (force UTF-8 for Unicode support on Windows)
    if sys.platform == "win32":
        # Reconfigure stderr to use UTF-8 encoding without closing the underlying buffer
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    console = Console(
        theme=QUESTFOUNDRY_THEME,
        stderr=True,  # Log to stderr by default
    )

    # Create Rich handler
    handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=False,  # Don't show local vars (too verbose)
        markup=True,  # Allow Rich markup in log messages
        log_time_format="[%X]",  # Time format
    )

    # Configure root logger
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,  # Override any existing configuration
    )

    # Reduce noise from verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    This is a convenience wrapper around logging.getLogger() that ensures
    consistent logger naming across the application.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing role: [bold]Plotwright[/bold]")
    """
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """
    Change the logging level at runtime.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Examples:
        >>> set_level("DEBUG")
        >>> logger = get_logger(__name__)
        >>> logger.debug("This will now be visible")
    """
    logging.getLogger().setLevel(level.upper())
