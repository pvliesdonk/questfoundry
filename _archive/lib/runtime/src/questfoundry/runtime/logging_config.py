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
QUESTFOUNDRY_THEME = Theme(
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
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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


def setup_file_logging(file_path: str, level: str = "DEBUG") -> logging.FileHandler:
    """
    Add file logging handler that captures detailed debug logs.

    This is used for option C: trace + debug logs to file while -v controls screen.
    The file handler captures all log messages at the specified level (default DEBUG),
    independent of the console handler's level.

    Args:
        file_path: Path to the log file
        level: Logging level for file (default DEBUG to capture everything)

    Returns:
        The FileHandler instance (for later removal if needed)

    Examples:
        >>> handler = setup_file_logging("debug.log")
        >>> # Later: logging.getLogger().removeHandler(handler)
    """
    from pathlib import Path

    # Ensure parent directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Create file handler with detailed formatting
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(level.upper())

    # Detailed format for file logs (includes timestamp, logger name, level, path)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Ensure root logger level allows DEBUG through (handler filters its own level)
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)

    return file_handler
