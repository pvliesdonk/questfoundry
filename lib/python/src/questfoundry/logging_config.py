"""
QuestFoundry Logging Configuration

Provides centralized logging setup for the QuestFoundry library with support for
multiple log levels including custom TRACE level for detailed diagnostic output.

Downstream applications can use this module to configure logging for their needs:

    from questfoundry.logging_config import setup_logging

    # Setup with default configuration
    setup_logging()

    # Or customize log level
    setup_logging(level="DEBUG")
"""

import logging
import sys
from typing import Any, Literal

# Define custom TRACE level (lower than DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(
    logger: logging.Logger, message: str, *args: object, **kwargs: object
) -> None:
    """Log a TRACE level message."""
    if logger.isEnabledFor(TRACE):
        logger._log(TRACE, message, args, **kwargs)  # type: ignore


# Attach trace method to Logger class
logging.Logger.trace = _trace  # type: ignore


class TraceFormatter(logging.Formatter):
    """Custom formatter for QuestFoundry logs with support for TRACE level."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        include_module: bool = True,
    ):
        if fmt is None:
            if include_module:
                fmt = (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "[%(filename)s:%(lineno)d] - %(message)s"
                )
            else:
                fmt = "%(asctime)s - %(levelname)s - %(message)s"

        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with custom TRACE level handling."""
        if record.levelno == TRACE:
            record.levelname = "TRACE"
        return super().format(record)


def setup_logging(
    level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    format_string: str | None = None,
    include_module: bool = True,
    handlers: list[logging.Handler] | None = None,
) -> None:
    """
    Configure logging for the QuestFoundry library.

    This function sets up the root logger for the questfoundry package with
    sensible defaults. It can be called by downstream applications to configure
    logging output.

    Args:
        level: Logging level as a string. Options: TRACE, DEBUG, INFO, WARNING,
               ERROR, CRITICAL. Default: INFO
        format_string: Custom format string for log messages. If None, uses a
                      default format including timestamp, logger name, level,
                      filename, and line number.
        include_module: If True (default), include module filename and line number
                       in log output.
        handlers: List of custom handlers. If None, defaults to a StreamHandler
                 writing to stderr.

    Example:
        >>> setup_logging(level="DEBUG")
        >>> logger = logging.getLogger("questfoundry.orchestrator")
        >>> logger.debug("Orchestrator initialized")

        >>> # Or use custom handler
        >>> import logging
        >>> file_handler = logging.FileHandler("app.log")
        >>> setup_logging(level="INFO", handlers=[file_handler])
    """
    # Get the root questfoundry logger
    questfoundry_logger = logging.getLogger("questfoundry")

    # Set level (convert string to logging level constant)
    if level.upper() == "TRACE":
        log_level = TRACE
    else:
        log_level = getattr(logging, level.upper(), logging.INFO)

    questfoundry_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    questfoundry_logger.handlers.clear()

    # Create formatter
    formatter = TraceFormatter(format_string, include_module=include_module)

    # Add handlers
    if handlers is None:
        # Default to stderr handler
        handler: Any = logging.StreamHandler(sys.stderr)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        questfoundry_logger.addHandler(handler)
    else:
        for handler in handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
            questfoundry_logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    questfoundry_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name within the questfoundry package.

    This is a convenience function that ensures loggers are part of the
    questfoundry logging hierarchy and will use the configured handlers.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)
