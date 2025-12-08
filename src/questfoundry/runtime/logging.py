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
from typing import TYPE_CHECKING, Any

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
    "get_llm_logger",
    "get_tools_logger",
    "log_llm_request",
    "log_llm_response",
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

    # Always suppress httpx/httpcore trace noise - not useful for debugging QF
    for always_quiet in ["httpx", "httpcore"]:
        logging.getLogger(always_quiet).setLevel(logging.WARNING)

    # Reduce noise from other verbose libraries unless at trace level (-vvv)
    if verbosity < 3:
        for noisy in [
            "openai",
            "anthropic",
            "urllib3",
            "langchain",
            "langchain_core",
            "langchain_google_genai",
            "langchain_openai",
            "langchain_ollama",
        ]:
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
        "qf.llm": "llm.jsonl",  # LLM prompts and responses
    }

    for name, filename in domains.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # mode='a' ensures logs are appended, not overwritten
        handler = logging.FileHandler(log_dir / filename, mode="a", encoding="utf-8")
        handler.setFormatter(_JsonlFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        _file_handlers.append(handler)

    # Catch-all debug.jsonl for all questfoundry messages (append mode)
    debug_handler = logging.FileHandler(log_dir / "debug.jsonl", mode="a", encoding="utf-8")
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


# Domain-specific logger accessors
_llm_logger: logging.Logger | None = None
_tools_logger: logging.Logger | None = None


def get_llm_logger() -> logging.Logger:
    """Get the LLM communication logger for prompts/responses."""
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = logging.getLogger("qf.llm")
    return _llm_logger


def get_tools_logger() -> logging.Logger:
    """Get the tools logger for tool invocations."""
    global _tools_logger
    if _tools_logger is None:
        _tools_logger = logging.getLogger("qf.tools")
    return _tools_logger


def log_llm_request(
    messages: list[dict[str, Any]],
    model: str | None = None,
    role: str | None = None,
    iteration: int | None = None,
) -> None:
    """Log an LLM request (prompt) to structured logs.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        The messages being sent to the LLM.
    model : str | None
        The model being used.
    role : str | None
        The role making the request.
    iteration : int | None
        The iteration number in the execution loop.
    """
    logger = get_llm_logger()
    # Format messages for logging (truncate long content)
    formatted_messages = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        formatted_messages.append({
            "role": msg.get("role", msg.get("type", "unknown")),
            "content": content,
        })

    logger.info(
        "llm_request",
        extra={
            "event": "llm_request",
            "model": model,
            "role": role,
            "iteration": iteration,
            "message_count": len(messages),
            "messages": formatted_messages,
        },
    )


def log_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str | None = None,
    role: str | None = None,
    iteration: int | None = None,
    duration_ms: float | None = None,
) -> None:
    """Log an LLM response to structured logs.

    Parameters
    ----------
    content : str | None
        The text content of the response.
    tool_calls : list[dict[str, Any]] | None
        Tool calls made by the LLM.
    model : str | None
        The model that responded.
    role : str | None
        The role that received the response.
    iteration : int | None
        The iteration number in the execution loop.
    duration_ms : float | None
        Response time in milliseconds.
    """
    logger = get_llm_logger()
    # Truncate long content
    truncated_content = content
    if content and len(content) > 2000:
        truncated_content = content[:2000] + "... [truncated]"

    logger.info(
        "llm_response",
        extra={
            "event": "llm_response",
            "model": model,
            "role": role,
            "iteration": iteration,
            "duration_ms": duration_ms,
            "content": truncated_content,
            "tool_calls": tool_calls,
            "has_tool_calls": bool(tool_calls),
        },
    )
