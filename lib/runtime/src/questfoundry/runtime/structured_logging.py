"""Structured logging for QuestFoundry runtime.

Provides domain-specific JSONL loggers for debugging:
- qf.tools: Tool invocations and results
- qf.sot: Hot/cold SOT mutations
- qf.bus: Protocol message routing
- qf.prompts: Prompt engineering traces

Usage:
    from questfoundry.runtime.structured_logging import (
        configure_structured_logging,
        get_tool_logger,
        get_sot_logger,
        get_bus_logger,
        get_prompt_logger,
    )

    # Configure at startup (e.g., in CLI)
    configure_structured_logging(Path("./logs"))

    # Get loggers where needed
    tool_log = get_tool_logger()
    tool_log.info("tool.invoke", tool="read_hot_sot", args={"key": "canon"})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import structlog

# Module-level state
_log_dir: Path | None = None
_configured: bool = False

# Cached logger instances
_tool_logger: structlog.stdlib.BoundLogger | None = None
_sot_logger: structlog.stdlib.BoundLogger | None = None
_bus_logger: structlog.stdlib.BoundLogger | None = None
_prompt_logger: structlog.stdlib.BoundLogger | None = None

# Domain configuration
DOMAINS = {
    "qf.tools": "tool-calls.jsonl",
    "qf.sot": "state-sot.jsonl",
    "qf.bus": "message-bus.jsonl",
    "qf.prompts": "prompts.jsonl",
}


def configure_structured_logging(log_dir: Path) -> None:
    """Configure domain-specific structured loggers.

    Creates JSONL log files in the specified directory:
    - tool-calls.jsonl: Tool invocations and results
    - state-sot.jsonl: Hot/cold SOT mutations
    - message-bus.jsonl: Protocol message routing
    - prompts.jsonl: Prompt engineering traces

    Args:
        log_dir: Directory for log files (created if needed)
    """
    global _log_dir, _configured
    import structlog

    # Ensure directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_dir = log_dir

    # Configure structlog with JSON rendering
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file handlers for each domain
    for name, filename in DOMAINS.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Remove existing handlers
        logger.handlers.clear()

        # Add file handler with JSON output
        handler = logging.FileHandler(
            log_dir / filename,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        # Prevent propagation to root logger
        logger.propagate = False

    _configured = True


def get_tool_logger() -> structlog.stdlib.BoundLogger:
    """Get the tool invocation logger.

    Logs tool calls, arguments, results, and execution times.

    Returns:
        Structured logger for tool domain

    Raises:
        RuntimeError: If configure_structured_logging() not called
    """
    global _tool_logger
    import structlog

    if not _configured:
        raise RuntimeError(
            "Structured logging not configured. Call configure_structured_logging() first."
        )
    if _tool_logger is None:
        _tool_logger = structlog.get_logger("qf.tools")
    return _tool_logger


def get_sot_logger() -> structlog.stdlib.BoundLogger:
    """Get the state (SOT) mutation logger.

    Logs hot/cold source-of-truth reads and writes.

    Returns:
        Structured logger for SOT domain

    Raises:
        RuntimeError: If configure_structured_logging() not called
    """
    global _sot_logger
    import structlog

    if not _configured:
        raise RuntimeError(
            "Structured logging not configured. Call configure_structured_logging() first."
        )
    if _sot_logger is None:
        _sot_logger = structlog.get_logger("qf.sot")
    return _sot_logger


def get_bus_logger() -> structlog.stdlib.BoundLogger:
    """Get the message bus logger.

    Logs protocol message routing, consumption tracking, and pending counts.

    Returns:
        Structured logger for message bus domain

    Raises:
        RuntimeError: If configure_structured_logging() not called
    """
    global _bus_logger
    import structlog

    if not _configured:
        raise RuntimeError(
            "Structured logging not configured. Call configure_structured_logging() first."
        )
    if _bus_logger is None:
        _bus_logger = structlog.get_logger("qf.bus")
    return _bus_logger


def get_prompt_logger() -> structlog.stdlib.BoundLogger:
    """Get the prompt engineering logger.

    Logs LLM prompts, responses, and token usage.

    Returns:
        Structured logger for prompts domain

    Raises:
        RuntimeError: If configure_structured_logging() not called
    """
    global _prompt_logger
    import structlog

    if not _configured:
        raise RuntimeError(
            "Structured logging not configured. Call configure_structured_logging() first."
        )
    if _prompt_logger is None:
        _prompt_logger = structlog.get_logger("qf.prompts")
    return _prompt_logger


def reset_loggers() -> None:
    """Reset cached loggers (for testing)."""
    global _tool_logger, _sot_logger, _bus_logger, _prompt_logger, _configured
    _tool_logger = None
    _sot_logger = None
    _bus_logger = None
    _prompt_logger = None
    _configured = False


def is_configured() -> bool:
    """Check if structured logging is configured."""
    return _configured
