"""Structured logging for QuestFoundry runtime.

Provides domain-specific JSONL loggers for debugging:
- qf.tools: Tool invocations and results
- qf.sot: Hot/cold SOT mutations
- qf.bus: Protocol message routing
- qf.prompts: Prompt engineering traces
- qf.reasoning: Agent reasoning extraction

Usage:
    from questfoundry.runtime.structured_logging import (
        configure_structured_logging,
        get_tool_logger,
        get_sot_logger,
        get_bus_logger,
        get_prompt_logger,
        get_reasoning_logger,
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
_health_logger: structlog.stdlib.BoundLogger | None = None
_reasoning_logger: structlog.stdlib.BoundLogger | None = None
_feedback_logger: structlog.stdlib.BoundLogger | None = None
_debug_file_handler: logging.FileHandler | None = None

# Domain configuration
DOMAINS = {
    "qf.tools": "tool-calls.jsonl",
    "qf.sot": "state-sot.jsonl",
    "qf.bus": "message-bus.jsonl",
    "qf.prompts": "prompts.jsonl",
    "qf.health": "loop-health.jsonl",
    "qf.reasoning": "reasoning.jsonl",
    "qf.feedback": "agent-feedback.jsonl",
}


class _JsonDebugFormatter(logging.Formatter):
    """Format log records as JSON for debug.jsonl."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_structured_logging(log_dir: Path) -> None:
    """Configure domain-specific structured loggers.

    Creates JSONL log files in the specified directory:
    - tool-calls.jsonl: Tool invocations and results
    - state-sot.jsonl: Hot/cold SOT mutations
    - message-bus.jsonl: Protocol message routing
    - prompts.jsonl: Prompt engineering traces
    - reasoning.jsonl: Agent reasoning extraction
    - debug.jsonl: All DEBUG+ messages from questfoundry loggers

    Args:
        log_dir: Directory for log files (created if needed)
    """
    global _log_dir, _configured, _debug_file_handler
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

    # Add debug.jsonl handler for all questfoundry DEBUG messages
    debug_handler = logging.FileHandler(
        log_dir / "debug.jsonl",
        encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(_JsonDebugFormatter())
    _debug_file_handler = debug_handler

    # Attach to questfoundry root logger to capture all DEBUG+ messages
    qf_logger = logging.getLogger("questfoundry")
    qf_logger.addHandler(debug_handler)
    # Note: Do NOT set logger level here - respect the console verbosity setting
    # The debug_handler has its own setLevel(DEBUG) to capture all messages to file

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


def get_health_logger() -> "structlog.stdlib.BoundLogger":
    """Get the loop health logger.

    Logs loop health metrics for detecting stuck loops.

    Returns:
        Structured logger for health domain

    Raises:
        RuntimeError: If configure_structured_logging() not called
    """
    global _health_logger
    import structlog

    if not _configured:
        raise RuntimeError(
            "Structured logging not configured. Call configure_structured_logging() first."
        )
    if _health_logger is None:
        _health_logger = structlog.get_logger("qf.health")
    return _health_logger


def get_reasoning_logger() -> "structlog.stdlib.BoundLogger":
    """Get the agent reasoning logger.

    Logs extracted agent reasoning, decision points, and thought processes.

    Returns:
        Structured logger for reasoning domain

    Raises:
        RuntimeError: If configure_structured_logging() not called
    """
    global _reasoning_logger
    import structlog

    if not _configured:
        raise RuntimeError(
            "Structured logging not configured. Call configure_structured_logging() first."
        )
    if _reasoning_logger is None:
        _reasoning_logger = structlog.get_logger("qf.reasoning")
    return _reasoning_logger


def get_feedback_logger() -> "structlog.stdlib.BoundLogger":
    """Get the agent feedback logger.

    Logs agent feedback collected at end of session. When structured logging
    is enabled, writes to agent-feedback.jsonl. When disabled, falls back
    to standard INFO-level logging.

    Returns:
        Structured logger for feedback domain

    Note:
        Unlike other domain loggers, this does NOT require structured logging
        to be configured. It gracefully falls back to standard logging when
        --structured-logs is not enabled.
    """
    global _feedback_logger
    import structlog

    # If structured logging is configured, use domain-specific logger
    if _configured:
        if _feedback_logger is None:
            _feedback_logger = structlog.get_logger("qf.feedback")
        return _feedback_logger

    # Otherwise, return a basic logger that writes to console at INFO level
    # This matches the user's requirement for fallback behavior
    if _feedback_logger is None:
        _feedback_logger = structlog.get_logger("questfoundry.feedback")
    return _feedback_logger


def reset_loggers() -> None:
    """Reset cached loggers (for testing)."""
    global _tool_logger, _sot_logger, _bus_logger, _prompt_logger, _health_logger, _reasoning_logger, _debug_file_handler, _configured

    # Clean up debug handler
    if _debug_file_handler is not None:
        qf_logger = logging.getLogger("questfoundry")
        qf_logger.removeHandler(_debug_file_handler)
        _debug_file_handler.close()
        _debug_file_handler = None

    _tool_logger = None
    _sot_logger = None
    _bus_logger = None
    _prompt_logger = None
    _health_logger = None
    _reasoning_logger = None
    _configured = False


def is_configured() -> bool:
    """Check if structured logging is configured."""
    return _configured
