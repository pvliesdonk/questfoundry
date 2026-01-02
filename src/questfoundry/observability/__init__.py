"""Observability module for QuestFoundry.

Provides structured logging and LLM call tracking.
"""

from questfoundry.observability.llm_logger import LLMLogEntry, LLMLogger
from questfoundry.observability.logging import (
    close_file_logging,
    configure_logging,
    get_logger,
    get_logs_dir,
)

__all__ = [
    "LLMLogEntry",
    "LLMLogger",
    "close_file_logging",
    "configure_logging",
    "get_logger",
    "get_logs_dir",
]
