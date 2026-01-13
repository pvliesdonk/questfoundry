"""Observability module for QuestFoundry.

Provides structured logging, LLM call tracking, and LangSmith tracing.
"""

from questfoundry.observability.langchain_callbacks import (
    LLMLoggingCallback,
    create_logging_callbacks,
)
from questfoundry.observability.llm_logger import LLMLogEntry, LLMLogger
from questfoundry.observability.logging import (
    close_file_logging,
    configure_logging,
    get_logger,
    get_logs_dir,
)
from questfoundry.observability.tracing import (
    LANGSMITH_AVAILABLE,
    build_runnable_config,
    generate_run_id,
    get_current_run_tree,
    get_pipeline_run_id,
    is_tracing_enabled,
    set_pipeline_run_id,
    trace_context,
    traceable,
)

__all__ = [
    "LANGSMITH_AVAILABLE",
    "LLMLogEntry",
    "LLMLogger",
    "LLMLoggingCallback",
    "build_runnable_config",
    "close_file_logging",
    "configure_logging",
    "create_logging_callbacks",
    "generate_run_id",
    "get_current_run_tree",
    "get_logger",
    "get_logs_dir",
    "get_pipeline_run_id",
    "is_tracing_enabled",
    "set_pipeline_run_id",
    "trace_context",
    "traceable",
]
