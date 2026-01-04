"""Tools for LLM interactions.

This package provides the tool protocol and implementations for
LLM tool calling, including finalization tools for structured output
and research tools for context retrieval.
"""

from questfoundry.tools.base import Tool, ToolCall, ToolDefinition
from questfoundry.tools.finalization import (
    SubmitBrainstormTool,
    SubmitDreamTool,
    get_finalization_tool,
)

__all__ = [
    "SubmitBrainstormTool",
    "SubmitDreamTool",
    "Tool",
    "ToolCall",
    "ToolDefinition",
    "get_finalization_tool",
]
