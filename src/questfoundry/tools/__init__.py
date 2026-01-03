"""Tools for LLM interactions.

This package provides the tool protocol and base types for
LLM tool calling.
"""

from questfoundry.tools.base import Tool, ToolCall, ToolDefinition

__all__ = [
    "Tool",
    "ToolCall",
    "ToolDefinition",
]
