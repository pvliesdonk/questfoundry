"""
Tool execution infrastructure for the QuestFoundry runtime.

This module provides:
- BaseTool: Abstract base class for tool implementations
- ToolRegistry: Registry for loading and filtering tools
- Tool implementations: consult_schema, validate_artifact, etc.

Usage:
    from questfoundry.runtime.tools import ToolRegistry, build_agent_tools

    # Get tools for an agent
    registry = ToolRegistry(studio, project)
    tools = registry.get_agent_tools(agent)

    # Execute a tool
    result = await tools[0].run({"artifact_type_id": "section"})
"""

# Import tool implementations to trigger registration
from questfoundry.runtime.tools import (
    consult_corpus,  # noqa: F401
    consult_schema,  # noqa: F401
    delegate,  # noqa: F401
    request_clarification,  # noqa: F401
    search_workspace,  # noqa: F401
    stubs,  # noqa: F401
    validate_artifact,  # noqa: F401
    web_fetch,  # noqa: F401
    web_search,  # noqa: F401
)
from questfoundry.runtime.tools.base import (
    BaseTool,
    CapabilityViolationError,
    ToolContext,
    ToolError,
    ToolExecutionError,
    ToolResult,
    ToolUnavailableError,
    ToolValidationError,
    UnavailableTool,
)
from questfoundry.runtime.tools.registry import (
    TOOL_IMPLEMENTATIONS,
    ToolRegistry,
    build_agent_tools,
    register_tool,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "UnavailableTool",
    # Exceptions
    "ToolError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolUnavailableError",
    "CapabilityViolationError",
    # Registry
    "ToolRegistry",
    "TOOL_IMPLEMENTATIONS",
    "register_tool",
    "build_agent_tools",
]
