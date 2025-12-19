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
    communicate,  # noqa: F401  # Phase 7: unified human communication
    consult_corpus,  # noqa: F401
    consult_knowledge,  # noqa: F401  # Phase 7: knowledge menu+consult pattern
    consult_playbook,  # noqa: F401
    consult_schema,  # noqa: F401
    delegate,  # noqa: F401
    lifecycle_transition,  # noqa: F401  # Phase 4: lifecycle transitions
    list_agents,  # noqa: F401
    list_artifact_types,  # noqa: F401
    list_stores,  # noqa: F401
    request_clarification,  # noqa: F401  # Legacy: to be removed after migration
    return_to_orchestrator,  # noqa: F401  # Hub-and-spoke: specialists return to orchestrator
    save_artifact,  # noqa: F401  # Phase 4: artifact persistence
    search_workspace,  # noqa: F401
    stubs,  # noqa: F401
    terminate_session,  # noqa: F401  # Explicit session termination for orchestrators
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
