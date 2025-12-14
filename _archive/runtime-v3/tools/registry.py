"""Tool registry - maps tool definitions to implementations.

This module provides:
- TOOL_IMPLEMENTATIONS: Registry mapping tool IDs to implementation classes
- UnavailableTool: Stub for tools that aren't implemented
- build_agent_tools(): Build tools from agent capabilities
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from pydantic import Field

from questfoundry.runtime.domain.models import Agent, Studio, ToolDefinition
from questfoundry.runtime.knowledge.retrieval import (
    create_consult_knowledge_tool,
    create_query_knowledge_tool,
)
from questfoundry.runtime.state import StudioState
from questfoundry.runtime.stores.cold_store import ColdStore

if TYPE_CHECKING:
    from questfoundry.runtime.playbook_tracker import PlaybookTracker

logger = logging.getLogger(__name__)


class UnavailableTool(BaseTool):
    """Stub tool that informs the LLM a capability is unavailable.

    This is returned when a tool is defined in the domain but has no
    implementation in the runtime. Provides clear feedback to both
    logging (for operators) and LLM (for graceful degradation).
    """

    name: str
    reason: str
    description: str = "This tool is currently unavailable."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return self.reason


# Registry mapping tool IDs (from domain-v4 tools/*.json) to implementation classes.
# New tools should be added here when implemented.
TOOL_IMPLEMENTATIONS: dict[str, type[BaseTool]] = {
    # Import implementations lazily to avoid circular imports
    # These are mapped when build_agent_tools is called
}


def _get_tool_implementations() -> dict[str, type[BaseTool] | None]:
    """Get tool implementations, importing lazily.

    Returns a fresh dict each time to allow for dynamic registration.
    """
    from questfoundry.runtime.tools.consult import (
        ConsultSchema,
    )
    from questfoundry.runtime.tools.corpus import ConsultCorpusTool
    from questfoundry.runtime.tools.create_artifact import CreateArtifactTool
    from questfoundry.runtime.tools.lifecycle import RequestLifecycleTransitionTool
    from questfoundry.runtime.tools.playbook import ConsultPlaybookV4
    from questfoundry.runtime.tools.role import (
        ReadHotSot,
        WriteHotSot,
    )
    from questfoundry.runtime.tools.searxng import WebSearchTool
    from questfoundry.runtime.tools.sr import (
        DelegateTo,
        ReadArtifact,
        WriteArtifact,
    )
    from questfoundry.runtime.tools.validate import ValidateArtifactTool
    from questfoundry.runtime.tools.web_fetch import WebFetchTool

    return {
        # Core delegation tool
        "delegate": DelegateTo,
        # Schema/validation tools
        "consult_schema": ConsultSchema,
        "validate_artifact": ValidateArtifactTool,
        # Search tools
        "search_workspace": ReadHotSot,  # Maps to workspace search
        # Web tools
        "web_search": WebSearchTool,
        "web_fetch": WebFetchTool,
        # Export tools
        "assemble_export": None,  # TODO: Implement (see issue #138)
        # Generation tools (external, may not be available)
        "generate_image": None,
        "generate_audio": None,
        # Additional mapped tools
        "consult_playbook": ConsultPlaybookV4,  # v4 playbook tool with tracker
        "get_playbook": ConsultPlaybookV4,  # Alias - meta/ name for contextual view
        "read_artifact": ReadArtifact,
        "write_artifact": WriteArtifact,
        "read_hot_sot": ReadHotSot,
        "write_hot_sot": WriteHotSot,
        # P1 Section 2 - Additional Tools from meta/
        "consult_corpus": ConsultCorpusTool,  # RAG search over corpus entries
        "create_artifact": CreateArtifactTool,  # Create with validation
        "request_lifecycle_transition": RequestLifecycleTransitionTool,  # State change protocol
    }


def build_agent_tools(
    agent: Agent,
    studio: Studio,
    state: StudioState | None = None,
    cold_store: ColdStore | None = None,
    playbook_tracker: "PlaybookTracker | None" = None,
) -> list[BaseTool]:
    """Build tools from agent capabilities.

    This is the main entry point for capability-driven tool building.
    Tools are built based on the agent's capabilities[], not hardcoded role IDs.

    Args:
        agent: The agent to build tools for
        studio: The loaded studio
        state: Current studio state (for stateful tools)
        cold_store: Cold store instance (for read-only access)
        playbook_tracker: Optional tracker for playbook nudging

    Returns:
        List of BaseTool instances configured for this agent
    """
    tools: list[BaseTool] = []
    seen_tool_names: set[str] = set()
    implementations = _get_tool_implementations()

    def add_tool(tool: BaseTool) -> None:
        """Add tool if not already added (deduplicate by name)."""
        if tool.name not in seen_tool_names:
            tools.append(tool)
            seen_tool_names.add(tool.name)

    for cap in agent.capabilities:
        if cap.category == "tool" and cap.tool_ref:
            # Tool capability - resolve to implementation
            tool = _instantiate_tool(
                cap.tool_ref,
                studio,
                state,
                cold_store,
                implementations,
                playbook_tracker,
            )
            if tool:
                add_tool(tool)

        elif cap.category == "store_access":
            # Store access capability - add read/write tools
            store_tools = _build_store_access_tools(cap, studio, state, cold_store)
            for t in store_tools:
                add_tool(t)

        elif cap.category == "artifact_action":
            # Artifact action capability - add action tools
            # Currently mapped through store_access
            pass

    # Always add knowledge tools for agents with knowledge requirements
    knowledge_tools = _build_knowledge_tools(agent, studio)
    for t in knowledge_tools:
        add_tool(t)

    # Add return/terminate tool based on entry_agent status
    flow_tool = _build_flow_control_tool(agent, studio)
    if flow_tool:
        add_tool(flow_tool)

    logger.info(f"Built {len(tools)} tools for agent {agent.id}")
    return tools


def _instantiate_tool(
    tool_id: str,
    studio: Studio,
    state: StudioState | None,
    cold_store: ColdStore | None,
    implementations: dict[str, type[BaseTool] | None],
    playbook_tracker: "PlaybookTracker | None" = None,
) -> BaseTool | None:
    """Instantiate a tool from its ID.

    Args:
        tool_id: The tool ID from domain-v4 tools/*.json
        studio: The loaded studio
        state: Current studio state
        cold_store: Cold store instance
        implementations: Tool implementations map
        playbook_tracker: Optional tracker for playbook tools

    Returns:
        Instantiated tool, or UnavailableTool if not implemented
    """
    # Get the tool definition from the studio
    tool_def = studio.tools.get(tool_id)
    if not tool_def:
        logger.warning(f"Tool definition not found: {tool_id}")
        return None

    # Get the implementation class
    impl_class = implementations.get(tool_id)

    if impl_class is None:
        # Tool defined but not implemented - return stub
        logger.warning(
            f"Tool '{tool_id}' has no implementation. "
            f"Agents will be informed this tool is unavailable."
        )
        return UnavailableTool(
            name=tool_id,
            reason=f"Tool '{tool_id}' is not available in this runtime. "
            f"Please proceed without this capability or suggest alternatives.",
            description=tool_def.description,
        )

    # Instantiate the tool
    try:
        tool = impl_class()

        # Inject dependencies as needed
        if hasattr(tool, "state") and state is not None:
            tool.state = state  # type: ignore
        if hasattr(tool, "cold_store") and cold_store is not None:
            tool.cold_store = cold_store  # type: ignore
        if hasattr(tool, "studio"):
            tool.studio = studio  # type: ignore
        if hasattr(tool, "tracker") and playbook_tracker is not None:
            tool.tracker = playbook_tracker  # type: ignore

        return tool

    except Exception as e:
        logger.error(f"Failed to instantiate tool {tool_id}: {e}")
        return UnavailableTool(
            name=tool_id,
            reason=f"Tool '{tool_id}' failed to initialize: {e}",
            description=tool_def.description,
        )


def _build_store_access_tools(
    cap: Any,  # Capability
    studio: Studio,
    state: StudioState | None,
    cold_store: ColdStore | None,
) -> list[BaseTool]:
    """Build store access tools from a store_access capability."""
    from questfoundry.runtime.tools.role import ReadHotSot, WriteHotSot

    tools: list[BaseTool] = []

    if not cap.stores:
        return tools

    # For now, we use generic read/write tools
    # Future: could create store-specific tools
    for store_id in cap.stores:
        store = studio.stores.get(store_id)
        if not store:
            logger.warning(f"Store not found: {store_id}")
            continue

        # Add read tool for read access
        if cap.access_level in ("read", "write", "admin"):
            read_tool = ReadHotSot()
            if state is not None and hasattr(read_tool, "state"):
                read_tool.state = state  # type: ignore
            # Only add once (not per-store for now)
            if not any(isinstance(t, ReadHotSot) for t in tools):
                tools.append(read_tool)

        # Add write tool for write/admin access
        if cap.access_level in ("write", "admin"):
            write_tool = WriteHotSot()
            if state is not None and hasattr(write_tool, "state"):
                write_tool.state = state  # type: ignore
            if not any(isinstance(t, WriteHotSot) for t in tools):
                tools.append(write_tool)

    return tools


def _build_knowledge_tools(agent: Agent, studio: Studio) -> list[BaseTool]:
    """Build knowledge consultation tools."""
    tools: list[BaseTool] = []

    kr = agent.knowledge_requirements

    # Add consult tool if agent has role_specific knowledge
    if kr.role_specific or kr.can_lookup:
        consult_tool = create_consult_knowledge_tool(studio, agent)
        tools.append(consult_tool)

    # Add query tool if agent has can_lookup knowledge
    if kr.can_lookup:
        query_tool = create_query_knowledge_tool(studio, agent)
        tools.append(query_tool)

    return tools


def _build_flow_control_tool(agent: Agent, studio: Studio) -> BaseTool | None:
    """Build flow control tool (terminate or return)."""
    from questfoundry.runtime.tools.role import ReturnToSR
    from questfoundry.runtime.tools.sr import Terminate

    if agent.is_entry_agent:
        return Terminate()
    else:
        return ReturnToSR()


def register_tool(tool_id: str, impl_class: type[BaseTool]) -> None:
    """Register a custom tool implementation.

    This allows external code to add tool implementations at runtime.

    Args:
        tool_id: The tool ID (must match domain-v4 tools/*.json)
        impl_class: The implementation class
    """
    # Get the current implementations and add to it
    implementations = _get_tool_implementations()
    implementations[tool_id] = impl_class
    logger.info(f"Registered tool implementation: {tool_id}")
