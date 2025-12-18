"""
Tool registry for managing tool implementations.

The registry:
1. Maps tool IDs to implementation classes
2. Loads and instantiates tools with context
3. Filters tools by agent capabilities
4. Logs capability violations
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.models import Tool
from questfoundry.runtime.tools.base import (
    BaseTool,
    CapabilityViolationError,
    ToolContext,
    UnavailableTool,
)

if TYPE_CHECKING:
    from questfoundry.runtime.messaging.broker import AsyncMessageBroker
    from questfoundry.runtime.models import Agent, Studio, Tool
    from questfoundry.runtime.storage import LifecycleManager, Project, StoreManager

logger = logging.getLogger(__name__)


# Implicit tool definitions injected when the domain does not enumerate
# persistence helpers. These fulfill the meta-model contract that store
# permissions imply basic CRUD access.
IMPLICIT_TOOL_DEFINITIONS: dict[str, Tool] = {
    "save_artifact": Tool(
        id="save_artifact",
        name="Save Artifact",
        description="Persist an artifact to a store (validates schema and permissions).",
    ),
    "update_artifact": Tool(
        id="update_artifact",
        name="Update Artifact",
        description="Update an existing artifact in-place when allowed by the store.",
    ),
    "delete_artifact": Tool(
        id="delete_artifact",
        name="Delete Artifact",
        description="Delete an artifact from a store when permitted.",
    ),
}

PERSISTENCE_TOOL_IDS = tuple(IMPLICIT_TOOL_DEFINITIONS.keys())

# Registry mapping tool IDs to implementation classes
# Populated by register_tool decorator or direct assignment
TOOL_IMPLEMENTATIONS: dict[str, type[BaseTool]] = {}


def register_tool(tool_id: str) -> Callable[[type[BaseTool]], type[BaseTool]]:
    """
    Decorator to register a tool implementation.

    Usage:
        @register_tool("consult_schema")
        class ConsultSchemaTool(BaseTool):
            ...
    """

    def decorator(cls: type[BaseTool]) -> type[BaseTool]:
        TOOL_IMPLEMENTATIONS[tool_id] = cls
        return cls

    return decorator


# Tools that are completely disabled in non-interactive mode.
# Note: The 'communicate' tool handles non-interactive mode internally:
# - status, notification, error types still work
# - question type auto-selects default_option or returns error
INTERACTIVE_ONLY_TOOLS = frozenset({"request_clarification"})  # Deprecated, replaced by communicate


class ToolRegistry:
    """
    Registry for tool management.

    Handles:
    - Loading tool implementations
    - Filtering by agent capabilities
    - Creating tool instances with context
    """

    def __init__(
        self,
        studio: Studio,
        project: Project | None = None,
        domain_path: Any = None,
        broker: AsyncMessageBroker | None = None,
        interactive: bool = True,
        store_manager: StoreManager | None = None,
        lifecycle_manager: LifecycleManager | None = None,
    ):
        """
        Initialize tool registry.

        Args:
            studio: Loaded studio with tool definitions
            project: Optional project for tools that need storage
            domain_path: Path to domain directory
            broker: Message broker for delegation routing
            interactive: Whether running in interactive mode (disables human-input tools if False)
            store_manager: Store manager for artifact storage
            lifecycle_manager: Lifecycle manager for artifact state transitions
        """
        self._studio = studio
        self._project = project
        self._domain_path = domain_path
        self._broker = broker
        self._interactive = interactive
        self._store_manager = store_manager
        self._lifecycle_manager = lifecycle_manager
        self._tool_cache: dict[str, BaseTool] = {}

    def get_tool_definition(self, tool_id: str) -> Tool | None:
        """Get a tool definition by ID.

        TODO: Consider building a dict[str, Tool] index on init for O(1) lookups.
        Current linear search is acceptable for typical studio sizes (~10-20 tools).
        """
        studio_tools = self._studio.tools or []
        for tool in studio_tools:
            if tool.id == tool_id:
                return tool
        implicit = IMPLICIT_TOOL_DEFINITIONS.get(tool_id)
        return implicit

    def get_tool(
        self,
        tool_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> BaseTool:
        """
        Get an instantiated tool.

        Args:
            tool_id: Tool ID to get
            agent_id: ID of agent using the tool
            session_id: Current session ID

        Returns:
            Instantiated tool

        Raises:
            KeyError: If tool not found
        """
        # Check cache
        cache_key = f"{tool_id}:{agent_id or 'anon'}:{session_id or 'none'}"
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        # Get definition
        definition = self.get_tool_definition(tool_id)
        if not definition:
            raise KeyError(f"Tool not found: {tool_id}")

        # Create context
        context = ToolContext(
            studio=self._studio,
            project=self._project,
            agent_id=agent_id,
            session_id=session_id,
            domain_path=self._domain_path,
            broker=self._broker,
            interactive=self._interactive,
            store_manager=self._store_manager,
            lifecycle_manager=self._lifecycle_manager,
        )

        # Get implementation class
        impl_class = TOOL_IMPLEMENTATIONS.get(tool_id)

        if impl_class:
            tool = impl_class(definition, context)
        else:
            # No implementation - return unavailable stub
            logger.warning(f"No implementation for tool '{tool_id}', using stub")
            tool = UnavailableTool(definition, context)

        self._tool_cache[cache_key] = tool
        return tool

    def get_agent_tools(
        self,
        agent: Agent,
        session_id: str | None = None,
    ) -> list[BaseTool]:
        """
        Get all tools an agent has capability to use.

        Args:
            agent: Agent to get tools for
            session_id: Current session ID

        Returns:
            List of instantiated tools the agent can use
        """
        tools = []
        allowed_tool_ids = self._get_agent_tool_refs(agent)
        allowed_tool_ids.update(self._get_implicit_tool_refs(agent))

        for tool_id in allowed_tool_ids:
            # Skip interactive-only tools when not in interactive mode
            if not self._interactive and tool_id in INTERACTIVE_ONLY_TOOLS:
                logger.debug(f"Skipping interactive-only tool '{tool_id}' in non-interactive mode")
                continue

            try:
                tool = self.get_tool(tool_id, agent.id, session_id)
                tools.append(tool)
            except KeyError:
                logger.warning(f"Agent '{agent.id}' has capability for unknown tool '{tool_id}'")

        return tools

    def _get_agent_tool_refs(self, agent: Agent) -> set[str]:
        """Extract tool_ref values from agent capabilities."""
        tool_refs = set()
        for cap in agent.capabilities:
            if cap.tool_ref:
                tool_refs.add(cap.tool_ref)
        return tool_refs

    def _get_implicit_tool_refs(self, agent: Agent) -> set[str]:
        """Determine implicit tool refs derived from store permissions."""

        implicit: set[str] = set()

        if self._agent_has_store_write_access(agent):
            implicit.update(PERSISTENCE_TOOL_IDS)

        return implicit

    def _agent_has_store_write_access(self, agent: Agent) -> bool:
        """Check if the agent has write access to any store."""

        for cap in agent.capabilities or []:
            if cap.category != "store_access":
                continue

            access_level = (cap.access_level or "read").lower()
            if "write" in access_level:
                return True

        return False

    def check_capability(self, agent: Agent, tool_id: str) -> bool:
        """
        Check if an agent has capability to use a tool.

        Args:
            agent: Agent to check
            tool_id: Tool ID to check

        Returns:
            True if agent can use the tool
        """
        explicit_tools = self._get_agent_tool_refs(agent)
        implicit_tools = self._get_implicit_tool_refs(agent)

        return tool_id in explicit_tools or tool_id in implicit_tools

    def enforce_capability(self, agent: Agent, tool_id: str) -> None:
        """
        Enforce capability check, raising if violation.

        Args:
            agent: Agent attempting to use tool
            tool_id: Tool being used

        Raises:
            CapabilityViolationError: If agent lacks capability
        """
        if not self.check_capability(agent, tool_id):
            logger.warning(
                f"Capability violation: agent '{agent.id}' attempted to use tool '{tool_id}'"
            )
            raise CapabilityViolationError(agent.id, tool_id)

    def get_langchain_tools(
        self,
        agent: Agent,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get LangChain-compatible tool schemas for an agent.

        Used with llm.bind_tools() for tool calling.

        Args:
            agent: Agent to get tools for
            session_id: Current session ID

        Returns:
            List of tool schemas for LangChain
        """
        tools = self.get_agent_tools(agent, session_id)
        return [tool.to_langchain_schema() for tool in tools if tool.check_availability()]

    def list_all_tools(self) -> list[str]:
        """List all tool IDs defined in the studio."""
        return [tool.id for tool in self._studio.tools]

    def list_implemented_tools(self) -> list[str]:
        """List tool IDs that have implementations."""
        return list(TOOL_IMPLEMENTATIONS.keys())

    def list_unimplemented_tools(self) -> list[str]:
        """List tool IDs without implementations."""
        all_tools = set(self.list_all_tools())
        implemented = set(self.list_implemented_tools())
        return list(all_tools - implemented)


def build_agent_tools(
    agent: Agent,
    studio: Studio,
    project: Project | None = None,
    domain_path: Any = None,
    session_id: str | None = None,
    broker: AsyncMessageBroker | None = None,
    store_manager: StoreManager | None = None,
) -> list[BaseTool]:
    """
    Convenience function to build tools for an agent.

    Args:
        agent: Agent to build tools for
        studio: Loaded studio
        project: Optional project for storage
        domain_path: Path to domain directory
        session_id: Current session ID
        broker: Message broker for delegation routing

    Returns:
        List of tools the agent can use
    """
    registry = ToolRegistry(
        studio,
        project,
        domain_path,
        broker,
        store_manager=store_manager,
    )
    return registry.get_agent_tools(agent, session_id)
