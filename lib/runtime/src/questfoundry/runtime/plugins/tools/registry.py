"""
Tool Registry - provides tool implementations for roles.

Based on spec: interfaces/tool_registry.yaml
Implements plugin provider pattern for tool access.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Tool:
    """Base class for tools."""

    def __init__(self, tool_id: str, name: str, description: str):
        """Initialize tool."""
        self.tool_id = tool_id
        self.name = name
        self.description = description

    def invoke(self, **kwargs: Any) -> Any:
        """Invoke the tool (override in subclasses)."""
        raise NotImplementedError(f"Tool {self.tool_id} not implemented")


class MockTool(Tool):
    """Mock tool for testing - returns dummy data."""

    def __init__(self, tool_id: str, name: str, description: str):
        """Initialize mock tool."""
        super().__init__(tool_id, name, description)

    def invoke(self, **kwargs: Any) -> Any:
        """Return mock response."""
        return {
            "status": "success",
            "tool": self.tool_id,
            "message": f"Mock execution of {self.name}",
            "input": kwargs,
        }


class ToolRegistry:
    """
    Registry for tool implementations.

    Provides tool lookup and registration.
    """

    def __init__(self):
        """Initialize registry with stub tools."""
        self._tools: dict[str, Tool] = {}
        self._register_stub_tools()

    def _register_stub_tools(self) -> None:
        """Register stub implementations for common tools."""
        # Image generation
        self._tools["stable_diffusion"] = MockTool(
            "stable_diffusion", "Stable Diffusion", "Generate images using Stable Diffusion"
        )

        # Audio synthesis
        self._tools["audio_synthesis"] = MockTool(
            "audio_synthesis", "Audio Synthesis", "Generate audio/music using synthesis"
        )

        # Document conversion
        self._tools["pandoc"] = MockTool("pandoc", "Pandoc", "Convert documents between formats")

        # Web search
        self._tools["web_search"] = MockTool(
            "web_search", "Web Search", "Search the web for information"
        )

        # Lore index lookup
        self._tools["lore_index"] = MockTool(
            "lore_index", "Lore Index", "Look up entries in the lore/codex index"
        )

        logger.info(f"Registered {len(self._tools)} stub tools")

    def get_tool(self, tool_id: str, config: dict[str, Any | None] = None) -> Tool | None:
        """
        Get tool by ID.

        Args:
            tool_id: Tool identifier
            config: Optional tool configuration

        Returns:
            Tool instance or None if not found
        """
        if tool_id not in self._tools:
            logger.warning(f"Tool not found: {tool_id}")
            return None

        tool = self._tools[tool_id]

        # Store config if provided (can be used by tool implementations)
        if config:
            tool.config = config

        return tool

    def register_tool(self, tool_id: str, tool: Tool) -> None:
        """
        Register a tool implementation.

        Args:
            tool_id: Tool identifier
            tool: Tool instance
        """
        self._tools[tool_id] = tool
        logger.info(f"Registered tool: {tool_id}")

    def list_available_tools(self) -> list[str]:
        """
        List all available tools.

        Returns:
            List of tool IDs
        """
        return list(self._tools.keys())

    def has_tool(self, tool_id: str) -> bool:
        """
        Check if tool is available.

        Args:
            tool_id: Tool identifier

        Returns:
            True if tool exists, False otherwise
        """
        return tool_id in self._tools

    def get_all_tools(self) -> dict[str, Tool]:
        """
        Get all tools.

        Returns:
            Dict of all tools
        """
        return self._tools.copy()


# Global registry instance
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """
    Get or create tool registry singleton.

    Returns:
        ToolRegistry instance
    """
    global _registry

    if _registry is None:
        _registry = ToolRegistry()

    return _registry


def get_tool(tool_id: str, config: dict[str, Any | None] = None) -> Tool | None:
    """
    Convenience function to get tool from registry.

    Args:
        tool_id: Tool identifier
        config: Optional tool configuration

    Returns:
        Tool instance or None if not found
    """
    registry = get_tool_registry()
    return registry.get_tool(tool_id, config)
