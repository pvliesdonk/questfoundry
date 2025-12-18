"""Tests for tool registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.tools.base import BaseTool, CapabilityViolationError, ToolResult
from questfoundry.runtime.tools.registry import (
    TOOL_IMPLEMENTATIONS,
    ToolRegistry,
    build_agent_tools,
    register_tool,
)


def make_mock_studio_with_tools():
    """Create a mock studio with tool definitions."""
    studio = MagicMock()

    # Tool definitions
    tool1 = MagicMock()
    tool1.id = "consult_schema"
    tool1.name = "Consult Schema"
    tool1.description = "Get schema info"
    tool1.timeout_ms = 30000
    tool1.input_schema = None

    tool2 = MagicMock()
    tool2.id = "delegate"
    tool2.name = "Delegate Work"
    tool2.description = "Delegate to agent"
    tool2.timeout_ms = 30000
    tool2.input_schema = None

    tool3 = MagicMock()
    tool3.id = "unknown_tool"
    tool3.name = "Unknown"
    tool3.description = "No implementation"
    tool3.timeout_ms = 30000
    tool3.input_schema = None

    studio.tools = [tool1, tool2, tool3]

    # Agent with capabilities
    agent = MagicMock()
    agent.id = "test_agent"
    agent.capabilities = []

    cap1 = MagicMock()
    cap1.tool_ref = "consult_schema"
    agent.capabilities.append(cap1)

    cap2 = MagicMock()
    cap2.tool_ref = "delegate"
    agent.capabilities.append(cap2)

    studio.agents = [agent]

    # Artifact types (for consult_schema)
    studio.artifact_types = []

    return studio, agent


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_get_tool_definition(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        tool_def = registry.get_tool_definition("consult_schema")
        assert tool_def is not None
        assert tool_def.id == "consult_schema"

    def test_get_tool_definition_not_found(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        tool_def = registry.get_tool_definition("nonexistent")
        assert tool_def is None

    def test_get_tool_definition_implicit_persistence(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        tool_def = registry.get_tool_definition("save_artifact")
        assert tool_def is not None
        assert tool_def.id == "save_artifact"

    def test_get_tool_returns_implementation(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        # consult_schema has an implementation
        tool = registry.get_tool("consult_schema")
        assert tool is not None
        assert tool.id == "consult_schema"
        # Should be the real implementation, not UnavailableTool
        assert "ConsultSchema" in type(tool).__name__

    def test_get_tool_returns_stub_for_unimplemented(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        tool = registry.get_tool("unknown_tool")
        assert tool is not None
        assert tool.id == "unknown_tool"
        assert not tool.check_availability()

    def test_get_tool_not_found(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        with pytest.raises(KeyError, match="Tool not found"):
            registry.get_tool("nonexistent")

    def test_get_agent_tools(self):
        studio, agent = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        tools = registry.get_agent_tools(agent)

        # Agent has capabilities for consult_schema and delegate
        tool_ids = {t.id for t in tools}
        assert "consult_schema" in tool_ids
        assert "delegate" in tool_ids
        # unknown_tool should not be included (no capability)
        assert "unknown_tool" not in tool_ids

    def test_check_capability_allowed(self):
        studio, agent = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        assert registry.check_capability(agent, "consult_schema") is True
        assert registry.check_capability(agent, "delegate") is True

    def test_check_capability_denied(self):
        studio, agent = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        assert registry.check_capability(agent, "unknown_tool") is False

    def test_check_capability_allows_implicit_persistence(self):
        studio, agent = make_mock_studio_with_tools()

        cap = MagicMock()
        cap.tool_ref = None
        cap.category = "store_access"
        cap.access_level = "write"
        cap.stores = ["workspace"]
        agent.capabilities.append(cap)

        registry = ToolRegistry(studio)

        assert registry.check_capability(agent, "save_artifact") is True

    def test_enforce_capability_allowed(self):
        studio, agent = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        # Should not raise
        registry.enforce_capability(agent, "consult_schema")

    def test_enforce_capability_allows_implicit_persistence(self):
        studio, agent = make_mock_studio_with_tools()

        cap = MagicMock()
        cap.tool_ref = None
        cap.category = "store_access"
        cap.access_level = "write"
        cap.stores = ["workspace"]
        agent.capabilities.append(cap)

        registry = ToolRegistry(studio)

        registry.enforce_capability(agent, "save_artifact")

    def test_enforce_capability_denied(self):
        studio, agent = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        with pytest.raises(CapabilityViolationError) as exc_info:
            registry.enforce_capability(agent, "unknown_tool")

        assert exc_info.value.agent_id == "test_agent"
        assert exc_info.value.tool_id == "unknown_tool"

    def test_get_langchain_tools(self):
        studio, agent = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        schemas = registry.get_langchain_tools(agent)

        # Should get schemas for available tools
        names = {s["name"] for s in schemas}
        assert "consult_schema" in names
        assert "delegate" in names

        # Check schema structure
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_list_all_tools(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        all_tools = registry.list_all_tools()
        assert "consult_schema" in all_tools
        assert "delegate" in all_tools
        assert "unknown_tool" in all_tools

    def test_list_implemented_tools(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        implemented = registry.list_implemented_tools()
        # These should be registered
        assert "consult_schema" in implemented
        assert "delegate" in implemented
        assert "validate_artifact" in implemented

    def test_list_unimplemented_tools(self):
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        unimplemented = registry.list_unimplemented_tools()
        # unknown_tool is in studio but not implemented
        assert "unknown_tool" in unimplemented

    def test_agent_with_store_write_gets_persistence_tools(self):
        studio, agent = make_mock_studio_with_tools()

        cap = MagicMock()
        cap.tool_ref = None
        cap.category = "store_access"
        cap.stores = ["workspace"]
        cap.access_level = "write"
        agent.capabilities.append(cap)

        registry = ToolRegistry(studio)
        tool_ids = {tool.id for tool in registry.get_agent_tools(agent)}

        assert {"save_artifact", "update_artifact", "delete_artifact"}.issubset(tool_ids)


class TestRegisterTool:
    """Tests for register_tool decorator."""

    def test_decorator_registers_class(self):
        # Create a test tool class
        @register_tool("test_decorator_tool")
        class TestDecoratorTool(BaseTool):
            async def execute(self, _args: dict) -> ToolResult:
                return ToolResult(success=True, data={})

        # Check it's registered
        assert "test_decorator_tool" in TOOL_IMPLEMENTATIONS
        assert TOOL_IMPLEMENTATIONS["test_decorator_tool"] is TestDecoratorTool

        # Cleanup
        del TOOL_IMPLEMENTATIONS["test_decorator_tool"]


class TestBuildAgentTools:
    """Tests for build_agent_tools convenience function."""

    def test_returns_tools_for_agent(self):
        studio, agent = make_mock_studio_with_tools()

        tools = build_agent_tools(agent, studio)

        tool_ids = {t.id for t in tools}
        assert "consult_schema" in tool_ids
        assert "delegate" in tool_ids


class TestToolRegistryInteractiveMode:
    """Tests for interactive mode handling in ToolRegistry."""

    def test_default_interactive_mode(self):
        """Registry should default to interactive mode."""
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio)

        tool = registry.get_tool("consult_schema")
        assert tool.context.interactive is True

    def test_non_interactive_mode_passed_to_tool(self):
        """Non-interactive flag should be passed to tool context."""
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio, interactive=False)

        tool = registry.get_tool("consult_schema")
        assert tool.context.interactive is False

    def test_interactive_mode_passed_to_tool(self):
        """Interactive flag should be passed to tool context."""
        studio, _ = make_mock_studio_with_tools()
        registry = ToolRegistry(studio, interactive=True)

        tool = registry.get_tool("consult_schema")
        assert tool.context.interactive is True

    def test_non_interactive_mode_for_agent_tools(self):
        """Non-interactive flag should be passed to all agent tools."""
        studio, agent = make_mock_studio_with_tools()
        registry = ToolRegistry(studio, interactive=False)

        tools = registry.get_agent_tools(agent)

        for tool in tools:
            assert tool.context.interactive is False
