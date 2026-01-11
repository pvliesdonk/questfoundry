"""Tests for tool protocol and base types."""

from __future__ import annotations

from questfoundry.tools import (
    ToolCall,
    ToolDefinition,
)

# --- ToolDefinition Tests ---


def test_tool_definition_creation() -> None:
    """ToolDefinition can be created with required fields."""
    definition = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
    )

    assert definition.name == "test_tool"
    assert definition.description == "A test tool"
    assert definition.parameters == {"type": "object", "properties": {}}


def test_tool_definition_default_parameters() -> None:
    """ToolDefinition has default empty parameters."""
    definition = ToolDefinition(name="simple", description="Simple tool")

    assert definition.parameters == {"type": "object", "properties": {}}


# --- ToolCall Tests ---


def test_tool_call_creation() -> None:
    """ToolCall can be created with all fields."""
    call = ToolCall(
        id="call_123",
        name="submit_dream",
        arguments={"genre": "fantasy", "tone": ["epic"]},
    )

    assert call.id == "call_123"
    assert call.name == "submit_dream"
    assert call.arguments["genre"] == "fantasy"


def test_tool_call_empty_arguments() -> None:
    """ToolCall works with empty arguments."""
    call = ToolCall(id="call_456", name="test", arguments={})

    assert call.arguments == {}


# --- Tool Protocol Tests ---


class MockTool:
    """Mock implementation of Tool protocol."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mock_tool",
            description="A mock tool for testing",
            parameters={"type": "object", "properties": {"input": {"type": "string"}}},
        )

    def execute(self, arguments: dict) -> str:
        return f"Executed with: {arguments.get('input', 'none')}"


def test_tool_protocol_implementation() -> None:
    """Custom tool implements Tool protocol correctly."""
    tool = MockTool()

    # Verify protocol compliance
    assert hasattr(tool, "definition")
    assert hasattr(tool, "execute")

    definition = tool.definition
    assert isinstance(definition, ToolDefinition)

    result = tool.execute({"input": "test"})
    assert "test" in result


def test_tool_definition_has_required_attributes() -> None:
    """ToolDefinition has attributes needed for LangChain."""
    definition = ToolDefinition(
        name="test",
        description="Test tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    )

    # These fields are needed for LangChain tool binding
    assert hasattr(definition, "name")
    assert hasattr(definition, "description")
    assert hasattr(definition, "parameters")

    # Parameters should be JSON Schema compatible
    params = definition.parameters
    assert params.get("type") == "object"
    assert "properties" in params
