"""Tests for tool protocol and finalization tools."""

from __future__ import annotations

from questfoundry.tools import (
    SubmitBrainstormTool,
    SubmitDreamTool,
    ToolCall,
    ToolDefinition,
    get_finalization_tool,
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


# --- SubmitDreamTool Tests ---


def test_submit_dream_tool_definition() -> None:
    """SubmitDreamTool has correct definition."""
    tool = SubmitDreamTool()
    definition = tool.definition

    assert definition.name == "submit_dream"
    assert "creative vision" in definition.description.lower()
    assert definition.parameters["type"] == "object"

    # Check required fields
    required = definition.parameters.get("required", [])
    assert "genre" in required
    assert "tone" in required
    assert "audience" in required
    assert "themes" in required


def test_submit_dream_tool_properties() -> None:
    """SubmitDreamTool has expected properties in schema."""
    tool = SubmitDreamTool()
    properties = tool.definition.parameters["properties"]

    # Core fields
    assert "genre" in properties
    assert "subgenre" in properties
    assert "tone" in properties
    assert "audience" in properties
    assert "themes" in properties
    assert "style_notes" in properties

    # Nested objects
    assert "scope" in properties
    assert "content_notes" in properties


def test_submit_dream_tool_execute() -> None:
    """SubmitDreamTool.execute returns confirmation message."""
    tool = SubmitDreamTool()
    result = tool.execute({"genre": "fantasy", "tone": ["epic"]})

    assert isinstance(result, str)
    assert "validation" in result.lower() or "submitted" in result.lower()


# --- SubmitBrainstormTool Tests ---


def test_submit_brainstorm_tool_definition() -> None:
    """SubmitBrainstormTool has correct definition."""
    tool = SubmitBrainstormTool()
    definition = tool.definition

    assert definition.name == "submit_brainstorm"
    assert "brainstorm" in definition.description.lower()

    # Check required fields
    required = definition.parameters.get("required", [])
    assert "characters" in required
    assert "plot_hooks" in required


def test_submit_brainstorm_tool_properties() -> None:
    """SubmitBrainstormTool has expected properties."""
    tool = SubmitBrainstormTool()
    properties = tool.definition.parameters["properties"]

    assert "characters" in properties
    assert "settings" in properties
    assert "plot_hooks" in properties
    assert "conflicts" in properties
    assert "notes" in properties


def test_submit_brainstorm_tool_execute() -> None:
    """SubmitBrainstormTool.execute returns confirmation message."""
    tool = SubmitBrainstormTool()
    result = tool.execute({"characters": [], "plot_hooks": []})

    assert isinstance(result, str)


# --- get_finalization_tool Tests ---


def test_get_finalization_tool_dream() -> None:
    """get_finalization_tool returns SubmitDreamTool for 'dream'."""
    tool = get_finalization_tool("dream")

    assert tool is not None
    assert isinstance(tool, SubmitDreamTool)
    assert tool.definition.name == "submit_dream"


def test_get_finalization_tool_brainstorm() -> None:
    """get_finalization_tool returns SubmitBrainstormTool for 'brainstorm'."""
    tool = get_finalization_tool("brainstorm")

    assert tool is not None
    assert isinstance(tool, SubmitBrainstormTool)
    assert tool.definition.name == "submit_brainstorm"


def test_get_finalization_tool_unknown() -> None:
    """get_finalization_tool returns None for unknown stage."""
    tool = get_finalization_tool("unknown_stage")

    assert tool is None


def test_get_finalization_tool_case_sensitive() -> None:
    """get_finalization_tool is case-sensitive."""
    # Uppercase should not match
    tool = get_finalization_tool("DREAM")

    assert tool is None


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


def test_tool_definition_as_dict() -> None:
    """ToolDefinition can be converted to dict for LangChain."""
    tool = SubmitDreamTool()
    definition = tool.definition

    # These fields are needed for LangChain tool binding
    assert hasattr(definition, "name")
    assert hasattr(definition, "description")
    assert hasattr(definition, "parameters")

    # Parameters should be JSON Schema compatible
    params = definition.parameters
    assert params.get("type") == "object"
    assert "properties" in params
