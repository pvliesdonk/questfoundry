"""Tests for finalization tools."""

from __future__ import annotations

from questfoundry.tools import Tool, ToolDefinition
from questfoundry.tools.finalization import (
    FINALIZATION_TOOLS,
    SubmitBrainstormTool,
    SubmitDreamTool,
    get_finalization_tool,
)

# --- SubmitDreamTool Tests ---


def test_submit_dream_tool_definition() -> None:
    """SubmitDreamTool has correct name and description."""
    tool = SubmitDreamTool()
    definition = tool.definition

    assert definition.name == "submit_dream"
    assert "creative vision" in definition.description.lower()


def test_submit_dream_tool_parameters() -> None:
    """SubmitDreamTool has expected required parameters."""
    tool = SubmitDreamTool()
    params = tool.definition.parameters

    assert params["type"] == "object"
    assert "genre" in params["properties"]
    assert "tone" in params["properties"]
    assert "audience" in params["properties"]
    assert "themes" in params["properties"]

    required = params.get("required", [])
    assert "genre" in required
    assert "tone" in required
    assert "audience" in required
    assert "themes" in required


def test_submit_dream_tool_optional_parameters() -> None:
    """SubmitDreamTool has optional parameters for scope and content_notes."""
    tool = SubmitDreamTool()
    params = tool.definition.parameters

    assert "scope" in params["properties"]
    assert "content_notes" in params["properties"]
    assert "style_notes" in params["properties"]
    assert "subgenre" in params["properties"]


def test_submit_dream_tool_execute() -> None:
    """SubmitDreamTool.execute returns confirmation message."""
    tool = SubmitDreamTool()
    result = tool.execute({"genre": "fantasy", "tone": ["dark"]})

    assert "submitted" in result.lower() or "validation" in result.lower()


def test_submit_dream_tool_implements_protocol() -> None:
    """SubmitDreamTool implements the Tool protocol."""
    tool = SubmitDreamTool()

    # Should be an instance of Tool (runtime_checkable protocol)
    assert isinstance(tool, Tool)


# --- SubmitBrainstormTool Tests ---


def test_submit_brainstorm_tool_definition() -> None:
    """SubmitBrainstormTool has correct name and description."""
    tool = SubmitBrainstormTool()
    definition = tool.definition

    assert definition.name == "submit_brainstorm"
    assert "brainstorm" in definition.description.lower()


def test_submit_brainstorm_tool_parameters() -> None:
    """SubmitBrainstormTool has expected required parameters."""
    tool = SubmitBrainstormTool()
    params = tool.definition.parameters

    assert params["type"] == "object"
    assert "characters" in params["properties"]
    assert "plot_hooks" in params["properties"]

    required = params.get("required", [])
    assert "characters" in required
    assert "plot_hooks" in required


def test_submit_brainstorm_tool_optional_parameters() -> None:
    """SubmitBrainstormTool has optional parameters."""
    tool = SubmitBrainstormTool()
    params = tool.definition.parameters

    assert "settings" in params["properties"]
    assert "conflicts" in params["properties"]
    assert "notes" in params["properties"]


def test_submit_brainstorm_tool_execute() -> None:
    """SubmitBrainstormTool.execute returns confirmation message."""
    tool = SubmitBrainstormTool()
    result = tool.execute({"characters": [], "plot_hooks": []})

    assert "submitted" in result.lower() or "validation" in result.lower()


def test_submit_brainstorm_tool_implements_protocol() -> None:
    """SubmitBrainstormTool implements the Tool protocol."""
    tool = SubmitBrainstormTool()

    assert isinstance(tool, Tool)


# --- get_finalization_tool Tests ---


def test_get_finalization_tool_dream() -> None:
    """get_finalization_tool returns SubmitDreamTool for 'dream' stage."""
    tool = get_finalization_tool("dream")

    assert tool is not None
    assert isinstance(tool, SubmitDreamTool)
    assert tool.definition.name == "submit_dream"


def test_get_finalization_tool_brainstorm() -> None:
    """get_finalization_tool returns SubmitBrainstormTool for 'brainstorm' stage."""
    tool = get_finalization_tool("brainstorm")

    assert tool is not None
    assert isinstance(tool, SubmitBrainstormTool)
    assert tool.definition.name == "submit_brainstorm"


def test_get_finalization_tool_unknown() -> None:
    """get_finalization_tool returns None for unknown stage."""
    tool = get_finalization_tool("unknown_stage")

    assert tool is None


def test_get_finalization_tool_returns_tool_protocol() -> None:
    """get_finalization_tool return type is Tool protocol compatible."""
    tool = get_finalization_tool("dream")

    # Return value should be usable as Tool
    assert tool is not None
    assert isinstance(tool, Tool)
    assert isinstance(tool.definition, ToolDefinition)


# --- FINALIZATION_TOOLS Registry Tests ---


def test_finalization_tools_registry_contains_dream() -> None:
    """FINALIZATION_TOOLS registry contains 'dream' entry."""
    assert "dream" in FINALIZATION_TOOLS
    assert FINALIZATION_TOOLS["dream"] is SubmitDreamTool


def test_finalization_tools_registry_contains_brainstorm() -> None:
    """FINALIZATION_TOOLS registry contains 'brainstorm' entry."""
    assert "brainstorm" in FINALIZATION_TOOLS
    assert FINALIZATION_TOOLS["brainstorm"] is SubmitBrainstormTool


def test_finalization_tools_registry_tools_are_callable() -> None:
    """All entries in FINALIZATION_TOOLS are callable and return tools."""
    for stage, tool_class in FINALIZATION_TOOLS.items():
        tool = tool_class()
        assert isinstance(tool, Tool), f"Tool for stage '{stage}' doesn't implement Tool protocol"
        assert hasattr(tool, "definition")
        assert hasattr(tool, "execute")
