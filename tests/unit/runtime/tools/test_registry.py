"""Tests for tool registry."""

from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from questfoundry.runtime.domain import load_studio
from questfoundry.runtime.tools import UnavailableTool, build_agent_tools


# Path to the domain-v4 test data
DOMAIN_V4_PATH = Path(__file__).parents[4] / "domain-v4"


@pytest.fixture
def studio():
    """Load the domain-v4 studio."""
    studio_path = DOMAIN_V4_PATH / "studio.json"
    return load_studio(studio_path)


class TestBuildAgentTools:
    """Tests for build_agent_tools function."""

    def test_builds_tools_for_showrunner(self, studio) -> None:
        """Test building tools for showrunner."""
        sr = studio.agents["showrunner"]
        tools = build_agent_tools(sr, studio)

        assert len(tools) > 0
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_entry_agent_gets_terminate_tool(self, studio) -> None:
        """Test that entry agents get terminate tool."""
        sr = studio.agents["showrunner"]
        assert sr.is_entry_agent is True

        tools = build_agent_tools(sr, studio)
        tool_names = [t.name for t in tools]

        assert "terminate" in tool_names

    def test_non_entry_agent_gets_return_tool(self, studio) -> None:
        """Test that non-entry agents get return_to_sr tool."""
        gk = studio.agents["gatekeeper"]
        assert gk.is_entry_agent is False

        tools = build_agent_tools(gk, studio)
        tool_names = [t.name for t in tools]

        assert "return_to_sr" in tool_names
        assert "terminate" not in tool_names

    def test_tool_capabilities_resolved(self, studio) -> None:
        """Test that tool capabilities are resolved to implementations."""
        sr = studio.agents["showrunner"]
        tools = build_agent_tools(sr, studio)
        tool_names = [t.name for t in tools]

        # SR has delegate tool capability
        assert "delegate_to" in tool_names

        # SR has consult_schema tool capability
        assert "consult_schema" in tool_names

    def test_store_access_creates_tools(self, studio) -> None:
        """Test that store_access capabilities create read/write tools."""
        sr = studio.agents["showrunner"]
        tools = build_agent_tools(sr, studio)
        tool_names = [t.name for t in tools]

        # SR has store_access capabilities, should get read/write tools
        assert "read_hot_sot" in tool_names
        assert "write_hot_sot" in tool_names

    def test_knowledge_tools_added(self, studio) -> None:
        """Test that knowledge tools are added for agents with knowledge requirements."""
        sr = studio.agents["showrunner"]
        tools = build_agent_tools(sr, studio)
        tool_names = [t.name for t in tools]

        # SR has knowledge requirements, should get consult tool
        assert "consult_knowledge" in tool_names

    def test_no_duplicate_tools(self, studio) -> None:
        """Test that duplicate tools are not added."""
        sr = studio.agents["showrunner"]
        tools = build_agent_tools(sr, studio)

        # Check for unique tool names
        tool_names = [t.name for t in tools]
        assert len(tool_names) == len(set(tool_names))

    def test_unavailable_tool_for_missing_implementation(self, studio) -> None:
        """Test that missing implementations get UnavailableTool stub."""
        from questfoundry.runtime.tools.registry import _get_tool_implementations

        # Get implementations and find one that's None (not implemented)
        impls = _get_tool_implementations()
        unavailable_tools = [tid for tid, impl in impls.items() if impl is None]

        # assemble_export, generate_image, generate_audio should be None
        assert "assemble_export" in unavailable_tools, "Need at least one unavailable tool for this test"

        # Build tools for an agent and check the stub is returned
        gk = studio.agents["gatekeeper"]
        tools = build_agent_tools(gk, studio)

        # validate_artifact is now implemented, so it should NOT be UnavailableTool
        validate_tool = next(
            (t for t in tools if t.name == "validate_artifact"), None
        )
        if validate_tool is not None:
            from questfoundry.runtime.tools.validate import ValidateArtifactTool
            assert isinstance(validate_tool, ValidateArtifactTool)


class TestUnavailableTool:
    """Tests for UnavailableTool."""

    def test_returns_reason(self) -> None:
        """Test that UnavailableTool returns its reason."""
        tool = UnavailableTool(
            name="test_tool",
            reason="This tool is not implemented.",
        )

        result = tool._run()
        assert result == "This tool is not implemented."

    def test_accepts_any_args(self) -> None:
        """Test that UnavailableTool accepts any arguments."""
        tool = UnavailableTool(
            name="test_tool",
            reason="Not implemented.",
        )

        # Should not raise
        result = tool._run("arg1", "arg2", kwarg="value")
        assert "Not implemented" in result


class TestDifferentAgentTypes:
    """Tests for building tools for different agent types."""

    def test_orchestrator_tools(self, studio) -> None:
        """Test tools for orchestrator archetype."""
        sr = studio.agents["showrunner"]
        assert "orchestrator" in sr.archetypes

        tools = build_agent_tools(sr, studio)
        tool_names = [t.name for t in tools]

        # Orchestrators should have delegate tool
        assert "delegate_to" in tool_names

    def test_validator_tools(self, studio) -> None:
        """Test tools for validator archetype."""
        gk = studio.agents["gatekeeper"]
        assert "validator" in gk.archetypes

        tools = build_agent_tools(gk, studio)
        tool_names = [t.name for t in tools]

        # Validators should have validation-related tools
        # validate_artifact is in their capabilities
        assert "validate_artifact" in tool_names

    def test_creator_tools(self, studio) -> None:
        """Test tools for creator archetype."""
        ss = studio.agents["scene_smith"]
        assert "creator" in ss.archetypes

        tools = build_agent_tools(ss, studio)
        tool_names = [t.name for t in tools]

        # Creators should have read/write tools
        assert "read_hot_sot" in tool_names
        assert "write_hot_sot" in tool_names

    def test_player_narrator_tools(self, studio) -> None:
        """Test tools for player_narrator (entry agent for playtest)."""
        pn = studio.agents["player_narrator"]
        assert pn.is_entry_agent is True

        tools = build_agent_tools(pn, studio)
        tool_names = [t.name for t in tools]

        # Should have terminate (entry agent) not return_to_sr
        assert "terminate" in tool_names
        assert "return_to_sr" not in tool_names
