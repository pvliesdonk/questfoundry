"""
Tests for TurnValidator - orchestrator enforcement.

Tests the tools-only output constraint for orchestrator agents:
- Orchestrators MUST end every turn with a terminating tool call
- Non-orchestrators don't require terminating tools
"""

import pytest

from questfoundry.runtime.agent.turn_validator import (
    TurnValidationConfig,
    TurnValidator,
    create_turn_validator,
)
from questfoundry.runtime.models import Agent, Studio, Tool
from questfoundry.runtime.models.base import Constraint
from questfoundry.runtime.models.enums import EnforcementType


@pytest.fixture
def mock_studio() -> Studio:
    """Create a mock studio with tools for testing."""
    return Studio(
        id="test-studio",
        name="Test Studio",
        tools=[
            Tool(
                id="delegate",
                name="Delegate",
                description="Delegate work to another agent",
                terminates_turn=True,
            ),
            Tool(
                id="communicate",
                name="Communicate",
                description="Send message to another agent",
                terminates_turn=True,
            ),
            Tool(
                id="search",
                name="Search",
                description="Search for information",
                terminates_turn=False,
            ),
            Tool(
                id="save_artifact",
                name="Save Artifact",
                description="Save an artifact",
                terminates_turn=False,
            ),
        ],
    )


@pytest.fixture
def orchestrator_agent() -> Agent:
    """Create an orchestrator agent for testing."""
    return Agent(
        id="showrunner",
        name="Showrunner",
        archetypes=["orchestrator"],
        constraints=[
            Constraint(
                id="tools_only_output",
                name="Tools-Only Output",
                rule="Must use tools for all output",
                enforcement=EnforcementType.RUNTIME,
            ),
        ],
    )


@pytest.fixture
def creator_agent() -> Agent:
    """Create a non-orchestrator agent for testing."""
    return Agent(
        id="scene-smith",
        name="Scene Smith",
        archetypes=["creator"],
    )


@pytest.fixture
def agent_with_runtime_constraint() -> Agent:
    """Create an agent with tools_only constraint but not orchestrator archetype."""
    return Agent(
        id="custom-agent",
        name="Custom Agent",
        archetypes=["creator"],
        constraints=[
            Constraint(
                id="tools_only_output",
                name="Tools-Only Output",
                rule="Must use tools for all output",
                enforcement=EnforcementType.RUNTIME,
            ),
        ],
    )


class MockToolCallRequest:
    """Mock ToolCallRequest for testing."""

    def __init__(self, name: str, arguments: dict | None = None):
        self.name = name
        self.arguments = arguments or {}
        self.id = f"call_{name}"


class TestTurnValidatorInit:
    """Tests for TurnValidator initialization."""

    def test_create_with_default_config(self, mock_studio: Studio):
        """Should create validator with default config."""
        validator = TurnValidator(mock_studio)
        assert validator is not None
        assert validator._config.max_retries == 3

    def test_create_with_custom_config(self, mock_studio: Studio):
        """Should create validator with custom config."""
        config = TurnValidationConfig(max_retries=5)
        validator = TurnValidator(mock_studio, config)
        assert validator._config.max_retries == 5

    def test_discovers_terminating_tools(self, mock_studio: Studio):
        """Should discover tools with terminates_turn=True."""
        validator = TurnValidator(mock_studio)
        assert "delegate" in validator.terminating_tools
        assert "communicate" in validator.terminating_tools
        assert "search" not in validator.terminating_tools
        assert "save_artifact" not in validator.terminating_tools

    def test_create_turn_validator_function(self, mock_studio: Studio):
        """Should work with factory function."""
        validator = create_turn_validator(mock_studio)
        assert validator is not None


class TestOrchestratorDetection:
    """Tests for orchestrator detection."""

    def test_is_orchestrator_true(self, mock_studio: Studio, orchestrator_agent: Agent):
        """Should detect orchestrator archetype."""
        validator = TurnValidator(mock_studio)
        assert validator.is_orchestrator(orchestrator_agent) is True

    def test_is_orchestrator_false(self, mock_studio: Studio, creator_agent: Agent):
        """Should not detect non-orchestrator as orchestrator."""
        validator = TurnValidator(mock_studio)
        assert validator.is_orchestrator(creator_agent) is False

    def test_has_tools_only_constraint(
        self, mock_studio: Studio, agent_with_runtime_constraint: Agent
    ):
        """Should detect tools_only constraint with runtime enforcement."""
        validator = TurnValidator(mock_studio)
        assert validator.has_tools_only_constraint(agent_with_runtime_constraint) is True

    def test_requires_terminating_tool_orchestrator(
        self, mock_studio: Studio, orchestrator_agent: Agent
    ):
        """Orchestrator should require terminating tool."""
        validator = TurnValidator(mock_studio)
        assert validator.requires_terminating_tool(orchestrator_agent) is True

    def test_requires_terminating_tool_creator(self, mock_studio: Studio, creator_agent: Agent):
        """Non-orchestrator should not require terminating tool."""
        validator = TurnValidator(mock_studio)
        assert validator.requires_terminating_tool(creator_agent) is False

    def test_requires_terminating_tool_with_constraint(
        self, mock_studio: Studio, agent_with_runtime_constraint: Agent
    ):
        """Agent with tools_only constraint should require terminating tool."""
        validator = TurnValidator(mock_studio)
        assert validator.requires_terminating_tool(agent_with_runtime_constraint) is True


class TestValidateTurn:
    """Tests for turn validation."""

    def test_valid_turn_with_terminating_tool(self, mock_studio: Studio, orchestrator_agent: Agent):
        """Should validate turn with terminating tool."""
        validator = TurnValidator(mock_studio)
        tool_calls = [MockToolCallRequest("delegate", {"task": "write chapter"})]

        result = validator.validate_turn(orchestrator_agent, tool_calls)

        assert result.valid is True
        assert result.has_terminating_tool is True
        assert result.terminating_tool_id == "delegate"
        assert result.is_orchestrator is True

    def test_invalid_turn_no_tools(self, mock_studio: Studio, orchestrator_agent: Agent):
        """Should reject turn without any tools for orchestrator."""
        validator = TurnValidator(mock_studio)

        result = validator.validate_turn(orchestrator_agent, None)

        assert result.valid is False
        assert result.has_terminating_tool is False
        assert result.is_orchestrator is True
        assert result.nudge_message is not None
        assert "delegate" in result.nudge_message.lower()

    def test_invalid_turn_no_terminating_tool(self, mock_studio: Studio, orchestrator_agent: Agent):
        """Should reject turn with only non-terminating tools for orchestrator."""
        validator = TurnValidator(mock_studio)
        tool_calls = [
            MockToolCallRequest("search", {"query": "test"}),
            MockToolCallRequest("save_artifact", {"data": "test"}),
        ]

        result = validator.validate_turn(orchestrator_agent, tool_calls)

        assert result.valid is False
        assert result.has_terminating_tool is False
        assert result.non_terminating_tools == ["search", "save_artifact"]
        assert result.nudge_message is not None
        assert "terminating tool" in result.nudge_message.lower()

    def test_valid_turn_non_orchestrator_no_tools(self, mock_studio: Studio, creator_agent: Agent):
        """Non-orchestrator should be valid without tools."""
        validator = TurnValidator(mock_studio)

        result = validator.validate_turn(creator_agent, None)

        assert result.valid is True
        assert result.is_orchestrator is False

    def test_valid_turn_non_orchestrator_with_tools(
        self, mock_studio: Studio, creator_agent: Agent
    ):
        """Non-orchestrator should be valid with any tools."""
        validator = TurnValidator(mock_studio)
        tool_calls = [MockToolCallRequest("search", {"query": "test"})]

        result = validator.validate_turn(creator_agent, tool_calls)

        assert result.valid is True
        assert result.is_orchestrator is False

    def test_mixed_tools_with_terminator(self, mock_studio: Studio, orchestrator_agent: Agent):
        """Should validate turn with mix of tools including terminator."""
        validator = TurnValidator(mock_studio)
        tool_calls = [
            MockToolCallRequest("search", {"query": "test"}),
            MockToolCallRequest("delegate", {"task": "write chapter"}),
        ]

        result = validator.validate_turn(orchestrator_agent, tool_calls)

        assert result.valid is True
        assert result.has_terminating_tool is True
        assert result.terminating_tool_id == "delegate"
        assert result.non_terminating_tools == ["search"]


class TestNudgeMessages:
    """Tests for nudge message generation."""

    def test_no_tools_nudge_mentions_delegate(self, mock_studio: Studio, orchestrator_agent: Agent):
        """No-tools nudge should mention delegate."""
        validator = TurnValidator(mock_studio)
        result = validator.validate_turn(orchestrator_agent, None)

        assert "delegate" in result.nudge_message.lower()

    def test_no_tools_nudge_mentions_communicate(
        self, mock_studio: Studio, orchestrator_agent: Agent
    ):
        """No-tools nudge should mention communicate."""
        validator = TurnValidator(mock_studio)
        result = validator.validate_turn(orchestrator_agent, None)

        assert "communicate" in result.nudge_message.lower()

    def test_missing_terminator_nudge_mentions_tools_used(
        self, mock_studio: Studio, orchestrator_agent: Agent
    ):
        """Missing terminator nudge should mention which tools were used."""
        validator = TurnValidator(mock_studio)
        tool_calls = [MockToolCallRequest("search", {"query": "test"})]

        result = validator.validate_turn(orchestrator_agent, tool_calls)

        assert "search" in result.nudge_message.lower()

    def test_nudge_includes_tool_hints(self, mock_studio: Studio, orchestrator_agent: Agent):
        """Nudge should include list of terminating tools when configured."""
        config = TurnValidationConfig(include_tool_hints=True)
        validator = TurnValidator(mock_studio, config)

        result = validator.validate_turn(orchestrator_agent, None)

        assert "delegate" in result.nudge_message
        assert "communicate" in result.nudge_message

    def test_nudge_without_tool_hints(self, mock_studio: Studio, orchestrator_agent: Agent):
        """Nudge should not include tool list when hints disabled."""
        config = TurnValidationConfig(include_tool_hints=False)
        validator = TurnValidator(mock_studio, config)

        result = validator.validate_turn(orchestrator_agent, None)

        # Should still have the core message
        assert "delegate" in result.nudge_message.lower()
        # But not the "Terminating tools available:" line
        assert "terminating tools available" not in result.nudge_message.lower()


class TestTurnValidationConfig:
    """Tests for TurnValidationConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = TurnValidationConfig()
        assert config.max_retries == 3
        assert config.allow_intermediate_tools is True
        assert config.include_tool_hints is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = TurnValidationConfig(
            max_retries=5,
            allow_intermediate_tools=False,
            include_tool_hints=False,
        )
        assert config.max_retries == 5
        assert config.allow_intermediate_tools is False
        assert config.include_tool_hints is False


class TestGetTerminatingToolNames:
    """Tests for getting terminating tool names."""

    def test_returns_sorted_list(self, mock_studio: Studio):
        """Should return sorted list of terminating tools."""
        validator = TurnValidator(mock_studio)
        names = validator.get_terminating_tool_names()

        assert names == ["communicate", "delegate"]  # Sorted alphabetically
