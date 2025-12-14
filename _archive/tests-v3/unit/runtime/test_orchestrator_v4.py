"""Tests for orchestrator v4."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from questfoundry.runtime.domain import load_studio
from questfoundry.runtime.orchestrator_v4 import OrchestratorV4, _build_entry_agent_tools
from questfoundry.runtime.playbook_tracker import PlaybookTracker
from questfoundry.runtime.state import create_initial_state


# Path to the domain-v4 test data
DOMAIN_V4_PATH = Path(__file__).parents[3] / "domain-v4"


@pytest.fixture
def studio():
    """Load the domain-v4 studio."""
    studio_path = DOMAIN_V4_PATH / "studio.json"
    return load_studio(studio_path)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MagicMock()


class TestOrchestratorV4Init:
    """Tests for OrchestratorV4 initialization."""

    def test_init_with_authoring_mode(self, studio, mock_llm) -> None:
        """Test initialization with authoring mode."""
        orch = OrchestratorV4(studio, mock_llm, entry_mode="authoring")

        assert orch.entry_agent_id == "showrunner"
        assert orch.entry_mode == "authoring"
        assert orch.playbook_tracker is not None

    def test_init_with_playtest_mode(self, studio, mock_llm) -> None:
        """Test initialization with playtest mode."""
        orch = OrchestratorV4(studio, mock_llm, entry_mode="playtest")

        assert orch.entry_agent_id == "player_narrator"
        assert orch.entry_mode == "playtest"

    def test_init_with_invalid_mode(self, studio, mock_llm) -> None:
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError) as exc_info:
            OrchestratorV4(studio, mock_llm, entry_mode="invalid")

        assert "No entry agent for mode 'invalid'" in str(exc_info.value)
        assert "Available modes" in str(exc_info.value)

    def test_default_parameters(self, studio, mock_llm) -> None:
        """Test default parameter values."""
        orch = OrchestratorV4(studio, mock_llm)

        assert orch.max_delegations == 50
        assert orch.cold_store is None
        assert orch.stream is False


class TestBuildEntryAgentTools:
    """Tests for _build_entry_agent_tools."""

    def test_builds_tools_for_showrunner(self, studio) -> None:
        """Test building tools for showrunner entry agent."""
        sr = studio.agents["showrunner"]
        state = create_initial_state("test", "test request")
        tracker = PlaybookTracker()

        tools = _build_entry_agent_tools(sr, studio, state, None, tracker)

        assert len(tools) > 0
        tool_names = [t.name for t in tools]

        # Should have orchestration tools
        assert "delegate_to" in tool_names

        # Should have consult tools
        assert "consult_playbook" in tool_names
        assert "consult_schema" in tool_names

        # Should have state tools
        assert "read_artifact" in tool_names or "read_hot_sot" in tool_names

        # Should have terminate (entry agent)
        assert "terminate" in tool_names

    def test_builds_tools_for_player_narrator(self, studio) -> None:
        """Test building tools for player_narrator entry agent."""
        pn = studio.agents["player_narrator"]
        state = create_initial_state("test", "test request")
        tracker = PlaybookTracker()

        tools = _build_entry_agent_tools(pn, studio, state, None, tracker)

        assert len(tools) > 0
        tool_names = [t.name for t in tools]

        # Should have terminate (entry agent)
        assert "terminate" in tool_names

        # Should NOT have return_to_sr (entry agents terminate, don't return)
        assert "return_to_sr" not in tool_names

    def test_injects_playbook_tracker(self, studio) -> None:
        """Test that playbook tracker is injected into tools."""
        sr = studio.agents["showrunner"]
        state = create_initial_state("test", "test request")
        tracker = PlaybookTracker()

        tools = _build_entry_agent_tools(sr, studio, state, None, tracker)

        # Find consult_playbook tool
        playbook_tool = next(
            (t for t in tools if t.name == "consult_playbook"), None
        )
        assert playbook_tool is not None
        assert playbook_tool.tracker is tracker


class TestOrchestratorV4Integration:
    """Integration tests for OrchestratorV4."""

    @pytest.mark.asyncio
    async def test_run_terminates_on_terminate_call(self, studio, mock_llm) -> None:
        """Test that run terminates when entry agent calls terminate."""
        orch = OrchestratorV4(studio, mock_llm, entry_mode="authoring")

        # Mock the executor to immediately return terminate
        with patch(
            "questfoundry.runtime.orchestrator_v4.ToolExecutor"
        ) as MockExecutor:
            mock_executor = AsyncMock()
            mock_executor.run.return_value = MagicMock(
                success=True,
                done_tool_result={
                    "_stop_tool": "terminate",
                    "termination": {"reason": "completed"},
                },
                tool_results=[],
            )
            MockExecutor.return_value = mock_executor

            result = await orch.run("Test request")

            assert "termination" in result["metadata"]
            assert result["metadata"]["termination"]["reason"] == "completed"

    @pytest.mark.asyncio
    async def test_run_handles_delegation(self, studio, mock_llm) -> None:
        """Test that run handles delegation correctly."""
        orch = OrchestratorV4(studio, mock_llm, entry_mode="authoring")

        call_count = 0

        async def mock_run(prompt):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: delegate
                return MagicMock(
                    success=True,
                    done_tool_result={
                        "_stop_tool": "delegate_to",
                        "success": True,
                        "delegation_request": {
                            "role": "plotwright",
                            "task": "Write a story",
                        },
                    },
                    tool_results=[],
                )
            else:
                # Second call: terminate
                return MagicMock(
                    success=True,
                    done_tool_result={
                        "_stop_tool": "terminate",
                        "termination": {"reason": "completed"},
                    },
                    tool_results=[],
                )

        with patch(
            "questfoundry.runtime.orchestrator_v4.ToolExecutor"
        ) as MockExecutor:
            mock_executor = AsyncMock()
            mock_executor.run.side_effect = mock_run
            MockExecutor.return_value = mock_executor

            result = await orch.run("Test request")

            assert "delegation_history" in result["metadata"]
            assert len(result["metadata"]["delegation_history"]) == 1
            assert result["metadata"]["delegation_history"][0]["role"] == "plotwright"

    @pytest.mark.asyncio
    async def test_run_stops_at_max_delegations(self, studio, mock_llm) -> None:
        """Test that run stops at max delegations."""
        orch = OrchestratorV4(studio, mock_llm, entry_mode="authoring", max_delegations=2)

        # Always delegate
        with patch(
            "questfoundry.runtime.orchestrator_v4.ToolExecutor"
        ) as MockExecutor:
            mock_executor = AsyncMock()
            mock_executor.run.return_value = MagicMock(
                success=True,
                done_tool_result={
                    "_stop_tool": "delegate_to",
                    "success": True,
                    "delegation_request": {
                        "role": "plotwright",
                        "task": "Write a story",
                    },
                },
                tool_results=[],
            )
            MockExecutor.return_value = mock_executor

            result = await orch.run("Test request")

            assert "error" in result["metadata"]
            assert "Max delegations" in result["metadata"]["error"]

    @pytest.mark.asyncio
    async def test_run_handles_execution_failure(self, studio, mock_llm) -> None:
        """Test that run handles execution failures gracefully."""
        orch = OrchestratorV4(studio, mock_llm, entry_mode="authoring")

        with patch(
            "questfoundry.runtime.orchestrator_v4.ToolExecutor"
        ) as MockExecutor:
            mock_executor = AsyncMock()
            mock_executor.run.return_value = MagicMock(
                success=False,
                error="LLM error",
                done_tool_result=None,
                tool_results=[],
            )
            MockExecutor.return_value = mock_executor

            result = await orch.run("Test request")

            assert "error" in result["metadata"]
            assert "LLM error" in result["metadata"]["error"]


class TestPlaybookTrackerIntegration:
    """Tests for playbook tracker integration."""

    @pytest.mark.asyncio
    async def test_artifacts_tracked_for_nudging(self, studio, mock_llm) -> None:
        """Test that artifacts are tracked for playbook nudging."""
        orch = OrchestratorV4(studio, mock_llm, entry_mode="authoring")

        call_count = 0

        async def mock_run(prompt):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Delegate
                return MagicMock(
                    success=True,
                    done_tool_result={
                        "_stop_tool": "delegate_to",
                        "success": True,
                        "delegation_request": {
                            "role": "plotwright",
                            "task": "Write a story",
                        },
                    },
                    tool_results=[],
                )
            elif call_count == 2:
                # Specialist returns with artifacts
                return MagicMock(
                    success=True,
                    done_tool_result={
                        "status": "completed",
                        "message": "Done",
                        "artifacts": ["section_brief/test"],
                    },
                    tool_results=[],
                )
            else:
                # Terminate
                return MagicMock(
                    success=True,
                    done_tool_result={
                        "_stop_tool": "terminate",
                        "termination": {"reason": "completed"},
                    },
                    tool_results=[],
                )

        with patch(
            "questfoundry.runtime.orchestrator_v4.ToolExecutor"
        ) as MockExecutor:
            mock_executor = AsyncMock()
            mock_executor.run.side_effect = mock_run
            MockExecutor.return_value = mock_executor

            result = await orch.run("Test request")

            # Verify tracker tracked the artifact
            assert "section_brief" in orch.playbook_tracker.produced_artifacts
