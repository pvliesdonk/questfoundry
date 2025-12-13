"""Tests for playbook consultation tools."""

from pathlib import Path

import pytest

from questfoundry.runtime.domain import load_studio
from questfoundry.runtime.playbook_tracker import PlaybookTracker
from questfoundry.runtime.tools.playbook import (
    ConsultPlaybookV4,
    create_consult_playbook_tool,
)


# Path to the domain-v4 test data
DOMAIN_V4_PATH = Path(__file__).parents[4] / "domain-v4"


@pytest.fixture
def studio():
    """Load the domain-v4 studio."""
    studio_path = DOMAIN_V4_PATH / "studio.json"
    return load_studio(studio_path)


@pytest.fixture
def tracker():
    """Create a fresh tracker."""
    return PlaybookTracker()


class TestConsultPlaybookV4:
    """Tests for ConsultPlaybookV4 tool."""

    def test_creates_tool(self, studio) -> None:
        """Test creating playbook tool."""
        tool = create_consult_playbook_tool(studio)

        assert isinstance(tool, ConsultPlaybookV4)
        assert tool.name == "consult_playbook"
        assert tool.studio is studio

    def test_creates_tool_with_tracker(self, studio, tracker) -> None:
        """Test creating tool with tracker."""
        tool = create_consult_playbook_tool(studio, tracker)

        assert tool.tracker is tracker

    def test_consult_existing_playbook(self, studio) -> None:
        """Test consulting an existing playbook."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "Story Spark" in result
        assert "topology_design" in result
        assert "brief_creation" in result

    def test_consult_playbook_shows_purpose(self, studio) -> None:
        """Test that playbook purpose is shown."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "Purpose" in result
        assert "structure" in result.lower() or "topology" in result.lower()

    def test_consult_playbook_shows_phases(self, studio) -> None:
        """Test that playbook phases are shown."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "Phases" in result
        assert "ENTRY" in result  # Entry point marker

    def test_consult_playbook_shows_outputs(self, studio) -> None:
        """Test that expected outputs are shown."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "Expected Outputs" in result or "Outputs" in result
        assert "section_brief" in result

    def test_consult_nonexistent_playbook(self, studio) -> None:
        """Test consulting a nonexistent playbook."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("nonexistent_playbook")

        assert "not found" in result.lower()
        assert "Available playbooks" in result

    def test_partial_match(self, studio) -> None:
        """Test partial matching of playbook names."""
        tool = create_consult_playbook_tool(studio)

        # "spark" should match "story_spark"
        result = tool._run("spark")

        assert "Story Spark" in result

    def test_notifies_tracker(self, studio, tracker) -> None:
        """Test that tracker is notified on consultation."""
        tool = create_consult_playbook_tool(studio, tracker)

        tool._run("story_spark")

        assert "story_spark" in tracker.consulted_playbooks
        assert tracker.active_playbook_id == "story_spark"

    def test_no_studio_error(self) -> None:
        """Test error when studio not configured."""
        tool = ConsultPlaybookV4()

        result = tool._run("story_spark")

        assert "Error" in result
        assert "not configured" in result

    def test_shows_triggers(self, studio) -> None:
        """Test that triggers are shown."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "Triggers" in result or "When to Use" in result

    def test_shows_quality_criteria(self, studio) -> None:
        """Test that quality criteria are shown."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "Quality" in result
        assert "integrity" in result.lower() or "reachability" in result.lower()


class TestPlaybookFormatting:
    """Tests for playbook output formatting."""

    def test_includes_step_details(self, studio) -> None:
        """Test that steps include agent and action."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        # Should include step agent assignments
        assert "plotwright" in result.lower() or "gatekeeper" in result.lower()

    def test_includes_completion_criteria(self, studio) -> None:
        """Test that completion criteria are shown."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "Completion Criteria" in result or "completion_criteria" in result.lower()

    def test_includes_transitions(self, studio) -> None:
        """Test that phase transitions are shown."""
        tool = create_consult_playbook_tool(studio)

        result = tool._run("story_spark")

        assert "On Success" in result or "next_phases" in result.lower()
