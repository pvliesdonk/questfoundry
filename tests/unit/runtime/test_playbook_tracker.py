"""Tests for playbook tracker."""

from pathlib import Path

import pytest

from questfoundry.runtime.domain import load_studio
from questfoundry.runtime.playbook_tracker import (
    PlaybookProgress,
    PlaybookTracker,
)


# Path to the domain-v4 test data
DOMAIN_V4_PATH = Path(__file__).parents[3] / "domain-v4"


@pytest.fixture
def studio():
    """Load the domain-v4 studio."""
    studio_path = DOMAIN_V4_PATH / "studio.json"
    return load_studio(studio_path)


@pytest.fixture
def tracker():
    """Create a fresh tracker."""
    return PlaybookTracker()


class TestPlaybookTracker:
    """Tests for PlaybookTracker."""

    def test_initial_state(self, tracker) -> None:
        """Test initial tracker state."""
        assert tracker.consulted_playbooks == {}
        assert tracker.active_playbook_id is None
        assert tracker.produced_artifacts == set()

    def test_on_playbook_consulted(self, tracker, studio) -> None:
        """Test tracking a consulted playbook."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        assert "story_spark" in tracker.consulted_playbooks
        assert tracker.active_playbook_id == "story_spark"

        progress = tracker.consulted_playbooks["story_spark"]
        assert progress.playbook_name == "Story Spark"
        assert progress.current_phase == "topology_design"

    def test_extracts_expected_outputs(self, tracker, studio) -> None:
        """Test that expected outputs are extracted from playbook."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        progress = tracker.consulted_playbooks["story_spark"]
        # Story Spark produces section_brief as required output
        assert "section_brief" in progress.expected_outputs

    def test_on_artifact_created(self, tracker, studio) -> None:
        """Test tracking artifact creation."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        tracker.on_artifact_created("section_brief")

        assert "section_brief" in tracker.produced_artifacts
        progress = tracker.consulted_playbooks["story_spark"]
        assert "section_brief" in progress.produced_outputs

    def test_on_phase_entered(self, tracker, studio) -> None:
        """Test tracking phase entry."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        tracker.on_phase_entered("story_spark", "brief_creation")

        progress = tracker.consulted_playbooks["story_spark"]
        assert progress.current_phase == "brief_creation"
        assert "brief_creation" in progress.phases_entered

    def test_on_phase_completed(self, tracker, studio) -> None:
        """Test tracking phase completion."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        tracker.on_phase_completed("story_spark", "topology_design")

        progress = tracker.consulted_playbooks["story_spark"]
        assert "topology_design" in progress.phases_completed

    def test_on_playbook_completed(self, tracker, studio) -> None:
        """Test marking playbook as complete."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        tracker.on_playbook_completed("story_spark")

        progress = tracker.consulted_playbooks["story_spark"]
        assert progress.is_complete is True
        assert tracker.active_playbook_id is None


class TestPlaybookNudges:
    """Tests for playbook nudging."""

    def test_no_nudge_without_playbook(self, tracker) -> None:
        """Test no nudge when no playbook consulted."""
        nudge = tracker.get_nudge()
        assert nudge is None

    def test_nudge_for_missing_outputs(self, tracker, studio) -> None:
        """Test nudge when outputs are missing."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        nudge = tracker.get_nudge()

        assert nudge is not None
        assert "section_brief" in nudge
        assert "not yet produced" in nudge

    def test_no_nudge_when_outputs_produced(self, tracker, studio) -> None:
        """Test no nudge when all outputs produced."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        # Produce all expected outputs
        for output_type in list(
            tracker.consulted_playbooks["story_spark"].expected_outputs
        ):
            tracker.on_artifact_created(output_type)

        nudge = tracker.get_nudge()
        assert nudge is None

    def test_no_nudge_when_playbook_complete(self, tracker, studio) -> None:
        """Test no nudge when playbook marked complete."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)
        tracker.on_playbook_completed("story_spark")

        nudge = tracker.get_nudge()
        assert nudge is None

    def test_nudge_rate_limiting(self, tracker, studio) -> None:
        """Test that nudges are rate limited."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        # Get nudges up to the limit
        for _ in range(tracker._max_nudges_per_playbook + 1):
            tracker.get_nudge()

        # Should be rate limited now
        nudge = tracker.get_nudge()
        assert nudge is None


class TestPhaseGuidance:
    """Tests for phase guidance."""

    def test_phase_guidance_when_active(self, tracker, studio) -> None:
        """Test phase guidance when playbook active."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)

        guidance = tracker.get_phase_guidance()

        assert guidance is not None
        assert "topology_design" in guidance
        assert "Story Spark" in guidance

    def test_no_guidance_without_playbook(self, tracker) -> None:
        """Test no guidance without active playbook."""
        guidance = tracker.get_phase_guidance()
        assert guidance is None


class TestProgressSummary:
    """Tests for progress summary."""

    def test_summary_no_playbooks(self, tracker) -> None:
        """Test summary when no playbooks consulted."""
        summary = tracker.get_progress_summary()
        assert "No playbooks consulted" in summary

    def test_summary_with_playbook(self, tracker, studio) -> None:
        """Test summary with active playbook."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)
        tracker.on_artifact_created("section_brief")

        summary = tracker.get_progress_summary()

        assert "Story Spark" in summary
        assert "In Progress" in summary
        assert "topology_design" in summary

    def test_reset(self, tracker, studio) -> None:
        """Test tracker reset."""
        playbook = studio.playbooks["story_spark"]
        tracker.on_playbook_consulted("story_spark", playbook)
        tracker.on_artifact_created("section_brief")

        tracker.reset()

        assert tracker.consulted_playbooks == {}
        assert tracker.active_playbook_id is None
        assert tracker.produced_artifacts == set()
