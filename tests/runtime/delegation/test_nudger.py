"""Tests for PlaybookNudger."""

import pytest

from questfoundry.runtime.delegation import (
    NudgeContext,
    PlaybookInstance,
    PlaybookNudger,
)
from questfoundry.runtime.messaging import MessageType

# Sample playbook for testing - phases and steps are dicts with IDs as keys
SAMPLE_PLAYBOOK = {
    "id": "test_playbook",
    "name": "Test Playbook",
    "phases": {
        "drafting": {
            "name": "Drafting Phase",
            "steps": {
                "write_draft": {
                    "outputs": [
                        {"artifact_type": "section_draft"},
                        {"artifact_type": "metadata"},
                    ],
                }
            },
            "quality_checkpoint": {
                "validator": "gatekeeper",
                "criteria": ["voice_consistency", "structure", "canon_compliance"],
            },
            "on_success": {"next_phases": ["review"]},
            "on_failure": {"next_phases": ["drafting"]},
            "is_rework_target": True,
        },
        "review": {
            "name": "Review Phase",
            "steps": {"validate": {"outputs": [{"artifact_type": "review_result"}]}},
            "on_success": {"next_phases": ["export"]},
            "on_failure": {"next_phases": ["drafting"]},
        },
        "export": {
            "name": "Export Phase",
            "steps": {"publish": {"outputs": []}},
        },
    },
}


class TestPlaybookNudger:
    """Tests for PlaybookNudger class."""

    @pytest.fixture
    def nudger(self):
        """Create a nudger with sample playbook."""
        return PlaybookNudger({"test_playbook": SAMPLE_PLAYBOOK})

    @pytest.fixture
    def ctx(self):
        """Create a sample nudge context."""
        return NudgeContext(
            playbook_id="test_playbook",
            instance_id="inst-001",
            phase_id="drafting",
            turn=5,
            agent_id="scene_smith",
        )

    def test_get_playbook(self, nudger):
        """Test getting a playbook by ID."""
        playbook = nudger.get_playbook("test_playbook")
        assert playbook is not None
        assert playbook["id"] == "test_playbook"

        missing = nudger.get_playbook("nonexistent")
        assert missing is None

    def test_check_phase_outputs_missing(self, nudger, ctx):
        """Test detecting missing outputs."""
        # No artifacts produced
        nudges = nudger.check_phase_outputs(ctx, artifacts_produced=[])

        # Should get nudges for both expected outputs
        assert len(nudges) == 2
        assert all(n.type == MessageType.NUDGE for n in nudges)
        assert all(n.payload["nudge_type"] == "missing_output" for n in nudges)

    def test_check_phase_outputs_partial(self, nudger, ctx):
        """Test detecting partially missing outputs."""
        # One artifact produced
        nudges = nudger.check_phase_outputs(
            ctx,
            artifacts_produced=["art-1"],
            artifact_types={"art-1": "section_draft"},
        )

        # Should only get nudge for missing metadata
        assert len(nudges) == 1
        assert "metadata" in nudges[0].payload["expected_output"]

    def test_check_phase_outputs_all_present(self, nudger, ctx):
        """Test no nudges when all outputs present."""
        nudges = nudger.check_phase_outputs(
            ctx,
            artifacts_produced=["art-1", "art-2"],
            artifact_types={"art-1": "section_draft", "art-2": "metadata"},
        )

        assert len(nudges) == 0

    def test_check_quality_checkpoint_exists(self, nudger, ctx):
        """Test detecting quality checkpoint."""
        nudge = nudger.check_quality_checkpoint(ctx)

        assert nudge is not None
        assert nudge.type == MessageType.NUDGE
        assert nudge.payload["nudge_type"] == "quality_gate_reminder"
        assert "gatekeeper" in nudge.payload["message"]
        assert "voice_consistency" in nudge.payload["message"]

    def test_check_quality_checkpoint_none(self, nudger, ctx):
        """Test no nudge when no quality checkpoint."""
        ctx.phase_id = "export"  # No quality checkpoint in export phase

        nudge = nudger.check_quality_checkpoint(ctx)
        assert nudge is None

    def test_check_rework_budget_healthy(self, nudger, ctx):
        """Test no warning when budget is healthy."""
        instance = PlaybookInstance(
            playbook_id="test_playbook",
            instance_id="inst-001",
            max_rework_cycles=3,
            rework_count=0,
        )

        nudge = nudger.check_rework_budget_warning(ctx, instance)
        assert nudge is None

    def test_check_rework_budget_low(self, nudger, ctx):
        """Test warning when budget is low."""
        instance = PlaybookInstance(
            playbook_id="test_playbook",
            instance_id="inst-001",
            max_rework_cycles=3,
            rework_count=2,  # Only 1 remaining
        )

        nudge = nudger.check_rework_budget_warning(ctx, instance)
        assert nudge is not None
        assert "Only 1 rework cycle remaining" in nudge.payload["message"]

    def test_check_rework_budget_exhausted(self, nudger, ctx):
        """Test critical warning when budget exhausted."""
        instance = PlaybookInstance(
            playbook_id="test_playbook",
            instance_id="inst-001",
            max_rework_cycles=3,
            rework_count=3,  # 0 remaining
        )

        nudge = nudger.check_rework_budget_warning(ctx, instance)
        assert nudge is not None
        assert "No rework cycles remaining" in nudge.payload["message"]

    def test_check_phase_consistency_valid(self, nudger, ctx):
        """Test no nudge for valid transitions."""
        ctx.phase_id = "review"
        previous_phase = "drafting"  # review is on_success of drafting

        nudge = nudger.check_phase_consistency(ctx, previous_phase)
        assert nudge is None

    def test_check_phase_consistency_invalid(self, nudger, ctx):
        """Test nudge for invalid transition."""
        ctx.phase_id = "export"
        previous_phase = "drafting"  # export is NOT a successor of drafting

        nudge = nudger.check_phase_consistency(ctx, previous_phase)
        assert nudge is not None
        assert nudge.payload["nudge_type"] == "unexpected_state"
        assert "not in expected successors" in nudge.payload["message"]

    def test_check_phase_consistency_first_phase(self, nudger, ctx):
        """Test no nudge for first phase (no previous)."""
        nudge = nudger.check_phase_consistency(ctx, previous_phase=None)
        assert nudge is None

    def test_generate_phase_entry_nudges(self, nudger, ctx):
        """Test generating all entry nudges at once."""
        instance = PlaybookInstance(
            playbook_id="test_playbook",
            instance_id="inst-001",
            max_rework_cycles=3,
            rework_count=2,  # Low budget
        )

        nudges = nudger.generate_phase_entry_nudges(ctx, instance)

        # Should have quality checkpoint and budget warning
        nudge_types = [n.payload["nudge_type"] for n in nudges]
        assert "quality_gate_reminder" in nudge_types
        assert "budget_warning" in nudge_types

    def test_unknown_playbook(self, nudger, ctx):
        """Test graceful handling of unknown playbook."""
        ctx.playbook_id = "nonexistent"

        nudges = nudger.check_phase_outputs(ctx, artifacts_produced=[])
        assert len(nudges) == 0

        quality = nudger.check_quality_checkpoint(ctx)
        assert quality is None


class TestPlaybookFollowupsNudge:
    """Tests for playbook followups nudging."""

    @pytest.fixture
    def nudger(self):
        """Create a nudger with sample playbook."""
        return PlaybookNudger({"test_playbook": SAMPLE_PLAYBOOK})

    @pytest.fixture
    def ctx(self):
        """Create a sample nudge context."""
        return NudgeContext(
            playbook_id="test_playbook",
            instance_id="inst-001",
            phase_id="drafting",
            turn=5,
            agent_id="showrunner",
        )

    def test_format_artifact_condition_basic(self, nudger):
        """Test formatting artifact condition with all fields."""
        cond = {
            "artifact_type": "section_brief",
            "state": "ready",
            "exists": True,
        }
        result = nudger._format_artifact_condition(cond)
        assert result == "section_brief in state 'ready' exists"

    def test_format_artifact_condition_not_exists(self, nudger):
        """Test formatting artifact condition with exists=False."""
        cond = {
            "artifact_type": "hook_card",
            "exists": False,
        }
        result = nudger._format_artifact_condition(cond)
        assert result == "hook_card does not exist"

    def test_format_artifact_condition_minimal(self, nudger):
        """Test formatting artifact condition with only type."""
        cond = {"artifact_type": "section"}
        result = nudger._format_artifact_condition(cond)
        assert result == "section exists"

    def test_check_playbook_followups_none(self, nudger, ctx):
        """Test no nudge when followups is None."""
        nudge = nudger.check_playbook_followups(ctx, None)
        assert nudge is None

    def test_check_playbook_followups_empty(self, nudger, ctx):
        """Test no nudge when followups is empty."""
        nudge = nudger.check_playbook_followups(ctx, {})
        assert nudge is None

    def test_check_playbook_followups_primary_only(self, nudger, ctx):
        """Test nudge with only primary followup."""
        followups = {
            "primary": {
                "playbook": "scene_weave",
                "description": "Write prose from briefs",
            }
        }
        nudge = nudger.check_playbook_followups(ctx, followups)

        assert nudge is not None
        assert nudge.type == MessageType.NUDGE
        assert nudge.payload["nudge_type"] == "playbook_followups"
        assert "Primary: scene_weave (Write prose from briefs)" in nudge.payload["message"]

    def test_check_playbook_followups_all_types(self, nudger, ctx):
        """Test nudge with all followup types."""
        followups = {
            "primary": {"playbook": "scene_weave"},
            "parallel": [{"playbook": "hook_harvest", "description": "Triage hooks"}],
            "conditional": [
                {"playbook": "lore_deepening", "condition": "canon gaps identified", "priority": 1},
                {"playbook": "story_spark", "condition": "structure changes needed", "priority": 2},
            ],
            "runtime_actions": [
                {"action": "commit_to_cold", "description": "Save approved sections"}
            ],
        }
        nudge = nudger.check_playbook_followups(ctx, followups)

        assert nudge is not None
        msg = nudge.payload["message"]
        assert "Primary: scene_weave" in msg
        assert "hook_harvest (Triage hooks)" in msg
        assert "lore_deepening (if canon gaps identified)" in msg
        assert "commit_to_cold (Save approved sections)" in msg

    def test_check_playbook_followups_conditional_sorted(self, nudger, ctx):
        """Test that conditional followups are sorted by priority."""
        followups = {
            "conditional": [
                {"playbook": "low_priority", "condition": "c1", "priority": 5},
                {"playbook": "high_priority", "condition": "c2", "priority": 1},
                {"playbook": "med_priority", "condition": "c3", "priority": 3},
            ],
        }
        nudge = nudger.check_playbook_followups(ctx, followups)

        msg = nudge.payload["message"]
        # high_priority should appear before med_priority before low_priority
        assert msg.index("high_priority") < msg.index("med_priority")
        assert msg.index("med_priority") < msg.index("low_priority")

    def test_check_playbook_followups_artifact_condition(self, nudger, ctx):
        """Test that artifact_condition is formatted when condition is absent."""
        followups = {
            "conditional": [
                {
                    "playbook": "scene_weave",
                    "artifact_condition": {
                        "artifact_type": "section_brief",
                        "state": "ready",
                    },
                },
            ],
        }
        nudge = nudger.check_playbook_followups(ctx, followups)

        msg = nudge.payload["message"]
        assert "section_brief in state 'ready' exists" in msg

    def test_generate_playbook_end_nudges_no_transition(self, nudger, ctx):
        """Test no nudges when transition is None."""
        nudges = nudger.generate_playbook_end_nudges(ctx, None)
        assert nudges == []

    def test_generate_playbook_end_nudges_not_ending(self, nudger, ctx):
        """Test no nudges when end_playbook is false."""
        transition = {"end_playbook": False, "followups": {"primary": {"playbook": "x"}}}
        nudges = nudger.generate_playbook_end_nudges(ctx, transition)
        assert nudges == []

    def test_generate_playbook_end_nudges_ending_with_followups(self, nudger, ctx):
        """Test nudge generated when playbook ends with followups."""
        transition = {
            "end_playbook": True,
            "followups": {"primary": {"playbook": "next_playbook"}},
        }
        nudges = nudger.generate_playbook_end_nudges(ctx, transition)

        assert len(nudges) == 1
        assert nudges[0].payload["nudge_type"] == "playbook_followups"

    def test_generate_playbook_end_nudges_ending_no_followups(self, nudger, ctx):
        """Test no nudge when playbook ends without followups."""
        transition = {"end_playbook": True}
        nudges = nudger.generate_playbook_end_nudges(ctx, transition)
        assert nudges == []


class TestNudgeContext:
    """Tests for NudgeContext dataclass."""

    def test_create_context(self):
        """Test creating a nudge context."""
        ctx = NudgeContext(
            playbook_id="playbook",
            instance_id="instance",
            phase_id="phase",
            turn=10,
            agent_id="agent",
        )

        assert ctx.playbook_id == "playbook"
        assert ctx.instance_id == "instance"
        assert ctx.phase_id == "phase"
        assert ctx.turn == 10
        assert ctx.agent_id == "agent"
