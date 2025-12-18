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
