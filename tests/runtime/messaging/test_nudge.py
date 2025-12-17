"""Tests for nudge message factory function."""

from questfoundry.runtime.messaging import (
    MessagePriority,
    MessageType,
    create_nudge,
)


class TestCreateNudge:
    """Tests for create_nudge factory function."""

    def test_create_basic_nudge(self):
        """Test creating a basic nudge message."""
        msg = create_nudge(
            from_agent="runtime",
            to_agent="scene_smith",
            nudge_type="missing_output",
            message="Expected section draft not produced",
        )

        assert msg.type == MessageType.NUDGE
        assert msg.from_agent == "runtime"
        assert msg.to_agent == "scene_smith"
        assert msg.payload["nudge_type"] == "missing_output"
        assert msg.payload["message"] == "Expected section draft not produced"
        assert msg.priority == MessagePriority.LOW

    def test_nudge_default_ttl(self):
        """Test that nudges have short TTL by default."""
        msg = create_nudge(
            from_agent="runtime",
            to_agent="agent",
            nudge_type="quality_gate_reminder",
            message="Quality checkpoint ahead",
        )

        assert msg.ttl_turns == 3  # Nudges expire quickly

    def test_nudge_with_playbook_context(self):
        """Test nudge with full playbook context."""
        msg = create_nudge(
            from_agent="runtime",
            to_agent="plotwright",
            nudge_type="unexpected_state",
            message="Phase transition unexpected",
            playbook_id="story_spark",
            playbook_instance_id="inst-123",
            phase_id="outline_creation",
            turn_created=10,
        )

        assert msg.playbook_id == "story_spark"
        assert msg.playbook_instance_id == "inst-123"
        assert msg.phase_id == "outline_creation"
        assert msg.turn_created == 10

    def test_nudge_with_expected_output(self):
        """Test nudge with expected output info."""
        msg = create_nudge(
            from_agent="runtime",
            to_agent="scene_smith",
            nudge_type="missing_output",
            message="Section draft expected",
            expected_output="section_draft",
        )

        assert msg.payload["expected_output"] == "section_draft"

    def test_nudge_with_current_state(self):
        """Test nudge with current state info."""
        msg = create_nudge(
            from_agent="runtime",
            to_agent="agent",
            nudge_type="unexpected_state",
            message="Unexpected phase",
            current_state="from=drafting, to=export",
        )

        assert msg.payload["current_state"] == "from=drafting, to=export"

    def test_nudge_custom_ttl(self):
        """Test nudge with custom TTL."""
        msg = create_nudge(
            from_agent="runtime",
            to_agent="agent",
            nudge_type="timeout_warning",
            message="Phase taking long",
            ttl_turns=10,
        )

        assert msg.ttl_turns == 10

    def test_all_nudge_types(self):
        """Test all defined nudge types can be created."""
        nudge_types = [
            "missing_output",
            "unexpected_state",
            "quality_gate_reminder",
            "timeout_warning",
            "consistency_concern",
        ]

        for nudge_type in nudge_types:
            msg = create_nudge(
                from_agent="runtime",
                to_agent="agent",
                nudge_type=nudge_type,
                message=f"Test {nudge_type}",
            )
            assert msg.payload["nudge_type"] == nudge_type
