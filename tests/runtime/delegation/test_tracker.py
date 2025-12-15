"""Tests for PlaybookTracker."""

import pytest

from questfoundry.runtime.delegation import (
    PlaybookInstance,
    PlaybookTracker,
)
from questfoundry.runtime.messaging.types import PlaybookStatus


class TestPlaybookInstance:
    """Tests for PlaybookInstance dataclass."""

    def test_create_instance(self):
        """Test creating a playbook instance."""
        instance = PlaybookInstance(
            playbook_id="scene_weave",
            instance_id="inst-123",
            max_rework_cycles=3,
        )

        assert instance.playbook_id == "scene_weave"
        assert instance.instance_id == "inst-123"
        assert instance.max_rework_cycles == 3
        assert instance.rework_count == 0
        assert instance.current_phase is None
        assert instance.status == PlaybookStatus.ACTIVE

    def test_to_dict(self):
        """Test serialization to dictionary."""
        instance = PlaybookInstance(
            playbook_id="story_spark",
            instance_id="inst-456",
            max_rework_cycles=3,
            rework_count=1,
            current_phase="brief_creation",
        )
        instance.rework_target_visits["brief_creation"] = 2

        data = instance.to_dict()

        assert data["playbook_id"] == "story_spark"
        assert data["instance_id"] == "inst-456"
        assert data["max_rework_cycles"] == 3
        assert data["rework_count"] == 1
        assert data["current_phase"] == "brief_creation"
        assert data["rework_target_visits"]["brief_creation"] == 2
        assert data["status"] == "active"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "playbook_id": "scene_weave",
            "instance_id": "inst-789",
            "max_rework_cycles": 3,
            "rework_count": 2,
            "current_phase": "prose_drafting",
            "rework_target_visits": {"prose_drafting": 3},
            "started_at": "2025-01-15T10:00:00",
            "completed_at": None,
            "status": "active",
        }

        instance = PlaybookInstance.from_dict(data)

        assert instance.playbook_id == "scene_weave"
        assert instance.instance_id == "inst-789"
        assert instance.rework_count == 2
        assert instance.rework_target_visits["prose_drafting"] == 3


class TestPlaybookTracker:
    """Tests for PlaybookTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a test tracker."""
        return PlaybookTracker()

    @pytest.mark.asyncio
    async def test_start_playbook(self, tracker):
        """Test starting a playbook instance."""
        instance = await tracker.start_playbook(
            playbook_id="scene_weave",
            max_rework_cycles=3,
            initiating_agent="showrunner",
        )

        assert instance.playbook_id == "scene_weave"
        assert instance.max_rework_cycles == 3
        assert instance.initiating_agent == "showrunner"
        assert instance.status == PlaybookStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_instance(self, tracker):
        """Test retrieving a playbook instance."""
        instance = await tracker.start_playbook("scene_weave", 3)

        retrieved = await tracker.get_instance(instance.instance_id)

        assert retrieved is not None
        assert retrieved.instance_id == instance.instance_id

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self, tracker):
        """Test retrieving non-existent instance."""
        result = await tracker.get_instance("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_record_phase_entry_first_visit(self, tracker):
        """Test first visit to a phase (not counted as rework)."""
        instance = await tracker.start_playbook("scene_weave", 3)

        result = await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        assert result.allowed is True
        assert result.remaining == 3  # No rework yet

        # Check instance was updated
        instance = await tracker.get_instance(instance.instance_id)
        assert instance.current_phase == "prose_drafting"
        assert instance.rework_target_visits["prose_drafting"] == 1
        assert instance.rework_count == 0  # First visit doesn't count

    @pytest.mark.asyncio
    async def test_record_phase_entry_rework(self, tracker):
        """Test re-entry to rework target phase (counts as rework)."""
        instance = await tracker.start_playbook("scene_weave", 3)

        # First visit
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Second visit (rework)
        result = await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        assert result.allowed is True
        assert result.remaining == 2  # One rework used

        instance = await tracker.get_instance(instance.instance_id)
        assert instance.rework_count == 1
        assert instance.rework_target_visits["prose_drafting"] == 2

    @pytest.mark.asyncio
    async def test_record_phase_entry_non_rework_target(self, tracker):
        """Test entry to non-rework-target phase (never counts)."""
        instance = await tracker.start_playbook("scene_weave", 3)

        # Multiple visits to non-rework phase
        for _ in range(5):
            result = await tracker.record_phase_entry(
                instance.instance_id,
                "style_review",
                is_rework_target=False,
            )
            assert result.allowed is True
            assert result.remaining == 3  # Budget unchanged

        instance = await tracker.get_instance(instance.instance_id)
        assert instance.rework_count == 0

    @pytest.mark.asyncio
    async def test_rework_budget_exhaustion(self, tracker):
        """Test that budget exhaustion triggers escalation."""
        instance = await tracker.start_playbook("scene_weave", 2)

        # First visit (not counted)
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Second visit (rework 1)
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Third visit (rework 2)
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Fourth visit (exceeds budget)
        result = await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        assert result.allowed is False
        assert result.should_escalate is True
        assert result.reason == "max_rework_exceeded"
        assert result.escalation_details is not None
        assert result.escalation_details["rework_count"] == 3
        assert result.escalation_details["max_rework_cycles"] == 2

    @pytest.mark.asyncio
    async def test_check_rework_budget(self, tracker):
        """Test checking budget without recording phase."""
        instance = await tracker.start_playbook("scene_weave", 3)

        # Record some rework
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        result = await tracker.check_rework_budget(instance.instance_id)

        assert result.allowed is True
        assert result.remaining == 2

    @pytest.mark.asyncio
    async def test_complete_playbook(self, tracker):
        """Test completing a playbook instance."""
        instance = await tracker.start_playbook("scene_weave", 3)

        success = await tracker.complete_playbook(instance.instance_id)

        assert success is True

        instance = await tracker.get_instance(instance.instance_id)
        assert instance.status == PlaybookStatus.COMPLETED
        assert instance.completed_at is not None

    @pytest.mark.asyncio
    async def test_escalate_playbook(self, tracker):
        """Test escalating a playbook instance."""
        instance = await tracker.start_playbook("scene_weave", 3)

        success = await tracker.escalate_playbook(instance.instance_id)

        assert success is True

        instance = await tracker.get_instance(instance.instance_id)
        assert instance.status == PlaybookStatus.ESCALATED

    @pytest.mark.asyncio
    async def test_get_active_instances(self, tracker):
        """Test getting all active instances."""
        inst1 = await tracker.start_playbook("scene_weave", 3)
        inst2 = await tracker.start_playbook("scene_weave", 3)
        # Third instance for story_spark - needed for count but not referenced
        await tracker.start_playbook("story_spark", 3)

        # Complete one
        await tracker.complete_playbook(inst2.instance_id)

        # Get all active
        active = await tracker.get_active_instances()
        assert len(active) == 2

        # Get active for specific playbook
        scene_weave_active = await tracker.get_active_instances("scene_weave")
        assert len(scene_weave_active) == 1
        assert scene_weave_active[0].instance_id == inst1.instance_id

    @pytest.mark.asyncio
    async def test_get_stats(self, tracker):
        """Test getting tracker statistics."""
        await tracker.start_playbook("scene_weave", 3)
        inst2 = await tracker.start_playbook("scene_weave", 3)
        inst3 = await tracker.start_playbook("story_spark", 3)

        await tracker.complete_playbook(inst2.instance_id)
        await tracker.escalate_playbook(inst3.instance_id)

        stats = await tracker.get_stats()

        assert stats["total_instances"] == 3
        assert stats["active"] == 1
        assert stats["completed"] == 1
        assert stats["escalated"] == 1

    @pytest.mark.asyncio
    async def test_clear_completed(self, tracker):
        """Test clearing completed instances."""
        inst1 = await tracker.start_playbook("scene_weave", 3)
        inst2 = await tracker.start_playbook("scene_weave", 3)

        await tracker.complete_playbook(inst2.instance_id)

        removed = await tracker.clear_completed()

        assert removed == 1
        assert await tracker.get_instance(inst1.instance_id) is not None
        assert await tracker.get_instance(inst2.instance_id) is None

    @pytest.mark.asyncio
    async def test_multiple_rework_target_phases(self, tracker):
        """Test tracking multiple rework target phases independently."""
        instance = await tracker.start_playbook("scene_weave", 3)

        # Visit prose_drafting twice
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Visit brief_creation twice
        await tracker.record_phase_entry(
            instance.instance_id,
            "brief_creation",
            is_rework_target=True,
        )
        await tracker.record_phase_entry(
            instance.instance_id,
            "brief_creation",
            is_rework_target=True,
        )

        instance = await tracker.get_instance(instance.instance_id)

        # Both phases were visited twice (1 initial + 1 rework each)
        assert instance.rework_target_visits["prose_drafting"] == 2
        assert instance.rework_target_visits["brief_creation"] == 2
        # Total rework count is 2 (one rework per phase)
        assert instance.rework_count == 2
