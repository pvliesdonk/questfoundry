"""Tests for checkpoint data models."""

from datetime import datetime

from questfoundry.runtime.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    CheckpointConfig,
    CheckpointInfo,
    ContextUsage,
    DelegationSnapshot,
)
from questfoundry.runtime.session import SessionStatus


class TestContextUsage:
    """Tests for ContextUsage dataclass."""

    def test_initial_values(self):
        """Test default initial values."""
        usage = ContextUsage(agent_id="test_agent")
        assert usage.agent_id == "test_agent"
        assert usage.total_tokens == 0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.limit == 128000
        assert usage.warning_threshold == 100000

    def test_add_usage(self):
        """Test adding token usage."""
        usage = ContextUsage(agent_id="test_agent")
        usage.add_usage(input_tokens=100, output_tokens=50)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_cumulative_usage(self):
        """Test cumulative usage tracking."""
        usage = ContextUsage(agent_id="test_agent")
        usage.add_usage(input_tokens=100, output_tokens=50)
        usage.add_usage(input_tokens=200, output_tokens=100)

        assert usage.input_tokens == 300
        assert usage.output_tokens == 150
        assert usage.total_tokens == 450

    def test_remaining_tokens(self):
        """Test remaining tokens calculation."""
        usage = ContextUsage(agent_id="test_agent", limit=1000)
        usage.add_usage(input_tokens=300, output_tokens=200)

        assert usage.remaining == 500

    def test_remaining_not_negative(self):
        """Test remaining is never negative."""
        usage = ContextUsage(agent_id="test_agent", limit=100)
        usage.add_usage(input_tokens=150, output_tokens=50)

        assert usage.remaining == 0

    def test_usage_percent(self):
        """Test usage percentage calculation."""
        usage = ContextUsage(agent_id="test_agent", limit=1000)
        usage.add_usage(input_tokens=250, output_tokens=0)

        assert usage.usage_percent == 25.0

    def test_usage_percent_zero_limit(self):
        """Test usage percentage with zero limit."""
        usage = ContextUsage(agent_id="test_agent", limit=0)
        assert usage.usage_percent == 0.0

    def test_at_warning(self):
        """Test warning threshold detection."""
        usage = ContextUsage(
            agent_id="test_agent",
            limit=100,
            warning_threshold=80,
        )

        # Below warning
        usage.add_usage(input_tokens=70, output_tokens=0)
        assert not usage.at_warning

        # At warning
        usage.add_usage(input_tokens=10, output_tokens=0)
        assert usage.at_warning

    def test_at_limit(self):
        """Test limit detection."""
        usage = ContextUsage(agent_id="test_agent", limit=100)

        # Below limit
        usage.add_usage(input_tokens=90, output_tokens=0)
        assert not usage.at_limit

        # At limit
        usage.add_usage(input_tokens=10, output_tokens=0)
        assert usage.at_limit

    def test_serialization(self):
        """Test to_dict and from_dict."""
        usage = ContextUsage(
            agent_id="test_agent",
            limit=50000,
            warning_threshold=40000,
        )
        usage.add_usage(input_tokens=1000, output_tokens=500)

        data = usage.to_dict()
        restored = ContextUsage.from_dict(data)

        assert restored.agent_id == usage.agent_id
        assert restored.total_tokens == usage.total_tokens
        assert restored.input_tokens == usage.input_tokens
        assert restored.output_tokens == usage.output_tokens
        assert restored.limit == usage.limit
        assert restored.warning_threshold == usage.warning_threshold


class TestDelegationSnapshot:
    """Tests for DelegationSnapshot dataclass."""

    def test_basic_snapshot(self):
        """Test basic delegation snapshot."""
        snapshot = DelegationSnapshot(
            delegation_id="del_001",
            from_agent="showrunner",
            to_agent="scene_smith",
            status="pending",
        )

        assert snapshot.delegation_id == "del_001"
        assert snapshot.from_agent == "showrunner"
        assert snapshot.to_agent == "scene_smith"
        assert snapshot.status == "pending"
        assert snapshot.task is None
        assert snapshot.correlation_id is None

    def test_full_snapshot(self):
        """Test snapshot with all fields."""
        snapshot = DelegationSnapshot(
            delegation_id="del_002",
            from_agent="showrunner",
            to_agent="gatekeeper",
            status="active",
            task="Review scene draft",
            correlation_id="corr_123",
            created_at="2024-12-16T10:30:00",
        )

        assert snapshot.task == "Review scene draft"
        assert snapshot.correlation_id == "corr_123"
        assert snapshot.created_at == "2024-12-16T10:30:00"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        snapshot = DelegationSnapshot(
            delegation_id="del_003",
            from_agent="a",
            to_agent="b",
            status="completed",
            task="Do something",
        )

        data = snapshot.to_dict()
        restored = DelegationSnapshot.from_dict(data)

        assert restored.delegation_id == snapshot.delegation_id
        assert restored.from_agent == snapshot.from_agent
        assert restored.to_agent == snapshot.to_agent
        assert restored.status == snapshot.status
        assert restored.task == snapshot.task


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_minimal_checkpoint(self):
        """Test creating a minimal checkpoint."""
        checkpoint = Checkpoint(
            id="cp_001",
            session_id="session_123",
            turn_number=5,
        )

        assert checkpoint.id == "cp_001"
        assert checkpoint.session_id == "session_123"
        assert checkpoint.turn_number == 5
        assert checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION
        assert checkpoint.session_status == SessionStatus.ACTIVE
        assert checkpoint.entry_agent == "showrunner"
        assert checkpoint.mailbox_states == {}
        assert checkpoint.active_delegations == []
        assert checkpoint.playbook_instances == []
        assert checkpoint.context_usage == {}
        assert checkpoint.summary is None

    def test_full_checkpoint(self):
        """Test checkpoint with all fields populated."""
        context_usage = {
            "showrunner": ContextUsage(agent_id="showrunner"),
        }
        context_usage["showrunner"].add_usage(input_tokens=1000, output_tokens=500)

        delegations = [
            DelegationSnapshot(
                delegation_id="del_001",
                from_agent="showrunner",
                to_agent="scene_smith",
                status="pending",
            )
        ]

        checkpoint = Checkpoint(
            id="cp_turn_005",
            session_id="sess_abc",
            turn_number=5,
            created_at=datetime(2024, 12, 16, 10, 30, 0),
            session_status=SessionStatus.ACTIVE,
            entry_agent="showrunner",
            turn_count=5,
            mailbox_states={"showrunner": [{"id": "msg_001"}]},
            active_delegations=delegations,
            playbook_instances=[{"playbook_id": "story_dev", "status": "active"}],
            context_usage=context_usage,
            summary="Test checkpoint",
        )

        assert checkpoint.turn_count == 5
        assert len(checkpoint.mailbox_states) == 1
        assert len(checkpoint.active_delegations) == 1
        assert checkpoint.summary == "Test checkpoint"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        context_usage = {
            "agent1": ContextUsage(agent_id="agent1"),
        }
        context_usage["agent1"].add_usage(input_tokens=500, output_tokens=250)

        checkpoint = Checkpoint(
            id="cp_test",
            session_id="sess_test",
            turn_number=3,
            created_at=datetime(2024, 12, 16, 12, 0, 0),
            session_status=SessionStatus.ACTIVE,
            entry_agent="showrunner",
            turn_count=3,
            mailbox_states={"agent1": [{"id": "m1"}, {"id": "m2"}]},
            active_delegations=[
                DelegationSnapshot(
                    delegation_id="d1",
                    from_agent="a",
                    to_agent="b",
                    status="pending",
                )
            ],
            playbook_instances=[{"id": "pb1"}],
            context_usage=context_usage,
            summary="A test",
        )

        data = checkpoint.to_dict()

        # Verify schema marker
        assert data["$schema"] == f"checkpoint-v{CHECKPOINT_SCHEMA_VERSION}"

        # Restore
        restored = Checkpoint.from_dict(data)

        assert restored.id == checkpoint.id
        assert restored.session_id == checkpoint.session_id
        assert restored.turn_number == checkpoint.turn_number
        assert restored.created_at == checkpoint.created_at
        assert restored.session_status == checkpoint.session_status
        assert restored.entry_agent == checkpoint.entry_agent
        assert restored.turn_count == checkpoint.turn_count
        assert restored.mailbox_states == checkpoint.mailbox_states
        assert len(restored.active_delegations) == 1
        assert restored.active_delegations[0].delegation_id == "d1"
        assert restored.playbook_instances == checkpoint.playbook_instances
        assert "agent1" in restored.context_usage
        assert restored.context_usage["agent1"].total_tokens == 750
        assert restored.summary == checkpoint.summary


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_basic_info(self):
        """Test basic checkpoint info."""
        info = CheckpointInfo(
            id="cp_001",
            session_id="sess_123",
            turn_number=5,
            created_at=datetime(2024, 12, 16, 10, 0, 0),
        )

        assert info.id == "cp_001"
        assert info.session_id == "sess_123"
        assert info.turn_number == 5
        assert info.summary is None

    def test_serialization(self):
        """Test to_dict."""
        info = CheckpointInfo(
            id="cp_002",
            session_id="sess_456",
            turn_number=10,
            created_at=datetime(2024, 12, 16, 11, 30, 0),
            summary="After major plot point",
        )

        data = info.to_dict()

        assert data["id"] == "cp_002"
        assert data["session_id"] == "sess_456"
        assert data["turn_number"] == 10
        assert data["summary"] == "After major plot point"
        assert "created_at" in data


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = CheckpointConfig()

        assert config.auto_checkpoint is True
        assert config.checkpoint_frequency == 1
        assert config.max_checkpoints == 10
        assert config.checkpoint_on_error is True
        assert config.default_context_limit == 128000
        assert config.default_warning_threshold == 100000

    def test_custom_config(self):
        """Test custom configuration."""
        config = CheckpointConfig(
            auto_checkpoint=False,
            checkpoint_frequency=3,
            max_checkpoints=5,
            default_context_limit=64000,
        )

        assert config.auto_checkpoint is False
        assert config.checkpoint_frequency == 3
        assert config.max_checkpoints == 5
        assert config.default_context_limit == 64000

    def test_serialization(self):
        """Test to_dict and from_dict."""
        config = CheckpointConfig(
            auto_checkpoint=False,
            checkpoint_frequency=2,
            max_checkpoints=20,
        )

        data = config.to_dict()
        restored = CheckpointConfig.from_dict(data)

        assert restored.auto_checkpoint == config.auto_checkpoint
        assert restored.checkpoint_frequency == config.checkpoint_frequency
        assert restored.max_checkpoints == config.max_checkpoints
