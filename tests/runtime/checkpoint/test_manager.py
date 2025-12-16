"""Tests for CheckpointManager."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.runtime.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointManager,
    ContextUsage,
)
from questfoundry.runtime.session import SessionStatus


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "checkpoints").mkdir()
    return project_dir


@pytest.fixture
def mock_project(temp_project_dir):
    """Create a mock project."""
    project = MagicMock()
    project.checkpoints_path = temp_project_dir / "checkpoints"
    return project


@pytest.fixture
def manager(mock_project):
    """Create a CheckpointManager instance."""
    return CheckpointManager(mock_project)


@pytest.fixture
def mock_session():
    """Create a mock session."""
    session = MagicMock()
    session.id = "session_123"
    session.turn_count = 5
    session.status = SessionStatus.ACTIVE
    session.entry_agent = "showrunner"
    return session


@pytest.fixture
def mock_broker():
    """Create a mock message broker."""
    broker = MagicMock()
    broker._mailboxes = {}
    return broker


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_initialization(self, mock_project):
        """Test manager initialization."""
        manager = CheckpointManager(mock_project)

        assert manager._project == mock_project
        assert manager._config.auto_checkpoint is True
        assert manager._checkpoints_dir == mock_project.checkpoints_path

    def test_initialization_with_config(self, mock_project):
        """Test manager initialization with custom config."""
        config = CheckpointConfig(
            auto_checkpoint=False,
            max_checkpoints=5,
        )
        manager = CheckpointManager(mock_project, config=config)

        assert manager._config.auto_checkpoint is False
        assert manager._config.max_checkpoints == 5

    def test_config_property(self, manager):
        """Test config property."""
        assert isinstance(manager.config, CheckpointConfig)

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, manager, mock_session, mock_broker):
        """Test creating a checkpoint."""
        checkpoint = await manager.create_checkpoint(
            session=mock_session,
            broker=mock_broker,
        )

        # Checkpoint ID includes session prefix (first 8 chars) and turn number
        # session_id="session_123" -> prefix="session_", turn=5 -> "cp_session__005"
        assert checkpoint.id == "cp_session__005"
        assert checkpoint.session_id == "session_123"
        assert checkpoint.turn_number == 5
        assert checkpoint.session_status == SessionStatus.ACTIVE

        # Verify file was created
        checkpoint_path = manager._checkpoints_dir / "cp_session__005.json"
        assert checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_create_checkpoint_custom_id(self, manager, mock_session, mock_broker):
        """Test creating a checkpoint with custom ID."""
        checkpoint = await manager.create_checkpoint(
            session=mock_session,
            broker=mock_broker,
            checkpoint_id="manual_save",
        )

        assert checkpoint.id == "manual_save"

        checkpoint_path = manager._checkpoints_dir / "manual_save.json"
        assert checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_create_checkpoint_with_summary(self, manager, mock_session, mock_broker):
        """Test creating a checkpoint with summary."""
        checkpoint = await manager.create_checkpoint(
            session=mock_session,
            broker=mock_broker,
            summary="After completing the first scene",
        )

        assert checkpoint.summary == "After completing the first scene"

    @pytest.mark.asyncio
    async def test_create_checkpoint_with_context_usage(self, manager, mock_session, mock_broker):
        """Test creating a checkpoint with context usage."""
        context_usage = {
            "showrunner": ContextUsage(agent_id="showrunner"),
        }
        context_usage["showrunner"].add_usage(input_tokens=1000, output_tokens=500)

        checkpoint = await manager.create_checkpoint(
            session=mock_session,
            broker=mock_broker,
            context_usage=context_usage,
        )

        assert "showrunner" in checkpoint.context_usage
        assert checkpoint.context_usage["showrunner"].total_tokens == 1500

    def test_load_checkpoint(self, manager):
        """Test loading a checkpoint."""
        # Create a checkpoint file
        checkpoint_data = {
            "$schema": "checkpoint-v1",
            "id": "cp_test",
            "session_id": "sess_test",
            "turn_number": 3,
            "created_at": "2024-12-16T10:00:00",
            "schema_version": 1,
            "session_status": "active",
            "entry_agent": "showrunner",
            "turn_count": 3,
            "mailbox_states": {},
            "active_delegations": [],
            "playbook_instances": [],
            "context_usage": {},
            "summary": "Test checkpoint",
        }

        checkpoint_path = manager._checkpoints_dir / "cp_test.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Load it
        checkpoint = manager.load_checkpoint("cp_test")

        assert checkpoint is not None
        assert checkpoint.id == "cp_test"
        assert checkpoint.session_id == "sess_test"
        assert checkpoint.turn_number == 3
        assert checkpoint.summary == "Test checkpoint"

    def test_load_checkpoint_not_found(self, manager):
        """Test loading a non-existent checkpoint."""
        checkpoint = manager.load_checkpoint("nonexistent")
        assert checkpoint is None

    def test_load_checkpoint_invalid_json(self, manager):
        """Test loading a checkpoint with invalid JSON."""
        checkpoint_path = manager._checkpoints_dir / "invalid.json"
        with open(checkpoint_path, "w") as f:
            f.write("not valid json")

        checkpoint = manager.load_checkpoint("invalid")
        assert checkpoint is None

    def test_list_checkpoints_empty(self, manager):
        """Test listing checkpoints when none exist."""
        checkpoints = manager.list_checkpoints()
        assert checkpoints == []

    def test_list_checkpoints(self, manager):
        """Test listing multiple checkpoints."""
        # Create some checkpoint files
        for i in [1, 3, 5]:
            checkpoint_data = {
                "id": f"cp_turn_{i:03d}",
                "session_id": "sess_test",
                "turn_number": i,
                "created_at": f"2024-12-16T10:{i:02d}:00",
                "schema_version": 1,
                "session_status": "active",
            }
            path = manager._checkpoints_dir / f"cp_turn_{i:03d}.json"
            with open(path, "w") as f:
                json.dump(checkpoint_data, f)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3
        # Should be sorted by turn number descending
        assert checkpoints[0].turn_number == 5
        assert checkpoints[1].turn_number == 3
        assert checkpoints[2].turn_number == 1

    def test_list_checkpoints_filter_by_session(self, manager):
        """Test listing checkpoints filtered by session."""
        # Create checkpoints for different sessions
        for sess_id, turn in [("sess_a", 1), ("sess_a", 2), ("sess_b", 3)]:
            checkpoint_data = {
                "id": f"cp_{sess_id}_{turn}",
                "session_id": sess_id,
                "turn_number": turn,
                "created_at": "2024-12-16T10:00:00",
                "schema_version": 1,
            }
            path = manager._checkpoints_dir / f"cp_{sess_id}_{turn}.json"
            with open(path, "w") as f:
                json.dump(checkpoint_data, f)

        # Filter by session
        checkpoints = manager.list_checkpoints(session_id="sess_a")

        assert len(checkpoints) == 2
        assert all(cp.session_id == "sess_a" for cp in checkpoints)

    def test_delete_checkpoint(self, manager):
        """Test deleting a checkpoint."""
        # Create a checkpoint file
        checkpoint_data = {"id": "cp_delete", "session_id": "sess", "turn_number": 1}
        path = manager._checkpoints_dir / "cp_delete.json"
        with open(path, "w") as f:
            json.dump(checkpoint_data, f)

        assert path.exists()

        # Delete it
        result = manager.delete_checkpoint("cp_delete")

        assert result is True
        assert not path.exists()

    def test_delete_checkpoint_not_found(self, manager):
        """Test deleting a non-existent checkpoint."""
        result = manager.delete_checkpoint("nonexistent")
        assert result is False

    def test_get_latest_checkpoint_none(self, manager):
        """Test getting latest when no checkpoints exist."""
        checkpoint = manager.get_latest_checkpoint()
        assert checkpoint is None

    def test_get_latest_checkpoint(self, manager):
        """Test getting the latest checkpoint."""
        # Create checkpoints
        for i in [1, 3, 5]:
            checkpoint_data = {
                "id": f"cp_turn_{i:03d}",
                "session_id": "sess_test",
                "turn_number": i,
                "created_at": f"2024-12-16T10:{i:02d}:00",
                "schema_version": 1,
                "session_status": "active",
                "entry_agent": "showrunner",
                "turn_count": i,
                "mailbox_states": {},
                "active_delegations": [],
                "playbook_instances": [],
                "context_usage": {},
            }
            path = manager._checkpoints_dir / f"cp_turn_{i:03d}.json"
            with open(path, "w") as f:
                json.dump(checkpoint_data, f)

        checkpoint = manager.get_latest_checkpoint()

        assert checkpoint is not None
        assert checkpoint.id == "cp_turn_005"
        assert checkpoint.turn_number == 5


class TestCheckpointRetention:
    """Tests for checkpoint retention policy."""

    def test_enforce_retention(self, mock_project, temp_project_dir):  # noqa: ARG002
        """Test that retention policy deletes old checkpoints."""
        config = CheckpointConfig(max_checkpoints=3)
        manager = CheckpointManager(mock_project, config=config)

        # Create 5 checkpoints
        for i in range(5):
            checkpoint_data = {
                "id": f"cp_{i}",
                "session_id": "sess",
                "turn_number": i,
                "created_at": f"2024-12-16T10:{i:02d}:00",
                "schema_version": 1,
            }
            path = manager._checkpoints_dir / f"cp_{i}.json"
            with open(path, "w") as f:
                json.dump(checkpoint_data, f)

        # Trigger retention
        manager._enforce_retention()

        # Should have 3 remaining
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Should keep the newest ones (highest turn numbers)
        turn_numbers = [cp.turn_number for cp in checkpoints]
        assert sorted(turn_numbers, reverse=True) == [4, 3, 2]

    def test_retention_unlimited(self, mock_project, temp_project_dir):  # noqa: ARG002
        """Test unlimited retention (max_checkpoints=0)."""
        config = CheckpointConfig(max_checkpoints=0)
        manager = CheckpointManager(mock_project, config=config)

        # Create 10 checkpoints
        for i in range(10):
            checkpoint_data = {
                "id": f"cp_{i}",
                "session_id": "sess",
                "turn_number": i,
                "created_at": f"2024-12-16T10:{i:02d}:00",
                "schema_version": 1,
            }
            path = manager._checkpoints_dir / f"cp_{i}.json"
            with open(path, "w") as f:
                json.dump(checkpoint_data, f)

        # Trigger retention (should do nothing)
        manager._enforce_retention()

        # All should remain
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 10


class TestCheckpointRestoration:
    """Tests for checkpoint restoration."""

    @pytest.mark.asyncio
    async def test_restore_empty_checkpoint(self, manager, mock_broker):
        """Test restoring from an empty checkpoint."""
        checkpoint = Checkpoint(
            id="cp_test",
            session_id="sess_test",
            turn_number=1,
        )

        result = await manager.restore_from_checkpoint(
            checkpoint=checkpoint,
            broker=mock_broker,
        )

        assert result["checkpoint_id"] == "cp_test"
        assert result["mailboxes_restored"] == 0
        assert result["messages_restored"] == 0
        assert result["playbooks_restored"] == 0

    @pytest.mark.asyncio
    async def test_restore_with_mailbox_messages(self, manager):
        """Test restoring mailbox messages."""
        # Create mock broker with get_mailbox
        mock_mailbox = AsyncMock()
        mock_mailbox.clear = AsyncMock()
        mock_mailbox.put = AsyncMock()

        mock_broker = MagicMock()
        mock_broker.get_mailbox = AsyncMock(return_value=mock_mailbox)

        # Create checkpoint with messages
        checkpoint = Checkpoint(
            id="cp_test",
            session_id="sess_test",
            turn_number=1,
            mailbox_states={
                "agent1": [
                    {
                        "id": "msg_001",
                        "type": "feedback",
                        "from_agent": "sender",
                        "to_agent": "agent1",
                        "timestamp": "2024-12-16T10:00:00",
                        "priority": 0,
                        "status": "pending",
                        "payload": {},
                    }
                ],
            },
        )

        result = await manager.restore_from_checkpoint(
            checkpoint=checkpoint,
            broker=mock_broker,
        )

        assert result["mailboxes_restored"] == 1
        assert result["messages_restored"] == 1
        mock_mailbox.clear.assert_called_once()
        mock_mailbox.put.assert_called_once()


class TestSchemaMigration:
    """Tests for checkpoint schema migration."""

    def test_migrate_current_version(self, manager):
        """Test migration of current schema version."""
        data = {"schema_version": 1}
        migrated = manager._migrate_checkpoint(data)
        assert migrated == data

    def test_migrate_old_version(self, manager):
        """Test migration logs warning for old versions."""
        data = {"schema_version": 0}

        with patch("questfoundry.runtime.checkpoint.manager.logger") as mock_logger:
            migrated = manager._migrate_checkpoint(data)

            # Should log warning
            mock_logger.warning.assert_called_once()
            # Data should be returned unchanged (future: apply migrations)
            assert migrated == data
