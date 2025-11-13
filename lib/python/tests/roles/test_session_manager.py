"""Tests for session manager."""

from pathlib import Path

import pytest

from questfoundry.roles.session import RoleSession
from questfoundry.roles.session_manager import SessionManager


@pytest.fixture
def manager(tmp_path):
    """Create a test session manager."""
    return SessionManager(workspace_path=tmp_path)


def test_manager_creation(manager, tmp_path):
    """Test creating a session manager."""
    assert manager.workspace_path == tmp_path
    assert len(manager.active_sessions) == 0
    assert manager.get_active_roles() == []


def test_wake_role(manager):
    """Test waking a role creates a new session."""
    session = manager.wake_role("scene_smith", "TU-2024-01-15-TEST01")

    assert isinstance(session, RoleSession)
    assert session.role == "scene_smith"
    assert session.tu_context == "TU-2024-01-15-TEST01"
    assert manager.is_role_awake("scene_smith")
    assert "scene_smith" in manager.get_active_roles()


def test_wake_role_already_awake(manager):
    """Test waking a role that's already awake returns existing session."""
    session1 = manager.wake_role("scene_smith", "TU-2024-01-15-TEST01")
    session2 = manager.wake_role("scene_smith", "TU-2024-01-15-TEST02")

    # Should be the same session
    assert session1 is session2
    # TU context should NOT change
    assert session1.tu_context == "TU-2024-01-15-TEST01"


def test_get_session(manager):
    """Test getting an active session."""
    manager.wake_role("scene_smith", "TU-2024-01-15-TEST01")

    session = manager.get_session("scene_smith")

    assert session is not None
    assert session.role == "scene_smith"


def test_get_session_not_awake(manager):
    """Test getting session for role that's not awake returns None."""
    session = manager.get_session("scene_smith")

    assert session is None


def test_dormant_role(manager, tmp_path):
    """Test making a role dormant."""
    session = manager.wake_role("scene_smith", "TU-2024-01-15-TEST01")
    session.add_dormancy_signal("task.complete")

    archive = manager.dormant_role("scene_smith")

    assert archive["role"] == "scene_smith"
    assert "task.complete" in archive["dormancy_signals"]
    assert not manager.is_role_awake("scene_smith")
    assert "scene_smith" not in manager.get_active_roles()

    # Verify session was saved to file
    sessions_dir = tmp_path / ".questfoundry" / "sessions" / "scene_smith"
    assert sessions_dir.exists()
    assert len(list(sessions_dir.glob("*.json"))) == 1


def test_dormant_role_not_awake(manager):
    """Test making a non-awake role dormant raises error."""
    with pytest.raises(KeyError):
        manager.dormant_role("scene_smith")


def test_get_active_roles(manager):
    """Test getting list of active roles."""
    manager.wake_role("scene_smith")
    manager.wake_role("showrunner")
    manager.wake_role("gatekeeper")

    active = manager.get_active_roles()

    assert len(active) == 3
    assert "scene_smith" in active
    assert "showrunner" in active
    assert "gatekeeper" in active


def test_is_role_awake(manager):
    """Test checking if role is awake."""
    assert not manager.is_role_awake("scene_smith")

    manager.wake_role("scene_smith")

    assert manager.is_role_awake("scene_smith")


def test_archive_all(manager):
    """Test archiving all sessions without making them dormant."""
    manager.wake_role("scene_smith", "TU-2024-01-15-TEST01")
    manager.wake_role("showrunner", "TU-2024-01-15-TEST01")

    archives = manager.archive_all()

    assert len(archives) == 2
    assert "scene_smith" in archives
    assert "showrunner" in archives
    # Roles should still be awake
    assert manager.is_role_awake("scene_smith")
    assert manager.is_role_awake("showrunner")


def test_dormant_all(manager):
    """Test making all roles dormant."""
    manager.wake_role("scene_smith", "TU-2024-01-15-TEST01")
    manager.wake_role("showrunner", "TU-2024-01-15-TEST01")

    archives = manager.dormant_all()

    assert len(archives) == 2
    assert "scene_smith" in archives
    assert "showrunner" in archives
    # All roles should now be dormant
    assert not manager.is_role_awake("scene_smith")
    assert not manager.is_role_awake("showrunner")
    assert manager.get_active_roles() == []


def test_get_total_message_count(manager):
    """Test getting total message count across all sessions."""
    from datetime import datetime, timezone

    from questfoundry.protocol.envelope import (
        Context,
        Envelope,
        Payload,
        Protocol,
        Receiver,
        Safety,
        Sender,
    )

    session1 = manager.wake_role("scene_smith")
    session2 = manager.wake_role("showrunner")

    # Add messages to both sessions
    env = Envelope(
        protocol=Protocol(name="qf-protocol", version="0.2.1"),
        id="test-msg-001",
        time=datetime.now(timezone.utc),
        sender=Sender(role="SR"),
        receiver=Receiver(role="SS"),
        intent="test",
        context=Context(hot_cold="hot"),
        safety=Safety(player_safe=True, spoilers="forbidden"),
        payload=Payload(type="test", data={}),
    )

    session1.add_to_history(env)
    session1.add_to_history(env)
    session2.add_to_history(env)

    assert manager.get_total_message_count() == 3


def test_clear_dormancy_signals(manager):
    """Test clearing dormancy signals for a role."""
    session = manager.wake_role("scene_smith")
    session.add_dormancy_signal("task.complete")

    assert session.should_dormant()

    manager.clear_dormancy_signals("scene_smith")

    assert not session.should_dormant()


def test_clear_dormancy_signals_not_awake(manager):
    """Test clearing signals for non-awake role raises error."""
    with pytest.raises(KeyError):
        manager.clear_dormancy_signals("scene_smith")


def test_get_sessions_needing_dormancy(manager):
    """Test getting list of sessions that need dormancy."""
    session1 = manager.wake_role("scene_smith")
    manager.wake_role("showrunner")  # No dormancy signals
    session3 = manager.wake_role("gatekeeper")

    # Add dormancy signals to some sessions
    session1.add_dormancy_signal("task.complete")
    session3.add_dormancy_signal("error.fatal")

    needing_dormancy = manager.get_sessions_needing_dormancy()

    assert len(needing_dormancy) == 2
    assert "scene_smith" in needing_dormancy
    assert "gatekeeper" in needing_dormancy
    assert "showrunner" not in needing_dormancy


def test_manager_repr(manager):
    """Test string representation of session manager."""
    manager.wake_role("scene_smith")
    manager.wake_role("showrunner")

    repr_str = repr(manager)

    assert "SessionManager" in repr_str
    assert "active_roles=2" in repr_str
    assert "total_messages=0" in repr_str


def test_manager_default_workspace():
    """Test manager with default workspace path."""
    manager = SessionManager()

    assert manager.workspace_path == Path.cwd()


def test_wake_dormant_cycle(manager, tmp_path):
    """Test full wake -> dormant -> wake cycle."""
    # Wake role
    session1 = manager.wake_role("scene_smith", "TU-2024-01-15-TEST01")
    session1.add_dormancy_signal("task.complete")

    # Make dormant
    manager.dormant_role("scene_smith")
    assert not manager.is_role_awake("scene_smith")

    # Wake again - should create new session
    session2 = manager.wake_role("scene_smith", "TU-2024-01-15-TEST02")

    assert session2 is not session1
    assert session2.tu_context == "TU-2024-01-15-TEST02"
    assert len(session2.dormancy_signals) == 0
