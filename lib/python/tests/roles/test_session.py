"""Tests for role session management."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from questfoundry.protocol.envelope import (
    Context,
    Envelope,
    Payload,
    Protocol,
    Receiver,
    Safety,
    Sender,
)
from questfoundry.roles.session import RoleSession


@pytest.fixture
def sample_envelope():
    """Create a sample envelope for testing."""
    return Envelope(
        protocol=Protocol(name="qf-protocol", version="0.2.1"),
        id="test-msg-001",
        time=datetime.now(timezone.utc),
        sender=Sender(role="SR"),  # Showrunner
        receiver=Receiver(role="SS"),  # Scene Smith
        intent="scene.write",
        context=Context(hot_cold="hot", tu="TU-2024-01-15-TEST01"),
        safety=Safety(player_safe=False, spoilers="allowed"),
        payload=Payload(type="tu_brief", data={"test": "data"}),
    )


@pytest.fixture
def session(tmp_path):
    """Create a test session."""
    return RoleSession(
        role="scene_smith",
        tu_context="TU-2024-01-15-TEST01",
        workspace_path=tmp_path,
    )


def test_session_creation(session):
    """Test creating a new session."""
    assert session.role == "scene_smith"
    assert session.tu_context == "TU-2024-01-15-TEST01"
    assert len(session.conversation_history) == 0
    assert len(session.dormancy_signals) == 0
    assert not session.should_dormant()
    assert isinstance(session.active_since, datetime)


def test_add_to_history(session, sample_envelope):
    """Test adding envelope to conversation history."""
    assert len(session.conversation_history) == 0

    session.add_to_history(sample_envelope)

    assert len(session.conversation_history) == 1
    assert session.conversation_history[0] == sample_envelope


def test_send_message(session, sample_envelope):
    """Test sending a message adds it to history."""
    session.send_message(sample_envelope)

    assert len(session.conversation_history) == 1
    assert session.conversation_history[0] == sample_envelope


def test_get_context_window_small(session, sample_envelope):
    """Test getting context window with few messages."""
    # Add 10 messages
    for i in range(10):
        env = sample_envelope.model_copy()
        env.id = f"test-msg-{i:03d}"
        session.add_to_history(env)

    # Request 50 messages - should get all 10
    context = session.get_context_window(max_messages=50)
    assert len(context) == 10


def test_get_context_window_large(session, sample_envelope):
    """Test getting context window with many messages."""
    # Add 100 messages
    for i in range(100):
        env = sample_envelope.model_copy()
        env.id = f"test-msg-{i:03d}"
        session.add_to_history(env)

    # Request 50 messages - should get most recent 50
    context = session.get_context_window(max_messages=50)
    assert len(context) == 50

    # Verify we got the most recent ones
    assert context[0].id == "test-msg-050"
    assert context[-1].id == "test-msg-099"


def test_add_dormancy_signal(session):
    """Test adding dormancy signals."""
    assert not session.should_dormant()

    session.add_dormancy_signal("task.complete")

    assert session.should_dormant()
    assert "task.complete" in session.dormancy_signals


def test_add_dormancy_signal_no_duplicates(session):
    """Test adding same signal twice doesn't duplicate."""
    session.add_dormancy_signal("task.complete")
    session.add_dormancy_signal("task.complete")

    assert len(session.dormancy_signals) == 1


def test_archive(session, sample_envelope):
    """Test archiving session state."""
    # Add some history and signals
    session.add_to_history(sample_envelope)
    session.add_dormancy_signal("task.complete")

    archive = session.archive()

    assert archive["role"] == "scene_smith"
    assert archive["tu_context"] == "TU-2024-01-15-TEST01"
    assert archive["message_count"] == 1
    assert "task.complete" in archive["dormancy_signals"]
    assert len(archive["conversation_history"]) == 1
    assert isinstance(archive["active_since"], str)
    assert isinstance(archive["active_duration_seconds"], float)


def test_save_to_file(session, sample_envelope, tmp_path):
    """Test saving session to file."""
    session.add_to_history(sample_envelope)
    session.add_dormancy_signal("task.complete")

    path = session.save_to_file()

    assert path.exists()
    assert path.parent.name == "scene_smith"
    assert path.parent.parent.name == "sessions"
    assert "session-" in path.name
    assert path.suffix == ".json"

    # Verify content
    with open(path) as f:
        data = json.load(f)

    assert data["role"] == "scene_smith"
    assert data["message_count"] == 1


def test_save_to_custom_path(session, tmp_path):
    """Test saving session to custom path."""
    custom_path = tmp_path / "custom-session.json"

    path = session.save_to_file(custom_path)

    assert path == custom_path
    assert path.exists()


def test_load_from_file(session, sample_envelope, tmp_path):
    """Test loading session from file."""
    # Create and save session
    session.add_to_history(sample_envelope)
    session.add_dormancy_signal("task.complete")
    saved_path = session.save_to_file()

    # Load it back
    loaded = RoleSession.load_from_file(saved_path)

    assert loaded.role == session.role
    assert loaded.tu_context == session.tu_context
    assert len(loaded.conversation_history) == 1
    assert "task.complete" in loaded.dormancy_signals
    assert loaded.should_dormant()


def test_load_from_nonexistent_file(tmp_path):
    """Test loading from nonexistent file raises error."""
    fake_path = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError):
        RoleSession.load_from_file(fake_path)


def test_session_repr(session):
    """Test string representation of session."""
    repr_str = repr(session)

    assert "RoleSession" in repr_str
    assert "scene_smith" in repr_str
    assert "messages=0" in repr_str
    assert "dormant=False" in repr_str


def test_session_with_multiple_messages(session, sample_envelope):
    """Test session with multiple messages from conversation."""
    # Simulate a conversation
    for i in range(5):
        env = sample_envelope.model_copy()
        env.id = f"msg-{i:03d}"
        session.send_message(env)

    assert len(session.conversation_history) == 5

    # Get context window
    context = session.get_context_window()
    assert len(context) == 5


def test_session_default_workspace_path():
    """Test session with default workspace path."""
    session = RoleSession(role="test_role")

    assert session.workspace_path == Path.cwd()
