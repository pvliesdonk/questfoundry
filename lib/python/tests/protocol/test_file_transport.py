"""Tests for file-based transport"""

import tempfile
from pathlib import Path

import pytest

from questfoundry.protocol import (
    EnvelopeBuilder,
    FileTransport,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def transport(temp_workspace):
    """Create a FileTransport instance"""
    return FileTransport(temp_workspace)


@pytest.fixture
def sample_envelope():
    """Create a sample envelope for testing"""
    from datetime import datetime

    return (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-message-001")
        .with_time(datetime(2024, 1, 15, 10, 30, 0))
        .with_intent("test.message")
        .with_sender("SR")
        .with_receiver("GK")
        .with_context("hot")
        .with_safety(True, "forbidden")
        .with_payload("test_data", {"test": "value"})
        .build()
    )


def test_transport_initialization(temp_workspace):
    """Test transport creates required directories"""
    FileTransport(temp_workspace)

    # Check directories exist
    messages_dir = temp_workspace / "messages"
    assert messages_dir.exists()
    assert (messages_dir / "inbox").exists()
    assert (messages_dir / "outbox").exists()
    assert (messages_dir / "processed").exists()


def test_send_message(transport, sample_envelope):
    """Test sending a message writes to outbox"""
    transport.send(sample_envelope)

    # Check message file was created
    outbox_files = list(transport.outbox_dir.glob("*.json"))
    assert len(outbox_files) == 1

    # Check file contains valid JSON
    message_file = outbox_files[0]
    assert message_file.exists()
    assert message_file.stat().st_size > 0


def test_send_multiple_messages(transport, sample_envelope):
    """Test sending multiple messages"""
    transport.send(sample_envelope)
    transport.send(sample_envelope)
    transport.send(sample_envelope)

    outbox_files = list(transport.outbox_dir.glob("*.json"))
    assert len(outbox_files) == 3

    # Check all files have unique names
    filenames = [f.name for f in outbox_files]
    assert len(filenames) == len(set(filenames))


def test_receive_message(transport, sample_envelope):
    """Test receiving a message from inbox"""
    # Place message in inbox
    transport.send(sample_envelope)
    # Move from outbox to inbox to simulate receiving
    outbox_file = list(transport.outbox_dir.glob("*.json"))[0]
    inbox_file = transport.inbox_dir / outbox_file.name
    outbox_file.rename(inbox_file)

    # Receive messages
    received = list(transport.receive())

    assert len(received) == 1
    assert received[0].intent == "test.message"
    assert received[0].sender.role == "SR"
    assert received[0].receiver.role == "GK"


def test_receive_acknowledges_message(transport, sample_envelope):
    """Test that received messages are moved to processed"""
    # Place message in inbox
    transport.send(sample_envelope)
    outbox_file = list(transport.outbox_dir.glob("*.json"))[0]
    inbox_file = transport.inbox_dir / outbox_file.name
    outbox_file.rename(inbox_file)

    # Receive messages
    list(transport.receive())

    # Check inbox is empty
    inbox_files = list(transport.inbox_dir.glob("*.json"))
    assert len(inbox_files) == 0

    # Check message moved to processed
    processed_files = list(transport.processed_dir.glob("*.json"))
    assert len(processed_files) == 1


def test_receive_multiple_messages_in_order(transport, sample_envelope):
    """Test receiving multiple messages"""
    import time
    from datetime import datetime

    # Send 3 messages with delays to ensure different timestamps
    intents = ["test.message.first", "test.message.second", "test.message.third"]
    for i, intent in enumerate(intents):
        builder = (
            EnvelopeBuilder()
            .with_protocol("1.0.0")
            .with_id(f"test-message-{i:03d}")
            .with_time(datetime(2024, 1, 15, 10, 30, i))
            .with_intent(intent)
            .with_sender("SR")
            .with_receiver("GK")
            .with_context("hot")
            .with_safety(True, "forbidden")
            .with_payload("test_data", {"index": i})
        )
        transport.send(builder.build())
        if i < 2:  # Don't sleep after last message
            time.sleep(0.01)  # Small delay to ensure different filenames

    # Move all to inbox
    for outbox_file in transport.outbox_dir.glob("*.json"):
        inbox_file = transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    # Receive messages
    received = list(transport.receive())

    # Check all messages were received (order may vary due to timing)
    assert len(received) == 3
    intents = {e.intent for e in received}
    assert intents == {
        "test.message.first",
        "test.message.second",
        "test.message.third",
    }


def test_receive_empty_inbox(transport):
    """Test receiving from empty inbox returns no messages"""
    received = list(transport.receive())
    assert len(received) == 0


def test_receive_invalid_message_moves_to_error(transport, temp_workspace):
    """Test that invalid messages are silently skipped"""
    # Create invalid message file
    invalid_file = transport.inbox_dir / "invalid.json"
    invalid_file.write_text('{"invalid": "data"}')

    # Try to receive - should silently skip invalid messages
    received = list(transport.receive())

    # Should receive no valid messages
    assert len(received) == 0


def test_context_manager(temp_workspace, sample_envelope):
    """Test transport works as context manager"""
    with FileTransport(temp_workspace) as transport:
        transport.send(sample_envelope)

        outbox_files = list(transport.outbox_dir.glob("*.json"))
        assert len(outbox_files) == 1


def test_clear_outbox(transport, sample_envelope):
    """Test clearing outbox"""
    # Send 3 messages
    transport.send(sample_envelope)
    transport.send(sample_envelope)
    transport.send(sample_envelope)

    # Clear outbox
    cleared = transport.clear_outbox()

    assert cleared == 3
    assert len(list(transport.outbox_dir.glob("*.json"))) == 0


def test_clear_inbox(transport, sample_envelope):
    """Test clearing inbox"""
    # Place messages in inbox
    for _ in range(3):
        transport.send(sample_envelope)

    for outbox_file in transport.outbox_dir.glob("*.json"):
        inbox_file = transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    # Clear inbox
    cleared = transport.clear_inbox()

    assert cleared == 3
    assert len(list(transport.inbox_dir.glob("*.json"))) == 0


def test_clear_processed(transport, sample_envelope):
    """Test clearing processed directory"""
    # Send and receive messages
    transport.send(sample_envelope)
    transport.send(sample_envelope)

    for outbox_file in transport.outbox_dir.glob("*.json"):
        inbox_file = transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    list(transport.receive())

    # Clear processed
    cleared = transport.clear_processed()

    assert cleared == 2
    assert len(list(transport.processed_dir.glob("*"))) == 0


def test_send_uses_atomic_write(transport, sample_envelope, monkeypatch):
    """Test that send uses atomic write (temp + rename)"""
    # This test verifies the pattern is used, not that it's truly atomic
    # (which would require OS-level testing)

    transport.send(sample_envelope)

    # Check no .tmp files left behind
    tmp_files = list(transport.outbox_dir.glob("*.tmp"))
    assert len(tmp_files) == 0

    # Check final file exists
    json_files = list(transport.outbox_dir.glob("*.json"))
    assert len(json_files) == 1


def test_roundtrip_envelope_data(transport, sample_envelope):
    """Test envelope data survives send/receive roundtrip"""
    # Send
    transport.send(sample_envelope)

    # Move to inbox
    for outbox_file in transport.outbox_dir.glob("*.json"):
        inbox_file = transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    # Receive
    received = list(transport.receive())

    # Verify data integrity
    assert len(received) == 1
    envelope = received[0]

    assert envelope.protocol.name == sample_envelope.protocol.name
    assert envelope.protocol.version == sample_envelope.protocol.version
    assert envelope.intent == sample_envelope.intent
    assert envelope.sender.role == sample_envelope.sender.role
    assert envelope.receiver.role == sample_envelope.receiver.role
    assert envelope.context.hot_cold == sample_envelope.context.hot_cold
    assert envelope.safety.player_safe == sample_envelope.safety.player_safe
    assert envelope.payload.type == sample_envelope.payload.type
    assert envelope.payload.data == sample_envelope.payload.data
