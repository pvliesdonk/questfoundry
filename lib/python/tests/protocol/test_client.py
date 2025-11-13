"""Tests for ProtocolClient"""

import tempfile
import time
from pathlib import Path

import pytest

from questfoundry.protocol import Envelope, ProtocolClient


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def client(temp_workspace):
    """Create a ProtocolClient instance"""
    return ProtocolClient.from_workspace(temp_workspace, "SR")


@pytest.fixture
def receiver_client(temp_workspace):
    """Create a second client for receiving (different role)"""
    return ProtocolClient.from_workspace(temp_workspace, "GK")


def test_client_initialization(temp_workspace):
    """Test client can be initialized"""
    client = ProtocolClient.from_workspace(temp_workspace, "SR")
    assert client.sender_role == "SR"


def test_create_envelope(client):
    """Test creating envelope with defaults"""
    envelope = client.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="hook_card",
        payload_data={
            "title": "Test Hook",
            "question": "Test Question?",
            "stakes": "Test Stakes",
            "approaches": ["Approach 1"],
        },
    )

    assert envelope.sender.role == "SR"
    assert envelope.receiver.role == "GK"
    assert envelope.intent == "test.message"
    assert envelope.payload.type == "hook_card"
    assert envelope.payload.data["title"] == "Test Hook"
    assert envelope.context.hot_cold == "hot"
    assert envelope.safety.player_safe is True
    assert envelope.safety.spoilers == "forbidden"


def test_create_envelope_with_options(client):
    """Test creating envelope with custom options"""
    envelope = client.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="simple",  # Use any type, we'll disable validation
        payload_data={"key": "value"},
        hot_cold="cold",
        player_safe=False,
        spoilers="allowed",
        tu="TU-2024-01-15-TEST01",
        refs=["artifact-1", "artifact-2"],
    )

    assert envelope.context.hot_cold == "cold"
    assert envelope.context.tu == "TU-2024-01-15-TEST01"
    assert envelope.safety.player_safe is False
    assert envelope.safety.spoilers == "allowed"
    assert envelope.refs == ["artifact-1", "artifact-2"]


def test_send_envelope(client):
    """Test sending an envelope"""
    envelope = client.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="simple",
        payload_data={"test": "value"},
    )

    client.send(envelope, validate=False)

    # Verify envelope was sent to outbox
    outbox_files = list(client.transport.outbox_dir.glob("*.json"))
    assert len(outbox_files) == 1


def test_send_envelope_validation_error(client):
    """Test sending envelope with validation errors fails"""
    envelope = client.create_envelope(
        receiver="PN",  # Player Narrator
        intent="test.message",
        payload_type="hook_card",
        payload_data={
            "title": "Test Hook",
            "question": "Test Question?",
            "stakes": "Test Stakes",
            "approaches": ["Approach 1"],
        },
        hot_cold="hot",  # PN should only receive cold messages
    )

    with pytest.raises(ValueError, match="conformance validation failed"):
        client.send(envelope)


def test_send_without_validation(client):
    """Test sending envelope without validation"""
    envelope = client.create_envelope(
        receiver="PN",
        intent="test.message",
        payload_type="simple",
        payload_data={"test": "value"},
        hot_cold="hot",  # Would fail validation
    )

    # Should not raise when validate=False
    client.send(envelope, validate=False)


def test_receive_envelope(client, temp_workspace):
    """Test receiving an envelope"""
    # Send an envelope from one client
    envelope = client.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="simple",
        payload_data={"test": "value"},
    )
    client.send(envelope, validate=False)

    # Move from outbox to inbox for receiving
    for outbox_file in client.transport.outbox_dir.glob("*.json"):
        inbox_file = client.transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    # Receive the envelope
    received = list(client.receive(validate=False))
    assert len(received) == 1
    assert received[0].intent == "test.message"


def test_subscribe_to_intents(client):
    """Test subscribing to intent patterns"""
    received_envelopes = []

    def callback(envelope: Envelope) -> None:
        received_envelopes.append(envelope)

    # Subscribe to all test.* intents
    client.subscribe(r"test\..*", callback)

    # Create and send some envelopes
    envelope1 = client.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="simple",
        payload_data={"id": 1},
    )
    envelope2 = client.create_envelope(
        receiver="GK",
        intent="test.another",
        payload_type="simple",
        payload_data={"id": 2},
    )
    envelope3 = client.create_envelope(
        receiver="GK",
        intent="other.message",
        payload_type="simple",
        payload_data={"id": 3},
    )

    client.send(envelope1, validate=False)
    client.send(envelope2, validate=False)
    client.send(envelope3, validate=False)

    # Move all to inbox
    for outbox_file in client.transport.outbox_dir.glob("*.json"):
        inbox_file = client.transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    # Receive messages (triggers subscriptions)
    list(client.receive(validate=False))

    # Only test.* intents should trigger callback
    assert len(received_envelopes) == 2
    intents = {e.intent for e in received_envelopes}
    assert intents == {"test.message", "test.another"}


def test_unsubscribe_all(client):
    """Test unsubscribing from all patterns"""
    received_envelopes = []

    def callback(envelope: Envelope) -> None:
        received_envelopes.append(envelope)

    client.subscribe(r"test\..*", callback)
    client.unsubscribe_all()

    envelope = client.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="simple",
        payload_data={"test": "value"},
    )
    client.send(envelope, validate=False)

    # Move to inbox
    for outbox_file in client.transport.outbox_dir.glob("*.json"):
        inbox_file = client.transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    list(client.receive(validate=False))

    # Callback should not be triggered
    assert len(received_envelopes) == 0


def test_send_and_wait_timeout(client):
    """Test send_and_wait returns None on timeout"""
    envelope = client.create_envelope(
        receiver="GK",
        intent="test.request",
        payload_type="simple",
        payload_data={"test": "value"},
    )

    # Should timeout since no response is sent
    response = client.send_and_wait(envelope, timeout=0.2, validate=False)
    assert response is None


def test_send_and_wait_with_response(client, temp_workspace):
    """Test send_and_wait receives correlated response"""
    import threading

    # Create request envelope
    request = client.create_envelope(
        receiver="GK",
        intent="test.request",
        payload_type="simple",
        payload_data={"test": "request"},
    )

    def send_response():
        """Simulate another client sending a response"""
        time.sleep(0.1)  # Small delay

        # Create response client
        response_client = ProtocolClient.from_workspace(temp_workspace, "GK")

        # Move request from outbox to inbox so we can read it
        for outbox_file in response_client.transport.outbox_dir.glob("*.json"):
            inbox_file = response_client.transport.inbox_dir / outbox_file.name
            outbox_file.rename(inbox_file)

        # Read the request to get correlation_id
        requests = list(response_client.receive(validate=False))
        if requests:
            req = requests[0]
            # Send response with same correlation_id
            response = response_client.create_envelope(
                receiver="SR",
                intent="test.response",
                payload_type="simple",
                payload_data={"test": "response"},
                correlation_id=req.correlation_id,
            )
            response_client.send(response, validate=False)

            # Move response to inbox
            for outbox_file in response_client.transport.outbox_dir.glob("*.json"):
                inbox_file = response_client.transport.inbox_dir / outbox_file.name
                outbox_file.rename(inbox_file)

    # Start response thread
    thread = threading.Thread(target=send_response)
    thread.start()

    # Send and wait for response
    response = client.send_and_wait(request, timeout=2.0, validate=False)

    thread.join()

    assert response is not None
    assert response.intent == "test.response"
    assert response.payload.data == {"test": "response"}


def test_context_manager(temp_workspace):
    """Test client works as context manager"""
    with ProtocolClient.from_workspace(temp_workspace, "SR") as client:
        envelope = client.create_envelope(
            receiver="GK",
            intent="test.message",
            payload_type="simple",
            payload_data={"test": "value"},
        )
        client.send(envelope, validate=False)

    # Transport should be closed after exiting context


def test_create_envelope_with_correlation_and_reply(client):
    """Test creating envelope with correlation_id and reply_to"""
    envelope = client.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="simple",
        payload_data={"test": "value"},
        correlation_id="corr-123",
        reply_to="msg-456",
    )

    assert envelope.correlation_id == "corr-123"
    assert envelope.reply_to == "msg-456"


def test_multiple_clients_communication(temp_workspace):
    """Test multiple clients can communicate"""
    # Create sender and receiver clients
    sender = ProtocolClient.from_workspace(temp_workspace, "SR")
    receiver = ProtocolClient.from_workspace(temp_workspace, "GK")

    # Sender sends message
    envelope = sender.create_envelope(
        receiver="GK",
        intent="test.message",
        payload_type="simple",
        payload_data={"from": "sender"},
    )
    sender.send(envelope, validate=False)

    # Move from sender's outbox to receiver's inbox
    for outbox_file in sender.transport.outbox_dir.glob("*.json"):
        inbox_file = receiver.transport.inbox_dir / outbox_file.name
        outbox_file.rename(inbox_file)

    # Receiver receives message
    messages = list(receiver.receive(validate=False))
    assert len(messages) == 1
    assert messages[0].sender.role == "SR"
    assert messages[0].receiver.role == "GK"
