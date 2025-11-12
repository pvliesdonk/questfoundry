"""Tests for protocol envelope models"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from questfoundry.protocol import (
    Context,
    Envelope,
    EnvelopeBuilder,
    HotCold,
    Payload,
    Protocol,
    Receiver,
    RoleName,
    Safety,
    Sender,
    SpoilerPolicy,
)


def test_protocol_model():
    """Test Protocol model"""
    protocol = Protocol(name="qf-protocol", version="1.0.0")

    assert protocol.name == "qf-protocol"
    assert protocol.version == "1.0.0"


def test_protocol_invalid_name():
    """Test Protocol rejects invalid names"""
    with pytest.raises(ValidationError):
        Protocol(name="invalid-protocol", version="1.0.0")


def test_sender_model():
    """Test Sender model"""
    sender = Sender(role=RoleName.SHOWRUNNER, agent="human:alice")

    assert sender.role == RoleName.SHOWRUNNER
    assert sender.agent == "human:alice"


def test_sender_without_agent():
    """Test Sender model without optional agent"""
    sender = Sender(role=RoleName.SCENE_SMITH)

    assert sender.role == RoleName.SCENE_SMITH
    assert sender.agent is None


def test_receiver_model():
    """Test Receiver model"""
    receiver = Receiver(role=RoleName.LORE_WEAVER)

    assert receiver.role == RoleName.LORE_WEAVER


def test_context_model():
    """Test Context model"""
    context = Context(
        hot_cold=HotCold.HOT, tu="TU-2025-10-30-SR01", loop="Hook Harvest"
    )

    assert context.hot_cold == HotCold.HOT
    assert context.tu == "TU-2025-10-30-SR01"
    assert context.loop == "Hook Harvest"


def test_context_cold_with_snapshot():
    """Test Context with cold snapshot"""
    context = Context(hot_cold=HotCold.COLD, snapshot="Cold @ 2025-10-28")

    assert context.hot_cold == HotCold.COLD
    assert context.snapshot == "Cold @ 2025-10-28"


def test_safety_model():
    """Test Safety model"""
    safety = Safety(player_safe=False, spoilers=SpoilerPolicy.ALLOWED)

    assert safety.player_safe is False
    assert safety.spoilers == SpoilerPolicy.ALLOWED


def test_payload_model():
    """Test Payload model"""
    payload = Payload(type="hook_card", data={"header": {"id": "HK-20251030-01"}})

    assert payload.type == "hook_card"
    assert payload.data["header"]["id"] == "HK-20251030-01"


def test_envelope_builder():
    """Test EnvelopeBuilder fluent API"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("urn:uuid:550e8400-e29b-41d4-a716-446655440000")
        .with_time(datetime(2025, 10, 30, 12, 19, 29, tzinfo=timezone.utc))
        .with_sender(RoleName.SHOWRUNNER, "human:alice")
        .with_receiver(RoleName.LORE_WEAVER)
        .with_intent("hook.create")
        .with_context(HotCold.HOT, tu="TU-2025-10-30-SR01", loop="Hook Harvest")
        .with_safety(False, SpoilerPolicy.ALLOWED)
        .with_payload("hook_card", {"header": {"id": "HK-20251030-01"}})
        .with_refs(["TU-2025-10-30-SR01"])
        .build()
    )

    assert envelope.protocol.version == "1.0.0"
    assert envelope.sender.role == RoleName.SHOWRUNNER
    assert envelope.receiver.role == RoleName.LORE_WEAVER
    assert envelope.intent == "hook.create"
    assert envelope.context.hot_cold == HotCold.HOT
    assert len(envelope.refs) == 1


def test_envelope_builder_missing_required():
    """Test EnvelopeBuilder raises error when required fields missing"""
    builder = EnvelopeBuilder().with_protocol("1.0.0")

    with pytest.raises(ValueError, match="Missing required fields"):
        builder.build()


def test_envelope_from_json():
    """Test parsing Envelope from JSON"""
    envelope_json = {
        "protocol": {"name": "qf-protocol", "version": "1.0.0"},
        "id": "urn:uuid:550e8400-e29b-41d4-a716-446655440000",
        "time": "2025-10-30T12:19:29Z",
        "sender": {"role": "SR", "agent": "human:alice"},
        "receiver": {"role": "LW"},
        "intent": "hook.create",
        "context": {
            "hot_cold": "hot",
            "tu": "TU-2025-10-30-SR01",
            "loop": "Hook Harvest",
        },
        "safety": {"player_safe": False, "spoilers": "allowed"},
        "payload": {
            "type": "hook_card",
            "data": {"header": {"id": "HK-20251030-01"}},
        },
        "refs": ["TU-2025-10-30-SR01"],
    }

    envelope = Envelope.model_validate(envelope_json)

    assert envelope.protocol.version == "1.0.0"
    assert envelope.sender.role == RoleName.SHOWRUNNER
    assert envelope.receiver.role == RoleName.LORE_WEAVER
    assert envelope.intent == "hook.create"


def test_envelope_to_json():
    """Test serializing Envelope to JSON"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-msg-001")
        .with_time(datetime(2025, 10, 30, 12, 19, 29, tzinfo=timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.SCENE_SMITH)
        .with_intent("scene.write")
        .with_context(HotCold.HOT, tu="TU-2025-10-30-SR01")
        .with_safety(False, SpoilerPolicy.ALLOWED)
        .with_payload("tu_brief", {"objective": "Test"})
        .build()
    )

    json_str = envelope.model_dump_json()
    data = json.loads(json_str)

    assert data["protocol"]["version"] == "1.0.0"
    assert data["sender"]["role"] == "SR"
    assert data["intent"] == "scene.write"


def test_parse_real_example_hook_create():
    """Test parsing real example from spec/04-protocol/EXAMPLES/"""
    spec_dir = Path(__file__).parent.parent.parent / "spec"
    example_file = spec_dir / "04-protocol" / "EXAMPLES" / "hook.create.json"

    if not example_file.exists():
        pytest.skip("Spec examples not available")

    with open(example_file) as f:
        envelope_data = json.load(f)

    envelope = Envelope.model_validate(envelope_data)

    assert envelope.protocol.version == "1.0.0"
    assert envelope.sender.role == RoleName.SHOWRUNNER
    assert envelope.receiver.role == RoleName.LORE_WEAVER
    assert envelope.intent == "hook.create"
    assert envelope.payload.type == "hook_card"
    assert envelope.context.hot_cold == HotCold.HOT


def test_envelope_validation_invalid_intent():
    """Test envelope validation rejects invalid intent format"""
    with pytest.raises(ValidationError):
        Envelope(
            protocol=Protocol(name="qf-protocol", version="1.0.0"),
            id="test-001",
            time=datetime.now(timezone.utc),
            sender=Sender(role=RoleName.SHOWRUNNER),
            receiver=Receiver(role=RoleName.SCENE_SMITH),
            intent="INVALID INTENT",  # Must be lowercase with dots/dashes
            context=Context(hot_cold=HotCold.HOT),
            safety=Safety(player_safe=False, spoilers=SpoilerPolicy.ALLOWED),
            payload=Payload(type="tu_brief", data={}),
        )


def test_envelope_with_correlation():
    """Test envelope with correlation_id and reply_to"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-msg-002")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.LORE_WEAVER)
        .with_receiver(RoleName.SHOWRUNNER)
        .with_intent("hook.update_status")
        .with_correlation_id("corr-hook-harvest-001")
        .with_reply_to("urn:uuid:550e8400-e29b-41d4-a716-446655440000")
        .with_context(HotCold.HOT, tu="TU-2025-10-30-SR01")
        .with_safety(False, SpoilerPolicy.ALLOWED)
        .with_payload("hook_card", {"status": "accepted"})
        .build()
    )

    assert envelope.correlation_id == "corr-hook-harvest-001"
    assert envelope.reply_to == "urn:uuid:550e8400-e29b-41d4-a716-446655440000"
