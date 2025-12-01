from __future__ import annotations

import pytest

from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.tools.protocol_tools import SendProtocolEnvelope, SendProtocolMessage


@pytest.fixture()
def base_state() -> dict:
    manager = StateManager()
    return manager.initialize_state(loop_id="story_spark", context={"scene_text": "x"})


def test_send_protocol_message_adds_message(base_state):
    tool = SendProtocolMessage()

    update = tool._run(
        receiver="gatekeeper",
        intent="review.request",
        payload={"type": "none", "data": {"note": "please check"}},
        role_id="plotwright",
        state=base_state,
    )

    assert "messages" in update
    assert update["messages"][0]["receiver"] == {"role": "GK"}
    assert update["messages"][0]["intent"] == "review.request"


def test_send_protocol_envelope_validates_sender(base_state):
    tool = SendProtocolEnvelope()

    with pytest.raises(Exception):
        tool._run(envelope={"sender": "someone_else", "receiver": "gatekeeper", "intent": "review.request", "payload": {}}, role_id="plotwright", state=base_state)


def test_send_protocol_envelope_success(base_state):
    tool = SendProtocolEnvelope()

    update = tool._run(
        envelope={
            "receiver": "gatekeeper",
            "intent": "review.request",
            "payload": {"type": "none", "data": {"msg": "hi"}},
        },
        role_id="plotwright",
        state=base_state,
    )

    assert update["messages"][0]["envelope"]["sender"] == {"role": "PW"}
    assert update["messages"][0]["envelope"]["intent"] == "review.request"
