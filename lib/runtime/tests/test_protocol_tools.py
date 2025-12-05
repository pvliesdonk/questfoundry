from __future__ import annotations

import pytest

from unittest.mock import MagicMock, patch

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


def test_send_protocol_message_tu_open_writes_current_tu(base_state):
    tool = SendProtocolMessage()

    tu_brief = {
        "id": base_state["meta"]["current_tu"],
        "opened": "2025-12-05",
    }

    # Patch WriteHotSOT at the protocol_tools import site so we don't depend
    # on the full tu_brief schema for this unit test.
    with patch("questfoundry.runtime.tools.protocol_tools.WriteHotSOT") as MockWriter:
        writer_instance = MockWriter.return_value
        writer_instance._run.return_value = {
            "hot_sot": {"current_tu": tu_brief},
            "success": True,
        }

        update = tool._run(
            receiver="gatekeeper",
            intent="tu.open",
            payload={"tu_brief": tu_brief},
            role_id="showrunner",
            state=base_state,
        )

        # Ensure WriteHotSOT was called correctly
        writer_instance._run.assert_called_once()
        _, kwargs = writer_instance._run.call_args
        assert kwargs["key"] == "current_tu"
        assert kwargs["value"] == tu_brief

        # Tool should still return protocol message
        assert update["messages"][0]["intent"] == "tu.open"

        # And propagate a hot_sot update with current_tu set
        assert "hot_sot" in update
        assert update["hot_sot"]["current_tu"]["id"] == tu_brief["id"]
