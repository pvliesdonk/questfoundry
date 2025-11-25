from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.exceptions import StateError
from questfoundry.runtime.models.state import StudioState
from questfoundry.runtime.protocol import Protocol
from questfoundry.runtime.protocol.types import Envelope

logger = logging.getLogger(__name__)


class _BaseProtocolTool(BaseTool):
    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}
    _protocol: Protocol = PrivateAttr()
    _state_manager: StateManager = PrivateAttr()

    def __init__(self, *, protocol: Protocol | None = None, state_manager: StateManager | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state_manager = state_manager or StateManager()
        self._protocol = protocol or Protocol(self._state_manager)


class SendProtocolMessage(_BaseProtocolTool):
    name: str = "send_protocol_message"
    description: str = "Send a TU Protocol message to a recipient"

    def _run(
        self,
        recipient: str,
        intent: str,
        payload: dict[str, Any],
        role_id: str,
        state: StudioState,
    ) -> dict[str, Any]:  # type: ignore[override]
        if not state:
            raise StateError("State payload is required")
        if role_id != state.get("current_node", role_id):
            logger.debug("role_id %s does not match current_node %s", role_id, state.get("current_node"))
        message = self._protocol.send_message(state, sender=role_id, recipient=recipient, intent=intent, payload=payload)
        return {"messages": [message]}


class SendProtocolEnvelope(_BaseProtocolTool):
    name: str = "send_protocol_envelope"
    description: str = "Send a full TU Protocol envelope"

    def _run(
        self,
        envelope: Envelope,
        role_id: str,
        state: StudioState,
    ) -> dict[str, Any]:  # type: ignore[override]
        if not state:
            raise StateError("State payload is required")
        if envelope.get("sender") and envelope.get("sender") != role_id:
            raise StateError("Envelope sender must match current role")
        envelope = {**envelope, "sender": envelope.get("sender", role_id)}
        message = self._protocol.send_envelope(state, envelope)
        return {"messages": [message]}
