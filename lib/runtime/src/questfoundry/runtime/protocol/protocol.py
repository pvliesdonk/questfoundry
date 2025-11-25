from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import jsonschema
from importlib import resources

from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.exceptions import StateError
from questfoundry.runtime.models.state import Message, StudioState
from questfoundry.runtime.protocol.types import Envelope

logger = logging.getLogger(__name__)


class Protocol:
    """Minimal TU protocol handler for message/envelope dispatch."""

    def __init__(self, state_manager: StateManager | None = None) -> None:
        self._state_manager = state_manager or StateManager()
        self._envelope_schema = self._load_envelope_schema()

    def send_message(
        self,
        state: StudioState,
        sender: str,
        recipient: str | list[str],
        intent: str,
        payload: dict[str, Any],
    ) -> Message:
        message_id = self._generate_message_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        envelope: Envelope = {
            "protocol": {"name": "qf-protocol", "version": "1.0.0"},
            "id": message_id,
            "time": timestamp,
            "sender": sender,
            "receiver": recipient,
            "intent": intent,
            "payload": payload,
            "tu_id": state["tu_id"],
            "snapshot_ref": state.get("snapshot_ref"),
            "loop_id": state.get("loop_id"),
            "message_id": message_id,
            "transport": "inproc",
            "context": {
                "hot_cold": "hot",
                "tu": state.get("tu_id"),
                "loop": state.get("loop_id"),
            },
            "safety": {"player_safe": True, "spoilers": "allowed"},
        }
        return self._dispatch(state, envelope)

    def send_envelope(self, state: StudioState, envelope: Envelope) -> Message:
        envelope = {**envelope}
        message_id = self._generate_message_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        envelope.setdefault("protocol", {"name": "qf-protocol", "version": "1.0.0"})
        envelope.setdefault("id", message_id)
        envelope.setdefault("time", timestamp)
        envelope.setdefault("tu_id", state["tu_id"])
        envelope.setdefault("snapshot_ref", state.get("snapshot_ref"))
        envelope.setdefault("loop_id", state.get("loop_id"))
        envelope.setdefault("transport", "inproc")
        envelope.setdefault("message_id", message_id)
        envelope.setdefault(
            "context",
            {"hot_cold": "hot", "tu": state.get("tu_id"), "loop": state.get("loop_id")},
        )
        envelope.setdefault("safety", {"player_safe": True, "spoilers": "allowed"})
        self._validate_envelope(envelope)
        return self._dispatch(state, envelope)

    def _dispatch(self, state: StudioState, envelope: Envelope) -> Message:
        sender = envelope.get("sender") or "system"
        receiver = envelope.get("receiver", "broadcast")
        timestamp = datetime.now(timezone.utc).isoformat()
        message: Message = {
            "sender": sender,
            "receiver": receiver if not isinstance(receiver, list) else "broadcast",
            "intent": envelope.get("intent", ""),
            "payload": envelope.get("payload", {}),
            "timestamp": timestamp,
            "envelope": {k: v for k, v in envelope.items() if k not in {"payload"}},
        }
        return self._state_manager.add_message(state, message)["messages"][-1]

    @staticmethod
    def _generate_message_id() -> str:
        now = datetime.now(timezone.utc)
        return f"MSG-{now.strftime('%Y%m%dT%H%M%S%fZ')}"

    def _load_envelope_schema(self) -> dict[str, Any] | None:
        try:
            with resources.files("questfoundry.runtime.resources.protocol").joinpath(
                "envelope.schema.json"
            ).open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover - schema optional fallback
            logger.debug("Envelope schema load failed: %s", exc)
            return None

    def _validate_envelope(self, envelope: Envelope) -> None:
        if not self._envelope_schema:
            return
        try:
            jsonschema.validate(envelope, self._envelope_schema)
        except jsonschema.ValidationError as exc:
            logger.debug("Envelope validation skipped: %s", exc)
