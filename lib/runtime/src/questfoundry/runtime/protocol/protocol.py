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

ROLE_ABBREVIATIONS = {
    "showrunner": "SR",
    "gatekeeper": "GK",
    "plotwright": "PW",
    "scene_smith": "SS",
    "style_lead": "ST",
    "lore_weaver": "LW",
    "codex_curator": "CC",
    "art_director": "AD",
    "illustrator": "IL",
    "audio_director": "AuD",
    "audio_producer": "AuP",
    "translator": "TR",
    "book_binder": "BB",
    "player_narrator": "PN",
    "researcher": "RS",
}

LOOP_DISPLAY = {
    "story_spark": "Story Spark",
    "hook_harvest": "Hook Harvest",
    "lore_deepening": "Lore Deepening",
    "codex_expansion": "Codex Expansion",
    "style_tune_up": "Style Tune-up",
    "art_touch_up": "Art Touch-up",
    "audio_pass": "Audio Pass",
    "translation_pass": "Translation Pass",
    "binding_run": "Binding Run",
    "narration_dry_run": "Narration Dry-Run",
    "gatecheck": "Gatecheck",
    "post_mortem": "Post-Mortem",
    "archive_snapshot": "Archive Snapshot",
}


def _to_abbrev(role_id: str) -> str:
    return ROLE_ABBREVIATIONS.get(role_id, role_id.upper() if len(role_id) <= 4 else "*")


def _wrap_sender(role_id: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(role_id, dict):
        if "role" in role_id:
            return {"role": _to_abbrev(str(role_id["role"]))}
        return role_id
    return {"role": _to_abbrev(role_id)}


def _wrap_receiver(val: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(val, dict):
        if "role" in val:
            return {"role": _to_abbrev(str(val["role"]))}
        return val
    return {"role": _to_abbrev(val)}


def _wrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "type" not in payload:
        return {"type": "none", "data": payload}
    return payload


def _normalize_tu(tu_id: str | None, fallback_role: str) -> str:
    import re

    pattern = re.compile(r"^TU-\\d{4}-\\d{2}-\\d{2}-[A-Z]{2,4}\\d{2}$")
    if tu_id and pattern.match(tu_id):
        return tu_id
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    role = _to_abbrev(fallback_role)
    return f"TU-{today}-{role}01"


class Protocol:
    """Minimal TU protocol handler for message/envelope dispatch."""

    def __init__(self, state_manager: StateManager | None = None) -> None:
        self._state_manager = state_manager or StateManager()
        self._envelope_schema = self._load_envelope_schema()

    def send_message(
        self,
        state: StudioState,
        sender: str,
        receiver: str | list[str],
        intent: str,
        payload: dict[str, Any],
    ) -> Message:
        message_id = self._generate_message_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        recv_val = _wrap_receiver(receiver)
        envelope: Envelope = {
            "protocol": {"name": "qf-protocol", "version": "1.0.0"},
            "id": message_id,
            "time": timestamp,
            "sender": _wrap_sender(sender),
            "receiver": recv_val,
            "intent": intent,
            "payload": _wrap_payload(payload),
            "tu_id": _normalize_tu(state.get("tu_id"), sender),
            "snapshot_ref": state.get("snapshot_ref"),
            "loop_id": state.get("loop_id"),
            "message_id": message_id,
            "transport": "inproc",
            "context": {
                "hot_cold": "hot",
                "tu": _normalize_tu(state.get("tu_id"), sender),
                "loop": LOOP_DISPLAY.get(state.get("loop_id"), state.get("loop_id")),
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
        sender_val = envelope.get("sender") or state.get("current_node")
        envelope["sender"] = _wrap_sender(sender_val)
        recv_val = envelope.get("receiver", "broadcast")
        envelope["receiver"] = _wrap_receiver(recv_val)
        envelope.setdefault("tu_id", _normalize_tu(state.get("tu_id"), sender_val if isinstance(sender_val, str) else sender_val.get("role", "*")))
        envelope.setdefault("snapshot_ref", state.get("snapshot_ref"))
        envelope.setdefault("loop_id", state.get("loop_id"))
        envelope.setdefault("transport", "inproc")
        envelope.setdefault("message_id", message_id)
        envelope.setdefault(
            "context",
            {
                "hot_cold": "hot",
                "tu": _normalize_tu(state.get("tu_id"), sender_val if isinstance(sender_val, str) else sender_val.get("role", "*")),
                "loop": LOOP_DISPLAY.get(state.get("loop_id"), state.get("loop_id")),
            },
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
            raise StateError(f"Envelope validation failed: {exc.message}")
