from __future__ import annotations

from typing import Any, Literal, TypedDict

from questfoundry.runtime.models.state import Message


class Envelope(TypedDict, total=False):
    sender: str
    receiver: str | list[str]
    intent: str
    payload: dict[str, Any]
    tu_id: str
    snapshot_ref: str | None
    causality_ref: str | None
    loop_id: str | None
    metadata: dict[str, Any]
    transport: Literal["inproc", "http"]
    message_id: str
