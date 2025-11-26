from __future__ import annotations

from typing import Any, Literal, TypedDict

from questfoundry.runtime.models.state import Message


# Special receiver values for mesh routing
TERMINATE = "__terminate__"
BROADCAST = "*"
SHOWRUNNER = "showrunner"


class Envelope(TypedDict, total=False):
    """Layer 4 protocol envelope for mesh routing.

    The envelope metadata accompanies every message and enables:
    - Traceability: tu_id, causality_ref link messages to Trace Units
    - Context: snapshot_ref, loop_id provide execution context
    - Routing: receiver field drives Control Plane routing decisions
    """

    sender: str  # Role ID or "human"
    receiver: str | list[str]  # Role ID, abbreviation, "*", list, or "__terminate__"
    intent: str  # Protocol intent (e.g., "tu.assign", "gate.report.submit")
    payload: dict[str, Any]  # Message payload
    tu_id: str  # Trace Unit ID for traceability
    snapshot_ref: str | None  # Cold snapshot reference
    causality_ref: str | None  # Parent message ID for causality chain
    loop_id: str | None  # Current loop context (informational, not routing)
    metadata: dict[str, Any]  # Additional metadata
    transport: Literal["inproc", "http"]  # Transport mechanism
    message_id: str  # Unique message identifier
