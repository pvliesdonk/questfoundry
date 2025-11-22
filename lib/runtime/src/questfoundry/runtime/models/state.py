"""
StudioState and related TypedDict/Pydantic models for runtime execution.

Based on spec: components/state_manager.md
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from datetime import datetime
import operator


class Artifact(TypedDict):
    """Single artifact in state."""
    artifact_type: str  # scene, character, lore_entry, illustration, etc.
    content: str | Dict[str, Any]  # Actual artifact content
    role_id: str  # Role that created it
    timestamp: str  # ISO format
    tu_id: str  # Associated Trace Unit
    state_key: str  # Where it lives in hot/cold sources
    metadata: Dict[str, Any]  # Additional metadata


class BarStatus(TypedDict):
    """Quality bar status."""
    status: Literal["green", "yellow", "red", "not_checked"]
    feedback: Optional[str]
    checked_by: Optional[str]  # Role ID of checker (usually gatekeeper)
    timestamp: Optional[str]  # ISO format


class Message(TypedDict):
    """Protocol message between roles."""
    sender: str  # Role ID
    receiver: str  # Role ID or "broadcast"
    intent: str  # Protocol intent (e.g., "request_review")
    payload: Dict[str, Any]  # Message payload
    timestamp: str  # ISO format
    envelope: Dict[str, Any]  # Envelope requirements (TU ID, snapshot ref, etc.)


class StudioState(TypedDict):
    """Complete state for loop execution.

    Uses Annotated types with reducers to handle concurrent updates from
    multiple nodes executing in the same LangGraph step.

    Reducers:
        - artifacts: Merge dicts (later values win for duplicate keys)
        - messages: Concatenate lists (all messages preserved)
        - quality_bars: Merge dicts (later values win)
    """
    # Core identity
    tu_id: str  # Trace Unit ID (e.g., "TU-2025-042")
    tu_lifecycle: Literal[
        "hot-proposed",
        "stabilizing",
        "gatecheck",
        "cold-merged"
    ]

    # Execution context
    current_node: str  # Currently executing role ID
    loop_id: str  # Which loop is running
    loop_context: Dict[str, Any]  # Loop-specific context

    # Artifacts and quality (with reducers for concurrent updates)
    artifacts: Annotated[Dict[str, Any], operator.or_]  # Merge dicts
    quality_bars: Annotated[Dict[str, BarStatus], operator.or_]  # Merge dicts

    # Protocol (with reducer for concurrent updates)
    messages: Annotated[List[Message], operator.add]  # Concatenate lists

    # Traceability
    snapshot_ref: Optional[str]  # Read-only snapshot reference
    parent_tu_id: Optional[str]  # Parent TU if derived

    # Error handling
    error: Optional[str]  # Error message if any
    retry_count: int  # Current retry count

    # Metadata
    created_at: str  # ISO format
    updated_at: str  # ISO format


# Quality bar dimensions
QUALITY_BARS = [
    "Integrity",
    "Reachability",
    "Nonlinearity",
    "Gateways",
    "Style",
    "Determinism",
    "Presentation",
    "Accessibility"
]

# Valid lifecycle transitions
VALID_TRANSITIONS = {
    "hot-proposed": ["stabilizing"],
    "stabilizing": ["gatecheck", "hot-proposed"],
    "gatecheck": ["stabilizing", "cold-merged"],
    "cold-merged": []  # Terminal state
}
