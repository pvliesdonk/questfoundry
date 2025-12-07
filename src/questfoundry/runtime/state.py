"""StudioState - the central state model for LangGraph execution.

Uses LangGraph's native patterns with langchain-core message types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# =============================================================================
# Artifact Model
# =============================================================================


class Artifact(BaseModel):
    """Base class for all artifacts in the studio."""

    id: str
    type: str
    status: str = "draft"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Intent Model
# =============================================================================


class Intent(BaseModel):
    """A role's declaration of work status for routing."""

    type: Literal["handoff", "escalation", "broadcast", "terminate"]
    source_role: str
    status: str = "completed"
    payload: dict[str, Any] | None = None
    reason: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# StudioState - LangGraph TypedDict
# =============================================================================


class StudioState(TypedDict, total=False):
    """Central state for LangGraph execution.

    Uses LangGraph's add_messages reducer for message accumulation.
    """

    # Artifact stores
    hot_store: dict[str, Artifact]
    cold_store: dict[str, Artifact]

    # LangGraph message history (uses add_messages reducer)
    messages: Annotated[list[AnyMessage], add_messages]

    # Routing
    current_role: str
    pending_intents: list[Intent]

    # Context
    loop_id: str
    iteration: int
    metadata: dict[str, Any]


def create_initial_state(loop_id: str, user_input: str | None = None) -> StudioState:
    """Create initial state for a loop execution."""
    from langchain_core.messages import HumanMessage

    messages: list[AnyMessage] = []
    if user_input:
        messages.append(HumanMessage(content=user_input))

    return StudioState(
        hot_store={},
        cold_store={},
        messages=messages,
        current_role="",
        pending_intents=[],
        loop_id=loop_id,
        iteration=0,
        metadata={"started_at": datetime.now().isoformat()},
    )
