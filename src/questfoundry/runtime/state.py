"""StudioState - the central state model for LangGraph workflow execution.

This module defines the runtime state model used by LangGraph to track
workflow execution. It follows LangGraph's native patterns using TypedDict
with annotated reducers for message accumulation.

Architecture
------------
The state model consists of three main components:

1. **Artifact Stores** (``hot_store``, ``cold_store``):
   - ``hot_store``: Mutable working drafts, visible to all roles
   - ``cold_store``: Immutable canonical data, append-only

2. **Message History** (``messages``):
   Uses LangGraph's ``add_messages`` reducer for automatic message
   accumulation across node invocations.

3. **Routing State** (``current_role``, ``pending_intents``):
   Tracks which role is active and what intents are pending for
   the router to process.

LangGraph Integration
---------------------
The ``StudioState`` TypedDict is designed to work directly with LangGraph's
StateGraph. Key features:

- ``messages`` uses ``Annotated[list[AnyMessage], add_messages]`` for
  automatic message list accumulation (no manual concatenation needed)
- All fields use ``total=False`` to allow partial state updates from nodes
- State is passed to each node and returned updates are merged

Example Usage
-------------
Create initial state for a workflow::

    from questfoundry.runtime import create_initial_state

    state = create_initial_state("story_spark", "Create a mystery story")

Use with LangGraph::

    from langgraph.graph import StateGraph

    graph = StateGraph(StudioState)
    graph.add_node("showrunner", showrunner_node)
    # ... add more nodes and edges
    compiled = graph.compile()

    result = compiled.invoke(state)

See Also
--------
:mod:`questfoundry.runtime.graph` : Graph builder using this state
:mod:`langgraph.graph` : LangGraph StateGraph documentation
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
    """Base class for all artifacts stored in the studio state.

    Artifacts are the primary data objects created and manipulated by roles
    during workflow execution. They can be story hooks, scenes, character
    profiles, or any other domain-specific content.

    Artifacts follow a lifecycle (typically draft -> review -> final) and
    can be stored in either the hot store (mutable) or cold store (immutable).

    Attributes
    ----------
    id : str
        Unique artifact identifier (e.g., "hook-001", "scene-intro").
    type : str
        Artifact type name matching an ArtifactTypeIR (e.g., "hook_card").
    status : str
        Current lifecycle status. Defaults to "draft".
    created_at : datetime
        When the artifact was created. Defaults to now.
    updated_at : datetime
        When the artifact was last modified. Defaults to now.
    created_by : str
        Role ID that created this artifact. Defaults to empty string.
    data : dict[str, Any]
        Artifact payload matching the type's field schema. Defaults to empty dict.

    Examples
    --------
    Create a hook card artifact::

        artifact = Artifact(
            id="hook-001",
            type="hook_card",
            status="draft",
            created_by="showrunner",
            data={
                "title": "The Missing Heirloom",
                "hook_type": "narrative",
                "description": "A family heirloom has vanished...",
            },
        )

    Move from hot to cold store when finalized::

        artifact.status = "final"
        artifact.updated_at = datetime.now()
        state["cold_store"][artifact.id] = artifact
        del state["hot_store"][artifact.id]
    """

    id: str
    """Unique artifact identifier."""

    type: str
    """Artifact type name (matches ArtifactTypeIR.id)."""

    status: str = "draft"
    """Current lifecycle status."""

    created_at: datetime = Field(default_factory=datetime.now)
    """Creation timestamp."""

    updated_at: datetime = Field(default_factory=datetime.now)
    """Last modification timestamp."""

    created_by: str = ""
    """Role ID that created this artifact."""

    data: dict[str, Any] = Field(default_factory=dict)
    """Artifact payload (type-specific fields)."""


# =============================================================================
# Intent Model
# =============================================================================


class DelegationResult(BaseModel):
    """Result returned by a specialist role to the Showrunner.

    When SR delegates work to a role via delegate_to(), the role executes
    autonomously and returns a DelegationResult when complete. SR then
    evaluates the result and decides the next action.

    Attributes
    ----------
    role_id : str
        The role that performed the work.
    status : str
        Work outcome: "completed", "blocked", "needs_review", "error".
    artifacts : list[str]
        IDs of artifacts created or modified during this delegation.
    message : str
        Summary of work done for SR to understand the outcome.
    recommendation : str | None
        Optional suggestion for SR's next action (e.g., "engage lorekeeper
        to verify canon consistency").

    Examples
    --------
    Successful completion::

        result = DelegationResult(
            role_id="plotwright",
            status="completed",
            artifacts=["topology-001"],
            message="Created story topology with 5 scenes and 3 branches.",
            recommendation="Engage scene_smith to draft scene content.",
        )

    Blocked on missing information::

        result = DelegationResult(
            role_id="scene_smith",
            status="blocked",
            artifacts=[],
            message="Cannot draft scene - missing character backstory.",
            recommendation="Engage lorekeeper to establish character canon.",
        )
    """

    role_id: str
    """The role that performed the work."""

    status: str
    """Work outcome: completed, blocked, needs_review, error."""

    artifacts: list[str] = Field(default_factory=list)
    """IDs of artifacts created or modified."""

    message: str
    """Summary of work done."""

    recommendation: str | None = None
    """Optional suggestion for next action."""


class Intent(BaseModel):
    """A role's declaration of work status used for routing decisions.

    Intents are the primary communication mechanism between roles and the
    runtime router. When a role completes work, it emits an intent that
    the router uses to determine the next node to visit.

    Intent Types
    ------------
    ``handoff``
        Normal completion - transfer work to next role in workflow.
    ``escalation``
        Exception case - bump to higher-agency role for handling.
    ``broadcast``
        Notification - inform multiple roles without routing change.
    ``terminate``
        Completion signal - end the workflow execution.

    Attributes
    ----------
    type : Literal["handoff", "escalation", "broadcast", "terminate"]
        Intent type determining routing behavior.
    source_role : str
        Role ID that emitted this intent.
    status : str
        Work status (e.g., "completed", "needs_revision"). Defaults to "completed".
    payload : dict[str, Any] | None
        Optional additional data for the target role. Defaults to None.
    reason : str | None
        Optional explanation for escalation/termination. Defaults to None.
    artifact_ids : list[str]
        IDs of artifacts related to this intent. Defaults to empty list.
    timestamp : datetime
        When the intent was created. Defaults to now.

    Examples
    --------
    Normal handoff after completing work::

        intent = Intent(
            type="handoff",
            source_role="showrunner",
            status="completed",
            artifact_ids=["brief-001"],
        )

    Escalation when work cannot proceed::

        intent = Intent(
            type="escalation",
            source_role="scene_smith",
            status="blocked",
            reason="Contradicts established canon - need Lorekeeper review",
        )

    Terminate workflow::

        intent = Intent(
            type="terminate",
            source_role="publisher",
            status="completed",
            artifact_ids=["output-001"],
        )
    """

    type: Literal["handoff", "escalation", "broadcast", "terminate"]
    """Intent type determining routing behavior."""

    source_role: str
    """Role ID that emitted this intent."""

    status: str = "completed"
    """Work status (e.g., "completed", "needs_revision")."""

    payload: dict[str, Any] | None = None
    """Optional additional data for target role."""

    reason: str | None = None
    """Optional explanation for escalation/termination."""

    artifact_ids: list[str] = Field(default_factory=list)
    """IDs of related artifacts."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """Intent creation timestamp."""


# =============================================================================
# StudioState - LangGraph TypedDict
# =============================================================================


class StudioState(TypedDict, total=False):
    """Central state TypedDict for LangGraph workflow execution.

    This TypedDict defines the complete state shape passed through the
    LangGraph workflow. It uses ``total=False`` to allow partial updates
    from node functions.

    The ``messages`` field uses LangGraph's ``add_messages`` reducer,
    which automatically handles message accumulation. Node functions
    can return ``{"messages": [new_message]}`` and LangGraph will
    append rather than replace.

    State Components
    ----------------
    **Artifact Stores** (``hot_store``, ``cold_store``):
        Dictionary stores for artifacts keyed by artifact ID.
        Hot store holds mutable drafts; cold store holds immutable finals.

    **Message History** (``messages``):
        LangChain message list with automatic accumulation via add_messages.
        Contains the conversation history including system, human, and AI messages.

    **Routing** (``current_role``, ``pending_intents``):
        Tracks active role and queued intents for router evaluation.

    **Context** (``loop_id``, ``iteration``, ``metadata``):
        Workflow execution context including loop identifier and iteration count.

    Examples
    --------
    Create state for testing::

        state: StudioState = {
            "hot_store": {},
            "cold_store": {},
            "messages": [],
            "current_role": "showrunner",
            "pending_intents": [],
            "loop_id": "story_spark",
            "iteration": 0,
            "metadata": {},
        }

    Node function returning partial update::

        def showrunner_node(state: StudioState) -> dict[str, Any]:
            # LangGraph merges this with existing state
            return {
                "messages": [AIMessage(content="Created brief")],
                "current_role": "showrunner",
                "iteration": state.get("iteration", 0) + 1,
            }

    See Also
    --------
    :func:`create_initial_state` : Factory function for initial state
    :class:`Artifact` : Artifact model for stores
    :class:`Intent` : Intent model for routing
    """

    # Artifact stores
    hot_store: dict[str, Artifact]
    """Mutable artifact store for working drafts."""

    cold_store: dict[str, Artifact]
    """Immutable artifact store for canonical data."""

    # LangGraph message history (uses add_messages reducer)
    messages: Annotated[list[AnyMessage], add_messages]
    """Conversation history with automatic accumulation."""

    # Routing
    current_role: str
    """Currently active role ID."""

    pending_intents: list[Intent]
    """Queue of intents awaiting router evaluation."""

    # Context
    loop_id: str
    """Identifier of the executing loop."""

    iteration: int
    """Current iteration count (incremented each node visit)."""

    metadata: dict[str, Any]
    """Additional execution metadata."""


def create_initial_state(loop_id: str, user_input: str | None = None) -> StudioState:
    """Create initial state for starting a workflow loop.

    Factory function that creates a properly initialized StudioState
    ready for LangGraph execution.

    Parameters
    ----------
    loop_id : str
        Identifier of the loop to execute (e.g., "story_spark").
    user_input : str | None, optional
        Optional initial user message to include in the conversation.
        If provided, creates a HumanMessage as the first message.
        Defaults to None.

    Returns
    -------
    StudioState
        Initialized state with empty stores, the specified loop_id,
        iteration 0, and optionally a user message.

    Examples
    --------
    Create state without user input::

        state = create_initial_state("story_spark")
        assert state["loop_id"] == "story_spark"
        assert len(state["messages"]) == 0

    Create state with user input::

        state = create_initial_state("story_spark", "Create a mystery story")
        assert len(state["messages"]) == 1
        assert "mystery" in state["messages"][0].content.lower()

    Use with compiled graph::

        from questfoundry.runtime import create_initial_state
        from questfoundry.runtime.graph import build_graph

        state = create_initial_state("story_spark", "Write a detective story")
        compiled = build_graph(loop_ir, roles, llm).compile()
        result = compiled.invoke(state)
    """
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
