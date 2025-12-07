"""LangGraph builder for QuestFoundry workflows.

This module provides utilities for building LangGraph StateGraphs
from compiled loop definitions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from questfoundry.runtime.state import Intent, StudioState

if TYPE_CHECKING:
    from questfoundry.compiler.models import LoopIR, RoleIR


def create_role_node(
    role: RoleIR,
    llm: BaseChatModel,
) -> Callable[[StudioState], dict[str, Any]]:
    """Create a LangGraph node function for a role.

    Args:
        role: The role definition.
        llm: The LLM to use for this role.

    Returns:
        A node function that takes StudioState and returns updates.
    """

    def node(state: StudioState) -> dict[str, Any]:
        """Execute the role and return state updates."""
        # Build system message from role
        system_content = f"""You are the {role.archetype}, responsible for: {role.mandate}

## Constraints
{chr(10).join(f"- {c}" for c in role.constraints)}

When you have completed your work, respond with your output.
"""
        messages = [SystemMessage(content=system_content), *state.get("messages", [])]

        # Invoke LLM
        response = llm.invoke(messages)

        # Create handoff intent (simplified - real impl would parse response)
        intent = Intent(
            type="handoff",
            source_role=role.id,
            status="completed",
        )

        return {
            "messages": [response],
            "current_role": role.id,
            "pending_intents": [intent],
            "iteration": state.get("iteration", 0) + 1,
        }

    return node


def create_router(loop: LoopIR) -> Callable[[StudioState], str]:
    """Create a routing function based on loop edges.

    Args:
        loop: The loop definition with edges.

    Returns:
        A router function that returns the next node name.
    """
    # Build edge lookup: source -> [(condition, target), ...]
    edge_map: dict[str, list[tuple[str, str]]] = {}
    for edge in loop.edges:
        if edge.source not in edge_map:
            edge_map[edge.source] = []
        edge_map[edge.source].append((edge.condition, edge.target))

    def router(state: StudioState) -> str:
        """Route to next node based on intent."""
        current = state.get("current_role", "")
        intents = state.get("pending_intents", [])

        if not intents:
            return END

        intent = intents[-1]

        # Check for termination
        if intent.type == "terminate":
            return END

        # Look up edges for current role
        edges = edge_map.get(current, [])

        for condition, target in edges:
            # Simple condition evaluation (real impl would be more sophisticated)
            # For now, just check if condition matches intent status
            if intent.status in condition or condition == "true":
                return target

        # No matching edge - end
        return END

    return router


def build_graph(
    loop: LoopIR,
    roles: dict[str, RoleIR],
    llm: BaseChatModel,
) -> StateGraph[StudioState]:
    """Build a LangGraph StateGraph from a loop definition.

    Args:
        loop: The loop definition.
        roles: Dictionary of role definitions.
        llm: The LLM to use for all roles.

    Returns:
        Compiled StateGraph ready for execution.
    """
    graph: StateGraph[StudioState] = StateGraph(StudioState)

    # Add nodes for each role in the loop
    for node in loop.nodes:
        role = roles.get(node.role)
        if role is None:
            raise ValueError(f"Unknown role '{node.role}' in loop '{loop.id}'")

        graph.add_node(node.id, create_role_node(role, llm))  # type: ignore[call-overload]

    # Add conditional edges based on routing
    router = create_router(loop)

    for node in loop.nodes:
        # Get possible targets from this node
        targets = {e.target for e in loop.edges if e.source == node.id}
        targets.add(END)  # Always allow ending

        graph.add_conditional_edges(
            node.id,
            router,
            {t: t for t in targets},
        )

    # Set entry point
    graph.set_entry_point(loop.entry_point)

    return graph


# =============================================================================
# Example usage (for testing)
# =============================================================================


def create_example_graph(llm: BaseChatModel) -> StateGraph[StudioState]:
    """Create a simple example graph for testing.

    This creates a minimal 2-node graph:
    showrunner -> plotwright -> END
    """
    from questfoundry.compiler.models import (
        Agency,
        GraphEdgeIR,
        GraphNodeIR,
        LoopIR,
        RoleIR,
    )

    # Define minimal roles
    showrunner = RoleIR(
        id="showrunner",
        abbr="SR",
        archetype="Product Owner",
        agency=Agency.HIGH,
        mandate="Manage by Exception",
        constraints=["Translate user requests into actionable briefs"],
    )

    plotwright = RoleIR(
        id="plotwright",
        abbr="PW",
        archetype="Architect",
        agency=Agency.MEDIUM,
        mandate="Design the Topology",
        constraints=["Create story structure", "Define scene flow"],
    )

    # Define minimal loop
    loop = LoopIR(
        id="example",
        name="Example Loop",
        trigger="user_request",
        entry_point="showrunner",
        nodes=[
            GraphNodeIR(id="showrunner", role="showrunner"),
            GraphNodeIR(id="plotwright", role="plotwright"),
        ],
        edges=[
            GraphEdgeIR(
                source="showrunner",
                target="plotwright",
                condition="completed",
            ),
            GraphEdgeIR(
                source="plotwright",
                target="__end__",
                condition="completed",
            ),
        ],
    )

    roles = {"showrunner": showrunner, "plotwright": plotwright}

    return build_graph(loop, roles, llm)
