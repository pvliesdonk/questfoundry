"""LangGraph builder for QuestFoundry workflows.

This module provides utilities for building LangGraph StateGraphs from compiled
loop definitions (LoopIR). It bridges the gap between the compiler's IR output
and LangGraph's runtime execution model.

Architecture
------------
The graph builder creates a LangGraph StateGraph with:

1. **Nodes**: Each loop node becomes a LangGraph node that:
   - Builds a system prompt from the role definition
   - Invokes the LLM with conversation history
   - Emits an intent for routing

2. **Router**: A conditional routing function that:
   - Evaluates edge conditions against intent status
   - Determines the next node to visit
   - Handles termination via ``END`` sentinel

3. **Edges**: Conditional edges connecting nodes based on:
   - Intent status (e.g., "completed", "needs_revision")
   - Simple condition matching

Workflow Execution
------------------
When the compiled graph is invoked:

1. Execution starts at the ``entry_point`` node
2. The node's role processes the current state
3. An intent is emitted declaring work status
4. The router evaluates edges to find the next node
5. Process repeats until ``END`` is reached

Example Usage
-------------
Build and run a workflow::

    from langchain_ollama import ChatOllama
    from questfoundry.runtime.graph import build_graph
    from questfoundry.runtime import create_initial_state

    # Get IR from compiler (or create manually)
    loop_ir = ...  # LoopIR instance
    roles = {...}  # dict[str, RoleIR]

    # Build graph
    llm = ChatOllama(model="llama3.2")
    graph = build_graph(loop_ir, roles, llm)

    # Compile and run
    compiled = graph.compile()
    initial = create_initial_state("story_spark", "Write a mystery story")
    result = compiled.invoke(initial)

    # Access results
    final_messages = result["messages"]
    artifacts = result["cold_store"]

See Also
--------
:class:`questfoundry.runtime.state.StudioState` : State model
:class:`questfoundry.compiler.models.LoopIR` : Loop IR definition
:mod:`langgraph.graph` : LangGraph StateGraph documentation
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

    Generates a closure that, when called with state, builds a system prompt
    from the role definition, invokes the LLM, and returns state updates.

    The generated node function:
    1. Constructs a system message from role archetype, mandate, and constraints
    2. Prepends the system message to the conversation history
    3. Invokes the LLM with the full message list
    4. Creates a handoff intent (simplified for Phase 1)
    5. Returns state updates for LangGraph to merge

    Parameters
    ----------
    role : RoleIR
        The role definition containing archetype, mandate, and constraints.
    llm : BaseChatModel
        The LangChain chat model to use for this role's invocations.

    Returns
    -------
    Callable[[StudioState], dict[str, Any]]
        A node function compatible with LangGraph's ``add_node()``.
        Takes StudioState and returns a dict of state updates.

    Notes
    -----
    The current implementation is simplified for Phase 1:
    - Always creates a "handoff" intent with "completed" status
    - Doesn't parse the response for structured output
    - Uses a basic system prompt template

    Future enhancements will add:
    - Response parsing for intent extraction
    - Tool integration
    - Jinja2 template rendering for prompts

    Examples
    --------
    Create and use a node function::

        from questfoundry.compiler.models import RoleIR, Agency
        from langchain_ollama import ChatOllama

        role = RoleIR(
            id="showrunner",
            abbr="SR",
            archetype="Product Owner",
            agency=Agency.HIGH,
            mandate="Translate user intent",
            constraints=["Keep responses focused"],
        )

        llm = ChatOllama(model="llama3.2")
        node_fn = create_role_node(role, llm)

        # Use in LangGraph
        graph.add_node("showrunner", node_fn)
    """

    def node(state: StudioState) -> dict[str, Any]:
        """Execute the role and return state updates.

        Parameters
        ----------
        state : StudioState
            Current workflow state including message history.

        Returns
        -------
        dict[str, Any]
            State updates to merge: messages, current_role, pending_intents, iteration.
        """
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
    """Create a routing function based on loop edge definitions.

    Generates a router closure that evaluates state to determine the next
    node. The router is used with LangGraph's ``add_conditional_edges()``.

    Routing Logic
    -------------
    1. If no pending intents, return END
    2. If latest intent is "terminate", return END
    3. For each edge from current node:
       - Check if condition matches intent status
       - Return first matching target
    4. If no edges match, return END

    Parameters
    ----------
    loop : LoopIR
        The loop definition containing edges with conditions.

    Returns
    -------
    Callable[[StudioState], str]
        A router function that takes state and returns a node name string
        (or the END sentinel).

    Notes
    -----
    The current condition evaluation is simplified:
    - Checks if intent.status substring appears in condition
    - Treats "true" as unconditional match

    Future enhancements will add:
    - Expression parsing for complex conditions
    - Access to artifact state in conditions
    - Priority-based edge evaluation

    Examples
    --------
    Create and use a router::

        router = create_router(loop_ir)

        # Use with conditional edges
        graph.add_conditional_edges(
            "showrunner",
            router,
            {"plotwright": "plotwright", END: END},
        )
    """
    # Build edge lookup: source -> [(condition, target), ...]
    edge_map: dict[str, list[tuple[str, str]]] = {}
    for edge in loop.edges:
        if edge.source not in edge_map:
            edge_map[edge.source] = []
        edge_map[edge.source].append((edge.condition, edge.target))

    def router(state: StudioState) -> str:
        """Route to next node based on intent status.

        Parameters
        ----------
        state : StudioState
            Current workflow state.

        Returns
        -------
        str
            Name of the next node to visit, or END sentinel.
        """
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

    This is the main entry point for converting compiled IR into an executable
    LangGraph workflow. It creates nodes for each role, wires up conditional
    edges based on the loop definition, and sets the entry point.

    The returned graph is uncompiled - call ``.compile()`` before invoking.

    Parameters
    ----------
    loop : LoopIR
        The loop definition containing nodes, edges, and entry point.
    roles : dict[str, RoleIR]
        Dictionary of role definitions keyed by role ID. All roles
        referenced by loop nodes must be present.
    llm : BaseChatModel
        The LangChain chat model to use for all role invocations.
        In the future, this may be per-role configurable.

    Returns
    -------
    StateGraph[StudioState]
        Uncompiled StateGraph ready for ``.compile()`` and ``.invoke()``.

    Raises
    ------
    ValueError
        If a loop node references a role not present in the roles dict.

    Notes
    -----
    The graph structure mirrors the loop definition:
    - Each GraphNodeIR becomes a LangGraph node
    - Each GraphEdgeIR becomes a conditional edge via the router
    - The entry_point sets where execution starts

    Examples
    --------
    Build and run a workflow::

        from langchain_ollama import ChatOllama
        from questfoundry.runtime import create_initial_state

        llm = ChatOllama(model="llama3.2")
        graph = build_graph(loop_ir, roles, llm)

        # Must compile before invoking
        compiled = graph.compile()

        # Create initial state and run
        state = create_initial_state("story_spark", "Write a story")
        result = compiled.invoke(state)

    Visualize the graph (requires graphviz)::

        graph = build_graph(loop_ir, roles, llm)
        compiled = graph.compile()
        print(compiled.get_graph().draw_mermaid())

    See Also
    --------
    :func:`create_role_node` : Node function generator
    :func:`create_router` : Router function generator
    :func:`create_example_graph` : Example for testing
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
    """Create a simple example graph for testing and demonstration.

    Creates a minimal 2-node workflow graph without requiring external
    IR files. Useful for testing the graph builder and understanding
    the execution flow.

    Graph Structure
    ---------------
    ::

        [START] -> showrunner -> plotwright -> [END]

    - showrunner: High-agency "Product Owner" role
    - plotwright: Medium-agency "Architect" role
    - Both transition on "completed" status

    Parameters
    ----------
    llm : BaseChatModel
        The LangChain chat model to use for both roles.

    Returns
    -------
    StateGraph[StudioState]
        Uncompiled example graph ready for ``.compile()`` and ``.invoke()``.

    Examples
    --------
    Create and run the example::

        from langchain_ollama import ChatOllama
        from questfoundry.runtime import create_initial_state
        from questfoundry.runtime.graph import create_example_graph

        llm = ChatOllama(model="llama3.2")
        graph = create_example_graph(llm)
        compiled = graph.compile()

        state = create_initial_state("example", "Create a short story")
        result = compiled.invoke(state)

        # Inspect results
        for msg in result["messages"]:
            print(f"{msg.__class__.__name__}: {msg.content[:100]}...")
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
