"""
Control Plane - Protocol-driven mesh routing for the Studio.

Replaces GraphFactory with envelope-based routing.
The Control Plane is the "laws of physics" for the Studio mesh -
it routes messages based on envelope metadata, not hardcoded edges.

Based on: Studio Graph Architecture gist (The Mesh & The Coordinator)
"""

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from questfoundry.runtime.core.node_factory import NodeFactory
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.models.state import Message, StudioState

logger = logging.getLogger(__name__)


# Special receiver values
TERMINATE = "__terminate__"
BROADCAST = "*"
SHOWRUNNER = "showrunner"


class DormancyRegistry:
    """Tracks which roles are active vs dormant."""

    def __init__(self) -> None:
        self._dormant: set[str] = set()
        self._always_on = {"showrunner", "gatekeeper"}

    def is_dormant(self, role_id: str) -> bool:
        """Check if a role is dormant."""
        if role_id in self._always_on:
            return False
        return role_id in self._dormant

    def wake(self, role_id: str) -> None:
        """Wake a dormant role."""
        self._dormant.discard(role_id)
        logger.info(f"Role woken: {role_id}")

    def sleep(self, role_id: str) -> None:
        """Put a role to sleep."""
        if role_id not in self._always_on:
            self._dormant.add(role_id)
            logger.info(f"Role dormant: {role_id}")

    def set_dormant_roles(self, role_ids: set[str]) -> None:
        """Set the initial dormant roles."""
        self._dormant = role_ids - self._always_on


class ControlPlane:
    """
    Protocol-driven mesh router for the Studio.

    The Control Plane builds a dynamic LangGraph where:
    - Every role can route to every other role
    - Routing is determined by message envelope `receiver` field
    - The Showrunner is coordinator, not bottleneck
    """

    def __init__(
        self,
        schema_registry: SchemaRegistry | None = None,
        node_factory: NodeFactory | None = None,
        state_manager: StateManager | None = None,
        preferred_provider: str | None = None,
    ):
        """Initialize the control plane."""
        self.schema_registry = schema_registry or SchemaRegistry()
        self.state_manager = state_manager or StateManager()
        self.node_factory = node_factory or NodeFactory(
            self.schema_registry,
            state_manager=self.state_manager,
            preferred_provider=preferred_provider,
        )
        self.dormancy = DormancyRegistry()
        self._graph_cache: dict[str, Any] = {}

        # Wire up orchestration tools to use our dormancy registry
        from questfoundry.runtime.tools.orchestration_tools import set_dormancy_registry
        set_dormancy_registry(self.dormancy)

        # Track message history for loop detection
        self._message_history: list[tuple[str, str, str]] = []  # (sender, receiver, intent)
        self._max_ping_pong = 3  # Max identical exchanges before intervention

    def get_available_roles(self) -> list[str]:
        """
        Get list of all available role IDs from spec definitions.

        Returns:
            List of role IDs (e.g., ["showrunner", "gatekeeper", "plotwright", ...])
        """
        # Core roles that are always available
        roles = [
            "showrunner",
            "gatekeeper",
            "plotwright",
            "scene_smith",
            "style_lead",
            "lore_weaver",
            "codex_curator",
            "researcher",
            "art_director",
            "illustrator",
            "audio_director",
            "audio_producer",
            "translator",
            "book_binder",
            "player_narrator",
        ]
        return roles

    def route_by_envelope(self, state: StudioState) -> str:
        """
        Core routing logic - inspects last message envelope to determine next node.

        This is the heart of the mesh architecture:
        - Direct messages route peer-to-peer
        - Reports to coordinator route to showrunner
        - Broadcasts trigger parallel execution
        - Termination signals end the graph

        Args:
            state: Current studio state with messages

        Returns:
            Next node ID or END
        """
        messages = state.get("messages", [])

        if not messages:
            # No messages yet - start with showrunner
            logger.debug("No messages, routing to showrunner")
            return SHOWRUNNER

        last_message = messages[-1]
        receiver = last_message.get("receiver", SHOWRUNNER)

        # Case: Termination signal
        if receiver == TERMINATE:
            logger.info("Termination signal received, ending graph")
            return END

        # Case: Broadcast (parallel execution)
        if receiver == BROADCAST:
            # For now, route to showrunner who will coordinate
            # Future: implement true parallel fan-out
            logger.debug("Broadcast message, routing to showrunner for coordination")
            return SHOWRUNNER

        # Case: Multi-receiver (parallel execution)
        if isinstance(receiver, list):
            # For now, route to showrunner who will coordinate
            # Future: implement true parallel fan-out
            logger.debug(f"Multi-receiver message to {receiver}, routing to showrunner")
            return SHOWRUNNER

        # Normalize receiver to role_id format
        receiver_id = self._normalize_role_id(receiver)

        # Case: Dormancy enforcement
        if self.dormancy.is_dormant(receiver_id):
            logger.info(f"Role {receiver_id} is dormant, routing to showrunner for wake decision")
            return SHOWRUNNER

        # Case: Loop detection
        if self._detect_ping_pong(last_message):
            logger.warning("Ping-pong detected, routing to showrunner for intervention")
            return SHOWRUNNER

        # Case: Direct peer-to-peer routing
        available_roles = self.get_available_roles()
        if receiver_id in available_roles:
            logger.debug(f"Direct routing to: {receiver_id}")
            return receiver_id

        # Fallback: unknown receiver, route to showrunner
        logger.warning(f"Unknown receiver '{receiver}', routing to showrunner")
        return SHOWRUNNER

    def _normalize_role_id(self, receiver: str) -> str:
        """
        Normalize receiver string to role_id format.

        Examples:
            "SR" -> "showrunner"
            "GK" -> "gatekeeper"
            "Scene Smith" -> "scene_smith"
        """
        # Abbreviation mapping
        abbreviations = {
            "SR": "showrunner",
            "GK": "gatekeeper",
            "PW": "plotwright",
            "SS": "scene_smith",
            "SL": "style_lead",
            "LW": "lore_weaver",
            "CC": "codex_curator",
            "RS": "researcher",
            "AD": "art_director",
            "IL": "illustrator",
            "AuD": "audio_director",
            "AuP": "audio_producer",
            "TR": "translator",
            "BB": "book_binder",
            "PN": "player_narrator",
        }

        if receiver in abbreviations:
            return abbreviations[receiver]

        # Convert to snake_case
        return receiver.lower().replace(" ", "_")

    def _detect_ping_pong(self, message: Message) -> bool:
        """
        Detect if roles are ping-ponging identical messages.

        Returns True if intervention is needed.
        """
        sender = message.get("sender", "")
        receiver = message.get("receiver", "")
        intent = message.get("intent", "")

        # Add to history
        self._message_history.append((sender, receiver, intent))

        # Keep only recent history
        if len(self._message_history) > 10:
            self._message_history = self._message_history[-10:]

        # Check for ping-pong pattern (A->B, B->A repeated)
        if len(self._message_history) >= self._max_ping_pong * 2:
            recent = self._message_history[-self._max_ping_pong * 2 :]
            # Check if it's alternating between same two roles
            pairs = [(recent[i], recent[i + 1]) for i in range(0, len(recent) - 1, 2)]
            if len(set(pairs)) == 1:
                return True

        return False

    def _create_showrunner_entry_node(self) -> Any:
        """
        Create the showrunner entry node that interprets human input.

        The showrunner always starts by receiving human input and
        emitting the first protocol messages to kick off work.
        """
        # Get the standard showrunner node
        return self.node_factory.create_role_node("showrunner")

    def _get_trace_handler(self) -> Any:
        """Get trace handler from state manager if available."""
        return getattr(self.state_manager, "_trace_handler", None)

    def _trace_message(self, message: Message) -> None:
        """Trace a message if trace handler is available."""
        trace_handler = self._get_trace_handler()
        if trace_handler:
            trace_handler.trace_message(message)

    def _wrap_node_with_envelope(self, role_id: str) -> Any:
        """
        Wrap a role node to ensure it emits proper envelope messages and traces them.

        The wrapped node:
        1. Traces role_started
        2. Executes the role logic
        3. Ensures output messages have proper envelope structure
        4. Traces all output messages
        5. Traces role_completed
        """
        base_node = self.node_factory.create_role_node(role_id)

        def wrapped_node(state: StudioState) -> dict[str, Any]:
            # Trace role started
            start_message: Message = {
                "sender": role_id,
                "receiver": "trace",
                "intent": "role_started",
                "payload": {"role_name": role_id},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "envelope": {"tu_id": state.get("tu_id", "")},
            }
            self._trace_message(start_message)

            # Execute base node
            result = base_node(state)

            # Ensure messages have proper envelope structure and trace them
            messages = result.get("messages", [])
            for msg in messages:
                if "envelope" not in msg:
                    msg["envelope"] = {}
                msg["envelope"]["tu_id"] = state.get("tu_id", "")
                msg["envelope"]["timestamp"] = datetime.now(timezone.utc).isoformat()

                # If no receiver specified, default to showrunner
                if not msg.get("receiver"):
                    msg["receiver"] = SHOWRUNNER

                # Trace the message
                self._trace_message(msg)

            # Trace role completed
            completed_message: Message = {
                "sender": role_id,
                "receiver": "trace",
                "intent": "role_completed",
                "payload": {
                    "role_name": role_id,
                    "insight": f"Generated {len(messages)} message(s)",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "envelope": {"tu_id": state.get("tu_id", "")},
            }
            self._trace_message(completed_message)

            return result

        return wrapped_node

    def create_studio_graph(self) -> Any:
        """
        Create the studio mesh graph.

        This builds a LangGraph where:
        - All roles are nodes
        - A universal router (route_by_envelope) handles all transitions
        - Showrunner is the entry point

        Returns:
            Compiled LangGraph ready for invocation
        """
        logger.info("Creating studio mesh graph")

        # Create StateGraph
        graph = StateGraph(StudioState)

        # Get all available roles
        roles = self.get_available_roles()

        # Add all role nodes
        for role_id in roles:
            try:
                node = self._wrap_node_with_envelope(role_id)
                graph.add_node(role_id, node)
                logger.debug(f"Added node: {role_id}")
            except Exception as e:
                logger.error(f"Failed to add node {role_id}: {e}")
                raise

        # Set showrunner as entry point
        graph.set_entry_point(SHOWRUNNER)

        # Add universal conditional edges from every node
        # Each node routes based on the envelope of the last message
        all_targets = {role_id: role_id for role_id in roles}
        all_targets[END] = END

        for role_id in roles:
            graph.add_conditional_edges(
                source=role_id,
                path=self.route_by_envelope,
                path_map=all_targets,
            )

        # Compile
        compiled = graph.compile()
        logger.info("Studio mesh graph compiled successfully")

        return compiled

    def run(
        self,
        human_input: str,
        context: dict[str, Any] | None = None,
        recursion_limit: int = 50,
    ) -> StudioState:
        """
        Run the studio with human input.

        Args:
            human_input: The user's request/directive
            context: Optional initial context
            recursion_limit: Max graph iterations

        Returns:
            Final studio state
        """
        # Initialize state
        initial_state = self.state_manager.initialize_state(
            tu_id=f"TU-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            loop_id="interactive",
            context=context or {},
        )

        # Add human input as first message
        human_message: Message = {
            "sender": "human",
            "receiver": SHOWRUNNER,
            "intent": "human.directive",
            "payload": {"content": human_input},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "envelope": {
                "tu_id": initial_state["tu_id"],
            },
        }
        initial_state["messages"] = [human_message]

        # Trace the human input
        self._trace_message(human_message)

        # Create and run graph
        graph = self.create_studio_graph()

        logger.info(f"Starting studio run with input: {human_input[:100]}...")

        try:
            final_state = graph.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit},
            )
            logger.info("Studio run completed successfully")
            return final_state

        except Exception as e:
            logger.error(f"Studio run failed: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear graph cache and message history."""
        self._graph_cache.clear()
        self._message_history.clear()
        logger.debug("Cleared control plane cache")
