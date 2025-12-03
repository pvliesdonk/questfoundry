"""
Control Plane - Protocol-driven mesh routing for the Studio.

Replaces GraphFactory with envelope-based routing.
The Control Plane is the "laws of physics" for the Studio mesh -
it routes messages based on envelope metadata, not hardcoded edges.

Based on: Studio Graph Architecture gist (The Mesh & The Coordinator)
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, StateGraph

from questfoundry.runtime.config import get_settings
from questfoundry.runtime.core.node_factory import NodeFactory
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.models.state import Message, StudioState

logger = logging.getLogger(__name__)

# Lazy getter for structured bus logger (configured at runtime by CLI)
def _get_bus_log():
    """Get bus logger if configured, None otherwise.

    Uses lazy evaluation because configure_structured_logging() is called
    by CLI after module import. Checking is_configured() avoids the
    RuntimeError from get_bus_logger() when logging isn't set up.
    """
    try:
        from questfoundry.runtime.structured_logging import get_bus_logger, is_configured

        if is_configured():
            return get_bus_logger()
    except ImportError:
        pass
    return None

# Optional LangSmith tracing
try:
    from langsmith import traceable
except ImportError:
    # LangSmith not available, use no-op decorator
    def traceable(**kwargs):
        def decorator(func):
            return func

        return decorator


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
    Protocol-driven mesh router for the Studio (async-first architecture).

    The Control Plane builds a dynamic LangGraph where:
    - Every role can route to every other role
    - Routing is determined by message envelope `receiver` field
    - Messages are queued per receiver (message bus pattern)
    - Roles only execute when they have pending messages
    - The Showrunner is coordinator, not bottleneck

    Async Pattern (Preferred):
    - Use run() for async execution
    - All role nodes execute asynchronously
    - Supports both sync and async base nodes via inspection (asyncio.iscoroutinefunction)

    Backward Compatibility:
    - Use run_sync() for synchronous contexts
    - Wraps run() with asyncio.run()
    """

    def __init__(
        self,
        schema_registry: SchemaRegistry | None = None,
        node_factory: NodeFactory | None = None,
        state_manager: StateManager | None = None,
        preferred_provider: str | None = None,
    ):
        """Initialize the control plane."""
        # Load runtime settings from centralized config
        settings = get_settings()

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

        # Track message history for loop detection (from centralized config)
        self._message_history: list[tuple[str, str, str]] = []  # (sender, receiver, intent)
        self._max_ping_pong = settings.runtime.max_ping_pong

        # Track role execution counts to detect infinite loops
        self._role_execution_counts: dict[str, int] = {}
        self._max_role_executions = settings.runtime.max_role_executions
        self._total_executions: int = 0  # Track total executions for periodic reset
        self._reset_threshold: int = settings.runtime.execution_reset_threshold

        # Fairness tracking - prevent same role running too many times consecutively
        self._last_executed_role: str | None = None
        self._consecutive_executions: int = 0
        self._max_consecutive: int = settings.runtime.max_consecutive_role_executions

        # Parallel execution - how many roles to run concurrently
        self._max_parallel: int = settings.runtime.max_parallel_roles

        # Role abbreviation mapping for message bus routing
        self._abbreviations = {
            "sr": "showrunner",
            "gk": "gatekeeper",
            "pw": "plotwright",
            "ss": "scene_smith",
            "sl": "style_lead",
            "lw": "lore_weaver",
            "cc": "codex_curator",
            "rs": "researcher",
            "ad": "art_director",
            "il": "illustrator",
            "aud": "audio_director",
            "audr": "audio_director",
            "aup": "audio_producer",
            "aupr": "audio_producer",
            "tr": "translator",
            "bb": "book_binder",
            "pn": "player_narrator",
        }

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

    def _get_pending_messages_by_role(
        self, state: StudioState
    ) -> tuple[dict[str, list[Message]], dict[str, int]]:
        """
        Get all unprocessed messages grouped by receiver role.

        Uses state["_consumed_messages"] to track which messages have been consumed.
        Messages not in the consumed set are considered pending.

        Args:
            state: Current studio state with messages

        Returns:
            Tuple of:
            - Dict mapping role_id -> list of pending messages for that role
            - Dict mapping role_id -> count of DIRECT messages (not broadcasts)
        """
        messages = state.get("messages", [])
        consumed = state.get("_consumed_messages", set())
        available_roles = set(self.get_available_roles())

        pending: dict[str, list[Message]] = {}
        direct_counts: dict[str, int] = {}  # Track direct (non-broadcast) messages

        for idx, msg in enumerate(messages):
            receiver = msg.get("receiver", "")

            # Skip system/trace messages
            if receiver in ("system", "trace", ""):
                continue

            # Normalize receiver
            if isinstance(receiver, dict):
                receiver = receiver.get("role", "") or receiver.get("id", "")
            receiver = str(receiver).lower().replace(" ", "_")

            # Map abbreviations to full role IDs
            receiver_id = self._abbreviations.get(receiver, receiver)

            # Check if this message has been consumed
            if idx in consumed:
                continue  # Already consumed by target role

            # Only track messages to actual roles
            if receiver_id in available_roles:
                pending.setdefault(receiver_id, []).append(msg)
                # This is a DIRECT message - increment direct count
                direct_counts[receiver_id] = direct_counts.get(receiver_id, 0) + 1
            elif receiver == BROADCAST or receiver == "*":
                # Broadcast messages go to all active roles that haven't consumed them
                # EXCEPT the sender (roles don't receive their own broadcasts)
                # NOTE: Broadcasts do NOT increment direct_counts
                sender = msg.get("sender", "")
                if isinstance(sender, dict):
                    sender = sender.get("role", "") or sender.get("id", "")
                sender = str(sender).lower().replace(" ", "_")
                sender_id = self._abbreviations.get(sender, sender)

                for role in available_roles:
                    if role == sender_id:
                        continue  # Sender doesn't receive their own broadcast
                    if not self.dormancy.is_dormant(role):
                        # Check role-specific broadcast consumption
                        if (idx, role) not in consumed:
                            pending.setdefault(role, []).append(msg)

        # Log pending message counts per role using structured logging
        bus_log = _get_bus_log()
        if bus_log:
            pending_counts = {role: len(msgs) for role, msgs in pending.items() if msgs}
            if pending_counts:
                bus_log.info(
                    "pending_messages_by_role",
                    pending_counts=pending_counts,
                    direct_counts=direct_counts,
                    total_pending=sum(pending_counts.values()),
                )

        return pending, direct_counts

    def route_by_envelope(self, state: StudioState) -> list[str]:
        """
        Core routing logic - message bus pattern with parallel execution.

        Routes to roles that have pending messages. Returns up to _max_parallel
        roles for concurrent execution. This ensures:
        - Roles only execute when they have work (messages) to process
        - Multiple roles can process messages in parallel
        - No role executes "empty" without a task
        - Graph ends when all messages are consumed

        Args:
            state: Current studio state with messages

        Returns:
            List of next node IDs (for parallel execution) or [END]
        """
        messages = state.get("messages", [])

        if not messages:
            # No messages yet - start with showrunner
            logger.debug("No messages, routing to showrunner")
            return [SHOWRUNNER]

        # Check for termination in the last message
        last_message = messages[-1]
        last_receiver = last_message.get("receiver", "")
        if last_receiver == TERMINATE:
            logger.info("Termination signal received, ending graph")

            # Log termination routing decision
            bus_log = _get_bus_log()
            if bus_log:
                bus_log.info(
                    "route_decision",
                    next_node=END,
                    next_nodes=[END],
                    reason="termination_signal",
                )

            return [END]

        # Check for human interaction
        if last_receiver in ("customer", "human"):
            logger.info(f"Message directed to {last_receiver}, handling human interaction")
            return [self._handle_human_interaction(last_message, state)]

        # Get pending messages by role (with direct message counts)
        pending, direct_counts = self._get_pending_messages_by_role(state)

        if not pending:
            # No pending messages - all work is done, end the graph
            logger.info("No pending messages for any role, ending graph")

            # Log END routing decision
            bus_log = _get_bus_log()
            if bus_log:
                bus_log.info(
                    "route_decision",
                    next_node=END,
                    next_nodes=[END],
                    reason="no_pending_messages",
                )

            return [END]

        # Priority order: workers first, coordinators last
        priority_order = [
            "plotwright",
            "lore_weaver",
            "scene_smith",
            "style_lead",
            "codex_curator",
            "researcher",
            "art_director",
            "illustrator",
            "audio_director",
            "audio_producer",
            "translator",
            "book_binder",
            "player_narrator",
            "gatekeeper",
            "showrunner",
        ]

        # Helper function to route to a role (checks dormancy, exec count, fairness)
        def try_route_to_role(
            role_id: str, is_direct: bool, skip_fairness: bool = False
        ) -> str | None:
            if role_id not in pending or not pending[role_id]:
                return None

            # Check dormancy
            if self.dormancy.is_dormant(role_id):
                logger.debug(f"Role {role_id} has messages but is dormant, skipping")
                return None

            # Check execution count - prevent infinite loops
            exec_count = self._role_execution_counts.get(role_id, 0)
            if exec_count >= self._max_role_executions:
                logger.warning(
                    f"Role {role_id} has executed {exec_count} times, "
                    "forcing termination to prevent infinite loop"
                )
                # Return special marker for termination
                return END

            # Fairness check - prevent same role running too many times consecutively
            if not skip_fairness and role_id == self._last_executed_role:
                if self._consecutive_executions >= self._max_consecutive:
                    # Check if other roles have pending work
                    other_roles_with_work = [
                        r for r in pending
                        if r != role_id and not self.dormancy.is_dormant(r)
                    ]
                    if other_roles_with_work:
                        logger.info(
                            f"Fairness: {role_id} has run {self._consecutive_executions}x "
                            f"consecutively, skipping for others: {other_roles_with_work}"
                        )
                        return None  # Skip this role, try others

            # Increment execution count
            self._role_execution_counts[role_id] = exec_count + 1
            self._total_executions += 1

            # Periodic reset: after N total executions, halve all per-role counts
            # This allows long-running sessions to continue making progress
            if self._total_executions >= self._reset_threshold:
                logger.info(
                    f"Progress checkpoint: {self._total_executions} total executions, "
                    f"halving per-role counts to allow continued progress"
                )
                for r in self._role_execution_counts:
                    self._role_execution_counts[r] = self._role_execution_counts[r] // 2
                self._total_executions = 0

            # Update fairness tracking
            if role_id == self._last_executed_role:
                self._consecutive_executions += 1
            else:
                self._consecutive_executions = 1
                self._last_executed_role = role_id

            msg_type = "direct" if is_direct else "broadcast"
            logger.info(
                f"Message bus: routing to {role_id} "
                f"({len(pending[role_id])} pending, {msg_type}, exec #{exec_count + 1})"
            )

            # Log routing decision with structured logging
            bus_log = _get_bus_log()
            if bus_log:
                pending_counts = {r: len(msgs) for r, msgs in pending.items() if msgs}
                bus_log.info(
                    "route_decision",
                    next_node=role_id,
                    next_nodes=[role_id],
                    pending_counts=pending_counts,
                    direct_counts=direct_counts,
                    message_count=len(pending[role_id]),
                    message_type=msg_type,
                    exec_count=exec_count + 1,
                    consecutive=self._consecutive_executions,
                )

            return role_id

        # Collect eligible roles for parallel execution (up to _max_parallel)
        eligible_roles: list[str] = []

        # PASS 1: Collect roles with DIRECT messages (addressed specifically)
        # Direct messages take priority over broadcasts
        for role_id in priority_order:
            if len(eligible_roles) >= self._max_parallel:
                break
            if direct_counts.get(role_id, 0) > 0:
                result = try_route_to_role(role_id, is_direct=True)
                if result == END:
                    return [END]
                if result:
                    eligible_roles.append(result)

        # PASS 2: Collect roles with only BROADCAST messages (if we have capacity)
        for role_id in priority_order:
            if len(eligible_roles) >= self._max_parallel:
                break
            if role_id in pending and direct_counts.get(role_id, 0) == 0:
                # Skip if already added in PASS 1
                if role_id in eligible_roles:
                    continue
                result = try_route_to_role(role_id, is_direct=False)
                if result == END:
                    return [END]
                if result:
                    eligible_roles.append(result)

        # If we found eligible roles, return them for parallel execution
        if eligible_roles:
            logger.info(f"Parallel routing to {len(eligible_roles)} roles: {eligible_roles}")
            bus_log = _get_bus_log()
            if bus_log:
                pending_counts = {r: len(msgs) for r, msgs in pending.items() if msgs}
                bus_log.info(
                    "parallel_route_decision",
                    next_nodes=eligible_roles,
                    parallel_count=len(eligible_roles),
                    pending_counts=pending_counts,
                    direct_counts=direct_counts,
                )
            return eligible_roles

        # All pending roles are dormant - route to showrunner to decide wake
        dormant_with_messages = [r for r in pending if self.dormancy.is_dormant(r)]
        if dormant_with_messages:
            # Only route to showrunner if it has pending messages to process
            if SHOWRUNNER in pending:
                logger.info(
                    f"Dormant roles have pending messages: {dormant_with_messages}, "
                    "routing to showrunner"
                )
                return [SHOWRUNNER]
            else:
                # SR has no pending messages - can't wake roles, end graph
                logger.warning(
                    f"Dormant roles {dormant_with_messages} have messages but "
                    "showrunner has no pending messages to wake them. Ending graph."
                )

                # Log END routing decision
                bus_log = _get_bus_log()
                if bus_log:
                    bus_log.info(
                        "route_decision",
                        next_node=END,
                        next_nodes=[END],
                        reason="dormant_roles_no_showrunner",
                        dormant_roles=dormant_with_messages,
                    )

                return [END]

        # No actionable pending messages - end graph
        logger.info("No actionable pending messages, ending graph")

        # Log END routing decision
        bus_log = _get_bus_log()
        if bus_log:
            bus_log.info(
                "route_decision",
                next_node=END,
                next_nodes=[END],
                reason="no_actionable_messages",
            )

        return [END]

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

    def _handle_human_interaction(self, message: Message, state: StudioState) -> str:
        """
        Handle human interaction for messages directed to customer.

        When a role sends human.question to customer, we need to:
        1. Display the question to the human user
        2. Collect their response
        3. Route it back to the sender with human.answer intent

        Args:
            message: The message with receiver="customer"
            state: Current studio state

        Returns:
            Next node to route to (typically the original sender)
        """
        sender = message.get("sender", SHOWRUNNER)
        intent = message.get("intent", "")
        content = message.get("content", "")

        if intent == "human.question":
            # Display question to console
            from rich.console import Console
            from rich.panel import Panel
            from rich.prompt import Prompt

            console = Console()

            # Display the question
            console.print(
                Panel(
                    f"[bold cyan]Question from {sender}:[/bold cyan]\n\n{content}",
                    title="🤔 Input Requested",
                    border_style="cyan",
                )
            )

            # Get user response
            response = Prompt.ask("[bold]Your response")

            # Create response message
            response_msg = {
                "sender": "customer",
                "receiver": sender,
                "intent": "human.answer",
                "content": response,
                "metadata": {
                    "responding_to": message.get("tu_id", ""),
                    "original_intent": intent,
                },
            }

            # Add response to state
            state.setdefault("messages", []).append(response_msg)

            # Route back to original sender
            return self._normalize_role_id(sender)

        # For other intents, just route back to sender
        logger.info(f"Customer received {intent}, routing back to {sender}")
        return self._normalize_role_id(sender)

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

        Returns an async function that:
        1. Traces role_started event
        2. Executes the base role node (supports both sync and async via inspection)
        3. Ensures output messages have proper envelope structure
        4. Traces all output messages
        5. Marks messages as consumed (per-role consumption tracking)
        6. Traces role_completed event

        The async wrapper allows LangGraph to manage concurrency while maintaining
        message bus semantics (role execution is driven by pending messages).

        Returns:
            An async callable that LangGraph will await automatically
        """
        base_node = self.node_factory.create_role_node(role_id)

        async def wrapped_node(state: StudioState) -> dict[str, Any]:
            # Trace role started
            start_message: Message = {
                "sender": role_id,
                "receiver": "trace",
                "intent": "role_started",
                "payload": {"role_name": role_id},
                "timestamp": datetime.now(UTC).isoformat(),
                "envelope": {"tu_id": state.get("tu_id", "")},
            }
            self._trace_message(start_message)

            # Log role execution start with structured logging
            bus_log = _get_bus_log()
            if bus_log:
                bus_log.info(
                    "role_start",
                    role=role_id,
                    tu_id=state.get("tu_id", ""),
                )

            # Execute base node - handle both sync and async nodes
            if asyncio.iscoroutinefunction(base_node):
                result = await base_node(state)
            else:
                result = base_node(state)

            # Ensure messages have proper envelope structure and trace them
            messages = result.get("messages", [])
            for msg in messages:
                if "envelope" not in msg:
                    msg["envelope"] = {}
                msg["envelope"]["tu_id"] = state.get("tu_id", "")
                msg["envelope"]["timestamp"] = datetime.now(UTC).isoformat()

                # If no receiver specified, default to showrunner
                if not msg.get("receiver"):
                    msg["receiver"] = SHOWRUNNER

                # Trace the message
                self._trace_message(msg)

            # Mark messages addressed TO this role as consumed by tracking them
            # We use a set of consumed message indices instead of a simple cursor
            # to allow multiple roles to have pending messages simultaneously
            consumed = set(state.get("_consumed_messages", set()))
            existing_messages = state.get("messages", [])
            set(self.get_available_roles())

            for idx, msg in enumerate(existing_messages):
                if idx in consumed:
                    continue  # Already consumed

                receiver = msg.get("receiver", "")
                if isinstance(receiver, dict):
                    receiver = receiver.get("role", "") or receiver.get("id", "")
                receiver = str(receiver).lower().replace(" ", "_")
                receiver_id = self._abbreviations.get(receiver, receiver)

                # Mark messages to THIS role as consumed
                if receiver_id == role_id:
                    consumed.add(idx)
                # Also consume broadcast messages for this role
                elif receiver in (BROADCAST, "*"):
                    # Add role-specific consumed marker for broadcasts
                    consumed.add((idx, role_id))

            result["_consumed_messages"] = consumed

            # Log consumption tracking via structured logging
            bus_log = _get_bus_log()
            if bus_log:
                consumed_indices = [idx for idx in consumed if isinstance(idx, int)]
                broadcast_consumed = [
                    item for item in consumed if isinstance(item, tuple)
                ]
                try:
                    bus_log.info(
                        "message_consumed",
                        role=role_id,
                        consumed_indices=consumed_indices,
                        broadcast_consumed_count=len(broadcast_consumed),
                        total_consumed=len(consumed),
                        message_count=len(existing_messages),
                    )
                except Exception:
                    # Silently ignore logging errors
                    pass

            # Trace role completed
            completed_message: Message = {
                "sender": role_id,
                "receiver": "trace",
                "intent": "role_completed",
                "payload": {
                    "role_name": role_id,
                    "insight": f"Generated {len(messages)} message(s)",
                },
                "timestamp": datetime.now(UTC).isoformat(),
                "envelope": {"tu_id": state.get("tu_id", "")},
            }
            self._trace_message(completed_message)

            # Log role execution completion with structured logging
            bus_log = _get_bus_log()
            if bus_log:
                bus_log.info(
                    "role_complete",
                    role=role_id,
                    message_count=len(messages),
                    tu_id=state.get("tu_id", ""),
                )

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

    @traceable(name="control_plane.run", tags=["control_plane"])
    async def run(
        self,
        human_input: str,
        context: dict[str, Any] | None = None,
        recursion_limit: int = 50,
    ) -> StudioState:
        """
        Run the studio with human input asynchronously (preferred entry point).

        Executes the protocol-driven mesh graph where:
        - Human input is wrapped in a human.directive message
        - The graph routes messages between roles via route_by_envelope
        - Roles process messages and emit new protocol messages
        - Execution continues until receiver="__terminate__"

        This method is async-first and should be called with await.
        For synchronous contexts, use run_sync() instead.

        Args:
            human_input: The user's request/directive
            context: Optional initial context
            recursion_limit: Max graph iterations (prevents infinite loops)

        Returns:
            Final studio state with all messages and accumulated context

        Raises:
            Exception: If graph execution fails (logged with full traceback)
        """
        # Reset execution counters for new run
        self._role_execution_counts.clear()
        self._total_executions = 0
        self._message_history.clear()

        # Initialize state
        initial_state = self.state_manager.initialize_state(
            tu_id=f"TU-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
            context=context or {},
            # loop_id intentionally omitted - SR will choose the appropriate playbook
        )

        # Surface the human directive in hot_sot for easy tool access (spec: SR should
        # route its interpretation of the customer ask).
        initial_state.setdefault("hot_sot", {})["customer_directives"] = human_input
        # Keep a direct copy for prompt rendering convenience
        initial_state["human_directive"] = human_input

        # Add human input as first message
        human_message: Message = {
            "sender": "human",
            "receiver": SHOWRUNNER,
            "intent": "human.directive",
            "payload": {"content": human_input},
            "timestamp": datetime.now(UTC).isoformat(),
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
            # Configure with thread_id for LangSmith tracing
            config = {
                "configurable": {
                    "thread_id": initial_state.get("tu_id", "default"),
                },
                "recursion_limit": recursion_limit,
            }

            # Use ainvoke for asynchronous graph execution
            final_state = await graph.ainvoke(initial_state, config=config)
            logger.info("Studio run completed successfully")
            return final_state

        except Exception as e:
            logger.error(f"Studio run failed: {e}")
            raise

    def run_sync(
        self,
        human_input: str,
        context: dict[str, Any] | None = None,
        recursion_limit: int = 50,
    ) -> StudioState:
        """
        Synchronous wrapper for run() - for CLI and synchronous contexts (backward compatibility).

        This method wraps the async run() using asyncio.run(), allowing synchronous code
        to invoke the async control plane. It blocks until the entire studio execution completes.

        Use this for:
        - CLI entry points
        - Synchronous test contexts
        - Legacy code integrations

        Prefer run() for async-capable contexts.

        Args:
            human_input: The user's request/directive
            context: Optional initial context
            recursion_limit: Max graph iterations (prevents infinite loops)

        Returns:
            Final studio state with all messages and accumulated context
        """
        return asyncio.run(self.run(human_input, context, recursion_limit))

    def clear_cache(self) -> None:
        """Clear graph cache and message history."""
        self._graph_cache.clear()
        self._message_history.clear()
        logger.debug("Cleared control plane cache")
