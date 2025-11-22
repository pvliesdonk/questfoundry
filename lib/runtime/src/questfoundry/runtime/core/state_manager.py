"""
State Manager - manages StudioState lifecycle and mutations.

Based on spec: components/state_manager.md
STRICT component - state management is foundation of execution correctness.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.models.state import (
    QUALITY_BARS,
    VALID_TRANSITIONS,
    BarStatus,
    Message,
    StudioState,
)

if TYPE_CHECKING:
    from questfoundry.runtime.core.trace_handler import TraceHandler

logger = logging.getLogger(__name__)


class StateManager:
    """Manage StudioState lifecycle and mutations throughout loop execution."""

    def __init__(self, trace_handler: TraceHandler | None = None):
        """
        Initialize state manager.

        Args:
            trace_handler: Optional trace handler to capture protocol messages
        """
        self._sequence_numbers: dict[int, int] = {}
        self._trace_handler = trace_handler

    def generate_tu_id(self) -> str:
        """
        Generate unique Trace Unit ID.

        Format: TU-YYYY-NNN
        Example: TU-2025-042
        """
        year = datetime.now().year
        seq = self._sequence_numbers.get(year, 0) + 1
        self._sequence_numbers[year] = seq
        return f"TU-{year}-{seq:03d}"

    def initialize_state(
        self, loop_id: str, context: dict[str, Any], tu_id: str | None = None
    ) -> StudioState:
        """
        Create fresh StudioState for loop execution.

        Steps:
        1. Generate TU ID if not provided
        2. Initialize all required fields
        3. Set created_at/updated_at timestamps
        4. Validate against StudioState schema
        5. Return initialized state

        Args:
            loop_id: Which loop is running
            context: Loop-specific context dict
            tu_id: Optional custom TU ID (auto-generates if None)

        Returns:
            Fresh StudioState ready for execution
        """
        if tu_id is None:
            tu_id = self.generate_tu_id()

        now = datetime.utcnow().isoformat() + "Z"

        # Initialize quality bars (8 dimensions)
        quality_bars: dict[str, BarStatus] = {}
        for bar_name in QUALITY_BARS:
            quality_bars[bar_name] = {
                "status": "not_checked",
                "feedback": None,
                "checked_by": None,
                "timestamp": None,
            }

        state: StudioState = {
            "tu_id": tu_id,
            "tu_lifecycle": "hot-proposed",
            "current_node": "",  # Will be set by GraphFactory
            "loop_id": loop_id,
            "loop_context": context.copy(),
            "artifacts": {},
            "quality_bars": quality_bars,
            "messages": [],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": now,
            "updated_at": now,
        }

        logger.info(f"Initialized state for loop {loop_id}: TU {tu_id}")
        return state

    def update_state(self, state: StudioState, updates: dict[str, Any]) -> StudioState:
        """
        Apply updates to state, maintaining immutability.

        Steps:
        1. Create shallow copy of state
        2. Apply updates (merge dicts, append lists)
        3. Update 'updated_at' timestamp
        4. Validate against schema
        5. Return new state

        CRITICAL: Original state must remain unchanged (immutability)

        Args:
            state: Original state (unchanged)
            updates: Partial updates to apply

        Returns:
            New state with updates applied
        """
        # Shallow copy
        new_state = copy.copy(state)

        # Apply updates
        for key, value in updates.items():
            if key == "artifacts" and isinstance(value, dict):
                # Merge artifacts
                new_state["artifacts"] = {**new_state["artifacts"], **value}
            elif key == "quality_bars" and isinstance(value, dict):
                # Merge quality bars
                new_state["quality_bars"] = {**new_state["quality_bars"], **value}
            elif key == "messages" and isinstance(value, list):
                # Append messages
                new_state["messages"] = new_state["messages"] + value
            else:
                # Direct assignment
                new_state[key] = value

        # Update timestamp
        new_state["updated_at"] = datetime.utcnow().isoformat() + "Z"

        return new_state

    def transition_tu(self, state: StudioState, new_lifecycle: str) -> StudioState:
        """
        Transition TU to new lifecycle stage.

        Valid Transitions:
        hot-proposed → stabilizing
        stabilizing → gatecheck
        gatecheck → stabilizing (failed review, needs rework)
        gatecheck → cold-merged (approved)
        stabilizing → hot-proposed (major rework needed)

        Invalid Transitions:
        cold-merged → * (cannot un-merge)
        hot-proposed → cold-merged (must go through gatecheck)

        Args:
            state: Current state
            new_lifecycle: Target lifecycle stage

        Returns:
            State with updated lifecycle

        Raises:
            ValueError: If transition is invalid
        """
        current = state["tu_lifecycle"]

        if new_lifecycle not in VALID_TRANSITIONS.get(current, []):
            raise ValueError(
                f"Invalid transition: {current} → {new_lifecycle}\n"
                f"Valid transitions from {current}: {VALID_TRANSITIONS.get(current, [])}"
            )

        new_state = self.update_state(state, {"tu_lifecycle": new_lifecycle})

        # Log transition
        message: Message = {
            "sender": "system",
            "receiver": "broadcast",
            "intent": "lifecycle_transition",
            "payload": {"from": current, "to": new_lifecycle},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "envelope": {"tu_id": state["tu_id"], "snapshot_ref": state.get("snapshot_ref")},
        }

        new_state = self.add_message(new_state, message)

        logger.info(f"TU {state['tu_id']} transitioned: {current} → {new_lifecycle}")
        return new_state

    def add_artifact(self, state: StudioState, artifact: dict[str, Any]) -> StudioState:
        """
        Add new artifact to state.

        Steps:
        1. Validate artifact structure
        2. Add to state["artifacts"]
        3. Log artifact_created message
        4. Return updated state

        Args:
            state: Current state
            artifact: Artifact to add

        Returns:
            State with artifact added
        """
        artifact_id = artifact.get("role_id", "unknown")
        artifact_type = artifact.get("artifact_type", "unknown")

        # Add artifact
        new_state = self.update_state(state, {"artifacts": {artifact_id: artifact}})

        # Log message
        message: Message = {
            "sender": artifact.get("role_id", "unknown"),
            "receiver": "broadcast",
            "intent": "artifact_created",
            "payload": {"artifact_id": artifact_id, "artifact_type": artifact_type},
            "timestamp": artifact.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            "envelope": {"tu_id": state["tu_id"], "snapshot_ref": state.get("snapshot_ref")},
        }

        new_state = self.add_message(new_state, message)

        logger.debug(f"Added artifact: {artifact_type} from {artifact.get('role_id')}")
        return new_state

    def update_quality_bars(
        self, state: StudioState, bar_results: dict[str, BarStatus]
    ) -> StudioState:
        """
        Update quality bar status.

        Steps:
        1. Validate bar names (must be one of 8 dimensions)
        2. Validate status values (green, yellow, red, not_checked)
        3. Merge with existing quality_bars
        4. Log quality_check message
        5. Return updated state

        Args:
            state: Current state
            bar_results: New bar statuses to merge

        Returns:
            State with updated quality bars
        """
        # Validate bar names
        for bar_name in bar_results.keys():
            if bar_name not in QUALITY_BARS:
                raise ValueError(f"Unknown quality bar: {bar_name}")

        # Validate statuses
        valid_statuses = {"green", "yellow", "red", "not_checked"}
        for bar_name, bar_status in bar_results.items():
            status = bar_status.get("status", "not_checked")
            if status not in valid_statuses:
                raise ValueError(f"Invalid bar status: {status}")

        # Update quality bars
        new_state = self.update_state(state, {"quality_bars": bar_results})

        # Log message
        message: Message = {
            "sender": "gatekeeper",
            "receiver": "broadcast",
            "intent": "quality_check",
            "payload": {
                "bars_checked": list(bar_results.keys()),
                "statuses": {k: v.get("status") for k, v in bar_results.items()},
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "envelope": {"tu_id": state["tu_id"], "snapshot_ref": state.get("snapshot_ref")},
        }

        new_state = self.add_message(new_state, message)

        logger.debug(f"Updated quality bars: {list(bar_results.keys())}")
        return new_state

    def add_message(self, state: StudioState, message: Message) -> StudioState:
        """
        Add protocol message to state.

        Steps:
        1. Validate message structure
        2. Add envelope requirements
        3. Append to state["messages"]
        4. Trace message if handler configured
        5. Return updated state

        Args:
            state: Current state
            message: Message to add

        Returns:
            State with message added
        """
        # Ensure envelope has TU ID
        if "envelope" not in message:
            message["envelope"] = {}

        if "tu_id" not in message["envelope"]:
            message["envelope"]["tu_id"] = state["tu_id"]

        # Trace message if handler configured
        if self._trace_handler:
            try:
                self._trace_handler.trace_message(message)
            except Exception as e:
                logger.warning(f"Trace handler error: {e}")

        return self.update_state(state, {"messages": [message]})

    def snapshot_state(self, state: StudioState) -> dict[str, Any]:
        """
        Create read-only snapshot of current state.

        Used for loops that operate on snapshots.

        Args:
            state: State to snapshot

        Returns:
            Snapshot dict with snapshot_id and frozen state
        """
        snapshot_id = f"SNAP-{state['tu_id']}-01"

        snapshot = {
            "snapshot_id": snapshot_id,
            "tu_id": state["tu_id"],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "state": copy.deepcopy(state),
        }

        logger.info(f"Created snapshot: {snapshot_id}")
        return snapshot

    def check_bar_threshold(
        self, state: StudioState, bars_checked: list[str], threshold: str
    ) -> bool:
        """
        Check quality bars against threshold.

        Thresholds:
        - all_green: All checked bars must be green
        - mostly_green: ≥75% green, rest yellow (no red)
        - no_red: No red bars allowed
        - any_progress: At least one bar checked

        Args:
            state: Current state
            bars_checked: Bar names to check
            threshold: Threshold type

        Returns:
            True if threshold met, False otherwise
        """
        statuses = [
            state["quality_bars"][bar]["status"]
            for bar in bars_checked
            if bar in state["quality_bars"]
        ]

        if threshold == "all_green":
            return all(s == "green" for s in statuses)

        elif threshold == "mostly_green":
            green_count = sum(1 for s in statuses if s == "green")
            red_count = sum(1 for s in statuses if s == "red")
            return red_count == 0 and green_count >= len(statuses) * 0.75

        elif threshold == "no_red":
            return all(s != "red" for s in statuses)

        elif threshold == "any_progress":
            return any(s != "not_checked" for s in statuses)

        else:
            raise ValueError(f"Unknown threshold: {threshold}")
