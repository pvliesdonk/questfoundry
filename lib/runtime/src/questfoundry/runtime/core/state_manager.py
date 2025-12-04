"""
State Manager - manages StudioState lifecycle and mutations.

Based on spec: components/state_manager.md
STRICT component - state management is foundation of execution correctness.
"""

from __future__ import annotations

import copy
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.config import get_settings
from questfoundry.runtime.core.cold_store import ColdStore
from questfoundry.runtime.exceptions import StateError
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

    def __init__(
        self,
        trace_handler: TraceHandler | None = None,
        project_id: str | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        """
        Initialize state manager.

        Args:
            trace_handler: Optional trace handler to capture protocol messages
            project_id: Optional project identifier for storage and tracing
            project_root: Optional base directory for projects; when omitted,
                defaults to config paths.project_dir.
        """
        settings = get_settings()
        self._sequence_numbers: dict[int, int] = {}
        self._trace_handler = trace_handler

        # Project identity (used for Cold SoT persistence)
        self._project_id = project_id

        # Resolve base dir for projects (from centralized config)
        if project_root is not None:
            base_dir = Path(project_root).expanduser()
        else:
            # Get from centralized config (already handles env var fallback)
            base_dir = Path(settings.paths.project_dir).expanduser()

        self._cold_store = ColdStore(base_dir=base_dir)

    def generate_tu_id(self, role_code: str) -> str:
        """
        Generate unique Trace Unit ID following spec format.

        Format: TU-YYYY-MM-DD-ROLESEQ
        Example: TU-2025-12-04-SR01 (Showrunner, sequence 01)

        Args:
            role_code: 2-4 letter role abbreviation (SR, PW, LW, ST, CC, etc.)

        Returns:
            Properly formatted TU ID
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # Track sequence per role per day
        key = f"{date_str}-{role_code}"
        seq = self._sequence_numbers.get(key, 0) + 1
        self._sequence_numbers[key] = seq

        return f"TU-{date_str}-{role_code.upper()}{seq:02d}"

    def initialize_state(
        self,
        context: dict[str, Any],
        tu_id: str | None = None,
        loop_id: str | None = None,
        role_code: str = "SR",
    ) -> StudioState:
        """
        Create fresh StudioState for execution.

        Steps:
        1. Generate TU ID if not provided
        2. Initialize all required fields
        3. Set created_at/updated_at timestamps
        4. Validate against StudioState schema
        5. Return initialized state

        Args:
            context: Execution context dict
            tu_id: Optional custom TU ID (auto-generates if None)
            loop_id: Optional loop identifier (None = SR decides dynamically)
            role_code: Role code for TU ID generation (default: "SR" for showrunner)

        Returns:
            Fresh StudioState ready for execution
        """
        if tu_id is None:
            tu_id = self.generate_tu_id(role_code)

        # Resolve project identity for storage (from centralized config)
        # Preference: explicit __init__ arg > config (handles env var)
        config_settings = get_settings()
        project_id = self._project_id or config_settings.paths.project_id

        now = datetime.now(UTC).isoformat()

        # Initialize quality bars (8 dimensions)
        quality_bars: dict[str, BarStatus] = {}
        for bar_name in QUALITY_BARS:
            quality_bars[bar_name] = {
                "status": "not_checked",
                "feedback": None,
                "checked_by": None,
                "timestamp": None,
            }

        # Initialize Hot/Cold Sources of Truth
        # Hot: working drafts, proposals, hooks, WIP
        # Includes all schema-defined arrays plus common ad-hoc keys used by roles
        hot_sot: dict[str, Any] = {
            # Schema-defined arrays (studio_state.schema.json)
            "hooks": [],
            "tus": [],
            "canon_packs": [],
            "style_addenda": [],
            "research_memos": [],
            "art_plans": [],
            "audio_plans": [],
            "edit_notes": [],
            "canon_transfer_packages": [],
            # Legacy/convenience keys
            "topology_notes": None,
            "section_briefs": [],
            "draft_sections": [],
            "hooks_generated": [],
            "style_notes": None,
            "lore_notes": None,
            "curator_notes": None,
            # Common ad-hoc keys used by roles
            "customer_directives": [],
            "drafts": [],
            "sections": [],
            # Dict-type keys used by roles
            "canon": {},
            "style": {},
            "topology": {},
            "world_genesis_manifest": {},
        }

        # Cold: stable canon, export-safe content
        # Start from default skeleton, then overlay any persisted Cold SoT for this project.
        # Includes all schema-defined arrays (studio_state.schema.json)
        cold_sot: dict[str, Any] = {
            "current_snapshot": "",  # Empty string per schema, not None
            "snapshots": [],
            "codex_entries": [],
            "language_packs": [],
            "sections": [],
            # Dict-type keys
            "canon": {},
            "codex": {},
            "manuscript": {},
        }
        try:
            persisted = self._cold_store.load_cold(project_id)
            if persisted is not None:
                cold_sot.update(persisted)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Failed to load Cold SoT for project '%s': %s", project_id, e)

        state: StudioState = {
            "tu_id": tu_id,
            "tu_lifecycle": "hot-proposed",
            "current_node": "",  # Will be set by GraphFactory
            "loop_id": loop_id,
            "loop_context": context.copy(),
            "hot_sot": hot_sot,
            "cold_sot": cold_sot,
            "artifacts": {},  # DEPRECATED - keep for backwards compatibility
            "quality_bars": quality_bars,
            "messages": [],
            "role_conversations": {},
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": now,
            "updated_at": now,
            # Message bus tracking - tracks which messages have been consumed
            "_consumed_messages": set(),
            # Schema-required fields (studio_state.schema.json)
            "meta": {
                "project_id": project_id,
                "current_tu": tu_id,
                "current_loop": loop_id or "",
                "timestamp": now,
            },
            "protocol": {
                "correlation_id": tu_id,
                "message_history": [],
                "pending_messages": [],
            },
            "execution": {
                "iteration_count": 0,
                "current_node": "",
                "node_history": [],
                "errors": [],
                "warnings": [],
            },
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
        new_state["updated_at"] = datetime.now(UTC).isoformat()

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
            valid_transitions = VALID_TRANSITIONS.get(current, [])
            raise StateError(
                f"Invalid lifecycle transition: {current} → {new_lifecycle}",
                tu_id=state["tu_id"],
                current_state=current,
                suggestions=[
                    f"Valid transitions from '{current}': {', '.join(valid_transitions)}"
                    if valid_transitions
                    else f"'{current}' is a terminal state with no valid transitions",
                    "Check spec/components/state_manager.md for lifecycle documentation",
                ],
            )

        new_state = self.update_state(state, {"tu_lifecycle": new_lifecycle})

        # If TU entered cold-merged, persist Cold SoT snapshot for this project
        if new_lifecycle == "cold-merged":
            envelope = new_state.get("envelope", {})
            project_id = (
                envelope.get("project_id")
                or self._project_id
                or get_settings().paths.project_id
            )
            try:
                # Save full cold_sot and append a snapshot for audit/history
                self._cold_store.save_cold(project_id, new_state.get("cold_sot", {}))
                snapshot = self.snapshot_state(new_state)
                self._cold_store.append_snapshot(project_id, new_state["tu_id"], snapshot)
            except Exception as e:  # pragma: no cover - defensive
                logger.error(
                    "Failed to persist Cold SoT for project '%s' (TU %s): %s",
                    project_id,
                    new_state.get("tu_id"),
                    e,
                )

        # Log transition
        message: Message = {
            "sender": "system",
            "receiver": "broadcast",
            "intent": "lifecycle_transition",
            "payload": {"from": current, "to": new_lifecycle},
            "timestamp": datetime.now(UTC).isoformat(),
            "envelope": {"tu_id": state["tu_id"], "snapshot_ref": state.get("snapshot_ref")},
        }

        new_state = self.add_message(new_state, message)

        logger.info(f"TU {state['tu_id']} transitioned: {current} → {new_lifecycle}")
        return new_state

    def add_artifact(self, state: StudioState, artifact: dict[str, Any]) -> StudioState:
        """
        Add new artifact to state.

        DEPRECATED: Use write_hot_sot() or write_cold_sot() tools instead.
        This method writes to both artifacts (deprecated) and hot_sot for
        backward compatibility.

        Steps:
        1. Validate artifact structure
        2. Add to state["artifacts"] (deprecated) AND state["hot_sot"]
        3. Log artifact_created message
        4. Return updated state

        Migration path:
        - Use ReadHotSOT/WriteHotSOT tools for in-progress work
        - Use ReadColdSOT/WriteColdSOT tools for finalized content
        - state["artifacts"] will be removed in a future version

        Args:
            state: Current state
            artifact: Artifact to add

        Returns:
            State with artifact added
        """
        warnings.warn(
            "add_artifact() is deprecated. Use write_hot_sot() or write_cold_sot() "
            "tools based on artifact lifecycle. See state_tools.py.",
            DeprecationWarning,
            stacklevel=2,
        )

        artifact_id = artifact.get("role_id", "unknown")
        artifact_type = artifact.get("artifact_type", "unknown")

        # Build hot_sot key based on artifact type and role
        hot_sot_key = f"{artifact_type}.{artifact_id}"

        # Add to BOTH artifacts (deprecated) and hot_sot (forward compatible)
        new_state = self.update_state(
            state,
            {
                "artifacts": {artifact_id: artifact},
                "hot_sot": {hot_sot_key: artifact},
            },
        )

        # Log message
        message: Message = {
            "sender": artifact.get("role_id", "unknown"),
            "receiver": "broadcast",
            "intent": "artifact_created",
            "payload": {"artifact_id": artifact_id, "artifact_type": artifact_type},
            "timestamp": artifact.get("timestamp", datetime.now(UTC).isoformat()),
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
                raise StateError(
                    f"Unknown quality bar: '{bar_name}'",
                    tu_id=state["tu_id"],
                    suggestions=[
                        f"Valid quality bars: {', '.join(QUALITY_BARS)}",
                        "Check spec/components/state_manager.md for quality bar definitions",
                    ],
                )

        # Validate statuses
        valid_statuses = {"green", "yellow", "red", "not_checked"}
        for bar_name, bar_status in bar_results.items():
            status = bar_status.get("status", "not_checked")
            if status not in valid_statuses:
                raise StateError(
                    f"Invalid status '{status}' for quality bar '{bar_name}'",
                    tu_id=state["tu_id"],
                    suggestions=[
                        f"Valid statuses: {', '.join(valid_statuses)}",
                        "Quality bar status must be one of: green, yellow, red, not_checked",
                    ],
                )

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
            "timestamp": datetime.now(UTC).isoformat(),
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
            "created_at": datetime.now(UTC).isoformat(),
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
            raise StateError(
                f"Unknown quality bar threshold: '{threshold}'",
                tu_id=state["tu_id"],
                suggestions=[
                    "Valid thresholds: all_green, mostly_green, no_red, any_progress",
                    "Check edge evaluator configuration for correct threshold value",
                ],
            )
