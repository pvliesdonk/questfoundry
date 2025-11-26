"""
Orchestration Tools - Tools for Showrunner role coordination.

These tools allow the Showrunner to coordinate the studio:
- State management (snapshots, TU updates)
- Role management (wake/sleep roles)
- Quality gate triggering
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any, Annotated

from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools.base import _is_injected_arg_type
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model
from pydantic.fields import PydanticUndefined

from questfoundry.runtime.core.cold_store import ColdStore
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


class _StrictToolSchemaMixin:
    """Preserve args_schema config and drop injected fields from tool schema."""

    @property
    def tool_call_schema(self):  # type: ignore[override]
        args_schema = getattr(self, "args_schema", None)
        if args_schema is None or not isinstance(args_schema, type):
            return super().tool_call_schema  # pragma: no cover - fallback

        if hasattr(args_schema, "model_fields"):
            pruned_fields = {}
            for name, field in args_schema.model_fields.items():
                annotated_type = field.annotation
                if field.metadata:
                    annotated_type = Annotated[field.annotation, *field.metadata]
                if _is_injected_arg_type(annotated_type):
                    continue
                pruned_fields[name] = field
            if len(pruned_fields) == len(args_schema.model_fields):
                return args_schema

            field_defs: dict[str, tuple[Any, Field]] = {}
            for name, field in pruned_fields.items():
                default = field.default if field.default is not PydanticUndefined else ...
                field_defs[name] = (
                    field.annotation,
                    Field(default=default, description=field.description),
                )

            return create_model(  # type: ignore[misc]
                f"{args_schema.__name__}Public",
                __config__=args_schema.model_config,
                __module__=args_schema.__module__,
                **field_defs,
            )

        return super().tool_call_schema


class CreateSnapshot(_StrictToolSchemaMixin, BaseTool):
    """
    Create a snapshot of the current Hot SoT for traceability.

    The Showrunner uses this before significant state changes
    to enable rollback and audit trails.
    """

    name: str = "create_snapshot"
    description: str = (
        "Create a snapshot of the current Hot Source of Truth state. "
        "Returns a snapshot_ref ID that can be used to restore state. "
        "Use before significant changes for traceability."
    )
    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        label: str | None = Field(default=None, description="Optional snapshot label")
        state: Annotated[StudioState, InjectedToolArg] = Field(
            ..., description="Current studio state"
        )
        project_id: str | None = Field(default=None, description="Project id override")
    args_schema = Args

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}
    _state_manager: StateManager = PrivateAttr()
    _cold_store: ColdStore = PrivateAttr()

    def __init__(
        self,
        state_manager: StateManager | None = None,
        cold_store: ColdStore | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state_manager = state_manager or StateManager()
        self._cold_store = cold_store or ColdStore()

    def _run(
        self,
        label: str = "",
        state: StudioState | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a snapshot of current state."""
        if state is None:
            return {"success": False, "error": "State payload is required for create_snapshot"}

        # Resolve project ID
        pid = (
            project_id
            or state.get("loop_context", {}).get("project_id")
            or os.getenv("QF_PROJECT_ID", "default")
        )
        tu_id = state.get("tu_id", "unknown")

        # Create snapshot via StateManager
        snapshot = self._state_manager.snapshot_state(state)
        if label:
            snapshot["label"] = label

        # Persist to cold store
        self._cold_store.append_snapshot(pid, tu_id, snapshot)

        snapshot_id = snapshot.get("snapshot_id", f"SNAP-{tu_id}")
        logger.info(f"Created snapshot {snapshot_id} for project {pid}")

        return {
            "success": True,
            "snapshot_id": snapshot_id,
            "tu_id": tu_id,
            "project_id": pid,
            "label": label or None,
            "created_at": snapshot.get("created_at"),
        }


class UpdateTU(_StrictToolSchemaMixin, BaseTool):
    """
    Update the current Trace Unit's lifecycle state.

    The Showrunner uses this to transition TUs through their lifecycle:
    hot-proposed → stabilizing → gatecheck → cold-merged
    """

    name: str = "update_tu"
    description: str = (
        "Update the current Trace Unit's lifecycle state. "
        "Valid transitions: hot-proposed → stabilizing → gatecheck → cold-merged. "
        "Input: new_state (one of: stabilizing, gatecheck, cold-merged)"
    )
    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        new_state: str = Field(..., description="New TU lifecycle state")
        state: Annotated[StudioState, InjectedToolArg] = Field(
            ..., description="Current studio state"
        )
    args_schema = Args

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}
    _state_manager: StateManager = PrivateAttr()

    def __init__(self, state_manager: StateManager | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state_manager = state_manager or StateManager()

    def _run(
        self,
        new_state: str,
        state: StudioState | None = None,
    ) -> dict[str, Any]:
        """Transition the TU to a new lifecycle state."""
        valid_states = ["hot-proposed", "stabilizing", "gatecheck", "cold-merged"]
        if new_state not in valid_states:
            return {
                "success": False,
                "error": f"Invalid state: {new_state}. Valid states: {valid_states}",
            }

        if state is None:
            return {"success": False, "error": "State payload is required for update_tu"}

        current_lifecycle = state.get("tu_lifecycle", "unknown")
        tu_id = state.get("tu_id", "unknown")

        try:
            # Use StateManager to perform the transition (validates and persists)
            new_state_obj = self._state_manager.transition_tu(state, new_state)
            logger.info(f"TU {tu_id} transitioned: {current_lifecycle} → {new_state}")
            return {
                "success": True,
                "tu_id": tu_id,
                "previous_lifecycle": current_lifecycle,
                "new_lifecycle": new_state,
                "updated_state": new_state_obj,
            }
        except Exception as exc:
            logger.error(f"Failed to transition TU {tu_id}: {exc}")
            return {
                "success": False,
                "tu_id": tu_id,
                "current_lifecycle": current_lifecycle,
                "requested_lifecycle": new_state,
                "error": str(exc),
            }


# Module-level dormancy registry reference (set by ControlPlane at runtime)
_dormancy_registry: Any = None


def set_dormancy_registry(registry: Any) -> None:
    """Set the shared dormancy registry for orchestration tools."""
    global _dormancy_registry
    _dormancy_registry = registry


def get_dormancy_registry() -> Any:
    """Get the shared dormancy registry."""
    return _dormancy_registry


class WakeRole(_StrictToolSchemaMixin, BaseTool):
    """
    Wake a dormant role so it can participate in the current work.

    The Showrunner uses this to activate optional roles (e.g., illustrator,
    translator) when they're needed for specific tasks.
    """

    name: str = "wake_role"
    description: str = (
        "Wake a dormant role to make it available for work. "
        "Some roles (showrunner, gatekeeper) are always-on and cannot be dormant. "
        "Input: role_id (e.g., 'illustrator', 'translator')"
    )

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        role_id: str = Field(..., description="Role id to wake")
    args_schema = Args

    def _run(self, role_id: str) -> dict[str, Any]:
        """Wake a dormant role."""
        always_on = {"showrunner", "gatekeeper"}
        if role_id in always_on:
            return {
                "success": True,
                "message": f"Role '{role_id}' is always-on, no action needed",
            }

        registry = get_dormancy_registry()
        if registry is None:
            # No registry available - likely running outside ControlPlane context
            logger.warning(f"wake_role: No dormancy registry available for {role_id}")
            return {
                "success": True,
                "message": f"Role '{role_id}' wake requested (no registry)",
                "note": "Running without DormancyRegistry - assuming role is active",
            }

        was_dormant = registry.is_dormant(role_id)
        registry.wake(role_id)
        logger.info(f"Woke role '{role_id}' (was_dormant={was_dormant})")

        return {
            "success": True,
            "role_id": role_id,
            "was_dormant": was_dormant,
            "is_dormant": False,
            "message": f"Role '{role_id}' is now active",
        }


class TriggerGatecheck(_StrictToolSchemaMixin, BaseTool):
    """
    Trigger a quality gate check on the current work.

    The Showrunner uses this to invoke the Gatekeeper for validation
    before finalizing artifacts. Returns a message envelope to be sent
    to the Gatekeeper role.
    """

    name: str = "trigger_gatecheck"
    description: str = (
        "Trigger a quality gate check on the current Hot SoT. "
        "The Gatekeeper will validate against the 8 quality bars. "
        "Input: quality_bars (list of bars to check, or 'all' for full check)"
    )

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        quality_bars: str | list[str] = Field(
            default="all", description="Which quality bars to evaluate"
        )
        state: Annotated[StudioState | None, InjectedToolArg] = Field(
            default=None, description="Current studio state"
        )
    args_schema = Args

    def _run(
        self,
        quality_bars: str | list[str] = "all",
        state: StudioState | None = None,
    ) -> dict[str, Any]:
        """Trigger quality gate validation."""
        all_bars = [
            "Integrity",
            "Reachability",
            "Nonlinearity",
            "Gateways",
            "Style",
            "Determinism",
            "Presentation",
            "Accessibility",
        ]

        if quality_bars == "all":
            bars_to_check = all_bars
        elif isinstance(quality_bars, str):
            bars_to_check = [quality_bars]
        else:
            bars_to_check = quality_bars

        # Validate bar names
        invalid = [b for b in bars_to_check if b not in all_bars]
        if invalid:
            return {
                "success": False,
                "error": f"Invalid quality bars: {invalid}. Valid bars: {all_bars}",
            }

        tu_id = state.get("tu_id", "unknown") if state else "unknown"

        # Create a gatecheck request message for the Gatekeeper
        # The ControlPlane will route this based on the receiver field
        gatecheck_message = {
            "sender": "showrunner",
            "receiver": "gatekeeper",
            "intent": "gatecheck_request",
            "payload": {
                "bars_to_check": bars_to_check,
                "tu_id": tu_id,
                "request_type": "quality_validation",
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "envelope": {
                "tu_id": tu_id,
                "priority": "normal",
            },
        }

        logger.info(f"Triggered gatecheck for bars: {bars_to_check}")

        return {
            "success": True,
            "bars_requested": bars_to_check,
            "message": gatecheck_message,
            "note": "Message should be added to state.messages for routing to Gatekeeper",
        }


class SleepRole(BaseTool):
    """
    Put a role to sleep (dormant) to conserve resources.

    The Showrunner uses this to deactivate optional roles that are
    no longer needed for the current work.
    """

    name: str = "sleep_role"
    description: str = (
        "Put a role to sleep (dormant) to conserve resources. "
        "Always-on roles (showrunner, gatekeeper) cannot be put to sleep. "
        "Input: role_id (e.g., 'illustrator', 'translator')"
    )

    def _run(self, role_id: str) -> dict[str, Any]:
        """Put a role to sleep."""
        always_on = {"showrunner", "gatekeeper"}
        if role_id in always_on:
            return {
                "success": False,
                "error": f"Role '{role_id}' is always-on and cannot be put to sleep",
            }

        registry = get_dormancy_registry()
        if registry is None:
            # No registry available - likely running outside ControlPlane context
            logger.warning(f"sleep_role: No dormancy registry available for {role_id}")
            return {
                "success": True,
                "message": f"Role '{role_id}' sleep requested (no registry)",
                "note": "Running without DormancyRegistry - request noted but not enforced",
            }

        was_dormant = registry.is_dormant(role_id)
        registry.sleep(role_id)
        logger.info(f"Put role '{role_id}' to sleep (was_dormant={was_dormant})")

        return {
            "success": True,
            "role_id": role_id,
            "was_dormant": was_dormant,
            "is_dormant": True,
            "message": f"Role '{role_id}' is now dormant",
        }
