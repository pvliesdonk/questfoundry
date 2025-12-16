"""
Lifecycle transition tool implementation.

Allows agents to request lifecycle state changes for artifacts.
Validates transitions against the lifecycle state machine and
enforces allowed_agents restrictions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

if TYPE_CHECKING:
    from questfoundry.runtime.storage.lifecycle import LifecycleManager

logger = logging.getLogger(__name__)


@register_tool("request_lifecycle_transition")
class RequestLifecycleTransitionTool(BaseTool):
    """
    Request a lifecycle state transition for an artifact.

    Validates the transition against the lifecycle state machine:
    - Checks if transition is defined
    - Checks if current agent is allowed to make the transition
    - Returns required validations if any

    Note: If transition requires_validation, the actual state change
    may be deferred until validations pass.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute lifecycle transition request."""
        artifact_id: str | None = args.get("artifact_id")
        target_state: str | None = args.get("target_state")
        force: bool = args.get("force", False)

        # Validate required fields
        if not artifact_id:
            return ToolResult(
                success=False,
                data={},
                error="artifact_id is required",
            )

        if not target_state:
            return ToolResult(
                success=False,
                data={},
                error="target_state is required",
            )

        # Validate we have a project
        if not self._context.project:
            return ToolResult(
                success=False,
                data={},
                error="No project available - cannot transition artifact",
            )

        # Get the artifact
        artifact = self._context.project.get_artifact(artifact_id)
        if not artifact:
            return ToolResult(
                success=False,
                data={},
                error=f"Artifact not found: {artifact_id}",
            )

        # Get current state
        current_state = artifact.get("_lifecycle_state", "draft")
        artifact_type = artifact.get("_type")

        # Check if already in target state
        if current_state == target_state:
            return ToolResult(
                success=True,
                data={
                    "artifact_id": artifact_id,
                    "current_state": current_state,
                    "target_state": target_state,
                    "transitioned": False,
                    "message": f"Artifact already in '{target_state}' state",
                },
            )

        # Get lifecycle manager if available
        lifecycle_manager = self._get_lifecycle_manager()

        # Validate transition
        if lifecycle_manager and artifact_type:
            allowed, reason = lifecycle_manager.validate_transition(
                artifact_type,
                current_state,
                target_state,
                agent_id=self._context.agent_id,
            )

            if not allowed:
                return ToolResult(
                    success=False,
                    data={
                        "artifact_id": artifact_id,
                        "current_state": current_state,
                        "target_state": target_state,
                    },
                    error=f"Transition not allowed: {reason}",
                )

            # Check for required validations
            required_validations = lifecycle_manager.get_required_validations(
                artifact_type, current_state, target_state
            )

            if required_validations and not force:
                # Return pending status with required validations
                return ToolResult(
                    success=True,
                    data={
                        "artifact_id": artifact_id,
                        "current_state": current_state,
                        "target_state": target_state,
                        "transitioned": False,
                        "status": "pending_validation",
                        "required_validations": required_validations,
                        "message": (
                            f"Transition requires validations: {', '.join(required_validations)}. "
                            "Use force=true to bypass (not recommended)."
                        ),
                    },
                )

            if required_validations and force:
                logger.warning(
                    f"Forcing transition {current_state} -> {target_state} for {artifact_id} "
                    f"without validations: {required_validations}"
                )

        # Perform the transition
        try:
            updated = self._context.project.update_artifact(
                artifact_id=artifact_id,
                data={"_lifecycle_state": target_state},
            )

            if updated:
                logger.info(
                    f"Lifecycle transition: {artifact_id} {current_state} -> {target_state} "
                    f"(by {self._context.agent_id})"
                )

                return ToolResult(
                    success=True,
                    data={
                        "artifact_id": artifact_id,
                        "previous_state": current_state,
                        "current_state": target_state,
                        "transitioned": True,
                        "transitioned_by": self._context.agent_id,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Failed to update artifact: {artifact_id}",
                )

        except Exception as e:
            logger.exception(f"Failed to transition artifact: {e}")
            return ToolResult(
                success=False,
                data={},
                error=f"Failed to transition artifact: {e}",
            )

    def _get_lifecycle_manager(self) -> LifecycleManager | None:
        """
        Get lifecycle manager from context.

        Note: LifecycleManager is not yet in ToolContext.
        This will be added when we integrate with the runtime.
        For now, return None to allow any transition.
        """
        # TODO: Add lifecycle_manager to ToolContext
        # return self._context.lifecycle_manager
        return getattr(self._context, "lifecycle_manager", None)


@register_tool("get_lifecycle_state")
class GetLifecycleStateTool(BaseTool):
    """
    Get the current lifecycle state of an artifact.

    Returns the current state and available transitions.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute lifecycle state retrieval."""
        artifact_id: str | None = args.get("artifact_id")

        if not artifact_id:
            return ToolResult(
                success=False,
                data={},
                error="artifact_id is required",
            )

        if not self._context.project:
            return ToolResult(
                success=False,
                data={},
                error="No project available - cannot get lifecycle state",
            )

        # Get the artifact
        artifact = self._context.project.get_artifact(artifact_id)
        if not artifact:
            return ToolResult(
                success=False,
                data={},
                error=f"Artifact not found: {artifact_id}",
            )

        current_state = artifact.get("_lifecycle_state", "draft")
        artifact_type = artifact.get("_type")

        result_data: dict[str, Any] = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "current_state": current_state,
        }

        # Get available transitions if lifecycle manager available
        lifecycle_manager = self._get_lifecycle_manager()
        if lifecycle_manager and artifact_type:
            lifecycle = lifecycle_manager.get_lifecycle(artifact_type)
            if lifecycle:
                # Get valid transitions for current agent
                valid_transitions = lifecycle.get_valid_transitions(
                    current_state, agent_id=self._context.agent_id
                )
                result_data["available_transitions"] = [
                    {
                        "target_state": t.to_state,
                        "requires_validation": t.requires_validation,
                    }
                    for t in valid_transitions
                ]

                # Check if current state is terminal
                result_data["is_terminal"] = lifecycle.is_terminal_state(current_state)

        return ToolResult(
            success=True,
            data=result_data,
        )

    def _get_lifecycle_manager(self) -> LifecycleManager | None:
        """Get lifecycle manager from context."""
        return getattr(self._context, "lifecycle_manager", None)
