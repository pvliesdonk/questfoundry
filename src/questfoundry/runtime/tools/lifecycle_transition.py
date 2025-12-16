"""
Lifecycle transition tool implementation.

Allows agents to request lifecycle state changes for artifacts.
Validates transitions against the lifecycle state machine and
enforces allowed_agents restrictions.

Transitions with `requires_validation` run quality bar checks
and commit/reject based on results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import TOOL_IMPLEMENTATIONS, register_tool

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

            if required_validations and force:
                logger.warning(
                    f"Forcing transition {current_state} -> {target_state} for {artifact_id} "
                    f"without validations: {required_validations}"
                )
            elif required_validations:
                # Run validations
                validation_result = await self._run_validations(
                    artifact_id, artifact_type, artifact, required_validations
                )

                if not validation_result["passed"]:
                    # Reject transition
                    return ToolResult(
                        success=True,
                        data={
                            "result": "rejected",
                            "artifact_id": artifact_id,
                            "current_state": current_state,
                            "target_state": target_state,
                            "transitioned": False,
                            "validation_results": validation_result["results"],
                            "rejection_reason": validation_result["reason"],
                            "guidance": validation_result["guidance"],
                        },
                    )

                # Validations passed - continue to commit
                logger.info(f"Validations passed for {artifact_id}: {required_validations}")

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
                        "result": "committed",
                        "artifact_id": artifact_id,
                        "previous_state": current_state,
                        "new_state": target_state,
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
        """Get lifecycle manager from context."""
        return getattr(self._context, "lifecycle_manager", None)

    async def _run_validations(
        self,
        artifact_id: str,
        artifact_type: str,
        artifact: dict[str, Any],
        required_validations: list[str],
    ) -> dict[str, Any]:
        """
        Run required validation checks on an artifact.

        Args:
            artifact_id: The artifact to validate
            artifact_type: The artifact type ID
            artifact: The artifact data
            required_validations: List of quality bar names to check

        Returns:
            Dict with:
                passed: bool - True if all validations pass
                results: list - Detailed results per validation
                reason: str | None - Rejection reason if failed
                guidance: str | None - Fix guidance if failed
        """
        # Try to get validate_artifact tool implementation
        validate_tool_class = TOOL_IMPLEMENTATIONS.get("validate_artifact")
        if not validate_tool_class:
            # No validate tool - assume validations pass (permissive)
            logger.warning(
                f"validate_artifact tool not found - skipping validations for {artifact_id}"
            )
            return {
                "passed": True,
                "results": [],
                "reason": None,
                "guidance": None,
            }

        # Create a mock tool definition for validate_artifact
        from questfoundry.runtime.models import Tool

        validate_def = Tool(
            id="validate_artifact",
            name="validate_artifact",
            description="Validate artifact against quality bars",
            timeout_ms=30000,
        )

        # Instantiate the validate tool with current context
        validate_tool = validate_tool_class(validate_def, self._context)

        # Extract user data from artifact (exclude system fields)
        artifact_data = {k: v for k, v in artifact.items() if not k.startswith("_")}

        # Run validation
        result = await validate_tool.execute(
            {
                "artifact_id": artifact_id,
                "artifact_type_id": artifact_type,
                "artifact_data": artifact_data,
                "validation_mode": "bars_only",  # Focus on quality bars
                "bars_to_check": required_validations,
            }
        )

        if not result.success:
            # Validation tool itself failed
            return {
                "passed": False,
                "results": [],
                "reason": f"Validation failed: {result.error}",
                "guidance": "Check artifact data and retry",
            }

        # Check bar results
        bar_results = result.data.get("bar_results", [])
        failed_bars = [br for br in bar_results if br.get("status") == "red"]

        if failed_bars:
            # Collect failure details
            failed_names = [br["bar"] for br in failed_bars]
            fixes = [br.get("smallest_fix") for br in failed_bars if br.get("smallest_fix")]

            return {
                "passed": False,
                "results": bar_results,
                "reason": f"Quality bar failures: {', '.join(failed_names)}",
                "guidance": "; ".join(fixes) if fixes else "Address the failing quality bars",
            }

        # All validations passed
        return {
            "passed": True,
            "results": bar_results,
            "reason": None,
            "guidance": None,
        }


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
