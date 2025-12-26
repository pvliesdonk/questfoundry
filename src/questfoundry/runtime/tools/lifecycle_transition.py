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
    from questfoundry.runtime.storage.store_manager import StoreManager

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

            # Check preconditions (artifact dependencies)
            preconditions = lifecycle_manager.get_preconditions(
                artifact_type, current_state, target_state
            )

            if preconditions:
                precondition_result = self._validate_preconditions(
                    artifact=artifact,
                    preconditions=preconditions,
                )

                if not precondition_result["satisfied"]:
                    return ToolResult(
                        success=False,
                        data={
                            "artifact_id": artifact_id,
                            "current_state": current_state,
                            "target_state": target_state,
                            "precondition_failures": precondition_result["failures"],
                        },
                        error=f"Preconditions not met: {precondition_result['summary']}",
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

        # Handle cold transitions specially - requires store migration
        if target_state == "cold":
            cold_result = self._handle_cold_transition(
                artifact_id=artifact_id,
                artifact=artifact,
                artifact_type=artifact_type,
                current_state=current_state,
            )
            if cold_result is not None:
                return cold_result

        # Check if transition has a target_store for store migration
        target_store = None
        if lifecycle_manager and artifact_type:
            lifecycle = lifecycle_manager.get_lifecycle(artifact_type)
            if lifecycle:
                target_store = lifecycle.get_transition_store(current_state, target_state)

        # Perform the transition
        try:
            update_data: dict[str, Any] = {"_lifecycle_state": target_state}

            # Include store migration if transition specifies target_store
            if target_store:
                current_store = artifact.get("_store", "workspace")
                update_data["_store"] = target_store
                logger.info(
                    f"Store migration: {artifact_id} {current_store} -> {target_store} "
                    f"(via transition to {target_state})"
                )

            updated = self._context.project.update_artifact(
                artifact_id=artifact_id,
                data=update_data,
            )

            if updated:
                logger.info(
                    f"Lifecycle transition: {artifact_id} {current_state} -> {target_state} "
                    f"(by {self._context.agent_id})"
                )

                result_data: dict[str, Any] = {
                    "result": "committed",
                    "artifact_id": artifact_id,
                    "previous_state": current_state,
                    "new_state": target_state,
                    "transitioned": True,
                    "transitioned_by": self._context.agent_id,
                }

                # Include store migration info if applicable
                if target_store:
                    result_data["previous_store"] = current_store
                    result_data["new_store"] = target_store

                return ToolResult(
                    success=True,
                    data=result_data,
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

    def _get_store_manager(self) -> StoreManager | None:
        """Get store manager from context."""
        return getattr(self._context, "store_manager", None)

    def _validate_preconditions(
        self,
        artifact: dict[str, Any],
        preconditions: list[Any],
    ) -> dict[str, Any]:
        """
        Validate transition preconditions against current project state.

        Args:
            artifact: The artifact being transitioned
            preconditions: List of TransitionPrecondition objects

        Returns:
            Dict with:
                satisfied: bool - True if all preconditions are met
                failures: list - Details of failed preconditions
                summary: str - Human-readable summary of failures
        """
        if not self._context.project:
            return {
                "satisfied": False,
                "failures": [{"type": "no_project", "message": "No project available"}],
                "summary": "No project available to check preconditions",
            }

        failures: list[dict[str, Any]] = []
        project = self._context.project

        for precondition in preconditions:
            # Handle requires_artifact precondition
            if precondition.requires_artifact:
                req = precondition.requires_artifact
                target_type = req.artifact_type
                match_field = req.match_field

                # Find artifacts of the required type
                matching_artifacts = project.list_artifacts(artifact_type=target_type)

                if not matching_artifacts:
                    failures.append(
                        {
                            "type": "requires_artifact",
                            "artifact_type": target_type,
                            "message": f"No '{target_type}' artifact exists",
                            "fix": f"Create a '{target_type}' artifact before this transition",
                        }
                    )
                    continue

                # If match_field specified, check for matching value
                if match_field:
                    source_value = artifact.get(match_field)
                    found_match = False
                    for candidate in matching_artifacts:
                        if candidate.get(match_field) == source_value:
                            found_match = True
                            break

                    if not found_match:
                        failures.append(
                            {
                                "type": "requires_artifact",
                                "artifact_type": target_type,
                                "match_field": match_field,
                                "expected_value": source_value,
                                "message": (
                                    f"No '{target_type}' artifact with "
                                    f"{match_field}='{source_value}'"
                                ),
                                "fix": (
                                    f"Create a '{target_type}' artifact with "
                                    f"{match_field}='{source_value}'"
                                ),
                            }
                        )

            # Handle field_references precondition
            if precondition.field_references:
                ref = precondition.field_references
                source_field = ref.source_field
                target_type = ref.target_artifact_type
                target_path = ref.target_path
                match_field = ref.match_field

                # Get source value from artifact
                source_value = artifact.get(source_field)
                if source_value is None:
                    failures.append(
                        {
                            "type": "field_references",
                            "source_field": source_field,
                            "message": f"Source field '{source_field}' is not set",
                            "fix": f"Set the '{source_field}' field before this transition",
                        }
                    )
                    continue

                # Find target artifact
                target_artifacts = project.list_artifacts(artifact_type=target_type)

                # Filter by match_field if specified
                if match_field:
                    match_value = artifact.get(match_field)
                    target_artifacts = [
                        a for a in target_artifacts if a.get(match_field) == match_value
                    ]

                if not target_artifacts:
                    failures.append(
                        {
                            "type": "field_references",
                            "target_artifact_type": target_type,
                            "message": f"No '{target_type}' artifact exists",
                            "fix": f"Create a '{target_type}' artifact first",
                        }
                    )
                    continue

                # Check if source_value exists in target_path
                # Parse target_path like "passages[].pid"
                found_in_any = False
                for target_artifact in target_artifacts:
                    if self._check_value_in_path(target_artifact, target_path, source_value):
                        found_in_any = True
                        break

                if not found_in_any:
                    failures.append(
                        {
                            "type": "field_references",
                            "source_field": source_field,
                            "source_value": source_value,
                            "target_artifact_type": target_type,
                            "target_path": target_path,
                            "message": (
                                f"Value '{source_value}' from '{source_field}' not found "
                                f"in {target_type}.{target_path}"
                            ),
                            "fix": (
                                f"Ensure '{source_value}' exists in the "
                                f"'{target_type}' artifact's {target_path}"
                            ),
                        }
                    )

        if failures:
            summary_parts = [f["message"] for f in failures]
            return {
                "satisfied": False,
                "failures": failures,
                "summary": "; ".join(summary_parts),
            }

        return {
            "satisfied": True,
            "failures": [],
            "summary": "",
        }

    def _check_value_in_path(
        self,
        artifact: dict[str, Any],
        path: str,
        value: Any,
    ) -> bool:
        """
        Check if a value exists at a JSONPath-like path in an artifact.

        Supports paths like:
        - "passages[].pid" - check pid field in each item of passages array
        - "name" - check top-level field

        Args:
            artifact: The artifact to search
            path: JSONPath-like expression
            value: Value to find

        Returns:
            True if value found at path
        """
        # Parse path like "passages[].pid"
        if "[]." in path:
            array_field, item_field = path.split("[].", 1)
            array_data = artifact.get(array_field, [])
            if not isinstance(array_data, list):
                return False

            for item in array_data:
                if isinstance(item, dict):
                    # Handle nested path like "pid"
                    if "." in item_field:
                        # Recursively check nested path
                        if self._check_value_in_path(item, item_field, value):
                            return True
                    elif item.get(item_field) == value:
                        return True
            return False

        # Simple field lookup
        if "." in path:
            parts = path.split(".", 1)
            nested = artifact.get(parts[0])
            if isinstance(nested, dict):
                return self._check_value_in_path(nested, parts[1], value)
            return False

        return bool(artifact.get(path) == value)

    def _handle_cold_transition(
        self,
        artifact_id: str,
        artifact: dict[str, Any],
        artifact_type: str | None,
        current_state: str,
    ) -> ToolResult | None:
        """
        Handle transition to cold state with store migration.

        Cold transitions require:
        1. Determining the target cold store from artifact type
        2. Verifying the caller is an exclusive writer for that store
        3. Idempotency check (artifact already in cold store)
        4. Updating both _lifecycle_state and _store atomically

        Returns:
            ToolResult if transition is handled (success or error),
            None if should fall through to normal transition logic
        """
        if not artifact_type:
            return ToolResult(
                success=False,
                data={},
                error="Cannot transition to cold: artifact has no type",
            )

        store_manager = self._get_store_manager()
        if not store_manager:
            # No store manager - cannot perform cold transition atomically
            logger.error(
                f"StoreManager not available, cannot perform cold transition for {artifact_id}"
            )
            return ToolResult(
                success=False,
                data={},
                error="StoreManager not available. Cannot perform cold transition.",
            )

        # 1. Determine target cold store
        target_store = store_manager.get_cold_store_for_artifact_type(artifact_type)
        if not target_store:
            return ToolResult(
                success=False,
                data={
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                },
                error=(
                    f"No cold store accepts artifact type '{artifact_type}'. "
                    "Check domain-v4/stores/*.json for artifact_types configuration."
                ),
            )

        # 2. Check exclusive writer permission
        agent_id = self._context.agent_id or ""
        if not store_manager.is_exclusive_writer(target_store, agent_id):
            exclusive_writers = store_manager.get_exclusive_writers(target_store)
            # Format writers list for human-readable error message
            if not exclusive_writers:
                writers_str = "the exclusive writers"  # Fallback
            elif len(exclusive_writers) == 1:
                writers_str = exclusive_writers[0]
            else:
                writers_str = f"{', '.join(exclusive_writers[:-1])} or {exclusive_writers[-1]}"
            return ToolResult(
                success=False,
                data={
                    "artifact_id": artifact_id,
                    "target_store": target_store,
                    "required_writers": exclusive_writers,
                    "current_agent": agent_id,
                },
                error=(
                    f"Only {writers_str} can promote to '{target_store}' store. "
                    f"Delegate to the exclusive writer to complete the cold transition."
                ),
            )

        # 3. Idempotency check - already in cold state and correct store?
        current_store = artifact.get("_store", "workspace")
        if current_state == "cold" and current_store == target_store:
            return ToolResult(
                success=True,
                data={
                    "artifact_id": artifact_id,
                    "already_cold": True,
                    "store": target_store,
                    "message": f"Artifact already in cold state in '{target_store}' store",
                },
            )

        # 4. Perform the transition with store migration
        # Note: project is already validated as non-None in execute() before this is called
        project = self._context.project
        if project is None:
            # Should never happen - checked in execute(), but handle gracefully
            return ToolResult(
                success=False,
                data={"artifact_id": artifact_id},
                error="Internal error: project context is unavailable for cold transition.",
            )

        try:
            updated = project.update_artifact(
                artifact_id=artifact_id,
                data={
                    "_lifecycle_state": "cold",
                    "_store": target_store,
                },
            )

            if updated:
                logger.info(
                    f"Cold transition: {artifact_id} {current_state} -> cold, "
                    f"store: {current_store} -> {target_store} (by {agent_id})"
                )

                return ToolResult(
                    success=True,
                    data={
                        "result": "committed",
                        "artifact_id": artifact_id,
                        "previous_state": current_state,
                        "new_state": "cold",
                        "previous_store": current_store,
                        "new_store": target_store,
                        "transitioned": True,
                        "transitioned_by": agent_id,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Failed to update artifact for cold transition: {artifact_id}",
                )

        except Exception as e:
            logger.exception(f"Failed cold transition for artifact: {e}")
            return ToolResult(
                success=False,
                data={},
                error=f"Failed cold transition: {e}",
            )

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
        # Import here to avoid circular dependency (models imports from tools)
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
        failed_bars = [br for br in bar_results if br.get("status") == "fail"]

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
