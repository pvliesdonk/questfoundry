"""
Save Artifact tool implementation.

Allows agents to persist artifacts to stores. Validates schema,
checks exclusive writer policy, and assigns initial lifecycle state.
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)


class ExclusiveWriterPolicy(Enum):
    """
    Policy for handling exclusive writer violations.

    WARN: Log warning, include in result, but allow the write (open floor principle)
    BLOCK: Reject writes from non-designated agents
    """

    WARN = "warn"
    BLOCK = "block"


# Default policy - can be changed to BLOCK for stricter enforcement
DEFAULT_EXCLUSIVE_WRITER_POLICY = ExclusiveWriterPolicy.WARN


@register_tool("save_artifact")
class SaveArtifactTool(BaseTool):
    """
    Save an artifact to a store.

    Validates artifact against its type schema, checks store permissions,
    and persists to the project database.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute artifact save."""
        artifact_id: str | None = args.get("artifact_id")
        artifact_type_id = (
            args.get("artifact_type")
            or args.get("artifact_type_id")
            or self._infer_artifact_type_from_id(artifact_id)
        )
        data = args.get("data", {})
        store_id: str | None = args.get("store")

        if not isinstance(data, dict):
            return self._failure_result(
                "Artifact data must be an object",
                errors=[
                    self._error_entry(
                        "data",
                        "Artifact data must be a JSON object",
                        "Provide a dictionary of field values that align with the artifact schema.",
                    )
                ],
            )

        if not self._context.project:
            return self._failure_result(
                "No project available - cannot save artifact",
                errors=[
                    self._error_entry(
                        "project",
                        "Project storage is unavailable",
                        "Restart the session or ensure the runtime is initialized with a project.",
                    )
                ],
                fatal=True,
            )

        if not artifact_type_id:
            return self._failure_result(
                "Unable to determine artifact type",
                errors=[
                    self._error_entry(
                        "artifact_type",
                        "Artifact type is missing or could not be inferred",
                        "Pass artifact_type explicitly or use an artifact_id with a known prefix (e.g., section_*).",
                    )
                ],
            )

        artifact_type = self._get_artifact_type(artifact_type_id)
        if not artifact_type:
            return self._failure_result(
                f"Unknown artifact type: {artifact_type_id}",
                errors=[
                    self._error_entry(
                        "artifact_type",
                        f"'{artifact_type_id}' is not defined in the studio",
                        "Confirm the artifact type exists in domain/ and provide a valid identifier.",
                    )
                ],
            )

        resolved_store_id = self._resolve_store(artifact_type_id, store_id)
        workflow_warning = None
        if self._context.store_manager:
            allowed, reason = self._context.store_manager.validate_write(
                resolved_store_id, artifact_type_id
            )
            if not allowed:
                return self._failure_result(
                    reason or f"Cannot write to store: {resolved_store_id}",
                    errors=[
                        self._error_entry(
                            "store",
                            reason
                            or f"Store '{resolved_store_id}' rejects '{artifact_type_id}' artifacts",
                            "Select a store that lists this artifact type or update the domain's store permissions.",
                        )
                    ],
                )

            workflow_warning = self._check_exclusive_writer(resolved_store_id)
            if workflow_warning:
                logger.warning(workflow_warning)
                if DEFAULT_EXCLUSIVE_WRITER_POLICY == ExclusiveWriterPolicy.BLOCK:
                    return self._failure_result(
                        f"Exclusive writer violation: {workflow_warning}",
                        errors=[
                            self._error_entry(
                                "store",
                                workflow_warning,
                                "Let the designated producer write to this store or adjust the store's workflow intent.",
                            )
                        ],
                    )

        validation_errors, validation_warnings = self._validate_artifact_data(artifact_type, data)
        feedback_warnings = list(validation_warnings)
        if workflow_warning:
            feedback_warnings.append(
                self._warning_entry(
                    "store",
                    workflow_warning,
                    "Let the designated producer persist to this store when possible.",
                )
            )

        if validation_errors:
            return self._failure_result(
                f"Artifact validation failed: {len(validation_errors)} error(s)",
                errors=validation_errors,
                warnings=feedback_warnings,
                include_validation_errors=True,
                workflow_warning=workflow_warning,
            )

        if not artifact_id:
            artifact_id = self._generate_artifact_id(artifact_type_id)

        initial_state = self._get_initial_lifecycle_state(artifact_type)

        try:
            artifact = self._context.project.create_artifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type_id,
                data=data,
                store=resolved_store_id,
                created_by=self._context.agent_id,
            )

            if initial_state and initial_state != "draft":
                self._context.project.update_artifact(
                    artifact_id=artifact_id,
                    data={"_lifecycle_state": initial_state},
                )
                artifact["_lifecycle_state"] = initial_state

            result_data = {
                "artifact": artifact,
                "artifact_id": artifact_id,
                "store": resolved_store_id,
                "lifecycle_state": artifact.get("_lifecycle_state", "draft"),
                "feedback": self._build_feedback(True, warnings=feedback_warnings),
            }

            if workflow_warning:
                result_data["workflow_warning"] = workflow_warning

            return ToolResult(success=True, data=result_data)

        except Exception as e:  # pragma: no cover - defensive
            logger.exception(f"Failed to save artifact: {e}")
            return self._failure_result(
                f"Failed to save artifact: {e}",
                errors=[
                    self._error_entry(
                        None,
                        "Unexpected error while writing the artifact",
                        "Check the runtime logs for details and retry once the issue is resolved.",
                    )
                ],
                workflow_warning=workflow_warning,
            )

    def _resolve_store(self, artifact_type_id: str, explicit_store: str | None) -> str:
        """
        Resolve which store to use.

        Priority:
        1. Explicit store parameter
        2. StoreManager default for artifact type
        3. Artifact type's default_store
        4. "workspace" fallback
        """
        if explicit_store:
            return explicit_store

        # Check store manager
        if self._context.store_manager:
            return self._context.store_manager.get_default_store(artifact_type_id)

        # Check artifact type definition
        for at in self._context.studio.artifact_types:
            if at.id == artifact_type_id and at.default_store:
                return at.default_store

        # Fallback
        return "workspace"

    def _check_exclusive_writer(self, store_id: str) -> str | None:
        """
        Check if current agent violates exclusive writer policy.

        Returns warning message if violation detected, None otherwise.
        """
        if not self._context.store_manager or not self._context.agent_id:
            return None

        exclusive_producer = self._context.store_manager.get_exclusive_producer(store_id)
        if exclusive_producer and exclusive_producer != self._context.agent_id:
            return (
                f"Workflow deviation: {self._context.agent_id} writing to {store_id} "
                f"(exclusive to {exclusive_producer})"
            )

        return None

    def _validate_artifact_data(
        self, artifact_type: Any, data: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Validate artifact data against type schema."""
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if not artifact_type.fields:
            return errors, warnings

        for field in artifact_type.fields:
            if field.required and field.name not in data:
                guidance = field.description or "Provide this field as described in the schema."
                errors.append(
                    self._error_entry(
                        field.name,
                        f"Required field '{field.name}' is missing",
                        guidance,
                    )
                )

        for field in artifact_type.fields:
            if field.name in data:
                value = data[field.name]
                type_error = self._check_field_type(field, value)
                if type_error:
                    errors.append(type_error)

        return errors, warnings

    def _check_field_type(self, field: Any, value: Any) -> dict[str, Any] | None:
        """Check if a field value matches the expected type."""
        expected_type = field.type.value if getattr(field, "type", None) else "string"
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "text": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "ref": lambda v: isinstance(v, str),
        }

        checker = type_checks.get(expected_type)
        if checker and not checker(value):
            guidance = field.description or "Use the schema's expected data type for this field."
            return self._error_entry(
                field.name,
                (
                    f"Field '{field.name}' has wrong type. "
                    f"Expected {expected_type}, got {type(value).__name__}"
                ),
                guidance,
            )

        return None

    def _generate_artifact_id(self, artifact_type_id: str) -> str:
        """Generate a unique artifact ID."""
        short_uuid = uuid.uuid4().hex[:8]
        return f"{artifact_type_id}_{short_uuid}"

    def _get_initial_lifecycle_state(self, artifact_type: Any) -> str | None:
        """Get initial lifecycle state from artifact type definition."""
        if hasattr(artifact_type, "lifecycle") and artifact_type.lifecycle:
            lifecycle = artifact_type.lifecycle
            if hasattr(lifecycle, "initial_state"):
                initial_state = lifecycle.initial_state
                return str(initial_state) if initial_state else "draft"
        return "draft"

    def _get_artifact_type(self, artifact_type_id: str) -> Any | None:
        for at in self._context.studio.artifact_types:
            if at.id == artifact_type_id:
                return at
        return None

    def _failure_result(
        self,
        error_message: str,
        *,
        errors: list[dict[str, Any]] | None = None,
        warnings: list[dict[str, Any]] | None = None,
        fatal: bool = False,
        include_validation_errors: bool = False,
        workflow_warning: str | None = None,
    ) -> ToolResult:
        feedback = self._build_feedback(False, errors=errors, warnings=warnings)
        data: dict[str, Any] = {"feedback": feedback}
        if include_validation_errors:
            data["validation_errors"] = errors or []
        if workflow_warning:
            data["workflow_warning"] = workflow_warning
        return ToolResult(
            success=False,
            data=data,
            error=error_message,
            fatal=fatal,
        )

    def _build_feedback(
        self,
        valid: bool,
        *,
        errors: list[dict[str, Any]] | None = None,
        warnings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "valid": valid,
            "errors": errors or [],
            "warnings": warnings or [],
        }

    def _error_entry(
        self,
        field: str | None,
        error: str,
        guidance: str | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {"field": field, "error": error}
        if guidance:
            entry["guidance"] = guidance
        return entry

    def _warning_entry(
        self,
        field: str | None,
        warning: str,
        suggestion: str | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {"field": field, "warning": warning}
        if suggestion:
            entry["suggestion"] = suggestion
        return entry

    def _infer_artifact_type_from_id(self, artifact_id: str | None) -> str | None:
        if not artifact_id or not self._context.studio.artifact_types:
            return None

        candidates = sorted(
            (at.id for at in self._context.studio.artifact_types),
            key=len,
            reverse=True,
        )
        for candidate in candidates:
            if artifact_id.startswith(f"{candidate}_"):
                return candidate
        return None


@register_tool("update_artifact")
class UpdateArtifactTool(BaseTool):
    """
    Update an existing artifact.

    Merges provided data with existing artifact data.
    Respects store semantics (blocks updates to cold/append_only stores).
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute artifact update."""
        artifact_id = args.get("artifact_id")
        data = args.get("data", {})

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
                error="No project available - cannot update artifact",
            )

        # Get existing artifact
        existing = self._context.project.get_artifact(artifact_id)
        if not existing:
            return ToolResult(
                success=False,
                data={},
                error=f"Artifact not found: {artifact_id}",
            )

        # Check store semantics
        store_id = existing.get("_store")
        save_version = False
        if store_id and self._context.store_manager:
            store = self._context.store_manager.get_store(store_id)
            if store and not store.allows_updates():
                return ToolResult(
                    success=False,
                    data={},
                    error=(
                        f"Cannot update artifact in {store.semantics.value} store '{store_id}'. "
                        "Store does not allow updates."
                    ),
                )
            # Check if store requires version history
            if store and store.requires_version_history():
                save_version = True

        # Update artifact
        try:
            # Save version history before update for versioned stores
            version_saved = None
            if save_version:
                version_saved = self._context.project.save_version(
                    artifact_id=artifact_id,
                    created_by=self._context.agent_id,
                )
                logger.debug(f"Saved version {version_saved} for {artifact_id}")

            updated = self._context.project.update_artifact(
                artifact_id=artifact_id,
                data=data,
            )

            if updated:
                result_data = {
                    "artifact": updated,
                    "artifact_id": artifact_id,
                }
                if version_saved is not None:
                    result_data["version_saved"] = version_saved

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
            logger.exception(f"Failed to update artifact: {e}")
            return ToolResult(
                success=False,
                data={},
                error=f"Failed to update artifact: {e}",
            )


@register_tool("get_artifact")
class GetArtifactTool(BaseTool):
    """
    Retrieve artifacts by ID.

    Accepts a single ID or list of IDs. Always returns an array of artifacts
    for consistent response shape.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute artifact retrieval."""
        artifact_ids = args.get("artifact_ids")

        if not artifact_ids:
            return ToolResult(
                success=False,
                data={},
                error="artifact_ids is required",
            )

        if not self._context.project:
            return ToolResult(
                success=False,
                data={},
                error="No project available - cannot retrieve artifact",
            )

        # Normalize to list
        if isinstance(artifact_ids, str):
            artifact_ids = [artifact_ids]

        # Fetch all requested artifacts
        artifacts = []
        not_found = []
        for aid in artifact_ids:
            artifact = self._context.project.get_artifact(aid)
            if artifact:
                artifacts.append(artifact)
            else:
                not_found.append(aid)

        # Build response
        data: dict[str, Any] = {
            "artifacts": artifacts,
            "count": len(artifacts),
        }
        if not_found:
            data["not_found"] = not_found

        if artifacts:
            return ToolResult(
                success=True,
                data=data,
            )
        else:
            return ToolResult(
                success=False,
                data=data,
                error=f"No artifacts found for IDs: {', '.join(not_found)}",
            )


@register_tool("delete_artifact")
class DeleteArtifactTool(BaseTool):
    """
    Delete an artifact from a store.

    Respects store semantics (blocks deletes from cold/append_only stores).
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute artifact deletion."""
        artifact_id = args.get("artifact_id")

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
                error="No project available - cannot delete artifact",
            )

        # Get existing artifact to check store
        existing = self._context.project.get_artifact(artifact_id)
        if not existing:
            return ToolResult(
                success=False,
                data={},
                error=f"Artifact not found: {artifact_id}",
            )

        # Check store semantics
        store_id = existing.get("_store")
        if store_id and self._context.store_manager:
            store = self._context.store_manager.get_store(store_id)
            if store and not store.allows_deletes():
                return ToolResult(
                    success=False,
                    data={},
                    error=(
                        f"Cannot delete artifact from {store.semantics.value} store '{store_id}'. "
                        "Store does not allow deletions."
                    ),
                )

        # Delete artifact
        try:
            deleted = self._context.project.delete_artifact(artifact_id)

            if deleted:
                return ToolResult(
                    success=True,
                    data={
                        "artifact_id": artifact_id,
                        "deleted": True,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Failed to delete artifact: {artifact_id}",
                )

        except Exception as e:
            logger.exception(f"Failed to delete artifact: {e}")
            return ToolResult(
                success=False,
                data={},
                error=f"Failed to delete artifact: {e}",
            )


@register_tool("list_artifacts")
class ListArtifactsTool(BaseTool):
    """
    Query artifacts with filters.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute artifact listing."""
        artifact_type = args.get("artifact_type")
        store = args.get("store")
        lifecycle_state = args.get("lifecycle_state")
        limit = args.get("limit", 20)

        if not self._context.project:
            return ToolResult(
                success=False,
                data={},
                error="No project available - cannot list artifacts",
            )

        artifacts = self._context.project.query_artifacts(
            artifact_type=artifact_type,
            store=store,
            lifecycle_state=lifecycle_state,
            limit=limit,
        )

        # Return summary view (exclude large data fields)
        summaries = []
        for artifact in artifacts:
            summaries.append(
                {
                    "_id": artifact.get("_id"),
                    "_type": artifact.get("_type"),
                    "_version": artifact.get("_version"),
                    "_lifecycle_state": artifact.get("_lifecycle_state"),
                    "_store": artifact.get("_store"),
                    "_updated_at": artifact.get("_updated_at"),
                    "_created_by": artifact.get("_created_by"),
                }
            )

        return ToolResult(
            success=True,
            data={
                "artifacts": summaries,
                "count": len(summaries),
            },
        )
