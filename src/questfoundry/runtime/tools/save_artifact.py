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
        artifact_type_id: str | None = args.get("artifact_type")
        data = args.get("data", {})
        artifact_id: str | None = args.get("artifact_id")
        store_id: str | None = args.get("store")

        # Validate required fields
        if not artifact_type_id:
            return ToolResult(
                success=False,
                data={},
                error="artifact_type is required",
            )

        # Validate we have a project
        if not self._context.project:
            return ToolResult(
                success=False,
                data={},
                error="No project available - cannot save artifact",
            )

        # Validate artifact type exists
        artifact_type = None
        for at in self._context.studio.artifact_types:
            if at.id == artifact_type_id:
                artifact_type = at
                break

        if not artifact_type:
            return ToolResult(
                success=False,
                data={},
                error=f"Unknown artifact type: {artifact_type_id}",
            )

        # Resolve store
        resolved_store_id = self._resolve_store(artifact_type_id, store_id)

        # Validate store accepts this artifact type
        if self._context.store_manager:
            allowed, reason = self._context.store_manager.validate_write(
                resolved_store_id, artifact_type_id
            )
            if not allowed:
                return ToolResult(
                    success=False,
                    data={},
                    error=reason or f"Cannot write to store: {resolved_store_id}",
                )

            # Check exclusive writer policy
            workflow_warning = self._check_exclusive_writer(resolved_store_id)
            if workflow_warning:
                logger.warning(workflow_warning)

                # Check policy - block or warn
                if DEFAULT_EXCLUSIVE_WRITER_POLICY == ExclusiveWriterPolicy.BLOCK:
                    return ToolResult(
                        success=False,
                        data={"workflow_violation": workflow_warning},
                        error=f"Exclusive writer violation: {workflow_warning}",
                    )
                # WARN policy: continue with save (open floor principle)
        else:
            workflow_warning = None

        # Validate artifact data against schema
        validation_errors = self._validate_artifact_data(artifact_type, data)
        if validation_errors:
            return ToolResult(
                success=False,
                data={"validation_errors": validation_errors},
                error=f"Artifact validation failed: {len(validation_errors)} error(s)",
            )

        # Generate artifact ID if not provided
        if not artifact_id:
            artifact_id = self._generate_artifact_id(artifact_type_id)

        # Get initial lifecycle state
        initial_state = self._get_initial_lifecycle_state(artifact_type)

        # Save to project
        try:
            artifact = self._context.project.create_artifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type_id,
                data=data,
                store=resolved_store_id,
                created_by=self._context.agent_id,
            )

            # Update lifecycle state if artifact type has lifecycle
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
            }

            # Include workflow warning if present (open floor principle)
            if workflow_warning:
                result_data["workflow_warning"] = workflow_warning

            return ToolResult(
                success=True,
                data=result_data,
            )

        except Exception as e:
            logger.exception(f"Failed to save artifact: {e}")
            return ToolResult(
                success=False,
                data={},
                error=f"Failed to save artifact: {e}",
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
    ) -> list[dict[str, str]]:
        """
        Validate artifact data against type schema.

        Returns list of validation errors.
        """
        errors: list[dict[str, str]] = []

        if not artifact_type.fields:
            return errors

        # Check required fields
        for field in artifact_type.fields:
            if field.required and field.name not in data:
                errors.append(
                    {
                        "field": field.name,
                        "error": f"Required field '{field.name}' is missing",
                    }
                )

        # Check field types
        for field in artifact_type.fields:
            if field.name in data:
                value = data[field.name]
                field_type = field.type.value if field.type else "string"
                type_error = self._check_field_type(field.name, value, field_type)
                if type_error:
                    errors.append(type_error)

        return errors

    def _check_field_type(
        self, field_name: str, value: Any, expected_type: str
    ) -> dict[str, str] | None:
        """Check if a field value matches the expected type."""
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
            return {
                "field": field_name,
                "error": (
                    f"Field '{field_name}' has wrong type. "
                    f"Expected {expected_type}, got {type(value).__name__}"
                ),
            }

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
    Retrieve an artifact by ID.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute artifact retrieval."""
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
                error="No project available - cannot retrieve artifact",
            )

        artifact = self._context.project.get_artifact(artifact_id)

        if artifact:
            return ToolResult(
                success=True,
                data={"artifact": artifact},
            )
        else:
            return ToolResult(
                success=False,
                data={},
                error=f"Artifact not found: {artifact_id}",
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
