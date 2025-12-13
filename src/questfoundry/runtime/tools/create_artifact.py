"""Create artifact tool with validation for v4 runtime.

This tool combines artifact creation with schema validation, providing
a more robust interface than raw write_artifact. It validates the artifact
against its type schema before writing to hot_store.

Per meta/ spec, artifacts are created in draft state and must go through
lifecycle transitions to reach published/canon state.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class CreateArtifactTool(BaseTool):
    """Create a new artifact with schema validation.

    Use this tool to create artifacts in hot_store with automatic schema
    validation. This ensures artifacts conform to their type definitions
    before being written.

    The artifact is created in 'draft' status by default. Use
    request_lifecycle_transition to move it through the lifecycle
    (draft -> review -> approved -> canon).
    """

    name: str = "create_artifact"
    description: str = (
        "Create a new artifact with schema validation. "
        "Validates the artifact against its type schema before writing to hot_store. "
        "Input: artifact_id (unique ID), artifact_type (e.g., 'scene', 'act'), "
        "data (artifact fields as dict), status (optional, default 'draft')"
    )

    # Injected by registry
    studio: Any = Field(default=None, exclude=True)
    state: Any = Field(default=None, exclude=True)
    agent_id: str = Field(default="unknown", exclude=True)

    def _run(
        self,
        artifact_id: str,
        artifact_type: str,
        data: dict[str, Any],
        status: str = "draft",
    ) -> str:
        """Create an artifact with validation.

        Parameters
        ----------
        artifact_id : str
            Unique identifier for the artifact
        artifact_type : str
            Type of artifact (must match a defined artifact type)
        data : dict
            Artifact content fields
        status : str
            Initial lifecycle status (default: 'draft')

        Returns
        -------
        str
            JSON result with creation status and any validation errors
        """
        # Validate inputs
        if not artifact_id or not artifact_id.strip():
            return json.dumps({
                "success": False,
                "error": "artifact_id is required",
                "hint": "Provide a unique identifier like 'scene_001' or 'act_1'",
            })

        if not artifact_type or not artifact_type.strip():
            return json.dumps({
                "success": False,
                "error": "artifact_type is required",
                "hint": "Use consult_schema to see available artifact types",
            })

        if data is None:
            data = {}

        artifact_id = artifact_id.strip()
        artifact_type = artifact_type.strip().lower()

        # Check for existing artifact
        hot_store = self._get_hot_store()
        if artifact_id in hot_store:
            return json.dumps({
                "success": False,
                "error": f"Artifact '{artifact_id}' already exists",
                "hint": "Use write_artifact to update, or choose a different ID",
            })

        # Validate against schema
        validation_result = self._validate_schema(artifact_type, data)
        if not validation_result["valid"]:
            return json.dumps({
                "success": False,
                "error": "Schema validation failed",
                "validation_errors": validation_result.get("errors", []),
                "hint": validation_result.get("hint", "Fix the validation errors and try again"),
            })

        # Create the artifact
        artifact = self._create_artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            data=data,
            status=status,
        )

        # Write to hot_store
        hot_store[artifact_id] = artifact

        return json.dumps({
            "success": True,
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "status": status,
            "store": "hot_store",
            "message": f"Artifact '{artifact_id}' created successfully in draft state.",
            "next_steps": [
                "Review the artifact content",
                "Use request_lifecycle_transition to move to 'review' when ready",
            ],
        })

    def _get_hot_store(self) -> dict[str, Any]:
        """Get the hot_store from state."""
        if self.state is None:
            return {}

        if hasattr(self.state, "hot_store"):
            return self.state.hot_store

        if isinstance(self.state, dict):
            return self.state.setdefault("hot_store", {})

        return {}

    def _validate_schema(self, artifact_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """Validate artifact data against its type schema."""
        try:
            from questfoundry.runtime.validation import validate_artifact

            result = validate_artifact(artifact_type, data)
            return {
                "valid": result.get("success", False),
                "errors": self._format_errors(result),
                "hint": result.get("hint"),
            }
        except ImportError:
            # Validation module not available - pass through
            logger.warning("Validation module not available, skipping schema check")
            return {"valid": True, "errors": []}
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
            return {
                "valid": False,
                "errors": [{"field": "_validation", "error": str(e)}],
                "hint": "An error occurred during validation. Check the artifact type and data.",
            }

    def _format_errors(self, result: dict[str, Any]) -> list[dict[str, str]]:
        """Format validation errors into a consistent structure."""
        errors: list[dict[str, str]] = []

        for field in result.get("missing_fields", []):
            errors.append({
                "field": field,
                "error": "Required field is missing",
            })

        for field_info in result.get("invalid_fields", []):
            if isinstance(field_info, dict):
                errors.append({
                    "field": field_info.get("field", "unknown"),
                    "error": field_info.get("error", "Invalid value"),
                })
            else:
                errors.append({
                    "field": str(field_info),
                    "error": "Invalid value",
                })

        return errors

    def _create_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        data: dict[str, Any],
        status: str,
    ) -> Any:
        """Create an artifact instance."""
        # Try to use the runtime Artifact model
        try:
            from questfoundry.runtime.state import Artifact

            return Artifact(
                id=artifact_id,
                type=artifact_type,
                status=status,
                created_by=self.agent_id,
                data=data,
            )
        except ImportError:
            # Fallback to dict
            return {
                "id": artifact_id,
                "type": artifact_type,
                "status": status,
                "created_by": self.agent_id,
                "created_at": datetime.now(UTC).isoformat(),
                "data": data,
            }


def create_create_artifact_tool(
    studio: Any = None,
    state: Any = None,
    agent_id: str = "unknown",
) -> CreateArtifactTool:
    """Factory function to create a CreateArtifactTool.

    Args:
        studio: The loaded studio (for schema lookup)
        state: The current studio state (for hot_store access)
        agent_id: ID of the agent creating artifacts

    Returns:
        Configured CreateArtifactTool
    """
    tool = CreateArtifactTool()
    tool.studio = studio
    tool.state = state
    tool.agent_id = agent_id
    return tool
