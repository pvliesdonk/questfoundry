"""Role tools - tools available to specialist roles.

These tools are used by specialist roles (not SR) to:
- Return results to SR when work is complete
- Read/write artifacts in hot_store

The return_to_sr tool is the "done" signal for role execution.
It requires the role to summarize its work for logging and SR decision-making.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class ReturnToSR(BaseTool):
    """Return control to the Showrunner with a work summary.

    Call this when your assigned task is complete (or cannot proceed).
    You MUST provide a summary of work done for logging and SR decision-making.

    Status values:
    - completed: Task finished successfully
    - blocked: Cannot proceed, need different input or another role
    - needs_review: Work done but needs validation
    - error: Something went wrong
    """

    name: str = "return_to_sr"
    description: str = (
        "Return control to Showrunner with work summary. "
        "MUST be called when your task is complete. "
        "Inputs: status (completed|blocked|needs_review|error), "
        "message (summary of work done), "
        "artifacts (list of artifact IDs created/modified), "
        "recommendation (optional suggested next action)"
    )

    # Role ID is injected by executor
    role_id: str = Field(default="unknown")

    def _run(
        self,
        status: str,
        message: str,
        artifacts: list[str] | None = None,
        recommendation: str | None = None,
        **kwargs: Any,  # Accept extra args LLM might pass
    ) -> str:
        """Return DelegationResult to SR."""
        # Log any unexpected kwargs for debugging
        if kwargs:
            logger.debug(f"return_to_sr received extra kwargs: {list(kwargs.keys())}")
        # Validate status
        valid_statuses = {"completed", "blocked", "needs_review", "error"}
        if status not in valid_statuses:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Invalid status '{status}'",
                    "valid_statuses": list(valid_statuses),
                    "hint": "Use one of: completed, blocked, needs_review, error",
                }
            )

        # Validate message
        if not message or not message.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": "Message is required",
                    "hint": "Summarize what work you did (or why you couldn't proceed).",
                }
            )

        # Build DelegationResult
        result = {
            "role_id": self.role_id,
            "status": status,
            "message": message.strip(),
            "artifacts": artifacts or [],
            "recommendation": recommendation.strip() if recommendation else None,
        }

        return json.dumps(
            {
                "success": True,
                "delegation_result": result,
            }
        )


class ReadHotSot(BaseTool):
    """Read from the hot Source of Truth (mutable draft storage).

    Use this to read artifacts or data that other roles have created.
    You can read by artifact_id or by key path.
    """

    name: str = "read_hot_sot"
    description: str = (
        "Read from hot_store (mutable draft storage). "
        "Input: key (artifact_id or dot-path like 'current_tu.title')"
    )

    # State is injected by executor
    state: dict[str, Any] = Field(default_factory=dict)

    def _run(self, key: str) -> str:
        """Read from hot_store."""
        if not key:
            return json.dumps(
                {
                    "success": False,
                    "error": "key is required",
                }
            )

        hot_store = self.state.get("hot_store", {})

        # Try direct artifact lookup first
        if key in hot_store:
            artifact = hot_store[key]
            return json.dumps(
                {
                    "success": True,
                    "key": key,
                    "value": artifact.model_dump() if hasattr(artifact, "model_dump") else artifact,
                }
            )

        # Try dot-path navigation
        value = self._get_nested(hot_store, key)
        if value is not None:
            return json.dumps(
                {
                    "success": True,
                    "key": key,
                    "value": value if not hasattr(value, "model_dump") else value.model_dump(),
                }
            )

        # Not found - list available keys
        available = list(hot_store.keys())[:20]
        return json.dumps(
            {
                "success": False,
                "error": f"Key '{key}' not found in hot_store",
                "available_keys": available,
            }
        )

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get nested value by dot-path."""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


class WriteHotSot(BaseTool):
    """Write to the hot Source of Truth (mutable draft storage).

    Use this to create or update artifacts. For known artifact keys
    (hooks, briefs, scenes, etc.), automatically validates against
    artifact schema before writing. Returns LLM-friendly validation
    errors if data doesn't match schema.

    Returns {success: true, ...} when write succeeds.
    Returns {success: false, missing_fields: [...], invalid_fields: [...], hint: '...'}
    when validation fails - use consult_schema to check field requirements.
    """

    name: str = "write_hot_sot"
    description: str = (
        "Write to hot_store (mutable draft storage). "
        "For artifact keys (hooks, briefs, scenes, etc.), validates against schema. "
        "Inputs: key (artifact_id or path like 'hooks'), value (data to write). "
        "Returns success:false with missing_fields, invalid_fields, hint if validation fails."
    )

    # State is injected by executor
    state: dict[str, Any] = Field(default_factory=dict)

    # Role ID for tracking who wrote
    role_id: str = Field(default="unknown")

    def _run(self, key: str, value: Any, **kwargs: Any) -> str:
        """Write to hot_store with artifact validation."""
        # Log any unexpected kwargs for debugging
        if kwargs:
            logger.debug(f"write_hot_sot received extra kwargs: {list(kwargs.keys())}")

        if not key:
            return json.dumps(
                {
                    "success": False,
                    "error": "key is required",
                }
            )

        if value is None:
            return json.dumps(
                {
                    "success": False,
                    "error": "value is required",
                }
            )

        # Artifact validation: detect artifact type from key
        if isinstance(value, dict):
            validation_result = self._validate_artifact(key, value)
            if validation_result is not None:
                if not validation_result.get("success", True):
                    # Validation failed - return LLM-friendly feedback
                    return json.dumps(validation_result)
                # Use validated/normalized data
                value = validation_result.get("validated", value)

        hot_store = self.state.setdefault("hot_store", {})

        # Check if updating existing artifact
        is_update = key in hot_store

        # Handle array keys (hooks, briefs, etc.) - append to list
        top_key = key.split(".")[0]
        if top_key in {"hooks", "briefs", "scenes", "canon_entries", "gatecheck_reports"}:
            existing = hot_store.get(top_key, [])
            if not isinstance(existing, list):
                existing = []
            hot_store[top_key] = existing + [value]
        elif isinstance(value, dict) and "type" in value:
            from questfoundry.runtime.state import Artifact

            # Create or update artifact
            existing = hot_store.get(key)
            if existing and hasattr(existing, "model_dump"):
                # Update existing artifact
                existing_data = existing.model_dump()
                existing_data.update(value)
                existing_data["updated_at"] = datetime.now()
                artifact = Artifact(**existing_data)
            else:
                # Create new artifact
                artifact = Artifact(
                    id=key,
                    type=value.get("type", "unknown"),
                    status=value.get("status", "draft"),
                    created_by=self.role_id,
                    data=value.get("data", value),
                )
            hot_store[key] = artifact
        else:
            # Write raw value
            hot_store[key] = value

        return json.dumps(
            {
                "success": True,
                "action": "updated" if is_update else "created",
                "key": key,
            }
        )

    def _validate_artifact(self, key: str, value: dict[str, Any]) -> dict[str, Any] | None:
        """Validate artifact data if key maps to a known artifact type.

        Returns:
            None if no validation needed (unknown key)
            {"success": True, "validated": data} if valid
            {"success": False, ...} with LLM-friendly errors if invalid
        """
        try:
            from questfoundry.runtime.validation import detect_artifact_type, validate_artifact

            artifact_type = detect_artifact_type(key)
            if artifact_type is None:
                return None  # No validation for this key

            return validate_artifact(artifact_type, value)
        except ImportError:
            # Validation module not available
            logger.debug(f"Validation module not available, skipping validation for {key}")
            return None
        except Exception as e:
            # Validation setup error - log and continue without validation
            logger.warning(f"Artifact validation error for {key}: {e}")
            return None


# Factory functions for creating tool instances with injected state


def read_hot_sot(state: dict[str, Any]) -> ReadHotSot:
    """Create ReadHotSot tool with injected state."""
    tool = ReadHotSot()
    tool.state = state
    return tool


def write_hot_sot(state: dict[str, Any], role_id: str) -> WriteHotSot:
    """Create WriteHotSot tool with injected state and role_id."""
    tool = WriteHotSot()
    tool.state = state
    tool.role_id = role_id
    return tool
