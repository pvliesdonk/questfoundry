"""Role tools - tools available to specialist roles.

These tools are used by specialist roles (not SR) to:
- Return results to SR when work is complete
- Read/write artifacts in hot_store
- Read from cold_store (canon lookup)
- Promote artifacts to cold_store (Gatekeeper only)

The return_to_sr tool is the "done" signal for role execution.
It requires the role to summarize its work for logging and SR decision-making.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from pydantic import Field

if TYPE_CHECKING:
    from questfoundry.runtime.stores import ColdStore

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


# =============================================================================
# Cold Store Tools
# =============================================================================


class ReadColdSot(BaseTool):
    """Read from the Cold Source of Truth (immutable canon storage).

    Use this to look up established canon, previously approved content,
    or player-safe material. Cold content has passed Gatekeeper validation
    and is safe for player-facing use.

    Available to ALL roles for canon lookup.
    """

    name: str = "read_cold_sot"
    description: str = (
        "Read from cold_store (immutable canon storage). "
        "Use for canon lookup, approved content, player-safe material. "
        "Input: key (section_id or 'list' to see available sections)"
    )

    # ColdStore is injected by executor
    cold_store: Any = Field(default=None)

    def _run(self, key: str) -> str:
        """Read from cold_store."""
        if self.cold_store is None:
            return json.dumps(
                {
                    "success": False,
                    "error": "Cold store not available",
                    "hint": "Cold store may not be initialized for this session.",
                }
            )

        if not key:
            return json.dumps(
                {
                    "success": False,
                    "error": "key is required",
                    "hint": "Use 'list' to see available sections, or provide an anchor.",
                }
            )

        # Handle 'list' command
        if key.lower() == "list":
            sections = self.cold_store.list_sections()
            snapshots = self.cold_store.list_snapshots()
            return json.dumps(
                {
                    "success": True,
                    "sections": sections[:50],  # Limit for readability
                    "section_count": len(sections),
                    "snapshots": snapshots[:10],
                    "snapshot_count": len(snapshots),
                }
            )

        # Handle 'latest_snapshot' command
        if key.lower() in ("latest_snapshot", "latest"):
            snapshot = self.cold_store.get_latest_snapshot()
            if snapshot is None:
                return json.dumps(
                    {
                        "success": False,
                        "error": "No snapshots exist yet",
                        "hint": "Content must be promoted to cold_store and snapshotted first.",
                    }
                )
            # Get section anchors for this snapshot
            section_anchors = self.cold_store.get_snapshot_section_anchors(snapshot.snapshot_id)
            return json.dumps(
                {
                    "success": True,
                    "snapshot_id": snapshot.snapshot_id,
                    "description": snapshot.description,
                    "section_count": snapshot.section_count,
                    "section_ids": section_anchors,  # Keep as section_ids for API compat
                    "created_at": snapshot.created_at.isoformat(),
                }
            )

        # Try to get section by anchor
        section = self.cold_store.get_section(key)
        if section is not None:
            return json.dumps(
                {
                    "success": True,
                    "section_id": section.anchor,  # Use anchor as external ID
                    "anchor": section.anchor,
                    "title": section.title,
                    "content": section.content,
                    "content_hash": section.content_hash,
                    "order": section.order,
                    "requires_gate": section.requires_gate,
                    "source_brief_id": section.source_brief_id,
                    "created_at": section.created_at.isoformat(),
                }
            )

        # Try to get snapshot by ID
        snapshot = self.cold_store.get_snapshot(key)
        if snapshot is not None:
            section_anchors = self.cold_store.get_snapshot_section_anchors(snapshot.snapshot_id)
            return json.dumps(
                {
                    "success": True,
                    "snapshot_id": snapshot.snapshot_id,
                    "description": snapshot.description,
                    "section_ids": section_anchors,
                    "manifest_hash": snapshot.manifest_hash,
                    "created_at": snapshot.created_at.isoformat(),
                }
            )

        # Not found
        available = self.cold_store.list_sections()[:20]
        return json.dumps(
            {
                "success": False,
                "error": f"Key '{key}' not found in cold_store",
                "available_sections": available,
                "hint": "Use 'list' to see all available sections.",
            }
        )


class PromoteToCanon(BaseTool):
    """Promote artifacts from hot_store to cold_store (Gatekeeper only).

    This tool is the ONLY mechanism for writing to cold_store.
    It should only be used AFTER all quality bars have passed.

    The promotion:
    1. Reads artifact(s) from hot_store
    2. Extracts player-safe content (no spoilers)
    3. Writes to cold_store as immutable section
    4. Optionally creates a snapshot

    IMPORTANT: This tool is restricted to Gatekeeper role.
    Other roles attempting to use it will receive an error.
    """

    name: str = "promote_to_canon"
    description: str = (
        "Promote hot_store artifacts to cold_store (GATEKEEPER ONLY). "
        "Use ONLY after all quality bars pass. "
        "Inputs: artifact_ids (list of hot_store keys to promote), "
        "create_snapshot (bool, whether to create snapshot after promotion), "
        "snapshot_description (optional description for snapshot)"
    )

    # Injected by executor
    state: dict[str, Any] = Field(default_factory=dict)
    cold_store: Any = Field(default=None)
    role_id: str = Field(default="unknown")

    def _run(
        self,
        artifact_ids: list[str],
        create_snapshot: bool = True,
        snapshot_description: str = "",
        **kwargs: Any,
    ) -> str:
        """Promote artifacts from hot_store to cold_store."""
        # Log any unexpected kwargs
        if kwargs:
            logger.debug(f"promote_to_canon received extra kwargs: {list(kwargs.keys())}")

        # CRITICAL: Only Gatekeeper can promote to canon
        if self.role_id.lower() != "gatekeeper":
            return json.dumps(
                {
                    "success": False,
                    "error": f"Role '{self.role_id}' cannot promote to canon",
                    "hint": "Only Gatekeeper can write to cold_store. "
                    "Request Gatekeeper to perform promotion after quality check.",
                }
            )

        if self.cold_store is None:
            return json.dumps(
                {
                    "success": False,
                    "error": "Cold store not available",
                }
            )

        if not artifact_ids:
            return json.dumps(
                {
                    "success": False,
                    "error": "artifact_ids is required",
                    "hint": "Provide list of hot_store artifact IDs to promote.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        promoted = []
        errors = []

        for artifact_id in artifact_ids:
            if artifact_id not in hot_store:
                errors.append(f"Artifact '{artifact_id}' not found in hot_store")
                continue

            artifact = hot_store[artifact_id]

            # Extract content for cold_store
            if hasattr(artifact, "model_dump"):
                artifact_data = artifact.model_dump()
            elif isinstance(artifact, dict):
                artifact_data = artifact
            else:
                artifact_data = {"value": artifact}

            # Extract player-safe content
            # For artifacts with 'data' field, use that as content
            if "data" in artifact_data and isinstance(artifact_data["data"], dict):
                content_data = artifact_data["data"]
            else:
                content_data = artifact_data

            # Extract title
            title = (
                content_data.get("title")
                or artifact_data.get("title")
                or artifact_id
            )

            # Convert to prose/string for cold section
            if isinstance(content_data, dict):
                # Try to get prose content
                content = content_data.get(
                    "content",
                    content_data.get(
                        "prose",
                        content_data.get("text", json.dumps(content_data, indent=2)),
                    ),
                )
            else:
                content = str(content_data)

            # Add to cold_store using new API
            try:
                self.cold_store.add_section(
                    anchor=artifact_id,
                    title=title,
                    content=content,
                    source_brief_id=artifact_id,  # Track lineage
                )
                promoted.append(artifact_id)
            except Exception as e:
                errors.append(f"Failed to promote '{artifact_id}': {e}")

        # Create snapshot if requested and we promoted something
        snapshot_id = None
        if create_snapshot and promoted:
            desc = snapshot_description or f"Promoted {len(promoted)} artifact(s)"
            snapshot_id = self.cold_store.create_snapshot(desc)

        return json.dumps(
            {
                "success": len(promoted) > 0,
                "promoted": promoted,
                "errors": errors if errors else None,
                "snapshot_id": snapshot_id,
            }
        )


# Factory functions for Cold Store tools


def read_cold_sot(cold_store: ColdStore | None) -> ReadColdSot:
    """Create ReadColdSot tool with injected cold_store."""
    tool = ReadColdSot()
    tool.cold_store = cold_store
    return tool


def promote_to_canon(
    state: dict[str, Any],
    cold_store: ColdStore | None,
    role_id: str,
) -> PromoteToCanon:
    """Create PromoteToCanon tool with injected state and cold_store.

    Note: Only Gatekeeper should be given this tool. The tool itself
    also validates role_id as a safety check.
    """
    tool = PromoteToCanon()
    tool.state = state
    tool.cold_store = cold_store
    tool.role_id = role_id
    return tool
