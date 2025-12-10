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
from pydantic import Field, ValidationError

if TYPE_CHECKING:
    from questfoundry.runtime.stores import ColdStore

logger = logging.getLogger(__name__)

# ============================================================================
# Artifact Schema Detection & Validation
# ============================================================================
# Cold promotion config is generated from domain definitions.
# See COLD_PROMOTION_CONFIG in questfoundry.generated.models.artifacts
#
# The config maps artifact class names to their cold promotion settings:
# - content_field: The field containing prose content to extract
# - requires_content: Whether empty content should fail validation


def _detect_artifact_type(data: dict[str, Any]) -> tuple[str | None, Any | None, list[str]]:
    """Detect artifact type by validating against known schemas.

    Tries each cold-store-eligible artifact model and returns the first that validates.

    Returns:
        Tuple of (artifact_type, validated_model, validation_errors).
        If detection fails, artifact_type and validated_model are None,
        and validation_errors contains details about why each schema failed.
    """
    # Lazy import to avoid circular dependencies
    from questfoundry.generated.models.artifacts import (
        Act,
        CanonEntry,
        Chapter,
        Character,
        Event,
        Fact,
        Item,
        Location,
        Relationship,
        Scene,
        Timeline,
    )

    # Order by specificity: more constrained schemas first
    # Chapter must come before Act (both have title+sequence, but Chapter has more fields)
    models = [
        ("Scene", Scene),
        ("CanonEntry", CanonEntry),
        ("Character", Character),
        ("Location", Location),
        ("Item", Item),
        ("Event", Event),
        ("Fact", Fact),
        ("Chapter", Chapter),  # Before Act - has act_id, scenes, summary
        ("Act", Act),
        ("Timeline", Timeline),
        ("Relationship", Relationship),
    ]

    validation_errors: list[str] = []

    for type_name, model_class in models:
        try:
            validated = model_class(**data)
            return type_name, validated, []
        except ValidationError as e:
            # Record why this schema didn't match
            errors = "; ".join(err["msg"] for err in e.errors()[:3])  # First 3 errors
            validation_errors.append(f"{type_name}: {errors}")

    return None, None, validation_errors


def _extract_content_for_cold(
    artifact_type: str,
    validated_model: Any,
    artifact_id: str,
) -> tuple[str | None, str | None]:
    """Extract the content field value for cold_store promotion.

    Uses PROMOTABLE_ARTIFACTS to check if the artifact can be promoted.
    Content extraction is type-dependent:
    - Scene/CanonEntry: Extract from 'content' field
    - Act/Chapter: Structural, no content to extract (returns empty string)
    - Others: Try common content fields

    Returns:
        Tuple of (content_value, error_message).
        If extraction succeeds, error_message is None.
        If extraction fails, content_value is None and error_message explains why.
    """
    from questfoundry.generated.models.artifacts import PROMOTABLE_ARTIFACTS

    # Normalize type for comparison (key-based detection returns lowercase)
    artifact_type_lower = artifact_type.lower()
    promotable_lower = {t.lower() for t in PROMOTABLE_ARTIFACTS}

    # Check if this artifact type is promotable
    if artifact_type_lower not in promotable_lower:
        return None, (
            f"Artifact type '{artifact_type}' cannot be promoted to cold_store. "
            f"Promotable types: {', '.join(sorted(PROMOTABLE_ARTIFACTS))}"
        )

    # Structural artifacts don't have prose content
    structural_types = {"act", "chapter", "choice", "gate"}
    if artifact_type_lower in structural_types:
        # These artifacts store metadata, not prose content
        # Return description or empty string
        content = getattr(validated_model, "description", None) or ""
        return content, None

    # Content-bearing artifacts: try common content fields
    content_fields = ["content", "statement", "description"]
    for field in content_fields:
        content = getattr(validated_model, field, None)
        if content:
            return content, None

    # No content found - this may be OK for some types
    logger.warning(
        f"Artifact '{artifact_id}' ({artifact_type}) has no content field. "
        f"Promoting with empty content."
    )
    return "", None


class ReturnToSR(BaseTool):
    """Return control to the Showrunner with a work summary.

    Call this when your assigned task is complete (or cannot proceed).
    You MUST provide a summary of work done for logging and SR decision-making.

    Status values (3 simple statuses):
    - completed: Work finished (success/failure details go in message)
    - blocked: Cannot proceed (need external input or another role)
    - error: Something broke internally
    """

    name: str = "return_to_sr"
    description: str = (
        "Return control to Showrunner with work summary. "
        "MUST be called when your task is complete. "
        "Inputs: status (completed|blocked|error), "
        "message (summary of work done - include success/failure details), "
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

        # Validate status (3 simple statuses)
        # - completed: work done (success or failure details go in message)
        # - blocked: cannot proceed (missing dependency, need external help)
        # - error: something broke internally
        valid_statuses = {"completed", "blocked", "error"}
        if status not in valid_statuses:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Invalid status '{status}'",
                    "error_count": 1,
                    "invalid_fields": [
                        {
                            "field": "status",
                            "provided": status,
                            "issue": f"'{status}' is not a valid status value",
                        }
                    ],
                    "valid_statuses": sorted(valid_statuses),
                    "hint": (
                        "Use 'completed' when work is done (success or failure details in message), "
                        "'blocked' if you cannot proceed and need external help, "
                        "'error' if something broke. "
                        "Use 'recommendation' field for suggested next steps."
                    ),
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
    """Promote artifacts from hot_store to cold_store.

    This tool is the ONLY mechanism for writing to cold_store.
    It should only be used AFTER Gatekeeper has validated quality bars.

    The promotion:
    1. Reads artifact(s) from hot_store
    2. Extracts player-safe content (no spoilers)
    3. Writes to cold_store as immutable section
    4. Optionally creates a snapshot

    IMPORTANT: This tool is restricted to Lorekeeper role only.
    Workflow: Gatekeeper validates → SR delegates to Lorekeeper → Lorekeeper promotes.
    Other roles attempting to use it will receive an error.
    """

    name: str = "promote_to_canon"
    description: str = (
        "Promote hot_store artifacts to cold_store (LOREKEEPER ONLY). "
        "Use ONLY after Gatekeeper validates quality bars. "
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

        # Only Lorekeeper can promote to canon
        # - Gatekeeper validates/approves, then SR delegates to Lorekeeper for promotion
        # - Lorekeeper is the canonical "Librarian" who maintains the truth
        if self.role_id.lower() != "lorekeeper":
            return json.dumps(
                {
                    "success": False,
                    "error": f"Role '{self.role_id}' cannot promote to canon",
                    "hint": "Only Lorekeeper can write to cold_store. "
                    "After Gatekeeper validates, delegate to Lorekeeper to promote.",
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

        # Import validation utilities
        from questfoundry.runtime.validation import detect_artifact_type, validate_artifact

        # Helper to return structured error per 9.4 validate-with-feedback pattern
        def _validation_error(artifact_id: str, error: str, invalid_fields: list, hint: str):
            return json.dumps({
                "success": False,
                "error": error,
                "artifact_id": artifact_id,
                "invalid_fields": invalid_fields,
                "hint": hint,
            })

        for artifact_id in artifact_ids:
            # Check artifact exists - fail immediately with feedback
            if artifact_id not in hot_store:
                return _validation_error(
                    artifact_id,
                    f"Artifact '{artifact_id}' not found in hot_store",
                    [{"field": "artifact_id", "provided": artifact_id, "issue": "not found"}],
                    f"Use list_hot_store_keys to see available artifacts. "
                    f"Create '{artifact_id}' with write_hot_sot before promoting.",
                )

            artifact = hot_store[artifact_id]

            # Extract artifact data for validation
            if hasattr(artifact, "model_dump"):
                artifact_data = artifact.model_dump()
            elif isinstance(artifact, dict):
                artifact_data = artifact
            else:
                return _validation_error(
                    artifact_id,
                    f"Artifact '{artifact_id}' has invalid type: {type(artifact).__name__}",
                    [{"field": "type", "provided": type(artifact).__name__,
                      "issue": "must be dict"}],
                    "Artifacts must be dicts. Use write_hot_sot with a dict value.",
                )

            # Handle nested 'data' field (some artifacts wrap content)
            if "data" in artifact_data and isinstance(artifact_data["data"], dict):
                content_data = artifact_data["data"]
            else:
                content_data = artifact_data

            # Key-based type detection (more reliable than schema matching)
            artifact_type = detect_artifact_type(artifact_id)
            if artifact_type is None:
                # Fall back to schema-based detection
                artifact_type, validated_model, validation_errors = _detect_artifact_type(content_data)
                if artifact_type is None:
                    return _validation_error(
                        artifact_id,
                        f"Cannot determine artifact type for '{artifact_id}'",
                        [{"field": "artifact_id", "provided": artifact_id,
                          "issue": f"Key must start with type prefix (e.g., scene_1, act_1). Errors: {'; '.join(validation_errors[:2])}"}],
                        "Use artifact keys like 'scene_1', 'act_1', 'chapter_1'. "
                        "Run consult_schema to see required fields for each type.",
                    )
            else:
                # Validate the data against the detected type
                # validate_artifact returns dict: {"success": bool, "validated": dict} or error dict
                validation_result = validate_artifact(artifact_type, content_data)
                if not validation_result.get("success", False):
                    # Extract error info from validation result
                    invalid_fields = validation_result.get("invalid_fields", [])
                    missing_fields = validation_result.get("missing_fields", [])

                    # Build combined invalid_fields list
                    all_invalid = []
                    for field in missing_fields:
                        all_invalid.append({"field": field, "issue": "Field required"})
                    all_invalid.extend(invalid_fields)

                    return _validation_error(
                        artifact_id,
                        f"Artifact '{artifact_id}' ({artifact_type}) failed schema validation",
                        (
                            all_invalid[:5]
                            if all_invalid
                            else [{"field": "unknown", "issue": "Validation failed"}]
                        ),
                        f"Use consult_schema(artifact_type='{artifact_type}') "
                        f"to see required fields. Update '{artifact_id}' with "
                        "write_hot_sot to add missing fields, then retry.",
                    )

                # Create a SimpleNamespace from validated dict for getattr compatibility
                from types import SimpleNamespace

                validated_data = validation_result.get("validated", content_data)
                validated_model = SimpleNamespace(**validated_data)

            # Extract content from the correct field for this artifact type
            content, extract_error = _extract_content_for_cold(
                artifact_type, validated_model, artifact_id
            )

            if extract_error:
                return _validation_error(
                    artifact_id,
                    f"Content extraction failed for '{artifact_id}'",
                    [{"field": "content", "issue": extract_error}],
                    f"Artifact type '{artifact_type}' requires a content field. "
                    f"Use consult_schema(artifact_type='{artifact_type}') to see requirements.",
                )

            # Extract title (try common fields, fall back to artifact_id)
            title = (
                getattr(validated_model, "title", None)
                or getattr(validated_model, "name", None)
                or content_data.get("title")
                or artifact_id
            )

            logger.info(f"Promoting '{artifact_id}' as {artifact_type} with content from validated schema")

            # Extract interactive fiction fields (choices, gates) from the model
            # Handle both Pydantic objects and raw dicts from LLM output
            from questfoundry.generated.models.artifacts import Choice, Gate

            raw_choices = getattr(validated_model, "choices", None) or content_data.get("choices")
            raw_gates = getattr(validated_model, "gates", None) or content_data.get("gates")

            # Convert to proper types if needed (handles raw dicts from LLM)
            choices = None
            if raw_choices:
                choices = [
                    c if isinstance(c, Choice) else Choice(**c)
                    for c in raw_choices
                ]
            gates = None
            if raw_gates:
                gates = [
                    g if isinstance(g, Gate) else Gate(**g)
                    for g in raw_gates
                ]
            requires_gate = bool(gates)

            # Route to appropriate cold_store method based on artifact type
            # Normalize type name (key-based detection returns lowercase)
            artifact_type_lower = artifact_type.lower()
            try:
                if artifact_type_lower == "act":
                    # Extract Act-specific fields
                    from questfoundry.generated.models.enums import Visibility
                    sequence = getattr(validated_model, "sequence", 1)
                    description = getattr(validated_model, "description", None)
                    visibility_val = getattr(validated_model, "visibility", None)
                    visibility = visibility_val if visibility_val else Visibility.PUBLIC

                    self.cold_store.add_act(
                        anchor=artifact_id,
                        title=title,
                        sequence=sequence,
                        description=description,
                        visibility=visibility,
                    )
                elif artifact_type_lower == "chapter":
                    # Extract Chapter-specific fields
                    from questfoundry.generated.models.enums import Visibility
                    sequence = getattr(validated_model, "sequence", 1)
                    act_anchor = getattr(validated_model, "act_id", None)
                    summary = getattr(validated_model, "summary", None)
                    visibility_val = getattr(validated_model, "visibility", None)
                    visibility = visibility_val if visibility_val else Visibility.PUBLIC

                    self.cold_store.add_chapter(
                        anchor=artifact_id,
                        title=title,
                        sequence=sequence,
                        act_anchor=act_anchor,
                        summary=summary,
                        visibility=visibility,
                    )
                else:
                    # Content-bearing artifacts use add_section
                    self.cold_store.add_section(
                        anchor=artifact_id,
                        title=title,
                        content=content,
                        source_brief_id=artifact_id,  # Track lineage
                        choices=choices,
                        gates=gates,
                        requires_gate=requires_gate,
                    )
                promoted.append(artifact_id)
            except Exception as e:
                return _validation_error(
                    artifact_id,
                    f"Cold store write failed for '{artifact_id}'",
                    [{"field": "cold_store", "issue": str(e)}],
                    "This is likely a database error. Check cold_store connection and retry.",
                )

        # Create snapshot if requested and we promoted something
        snapshot_id = None
        if create_snapshot and promoted:
            desc = snapshot_description or f"Promoted {len(promoted)} artifact(s)"
            snapshot_id = self.cold_store.create_snapshot(desc)

        return json.dumps(
            {
                "success": True,
                "promoted": promoted,
                "snapshot_id": snapshot_id,
            }
        )


class ListHotStoreKeys(BaseTool):
    """List all keys in hot_store for artifact discovery.

    Use this to discover what artifacts exist in hot_store before
    reading or working with them. Useful when you need to know
    what artifacts have been created by previous roles.

    Also indicates which promotable artifacts (act_*, chapter_*, scene_*)
    are NOT yet in cold_store - useful for SR to verify promotion completeness.
    """

    name: str = "list_hot_store_keys"
    description: str = (
        "List all artifact keys in hot_store. "
        "Use this to discover what artifacts exist. "
        "Returns a list of keys and their types, plus which promotable "
        "artifacts are NOT yet in cold_store."
    )

    # State is injected by executor
    state: dict[str, Any] = Field(default_factory=dict)
    # ColdStore is injected for comparison (optional)
    cold_store: Any = Field(default=None)

    def _run(self) -> str:
        """List all keys in hot_store."""
        hot_store = self.state.get("hot_store", {})

        if not hot_store:
            return json.dumps(
                {
                    "success": True,
                    "keys": [],
                    "message": "hot_store is empty",
                }
            )

        # Build key info with types
        key_info = []
        promotable_keys = []  # Keys that could be promoted (act_*, chapter_*, scene_*)

        for key, value in hot_store.items():
            info = {"key": key}
            if hasattr(value, "model_dump"):
                # Pydantic model
                info["type"] = type(value).__name__
            elif isinstance(value, dict):
                info["type"] = value.get("type", "dict")
            elif isinstance(value, list):
                info["type"] = f"list[{len(value)} items]"
            else:
                info["type"] = type(value).__name__
            key_info.append(info)

            # Track promotable keys
            if key.startswith(("act_", "chapter_", "scene_")):
                promotable_keys.append(key)

        result = {
            "success": True,
            "keys": key_info,
            "count": len(key_info),
        }

        # If we have cold_store and promotable keys, check what's NOT in cold
        if self.cold_store is not None and promotable_keys:
            cold_sections = set(self.cold_store.list_sections())
            not_in_cold = [k for k in promotable_keys if k not in cold_sections]

            if not_in_cold:
                result["promotable_not_in_cold"] = not_in_cold
                result["promotion_hint"] = (
                    f"{len(not_in_cold)} promotable artifact(s) not yet in cold_store. "
                    f"Delegate to Lorekeeper to promote: {', '.join(not_in_cold)}"
                )

        return json.dumps(result)


class ListColdStoreKeys(BaseTool):
    """List all sections and snapshots in cold_store.

    Use this to discover what canon content exists in cold_store
    before reading specific sections. Returns sections (anchors)
    and available snapshots.
    """

    name: str = "list_cold_store_keys"
    description: str = (
        "List all sections and snapshots in cold_store. "
        "Use this to discover what canon content exists. "
        "Returns section anchors and snapshot IDs."
    )

    # ColdStore is injected by executor
    cold_store: Any = Field(default=None)

    def _run(self) -> str:
        """List all sections and snapshots in cold_store."""
        if self.cold_store is None:
            return json.dumps(
                {
                    "success": False,
                    "error": "Cold store not available",
                    "hint": "Cold store may not be initialized for this session.",
                }
            )

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

    Note: Only Lorekeeper should be given this tool.
    The tool itself also validates role_id as a safety check.
    Workflow: GK validates → SR delegates to LK → LK promotes.
    """
    tool = PromoteToCanon()
    tool.state = state
    tool.cold_store = cold_store
    tool.role_id = role_id
    return tool
