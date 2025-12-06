from __future__ import annotations

# ruff: noqa: E402, I001

"""Internal state tools for hot/cold Sources of Truth.

These are lightweight LangChain-compatible tools used by roles to read/write
state. They rely on StateManager/ColdStore for storage semantics but avoid
additional permission checks (tool exposure already encodes access control).
"""

import hashlib
import json
import logging
import threading
from copy import deepcopy
from datetime import date
from typing import Any, Annotated

import jsonschema
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools.base import _is_injected_arg_type
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, create_model
from pydantic.fields import PydanticUndefined

from questfoundry.runtime.core.cold_store import ColdStore
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.exceptions import StateError
from questfoundry.runtime.models.state import StudioState

_STATE_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


def _get_sot_log():
    """Lazy getter for SOT logger (configured at runtime by CLI)."""
    try:
        from questfoundry.runtime.structured_logging import get_sot_logger, is_configured

        if is_configured():
            return get_sot_logger()
    except ImportError:
        pass
    return None


def _compute_content_hash(value: Any) -> str:
    """Compute a short hash for artifact evolution tracking."""
    try:
        serialized = json.dumps(value, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:8]
    except (TypeError, ValueError):
        return "unhashable"


def _safe_serialize(value: Any, max_size: int = 50000) -> Any:
    """Safely serialize a value for logging, truncating if too large."""
    try:
        serialized = json.dumps(value, default=str)
        if len(serialized) > max_size:
            return {"_truncated": True, "_size": len(serialized), "_preview": serialized[:1000]}
        return value
    except (TypeError, ValueError):
        return {"_error": "not_serializable", "_type": type(value).__name__}


class _StrictToolSchemaMixin:
    """Preserve args_schema config and drop injected fields from tool schema."""

    @property
    def tool_call_schema(self):  # type: ignore[override]
        args_schema = getattr(self, "args_schema", None)
        if args_schema is None or not isinstance(args_schema, type):
            return super().tool_call_schema  # pragma: no cover - fallback

        # Remove injected fields (e.g., ToolRuntime) while retaining config like
        # extra="forbid" and json_schema_extra.
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


# Schema-defined array keys in hot_sot and cold_sot
# When these keys are missing, return [] instead of None
_ARRAY_KEYS = frozenset(
    {
        # hot_sot arrays (from studio_state.schema.json)
        "hooks",
        "tus",
        "canon_packs",
        "style_addenda",
        "research_memos",
        "art_plans",
        "audio_plans",
        "edit_notes",
        "canon_transfer_packages",
        # cold_sot arrays
        "snapshots",
        "codex_entries",
        "language_packs",
        "sections",
        # Common ad-hoc arrays used by roles
        "customer_directives",
        "section_briefs",
        "drafts",
    }
)

# Dict-type keys that should return {} instead of None
_DICT_KEYS = frozenset(
    {
        "canon",
        "style",
        "topology",
        "codex",
        "manuscript",
        "world_genesis_manifest",
    }
)

# Cached key-to-artifact mapping (populated lazily)
_KEY_TO_ARTIFACT_CACHE: dict[str, str] | None = None


def _get_key_to_artifact_mapping() -> dict[str, str]:
    """Build reverse mapping: hot_sot_key -> artifact_type.

    Maps hot_sot keys to their corresponding artifact schema types.
    Example: {"current_tu": "tu_brief", "hooks": "hook_card", ...}

    Uses cached value after first call to avoid repeated file reads.
    """
    global _KEY_TO_ARTIFACT_CACHE
    if _KEY_TO_ARTIFACT_CACHE is not None:
        return _KEY_TO_ARTIFACT_CACHE

    try:
        from questfoundry.runtime.core.schema_tool_generator import _discover_artifact_mappings

        artifact_mappings = _discover_artifact_mappings()
        # Reverse: artifact_type -> hot_sot_key becomes hot_sot_key -> artifact_type
        _KEY_TO_ARTIFACT_CACHE = {v: k for k, v in artifact_mappings.items()}
        logger.debug(f"Built key-to-artifact mapping: {_KEY_TO_ARTIFACT_CACHE}")
    except Exception as e:
        logger.warning(f"Failed to build key-to-artifact mapping: {e}")
        _KEY_TO_ARTIFACT_CACHE = {}

    return _KEY_TO_ARTIFACT_CACHE


def create_empty_hot_sot() -> dict[str, Any]:
    """Create an empty schema-compliant hot_sot structure.

    All array fields initialized to [] per studio_state.schema.json.
    Dict fields initialized to {} for role convenience.
    """
    return {
        # Schema-defined arrays
        "hooks": [],
        "tus": [],
        "canon_packs": [],
        "style_addenda": [],
        "research_memos": [],
        "art_plans": [],
        "audio_plans": [],
        "edit_notes": [],
        "canon_transfer_packages": [],
        # Common ad-hoc arrays used by roles
        "customer_directives": [],
        "section_briefs": [],
        "drafts": [],
        "sections": [],
        # Dict-type keys used by roles
        "canon": {},
        "style": {},
        "topology": {},
        "world_genesis_manifest": {},
    }


def create_empty_cold_sot() -> dict[str, Any]:
    """Create an empty schema-compliant cold_sot structure.

    All array fields initialized to [] per studio_state.schema.json.
    Dict fields initialized to {} for role convenience.
    current_snapshot initialized with today's date per schema pattern requirement:
    ^Cold @ \\d{4}-\\d{2}-\\d{2}$
    """
    today = date.today().strftime("%Y-%m-%d")
    return {
        "current_snapshot": f"Cold @ {today}",
        "snapshots": [],
        "codex_entries": [],
        "language_packs": [],
        "sections": [],
        # Dict-type keys
        "canon": {},
        "codex": {},
        "manuscript": {},
    }


def ensure_sot_initialized(state: StudioState) -> StudioState:
    """Ensure hot_sot and cold_sot are properly initialized.

    If hot_sot or cold_sot are missing or empty, initializes them with
    schema-compliant empty structures.

    Args:
        state: Current studio state

    Returns:
        State with initialized hot_sot and cold_sot
    """
    updated = state.copy()
    if not updated.get("hot_sot"):
        updated["hot_sot"] = create_empty_hot_sot()
        logger.debug("Initialized empty hot_sot")
    if not updated.get("cold_sot"):
        updated["cold_sot"] = create_empty_cold_sot()
        logger.debug("Initialized empty cold_sot")
    return updated


def _get_nested(data: dict[str, Any], path: str | None, *, default: Any = None) -> Any:
    """Get nested value from dict by dot-path.

    Returns appropriate default instead of raising for missing keys:
    - Known array keys (hooks, tus, sections, etc.) → []
    - Known dict keys (canon, style, topology, etc.) → {}
    - Other missing keys → default (None by default)
    """
    if not path:
        return data

    parts = path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            # Key not found - return appropriate default based on key type
            final_key = parts[-1]
            if final_key in _ARRAY_KEYS:
                logger.debug("Key '%s' not found, returning empty list", path)
                return []
            if final_key in _DICT_KEYS:
                logger.debug("Key '%s' not found, returning empty dict", path)
                return {}
            logger.debug("Key '%s' not found, returning default", path)
            return default
    return deepcopy(current)


def _set_nested(root: dict[str, Any], path: str | None, value: Any) -> dict[str, Any]:
    if not path:
        return deepcopy(value) if isinstance(value, dict) else value

    parts = path.split(".")
    updated = deepcopy(root)
    cursor: Any = updated
    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        if is_last:
            cursor[part] = _merge_values(cursor.get(part), value)
        else:
            cursor.setdefault(part, {})
            if not isinstance(cursor[part], dict):
                raise StateError(f"Cannot descend into non-dict at '{part}' for key '{path}'")
            cursor = cursor[part]
    return updated


def _merge_values(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = existing.copy()
        merged.update(incoming)
        return merged
    if isinstance(existing, list):
        if incoming is None:
            return existing
        if isinstance(incoming, list):
            return existing + incoming
        return existing + [incoming]
    # Replace for other types or None
    return deepcopy(incoming)


class _BaseStateTool(_StrictToolSchemaMixin, BaseTool):
    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}
    _state_manager: StateManager = PrivateAttr()
    _cold_store: ColdStore = PrivateAttr()
    _schema_registry: SchemaRegistry = PrivateAttr()

    def __init__(
        self,
        *,
        state_manager: StateManager | None = None,
        cold_store: ColdStore | None = None,
        schema_registry: SchemaRegistry | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state_manager = state_manager or StateManager()
        self._cold_store = cold_store or ColdStore()
        self._schema_registry = schema_registry or SchemaRegistry()

    @staticmethod
    def _validate_state(candidate: StudioState) -> dict[str, Any]:
        """
        Validate state against schema and return structured validation results.

        Returns:
            Dict with:
                - success (bool): True if valid, False if validation errors
                - errors (list): Structured error details (empty if valid)
                - message (str): Human-readable summary
        """
        # Normalize candidate to satisfy StudioState schema invariants.
        # This is defensive: older or hand-constructed states may be missing
        # keys that are now required at the root level.
        # - ensure_sot_initialized() guarantees hot_sot/cold_sot skeletons
        # - exports is required by studio_state.schema.json, even for loops
        #   that never touch view/export flows.
        candidate = ensure_sot_initialized(candidate)
        if "exports" not in candidate:
            normalized = candidate.copy()
            normalized["exports"] = {}
            candidate = normalized
            logger.debug("Initialized missing 'exports' block on StudioState before validation")

        try:
            registry = SchemaRegistry()
            schema = registry.load_schema("studio_state.schema.json")
            registry.validate_against_schema(candidate, schema)
            return {"success": True, "errors": [], "message": "Validation passed"}
        except jsonschema.ValidationError as exc:
            # Extract field-level details from validation error
            field_path = ".".join(str(p) for p in exc.path) if exc.path else "(root)"

            # Build structured error information
            error_detail = {
                "field": field_path,
                "message": exc.message,
                "validator": exc.validator,
                "constraint": exc.validator_value,
            }

            # Add instance value if available
            if hasattr(exc, "instance"):
                error_detail["actual_value"] = str(exc.instance)[:200]  # Truncate long values

            # Create actionable feedback message
            feedback_parts = [
                f"Validation failed for field '{field_path}':",
                f"  Error: {exc.message}",
            ]

            if exc.validator == "pattern":
                feedback_parts.append(f"  Expected pattern: {exc.validator_value}")
                feedback_parts.append("  → Use consult_schema to review field format requirements")
            elif exc.validator == "required":
                feedback_parts.append(f"  Missing required fields: {exc.validator_value}")
                feedback_parts.append("  → Use consult_schema to see all required fields")
            elif exc.validator == "type":
                feedback_parts.append(f"  Expected type: {exc.validator_value}")
                feedback_parts.append("  → Check the schema for correct data types")
            else:
                feedback_parts.append("  → Use consult_schema for detailed field requirements")

            message = "\n".join(feedback_parts)

            # Treat schema validation failures as hard errors in logs so they
            # are visible in monitoring; the tool still returns a structured
            # {success: False, ...} payload to the caller.
            logger.error("State validation failed: %s", message)
            return {
                "success": False,
                "errors": [error_detail],
                "message": message,
            }
        except Exception as exc:
            # Catch-all for non-ValidationError exceptions
            error_msg = str(exc)
            # If the schema contains unresolved $ref pointers (e.g., when running
            # in an environment without bundled artifact schemas), treat this as
            # a soft failure and skip state validation rather than blocking all
            # writes. Artifact-level validation still runs separately.
            if "Unresolvable:" in error_msg:
                logger.warning(
                    "State validation schema reference unresolved, skipping full "
                    "state validation: %s",
                    error_msg,
                )
                return {
                    "success": True,
                    "errors": [],
                    "message": f"Validation skipped due to unresolved schema reference: {error_msg}",
                }

            logger.error("State validation error: %s", error_msg)
            return {
                "success": False,
                "errors": [{"message": error_msg}],
                "message": f"Validation error: {error_msg}",
            }


class ReadHotSOT(_BaseStateTool):
    name: str = "read_hot_sot"
    description: str = "Read from in-memory hot source of truth"

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        key: str | None = Field(default=None, description="Dot-path within hot_sot to read")
        state: Annotated[StudioState | None, InjectedToolArg] = Field(
            default=None, description="Current studio state"
        )
        role_id: Annotated[str | None, InjectedToolArg] = Field(
            default=None, description="Caller role id"
        )

    args_schema = Args

    def _run(
        self,
        key: str | None = None,
        state: StudioState | None = None,
        role_id: str | None = None,
    ) -> Any:  # type: ignore[override]
        if state is None:
            raise StateError("State payload is required for read_hot_sot")
        result = _get_nested(state.get("hot_sot", {}), key)

        # Log to structured logging (full content + hash for evolution tracking)
        sot_log = _get_sot_log()
        if sot_log:
            sot_log.info(
                "read_hot_sot",
                role=role_id,
                key=key,
                result_type=type(result).__name__,
                result_size=len(result) if isinstance(result, (list, dict, str)) else 1,
                content_hash=_compute_content_hash(result),
                value=_safe_serialize(result),
            )
        return result


class WriteHotSOT(_BaseStateTool):
    name: str = "write_hot_sot"
    description: str = (
        "Write to Hot State of Things. For artifact keys (current_tu, hooks, drafts, etc.), "
        "automatically validates value against artifact schema before writing. "
        "Returns {hot_sot: ..., success: true} when the write command executes and passes "
        "validation; domain-level outcomes (e.g., no-op, duplicate content) are reflected in "
        "the payload, not in the success flag. "
        "If validation fails, returns {success: false, missing_fields: [...], invalid_fields: [...], hint: '...'} "
        "with LLM-friendly feedback. Use consult_schema to check field requirements before writing."
    )

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        key: str | None = Field(
            default=None,
            description=(
                "Dot-path within hot_sot to write. Use keys from your role's 'Your Output Keys' section. "
                "Common keys: drafts, topology_notes, section_briefs, style, canon, hooks"
            ),
        )
        value: (
            dict[str, Any]
            | list[str | int | float | bool | dict[str, Any] | None]
            | str
            | int
            | float
            | bool
            | None
        ) = Field(default=None, description="Value to write")
        state: Annotated[StudioState | None, InjectedToolArg] = Field(
            default=None, description="Current studio state"
        )
        role_id: Annotated[str | None, InjectedToolArg] = Field(
            default=None, description="Caller role id"
        )

    args_schema = Args

    def _run(
        self,
        key: str | None = None,
        value: Any | None = None,
        state: StudioState | None = None,
        role_id: str | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        if state is None:
            raise StateError("State payload is required for write_hot_sot")
        if value is None:
            raise StateError("Value is required for write_hot_sot")

        # Artifact validation: detect artifact type from key and validate before write
        top_key = key.split(".")[0] if key else None
        key_to_artifact = _get_key_to_artifact_mapping()
        artifact_type = key_to_artifact.get(top_key) if top_key else None

        if artifact_type and isinstance(value, dict):
            # Import validation utilities from schema_tool_generator
            try:
                from questfoundry.runtime.core.schema_tool_generator import (
                    SchemaToolGenerator,
                    _format_validation_errors,
                    _normalize_artifact_input,
                )

                generator = SchemaToolGenerator()
                schema = generator._load_schema(artifact_type)
                model = generator.generate_pydantic_model(artifact_type, schema)

                # Apply normalizations (e.g., checkpoint format fixes for tu_brief)
                normalized_value = _normalize_artifact_input(artifact_type, value)

                # Validate using Pydantic model
                try:
                    validated = model(**normalized_value)
                    # Use validated/normalized data, omitting unset optional fields so
                    # we don't inject schema-defaulted or None-valued optionals into
                    # StudioState. The StudioState JSON Schema remains the single
                    # source of truth for required vs optional.
                    value = validated.model_dump(exclude_unset=True)
                except ValidationError as e:
                    # Return LLM-friendly validation feedback
                    field_names = list(model.model_fields.keys())
                    required_fields = set(schema.get("required", []))
                    return _format_validation_errors(
                        e, artifact_type, field_names, required_fields
                    )
            except ImportError:
                # schema_tool_generator not available, skip artifact validation
                logger.debug(f"Schema validation skipped for {artifact_type}: import error")
            except FileNotFoundError:
                # Schema file not found, skip artifact validation
                logger.debug(f"Schema validation skipped for {artifact_type}: schema not found")
            except Exception as e:
                # Other errors during validation setup, log and continue with generic validation
                logger.warning(f"Artifact validation error for {artifact_type}: {e}")

        with _STATE_LOCK:
            # Get previous value for evolution tracking
            prev_value = _get_nested(state.get("hot_sot", {}), key)
            prev_hash = _compute_content_hash(prev_value) if prev_value is not None else None

            new_hot = _set_nested(state.get("hot_sot", {}), key, value)
            updated_state = state.copy()
            updated_state["hot_sot"] = new_hot
            # Validate whole state and capture structured errors for agent feedback
            validation_result = self._validate_state(updated_state)

            # Log to structured logging (full content + evolution tracking)
            sot_log = _get_sot_log()
            if sot_log:
                new_hash = _compute_content_hash(value)
                sot_log.info(
                    "write_hot_sot",
                    role=role_id,
                    key=key,
                    value_type=type(value).__name__,
                    value_size=len(value) if isinstance(value, (list, dict, str)) else 1,
                    content_hash=new_hash,
                    prev_hash=prev_hash,
                    evolved=prev_hash is not None and prev_hash != new_hash,
                    value=_safe_serialize(value),
                    validation_result=validation_result,
                )

            # Return success=false with structured errors if validation failed
            if not validation_result["success"]:
                return {
                    "success": False,
                    "error": validation_result["message"],
                    "errors": validation_result["errors"],
                    "key": key,
                }

            # Success: return updated state
            return {"hot_sot": new_hot, "success": True}


class ReadColdSOT(_BaseStateTool):
    name: str = "read_cold_sot"
    description: str = "Read from persistent cold source of truth"

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        key: str | None = Field(default=None, description="Dot-path within cold_sot to read")
        project_id: str | None = Field(default=None, description="Project id override")
        state: Annotated[StudioState | None, InjectedToolArg] = Field(
            default=None, description="Current studio state"
        )
        role_id: Annotated[str | None, InjectedToolArg] = Field(
            default=None, description="Caller role id"
        )

    args_schema = Args

    def _run(
        self,
        key: str | None = None,
        state: StudioState | None = None,
        role_id: str | None = None,
        project_id: str | None = None,
    ) -> Any:  # type: ignore[override]
        # Prefer provided state; fall back to persisted cold store
        base_cold = state.get("cold_sot", {}) if state else {}
        if not base_cold:
            pid = project_id or (state.get("loop_context", {}).get("project_id") if state else None)
            pid = pid or "default"
            stored = self._cold_store.load_cold(pid)
            base_cold = stored or {}
        result = _get_nested(base_cold, key)

        # Log to structured logging (full content + hash for evolution tracking)
        sot_log = _get_sot_log()
        if sot_log:
            sot_log.info(
                "read_cold_sot",
                role=role_id,
                key=key,
                result_type=type(result).__name__,
                result_size=len(result) if isinstance(result, (list, dict, str)) else 1,
                content_hash=_compute_content_hash(result),
                value=_safe_serialize(result),
            )
        return result


class WriteColdSOT(_BaseStateTool):
    name: str = "write_cold_sot"
    description: str = (
        "Persist to cold source of truth (SQLite/disk). "
        "Returns {cold_sot: ..., success: true} when the persistence command executes and passes "
        "validation; whether the resulting content is semantically \"good\" is encoded in the "
        "returned structure, not in the success flag. "
        "If validation fails, returns {success: false, error: '...', errors: [...]} with detailed field-level errors. "
        "Use consult_schema to understand requirements, then retry with corrected data."
    )

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        key: str | None = Field(
            default=None,
            description=(
                "Dot-path within cold_sot to write. Use for finalized, validated content. "
                "Common keys: sections, codex_entries, snapshots"
            ),
        )
        value: (
            dict[str, Any]
            | list[str | int | float | bool | dict[str, Any] | None]
            | str
            | int
            | float
            | bool
            | None
        ) = Field(default=None, description="Value to write")
        project_id: str | None = Field(default=None, description="Project id override")
        state: Annotated[StudioState | None, InjectedToolArg] = Field(
            default=None, description="Current studio state"
        )
        role_id: Annotated[str | None, InjectedToolArg] = Field(
            default=None, description="Caller role id"
        )

    args_schema = Args

    def _run(
        self,
        key: str | None = None,
        value: Any | None = None,
        state: StudioState | None = None,
        role_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        if state is None:
            raise StateError("State payload is required for write_cold_sot")
        if value is None:
            raise StateError("Value is required for write_cold_sot")

        pid = project_id or state.get("loop_context", {}).get("project_id") or "default"

        with _STATE_LOCK:
            # Get previous value for evolution tracking
            prev_value = _get_nested(state.get("cold_sot", {}), key)
            prev_hash = _compute_content_hash(prev_value) if prev_value is not None else None

            new_cold = _set_nested(state.get("cold_sot", {}), key, value)
            # Persist full cold_sot snapshot
            self._cold_store.save_cold(pid, new_cold)

            updated_state = state.copy()
            updated_state["cold_sot"] = new_cold
            # Validate whole state and capture structured errors for agent feedback
            validation_result = self._validate_state(updated_state)

            # Log to structured logging (full content + evolution tracking)
            sot_log = _get_sot_log()
            if sot_log:
                new_hash = _compute_content_hash(value)
                sot_log.info(
                    "write_cold_sot",
                    role=role_id,
                    key=key,
                    value_type=type(value).__name__,
                    value_size=len(value) if isinstance(value, (list, dict, str)) else 1,
                    project_id=pid,
                    content_hash=new_hash,
                    prev_hash=prev_hash,
                    evolved=prev_hash is not None and prev_hash != new_hash,
                    value=_safe_serialize(value),
                    validation_result=validation_result,
                )

            # Return success=false with structured errors if validation failed
            if not validation_result["success"]:
                return {
                    "success": False,
                    "error": validation_result["message"],
                    "errors": validation_result["errors"],
                    "key": key,
                }

            # Success: return updated state
            return {"cold_sot": new_cold, "success": True}
