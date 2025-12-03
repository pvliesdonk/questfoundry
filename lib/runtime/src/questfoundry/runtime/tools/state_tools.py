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
from typing import Any, Annotated

from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools.base import _is_injected_arg_type
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model
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
    """
    return {
        "current_snapshot": "",
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
    def _validate_state(candidate: StudioState) -> None:
        try:
            registry = SchemaRegistry()
            schema = registry.load_schema("studio_state.schema.json")
            registry.validate_against_schema(candidate, schema)
        except Exception as exc:  # pragma: no cover - best-effort guard
            logger.debug("State validation skipped due to error: %s", exc)


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
    description: str = "Write to in-memory hot source of truth"

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        key: str | None = Field(default=None, description="Dot-path within hot_sot to write")
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

        with _STATE_LOCK:
            # Get previous value for evolution tracking
            prev_value = _get_nested(state.get("hot_sot", {}), key)
            prev_hash = _compute_content_hash(prev_value) if prev_value is not None else None

            new_hot = _set_nested(state.get("hot_sot", {}), key, value)
            updated_state = state.copy()
            updated_state["hot_sot"] = new_hot
            # Validate whole state to catch structural regressions
            self._validate_state(updated_state)

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
                )

            return {"hot_sot": new_hot}


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
    description: str = "Persist to cold source of truth (SQLite/disk)"

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        key: str | None = Field(default=None, description="Dot-path within cold_sot to write")
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
            self._validate_state(updated_state)

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
                )
            return {"cold_sot": new_cold}
