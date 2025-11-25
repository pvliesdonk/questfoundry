from __future__ import annotations

"""Internal state tools for hot/cold Sources of Truth.

These are lightweight LangChain-compatible tools used by roles to read/write
state. They rely on StateManager/ColdStore for storage semantics but avoid
additional permission checks (tool exposure already encodes access control).
"""

import threading
import logging
from copy import deepcopy
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from questfoundry.runtime.core.cold_store import ColdStore
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.exceptions import StateError
from questfoundry.runtime.models.state import StudioState


_STATE_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


def _get_nested(data: dict[str, Any], path: str | None) -> Any:
    if not path:
        return data

    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise StateError(f"State key not found: {path}")
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


class _BaseStateTool(BaseTool):
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

    def _run(self, key: str | None = None, state: StudioState | None = None, role_id: str | None = None) -> Any:  # type: ignore[override]
        if state is None:
            raise StateError("State payload is required for read_hot_sot")
        return _get_nested(state.get("hot_sot", {}), key)


class WriteHotSOT(_BaseStateTool):
    name: str = "write_hot_sot"
    description: str = "Write to in-memory hot source of truth"

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
            new_hot = _set_nested(state.get("hot_sot", {}), key, value)
            updated_state = state.copy()
            updated_state["hot_sot"] = new_hot
            # Validate whole state to catch structural regressions
            self._validate_state(updated_state)
            return {"hot_sot": new_hot}


class ReadColdSOT(_BaseStateTool):
    name: str = "read_cold_sot"
    description: str = "Read from persistent cold source of truth"

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
        return _get_nested(base_cold, key)


class WriteColdSOT(_BaseStateTool):
    name: str = "write_cold_sot"
    description: str = "Persist to cold source of truth (SQLite/disk)"

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
            new_cold = _set_nested(state.get("cold_sot", {}), key, value)
            # Persist full cold_sot snapshot
            self._cold_store.save_cold(pid, new_cold)

            updated_state = state.copy()
            updated_state["cold_sot"] = new_cold
            self._validate_state(updated_state)
            return {"cold_sot": new_cold}
