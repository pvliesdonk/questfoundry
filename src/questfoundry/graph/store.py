"""Graph storage backend protocol and dict-based implementation.

The GraphStore protocol defines the low-level storage operations that Graph
delegates to. Implementations handle raw CRUD; Graph provides the public API
with validation, error messages, and business logic.

DictGraphStore is the default backend, preserving the original in-memory dict
behavior. SqliteGraphStore provides SQLite-backed storage with mutation
recording and savepoint support.
"""

from __future__ import annotations

import copy
from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class GraphStore(Protocol):
    """Storage backend protocol for Graph.

    Implementations provide low-level CRUD for nodes, edges, and metadata.
    Graph wraps a GraphStore and adds validation, error handling, and
    business logic on top.

    Methods raise no domain-specific errors — Graph is responsible for
    translating storage failures into NodeNotFoundError, etc.
    """

    # -- Nodes -----------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get node data by ID, or None if not found."""
        ...

    def has_node(self, node_id: str) -> bool:
        """Check whether a node exists."""
        ...

    def set_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Set node data (create or overwrite)."""
        ...

    def update_node_fields(self, node_id: str, **updates: Any) -> None:
        """Merge keyword updates into an existing node's data dict."""
        ...

    def delete_node(self, node_id: str) -> None:
        """Delete a node by ID. No cascade — caller handles edges first."""
        ...

    def get_nodes_by_type(self, node_type: str) -> dict[str, dict[str, Any]]:
        """Return all nodes whose ``type`` field matches *node_type*."""
        ...

    def all_node_ids(self) -> list[str]:
        """Return all node IDs."""
        ...

    def node_ids_with_prefix(self, prefix: str) -> list[str]:
        """Return node IDs that start with *prefix*."""
        ...

    def node_count(self) -> int:
        """Return total number of nodes."""
        ...

    # -- Edges -----------------------------------------------------------------

    def add_edge(self, edge: dict[str, Any]) -> None:
        """Append a pre-built edge dict (no validation)."""
        ...

    def remove_edge(self, edge_type: str, from_id: str, to_id: str) -> bool:
        """Remove first matching edge. Return True if removed, False if absent."""
        ...

    def get_edges(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return edges matching optional filters (all edges if no filters)."""
        ...

    def edges_referencing(self, node_id: str) -> list[dict[str, Any]]:
        """Return all edges where *node_id* appears as ``from`` or ``to``."""
        ...

    def remove_edges_referencing(self, node_id: str) -> None:
        """Remove all edges where *node_id* appears as ``from`` or ``to``."""
        ...

    def edge_count(self) -> int:
        """Return total number of edges."""
        ...

    # -- Meta ------------------------------------------------------------------

    def get_meta(self, key: str) -> Any:
        """Get a metadata value by key, or None if absent."""
        ...

    def set_meta(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        ...

    def all_meta(self) -> dict[str, Any]:
        """Return the full metadata dict."""
        ...

    # -- Mutation context ------------------------------------------------------

    def set_mutation_context(self, stage: str = "", phase: str = "") -> None:
        """Set the current stage/phase for mutation recording."""
        ...

    # -- Savepoints ------------------------------------------------------------

    def savepoint(self, name: str) -> None:
        """Create a named savepoint of the current state."""
        ...

    def rollback_to(self, name: str) -> None:
        """Rollback to a named savepoint."""
        ...

    def release(self, name: str) -> None:
        """Release (discard) a named savepoint."""
        ...

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire store to a graph dict (deep copy)."""
        ...


class DictGraphStore:
    """In-memory dict-based graph store.

    This is the default backend, preserving the original Graph behavior
    of storing all state in a nested dict.
    """

    VERSION = "5.0"

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {
            "version": self.VERSION,
            "meta": {
                "project_name": None,
                "last_stage": None,
                "last_modified": None,
                "stage_history": [],
            },
            "nodes": {},
            "edges": [],
        }
        self._savepoints: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DictGraphStore:
        """Create a DictGraphStore from a graph data dict."""
        return cls(data)

    # -- Nodes -----------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        result = self._data["nodes"].get(node_id)
        return cast("dict[str, Any] | None", result)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._data["nodes"]

    def set_node(self, node_id: str, data: dict[str, Any]) -> None:
        self._data["nodes"][node_id] = data

    def update_node_fields(self, node_id: str, **updates: Any) -> None:
        self._data["nodes"][node_id].update(updates)

    def delete_node(self, node_id: str) -> None:
        del self._data["nodes"][node_id]

    def get_nodes_by_type(self, node_type: str) -> dict[str, dict[str, Any]]:
        return {
            nid: ndata
            for nid, ndata in self._data["nodes"].items()
            if ndata.get("type") == node_type
        }

    def all_node_ids(self) -> list[str]:
        return list(self._data["nodes"].keys())

    def node_ids_with_prefix(self, prefix: str) -> list[str]:
        return [nid for nid in self._data["nodes"] if nid.startswith(prefix)]

    def node_count(self) -> int:
        return len(self._data["nodes"])

    # -- Edges -----------------------------------------------------------------

    def add_edge(self, edge: dict[str, Any]) -> None:
        self._data["edges"].append(edge)

    def remove_edge(self, edge_type: str, from_id: str, to_id: str) -> bool:
        for i, edge in enumerate(self._data["edges"]):
            if (
                edge.get("type") == edge_type
                and edge.get("from") == from_id
                and edge.get("to") == to_id
            ):
                self._data["edges"].pop(i)
                return True
        return False

    def get_edges(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = self._data["edges"]
        if from_id is not None:
            edges = [e for e in edges if e.get("from") == from_id]
        if to_id is not None:
            edges = [e for e in edges if e.get("to") == to_id]
        if edge_type is not None:
            edges = [e for e in edges if e.get("type") == edge_type]
        return edges

    def edges_referencing(self, node_id: str) -> list[dict[str, Any]]:
        return [e for e in self._data["edges"] if node_id in (e.get("from"), e.get("to"))]

    def remove_edges_referencing(self, node_id: str) -> None:
        self._data["edges"] = [
            e for e in self._data["edges"] if node_id not in (e.get("from"), e.get("to"))
        ]

    def edge_count(self) -> int:
        return len(self._data["edges"])

    # -- Meta ------------------------------------------------------------------

    def get_meta(self, key: str) -> Any:
        return self._data["meta"].get(key)

    def set_meta(self, key: str, value: Any) -> None:
        self._data["meta"][key] = value

    def all_meta(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self._data["meta"])

    # -- Mutation context (no-op for dict backend) ----------------------------

    def set_mutation_context(self, stage: str = "", phase: str = "") -> None:
        """No-op — DictGraphStore does not record mutations."""

    # -- Savepoints (deepcopy-based) -------------------------------------------

    def savepoint(self, name: str) -> None:
        """Save a named snapshot of current state."""
        self._savepoints[name] = copy.deepcopy(self._data)

    def rollback_to(self, name: str) -> None:
        """Restore state from a named snapshot."""
        if name not in self._savepoints:
            raise ValueError(f"No savepoint named '{name}'")
        self._data = copy.deepcopy(self._savepoints[name])

    def release(self, name: str) -> None:
        """Discard a named snapshot."""
        self._savepoints.pop(name, None)

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)
