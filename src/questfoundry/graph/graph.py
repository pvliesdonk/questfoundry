"""Unified story graph storage.

The graph is the single source of truth for story state. All stages read from
and write to the graph through structured mutations.

The graph enforces referential integrity similar to foreign keys in databases:
- Node creation is explicit (create_node fails if exists)
- Node updates require the node to exist (update_node fails if not found)
- Edges validate that both endpoints exist
- Errors provide semantic, actionable feedback for LLM retry loops

Graph delegates storage operations to a GraphStore backend (DictGraphStore by
default). See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from questfoundry.graph.errors import (
    EdgeEndpointError,
    NodeExistsError,
    NodeNotFoundError,
    NodeReferencedError,
)
from questfoundry.graph.store import DictGraphStore, GraphStore

if TYPE_CHECKING:
    from pathlib import Path


class Graph:
    """Unified story graph storage.

    The graph stores all story state as nodes and edges. Stages produce
    structured output that the runtime applies as mutations.

    Storage is delegated to a GraphStore backend. By default, DictGraphStore
    provides the original in-memory dict behavior.

    Attributes:
        _store: The underlying storage backend.
    """

    VERSION = "5.0"

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        store: GraphStore | None = None,
    ) -> None:
        """Initialize graph with optional data or store.

        Args:
            data: Graph data dict. If None, creates an empty graph.
                  Ignored if *store* is provided.
            store: Pre-built storage backend. If provided, *data* is ignored.
        """
        if store is not None:
            self._store = store
        else:
            self._store = DictGraphStore(data)

    @property
    def _data(self) -> dict[str, Any]:
        """Backward-compat access to underlying dict (DictGraphStore only).

        External code that accesses ``graph._data`` directly should migrate
        to use Graph public methods instead.
        """
        return self._store._data  # type: ignore[attr-defined, no-any-return]

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, project_path: Path) -> Graph:
        """Load graph from project directory.

        Args:
            project_path: Path to project root directory.

        Returns:
            Loaded graph, or empty graph if no graph.json exists.
        """
        graph_file = project_path / "graph.json"
        if not graph_file.exists():
            return cls.empty()
        return cls.load_from_file(graph_file)

    @classmethod
    def load_from_file(cls, file_path: Path) -> Graph:
        """Load graph from a specific file (e.g., snapshot).

        Args:
            file_path: Path to graph JSON file.

        Returns:
            Loaded graph.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file contains invalid JSON.
        """
        with file_path.open() as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def empty(cls) -> Graph:
        """Create an empty graph.

        Returns:
            New empty graph with default structure.
        """
        return cls()

    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------

    def save(self, file_path: Path) -> None:
        """Persist graph to a file (atomic write).

        Uses a temporary file and rename to ensure atomicity.
        No partial writes on failure.

        Args:
            file_path: Path to save graph JSON.
        """
        # Update last_modified timestamp
        self._store.set_meta("last_modified", datetime.now(UTC).isoformat())

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize via store
        data = self._store.to_dict()

        # Atomic write: write to temp file, then rename
        # Use PID-based suffix to avoid collisions with concurrent processes
        temp_file = file_path.with_name(f"{file_path.name}.{os.getpid()}.tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(file_path)
        except Exception:
            # Clean up temp file on failure
            if temp_file.exists():
                temp_file.unlink()
            raise

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a node by ID.

        Args:
            node_id: Unique node identifier.

        Returns:
            Node data dict, or None if not found.
        """
        return self._store.get_node(node_id)

    def create_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Create a new node. Fails if node already exists.

        Use this when adding new data (e.g., BRAINSTORM creating entities).
        For modifying existing nodes, use update_node().

        Args:
            node_id: Unique node identifier.
            data: Node data dict.

        Raises:
            NodeExistsError: If node already exists.
        """
        if self._store.has_node(node_id):
            raise NodeExistsError(node_id)
        self._store.set_node(node_id, data)

    def update_node(self, node_id: str, **updates: Any) -> None:
        """Update an existing node. Fails if node doesn't exist.

        Use this when modifying existing data (e.g., SEED updating entity disposition).
        For creating new nodes, use create_node().

        Args:
            node_id: Unique node identifier.
            **updates: Fields to update (keyword arguments).

        Raises:
            NodeNotFoundError: If node doesn't exist.
        """
        if not self._store.has_node(node_id):
            # Provide helpful context based on node ID prefix
            node_type = self._infer_type_from_id(node_id)
            available = self._get_node_ids_by_type(node_type)
            raise NodeNotFoundError(
                node_id,
                available=available,
                context="update_node - node must exist before updating",
            )
        self._store.update_node_fields(node_id, **updates)

    def upsert_node(self, node_id: str, data: dict[str, Any]) -> bool:
        """Create or update a node. Use sparingly - prefer explicit create/update.

        Returns True if created, False if updated. This is useful for idempotent
        operations but hides intent. Prefer create_node() or update_node() for clarity.

        Args:
            node_id: Unique node identifier.
            data: Node data dict.

        Returns:
            True if node was created, False if updated.
        """
        created = not self._store.has_node(node_id)
        self._store.set_node(node_id, data)
        return created

    def delete_node(self, node_id: str, *, cascade: bool = False) -> None:
        """Delete a node. Fails if node is referenced by edges.

        Args:
            node_id: Node to delete.
            cascade: If True, also delete edges referencing this node.

        Raises:
            NodeNotFoundError: If node doesn't exist.
            NodeReferencedError: If node is referenced by edges and cascade=False.
        """
        if not self._store.has_node(node_id):
            raise NodeNotFoundError(node_id, context="delete_node")

        # Find edges referencing this node
        refs = self._store.edges_referencing(node_id)
        if refs and not cascade:
            raise NodeReferencedError(node_id, referenced_by=refs)

        # Delete referencing edges if cascade
        if cascade and refs:
            self._store.remove_edges_referencing(node_id)

        self._store.delete_node(node_id)

    def set_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Set a node, creating or replacing it.

        .. deprecated:: 5.1
            Use create_node() for new nodes or update_node() for existing nodes.
            set_node() hides intent and bypasses referential integrity checks.

        Args:
            node_id: Unique node identifier.
            data: Node data dict.
        """
        warnings.warn(
            "set_node() is deprecated. Use create_node() for new nodes or "
            "update_node() for existing nodes.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._store.set_node(node_id, data)

    def add_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Add a new node (alias for create_node).

        .. deprecated:: 5.1
            Use create_node() instead for consistency with update_node().

        Args:
            node_id: Unique node identifier.
            data: Node data dict.

        Raises:
            NodeExistsError: If node already exists.
        """
        warnings.warn(
            "add_node() is deprecated. Use create_node() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.create_node(node_id, data)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists.

        Args:
            node_id: Unique node identifier.

        Returns:
            True if node exists, False otherwise.
        """
        return self._store.has_node(node_id)

    def ref(self, node_type: str, raw_id: str) -> str:
        """Get a validated node reference. Raises if node doesn't exist.

        Use this to get node IDs for edge creation - it validates the node
        exists and returns the properly prefixed ID.

        Args:
            node_type: Type prefix (e.g., "entity", "path", "dilemma").
            raw_id: The raw ID without prefix.

        Returns:
            Full node ID (e.g., "entity::kay").

        Raises:
            ValueError: If raw_id contains '::' separator.
            NodeNotFoundError: If node doesn't exist.

        Example:
            >>> path_ref = graph.ref("path", "trust_path")
            >>> graph.add_edge("belongs_to", beat_ref, path_ref)
        """
        if "::" in raw_id:
            raise ValueError(
                f"raw_id should not contain '::' separator. "
                f"Got '{raw_id}', expected just the ID part (e.g., 'kay' not 'entity::kay')"
            )
        node_id = f"{node_type}::{raw_id}"
        if not self._store.has_node(node_id):
            available = self._get_node_ids_by_type(node_type)
            raise NodeNotFoundError(
                node_id,
                available=available,
                context=f"reference to {node_type}",
            )
        return node_id

    def get_nodes_by_type(self, node_type: str) -> dict[str, dict[str, Any]]:
        """Get all nodes of a specific type.

        Args:
            node_type: Type to filter by (e.g., "entity", "dilemma").

        Returns:
            Dict of node_id -> node_data for matching nodes.
        """
        return self._store.get_nodes_by_type(node_type)

    def _infer_type_from_id(self, node_id: str) -> str | None:
        """Infer node type from ID prefix (e.g., 'entity::kay' -> 'entity')."""
        if "::" in node_id:
            return node_id.split("::")[0]
        return None

    def _get_node_ids_by_type(self, node_type: str | None) -> list[str]:
        """Get all node IDs matching a type prefix.

        When node_type is None (for non-prefixed IDs), returns only non-prefixed
        IDs to provide relevant suggestions.
        """
        if node_type is None:
            return [nid for nid in self._store.all_node_ids() if "::" not in nid]
        return self._store.node_ids_with_prefix(f"{node_type}::")

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_edge(
        self,
        edge_type: str,
        from_id: str,
        to_id: str,
        *,
        validate: bool = True,
        **props: Any,
    ) -> None:
        """Add an edge between two nodes. Validates endpoints exist by default.

        Args:
            edge_type: Type of edge (e.g., "choice", "has_answer").
            from_id: Source node ID.
            to_id: Target node ID.
            validate: If True (default), validates both endpoints exist.
                Set to False only for special cases like building test fixtures.
            **props: Additional edge properties.

        Raises:
            EdgeEndpointError: If validate=True and either endpoint doesn't exist.
        """
        if validate:
            from_exists = self._store.has_node(from_id)
            to_exists = self._store.has_node(to_id)

            if not from_exists or not to_exists:
                # Determine what's missing
                if not from_exists and not to_exists:
                    missing = "both"
                elif not from_exists:
                    missing = "from"
                else:
                    missing = "to"

                # Get available IDs for helpful feedback
                from_type = self._infer_type_from_id(from_id)
                to_type = self._infer_type_from_id(to_id)

                raise EdgeEndpointError(
                    edge_type=edge_type,
                    from_id=from_id,
                    to_id=to_id,
                    missing=missing,
                    available_from=self._get_node_ids_by_type(from_type),
                    available_to=self._get_node_ids_by_type(to_type),
                )

        edge = {
            "type": edge_type,
            "from": from_id,
            "to": to_id,
            **props,
        }
        self._store.add_edge(edge)

    def remove_edge(
        self,
        edge_type: str,
        from_id: str,
        to_id: str,
    ) -> bool:
        """Remove a specific edge by type and endpoints.

        Removes the first matching edge. If no matching edge exists,
        returns False without raising an error.

        Args:
            edge_type: Type of edge (e.g., "describes_visual", "targets").
            from_id: Source node ID.
            to_id: Target node ID.

        Returns:
            True if an edge was removed, False if no match found.
        """
        return self._store.remove_edge(edge_type, from_id, to_id)

    def get_edges(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get edges matching filter criteria.

        All filter parameters are optional. If none provided, returns all edges.

        Args:
            from_id: Filter by source node ID.
            to_id: Filter by target node ID.
            edge_type: Filter by edge type.

        Returns:
            List of matching edge dicts.
        """
        return self._store.get_edges(from_id=from_id, to_id=to_id, edge_type=edge_type)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_invariants(self) -> list[str]:
        """Check graph invariants and return any violations.

        Invariants checked:
        1. All edge endpoints exist (referential integrity)
        2. All edges have required fields (type, from, to)

        This is for detecting code bugs/data corruption, not LLM validation.
        Call after mutations to ensure graph is in a valid state.

        Returns:
            List of violation messages (empty if valid).
        """
        violations: list[str] = []

        for i, edge in enumerate(self._store.get_edges()):
            # Check edge has required fields
            edge_type = edge.get("type")
            from_id = edge.get("from")
            to_id = edge.get("to")

            if not edge_type:
                violations.append(f"Edge {i} missing 'type' field")
            if not from_id:
                violations.append(f"Edge {i} missing 'from' field")
            if not to_id:
                violations.append(f"Edge {i} missing 'to' field")

            # Check endpoints exist
            if from_id and not self._store.has_node(from_id):
                violations.append(f"Edge {i} ({edge_type}): source '{from_id}' does not exist")
            if to_id and not self._store.has_node(to_id):
                violations.append(f"Edge {i} ({edge_type}): target '{to_id}' does not exist")

        return violations

    # -------------------------------------------------------------------------
    # Metadata Operations
    # -------------------------------------------------------------------------

    def set_last_stage(self, stage_name: str) -> None:
        """Record completion of a stage.

        Updates last_stage and appends to stage_history.

        Args:
            stage_name: Name of completed stage.
        """
        self._store.set_meta("last_stage", stage_name)
        history = self._store.get_meta("stage_history") or []
        history.append(
            {
                "stage": stage_name,
                "completed": datetime.now(UTC).isoformat(),
            }
        )
        self._store.set_meta("stage_history", history)

    def get_last_stage(self) -> str | None:
        """Get the name of the last completed stage.

        Returns:
            Stage name, or None if no stages completed.
        """
        result = self._store.get_meta("last_stage")
        return cast("str | None", result)

    def set_project_name(self, name: str) -> None:
        """Set the project name in metadata.

        Args:
            name: Project name.
        """
        self._store.set_meta("project_name", name)

    def get_project_name(self) -> str | None:
        """Get the project name from metadata.

        Returns:
            Project name, or None if not set.
        """
        result = self._store.get_meta("project_name")
        return cast("str | None", result)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary.

        Returns a deep copy to prevent external mutation of internal state.

        Returns:
            Graph data as dict (deep copy).
        """
        return self._store.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        """Create graph from dictionary.

        Args:
            data: Graph data dict.

        Returns:
            New Graph instance.
        """
        return cls(data)

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return string representation of graph."""
        node_count = self._store.node_count()
        edge_count = self._store.edge_count()
        last_stage = self.get_last_stage() or "none"
        return f"Graph(nodes={node_count}, edges={edge_count}, last_stage={last_stage})"
