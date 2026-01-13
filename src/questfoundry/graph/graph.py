"""Unified story graph storage.

The graph is the single source of truth for story state. All stages read from
and write to the graph through structured mutations.

See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path


class Graph:
    """Unified story graph storage.

    The graph stores all story state as nodes and edges in a single JSON file.
    Stages produce structured output that the runtime applies as mutations.

    Attributes:
        _data: Internal graph data structure with version, meta, nodes, edges.
    """

    VERSION = "5.0"

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize graph with optional data.

        Args:
            data: Graph data dict. If None, creates an empty graph.
        """
        self._data = data or {
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
        self._data["meta"]["last_modified"] = datetime.now(UTC).isoformat()

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_file = file_path.with_suffix(".tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(self._data, f, indent=2)
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
        result = self._data["nodes"].get(node_id)
        return cast("dict[str, Any] | None", result)

    def set_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Set a node, creating or replacing it.

        Args:
            node_id: Unique node identifier.
            data: Node data dict.
        """
        self._data["nodes"][node_id] = data

    def add_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Add a new node.

        Args:
            node_id: Unique node identifier.
            data: Node data dict.

        Raises:
            ValueError: If node already exists.
        """
        if node_id in self._data["nodes"]:
            raise ValueError(f"Node '{node_id}' already exists")
        self._data["nodes"][node_id] = data

    def update_node(self, node_id: str, updates: dict[str, Any]) -> None:
        """Update an existing node with new data.

        Args:
            node_id: Unique node identifier.
            updates: Dict of fields to update.

        Raises:
            ValueError: If node doesn't exist.
        """
        if node_id not in self._data["nodes"]:
            raise ValueError(f"Node '{node_id}' does not exist")
        self._data["nodes"][node_id].update(updates)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists.

        Args:
            node_id: Unique node identifier.

        Returns:
            True if node exists, False otherwise.
        """
        return node_id in self._data["nodes"]

    def get_nodes_by_type(self, node_type: str) -> dict[str, dict[str, Any]]:
        """Get all nodes of a specific type.

        Args:
            node_type: Type to filter by (e.g., "entity", "tension").

        Returns:
            Dict of node_id -> node_data for matching nodes.
        """
        return {
            node_id: node_data
            for node_id, node_data in self._data["nodes"].items()
            if node_data.get("type") == node_type
        }

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_edge(
        self,
        edge_type: str,
        from_id: str,
        to_id: str,
        **props: Any,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            edge_type: Type of edge (e.g., "choice", "has_alternative").
            from_id: Source node ID.
            to_id: Target node ID.
            **props: Additional edge properties.
        """
        edge = {
            "type": edge_type,
            "from": from_id,
            "to": to_id,
            **props,
        }
        self._data["edges"].append(edge)

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
        edges: list[dict[str, Any]] = self._data["edges"]

        if from_id is not None:
            edges = [e for e in edges if e.get("from") == from_id]
        if to_id is not None:
            edges = [e for e in edges if e.get("to") == to_id]
        if edge_type is not None:
            edges = [e for e in edges if e.get("type") == edge_type]

        return edges

    # -------------------------------------------------------------------------
    # Metadata Operations
    # -------------------------------------------------------------------------

    def set_last_stage(self, stage_name: str) -> None:
        """Record completion of a stage.

        Updates last_stage and appends to stage_history.

        Args:
            stage_name: Name of completed stage.
        """
        self._data["meta"]["last_stage"] = stage_name
        self._data["meta"]["stage_history"].append(
            {
                "stage": stage_name,
                "completed": datetime.now(UTC).isoformat(),
            }
        )

    def get_last_stage(self) -> str | None:
        """Get the name of the last completed stage.

        Returns:
            Stage name, or None if no stages completed.
        """
        result = self._data["meta"].get("last_stage")
        return cast("str | None", result)

    def set_project_name(self, name: str) -> None:
        """Set the project name in metadata.

        Args:
            name: Project name.
        """
        self._data["meta"]["project_name"] = name

    def get_project_name(self) -> str | None:
        """Get the project name from metadata.

        Returns:
            Project name, or None if not set.
        """
        result = self._data["meta"].get("project_name")
        return cast("str | None", result)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary.

        Returns:
            Graph data as dict.
        """
        return self._data

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
        node_count = len(self._data["nodes"])
        edge_count = len(self._data["edges"])
        last_stage = self.get_last_stage() or "none"
        return f"Graph(nodes={node_count}, edges={edge_count}, last_stage={last_stage})"
