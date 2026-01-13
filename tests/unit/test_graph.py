"""Tests for the Graph class and snapshot management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from questfoundry.graph import Graph
from questfoundry.graph.snapshots import (
    cleanup_old_snapshots,
    delete_snapshot,
    list_snapshots,
    rollback_to_snapshot,
    save_snapshot,
)


class TestGraphBasics:
    """Test basic Graph operations."""

    def test_empty_graph(self) -> None:
        """Empty graph has correct structure."""
        graph = Graph.empty()
        data = graph.to_dict()

        assert data["version"] == "5.0"
        assert data["meta"]["last_stage"] is None
        assert data["nodes"] == {}
        assert data["edges"] == []

    def test_repr(self) -> None:
        """Graph repr shows useful info."""
        graph = Graph.empty()
        assert "nodes=0" in repr(graph)
        assert "edges=0" in repr(graph)
        assert "last_stage=none" in repr(graph)

        graph.add_node("test", {"type": "test"})
        graph.set_last_stage("dream")
        assert "nodes=1" in repr(graph)
        assert "last_stage=dream" in repr(graph)


class TestNodeOperations:
    """Test node CRUD operations."""

    def test_add_node(self) -> None:
        """Can add a new node."""
        graph = Graph.empty()
        graph.add_node("char_001", {"type": "entity", "name": "Alice"})

        assert graph.has_node("char_001")
        node = graph.get_node("char_001")
        assert node is not None
        assert node["type"] == "entity"
        assert node["name"] == "Alice"

    def test_add_duplicate_node_raises(self) -> None:
        """Adding duplicate node raises ValueError."""
        graph = Graph.empty()
        graph.add_node("char_001", {"type": "entity"})

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("char_001", {"type": "entity"})

    def test_set_node_creates_or_replaces(self) -> None:
        """set_node creates if missing, replaces if exists."""
        graph = Graph.empty()

        # Create new
        graph.set_node("vision", {"type": "vision", "genre": "fantasy"})
        assert graph.get_node("vision")["genre"] == "fantasy"

        # Replace existing
        graph.set_node("vision", {"type": "vision", "genre": "noir"})
        assert graph.get_node("vision")["genre"] == "noir"

    def test_update_node(self) -> None:
        """Can update existing node fields."""
        graph = Graph.empty()
        graph.add_node("char_001", {"type": "entity", "name": "Alice"})

        graph.update_node("char_001", {"disposition": "retained"})

        node = graph.get_node("char_001")
        assert node is not None
        assert node["name"] == "Alice"  # Original preserved
        assert node["disposition"] == "retained"  # New field added

    def test_update_nonexistent_node_raises(self) -> None:
        """Updating nonexistent node raises ValueError."""
        graph = Graph.empty()

        with pytest.raises(ValueError, match="does not exist"):
            graph.update_node("missing", {"foo": "bar"})

    def test_get_node_returns_none_for_missing(self) -> None:
        """get_node returns None for missing node."""
        graph = Graph.empty()
        assert graph.get_node("missing") is None

    def test_has_node(self) -> None:
        """has_node returns correct boolean."""
        graph = Graph.empty()
        assert not graph.has_node("char_001")

        graph.add_node("char_001", {"type": "entity"})
        assert graph.has_node("char_001")

    def test_get_nodes_by_type(self) -> None:
        """Can filter nodes by type."""
        graph = Graph.empty()
        graph.add_node("char_001", {"type": "entity", "entity_type": "character"})
        graph.add_node("char_002", {"type": "entity", "entity_type": "character"})
        graph.add_node("tension_001", {"type": "tension", "question": "?"})
        graph.set_node("vision", {"type": "vision", "genre": "noir"})

        entities = graph.get_nodes_by_type("entity")
        assert len(entities) == 2
        assert "char_001" in entities
        assert "char_002" in entities

        tensions = graph.get_nodes_by_type("tension")
        assert len(tensions) == 1
        assert "tension_001" in tensions

        visions = graph.get_nodes_by_type("vision")
        assert len(visions) == 1


class TestEdgeOperations:
    """Test edge operations."""

    def test_add_edge(self) -> None:
        """Can add edges between nodes."""
        graph = Graph.empty()
        graph.add_node("tension_001", {"type": "tension"})
        graph.add_node("alt_001", {"type": "alternative"})

        graph.add_edge("has_alternative", "tension_001", "alt_001")

        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[0]["type"] == "has_alternative"
        assert edges[0]["from"] == "tension_001"
        assert edges[0]["to"] == "alt_001"

    def test_add_edge_with_props(self) -> None:
        """Can add edges with additional properties."""
        graph = Graph.empty()
        graph.add_edge(
            "choice",
            "scene_001",
            "scene_002",
            label="Go north",
            requires=["has_key"],
        )

        edges = graph.get_edges()
        assert edges[0]["label"] == "Go north"
        assert edges[0]["requires"] == ["has_key"]

    def test_get_edges_filter_by_from(self) -> None:
        """Can filter edges by source node."""
        graph = Graph.empty()
        graph.add_edge("has_alt", "t1", "a1")
        graph.add_edge("has_alt", "t1", "a2")
        graph.add_edge("has_alt", "t2", "a3")

        edges = graph.get_edges(from_id="t1")
        assert len(edges) == 2

    def test_get_edges_filter_by_to(self) -> None:
        """Can filter edges by target node."""
        graph = Graph.empty()
        graph.add_edge("belongs_to", "beat_1", "thread_1")
        graph.add_edge("belongs_to", "beat_2", "thread_1")
        graph.add_edge("belongs_to", "beat_3", "thread_2")

        edges = graph.get_edges(to_id="thread_1")
        assert len(edges) == 2

    def test_get_edges_filter_by_type(self) -> None:
        """Can filter edges by type."""
        graph = Graph.empty()
        graph.add_edge("has_alternative", "t1", "a1")
        graph.add_edge("explores", "thread_1", "a1")
        graph.add_edge("has_alternative", "t2", "a2")

        edges = graph.get_edges(edge_type="has_alternative")
        assert len(edges) == 2

    def test_get_edges_multiple_filters(self) -> None:
        """Can combine multiple filters."""
        graph = Graph.empty()
        graph.add_edge("has_alt", "t1", "a1")
        graph.add_edge("has_alt", "t1", "a2")
        graph.add_edge("explores", "th1", "a1")

        edges = graph.get_edges(from_id="t1", edge_type="has_alt")
        assert len(edges) == 2

        edges = graph.get_edges(to_id="a1", edge_type="has_alt")
        assert len(edges) == 1


class TestMetadata:
    """Test metadata operations."""

    def test_set_and_get_last_stage(self) -> None:
        """Can set and get last completed stage."""
        graph = Graph.empty()
        assert graph.get_last_stage() is None

        graph.set_last_stage("dream")
        assert graph.get_last_stage() == "dream"

    def test_stage_history_recorded(self) -> None:
        """Stage completions are recorded in history."""
        graph = Graph.empty()
        graph.set_last_stage("dream")
        graph.set_last_stage("brainstorm")

        history = graph.to_dict()["meta"]["stage_history"]
        assert len(history) == 2
        assert history[0]["stage"] == "dream"
        assert history[1]["stage"] == "brainstorm"
        assert "completed" in history[0]

    def test_set_and_get_project_name(self) -> None:
        """Can set and get project name."""
        graph = Graph.empty()
        assert graph.get_project_name() is None

        graph.set_project_name("noir_mystery")
        assert graph.get_project_name() == "noir_mystery"


class TestPersistence:
    """Test load/save operations."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Can save and load graph."""
        graph = Graph.empty()
        graph.set_project_name("test_project")
        graph.add_node("char_001", {"type": "entity", "name": "Alice"})
        graph.set_last_stage("dream")

        # Save
        graph_file = tmp_path / "graph.json"
        graph.save(graph_file)
        assert graph_file.exists()

        # Load
        loaded = Graph.load_from_file(graph_file)
        assert loaded.get_project_name() == "test_project"
        assert loaded.get_last_stage() == "dream"
        assert loaded.get_node("char_001")["name"] == "Alice"

    def test_load_from_project_path(self, tmp_path: Path) -> None:
        """Can load graph from project directory."""
        graph = Graph.empty()
        graph.add_node("test", {"type": "test"})
        graph.save(tmp_path / "graph.json")

        loaded = Graph.load(tmp_path)
        assert loaded.has_node("test")

    def test_load_returns_empty_if_no_file(self, tmp_path: Path) -> None:
        """Loading from empty directory returns empty graph."""
        graph = Graph.load(tmp_path)
        assert graph.get_last_stage() is None
        assert len(graph.to_dict()["nodes"]) == 0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Save creates parent directories if needed."""
        graph = Graph.empty()
        nested_path = tmp_path / "a" / "b" / "c" / "graph.json"
        graph.save(nested_path)
        assert nested_path.exists()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """Save uses atomic write pattern."""
        graph = Graph.empty()
        graph_file = tmp_path / "graph.json"
        graph.save(graph_file)

        # Verify no temp file left behind
        assert not (tmp_path / "graph.tmp").exists()
        assert graph_file.exists()

    def test_save_updates_last_modified(self, tmp_path: Path) -> None:
        """Save updates last_modified timestamp."""
        graph = Graph.empty()
        graph_file = tmp_path / "graph.json"
        graph.save(graph_file)

        data = json.loads(graph_file.read_text())
        assert data["meta"]["last_modified"] is not None


class TestSerialization:
    """Test to_dict/from_dict."""

    def test_round_trip(self) -> None:
        """Data survives round trip through dict."""
        graph = Graph.empty()
        graph.set_project_name("test")
        graph.add_node("n1", {"type": "entity", "data": [1, 2, 3]})
        graph.add_edge("link", "n1", "n2", prop="value")

        data = graph.to_dict()
        restored = Graph.from_dict(data)

        assert restored.get_project_name() == "test"
        assert restored.get_node("n1")["data"] == [1, 2, 3]
        assert restored.get_edges()[0]["prop"] == "value"


class TestSnapshots:
    """Test snapshot management."""

    def test_save_snapshot(self, tmp_path: Path) -> None:
        """Can save pre-stage snapshot."""
        graph = Graph.empty()
        graph.add_node("before", {"type": "test"})

        snapshot_path = save_snapshot(graph, tmp_path, "dream")

        assert snapshot_path.exists()
        assert snapshot_path.name == "pre-dream.json"

        # Verify snapshot contains original data
        loaded = Graph.load_from_file(snapshot_path)
        assert loaded.has_node("before")

    def test_rollback_to_snapshot(self, tmp_path: Path) -> None:
        """Can rollback to pre-stage snapshot."""
        # Create initial state and snapshot
        graph = Graph.empty()
        graph.add_node("original", {"type": "test"})
        save_snapshot(graph, tmp_path, "dream")

        # Modify graph (simulating stage execution)
        graph.add_node("added_by_dream", {"type": "test"})
        graph.save(tmp_path / "graph.json")

        # Verify current state has new node
        current = Graph.load(tmp_path)
        assert current.has_node("added_by_dream")

        # Rollback
        restored = rollback_to_snapshot(tmp_path, "dream")

        assert restored.has_node("original")
        assert not restored.has_node("added_by_dream")

        # Verify graph.json was also updated
        reloaded = Graph.load(tmp_path)
        assert not reloaded.has_node("added_by_dream")

    def test_rollback_nonexistent_snapshot_raises(self, tmp_path: Path) -> None:
        """Rolling back to nonexistent snapshot raises ValueError."""
        with pytest.raises(ValueError, match="No snapshot"):
            rollback_to_snapshot(tmp_path, "nonexistent")

    def test_list_snapshots(self, tmp_path: Path) -> None:
        """Can list available snapshots."""
        graph = Graph.empty()

        # Initially empty
        assert list_snapshots(tmp_path) == []

        # Add snapshots
        save_snapshot(graph, tmp_path, "dream")
        save_snapshot(graph, tmp_path, "brainstorm")
        save_snapshot(graph, tmp_path, "seed")

        snapshots = list_snapshots(tmp_path)
        assert "dream" in snapshots
        assert "brainstorm" in snapshots
        assert "seed" in snapshots

    def test_delete_snapshot(self, tmp_path: Path) -> None:
        """Can delete a specific snapshot."""
        graph = Graph.empty()
        save_snapshot(graph, tmp_path, "dream")

        assert delete_snapshot(tmp_path, "dream")
        assert "dream" not in list_snapshots(tmp_path)

        # Deleting nonexistent returns False
        assert not delete_snapshot(tmp_path, "nonexistent")

    def test_cleanup_old_snapshots(self, tmp_path: Path) -> None:
        """Can cleanup old snapshots keeping only recent ones."""
        import time

        graph = Graph.empty()

        # Create snapshots with slight time gaps
        for stage in ["dream", "brainstorm", "seed", "grow"]:
            save_snapshot(graph, tmp_path, stage)
            time.sleep(0.01)  # Ensure different mtime

        # Keep only 2 most recent
        deleted = cleanup_old_snapshots(tmp_path, keep_count=2)

        assert len(deleted) == 2
        remaining = list_snapshots(tmp_path)
        assert len(remaining) == 2
        # Most recent should remain
        assert "grow" in remaining
        assert "seed" in remaining
