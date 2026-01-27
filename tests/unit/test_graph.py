"""Tests for the Graph class and snapshot management."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from questfoundry.graph import Graph
from questfoundry.graph.errors import (
    EdgeEndpointError,
    GraphIntegrityError,
    NodeExistsError,
    NodeNotFoundError,
    NodeReferencedError,
)
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

        graph.create_node("test", {"type": "test"})
        graph.set_last_stage("dream")
        assert "nodes=1" in repr(graph)
        assert "last_stage=dream" in repr(graph)


class TestNodeOperations:
    """Test node CRUD operations."""

    def test_create_node(self) -> None:
        """Can create a new node."""
        graph = Graph.empty()
        graph.create_node("char_001", {"type": "entity", "name": "Alice"})

        assert graph.has_node("char_001")
        node = graph.get_node("char_001")
        assert node is not None
        assert node["type"] == "entity"
        assert node["name"] == "Alice"

    def test_create_duplicate_node_raises_node_exists_error(self) -> None:
        """Creating duplicate node raises NodeExistsError."""
        graph = Graph.empty()
        graph.create_node("char_001", {"type": "entity"})

        with pytest.raises(NodeExistsError) as exc_info:
            graph.create_node("char_001", {"type": "entity"})

        assert exc_info.value.node_id == "char_001"

    def test_update_node_with_kwargs(self) -> None:
        """Can update existing node fields using kwargs."""
        graph = Graph.empty()
        graph.create_node("char_001", {"type": "entity", "name": "Alice"})

        graph.update_node("char_001", disposition="retained", status="active")

        node = graph.get_node("char_001")
        assert node is not None
        assert node["name"] == "Alice"  # Original preserved
        assert node["disposition"] == "retained"  # New field added
        assert node["status"] == "active"  # Another new field

    def test_update_nonexistent_node_raises_node_not_found_error(self) -> None:
        """Updating nonexistent node raises NodeNotFoundError with context."""
        graph = Graph.empty()
        graph.create_node("entity::alice", {"type": "entity"})
        graph.create_node("entity::bob", {"type": "entity"})

        with pytest.raises(NodeNotFoundError) as exc_info:
            graph.update_node("entity::charlie", disposition="cut")

        error = exc_info.value
        assert error.node_id == "entity::charlie"
        assert "entity::alice" in error.available
        assert "entity::bob" in error.available
        assert "update_node" in error.context

    def test_upsert_node_creates(self) -> None:
        """upsert_node creates new node and returns True."""
        graph = Graph.empty()
        created = graph.upsert_node("char_001", {"type": "entity"})

        assert created is True
        assert graph.has_node("char_001")

    def test_upsert_node_updates(self) -> None:
        """upsert_node updates existing node and returns False."""
        graph = Graph.empty()
        graph.create_node("char_001", {"type": "entity", "name": "Alice"})

        created = graph.upsert_node("char_001", {"type": "entity", "name": "Bob"})

        assert created is False
        assert graph.get_node("char_001")["name"] == "Bob"

    def test_delete_node(self) -> None:
        """Can delete an unreferenced node."""
        graph = Graph.empty()
        graph.create_node("char_001", {"type": "entity"})
        assert graph.has_node("char_001")

        graph.delete_node("char_001")
        assert not graph.has_node("char_001")

    def test_delete_nonexistent_node_raises(self) -> None:
        """Deleting nonexistent node raises NodeNotFoundError."""
        graph = Graph.empty()

        with pytest.raises(NodeNotFoundError) as exc_info:
            graph.delete_node("missing")

        assert exc_info.value.node_id == "missing"

    def test_delete_referenced_node_raises(self) -> None:
        """Deleting node referenced by edge raises NodeReferencedError."""
        graph = Graph.empty()
        graph.create_node("path::main", {"type": "path"})
        graph.create_node("beat::intro", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::intro", "path::main")

        with pytest.raises(NodeReferencedError) as exc_info:
            graph.delete_node("path::main")

        error = exc_info.value
        assert error.node_id == "path::main"
        assert len(error.referenced_by) == 1

    def test_delete_node_cascade(self) -> None:
        """Can delete referenced node with cascade=True."""
        graph = Graph.empty()
        graph.create_node("path::main", {"type": "path"})
        graph.create_node("beat::intro", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::intro", "path::main")

        # Verify edge exists
        assert len(graph.get_edges(to_id="path::main")) == 1

        # Delete with cascade
        graph.delete_node("path::main", cascade=True)

        assert not graph.has_node("path::main")
        assert len(graph.get_edges(to_id="path::main")) == 0

    def test_get_node_returns_none_for_missing(self) -> None:
        """get_node returns None for missing node."""
        graph = Graph.empty()
        assert graph.get_node("missing") is None

    def test_has_node(self) -> None:
        """has_node returns correct boolean."""
        graph = Graph.empty()
        assert not graph.has_node("char_001")

        graph.create_node("char_001", {"type": "entity"})
        assert graph.has_node("char_001")

    def test_get_nodes_by_type(self) -> None:
        """Can filter nodes by type."""
        graph = Graph.empty()
        graph.create_node("char_001", {"type": "entity", "entity_type": "character"})
        graph.create_node("char_002", {"type": "entity", "entity_type": "character"})
        graph.create_node("dilemma_001", {"type": "dilemma", "question": "?"})
        graph.create_node("vision", {"type": "vision", "genre": "noir"})

        entities = graph.get_nodes_by_type("entity")
        assert len(entities) == 2
        assert "char_001" in entities
        assert "char_002" in entities

        dilemmas = graph.get_nodes_by_type("dilemma")
        assert len(dilemmas) == 1
        assert "dilemma_001" in dilemmas

        visions = graph.get_nodes_by_type("vision")
        assert len(visions) == 1


class TestDeprecatedNodeMethods:
    """Test deprecated node methods emit warnings."""

    def test_add_node_deprecated(self) -> None:
        """add_node() emits deprecation warning."""
        graph = Graph.empty()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graph.add_node("test", {"type": "test"})

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "create_node" in str(w[0].message)

    def test_set_node_deprecated(self) -> None:
        """set_node() emits deprecation warning."""
        graph = Graph.empty()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graph.set_node("test", {"type": "test"})

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "create_node" in str(w[0].message) or "update_node" in str(w[0].message)


class TestNodeReference:
    """Test ref() validated reference helper."""

    def test_ref_returns_full_id(self) -> None:
        """ref() returns full node ID for existing node."""
        graph = Graph.empty()
        graph.create_node("entity::kay", {"type": "entity"})

        ref_id = graph.ref("entity", "kay")
        assert ref_id == "entity::kay"

    def test_ref_raises_for_missing_node(self) -> None:
        """ref() raises NodeNotFoundError for missing node."""
        graph = Graph.empty()
        graph.create_node("entity::alice", {"type": "entity"})

        with pytest.raises(NodeNotFoundError) as exc_info:
            graph.ref("entity", "bob")

        error = exc_info.value
        assert error.node_id == "entity::bob"
        assert "entity::alice" in error.available

    def test_ref_provides_available_ids_by_type(self) -> None:
        """ref() error includes available IDs of the same type."""
        graph = Graph.empty()
        graph.create_node("entity::alice", {"type": "entity"})
        graph.create_node("entity::bob", {"type": "entity"})
        graph.create_node("path::main", {"type": "path"})  # Different type

        with pytest.raises(NodeNotFoundError) as exc_info:
            graph.ref("entity", "charlie")

        error = exc_info.value
        # Should include entity IDs but not path IDs
        assert "entity::alice" in error.available
        assert "entity::bob" in error.available
        assert "path::main" not in error.available

    def test_ref_rejects_double_prefixed_id(self) -> None:
        """ref() raises ValueError if raw_id already contains prefix."""
        graph = Graph.empty()
        graph.create_node("entity::kay", {"type": "entity"})

        with pytest.raises(ValueError) as exc_info:
            graph.ref("entity", "entity::kay")

        assert "::" in str(exc_info.value)
        assert "kay" in str(exc_info.value)


class TestEdgeOperations:
    """Test edge operations."""

    def test_add_edge(self) -> None:
        """Can add edges between existing nodes."""
        graph = Graph.empty()
        graph.create_node("tension_001", {"type": "dilemma"})
        graph.create_node("alt_001", {"type": "alternative"})

        graph.add_edge("has_answer", "tension_001", "alt_001")

        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[0]["type"] == "has_answer"
        assert edges[0]["from"] == "tension_001"
        assert edges[0]["to"] == "alt_001"

    def test_add_edge_with_props(self) -> None:
        """Can add edges with additional properties."""
        graph = Graph.empty()
        graph.create_node("scene_001", {"type": "scene"})
        graph.create_node("scene_002", {"type": "scene"})
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

    def test_add_edge_validates_from_endpoint(self) -> None:
        """add_edge raises EdgeEndpointError if source doesn't exist."""
        graph = Graph.empty()
        graph.create_node("entity::bob", {"type": "entity"})

        with pytest.raises(EdgeEndpointError) as exc_info:
            graph.add_edge("relates_to", "entity::alice", "entity::bob")

        error = exc_info.value
        assert error.edge_type == "relates_to"
        assert error.from_id == "entity::alice"
        assert error.missing == "from"
        # available_from shows all entity IDs that could be used as source
        assert "entity::bob" in error.available_from

    def test_add_edge_validates_to_endpoint(self) -> None:
        """add_edge raises EdgeEndpointError if target doesn't exist."""
        graph = Graph.empty()
        graph.create_node("entity::alice", {"type": "entity"})

        with pytest.raises(EdgeEndpointError) as exc_info:
            graph.add_edge("relates_to", "entity::alice", "entity::bob")

        error = exc_info.value
        assert error.edge_type == "relates_to"
        assert error.to_id == "entity::bob"
        assert error.missing == "to"

    def test_add_edge_validates_both_endpoints(self) -> None:
        """add_edge raises EdgeEndpointError if both endpoints missing."""
        graph = Graph.empty()

        with pytest.raises(EdgeEndpointError) as exc_info:
            graph.add_edge("relates_to", "entity::alice", "entity::bob")

        error = exc_info.value
        assert error.missing == "both"

    def test_add_edge_skip_validation(self) -> None:
        """add_edge with validate=False allows missing endpoints."""
        graph = Graph.empty()

        # Should not raise
        graph.add_edge("relates_to", "missing_a", "missing_b", validate=False)

        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[0]["from"] == "missing_a"
        assert edges[0]["to"] == "missing_b"

    def test_add_edge_provides_available_ids(self) -> None:
        """EdgeEndpointError includes available IDs of matching type."""
        graph = Graph.empty()
        graph.create_node("path::main", {"type": "path"})
        graph.create_node("path::side", {"type": "path"})
        graph.create_node("beat::intro", {"type": "beat"})

        with pytest.raises(EdgeEndpointError) as exc_info:
            graph.add_edge("belongs_to", "beat::intro", "path::missing")

        error = exc_info.value
        # Should suggest path IDs for the target
        assert "path::main" in error.available_to
        assert "path::side" in error.available_to

    def test_get_edges_filter_by_from(self) -> None:
        """Can filter edges by source node."""
        graph = Graph.empty()
        graph.create_node("t1", {"type": "dilemma"})
        graph.create_node("t2", {"type": "dilemma"})
        graph.create_node("a1", {"type": "alternative"})
        graph.create_node("a2", {"type": "alternative"})
        graph.create_node("a3", {"type": "alternative"})
        graph.add_edge("has_alt", "t1", "a1")
        graph.add_edge("has_alt", "t1", "a2")
        graph.add_edge("has_alt", "t2", "a3")

        edges = graph.get_edges(from_id="t1")
        assert len(edges) == 2

    def test_get_edges_filter_by_to(self) -> None:
        """Can filter edges by target node."""
        graph = Graph.empty()
        graph.create_node("beat_1", {"type": "beat"})
        graph.create_node("beat_2", {"type": "beat"})
        graph.create_node("beat_3", {"type": "beat"})
        graph.create_node("thread_1", {"type": "path"})
        graph.create_node("thread_2", {"type": "path"})
        graph.add_edge("belongs_to", "beat_1", "thread_1")
        graph.add_edge("belongs_to", "beat_2", "thread_1")
        graph.add_edge("belongs_to", "beat_3", "thread_2")

        edges = graph.get_edges(to_id="thread_1")
        assert len(edges) == 2

    def test_get_edges_filter_by_type(self) -> None:
        """Can filter edges by type."""
        graph = Graph.empty()
        graph.create_node("t1", {"type": "dilemma"})
        graph.create_node("t2", {"type": "dilemma"})
        graph.create_node("a1", {"type": "alternative"})
        graph.create_node("a2", {"type": "alternative"})
        graph.create_node("thread_1", {"type": "path"})
        graph.add_edge("has_answer", "t1", "a1")
        graph.add_edge("explores", "thread_1", "a1")
        graph.add_edge("has_answer", "t2", "a2")

        edges = graph.get_edges(edge_type="has_answer")
        assert len(edges) == 2

    def test_get_edges_multiple_filters(self) -> None:
        """Can combine multiple filters."""
        graph = Graph.empty()
        graph.create_node("t1", {"type": "dilemma"})
        graph.create_node("a1", {"type": "alternative"})
        graph.create_node("a2", {"type": "alternative"})
        graph.create_node("th1", {"type": "path"})
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
        graph.create_node("char_001", {"type": "entity", "name": "Alice"})
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
        graph.create_node("test", {"type": "test"})
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
        graph.create_node("n1", {"type": "entity", "data": [1, 2, 3]})
        graph.create_node("n2", {"type": "entity"})
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
        graph.create_node("before", {"type": "test"})

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
        graph.create_node("original", {"type": "test"})
        save_snapshot(graph, tmp_path, "dream")

        # Modify graph (simulating stage execution)
        graph.create_node("added_by_dream", {"type": "test"})
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


class TestGraphIntegrityErrors:
    """Test error types and their LLM feedback formatting."""

    def test_node_not_found_error_basic(self) -> None:
        """NodeNotFoundError has correct attributes."""
        error = NodeNotFoundError(
            node_id="entity::missing",
            available=["entity::alice", "entity::bob"],
            context="update_node",
        )

        assert error.node_id == "entity::missing"
        assert "entity::alice" in error.available
        assert "update_node" in error.context
        assert "entity::missing" in str(error)

    def test_node_not_found_error_llm_feedback(self) -> None:
        """NodeNotFoundError produces actionable LLM feedback."""
        error = NodeNotFoundError(
            node_id="entity::charlei",  # Typo
            available=["entity::alice", "entity::bob", "entity::charlie"],
            context="edge creation",
        )

        feedback = error.to_llm_feedback()

        # Contains the problematic ID
        assert "entity::charlei" in feedback
        # Contains available IDs
        assert "entity::alice" in feedback
        # Suggests similar IDs (typo correction)
        assert "entity::charlie" in feedback
        # Includes context
        assert "edge creation" in feedback

    def test_node_not_found_error_suggestions(self) -> None:
        """NodeNotFoundError suggests similar IDs for typos."""
        error = NodeNotFoundError(
            node_id="dilemma::strom_aftermath",  # Typo: strom -> storm
            available=[
                "dilemma::storm_aftermath",
                "dilemma::family_secret",
                "dilemma::power_struggle",
            ],
        )

        feedback = error.to_llm_feedback()

        # Should suggest storm_aftermath as close match
        assert "storm_aftermath" in feedback

    def test_node_exists_error(self) -> None:
        """NodeExistsError has correct attributes and feedback."""
        error = NodeExistsError(node_id="entity::alice")

        assert error.node_id == "entity::alice"
        assert "entity::alice" in str(error)
        assert "already exists" in str(error)

        feedback = error.to_llm_feedback()
        assert "entity::alice" in feedback
        assert "already exists" in feedback

    def test_node_referenced_error(self) -> None:
        """NodeReferencedError shows referencing edges."""
        error = NodeReferencedError(
            node_id="path::main",
            referenced_by=[
                {"type": "belongs_to", "from": "beat::intro", "to": "path::main"},
                {"type": "belongs_to", "from": "beat::climax", "to": "path::main"},
            ],
        )

        assert error.node_id == "path::main"
        assert len(error.referenced_by) == 2

        feedback = error.to_llm_feedback()
        assert "path::main" in feedback
        assert "belongs_to" in feedback
        assert "beat::intro" in feedback

    def test_edge_endpoint_error_from_missing(self) -> None:
        """EdgeEndpointError for missing source."""
        error = EdgeEndpointError(
            edge_type="relates_to",
            from_id="entity::missing",
            to_id="entity::alice",
            missing="from",
            available_from=["entity::bob", "entity::charlie"],
            available_to=["entity::alice"],
        )

        assert error.missing == "from"
        feedback = error.to_llm_feedback()

        assert "entity::missing" in feedback
        assert "entity::bob" in feedback or "entity::charlie" in feedback
        assert "Source node" in feedback or "source" in feedback.lower()

    def test_edge_endpoint_error_to_missing(self) -> None:
        """EdgeEndpointError for missing target."""
        error = EdgeEndpointError(
            edge_type="belongs_to",
            from_id="beat::intro",
            to_id="path::missing",
            missing="to",
            available_from=["beat::intro"],
            available_to=["path::main", "path::side"],
        )

        assert error.missing == "to"
        feedback = error.to_llm_feedback()

        assert "path::missing" in feedback
        assert "path::main" in feedback

    def test_edge_endpoint_error_both_missing(self) -> None:
        """EdgeEndpointError for both endpoints missing."""
        error = EdgeEndpointError(
            edge_type="relates_to",
            from_id="entity::a",
            to_id="entity::b",
            missing="both",
            available_from=[],
            available_to=[],
        )

        assert error.missing == "both"
        feedback = error.to_llm_feedback()

        assert "entity::a" in feedback
        assert "entity::b" in feedback

    def test_graph_integrity_error_is_exception(self) -> None:
        """All error types inherit from GraphIntegrityError and Exception."""
        errors = [
            NodeNotFoundError("test"),
            NodeExistsError("test"),
            NodeReferencedError("test"),
            EdgeEndpointError("type", "from", "to", "from"),
        ]

        for error in errors:
            assert isinstance(error, GraphIntegrityError)
            assert isinstance(error, Exception)


class TestGraphCorruptionError:
    """Tests for GraphCorruptionError."""

    def test_basic_attributes(self) -> None:
        """GraphCorruptionError stores violations and stage."""
        from questfoundry.graph.errors import GraphCorruptionError

        error = GraphCorruptionError(
            violations=["Edge 0 missing 'to' field", "Edge 1 source does not exist"],
            stage="brainstorm",
        )

        assert error.violations == [
            "Edge 0 missing 'to' field",
            "Edge 1 source does not exist",
        ]
        assert error.stage == "brainstorm"
        assert "brainstorm" in str(error)
        # Both violations should appear in string representation
        assert "Edge 0" in str(error)
        assert "Edge 1" in str(error)

    def test_str_shows_violations(self) -> None:
        """String representation shows violations."""
        from questfoundry.graph.errors import GraphCorruptionError

        error = GraphCorruptionError(
            violations=["Dangling edge to missing node"],
            stage="seed",
        )

        s = str(error)
        assert "seed" in s
        assert "Dangling edge" in s

    def test_str_truncates_many_violations(self) -> None:
        """String truncates when there are many violations."""
        from questfoundry.graph.errors import GraphCorruptionError

        violations = [f"Violation {i}" for i in range(10)]
        error = GraphCorruptionError(violations=violations, stage="grow")

        s = str(error)
        # Should show first 5 (matches log limit in orchestrator)
        assert "Violation 0" in s
        assert "Violation 4" in s
        # Violation 5+ should not be shown directly
        assert "Violation 5" not in s or "5 more" in s
        # Should indicate more
        assert "5 more" in s


class TestGraphValidateInvariants:
    """Tests for Graph.validate_invariants()."""

    def test_valid_graph_returns_empty(self) -> None:
        """Valid graph with correct edges returns no violations."""
        graph = Graph.empty()
        graph.create_node("entity::alice", {"type": "entity"})
        graph.create_node("entity::bob", {"type": "entity"})
        graph.add_edge("relates_to", "entity::alice", "entity::bob")

        violations = graph.validate_invariants()
        assert violations == []

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph has no violations."""
        graph = Graph.empty()
        violations = graph.validate_invariants()
        assert violations == []

    def test_detects_missing_edge_source(self) -> None:
        """Detects edge with non-existent source."""
        graph = Graph.empty()
        graph.create_node("entity::bob", {"type": "entity"})
        # Add edge with validation disabled (simulating data corruption)
        graph.add_edge("relates_to", "entity::missing", "entity::bob", validate=False)

        violations = graph.validate_invariants()
        assert len(violations) == 1
        assert "entity::missing" in violations[0]
        assert "source" in violations[0]

    def test_detects_missing_edge_target(self) -> None:
        """Detects edge with non-existent target."""
        graph = Graph.empty()
        graph.create_node("entity::alice", {"type": "entity"})
        # Add edge with validation disabled (simulating data corruption)
        graph.add_edge("relates_to", "entity::alice", "entity::missing", validate=False)

        violations = graph.validate_invariants()
        assert len(violations) == 1
        assert "entity::missing" in violations[0]
        assert "target" in violations[0]

    def test_detects_edge_missing_type(self) -> None:
        """Detects edge missing type field."""
        graph = Graph.empty()
        graph.create_node("a", {"type": "test"})
        graph.create_node("b", {"type": "test"})
        # Manually add malformed edge
        graph._data["edges"].append({"from": "a", "to": "b"})

        violations = graph.validate_invariants()
        assert len(violations) == 1
        assert "missing 'type'" in violations[0]

    def test_detects_edge_missing_from(self) -> None:
        """Detects edge missing from field and target not existing."""
        graph = Graph.empty()
        graph._data["edges"].append({"type": "test", "to": "b"})

        violations = graph.validate_invariants()
        assert len(violations) == 2
        expected_violations = {
            "Edge 0 missing 'from' field",
            "Edge 0 (test): target 'b' does not exist",
        }
        assert set(violations) == expected_violations

    def test_detects_edge_missing_to(self) -> None:
        """Detects edge missing to field and source not existing."""
        graph = Graph.empty()
        graph._data["edges"].append({"type": "test", "from": "a"})

        violations = graph.validate_invariants()
        assert len(violations) == 2
        expected_violations = {
            "Edge 0 missing 'to' field",
            "Edge 0 (test): source 'a' does not exist",
        }
        assert set(violations) == expected_violations

    def test_returns_multiple_violations(self) -> None:
        """Returns all violations found."""
        graph = Graph.empty()
        # Add two bad edges
        graph._data["edges"].append({"type": "test", "from": "x", "to": "y"})
        graph._data["edges"].append({"type": "test", "from": "a", "to": "b"})

        violations = graph.validate_invariants()
        # Each edge has 2 violations (source and target don't exist)
        assert len(violations) == 4
