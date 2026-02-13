"""Tests for DictGraphStore — the in-memory dict storage backend.

These tests exercise the GraphStore protocol through DictGraphStore directly,
independent of the Graph facade. Graph-level tests in test_graph.py continue
to verify the full stack.
"""

from __future__ import annotations

from questfoundry.graph.store import DictGraphStore, GraphStore


class TestDictGraphStoreProtocol:
    """Verify DictGraphStore satisfies the GraphStore protocol."""

    def test_is_runtime_checkable(self) -> None:
        """DictGraphStore passes isinstance check for GraphStore."""
        store = DictGraphStore()
        assert isinstance(store, GraphStore)

    def test_default_state(self) -> None:
        """Default store has empty nodes, edges, and default meta."""
        store = DictGraphStore()
        assert store.node_count() == 0
        assert store.edge_count() == 0
        assert store.get_meta("last_stage") is None
        assert store.get_meta("project_name") is None
        assert store.get_meta("stage_history") == []


class TestDictGraphStoreNodes:
    """Test node CRUD on DictGraphStore."""

    def test_set_and_get_node(self) -> None:
        """Can set and retrieve a node."""
        store = DictGraphStore()
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})

        node = store.get_node("entity::alice")
        assert node is not None
        assert node["name"] == "Alice"

    def test_get_missing_node_returns_none(self) -> None:
        """Getting a non-existent node returns None."""
        store = DictGraphStore()
        assert store.get_node("missing") is None

    def test_has_node(self) -> None:
        """has_node returns correct boolean."""
        store = DictGraphStore()
        assert not store.has_node("entity::alice")
        store.set_node("entity::alice", {"type": "entity"})
        assert store.has_node("entity::alice")

    def test_update_node_fields(self) -> None:
        """update_node_fields merges into existing data."""
        store = DictGraphStore()
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})
        store.update_node_fields("entity::alice", disposition="retained", status="active")

        node = store.get_node("entity::alice")
        assert node is not None
        assert node["name"] == "Alice"  # preserved
        assert node["disposition"] == "retained"  # added
        assert node["status"] == "active"  # added

    def test_delete_node(self) -> None:
        """delete_node removes the node."""
        store = DictGraphStore()
        store.set_node("entity::alice", {"type": "entity"})
        store.delete_node("entity::alice")
        assert not store.has_node("entity::alice")

    def test_get_nodes_by_type(self) -> None:
        """get_nodes_by_type filters by type field."""
        store = DictGraphStore()
        store.set_node("entity::alice", {"type": "entity"})
        store.set_node("entity::bob", {"type": "entity"})
        store.set_node("dilemma::d1", {"type": "dilemma"})

        entities = store.get_nodes_by_type("entity")
        assert len(entities) == 2
        assert "entity::alice" in entities
        assert "entity::bob" in entities
        assert "dilemma::d1" not in entities

    def test_all_node_ids(self) -> None:
        """all_node_ids returns all node IDs."""
        store = DictGraphStore()
        store.set_node("a", {"type": "test"})
        store.set_node("b", {"type": "test"})
        ids = store.all_node_ids()
        assert set(ids) == {"a", "b"}

    def test_node_ids_with_prefix(self) -> None:
        """node_ids_with_prefix filters by prefix."""
        store = DictGraphStore()
        store.set_node("entity::alice", {"type": "entity"})
        store.set_node("entity::bob", {"type": "entity"})
        store.set_node("path::main", {"type": "path"})

        entity_ids = store.node_ids_with_prefix("entity::")
        assert set(entity_ids) == {"entity::alice", "entity::bob"}

    def test_node_count(self) -> None:
        """node_count tracks additions and deletions."""
        store = DictGraphStore()
        assert store.node_count() == 0
        store.set_node("a", {"type": "test"})
        assert store.node_count() == 1
        store.set_node("b", {"type": "test"})
        assert store.node_count() == 2
        store.delete_node("a")
        assert store.node_count() == 1


class TestDictGraphStoreEdges:
    """Test edge CRUD on DictGraphStore."""

    def test_add_and_get_edge(self) -> None:
        """Can add and retrieve an edge."""
        store = DictGraphStore()
        store.add_edge({"type": "relates_to", "from": "a", "to": "b"})

        edges = store.get_edges()
        assert len(edges) == 1
        assert edges[0]["type"] == "relates_to"

    def test_get_edges_with_filters(self) -> None:
        """get_edges filters by from, to, and type."""
        store = DictGraphStore()
        store.add_edge({"type": "belongs_to", "from": "beat::1", "to": "path::a"})
        store.add_edge({"type": "belongs_to", "from": "beat::2", "to": "path::a"})
        store.add_edge({"type": "explores", "from": "path::a", "to": "dilemma::d"})

        assert len(store.get_edges(edge_type="belongs_to")) == 2
        assert len(store.get_edges(to_id="path::a")) == 2
        assert len(store.get_edges(from_id="beat::1")) == 1
        assert len(store.get_edges(from_id="path::a", edge_type="explores")) == 1

    def test_remove_edge(self) -> None:
        """remove_edge removes first match and returns True."""
        store = DictGraphStore()
        store.add_edge({"type": "link", "from": "a", "to": "b"})
        assert store.remove_edge("link", "a", "b") is True
        assert store.edge_count() == 0

    def test_remove_edge_no_match(self) -> None:
        """remove_edge returns False when no match found."""
        store = DictGraphStore()
        assert store.remove_edge("link", "a", "b") is False

    def test_edges_referencing(self) -> None:
        """edges_referencing finds all edges involving a node."""
        store = DictGraphStore()
        store.add_edge({"type": "a", "from": "x", "to": "y"})
        store.add_edge({"type": "b", "from": "y", "to": "z"})
        store.add_edge({"type": "c", "from": "w", "to": "x"})

        refs = store.edges_referencing("x")
        assert len(refs) == 2  # from=x and to=x

        refs_y = store.edges_referencing("y")
        assert len(refs_y) == 2  # to=y and from=y

    def test_remove_edges_referencing(self) -> None:
        """remove_edges_referencing deletes all edges involving a node."""
        store = DictGraphStore()
        store.add_edge({"type": "a", "from": "x", "to": "y"})
        store.add_edge({"type": "b", "from": "y", "to": "z"})
        store.add_edge({"type": "c", "from": "w", "to": "v"})

        store.remove_edges_referencing("y")
        assert store.edge_count() == 1
        assert store.get_edges()[0]["from"] == "w"

    def test_edge_count(self) -> None:
        """edge_count tracks additions and removals."""
        store = DictGraphStore()
        assert store.edge_count() == 0
        store.add_edge({"type": "t", "from": "a", "to": "b"})
        assert store.edge_count() == 1
        store.add_edge({"type": "t", "from": "c", "to": "d"})
        assert store.edge_count() == 2
        store.remove_edge("t", "a", "b")
        assert store.edge_count() == 1


class TestDictGraphStoreMeta:
    """Test metadata operations on DictGraphStore."""

    def test_set_and_get_meta(self) -> None:
        """Can set and get metadata values."""
        store = DictGraphStore()
        store.set_meta("project_name", "test")
        assert store.get_meta("project_name") == "test"

    def test_get_meta_missing_key(self) -> None:
        """get_meta returns None for missing keys."""
        store = DictGraphStore()
        assert store.get_meta("nonexistent") is None

    def test_all_meta(self) -> None:
        """all_meta returns the full metadata dict."""
        store = DictGraphStore()
        meta = store.all_meta()
        assert "project_name" in meta
        assert "last_stage" in meta
        assert "stage_history" in meta


class TestDictGraphStoreSerialization:
    """Test serialization round-trips."""

    def test_to_dict_returns_deep_copy(self) -> None:
        """to_dict returns a deep copy (mutations don't leak)."""
        store = DictGraphStore()
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})

        d = store.to_dict()
        d["nodes"]["entity::alice"]["name"] = "MUTATED"

        # Original store unchanged
        assert store.get_node("entity::alice")["name"] == "Alice"

    def test_from_dict_round_trip(self) -> None:
        """from_dict creates a store from to_dict output."""
        store = DictGraphStore()
        store.set_node("a", {"type": "test", "val": 42})
        store.add_edge({"type": "link", "from": "a", "to": "a", "weight": 1.5})
        store.set_meta("project_name", "roundtrip")

        data = store.to_dict()
        restored = DictGraphStore.from_dict(data)

        assert restored.get_node("a")["val"] == 42
        assert restored.edge_count() == 1
        assert restored.get_meta("project_name") == "roundtrip"

    def test_to_dict_has_version(self) -> None:
        """to_dict output includes version field."""
        store = DictGraphStore()
        d = store.to_dict()
        assert d["version"] == "5.0"
