"""Tests for SqliteGraphStore — the SQLite storage backend.

All tests use :memory: databases. Tests mirror the DictGraphStore test
patterns to verify both backends behave identically for the GraphStore protocol.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from questfoundry.graph.sqlite_store import SqliteGraphStore
from questfoundry.graph.store import GraphStore

if TYPE_CHECKING:
    from pathlib import Path


class TestSqliteStoreProtocol:
    """Verify SqliteGraphStore satisfies the GraphStore protocol."""

    def test_is_runtime_checkable(self) -> None:
        """SqliteGraphStore passes isinstance check for GraphStore."""
        store = SqliteGraphStore()
        assert isinstance(store, GraphStore)

    def test_default_state(self) -> None:
        """Default store has empty nodes and edges."""
        store = SqliteGraphStore()
        assert store.node_count() == 0
        assert store.edge_count() == 0
        assert store.get_meta("last_stage") is None


class TestSqliteStoreNodes:
    """Test node CRUD on SqliteGraphStore."""

    def test_set_and_get_node(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})
        node = store.get_node("entity::alice")
        assert node is not None
        assert node["name"] == "Alice"
        assert node["type"] == "entity"

    def test_get_missing_node_returns_none(self) -> None:
        store = SqliteGraphStore()
        assert store.get_node("missing") is None

    def test_has_node(self) -> None:
        store = SqliteGraphStore()
        assert not store.has_node("entity::alice")
        store.set_node("entity::alice", {"type": "entity"})
        assert store.has_node("entity::alice")

    def test_set_node_overwrites(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})
        store.set_node("entity::alice", {"type": "entity", "name": "Bob"})
        assert store.get_node("entity::alice")["name"] == "Bob"

    def test_update_node_fields(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})
        store.update_node_fields("entity::alice", disposition="retained", status="active")

        node = store.get_node("entity::alice")
        assert node is not None
        assert node["name"] == "Alice"  # preserved
        assert node["disposition"] == "retained"
        assert node["status"] == "active"

    def test_delete_node(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity"})
        store.delete_node("entity::alice")
        assert not store.has_node("entity::alice")

    def test_get_nodes_by_type(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity"})
        store.set_node("entity::bob", {"type": "entity"})
        store.set_node("dilemma::d1", {"type": "dilemma"})

        entities = store.get_nodes_by_type("entity")
        assert len(entities) == 2
        assert "entity::alice" in entities
        assert "entity::bob" in entities
        assert "dilemma::d1" not in entities

    def test_all_node_ids(self) -> None:
        store = SqliteGraphStore()
        store.set_node("a", {"type": "test"})
        store.set_node("b", {"type": "test"})
        assert set(store.all_node_ids()) == {"a", "b"}

    def test_node_ids_with_prefix(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity"})
        store.set_node("entity::bob", {"type": "entity"})
        store.set_node("path::main", {"type": "path"})

        entity_ids = store.node_ids_with_prefix("entity::")
        assert set(entity_ids) == {"entity::alice", "entity::bob"}

    def test_node_count(self) -> None:
        store = SqliteGraphStore()
        assert store.node_count() == 0
        store.set_node("a", {"type": "test"})
        assert store.node_count() == 1
        store.delete_node("a")
        assert store.node_count() == 0


class TestSqliteStoreEdges:
    """Test edge CRUD on SqliteGraphStore."""

    def test_add_and_get_edge(self) -> None:
        store = SqliteGraphStore()
        store.add_edge({"type": "relates_to", "from": "a", "to": "b"})
        edges = store.get_edges()
        assert len(edges) == 1
        assert edges[0]["type"] == "relates_to"
        assert edges[0]["from"] == "a"
        assert edges[0]["to"] == "b"

    def test_edge_with_extra_props(self) -> None:
        """Extra edge properties are preserved in data column."""
        store = SqliteGraphStore()
        store.add_edge(
            {
                "type": "choice",
                "from": "s1",
                "to": "s2",
                "label": "Go north",
                "requires": ["has_key"],
            }
        )
        edges = store.get_edges()
        assert edges[0]["label"] == "Go north"
        assert edges[0]["requires"] == ["has_key"]

    def test_get_edges_with_filters(self) -> None:
        store = SqliteGraphStore()
        store.add_edge({"type": "belongs_to", "from": "beat::1", "to": "path::a"})
        store.add_edge({"type": "belongs_to", "from": "beat::2", "to": "path::a"})
        store.add_edge({"type": "explores", "from": "path::a", "to": "dilemma::d"})

        assert len(store.get_edges(edge_type="belongs_to")) == 2
        assert len(store.get_edges(to_id="path::a")) == 2
        assert len(store.get_edges(from_id="beat::1")) == 1
        assert len(store.get_edges(from_id="path::a", edge_type="explores")) == 1

    def test_get_edges_preserves_insertion_order(self) -> None:
        """Edges are returned in insertion order (rowid)."""
        store = SqliteGraphStore()
        store.add_edge({"type": "a", "from": "x", "to": "y"})
        store.add_edge({"type": "b", "from": "x", "to": "z"})
        store.add_edge({"type": "c", "from": "x", "to": "w"})

        edges = store.get_edges()
        types = [e["type"] for e in edges]
        assert types == ["a", "b", "c"]

    def test_remove_edge(self) -> None:
        store = SqliteGraphStore()
        store.add_edge({"type": "link", "from": "a", "to": "b"})
        assert store.remove_edge("link", "a", "b") is True
        assert store.edge_count() == 0

    def test_remove_edge_no_match(self) -> None:
        store = SqliteGraphStore()
        assert store.remove_edge("link", "a", "b") is False

    def test_remove_edge_first_only(self) -> None:
        """remove_edge removes only the first matching edge."""
        store = SqliteGraphStore()
        store.add_edge({"type": "link", "from": "a", "to": "b"})
        store.add_edge({"type": "link", "from": "a", "to": "b"})
        assert store.edge_count() == 2

        store.remove_edge("link", "a", "b")
        assert store.edge_count() == 1

    def test_edges_referencing(self) -> None:
        store = SqliteGraphStore()
        store.add_edge({"type": "a", "from": "x", "to": "y"})
        store.add_edge({"type": "b", "from": "y", "to": "z"})
        store.add_edge({"type": "c", "from": "w", "to": "x"})

        refs = store.edges_referencing("x")
        assert len(refs) == 2  # from=x and to=x

    def test_remove_edges_referencing(self) -> None:
        store = SqliteGraphStore()
        store.add_edge({"type": "a", "from": "x", "to": "y"})
        store.add_edge({"type": "b", "from": "y", "to": "z"})
        store.add_edge({"type": "c", "from": "w", "to": "v"})

        store.remove_edges_referencing("y")
        assert store.edge_count() == 1
        assert store.get_edges()[0]["from"] == "w"

    def test_edge_count(self) -> None:
        store = SqliteGraphStore()
        assert store.edge_count() == 0
        store.add_edge({"type": "t", "from": "a", "to": "b"})
        assert store.edge_count() == 1
        store.remove_edge("t", "a", "b")
        assert store.edge_count() == 0


class TestSqliteStoreMeta:
    """Test metadata operations on SqliteGraphStore."""

    def test_set_and_get_meta(self) -> None:
        store = SqliteGraphStore()
        store.set_meta("project_name", "test")
        assert store.get_meta("project_name") == "test"

    def test_get_meta_missing_key(self) -> None:
        store = SqliteGraphStore()
        assert store.get_meta("nonexistent") is None

    def test_set_meta_overwrites(self) -> None:
        store = SqliteGraphStore()
        store.set_meta("key", "value1")
        store.set_meta("key", "value2")
        assert store.get_meta("key") == "value2"

    def test_all_meta(self) -> None:
        store = SqliteGraphStore()
        store.set_meta("project_name", "test")
        store.set_meta("last_stage", "dream")
        meta = store.all_meta()
        assert meta["project_name"] == "test"
        assert meta["last_stage"] == "dream"

    def test_meta_stores_complex_values(self) -> None:
        """Meta values can be lists, dicts, etc. (JSON-serialized)."""
        store = SqliteGraphStore()
        store.set_meta(
            "stage_history",
            [
                {"stage": "dream", "completed": "2024-01-01T00:00:00"},
            ],
        )
        history = store.get_meta("stage_history")
        assert len(history) == 1
        assert history[0]["stage"] == "dream"


class TestSqliteStoreMutations:
    """Test mutation recording."""

    def test_create_node_records_mutation(self) -> None:
        store = SqliteGraphStore()
        store.set_mutation_context(stage="seed", phase="serialize")
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})

        rows = store._conn.execute("SELECT * FROM mutations").fetchall()
        assert len(rows) == 1
        assert rows[0]["operation"] == "create_node"
        assert rows[0]["target_id"] == "entity::alice"
        assert rows[0]["stage"] == "seed"
        assert rows[0]["phase"] == "serialize"
        delta = json.loads(rows[0]["delta"])
        assert delta["name"] == "Alice"

    def test_update_node_records_mutation(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity"})

        store.set_mutation_context(stage="grow", phase="path_agnostic")
        store.update_node_fields("entity::alice", disposition="retained")

        rows = store._conn.execute(
            "SELECT * FROM mutations WHERE operation = 'update_node'"
        ).fetchall()
        assert len(rows) == 1
        delta = json.loads(rows[0]["delta"])
        assert delta == {"disposition": "retained"}

    def test_delete_node_records_mutation(self) -> None:
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity"})
        store.delete_node("entity::alice")

        rows = store._conn.execute(
            "SELECT * FROM mutations WHERE operation = 'delete_node'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["target_id"] == "entity::alice"

    def test_add_edge_records_mutation(self) -> None:
        store = SqliteGraphStore()
        store.add_edge({"type": "belongs_to", "from": "beat::1", "to": "path::a"})

        rows = store._conn.execute(
            "SELECT * FROM mutations WHERE operation = 'add_edge'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["target_id"] == "edge:belongs_to:beat::1:path::a"

    def test_remove_edge_records_mutation(self) -> None:
        store = SqliteGraphStore()
        store.add_edge({"type": "link", "from": "a", "to": "b"})
        store.remove_edge("link", "a", "b")

        rows = store._conn.execute(
            "SELECT * FROM mutations WHERE operation = 'remove_edge'"
        ).fetchall()
        assert len(rows) == 1

    def test_overwrite_node_records_update_mutation(self) -> None:
        """Second set_node on same ID records update_node."""
        store = SqliteGraphStore()
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})
        store.set_node("entity::alice", {"type": "entity", "name": "Bob"})

        rows = store._conn.execute("SELECT operation FROM mutations").fetchall()
        ops = [row["operation"] for row in rows]
        assert ops == ["create_node", "update_node"]

    def test_mutation_context_default_empty(self) -> None:
        """Without set_mutation_context, stage/phase are empty strings."""
        store = SqliteGraphStore()
        store.set_node("a", {"type": "test"})

        row = store._conn.execute("SELECT stage, phase FROM mutations").fetchone()
        assert row["stage"] == ""
        assert row["phase"] == ""

    def test_remove_edges_referencing_records_mutations(self) -> None:
        """remove_edges_referencing records a mutation for each removed edge."""
        store = SqliteGraphStore()
        store.add_edge({"type": "a", "from": "x", "to": "y"})
        store.add_edge({"type": "b", "from": "y", "to": "z"})
        # Clear the add_edge mutations
        store._conn.execute("DELETE FROM mutations")

        store.remove_edges_referencing("y")

        rows = store._conn.execute(
            "SELECT * FROM mutations WHERE operation = 'remove_edge'"
        ).fetchall()
        assert len(rows) == 2


class TestSqliteStoreSavepoints:
    """Test savepoint/rollback/release."""

    def test_savepoint_and_rollback(self) -> None:
        """Can rollback to a savepoint, undoing changes."""
        store = SqliteGraphStore()
        store.set_node("before", {"type": "test"})

        store.savepoint("phase1")
        store.set_node("during", {"type": "test"})
        assert store.has_node("during")

        store.rollback_to("phase1")
        assert not store.has_node("during")
        assert store.has_node("before")

    def test_savepoint_and_release(self) -> None:
        """Releasing a savepoint commits its changes."""
        store = SqliteGraphStore()
        store.savepoint("phase1")
        store.set_node("new", {"type": "test"})
        store.release("phase1")

        assert store.has_node("new")

    def test_nested_savepoints(self) -> None:
        """Nested savepoints work correctly."""
        store = SqliteGraphStore()
        store.set_node("base", {"type": "test"})

        store.savepoint("outer")
        store.set_node("outer_node", {"type": "test"})

        store.savepoint("inner")
        store.set_node("inner_node", {"type": "test"})

        # Rollback inner only
        store.rollback_to("inner")
        assert not store.has_node("inner_node")
        assert store.has_node("outer_node")

        # Release inner, rollback outer
        store.release("inner")
        store.rollback_to("outer")
        assert not store.has_node("outer_node")
        assert store.has_node("base")

    def test_rollback_edges(self) -> None:
        """Savepoint rollback also undoes edge changes."""
        store = SqliteGraphStore()
        store.add_edge({"type": "keep", "from": "a", "to": "b"})

        store.savepoint("phase")
        store.add_edge({"type": "temp", "from": "c", "to": "d"})
        assert store.edge_count() == 2

        store.rollback_to("phase")
        assert store.edge_count() == 1
        assert store.get_edges()[0]["type"] == "keep"

    def test_rollback_meta(self) -> None:
        """Savepoint rollback also undoes meta changes."""
        store = SqliteGraphStore()
        store.set_meta("last_stage", "seed")

        store.savepoint("phase")
        store.set_meta("last_stage", "grow")
        assert store.get_meta("last_stage") == "grow"

        store.rollback_to("phase")
        assert store.get_meta("last_stage") == "seed"


class TestSqliteStoreSerialization:
    """Test to_dict/from_dict round-trips."""

    def test_to_dict_format(self) -> None:
        """to_dict returns the expected graph dict format."""
        store = SqliteGraphStore()
        store.set_meta("project_name", "test")
        store.set_meta("last_stage", "dream")
        store.set_node("entity::alice", {"type": "entity", "name": "Alice"})
        store.add_edge({"type": "link", "from": "entity::alice", "to": "entity::alice"})

        d = store.to_dict()
        assert d["version"] == "5.0"
        assert d["meta"]["project_name"] == "test"
        assert "entity::alice" in d["nodes"]
        assert len(d["edges"]) == 1
        assert d["edges"][0]["type"] == "link"

    def test_from_dict_round_trip(self) -> None:
        """from_dict creates a store from to_dict output."""
        original = SqliteGraphStore()
        original.set_meta("project_name", "roundtrip")
        original.set_meta("stage_history", [])
        original.set_node("a", {"type": "test", "val": 42})
        original.add_edge({"type": "link", "from": "a", "to": "a", "weight": 1.5})

        data = original.to_dict()
        restored = SqliteGraphStore.from_dict(data)

        assert restored.get_node("a")["val"] == 42
        assert restored.edge_count() == 1
        assert restored.get_edges()[0]["weight"] == 1.5
        assert restored.get_meta("project_name") == "roundtrip"

    def test_from_dict_no_mutations_recorded(self) -> None:
        """from_dict bulk import does not record mutations."""
        data = {
            "version": "5.0",
            "meta": {"last_stage": "seed"},
            "nodes": {"a": {"type": "test"}},
            "edges": [{"type": "link", "from": "a", "to": "a"}],
        }
        store = SqliteGraphStore.from_dict(data)
        rows = store._conn.execute("SELECT COUNT(*) AS cnt FROM mutations").fetchone()
        assert rows["cnt"] == 0

    def test_from_dict_dict_store_compat(self) -> None:
        """SqliteGraphStore.from_dict is compatible with DictGraphStore.to_dict."""
        from questfoundry.graph.store import DictGraphStore

        dict_store = DictGraphStore()
        dict_store.set_node("entity::alice", {"type": "entity", "name": "Alice"})
        dict_store.set_node("entity::bob", {"type": "entity", "name": "Bob"})
        dict_store.add_edge({"type": "relates", "from": "entity::alice", "to": "entity::bob"})
        dict_store.set_meta("project_name", "cross_compat")
        dict_store.set_meta("last_stage", "dream")
        dict_store.set_meta("stage_history", [{"stage": "dream", "completed": "2024-01-01"}])

        data = dict_store.to_dict()
        sqlite_store = SqliteGraphStore.from_dict(data)

        # Verify all data migrated correctly
        assert sqlite_store.get_node("entity::alice")["name"] == "Alice"
        assert sqlite_store.get_node("entity::bob")["name"] == "Bob"
        assert sqlite_store.edge_count() == 1
        assert sqlite_store.get_meta("project_name") == "cross_compat"
        assert sqlite_store.get_meta("last_stage") == "dream"

        # And round-trip back to dict
        roundtrip = sqlite_store.to_dict()
        assert roundtrip["nodes"]["entity::alice"]["name"] == "Alice"
        assert len(roundtrip["edges"]) == 1


class TestSqliteStoreFileBackend:
    """Test file-based SQLite (not :memory:)."""

    def test_file_persistence(self, tmp_path: Path) -> None:
        """Data persists across open/close cycles."""
        db_path = tmp_path / "test.db"

        store = SqliteGraphStore(db_path)
        store.set_node("entity::alice", {"type": "entity"})
        store.set_meta("last_stage", "dream")
        store.close()

        store2 = SqliteGraphStore(db_path)
        assert store2.has_node("entity::alice")
        assert store2.get_meta("last_stage") == "dream"
        store2.close()
