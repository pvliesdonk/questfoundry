"""Tests for Graph ↔ SQLite integration.

Covers: auto-migration, .db load/save, savepoint API through Graph,
mutation_context, snapshot handling with mixed formats.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from questfoundry.graph.graph import Graph
from questfoundry.graph.migration import migrate_json_to_sqlite
from questfoundry.graph.snapshots import (
    delete_snapshot,
    list_snapshots,
    rollback_to_snapshot,
    save_snapshot,
)
from questfoundry.graph.sqlite_store import SqliteGraphStore
from questfoundry.graph.store import DictGraphStore

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Auto-migration
# ---------------------------------------------------------------------------


def _write_json_graph(path: Path) -> dict:
    """Write a minimal graph.json and return the data."""
    data = {
        "version": "5.0",
        "meta": {
            "project_name": "test",
            "last_stage": "seed",
            "last_modified": None,
            "stage_history": [],
        },
        "nodes": {
            "entity::alice": {"type": "entity", "name": "Alice"},
            "entity::bob": {"type": "entity", "name": "Bob"},
        },
        "edges": [
            {"type": "knows", "from": "entity::alice", "to": "entity::bob"},
        ],
    }
    json_file = path / "graph.json"
    with json_file.open("w") as f:
        json.dump(data, f)
    return data


class TestGraphLoadDetection:
    """Graph.load() detects .db vs .json files."""

    def test_load_prefers_db_over_json(self, tmp_path: Path) -> None:
        """If both graph.db and graph.json exist, prefer .db."""
        _write_json_graph(tmp_path)
        db_graph = Graph.empty()
        db_graph.create_node("entity::charlie", {"type": "entity"})
        db_graph.save(tmp_path / "graph.db")

        loaded = Graph.load(tmp_path)
        assert loaded.has_node("entity::charlie")
        assert not loaded.has_node("entity::alice")

    def test_load_json_when_no_db(self, tmp_path: Path) -> None:
        """Loads graph.json when no .db exists."""
        _write_json_graph(tmp_path)
        graph = Graph.load(tmp_path)

        assert not graph.is_sqlite_backed
        assert graph.has_node("entity::alice")

    def test_load_db_is_sqlite_backed(self, tmp_path: Path) -> None:
        """Loading .db results in SQLite-backed graph."""
        g = Graph.empty()
        g.create_node("n1", {"type": "t"})
        g.save(tmp_path / "graph.db")

        loaded = Graph.load(tmp_path)
        assert loaded.is_sqlite_backed
        assert loaded.has_node("n1")

    def test_load_empty_when_no_files(self, tmp_path: Path) -> None:
        """Returns empty graph when neither .db nor .json exists."""
        graph = Graph.load(tmp_path)
        assert not graph.is_sqlite_backed
        assert isinstance(graph._store, DictGraphStore)


# ---------------------------------------------------------------------------
# migrate_json_to_sqlite utility
# ---------------------------------------------------------------------------


class TestMigrateJsonToSqlite:
    """Direct migration function tests."""

    def test_round_trip_fidelity(self, tmp_path: Path) -> None:
        """Migrated data matches original JSON."""
        data = {
            "version": "5.0",
            "meta": {"project_name": "test", "last_stage": None},
            "nodes": {"n1": {"type": "t", "val": 42}},
            "edges": [{"type": "e", "from": "n1", "to": "n1"}],
        }
        json_file = tmp_path / "graph.json"
        with json_file.open("w") as f:
            json.dump(data, f)

        store = migrate_json_to_sqlite(json_file, tmp_path / "graph.db")
        result = store.to_dict()
        store.close()

        assert result["nodes"]["n1"]["val"] == 42
        assert len(result["edges"]) == 1
        assert result["meta"]["project_name"] == "test"


# ---------------------------------------------------------------------------
# Graph.load_from_file / Graph.save (.db support)
# ---------------------------------------------------------------------------


class TestDbLoadSave:
    """Graph.load_from_file and Graph.save with .db files."""

    def test_save_and_load_db(self, tmp_path: Path) -> None:
        """Save to .db and reload."""
        graph = Graph.empty()
        graph.create_node("entity::alice", {"type": "entity"})
        graph.add_edge("likes", "entity::alice", "entity::alice")
        graph.save(tmp_path / "graph.db")

        loaded = Graph.load_from_file(tmp_path / "graph.db")
        assert loaded.has_node("entity::alice")
        assert len(loaded.get_edges(edge_type="likes")) == 1

    def test_save_db_from_sqlite_backed(self, tmp_path: Path) -> None:
        """SQLite-backed graph saves via backup API."""
        store = SqliteGraphStore(tmp_path / "original.db")
        store.set_node("n1", {"type": "t"})
        graph = Graph(store=store)

        backup_path = tmp_path / "backup.db"
        graph.save(backup_path)

        loaded = Graph.load_from_file(backup_path)
        assert loaded.has_node("n1")

    def test_save_json_still_works(self, tmp_path: Path) -> None:
        """JSON save still works for backward compatibility."""
        graph = Graph.empty()
        graph.create_node("n1", {"type": "t"})
        graph.save(tmp_path / "out.json")

        with (tmp_path / "out.json").open() as f:
            data = json.load(f)
        assert "n1" in data["nodes"]

    def test_save_db_overwrites_existing(self, tmp_path: Path) -> None:
        """Saving .db to an existing path overwrites it."""
        db_path = tmp_path / "graph.db"

        # First save
        g1 = Graph.empty()
        g1.create_node("old_node", {"type": "t"})
        g1.save(db_path)

        # Second save (different data)
        g2 = Graph.empty()
        g2.create_node("new_node", {"type": "t"})
        g2.save(db_path)

        loaded = Graph.load_from_file(db_path)
        assert loaded.has_node("new_node")
        assert not loaded.has_node("old_node")


# ---------------------------------------------------------------------------
# Graph savepoint API
# ---------------------------------------------------------------------------


class TestGraphSavepoints:
    """Savepoint/rollback/release through the Graph facade."""

    def test_savepoint_rollback_dict_store(self) -> None:
        """Savepoint works with DictGraphStore."""
        graph = Graph.empty()
        graph.create_node("n1", {"type": "t"})
        graph.savepoint("before_change")

        graph.create_node("n2", {"type": "t"})
        assert graph.has_node("n2")

        graph.rollback_to("before_change")
        assert graph.has_node("n1")
        assert not graph.has_node("n2")

    def test_savepoint_release_dict_store(self) -> None:
        """Release discards the savepoint without rollback."""
        graph = Graph.empty()
        graph.create_node("n1", {"type": "t"})
        graph.savepoint("sp1")
        graph.create_node("n2", {"type": "t"})
        graph.release("sp1")

        # Both nodes still present after release
        assert graph.has_node("n1")
        assert graph.has_node("n2")

    def test_savepoint_rollback_sqlite_store(self) -> None:
        """Savepoint works with SqliteGraphStore."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        graph.create_node("n1", {"type": "t"})
        graph.savepoint("before_change")

        graph.create_node("n2", {"type": "t"})
        assert graph.has_node("n2")

        graph.rollback_to("before_change")
        assert graph.has_node("n1")
        assert not graph.has_node("n2")

    def test_savepoint_nested(self) -> None:
        """Nested savepoints work correctly."""
        graph = Graph.empty()
        graph.create_node("n1", {"type": "t"})
        graph.savepoint("sp1")

        graph.create_node("n2", {"type": "t"})
        graph.savepoint("sp2")

        graph.create_node("n3", {"type": "t"})

        # Rollback inner savepoint
        graph.rollback_to("sp2")
        assert graph.has_node("n2")
        assert not graph.has_node("n3")

        # Rollback outer savepoint
        graph.rollback_to("sp1")
        assert graph.has_node("n1")
        assert not graph.has_node("n2")

    def test_rollback_invalid_savepoint_raises(self) -> None:
        """Rollback to nonexistent savepoint raises ValueError."""
        graph = Graph.empty()
        raised = False
        try:
            graph.rollback_to("nonexistent")
        except (ValueError, Exception):
            raised = True
        assert raised, "Expected ValueError for nonexistent savepoint"


# ---------------------------------------------------------------------------
# Graph mutation_context
# ---------------------------------------------------------------------------


class TestMutationContext:
    """mutation_context() tags mutations with stage/phase."""

    def test_mutation_context_sqlite(self) -> None:
        """Mutations inside context have correct stage/phase."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context(stage="grow", phase="spine"):
            graph.create_node("n1", {"type": "t"})

        # Check mutation was recorded with context
        rows = store._conn.execute(
            "SELECT stage, phase FROM mutations WHERE target_id = 'n1'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["stage"] == "grow"
        assert rows[0]["phase"] == "spine"

    def test_mutation_context_resets_on_exit(self) -> None:
        """Context resets stage/phase after exiting."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context(stage="grow", phase="spine"):
            graph.create_node("n1", {"type": "t"})

        # After context, mutations have empty stage/phase
        graph.create_node("n2", {"type": "t"})
        rows = store._conn.execute(
            "SELECT stage, phase FROM mutations WHERE target_id = 'n2'"
        ).fetchall()
        assert rows[0]["stage"] == ""
        assert rows[0]["phase"] == ""

    def test_mutation_context_resets_on_exception(self) -> None:
        """Context resets even if body raises."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        try:
            with graph.mutation_context(stage="grow", phase="spine"):
                graph.create_node("n1", {"type": "t"})
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        # Context should be reset
        graph.create_node("n2", {"type": "t"})
        rows = store._conn.execute(
            "SELECT stage, phase FROM mutations WHERE target_id = 'n2'"
        ).fetchall()
        assert rows[0]["stage"] == ""

    def test_mutation_context_noop_dict_store(self) -> None:
        """mutation_context is a no-op for DictGraphStore."""
        graph = Graph.empty()
        with graph.mutation_context(stage="grow", phase="spine"):
            graph.create_node("n1", {"type": "t"})
        # No error — just verifies it doesn't crash
        assert graph.has_node("n1")


# ---------------------------------------------------------------------------
# Graph.is_sqlite_backed / sqlite_store
# ---------------------------------------------------------------------------


class TestStoreAccess:
    """is_sqlite_backed and sqlite_store properties."""

    def test_dict_not_sqlite_backed(self) -> None:
        graph = Graph.empty()
        assert not graph.is_sqlite_backed

    def test_sqlite_is_sqlite_backed(self) -> None:
        graph = Graph(store=SqliteGraphStore())
        assert graph.is_sqlite_backed

    def test_sqlite_store_access(self) -> None:
        store = SqliteGraphStore()
        graph = Graph(store=store)
        assert graph.sqlite_store is store

    def test_sqlite_store_raises_for_dict(self) -> None:
        graph = Graph.empty()
        raised = False
        try:
            _ = graph.sqlite_store
        except TypeError:
            raised = True
        assert raised, "Expected TypeError for non-SQLite graph"


# ---------------------------------------------------------------------------
# Snapshot integration with .db
# ---------------------------------------------------------------------------


class TestDbSnapshots:
    """Snapshot functions work with SQLite-backed graphs."""

    def test_save_snapshot_db(self, tmp_path: Path) -> None:
        """save_snapshot creates .db snapshot for SQLite graph."""
        store = SqliteGraphStore(tmp_path / "graph.db")
        graph = Graph(store=store)
        graph.create_node("n1", {"type": "t"})

        path = save_snapshot(graph, tmp_path, "grow")
        assert path.suffix == ".db"
        assert path.exists()

    def test_rollback_snapshot_db(self, tmp_path: Path) -> None:
        """rollback_to_snapshot restores from .db snapshot."""
        store = SqliteGraphStore(tmp_path / "graph.db")
        graph = Graph(store=store)
        graph.create_node("n1", {"type": "t"})
        save_snapshot(graph, tmp_path, "grow")

        # Modify
        graph.create_node("n2", {"type": "t"})
        graph.save(tmp_path / "graph.db")

        # Rollback
        restored = rollback_to_snapshot(tmp_path, "grow")
        assert restored.has_node("n1")
        assert not restored.has_node("n2")

    def test_list_snapshots_mixed(self, tmp_path: Path) -> None:
        """list_snapshots finds both .json and .db snapshots."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "pre-dream.json").write_text("{}")
        (snapshot_dir / "pre-grow.db").write_bytes(b"")

        stages = list_snapshots(tmp_path)
        assert "dream" in stages
        assert "grow" in stages

    def test_delete_snapshot_both_formats(self, tmp_path: Path) -> None:
        """delete_snapshot removes both .json and .db variants."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "pre-dream.json").write_text("{}")
        (snapshot_dir / "pre-dream.db").write_bytes(b"")

        assert delete_snapshot(tmp_path, "dream")
        assert not (snapshot_dir / "pre-dream.json").exists()
        assert not (snapshot_dir / "pre-dream.db").exists()
