"""Tests for Graph ↔ SQLite integration.

Covers: .db load/save, savepoint API through Graph,
mutation_context, snapshot handling.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from questfoundry.graph.graph import Graph
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


class TestGraphLoadDetection:
    """Graph.load() detects .db files."""

    def test_load_db_is_sqlite_backed(self, tmp_path: Path) -> None:
        """Loading .db results in SQLite-backed graph."""
        g = Graph.empty()
        g.create_node("n1", {"type": "t"})
        g.save(tmp_path / "graph.db")

        loaded = Graph.load(tmp_path)
        assert loaded.is_sqlite_backed
        assert loaded.has_node("n1")

    def test_load_empty_when_no_db(self, tmp_path: Path) -> None:
        """Returns empty graph when no .db exists."""
        graph = Graph.load(tmp_path)
        assert not graph.is_sqlite_backed
        assert isinstance(graph._store, DictGraphStore)


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

    def test_list_snapshots(self, tmp_path: Path) -> None:
        """list_snapshots finds .db snapshots."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "pre-dream.db").write_bytes(b"")
        (snapshot_dir / "pre-grow.db").write_bytes(b"")

        stages = list_snapshots(tmp_path)
        assert "dream" in stages
        assert "grow" in stages

    def test_delete_snapshot(self, tmp_path: Path) -> None:
        """delete_snapshot removes .db snapshot."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "pre-dream.db").write_bytes(b"")

        assert delete_snapshot(tmp_path, "dream")
        assert not (snapshot_dir / "pre-dream.db").exists()


# ---------------------------------------------------------------------------
# Mutation before_state recording
# ---------------------------------------------------------------------------


class TestBeforeStateMutations:
    """Verify that mutating operations capture before_state for reversibility."""

    def _get_mutations(self, store: SqliteGraphStore) -> list[dict]:
        rows = store._conn.execute(
            "SELECT operation, target_id, delta, before_state FROM mutations ORDER BY id"
        ).fetchall()
        return [
            {
                "operation": r["operation"],
                "target_id": r["target_id"],
                "delta": json.loads(r["delta"]) if r["delta"] else None,
                "before_state": (json.loads(r["before_state"]) if r["before_state"] else None),
            }
            for r in rows
        ]

    def test_create_node_has_no_before_state(self) -> None:
        """create_node records delta but no before_state."""
        store = SqliteGraphStore()
        store.set_node("n1", {"type": "t", "name": "Alice"})

        muts = self._get_mutations(store)
        assert len(muts) == 1
        assert muts[0]["operation"] == "create_node"
        assert muts[0]["delta"] == {"type": "t", "name": "Alice"}
        assert muts[0]["before_state"] is None

    def test_set_node_update_captures_before(self) -> None:
        """set_node on existing node captures old data in before_state."""
        store = SqliteGraphStore()
        store.set_node("n1", {"type": "t", "name": "Alice"})
        store.set_node("n1", {"type": "t", "name": "Bob"})

        muts = self._get_mutations(store)
        assert len(muts) == 2
        assert muts[1]["operation"] == "replace_node"
        assert muts[1]["delta"] == {"type": "t", "name": "Bob"}
        assert muts[1]["before_state"] == {"type": "t", "name": "Alice"}

    def test_update_node_fields_captures_old_values(self) -> None:
        """update_node_fields records old values of changed fields."""
        store = SqliteGraphStore()
        store.set_node("n1", {"type": "t", "name": "Alice", "age": 30})
        store.update_node_fields("n1", name="Bob", age=31)

        muts = self._get_mutations(store)
        update_mut = muts[-1]
        assert update_mut["operation"] == "update_node"
        assert update_mut["delta"] == {"name": "Bob", "age": 31}
        assert update_mut["before_state"] == {"name": "Alice", "age": 30}

    def test_update_node_fields_captures_none_for_new_fields(self) -> None:
        """update_node_fields records None for fields that didn't exist before."""
        store = SqliteGraphStore()
        store.set_node("n1", {"type": "t", "name": "Alice"})
        store.update_node_fields("n1", mood="happy")

        muts = self._get_mutations(store)
        update_mut = muts[-1]
        assert update_mut["before_state"] == {"mood": None}

    def test_delete_node_captures_before_state(self) -> None:
        """delete_node records full node data in before_state."""
        store = SqliteGraphStore()
        store.set_node("n1", {"type": "t", "name": "Alice", "age": 30})
        store.delete_node("n1")

        muts = self._get_mutations(store)
        delete_mut = muts[-1]
        assert delete_mut["operation"] == "delete_node"
        assert delete_mut["before_state"] == {"type": "t", "name": "Alice", "age": 30}

    def test_add_edge_stores_full_delta(self) -> None:
        """add_edge records full edge dict in delta."""
        store = SqliteGraphStore()
        store.add_edge({"type": "knows", "from": "a", "to": "b", "weight": 5})

        muts = self._get_mutations(store)
        assert len(muts) == 1
        assert muts[0]["operation"] == "add_edge"
        assert muts[0]["delta"] == {"type": "knows", "from": "a", "to": "b", "weight": 5}

    def test_remove_edge_captures_before_state(self) -> None:
        """remove_edge records full edge dict in before_state."""
        store = SqliteGraphStore()
        store.add_edge({"type": "knows", "from": "a", "to": "b", "weight": 5})
        store.remove_edge("knows", "a", "b")

        muts = self._get_mutations(store)
        remove_mut = muts[-1]
        assert remove_mut["operation"] == "remove_edge"
        assert remove_mut["before_state"]["type"] == "knows"
        assert remove_mut["before_state"]["from"] == "a"
        assert remove_mut["before_state"]["to"] == "b"
        assert remove_mut["before_state"]["weight"] == 5

    def test_remove_edges_referencing_captures_all(self) -> None:
        """remove_edges_referencing captures before_state for each removed edge."""
        store = SqliteGraphStore()
        store.add_edge({"type": "knows", "from": "a", "to": "b"})
        store.add_edge({"type": "likes", "from": "c", "to": "a"})
        store.add_edge({"type": "hates", "from": "d", "to": "e"})
        store.remove_edges_referencing("a")

        muts = self._get_mutations(store)
        # 3 add_edge + 2 remove_edge (only edges referencing "a")
        remove_muts = [m for m in muts if m["operation"] == "remove_edge"]
        assert len(remove_muts) == 2
        edge_types = {m["before_state"]["type"] for m in remove_muts}
        assert edge_types == {"knows", "likes"}
        # All have before_state
        assert all(m["before_state"] is not None for m in remove_muts)

    def test_schema_migration_adds_column(self, tmp_path: Path) -> None:
        """Opening a DB without before_state column adds it automatically."""
        import sqlite3

        db_path = tmp_path / "test.db"
        # Create a DB with old schema (no before_state column)
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """\
            CREATE TABLE nodes (
                node_id TEXT PRIMARY KEY, type TEXT, data JSON,
                created_stage TEXT DEFAULT '', created_phase TEXT DEFAULT '',
                modified_stage TEXT DEFAULT '', modified_phase TEXT DEFAULT ''
            );
            CREATE TABLE edges (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                edge_type TEXT, from_id TEXT, to_id TEXT, data JSON,
                created_stage TEXT DEFAULT '', created_phase TEXT DEFAULT ''
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value JSON);
            CREATE TABLE mutations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT '', stage TEXT DEFAULT '',
                phase TEXT DEFAULT '', operation TEXT,
                target_id TEXT, delta JSON, rationale TEXT DEFAULT ''
            );
            CREATE TABLE phase_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT, phase TEXT, started_at TEXT,
                completed_at TEXT, status TEXT, mutation_count INTEGER,
                detail TEXT
            );
            """
        )
        # Insert an old mutation without before_state
        conn.execute(
            "INSERT INTO mutations (operation, target_id) VALUES (?, ?)",
            ("create_node", "old_node"),
        )
        conn.commit()
        conn.close()

        # Reopen with SqliteGraphStore — should migrate
        store = SqliteGraphStore(db_path)
        # Verify column exists by inserting with before_state
        store.set_node("n1", {"type": "t", "name": "X"})
        store.delete_node("n1")

        # Old mutation should have NULL before_state
        row = store._conn.execute(
            "SELECT before_state FROM mutations WHERE target_id = 'old_node'"
        ).fetchone()
        assert row["before_state"] is None

        # New delete should have before_state
        row = store._conn.execute(
            "SELECT before_state FROM mutations WHERE operation = 'delete_node'"
        ).fetchone()
        assert row["before_state"] is not None
        store.close()


# ---------------------------------------------------------------------------
# Rewind API
# ---------------------------------------------------------------------------


class TestRewind:
    """Verify rewind_to_phase and rewind_stage reverse mutations correctly."""

    def test_rewind_create_nodes(self) -> None:
        """Rewinding create_node mutations deletes the created nodes."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("grow", "spine"):
            graph.create_node("n1", {"type": "t", "name": "A"})
            graph.create_node("n2", {"type": "t", "name": "B"})
            graph.create_node("n3", {"type": "t", "name": "C"})

        assert graph.has_node("n1")
        assert graph.has_node("n2")
        assert graph.has_node("n3")

        count = graph.rewind_to_phase("grow", "spine")
        assert count == 3
        assert not graph.has_node("n1")
        assert not graph.has_node("n2")
        assert not graph.has_node("n3")

    def test_rewind_update_restores_old_values(self) -> None:
        """Rewinding update_node restores previous field values."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("n1", {"type": "t", "name": "Alice", "age": 30})

        with graph.mutation_context("grow", "spine"):
            graph.update_node("n1", name="Bob", age=31)

        assert graph.get_node("n1")["name"] == "Bob"

        count = graph.rewind_to_phase("grow", "spine")
        assert count == 1
        node = graph.get_node("n1")
        assert node["name"] == "Alice"
        assert node["age"] == 30

    def test_rewind_replace_node_restores_full_data(self) -> None:
        """Rewinding replace_node (set_node update) restores full old data."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("n1", {"type": "t", "name": "Alice", "extra": "x"})

        with graph.mutation_context("grow", "spine"):
            graph.upsert_node("n1", {"type": "t", "name": "Bob"})

        node = graph.get_node("n1")
        assert node["name"] == "Bob"
        assert "extra" not in node

        count = graph.rewind_to_phase("grow", "spine")
        assert count == 1
        node = graph.get_node("n1")
        assert node["name"] == "Alice"
        assert node["extra"] == "x"

    def test_rewind_delete_recreates_node(self) -> None:
        """Rewinding delete_node recreates the node with original data."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("n1", {"type": "t", "name": "Alice"})

        with graph.mutation_context("grow", "collapse"):
            graph.delete_node("n1")

        assert not graph.has_node("n1")

        count = graph.rewind_to_phase("grow", "collapse")
        assert count == 1
        node = graph.get_node("n1")
        assert node is not None
        assert node["name"] == "Alice"

    def test_rewind_add_edge_removes(self) -> None:
        """Rewinding add_edge removes the created edge."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("a", {"type": "t"})
            graph.create_node("b", {"type": "t"})

        with graph.mutation_context("grow", "spine"):
            graph.add_edge("knows", "a", "b", weight=5)

        assert len(graph.get_edges(edge_type="knows")) == 1

        count = graph.rewind_to_phase("grow", "spine")
        assert count == 1
        assert len(graph.get_edges(edge_type="knows")) == 0

    def test_rewind_remove_edge_recreates(self) -> None:
        """Rewinding remove_edge recreates the edge with original data."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("a", {"type": "t"})
            graph.create_node("b", {"type": "t"})
            graph.add_edge("knows", "a", "b", weight=5)

        with graph.mutation_context("grow", "collapse"):
            graph.remove_edge("knows", "a", "b")

        assert len(graph.get_edges(edge_type="knows")) == 0

        count = graph.rewind_to_phase("grow", "collapse")
        assert count == 1
        edges = graph.get_edges(edge_type="knows")
        assert len(edges) == 1
        assert edges[0]["weight"] == 5

    def test_rewind_mixed_operations(self) -> None:
        """Rewind correctly reverses a mix of create, update, edge ops."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        # Phase 1: create base data
        with graph.mutation_context("seed", "core"):
            graph.create_node("n1", {"type": "t", "val": 1})
            graph.create_node("n2", {"type": "t", "val": 2})

        # Phase 2: mixed operations
        with graph.mutation_context("grow", "spine"):
            graph.create_node("n3", {"type": "t", "val": 3})
            graph.update_node("n1", val=10)
            graph.add_edge("link", "n1", "n2")
            graph.add_edge("link", "n2", "n3")

        assert graph.has_node("n3")
        assert graph.get_node("n1")["val"] == 10
        assert len(graph.get_edges(edge_type="link")) == 2

        count = graph.rewind_to_phase("grow", "spine")
        assert count == 4
        assert not graph.has_node("n3")
        assert graph.get_node("n1")["val"] == 1
        assert graph.has_node("n2")
        assert len(graph.get_edges(edge_type="link")) == 0

    def test_rewind_preserves_earlier_phases(self) -> None:
        """Rewind only affects mutations from the target phase onward."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("seed_node", {"type": "t", "from": "seed"})

        with graph.mutation_context("grow", "phase1"):
            graph.create_node("p1_node", {"type": "t", "from": "p1"})

        with graph.mutation_context("grow", "phase2"):
            graph.create_node("p2_node", {"type": "t", "from": "p2"})

        # Rewind from phase2 — phase1 and seed should be preserved
        count = graph.rewind_to_phase("grow", "phase2")
        assert count == 1
        assert graph.has_node("seed_node")
        assert graph.has_node("p1_node")
        assert not graph.has_node("p2_node")

    def test_rewind_from_middle_phase_reverses_later(self) -> None:
        """Rewinding from an early phase also reverses later phases."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("grow", "phase1"):
            graph.create_node("p1", {"type": "t"})

        with graph.mutation_context("grow", "phase2"):
            graph.create_node("p2", {"type": "t"})

        with graph.mutation_context("grow", "phase3"):
            graph.create_node("p3", {"type": "t"})

        # Rewind from phase1 — reverses all three phases
        count = graph.rewind_to_phase("grow", "phase1")
        assert count == 3
        assert not graph.has_node("p1")
        assert not graph.has_node("p2")
        assert not graph.has_node("p3")

    def test_rewind_nonexistent_phase_is_noop(self) -> None:
        """Rewind returns 0 for a phase with no mutations."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("grow", "spine"):
            graph.create_node("n1", {"type": "t"})

        count = graph.rewind_to_phase("grow", "nonexistent")
        assert count == 0
        # Original node should still exist
        assert graph.has_node("n1")

    def test_rewind_stage_reverses_all(self) -> None:
        """rewind_stage reverses all phases within the stage."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("seed_node", {"type": "t"})

        with graph.mutation_context("grow", "phase1"):
            graph.create_node("g1", {"type": "t"})

        with graph.mutation_context("grow", "phase2"):
            graph.create_node("g2", {"type": "t"})
            graph.update_node("g1", val=99)

        count = graph.rewind_stage("grow")
        assert count == 3  # g1 create + g2 create + g1 update
        assert graph.has_node("seed_node")
        assert not graph.has_node("g1")
        assert not graph.has_node("g2")

    def test_rewind_destructive_phase(self) -> None:
        """Rewind restores nodes deleted during a collapse-like phase."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("keep", {"type": "t", "name": "keeper"})
            graph.create_node("remove_me", {"type": "t", "name": "doomed"})
            graph.add_edge("link", "keep", "remove_me")

        with graph.mutation_context("grow", "collapse"):
            graph.remove_edge("link", "keep", "remove_me")
            graph.delete_node("remove_me")
            graph.update_node("keep", status="alone")

        assert not graph.has_node("remove_me")
        assert graph.get_node("keep")["status"] == "alone"

        count = graph.rewind_to_phase("grow", "collapse")
        assert count == 3
        assert graph.has_node("remove_me")
        assert graph.get_node("remove_me")["name"] == "doomed"
        assert graph.get_node("keep").get("status") is None
        edges = graph.get_edges(edge_type="link")
        assert len(edges) == 1

    def test_rewind_not_sqlite_raises(self) -> None:
        """Rewind on a dict-backed graph raises TypeError."""
        graph = Graph.empty()
        graph.create_node("n1", {"type": "t"})

        raised = False
        try:
            graph.rewind_to_phase("grow", "spine")
        except TypeError:
            raised = True
        assert raised, "Expected TypeError for non-SQLite graph"

    def test_rewind_atomicity_on_failure(self) -> None:
        """If rewind fails partway, graph state is unchanged."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("grow", "phase1"):
            graph.create_node("n1", {"type": "t"})

        # Manually insert a mutation with missing before_state that needs it
        store._conn.execute(
            "INSERT INTO mutations (stage, phase, operation, target_id) VALUES (?, ?, ?, ?)",
            ("grow", "phase1", "update_node", "n1"),
        )

        raised = False
        try:
            graph.rewind_to_phase("grow", "phase1")
        except RuntimeError:
            raised = True
        assert raised, "Expected RuntimeError for missing before_state"

        # Graph state should be unchanged due to SAVEPOINT rollback
        assert graph.has_node("n1")

    def test_mutations_deleted_after_rewind(self) -> None:
        """Rewind deletes the reversed mutation records."""
        store = SqliteGraphStore()
        graph = Graph(store=store)

        with graph.mutation_context("seed", "core"):
            graph.create_node("n1", {"type": "t"})

        with graph.mutation_context("grow", "spine"):
            graph.create_node("n2", {"type": "t"})
            graph.create_node("n3", {"type": "t"})

        # Verify 3 mutations exist
        count_before = store._conn.execute("SELECT COUNT(*) AS cnt FROM mutations").fetchone()[
            "cnt"
        ]
        assert count_before == 3

        graph.rewind_to_phase("grow", "spine")

        # Only seed mutation should remain
        count_after = store._conn.execute("SELECT COUNT(*) AS cnt FROM mutations").fetchone()["cnt"]
        assert count_after == 1

        remaining = store._conn.execute("SELECT stage, phase FROM mutations").fetchone()
        assert remaining["stage"] == "seed"
        assert remaining["phase"] == "core"
