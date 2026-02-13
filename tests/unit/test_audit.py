"""Tests for graph mutation audit trail queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.audit import mutation_summary, query_mutations, query_phase_history
from questfoundry.graph.sqlite_store import SqliteGraphStore

if TYPE_CHECKING:
    from pathlib import Path


def _populate_store(tmp_path: Path) -> Path:
    """Create a graph.db with some mutations and return the path."""
    db_path = tmp_path / "graph.db"
    store = SqliteGraphStore(db_path)

    store.set_mutation_context("grow", "spine")
    store.set_node("beat::intro", {"type": "beat", "summary": "Introduction"})
    store.set_node("beat::climax", {"type": "beat", "summary": "Climax"})
    store.add_edge({"type": "sequence", "from": "beat::intro", "to": "beat::climax"})

    store.set_mutation_context("grow", "path_agnostic")
    store.set_node("beat::outro", {"type": "beat", "summary": "Outro"})

    store.set_mutation_context("fill", "prose")
    store.update_node_fields("beat::intro", prose="Once upon a time...")

    store.close()
    return db_path


class TestQueryMutations:
    """Test mutation query function."""

    def test_returns_all_mutations(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path)
        # 3 create_node + 1 add_edge + 1 update_node = 5
        assert len(results) == 5

    def test_filter_by_stage(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path, stage="grow")
        assert len(results) == 4
        assert all(r["stage"] == "grow" for r in results)

    def test_filter_by_phase(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path, phase="spine")
        assert len(results) == 3
        assert all(r["phase"] == "spine" for r in results)

    def test_filter_by_operation(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path, operation="create_node")
        assert len(results) == 3

    def test_filter_by_target(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path, target="beat::intro")
        # create + update + add_edge (target contains "beat::intro") = 3
        assert len(results) == 3

    def test_limit(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path, limit=2)
        assert len(results) == 2

    def test_most_recent_first(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path)
        ids = [r["id"] for r in results]
        assert ids == sorted(ids, reverse=True)

    def test_delta_parsed(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path, operation="update_node")
        assert len(results) == 1
        assert results[0]["delta"] == {"prose": "Once upon a time..."}

    def test_combined_filters(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        results = query_mutations(db_path, stage="grow", operation="create_node")
        assert len(results) == 3


class TestMutationSummary:
    """Test mutation summary function."""

    def test_total_count(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        result = mutation_summary(db_path)
        assert result["total"] == 5

    def test_by_stage(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        result = mutation_summary(db_path)
        assert result["by_stage"]["grow"] == 4
        assert result["by_stage"]["fill"] == 1

    def test_by_operation(self, tmp_path: Path) -> None:
        db_path = _populate_store(tmp_path)
        result = mutation_summary(db_path)
        assert result["by_operation"]["create_node"] == 3
        assert result["by_operation"]["add_edge"] == 1
        assert result["by_operation"]["update_node"] == 1


class TestQueryPhaseHistory:
    """Test phase history query."""

    def test_empty_history(self, tmp_path: Path) -> None:
        db_path = tmp_path / "graph.db"
        store = SqliteGraphStore(db_path)
        store.close()

        result = query_phase_history(db_path)
        assert result == []
