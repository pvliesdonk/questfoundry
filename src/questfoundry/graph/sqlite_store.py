"""SQLite-backed graph storage with mutation audit trail.

SqliteGraphStore implements the GraphStore protocol using stdlib sqlite3.
Every mutating operation (create/update/delete node or edge) is recorded
in the ``mutations`` table with stage/phase context for auditing.

Savepoint support enables efficient rollback during GROW phases without
JSON serialization.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS nodes (
    node_id        TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    data           JSON NOT NULL,
    created_stage  TEXT DEFAULT '',
    created_phase  TEXT DEFAULT '',
    modified_stage TEXT DEFAULT '',
    modified_phase TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);

CREATE TABLE IF NOT EXISTS edges (
    rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_type TEXT NOT NULL,
    from_id   TEXT NOT NULL,
    to_id     TEXT NOT NULL,
    data      JSON,
    created_stage TEXT DEFAULT '',
    created_phase TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS mutations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
    stage     TEXT NOT NULL DEFAULT '',
    phase     TEXT NOT NULL DEFAULT '',
    operation TEXT NOT NULL,
    target_id TEXT NOT NULL,
    delta     JSON,
    rationale TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_mutations_stage  ON mutations(stage);
CREATE INDEX IF NOT EXISTS idx_mutations_target ON mutations(target_id);

CREATE TABLE IF NOT EXISTS phase_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stage          TEXT NOT NULL,
    phase          TEXT NOT NULL,
    started_at     TEXT,
    completed_at   TEXT,
    status         TEXT DEFAULT 'started',
    mutation_count INTEGER DEFAULT 0,
    detail         TEXT DEFAULT ''
);
"""

VERSION = "5.0"


class SqliteGraphStore:
    """SQLite-backed graph store with mutation recording.

    Every mutating operation records a row in the ``mutations`` table.
    Use :meth:`set_mutation_context` to set the current stage/phase
    before running operations — this tags each mutation for auditing.

    Supports SQLite savepoints for efficient in-process rollback.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        _conn: sqlite3.Connection | None = None,
    ) -> None:
        """Open or create a SQLite graph database.

        Args:
            db_path: Path to ``.db`` file, or ``":memory:"`` for in-memory.
            _conn: Pre-existing connection (for testing). If provided,
                   *db_path* is ignored.
        """
        if _conn is not None:
            self._conn = _conn
        else:
            self._conn = sqlite3.connect(
                str(db_path) if isinstance(db_path, Path) else db_path,
                isolation_level=None,  # autocommit — we manage transactions
            )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

        self._stage: str = ""
        self._phase: str = ""

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # -- Mutation context ------------------------------------------------------

    def set_mutation_context(self, stage: str = "", phase: str = "") -> None:
        """Set the current stage/phase for mutation recording.

        Args:
            stage: Current pipeline stage (e.g., "grow").
            phase: Current phase within the stage (e.g., "path_agnostic").
        """
        self._stage = stage
        self._phase = phase

    def _record_mutation(
        self,
        operation: str,
        target_id: str,
        delta: dict[str, Any] | None = None,
    ) -> None:
        """Append a mutation record."""
        self._conn.execute(
            "INSERT INTO mutations (stage, phase, operation, target_id, delta) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                self._stage,
                self._phase,
                operation,
                target_id,
                json.dumps(delta) if delta is not None else None,
            ),
        )

    # -- Savepoints ------------------------------------------------------------

    def savepoint(self, name: str) -> None:
        """Create a SQLite savepoint.

        Args:
            name: Savepoint name (alphanumeric + underscores).
        """
        self._conn.execute(f"SAVEPOINT sp_{name}")

    def rollback_to(self, name: str) -> None:
        """Rollback to a named savepoint.

        The savepoint remains active after rollback (SQLite behavior).

        Args:
            name: Savepoint name previously created with :meth:`savepoint`.
        """
        self._conn.execute(f"ROLLBACK TO sp_{name}")

    def release(self, name: str) -> None:
        """Release (commit) a named savepoint.

        Args:
            name: Savepoint name to release.
        """
        self._conn.execute(f"RELEASE SAVEPOINT sp_{name}")

    # -- Nodes -----------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT data FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return cast("dict[str, Any]", json.loads(row["data"]))

    def has_node(self, node_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return row is not None

    def set_node(self, node_id: str, data: dict[str, Any]) -> None:
        node_type = data.get("type", "")
        existing = self.has_node(node_id)
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes "
            "(node_id, type, data, created_stage, created_phase, modified_stage, modified_phase) "
            "VALUES (?, ?, ?, "
            "  COALESCE((SELECT created_stage FROM nodes WHERE node_id = ?), ?), "
            "  COALESCE((SELECT created_phase FROM nodes WHERE node_id = ?), ?), "
            "  ?, ?)",
            (
                node_id,
                node_type,
                json.dumps(data),
                node_id,
                self._stage,
                node_id,
                self._phase,
                self._stage,
                self._phase,
            ),
        )
        op = "update_node" if existing else "create_node"
        self._record_mutation(op, node_id, delta=data)

    def update_node_fields(self, node_id: str, **updates: Any) -> None:
        row = self._conn.execute("SELECT data FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return  # Graph checks existence before calling
        current = json.loads(row["data"])
        current.update(updates)
        node_type = current.get("type", "")
        self._conn.execute(
            "UPDATE nodes SET data = ?, type = ?, modified_stage = ?, modified_phase = ? "
            "WHERE node_id = ?",
            (json.dumps(current), node_type, self._stage, self._phase, node_id),
        )
        self._record_mutation("update_node", node_id, delta=dict(updates))

    def delete_node(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
        self._record_mutation("delete_node", node_id)

    def get_nodes_by_type(self, node_type: str) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT node_id, data FROM nodes WHERE type = ?", (node_type,)
        ).fetchall()
        return {row["node_id"]: json.loads(row["data"]) for row in rows}

    def all_node_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT node_id FROM nodes").fetchall()
        return [row["node_id"] for row in rows]

    def node_ids_with_prefix(self, prefix: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT node_id FROM nodes WHERE node_id LIKE ?",
            (prefix + "%",),
        ).fetchall()
        return [row["node_id"] for row in rows]

    def node_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM nodes").fetchone()
        return row["cnt"]  # type: ignore[no-any-return]

    # -- Edges -----------------------------------------------------------------

    def add_edge(self, edge: dict[str, Any]) -> None:
        edge_type = edge.get("type", "")
        from_id = edge.get("from", "")
        to_id = edge.get("to", "")
        # Extra props beyond type/from/to
        extra = {k: v for k, v in edge.items() if k not in ("type", "from", "to")}
        self._conn.execute(
            "INSERT INTO edges (edge_type, from_id, to_id, data, created_stage, created_phase) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                edge_type,
                from_id,
                to_id,
                json.dumps(extra) if extra else None,
                self._stage,
                self._phase,
            ),
        )
        target_id = f"edge:{edge_type}:{from_id}:{to_id}"
        self._record_mutation("add_edge", target_id)

    def remove_edge(self, edge_type: str, from_id: str, to_id: str) -> bool:
        # Find the first matching row to delete
        row = self._conn.execute(
            "SELECT rowid FROM edges WHERE edge_type = ? AND from_id = ? AND to_id = ? LIMIT 1",
            (edge_type, from_id, to_id),
        ).fetchone()
        if row is None:
            return False
        self._conn.execute("DELETE FROM edges WHERE rowid = ?", (row["rowid"],))
        target_id = f"edge:{edge_type}:{from_id}:{to_id}"
        self._record_mutation("remove_edge", target_id)
        return True

    def get_edges(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[str] = []
        if from_id is not None:
            clauses.append("from_id = ?")
            params.append(from_id)
        if to_id is not None:
            clauses.append("to_id = ?")
            params.append(to_id)
        if edge_type is not None:
            clauses.append("edge_type = ?")
            params.append(edge_type)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT edge_type, from_id, to_id, data FROM edges{where} ORDER BY rowid",
            params,
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def edges_referencing(self, node_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT edge_type, from_id, to_id, data FROM edges "
            "WHERE from_id = ? OR to_id = ? ORDER BY rowid",
            (node_id, node_id),
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def remove_edges_referencing(self, node_id: str) -> None:
        # Record mutations before deleting
        rows = self._conn.execute(
            "SELECT edge_type, from_id, to_id FROM edges WHERE from_id = ? OR to_id = ?",
            (node_id, node_id),
        ).fetchall()
        for row in rows:
            target_id = f"edge:{row['edge_type']}:{row['from_id']}:{row['to_id']}"
            self._record_mutation("remove_edge", target_id)
        self._conn.execute(
            "DELETE FROM edges WHERE from_id = ? OR to_id = ?",
            (node_id, node_id),
        )

    def edge_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM edges").fetchone()
        return row["cnt"]  # type: ignore[no-any-return]

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> dict[str, Any]:
        """Convert an edge row to the dict format expected by Graph."""
        edge: dict[str, Any] = {
            "type": row["edge_type"],
            "from": row["from_id"],
            "to": row["to_id"],
        }
        if row["data"]:
            extra = json.loads(row["data"])
            edge.update(extra)
        return edge

    # -- Meta ------------------------------------------------------------------

    def get_meta(self, key: str) -> Any:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )

    def all_meta(self) -> dict[str, Any]:
        rows = self._conn.execute("SELECT key, value FROM meta").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Reconstruct the full graph dict from SQLite tables."""
        # Nodes
        nodes: dict[str, dict[str, Any]] = {}
        for row in self._conn.execute("SELECT node_id, data FROM nodes").fetchall():
            nodes[row["node_id"]] = json.loads(row["data"])

        # Edges (preserve insertion order via rowid)
        edges: list[dict[str, Any]] = []
        for row in self._conn.execute(
            "SELECT edge_type, from_id, to_id, data FROM edges ORDER BY rowid"
        ).fetchall():
            edges.append(self._row_to_edge(row))

        # Meta
        meta = self.all_meta()

        return {
            "version": VERSION,
            "meta": meta,
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        db_path: str | Path = ":memory:",
    ) -> SqliteGraphStore:
        """Bulk-import a graph dict into a new SqliteGraphStore.

        Args:
            data: Full graph data dict (version, meta, nodes, edges).
            db_path: Where to create the database.

        Returns:
            New SqliteGraphStore populated with the data.
        """
        store = cls(db_path)

        # Meta
        meta = data.get("meta", {})
        for key, value in meta.items():
            store._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

        # Nodes (bulk insert, no mutation recording)
        nodes = data.get("nodes", {})
        for node_id, node_data in nodes.items():
            node_type = node_data.get("type", "")
            store._conn.execute(
                "INSERT INTO nodes (node_id, type, data) VALUES (?, ?, ?)",
                (node_id, node_type, json.dumps(node_data)),
            )

        # Edges (bulk insert, no mutation recording)
        edges = data.get("edges", [])
        for edge in edges:
            edge_type = edge.get("type", "")
            from_id = edge.get("from", "")
            to_id = edge.get("to", "")
            extra = {k: v for k, v in edge.items() if k not in ("type", "from", "to")}
            store._conn.execute(
                "INSERT INTO edges (edge_type, from_id, to_id, data) VALUES (?, ?, ?, ?)",
                (edge_type, from_id, to_id, json.dumps(extra) if extra else None),
            )

        return store
