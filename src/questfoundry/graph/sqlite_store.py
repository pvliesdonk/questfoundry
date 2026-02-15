"""SQLite-backed graph storage with mutation audit trail.

SqliteGraphStore implements the GraphStore protocol using stdlib sqlite3.
Every mutating operation (create/update/delete node or edge) is recorded
in the ``mutations`` table with stage/phase context for auditing.

Savepoint support enables efficient rollback during GROW phases without
JSON serialization.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, cast

from questfoundry.graph.errors import NodeNotFoundError

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
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
    stage        TEXT NOT NULL DEFAULT '',
    phase        TEXT NOT NULL DEFAULT '',
    operation    TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    delta        JSON,
    before_state JSON,
    rationale    TEXT DEFAULT ''
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
            self._db_path: str = ":memory:"
        else:
            self._db_path = str(db_path) if isinstance(db_path, Path) else db_path
            self._conn = sqlite3.connect(
                self._db_path,
                isolation_level=None,  # autocommit — we manage transactions
            )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._ensure_before_state_column()

        self._stage: str = ""
        self._phase: str = ""

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def backup_to(self, dest_path: Path) -> None:
        """Copy the live database to a destination file.

        Uses SQLite's online backup API for a consistent copy,
        even while the source database is in use.

        Args:
            dest_path: Path for the backup file.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(str(dest_path))
        try:
            self._conn.backup(dest)
        except Exception:
            dest.close()
            # Remove partial file on failure
            if dest_path.exists():
                dest_path.unlink()
            raise
        else:
            dest.close()

    # -- Schema migration ------------------------------------------------------

    def _ensure_before_state_column(self) -> None:
        """Add ``before_state`` column if absent (schema migration).

        Existing databases created before rewind support lack the column.
        ``ALTER TABLE ADD COLUMN`` is safe for nullable columns in SQLite.
        """
        cursor = self._conn.execute("PRAGMA table_info(mutations)")
        columns = {row[1] for row in cursor.fetchall()}
        if "before_state" not in columns:
            self._conn.execute("ALTER TABLE mutations ADD COLUMN before_state JSON")

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
        before_state: dict[str, Any] | None = None,
    ) -> None:
        """Append a mutation record.

        Args:
            operation: Mutation type (create_node, update_node, etc.).
            target_id: Affected node ID or edge reference.
            delta: New data (node data for creates/updates, edge dict for add_edge).
            before_state: State before mutation for reversibility. For create_node,
                this is None. For update_node via update_node_fields, contains old
                values of changed fields (None if field was newly added). For
                delete_node and remove_edge, contains full node/edge data.
                Required for update_node, delete_node, and remove_edge to enable rewind.
        """
        self._conn.execute(
            "INSERT INTO mutations "
            "(stage, phase, operation, target_id, delta, before_state) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._stage,
                self._phase,
                operation,
                target_id,
                json.dumps(delta) if delta is not None else None,
                json.dumps(before_state) if before_state is not None else None,
            ),
        )

    # -- Savepoints ------------------------------------------------------------

    _SAVEPOINT_RE = re.compile(r"^[A-Za-z0-9_]+$")

    def _validate_savepoint_name(self, name: str) -> None:
        """Validate savepoint name to prevent SQL injection."""
        if not self._SAVEPOINT_RE.match(name):
            msg = f"Invalid savepoint name {name!r}: must be alphanumeric/underscores only"
            raise ValueError(msg)

    def savepoint(self, name: str) -> None:
        """Create a SQLite savepoint.

        Args:
            name: Savepoint name (alphanumeric + underscores only).

        Raises:
            ValueError: If *name* contains invalid characters.
        """
        self._validate_savepoint_name(name)
        self._conn.execute(f"SAVEPOINT sp_{name}")

    def rollback_to(self, name: str) -> None:
        """Rollback to a named savepoint.

        The savepoint remains active after rollback (SQLite behavior).

        Args:
            name: Savepoint name previously created with :meth:`savepoint`.

        Raises:
            ValueError: If *name* contains invalid characters.
        """
        self._validate_savepoint_name(name)
        self._conn.execute(f"ROLLBACK TO sp_{name}")

    def release(self, name: str) -> None:
        """Release (commit) a named savepoint.

        Args:
            name: Savepoint name to release.

        Raises:
            ValueError: If *name* contains invalid characters.
        """
        self._validate_savepoint_name(name)
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

    def _node_meta(self, node_id: str) -> dict[str, str | None]:
        """Fetch audit metadata columns for a node."""
        row = self._conn.execute(
            "SELECT created_stage, created_phase, modified_stage, modified_phase "
            "FROM nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return {}
        return {
            "created_stage": row["created_stage"],
            "created_phase": row["created_phase"],
            "modified_stage": row["modified_stage"],
            "modified_phase": row["modified_phase"],
        }

    def set_node(self, node_id: str, data: dict[str, Any]) -> None:
        node_type = data.get("type", "")
        row = self._conn.execute("SELECT data FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        existing = row is not None
        before: dict[str, Any] | None = None
        if existing and row is not None:
            old_data: dict[str, Any] = json.loads(row["data"])
            old_data["_meta"] = self._node_meta(node_id)
            before = old_data
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
        op = "replace_node" if existing else "create_node"
        self._record_mutation(op, node_id, delta=data, before_state=before)

    def update_node_fields(self, node_id: str, **updates: Any) -> None:
        row = self._conn.execute("SELECT data FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            raise NodeNotFoundError(node_id=node_id)
        current = json.loads(row["data"])
        old_values: dict[str, Any] = {k: current.get(k) for k in updates}
        old_values["_meta"] = self._node_meta(node_id)
        current.update(updates)
        node_type = current.get("type", "")
        self._conn.execute(
            "UPDATE nodes SET data = ?, type = ?, modified_stage = ?, modified_phase = ? "
            "WHERE node_id = ?",
            (json.dumps(current), node_type, self._stage, self._phase, node_id),
        )
        self._record_mutation(
            "update_node",
            node_id,
            delta=dict(updates),
            before_state=old_values,
        )

    def delete_node(self, node_id: str) -> None:
        row = self._conn.execute("SELECT data FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            raise NodeNotFoundError(node_id=node_id)
        before = json.loads(row["data"])
        before["_meta"] = self._node_meta(node_id)
        self._conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
        self._record_mutation("delete_node", node_id, before_state=before)

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
        self._record_mutation("add_edge", target_id, delta=dict(edge))

    def remove_edge(self, edge_type: str, from_id: str, to_id: str) -> bool:
        # Find the first matching row to delete (fetch full data for before_state)
        row = self._conn.execute(
            "SELECT rowid, edge_type, from_id, to_id, data FROM edges "
            "WHERE edge_type = ? AND from_id = ? AND to_id = ? LIMIT 1",
            (edge_type, from_id, to_id),
        ).fetchone()
        if row is None:
            return False
        before = self._row_to_edge(row)
        self._conn.execute("DELETE FROM edges WHERE rowid = ?", (row["rowid"],))
        target_id = f"edge:{edge_type}:{from_id}:{to_id}"
        self._record_mutation("remove_edge", target_id, before_state=before)
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
        # Atomic: record mutations and delete in one transaction
        self._conn.execute("SAVEPOINT sp_remove_refs")
        try:
            rows = self._conn.execute(
                "SELECT edge_type, from_id, to_id, data FROM edges WHERE from_id = ? OR to_id = ?",
                (node_id, node_id),
            ).fetchall()
            for row in rows:
                target_id = f"edge:{row['edge_type']}:{row['from_id']}:{row['to_id']}"
                before = self._row_to_edge(row)
                self._record_mutation("remove_edge", target_id, before_state=before)
            self._conn.execute(
                "DELETE FROM edges WHERE from_id = ? OR to_id = ?",
                (node_id, node_id),
            )
            self._conn.execute("RELEASE SAVEPOINT sp_remove_refs")
        except Exception:
            self._conn.execute("ROLLBACK TO sp_remove_refs")
            self._conn.execute("RELEASE SAVEPOINT sp_remove_refs")
            raise

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

    # -- Rewind ----------------------------------------------------------------

    def rewind_to_phase(self, stage: str, phase: str) -> int:
        """Rewind graph by reversing all mutations from a phase onward.

        Finds the first mutation matching *(stage, phase)* and applies the
        inverse of every mutation from that point forward, in reverse order.
        The reversed mutation records are deleted.

        Args:
            stage: Pipeline stage (e.g., ``"grow"``).
            phase: Phase within the stage (e.g., ``"path_agnostic"``).

        Returns:
            Number of mutations reversed.

        Raises:
            ValueError: If no mutations found for *(stage, phase)*.
            RuntimeError: If a mutation lacks ``before_state`` needed for
                reversal (e.g., pre-migration data).
        """
        row = self._conn.execute(
            "SELECT MIN(id) AS min_id FROM mutations WHERE stage = ? AND phase = ?",
            (stage, phase),
        ).fetchone()

        if row is None or row["min_id"] is None:
            raise ValueError(f"No mutations found for stage={stage!r}, phase={phase!r}")

        return self._rewind_from_id(row["min_id"])

    def rewind_stage(self, stage: str) -> int:
        """Rewind all mutations for an entire stage.

        Finds the first mutation for *stage* (any phase) and reverses
        everything from that point forward.

        Args:
            stage: Pipeline stage to rewind (e.g., ``"grow"``).

        Returns:
            Number of mutations reversed.

        Raises:
            ValueError: If no mutations found for the stage.
            RuntimeError: If a mutation lacks ``before_state`` needed for
                reversal.
        """
        row = self._conn.execute(
            "SELECT MIN(id) AS min_id FROM mutations WHERE stage = ?",
            (stage,),
        ).fetchone()

        if row is None or row["min_id"] is None:
            raise ValueError(f"No mutations found for stage={stage!r}")

        return self._rewind_from_id(row["min_id"])

    def _rewind_from_id(self, first_id: int) -> int:
        """Reverse all mutations with ``id >= first_id``.

        Processes mutations in reverse chronological order, applying the
        inverse of each. The entire rewind is wrapped in a SAVEPOINT for
        atomicity — on any failure, all changes are rolled back.

        Args:
            first_id: Mutation ID to start rewinding from (inclusive).

        Returns:
            Number of mutations reversed.
        """
        mutations = self._conn.execute(
            "SELECT id, operation, target_id, delta, before_state "
            "FROM mutations WHERE id >= ? ORDER BY id DESC",
            (first_id,),
        ).fetchall()

        if not mutations:
            return 0

        self._conn.execute("SAVEPOINT sp_rewind")
        try:
            for mutation in mutations:
                self._apply_inverse(mutation)

            self._conn.execute("DELETE FROM mutations WHERE id >= ?", (first_id,))
            self._conn.execute("RELEASE SAVEPOINT sp_rewind")
        except Exception:
            self._conn.execute("ROLLBACK TO sp_rewind")
            self._conn.execute("RELEASE SAVEPOINT sp_rewind")
            raise

        return len(mutations)

    def _apply_inverse(self, mutation: sqlite3.Row) -> None:
        """Apply the inverse of a single mutation.

        Args:
            mutation: Row from the ``mutations`` table with columns:
                ``operation``, ``target_id``, ``delta``, ``before_state``.

        Raises:
            RuntimeError: If ``before_state`` is required but missing, or
                if the operation is unknown.
        """
        operation = mutation["operation"]
        target_id = mutation["target_id"]
        delta = json.loads(mutation["delta"]) if mutation["delta"] else None
        before = json.loads(mutation["before_state"]) if mutation["before_state"] else None

        if operation == "create_node":
            self._conn.execute("DELETE FROM nodes WHERE node_id = ?", (target_id,))

        elif operation == "replace_node":
            if before is None:
                raise RuntimeError(
                    f"Cannot reverse replace_node for {target_id!r}: "
                    f"before_state is missing (pre-migration mutation?)"
                )
            meta = before.pop("_meta", {})
            self._conn.execute(
                "INSERT OR REPLACE INTO nodes "
                "(node_id, type, data, created_stage, created_phase, modified_stage, modified_phase) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    target_id,
                    before.get("type", ""),
                    json.dumps(before),
                    meta.get("created_stage"),
                    meta.get("created_phase"),
                    meta.get("modified_stage"),
                    meta.get("modified_phase"),
                ),
            )

        elif operation == "update_node":
            if before is None:
                raise RuntimeError(
                    f"Cannot reverse update_node for {target_id!r}: "
                    f"before_state is missing (pre-migration mutation?)"
                )
            meta = before.pop("_meta", {})
            row = self._conn.execute(
                "SELECT data FROM nodes WHERE node_id = ?", (target_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"Cannot reverse update_node: node {target_id!r} not found in database"
                )
            current = json.loads(row["data"])
            for k, v in before.items():
                if v is None:
                    current.pop(k, None)
                else:
                    current[k] = v
            update_sql = "UPDATE nodes SET data = ?, type = ?"
            params: list[Any] = [json.dumps(current), current.get("type", "")]
            if meta:
                update_sql += ", modified_stage = ?, modified_phase = ?"
                params.extend([meta.get("modified_stage"), meta.get("modified_phase")])
            update_sql += " WHERE node_id = ?"
            params.append(target_id)
            self._conn.execute(update_sql, params)

        elif operation == "delete_node":
            if before is None:
                raise RuntimeError(
                    f"Cannot reverse delete_node for {target_id!r}: "
                    f"before_state is missing (pre-migration mutation?)"
                )
            meta = before.pop("_meta", {})
            self._conn.execute(
                "INSERT INTO nodes "
                "(node_id, type, data, created_stage, created_phase, modified_stage, modified_phase) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    target_id,
                    before.get("type", ""),
                    json.dumps(before),
                    meta.get("created_stage"),
                    meta.get("created_phase"),
                    meta.get("modified_stage"),
                    meta.get("modified_phase"),
                ),
            )

        elif operation == "add_edge":
            if delta is None:
                raise RuntimeError(f"Cannot reverse add_edge for {target_id!r}: delta is missing")
            edge_type = delta.get("type", "")
            from_id = delta.get("from", "")
            to_id = delta.get("to", "")
            row = self._conn.execute(
                "SELECT rowid FROM edges WHERE edge_type = ? AND from_id = ? AND to_id = ? LIMIT 1",
                (edge_type, from_id, to_id),
            ).fetchone()
            if row is not None:
                self._conn.execute("DELETE FROM edges WHERE rowid = ?", (row["rowid"],))

        elif operation == "remove_edge":
            if before is None:
                raise RuntimeError(
                    f"Cannot reverse remove_edge for {target_id!r}: "
                    f"before_state is missing (pre-migration mutation?)"
                )
            edge_type = before.get("type", "")
            from_id = before.get("from", "")
            to_id = before.get("to", "")
            extra = {k: v for k, v in before.items() if k not in ("type", "from", "to")}
            self._conn.execute(
                "INSERT INTO edges (edge_type, from_id, to_id, data) VALUES (?, ?, ?, ?)",
                (
                    edge_type,
                    from_id,
                    to_id,
                    json.dumps(extra) if extra else None,
                ),
            )

        else:
            raise RuntimeError(f"Unknown mutation operation: {operation!r}")

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

        # Bulk import in a single transaction for performance
        store._conn.execute("BEGIN")
        try:
            # Meta
            meta = data.get("meta", {})
            if meta:
                store._conn.executemany(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    [(key, json.dumps(value)) for key, value in meta.items()],
                )

            # Nodes (bulk insert, no mutation recording)
            nodes = data.get("nodes", {})
            if nodes:
                store._conn.executemany(
                    "INSERT INTO nodes (node_id, type, data) VALUES (?, ?, ?)",
                    [
                        (node_id, node_data.get("type", ""), json.dumps(node_data))
                        for node_id, node_data in nodes.items()
                    ],
                )

            # Edges (bulk insert, no mutation recording)
            edges = data.get("edges", [])
            if edges:
                store._conn.executemany(
                    "INSERT INTO edges (edge_type, from_id, to_id, data) VALUES (?, ?, ?, ?)",
                    [
                        (
                            edge.get("type", ""),
                            edge.get("from", ""),
                            edge.get("to", ""),
                            json.dumps(
                                {k: v for k, v in edge.items() if k not in ("type", "from", "to")}
                            )
                            or None,
                        )
                        for edge in edges
                    ],
                )

            store._conn.execute("COMMIT")
        except Exception:
            store._conn.execute("ROLLBACK")
            raise

        return store
