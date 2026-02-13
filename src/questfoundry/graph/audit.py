"""Mutation audit trail queries.

Provides functions to query and format the mutations table from a
SqliteGraphStore database for debugging and inspection.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def _open_db(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a read-only connection to the audit database.

    Raises:
        sqlite3.Error: If the database cannot be opened or is corrupted.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def query_mutations(
    db_path: Path,
    *,
    stage: str | None = None,
    phase: str | None = None,
    operation: str | None = None,
    target: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query mutations from a SQLite graph database.

    Args:
        db_path: Path to the ``.db`` file.
        stage: Filter by stage name.
        phase: Filter by phase name.
        operation: Filter by operation type (e.g., "create_node").
        target: Filter by target ID (substring match).
        limit: Maximum number of results.

    Returns:
        List of mutation dicts, most recent first.

    Raises:
        sqlite3.Error: If the database cannot be read.
    """
    with _open_db(db_path) as conn:
        clauses: list[str] = []
        params: list[Any] = []

        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        if phase is not None:
            clauses.append("phase = ?")
            params.append(phase)
        if operation is not None:
            clauses.append("operation = ?")
            params.append(operation)
        if target is not None:
            clauses.append("target_id LIKE ?")
            params.append(f"%{target}%")

        # WHERE clause is built from hardcoded column names; all values
        # are parameterized via ? — no injection risk.
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        rows = conn.execute(
            f"SELECT id, timestamp, stage, phase, operation, target_id, delta "
            f"FROM mutations{where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()

        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "stage": row["stage"],
                "phase": row["phase"],
                "operation": row["operation"],
                "target_id": row["target_id"],
                "delta": json.loads(row["delta"]) if row["delta"] else None,
            }
            for row in rows
        ]


def query_phase_history(db_path: Path) -> list[dict[str, Any]]:
    """Query phase history from a SQLite graph database.

    Args:
        db_path: Path to the ``.db`` file.

    Returns:
        List of phase history dicts, ordered by ID.

    Raises:
        sqlite3.Error: If the database cannot be read.
    """
    with _open_db(db_path) as conn:
        rows = conn.execute(
            "SELECT id, stage, phase, started_at, completed_at, status, "
            "mutation_count, detail FROM phase_history ORDER BY id"
        ).fetchall()

        return [
            {
                "id": row["id"],
                "stage": row["stage"],
                "phase": row["phase"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "status": row["status"],
                "mutation_count": row["mutation_count"],
                "detail": row["detail"],
            }
            for row in rows
        ]


def mutation_summary(db_path: Path) -> dict[str, Any]:
    """Get a summary of mutation counts by stage and operation.

    Args:
        db_path: Path to the ``.db`` file.

    Returns:
        Summary dict with total count, per-stage counts, and per-operation counts.

    Raises:
        sqlite3.Error: If the database cannot be read.
    """
    with _open_db(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM mutations").fetchone()["cnt"]

        stage_rows = conn.execute(
            "SELECT stage, COUNT(*) AS cnt FROM mutations GROUP BY stage ORDER BY cnt DESC"
        ).fetchall()

        op_rows = conn.execute(
            "SELECT operation, COUNT(*) AS cnt FROM mutations GROUP BY operation ORDER BY cnt DESC"
        ).fetchall()

        return {
            "total": total,
            "by_stage": {row["stage"] or "(none)": row["cnt"] for row in stage_rows},
            "by_operation": {row["operation"]: row["cnt"] for row in op_rows},
        }
