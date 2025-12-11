"""Checkpoint store for workflow state persistence and resume.

Provides per-delegation checkpointing to enable:
- Resume after crash/timeout
- Development iteration (restart from specific point)
- Debugging specific workflow stages

Storage: SQLite database at {project}/checkpoints.db
Delete the file to clear all checkpoints.

Usage:
    from questfoundry.runtime.checkpoint import CheckpointStore

    # Create store for a project
    store = CheckpointStore(project_path)

    # Start a new run
    run_id = store.start_run("create a mystery story", loop_id="story_spark")

    # Save checkpoint after each delegation
    store.save_checkpoint(
        run_id=run_id,
        sr_turn=2,
        role_id="gatekeeper",
        hot_store=state["hot_store"],
        sr_messages=sr_executor.messages,
        role_messages=role_agent.messages,
        delegation_history=history,
    )

    # Resume from latest checkpoint
    checkpoint = store.get_latest_checkpoint(run_id)

    # Or resume from specific checkpoint
    checkpoint = store.get_checkpoint(checkpoint_id=3)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointStore",
    "Run",
    "Checkpoint",
]


@dataclass
class Run:
    """A workflow run."""

    id: str
    request: str
    loop_id: str
    started_at: str
    completed_at: str | None
    status: str  # running, completed, failed


@dataclass
class Checkpoint:
    """A checkpoint within a run."""

    id: int
    run_id: str
    sr_turn: int
    role_id: str | None  # None for SR-only checkpoints
    hot_store: dict[str, Any]
    cold_snapshot_ref: str | None
    sr_messages: list[dict[str, Any]]
    role_messages: list[dict[str, Any]] | None
    delegation_history: list[dict[str, Any]]
    created_at: str
    domain_version: int | None  # Domain version at checkpoint time (None for legacy)


_SCHEMA = """
-- Workflow runs
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    request TEXT NOT NULL,
    loop_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running'
);

-- Per-delegation checkpoints
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    sr_turn INTEGER NOT NULL,
    role_id TEXT,
    hot_store TEXT NOT NULL,
    cold_snapshot_ref TEXT,
    sr_messages TEXT NOT NULL,
    role_messages TEXT,
    delegation_history TEXT NOT NULL,
    created_at TEXT NOT NULL,
    domain_version INTEGER
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run_turn ON checkpoints(run_id, sr_turn);
"""


class CheckpointStore:
    """SQLite-backed checkpoint storage for workflow state.

    Parameters
    ----------
    project_path : Path
        Path to the project directory. checkpoints.db will be created here.
    """

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.db_path = self.project_path / "checkpoints.db"
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist and run migrations."""
        self.project_path.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        self._run_migrations(conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run schema migrations for existing databases."""
        # Check if domain_version column exists
        cursor = conn.execute("PRAGMA table_info(checkpoints)")
        columns = {row[1] for row in cursor.fetchall()}

        if "domain_version" not in columns:
            logger.info("Migrating checkpoints: adding domain_version column")
            conn.execute("ALTER TABLE checkpoints ADD COLUMN domain_version INTEGER")
            conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------------------------
    # Run Management
    # -------------------------------------------------------------------------

    def generate_run_id(self) -> str:
        """Generate a unique run ID."""
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        # Count existing runs for today
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE id LIKE ?",
            (f"run-{date_str}-%",),
        )
        count = cursor.fetchone()[0]
        return f"run-{date_str}-{count + 1:03d}"

    def start_run(self, request: str, loop_id: str = "default") -> str:
        """Start a new workflow run.

        Parameters
        ----------
        request : str
            The user's request.
        loop_id : str
            The workflow loop ID.

        Returns
        -------
        str
            The generated run ID.
        """
        run_id = self.generate_run_id()
        now = datetime.now(UTC).isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO runs (id, request, loop_id, started_at, status) VALUES (?, ?, ?, ?, ?)",
            (run_id, request, loop_id, now, "running"),
        )
        conn.commit()
        logger.info(f"Started run: {run_id}")
        return run_id

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as completed.

        Parameters
        ----------
        run_id : str
            The run ID.
        status : str
            Final status (completed, failed).
        """
        now = datetime.now(UTC).isoformat()
        conn = self._get_conn()
        conn.execute(
            "UPDATE runs SET completed_at = ?, status = ? WHERE id = ?",
            (now, status, run_id),
        )
        conn.commit()
        logger.info(f"Completed run: {run_id} ({status})")

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Run(
            id=row["id"],
            request=row["request"],
            loop_id=row["loop_id"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            status=row["status"],
        )

    def get_latest_run(self) -> Run | None:
        """Get the most recent run."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            return None
        return Run(
            id=row["id"],
            request=row["request"],
            loop_id=row["loop_id"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            status=row["status"],
        )

    def list_runs(self, limit: int = 10) -> list[Run]:
        """List recent runs."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [
            Run(
                id=row["id"],
                request=row["request"],
                loop_id=row["loop_id"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                status=row["status"],
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------

    def save_checkpoint(
        self,
        run_id: str,
        sr_turn: int,
        hot_store: dict[str, Any],
        sr_messages: list[dict[str, Any]],
        delegation_history: list[dict[str, Any]],
        role_id: str | None = None,
        role_messages: list[dict[str, Any]] | None = None,
        cold_snapshot_ref: str | None = None,
        domain_version: int | None = None,
    ) -> int:
        """Save a checkpoint after a delegation completes.

        Parameters
        ----------
        run_id : str
            The run ID.
        sr_turn : int
            Current SR turn number.
        hot_store : dict[str, Any]
            Full hot_store state.
        sr_messages : list[dict[str, Any]]
            SR's full message history.
        delegation_history : list[dict[str, Any]]
            List of completed delegations.
        role_id : str | None
            The role that just completed (None for SR-only checkpoints).
        role_messages : list[dict[str, Any]] | None
            The role's message history.
        cold_snapshot_ref : str | None
            Reference to cold_store snapshot if one was created.
        domain_version : int | None
            The domain version at checkpoint creation time.

        Returns
        -------
        int
            The checkpoint ID.
        """
        now = datetime.now(UTC).isoformat()
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO checkpoints
            (run_id, sr_turn, role_id, hot_store, cold_snapshot_ref,
             sr_messages, role_messages, delegation_history, created_at, domain_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sr_turn,
                role_id,
                json.dumps(hot_store, default=str),
                cold_snapshot_ref,
                json.dumps(sr_messages, default=str),
                json.dumps(role_messages, default=str) if role_messages else None,
                json.dumps(delegation_history, default=str),
                now,
                domain_version,
            ),
        )
        conn.commit()
        checkpoint_id = cursor.lastrowid
        logger.info(f"Checkpoint {checkpoint_id}: {role_id or 'SR'} completed (turn {sr_turn})")
        return checkpoint_id  # type: ignore[return-value]

    def get_checkpoint(self, checkpoint_id: int) -> Checkpoint | None:
        """Get a checkpoint by ID."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a run."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        """List all checkpoints for a run."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        return [self._row_to_checkpoint(row) for row in cursor.fetchall()]

    def get_checkpoint_count(self, run_id: str) -> int:
        """Get the number of checkpoints for a run."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?",
            (run_id,),
        )
        count: int = cursor.fetchone()[0]
        return count

    def _row_to_checkpoint(self, row: sqlite3.Row) -> Checkpoint:
        """Convert a database row to a Checkpoint."""
        # Handle legacy checkpoints without domain_version column
        row_dict = dict(row)
        domain_version = row_dict.get("domain_version")

        return Checkpoint(
            id=row["id"],
            run_id=row["run_id"],
            sr_turn=row["sr_turn"],
            role_id=row["role_id"],
            hot_store=json.loads(row["hot_store"]),
            cold_snapshot_ref=row["cold_snapshot_ref"],
            sr_messages=json.loads(row["sr_messages"]),
            role_messages=json.loads(row["role_messages"]) if row["role_messages"] else None,
            delegation_history=json.loads(row["delegation_history"]),
            created_at=row["created_at"],
            domain_version=domain_version,
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its checkpoints."""
        conn = self._get_conn()
        conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
        logger.info(f"Deleted run: {run_id}")

    def delete_old_runs(self, keep_count: int = 10) -> int:
        """Delete old runs, keeping only the most recent ones.

        Parameters
        ----------
        keep_count : int
            Number of recent runs to keep.

        Returns
        -------
        int
            Number of runs deleted.
        """
        conn = self._get_conn()
        # Get IDs of runs to delete
        cursor = conn.execute(
            """
            SELECT id FROM runs
            WHERE id NOT IN (
                SELECT id FROM runs ORDER BY started_at DESC LIMIT ?
            )
            """,
            (keep_count,),
        )
        ids_to_delete = [row["id"] for row in cursor.fetchall()]

        if not ids_to_delete:
            return 0

        # Delete checkpoints and runs
        placeholders = ",".join("?" * len(ids_to_delete))
        conn.execute(f"DELETE FROM checkpoints WHERE run_id IN ({placeholders})", ids_to_delete)
        conn.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", ids_to_delete)
        conn.commit()

        logger.info(f"Deleted {len(ids_to_delete)} old runs")
        return len(ids_to_delete)
