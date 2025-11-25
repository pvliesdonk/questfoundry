from __future__ import annotations

"""
Cold Store - durable storage for Cold Source of Truth (SoT).

Design:
    - One project directory per project_id, rooted under a configurable base dir.
    - Each project directory contains:
        - cold_sot.db   (SQLite file for canonical cold_sot and snapshots)

    - We currently persist:
        - cold_state(project_id, data JSON, updated_at)
        - snapshots(id, project_id, tu_id, created_at, data JSON)

The runtime works with Python dicts for `cold_sot`; this module handles
JSON <-> SQLite mapping and directory management.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_project_root() -> Path:
    """
    Default root for QuestFoundry projects under the user's home directory.

    Example: ~/.questfoundry/projects
    """
    home = Path(os.path.expanduser("~"))
    return home / ".questfoundry" / "projects"


@dataclass
class ColdStore:
    """
    SQLite-backed storage for Cold SoT per project.

    The base_dir is the root under which per-project directories are created.
    Each project directory stores a single SQLite file `cold_sot.db`.
    """

    base_dir: Path

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            self.base_dir = _default_project_root()
        else:
            self.base_dir = Path(base_dir).expanduser()

    # --- Public API -----------------------------------------------------

    def load_cold(self, project_id: str) -> dict[str, Any] | None:
        """
        Load the canonical cold_sot dict for a project, if it exists.
        """
        conn = self._get_connection(project_id, create_if_missing=False)
        if conn is None:
            return None
        try:
            cur = conn.execute(
                "SELECT data FROM cold_state WHERE project_id = ?", (project_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def save_cold(self, project_id: str, cold_sot: dict[str, Any]) -> None:
        """
        Upsert the canonical cold_sot for a project.
        """
        conn = self._get_connection(project_id, create_if_missing=True)
        try:
            data = json.dumps(cold_sot or {})
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO cold_state (project_id, data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (project_id, data, now),
            )
            conn.commit()
        finally:
            conn.close()

    def append_snapshot(
        self, project_id: str, tu_id: str, snapshot: dict[str, Any]
    ) -> None:
        """
        Append a snapshot row for audit/history.

        The snapshot dict is expected to be produced by StateManager.snapshot_state().
        """
        conn = self._get_connection(project_id, create_if_missing=True)
        try:
            data = json.dumps(snapshot or {})
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO snapshots (project_id, tu_id, created_at, data)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, tu_id, now, data),
            )
            conn.commit()
        finally:
            conn.close()

    # --- Internal helpers -----------------------------------------------

    def _ensure_project_dir(self, project_id: str) -> Path:
        project_dir = self.base_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def _get_db_path(self, project_id: str) -> Path:
        project_dir = self._ensure_project_dir(project_id)
        return project_dir / "cold_sot.db"

    def _get_connection(
        self, project_id: str, create_if_missing: bool
    ) -> sqlite3.Connection | None:
        """
        Get a SQLite connection for a project, creating schema if needed.

        When create_if_missing is False and the DB file does not exist,
        returns None.
        """
        db_path = self._get_db_path(project_id)
        if not db_path.exists() and not create_if_missing:
            return None

        conn = sqlite3.connect(db_path)
        # Basic pragmas for small-scale local DB
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        self._ensure_schema(conn)
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        """
        Create tables if they do not exist.
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cold_state (
                project_id TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                tu_id      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data       TEXT NOT NULL
            )
            """
        )
        conn.commit()
