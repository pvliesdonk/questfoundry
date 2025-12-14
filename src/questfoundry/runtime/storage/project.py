"""
Project storage management.

A project represents a single story/game being developed. Each project has:
- project.json: Project metadata
- project.sqlite: Artifact storage
- assets/: Binary assets (images, audio)
- logs/: Event logs (created with --log flag)
- checkpoints/: Session checkpoints for resumption

The SQLite database contains:
- artifacts: All artifacts with system fields
- artifact_versions: Version history for versioned stores
- messages: Inter-agent messages
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ProjectInfo:
    """Project metadata from project.json."""

    id: str
    name: str
    description: str | None = None
    studio_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "studio_id": self.studio_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectInfo:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            studio_id=data.get("studio_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class Project:
    """
    A QuestFoundry project.

    Manages the project directory structure and SQLite database.
    """

    def __init__(self, path: Path):
        """
        Initialize project from path.

        Args:
            path: Path to project directory
        """
        self._path = path
        self._info: ProjectInfo | None = None
        self._conn: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        """Project directory path."""
        return self._path

    @property
    def info(self) -> ProjectInfo | None:
        """Project metadata (loaded lazily)."""
        if self._info is None:
            self._load_info()
        return self._info

    @property
    def db_path(self) -> Path:
        """Path to SQLite database."""
        return self._path / "project.sqlite"

    @property
    def assets_path(self) -> Path:
        """Path to assets directory."""
        return self._path / "assets"

    @property
    def logs_path(self) -> Path:
        """Path to logs directory."""
        return self._path / "logs"

    @property
    def checkpoints_path(self) -> Path:
        """Path to checkpoints directory."""
        return self._path / "checkpoints"

    def exists(self) -> bool:
        """Check if project exists."""
        return (self._path / "project.json").exists()

    def _load_info(self) -> None:
        """Load project info from project.json."""
        info_path = self._path / "project.json"
        if info_path.exists():
            data = json.loads(info_path.read_text())
            self._info = ProjectInfo.from_dict(data)

    def _save_info(self) -> None:
        """Save project info to project.json."""
        if self._info:
            info_path = self._path / "project.json"
            info_path.write_text(json.dumps(self._info.to_dict(), indent=2))

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    @classmethod
    def create(
        cls,
        path: Path,
        name: str,
        description: str | None = None,
        studio_id: str | None = None,
    ) -> Project:
        """
        Create a new project.

        Args:
            path: Directory for the project
            name: Human-readable project name
            description: Optional description
            studio_id: ID of the studio this project uses

        Returns:
            The created Project
        """
        # Create directory structure
        path.mkdir(parents=True, exist_ok=True)
        (path / "assets").mkdir(exist_ok=True)

        # Create project info
        project = cls(path)
        project._info = ProjectInfo(
            id=path.name,  # Use directory name as ID
            name=name,
            description=description,
            studio_id=studio_id,
        )
        project._save_info()

        # Initialize database
        project._init_database()

        return project

    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        conn = self._get_connection()

        conn.executescript(
            """
            -- Artifacts table: stores all artifacts
            CREATE TABLE IF NOT EXISTS artifacts (
                _id TEXT PRIMARY KEY,
                _type TEXT NOT NULL,
                _version INTEGER NOT NULL DEFAULT 1,
                _created_at TEXT NOT NULL,
                _updated_at TEXT NOT NULL,
                _created_by TEXT,
                _lifecycle_state TEXT DEFAULT 'draft',
                _store TEXT,
                data JSON NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(_type);
            CREATE INDEX IF NOT EXISTS idx_artifacts_store ON artifacts(_store);
            CREATE INDEX IF NOT EXISTS idx_artifacts_lifecycle ON artifacts(_lifecycle_state);

            -- Artifact versions: history for versioned stores
            CREATE TABLE IF NOT EXISTS artifact_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                data JSON NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT,
                FOREIGN KEY (artifact_id) REFERENCES artifacts(_id)
            );

            CREATE INDEX IF NOT EXISTS idx_versions_artifact ON artifact_versions(artifact_id);

            -- Messages: inter-agent communication
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE NOT NULL,
                message_type TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                payload JSON NOT NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT DEFAULT 'pending'
            );

            CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_agent, status);
            CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type);

            -- Sessions: track interaction sessions
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                entry_agent TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            );

            -- Turns: individual turns within sessions
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                input TEXT,
                output TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                token_usage JSON,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        """
        )
        conn.commit()

    @classmethod
    def open(cls, path: Path) -> Project:
        """
        Open an existing project.

        Args:
            path: Path to project directory

        Returns:
            The Project

        Raises:
            FileNotFoundError: If project doesn't exist
        """
        project = cls(path)
        if not project.exists():
            raise FileNotFoundError(f"Project not found at {path}")
        return project

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # Artifact operations

    def create_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        data: dict[str, Any],
        store: str | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new artifact.

        Args:
            artifact_id: Unique artifact ID
            artifact_type: Artifact type ID
            data: Artifact data (user fields)
            store: Store to place artifact in
            created_by: Agent ID creating the artifact

        Returns:
            Complete artifact with system fields
        """
        conn = self._get_connection()
        now = datetime.now().isoformat()

        artifact = {
            "_id": artifact_id,
            "_type": artifact_type,
            "_version": 1,
            "_created_at": now,
            "_updated_at": now,
            "_created_by": created_by,
            "_lifecycle_state": "draft",
            "_store": store,
            **data,
        }

        conn.execute(
            """
            INSERT INTO artifacts (_id, _type, _version, _created_at, _updated_at,
                                   _created_by, _lifecycle_state, _store, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                artifact_type,
                1,
                now,
                now,
                created_by,
                "draft",
                store,
                json.dumps(data),
            ),
        )
        conn.commit()

        return artifact

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Get an artifact by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM artifacts WHERE _id = ?", (artifact_id,)
        ).fetchone()

        if row is None:
            return None

        # Combine system fields with data
        data = json.loads(row["data"])
        return {
            "_id": row["_id"],
            "_type": row["_type"],
            "_version": row["_version"],
            "_created_at": row["_created_at"],
            "_updated_at": row["_updated_at"],
            "_created_by": row["_created_by"],
            "_lifecycle_state": row["_lifecycle_state"],
            "_store": row["_store"],
            **data,
        }

    def update_artifact(
        self,
        artifact_id: str,
        data: dict[str, Any],
        updated_by: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Update an artifact.

        Args:
            artifact_id: Artifact ID to update
            data: New data (merged with existing)
            updated_by: Agent ID making the update

        Returns:
            Updated artifact or None if not found
        """
        existing = self.get_artifact(artifact_id)
        if existing is None:
            return None

        conn = self._get_connection()
        now = datetime.now().isoformat()
        new_version = existing["_version"] + 1

        # Merge data
        existing_data = {k: v for k, v in existing.items() if not k.startswith("_")}
        merged_data = {**existing_data, **data}

        conn.execute(
            """
            UPDATE artifacts
            SET _version = ?, _updated_at = ?, data = ?
            WHERE _id = ?
            """,
            (new_version, now, json.dumps(merged_data), artifact_id),
        )
        conn.commit()

        return self.get_artifact(artifact_id)

    def query_artifacts(
        self,
        artifact_type: str | None = None,
        store: str | None = None,
        lifecycle_state: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query artifacts with filters.

        Args:
            artifact_type: Filter by type
            store: Filter by store
            lifecycle_state: Filter by lifecycle state
            limit: Maximum results

        Returns:
            List of matching artifacts
        """
        conn = self._get_connection()

        query = "SELECT * FROM artifacts WHERE 1=1"
        params: list[Any] = []

        if artifact_type:
            query += " AND _type = ?"
            params.append(artifact_type)

        if store:
            query += " AND _store = ?"
            params.append(store)

        if lifecycle_state:
            query += " AND _lifecycle_state = ?"
            params.append(lifecycle_state)

        query += " ORDER BY _updated_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            data = json.loads(row["data"])
            results.append(
                {
                    "_id": row["_id"],
                    "_type": row["_type"],
                    "_version": row["_version"],
                    "_created_at": row["_created_at"],
                    "_updated_at": row["_updated_at"],
                    "_created_by": row["_created_by"],
                    "_lifecycle_state": row["_lifecycle_state"],
                    "_store": row["_store"],
                    **data,
                }
            )

        return results


def list_projects(projects_dir: Path) -> list[Project]:
    """
    List all projects in a directory.

    Args:
        projects_dir: Directory containing project subdirectories

    Returns:
        List of Project objects
    """
    projects = []
    if projects_dir.exists():
        for subdir in projects_dir.iterdir():
            if subdir.is_dir():
                project = Project(subdir)
                if project.exists():
                    projects.append(project)
    return projects
