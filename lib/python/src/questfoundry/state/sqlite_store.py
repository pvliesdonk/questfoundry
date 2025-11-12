"""SQLite-based state store implementation"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from .store import StateStore
from .types import ProjectInfo, SnapshotInfo, TUState

logger = logging.getLogger(__name__)


class SQLiteStore(StateStore):
    """
    SQLite implementation of StateStore for .qfproj files.

    Provides cold storage with ACID transactions, JSON querying,
    and audit history. Suitable for archived/finalized project state.

    Thread Safety:
        This class is NOT thread-safe by default. Each thread should create
        its own SQLiteStore instance. While check_same_thread=False is enabled
        for the connection, SQLite itself has limitations with concurrent writes.
        The WAL mode improves concurrent read performance, but writes are still
        serialized by SQLite's locking mechanism.

        For multi-threaded applications:
        - Use separate store instances per thread, OR
        - Implement external synchronization (locks/queues) around write operations

    Example:
        >>> store = SQLiteStore("my_game.qfproj")
        >>> store.init_database()
        >>> info = ProjectInfo(name="My Game")
        >>> store.save_project_info(info)
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to .qfproj file (SQLite database)
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

        logger.debug("Initializing SQLiteStore at %s", self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create database connection.

        Configures connection for optimal concurrent access with WAL mode
        and appropriate timeouts. Note that while the connection is configured
        with check_same_thread=False, proper thread safety requires either:
        1. One store instance per thread, OR
        2. External synchronization around database operations

        Returns:
            SQLite database connection
        """
        if self._conn is None:
            # Use check_same_thread=False to allow connection sharing
            # WARNING: This does not make the store thread-safe. See class docstring.
            self._conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # Wait up to 30 seconds for locks
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row

            # Enable Write-Ahead Logging for better concurrent read access
            self._conn.execute("PRAGMA journal_mode=WAL")

            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")

            # Commit pragma changes
            self._conn.commit()

        return self._conn

    def close(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def init_database(self) -> None:
        """
        Initialize database schema.

        Creates all tables, indexes, and initial data if database is new.

        Raises:
            IOError: If schema initialization fails
        """
        logger.info("Initializing database at %s", self.db_path)
        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()

        conn = self._get_connection()
        logger.trace("Executing schema initialization script")
        conn.executescript(schema_sql)
        conn.commit()
        logger.info("Database initialized successfully")

    def get_schema_version(self) -> int:
        """Get current database schema version"""
        conn = self._get_connection()
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        return result[0] if result and result[0] else 0

    def get_project_info(self) -> ProjectInfo:
        """Get project metadata"""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT name, description, version, author, created, modified, metadata
            FROM project WHERE id = 1
            """
        )
        row = cursor.fetchone()

        if not row:
            raise FileNotFoundError("Project metadata not found. Initialize first.")

        return ProjectInfo(
            name=row["name"],
            description=row["description"],
            version=row["version"],
            author=row["author"],
            created=datetime.fromisoformat(row["created"]),
            modified=datetime.fromisoformat(row["modified"]),
            metadata=json.loads(row["metadata"]),
        )

    def save_project_info(self, info: ProjectInfo) -> None:
        """Save project metadata"""
        conn = self._get_connection()

        # Update modified timestamp
        info.modified = datetime.now()

        conn.execute(
            """
            INSERT OR REPLACE INTO project
            (id, name, description, version, author, created, modified, metadata)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                info.name,
                info.description,
                info.version,
                info.author,
                info.created.isoformat(),
                info.modified.isoformat(),
                json.dumps(info.metadata),
            ),
        )
        conn.commit()

    def save_artifact(self, artifact: Artifact) -> None:
        """Save an artifact"""
        conn = self._get_connection()

        # Ensure artifact has an ID
        artifact_id = artifact.metadata.get("id")
        if not artifact_id:
            logger.error("Attempted to save artifact without 'id' in metadata")
            raise ValueError("Artifact must have an 'id' in metadata")

        # Check if artifact already exists to determine create vs update
        existing = self.get_artifact(artifact_id)
        is_create = existing is None

        action = "create" if is_create else "update"
        logger.debug(
            "Saving artifact '%s' to cold storage (action=%s, type=%s)",
            artifact_id,
            action,
            artifact.type,
        )

        now = datetime.now().isoformat()

        # Handle created timestamp: preserve if exists, set if new
        if is_create:
            created = now
            # Update metadata with created timestamp
            artifact.metadata["created"] = created
            logger.trace("New artifact - setting created timestamp")
        else:
            # Preserve existing created timestamp
            cursor = conn.execute(
                "SELECT created FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            )
            row = cursor.fetchone()
            created = row["created"] if row else now
            logger.trace("Updating existing artifact - preserving created timestamp")

        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts
            (artifact_id, artifact_type, created, modified, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                artifact.type,
                created,
                now,
                json.dumps(artifact.data),
                json.dumps(artifact.metadata),
            ),
        )

        # Log to history with actual changes
        changes = {
            "type": artifact.type,
            "data": artifact.data,
            "metadata": artifact.metadata,
        }
        self._log_history("artifact", artifact_id, action, changes)

        conn.commit()
        logger.info("Successfully saved artifact '%s' to cold storage", artifact_id)

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Retrieve an artifact by ID"""
        logger.trace("Retrieving artifact '%s' from cold storage", artifact_id)

        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT artifact_type, data, metadata
            FROM artifacts
            WHERE artifact_id = ?
            """,
            (artifact_id,),
        )
        row = cursor.fetchone()

        if not row:
            logger.debug("Artifact '%s' not found in cold storage", artifact_id)
            return None

        logger.debug("Found artifact '%s' in cold storage", artifact_id)
        return Artifact(
            type=row["artifact_type"],
            data=json.loads(row["data"]),
            metadata=json.loads(row["metadata"]),
        )

    def list_artifacts(
        self, artifact_type: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[Artifact]:
        """List artifacts with optional filtering"""
        conn = self._get_connection()

        # Whitelist of allowed filter keys to prevent SQL injection
        ALLOWED_FILTER_KEYS = {"status", "author", "name", "trigger"}

        # Build query
        query = "SELECT artifact_type, data, metadata FROM artifacts"
        conditions = []
        params = []

        if artifact_type:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)

        if filters:
            for key, value in filters.items():
                # Validate key against whitelist
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(
                        f"Invalid filter key '{key}'. "
                        f"Allowed keys: {', '.join(sorted(ALLOWED_FILTER_KEYS))}"
                    )
                # Use JSON extraction for filtering
                conditions.append(f"json_extract(data, '$.{key}') = ?")
                params.append(value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY modified DESC"

        cursor = conn.execute(query, params)
        artifacts = []

        for row in cursor.fetchall():
            artifacts.append(
                Artifact(
                    type=row["artifact_type"],
                    data=json.loads(row["data"]),
                    metadata=json.loads(row["metadata"]),
                )
            )

        return artifacts

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact"""
        logger.debug("Deleting artifact '%s' from cold storage", artifact_id)

        conn = self._get_connection()

        cursor = conn.execute(
            "DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )

        deleted = cursor.rowcount > 0
        if deleted:
            logger.trace("Logging artifact '%s' deletion to history", artifact_id)
            self._log_history("artifact", artifact_id, "delete", {})
            logger.info(
                "Successfully deleted artifact '%s' from cold storage", artifact_id
            )
        else:
            logger.warning("Artifact '%s' not found for deletion", artifact_id)

        conn.commit()
        return deleted

    def save_tu(self, tu: TUState) -> None:
        """Save TU state"""
        conn = self._get_connection()

        # Update modified timestamp
        tu.modified = datetime.now()

        conn.execute(
            """
            INSERT OR REPLACE INTO tus
            (tu_id, status, snapshot_id, created, modified, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tu.tu_id,
                tu.status,
                tu.snapshot_id,
                tu.created.isoformat(),
                tu.modified.isoformat(),
                json.dumps(tu.data),
                json.dumps(tu.metadata),
            ),
        )
        conn.commit()

        self._log_history("tu", tu.tu_id, "update", {})

    def get_tu(self, tu_id: str) -> TUState | None:
        """Retrieve TU state by ID"""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT tu_id, status, snapshot_id, created, modified, data, metadata
            FROM tus
            WHERE tu_id = ?
            """,
            (tu_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return TUState(
            tu_id=row["tu_id"],
            status=row["status"],
            snapshot_id=row["snapshot_id"],
            created=datetime.fromisoformat(row["created"]),
            modified=datetime.fromisoformat(row["modified"]),
            data=json.loads(row["data"]),
            metadata=json.loads(row["metadata"]),
        )

    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]:
        """List TUs with optional filtering"""
        conn = self._get_connection()

        query = (
            "SELECT tu_id, status, snapshot_id, created, modified, "
            "data, metadata FROM tus"
        )
        conditions = []
        params = []

        if filters:
            for key, value in filters.items():
                if key in ["status", "snapshot_id"]:
                    conditions.append(f"{key} = ?")
                    params.append(value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY modified DESC"

        cursor = conn.execute(query, params)
        tus = []

        for row in cursor.fetchall():
            tus.append(
                TUState(
                    tu_id=row["tu_id"],
                    status=row["status"],
                    snapshot_id=row["snapshot_id"],
                    created=datetime.fromisoformat(row["created"]),
                    modified=datetime.fromisoformat(row["modified"]),
                    data=json.loads(row["data"]),
                    metadata=json.loads(row["metadata"]),
                )
            )

        return tus

    def save_snapshot(self, snapshot: SnapshotInfo) -> None:
        """
        Save snapshot metadata.

        Snapshots are immutable - attempting to save a snapshot with an
        existing ID will raise an error.

        Raises:
            ValueError: If snapshot with same ID already exists
        """
        conn = self._get_connection()

        # Check if snapshot already exists
        existing = self.get_snapshot(snapshot.snapshot_id)
        if existing is not None:
            raise ValueError(
                f"Snapshot '{snapshot.snapshot_id}' already exists. "
                "Snapshots are immutable and cannot be updated."
            )

        conn.execute(
            """
            INSERT INTO snapshots
            (snapshot_id, tu_id, created, description, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.tu_id,
                snapshot.created.isoformat(),
                snapshot.description,
                json.dumps(snapshot.metadata),
            ),
        )

        # Log to history
        changes = {
            "tu_id": snapshot.tu_id,
            "description": snapshot.description,
            "metadata": snapshot.metadata,
        }
        self._log_history("snapshot", snapshot.snapshot_id, "create", changes)

        conn.commit()

    def get_snapshot(self, snapshot_id: str) -> SnapshotInfo | None:
        """Retrieve snapshot by ID"""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT snapshot_id, tu_id, created, description, metadata
            FROM snapshots
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return SnapshotInfo(
            snapshot_id=row["snapshot_id"],
            tu_id=row["tu_id"],
            created=datetime.fromisoformat(row["created"]),
            description=row["description"],
            metadata=json.loads(row["metadata"]),
        )

    def list_snapshots(
        self, filters: dict[str, Any] | None = None
    ) -> list[SnapshotInfo]:
        """List snapshots with optional filtering"""
        conn = self._get_connection()

        query = (
            "SELECT snapshot_id, tu_id, created, description, metadata FROM snapshots"
        )
        conditions = []
        params = []

        if filters and "tu_id" in filters:
            conditions.append("tu_id = ?")
            params.append(filters["tu_id"])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created DESC"

        cursor = conn.execute(query, params)
        snapshots = []

        for row in cursor.fetchall():
            snapshots.append(
                SnapshotInfo(
                    snapshot_id=row["snapshot_id"],
                    tu_id=row["tu_id"],
                    created=datetime.fromisoformat(row["created"]),
                    description=row["description"],
                    metadata=json.loads(row["metadata"]),
                )
            )

        return snapshots

    def save_or_replace_snapshot(self, snapshot: SnapshotInfo) -> None:
        """
        Save or replace snapshot metadata.

        Unlike save_snapshot(), this allows replacing existing snapshots.
        Useful for import operations.

        Args:
            snapshot: Snapshot to save
        """
        conn = self._get_connection()

        conn.execute(
            """
            INSERT OR REPLACE INTO snapshots
            (snapshot_id, tu_id, created, description, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.tu_id,
                snapshot.created.isoformat(),
                snapshot.description,
                json.dumps(snapshot.metadata),
            ),
        )

        # Log to history
        changes = {
            "tu_id": snapshot.tu_id,
            "description": snapshot.description,
            "metadata": snapshot.metadata,
        }
        self._log_history("snapshot", snapshot.snapshot_id, "upsert", changes)
        conn.commit()

    def get_artifacts_by_snapshot_id(self, snapshot_id: str) -> list[Artifact]:
        """
        Get all artifacts associated with a snapshot.

        Retrieves artifacts that explicitly reference the snapshot ID
        in their metadata.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            List of artifacts
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT artifact_id, artifact_type, data, metadata, created, modified
            FROM artifacts
            WHERE json_extract(metadata, '$.snapshot_id') = ?
            ORDER BY modified DESC
            """,
            (snapshot_id,),
        )

        artifacts = []
        for row in cursor.fetchall():
            artifact = Artifact(
                type=row["artifact_type"],
                data=json.loads(row["data"]),
                metadata=json.loads(row["metadata"]),
            )
            artifacts.append(artifact)

        return artifacts

    def get_artifacts_by_ids(self, artifact_ids: list[str]) -> list[Artifact]:
        """
        Get multiple artifacts by IDs in a single query.

        This is more efficient than calling get_artifact() in a loop.

        Args:
            artifact_ids: List of artifact IDs to retrieve

        Returns:
            List of artifacts (may be fewer than requested if some don't exist)
        """
        if not artifact_ids:
            return []

        conn = self._get_connection()

        # Build parameterized query with correct number of placeholders
        placeholders = ",".join("?" * len(artifact_ids))
        query = f"""
            SELECT artifact_id, artifact_type, data, metadata
            FROM artifacts
            WHERE artifact_id IN ({placeholders})
        """

        cursor = conn.execute(query, artifact_ids)

        artifacts = []
        for row in cursor.fetchall():
            artifact = Artifact(
                type=row["artifact_type"],
                data=json.loads(row["data"]),
                metadata=json.loads(row["metadata"]),
            )
            artifacts.append(artifact)

        return artifacts

    def _log_history(
        self, entity_type: str, entity_id: str, action: str, changes: dict[str, Any]
    ) -> None:
        """Log action to audit history"""
        conn = self._get_connection()

        conn.execute(
            """
            INSERT INTO history (entity_type, entity_id, action, changes, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entity_type,
                entity_id,
                action,
                json.dumps(changes),
                json.dumps({}),
            ),
        )
        # Note: commit is done by calling function

    def __enter__(self) -> "SQLiteStore":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        self.close()
