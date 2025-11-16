"""PostgreSQL-based implementation of StateStore protocol

This module provides a multi-tenant storage backend that implements the
StateStore protocol from questfoundry-py. All operations are scoped by
project_id to enable multiple projects in a shared database.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool
from questfoundry.models.artifact import Artifact
from questfoundry.state.store import StateStore
from questfoundry.state.types import ProjectInfo, SnapshotInfo, TUState


class PostgresStore(StateStore):
    """
    PostgreSQL implementation of StateStore for multi-tenant cold storage.

    This replaces SQLiteStore for concurrent multi-tenant web access.
    All operations are scoped by project_id to isolate tenant data.

    Key features:
    - Project-scoped queries (all tables include project_id column)
    - JSONB columns for efficient artifact data storage and querying
    - Connection pooling for concurrent access
    - ACID transactions

    Thread Safety:
        This class uses connection pooling and is safe for concurrent access
        from multiple threads/requests.

    Example:
        >>> store = PostgresStore(connection_string, project_id="project-123")
        >>> artifact = Artifact(type="hook_card", data={...})
        >>> store.save_artifact(artifact)
    """

    def __init__(self, connection_string: str, project_id: str):
        """
        Initialize PostgreSQL store.

        Args:
            connection_string: PostgreSQL connection URL
            project_id: Project identifier for scoping all queries
        """
        self.connection_string = connection_string
        self.project_id = project_id
        self.pool = ConnectionPool(
            connection_string,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )

    def close(self) -> None:
        """Close connection pool"""
        self.pool.close()

    def get_project_info(self) -> ProjectInfo:
        """Get project metadata scoped to current project_id"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        name,
                        description,
                        version,
                        author,
                        created,
                        modified,
                        metadata
                    FROM project_info WHERE project_id = %s
                    """,
                    (self.project_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise FileNotFoundError(f"Project {self.project_id} not found")
                return ProjectInfo(
                    name=row["name"],
                    description=row["description"],
                    version=row["version"],
                    author=row["author"],
                    created=row["created"],
                    modified=row["modified"],
                    metadata=row["metadata"],
                )

    def save_project_info(self, info: ProjectInfo) -> None:
        """Save project metadata"""
        info.modified = datetime.now()

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_info
                        (
                            project_id,
                            name,
                            description,
                            version,
                            author,
                            created,
                            modified,
                            metadata
                        )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_id)
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        version = EXCLUDED.version,
                        author = EXCLUDED.author,
                        modified = EXCLUDED.modified,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        self.project_id,
                        info.name,
                        info.description,
                        info.version,
                        info.author,
                        info.created,
                        info.modified,
                        Json(info.metadata),
                    ),
                )
                conn.commit()

    def save_artifact(self, artifact: Artifact) -> None:
        """Save artifact with project_id scoping"""
        artifact_id = artifact.metadata.get("id")
        if not artifact_id:
            raise ValueError("Artifact must have 'id' in metadata")

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO artifacts
                        (project_id, artifact_id, artifact_type, data, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, artifact_id)
                    DO UPDATE SET
                        artifact_type = EXCLUDED.artifact_type,
                        data = EXCLUDED.data,
                        metadata = EXCLUDED.metadata,
                        modified = NOW()
                    """,
                    (
                        self.project_id,
                        artifact_id,
                        artifact.type,
                        Json(artifact.data),
                        Json(artifact.metadata),
                    ),
                )
                conn.commit()

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Get artifact by ID within current project_id scope"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT artifact_type, data, metadata
                    FROM artifacts
                    WHERE project_id = %s AND artifact_id = %s
                    """,
                    (self.project_id, artifact_id),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return Artifact(
                    type=row["artifact_type"],
                    data=row["data"],
                    metadata=row["metadata"],
                )

    def list_artifacts(
        self, artifact_type: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[Artifact]:
        """List artifacts within current project_id scope"""
        query = """
            SELECT artifact_type, data, metadata
            FROM artifacts
            WHERE project_id = %s
        """
        params: list[Any] = [self.project_id]

        if artifact_type:
            query += " AND artifact_type = %s"
            params.append(artifact_type)

        # Add JSONB filters
        if filters:
            for key, value in filters.items():
                query += " AND data->>%s = %s"
                params.append(key)
                params.append(str(value))

        query += " ORDER BY modified DESC"

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [
                    Artifact(
                        type=row["artifact_type"],
                        data=row["data"],
                        metadata=row["metadata"],
                    )
                    for row in cur.fetchall()
                ]

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact within current project_id scope"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM artifacts
                    WHERE project_id = %s AND artifact_id = %s
                    """,
                    (self.project_id, artifact_id),
                )
                conn.commit()
                return cur.rowcount > 0

    def save_tu(self, tu: TUState) -> None:
        """Save TU state with project_id scoping"""
        tu.modified = datetime.now()

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tus
                        (
                            project_id,
                            tu_id,
                            status,
                            snapshot_id,
                            created,
                            modified,
                            data,
                            metadata
                        )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, tu_id)
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        snapshot_id = EXCLUDED.snapshot_id,
                        modified = EXCLUDED.modified,
                        data = EXCLUDED.data,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        self.project_id,
                        tu.tu_id,
                        tu.status,
                        tu.snapshot_id,
                        tu.created,
                        tu.modified,
                        Json(tu.data),
                        Json(tu.metadata),
                    ),
                )
                conn.commit()

    def get_tu(self, tu_id: str) -> TUState | None:
        """Get TU by ID within current project_id scope"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tu_id, status, snapshot_id, created, modified, data, metadata
                    FROM tus
                    WHERE project_id = %s AND tu_id = %s
                    """,
                    (self.project_id, tu_id),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return TUState(
                    tu_id=row["tu_id"],
                    status=row["status"],
                    snapshot_id=row["snapshot_id"],
                    created=row["created"],
                    modified=row["modified"],
                    data=row["data"],
                    metadata=row["metadata"],
                )

    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]:
        """List TUs within current project_id scope"""
        query = """
            SELECT tu_id, status, snapshot_id, created, modified, data, metadata
            FROM tus
            WHERE project_id = %s
        """
        params: list[Any] = [self.project_id]

        if filters:
            for key, value in filters.items():
                if key in ["status", "snapshot_id"]:
                    query += f" AND {key} = %s"
                    params.append(value)

        query += " ORDER BY modified DESC"

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [
                    TUState(
                        tu_id=row["tu_id"],
                        status=row["status"],
                        snapshot_id=row["snapshot_id"],
                        created=row["created"],
                        modified=row["modified"],
                        data=row["data"],
                        metadata=row["metadata"],
                    )
                    for row in cur.fetchall()
                ]

    def save_snapshot(self, snapshot: SnapshotInfo) -> None:
        """Save snapshot with project_id scoping"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # Check if snapshot already exists (immutability)
                cur.execute(
                    """
                    SELECT 1 FROM snapshots
                    WHERE project_id = %s AND snapshot_id = %s
                    """,
                    (self.project_id, snapshot.snapshot_id),
                )
                if cur.fetchone():
                    raise ValueError(
                        f"Snapshot '{snapshot.snapshot_id}' already exists. "
                        "Snapshots are immutable and cannot be updated."
                    )

                cur.execute(
                    """
                    INSERT INTO snapshots
                        (project_id, snapshot_id, tu_id, created, description, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        self.project_id,
                        snapshot.snapshot_id,
                        snapshot.tu_id,
                        snapshot.created,
                        snapshot.description,
                        Json(snapshot.metadata),
                    ),
                )
                conn.commit()

    def get_snapshot(self, snapshot_id: str) -> SnapshotInfo | None:
        """Get snapshot by ID within current project_id scope"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT snapshot_id, tu_id, created, description, metadata
                    FROM snapshots
                    WHERE project_id = %s AND snapshot_id = %s
                    """,
                    (self.project_id, snapshot_id),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return SnapshotInfo(
                    snapshot_id=row["snapshot_id"],
                    tu_id=row["tu_id"],
                    created=row["created"],
                    description=row["description"],
                    metadata=row["metadata"],
                )

    def list_snapshots(
        self, filters: dict[str, Any] | None = None
    ) -> list[SnapshotInfo]:
        """List snapshots within current project_id scope"""
        query = """
            SELECT snapshot_id, tu_id, created, description, metadata
            FROM snapshots
            WHERE project_id = %s
        """
        params: list[Any] = [self.project_id]

        if filters and "tu_id" in filters:
            query += " AND tu_id = %s"
            params.append(filters["tu_id"])

        query += " ORDER BY created DESC"

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [
                    SnapshotInfo(
                        snapshot_id=row["snapshot_id"],
                        tu_id=row["tu_id"],
                        created=row["created"],
                        description=row["description"],
                        metadata=row["metadata"],
                    )
                    for row in cur.fetchall()
                ]
