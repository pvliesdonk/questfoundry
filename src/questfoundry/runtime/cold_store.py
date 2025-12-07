"""Cold Store - SQLite-based persistent storage for player-safe content.

This module implements the Cold SoT (Source of Truth) for QuestFoundry v3.
The Cold Store contains immutable, player-safe content that can be surfaced
to players without spoilers.

Architecture
------------
The Cold Store uses SQLite for persistence with a hybrid approach:
- Metadata and relationships stored in SQLite tables
- Large content (book sections, art) stored as external files with SHA-256 hashes
- Snapshots provide deterministic builds via content hashing

Key Concepts
------------
**Cold vs Hot**:
    - Cold: Immutable, player-safe, no spoilers
    - Hot: Mutable, work-in-progress, may contain spoilers

**Cold-Only Rule**:
    Player-facing components (Player Navigator, Book Binder) receive ONLY Cold
    content. Never mix Hot and Cold in player views.

**Single Snapshot Sourcing**:
    Any view the player sees comes from a SINGLE Cold snapshot.
    No mixing of snapshots to prevent inconsistencies.

**Safety Triple**:
    Cold content must satisfy: hot_cold="cold", player_safe=True, spoilers="forbidden"

Usage
-----
Create a new project cold store::

    cold = ColdStore.create("my_project.qfproj")

Load existing project::

    cold = ColdStore.load("my_project.qfproj")

Add content and create snapshot::

    cold.add_section("chapter_1", "The story begins...", {"title": "Chapter 1"})
    snapshot_id = cold.create_snapshot("First chapter complete")

TODO (v3 Missing Features)
--------------------------
The following v2 features are not yet implemented:

- [ ] Art asset storage (add_art_asset, get_art_assets)
- [ ] Manifest generation for build reproducibility
- [ ] Trace unit integration (hot_sot -> cold_sot promotion)
- [ ] Quality gate check tracking
- [ ] View log audit trail
- [ ] Cold file external storage (currently stores content directly)
- [ ] SHA-256 integrity validation on load
- [ ] Snapshot diff/comparison utilities
- [ ] Export/import between projects
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ColdSection(BaseModel):
    """A section of player-safe book content.

    Sections are the primary content units in the Cold Store.
    They represent finalized prose that can be shown to players.

    Attributes
    ----------
    id : str
        Unique section identifier (e.g., "chapter_1", "scene_intro").
    content : str
        The prose content of the section.
    metadata : dict[str, Any]
        Additional metadata (title, author, tags, etc.).
    content_hash : str
        SHA-256 hash of content for integrity validation.
    created_at : datetime
        When the section was created.
    source_artifact_id : str | None
        ID of the hot artifact this was promoted from (if any).
    """

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    source_artifact_id: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Compute content hash if not provided."""
        if not self.content_hash and self.content:
            self.content_hash = _compute_hash(self.content)


class ColdSnapshot(BaseModel):
    """A point-in-time snapshot of the Cold Store.

    Snapshots enable deterministic builds - the same snapshot ID
    always produces the same output. Used for version control
    and reproducibility.

    Attributes
    ----------
    id : str
        Unique snapshot identifier (typically a hash).
    created_at : datetime
        When the snapshot was created.
    description : str
        Human-readable description of this snapshot.
    section_ids : list[str]
        IDs of sections included in this snapshot.
    manifest_hash : str
        SHA-256 hash of the manifest for integrity validation.
    """

    id: str
    created_at: datetime = Field(default_factory=datetime.now)
    description: str = ""
    section_ids: list[str] = Field(default_factory=list)
    manifest_hash: str = ""


class ColdStoreStats(BaseModel):
    """Statistics about the Cold Store.

    Attributes
    ----------
    section_count : int
        Number of sections in the store.
    snapshot_count : int
        Number of snapshots created.
    total_content_bytes : int
        Total size of all section content.
    last_snapshot_id : str | None
        ID of the most recent snapshot.
    last_snapshot_at : datetime | None
        When the last snapshot was created.
    """

    section_count: int = 0
    snapshot_count: int = 0
    total_content_bytes: int = 0
    last_snapshot_id: str | None = None
    last_snapshot_at: datetime | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _compute_manifest_hash(section_ids: list[str], sections: dict[str, str]) -> str:
    """Compute hash of manifest (ordered section hashes)."""
    # Create deterministic manifest: sorted section_ids with their content hashes
    manifest_parts = []
    for sid in sorted(section_ids):
        content_hash = sections.get(sid, "")
        manifest_parts.append(f"{sid}:{content_hash}")
    manifest_str = "\n".join(manifest_parts)
    return _compute_hash(manifest_str)


# =============================================================================
# ColdStore Implementation
# =============================================================================


class ColdStore:
    """SQLite-based persistent Cold Store for player-safe content.

    The Cold Store is the canonical source of truth for player-facing
    content. It provides:

    - Persistent storage (SQLite database file)
    - Content integrity via SHA-256 hashing
    - Snapshots for deterministic builds
    - Isolation from Hot (work-in-progress) content

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file (.qfproj).

    Examples
    --------
    Create a new project::

        cold = ColdStore.create("my_project.qfproj")
        cold.add_section("intro", "Welcome to the adventure...")
        cold.create_snapshot("Initial content")

    Load existing project::

        cold = ColdStore.load("my_project.qfproj")
        sections = cold.list_sections()

    Notes
    -----
    **Thread Safety**: Each ColdStore instance maintains its own connection.
    For multi-threaded access, create separate instances or use connection pooling.

    **File Locking**: SQLite provides automatic file locking. Concurrent writes
    from multiple processes may fail - coordinate writes at application level.
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        """Initialize Cold Store with database path.

        Use ColdStore.create() or ColdStore.load() factory methods instead
        of calling this directly.
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def create(cls, path: str | Path) -> ColdStore:
        """Create a new Cold Store project file.

        Parameters
        ----------
        path : str | Path
            Path for the new database file. Will be created.

        Returns
        -------
        ColdStore
            New Cold Store instance with initialized schema.

        Raises
        ------
        FileExistsError
            If the file already exists.

        Examples
        --------
        ::

            cold = ColdStore.create("new_project.qfproj")
        """
        db_path = Path(path)
        if db_path.exists():
            raise FileExistsError(f"Cold Store already exists: {db_path}")

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        store = cls(db_path)
        store._init_schema()
        logger.info(f"Created new Cold Store: {db_path}")
        return store

    @classmethod
    def load(cls, path: str | Path) -> ColdStore:
        """Load an existing Cold Store project file.

        Parameters
        ----------
        path : str | Path
            Path to the existing database file.

        Returns
        -------
        ColdStore
            Cold Store instance connected to the database.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        ::

            cold = ColdStore.load("existing_project.qfproj")
        """
        db_path = Path(path)
        if not db_path.exists():
            raise FileNotFoundError(f"Cold Store not found: {db_path}")

        store = cls(db_path)
        store._verify_schema()
        logger.info(f"Loaded Cold Store: {db_path}")
        return store

    @classmethod
    def load_or_create(cls, path: str | Path) -> ColdStore:
        """Load existing Cold Store or create new one if not found.

        This is the recommended method for workflow initialization.
        It handles both new projects and resuming existing ones.

        Parameters
        ----------
        path : str | Path
            Path to the database file.

        Returns
        -------
        ColdStore
            Cold Store instance (new or existing).

        Examples
        --------
        ::

            # At workflow start
            cold = ColdStore.load_or_create("project.qfproj")
        """
        db_path = Path(path)
        if db_path.exists():
            return cls.load(db_path)
        return cls.create(db_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper cleanup."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        yield self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript("""
                -- Schema version tracking
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                -- Cold sections (player-safe book content)
                CREATE TABLE IF NOT EXISTS sections (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    source_artifact_id TEXT
                );

                -- Snapshots for deterministic builds
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    section_ids TEXT NOT NULL DEFAULT '[]',
                    manifest_hash TEXT NOT NULL
                );

                -- Snapshot-section junction for efficient queries
                CREATE TABLE IF NOT EXISTS snapshot_sections (
                    snapshot_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, section_id),
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id),
                    FOREIGN KEY (section_id) REFERENCES sections(id)
                );

                -- TODO: Art assets table (not yet implemented)
                -- CREATE TABLE IF NOT EXISTS art_assets (
                --     id TEXT PRIMARY KEY,
                --     file_path TEXT NOT NULL,
                --     content_hash TEXT NOT NULL,
                --     metadata TEXT NOT NULL DEFAULT '{}',
                --     created_at TEXT NOT NULL
                -- );

                -- TODO: Trace unit tracking (not yet implemented)
                -- CREATE TABLE IF NOT EXISTS trace_units (
                --     id TEXT PRIMARY KEY,
                --     status TEXT NOT NULL,
                --     created_at TEXT NOT NULL,
                --     closed_at TEXT,
                --     hot_artifacts TEXT NOT NULL DEFAULT '[]',
                --     cold_sections TEXT NOT NULL DEFAULT '[]'
                -- );

                -- TODO: Quality check records (not yet implemented)
                -- CREATE TABLE IF NOT EXISTS quality_checks (
                --     id TEXT PRIMARY KEY,
                --     trace_unit_id TEXT NOT NULL,
                --     check_type TEXT NOT NULL,
                --     passed INTEGER NOT NULL,
                --     details TEXT NOT NULL DEFAULT '{}',
                --     checked_at TEXT NOT NULL
                -- );

                -- TODO: View log for audit trail (not yet implemented)
                -- CREATE TABLE IF NOT EXISTS view_log (
                --     id INTEGER PRIMARY KEY AUTOINCREMENT,
                --     snapshot_id TEXT NOT NULL,
                --     viewer_type TEXT NOT NULL,
                --     viewed_at TEXT NOT NULL,
                --     details TEXT NOT NULL DEFAULT '{}'
                -- );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_sections_created_at ON sections(created_at);
                CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON snapshots(created_at);
            """)

            # Set schema version
            conn.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION)),
            )
            conn.commit()

    def _verify_schema(self) -> None:
        """Verify database schema is compatible."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM schema_info WHERE key = ?",
                ("schema_version",),
            )
            row = cursor.fetchone()
            if row is None:
                # Old database without version - attempt migration
                logger.warning("Cold Store has no schema version, attempting upgrade")
                self._init_schema()
                return

            version = int(row["value"])
            if version > self.SCHEMA_VERSION:
                raise ValueError(
                    f"Cold Store schema version {version} is newer than supported {self.SCHEMA_VERSION}"
                )
            if version < self.SCHEMA_VERSION:
                # TODO: Implement schema migrations
                logger.warning(f"Cold Store schema version {version} needs migration")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ColdStore:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close connection."""
        self.close()

    # =========================================================================
    # Section Operations
    # =========================================================================

    def add_section(
        self,
        section_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        source_artifact_id: str | None = None,
    ) -> ColdSection:
        """Add or update a section in the Cold Store.

        Sections are the primary content units. Each section represents
        a piece of player-safe prose content.

        Parameters
        ----------
        section_id : str
            Unique identifier for the section.
        content : str
            The prose content.
        metadata : dict[str, Any] | None, optional
            Additional metadata (title, tags, etc.). Defaults to None.
        source_artifact_id : str | None, optional
            ID of hot artifact this was promoted from. Defaults to None.

        Returns
        -------
        ColdSection
            The created/updated section.

        Examples
        --------
        ::

            section = cold.add_section(
                "chapter_1",
                "It was a dark and stormy night...",
                metadata={"title": "Chapter 1: The Beginning"}
            )
        """
        metadata = metadata or {}
        content_hash = _compute_hash(content)
        created_at = datetime.now()

        section = ColdSection(
            id=section_id,
            content=content,
            metadata=metadata,
            content_hash=content_hash,
            created_at=created_at,
            source_artifact_id=source_artifact_id,
        )

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sections
                (id, content, content_hash, metadata, created_at, source_artifact_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    section_id,
                    content,
                    content_hash,
                    json.dumps(metadata),
                    created_at.isoformat(),
                    source_artifact_id,
                ),
            )
            conn.commit()

        logger.debug(f"Added section: {section_id}")
        return section

    def get_section(self, section_id: str) -> ColdSection | None:
        """Get a section by ID.

        Parameters
        ----------
        section_id : str
            The section identifier.

        Returns
        -------
        ColdSection | None
            The section, or None if not found.

        Examples
        --------
        ::

            section = cold.get_section("chapter_1")
            if section:
                print(section.content)
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sections WHERE id = ?",
                (section_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return ColdSection(
                id=row["id"],
                content=row["content"],
                content_hash=row["content_hash"],
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                source_artifact_id=row["source_artifact_id"],
            )

    def list_sections(self) -> list[str]:
        """List all section IDs in the store.

        Returns
        -------
        list[str]
            List of section IDs, sorted by creation time.

        Examples
        --------
        ::

            for section_id in cold.list_sections():
                section = cold.get_section(section_id)
                print(f"{section_id}: {section.metadata.get('title', 'Untitled')}")
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT id FROM sections ORDER BY created_at")
            return [row["id"] for row in cursor.fetchall()]

    def delete_section(self, section_id: str) -> bool:
        """Delete a section from the store.

        Note: Sections included in snapshots cannot be deleted.
        This preserves snapshot integrity.

        Parameters
        ----------
        section_id : str
            The section identifier.

        Returns
        -------
        bool
            True if deleted, False if not found or protected.

        Examples
        --------
        ::

            if cold.delete_section("draft_chapter"):
                print("Deleted")
        """
        with self._connection() as conn:
            # Check if section is in any snapshot
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM snapshot_sections WHERE section_id = ?",
                (section_id,),
            )
            if cursor.fetchone()["count"] > 0:
                logger.warning(f"Cannot delete section {section_id}: included in snapshot")
                return False

            cursor = conn.execute(
                "DELETE FROM sections WHERE id = ?",
                (section_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Snapshot Operations
    # =========================================================================

    def create_snapshot(self, description: str = "") -> str:
        """Create a snapshot of the current Cold Store state.

        Snapshots capture the current state of all sections for
        deterministic builds. The same snapshot ID always produces
        the same output.

        Parameters
        ----------
        description : str, optional
            Human-readable description. Defaults to "".

        Returns
        -------
        str
            The snapshot ID (SHA-256 hash of manifest).

        Examples
        --------
        ::

            snapshot_id = cold.create_snapshot("Chapter 1 complete")
            print(f"Created snapshot: {snapshot_id}")
        """
        with self._connection() as conn:
            # Get all sections with their content hashes
            cursor = conn.execute("SELECT id, content_hash FROM sections ORDER BY id")
            sections = {row["id"]: row["content_hash"] for row in cursor.fetchall()}
            section_ids = list(sections.keys())

            # Compute manifest hash (this becomes the snapshot ID)
            manifest_hash = _compute_manifest_hash(section_ids, sections)
            snapshot_id = manifest_hash[:16]  # Use first 16 chars for readability

            # Check if this exact snapshot already exists (deterministic)
            cursor = conn.execute(
                "SELECT id FROM snapshots WHERE id = ?",
                (snapshot_id,),
            )
            if cursor.fetchone() is not None:
                # Same content = same snapshot, just return existing ID
                logger.debug(f"Snapshot {snapshot_id} already exists (same content)")
                return snapshot_id

            created_at = datetime.now()

            # Insert snapshot
            conn.execute(
                """
                INSERT INTO snapshots
                (id, created_at, description, section_ids, manifest_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    created_at.isoformat(),
                    description,
                    json.dumps(section_ids),
                    manifest_hash,
                ),
            )

            # Insert junction records
            for sid in section_ids:
                conn.execute(
                    "INSERT INTO snapshot_sections (snapshot_id, section_id) VALUES (?, ?)",
                    (snapshot_id, sid),
                )

            conn.commit()

        logger.info(f"Created snapshot: {snapshot_id} ({len(section_ids)} sections)")
        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> ColdSnapshot | None:
        """Get a snapshot by ID.

        Parameters
        ----------
        snapshot_id : str
            The snapshot identifier.

        Returns
        -------
        ColdSnapshot | None
            The snapshot, or None if not found.

        Examples
        --------
        ::

            snapshot = cold.get_snapshot("abc123")
            if snapshot:
                print(f"Sections: {snapshot.section_ids}")
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM snapshots WHERE id = ?",
                (snapshot_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return ColdSnapshot(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                description=row["description"],
                section_ids=json.loads(row["section_ids"]),
                manifest_hash=row["manifest_hash"],
            )

    def list_snapshots(self) -> list[str]:
        """List all snapshot IDs.

        Returns
        -------
        list[str]
            List of snapshot IDs, sorted by creation time (newest first).

        Examples
        --------
        ::

            for snapshot_id in cold.list_snapshots():
                snapshot = cold.get_snapshot(snapshot_id)
                print(f"{snapshot_id}: {snapshot.description}")
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT id FROM snapshots ORDER BY created_at DESC")
            return [row["id"] for row in cursor.fetchall()]

    def get_latest_snapshot(self) -> ColdSnapshot | None:
        """Get the most recent snapshot.

        Returns
        -------
        ColdSnapshot | None
            The latest snapshot, or None if no snapshots exist.

        Examples
        --------
        ::

            latest = cold.get_latest_snapshot()
            if latest:
                print(f"Using snapshot: {latest.id}")
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM snapshots ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                return None

            return ColdSnapshot(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                description=row["description"],
                section_ids=json.loads(row["section_ids"]),
                manifest_hash=row["manifest_hash"],
            )

    def get_snapshot_sections(self, snapshot_id: str) -> list[ColdSection]:
        """Get all sections for a snapshot.

        This is the primary method for retrieving player-facing content.
        It returns exactly the sections that were present when the
        snapshot was created.

        Parameters
        ----------
        snapshot_id : str
            The snapshot identifier.

        Returns
        -------
        list[ColdSection]
            List of sections in the snapshot, in creation order.

        Raises
        ------
        ValueError
            If snapshot not found.

        Examples
        --------
        ::

            sections = cold.get_snapshot_sections("abc123")
            for section in sections:
                print(section.content)
        """
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        sections = []
        for section_id in snapshot.section_ids:
            section = self.get_section(section_id)
            if section:
                sections.append(section)
            else:
                logger.warning(f"Section {section_id} in snapshot {snapshot_id} not found")

        return sections

    # =========================================================================
    # Statistics and Utilities
    # =========================================================================

    def get_stats(self) -> ColdStoreStats:
        """Get statistics about the Cold Store.

        Returns
        -------
        ColdStoreStats
            Statistics including counts and sizes.

        Examples
        --------
        ::

            stats = cold.get_stats()
            print(f"Sections: {stats.section_count}")
            print(f"Snapshots: {stats.snapshot_count}")
        """
        with self._connection() as conn:
            # Section count
            cursor = conn.execute("SELECT COUNT(*) as count FROM sections")
            section_count = cursor.fetchone()["count"]

            # Snapshot count
            cursor = conn.execute("SELECT COUNT(*) as count FROM snapshots")
            snapshot_count = cursor.fetchone()["count"]

            # Total content size
            cursor = conn.execute("SELECT COALESCE(SUM(LENGTH(content)), 0) as total FROM sections")
            total_bytes = cursor.fetchone()["total"]

            # Latest snapshot
            cursor = conn.execute(
                "SELECT id, created_at FROM snapshots ORDER BY created_at DESC LIMIT 1"
            )
            latest = cursor.fetchone()

            return ColdStoreStats(
                section_count=section_count,
                snapshot_count=snapshot_count,
                total_content_bytes=total_bytes,
                last_snapshot_id=latest["id"] if latest else None,
                last_snapshot_at=datetime.fromisoformat(latest["created_at"]) if latest else None,
            )

    def export_to_dict(self) -> dict[str, Any]:
        """Export Cold Store contents to a dictionary.

        Useful for debugging, backup, or serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary with sections and snapshots.

        Examples
        --------
        ::

            data = cold.export_to_dict()
            json.dump(data, open("backup.json", "w"), indent=2)
        """
        with self._connection() as conn:
            # Get all sections
            cursor = conn.execute("SELECT * FROM sections ORDER BY created_at")
            sections = []
            for row in cursor.fetchall():
                sections.append(
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "content_hash": row["content_hash"],
                        "metadata": json.loads(row["metadata"]),
                        "created_at": row["created_at"],
                        "source_artifact_id": row["source_artifact_id"],
                    }
                )

            # Get all snapshots
            cursor = conn.execute("SELECT * FROM snapshots ORDER BY created_at")
            snapshots = []
            for row in cursor.fetchall():
                snapshots.append(
                    {
                        "id": row["id"],
                        "created_at": row["created_at"],
                        "description": row["description"],
                        "section_ids": json.loads(row["section_ids"]),
                        "manifest_hash": row["manifest_hash"],
                    }
                )

            return {
                "schema_version": self.SCHEMA_VERSION,
                "sections": sections,
                "snapshots": snapshots,
            }


# =============================================================================
# Factory Function for Runtime Integration
# =============================================================================


def get_cold_store(project_path: str | Path | None = None) -> ColdStore:
    """Get or create a Cold Store for the current project.

    This is the primary entry point for runtime Cold Store access.
    It handles the "preload at start from file" requirement.

    Parameters
    ----------
    project_path : str | Path | None, optional
        Path to the project file. If None, uses default location.
        Defaults to None.

    Returns
    -------
    ColdStore
        Cold Store instance (loaded or newly created).

    Examples
    --------
    ::

        # At workflow start
        cold = get_cold_store("my_project.qfproj")

        # Or use default location
        cold = get_cold_store()
    """
    resolved_path = Path.cwd() / "project.qfproj" if project_path is None else Path(project_path)
    return ColdStore.load_or_create(resolved_path)
