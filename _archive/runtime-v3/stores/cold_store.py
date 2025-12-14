"""ColdStore - Persistent SQLite + files for all approved canon.

The ColdStore is the canonical source of truth for all approved content.
It contains both player-visible and internal content; visibility filtering
is handled at export time by Publisher.

Three-Tier Storage Model:
- hot_store: Working drafts, mutable, process artifacts (ephemeral)
- cold_store: All approved content, append-only (this module)
- Views/Exports: Filtered snapshots for different audiences

Characteristics:
- Lifetime: Permanent (project lifetime)
- Persistence: SQLite database + external asset files
- Contents: All approved, gatekeeper-validated content
- Visibility: Per-artifact visibility field controls export filtering

Components (per v2 spec):
- Book: Story structure metadata
- Acts/Chapters: Structural hierarchy
- Sections: Narrative prose (scenes)
- Codex: Player-safe encyclopedia (character, location, item, relationship)
         These are ALWAYS player-safe - no spoilers allowed.
- Canon: Internal world facts (canon_entry, event, fact, timeline)
         These CAN contain spoilers (spoiler_level: hot/cold).
- Assets: Binary files (images, audio, fonts)
- Snapshots: Point-in-time captures for deterministic builds

Content Routing:
- scene → sections table
- character, location, item, relationship → codex table (player-safe)
- canon_entry, event, fact, timeline → canon table (can have spoilers)
- act, chapter → acts, chapters tables (structural)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# TODO: This file needs v4 overhaul - see issue #142
# These imports are from v3 generated/ and need to be replaced with v4 domain loader
from questfoundry.generated.models.artifacts import Choice, ColdAct, ColdChapter, ColdSection, Gate
from questfoundry.generated.models.enums import Visibility

__all__ = [
    "AssetProvenance",
    "AssetType",
    "BookMetadata",
    "ColdAsset",
    "ColdSection",
    "ColdSnapshot",
    "ColdStore",
    "StoredArtifact",
    "get_cold_store",
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 5  # Added artifacts table for persistent work artifacts


# =============================================================================
# Enums
# =============================================================================


class AssetType(str, Enum):
    """Classification of external binary assets."""

    PLATE = "plate"  # Illustration for a section
    COVER = "cover"  # Book cover image
    ICON = "icon"  # Small graphic (character portrait, item icon)
    AUDIO = "audio"  # Sound file (ambient, music, SFX)
    FONT = "font"  # Custom typography file
    ORNAMENT = "ornament"  # Decorative element (divider, flourish)


# =============================================================================
# Data Models (from stores.md)
# =============================================================================


class BookMetadata(BaseModel):
    """Book metadata (singleton in cold_store)."""

    title: str = "Untitled"
    subtitle: str | None = None
    language: str = "en"  # ISO 639-1
    author: str | None = None
    start_anchor: str | None = None
    domain_version: int | None = None  # Domain spec version used to create this project


class AssetProvenance(BaseModel):
    """Creation metadata for reproducibility."""

    created_by: str  # Role that created it
    prompt: str | None = None  # Generation prompt (if AI-generated)
    seed: int | None = None  # Random seed (if applicable)
    model: str | None = None  # Model/tool used
    policy_notes: str | None = None  # Policy constraints applied


class ColdAsset(BaseModel):
    """External binary file with provenance tracking."""

    id: int
    anchor: str  # Section anchor or 'cover', 'logo'
    asset_type: AssetType
    filename: str  # Filename in assets directory
    file_hash: str  # SHA-256 hash of file contents
    file_size: int  # File size in bytes
    mime_type: str  # MIME type
    approved_by: str  # Role ID that approved
    approved_at: datetime
    provenance: AssetProvenance | None = None


class ColdSnapshot(BaseModel):
    """Point-in-time capture for deterministic builds."""

    id: int
    snapshot_id: str  # e.g., 'cold-2025-12-08-001'
    created_at: datetime
    description: str = ""
    manifest_hash: str  # SHA-256 of all section + asset hashes
    section_count: int
    asset_count: int


class ColdCodex(BaseModel):
    """Player-safe encyclopedia entry.

    Per v2 spec: "Codex entries are the player-facing encyclopedia containing
    world knowledge that is explicitly safe for players to know."

    Categories: character, location, item, relationship
    These NEVER contain spoilers - they're always player-safe.
    """

    id: int
    anchor: str  # Unique identifier (e.g., 'char_protagonist', 'loc_marketplace')
    category: str  # character, location, item, relationship
    title: str  # Display name
    content: str  # Description/information
    content_hash: str  # SHA-256 for integrity
    metadata: dict[str, Any] | None = None  # Category-specific fields as JSON
    visibility: Visibility = Visibility.PUBLIC
    created_at: datetime


class ColdCanon(BaseModel):
    """Internal world facts (may contain spoilers).

    Per v2 spec: "Canon can contain spoilers; never leaves Hot until
    Gatekeeper approves player-safe summaries."

    Categories: canon_entry, event, fact, timeline
    spoiler_level distinguishes internal (hot) from player-safe (cold) summaries.
    """

    id: int
    anchor: str  # Unique identifier (e.g., 'fact_magic_system', 'event_war')
    category: str  # canon_entry, event, fact, timeline
    title: str  # Display name
    content: str  # The canonical information
    content_hash: str  # SHA-256 for integrity
    spoiler_level: str = "hot"  # hot (internal only) or cold (player-safe)
    metadata: dict[str, Any] | None = None  # Category-specific fields as JSON
    visibility: Visibility = Visibility.INTERNAL
    created_at: datetime


class StoredArtifact(BaseModel):
    """Persistent work artifact (HookCard, Brief, GatecheckReport, etc.).

    This model wraps any work artifact for storage in the unified artifacts table.
    The actual artifact data is stored as a JSON blob, allowing Pydantic to handle
    schema evolution without database migrations.
    """

    id: int
    anchor: str  # Unique identifier (e.g., 'hook_123', 'brief_abc')
    artifact_type: str  # hook_card, brief, gatecheck_report, shotlist, etc.
    status: str  # Lifecycle status from the artifact's lifecycle
    data: dict[str, Any]  # JSON blob of the full Pydantic model
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _compute_manifest_hash(sections: list[tuple[str, str]], assets: list[tuple[str, str]]) -> str:
    """Compute manifest hash from section and asset hashes."""
    parts = []
    for anchor, hash_ in sorted(sections):
        parts.append(f"section:{anchor}:{hash_}")
    for filename, hash_ in sorted(assets):
        parts.append(f"asset:{filename}:{hash_}")
    manifest = "\n".join(parts)
    return _compute_hash(manifest)


def _now_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(UTC).isoformat()


# =============================================================================
# SQL Schema
# =============================================================================

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Book metadata (singleton row)
CREATE TABLE IF NOT EXISTS book_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    title TEXT NOT NULL DEFAULT 'Untitled',
    subtitle TEXT,
    language TEXT NOT NULL DEFAULT 'en',
    author TEXT,
    start_section_id INTEGER REFERENCES sections(id),
    domain_version INTEGER  -- Domain spec version used to create this project
);

-- Acts (structural organization)
CREATE TABLE IF NOT EXISTS acts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    description TEXT,
    visibility TEXT DEFAULT 'public',
    created_at TEXT NOT NULL
);

-- Chapters (content divisions within acts)
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    act_id INTEGER REFERENCES acts(id),
    title TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    summary TEXT,
    visibility TEXT DEFAULT 'public',
    created_at TEXT NOT NULL
);

-- Sections with auto-increment ID
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    chapter_id INTEGER REFERENCES chapters(id),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    order_num INTEGER NOT NULL UNIQUE,
    requires_gate INTEGER DEFAULT 0,
    source_brief_id TEXT,
    choices TEXT,  -- JSON array of choice strings for interactive fiction
    gates TEXT,    -- JSON array of gate condition strings
    visibility TEXT DEFAULT 'public',
    created_at TEXT NOT NULL
);

-- External assets with provenance
CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id INTEGER REFERENCES sections(id),
    anchor TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    filename TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    provenance TEXT
);

-- Codex entries (player-safe encyclopedia)
-- Categories: character, location, item, relationship
-- These are ALWAYS player-safe - no spoiler_level field
-- Per v2 spec: "Codex entries are the player-facing encyclopedia"
CREATE TABLE IF NOT EXISTS codex (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,  -- character, location, item, relationship
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata TEXT,  -- JSON for category-specific fields
    visibility TEXT DEFAULT 'public',
    created_at TEXT NOT NULL
);

-- Canon entries (internal world facts - can contain spoilers)
-- Categories: canon_entry, event, fact, timeline
-- Per v2 spec: "Canon can contain spoilers; never leaves Hot until Gatekeeper approves"
-- spoiler_level: hot (internal only) or cold (player-safe summary)
CREATE TABLE IF NOT EXISTS canon (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,  -- canon_entry, event, fact, timeline
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    spoiler_level TEXT DEFAULT 'hot',  -- hot (internal) or cold (player-safe)
    metadata TEXT,  -- JSON for category-specific fields (e.g., timeline refs, event dates)
    visibility TEXT DEFAULT 'internal',
    created_at TEXT NOT NULL
);

-- Unified artifacts table for persistent work artifacts
-- This replaces ephemeral hot_store for HookCard, Brief, GatecheckReport, etc.
-- Uses JSON blobs for data - Pydantic handles serialization/evolution
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,         -- Unique identifier (e.g., 'hook_123', 'brief_abc')
    artifact_type TEXT NOT NULL,         -- hook_card, brief, gatecheck_report, etc.
    status TEXT NOT NULL,                -- Lifecycle status (proposed, active, completed, etc.)
    data TEXT NOT NULL,                  -- JSON blob of Pydantic model
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Snapshots for deterministic builds
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    description TEXT,
    manifest_hash TEXT NOT NULL,
    section_count INTEGER NOT NULL,
    asset_count INTEGER NOT NULL
);

-- Snapshot membership (which sections/assets in each snapshot)
CREATE TABLE IF NOT EXISTS snapshot_sections (
    snapshot_id INTEGER REFERENCES snapshots(id),
    section_id INTEGER REFERENCES sections(id),
    PRIMARY KEY (snapshot_id, section_id)
);

CREATE TABLE IF NOT EXISTS snapshot_assets (
    snapshot_id INTEGER REFERENCES snapshots(id),
    asset_id INTEGER REFERENCES assets(id),
    PRIMARY KEY (snapshot_id, asset_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_acts_anchor ON acts(anchor);
CREATE INDEX IF NOT EXISTS idx_acts_sequence ON acts(sequence);
CREATE INDEX IF NOT EXISTS idx_chapters_anchor ON chapters(anchor);
CREATE INDEX IF NOT EXISTS idx_chapters_act ON chapters(act_id);
CREATE INDEX IF NOT EXISTS idx_sections_anchor ON sections(anchor);
CREATE INDEX IF NOT EXISTS idx_sections_order ON sections(order_num);
CREATE INDEX IF NOT EXISTS idx_sections_chapter ON sections(chapter_id);
CREATE INDEX IF NOT EXISTS idx_assets_anchor ON assets(anchor);
CREATE INDEX IF NOT EXISTS idx_codex_anchor ON codex(anchor);
CREATE INDEX IF NOT EXISTS idx_codex_category ON codex(category);
CREATE INDEX IF NOT EXISTS idx_canon_anchor ON canon(anchor);
CREATE INDEX IF NOT EXISTS idx_canon_category ON canon(category);
CREATE INDEX IF NOT EXISTS idx_canon_spoiler ON canon(spoiler_level);
CREATE INDEX IF NOT EXISTS idx_artifacts_anchor ON artifacts(anchor);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status);
CREATE INDEX IF NOT EXISTS idx_artifacts_type_status ON artifacts(artifact_type, status);
CREATE INDEX IF NOT EXISTS idx_snapshots_created ON snapshots(created_at);
"""


# =============================================================================
# ColdStore Class
# =============================================================================


class ColdStore:
    """Persistent SQLite + files storage for player-safe canon.

    A ColdStore manages a project directory containing:
    - project.qfdb: SQLite database
    - assets/images/: Visual assets
    - assets/audio/: Sound assets
    - assets/fonts/: Typography files
    - .qf/: Runtime working directory

    Examples
    --------
    Create a new project::

        cold = ColdStore.create("my_project")
        cold.add_section("intro", "Introduction", "Welcome to the story...")
        cold.create_snapshot("Initial content")
        cold.close()

    Load an existing project::

        cold = ColdStore.load("my_project")
        section = cold.get_section("intro")
        cold.close()
    """

    def __init__(self, project_root: Path):
        """Initialize ColdStore (use create() or load() instead)."""
        self.project_root = project_root
        self.db_path = project_root / "project.qfdb"
        self.assets_dir = project_root / "assets"
        self._conn: sqlite3.Connection | None = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @classmethod
    def create(cls, path: str | Path) -> ColdStore:
        """Create a new ColdStore project.

        Parameters
        ----------
        path : str | Path
            Path to project directory.

        Returns
        -------
        ColdStore
            New cold store instance.

        Raises
        ------
        FileExistsError
            If project already exists.
        """
        project_root = Path(path)
        db_path = project_root / "project.qfdb"

        if db_path.exists():
            raise FileExistsError(f"Project already exists: {db_path}")

        # Create directory structure
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / "assets" / "images").mkdir(parents=True, exist_ok=True)
        (project_root / "assets" / "audio").mkdir(parents=True, exist_ok=True)
        (project_root / "assets" / "fonts").mkdir(parents=True, exist_ok=True)
        (project_root / ".qf").mkdir(parents=True, exist_ok=True)

        store = cls(project_root)
        store._init_database()
        logger.info(f"Created ColdStore: {project_root}")
        return store

    @classmethod
    def load(cls, path: str | Path) -> ColdStore:
        """Load an existing ColdStore project.

        Parameters
        ----------
        path : str | Path
            Path to project directory.

        Returns
        -------
        ColdStore
            Loaded cold store instance.

        Raises
        ------
        FileNotFoundError
            If project doesn't exist.
        """
        project_root = Path(path)
        db_path = project_root / "project.qfdb"

        if not db_path.exists():
            raise FileNotFoundError(f"Project not found: {db_path}")

        store = cls(project_root)
        store._check_schema_version()
        logger.info(f"Loaded ColdStore: {project_root}")
        return store

    @classmethod
    def load_or_create(cls, path: str | Path) -> ColdStore:
        """Load existing project or create new one."""
        project_root = Path(path)
        db_path = project_root / "project.qfdb"

        if db_path.exists():
            return cls.load(path)
        return cls.create(path)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ColdStore:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # =========================================================================
    # Database Management
    # =========================================================================

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory.

        Note: check_same_thread=False allows the connection to be used from
        multiple threads (required since LangGraph may run tools in different
        threads). This is safe as writes are serialized by the orchestrator.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        yield self._conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            # Initialize book metadata singleton
            conn.execute("INSERT OR IGNORE INTO book_metadata (id) VALUES (1)")
            conn.commit()

    def _check_schema_version(self) -> None:
        """Check schema version compatibility and run migrations if needed."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            if row is None:
                raise ValueError("Invalid database: missing schema version")
            db_version = row["version"]
            if db_version > SCHEMA_VERSION:
                raise ValueError(
                    f"Database schema v{db_version} is newer than supported v{SCHEMA_VERSION}"
                )
            # Run migrations for older databases
            if db_version < SCHEMA_VERSION:
                self._migrate_schema(conn, db_version)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations from older versions."""
        if from_version < 3:
            # v2 → v3: Add domain_version column to book_metadata
            logger.info("Migrating database schema from v2 to v3...")
            conn.execute("ALTER TABLE book_metadata ADD COLUMN domain_version INTEGER")
            conn.execute("UPDATE schema_version SET version = 3")
            conn.commit()
            logger.info("Database migrated to schema v3")
            from_version = 3

        if from_version < 4:
            # v3 → v4: Add codex and canon tables (already handled by SCHEMA_SQL)
            logger.info("Migrating database schema from v3 to v4...")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS codex (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anchor TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata TEXT,
                    visibility TEXT DEFAULT 'public',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS canon (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anchor TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    spoiler_level TEXT DEFAULT 'hot',
                    metadata TEXT,
                    visibility TEXT DEFAULT 'internal',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_codex_anchor ON codex(anchor);
                CREATE INDEX IF NOT EXISTS idx_codex_category ON codex(category);
                CREATE INDEX IF NOT EXISTS idx_canon_anchor ON canon(anchor);
                CREATE INDEX IF NOT EXISTS idx_canon_category ON canon(category);
                CREATE INDEX IF NOT EXISTS idx_canon_spoiler ON canon(spoiler_level);
                """
            )
            conn.execute("UPDATE schema_version SET version = 4")
            conn.commit()
            logger.info("Database migrated to schema v4")
            from_version = 4

        if from_version < 5:
            # v4 → v5: Add artifacts table for persistent work artifacts
            logger.info("Migrating database schema from v4 to v5...")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anchor TEXT NOT NULL UNIQUE,
                    artifact_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_anchor ON artifacts(anchor);
                CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
                CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status);
                CREATE INDEX IF NOT EXISTS idx_artifacts_type_status ON artifacts(artifact_type, status);
                """
            )
            conn.execute("UPDATE schema_version SET version = 5")
            conn.commit()
            logger.info("Database migrated to schema v5")

    # =========================================================================
    # Book Metadata
    # =========================================================================

    def get_book_metadata(self) -> BookMetadata:
        """Get book metadata."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM book_metadata WHERE id = 1")
            row = cursor.fetchone()
            if row is None:
                return BookMetadata()

            # Get start_anchor from section if set
            start_anchor = None
            if row["start_section_id"]:
                section_cursor = conn.execute(
                    "SELECT anchor FROM sections WHERE id = ?",
                    (row["start_section_id"],),
                )
                section_row = section_cursor.fetchone()
                if section_row:
                    start_anchor = section_row["anchor"]

            return BookMetadata(
                title=row["title"],
                subtitle=row["subtitle"],
                language=row["language"],
                author=row["author"],
                start_anchor=start_anchor,
                domain_version=row["domain_version"] if "domain_version" in row else None,  # noqa: SIM401
            )

    def set_book_metadata(self, metadata: BookMetadata) -> None:
        """Set book metadata."""
        with self._connection() as conn:
            # Resolve start_anchor to section_id
            start_section_id = None
            if metadata.start_anchor:
                cursor = conn.execute(
                    "SELECT id FROM sections WHERE anchor = ?",
                    (metadata.start_anchor,),
                )
                row = cursor.fetchone()
                if row:
                    start_section_id = row["id"]

            conn.execute(
                """
                UPDATE book_metadata SET
                    title = ?, subtitle = ?, language = ?,
                    author = ?, start_section_id = ?, domain_version = ?
                WHERE id = 1
                """,
                (
                    metadata.title,
                    metadata.subtitle,
                    metadata.language,
                    metadata.author,
                    start_section_id,
                    metadata.domain_version,
                ),
            )
            conn.commit()

    def get_domain_version(self) -> int | None:
        """Get the domain version used to create this project.

        Returns
        -------
        int | None
            The domain version, or None if not set.
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT domain_version FROM book_metadata WHERE id = 1")
            row = cursor.fetchone()
            if row is None:
                return None
            result: int | None = row["domain_version"]
            return result

    def set_domain_version(self, version: int) -> None:
        """Set the domain version for this project.

        Parameters
        ----------
        version : int
            The domain spec version.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE book_metadata SET domain_version = ? WHERE id = 1",
                (version,),
            )
            conn.commit()

    # =========================================================================
    # Section Operations
    # =========================================================================

    def add_section(
        self,
        anchor: str,
        title: str,
        content: str,
        *,
        order: int | None = None,
        chapter_anchor: str | None = None,
        requires_gate: bool = False,
        source_brief_id: str | None = None,
        choices: list[Choice] | None = None,
        gates: list[Gate] | None = None,
        visibility: Visibility = Visibility.PUBLIC,
    ) -> ColdSection:
        """Add or update a section.

        Parameters
        ----------
        anchor : str
            Unique identifier for navigation.
        title : str
            Section title.
        content : str
            Prose content.
        order : int | None
            Display order (auto-assigned if None).
        chapter_anchor : str | None
            Anchor of the parent chapter.
        requires_gate : bool
            Whether this section has access conditions.
        source_brief_id : str | None
            ID of the Brief that produced this section.
        choices : list[Choice] | None
            Available choices/exits for interactive fiction.
        gates : list[Gate] | None
            Gate conditions that control access.
        visibility : Visibility
            Export visibility (default: public).

        Returns
        -------
        ColdSection
            The created/updated section.
        """
        content_hash = _compute_hash(content)
        created_at = _now_iso()

        # Serialize choices and gates as JSON (using model_dump for Pydantic objects)
        choices_json = json.dumps([c.model_dump() for c in choices]) if choices else None
        gates_json = json.dumps([g.model_dump() for g in gates]) if gates else None

        with self._connection() as conn:
            # Resolve chapter_anchor to chapter_id
            chapter_id = None
            if chapter_anchor:
                cursor = conn.execute("SELECT id FROM chapters WHERE anchor = ?", (chapter_anchor,))
                row = cursor.fetchone()
                if row:
                    chapter_id = row["id"]

            # Check if anchor exists
            cursor = conn.execute("SELECT id FROM sections WHERE anchor = ?", (anchor,))
            existing = cursor.fetchone()

            if existing:
                # Update existing section
                conn.execute(
                    """
                    UPDATE sections SET
                        chapter_id = ?, title = ?, content = ?, content_hash = ?,
                        requires_gate = ?, source_brief_id = ?,
                        choices = ?, gates = ?, visibility = ?
                    WHERE anchor = ?
                    """,
                    (
                        chapter_id,
                        title,
                        content,
                        content_hash,
                        requires_gate,
                        source_brief_id,
                        choices_json,
                        gates_json,
                        visibility.value,
                        anchor,
                    ),
                )
                section_id = existing["id"]
                # Get current order
                cursor = conn.execute("SELECT order_num FROM sections WHERE id = ?", (section_id,))
                order = cursor.fetchone()["order_num"]
            else:
                # Auto-assign order if not provided
                if order is None:
                    cursor = conn.execute(
                        "SELECT COALESCE(MAX(order_num), 0) + 1 as next_order FROM sections"
                    )
                    order = cursor.fetchone()["next_order"]

                # Insert new section
                cursor = conn.execute(
                    """
                    INSERT INTO sections
                    (anchor, chapter_id, title, content, content_hash, order_num, requires_gate, source_brief_id, choices, gates, visibility, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        anchor,
                        chapter_id,
                        title,
                        content,
                        content_hash,
                        order,
                        requires_gate,
                        source_brief_id,
                        choices_json,
                        gates_json,
                        visibility.value,
                        created_at,
                    ),
                )
                section_id = cursor.lastrowid

            conn.commit()

        return ColdSection(
            id=section_id,
            anchor=anchor,
            chapter_id=chapter_id,
            title=title,
            content=content,
            content_hash=content_hash,
            order=order,
            requires_gate=requires_gate,
            source_brief_id=source_brief_id,
            choices=choices,
            gates=gates,
            visibility=visibility,
            created_at=datetime.fromisoformat(created_at),
        )

    def get_section(self, anchor: str) -> ColdSection | None:
        """Get a section by anchor."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM sections WHERE anchor = ?", (anchor,))
            row = cursor.fetchone()
            if row is None:
                return None

            # Parse JSON fields into Choice/Gate objects
            choices_data = json.loads(row["choices"]) if row["choices"] else None
            gates_data = json.loads(row["gates"]) if row["gates"] else None
            choices = [Choice(**c) for c in choices_data] if choices_data else None
            gates = [Gate(**g) for g in gates_data] if gates_data else None

            return ColdSection(
                id=row["id"],
                anchor=row["anchor"],
                chapter_id=row["chapter_id"],
                title=row["title"],
                content=row["content"],
                content_hash=row["content_hash"],
                order=row["order_num"],
                requires_gate=bool(row["requires_gate"]),
                source_brief_id=row["source_brief_id"],
                choices=choices,
                gates=gates,
                visibility=Visibility(row["visibility"])
                if row["visibility"]
                else Visibility.PUBLIC,
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def list_sections(self) -> list[str]:
        """List all section anchors in display order."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT anchor FROM sections ORDER BY order_num")
            return [row["anchor"] for row in cursor.fetchall()]

    def get_all_sections(self) -> list[ColdSection]:
        """Get all sections in display order."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM sections ORDER BY order_num")
            sections = []
            for row in cursor.fetchall():
                # Parse JSON fields into Choice/Gate objects
                choices_data = json.loads(row["choices"]) if row["choices"] else None
                gates_data = json.loads(row["gates"]) if row["gates"] else None
                choices = [Choice(**c) for c in choices_data] if choices_data else None
                gates = [Gate(**g) for g in gates_data] if gates_data else None
                sections.append(
                    ColdSection(
                        id=row["id"],
                        anchor=row["anchor"],
                        chapter_id=row["chapter_id"],
                        title=row["title"],
                        content=row["content"],
                        content_hash=row["content_hash"],
                        order=row["order_num"],
                        requires_gate=bool(row["requires_gate"]),
                        source_brief_id=row["source_brief_id"],
                        choices=choices,
                        gates=gates,
                        visibility=Visibility(row["visibility"])
                        if row["visibility"]
                        else Visibility.PUBLIC,
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
            return sections

    def delete_section(self, anchor: str) -> bool:
        """Delete a section (fails if in a snapshot)."""
        with self._connection() as conn:
            # Check if in any snapshot
            cursor = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM snapshot_sections ss
                JOIN sections s ON ss.section_id = s.id
                WHERE s.anchor = ?
                """,
                (anchor,),
            )
            if cursor.fetchone()["cnt"] > 0:
                logger.warning(f"Cannot delete section '{anchor}': in snapshot")
                return False

            cursor = conn.execute("DELETE FROM sections WHERE anchor = ?", (anchor,))
            conn.commit()
            return cursor.rowcount > 0

    def rename_anchor(self, old_anchor: str, new_anchor: str) -> bool:
        """Rename a section's anchor (safe operation)."""
        with self._connection() as conn:
            try:
                conn.execute(
                    "UPDATE sections SET anchor = ? WHERE anchor = ?",
                    (new_anchor, old_anchor),
                )
                # Update assets referencing this anchor
                conn.execute(
                    "UPDATE assets SET anchor = ? WHERE anchor = ?",
                    (new_anchor, old_anchor),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    # =========================================================================
    # Codex Operations (player-safe encyclopedia)
    # Categories: character, location, item, relationship
    # Per v2 spec: "Codex entries are the player-facing encyclopedia"
    # =========================================================================

    def add_codex(
        self,
        anchor: str,
        category: str,
        title: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        visibility: Visibility = Visibility.PUBLIC,
    ) -> ColdCodex:
        """Add or update a codex entry.

        Parameters
        ----------
        anchor : str
            Unique identifier (e.g., 'char_protagonist', 'loc_marketplace').
        category : str
            Type of entry: character, location, item, relationship.
        title : str
            Display name.
        content : str
            Description/information.
        metadata : dict | None
            Category-specific fields as JSON.
        visibility : Visibility
            Export visibility (default: public).

        Returns
        -------
        ColdCodex
            The created/updated codex entry.
        """
        content_hash = _compute_hash(content)
        created_at = _now_iso()
        metadata_json = json.dumps(metadata) if metadata else None

        with self._connection() as conn:
            # Check if anchor exists
            cursor = conn.execute("SELECT id FROM codex WHERE anchor = ?", (anchor,))
            existing = cursor.fetchone()

            if existing:
                # Update existing entry
                conn.execute(
                    """
                    UPDATE codex SET
                        category = ?, title = ?, content = ?, content_hash = ?,
                        metadata = ?, visibility = ?
                    WHERE anchor = ?
                    """,
                    (
                        category,
                        title,
                        content,
                        content_hash,
                        metadata_json,
                        visibility.value,
                        anchor,
                    ),
                )
                codex_id = existing["id"]
            else:
                # Insert new entry
                cursor = conn.execute(
                    """
                    INSERT INTO codex
                    (anchor, category, title, content, content_hash, metadata, visibility, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        anchor,
                        category,
                        title,
                        content,
                        content_hash,
                        metadata_json,
                        visibility.value,
                        created_at,
                    ),
                )
                codex_id = cursor.lastrowid

            conn.commit()

        return ColdCodex(
            id=codex_id,
            anchor=anchor,
            category=category,
            title=title,
            content=content,
            content_hash=content_hash,
            metadata=metadata,
            visibility=visibility,
            created_at=datetime.fromisoformat(created_at),
        )

    def get_codex(self, anchor: str) -> ColdCodex | None:
        """Get a codex entry by anchor."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM codex WHERE anchor = ?", (anchor,))
            row = cursor.fetchone()
            if row is None:
                return None

            metadata = json.loads(row["metadata"]) if row["metadata"] else None

            return ColdCodex(
                id=row["id"],
                anchor=row["anchor"],
                category=row["category"],
                title=row["title"],
                content=row["content"],
                content_hash=row["content_hash"],
                metadata=metadata,
                visibility=Visibility(row["visibility"])
                if row["visibility"]
                else Visibility.PUBLIC,
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def list_codex(self, category: str | None = None) -> list[str]:
        """List codex entry anchors, optionally filtered by category."""
        with self._connection() as conn:
            if category:
                cursor = conn.execute(
                    "SELECT anchor FROM codex WHERE category = ? ORDER BY title",
                    (category,),
                )
            else:
                cursor = conn.execute("SELECT anchor FROM codex ORDER BY category, title")
            return [row["anchor"] for row in cursor.fetchall()]

    # =========================================================================
    # Canon Operations (internal world facts - can contain spoilers)
    # Categories: canon_entry, event, fact, timeline
    # Per v2 spec: "Canon can contain spoilers; never leaves Hot until approved"
    # =========================================================================

    def add_canon(
        self,
        anchor: str,
        category: str,
        title: str,
        content: str,
        *,
        spoiler_level: str = "hot",
        metadata: dict[str, Any] | None = None,
        visibility: Visibility = Visibility.INTERNAL,
    ) -> ColdCanon:
        """Add or update a canon entry.

        Parameters
        ----------
        anchor : str
            Unique identifier (e.g., 'fact_magic_system', 'event_war').
        category : str
            Type of entry: canon_entry, event, fact, timeline.
        title : str
            Display name.
        content : str
            The canonical information.
        spoiler_level : str
            'hot' (internal only) or 'cold' (player-safe summary).
        metadata : dict | None
            Category-specific fields as JSON.
        visibility : Visibility
            Export visibility (default: internal).

        Returns
        -------
        ColdCanon
            The created/updated canon entry.
        """
        content_hash = _compute_hash(content)
        created_at = _now_iso()
        metadata_json = json.dumps(metadata) if metadata else None

        with self._connection() as conn:
            # Check if anchor exists
            cursor = conn.execute("SELECT id FROM canon WHERE anchor = ?", (anchor,))
            existing = cursor.fetchone()

            if existing:
                # Update existing entry
                conn.execute(
                    """
                    UPDATE canon SET
                        category = ?, title = ?, content = ?, content_hash = ?,
                        spoiler_level = ?, metadata = ?, visibility = ?
                    WHERE anchor = ?
                    """,
                    (
                        category,
                        title,
                        content,
                        content_hash,
                        spoiler_level,
                        metadata_json,
                        visibility.value,
                        anchor,
                    ),
                )
                canon_id = existing["id"]
            else:
                # Insert new entry
                cursor = conn.execute(
                    """
                    INSERT INTO canon
                    (anchor, category, title, content, content_hash, spoiler_level, metadata, visibility, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        anchor,
                        category,
                        title,
                        content,
                        content_hash,
                        spoiler_level,
                        metadata_json,
                        visibility.value,
                        created_at,
                    ),
                )
                canon_id = cursor.lastrowid

            conn.commit()

        return ColdCanon(
            id=canon_id,
            anchor=anchor,
            category=category,
            title=title,
            content=content,
            content_hash=content_hash,
            spoiler_level=spoiler_level,
            metadata=metadata,
            visibility=visibility,
            created_at=datetime.fromisoformat(created_at),
        )

    def get_canon(self, anchor: str) -> ColdCanon | None:
        """Get a canon entry by anchor."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM canon WHERE anchor = ?", (anchor,))
            row = cursor.fetchone()
            if row is None:
                return None

            metadata = json.loads(row["metadata"]) if row["metadata"] else None

            return ColdCanon(
                id=row["id"],
                anchor=row["anchor"],
                category=row["category"],
                title=row["title"],
                content=row["content"],
                content_hash=row["content_hash"],
                spoiler_level=row["spoiler_level"],
                metadata=metadata,
                visibility=Visibility(row["visibility"])
                if row["visibility"]
                else Visibility.INTERNAL,
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def list_canon(
        self, category: str | None = None, spoiler_level: str | None = None
    ) -> list[str]:
        """List canon entry anchors, optionally filtered by category and/or spoiler level."""
        with self._connection() as conn:
            conditions = []
            params = []
            if category:
                conditions.append("category = ?")
                params.append(category)
            if spoiler_level:
                conditions.append("spoiler_level = ?")
                params.append(spoiler_level)

            query = "SELECT anchor FROM canon"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY category, title"

            cursor = conn.execute(query, params)
            return [row["anchor"] for row in cursor.fetchall()]

    # =========================================================================
    # Artifact Operations (persistent work artifacts)
    # Types: hook_card, brief, gatecheck_report, shotlist, audio_plan, etc.
    # =========================================================================

    def save_artifact(
        self,
        anchor: str,
        artifact_type: str,
        status: str,
        data: dict[str, Any],
    ) -> StoredArtifact:
        """Save or update a work artifact.

        Parameters
        ----------
        anchor : str
            Unique identifier (e.g., 'hook_123', 'brief_abc').
        artifact_type : str
            Type of artifact: hook_card, brief, gatecheck_report, etc.
        status : str
            Lifecycle status (proposed, active, completed, etc.).
        data : dict
            Full artifact data as a dict (from Pydantic model_dump()).

        Returns
        -------
        StoredArtifact
            The saved artifact wrapper.
        """
        now = _now_iso()
        data_json = json.dumps(data, default=str)

        with self._connection() as conn:
            # Check if anchor exists
            cursor = conn.execute(
                "SELECT id, created_at FROM artifacts WHERE anchor = ?", (anchor,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing artifact
                conn.execute(
                    """
                    UPDATE artifacts SET
                        artifact_type = ?, status = ?, data = ?, updated_at = ?
                    WHERE anchor = ?
                    """,
                    (artifact_type, status, data_json, now, anchor),
                )
                artifact_id = existing["id"]
                created_at = existing["created_at"]
            else:
                # Insert new artifact
                cursor = conn.execute(
                    """
                    INSERT INTO artifacts
                    (anchor, artifact_type, status, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (anchor, artifact_type, status, data_json, now, now),
                )
                artifact_id = cursor.lastrowid
                created_at = now

            conn.commit()

        return StoredArtifact(
            id=artifact_id,
            anchor=anchor,
            artifact_type=artifact_type,
            status=status,
            data=data,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(now),
        )

    def get_artifact(self, anchor: str) -> StoredArtifact | None:
        """Get an artifact by anchor.

        Parameters
        ----------
        anchor : str
            The artifact's unique identifier.

        Returns
        -------
        StoredArtifact | None
            The artifact wrapper, or None if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM artifacts WHERE anchor = ?", (anchor,))
            row = cursor.fetchone()
            if row is None:
                return None

            return StoredArtifact(
                id=row["id"],
                anchor=row["anchor"],
                artifact_type=row["artifact_type"],
                status=row["status"],
                data=json.loads(row["data"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def delete_artifact(self, anchor: str) -> bool:
        """Delete an artifact.

        Parameters
        ----------
        anchor : str
            The artifact's unique identifier.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM artifacts WHERE anchor = ?", (anchor,))
            conn.commit()
            return cursor.rowcount > 0

    def list_artifacts(
        self,
        artifact_type: str | None = None,
        status: str | None = None,
    ) -> list[str]:
        """List artifact anchors, optionally filtered.

        Parameters
        ----------
        artifact_type : str | None
            Filter by artifact type (e.g., 'hook_card').
        status : str | None
            Filter by status (e.g., 'proposed').

        Returns
        -------
        list[str]
            List of matching artifact anchors.
        """
        with self._connection() as conn:
            conditions = []
            params: list[str] = []
            if artifact_type:
                conditions.append("artifact_type = ?")
                params.append(artifact_type)
            if status:
                conditions.append("status = ?")
                params.append(status)

            query = "SELECT anchor FROM artifacts"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY updated_at DESC"

            cursor = conn.execute(query, params)
            return [row["anchor"] for row in cursor.fetchall()]

    def get_all_artifacts(
        self,
        artifact_type: str | None = None,
        status: str | None = None,
    ) -> list[StoredArtifact]:
        """Get all artifacts, optionally filtered.

        Parameters
        ----------
        artifact_type : str | None
            Filter by artifact type (e.g., 'hook_card').
        status : str | None
            Filter by status (e.g., 'proposed').

        Returns
        -------
        list[StoredArtifact]
            List of matching artifacts.
        """
        with self._connection() as conn:
            conditions = []
            params: list[str] = []
            if artifact_type:
                conditions.append("artifact_type = ?")
                params.append(artifact_type)
            if status:
                conditions.append("status = ?")
                params.append(status)

            query = "SELECT * FROM artifacts"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY updated_at DESC"

            cursor = conn.execute(query, params)
            return [
                StoredArtifact(
                    id=row["id"],
                    anchor=row["anchor"],
                    artifact_type=row["artifact_type"],
                    status=row["status"],
                    data=json.loads(row["data"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in cursor.fetchall()
            ]

    def update_artifact_status(self, anchor: str, status: str) -> bool:
        """Update an artifact's status.

        Parameters
        ----------
        anchor : str
            The artifact's unique identifier.
        status : str
            New status value.

        Returns
        -------
        bool
            True if updated, False if not found.
        """
        now = _now_iso()
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE artifacts SET status = ?, updated_at = ? WHERE anchor = ?",
                (status, now, anchor),
            )
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Act Operations
    # =========================================================================

    def add_act(
        self,
        anchor: str,
        title: str,
        sequence: int,
        *,
        description: str | None = None,
        visibility: Visibility = Visibility.PUBLIC,
    ) -> ColdAct:
        """Add or update an act.

        Parameters
        ----------
        anchor : str
            Unique identifier for the act.
        title : str
            Act title.
        sequence : int
            Order within the story.
        description : str | None
            Summary of the act's narrative purpose.
        visibility : Visibility
            Export visibility (default: public).

        Returns
        -------
        ColdAct
            The created/updated act.
        """
        created_at = _now_iso()

        with self._connection() as conn:
            # Check if anchor exists
            cursor = conn.execute("SELECT id FROM acts WHERE anchor = ?", (anchor,))
            existing = cursor.fetchone()

            if existing:
                # Update existing act
                conn.execute(
                    """
                    UPDATE acts SET
                        title = ?, sequence = ?, description = ?, visibility = ?
                    WHERE anchor = ?
                    """,
                    (title, sequence, description, visibility.value, anchor),
                )
                act_id = existing["id"]
            else:
                # Insert new act
                cursor = conn.execute(
                    """
                    INSERT INTO acts
                    (anchor, title, sequence, description, visibility, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (anchor, title, sequence, description, visibility.value, created_at),
                )
                act_id = cursor.lastrowid

            conn.commit()

        return ColdAct(
            id=act_id,
            anchor=anchor,
            title=title,
            sequence=sequence,
            description=description,
            visibility=visibility,
        )

    def get_act(self, anchor: str) -> ColdAct | None:
        """Get an act by anchor."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM acts WHERE anchor = ?", (anchor,))
            row = cursor.fetchone()
            if row is None:
                return None

            return ColdAct(
                id=row["id"],
                anchor=row["anchor"],
                title=row["title"],
                sequence=row["sequence"],
                description=row["description"],
                visibility=Visibility(row["visibility"])
                if row["visibility"]
                else Visibility.PUBLIC,
            )

    def list_acts(self) -> list[ColdAct]:
        """List all acts in sequence order."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM acts ORDER BY sequence")
            return [
                ColdAct(
                    id=row["id"],
                    anchor=row["anchor"],
                    title=row["title"],
                    sequence=row["sequence"],
                    description=row["description"],
                    visibility=Visibility(row["visibility"])
                    if row["visibility"]
                    else Visibility.PUBLIC,
                )
                for row in cursor.fetchall()
            ]

    # =========================================================================
    # Chapter Operations
    # =========================================================================

    def add_chapter(
        self,
        anchor: str,
        title: str,
        sequence: int,
        *,
        act_anchor: str | None = None,
        summary: str | None = None,
        visibility: Visibility = Visibility.PUBLIC,
    ) -> ColdChapter:
        """Add or update a chapter.

        Parameters
        ----------
        anchor : str
            Unique identifier for the chapter.
        title : str
            Chapter title.
        sequence : int
            Order within the act.
        act_anchor : str | None
            Anchor of the parent act.
        summary : str | None
            Chapter summary.
        visibility : Visibility
            Export visibility (default: public).

        Returns
        -------
        ColdChapter
            The created/updated chapter.
        """
        created_at = _now_iso()

        with self._connection() as conn:
            # Resolve act_anchor to act_id
            act_id = None
            if act_anchor:
                cursor = conn.execute("SELECT id FROM acts WHERE anchor = ?", (act_anchor,))
                row = cursor.fetchone()
                if row:
                    act_id = row["id"]

            # Check if anchor exists
            cursor = conn.execute("SELECT id FROM chapters WHERE anchor = ?", (anchor,))
            existing = cursor.fetchone()

            if existing:
                # Update existing chapter
                conn.execute(
                    """
                    UPDATE chapters SET
                        act_id = ?, title = ?, sequence = ?, summary = ?, visibility = ?
                    WHERE anchor = ?
                    """,
                    (act_id, title, sequence, summary, visibility.value, anchor),
                )
                chapter_id = existing["id"]
            else:
                # Insert new chapter
                cursor = conn.execute(
                    """
                    INSERT INTO chapters
                    (anchor, act_id, title, sequence, summary, visibility, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (anchor, act_id, title, sequence, summary, visibility.value, created_at),
                )
                chapter_id = cursor.lastrowid

            conn.commit()

        return ColdChapter(
            id=chapter_id,
            anchor=anchor,
            title=title,
            sequence=sequence,
            act_id=act_id,
            summary=summary,
            visibility=visibility,
        )

    def get_chapter(self, anchor: str) -> ColdChapter | None:
        """Get a chapter by anchor."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM chapters WHERE anchor = ?", (anchor,))
            row = cursor.fetchone()
            if row is None:
                return None

            return ColdChapter(
                id=row["id"],
                anchor=row["anchor"],
                title=row["title"],
                sequence=row["sequence"],
                act_id=row["act_id"],
                summary=row["summary"],
                visibility=Visibility(row["visibility"])
                if row["visibility"]
                else Visibility.PUBLIC,
            )

    def list_chapters(self, act_anchor: str | None = None) -> list[ColdChapter]:
        """List chapters, optionally filtered by act."""
        with self._connection() as conn:
            if act_anchor:
                cursor = conn.execute(
                    """
                    SELECT c.* FROM chapters c
                    JOIN acts a ON c.act_id = a.id
                    WHERE a.anchor = ?
                    ORDER BY c.sequence
                    """,
                    (act_anchor,),
                )
            else:
                cursor = conn.execute("SELECT * FROM chapters ORDER BY sequence")

            return [
                ColdChapter(
                    id=row["id"],
                    anchor=row["anchor"],
                    title=row["title"],
                    sequence=row["sequence"],
                    act_id=row["act_id"],
                    summary=row["summary"],
                    visibility=Visibility(row["visibility"])
                    if row["visibility"]
                    else Visibility.PUBLIC,
                )
                for row in cursor.fetchall()
            ]

    # =========================================================================
    # Asset Operations
    # =========================================================================

    def add_asset(
        self,
        anchor: str,
        asset_type: AssetType,
        filename: str,
        file_path: Path,
        *,
        approved_by: str,
        mime_type: str | None = None,
        provenance: AssetProvenance | None = None,
    ) -> ColdAsset:
        """Add an asset from a file.

        Parameters
        ----------
        anchor : str
            Section anchor or 'cover', 'logo'.
        asset_type : AssetType
            Type of asset.
        filename : str
            Filename to use in assets directory.
        file_path : Path
            Source file to copy.
        approved_by : str
            Role ID that approved this asset.
        mime_type : str | None
            MIME type (auto-detected if None).
        provenance : AssetProvenance | None
            Creation metadata.

        Returns
        -------
        ColdAsset
            The added asset.
        """
        # Compute hash and size
        file_hash = _compute_file_hash(file_path)
        file_size = file_path.stat().st_size

        # Auto-detect MIME type
        if mime_type is None:
            suffix = file_path.suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".woff": "font/woff",
                ".woff2": "font/woff2",
                ".ttf": "font/ttf",
                ".otf": "font/otf",
            }
            mime_type = mime_map.get(suffix, "application/octet-stream")

        # Determine destination directory
        if asset_type in (AssetType.PLATE, AssetType.COVER, AssetType.ICON, AssetType.ORNAMENT):
            dest_dir = self.assets_dir / "images"
        elif asset_type == AssetType.AUDIO:
            dest_dir = self.assets_dir / "audio"
        elif asset_type == AssetType.FONT:
            dest_dir = self.assets_dir / "fonts"
        else:
            dest_dir = self.assets_dir

        # Copy file
        dest_path = dest_dir / filename
        import shutil

        shutil.copy2(file_path, dest_path)

        approved_at = _now_iso()

        with self._connection() as conn:
            # Get section_id if anchor refers to a section
            section_id = None
            if anchor not in ("cover", "logo"):
                cursor = conn.execute("SELECT id FROM sections WHERE anchor = ?", (anchor,))
                row = cursor.fetchone()
                if row:
                    section_id = row["id"]

            # Serialize provenance
            provenance_json = None
            if provenance:
                provenance_json = provenance.model_dump_json()

            cursor = conn.execute(
                """
                INSERT INTO assets
                (section_id, anchor, asset_type, filename, file_hash, file_size, mime_type, approved_by, approved_at, provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section_id,
                    anchor,
                    asset_type.value,
                    filename,
                    file_hash,
                    file_size,
                    mime_type,
                    approved_by,
                    approved_at,
                    provenance_json,
                ),
            )
            asset_id = cursor.lastrowid
            conn.commit()

        return ColdAsset(
            id=asset_id,
            anchor=anchor,
            asset_type=asset_type,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            mime_type=mime_type,
            approved_by=approved_by,
            approved_at=datetime.fromisoformat(approved_at),
            provenance=provenance,
        )

    def get_asset(self, filename: str) -> ColdAsset | None:
        """Get an asset by filename."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM assets WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            if row is None:
                return None

            provenance = None
            if row["provenance"]:
                provenance = AssetProvenance(**json.loads(row["provenance"]))

            return ColdAsset(
                id=row["id"],
                anchor=row["anchor"],
                asset_type=AssetType(row["asset_type"]),
                filename=row["filename"],
                file_hash=row["file_hash"],
                file_size=row["file_size"],
                mime_type=row["mime_type"],
                approved_by=row["approved_by"],
                approved_at=datetime.fromisoformat(row["approved_at"]),
                provenance=provenance,
            )

    def get_assets_for_anchor(self, anchor: str) -> list[ColdAsset]:
        """Get all assets for a section anchor."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM assets WHERE anchor = ? ORDER BY asset_type",
                (anchor,),
            )
            assets = []
            for row in cursor.fetchall():
                provenance = None
                if row["provenance"]:
                    provenance = AssetProvenance(**json.loads(row["provenance"]))
                assets.append(
                    ColdAsset(
                        id=row["id"],
                        anchor=row["anchor"],
                        asset_type=AssetType(row["asset_type"]),
                        filename=row["filename"],
                        file_hash=row["file_hash"],
                        file_size=row["file_size"],
                        mime_type=row["mime_type"],
                        approved_by=row["approved_by"],
                        approved_at=datetime.fromisoformat(row["approved_at"]),
                        provenance=provenance,
                    )
                )
            return assets

    def list_assets(self) -> list[ColdAsset]:
        """List all assets."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM assets ORDER BY anchor, asset_type")
            assets = []
            for row in cursor.fetchall():
                provenance = None
                if row["provenance"]:
                    provenance = AssetProvenance(**json.loads(row["provenance"]))
                assets.append(
                    ColdAsset(
                        id=row["id"],
                        anchor=row["anchor"],
                        asset_type=AssetType(row["asset_type"]),
                        filename=row["filename"],
                        file_hash=row["file_hash"],
                        file_size=row["file_size"],
                        mime_type=row["mime_type"],
                        approved_by=row["approved_by"],
                        approved_at=datetime.fromisoformat(row["approved_at"]),
                        provenance=provenance,
                    )
                )
            return assets

    def get_asset_path(self, filename: str) -> Path | None:
        """Get the full path to an asset file."""
        asset = self.get_asset(filename)
        if asset is None:
            return None

        if asset.asset_type in (
            AssetType.PLATE,
            AssetType.COVER,
            AssetType.ICON,
            AssetType.ORNAMENT,
        ):
            return self.assets_dir / "images" / filename
        elif asset.asset_type == AssetType.AUDIO:
            return self.assets_dir / "audio" / filename
        elif asset.asset_type == AssetType.FONT:
            return self.assets_dir / "fonts" / filename
        return self.assets_dir / filename

    # =========================================================================
    # Snapshot Operations
    # =========================================================================

    def create_snapshot(self, description: str = "") -> str:
        """Create a snapshot of the current state.

        Returns
        -------
        str
            The snapshot_id (e.g., 'cold-2025-12-08-001').
        """
        with self._connection() as conn:
            # Get all sections
            cursor = conn.execute("SELECT id, anchor, content_hash FROM sections ORDER BY anchor")
            sections = [(row["anchor"], row["content_hash"]) for row in cursor.fetchall()]
            section_ids = [row["id"] for row in conn.execute("SELECT id FROM sections").fetchall()]

            # Get all assets
            cursor = conn.execute("SELECT id, filename, file_hash FROM assets ORDER BY filename")
            assets = [(row["filename"], row["file_hash"]) for row in cursor.fetchall()]
            asset_ids = [row["id"] for row in conn.execute("SELECT id FROM assets").fetchall()]

            # Compute manifest hash
            manifest_hash = _compute_manifest_hash(sections, assets)

            # Generate snapshot_id
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM snapshots WHERE snapshot_id LIKE ?",
                (f"cold-{today}-%",),
            )
            count = cursor.fetchone()["count"]
            snapshot_id = f"cold-{today}-{count + 1:03d}"

            created_at = _now_iso()

            # Insert snapshot
            cursor = conn.execute(
                """
                INSERT INTO snapshots
                (snapshot_id, created_at, description, manifest_hash, section_count, asset_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, created_at, description, manifest_hash, len(sections), len(assets)),
            )
            internal_id = cursor.lastrowid

            # Insert junction records
            for section_id in section_ids:
                conn.execute(
                    "INSERT INTO snapshot_sections (snapshot_id, section_id) VALUES (?, ?)",
                    (internal_id, section_id),
                )
            for asset_id in asset_ids:
                conn.execute(
                    "INSERT INTO snapshot_assets (snapshot_id, asset_id) VALUES (?, ?)",
                    (internal_id, asset_id),
                )

            conn.commit()

        logger.info(
            f"Created snapshot: {snapshot_id} ({len(sections)} sections, {len(assets)} assets)"
        )
        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> ColdSnapshot | None:
        """Get a snapshot by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return ColdSnapshot(
                id=row["id"],
                snapshot_id=row["snapshot_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                description=row["description"] or "",
                manifest_hash=row["manifest_hash"],
                section_count=row["section_count"],
                asset_count=row["asset_count"],
            )

    def list_snapshots(self) -> list[str]:
        """List all snapshot IDs (newest first)."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT snapshot_id FROM snapshots ORDER BY created_at DESC")
            return [row["snapshot_id"] for row in cursor.fetchall()]

    def get_latest_snapshot(self) -> ColdSnapshot | None:
        """Get the most recent snapshot."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM snapshots ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                return None

            return ColdSnapshot(
                id=row["id"],
                snapshot_id=row["snapshot_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                description=row["description"] or "",
                manifest_hash=row["manifest_hash"],
                section_count=row["section_count"],
                asset_count=row["asset_count"],
            )

    def get_snapshot_sections(self, snapshot_id: str) -> list[ColdSection]:
        """Get all sections for a snapshot in display order."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.* FROM sections s
                JOIN snapshot_sections ss ON s.id = ss.section_id
                JOIN snapshots sn ON ss.snapshot_id = sn.id
                WHERE sn.snapshot_id = ?
                ORDER BY s.order_num
                """,
                (snapshot_id,),
            )
            return [
                ColdSection(
                    id=row["id"],
                    anchor=row["anchor"],
                    chapter_id=row["chapter_id"],
                    title=row["title"],
                    content=row["content"],
                    content_hash=row["content_hash"],
                    order=row["order_num"],
                    requires_gate=bool(row["requires_gate"]),
                    source_brief_id=row["source_brief_id"],
                    visibility=Visibility(row["visibility"])
                    if row["visibility"]
                    else Visibility.PUBLIC,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]

    def get_snapshot_section_anchors(self, snapshot_id: str) -> list[str]:
        """Get all section anchors for a snapshot in display order."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.anchor FROM sections s
                JOIN snapshot_sections ss ON s.id = ss.section_id
                JOIN snapshots sn ON ss.snapshot_id = sn.id
                WHERE sn.snapshot_id = ?
                ORDER BY s.order_num
                """,
                (snapshot_id,),
            )
            return [row["anchor"] for row in cursor.fetchall()]

    def get_snapshot_assets(self, snapshot_id: str) -> list[ColdAsset]:
        """Get all assets for a snapshot."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT a.* FROM assets a
                JOIN snapshot_assets sa ON a.id = sa.asset_id
                JOIN snapshots sn ON sa.snapshot_id = sn.id
                WHERE sn.snapshot_id = ?
                ORDER BY a.anchor, a.asset_type
                """,
                (snapshot_id,),
            )
            assets = []
            for row in cursor.fetchall():
                provenance = None
                if row["provenance"]:
                    provenance = AssetProvenance(**json.loads(row["provenance"]))
                assets.append(
                    ColdAsset(
                        id=row["id"],
                        anchor=row["anchor"],
                        asset_type=AssetType(row["asset_type"]),
                        filename=row["filename"],
                        file_hash=row["file_hash"],
                        file_size=row["file_size"],
                        mime_type=row["mime_type"],
                        approved_by=row["approved_by"],
                        approved_at=datetime.fromisoformat(row["approved_at"]),
                        provenance=provenance,
                    )
                )
            return assets

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_integrity(self) -> list[str]:
        """Validate cold store integrity.

        Returns
        -------
        list[str]
            List of errors (empty if valid).
        """
        errors = []

        with self._connection() as conn:
            # Check all asset files exist with correct hashes
            cursor = conn.execute("SELECT filename, file_hash, asset_type FROM assets")
            for row in cursor.fetchall():
                path = self.get_asset_path(row["filename"])
                if path is None or not path.exists():
                    errors.append(f"Missing asset file: {row['filename']}")
                else:
                    actual_hash = _compute_file_hash(path)
                    if actual_hash != row["file_hash"]:
                        errors.append(
                            f"Hash mismatch for {row['filename']}: "
                            f"expected {row['file_hash'][:16]}..., "
                            f"got {actual_hash[:16]}..."
                        )

            # Check book metadata
            cursor = conn.execute("SELECT start_section_id FROM book_metadata WHERE id = 1")
            row = cursor.fetchone()
            if row and row["start_section_id"]:
                cursor = conn.execute(
                    "SELECT id FROM sections WHERE id = ?",
                    (row["start_section_id"],),
                )
                if cursor.fetchone() is None:
                    errors.append(f"Invalid start_section_id: {row['start_section_id']}")

            # Check section order contiguity
            cursor = conn.execute("SELECT order_num FROM sections ORDER BY order_num")
            orders = [row["order_num"] for row in cursor.fetchall()]
            if orders:
                expected = list(range(1, len(orders) + 1))
                if orders != expected:
                    errors.append(f"Non-contiguous section order: {orders} (expected {expected})")

        return errors

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        with self._connection() as conn:
            section_count = conn.execute("SELECT COUNT(*) as cnt FROM sections").fetchone()["cnt"]

            asset_count = conn.execute("SELECT COUNT(*) as cnt FROM assets").fetchone()["cnt"]

            snapshot_count = conn.execute("SELECT COUNT(*) as cnt FROM snapshots").fetchone()["cnt"]

            total_content = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) as total FROM sections"
            ).fetchone()["total"]

            latest = self.get_latest_snapshot()

            return {
                "section_count": section_count,
                "asset_count": asset_count,
                "snapshot_count": snapshot_count,
                "total_content_bytes": total_content,
                "latest_snapshot_id": latest.snapshot_id if latest else None,
                "latest_snapshot_at": latest.created_at.isoformat() if latest else None,
            }


# =============================================================================
# Factory Function
# =============================================================================


def get_cold_store(path: str | Path) -> ColdStore:
    """Get a ColdStore, creating it if it doesn't exist.

    This is the recommended way to obtain a ColdStore instance.

    Parameters
    ----------
    path : str | Path
        Path to project directory.

    Returns
    -------
    ColdStore
        Cold store instance (created or loaded).
    """
    return ColdStore.load_or_create(path)
