"""
Aligned Cold Store - Schema-aligned durable storage for Cold Source of Truth (SoT).

Design:
    - SQLite database with schema aligned to QuestFoundry JSON schemas
    - Hybrid storage: metadata in database, content as external files
    - SHA-256 hashing for deterministic builds
    - Views for JSON generation matching schema format exactly
    - Triggers enforce immutability of Cold content
    - One portable .qfdb file per project

Database stores:
    - Manifest data (paths, hashes, sizes)
    - Book metadata and structure
    - Art asset references and provenance
    - Hot artifacts as JSON documents
    - Trace units and lifecycle tracking
    - Event log for audit trail

File system stores:
    - Section content (sections/*.md)
    - Images (assets/*.png)
    - Fonts (fonts/*.woff2)
    - Hot drafts (hot/sections/*.md)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _default_project_root() -> Path:
    """Default root for QuestFoundry projects under the user's home directory."""
    import os

    home = Path(os.path.expanduser("~"))
    return home / ".questfoundry" / "projects"


class AlignedColdStore:
    """
    Schema-aligned SQLite storage for Cold SoT per project.

    Implements hybrid storage: metadata in database, content as external files.
    Database structure matches QuestFoundry JSON schemas exactly.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        project_root: str | Path | None = None,
        project_id: str | None = None,
    ) -> None:
        """
        Initialize Aligned Cold Store.

        Args:
            db_path: Path to .qfdb SQLite database file. If None, derived from project_id.
            project_root: Root directory containing assets/, sections/, etc.
            project_id: Project identifier (used to derive paths if others not provided)
        """
        if project_root is not None:
            self.project_root = Path(project_root).expanduser()
        elif project_id is not None:
            self.project_root = _default_project_root() / project_id
        else:
            msg = "Either project_root or project_id must be provided"
            raise ValueError(msg)

        if db_path is not None:
            self.db_path = Path(db_path).expanduser()
        else:
            self.db_path = self.project_root / "project.qfdb"

        # Standard directories
        self.cold_dir = self.project_root / "cold"
        self.assets_dir = self.project_root / "assets"
        self.sections_dir = self.project_root / "sections"
        self.hot_dir = self.project_root / "hot"

        # Ensure directories exist
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.cold_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)
        self.sections_dir.mkdir(exist_ok=True)
        self.hot_dir.mkdir(exist_ok=True)
        (self.hot_dir / "sections").mkdir(exist_ok=True)
        (self.hot_dir / "assets").mkdir(exist_ok=True)

        # Initialize database
        if not self.db_path.exists():
            self._initialize_database()
        else:
            self._validate_database()

    @contextmanager
    def connection(self):
        """Database connection context manager with WAL mode and foreign keys."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.exception("Database error: %s", e)
            raise
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        """Create new database with schema from SQL file."""
        logger.info("Initializing aligned database: %s", self.db_path)

        # Read schema from SQL file
        schema_path = Path(__file__).parent / "aligned_cold_store.sql"
        if schema_path.exists():
            schema_sql = schema_path.read_text(encoding="utf-8")
        else:
            msg = f"Schema file not found: {schema_path}"
            raise FileNotFoundError(msg)

        with self.connection() as conn:
            conn.executescript(schema_sql)

            # Insert default singleton rows
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT OR IGNORE INTO cold_manifest (id, version, created_at, snapshot_id)
                VALUES (1, '1.0.0', ?, 'cold-init')
                """,
                (now,),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO cold_book_metadata (id, title, language, start_section)
                VALUES (1, 'Untitled Project', 'en', 'anchor001')
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO hot_manifest (id, version, snapshot_at, snapshot_id)
                VALUES (1, '1.0.0', ?, 'hot-init')
                """,
                (now,),
            )

        logger.info("Database initialized successfully")

    def _validate_database(self) -> None:
        """Validate existing database has correct schema."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN
                ('cold_manifest', 'cold_book_metadata', 'cold_art_assets')
                """
            )
            tables = [row["name"] for row in cursor]

            if len(tables) != 3:
                msg = f"Invalid database schema, missing tables. Found: {tables}"
                raise ValueError(msg)

    def _hash_file(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)
        return file_path.stat().st_size

    # =====================================================
    # COLD MANIFEST OPERATIONS
    # =====================================================

    def create_cold_snapshot(self, snapshot_id: str) -> None:
        """
        Create a new Cold snapshot with all file hashes.

        Args:
            snapshot_id: Snapshot identifier (e.g., 'cold-20251124')
        """
        if not snapshot_id.startswith("cold-"):
            snapshot_id = f"cold-{snapshot_id}"

        with self.connection() as conn:
            # Clear old file entries (but not manifest itself for fresh snapshots)
            conn.execute("DELETE FROM cold_manifest_files")

            # Update manifest metadata - use INSERT OR REPLACE to handle immutability
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO cold_manifest (id, version, created_at, snapshot_id)
                VALUES (1, '1.0.0', ?, ?)
                """,
                (now, snapshot_id),
            )

            # Add manifest files themselves (placeholders, updated after export)
            for manifest_file in ["manifest.json", "book.json", "art_manifest.json"]:
                path = f"cold/{manifest_file}"
                conn.execute(
                    """
                    INSERT INTO cold_manifest_files (path, sha256, size_bytes)
                    VALUES (?, ?, ?)
                    """,
                    (path, "0" * 64, 0),
                )

            # Add all section files from cold_book_sections
            cursor = conn.execute("SELECT text_file FROM cold_book_sections")
            for row in cursor:
                file_path = self.project_root / row["text_file"]
                if file_path.exists():
                    conn.execute(
                        """
                        INSERT INTO cold_manifest_files (path, sha256, size_bytes)
                        VALUES (?, ?, ?)
                        """,
                        (
                            row["text_file"],
                            self._hash_file(file_path),
                            self._get_file_size(file_path),
                        ),
                    )

            # Add all asset files from cold_art_assets
            cursor = conn.execute("SELECT filename FROM cold_art_assets")
            for row in cursor:
                file_path = self.assets_dir / row["filename"]
                if file_path.exists():
                    path = f"assets/{row['filename']}"
                    conn.execute(
                        """
                        INSERT INTO cold_manifest_files (path, sha256, size_bytes)
                        VALUES (?, ?, ?)
                        """,
                        (
                            path,
                            self._hash_file(file_path),
                            self._get_file_size(file_path),
                        ),
                    )

        logger.debug("Created Cold snapshot: %s", snapshot_id)

    def export_cold_manifest(self) -> dict[str, Any]:
        """Export cold_manifest.json structure."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT manifest_json FROM cold_manifest_json")
            row = cursor.fetchone()
            if row and row["manifest_json"]:
                return json.loads(row["manifest_json"])
        return {}

    def get_cold_snapshot_id(self) -> str | None:
        """Get current Cold snapshot ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT snapshot_id FROM cold_manifest WHERE id = 1")
            row = cursor.fetchone()
            if row:
                return row["snapshot_id"]
        return None

    # =====================================================
    # COLD BOOK OPERATIONS
    # =====================================================

    def set_book_metadata(self, **kwargs) -> None:
        """
        Set Cold book metadata.

        Args:
            title: Book title
            subtitle: Book subtitle
            language: ISO 639-1/3 language code
            author: Author name
            isbn: ISBN
            published_at: Publication date (YYYY-MM-DD)
            edition: Edition information
            copyright: Copyright statement
            publisher: Publisher name
            start_section: Starting anchor (e.g., 'anchor001')
        """
        allowed_fields = [
            "title",
            "subtitle",
            "language",
            "author",
            "isbn",
            "published_at",
            "edition",
            "copyright",
            "publisher",
            "start_section",
        ]

        fields = []
        values = []
        for key, value in kwargs.items():
            if key in allowed_fields:
                fields.append(f"{key} = ?")
                values.append(value)

        if fields:
            with self.connection() as conn:
                sql = f"UPDATE cold_book_metadata SET {', '.join(fields)} WHERE id = 1"
                conn.execute(sql, values)
                logger.debug("Updated book metadata: %s", ", ".join(kwargs.keys()))

    def get_book_metadata(self) -> dict[str, Any]:
        """Get Cold book metadata."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT title, subtitle, language, author, isbn, published_at,
                       edition, copyright, publisher, start_section
                FROM cold_book_metadata WHERE id = 1
                """
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return {}

    def add_book_section(
        self,
        anchor: str,
        title: str,
        text_file: str,
        order_num: int,
        requires_gate: bool = False,
    ) -> None:
        """
        Add a section to the Cold book.

        Args:
            anchor: Section anchor (e.g., 'anchor001')
            title: Section title
            text_file: Path to markdown file (e.g., 'sections/001.md')
            order_num: Section order number
            requires_gate: Whether section requires gateway condition
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cold_book_sections
                (anchor, title, text_file, order_num, player_safe, requires_gate)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (anchor, title, text_file, order_num, 1 if requires_gate else 0),
            )
            logger.debug("Added book section: %s", anchor)

    def get_book_sections(self) -> list[dict[str, Any]]:
        """Get all Cold book sections ordered by order_num."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT anchor, title, text_file, order_num, player_safe, requires_gate
                FROM cold_book_sections ORDER BY order_num
                """
            )
            return [dict(row) for row in cursor]

    def export_cold_book(self) -> dict[str, Any]:
        """Export cold_book.json structure."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT book_json FROM cold_book_json")
            row = cursor.fetchone()
            if row and row["book_json"]:
                return json.loads(row["book_json"])
        return {}

    # =====================================================
    # COLD ART OPERATIONS
    # =====================================================

    def add_art_asset(
        self,
        anchor: str,
        asset_type: str,
        filename: str,
        width_px: int,
        height_px: int,
        format_: str,
        approved_by: str,
        provenance: dict[str, Any],
    ) -> None:
        """
        Add an art asset to Cold.

        Args:
            anchor: Asset anchor (e.g., 'anchor001', 'cover')
            asset_type: Type ('plate', 'cover', 'icon', etc.)
            filename: Asset filename
            width_px: Width in pixels
            height_px: Height in pixels
            format_: File format ('PNG', 'JPG', 'SVG', etc.)
            approved_by: Role abbreviation who approved
            provenance: Dict with role, prompt_snippet, version, policy_notes
        """
        # Validate file exists and calculate hash
        file_path = self.assets_dir / filename
        if not file_path.exists():
            msg = f"Asset file not found: {file_path}"
            raise FileNotFoundError(msg)

        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cold_art_assets
                (anchor, type, filename, sha256, size_bytes,
                 width_px, height_px, format, approved_at, approved_by, provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    anchor,
                    asset_type,
                    filename,
                    self._hash_file(file_path),
                    self._get_file_size(file_path),
                    width_px,
                    height_px,
                    format_,
                    datetime.now(UTC).isoformat(),
                    approved_by,
                    json.dumps(provenance),
                ),
            )
            logger.debug("Added art asset: %s/%s", anchor, asset_type)

    def get_art_assets(self) -> list[dict[str, Any]]:
        """Get all Cold art assets."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT anchor, type, filename, sha256, size_bytes,
                       width_px, height_px, format, approved_at, approved_by, provenance
                FROM cold_art_assets
                """
            )
            results = []
            for row in cursor:
                item = dict(row)
                item["provenance"] = json.loads(item["provenance"])
                results.append(item)
            return results

    def export_cold_art_manifest(self) -> dict[str, Any]:
        """Export cold_art_manifest.json structure."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT art_json FROM cold_art_manifest_json")
            row = cursor.fetchone()
            if row and row["art_json"]:
                return json.loads(row["art_json"])
        return {}

    # =====================================================
    # HOT ARTIFACT OPERATIONS
    # =====================================================

    def create_hot_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        content: dict[str, Any],
        path: str | None = None,
        status: str = "proposed",
    ) -> None:
        """
        Create a Hot artifact (stored as JSON in database).

        Args:
            artifact_id: Unique artifact ID (e.g., 'TU-2025-11-24-SR01')
            artifact_type: Type of artifact
            content: Artifact content as dict
            path: Optional path reference
            status: Initial status (default: 'proposed')
        """
        if not path:
            path = f"hot/{artifact_type}s/{artifact_id}.json"

        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hot_artifacts
                (artifact_id, artifact_type, path, status, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                (artifact_id, artifact_type, path, status, json.dumps(content)),
            )
            logger.debug("Created Hot artifact: %s", artifact_id)

    def update_hot_artifact_status(self, artifact_id: str, new_status: str) -> None:
        """Update Hot artifact status."""
        valid_statuses = ["proposed", "in-progress", "stabilizing", "gatecheck", "resolved"]
        if new_status not in valid_statuses:
            msg = f"Invalid status: {new_status}. Must be one of {valid_statuses}"
            raise ValueError(msg)

        with self.connection() as conn:
            conn.execute(
                """
                UPDATE hot_artifacts
                SET status = ?, updated_at = datetime('now')
                WHERE artifact_id = ?
                """,
                (new_status, artifact_id),
            )
            logger.debug("Updated %s status to %s", artifact_id, new_status)

    def get_hot_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Get a Hot artifact by ID."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM hot_artifacts WHERE artifact_id = ?
                """,
                (artifact_id,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["artifact_id"],
                    "type": row["artifact_type"],
                    "path": row["path"],
                    "status": row["status"],
                    "content": json.loads(row["content"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
        return None

    def list_hot_artifacts(
        self, artifact_type: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """List Hot artifacts with optional filtering."""
        with self.connection() as conn:
            sql = "SELECT * FROM hot_artifacts WHERE 1=1"
            params: list[Any] = []

            if artifact_type:
                sql += " AND artifact_type = ?"
                params.append(artifact_type)
            if status:
                sql += " AND status = ?"
                params.append(status)

            cursor = conn.execute(sql, params)
            return [
                {
                    "id": row["artifact_id"],
                    "type": row["artifact_type"],
                    "path": row["path"],
                    "status": row["status"],
                    "content": json.loads(row["content"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in cursor
            ]

    def export_hot_manifest(self) -> dict[str, Any]:
        """Export hot_manifest.json structure."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT manifest_json FROM hot_manifest_json")
            row = cursor.fetchone()
            if row and row["manifest_json"]:
                return json.loads(row["manifest_json"])
        return {}

    # =====================================================
    # HOT SECTION OPERATIONS
    # =====================================================

    def add_hot_section(
        self,
        anchor: str,
        text_file: str,
        status: str = "draft",
        title: str | None = None,
        tu_id: str | None = None,
    ) -> None:
        """Add a section to Hot."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hot_sections
                (anchor, title, text_file, status, tu_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (anchor, title, text_file, status, tu_id),
            )
            logger.debug("Added Hot section: %s", anchor)

    def update_hot_section_status(self, anchor: str, new_status: str) -> None:
        """Update Hot section status."""
        valid_statuses = ["draft", "revising", "stabilizing", "gatecheck", "approved"]
        if new_status not in valid_statuses:
            msg = f"Invalid status: {new_status}"
            raise ValueError(msg)

        with self.connection() as conn:
            conn.execute(
                "UPDATE hot_sections SET status = ? WHERE anchor = ?",
                (new_status, anchor),
            )

    def get_hot_sections(self, status: str | None = None) -> list[dict[str, Any]]:
        """Get Hot sections with optional status filter."""
        with self.connection() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM hot_sections WHERE status = ?", (status,)
                )
            else:
                cursor = conn.execute("SELECT * FROM hot_sections")
            return [dict(row) for row in cursor]

    # =====================================================
    # TRACE UNIT OPERATIONS
    # =====================================================

    def create_trace_unit(self, tu_brief: dict[str, Any]) -> None:
        """
        Create a TU from a tu_brief following the schema.

        Args:
            tu_brief: TU brief dict following tu_brief.schema.json
        """
        tu_id = tu_brief["id"]
        path = f"hot/tu_briefs/{tu_id}.json"

        with self.connection() as conn:
            # Insert into trace_units table
            conn.execute(
                """
                INSERT INTO trace_units (
                    tu_id, opened, owner_a, responsible_r, loop, slice,
                    snapshot_context, awake, dormant, deferral_tags,
                    press, monitor, pre_gate_risks, inputs, deliverables,
                    bars_green, merge_view, timebox, gatecheck, linkage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tu_id,
                    tu_brief["opened"],
                    tu_brief["owner_a"],
                    json.dumps(tu_brief.get("responsible_r", [])),
                    tu_brief["loop"],
                    tu_brief["slice"],
                    tu_brief.get("snapshot_context"),
                    json.dumps(tu_brief.get("awake", [])),
                    json.dumps(tu_brief.get("dormant", [])),
                    json.dumps(tu_brief.get("deferral_tags", [])),
                    json.dumps(tu_brief.get("press", [])),
                    json.dumps(tu_brief.get("monitor", [])),
                    json.dumps(tu_brief.get("pre_gate_risks", [])),
                    json.dumps(tu_brief.get("inputs", [])),
                    json.dumps(tu_brief.get("deliverables", [])),
                    json.dumps(tu_brief.get("bars_green", [])),
                    tu_brief.get("merge_view"),
                    tu_brief.get("timebox", "90 min"),
                    tu_brief.get("gatecheck"),
                    tu_brief.get("linkage"),
                ),
            )

            # Also create as Hot artifact (in same transaction to avoid lock)
            conn.execute(
                """
                INSERT OR REPLACE INTO hot_artifacts
                (artifact_id, artifact_type, path, status, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                (tu_id, "tu_brief", path, "proposed", json.dumps(tu_brief)),
            )

        logger.debug("Created trace unit: %s", tu_id)

    def get_trace_unit(self, tu_id: str) -> dict[str, Any] | None:
        """Get a trace unit by ID."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM trace_units WHERE tu_id = ?", (tu_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                for field in [
                    "responsible_r",
                    "awake",
                    "dormant",
                    "deferral_tags",
                    "press",
                    "monitor",
                    "pre_gate_risks",
                    "inputs",
                    "deliverables",
                    "bars_green",
                ]:
                    if result.get(field):
                        result[field] = json.loads(result[field])
                return result
        return None

    def update_tu_lifecycle(self, tu_id: str, new_stage: str) -> None:
        """Update TU lifecycle stage."""
        valid_stages = ["hot-proposed", "stabilizing", "gatecheck", "cold-merged", "archived"]
        if new_stage not in valid_stages:
            msg = f"Invalid lifecycle stage: {new_stage}"
            raise ValueError(msg)

        with self.connection() as conn:
            # Update stage
            conn.execute(
                """
                UPDATE trace_units
                SET lifecycle_stage = ?
                WHERE tu_id = ?
                """,
                (new_stage, tu_id),
            )

            # Set appropriate timestamp
            if new_stage == "stabilizing":
                conn.execute(
                    "UPDATE trace_units SET stabilized_at = datetime('now') WHERE tu_id = ?",
                    (tu_id,),
                )
            elif new_stage == "gatecheck":
                conn.execute(
                    "UPDATE trace_units SET gatecheck_at = datetime('now') WHERE tu_id = ?",
                    (tu_id,),
                )
            elif new_stage == "cold-merged":
                conn.execute(
                    "UPDATE trace_units SET merged_at = datetime('now') WHERE tu_id = ?",
                    (tu_id,),
                )

        logger.debug("Updated %s to lifecycle stage: %s", tu_id, new_stage)

    def list_trace_units(self, lifecycle_stage: str | None = None) -> list[dict[str, Any]]:
        """List trace units with optional lifecycle filter."""
        with self.connection() as conn:
            if lifecycle_stage:
                cursor = conn.execute(
                    "SELECT * FROM trace_units WHERE lifecycle_stage = ?",
                    (lifecycle_stage,),
                )
            else:
                cursor = conn.execute("SELECT * FROM trace_units")

            results = []
            for row in cursor:
                result = dict(row)
                # Parse JSON fields
                for field in [
                    "responsible_r",
                    "awake",
                    "dormant",
                    "deferral_tags",
                    "press",
                    "monitor",
                    "pre_gate_risks",
                    "inputs",
                    "deliverables",
                    "bars_green",
                ]:
                    if result.get(field):
                        result[field] = json.loads(result[field])
                results.append(result)
            return results

    # =====================================================
    # QUALITY CHECK OPERATIONS
    # =====================================================

    def add_quality_check(
        self,
        tu_id: str,
        bar_name: str,
        status: str,
        checked_by: str,
        feedback: str | None = None,
    ) -> None:
        """Add a quality check result for a TU."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO quality_checks
                (tu_id, bar_name, status, feedback, checked_by, checked_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (tu_id, bar_name, status, feedback, checked_by),
            )

    def get_quality_checks(self, tu_id: str) -> list[dict[str, Any]]:
        """Get quality checks for a TU."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM quality_checks WHERE tu_id = ?", (tu_id,)
            )
            return [dict(row) for row in cursor]

    # =====================================================
    # VIEW LOG OPERATIONS
    # =====================================================

    def add_view_log(self, view_log: dict[str, Any]) -> int:
        """Add a view log entry."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO view_logs (
                    title, bound, binder, tu, cold_snapshot, targets,
                    options_and_coverage, dormancy, anchor_map,
                    presentation_status, presentation_notes,
                    accessibility_status, accessibility_notes,
                    gatekeeper, gatecheck_id, export_artifacts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    view_log["title"],
                    view_log["bound"],
                    view_log["binder"],
                    view_log["tu"],
                    view_log["cold_snapshot"],
                    json.dumps(view_log["targets"]),
                    view_log.get("options_and_coverage"),
                    json.dumps(view_log.get("dormancy", [])),
                    view_log.get("anchor_map"),
                    view_log["presentation_status"],
                    view_log.get("presentation_notes"),
                    view_log["accessibility_status"],
                    view_log.get("accessibility_notes"),
                    view_log["gatekeeper"],
                    view_log.get("gatecheck_id"),
                    json.dumps(view_log["export_artifacts"]),
                ),
            )
            return cursor.lastrowid or 0

    # =====================================================
    # EVENT LOG OPERATIONS
    # =====================================================

    def log_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any],
        actor_role: str,
    ) -> None:
        """Log an event to the audit trail."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO events (event_type, entity_type, entity_id, payload, actor_role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_type, entity_type, entity_id, json.dumps(payload), actor_role),
            )

    def get_events(
        self,
        event_type: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get events from audit log with optional filtering."""
        with self.connection() as conn:
            sql = "SELECT * FROM events WHERE 1=1"
            params: list[Any] = []

            if event_type:
                sql += " AND event_type = ?"
                params.append(event_type)
            if entity_type:
                sql += " AND entity_type = ?"
                params.append(entity_type)
            if entity_id:
                sql += " AND entity_id = ?"
                params.append(entity_id)

            sql += " ORDER BY occurred_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            results = []
            for row in cursor:
                result = dict(row)
                result["payload"] = json.loads(result["payload"])
                results.append(result)
            return results

    # =====================================================
    # EXPORT OPERATIONS
    # =====================================================

    def export_all_manifests(self) -> None:
        """Export all Cold manifests to JSON files."""
        self.cold_dir.mkdir(exist_ok=True)

        # Export manifest.json
        manifest = self.export_cold_manifest()
        if manifest:
            manifest_path = self.cold_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            logger.info("Exported: %s", manifest_path)

        # Export book.json
        book = self.export_cold_book()
        if book:
            book_path = self.cold_dir / "book.json"
            book_path.write_text(json.dumps(book, indent=2), encoding="utf-8")
            logger.info("Exported: %s", book_path)

        # Export art_manifest.json
        art = self.export_cold_art_manifest()
        if art:
            art_path = self.cold_dir / "art_manifest.json"
            art_path.write_text(json.dumps(art, indent=2), encoding="utf-8")
            logger.info("Exported: %s", art_path)

        # Now update manifest with actual hashes of the JSON files
        self._update_manifest_hashes()

    def _update_manifest_hashes(self) -> None:
        """Update manifest with actual hashes of generated JSON files."""
        with self.connection() as conn:
            for manifest_file in ["manifest.json", "book.json", "art_manifest.json"]:
                file_path = self.cold_dir / manifest_file
                if file_path.exists():
                    conn.execute(
                        """
                        UPDATE cold_manifest_files
                        SET sha256 = ?, size_bytes = ?
                        WHERE path = ?
                        """,
                        (
                            self._hash_file(file_path),
                            self._get_file_size(file_path),
                            f"cold/{manifest_file}",
                        ),
                    )

    # =====================================================
    # VALIDATION OPERATIONS
    # =====================================================

    def validate_cold_integrity(self) -> list[str]:
        """
        Validate Cold integrity (hashes, references, etc.).

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        with self.connection() as conn:
            # Check all files exist with correct hashes
            cursor = conn.execute("SELECT path, sha256 FROM cold_manifest_files")
            for row in cursor:
                if row["path"].startswith("cold/"):
                    # Skip manifest files during validation (they may not exist yet)
                    continue

                file_path = self.project_root / row["path"]
                if not file_path.exists():
                    errors.append(f"Missing file: {row['path']}")
                else:
                    actual_hash = self._hash_file(file_path)
                    if actual_hash != row["sha256"]:
                        errors.append(f"Hash mismatch: {row['path']}")

            # Check all sections have text files
            cursor = conn.execute("SELECT anchor, text_file FROM cold_book_sections")
            for row in cursor:
                file_path = self.project_root / row["text_file"]
                if not file_path.exists():
                    errors.append(f"Missing section file: {row['text_file']} for {row['anchor']}")

            # Check start_section exists
            cursor = conn.execute(
                """
                SELECT cbm.start_section
                FROM cold_book_metadata cbm
                LEFT JOIN cold_book_sections cbs ON cbm.start_section = cbs.anchor
                WHERE cbm.id = 1 AND cbs.id IS NULL
                """
            )
            if cursor.fetchone():
                errors.append("start_section references non-existent anchor")

        return errors

    def get_statistics(self) -> dict[str, Any]:
        """Get project statistics."""
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Count sections
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM cold_book_sections")
            row = cursor.fetchone()
            stats["sections"] = row["cnt"] if row else 0

            # Count assets
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM cold_art_assets")
            row = cursor.fetchone()
            stats["assets"] = row["cnt"] if row else 0

            # Count TUs
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM trace_units")
            row = cursor.fetchone()
            stats["trace_units"] = row["cnt"] if row else 0

            # Count Hot artifacts by type
            cursor = conn.execute(
                """
                SELECT artifact_type, COUNT(*) as cnt
                FROM hot_artifacts
                GROUP BY artifact_type
                """
            )
            stats["hot_artifacts"] = {row["artifact_type"]: row["cnt"] for row in cursor}

            # Quality check summary
            cursor = conn.execute(
                """
                SELECT bar_name, status, COUNT(*) as cnt
                FROM quality_checks
                GROUP BY bar_name, status
                """
            )
            quality: dict[str, dict[str, int]] = {}
            for row in cursor:
                if row["bar_name"] not in quality:
                    quality[row["bar_name"]] = {}
                quality[row["bar_name"]][row["status"]] = row["cnt"]
            stats["quality_checks"] = quality

            return stats

    # =====================================================
    # BACKWARD COMPATIBILITY WITH ColdStore
    # =====================================================

    def load_cold(self, project_id: str) -> dict[str, Any] | None:
        """
        Load Cold SoT in a format compatible with the old ColdStore.

        Returns a dict with 'manifest', 'book', 'art_manifest' keys.
        """
        return {
            "manifest": self.export_cold_manifest(),
            "book": self.export_cold_book(),
            "art_manifest": self.export_cold_art_manifest(),
            "metadata": self.get_book_metadata(),
            "sections": self.get_book_sections(),
            "assets": self.get_art_assets(),
        }

    def save_cold(self, project_id: str, cold_sot: dict[str, Any]) -> None:
        """
        Save Cold SoT from a dict (backward compatible with old ColdStore).

        This imports data from the dict format into the structured tables.
        """
        # Import book metadata if present
        if "metadata" in cold_sot:
            self.set_book_metadata(**cold_sot["metadata"])

        # Import sections if present
        if "sections" in cold_sot:
            for section in cold_sot["sections"]:
                self.add_book_section(
                    anchor=section["anchor"],
                    title=section["title"],
                    text_file=section["text_file"],
                    order_num=section["order_num"],
                    requires_gate=section.get("requires_gate", False),
                )

        logger.info("Saved Cold SoT for project: %s", project_id)
