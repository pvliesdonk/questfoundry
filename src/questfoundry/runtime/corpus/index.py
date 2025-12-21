"""
Corpus SQLite index.

Stores file metadata and sections for fast lookup and search.
Located at domain level: domain-v4/knowledge/.corpus_index.sqlite
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from questfoundry.runtime.corpus.parser import parse_corpus_directory

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1

CREATE_SCHEMA = """
-- File metadata from frontmatter
CREATE TABLE IF NOT EXISTS corpus_files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    topics TEXT NOT NULL,        -- JSON array
    cluster TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- SHA-256 prefix for invalidation
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Section headers extracted from markdown
CREATE TABLE IF NOT EXISTS corpus_sections (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    heading TEXT NOT NULL,
    level INTEGER NOT NULL,      -- 1=H1, 2=H2, 3=H3
    line_start INTEGER NOT NULL,
    content TEXT NOT NULL,       -- Section content for search
    FOREIGN KEY (file_id) REFERENCES corpus_files(id) ON DELETE CASCADE
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_files_cluster ON corpus_files(cluster);
CREATE INDEX IF NOT EXISTS idx_files_path ON corpus_files(path);
CREATE INDEX IF NOT EXISTS idx_sections_file ON corpus_sections(file_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS corpus_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class IndexedFile:
    """A file record from the index."""

    id: int
    path: str
    title: str
    summary: str
    topics: list[str]
    cluster: str
    content_hash: str


@dataclass
class IndexedSection:
    """A section record from the index."""

    id: int
    file_id: int
    heading: str
    level: int
    line_start: int
    content: str


@dataclass
class IndexStatus:
    """Status of the corpus index."""

    exists: bool
    file_count: int
    section_count: int
    clusters: dict[str, int]  # cluster -> file count
    stale_files: list[str]  # files that need re-indexing


class CorpusIndex:
    """
    SQLite-based corpus index.

    Stores file metadata and sections for fast lookup.
    Supports incremental updates based on content hash.
    """

    def __init__(self, index_path: Path):
        """
        Initialize corpus index.

        Args:
            index_path: Path to the SQLite database file
        """
        self._index_path = index_path
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def get_index_path(cls, domain_path: Path) -> Path:
        """Get the standard index path for a domain."""
        return domain_path / "knowledge" / ".corpus_index.sqlite"

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._index_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        """Create schema if not exists."""
        conn = self._conn
        if conn is None:
            return

        conn.executescript(CREATE_SCHEMA)

        # Set schema version if not set
        cursor = conn.execute("SELECT value FROM corpus_meta WHERE key = 'schema_version'")
        row = cursor.fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO corpus_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def build(self, corpus_dir: Path, force: bool = False) -> int:
        """
        Build or update the corpus index.

        Args:
            corpus_dir: Path to the corpus directory
            force: If True, rebuild all files regardless of hash

        Returns:
            Number of files indexed
        """
        conn = self._get_connection()
        files = parse_corpus_directory(corpus_dir)

        indexed_count = 0
        for corpus_file in files:
            relative_path = str(corpus_file.path.relative_to(corpus_dir.parent.parent))

            # Check if file needs update
            if not force:
                cursor = conn.execute(
                    "SELECT content_hash FROM corpus_files WHERE path = ?",
                    (relative_path,),
                )
                row = cursor.fetchone()
                if row and row["content_hash"] == corpus_file.content_hash:
                    logger.debug(f"Skipping unchanged file: {relative_path}")
                    continue

            # Delete existing records for this file
            cursor = conn.execute("SELECT id FROM corpus_files WHERE path = ?", (relative_path,))
            row = cursor.fetchone()
            if row:
                conn.execute("DELETE FROM corpus_files WHERE id = ?", (row["id"],))

            # Insert file record
            cursor = conn.execute(
                """
                INSERT INTO corpus_files (path, title, summary, topics, cluster, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    relative_path,
                    corpus_file.frontmatter.title,
                    corpus_file.frontmatter.summary,
                    json.dumps(corpus_file.frontmatter.topics),
                    corpus_file.frontmatter.cluster,
                    corpus_file.content_hash,
                ),
            )
            file_id = cursor.lastrowid

            # Insert sections
            for section in corpus_file.sections:
                conn.execute(
                    """
                    INSERT INTO corpus_sections (file_id, heading, level, line_start, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        section.heading,
                        section.level,
                        section.line_start,
                        section.content,
                    ),
                )

            indexed_count += 1
            logger.debug(f"Indexed: {relative_path} ({len(corpus_file.sections)} sections)")

        conn.commit()
        logger.info(f"Indexed {indexed_count} files")
        return indexed_count

    def get_status(self, corpus_dir: Path | None = None) -> IndexStatus:
        """
        Get index status.

        Args:
            corpus_dir: Optional corpus directory to check for stale files

        Returns:
            IndexStatus with counts and cluster breakdown
        """
        if not self._index_path.exists():
            return IndexStatus(
                exists=False,
                file_count=0,
                section_count=0,
                clusters={},
                stale_files=[],
            )

        conn = self._get_connection()

        # File count
        cursor = conn.execute("SELECT COUNT(*) as count FROM corpus_files")
        file_count = cursor.fetchone()["count"]

        # Section count
        cursor = conn.execute("SELECT COUNT(*) as count FROM corpus_sections")
        section_count = cursor.fetchone()["count"]

        # Cluster breakdown
        cursor = conn.execute(
            "SELECT cluster, COUNT(*) as count FROM corpus_files GROUP BY cluster"
        )
        clusters = {row["cluster"]: row["count"] for row in cursor.fetchall()}

        # Check for stale files
        stale_files: list[str] = []
        if corpus_dir:
            cursor = conn.execute("SELECT path, content_hash FROM corpus_files")
            for row in cursor.fetchall():
                file_path = corpus_dir.parent.parent / row["path"]
                if file_path.exists():
                    import hashlib

                    content = file_path.read_text(encoding="utf-8")
                    current_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                    if current_hash != row["content_hash"]:
                        stale_files.append(row["path"])
                else:
                    stale_files.append(row["path"])

        return IndexStatus(
            exists=True,
            file_count=file_count,
            section_count=section_count,
            clusters=clusters,
            stale_files=stale_files,
        )

    def get_toc(self) -> list[dict[str, Any]]:
        """
        Get table of contents (all files with metadata).

        Returns:
            List of file records with title, summary, cluster, topics
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT path, title, summary, topics, cluster
            FROM corpus_files
            ORDER BY cluster, title
            """
        )

        toc = []
        for row in cursor.fetchall():
            toc.append(
                {
                    "path": row["path"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "topics": json.loads(row["topics"]),
                    "cluster": row["cluster"],
                }
            )

        return toc

    def get_file(self, filename: str) -> dict[str, Any] | None:
        """
        Get a specific file by name.

        Args:
            filename: Filename (with or without path/extension)

        Returns:
            File record with sections, or None if not found
        """
        conn = self._get_connection()

        # Normalize filename
        if not filename.endswith(".md"):
            filename = filename + ".md"
        if "/" not in filename:
            filename = f"knowledge/corpus/{filename}"

        cursor = conn.execute(
            """
            SELECT id, path, title, summary, topics, cluster
            FROM corpus_files
            WHERE path = ? OR path LIKE ?
            """,
            (filename, f"%/{filename}"),
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Get sections
        section_cursor = conn.execute(
            """
            SELECT heading, level, line_start, content
            FROM corpus_sections
            WHERE file_id = ?
            ORDER BY line_start
            """,
            (row["id"],),
        )

        sections = [
            {
                "heading": s["heading"],
                "level": s["level"],
                "line_start": s["line_start"],
                "content": s["content"],
            }
            for s in section_cursor.fetchall()
        ]

        return {
            "path": row["path"],
            "title": row["title"],
            "summary": row["summary"],
            "topics": json.loads(row["topics"]),
            "cluster": row["cluster"],
            "sections": sections,
        }

    def get_cluster(self, cluster: str) -> list[dict[str, Any]]:
        """
        Get all files in a cluster.

        Args:
            cluster: Cluster name

        Returns:
            List of file records in the cluster
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT path, title, summary, topics
            FROM corpus_files
            WHERE cluster = ?
            ORDER BY title
            """,
            (cluster,),
        )

        return [
            {
                "path": row["path"],
                "title": row["title"],
                "summary": row["summary"],
                "topics": json.loads(row["topics"]),
            }
            for row in cursor.fetchall()
        ]

    def keyword_search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """
        Search sections by keyword.

        Simple text search in section content.
        For better results, use hybrid search when vectors are available.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of matching sections with file info
        """
        conn = self._get_connection()

        # Simple LIKE search for now
        # TODO: Improve with FTS5 or vector search
        search_term = f"%{query}%"

        cursor = conn.execute(
            """
            SELECT
                s.heading,
                s.level,
                s.content,
                f.path,
                f.title,
                f.cluster
            FROM corpus_sections s
            JOIN corpus_files f ON s.file_id = f.id
            WHERE s.content LIKE ? OR s.heading LIKE ?
            LIMIT ?
            """,
            (search_term, search_term, max_results),
        )

        results = []
        for row in cursor.fetchall():
            # Calculate simple relevance score
            content_lower = row["content"].lower()
            query_lower = query.lower()
            count = content_lower.count(query_lower)
            score = min(count / 10.0, 1.0)  # Normalize to 0-1

            results.append(
                {
                    "heading": row["heading"],
                    "level": row["level"],
                    "content": row["content"][:500] + "..."
                    if len(row["content"]) > 500
                    else row["content"],
                    "source_file": row["path"],
                    "title": row["title"],
                    "cluster": row["cluster"],
                    "relevance_score": round(score, 3),
                }
            )

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]

    def list_clusters(self) -> list[dict[str, Any]]:
        """
        List all clusters with file counts.

        Returns:
            List of cluster info dicts
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT cluster, COUNT(*) as file_count
            FROM corpus_files
            GROUP BY cluster
            ORDER BY cluster
            """
        )

        cluster_descriptions = {
            "narrative-structure": "Plot organization, pacing, branching, endings",
            "prose-and-language": "Voice, dialogue, exposition, subtext",
            "genre-conventions": "Genre-specific tropes and conventions",
            "audience-and-access": "Age targeting, accessibility, localization",
            "world-and-setting": "Worldbuilding, setting as character",
            "emotional-design": "Emotional beats, conflict patterns",
            "scope-and-planning": "Project sizing and metrics",
        }

        return [
            {
                "cluster": row["cluster"],
                "file_count": row["file_count"],
                "description": cluster_descriptions.get(row["cluster"], ""),
            }
            for row in cursor.fetchall()
        ]
