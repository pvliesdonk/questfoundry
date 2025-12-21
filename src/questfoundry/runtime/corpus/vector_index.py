"""
Vector index for corpus semantic search.

Extends the SQLite corpus index with sqlite-vec for vector similarity search.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.corpus.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


def serialize_float32(vector: list[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_float32(data: bytes) -> list[float]:
    """Deserialize bytes to a float32 vector."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


class VectorIndex:
    """
    Vector index using sqlite-vec.

    Stores embeddings for corpus sections and provides KNN search.
    Integrated with the main corpus index database.
    """

    def __init__(self, index_path: Path, dimension: int = 768):
        """
        Initialize vector index.

        Args:
            index_path: Path to the SQLite database (same as corpus index)
            dimension: Embedding vector dimension
        """
        self._index_path = index_path
        self._dimension = dimension
        self._conn: sqlite3.Connection | None = None
        self._vec_available = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection with sqlite-vec."""
        if self._conn is None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._index_path))
            self._conn.row_factory = sqlite3.Row

            # Try to load sqlite-vec extension
            try:
                self._conn.enable_load_extension(True)
                import sqlite_vec  # type: ignore[import-untyped]

                sqlite_vec.load(self._conn)
                self._vec_available = True
                logger.debug("sqlite-vec extension loaded")
            except Exception as e:
                logger.warning(f"Failed to load sqlite-vec: {e}")
                self._vec_available = False

            self._ensure_schema()

        return self._conn

    def _ensure_schema(self) -> None:
        """Create vector table if not exists."""
        if not self._vec_available:
            return

        conn = self._conn
        if conn is None:
            return

        # Create virtual table for vector search
        # Note: vec0 requires exact dimension in schema
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS corpus_vectors USING vec0(
                section_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self._dimension}]
            )
            """
        )

        # Track embedding metadata
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_embedding_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        conn.commit()

    @property
    def is_available(self) -> bool:
        """Check if vector search is available."""
        self._get_connection()
        return self._vec_available

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_status(self) -> dict[str, Any]:
        """Get vector index status."""
        if not self.is_available:
            return {
                "available": False,
                "reason": "sqlite-vec not loaded",
            }

        conn = self._get_connection()

        # Count vectors
        cursor = conn.execute("SELECT COUNT(*) as count FROM corpus_vectors")
        vector_count = cursor.fetchone()["count"]

        # Get metadata
        cursor = conn.execute("SELECT key, value FROM corpus_embedding_meta")
        meta = {row["key"]: row["value"] for row in cursor.fetchall()}

        return {
            "available": True,
            "vector_count": vector_count,
            "dimension": self._dimension,
            "model": meta.get("model"),
            "indexed_at": meta.get("indexed_at"),
        }

    async def build_vectors(
        self,
        provider: EmbeddingProvider,
        force: bool = False,
    ) -> int:
        """
        Build vector embeddings for all corpus sections.

        Args:
            provider: Embedding provider to use
            force: Force rebuild even if vectors exist

        Returns:
            Number of sections embedded
        """
        if not self.is_available:
            raise RuntimeError("sqlite-vec not available")

        conn = self._get_connection()

        # Check if rebuild needed
        if not force:
            cursor = conn.execute("SELECT COUNT(*) as count FROM corpus_vectors")
            existing = cursor.fetchone()["count"]
            if existing > 0:
                logger.info(f"Vector index already has {existing} vectors, skipping")
                return 0

        # Get all sections from corpus index
        cursor = conn.execute(
            """
            SELECT s.id, s.heading, s.content, f.title
            FROM corpus_sections s
            JOIN corpus_files f ON s.file_id = f.id
            ORDER BY s.id
            """
        )
        sections = cursor.fetchall()

        if not sections:
            logger.warning("No sections found to embed")
            return 0

        # Clear existing vectors if force rebuild
        if force:
            conn.execute("DELETE FROM corpus_vectors")

        # Batch embed sections
        batch_size = 20
        total_embedded = 0

        for i in range(0, len(sections), batch_size):
            batch = sections[i : i + batch_size]

            # Prepare texts for embedding
            # Include heading and title for context
            texts = [f"{row['title']} - {row['heading']}\n\n{row['content']}" for row in batch]

            # Embed batch
            try:
                result = await provider.embed(texts)

                # Insert vectors
                for j, row in enumerate(batch):
                    vector_bytes = serialize_float32(result.embeddings[j])
                    conn.execute(
                        "INSERT INTO corpus_vectors (section_id, embedding) VALUES (?, ?)",
                        (row["id"], vector_bytes),
                    )

                total_embedded += len(batch)
                logger.debug(f"Embedded {total_embedded}/{len(sections)} sections")

            except Exception as e:
                logger.error(f"Failed to embed batch {i}: {e}")
                raise

        # Update metadata
        from datetime import datetime

        conn.execute(
            """
            INSERT OR REPLACE INTO corpus_embedding_meta (key, value)
            VALUES ('model', ?), ('indexed_at', ?), ('dimension', ?)
            """,
            (provider.model, datetime.now().isoformat(), str(provider.dimension)),
        )

        conn.commit()
        logger.info(f"Embedded {total_embedded} sections")
        return total_embedded

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for similar sections using vector similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results to return

        Returns:
            List of matching sections with similarity scores
        """
        if not self.is_available:
            return []

        conn = self._get_connection()
        query_bytes = serialize_float32(query_embedding)

        # KNN search using sqlite-vec
        cursor = conn.execute(
            """
            SELECT
                v.section_id,
                v.distance,
                s.heading,
                s.content,
                f.path,
                f.title,
                f.cluster
            FROM corpus_vectors v
            JOIN corpus_sections s ON v.section_id = s.id
            JOIN corpus_files f ON s.file_id = f.id
            WHERE v.embedding MATCH ?
            ORDER BY v.distance
            LIMIT ?
            """,
            (query_bytes, limit),
        )

        results = []
        for row in cursor.fetchall():
            # Convert distance to similarity score (1 - normalized distance)
            # sqlite-vec returns L2 distance
            distance = row["distance"]
            similarity = 1.0 / (1.0 + distance)  # Convert to 0-1 similarity

            results.append(
                {
                    "section_id": row["section_id"],
                    "heading": row["heading"],
                    "content": row["content"][:500] + "..."
                    if len(row["content"]) > 500
                    else row["content"],
                    "source_file": row["path"],
                    "title": row["title"],
                    "cluster": row["cluster"],
                    "similarity_score": round(similarity, 4),
                    "distance": round(distance, 4),
                }
            )

        return results

    def has_vectors(self) -> bool:
        """Check if vector index has been built."""
        if not self.is_available:
            return False

        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as count FROM corpus_vectors")
        row = cursor.fetchone()
        return bool(row["count"] > 0) if row else False
