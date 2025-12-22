"""
Vector index for corpus semantic search.

Extends the SQLite corpus index with sqlite-vec for vector similarity search.
Supports multiple embedding models by using model-specific tables
(e.g., corpus_vectors_nomic_embed_text, corpus_vectors_text_embedding_3_small).
"""

from __future__ import annotations

import logging
import re
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


def _sanitize_model_name(model: str) -> str:
    """Convert model name to valid SQLite table suffix."""
    # Replace non-alphanumeric chars with underscores, collapse multiple
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", model)
    return sanitized.strip("_").lower()


class VectorIndex:
    """
    Vector index using sqlite-vec.

    Stores embeddings for corpus sections and provides KNN search.
    Supports multiple embedding models via separate tables.
    """

    def __init__(self, index_path: Path, model: str | None = None, dimension: int = 768):
        """
        Initialize vector index.

        Args:
            index_path: Path to the SQLite database (same as corpus index)
            model: Embedding model name (required for building, optional for status)
            dimension: Embedding vector dimension
        """
        self._index_path = index_path
        self._model = model
        self._dimension = dimension
        self._conn: sqlite3.Connection | None = None
        self._vec_available = False

    @property
    def _table_name(self) -> str:
        """Get model-specific table name."""
        if self._model is None:
            raise ValueError("Model name required for table operations")
        return f"corpus_vectors_{_sanitize_model_name(self._model)}"

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection with sqlite-vec."""
        if self._conn is None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._index_path))
            self._conn.row_factory = sqlite3.Row

            # Try to load sqlite-vec extension
            try:
                self._conn.enable_load_extension(True)
                import sqlite_vec  # type: ignore[import-untyped,import-not-found,unused-ignore]

                sqlite_vec.load(self._conn)
                self._vec_available = True
                logger.debug("sqlite-vec extension loaded")
            except Exception as e:
                logger.warning(f"Failed to load sqlite-vec: {e}")
                self._vec_available = False

            self._ensure_schema()

        return self._conn

    def _ensure_schema(self) -> None:
        """Create or migrate metadata table (schema only, not model-specific tables)."""
        if not self._vec_available:
            return

        conn = self._conn
        if conn is None:
            return

        # Check if metadata table exists and has correct schema
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='corpus_embedding_meta'"
        )
        if cursor.fetchone() is not None:
            # Table exists, check if it has the new schema (model column)
            cursor = conn.execute("PRAGMA table_info(corpus_embedding_meta)")
            columns = {row[1] for row in cursor.fetchall()}
            if "model" not in columns:
                # Old schema (any version without model column) - drop and recreate
                logger.info("Migrating corpus_embedding_meta to new model-based schema")
                conn.execute("DROP TABLE corpus_embedding_meta")
                conn.commit()

        # Create metadata table with new schema (shared across all models)
        # Keys: model -> (dimension, indexed_at)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_embedding_meta (
                model TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (model, key)
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

    def get_all_embeddings(self) -> list[dict[str, Any]]:
        """Get status of all available embedding sets (all models)."""
        if not self.is_available:
            return []

        conn = self._get_connection()
        embeddings = []

        # Try to get models from metadata
        try:
            cursor = conn.execute("SELECT DISTINCT model FROM corpus_embedding_meta")
            models = [row["model"] for row in cursor.fetchall()]

            for model in models:
                table_name = f"corpus_vectors_{_sanitize_model_name(model)}"

                # Check if table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                if cursor.fetchone() is None:
                    continue

                # Count vectors in this table
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}")  # noqa: S608
                row = cursor.fetchone()
                vector_count = row["count"] if row else 0

                if vector_count == 0:
                    continue

                # Get metadata for this model
                cursor = conn.execute(
                    "SELECT key, value FROM corpus_embedding_meta WHERE model = ?",
                    (model,),
                )
                meta = {row["key"]: row["value"] for row in cursor.fetchall()}

                embeddings.append(
                    {
                        "model": model,
                        "dimension": int(meta.get("dimension", 0)),
                        "vector_count": vector_count,
                        "indexed_at": meta.get("indexed_at"),
                    }
                )
        except Exception as e:
            logger.debug(f"Error querying embeddings metadata: {e}")
            # If metadata table doesn't exist or has wrong schema, return empty
            pass

        return embeddings

    def get_status(self) -> dict[str, Any]:
        """Get vector index status for current model (or general status if no model)."""
        if not self.is_available:
            return {
                "available": False,
                "reason": "sqlite-vec not loaded",
            }

        conn = self._get_connection()

        # If no model specified, return general status
        if self._model is None:
            all_embeddings = self.get_all_embeddings()
            return {
                "available": True,
                "embeddings": all_embeddings,
                "embedding_count": len(all_embeddings),
            }

        # Check if our model's table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self._table_name,),
        )
        if cursor.fetchone() is None:
            return {
                "available": True,
                "vector_count": 0,
                "model": self._model,
                "dimension": self._dimension,
                "indexed_at": None,
            }

        # Count vectors
        cursor = conn.execute(f"SELECT COUNT(*) as count FROM {self._table_name}")  # noqa: S608
        row = cursor.fetchone()
        vector_count = row["count"] if row else 0

        # Get metadata for this model
        cursor = conn.execute(
            "SELECT key, value FROM corpus_embedding_meta WHERE model = ?",
            (self._model,),
        )
        meta = {row["key"]: row["value"] for row in cursor.fetchall()}

        return {
            "available": True,
            "vector_count": vector_count,
            "model": self._model,
            "dimension": int(meta.get("dimension", self._dimension)),
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

        # Set model and dimension from provider
        self._model = provider.model
        self._dimension = provider.dimension

        conn = self._get_connection()
        table_name = self._table_name

        # Ensure model-specific table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cursor.fetchone() is None:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE {table_name} USING vec0(
                    section_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self._dimension}]
                )
                """
            )
            conn.commit()

        # Check if rebuild needed
        if not force:
            cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}")  # noqa: S608
            row = cursor.fetchone()
            existing = row["count"] if row else 0
            if existing > 0:
                logger.info(
                    f"Vector index already has {existing} vectors for {self._model}, skipping"
                )
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
            conn.execute(f"DELETE FROM {table_name}")  # noqa: S608

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
                        f"INSERT INTO {table_name} (section_id, embedding) VALUES (?, ?)",  # noqa: S608
                        (row["id"], vector_bytes),
                    )

                total_embedded += len(batch)
                logger.debug(f"Embedded {total_embedded}/{len(sections)} sections")

            except Exception as e:
                logger.error(f"Failed to embed batch {i}: {e}")
                raise

        # Update metadata for this model
        from datetime import datetime

        conn.execute(
            "DELETE FROM corpus_embedding_meta WHERE model = ?",
            (self._model,),
        )
        conn.execute(
            """
            INSERT INTO corpus_embedding_meta (model, key, value)
            VALUES (?, 'dimension', ?), (?, 'indexed_at', ?)
            """,
            (self._model, str(self._dimension), self._model, datetime.now().isoformat()),
        )

        conn.commit()
        logger.info(f"Embedded {total_embedded} sections with {self._model} ({self._dimension}d)")
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

        if self._model is None:
            logger.warning("No model specified for vector search")
            return []

        conn = self._get_connection()
        table_name = self._table_name

        # Check if table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cursor.fetchone() is None:
            return []

        query_bytes = serialize_float32(query_embedding)

        # KNN search using sqlite-vec
        cursor = conn.execute(
            f"""
            SELECT
                v.section_id,
                v.distance,
                s.heading,
                s.content,
                f.path,
                f.title,
                f.cluster
            FROM {table_name} v
            JOIN corpus_sections s ON v.section_id = s.id
            JOIN corpus_files f ON s.file_id = f.id
            WHERE v.embedding MATCH ?
            ORDER BY v.distance
            LIMIT ?
            """,  # noqa: S608
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
        """Check if vector index has been built (for current model or any model)."""
        if not self.is_available:
            return False

        conn = self._get_connection()

        # If no model specified, check if ANY model has vectors
        if self._model is None:
            return len(self.get_all_embeddings()) > 0

        table_name = self._table_name

        # Check if table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cursor.fetchone() is None:
            return False

        cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}")  # noqa: S608
        row = cursor.fetchone()
        return bool(row["count"] > 0) if row else False

    def get_first_available_model(self) -> str | None:
        """Get the first available model with vectors (for auto-detection)."""
        embeddings = self.get_all_embeddings()
        if embeddings:
            model: str = embeddings[0]["model"]
            return model
        return None

    def set_model(self, model: str, dimension: int) -> None:
        """Set the model and dimension for this index (for search operations)."""
        self._model = model
        self._dimension = dimension
