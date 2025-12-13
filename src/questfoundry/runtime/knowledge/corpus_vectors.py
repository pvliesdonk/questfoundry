"""Corpus vector store for semantic search over studio knowledge.

This module provides vector-based semantic search over corpus knowledge entries
(genre conventions, writing craft guides, etc.). It uses sentence-transformers
for embeddings and sqlite-vec for vector storage/search.

The corpus is STUDIO knowledge (shared across all projects), not project
knowledge. Vectors are cached in ~/.questfoundry/cache/ and automatically
rebuilt when corpus content changes.

Dependencies (optional [rag] extra):
- sentence-transformers: For computing text embeddings
- sqlite-vec: SQLite extension for vector similarity search

Falls back gracefully to keyword search if dependencies aren't installed.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check for optional dependencies
VECTOR_SEARCH_AVAILABLE = False
_import_error: str | None = None

try:
    import sqlite_vec
    from sentence_transformers import SentenceTransformer

    VECTOR_SEARCH_AVAILABLE = True
except ImportError as e:
    _import_error = str(e)
    logger.debug(f"Vector search not available: {e}")


# Default embedding model - good balance of speed and quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Chunking parameters
CHUNK_SIZE = 500  # Target characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks


def get_cache_dir() -> Path:
    """Get the QuestFoundry cache directory."""
    cache_dir = Path.home() / ".questfoundry" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_corpus_hash(corpus_dir: Path) -> str:
    """Compute a hash of all corpus files for cache invalidation."""
    hasher = hashlib.sha256()

    if not corpus_dir.exists():
        return "empty"

    # Sort files for deterministic ordering
    files = sorted(corpus_dir.glob("**/*.md"))
    for file_path in files:
        # Include filename and content in hash
        hasher.update(file_path.name.encode())
        hasher.update(file_path.read_bytes())

    return hasher.hexdigest()[:16]  # Short hash is sufficient


class CorpusVectorStore:
    """Vector store for corpus knowledge with automatic indexing.

    Provides semantic search over corpus markdown files using
    sentence-transformers embeddings and sqlite-vec for storage.

    Usage:
        store = CorpusVectorStore(corpus_dir)
        store.ensure_indexed()  # Build/update index if needed
        results = store.search("chapter length conventions", k=3)
    """

    def __init__(
        self,
        corpus_dir: Path,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
    ):
        """Initialize the corpus vector store.

        Args:
            corpus_dir: Path to the corpus markdown files
            model_name: Sentence transformer model to use
            cache_dir: Where to store the vector database (default: ~/.questfoundry/cache)
        """
        self.corpus_dir = Path(corpus_dir)
        self.model_name = model_name
        self.cache_dir = cache_dir or get_cache_dir()

        self._model: Any = None
        self._db_path: Path | None = None
        self._conn: sqlite3.Connection | None = None
        self._indexed = False

    @property
    def is_available(self) -> bool:
        """Check if vector search is available (dependencies installed)."""
        return VECTOR_SEARCH_AVAILABLE

    @property
    def unavailable_reason(self) -> str | None:
        """Get reason why vector search is unavailable."""
        return _import_error

    def ensure_indexed(self, force: bool = False) -> bool:
        """Ensure the corpus is indexed, building if necessary.

        Args:
            force: Force rebuild even if cache is valid

        Returns:
            True if index is ready, False if unavailable
        """
        if not VECTOR_SEARCH_AVAILABLE:
            logger.info(
                "Vector search not available. Install with: "
                "uv pip install questfoundry[rag]"
            )
            return False

        # Compute current corpus hash
        corpus_hash = compute_corpus_hash(self.corpus_dir)
        self._db_path = self.cache_dir / f"corpus_vectors_{corpus_hash}.db"

        # Check if we need to rebuild
        if not force and self._db_path.exists():
            logger.debug(f"Using cached corpus vectors: {self._db_path}")
            self._indexed = True
            return True

        # Need to build index
        logger.info(
            "Indexing corpus for vector search... "
            "(this is a one-time operation)"
        )

        try:
            self._build_index()
            self._indexed = True
            logger.info(f"Corpus indexed successfully: {self._db_path}")

            # Clean up old cache files
            self._cleanup_old_caches(corpus_hash)

            return True
        except Exception as e:
            logger.error(f"Failed to build corpus index: {e}")
            return False

    def _build_index(self) -> None:
        """Build the vector index from corpus files."""
        if not VECTOR_SEARCH_AVAILABLE:
            raise RuntimeError("Vector search dependencies not installed")

        # Load the embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)

        # Create database
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))

        # Load sqlite-vec extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Create schema
        conn.executescript("""
            -- Chunk text storage
            CREATE TABLE IF NOT EXISTS corpus_chunks (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                UNIQUE(file_name, chunk_idx)
            );

            -- Metadata
            CREATE TABLE IF NOT EXISTS corpus_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)

        # Create vector table with sqlite-vec
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS corpus_embeddings
            USING vec0(embedding float[{EMBEDDING_DIM}])
        """)

        # Process each corpus file
        corpus_files = list(self.corpus_dir.glob("**/*.md"))
        total_chunks = 0

        for file_path in corpus_files:
            logger.info(f"  Processing: {file_path.name}")
            content = file_path.read_text(encoding="utf-8")
            chunks = self._chunk_text(content)

            for idx, chunk in enumerate(chunks):
                # Insert chunk text
                cursor = conn.execute(
                    "INSERT INTO corpus_chunks (file_name, chunk_idx, chunk_text) "
                    "VALUES (?, ?, ?)",
                    (file_path.name, idx, chunk),
                )
                chunk_id = cursor.lastrowid

                # Compute and insert embedding
                embedding = self._model.encode(chunk, normalize_embeddings=True)
                conn.execute(
                    "INSERT INTO corpus_embeddings (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, embedding.tobytes()),
                )

                total_chunks += 1

        # Store metadata
        conn.execute(
            "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES (?, ?)",
            ("model", self.model_name),
        )
        conn.execute(
            "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES (?, ?)",
            ("chunk_count", str(total_chunks)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES (?, ?)",
            ("file_count", str(len(corpus_files))),
        )

        conn.commit()
        conn.close()

        logger.info(f"  Indexed {total_chunks} chunks from {len(corpus_files)} files")

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Tries to split on paragraph boundaries, falling back to
        sentence boundaries, then fixed-size chunks.
        """
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size, start new chunk
            if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: keep last portion of current chunk
                overlap_start = max(0, len(current_chunk) - CHUNK_OVERLAP)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _cleanup_old_caches(self, current_hash: str) -> None:
        """Remove old cache files with different hashes."""
        for old_db in self.cache_dir.glob("corpus_vectors_*.db"):
            if current_hash not in old_db.name:
                logger.debug(f"Removing outdated cache: {old_db.name}")
                old_db.unlink()

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Search the corpus for relevant chunks.

        Args:
            query: Natural language search query
            k: Number of results to return

        Returns:
            List of dicts with: file_name, chunk_text, score
        """
        if not self._indexed or not VECTOR_SEARCH_AVAILABLE:
            raise RuntimeError("Corpus not indexed. Call ensure_indexed() first.")

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        # Encode query
        query_embedding = self._model.encode(query, normalize_embeddings=True)

        # Open connection if needed
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Vector similarity search
        results = conn.execute(
            """
            SELECT
                c.file_name,
                c.chunk_text,
                e.distance
            FROM corpus_embeddings e
            JOIN corpus_chunks c ON c.id = e.rowid
            WHERE e.embedding MATCH ?
            ORDER BY e.distance
            LIMIT ?
            """,
            (query_embedding.tobytes(), k),
        ).fetchall()

        conn.close()

        return [
            {
                "file_name": row[0],
                "chunk_text": row[1],
                "score": 1.0 - row[2],  # Convert distance to similarity
            }
            for row in results
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the indexed corpus."""
        if not self._indexed or self._db_path is None:
            return {"indexed": False}

        conn = sqlite3.connect(str(self._db_path))
        meta = dict(conn.execute("SELECT key, value FROM corpus_meta").fetchall())
        conn.close()

        return {
            "indexed": True,
            "db_path": str(self._db_path),
            "model": meta.get("model", "unknown"),
            "chunk_count": int(meta.get("chunk_count", 0)),
            "file_count": int(meta.get("file_count", 0)),
        }


# Singleton for the default corpus store
_default_store: CorpusVectorStore | None = None


def get_corpus_vector_store(corpus_dir: Path | None = None) -> CorpusVectorStore:
    """Get the default corpus vector store.

    Args:
        corpus_dir: Path to corpus directory (default: auto-detect from domain-v4)

    Returns:
        CorpusVectorStore instance
    """
    global _default_store

    if _default_store is None:
        if corpus_dir is None:
            # Try to find domain-v4 corpus
            # Look relative to this file's location
            module_dir = Path(__file__).parent
            possible_paths = [
                module_dir.parents[3] / "domain-v4" / "knowledge" / "corpus",
                Path.cwd() / "domain-v4" / "knowledge" / "corpus",
            ]
            for path in possible_paths:
                if path.exists():
                    corpus_dir = path
                    break

            if corpus_dir is None:
                # Create empty path - will just have no content
                corpus_dir = possible_paths[0]

        _default_store = CorpusVectorStore(corpus_dir)

    return _default_store
