"""
Consult Corpus tool implementation.

RAG-style search in the knowledge base with:
- Vector search (when embeddings available)
- Keyword fallback (always available)

The corpus contains craft guidance, genre conventions, and writing patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)

# Flag for vector search availability
# Set to True when sqlite-vec and embeddings are configured
VECTOR_SEARCH_AVAILABLE = False

# Common stop words to filter out during tokenization
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }
)


@dataclass
class CorpusExcerpt:
    """An excerpt from the corpus."""

    excerpt: str
    source_file: str
    relevance_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "excerpt": self.excerpt,
            "source_file": self.source_file,
            "relevance_score": round(self.relevance_score, 3),
        }


@register_tool("consult_corpus")
class ConsultCorpusTool(BaseTool):
    """
    Search the craft corpus for writing guidance and genre conventions.

    Uses vector search when available, falls back to keyword matching.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._corpus_cache: dict[str, str] | None = None

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute corpus search."""
        query = args.get("query", "")
        max_results = args.get("max_results", 3)

        if not query.strip():
            return ToolResult(
                success=False,
                data={"success": False, "excerpts": []},
                error="Query cannot be empty",
            )

        # Load corpus if needed
        corpus = self._load_corpus()

        if not corpus:
            return ToolResult(
                success=False,
                data={"success": False, "excerpts": []},
                error="No corpus files found",
            )

        # Choose search method
        if VECTOR_SEARCH_AVAILABLE:
            excerpts = await self._vector_search(query, corpus, max_results)
            search_method = "vector"
        else:
            excerpts = self._keyword_search(query, corpus, max_results)
            search_method = "keyword"

        return ToolResult(
            success=True,
            data={
                "success": True,
                "source": "domain_corpus",
                "search_method": search_method,
                "excerpt_count": len(excerpts),
                "excerpts": [e.to_dict() for e in excerpts],
            },
        )

    def _load_corpus(self) -> dict[str, str]:
        """
        Load corpus files from domain knowledge directory.

        Returns dict mapping filename to content.
        """
        if self._corpus_cache is not None:
            return self._corpus_cache

        corpus: dict[str, str] = {}
        domain_path = self._context.domain_path

        if not domain_path:
            logger.warning("No domain_path configured for corpus search")
            return corpus

        domain_path = Path(domain_path)
        knowledge_dirs = [
            domain_path / "knowledge",
            domain_path / "corpus",
        ]

        for knowledge_dir in knowledge_dirs:
            if not knowledge_dir.exists():
                continue

            # Load markdown and text files
            for ext in ["*.md", "*.txt"]:
                for file_path in knowledge_dir.rglob(ext):
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        relative_path = file_path.relative_to(domain_path)
                        corpus[str(relative_path)] = content
                    except Exception as e:
                        logger.warning(f"Failed to load corpus file {file_path}: {e}")

        self._corpus_cache = corpus
        logger.debug(f"Loaded {len(corpus)} corpus files")
        return corpus

    def _keyword_search(
        self, query: str, corpus: dict[str, str], max_results: int
    ) -> list[CorpusExcerpt]:
        """
        Keyword-based search with relevance scoring.

        Score formula: (word_coverage * 0.7) + (density * 0.3)
        """
        # Tokenize query
        query_words = set(self._tokenize(query.lower()))

        if not query_words:
            return []

        candidates: list[tuple[float, str, str]] = []

        for source_file, content in corpus.items():
            # Find relevant excerpts in content
            excerpts = self._extract_excerpts(content, query_words)

            for excerpt, score in excerpts:
                candidates.append((score, source_file, excerpt))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Return top results
        results = []
        for score, source_file, excerpt in candidates[:max_results]:
            results.append(
                CorpusExcerpt(
                    excerpt=excerpt,
                    source_file=source_file,
                    relevance_score=min(score, 1.0),
                )
            )

        return results

    def _extract_excerpts(self, content: str, query_words: set[str]) -> list[tuple[str, float]]:
        """
        Extract relevant excerpts from content.

        Returns list of (excerpt, score) tuples.
        """
        excerpts = []

        # Split into paragraphs
        paragraphs = re.split(r"\n\n+", content)

        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip short paragraphs
                continue

            # Tokenize paragraph
            para_words = self._tokenize(para.lower())
            if not para_words:
                continue

            # Calculate word coverage
            matching_words = query_words & set(para_words)
            word_coverage = len(matching_words) / len(query_words) if query_words else 0

            # Calculate density (how concentrated query words are)
            density = len(matching_words) / len(para_words) if para_words else 0

            # Combined score
            score = (word_coverage * 0.7) + (density * 0.3)

            if score > 0.1:  # Minimum relevance threshold
                # Truncate excerpt if too long
                if len(para) > 500:
                    para = para[:497] + "..."
                excerpts.append((para, score))

        return excerpts

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words, filtering stop words."""
        # Simple word tokenization
        words = re.findall(r"\b[a-z]{2,}\b", text)
        # Remove common stop words (using module-level constant)
        return [w for w in words if w not in _STOP_WORDS]

    async def _vector_search(
        self, query: str, corpus: dict[str, str], max_results: int
    ) -> list[CorpusExcerpt]:
        """
        Vector-based semantic search.

        TODO: Implement when sqlite-vec and embedding model are configured.
        Currently falls back to keyword search.
        """
        # Placeholder for vector search implementation
        # When implemented, this would:
        # 1. Embed the query using an embedding model
        # 2. Search the vector index for similar embeddings
        # 3. Return excerpts with cosine similarity scores

        logger.debug("Vector search not available, using keyword fallback")
        return self._keyword_search(query, corpus, max_results)


# =============================================================================
# Vector Search Infrastructure (TODO: Phase 3 or later)
# =============================================================================

# TODO: Implement vector search when ready
# Required components:
# 1. Embedding model (e.g., sentence-transformers)
# 2. sqlite-vec extension for vector storage
# 3. Index building during domain load
# 4. Query embedding and similarity search
#
# Example structure:
#
# class VectorIndex:
#     def __init__(self, db_path: Path):
#         self._conn = sqlite3.connect(str(db_path))
#         self._conn.enable_load_extension(True)
#         self._conn.load_extension("vec0")
#
#     def build_index(self, corpus: dict[str, str], embedder):
#         # Chunk documents, embed, store in sqlite-vec
#         pass
#
#     def search(self, query_embedding: list[float], limit: int):
#         # KNN search using sqlite-vec
#         pass
