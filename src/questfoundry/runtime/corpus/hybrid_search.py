"""
Hybrid search combining keyword and vector search.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
search methods for improved relevance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.corpus.embeddings import EmbeddingProvider
    from questfoundry.runtime.corpus.index import CorpusIndex
    from questfoundry.runtime.corpus.vector_index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result with combined scoring."""

    section_id: int
    heading: str
    content: str
    source_file: str
    title: str
    cluster: str
    combined_score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    keyword_score: float | None = None
    vector_score: float | None = None

    def to_dict(self, rank: int | None = None) -> dict[str, Any]:
        """Convert to dictionary for tool output.

        Args:
            rank: Optional position in results (1-indexed). If provided,
                  included in output for clarity.

        Returns:
            Dictionary with source, section, content. Relevance scores are
            intentionally omitted as they confuse agents (RRF scores ~0.016
            look like "1.6% relevant" which is misleading).
        """
        result: dict[str, Any] = {
            "source": self.title,
            "section": self.heading,
            "content": self.content,
        }
        if rank is not None:
            result["rank"] = rank
        return result


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    id_key: str = "section_id",
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for all lists where item appears

    Args:
        ranked_lists: List of ranked result lists
        id_key: Key to use as unique identifier
        k: RRF constant (default 60, as in original paper)

    Returns:
        Combined and re-ranked list
    """
    scores: dict[Any, float] = {}
    items: dict[Any, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item.get(id_key) or item.get("source_file", "") + item.get("heading", "")
            rrf_score = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0) + rrf_score
            if item_id not in items:
                items[item_id] = item

    # Sort by combined RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build result list with scores
    results = []
    for item_id in sorted_ids:
        item = items[item_id].copy()
        item["rrf_score"] = scores[item_id]
        results.append(item)

    return results


class HybridSearcher:
    """
    Hybrid search combining keyword and vector search.

    Uses both the SQLite corpus index (keyword search) and
    the vector index (semantic search) with RRF fusion.
    """

    def __init__(
        self,
        corpus_index: CorpusIndex,
        vector_index: VectorIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """
        Initialize hybrid searcher.

        Args:
            corpus_index: The keyword search index
            vector_index: Optional vector index for semantic search
            embedding_provider: Provider for query embedding
        """
        self._corpus_index = corpus_index
        self._vector_index = vector_index
        self._embedding_provider = embedding_provider

    @property
    def vector_enabled(self) -> bool:
        """Check if vector search is available."""
        return (
            self._vector_index is not None
            and self._vector_index.is_available
            and self._vector_index.has_vectors()
            and self._embedding_provider is not None
        )

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query text
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects with combined scores
        """
        # Get keyword results
        keyword_results = self._corpus_index.keyword_search(query, max_results=max_results * 2)

        # If vector search not available, return keyword results only
        if not self.vector_enabled:
            return self._convert_keyword_results(keyword_results, max_results)

        # Get vector results (we know these are not None due to vector_enabled check)
        assert self._embedding_provider is not None
        assert self._vector_index is not None
        try:
            query_embedding = await self._embedding_provider.embed_single(query)
            vector_results = self._vector_index.vector_search(
                query_embedding, limit=max_results * 2
            )
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to keyword: {e}")
            return self._convert_keyword_results(keyword_results, max_results)

        # Combine with RRF
        combined = reciprocal_rank_fusion([vector_results, keyword_results])

        # Convert to SearchResult objects
        results = []
        for item in combined[:max_results]:
            # Find original ranks
            keyword_rank = next(
                (
                    j + 1
                    for j, kr in enumerate(keyword_results)
                    if kr.get("source_file") == item.get("source_file")
                    and kr.get("heading") == item.get("heading")
                ),
                None,
            )
            vector_rank = next(
                (
                    j + 1
                    for j, vr in enumerate(vector_results)
                    if vr.get("section_id") == item.get("section_id")
                ),
                None,
            )

            results.append(
                SearchResult(
                    section_id=item.get("section_id", 0),
                    heading=item.get("heading", ""),
                    content=item.get("content", ""),
                    source_file=item.get("source_file", ""),
                    title=item.get("title", ""),
                    cluster=item.get("cluster", ""),
                    combined_score=item.get("rrf_score", 0),
                    keyword_rank=keyword_rank,
                    vector_rank=vector_rank,
                    keyword_score=item.get("relevance_score"),
                    vector_score=item.get("similarity_score"),
                )
            )

        return results

    def _convert_keyword_results(
        self, keyword_results: list[dict[str, Any]], max_results: int
    ) -> list[SearchResult]:
        """Convert keyword results to SearchResult objects."""
        results = []
        for i, item in enumerate(keyword_results[:max_results]):
            results.append(
                SearchResult(
                    section_id=0,  # Not available from keyword search
                    heading=item.get("heading", ""),
                    content=item.get("content", ""),
                    source_file=item.get("source_file", ""),
                    title=item.get("title", ""),
                    cluster=item.get("cluster", ""),
                    combined_score=item.get("relevance_score", 0),
                    keyword_rank=i + 1,
                    keyword_score=item.get("relevance_score"),
                )
            )
        return results

    def get_search_method(self) -> str:
        """Get the search method being used."""
        if self.vector_enabled:
            return "hybrid"
        return "keyword"
