"""
Tests for hybrid search.

Tests the Reciprocal Rank Fusion algorithm and HybridSearcher.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.corpus.hybrid_search import (
    HybridSearcher,
    SearchResult,
    reciprocal_rank_fusion,
)
from questfoundry.runtime.corpus.index import CorpusIndex


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_basic_result(self):
        """SearchResult should store all fields."""
        result = SearchResult(
            section_id=1,
            heading="Test Heading",
            content="Test content",
            source_file="test.md",
            title="Test Title",
            cluster="test-cluster",
            combined_score=0.5,
        )

        assert result.section_id == 1
        assert result.heading == "Test Heading"
        assert result.content == "Test content"
        assert result.combined_score == 0.5

    def test_optional_fields(self):
        """SearchResult should allow optional rank fields."""
        result = SearchResult(
            section_id=1,
            heading="Test",
            content="Content",
            source_file="test.md",
            title="Title",
            cluster="cluster",
            combined_score=0.5,
            keyword_rank=1,
            vector_rank=2,
            keyword_score=0.8,
            vector_score=0.6,
        )

        assert result.keyword_rank == 1
        assert result.vector_rank == 2
        assert result.keyword_score == 0.8
        assert result.vector_score == 0.6

    def test_to_dict(self):
        """to_dict should return tool-friendly output."""
        result = SearchResult(
            section_id=1,
            heading="Test Heading",
            content="Test content",
            source_file="test.md",
            title="Test Title",
            cluster="test-cluster",
            combined_score=0.5678,
        )

        d = result.to_dict()

        assert d["heading"] == "Test Heading"
        assert d["content"] == "Test content"
        assert d["source_file"] == "test.md"
        assert d["title"] == "Test Title"
        assert d["cluster"] == "test-cluster"
        assert d["relevance_score"] == 0.5678
        # section_id not in dict (internal field)
        assert "section_id" not in d


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_single_list(self):
        """RRF with single list should preserve order."""
        items = [
            {"section_id": 1, "content": "first"},
            {"section_id": 2, "content": "second"},
            {"section_id": 3, "content": "third"},
        ]

        result = reciprocal_rank_fusion([items])

        assert len(result) == 3
        assert result[0]["section_id"] == 1
        assert result[1]["section_id"] == 2
        assert result[2]["section_id"] == 3

    def test_two_identical_lists(self):
        """RRF with identical lists should boost scores."""
        items = [
            {"section_id": 1, "content": "first"},
            {"section_id": 2, "content": "second"},
        ]

        result = reciprocal_rank_fusion([items, items])

        # Items should have higher RRF scores (appears in both)
        assert len(result) == 2
        assert result[0]["section_id"] == 1
        assert result[0]["rrf_score"] > 0

    def test_two_different_lists(self):
        """RRF should merge different lists."""
        list1 = [
            {"section_id": 1, "content": "a"},
            {"section_id": 2, "content": "b"},
        ]
        list2 = [
            {"section_id": 3, "content": "c"},
            {"section_id": 1, "content": "a"},
        ]

        result = reciprocal_rank_fusion([list1, list2])

        # Section 1 appears in both lists, should rank highest
        assert len(result) == 3
        assert result[0]["section_id"] == 1

    def test_rrf_score_calculation(self):
        """RRF scores should follow 1/(k+rank) formula."""
        items = [{"section_id": 1, "content": "only"}]

        # With k=60 (default) and rank=1: 1/(60+1) = 0.0164
        result = reciprocal_rank_fusion([items], k=60)

        expected_score = 1.0 / (60 + 1)
        assert abs(result[0]["rrf_score"] - expected_score) < 0.0001

    def test_custom_k_value(self):
        """RRF should use custom k value."""
        items = [{"section_id": 1, "content": "only"}]

        result = reciprocal_rank_fusion([items], k=10)

        expected_score = 1.0 / (10 + 1)
        assert abs(result[0]["rrf_score"] - expected_score) < 0.0001

    def test_custom_id_key(self):
        """RRF should use custom id key."""
        items = [
            {"custom_id": "a", "content": "first"},
            {"custom_id": "b", "content": "second"},
        ]

        result = reciprocal_rank_fusion([items], id_key="custom_id")

        assert len(result) == 2

    def test_fallback_id_generation(self):
        """RRF should generate ID from source_file + heading if no section_id."""
        items = [
            {"source_file": "test.md", "heading": "Section 1", "content": "a"},
            {"source_file": "test.md", "heading": "Section 2", "content": "b"},
        ]

        result = reciprocal_rank_fusion([items])

        # Should not raise, items should be uniquely identified
        assert len(result) == 2


class TestHybridSearcher:
    """Tests for HybridSearcher class."""

    @pytest.fixture
    def corpus_index(self, tmp_path: Path) -> CorpusIndex:
        """Create a test corpus index."""
        corpus_dir = tmp_path / "knowledge" / "corpus"
        corpus_dir.mkdir(parents=True)

        file1 = corpus_dir / "test.md"
        file1.write_text(
            dedent("""\
            ---
            title: Test File
            summary: This is a test summary that is long enough for validation.
            topics:
              - test-topic
              - another-topic
              - third-topic
            cluster: narrative-structure
            ---

            # Test File

            Test content.

            ## First Section

            First section content about dialogue.

            ## Second Section

            Second section about pacing.
        """)
        )

        index_path = CorpusIndex.get_index_path(tmp_path)
        index = CorpusIndex(index_path)
        index.build(corpus_dir)

        yield index

        index.close()

    def test_keyword_only_mode(self, corpus_index: CorpusIndex):
        """HybridSearcher without vector index uses keyword only."""
        searcher = HybridSearcher(
            corpus_index=corpus_index,
            vector_index=None,
            embedding_provider=None,
        )

        assert searcher.vector_enabled is False
        assert searcher.get_search_method() == "keyword"

    def test_hybrid_mode_with_mocks(self, corpus_index: CorpusIndex):
        """HybridSearcher with mocked vector index reports hybrid mode."""
        mock_vector_index = MagicMock()
        mock_vector_index.is_available = True
        mock_vector_index.has_vectors.return_value = True

        mock_provider = MagicMock()

        searcher = HybridSearcher(
            corpus_index=corpus_index,
            vector_index=mock_vector_index,
            embedding_provider=mock_provider,
        )

        assert searcher.vector_enabled is True
        assert searcher.get_search_method() == "hybrid"

    @pytest.mark.asyncio
    async def test_search_keyword_fallback(self, corpus_index: CorpusIndex):
        """search should fall back to keyword if no vector index."""
        searcher = HybridSearcher(
            corpus_index=corpus_index,
            vector_index=None,
            embedding_provider=None,
        )

        results = await searcher.search("dialogue", max_results=3)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        # Should have keyword rank but not vector rank
        assert results[0].keyword_rank is not None

    @pytest.mark.asyncio
    async def test_search_hybrid_with_mocks(self, corpus_index: CorpusIndex):
        """search should combine keyword and vector results."""
        # Mock vector index
        mock_vector_index = MagicMock()
        mock_vector_index.is_available = True
        mock_vector_index.has_vectors.return_value = True
        mock_vector_index.vector_search.return_value = [
            {
                "section_id": 1,
                "heading": "First Section",
                "content": "First section content about dialogue.",
                "source_file": "test.md",
                "title": "Test File",
                "cluster": "narrative-structure",
                "similarity_score": 0.9,
            }
        ]

        # Mock embedding provider
        mock_provider = AsyncMock()
        mock_provider.embed_single = AsyncMock(return_value=[0.1] * 768)

        searcher = HybridSearcher(
            corpus_index=corpus_index,
            vector_index=mock_vector_index,
            embedding_provider=mock_provider,
        )

        results = await searcher.search("dialogue", max_results=3)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_handles_vector_error(self, corpus_index: CorpusIndex):
        """search should fall back to keyword if vector search fails."""
        # Mock vector index that raises error
        mock_vector_index = MagicMock()
        mock_vector_index.is_available = True
        mock_vector_index.has_vectors.return_value = True
        mock_vector_index.vector_search.side_effect = Exception("Vector search failed")

        # Mock embedding provider that raises error
        mock_provider = AsyncMock()
        mock_provider.embed_single = AsyncMock(side_effect=Exception("Embed failed"))

        searcher = HybridSearcher(
            corpus_index=corpus_index,
            vector_index=mock_vector_index,
            embedding_provider=mock_provider,
        )

        # Should not raise, should fall back to keyword
        results = await searcher.search("dialogue", max_results=3)

        assert len(results) > 0

    def test_convert_keyword_results(self, corpus_index: CorpusIndex):
        """_convert_keyword_results should create SearchResult objects."""
        searcher = HybridSearcher(
            corpus_index=corpus_index,
            vector_index=None,
            embedding_provider=None,
        )

        keyword_results = [
            {
                "heading": "Test",
                "content": "Content",
                "source_file": "test.md",
                "title": "Title",
                "cluster": "cluster",
                "relevance_score": 0.5,
            }
        ]

        results = searcher._convert_keyword_results(keyword_results, max_results=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].heading == "Test"
        assert results[0].keyword_rank == 1
        assert results[0].vector_rank is None
