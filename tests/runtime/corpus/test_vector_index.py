"""
Tests for vector index.

Tests the vector storage and search functionality.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.corpus.index import CorpusIndex
from questfoundry.runtime.corpus.vector_index import (
    VectorIndex,
    deserialize_float32,
    serialize_float32,
)


class TestVectorSerialization:
    """Tests for vector serialization functions."""

    def test_serialize_float32(self):
        """serialize_float32 should convert list to bytes."""
        vector = [1.0, 2.0, 3.0]
        result = serialize_float32(vector)

        assert isinstance(result, bytes)
        assert len(result) == 12  # 3 floats * 4 bytes each

    def test_deserialize_float32(self):
        """deserialize_float32 should convert bytes to list."""
        vector = [1.0, 2.0, 3.0]
        serialized = serialize_float32(vector)
        result = deserialize_float32(serialized)

        assert result == vector

    def test_roundtrip(self):
        """Serialization should be reversible."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        serialized = serialize_float32(original)
        restored = deserialize_float32(serialized)

        # Compare with tolerance for floating point
        for a, b in zip(original, restored, strict=True):
            assert abs(a - b) < 1e-6


class TestVectorIndex:
    """Tests for VectorIndex class."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database path."""
        return tmp_path / "test_index.sqlite"

    @pytest.fixture
    def corpus_with_index(self, tmp_path: Path) -> tuple[Path, CorpusIndex]:
        """Create a corpus with index for testing."""
        # Create corpus files
        corpus_dir = tmp_path / "knowledge" / "corpus"
        corpus_dir.mkdir(parents=True)

        file1 = corpus_dir / "test1.md"
        file1.write_text(
            dedent("""\
            ---
            title: Test File One
            summary: This is a test summary for file one that is long enough.
            topics:
              - test-topic-one
              - test-topic-two
              - test-topic-three
            cluster: narrative-structure
            ---

            # Test File One

            Content for test file one.

            ## First Section

            First section content goes here.

            ## Second Section

            Second section content goes here.
        """)
        )

        file2 = corpus_dir / "test2.md"
        file2.write_text(
            dedent("""\
            ---
            title: Test File Two
            summary: This is a test summary for file two that is long enough.
            topics:
              - test-topic-three
              - test-topic-four
              - test-topic-five
            cluster: prose-and-language
            ---

            # Test File Two

            Content for test file two.

            ## Another Section

            Another section content here.
        """)
        )

        # Build corpus index
        index_path = CorpusIndex.get_index_path(tmp_path)
        index = CorpusIndex(index_path)
        index.build(corpus_dir)

        return index_path, index

    def test_init_creates_vector_index(self, temp_db: Path):
        """VectorIndex should initialize without errors."""
        vector_index = VectorIndex(temp_db)

        # Won't have sqlite-vec available in most test environments
        # but should still initialize
        assert vector_index is not None
        assert vector_index.dimension == 768

    def test_custom_dimension(self, temp_db: Path):
        """VectorIndex should accept custom dimension."""
        vector_index = VectorIndex(temp_db, dimension=1536)

        assert vector_index.dimension == 1536

    def test_close(self, temp_db: Path):
        """close() should release connection."""
        vector_index = VectorIndex(temp_db)
        _ = vector_index.is_available  # Force connection

        vector_index.close()

        assert vector_index._conn is None

    def test_get_status_unavailable(self, temp_db: Path):
        """get_status should report unavailable if sqlite-vec not loaded."""
        vector_index = VectorIndex(temp_db)

        # Force connection to check availability
        _ = vector_index.is_available

        if not vector_index._vec_available:
            status = vector_index.get_status()

            assert status["available"] is False
            assert "reason" in status

        vector_index.close()

    def test_has_vectors_returns_false_initially(self, temp_db: Path):
        """has_vectors should return False before building."""
        vector_index = VectorIndex(temp_db)

        if vector_index.is_available:
            assert vector_index.has_vectors() is False
        else:
            # If sqlite-vec not available, has_vectors returns False
            assert vector_index.has_vectors() is False

        vector_index.close()


class TestVectorIndexWithSqliteVec:
    """Tests that require sqlite-vec (skipped if not available)."""

    @pytest.fixture
    def vector_index_with_data(self, tmp_path: Path) -> tuple[VectorIndex, CorpusIndex]:
        """Create vector index with test data."""
        # Create corpus files
        corpus_dir = tmp_path / "knowledge" / "corpus"
        corpus_dir.mkdir(parents=True)

        file1 = corpus_dir / "dialogue.md"
        file1.write_text(
            dedent("""\
            ---
            title: Dialogue Craft
            summary: This is about writing dialogue for interactive fiction games.
            topics:
              - dialogue
              - character-voice
              - conversation
            cluster: prose-and-language
            ---

            # Dialogue Craft

            Writing compelling dialogue.

            ## Voice and Tone

            Each character needs a distinct voice.
        """)
        )

        # Build corpus index
        index_path = CorpusIndex.get_index_path(tmp_path)
        corpus_index = CorpusIndex(index_path)
        corpus_index.build(corpus_dir)

        # Create vector index
        vector_index = VectorIndex(index_path, dimension=4)

        return vector_index, corpus_index

    def test_build_vectors_requires_sqlite_vec(self, vector_index_with_data):
        """build_vectors should fail if sqlite-vec not available."""
        vector_index, corpus_index = vector_index_with_data

        if not vector_index.is_available:
            with pytest.raises(RuntimeError, match="sqlite-vec not available"):
                import asyncio

                mock_provider = MagicMock()
                asyncio.run(vector_index.build_vectors(mock_provider))

        corpus_index.close()
        vector_index.close()

    def test_vector_search_returns_empty_if_unavailable(self, vector_index_with_data):
        """vector_search should return empty if sqlite-vec not available."""
        vector_index, corpus_index = vector_index_with_data

        if not vector_index.is_available:
            results = vector_index.vector_search([0.1, 0.2, 0.3, 0.4])
            assert results == []

        corpus_index.close()
        vector_index.close()
