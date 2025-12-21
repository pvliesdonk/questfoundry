"""Tests for corpus SQLite index."""

from pathlib import Path
from textwrap import dedent

import pytest

from questfoundry.runtime.corpus.index import CorpusIndex


def create_test_corpus_file(path: Path, title: str, cluster: str, content: str = "") -> None:
    """Helper to create a test corpus file."""
    frontmatter = dedent(f"""
        ---
        title: {title}
        summary: This is a test summary for {title} that is long enough.
        topics:
          - test-topic-one
          - test-topic-two
          - test-topic-three
        cluster: {cluster}
        ---

        # {title}

        {content or f"Content for {title}."}

        ## Section One

        First section content.

        ## Section Two

        Second section content.
    """).strip()

    path.write_text(frontmatter)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Create a test corpus directory with sample files."""
    corpus = tmp_path / "knowledge" / "corpus"
    corpus.mkdir(parents=True)

    create_test_corpus_file(
        corpus / "dialogue.md",
        "Dialogue Craft",
        "prose-and-language",
        "Writing compelling dialogue.",
    )
    create_test_corpus_file(
        corpus / "fantasy.md",
        "Fantasy Conventions",
        "genre-conventions",
        "Fantasy genre tropes.",
    )
    create_test_corpus_file(
        corpus / "pacing.md",
        "Pacing Techniques",
        "narrative-structure",
        "Controlling story rhythm.",
    )

    return corpus


@pytest.fixture
def index(tmp_path: Path) -> CorpusIndex:
    """Create a test index."""
    index_path = tmp_path / ".corpus_index.sqlite"
    return CorpusIndex(index_path)


class TestCorpusIndex:
    """Tests for CorpusIndex."""

    def test_build_creates_index(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test that build creates the index file."""
        count = index.build(corpus_dir)

        assert count == 3
        assert index._index_path.exists()

    def test_build_indexes_all_files(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test that build indexes all corpus files."""
        index.build(corpus_dir)
        toc = index.get_toc()

        assert len(toc) == 3
        titles = {f["title"] for f in toc}
        assert "Dialogue Craft" in titles
        assert "Fantasy Conventions" in titles
        assert "Pacing Techniques" in titles

    def test_build_skips_unchanged(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test that build skips unchanged files on second run."""
        # First build
        count1 = index.build(corpus_dir)
        assert count1 == 3

        # Second build - no changes
        count2 = index.build(corpus_dir)
        assert count2 == 0

    def test_build_force_reindexes(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test that force=True reindexes all files."""
        index.build(corpus_dir)

        # Force rebuild
        count = index.build(corpus_dir, force=True)
        assert count == 3

    def test_get_status_empty(self, index: CorpusIndex) -> None:
        """Test status on empty index."""
        status = index.get_status()

        assert not status.exists
        assert status.file_count == 0
        assert status.section_count == 0

    def test_get_status_populated(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test status on populated index."""
        index.build(corpus_dir)
        status = index.get_status()

        assert status.exists
        assert status.file_count == 3
        assert status.section_count > 0
        assert "prose-and-language" in status.clusters
        assert "genre-conventions" in status.clusters
        assert "narrative-structure" in status.clusters

    def test_get_toc(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test table of contents retrieval."""
        index.build(corpus_dir)
        toc = index.get_toc()

        assert len(toc) == 3
        for entry in toc:
            assert "path" in entry
            assert "title" in entry
            assert "summary" in entry
            assert "topics" in entry
            assert "cluster" in entry

    def test_get_file_by_name(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test retrieving a specific file."""
        index.build(corpus_dir)
        result = index.get_file("dialogue.md")

        assert result is not None
        assert result["title"] == "Dialogue Craft"
        assert result["cluster"] == "prose-and-language"
        assert len(result["sections"]) >= 2

    def test_get_file_without_extension(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test retrieving file without .md extension."""
        index.build(corpus_dir)
        result = index.get_file("fantasy")

        assert result is not None
        assert result["title"] == "Fantasy Conventions"

    def test_get_file_not_found(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test retrieving nonexistent file returns None."""
        index.build(corpus_dir)
        result = index.get_file("nonexistent")

        assert result is None

    def test_get_cluster(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test retrieving files by cluster."""
        index.build(corpus_dir)
        files = index.get_cluster("prose-and-language")

        assert len(files) == 1
        assert files[0]["title"] == "Dialogue Craft"

    def test_get_cluster_empty(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test retrieving empty cluster returns empty list."""
        index.build(corpus_dir)
        files = index.get_cluster("scope-and-planning")

        assert files == []

    def test_keyword_search(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test keyword search finds matching sections."""
        index.build(corpus_dir)
        results = index.keyword_search("dialogue")

        assert len(results) > 0
        assert any("Dialogue" in r["title"] for r in results)

    def test_keyword_search_no_results(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test keyword search with no matches returns empty."""
        index.build(corpus_dir)
        results = index.keyword_search("xyznonexistent")

        assert results == []

    def test_list_clusters(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test listing all clusters."""
        index.build(corpus_dir)
        clusters = index.list_clusters()

        assert len(clusters) == 3
        cluster_names = {c["cluster"] for c in clusters}
        assert "prose-and-language" in cluster_names
        assert "genre-conventions" in cluster_names
        assert "narrative-structure" in cluster_names

        for cluster in clusters:
            assert "file_count" in cluster
            assert "description" in cluster

    def test_close(self, index: CorpusIndex, corpus_dir: Path) -> None:
        """Test that close properly closes connection."""
        index.build(corpus_dir)
        index.close()

        # Should be able to reopen
        index.build(corpus_dir, force=True)
        assert index.get_status().file_count == 3


class TestIndexWithRealCorpus:
    """Integration tests with real corpus files."""

    def test_index_real_corpus(self) -> None:
        """Test indexing the actual domain corpus."""
        # Navigate from tests/runtime/corpus/test_index.py to domain-v4
        domain_path = Path(__file__).resolve().parents[3] / "domain-v4"
        corpus_dir = domain_path / "knowledge" / "corpus"

        if not corpus_dir.exists():
            pytest.skip("Domain corpus not found")

        # Use temp index
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / ".corpus_index.sqlite"
            index = CorpusIndex(index_path)

            count = index.build(corpus_dir)

            assert count == 22  # Expected number of corpus files

            status = index.get_status()
            assert status.file_count == 22

            # Check all clusters are represented
            assert len(status.clusters) == 7

            # Test TOC
            toc = index.get_toc()
            assert len(toc) == 22

            # Test file retrieval
            dialogue = index.get_file("dialogue_craft")
            assert dialogue is not None
            assert dialogue["cluster"] == "prose-and-language"

            index.close()
