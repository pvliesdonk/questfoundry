"""
Tests for consult_corpus tool.

This tool provides access to the craft corpus with multiple modes:
- toc: Table of contents
- file: Get specific file
- cluster: Browse by cluster
- search: Search for excerpts
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.corpus.index import CorpusIndex
from questfoundry.runtime.tools import TOOL_IMPLEMENTATIONS, ToolContext
from questfoundry.runtime.tools.consult_corpus import ConsultCorpusTool


def create_test_corpus_file(path: Path, title: str, cluster: str, content: str = "") -> None:
    """Helper to create a test corpus file."""
    frontmatter = dedent(f"""\
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
    """)

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
        "Writing compelling dialogue for interactive fiction.",
    )
    create_test_corpus_file(
        corpus / "fantasy.md",
        "Fantasy Conventions",
        "genre-conventions",
        "Fantasy genre tropes and conventions.",
    )
    create_test_corpus_file(
        corpus / "pacing.md",
        "Pacing Techniques",
        "narrative-structure",
        "Controlling story rhythm and pacing.",
    )

    return corpus


@pytest.fixture
def indexed_domain(tmp_path: Path, corpus_dir: Path) -> Path:
    """Create a domain with indexed corpus."""
    domain_path = tmp_path
    index_path = CorpusIndex.get_index_path(domain_path)
    index = CorpusIndex(index_path)
    index.build(corpus_dir)
    index.close()
    return domain_path


@pytest.fixture
def tool_context(indexed_domain: Path) -> ToolContext:
    """Create tool context with indexed domain."""
    context = MagicMock(spec=ToolContext)
    context.domain_path = str(indexed_domain)
    context.agent_id = "test_agent"
    context.broker = None
    return context


@pytest.fixture
def consult_corpus_tool(tool_context: ToolContext) -> ConsultCorpusTool:
    """Create ConsultCorpusTool instance for testing."""
    definition = MagicMock()
    definition.id = "consult_corpus"
    definition.name = "Consult Corpus"
    definition.input_schema = MagicMock()
    definition.input_schema.properties = {
        "mode": {"type": "string"},
        "query": {"type": "string"},
        "file": {"type": "string"},
        "cluster": {"type": "string"},
        "max_results": {"type": "integer"},
    }
    definition.input_schema.required = []

    tool = ConsultCorpusTool(definition=definition, context=tool_context)
    return tool


class TestConsultCorpusToolRegistration:
    """Tests for tool registration."""

    def test_tool_is_registered(self):
        """consult_corpus should be registered in TOOL_IMPLEMENTATIONS."""
        assert "consult_corpus" in TOOL_IMPLEMENTATIONS

    def test_tool_class_is_correct(self):
        """Registered tool should be ConsultCorpusTool."""
        assert TOOL_IMPLEMENTATIONS["consult_corpus"] is ConsultCorpusTool


class TestConsultCorpusTocMode:
    """Tests for toc mode."""

    async def test_toc_returns_all_files(self, consult_corpus_tool: ConsultCorpusTool):
        """toc mode should return table of contents."""
        result = await consult_corpus_tool.execute({"mode": "toc"})

        assert result.success is True
        assert result.data["mode"] == "toc"
        assert "toc" in result.data
        assert len(result.data["toc"]) == 3

    async def test_toc_includes_file_metadata(self, consult_corpus_tool: ConsultCorpusTool):
        """toc entries should include title, summary, cluster, topics."""
        result = await consult_corpus_tool.execute({"mode": "toc"})

        assert result.success is True
        for entry in result.data["toc"]:
            assert "path" in entry
            assert "title" in entry
            assert "summary" in entry
            assert "cluster" in entry
            assert "topics" in entry


class TestConsultCorpusFileMode:
    """Tests for file mode."""

    async def test_file_retrieves_by_name(self, consult_corpus_tool: ConsultCorpusTool):
        """file mode should retrieve specific file by name."""
        result = await consult_corpus_tool.execute({"mode": "file", "file": "dialogue"})

        assert result.success is True
        assert result.data["mode"] == "file"
        assert "file_content" in result.data
        assert result.data["file_content"]["title"] == "Dialogue Craft"

    async def test_file_includes_sections(self, consult_corpus_tool: ConsultCorpusTool):
        """file content should include sections."""
        result = await consult_corpus_tool.execute({"mode": "file", "file": "dialogue"})

        assert result.success is True
        sections = result.data["file_content"]["sections"]
        assert len(sections) >= 2
        assert any(s["heading"] == "Section One" for s in sections)

    async def test_file_not_found_error(self, consult_corpus_tool: ConsultCorpusTool):
        """file mode should error for nonexistent file."""
        result = await consult_corpus_tool.execute({"mode": "file", "file": "nonexistent"})

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_file_requires_parameter(self, consult_corpus_tool: ConsultCorpusTool):
        """file mode should error without file parameter."""
        result = await consult_corpus_tool.execute({"mode": "file"})

        assert result.success is False
        assert "file parameter" in result.error.lower()


class TestConsultCorpusClusterMode:
    """Tests for cluster mode."""

    async def test_cluster_without_name_lists_clusters(
        self, consult_corpus_tool: ConsultCorpusTool
    ):
        """cluster mode without cluster param should list available clusters."""
        result = await consult_corpus_tool.execute({"mode": "cluster"})

        assert result.success is True
        assert "clusters" in result.data
        clusters = result.data["clusters"]
        assert len(clusters) == 3
        cluster_names = {c["cluster"] for c in clusters}
        assert "prose-and-language" in cluster_names

    async def test_cluster_returns_files_in_cluster(self, consult_corpus_tool: ConsultCorpusTool):
        """cluster mode should return files in specified cluster."""
        result = await consult_corpus_tool.execute(
            {"mode": "cluster", "cluster": "prose-and-language"}
        )

        assert result.success is True
        assert result.data["cluster"] == "prose-and-language"
        assert "cluster_files" in result.data
        assert len(result.data["cluster_files"]) == 1
        assert result.data["cluster_files"][0]["title"] == "Dialogue Craft"

    async def test_cluster_invalid_name_error(self, consult_corpus_tool: ConsultCorpusTool):
        """cluster mode should error for invalid cluster name."""
        result = await consult_corpus_tool.execute(
            {"mode": "cluster", "cluster": "invalid-cluster"}
        )

        assert result.success is False
        assert "invalid cluster" in result.error.lower()


class TestConsultCorpusSearchMode:
    """Tests for search mode."""

    async def test_search_finds_matching_content(self, consult_corpus_tool: ConsultCorpusTool):
        """search mode should find matching content."""
        result = await consult_corpus_tool.execute({"mode": "search", "query": "dialogue"})

        assert result.success is True
        assert result.data["mode"] == "search"
        assert result.data["search_method"] == "keyword"
        assert result.data["excerpt_count"] > 0
        assert len(result.data["excerpts"]) > 0

    async def test_search_respects_max_results(self, consult_corpus_tool: ConsultCorpusTool):
        """search should respect max_results parameter."""
        result = await consult_corpus_tool.execute(
            {"mode": "search", "query": "section", "max_results": 2}
        )

        assert result.success is True
        assert len(result.data["excerpts"]) <= 2

    async def test_search_requires_query(self, consult_corpus_tool: ConsultCorpusTool):
        """search mode should error without query."""
        result = await consult_corpus_tool.execute({"mode": "search"})

        assert result.success is False
        assert "query" in result.error.lower()

    async def test_search_default_mode(self, consult_corpus_tool: ConsultCorpusTool):
        """search should be the default mode when no mode specified."""
        result = await consult_corpus_tool.execute({"query": "fantasy"})

        assert result.success is True
        assert result.data["mode"] == "search"
        assert result.data["excerpt_count"] > 0


class TestConsultCorpusErrorHandling:
    """Tests for error handling."""

    async def test_unknown_mode_error(self, consult_corpus_tool: ConsultCorpusTool):
        """Unknown mode should return error."""
        result = await consult_corpus_tool.execute({"mode": "invalid"})

        assert result.success is False
        assert "unknown mode" in result.error.lower()

    async def test_no_domain_path_error(self):
        """Missing domain_path should return error."""
        context = MagicMock(spec=ToolContext)
        context.domain_path = None
        context.agent_id = "test_agent"
        context.broker = None

        definition = MagicMock()
        definition.id = "consult_corpus"

        tool = ConsultCorpusTool(definition=definition, context=context)
        result = await tool.execute({"mode": "toc"})

        assert result.success is False
        assert "not available" in result.error.lower()


class TestConsultCorpusIntegration:
    """Integration tests with real domain data."""

    @pytest.fixture
    def domain_v4_path(self) -> Path:
        """Return path to domain-v4 for integration tests."""
        return Path(__file__).resolve().parents[3] / "domain-v4"

    async def test_with_real_corpus(self, domain_v4_path: Path):
        """Test with actual domain-v4 corpus."""
        corpus_dir = domain_v4_path / "knowledge" / "corpus"
        if not corpus_dir.exists():
            pytest.skip("Domain corpus not found")

        # Build index in temp location
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            knowledge_dir = tmp_path / "knowledge"
            knowledge_dir.mkdir()

            # Copy corpus to temp
            import shutil

            shutil.copytree(corpus_dir, knowledge_dir / "corpus")

            # Build index
            index_path = CorpusIndex.get_index_path(tmp_path)
            index = CorpusIndex(index_path)
            index.build(knowledge_dir / "corpus")

            # Create context and tool
            context = MagicMock(spec=ToolContext)
            context.domain_path = str(tmp_path)
            context.agent_id = "test_agent"
            context.broker = None

            definition = MagicMock()
            definition.id = "consult_corpus"

            tool = ConsultCorpusTool(definition=definition, context=context)

            # Test toc
            result = await tool.execute({"mode": "toc"})
            assert result.success is True
            assert len(result.data["toc"]) == 22

            # Test file
            result = await tool.execute({"mode": "file", "file": "dialogue_craft"})
            assert result.success is True
            assert result.data["file_content"]["cluster"] == "prose-and-language"

            # Test cluster
            result = await tool.execute({"mode": "cluster", "cluster": "genre-conventions"})
            assert result.success is True
            assert len(result.data["cluster_files"]) == 4

            # Test search
            result = await tool.execute({"mode": "search", "query": "dialogue"})
            assert result.success is True
            assert result.data["excerpt_count"] > 0

            index.close()
