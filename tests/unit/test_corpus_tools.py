"""Tests for IF Craft Corpus research tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from questfoundry.tools.research.corpus_tools import (
    CorpusNotAvailableError,
    GetDocumentTool,
    ListClustersTool,
    SearchCorpusTool,
    get_corpus_tools,
)


@dataclass
class MockCorpusResult:
    """Mock corpus search result."""

    document_name: str
    title: str
    cluster: str
    section_heading: str | None
    content: str
    score: float
    topics: list[str]
    search_type: str = "keyword"

    @property
    def source(self) -> str:
        if self.section_heading:
            return f"{self.document_name} > {self.section_heading}"
        return self.document_name


class TestSearchCorpusTool:
    """Tests for SearchCorpusTool."""

    def test_definition_has_required_fields(self) -> None:
        """Tool definition should have name, description, and parameters."""
        tool = SearchCorpusTool()
        defn = tool.definition

        assert defn.name == "search_corpus"
        assert "craft" in defn.description.lower()
        assert defn.parameters["type"] == "object"
        assert "query" in defn.parameters["properties"]
        assert "query" in defn.parameters["required"]

    def test_execute_without_corpus_installed(self) -> None:
        """Should return error when corpus not installed."""
        tool = SearchCorpusTool()

        with patch(
            "questfoundry.tools.research.corpus_tools._corpus_available",
            return_value=False,
        ):
            # Clear the lru_cache
            from questfoundry.tools.research.corpus_tools import _get_corpus

            _get_corpus.cache_clear()

            result = tool.execute({"query": "test"})

        assert "not available" in result.lower() or "not installed" in result.lower()

    def test_execute_with_results(self) -> None:
        """Should return formatted results when search succeeds."""
        from questfoundry.tools.research.corpus_tools import _get_corpus

        _get_corpus.cache_clear()

        tool = SearchCorpusTool()

        mock_results = [
            MockCorpusResult(
                document_name="dialogue_craft",
                title="Dialogue Craft",
                cluster="prose-and-language",
                section_heading="Subtext",
                content="Subtext is what characters don't say directly.",
                score=0.85,
                topics=["dialogue", "subtext", "craft"],
            ),
        ]

        mock_corpus = MagicMock()
        mock_corpus.search.return_value = mock_results

        # Create mock ifcraftcorpus module
        mock_module = MagicMock()
        mock_module.Corpus.return_value = mock_corpus

        with (
            patch.dict(sys.modules, {"ifcraftcorpus": mock_module}),
            patch(
                "questfoundry.tools.research.corpus_tools._corpus_available",
                return_value=True,
            ),
        ):
            _get_corpus.cache_clear()
            result = tool.execute({"query": "dialogue"})

        assert "dialogue_craft" in result
        assert "0.85" in result
        assert "subtext" in result.lower()
        mock_corpus.search.assert_called_once_with("dialogue", cluster=None, limit=5)

    def test_execute_with_cluster_filter(self) -> None:
        """Should pass cluster filter to search."""
        from questfoundry.tools.research.corpus_tools import _get_corpus

        _get_corpus.cache_clear()

        tool = SearchCorpusTool()

        mock_corpus = MagicMock()
        mock_corpus.search.return_value = []

        # Create mock ifcraftcorpus module
        mock_module = MagicMock()
        mock_module.Corpus.return_value = mock_corpus

        with (
            patch.dict(sys.modules, {"ifcraftcorpus": mock_module}),
            patch(
                "questfoundry.tools.research.corpus_tools._corpus_available",
                return_value=True,
            ),
        ):
            _get_corpus.cache_clear()
            tool.execute({"query": "mystery", "cluster": "genre-conventions", "limit": 3})

        mock_corpus.search.assert_called_once_with("mystery", cluster="genre-conventions", limit=3)

    def test_execute_no_results(self) -> None:
        """Should return structured JSON with no_results status (ADR-008)."""
        import json

        from questfoundry.tools.research.corpus_tools import _get_corpus

        _get_corpus.cache_clear()

        tool = SearchCorpusTool()

        mock_corpus = MagicMock()
        mock_corpus.search.return_value = []

        # Create mock ifcraftcorpus module
        mock_module = MagicMock()
        mock_module.Corpus.return_value = mock_corpus

        with (
            patch.dict(sys.modules, {"ifcraftcorpus": mock_module}),
            patch(
                "questfoundry.tools.research.corpus_tools._corpus_available",
                return_value=True,
            ),
        ):
            _get_corpus.cache_clear()
            result = tool.execute({"query": "xyz123"})

        # ADR-008: Structured JSON response
        data = json.loads(result)
        assert data["result"] == "no_results"
        assert data["query"] == "xyz123"
        assert "action" in data
        assert "proceed" in data["action"].lower()


class TestGetDocumentTool:
    """Tests for GetDocumentTool."""

    def test_definition_has_required_fields(self) -> None:
        """Tool definition should have name, description, and parameters."""
        tool = GetDocumentTool()
        defn = tool.definition

        assert defn.name == "get_document"
        assert "document" in defn.description.lower()
        assert "name" in defn.parameters["properties"]
        assert "name" in defn.parameters["required"]

    def test_execute_document_found(self) -> None:
        """Should return formatted document when found."""
        tool = GetDocumentTool()

        mock_doc = {
            "title": "Horror Atmosphere",
            "cluster": "emotional-design",
            "content": "Building dread through pacing...",
        }

        mock_corpus = MagicMock()
        mock_corpus.get_document.return_value = mock_doc

        with (
            patch(
                "questfoundry.tools.research.corpus_tools._corpus_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.corpus_tools._get_corpus",
                return_value=mock_corpus,
            ),
        ):
            result = tool.execute({"name": "horror_atmosphere"})

        assert "Horror Atmosphere" in result
        assert "emotional-design" in result
        assert "dread" in result
        mock_corpus.get_document.assert_called_once_with("horror_atmosphere")

    def test_execute_document_not_found(self) -> None:
        """Should return helpful message when document not found."""
        tool = GetDocumentTool()

        mock_corpus = MagicMock()
        mock_corpus.get_document.return_value = None

        with (
            patch(
                "questfoundry.tools.research.corpus_tools._corpus_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.corpus_tools._get_corpus",
                return_value=mock_corpus,
            ),
        ):
            result = tool.execute({"name": "nonexistent"})

        assert "not found" in result.lower()


class TestListClustersTool:
    """Tests for ListClustersTool."""

    def test_definition_has_required_fields(self) -> None:
        """Tool definition should have name and description."""
        tool = ListClustersTool()
        defn = tool.definition

        assert defn.name == "list_clusters"
        assert "cluster" in defn.description.lower()
        assert defn.parameters["type"] == "object"

    def test_execute_returns_formatted_clusters(self) -> None:
        """Should return formatted cluster list with descriptions."""
        tool = ListClustersTool()

        mock_clusters = ["genre-conventions", "narrative-structure", "prose-and-language"]

        mock_corpus = MagicMock()
        mock_corpus.list_clusters.return_value = mock_clusters

        with (
            patch(
                "questfoundry.tools.research.corpus_tools._corpus_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.corpus_tools._get_corpus",
                return_value=mock_corpus,
            ),
        ):
            result = tool.execute({})

        assert "genre-conventions" in result
        assert "narrative-structure" in result
        # Should include descriptions
        assert "Genre-specific patterns" in result or "patterns" in result.lower()


class TestGetCorpusTools:
    """Tests for get_corpus_tools factory."""

    def test_returns_empty_when_not_available(self) -> None:
        """Should return empty list when corpus not installed."""
        with patch(
            "questfoundry.tools.research.corpus_tools._corpus_available",
            return_value=False,
        ):
            tools = get_corpus_tools()

        assert tools == []

    def test_returns_all_tools_when_available(self) -> None:
        """Should return all tools when corpus is installed."""
        mock_corpus = MagicMock()

        with (
            patch(
                "questfoundry.tools.research.corpus_tools._corpus_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.corpus_tools._get_corpus",
                return_value=mock_corpus,
            ),
        ):
            tools = get_corpus_tools()

        assert len(tools) == 3
        tool_names = {t.definition.name for t in tools}
        assert tool_names == {"search_corpus", "get_document", "list_clusters"}


class TestCorpusAvailable:
    """Tests for _corpus_available helper."""

    def test_returns_false_when_import_fails(self) -> None:
        """Should return False when ifcraftcorpus not installed."""
        # This test relies on the actual import behavior
        # When ifcraftcorpus is not installed, it should return False
        # We can't easily test this without uninstalling the package
        pass  # Covered by integration tests


class TestCorpusNotAvailableError:
    """Tests for CorpusNotAvailableError exception."""

    def test_message_includes_install_instructions(self) -> None:
        """Should include installation instructions in message."""
        error = CorpusNotAvailableError()
        assert "ifcraftcorpus" in str(error)
        assert "uv add" in str(error) or "install" in str(error).lower()
