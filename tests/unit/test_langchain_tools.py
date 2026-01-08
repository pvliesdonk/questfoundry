"""Tests for LangChain tool wrappers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_corpus_tools,
    get_document,
    get_web_tools,
    list_clusters,
    search_corpus,
    web_fetch,
    web_search,
)


class TestToolMetadata:
    """Test that tools have proper LangChain metadata."""

    def test_search_corpus_is_tool(self) -> None:
        """search_corpus should have LangChain tool metadata."""
        assert hasattr(search_corpus, "name")
        assert search_corpus.name == "search_corpus"
        assert hasattr(search_corpus, "description")
        assert "corpus" in search_corpus.description.lower()

    def test_get_document_is_tool(self) -> None:
        """get_document should have LangChain tool metadata."""
        assert hasattr(get_document, "name")
        assert get_document.name == "get_document"

    def test_list_clusters_is_tool(self) -> None:
        """list_clusters should have LangChain tool metadata."""
        assert hasattr(list_clusters, "name")
        assert list_clusters.name == "list_clusters"

    def test_web_search_is_tool(self) -> None:
        """web_search should have LangChain tool metadata."""
        assert hasattr(web_search, "name")
        assert web_search.name == "web_search"

    def test_web_fetch_is_tool(self) -> None:
        """web_fetch should have LangChain tool metadata."""
        assert hasattr(web_fetch, "name")
        assert web_fetch.name == "web_fetch"


class TestToolCollections:
    """Test tool collection functions."""

    def test_get_all_research_tools(self) -> None:
        """get_all_research_tools should return all 5 tools."""
        tools = get_all_research_tools()
        assert len(tools) == 5
        names = [t.name for t in tools]
        assert "search_corpus" in names
        assert "get_document" in names
        assert "list_clusters" in names
        assert "web_search" in names
        assert "web_fetch" in names

    def test_get_corpus_tools(self) -> None:
        """get_corpus_tools should return 3 corpus tools."""
        tools = get_corpus_tools()
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "search_corpus" in names
        assert "get_document" in names
        assert "list_clusters" in names

    def test_get_web_tools(self) -> None:
        """get_web_tools should return 2 web tools."""
        tools = get_web_tools()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "web_fetch" in names


class TestSearchCorpusExecution:
    """Test search_corpus tool execution."""

    @patch("questfoundry.tools.langchain_tools._search_corpus_tool")
    def test_search_corpus_delegates(self, mock_tool: MagicMock) -> None:
        """search_corpus should delegate to SearchCorpusTool."""
        mock_tool.execute.return_value = json.dumps(
            {"result": "success", "content": "test content", "action": "use this"}
        )

        result = search_corpus.invoke({"query": "test", "limit": 3})

        mock_tool.execute.assert_called_once_with({"query": "test", "limit": 3, "cluster": None})
        assert "success" in result
        assert "test content" in result

    @patch("questfoundry.tools.langchain_tools._search_corpus_tool")
    def test_search_corpus_with_cluster(self, mock_tool: MagicMock) -> None:
        """search_corpus should pass cluster parameter."""
        mock_tool.execute.return_value = json.dumps({"result": "success"})

        search_corpus.invoke({"query": "mystery", "cluster": "genre-conventions", "limit": 5})

        call_args = mock_tool.execute.call_args[0][0]
        assert call_args["query"] == "mystery"
        assert call_args["cluster"] == "genre-conventions"
        assert call_args["limit"] == 5

    @patch("questfoundry.tools.langchain_tools._search_corpus_tool")
    def test_search_corpus_default_limit(self, mock_tool: MagicMock) -> None:
        """search_corpus should use default limit of 5."""
        mock_tool.execute.return_value = json.dumps({"result": "success"})

        search_corpus.invoke({"query": "test"})

        call_args = mock_tool.execute.call_args[0][0]
        assert call_args["limit"] == 5


class TestGetDocumentExecution:
    """Test get_document tool execution."""

    @patch("questfoundry.tools.langchain_tools._get_document_tool")
    def test_get_document_delegates(self, mock_tool: MagicMock) -> None:
        """get_document should delegate to GetDocumentTool."""
        mock_tool.execute.return_value = json.dumps(
            {
                "result": "success",
                "title": "Test Doc",
                "content": "document content",
                "action": "use this",
            }
        )

        result = get_document.invoke({"name": "test_doc"})

        mock_tool.execute.assert_called_once_with({"name": "test_doc"})
        assert "success" in result
        assert "Test Doc" in result


class TestListClustersExecution:
    """Test list_clusters tool execution."""

    @patch("questfoundry.tools.langchain_tools._list_clusters_tool")
    def test_list_clusters_delegates(self, mock_tool: MagicMock) -> None:
        """list_clusters should delegate to ListClustersTool."""
        mock_tool.execute.return_value = json.dumps(
            {
                "result": "success",
                "clusters": ["genre-conventions", "narrative-structure"],
                "content": "cluster list",
                "action": "use search_corpus",
            }
        )

        result = list_clusters.invoke({})

        mock_tool.execute.assert_called_once_with({})
        assert "success" in result
        assert "genre-conventions" in result


class TestWebSearchExecution:
    """Test web_search tool execution."""

    @patch("questfoundry.tools.langchain_tools._web_search_tool")
    def test_web_search_delegates(self, mock_tool: MagicMock) -> None:
        """web_search should delegate to WebSearchTool."""
        mock_tool.execute.return_value = json.dumps(
            {"result": "success", "content": "search results", "action": "use this"}
        )

        result = web_search.invoke({"query": "test search", "max_results": 3})

        mock_tool.execute.assert_called_once_with(
            {"query": "test search", "max_results": 3, "recency": "all_time"}
        )
        assert "success" in result

    @patch("questfoundry.tools.langchain_tools._web_search_tool")
    def test_web_search_with_recency(self, mock_tool: MagicMock) -> None:
        """web_search should pass recency parameter."""
        mock_tool.execute.return_value = json.dumps({"result": "success"})

        web_search.invoke({"query": "recent news", "max_results": 5, "recency": "week"})

        call_args = mock_tool.execute.call_args[0][0]
        assert call_args["recency"] == "week"

    @patch("questfoundry.tools.langchain_tools._web_search_tool")
    def test_web_search_defaults(self, mock_tool: MagicMock) -> None:
        """web_search should use default values."""
        mock_tool.execute.return_value = json.dumps({"result": "success"})

        web_search.invoke({"query": "test"})

        call_args = mock_tool.execute.call_args[0][0]
        assert call_args["max_results"] == 5
        assert call_args["recency"] == "all_time"


class TestWebFetchExecution:
    """Test web_fetch tool execution."""

    @patch("questfoundry.tools.langchain_tools._web_fetch_tool")
    def test_web_fetch_delegates(self, mock_tool: MagicMock) -> None:
        """web_fetch should delegate to WebFetchTool."""
        mock_tool.execute.return_value = json.dumps(
            {
                "result": "success",
                "url": "https://example.com",
                "content": "page content",
                "action": "use this",
            }
        )

        result = web_fetch.invoke({"url": "https://example.com"})

        mock_tool.execute.assert_called_once_with(
            {"url": "https://example.com", "extract_mode": "markdown"}
        )
        assert "success" in result
        assert "page content" in result

    @patch("questfoundry.tools.langchain_tools._web_fetch_tool")
    def test_web_fetch_with_extract_mode(self, mock_tool: MagicMock) -> None:
        """web_fetch should pass extract_mode parameter."""
        mock_tool.execute.return_value = json.dumps({"result": "success"})

        web_fetch.invoke({"url": "https://example.com", "extract_mode": "article"})

        call_args = mock_tool.execute.call_args[0][0]
        assert call_args["extract_mode"] == "article"


class TestToolReturnTypes:
    """Test that all tools return strings (required for LangChain)."""

    @patch("questfoundry.tools.langchain_tools._search_corpus_tool")
    def test_search_corpus_returns_string(self, mock_tool: MagicMock) -> None:
        """search_corpus must return a string."""
        mock_tool.execute.return_value = '{"result": "success"}'

        result = search_corpus.invoke({"query": "test"})
        assert isinstance(result, str)

    @patch("questfoundry.tools.langchain_tools._get_document_tool")
    def test_get_document_returns_string(self, mock_tool: MagicMock) -> None:
        """get_document must return a string."""
        mock_tool.execute.return_value = '{"result": "success"}'

        result = get_document.invoke({"name": "test"})
        assert isinstance(result, str)

    @patch("questfoundry.tools.langchain_tools._list_clusters_tool")
    def test_list_clusters_returns_string(self, mock_tool: MagicMock) -> None:
        """list_clusters must return a string."""
        mock_tool.execute.return_value = '{"result": "success"}'

        result = list_clusters.invoke({})
        assert isinstance(result, str)

    @patch("questfoundry.tools.langchain_tools._web_search_tool")
    def test_web_search_returns_string(self, mock_tool: MagicMock) -> None:
        """web_search must return a string."""
        mock_tool.execute.return_value = '{"result": "success"}'

        result = web_search.invoke({"query": "test"})
        assert isinstance(result, str)

    @patch("questfoundry.tools.langchain_tools._web_fetch_tool")
    def test_web_fetch_returns_string(self, mock_tool: MagicMock) -> None:
        """web_fetch must return a string."""
        mock_tool.execute.return_value = '{"result": "success"}'

        result = web_fetch.invoke({"url": "https://example.com"})
        assert isinstance(result, str)


class TestErrorHandling:
    """Test that tools handle errors gracefully."""

    @patch("questfoundry.tools.langchain_tools._search_corpus_tool")
    def test_search_corpus_error_propagates(self, mock_tool: MagicMock) -> None:
        """search_corpus should return error JSON from underlying tool."""
        mock_tool.execute.return_value = json.dumps(
            {"result": "error", "error": "Corpus not available", "action": "proceed"}
        )

        result = search_corpus.invoke({"query": "test"})

        data = json.loads(result)
        assert data["result"] == "error"
        assert "error" in data

    @patch("questfoundry.tools.langchain_tools._web_search_tool")
    def test_web_search_error_propagates(self, mock_tool: MagicMock) -> None:
        """web_search should return error JSON from underlying tool."""
        mock_tool.execute.return_value = json.dumps(
            {"result": "error", "error": "SEARXNG_URL not configured", "action": "proceed"}
        )

        result = web_search.invoke({"query": "test"})

        data = json.loads(result)
        assert data["result"] == "error"
