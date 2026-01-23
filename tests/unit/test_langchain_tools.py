"""Tests for LangChain tool wrappers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

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

    @pytest.mark.asyncio
    async def test_search_corpus_delegates(self) -> None:
        """search_corpus should delegate to SearchCorpusTool."""
        with patch("questfoundry.tools.langchain_tools._search_corpus_tool") as mock_tool:
            mock_tool.execute = AsyncMock(
                return_value=json.dumps(
                    {"result": "success", "content": "test content", "action": "use this"}
                )
            )

            result = await search_corpus.ainvoke({"query": "test", "limit": 3})

            mock_tool.execute.assert_called_once_with(
                {"query": "test", "limit": 3, "cluster": None}
            )
            assert "success" in result
            assert "test content" in result

    @pytest.mark.asyncio
    async def test_search_corpus_with_cluster(self) -> None:
        """search_corpus should pass cluster parameter."""
        with patch("questfoundry.tools.langchain_tools._search_corpus_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value=json.dumps({"result": "success"}))

            await search_corpus.ainvoke(
                {"query": "mystery", "cluster": "genre-conventions", "limit": 5}
            )

            call_args = mock_tool.execute.call_args[0][0]
            assert call_args["query"] == "mystery"
            assert call_args["cluster"] == "genre-conventions"
            assert call_args["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_corpus_default_limit(self) -> None:
        """search_corpus should use default limit of 5."""
        with patch("questfoundry.tools.langchain_tools._search_corpus_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value=json.dumps({"result": "success"}))

            await search_corpus.ainvoke({"query": "test"})

            call_args = mock_tool.execute.call_args[0][0]
            assert call_args["limit"] == 5


class TestGetDocumentExecution:
    """Test get_document tool execution."""

    @pytest.mark.asyncio
    async def test_get_document_delegates(self) -> None:
        """get_document should delegate to GetDocumentTool."""
        with patch("questfoundry.tools.langchain_tools._get_document_tool") as mock_tool:
            mock_tool.execute = AsyncMock(
                return_value=json.dumps(
                    {
                        "result": "success",
                        "title": "Test Doc",
                        "content": "document content",
                        "action": "use this",
                    }
                )
            )

            result = await get_document.ainvoke({"name": "test_doc"})

            mock_tool.execute.assert_called_once_with({"name": "test_doc"})
            assert "success" in result
            assert "Test Doc" in result


class TestListClustersExecution:
    """Test list_clusters tool execution."""

    @pytest.mark.asyncio
    async def test_list_clusters_delegates(self) -> None:
        """list_clusters should delegate to ListClustersTool."""
        with patch("questfoundry.tools.langchain_tools._list_clusters_tool") as mock_tool:
            mock_tool.execute = AsyncMock(
                return_value=json.dumps(
                    {
                        "result": "success",
                        "clusters": ["genre-conventions", "narrative-structure"],
                        "content": "cluster list",
                        "action": "use search_corpus",
                    }
                )
            )

            result = await list_clusters.ainvoke({})

            mock_tool.execute.assert_called_once_with({})
            assert "success" in result
            assert "genre-conventions" in result


class TestWebSearchExecution:
    """Test web_search tool execution."""

    @pytest.mark.asyncio
    async def test_web_search_delegates(self) -> None:
        """web_search should delegate to WebSearchTool."""
        with patch("questfoundry.tools.langchain_tools._web_search_tool") as mock_tool:
            mock_tool.execute = AsyncMock(
                return_value=json.dumps(
                    {"result": "success", "content": "search results", "action": "use this"}
                )
            )

            result = await web_search.ainvoke({"query": "test search", "max_results": 3})

            mock_tool.execute.assert_called_once_with(
                {"query": "test search", "max_results": 3, "recency": "all_time"}
            )
            assert "success" in result

    @pytest.mark.asyncio
    async def test_web_search_with_recency(self) -> None:
        """web_search should pass recency parameter."""
        with patch("questfoundry.tools.langchain_tools._web_search_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value=json.dumps({"result": "success"}))

            await web_search.ainvoke({"query": "recent news", "max_results": 5, "recency": "week"})

            call_args = mock_tool.execute.call_args[0][0]
            assert call_args["recency"] == "week"

    @pytest.mark.asyncio
    async def test_web_search_defaults(self) -> None:
        """web_search should use default values."""
        with patch("questfoundry.tools.langchain_tools._web_search_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value=json.dumps({"result": "success"}))

            await web_search.ainvoke({"query": "test"})

            call_args = mock_tool.execute.call_args[0][0]
            assert call_args["max_results"] == 5
            assert call_args["recency"] == "all_time"


class TestWebFetchExecution:
    """Test web_fetch tool execution."""

    @pytest.mark.asyncio
    async def test_web_fetch_delegates(self) -> None:
        """web_fetch should delegate to WebFetchTool."""
        with patch("questfoundry.tools.langchain_tools._web_fetch_tool") as mock_tool:
            mock_tool.execute = AsyncMock(
                return_value=json.dumps(
                    {
                        "result": "success",
                        "url": "https://example.com",
                        "content": "page content",
                        "action": "use this",
                    }
                )
            )

            result = await web_fetch.ainvoke({"url": "https://example.com"})

            mock_tool.execute.assert_called_once_with(
                {"url": "https://example.com", "extract_mode": "markdown"}
            )
            assert "success" in result
            assert "page content" in result

    @pytest.mark.asyncio
    async def test_web_fetch_with_extract_mode(self) -> None:
        """web_fetch should pass extract_mode parameter."""
        with patch("questfoundry.tools.langchain_tools._web_fetch_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value=json.dumps({"result": "success"}))

            await web_fetch.ainvoke({"url": "https://example.com", "extract_mode": "article"})

            call_args = mock_tool.execute.call_args[0][0]
            assert call_args["extract_mode"] == "article"


class TestToolReturnTypes:
    """Test that all tools return strings (required for LangChain)."""

    @pytest.mark.asyncio
    async def test_search_corpus_returns_string(self) -> None:
        """search_corpus must return a string."""
        with patch("questfoundry.tools.langchain_tools._search_corpus_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value='{"result": "success"}')

            result = await search_corpus.ainvoke({"query": "test"})
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_document_returns_string(self) -> None:
        """get_document must return a string."""
        with patch("questfoundry.tools.langchain_tools._get_document_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value='{"result": "success"}')

            result = await get_document.ainvoke({"name": "test"})
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_list_clusters_returns_string(self) -> None:
        """list_clusters must return a string."""
        with patch("questfoundry.tools.langchain_tools._list_clusters_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value='{"result": "success"}')

            result = await list_clusters.ainvoke({})
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_web_search_returns_string(self) -> None:
        """web_search must return a string."""
        with patch("questfoundry.tools.langchain_tools._web_search_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value='{"result": "success"}')

            result = await web_search.ainvoke({"query": "test"})
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_web_fetch_returns_string(self) -> None:
        """web_fetch must return a string."""
        with patch("questfoundry.tools.langchain_tools._web_fetch_tool") as mock_tool:
            mock_tool.execute = AsyncMock(return_value='{"result": "success"}')

            result = await web_fetch.ainvoke({"url": "https://example.com"})
            assert isinstance(result, str)


class TestErrorHandling:
    """Test that tools handle errors gracefully."""

    @pytest.mark.asyncio
    async def test_search_corpus_error_propagates(self) -> None:
        """search_corpus should return error JSON from underlying tool."""
        with patch("questfoundry.tools.langchain_tools._search_corpus_tool") as mock_tool:
            mock_tool.execute = AsyncMock(
                return_value=json.dumps(
                    {"result": "error", "error": "Corpus not available", "action": "proceed"}
                )
            )

            result = await search_corpus.ainvoke({"query": "test"})

            data = json.loads(result)
            assert data["result"] == "error"
            assert "error" in data

    @pytest.mark.asyncio
    async def test_web_search_error_propagates(self) -> None:
        """web_search should return error JSON from underlying tool."""
        with patch("questfoundry.tools.langchain_tools._web_search_tool") as mock_tool:
            mock_tool.execute = AsyncMock(
                return_value=json.dumps(
                    {
                        "result": "error",
                        "error": "SEARXNG_URL not configured",
                        "action": "proceed",
                    }
                )
            )

            result = await web_search.ainvoke({"query": "test"})

            data = json.loads(result)
            assert data["result"] == "error"
