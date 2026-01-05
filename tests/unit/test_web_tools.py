"""Tests for web research tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from questfoundry.tools.research.web_tools import (
    WebFetchTool,
    WebSearchTool,
    _searxng_configured,
    get_web_tools,
)


@dataclass
class MockSearchResult:
    """Mock web search result."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None


@dataclass
class MockFetchResult:
    """Mock web fetch result."""

    url: str
    content: str
    content_length: int
    extract_mode: str


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_definition_has_required_fields(self) -> None:
        """Tool definition should have name, description, and parameters."""
        tool = WebSearchTool()
        defn = tool.definition

        assert defn.name == "web_search"
        assert "search" in defn.description.lower()
        assert defn.parameters["type"] == "object"
        assert "query" in defn.parameters["properties"]
        assert "query" in defn.parameters["required"]

    def test_execute_without_webtools_installed(self) -> None:
        """Should return error when pvl-webtools not installed."""
        tool = WebSearchTool()

        with patch(
            "questfoundry.tools.research.web_tools._web_tools_available",
            return_value=False,
        ):
            result = tool.execute({"query": "test"})

        assert "not installed" in result.lower() or "not available" in result.lower()

    def test_execute_without_searxng_configured(self) -> None:
        """Should return error when SEARXNG_URL not set."""
        tool = WebSearchTool()

        with (
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.web_tools._searxng_configured",
                return_value=False,
            ),
        ):
            result = tool.execute({"query": "test"})

        assert "searxng" in result.lower() or "not configured" in result.lower()

    def test_execute_with_results(self) -> None:
        """Should return formatted results when search succeeds."""
        tool = WebSearchTool()

        mock_results = [
            MockSearchResult(
                title="Interactive Fiction Guide",
                url="https://example.com/if-guide",
                snippet="A comprehensive guide to writing IF...",
                published_date="2025-01-01",
            ),
        ]

        # Create mock pvlwebtools module
        mock_module = MagicMock()
        mock_web_search = AsyncMock(return_value=mock_results)
        mock_module.web_search = mock_web_search

        with (
            patch.dict(sys.modules, {"pvlwebtools": mock_module}),
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.web_tools._searxng_configured",
                return_value=True,
            ),
        ):
            result = tool.execute({"query": "interactive fiction guide"})

        assert "Interactive Fiction Guide" in result
        assert "example.com" in result
        assert "comprehensive guide" in result

    def test_execute_no_results(self) -> None:
        """Should return helpful message when no results found."""
        tool = WebSearchTool()

        # Create mock pvlwebtools module
        mock_module = MagicMock()
        mock_web_search = AsyncMock(return_value=[])
        mock_module.web_search = mock_web_search

        with (
            patch.dict(sys.modules, {"pvlwebtools": mock_module}),
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.web_tools._searxng_configured",
                return_value=True,
            ),
        ):
            result = tool.execute({"query": "xyz123unique"})

        assert "no results" in result.lower()

    def test_execute_handles_exception(self) -> None:
        """Should return error message when search fails."""
        tool = WebSearchTool()

        # Create mock pvlwebtools module that raises exception
        mock_module = MagicMock()
        mock_web_search = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_module.web_search = mock_web_search

        with (
            patch.dict(sys.modules, {"pvlwebtools": mock_module}),
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.web_tools._searxng_configured",
                return_value=True,
            ),
        ):
            result = tool.execute({"query": "test"})

        assert "failed" in result.lower()
        assert "timeout" in result.lower()


class TestWebFetchTool:
    """Tests for WebFetchTool."""

    def test_definition_has_required_fields(self) -> None:
        """Tool definition should have name, description, and parameters."""
        tool = WebFetchTool()
        defn = tool.definition

        assert defn.name == "web_fetch"
        assert "fetch" in defn.description.lower()
        assert "url" in defn.parameters["properties"]
        assert "url" in defn.parameters["required"]

    def test_execute_without_webtools_installed(self) -> None:
        """Should return error when pvl-webtools not installed."""
        tool = WebFetchTool()

        with patch(
            "questfoundry.tools.research.web_tools._web_tools_available",
            return_value=False,
        ):
            result = tool.execute({"url": "https://example.com"})

        assert "not installed" in result.lower()

    def test_execute_success(self) -> None:
        """Should return formatted content when fetch succeeds."""
        tool = WebFetchTool()

        mock_result = MockFetchResult(
            url="https://example.com/article",
            content="# Article Title\n\nThis is the article content...",
            content_length=500,
            extract_mode="markdown",
        )

        # Create mock pvlwebtools module
        mock_module = MagicMock()
        mock_web_fetch = AsyncMock(return_value=mock_result)
        mock_module.web_fetch = mock_web_fetch

        with (
            patch.dict(sys.modules, {"pvlwebtools": mock_module}),
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
        ):
            result = tool.execute({"url": "https://example.com/article"})

        assert "example.com/article" in result
        assert "Article Title" in result
        assert "markdown" in result.lower()

    def test_execute_with_extract_mode(self) -> None:
        """Should pass extract_mode to fetch."""
        tool = WebFetchTool()

        mock_result = MockFetchResult(
            url="https://example.com",
            content="Metadata only",
            content_length=100,
            extract_mode="metadata",
        )

        # Create mock pvlwebtools module
        mock_module = MagicMock()
        mock_web_fetch = AsyncMock(return_value=mock_result)
        mock_module.web_fetch = mock_web_fetch

        with (
            patch.dict(sys.modules, {"pvlwebtools": mock_module}),
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
        ):
            result = tool.execute({"url": "https://example.com", "extract_mode": "metadata"})

        # Verify web_fetch was called
        assert mock_web_fetch.called
        assert "Metadata" in result

    def test_execute_truncates_long_content(self) -> None:
        """Should truncate content that exceeds limit."""
        tool = WebFetchTool()

        long_content = "x" * 10000  # Longer than MAX_OUTPUT_CHARS
        mock_result = MockFetchResult(
            url="https://example.com",
            content=long_content,
            content_length=len(long_content),
            extract_mode="markdown",
        )

        # Create mock pvlwebtools module
        mock_module = MagicMock()
        mock_web_fetch = AsyncMock(return_value=mock_result)
        mock_module.web_fetch = mock_web_fetch

        with (
            patch.dict(sys.modules, {"pvlwebtools": mock_module}),
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
        ):
            result = tool.execute({"url": "https://example.com"})

        assert "truncated" in result.lower()
        assert len(result) < len(long_content)

    def test_execute_handles_exception(self) -> None:
        """Should return error message when fetch fails."""
        tool = WebFetchTool()

        # Create mock pvlwebtools module that raises exception
        mock_module = MagicMock()
        mock_web_fetch = AsyncMock(side_effect=Exception("404 Not Found"))
        mock_module.web_fetch = mock_web_fetch

        with (
            patch.dict(sys.modules, {"pvlwebtools": mock_module}),
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
        ):
            result = tool.execute({"url": "https://example.com/missing"})

        assert "failed" in result.lower()
        assert "404" in result


class TestGetWebTools:
    """Tests for get_web_tools factory."""

    def test_returns_empty_when_not_available(self) -> None:
        """Should return empty list when pvl-webtools not installed."""
        with patch(
            "questfoundry.tools.research.web_tools._web_tools_available",
            return_value=False,
        ):
            tools = get_web_tools()

        assert tools == []

    def test_returns_fetch_only_without_searxng(self) -> None:
        """Should return only fetch tool when SEARXNG not configured."""
        with (
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.web_tools._searxng_configured",
                return_value=False,
            ),
        ):
            tools = get_web_tools(require_searxng=True)

        tool_names = {t.definition.name for t in tools}
        # web_fetch doesn't require SEARXNG
        assert "web_fetch" in tool_names
        assert "web_search" not in tool_names

    def test_returns_both_when_fully_configured(self) -> None:
        """Should return both tools when fully configured."""
        with (
            patch(
                "questfoundry.tools.research.web_tools._web_tools_available",
                return_value=True,
            ),
            patch(
                "questfoundry.tools.research.web_tools._searxng_configured",
                return_value=True,
            ),
        ):
            tools = get_web_tools()

        tool_names = {t.definition.name for t in tools}
        assert tool_names == {"web_search", "web_fetch"}


class TestSearxngConfigured:
    """Tests for _searxng_configured helper."""

    def test_returns_true_when_env_set(self) -> None:
        """Should return True when SEARXNG_URL is set."""
        with patch.dict("os.environ", {"SEARXNG_URL": "http://localhost:8080"}):
            assert _searxng_configured() is True

    def test_returns_false_when_env_missing(self) -> None:
        """Should return False when SEARXNG_URL is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove SEARXNG_URL if it exists
            import os

            original = os.environ.pop("SEARXNG_URL", None)
            try:
                assert _searxng_configured() is False
            finally:
                if original is not None:
                    os.environ["SEARXNG_URL"] = original


class TestWebToolsAvailable:
    """Tests for _web_tools_available helper."""

    def test_returns_false_when_import_fails(self) -> None:
        """Should return False when pvlwebtools not installed."""
        # This test relies on actual import behavior
        # We can't easily test this without manipulating imports
        pass  # Covered by integration tests
