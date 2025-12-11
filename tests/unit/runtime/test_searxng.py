"""Tests for SearXNG web search tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from questfoundry.runtime.config import SearXNGConfig
from questfoundry.runtime.tools.searxng import WebSearchTool, create_web_search_tool


class TestSearXNGConfig:
    """Tests for SearXNGConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = SearXNGConfig()
        assert config.url is None
        assert config.timeout == 10
        assert config.max_results == 5

    def test_enabled_with_url(self) -> None:
        """Config is enabled when URL is set."""
        config = SearXNGConfig(url="http://localhost:8080")
        assert config.enabled is True

    def test_disabled_without_url(self) -> None:
        """Config is disabled when URL is not set."""
        config = SearXNGConfig()
        assert config.enabled is False

    def test_custom_values(self) -> None:
        """Custom values can be set."""
        config = SearXNGConfig(
            url="http://search.example.com",
            timeout=30,
            max_results=10,
        )
        assert config.url == "http://search.example.com"
        assert config.timeout == 30
        assert config.max_results == 10

    def test_validation(self) -> None:
        """Values are validated."""
        with pytest.raises(ValueError):
            SearXNGConfig(timeout=0)  # Must be >= 1

        with pytest.raises(ValueError):
            SearXNGConfig(timeout=100)  # Must be <= 60

        with pytest.raises(ValueError):
            SearXNGConfig(max_results=0)  # Must be >= 1

        with pytest.raises(ValueError):
            SearXNGConfig(max_results=50)  # Must be <= 20


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_tool_attributes(self) -> None:
        """Tool has correct name and description."""
        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert "search" in tool.description.lower()
        assert "searxng" in tool.description.lower()

    def test_not_configured(self) -> None:
        """Tool returns helpful message when not configured."""
        tool = WebSearchTool()
        tool.searxng_url = None

        result = json.loads(tool._run("test query"))
        assert result["success"] is True
        assert result["available"] is False
        assert "not configured" in result["message"]

    def test_empty_query(self) -> None:
        """Tool returns error for empty query."""
        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"

        result = json.loads(tool._run(""))
        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_whitespace_only_query(self) -> None:
        """Tool returns error for whitespace-only query."""
        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"

        result = json.loads(tool._run("   "))
        assert result["success"] is False
        assert "required" in result["error"].lower()

    @patch("questfoundry.runtime.tools.searxng.httpx.Client")
    def test_unavailable_instance(self, mock_client_class: MagicMock) -> None:
        """Tool handles unavailable SearXNG instance gracefully."""
        # Mock availability check fails
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client_class.return_value = mock_client

        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"
        tool.availability_cached = None  # Reset cache

        result = json.loads(tool._run("test query"))
        assert result["success"] is True
        assert result["available"] is False
        assert "unavailable" in result["message"].lower()

    @patch("questfoundry.runtime.tools.searxng.httpx.Client")
    def test_successful_search(self, mock_client_class: MagicMock) -> None:
        """Tool returns formatted results on successful search."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Castle Architecture Guide",
                    "url": "https://example.com/castles",
                    "content": "Medieval castles were built...",
                    "engine": "google",
                },
                {
                    "title": "History of Fortifications",
                    "url": "https://example.com/forts",
                    "content": "Stone fortifications emerged...",
                    "engine": "bing",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"
        tool.availability_cached = True  # Skip availability check
        tool.max_results = 5

        result = json.loads(tool._run("medieval castle architecture"))
        assert result["success"] is True
        assert result["available"] is True
        assert result["query"] == "medieval castle architecture"
        assert result["result_count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Castle Architecture Guide"
        assert result["results"][1]["title"] == "History of Fortifications"

    @patch("questfoundry.runtime.tools.searxng.httpx.Client")
    def test_max_results_limit(self, mock_client_class: MagicMock) -> None:
        """Tool respects max_results limit."""
        # Mock response with many results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"http://example.com/{i}",
                    "content": f"Content {i}",
                }
                for i in range(20)
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"
        tool.availability_cached = True
        tool.max_results = 3

        result = json.loads(tool._run("test"))
        assert result["result_count"] == 3
        assert len(result["results"]) == 3

    @patch("questfoundry.runtime.tools.searxng.httpx.Client")
    def test_categories_parameter(self, mock_client_class: MagicMock) -> None:
        """Tool passes categories to SearXNG."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"
        tool.availability_cached = True

        # Test with list of categories
        result = json.loads(tool._run("test", categories=["news", "science"]))
        assert result["success"] is True
        assert result["categories"] == ["news", "science"]

        # Test with string category
        result = json.loads(tool._run("test", categories="images"))
        assert result["success"] is True
        assert result["categories"] == ["images"]

    @patch("questfoundry.runtime.tools.searxng.httpx.Client")
    def test_timeout_error(self, mock_client_class: MagicMock) -> None:
        """Tool handles timeout gracefully."""
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
        mock_client_class.return_value = mock_client

        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"
        tool.availability_cached = True

        result = json.loads(tool._run("test"))
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @patch("questfoundry.runtime.tools.searxng.httpx.Client")
    def test_http_error(self, mock_client_class: MagicMock) -> None:
        """Tool handles HTTP errors gracefully."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )
        mock_client_class.return_value = mock_client

        tool = WebSearchTool()
        tool.searxng_url = "http://localhost:8080"
        tool.availability_cached = True

        result = json.loads(tool._run("test"))
        assert result["success"] is False
        assert "500" in result["error"]


class TestCreateWebSearchTool:
    """Tests for create_web_search_tool factory function."""

    def test_factory_defaults(self) -> None:
        """Factory creates tool with default settings."""
        tool = create_web_search_tool()
        assert tool.searxng_url is None
        assert tool.timeout == 10
        assert tool.max_results == 5

    def test_factory_custom_settings(self) -> None:
        """Factory creates tool with custom settings."""
        tool = create_web_search_tool(
            searxng_url="http://search.example.com",
            timeout=30,
            max_results=10,
        )
        assert tool.searxng_url == "http://search.example.com"
        assert tool.timeout == 30
        assert tool.max_results == 10


class TestWebSearchToolIntegration:
    """Integration tests for web search tool with mocked responses."""

    @patch("questfoundry.runtime.tools.searxng.httpx.Client")
    def test_research_workflow(self, mock_client_class: MagicMock) -> None:
        """Test typical research workflow."""
        # Mock successful search
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Viking Longships - History",
                    "url": "https://example.com/vikings",
                    "content": "Viking longships were remarkable vessels...",
                    "engine": "duckduckgo",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create tool like Lorekeeper would use it
        tool = create_web_search_tool(
            searxng_url="http://localhost:8080",
            timeout=10,
            max_results=5,
        )
        tool.availability_cached = True  # Skip availability check in test

        # Execute research query
        result = json.loads(tool._run("viking longship construction"))

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert "viking" in result["results"][0]["title"].lower()
