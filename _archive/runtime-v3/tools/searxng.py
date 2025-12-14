"""SearXNG web search tool for Lorekeeper research.

This tool provides web search capabilities via a self-hosted SearXNG instance.
SearXNG is a privacy-respecting metasearch engine that aggregates results
from multiple search engines.

The tool is optional - if SearXNG is not configured, the tool will return
a helpful message indicating that web search is unavailable, allowing
Lorekeeper to continue without it (graceful degradation).

Configuration
-------------
Set the SearXNG URL via environment variable or config file::

    # Environment variable
    export QF_SEARXNG__URL="http://localhost:8080"

    # Or in questfoundry.yaml
    searxng:
      url: "http://localhost:8080"
      timeout: 10
      max_results: 5

Use Cases
---------
- Research real-world facts for world-building
- Verify historical/cultural accuracy
- Find inspiration for lore elements
- Cross-reference existing fiction (avoid plagiarism)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Search the web using a SearXNG instance.

    Use this tool to research facts, verify information, or find inspiration
    for world-building and lore creation. Results include title, URL, and
    snippet for each result.

    If SearXNG is not configured, this tool will return a message indicating
    that web search is unavailable, allowing you to continue without it.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for information using SearXNG. "
        "Use for research, fact-checking, and inspiration during lore creation. "
        "Input: query (search terms), categories (optional: 'general', 'news', 'science', 'images'). "
        "Returns search results with title, URL, and snippet."
    )

    # Configuration - injected by executor
    searxng_url: str | None = Field(default=None)
    timeout: int = Field(default=10)
    max_results: int = Field(default=5)

    # Availability cache (use model_config to exclude from serialization)
    availability_cached: bool | None = Field(default=None, exclude=True)

    def _check_availability(self) -> bool:
        """Check if SearXNG instance is reachable."""
        if not self.searxng_url:
            return False

        try:
            # Quick health check - just check if the base URL responds
            with httpx.Client(timeout=5) as client:
                response = client.get(self.searxng_url)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"SearXNG availability check failed: {e}")
            return False

    @property
    def available(self) -> bool:
        """Check if web search is available (cached)."""
        if self.availability_cached is None:
            self.availability_cached = self._check_availability()
        return self.availability_cached

    def _run(
        self,
        query: str,
        categories: list[str] | str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute web search.

        Parameters
        ----------
        query : str
            Search terms.
        categories : list[str] | str | None
            Search categories (general, news, science, images, etc.).
            Defaults to general if not specified.

        Returns
        -------
        str
            JSON string with search results or unavailability message.
        """
        # Log any unexpected kwargs for debugging
        if kwargs:
            logger.debug(f"web_search received extra kwargs: {list(kwargs.keys())}")

        # Check if query is provided
        if not query or not query.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": "Query is required",
                    "hint": "Provide search terms, e.g., 'medieval castle architecture'",
                }
            )

        # Check availability
        if not self.searxng_url:
            return json.dumps(
                {
                    "success": True,
                    "available": False,
                    "message": "Web search is not configured. "
                    "You can continue without web search - use your existing knowledge "
                    "or ask SR for more context.",
                    "hint": "To enable web search, set QF_SEARXNG__URL environment variable.",
                }
            )

        if not self.available:
            return json.dumps(
                {
                    "success": True,
                    "available": False,
                    "message": f"Web search is temporarily unavailable (SearXNG at {self.searxng_url} is not responding). "
                    "You can continue without web search.",
                }
            )

        # Normalize categories
        if categories is None:
            categories = ["general"]
        elif isinstance(categories, str):
            categories = [categories]

        # Build SearXNG API request
        params = {
            "q": query.strip(),
            "format": "json",
            "categories": ",".join(categories),
        }

        try:
            headers = {"Accept": "application/json"}
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.searxng_url.rstrip('/')}/search",
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.TimeoutException:
            return json.dumps(
                {
                    "success": False,
                    "error": "Search request timed out",
                    "hint": "Try a more specific query or try again later.",
                }
            )
        except httpx.HTTPStatusError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Search request failed: HTTP {e.response.status_code}",
                    "hint": "The search service may be temporarily unavailable.",
                }
            )
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return json.dumps(
                {
                    "success": False,
                    "error": f"Search failed: {e}",
                    "hint": "Try again or continue without web search.",
                }
            )

        # Extract and format results
        results = data.get("results", [])[: self.max_results]

        formatted_results = []
        for r in results:
            formatted_results.append(
                {
                    "title": r.get("title", "Untitled"),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "engine": r.get("engine", "unknown"),
                }
            )

        return json.dumps(
            {
                "success": True,
                "available": True,
                "query": query.strip(),
                "categories": categories,
                "result_count": len(formatted_results),
                "results": formatted_results,
            }
        )


def create_web_search_tool(
    searxng_url: str | None = None,
    timeout: int = 10,
    max_results: int = 5,
) -> WebSearchTool:
    """Create a WebSearchTool with configuration.

    Parameters
    ----------
    searxng_url : str | None
        SearXNG instance URL. If None, tool will gracefully degrade.
    timeout : int
        Request timeout in seconds.
    max_results : int
        Maximum number of results to return.

    Returns
    -------
    WebSearchTool
        Configured web search tool.
    """
    tool = WebSearchTool()
    tool.searxng_url = searxng_url
    tool.timeout = timeout
    tool.max_results = max_results
    return tool
