"""
Web Search tool implementation.

Provides web search via SearXNG (self-hosted metasearch engine).
Falls back to unavailable if SearXNG is not configured.

Configuration:
- QF_SEARXNG__URL: Base URL for SearXNG instance (e.g., http://localhost:8888)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolExecutionError, ToolResult
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)

# Valid recency values
VALID_RECENCY_VALUES = {"all_time", "day", "week", "month", "year"}

# Domain filter validation pattern - allows valid domain characters only
# Matches: example.com, sub.example.com, .edu, gov, etc.
DOMAIN_FILTER_PATTERN = re.compile(
    r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*$"
)

# SearXNG configuration
SEARXNG_URL = os.environ.get("QF_SEARXNG__URL", "")
SEARXNG_TIMEOUT = 10.0  # seconds


@register_tool("web_search")
class WebSearchTool(BaseTool):
    """
    Search the web using SearXNG metasearch engine.

    Requires QF_SEARXNG__URL environment variable to be configured.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._health_checked = False
        self._is_healthy = False

    def check_availability(self) -> bool:
        """Check if SearXNG is configured and reachable."""
        if not SEARXNG_URL:
            logger.debug("SearXNG not configured (QF_SEARXNG__URL not set)")
            return False

        # Cache health check result
        if not self._health_checked:
            self._is_healthy = self._check_searxng_health()
            self._health_checked = True

        return self._is_healthy

    def _check_searxng_health(self) -> bool:
        """Check if SearXNG is reachable.

        NOTE: This is a blocking HTTP call. While not ideal in async context,
        it's only called once per tool instance (result is cached) and happens
        during tool availability checking, not during execute().

        TODO: Consider lazy async health check on first execute() call instead.
        """
        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{SEARXNG_URL}/healthz")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"SearXNG health check failed: {e}")
            return False

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute web search."""
        query = args.get("query", "")
        max_results = args.get("max_results", 5)
        domain_filter = args.get("domain_filter")
        recency = args.get("recency", "all_time")

        if not query.strip():
            return ToolResult(
                success=False,
                data={"results": []},
                error="Search query cannot be empty",
            )

        # Validate domain_filter to prevent query injection
        if domain_filter and not DOMAIN_FILTER_PATTERN.match(domain_filter):
            return ToolResult(
                success=False,
                data={"results": []},
                error=f"Invalid domain_filter: '{domain_filter}'. Must be a valid domain (e.g., 'wikipedia.org', 'gov').",
            )

        # Validate recency value
        if recency not in VALID_RECENCY_VALUES:
            logger.warning(
                "WebSearchTool: Unrecognized recency value '%s'; defaulting to 'all_time'.",
                recency,
            )
            recency = "all_time"

        try:
            # Build actual search query with domain filter if specified
            search_query = f"site:{domain_filter} {query}" if domain_filter else query

            results = await self._search(search_query, max_results, recency)

            return ToolResult(
                success=True,
                data={
                    "query_used": search_query,
                    "result_count": len(results),
                    "results": results,
                },
            )

        except Exception as e:
            raise ToolExecutionError(f"Web search failed: {e}") from e

    async def _search(
        self,
        query: str,
        max_results: int,
        recency: str,
    ) -> list[dict[str, Any]]:
        """Perform search via SearXNG API.

        Args:
            query: Search query string (may include site: prefix for domain filtering)
            max_results: Maximum results to return (applied client-side; SearXNG
                doesn't support a results limit parameter)
            recency: Time filter - 'all_time', 'day', 'week', 'month', 'year'
        """
        import httpx

        # Map recency to SearXNG time_range parameter
        # Note: recency is already validated in execute(), so we use direct lookup
        time_range_map: dict[str, str | None] = {
            "all_time": None,
            "day": "day",
            "week": "week",
            "month": "month",
            "year": "year",
        }
        time_range = time_range_map[recency]

        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": "general",
        }
        if time_range:
            params["time_range"] = time_range

        async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT) as client:
            response = await client.get(f"{SEARXNG_URL}/search", params=params)
            response.raise_for_status()

            data = response.json()

        # Extract results (max_results applied client-side as SearXNG returns full page)
        results = []
        for item in data.get("results", [])[:max_results]:
            result: dict[str, Any] = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            # Include published_date if available
            if item.get("publishedDate"):
                result["published_date"] = item.get("publishedDate")
            results.append(result)

        return results


# =============================================================================
# Alternative: Mock search for testing
# =============================================================================


class MockWebSearchTool(BaseTool):
    """
    Mock web search for testing without SearXNG.

    Returns placeholder results.
    """

    def check_availability(self) -> bool:
        return True

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query", "")

        return ToolResult(
            success=True,
            data={
                "query": query,
                "result_count": 0,
                "results": [],
                "_mock": True,
                "_message": "Web search not available - SearXNG not configured",
            },
        )
