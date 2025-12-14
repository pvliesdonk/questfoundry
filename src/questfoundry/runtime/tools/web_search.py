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
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolExecutionError, ToolResult
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)

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
        categories = args.get("categories", ["general"])

        if not query.strip():
            return ToolResult(
                success=False,
                data={"results": []},
                error="Search query cannot be empty",
            )

        try:
            results = await self._search(query, max_results, categories)

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "result_count": len(results),
                    "results": results,
                },
            )

        except Exception as e:
            raise ToolExecutionError(f"Web search failed: {e}") from e

    async def _search(
        self, query: str, max_results: int, categories: list[str]
    ) -> list[dict[str, Any]]:
        """Perform search via SearXNG API."""
        import httpx

        params = {
            "q": query,
            "format": "json",
            "categories": ",".join(categories),
        }

        async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT) as client:
            response = await client.get(f"{SEARXNG_URL}/search", params=params)
            response.raise_for_status()

            data = response.json()

        # Extract results
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "engine": item.get("engine", ""),
                }
            )

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
