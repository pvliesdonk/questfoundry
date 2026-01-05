"""Web research tools for current information.

This module provides LangChain-compatible tools for web research:
- WebSearchTool: Search the web via SearXNG
- WebFetchTool: Fetch and extract content from URLs

Requires:
    - SEARXNG_URL environment variable for web search
    - pvl-webtools package: uv add pvl-webtools

All tools return structured JSON following ADR-008:
- result: semantic status (success, no_results, error)
- data/content: the actual result data
- action: guidance on what to do next (never instructs looping)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from questfoundry.tools.base import Tool, ToolDefinition

logger = logging.getLogger(__name__)

# Maximum characters for tool output
MAX_OUTPUT_CHARS = 4000  # ~1000 tokens


class WebToolsNotAvailableError(Exception):
    """Raised when pvl-webtools is not installed."""

    def __init__(self) -> None:
        super().__init__("pvl-webtools not available. Install with: uv add pvl-webtools")


class SearXNGNotConfiguredError(Exception):
    """Raised when SEARXNG_URL is not configured."""

    def __init__(self) -> None:
        super().__init__("SEARXNG_URL not configured. Set environment variable for web search.")


def _web_tools_available() -> bool:
    """Check if pvl-webtools is installed."""
    try:
        import pvlwebtools  # noqa: F401

        return True
    except ImportError:
        return False


def _searxng_configured() -> bool:
    """Check if SEARXNG_URL is configured."""
    return bool(os.environ.get("SEARXNG_URL"))


class WebSearchTool:
    """Search the web for current information.

    Uses SearXNG metasearch engine. Requires SEARXNG_URL
    environment variable to be configured.

    Returns title, URL, and snippet for each result.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information and trends.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 5)",
                    },
                    "recency": {
                        "type": "string",
                        "description": "Filter by time: all_time, day, week, month, year",
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute web search.

        Args:
            arguments: query (required), max_results (optional), recency (optional)

        Returns:
            Structured JSON response following ADR-008.
        """
        if not _web_tools_available():
            return json.dumps(
                {
                    "result": "error",
                    "error": "pvl-webtools not installed",
                    "action": "Web search unavailable. Proceed with your own knowledge.",
                }
            )

        if not _searxng_configured():
            return json.dumps(
                {
                    "result": "error",
                    "error": "SEARXNG_URL not configured",
                    "action": "Web search unavailable. Proceed with your own knowledge.",
                }
            )

        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        recency = arguments.get("recency", "all_time")

        try:
            from pvlwebtools import web_search

            # Run async function in sync context
            # Try to get running loop first (if called from async context),
            # otherwise create a new one
            try:
                asyncio.get_running_loop()
                # We're in an async context - run in thread to avoid blocking
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        web_search(query, max_results=max_results, recency=recency),
                    )
                    results = future.result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run()
                results = asyncio.run(web_search(query, max_results=max_results, recency=recency))
        except Exception as e:
            logger.warning("Web search failed: %s", e)
            return json.dumps(
                {
                    "result": "error",
                    "error": f"Search failed: {e}",
                    "action": "Web search failed. Proceed with your own knowledge.",
                }
            )

        if not results:
            return json.dumps(
                {
                    "result": "no_results",
                    "query": query,
                    "action": "No web results found. Proceed with your own knowledge.",
                }
            )

        # Format results with output size tracking
        formatted = []
        total_chars = 0

        for r in results:
            entry = f"**{r.title}**\n{r.url}\n{r.snippet}"
            if hasattr(r, "published_date") and r.published_date:
                entry += f"\n*Published: {r.published_date}*"

            # Check total output size
            if total_chars + len(entry) > MAX_OUTPUT_CHARS:
                formatted.append("\n*...additional results truncated*")
                break

            formatted.append(entry)
            total_chars += len(entry)

        return json.dumps(
            {
                "result": "success",
                "query": query,
                "count": len(results),
                "content": "\n\n---\n\n".join(formatted),
                "action": "Use this information to inform your creative decisions.",
            }
        )


class WebFetchTool:
    """Fetch and extract content from a URL.

    Retrieves web page content with intelligent extraction:
    - Strips navigation, ads, and boilerplate
    - Extracts main content as markdown
    - Truncates to fit context window

    Rate-limited to avoid overloading servers.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="web_fetch",
            description="Fetch page content from URL. Extracts main text as markdown.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch",
                    },
                    "extract_mode": {
                        "type": "string",
                        "description": "Extraction: markdown (default), article, metadata",
                    },
                },
                "required": ["url"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute web fetch.

        Args:
            arguments: url (required), extract_mode (optional)

        Returns:
            Structured JSON response following ADR-008.
        """
        if not _web_tools_available():
            return json.dumps(
                {
                    "result": "error",
                    "error": "pvl-webtools not installed",
                    "action": "Web fetch unavailable. Proceed with your own knowledge.",
                }
            )

        url = arguments["url"]
        extract_mode = arguments.get("extract_mode", "markdown")

        try:
            from pvlwebtools import web_fetch

            # Run async function in sync context
            # Try to get running loop first (if called from async context),
            # otherwise create a new one
            try:
                asyncio.get_running_loop()
                # We're in an async context - run in thread to avoid blocking
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        web_fetch(url, extract_mode=extract_mode),
                    )
                    result = future.result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run()
                result = asyncio.run(web_fetch(url, extract_mode=extract_mode))
        except Exception as e:
            logger.warning("Web fetch failed for %s: %s", url, e)
            return json.dumps(
                {
                    "result": "error",
                    "url": url,
                    "error": f"Fetch failed: {e}",
                    "action": "Could not fetch URL. Proceed with your own knowledge.",
                }
            )

        content = result.content

        # Truncate if too long
        if len(content) > MAX_OUTPUT_CHARS:
            content = content[:MAX_OUTPUT_CHARS] + "\n\n*...content truncated*"

        return json.dumps(
            {
                "result": "success",
                "url": result.url,
                "extract_mode": result.extract_mode,
                "content": content,
                "action": "Use this information to inform your creative decisions.",
            }
        )


def get_web_tools(require_searxng: bool = True) -> list[Tool]:
    """Get all web tools if library is available.

    Args:
        require_searxng: If True, only return tools if SEARXNG_URL is configured.

    Returns:
        List of web tools, or empty list if not available.
    """
    if not _web_tools_available():
        logger.info("pvl-webtools not installed, web tools disabled")
        return []

    tools: list[Tool] = []

    # WebSearch requires SEARXNG_URL
    if _searxng_configured():
        tools.append(WebSearchTool())
    elif require_searxng:
        logger.info("SEARXNG_URL not configured, web search disabled")
    else:
        # Still add it - will return helpful error message if called
        tools.append(WebSearchTool())

    # WebFetch doesn't require SEARXNG
    tools.append(WebFetchTool())

    return tools
