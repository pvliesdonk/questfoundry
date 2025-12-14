"""
Web Fetch tool implementation.

Fetches content from a URL and extracts article text.
Uses trafilatura for extraction with regex fallback.

Features:
- Article text extraction
- Rate limiting to avoid abuse
- Timeout handling
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolExecutionError, ToolResult
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)

# Rate limiting
MIN_REQUEST_INTERVAL = 3.0  # seconds between requests
_last_request_time: float = 0.0

# Request configuration
REQUEST_TIMEOUT = 15.0  # seconds
MAX_CONTENT_LENGTH = 1_000_000  # 1MB max
USER_AGENT = "QuestFoundry/1.0 (https://github.com/pvliesdonk/questfoundry)"


@register_tool("web_fetch")
class WebFetchTool(BaseTool):
    """
    Fetch and extract content from a URL.

    Extracts article text using trafilatura, with regex fallback.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_lock = asyncio.Lock()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute URL fetch."""
        url = args.get("url", "")
        extract_mode = args.get("extract_mode", "article")  # article, raw, or metadata

        if not url.strip():
            return ToolResult(
                success=False,
                data={},
                error="URL cannot be empty",
            )

        # Validate URL
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False,
                data={},
                error="URL must start with http:// or https://",
            )

        # Rate limiting
        await self._enforce_rate_limit()

        try:
            # Fetch content
            html = await self._fetch_url(url)

            # Extract based on mode
            if extract_mode == "raw":
                content = html[:50000]  # Truncate raw HTML
            elif extract_mode == "metadata":
                content = self._extract_metadata(html)
            else:
                content = self._extract_article(html, url)

            return ToolResult(
                success=True,
                data={
                    "url": url,
                    "content": content,
                    "content_length": len(content),
                    "extract_mode": extract_mode,
                },
            )

        except Exception as e:
            raise ToolExecutionError(f"Web fetch failed: {e}") from e

    async def _enforce_rate_limit(self) -> None:
        """Enforce minimum interval between requests (async-safe)."""
        global _last_request_time

        async with self._rate_limit_lock:
            now = time.time()
            elapsed = now - _last_request_time

            if elapsed < MIN_REQUEST_INTERVAL:
                await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)

            _last_request_time = time.time()

    async def _fetch_url(self, url: str) -> str:
        """Fetch URL content."""
        import httpx

        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_CONTENT_LENGTH:
                raise ToolExecutionError(f"Content too large: {content_length} bytes")

            return response.text

    def _extract_article(self, html: str, _url: str) -> str:
        """
        Extract article text from HTML.

        Uses trafilatura if available, falls back to regex extraction.
        """
        # Try trafilatura first
        try:
            import trafilatura  # type: ignore[import-not-found]

            result: str | None = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )

            if result:
                return result

        except ImportError:
            logger.debug("trafilatura not available, using regex fallback")
        except Exception as e:
            logger.debug(f"trafilatura extraction failed: {e}")

        # Regex fallback
        return self._regex_extract(html)

    def _regex_extract(self, html: str) -> str:
        """
        Basic regex-based text extraction.

        Removes scripts, styles, and HTML tags.
        """
        # Remove script and style elements
        html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html, flags=re.IGNORECASE)

        # Remove HTML comments
        html = re.sub(r"<!--[\s\S]*?-->", "", html)

        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Decode HTML entities
        text = self._decode_html_entities(text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Truncate if too long
        if len(text) > 20000:
            text = text[:20000] + "..."

        return text

    def _decode_html_entities(self, text: str) -> str:
        """Decode common HTML entities."""
        entities = {
            "&nbsp;": " ",
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
            "&#39;": "'",
        }

        for entity, char in entities.items():
            text = text.replace(entity, char)

        # Handle numeric entities
        text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
        text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)

        return text

    def _extract_metadata(self, html: str) -> str:
        """Extract page metadata (title, description, etc.)."""
        metadata = {}

        # Title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # Meta description
        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
            html,
            re.IGNORECASE,
        )
        if desc_match:
            metadata["description"] = desc_match.group(1).strip()

        # Open Graph
        og_matches = re.findall(
            r'<meta[^>]*property=["\']og:(\w+)["\'][^>]*content=["\']([^"\']*)["\']',
            html,
            re.IGNORECASE,
        )
        for prop, value in og_matches:
            metadata[f"og_{prop}"] = value.strip()

        return "\n".join(f"{k}: {v}" for k, v in metadata.items())
