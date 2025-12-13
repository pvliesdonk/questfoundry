"""Web fetch tool for v4 runtime.

Fetches and extracts content from URLs for research purposes.
Handles HTML cleanup and content extraction with graceful degradation.

If trafilatura is available, uses it for high-quality article extraction.
Otherwise falls back to basic HTML-to-text conversion.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time

import httpx
from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)

# Try to import trafilatura for better content extraction
try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.debug("trafilatura not available, using basic HTML extraction")


class WebFetchTool(BaseTool):
    """Fetch and extract content from a URL.

    Use this tool to read web pages for research, fact-checking, and
    gathering information. Returns cleaned text content suitable for
    analysis.

    For discovering URLs, use web_search first, then web_fetch to read
    specific pages in detail.
    """

    name: str = "web_fetch"
    description: str = (
        "Fetch and extract content from a URL. "
        "Use extract_mode='article' for main content only, "
        "'full' for all text, or 'summary' for a brief overview. "
        "Returns cleaned text content with metadata."
    )

    # Configuration
    timeout: int = Field(default=30)
    max_length: int = Field(default=10000)
    user_agent: str = Field(
        default="QuestFoundry/1.0 (Research Bot; +https://github.com/pvliesdonk/questfoundry)"
    )

    # Rate limiting - class level (not per-instance)
    last_request_time: float = Field(default=0.0, exclude=True)
    min_request_interval: float = Field(default=3.0, exclude=True)  # 20 req/min = 3s interval

    def _run(
        self,
        url: str,
        extract_mode: str = "article",
        max_length: int | None = None,
    ) -> str:
        """Fetch and extract content from a URL.

        Parameters
        ----------
        url : str
            The URL to fetch content from
        extract_mode : str
            Extraction mode: 'article' (main content), 'full' (all text),
            or 'summary' (truncated overview)
        max_length : int | None
            Maximum characters to return (default: 10000)

        Returns
        -------
        str
            JSON result with extracted content and metadata
        """
        # Validate extract_mode
        if extract_mode not in ("article", "full", "summary"):
            return json.dumps({
                "success": False,
                "error": f"Invalid extract_mode: {extract_mode}. "
                "Use 'article', 'full', or 'summary'.",
            })

        # Validate URL
        if not url.startswith(("http://", "https://")):
            return json.dumps({
                "success": False,
                "error": "URL must start with http:// or https://",
            })

        # Rate limiting
        self._apply_rate_limit()

        effective_max_length = max_length or self.max_length

        # Fetch the URL
        try:
            response = self._fetch_url(url)
        except httpx.TimeoutException:
            return json.dumps({
                "success": False,
                "error": f"Request timed out after {self.timeout}s. Try again later.",
            })
        except httpx.HTTPStatusError as e:
            return json.dumps({
                "success": False,
                "error": f"HTTP error {e.response.status_code}: {e.response.reason_phrase}",
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to fetch URL: {e}",
            })

        # Extract content based on mode
        html_content = response.text
        final_url = str(response.url)  # May differ if redirected

        if extract_mode == "article":
            content = self._extract_article(html_content, url)
        elif extract_mode == "full":
            content = self._extract_full(html_content)
        else:  # summary
            content = self._extract_article(html_content, url)
            # Truncate for summary
            if len(content) > 2000:
                content = content[:2000] + "..."

        # Extract metadata
        title = self._extract_title(html_content)
        published_date = self._extract_date(html_content)

        # Apply max_length
        truncated = False
        if len(content) > effective_max_length:
            content = content[:effective_max_length]
            truncated = True

        word_count = len(content.split())

        return json.dumps({
            "success": True,
            "title": title,
            "content": content,
            "url": final_url,
            "published_date": published_date,
            "word_count": word_count,
            "truncated": truncated,
        })

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _fetch_url(self, url: str) -> httpx.Response:
        """Fetch URL content with retry logic."""
        headers = {"User-Agent": self.user_agent}

        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response

    def _extract_article(self, html_content: str, url: str) -> str:
        """Extract main article content from HTML."""
        if TRAFILATURA_AVAILABLE:
            try:
                content = trafilatura.extract(
                    html_content,
                    url=url,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                )
                if content:
                    return content
            except Exception as e:
                logger.debug(f"trafilatura extraction failed: {e}")

        # Fallback to basic extraction
        return self._basic_extract(html_content)

    def _extract_full(self, html_content: str) -> str:
        """Extract all text content from HTML."""
        return self._basic_extract(html_content)

    def _basic_extract(self, html_content: str) -> str:
        """Basic HTML-to-text extraction without external libraries."""
        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<noscript[^>]*>.*?</noscript>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Remove nav, header, footer elements (common boilerplate)
        for tag in ["nav", "header", "footer", "aside", "menu"]:
            text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Convert common block elements to newlines
        text = re.sub(r"<(p|div|br|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", text, flags=re.IGNORECASE)

        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode HTML entities
        text = html.unescape(text)

        # Clean up whitespace
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Multiple newlines to double
        text = text.strip()

        return text

    def _extract_title(self, html_content: str) -> str:
        """Extract page title from HTML."""
        # Try <title> tag
        match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if match:
            title = html.unescape(match.group(1))
            return title.strip()

        # Try og:title meta tag
        match = re.search(
            r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
            html_content,
            re.IGNORECASE,
        )
        if match:
            return html.unescape(match.group(1)).strip()

        return ""

    def _extract_date(self, html_content: str) -> str | None:
        """Extract publication date from HTML if available."""
        # Try article:published_time meta tag
        match = re.search(
            r'<meta[^>]*property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)["\']',
            html_content,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)

        # Try datePublished schema.org
        match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html_content)
        if match:
            return match.group(1)

        return None
