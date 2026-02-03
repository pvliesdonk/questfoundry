"""Utilities for normalizing LLM message content across providers.

Some providers (notably Google Gemini) return ``AIMessage.content`` as a list of
content-block dicts rather than a plain string.  This module provides a single
helper to extract readable text regardless of the underlying format.
"""

from __future__ import annotations

from typing import Any


def extract_text(content: str | list[Any]) -> str:
    """Extract plain text from an LLM message content field.

    Handles two formats:
    - ``str``: returned as-is.
    - ``list[dict]``: content blocks (e.g. Gemini).  Text is extracted from
      each block that has ``type == "text"`` and a ``text`` key, then joined
      with newlines.

    Falls back to ``str(content)`` for unexpected shapes so callers never crash.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    # Unexpected shape — degrade gracefully but at least stringify.
    return str(content)
