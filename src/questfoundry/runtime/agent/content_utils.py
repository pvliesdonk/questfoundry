"""
Shared utilities for knowledge content extraction.

This module provides common functionality for extracting content from
knowledge entries, used by both knowledge.py and consult_knowledge.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.agent.structured_renderer import render_structured_entry

if TYPE_CHECKING:
    from questfoundry.runtime.models.base import KnowledgeContent, KnowledgeEntry

logger = logging.getLogger(__name__)


def extract_knowledge_content(
    entry: KnowledgeEntry,
    domain_path: Path | None = None,
) -> str | None:
    """Extract text content from a knowledge entry.

    For structured content, renders semantic types (rules, contracts, etc.)
    to readable markdown using the structured_renderer.

    For file_ref content, loads the file if domain_path is provided.

    Args:
        entry: The knowledge entry to extract content from
        domain_path: Optional path to domain directory for file_ref loading

    Returns:
        Rendered content string, or None if no content available
    """
    content = entry.content
    if content is None:
        return None

    # Handle dict content (raw from JSON)
    if isinstance(content, dict):
        return _extract_from_dict(content, domain_path)

    # Handle KnowledgeContent model
    # Import here to avoid circular dependency
    from questfoundry.runtime.models.base import KnowledgeContent as KC

    if isinstance(content, KC):
        return _extract_from_model(content, domain_path)

    return str(content) if content else None


def _extract_from_dict(
    content: dict[str, Any],
    domain_path: Path | None = None,
) -> str | None:
    """Extract content from a dict (raw JSON form)."""
    content_type = content.get("type", "structured")

    if content_type == "structured":
        data = content.get("data", {})
        if data:
            return render_structured_entry(data)
        return None

    elif content_type == "file_ref":
        file_path = content.get("file_path")
        return _load_file_content(file_path, domain_path)

    elif content_type == "corpus":
        return "(Corpus entry - use consult_corpus tool)"

    return None


def _extract_from_model(
    content: KnowledgeContent,
    domain_path: Path | None = None,
) -> str | None:
    """Extract content from a KnowledgeContent model."""
    if content.type == "structured":
        if content.data:
            return render_structured_entry(content.data)
        return None

    elif content.type == "file_ref":
        return _load_file_content(content.file_path, domain_path)

    elif content.type == "corpus":
        return "(Corpus entry - use consult_corpus tool)"

    return None


def _load_file_content(
    file_path: str | None,
    domain_path: Path | None = None,
) -> str:
    """Load content from a file reference.

    Args:
        file_path: Relative path to the content file
        domain_path: Base path for resolving file references

    Returns:
        File content or placeholder message if loading fails
    """
    if not file_path:
        return "(No file path specified)"

    if not domain_path:
        logger.warning(
            "file_ref content type encountered without domain_path; "
            "consider migrating to structured content: %s",
            file_path,
        )
        return f"(Content in file: {file_path})"

    full_path = domain_path / file_path

    if not full_path.is_file():
        logger.warning("Knowledge file not found: %s", full_path)
        return f"(Content file not found: {file_path})"

    try:
        return full_path.read_text()
    except OSError as e:
        logger.warning("Failed to read knowledge file '%s': %s", full_path, e)
        return f"(Error loading content from file: {file_path})"
