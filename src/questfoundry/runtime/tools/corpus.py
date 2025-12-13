"""Corpus consultation tool for v4 runtime.

Provides RAG-style vector search over corpus-type knowledge entries.
Corpus entries are large reference materials (genre conventions, writing craft,
world lore) that are too large to embed in prompts but can be searched.

For now, implements keyword-based search with future hooks for vector stores.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class ConsultCorpusTool(BaseTool):
    """Search corpus-type knowledge entries for relevant excerpts.

    Use this to find specific information within large reference materials
    like genre conventions, writing craft guides, or world lore compendiums.

    This tool performs semantic search (keyword-based for now, vector-based
    in future) and returns relevant excerpts with context.
    """

    name: str = "consult_corpus"
    description: str = (
        "Search a corpus knowledge entry for relevant excerpts. "
        "Use when you need specific information from large reference materials. "
        "Input: knowledge_id (the corpus entry to search), query (what to find), "
        "max_results (optional, default 3)"
    )

    # Injected by registry
    studio: Any = Field(default=None, exclude=True)
    agent: Any = Field(default=None, exclude=True)

    def _run(
        self,
        knowledge_id: str,
        query: str,
        max_results: int = 3,
    ) -> str:
        """Search a corpus entry for relevant excerpts.

        Parameters
        ----------
        knowledge_id : str
            ID of the corpus knowledge entry to search
        query : str
            Natural language search query
        max_results : int
            Maximum number of excerpts to return (default: 3)

        Returns
        -------
        str
            JSON result with matching excerpts and metadata
        """
        if not self.studio:
            return json.dumps({
                "success": False,
                "error": "Studio not configured. Cannot search corpus.",
            })

        if not knowledge_id:
            return json.dumps({
                "success": False,
                "error": "knowledge_id is required",
                "hint": "Provide the ID of a corpus knowledge entry to search.",
            })

        if not query or not query.strip():
            return json.dumps({
                "success": False,
                "error": "query is required",
                "hint": "Provide a search query describing what you're looking for.",
            })

        # Get the knowledge entry
        entry = self.studio.knowledge_entries.get(knowledge_id)
        if not entry:
            available = self._get_corpus_entries()
            return json.dumps({
                "success": False,
                "error": f"Knowledge entry '{knowledge_id}' not found",
                "available_corpus_entries": available,
            })

        # Check if it's a corpus-type entry
        if entry.content.type != "corpus":
            return json.dumps({
                "success": False,
                "error": f"Knowledge entry '{knowledge_id}' is not a corpus type",
                "hint": (
                    "consult_corpus is for searching large corpus entries. "
                    "For inline content, use consult_knowledge instead."
                ),
            })

        # Check access
        if not self._can_access(entry):
            return json.dumps({
                "success": False,
                "error": f"Knowledge entry '{knowledge_id}' is not accessible to your role.",
            })

        # Get corpus content
        content = self._get_corpus_content(entry)
        if not content:
            return json.dumps({
                "success": False,
                "error": f"Corpus '{knowledge_id}' has no searchable content.",
            })

        # Search the corpus
        excerpts = self._search_corpus(content, query.strip(), max_results)

        if not excerpts:
            return json.dumps({
                "success": True,
                "knowledge_id": knowledge_id,
                "query": query,
                "excerpts": [],
                "message": f"No relevant excerpts found for '{query}' in {knowledge_id}.",
            })

        return json.dumps({
            "success": True,
            "knowledge_id": knowledge_id,
            "query": query,
            "excerpt_count": len(excerpts),
            "excerpts": excerpts,
        })

    def _get_corpus_entries(self) -> list[str]:
        """Get list of corpus-type knowledge entries."""
        if not self.studio:
            return []

        corpus_ids = []
        for entry_id, entry in self.studio.knowledge_entries.items():
            if entry.content.type == "corpus" and self._can_access(entry):
                corpus_ids.append(entry_id)

        return sorted(corpus_ids)

    def _can_access(self, entry: Any) -> bool:
        """Check if the agent can access this entry."""
        if not entry.applicable_to:
            return True

        if not self.agent:
            return True

        if entry.applicable_to.agents and self.agent.id in entry.applicable_to.agents:
            return True

        if entry.applicable_to.archetypes:
            for archetype in self.agent.archetypes:
                if archetype in entry.applicable_to.archetypes:
                    return True

        return not (entry.applicable_to.agents or entry.applicable_to.archetypes)

    def _get_corpus_content(self, entry: Any) -> str | None:
        """Get searchable content from a corpus entry."""
        content = entry.content

        # Corpus entries can have:
        # - text: inline large text
        # - path: file reference
        # - chunks: pre-chunked content for vector search

        if content.text:
            return content.text

        if content.path:
            # TODO: Load from file
            logger.warning(f"Corpus file ref not supported: {content.path}")
            return None

        if hasattr(content, "chunks") and content.chunks:
            # Concatenate chunks for keyword search
            return "\n\n".join(content.chunks)

        return None

    def _search_corpus(
        self,
        content: str,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search corpus content for relevant excerpts.

        Current implementation: keyword-based paragraph search
        Future: vector embedding similarity search
        """
        # Normalize query
        query_words = set(re.findall(r'\w+', query.lower()))

        # Split content into paragraphs/chunks
        paragraphs = self._split_into_chunks(content)

        # Score each paragraph
        scored: list[tuple[float, int, str]] = []
        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue

            score = self._score_paragraph(para, query_words)
            if score > 0:
                scored.append((score, idx, para))

        # Sort by score descending
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Take top results
        results: list[dict[str, Any]] = []
        for score, idx, para in scored[:max_results]:
            results.append({
                "excerpt": para.strip(),
                "relevance_score": round(score, 2),
                "position": idx,
            })

        return results

    def _split_into_chunks(self, content: str) -> list[str]:
        """Split content into searchable chunks.

        Tries to split on:
        1. Double newlines (paragraphs)
        2. Section headers (markdown)
        3. Fixed-size chunks as fallback
        """
        # Try paragraph split first
        paragraphs = re.split(r'\n\s*\n', content)

        # If we get reasonable chunks, use them
        if len(paragraphs) > 1 and all(len(p) < 2000 for p in paragraphs):
            return paragraphs

        # Otherwise, use markdown header splits
        sections = re.split(r'\n(?=#{1,3}\s)', content)
        if len(sections) > 1:
            return sections

        # Fallback: fixed-size chunks with overlap
        chunk_size = 500
        overlap = 100
        chunks: list[str] = []
        pos = 0
        while pos < len(content):
            end = pos + chunk_size
            chunk = content[pos:end]
            # Try to end at sentence boundary
            if end < len(content):
                last_period = chunk.rfind('. ')
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = pos + last_period + 1
            chunks.append(chunk)
            pos = end - overlap if end < len(content) else len(content)

        return chunks

    def _score_paragraph(self, paragraph: str, query_words: set[str]) -> float:
        """Score a paragraph's relevance to query words."""
        para_lower = paragraph.lower()
        para_words = set(re.findall(r'\w+', para_lower))

        # Base score: word overlap
        overlap = query_words & para_words
        if not overlap:
            return 0.0

        # Score based on:
        # - Percentage of query words found
        # - Density of matches
        word_coverage = len(overlap) / len(query_words)

        # Count word occurrences for density
        match_count = sum(para_lower.count(word) for word in overlap)
        word_count = len(para_words)
        density = match_count / max(word_count, 1)

        # Combine scores
        score = (word_coverage * 0.7) + (density * 0.3)

        # Boost for exact phrase match
        query_phrase = ' '.join(sorted(query_words))
        if query_phrase in para_lower:
            score *= 1.5

        return score


def create_consult_corpus_tool(studio: Any, agent: Any = None) -> ConsultCorpusTool:
    """Factory function to create a consult corpus tool.

    Args:
        studio: The loaded studio
        agent: The agent who will use this tool (for access control)

    Returns:
        Configured ConsultCorpusTool
    """
    tool = ConsultCorpusTool()
    tool.studio = studio
    tool.agent = agent
    return tool
