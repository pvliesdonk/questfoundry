"""Corpus consultation tool for v4 runtime.

Provides RAG-style search over corpus knowledge:
1. Studio-specific corpus entries (if defined in studio)
2. Shared QuestFoundry corpus (genre conventions, writing craft, etc.)

When the [rag] extra is installed (sentence-transformers + sqlite-vec),
uses vector similarity search. Otherwise falls back to keyword search.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)

# Try to import vector search
try:
    from questfoundry.runtime.knowledge.corpus_vectors import (
        VECTOR_SEARCH_AVAILABLE,
        CorpusVectorStore,
        get_corpus_vector_store,
    )
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    CorpusVectorStore = None  # type: ignore
    get_corpus_vector_store = None  # type: ignore


class ConsultCorpusTool(BaseTool):
    """Search corpus knowledge for relevant excerpts.

    Searches two types of corpus:
    1. Studio-specific corpus entries (defined in studio knowledge)
    2. Shared QuestFoundry corpus (writing craft, genre conventions)

    If [rag] extra is installed, uses semantic vector search.
    Otherwise falls back to keyword-based search.
    """

    name: str = "consult_corpus"
    description: str = (
        "Search corpus knowledge for relevant excerpts. "
        "Use for writing craft advice, genre conventions, and reference material. "
        "Input: query (what to find), max_results (optional, default 3). "
        "Optionally: knowledge_id to search a specific corpus entry."
    )

    # Injected by registry
    studio: Any = Field(default=None, exclude=True)
    agent: Any = Field(default=None, exclude=True)

    # Vector store (lazy loaded)
    vector_store: Any = Field(default=None, exclude=True)
    vector_store_checked: bool = Field(default=False, exclude=True)

    def _run(
        self,
        query: str,
        max_results: int = 3,
        knowledge_id: str | None = None,
    ) -> str:
        """Search corpus for relevant excerpts.

        Parameters
        ----------
        query : str
            Natural language search query
        max_results : int
            Maximum number of excerpts to return (default: 3)
        knowledge_id : str | None
            Optional: specific corpus entry to search (studio-defined)

        Returns
        -------
        str
            JSON result with matching excerpts and metadata
        """
        if not query or not query.strip():
            return json.dumps({
                "success": False,
                "error": "query is required",
                "hint": "Provide a search query describing what you're looking for.",
            })

        query = query.strip()

        # If specific knowledge_id provided, search that entry
        if knowledge_id:
            return self._search_studio_corpus(knowledge_id, query, max_results)

        # Otherwise, search shared corpus (with vector if available)
        return self._search_shared_corpus(query, max_results)

    def _search_shared_corpus(self, query: str, max_results: int) -> str:
        """Search the shared QuestFoundry corpus."""
        # Try vector search first
        if self._try_vector_search():
            try:
                results = self.vector_store.search(query, k=max_results)
                if results:
                    return json.dumps({
                        "success": True,
                        "source": "shared_corpus",
                        "search_method": "vector",
                        "query": query,
                        "excerpt_count": len(results),
                        "excerpts": [
                            {
                                "excerpt": r["chunk_text"],
                                "source_file": r["file_name"],
                                "relevance_score": round(r["score"], 3),
                            }
                            for r in results
                        ],
                    })
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keyword: {e}")

        # Fallback: keyword search on corpus files
        return self._keyword_search_corpus_files(query, max_results)

    def _try_vector_search(self) -> bool:
        """Try to initialize vector search, return True if available."""
        if self.vector_store_checked:
            return self.vector_store is not None

        self.vector_store_checked = True

        if not VECTOR_SEARCH_AVAILABLE:
            logger.debug(
                "Vector search not available. "
                "Install with: uv pip install questfoundry[rag]"
            )
            return False

        try:
            self.vector_store = get_corpus_vector_store()
            if self.vector_store.ensure_indexed():
                stats = self.vector_store.get_stats()
                logger.info(
                    f"Corpus vector search ready: "
                    f"{stats.get('chunk_count', 0)} chunks from "
                    f"{stats.get('file_count', 0)} files"
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to initialize vector store: {e}")
            self.vector_store = None

        return False

    def _keyword_search_corpus_files(self, query: str, max_results: int) -> str:
        """Keyword search over corpus markdown files."""
        # Find corpus directory
        corpus_dir = self._find_corpus_dir()
        if not corpus_dir or not corpus_dir.exists():
            return json.dumps({
                "success": False,
                "error": "No corpus files found",
                "hint": "Corpus files should be in domain-v4/knowledge/corpus/",
            })

        # Load and search all corpus files
        all_excerpts: list[dict[str, Any]] = []
        query_words = set(re.findall(r"\w+", query.lower()))

        for file_path in corpus_dir.glob("**/*.md"):
            content = file_path.read_text(encoding="utf-8")
            excerpts = self._keyword_search_content(
                content, query_words, file_path.name
            )
            all_excerpts.extend(excerpts)

        # Sort by score and take top results
        all_excerpts.sort(key=lambda x: -x["relevance_score"])
        top_excerpts = all_excerpts[:max_results]

        if not top_excerpts:
            return json.dumps({
                "success": True,
                "source": "shared_corpus",
                "search_method": "keyword",
                "query": query,
                "excerpts": [],
                "message": f"No relevant excerpts found for '{query}'.",
            })

        return json.dumps({
            "success": True,
            "source": "shared_corpus",
            "search_method": "keyword",
            "query": query,
            "excerpt_count": len(top_excerpts),
            "excerpts": top_excerpts,
        })

    def _find_corpus_dir(self) -> Path | None:
        """Find the corpus directory."""
        # Try relative to module
        module_dir = Path(__file__).parent
        possible_paths = [
            module_dir.parents[3] / "domain-v4" / "knowledge" / "corpus",
            Path.cwd() / "domain-v4" / "knowledge" / "corpus",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return None

    def _keyword_search_content(
        self,
        content: str,
        query_words: set[str],
        source_file: str,
    ) -> list[dict[str, Any]]:
        """Keyword search within content."""
        paragraphs = self._split_into_chunks(content)
        results: list[dict[str, Any]] = []

        for para in paragraphs:
            if not para.strip():
                continue

            score = self._score_paragraph(para, query_words)
            if score > 0:
                results.append({
                    "excerpt": para.strip()[:500],  # Truncate long excerpts
                    "source_file": source_file,
                    "relevance_score": round(score, 3),
                })

        return results

    def _search_studio_corpus(
        self, knowledge_id: str, query: str, max_results: int
    ) -> str:
        """Search a specific studio-defined corpus entry."""
        if not self.studio:
            return json.dumps({
                "success": False,
                "error": "Studio not configured. Cannot search corpus entries.",
            })

        entry = self.studio.knowledge_entries.get(knowledge_id)
        if not entry:
            available = self._get_corpus_entries()
            return json.dumps({
                "success": False,
                "error": f"Knowledge entry '{knowledge_id}' not found",
                "available_corpus_entries": available,
                "hint": "Omit knowledge_id to search the shared corpus.",
            })

        if entry.content.type != "corpus":
            return json.dumps({
                "success": False,
                "error": f"Knowledge entry '{knowledge_id}' is not a corpus type",
                "hint": "Use consult_knowledge for inline content.",
            })

        if not self._can_access(entry):
            return json.dumps({
                "success": False,
                "error": f"Knowledge entry '{knowledge_id}' is not accessible.",
            })

        content = self._get_corpus_content(entry)
        if not content:
            return json.dumps({
                "success": False,
                "error": f"Corpus '{knowledge_id}' has no searchable content.",
            })

        query_words = set(re.findall(r"\w+", query.lower()))
        excerpts = self._keyword_search_content(content, query_words, knowledge_id)
        excerpts.sort(key=lambda x: -x["relevance_score"])
        top_excerpts = excerpts[:max_results]

        return json.dumps({
            "success": True,
            "source": "studio_corpus",
            "knowledge_id": knowledge_id,
            "search_method": "keyword",
            "query": query,
            "excerpt_count": len(top_excerpts),
            "excerpts": top_excerpts,
        })

    def _get_corpus_entries(self) -> list[str]:
        """Get list of corpus-type knowledge entries from studio."""
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

        if content.text:
            return content.text

        if content.path:
            logger.warning(f"Corpus file ref not supported: {content.path}")
            return None

        if hasattr(content, "chunks") and content.chunks:
            return "\n\n".join(content.chunks)

        return None

    def _split_into_chunks(self, content: str) -> list[str]:
        """Split content into searchable chunks."""
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)

        if len(paragraphs) > 1 and all(len(p) < 2000 for p in paragraphs):
            return paragraphs

        # Try markdown header splits
        sections = re.split(r"\n(?=#{1,3}\s)", content)
        if len(sections) > 1:
            return sections

        # Fallback: fixed-size chunks
        chunk_size = 500
        overlap = 100
        chunks: list[str] = []
        pos = 0
        while pos < len(content):
            end = pos + chunk_size
            chunk = content[pos:end]
            if end < len(content):
                last_period = chunk.rfind(". ")
                if last_period > chunk_size // 2:
                    chunk = chunk[: last_period + 1]
                    end = pos + last_period + 1
            chunks.append(chunk)
            pos = end - overlap if end < len(content) else len(content)

        return chunks

    def _score_paragraph(self, paragraph: str, query_words: set[str]) -> float:
        """Score a paragraph's relevance to query words."""
        para_lower = paragraph.lower()
        para_words = set(re.findall(r"\w+", para_lower))

        overlap = query_words & para_words
        if not overlap:
            return 0.0

        word_coverage = len(overlap) / len(query_words)
        match_count = sum(para_lower.count(word) for word in overlap)
        word_count = len(para_words)
        density = match_count / max(word_count, 1)

        score = (word_coverage * 0.7) + (density * 0.3)

        # Boost for exact phrase match
        query_phrase = " ".join(sorted(query_words))
        if query_phrase in para_lower:
            score *= 1.5

        return score


def create_consult_corpus_tool(studio: Any = None, agent: Any = None) -> ConsultCorpusTool:
    """Factory function to create a consult corpus tool.

    Args:
        studio: The loaded studio (for studio-specific corpus entries)
        agent: The agent who will use this tool (for access control)

    Returns:
        Configured ConsultCorpusTool
    """
    tool = ConsultCorpusTool()
    tool.studio = studio
    tool.agent = agent
    return tool
