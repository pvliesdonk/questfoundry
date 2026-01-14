"""IF Craft Corpus tools for writing guidance.

This module provides LangChain-compatible tools for searching the
IF Craft Corpus, a curated knowledge base for interactive fiction writing.

Tools:
    SearchCorpusTool: Search for craft guidance by topic/query
    GetDocumentTool: Get full document by name
    ListClustersTool: Discover available topic clusters

The corpus is a singleton to avoid repeated index building.

All tools return structured JSON following ADR-008:
- result: semantic status (success, no_results, error)
- data/content: the actual result data
- action: guidance on what to do next (never instructs looping)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.tools.base import Tool, ToolDefinition

if TYPE_CHECKING:
    from ifcraftcorpus import Corpus

logger = logging.getLogger(__name__)

# Maximum tokens (approx chars / 4) for tool output
MAX_OUTPUT_CHARS = 4000  # ~1000 tokens

# Cache embeddings in user's cache directory
EMBEDDINGS_CACHE_DIR = Path.home() / ".cache" / "questfoundry" / "corpus-embeddings"


class CorpusNotAvailableError(Exception):
    """Raised when IF Craft Corpus is not installed."""

    def __init__(self) -> None:
        super().__init__("IF Craft Corpus not available. Install with: uv add ifcraftcorpus")


def _corpus_available() -> bool:
    """Check if IF Craft Corpus is installed."""
    try:
        import ifcraftcorpus  # noqa: F401

        return True
    except ImportError:
        return False


# Thread-local storage for corpus instances
# Each thread gets its own Corpus with its own SQLite connection
_thread_local = threading.local()


def _clear_corpus_cache() -> None:
    """Clear thread-local corpus cache. Used for testing."""
    if hasattr(_thread_local, "corpus"):
        delattr(_thread_local, "corpus")


def _get_corpus() -> Corpus:
    """Get thread-local corpus instance with semantic search enabled.

    Uses thread-local storage to ensure each thread has its own Corpus
    instance with its own SQLite connection. This avoids SQLite's
    "objects created in a thread can only be used in that same thread" error.

    Embeddings are cached in ~/.cache/questfoundry/corpus-embeddings/.
    If embeddings don't exist, they are built on first use.

    Returns:
        Corpus instance with semantic search enabled.

    Raises:
        CorpusNotAvailableError: If ifcraftcorpus not installed.
    """
    # Check if this thread already has a corpus instance
    if hasattr(_thread_local, "corpus"):
        return _thread_local.corpus

    if not _corpus_available():
        raise CorpusNotAvailableError()

    from ifcraftcorpus import Corpus
    from ifcraftcorpus.providers import get_embedding_provider

    # Get embedding provider (uses OpenAI or Ollama based on env vars)
    try:
        provider = get_embedding_provider()
    except Exception as e:
        logger.warning("Could not get embedding provider: %s. Using keyword search only.", e)
        corpus = Corpus()
        _thread_local.corpus = corpus
        return corpus

    # Create corpus with embeddings support
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(
        embeddings_path=EMBEDDINGS_CACHE_DIR,
        embedding_provider=provider,
    )

    # Build embeddings if not already built
    if not corpus.has_semantic_search:
        logger.info("Building corpus embeddings (one-time operation)...")
        try:
            count = corpus.build_embeddings()
            logger.info("Built embeddings for %d documents", count)
        except Exception as e:
            logger.warning("Failed to build embeddings: %s. Using keyword search.", e)

    # Store in thread-local storage
    _thread_local.corpus = corpus
    return corpus


class SearchCorpusTool:
    """Search IF craft knowledge base for guidance.

    Returns formatted markdown results with source, score,
    and content excerpts. Results are truncated to avoid
    context overflow.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="search_corpus",
            description="Search IF craft knowledge base. Returns curated writing guidance.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to search (e.g., 'cozy mystery', 'dialogue')",
                    },
                    "cluster": {
                        "type": "string",
                        "description": "Filter by cluster: genre-conventions, narrative-structure, "
                        "prose-and-language, emotional-design, scope-and-planning, craft-foundations",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 5)",
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute corpus search.

        Args:
            arguments: query (required), cluster (optional), limit (optional)

        Returns:
            Structured JSON response following ADR-008.
        """
        query = arguments["query"]
        cluster = arguments.get("cluster")
        limit = arguments.get("limit", 5)

        try:
            corpus = _get_corpus()
        except CorpusNotAvailableError as e:
            return json.dumps(
                {
                    "result": "error",
                    "error": str(e),
                    "action": "Corpus unavailable. Proceed with your own creative knowledge.",
                }
            )

        try:
            results = corpus.search(query, cluster=cluster, limit=limit, mode="hybrid")
        except Exception as e:
            logger.warning("Corpus search failed: %s", e)
            return json.dumps(
                {
                    "result": "error",
                    "error": f"Search failed: {e}",
                    "action": "Search unavailable. Proceed with your own creative knowledge.",
                }
            )

        if not results:
            return json.dumps(
                {
                    "result": "no_results",
                    "query": query,
                    "cluster": cluster,
                    "action": "No matching craft guidance found. Proceed with your creative instincts.",
                }
            )

        # Format results as markdown content
        formatted = []
        total_chars = 0

        for r in results:
            # Build result entry
            source = r.source if hasattr(r, "source") else r.document_name
            topics = ", ".join(r.topics) if hasattr(r, "topics") and r.topics else ""
            score = f"{r.score:.2f}" if hasattr(r, "score") else ""

            # Truncate content to fit within budget
            content = r.content[:1500] if len(r.content) > 1500 else r.content

            entry = f"### {source}"
            if score:
                entry += f" (score: {score})"
            entry += "\n"
            if topics:
                entry += f"*Topics: {topics}*\n\n"
            else:
                entry += "\n"
            entry += f"{content}\n"

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
                "cluster": cluster,
                "count": len(results),
                "content": "\n---\n".join(formatted),
                "action": "Use this craft guidance to inform your creative decisions.",
            }
        )


class GetDocumentTool:
    """Get full document from IF Craft Corpus.

    Retrieves complete document content by name, useful when
    search results indicate a document needs deeper reading.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="get_document",
            description="Get full craft document by name. Use after search identifies relevant doc.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Document name (e.g., 'dialogue_craft', 'cozy_mystery')",
                    },
                },
                "required": ["name"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute document retrieval.

        Args:
            arguments: name (required)

        Returns:
            Structured JSON response following ADR-008.
        """
        name = arguments["name"]

        try:
            corpus = _get_corpus()
        except CorpusNotAvailableError as e:
            return json.dumps(
                {
                    "result": "error",
                    "error": str(e),
                    "action": "Corpus unavailable. Proceed with your own creative knowledge.",
                }
            )

        try:
            doc = corpus.get_document(name)
        except Exception as e:
            logger.warning("Document retrieval failed: %s", e)
            return json.dumps(
                {
                    "result": "error",
                    "error": f"Failed to get document: {e}",
                    "action": "Document retrieval failed. Proceed with available information.",
                }
            )

        if not doc:
            return json.dumps(
                {
                    "result": "no_results",
                    "document": name,
                    "action": "Document not found. Use list_clusters to discover available topics, or proceed with your creative instincts.",
                }
            )

        # Format document
        title = doc.get("title", name)
        cluster = doc.get("cluster", "unknown")
        content = doc.get("content", "")

        # Truncate if too long
        if len(content) > MAX_OUTPUT_CHARS:
            content = content[:MAX_OUTPUT_CHARS] + "\n\n*...content truncated*"

        return json.dumps(
            {
                "result": "success",
                "document": name,
                "title": title,
                "cluster": cluster,
                "content": f"# {title}\n*Cluster: {cluster}*\n\n{content}",
                "action": "Use this craft guidance to inform your creative decisions.",
            }
        )


class ListClustersTool:
    """List available topic clusters in IF Craft Corpus.

    Helps discover what craft knowledge is available for
    targeted searches.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="list_clusters",
            description="List available craft topic clusters for targeted searches.",
            parameters={
                "type": "object",
                "properties": {},
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        """Execute cluster listing.

        Args:
            arguments: Not used.

        Returns:
            Structured JSON response following ADR-008.
        """
        try:
            corpus = _get_corpus()
        except CorpusNotAvailableError as e:
            return json.dumps(
                {
                    "result": "error",
                    "error": str(e),
                    "action": "Corpus unavailable. Proceed with your own creative knowledge.",
                }
            )

        try:
            clusters = corpus.list_clusters()
        except Exception as e:
            logger.warning("Cluster listing failed: %s", e)
            return json.dumps(
                {
                    "result": "error",
                    "error": f"Failed to list clusters: {e}",
                    "action": "Cluster listing failed. Proceed with available information.",
                }
            )

        if not clusters:
            return json.dumps(
                {
                    "result": "no_results",
                    "action": "No clusters available in corpus. Proceed with your creative instincts.",
                }
            )

        # Format as bullet list with descriptions
        cluster_descriptions = {
            "craft-foundations": "Core craft concepts, workflow, tools",
            "narrative-structure": "Story arcs, pacing, branching, endings",
            "prose-and-language": "Dialogue, voice, POV, exposition",
            "genre-conventions": "Genre-specific patterns and expectations",
            "world-and-setting": "Worldbuilding, immersion, locations",
            "emotional-design": "Tone, atmosphere, emotional resonance",
            "game-design": "Mechanics, interaction patterns, player agency",
            "agent-design": "NPC behavior, dialogue systems",
            "scope-and-planning": "Project scope, planning, estimation",
            "audience-and-access": "Target audience, accessibility",
        }

        formatted = ["**Available Clusters:**\n"]
        for cluster in clusters:
            desc = cluster_descriptions.get(cluster, "")
            if desc:
                formatted.append(f"- **{cluster}**: {desc}")
            else:
                formatted.append(f"- **{cluster}**")

        return json.dumps(
            {
                "result": "success",
                "clusters": clusters,
                "content": "\n".join(formatted),
                "action": "Use search_corpus with a cluster filter for targeted results.",
            }
        )


def get_corpus_tools() -> list[Tool]:
    """Get all corpus tools if library is available.

    Returns:
        List of corpus tools, or empty list if not available.
    """
    if not _corpus_available():
        logger.info("IF Craft Corpus not installed, corpus tools disabled")
        return []

    return [
        SearchCorpusTool(),
        GetDocumentTool(),
        ListClustersTool(),
    ]
