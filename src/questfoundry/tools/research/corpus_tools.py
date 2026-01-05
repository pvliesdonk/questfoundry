"""IF Craft Corpus tools for writing guidance.

This module provides LangChain-compatible tools for searching the
IF Craft Corpus, a curated knowledge base for interactive fiction writing.

Tools:
    SearchCorpusTool: Search for craft guidance by topic/query
    GetDocumentTool: Get full document by name
    ListClustersTool: Discover available topic clusters

The corpus is a singleton to avoid repeated index building.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from questfoundry.tools.base import Tool, ToolDefinition

if TYPE_CHECKING:
    from ifcraftcorpus import Corpus

logger = logging.getLogger(__name__)

# Maximum tokens (approx chars / 4) for tool output
MAX_OUTPUT_CHARS = 4000  # ~1000 tokens


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


@lru_cache(maxsize=1)
def _get_corpus() -> Corpus:
    """Get singleton corpus instance.

    Returns:
        Corpus instance (lazily initialized).

    Raises:
        CorpusNotAvailableError: If ifcraftcorpus not installed.
    """
    if not _corpus_available():
        raise CorpusNotAvailableError()

    from ifcraftcorpus import Corpus

    return Corpus()


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
            Formatted markdown results or error message.
        """
        query = arguments["query"]
        cluster = arguments.get("cluster")
        limit = arguments.get("limit", 5)

        try:
            corpus = _get_corpus()
        except CorpusNotAvailableError as e:
            return f"Error: {e}"

        try:
            results = corpus.search(query, cluster=cluster, limit=limit)
        except Exception as e:
            logger.warning("Corpus search failed: %s", e)
            return f"Search failed: {e}"

        if not results:
            return f"No craft guidance found for '{query}'. Try broader terms."

        # Format results as markdown
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

        return "\n---\n".join(formatted)


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
            Formatted document or error message.
        """
        name = arguments["name"]

        try:
            corpus = _get_corpus()
        except CorpusNotAvailableError as e:
            return f"Error: {e}"

        try:
            doc = corpus.get_document(name)
        except Exception as e:
            logger.warning("Document retrieval failed: %s", e)
            return f"Failed to get document: {e}"

        if not doc:
            return f"Document '{name}' not found. Use list_clusters to see available topics."

        # Format document
        title = doc.get("title", name)
        cluster = doc.get("cluster", "unknown")
        content = doc.get("content", "")

        # Truncate if too long
        if len(content) > MAX_OUTPUT_CHARS:
            content = content[:MAX_OUTPUT_CHARS] + "\n\n*...content truncated*"

        return f"# {title}\n*Cluster: {cluster}*\n\n{content}"


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
            Formatted list of clusters.
        """
        try:
            corpus = _get_corpus()
        except CorpusNotAvailableError as e:
            return f"Error: {e}"

        try:
            clusters = corpus.list_clusters()
        except Exception as e:
            logger.warning("Cluster listing failed: %s", e)
            return f"Failed to list clusters: {e}"

        if not clusters:
            return "No clusters available."

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

        return "\n".join(formatted)


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
