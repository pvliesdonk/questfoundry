"""Knowledge retrieval tools for agents.

Provides:
- ConsultKnowledgeTool: Retrieve full content of a knowledge entry by ID
- QueryKnowledgeTool: Search knowledge base (future: RAG integration)
"""

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from questfoundry.runtime.domain.models import (
    Agent,
    KnowledgeEntry,
    Studio,
)

logger = logging.getLogger(__name__)


class ConsultKnowledgeTool(BaseTool):
    """Tool for retrieving full content of a knowledge entry.

    Agents use this to get detailed information from entries
    listed in their role_specific knowledge requirements.
    """

    name: str = "consult_knowledge"
    description: str = (
        "Retrieve full content of a knowledge entry by ID. "
        "Use this to get detailed guidance, procedures, or reference material. "
        "The ID should match an entry from the Available Reference Material section."
    )

    studio: Studio = Field(exclude=True)
    agent: Agent = Field(exclude=True)

    def _run(self, entry_id: str) -> str:
        """Retrieve a knowledge entry by ID.

        Args:
            entry_id: The ID of the knowledge entry to retrieve

        Returns:
            Full content of the entry, or an error message
        """
        entry = self.studio.knowledge_entries.get(entry_id)

        if not entry:
            available = self._get_available_entries()
            return (
                f"Knowledge entry '{entry_id}' not found. "
                f"Available entries: {', '.join(available) if available else 'none'}"
            )

        if not self._can_access(entry):
            return f"Knowledge entry '{entry_id}' is not accessible to your role."

        content = self._get_content(entry)
        if not content:
            return f"Knowledge entry '{entry_id}' has no content available."

        # Format response
        response = f"# {entry.name}\n\n"
        if entry.summary:
            response += f"*{entry.summary}*\n\n"
        response += content

        return response

    def _can_access(self, entry: KnowledgeEntry) -> bool:
        """Check if the agent can access this entry."""
        if not entry.applicable_to:
            return True

        if entry.applicable_to.agents and self.agent.id in entry.applicable_to.agents:
            return True

        if entry.applicable_to.archetypes:
            for archetype in self.agent.archetypes:
                if archetype in entry.applicable_to.archetypes:
                    return True

        if entry.applicable_to.agents or entry.applicable_to.archetypes:
            return False

        return True

    def _get_available_entries(self) -> list[str]:
        """Get list of entry IDs accessible to this agent."""
        kr = self.agent.knowledge_requirements
        available = set()

        # Include role_specific entries
        for entry_id in kr.role_specific:
            entry = self.studio.knowledge_entries.get(entry_id)
            if entry and self._can_access(entry):
                available.add(entry_id)

        # Include can_lookup entries
        for entry_id in kr.can_lookup:
            entry = self.studio.knowledge_entries.get(entry_id)
            if entry and self._can_access(entry):
                available.add(entry_id)

        return sorted(available)

    def _get_content(self, entry: KnowledgeEntry) -> str | None:
        """Extract content from entry."""
        content = entry.content

        if content.type == "inline" and content.text:
            return content.text

        if content.type == "file_ref" and content.path:
            logger.warning(f"File ref not supported: {content.path}")
            return None

        return None


class QueryKnowledgeTool(BaseTool):
    """Tool for searching the knowledge base.

    Provides semantic search over can_lookup entries.
    (Future: integrate with RAG/vector store)
    """

    name: str = "query_knowledge"
    description: str = (
        "Search the knowledge base for information on a topic. "
        "Use this when you need to find relevant reference material "
        "but don't know the exact entry ID."
    )

    studio: Studio = Field(exclude=True)
    agent: Agent = Field(exclude=True)

    def _run(self, query: str) -> str:
        """Search knowledge base for relevant entries.

        Args:
            query: Search query

        Returns:
            Matching entries with summaries
        """
        # Simple keyword search for now
        # Future: integrate with vector store for semantic search
        query_lower = query.lower()
        matches: list[dict[str, Any]] = []

        kr = self.agent.knowledge_requirements

        # Search in can_lookup entries
        searchable_ids = set(kr.can_lookup) | set(kr.role_specific)

        for entry_id in searchable_ids:
            entry = self.studio.knowledge_entries.get(entry_id)
            if not entry:
                continue

            if not self._can_access(entry):
                continue

            # Check if query matches keywords, name, or summary
            score = 0
            if query_lower in entry.name.lower():
                score += 10
            if entry.summary and query_lower in entry.summary.lower():
                score += 5
            for keyword in entry.keywords:
                if query_lower in keyword.lower():
                    score += 3
            for trigger in entry.triggers:
                if query_lower in trigger.lower():
                    score += 2

            if score > 0:
                matches.append({
                    "id": entry.id,
                    "name": entry.name,
                    "summary": entry.summary,
                    "score": score,
                })

        if not matches:
            return f"No knowledge entries found matching '{query}'."

        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)

        # Format response
        lines = [f"Found {len(matches)} relevant entries:\n"]
        for m in matches[:5]:  # Top 5
            summary = m["summary"] or "No summary available"
            lines.append(f"- **{m['name']}** (`{m['id']}`): {summary}")

        lines.append(
            "\nUse `consult_knowledge(entry_id)` to retrieve full content."
        )

        return "\n".join(lines)

    def _can_access(self, entry: KnowledgeEntry) -> bool:
        """Check if the agent can access this entry."""
        if not entry.applicable_to:
            return True

        if entry.applicable_to.agents and self.agent.id in entry.applicable_to.agents:
            return True

        if entry.applicable_to.archetypes:
            for archetype in self.agent.archetypes:
                if archetype in entry.applicable_to.archetypes:
                    return True

        if entry.applicable_to.agents or entry.applicable_to.archetypes:
            return False

        return True


def create_consult_knowledge_tool(studio: Studio, agent: Agent) -> ConsultKnowledgeTool:
    """Factory function to create a consult knowledge tool.

    Args:
        studio: The loaded studio
        agent: The agent who will use this tool

    Returns:
        Configured ConsultKnowledgeTool
    """
    return ConsultKnowledgeTool(studio=studio, agent=agent)


def create_query_knowledge_tool(studio: Studio, agent: Agent) -> QueryKnowledgeTool:
    """Factory function to create a query knowledge tool.

    Args:
        studio: The loaded studio
        agent: The agent who will use this tool

    Returns:
        Configured QueryKnowledgeTool
    """
    return QueryKnowledgeTool(studio=studio, agent=agent)
