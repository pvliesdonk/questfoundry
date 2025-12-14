"""
Context builder for agent activation.

Gathers and prepares all context needed for an agent, including
knowledge injection and conversation history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent, KnowledgeEntry, Studio


@dataclass
class AgentContext:
    """
    Context prepared for agent activation.

    Contains all the knowledge and information needed to build
    the system prompt and run the agent.
    """

    # Constitution text (if agent requires it)
    constitution_text: str | None = None

    # Must-know knowledge entries (always injected)
    must_know_entries: list[dict[str, str]] = field(default_factory=list)

    # Role-specific knowledge menu (available for lookup)
    role_specific_menu: list[dict[str, str]] = field(default_factory=list)

    # Token estimates for context budget
    constitution_tokens: int = 0
    must_know_tokens: int = 0
    menu_tokens: int = 0

    # Rough estimate: 4 chars per token
    CHARS_PER_TOKEN: int = 4

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens for all context."""
        return self.constitution_tokens + self.must_know_tokens + self.menu_tokens


class ContextBuilder:
    """
    Builds context for agent activation.

    Gathers knowledge based on agent's knowledge_requirements and
    prepares it for injection into the system prompt.
    """

    # Rough estimate: 4 chars per token
    CHARS_PER_TOKEN = 4

    def __init__(self, domain_path: Path | None = None):
        """
        Initialize context builder.

        Args:
            domain_path: Path to domain directory (for loading knowledge files)
        """
        self._domain_path = domain_path
        self._knowledge_cache: dict[str, dict[str, Any]] = {}

    def build(self, agent: Agent, studio: Studio) -> AgentContext:
        """
        Build context for agent activation.

        Args:
            agent: The agent to build context for
            studio: The studio containing knowledge entries

        Returns:
            AgentContext with all prepared knowledge
        """
        context = AgentContext()

        if not agent.knowledge_requirements:
            return context

        reqs = agent.knowledge_requirements

        # 1. Constitution (if required)
        if reqs.constitution:
            constitution_text = self._get_constitution_text(studio)
            if constitution_text:
                context.constitution_text = constitution_text
                context.constitution_tokens = len(constitution_text) // self.CHARS_PER_TOKEN

        # 2. Must-know entries
        for entry_id in reqs.must_know:
            entry = self._find_knowledge_entry(entry_id, studio)
            if entry:
                entry_dict = self._format_entry_for_injection(entry)
                context.must_know_entries.append(entry_dict)
                context.must_know_tokens += (
                    len(entry_dict.get("content", "")) // self.CHARS_PER_TOKEN
                )

        # 3. Role-specific entries (as menu items)
        for entry_id in reqs.role_specific:
            entry = self._find_knowledge_entry(entry_id, studio)
            if entry:
                menu_item = self._format_entry_for_menu(entry)
                context.role_specific_menu.append(menu_item)
                context.menu_tokens += len(menu_item.get("summary", "")) // self.CHARS_PER_TOKEN

        return context

    def _get_constitution_text(self, studio: Studio) -> str | None:
        """Get constitution text from studio."""
        # Try to load constitution from knowledge_config or file
        if studio.knowledge_config and "constitution" in studio.knowledge_config:
            const_data = studio.knowledge_config["constitution"]
            if isinstance(const_data, dict):
                # Check for principles
                principles = const_data.get("principles", [])
                if principles:
                    return self._format_constitution(const_data)

        # Try loading from file if domain_path is set
        if self._domain_path:
            return self._load_constitution_from_file()

        return None

    def _load_constitution_from_file(self) -> str | None:
        """Load constitution from governance/constitution.json."""
        if not self._domain_path:
            return None

        const_path = self._domain_path / "governance" / "constitution.json"
        if const_path.exists():
            import json

            data = json.loads(const_path.read_text())
            return self._format_constitution(data)

        return None

    def _format_constitution(self, data: dict[str, Any]) -> str:
        """Format constitution data into text."""
        lines = []

        # Preamble
        preamble = data.get("preamble")
        if preamble:
            lines.append(preamble)
            lines.append("")

        # Principles
        principles = data.get("principles", [])
        for p in principles:
            if isinstance(p, dict):
                statement = p.get("statement", "")
                if statement:
                    lines.append(f"- {statement}")

        return "\n".join(lines)

    def _find_knowledge_entry(
        self,
        entry_id: str,
        _studio: Studio,
    ) -> KnowledgeEntry | None:
        """Find a knowledge entry by ID."""
        # Check if studio has knowledge entries loaded
        # For now, we'll need to load from files if domain_path is set
        # _studio reserved for future use when knowledge is embedded in Studio
        if self._domain_path:
            return self._load_knowledge_entry(entry_id)

        return None

    def _load_knowledge_entry(self, entry_id: str) -> KnowledgeEntry | None:
        """Load a knowledge entry from the domain."""
        if not self._domain_path:
            return None

        # Check cache first
        if entry_id in self._knowledge_cache:
            from questfoundry.runtime.models.base import KnowledgeEntry

            return KnowledgeEntry(**self._knowledge_cache[entry_id])

        # Search in knowledge directories
        import json

        knowledge_dirs = [
            self._domain_path / "knowledge" / "must_know",
            self._domain_path / "knowledge" / "should_know",
            self._domain_path / "knowledge" / "role_specific",
        ]

        for dir_path in knowledge_dirs:
            if dir_path.exists():
                for file_path in dir_path.glob("*.json"):
                    try:
                        data = json.loads(file_path.read_text())
                        if data.get("id") == entry_id:
                            self._knowledge_cache[entry_id] = data
                            from questfoundry.runtime.models.base import KnowledgeEntry

                            return KnowledgeEntry(**data)
                    except (json.JSONDecodeError, KeyError):
                        continue

        return None

    def _format_entry_for_injection(self, entry: KnowledgeEntry) -> dict[str, str]:
        """Format a knowledge entry for prompt injection."""
        content = ""

        if entry.content:
            if isinstance(entry.content, dict):
                # Inline content
                if entry.content.get("type") == "inline":
                    content = entry.content.get("text", "")
            elif isinstance(entry.content, str):
                content = entry.content

        return {
            "id": entry.id,
            "name": entry.name or entry.id,
            "content": content,
        }

    def _format_entry_for_menu(self, entry: KnowledgeEntry) -> dict[str, str]:
        """Format a knowledge entry for the knowledge menu."""
        return {
            "id": entry.id,
            "name": entry.name or entry.id,
            "summary": entry.summary or "",
        }


# Convenience function
def build_context(
    agent: Agent,
    studio: Studio,
    domain_path: Path | None = None,
) -> AgentContext:
    """Build context for an agent."""
    builder = ContextBuilder(domain_path=domain_path)
    return builder.build(agent, studio)
