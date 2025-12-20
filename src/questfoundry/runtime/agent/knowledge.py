"""
Knowledge context builder for agent prompts.

Implements budget-aware knowledge injection following the menu+consult pattern
from meta/docs/knowledge-patterns.md.

Knowledge layers:
- constitution: Always inline (inviolable principles)
- must_know: Inline up to budget, overflow to menu
- should_know: Menu only (summary with consult hint)
- role_specific: Menu only (specialist reference)
- lookup: Never shown (query via tool only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent, KnowledgeEntry, Studio

import logging

from questfoundry.runtime.models.base import KnowledgeContent

logger = logging.getLogger(__name__)

# Valid knowledge layers for archetype-based inclusion
VALID_LAYERS = frozenset({"must_know", "should_know", "role_specific"})


def _combine_entry_lists(explicit: list[str], archetype_matched: list[str]) -> list[str]:
    """Combine explicit and archetype-matched entries, preserving order.

    Explicit entries come first, then archetype-matched entries that aren't
    already in the explicit list (to avoid duplicates).
    """
    return list(explicit) + [eid for eid in archetype_matched if eid not in set(explicit)]


@dataclass
class KnowledgeBudgetConfig:
    """Budget configuration for knowledge injection."""

    # Total token budget for all knowledge in system prompt
    total_prompt_budget_tokens: int = 4000

    # Budget for constitution (always inline)
    constitution_tokens: int = 500

    # Budget for must_know entries
    must_know_tokens: int = 1500

    # Budget for menu section
    menu_tokens: int = 500

    # Chars per token estimate (rough approximation)
    chars_per_token: int = 4


@dataclass
class KnowledgeContext:
    """Result of knowledge context building."""

    # Full context text for injection into prompt
    context_text: str

    # Separate sections for flexible injection
    constitution_section: str | None = None
    must_know_section: str | None = None
    menu_section: str | None = None

    # Tracking
    entries_inlined: list[str] = field(default_factory=list)
    entries_in_menu: list[str] = field(default_factory=list)
    tokens_used: int = 0


class KnowledgeContextBuilder:
    """
    Builds budget-aware knowledge context for agent prompts.

    Implements the layered knowledge injection strategy:
    1. Constitution always inlined
    2. Must-know inlined up to budget, overflow to menu
    3. Should-know and role-specific shown in menu only
    4. Lookup entries not shown (available via consult_knowledge tool)

    Usage:
        builder = KnowledgeContextBuilder()
        context = builder.build_context(agent, studio)
        # Use context.context_text in system prompt
    """

    def __init__(self, config: KnowledgeBudgetConfig | None = None) -> None:
        """Initialize with optional budget configuration."""
        self.config = config or KnowledgeBudgetConfig()

    def build_context(self, agent: Agent, studio: Studio) -> KnowledgeContext:
        """
        Build knowledge context for an agent.

        Includes both explicitly listed entries from agent.knowledge_requirements
        and entries that match the agent's archetypes via applicable_to.archetypes.

        Args:
            agent: Agent definition with knowledge_requirements
            studio: Studio with knowledge entries

        Returns:
            KnowledgeContext with formatted text and tracking info
        """
        sections: list[str] = []
        menu_items: list[dict[str, str]] = []
        entries_inlined: list[str] = []
        entries_in_menu: list[str] = []
        used_tokens = 0

        knowledge_req = agent.knowledge_requirements

        # Find entries that apply to this agent's archetypes
        archetype_entries = self._find_entries_by_archetype(agent, studio)

        # 1. Constitution - always inline
        constitution_section = None
        if knowledge_req and knowledge_req.constitution:
            constitution_text = self._get_constitution_text(studio)
            if constitution_text:
                constitution_section = self._format_constitution(constitution_text)
                sections.append(constitution_section)
                used_tokens += self._count_tokens(constitution_section)

        # 2. Must-know - inline up to budget, overflow to menu
        # Combine explicit must_know with archetype-matched must_know entries
        must_know_section = None
        explicit_must_know = (knowledge_req.must_know or []) if knowledge_req else []
        archetype_must_know = archetype_entries.get("must_know", [])
        all_must_know = _combine_entry_lists(explicit_must_know, archetype_must_know)

        if all_must_know:
            must_know_lines: list[str] = []
            must_know_budget = self.config.must_know_tokens
            must_know_tokens_used = 0

            for entry_id in all_must_know:
                entry = studio.knowledge.get(entry_id)
                if not entry:
                    continue

                content = self._get_entry_content(entry)
                if not content:
                    # No content - add to menu instead (if not already there)
                    if entry_id not in entries_in_menu:
                        menu_items.append(self._format_menu_item(entry))
                        entries_in_menu.append(entry_id)
                    continue

                entry_tokens = self._count_tokens(content)

                # Check if fits in must_know budget
                if must_know_tokens_used + entry_tokens <= must_know_budget:
                    must_know_lines.append(self._format_inline_entry(entry, content))
                    entries_inlined.append(entry_id)
                    used_tokens += entry_tokens
                    must_know_tokens_used += entry_tokens
                else:
                    # Over budget - add to menu instead (if not already there)
                    if entry_id not in entries_in_menu:
                        menu_items.append(self._format_menu_item(entry))
                        entries_in_menu.append(entry_id)

            if must_know_lines:
                must_know_section = "## Critical Knowledge\n\n" + "\n\n".join(must_know_lines)
                sections.append(must_know_section)

        # 3. Should-know - menu only (skip if already in menu or inlined)
        explicit_should_know = (knowledge_req.should_know or []) if knowledge_req else []
        archetype_should_know = archetype_entries.get("should_know", [])
        all_should_know = _combine_entry_lists(explicit_should_know, archetype_should_know)
        for entry_id in all_should_know:
            if entry_id in entries_in_menu or entry_id in entries_inlined:
                continue
            entry = studio.knowledge.get(entry_id)
            if entry:
                menu_items.append(self._format_menu_item(entry))
                entries_in_menu.append(entry_id)

        # 4. Role-specific - menu only (skip if already in menu or inlined)
        explicit_role_specific = (knowledge_req.role_specific or []) if knowledge_req else []
        archetype_role_specific = archetype_entries.get("role_specific", [])
        all_role_specific = _combine_entry_lists(explicit_role_specific, archetype_role_specific)
        for entry_id in all_role_specific:
            if entry_id in entries_in_menu or entry_id in entries_inlined:
                continue
            entry = studio.knowledge.get(entry_id)
            if entry:
                menu_items.append(self._format_menu_item(entry))
                entries_in_menu.append(entry_id)

        # 5. Build menu section
        menu_section = None
        if menu_items:
            menu_section = self._format_menu(menu_items)
            sections.append(menu_section)
            used_tokens += self._count_tokens(menu_section)

        # Assemble final context
        context_text = "\n\n".join(sections)

        return KnowledgeContext(
            context_text=context_text,
            constitution_section=constitution_section,
            must_know_section=must_know_section,
            menu_section=menu_section,
            entries_inlined=entries_inlined,
            entries_in_menu=entries_in_menu,
            tokens_used=used_tokens,
        )

    def _get_constitution_text(self, studio: Studio) -> str | None:
        """Get constitution text from studio.

        Constitution is typically stored in a separate file referenced
        by constitution_ref, but may also be in knowledge entries.
        """
        # TODO: Load from constitution_ref if available
        # For now, look for a 'constitution' knowledge entry
        entry = studio.knowledge.get("constitution")
        if entry:
            return self._get_entry_content(entry)
        return None

    def _get_entry_content(self, entry: KnowledgeEntry) -> str | None:
        """Extract text content from a knowledge entry."""
        content = entry.content
        if content is None:
            return None

        # Handle dict content (raw from JSON)
        if isinstance(content, dict):
            content_type = content.get("type", "inline")
            if content_type == "inline":
                return content.get("text")
            elif content_type == "structured":
                return json.dumps(content.get("data", {}), indent=2)
            return content.get("text")

        # Handle KnowledgeContent model
        if isinstance(content, KnowledgeContent):
            if content.type == "inline":
                return content.text
            elif content.type == "structured":
                return json.dumps(content.data or {}, indent=2)

        return str(content) if content else None

    def _format_constitution(self, text: str) -> str:
        """Format constitution for prompt injection."""
        return f"## Constitution (Inviolable Principles)\n\n{text}"

    def _format_inline_entry(self, entry: KnowledgeEntry, content: str) -> str:
        """Format a knowledge entry for inline injection."""
        name = entry.name or entry.id
        return f"### {name}\n\n{content}"

    def _format_menu_item(self, entry: KnowledgeEntry) -> dict[str, str]:
        """Format a knowledge entry for the menu."""
        return {
            "id": entry.id,
            "name": entry.name or entry.id,
            "summary": entry.summary or "(No summary)",
        }

    def _format_menu(self, items: list[dict[str, str]]) -> str:
        """Format the knowledge menu section."""
        lines = [
            "## Knowledge Menu\n",
            "Use `consult_knowledge(entry_id)` for full details.\n",
        ]

        for item in items:
            lines.append(f"- **{item['name']}** (`{item['id']}`): {item['summary']}")

        return "\n".join(lines)

    def _count_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Uses a simple chars/token ratio. For more accuracy,
        use a proper tokenizer like tiktoken.
        """
        return len(text) // self.config.chars_per_token

    def _find_entries_by_archetype(self, agent: Agent, studio: Studio) -> dict[str, list[str]]:
        """
        Find knowledge entries that apply to this agent's archetypes.

        Checks each knowledge entry's applicable_to.archetypes field against
        the agent's archetypes. Returns entries grouped by their layer.

        Args:
            agent: Agent definition with archetypes
            studio: Studio with knowledge entries

        Returns:
            Dict mapping layer names to lists of matching entry IDs
        """
        result: dict[str, list[str]] = {
            "must_know": [],
            "should_know": [],
            "role_specific": [],
        }

        agent_archetypes = set(getattr(agent, "archetypes", None) or [])
        if not agent_archetypes:
            return result

        for entry_id, entry in studio.knowledge.items():
            # Check if entry has applicable_to with archetypes
            applicable_to = entry.applicable_to
            if not applicable_to:
                continue

            # Handle both dict and model forms
            if isinstance(applicable_to, dict):
                entry_archetypes = set(applicable_to.get("archetypes", []))
            else:
                entry_archetypes = set(applicable_to.archetypes or [])

            # Check for archetype match
            if not entry_archetypes.intersection(agent_archetypes):
                continue

            # Add to appropriate layer bucket
            layer = entry.layer
            if layer in VALID_LAYERS:
                result[layer].append(entry_id)
            elif layer not in {"constitution", "lookup"}:
                # Warn about unknown layers (constitution and lookup are handled separately)
                logger.warning(
                    "Knowledge entry '%s' has unknown layer '%s', skipping archetype inclusion",
                    entry_id,
                    layer,
                )

        return result


def build_knowledge_context(
    agent: Agent,
    studio: Studio,
    config: KnowledgeBudgetConfig | None = None,
) -> KnowledgeContext:
    """
    Convenience function to build knowledge context for an agent.

    Args:
        agent: Agent definition
        studio: Studio with knowledge entries
        config: Optional budget configuration

    Returns:
        KnowledgeContext with formatted text
    """
    builder = KnowledgeContextBuilder(config)
    return builder.build_context(agent, studio)
