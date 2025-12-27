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

Model class support:
- Large models: Use normal summary/content, all must_know entries
- Small models: Use concise_summary/concise_description, small_model_must_know list
- Empty string in concise_* means exclude for small models
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent, KnowledgeEntry, Studio

import logging

from questfoundry.runtime.agent.content_utils import extract_knowledge_content
from questfoundry.runtime.models.enums import ModelClass

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

    def __init__(
        self,
        config: KnowledgeBudgetConfig | None = None,
        domain_path: Path | str | None = None,
    ) -> None:
        """Initialize with optional budget configuration.

        Args:
            config: Budget configuration for knowledge injection
            domain_path: Path to domain directory for loading constitution file
        """
        self.config = config or KnowledgeBudgetConfig()
        self._domain_path = Path(domain_path) if domain_path else None

    def build_context(
        self,
        agent: Agent,
        studio: Studio,
        model_class: ModelClass = ModelClass.LARGE,
    ) -> KnowledgeContext:
        """
        Build knowledge context for an agent.

        Includes both explicitly listed entries from agent.knowledge_requirements
        and entries that match the agent's archetypes via applicable_to.archetypes.

        For small models, uses concise_summary/concise_description when available,
        and respects small_model_must_know override.

        Args:
            agent: Agent definition with knowledge_requirements
            studio: Studio with knowledge entries
            model_class: Model size class ("small", "medium", "large")

        Returns:
            KnowledgeContext with formatted text and tracking info
        """
        sections: list[str] = []
        menu_items: list[dict[str, str]] = []
        entries_inlined: list[str] = []
        entries_in_menu: list[str] = []
        used_tokens = 0
        use_concise = model_class == ModelClass.SMALL

        knowledge_req = agent.knowledge_requirements

        # Find entries that apply to this agent (via archetypes or agent ID)
        matched_entries = self._find_entries_by_applicability(agent, studio)

        # 1. Constitution - always inline
        constitution_section = None
        if knowledge_req and knowledge_req.constitution:
            constitution_text = self._get_constitution_text(studio)
            if constitution_text:
                constitution_section = self._format_constitution(constitution_text)
                sections.append(constitution_section)
                used_tokens += self._count_tokens(constitution_section)

        # 2. Must-know - inline up to budget, overflow to menu
        # For small models, use small_model_must_know if available
        # When using small_model_must_know, skip archetype matching to keep prompt minimal
        must_know_section = None
        using_small_model_override = False
        if knowledge_req:
            if use_concise and knowledge_req.small_model_must_know:
                explicit_must_know = knowledge_req.small_model_must_know
                using_small_model_override = True
            else:
                explicit_must_know = knowledge_req.must_know or []
        else:
            explicit_must_know = []

        # Skip archetype matching for small models with explicit override
        if using_small_model_override:
            all_must_know = list(explicit_must_know)
        else:
            archetype_must_know = matched_entries.get("must_know", [])
            all_must_know = _combine_entry_lists(explicit_must_know, archetype_must_know)

        if all_must_know:
            must_know_lines: list[str] = []
            must_know_budget = self.config.must_know_tokens
            must_know_tokens_used = 0

            for entry_id in all_must_know:
                entry = studio.knowledge.get(entry_id)
                if not entry:
                    continue

                content = self._get_entry_content_for_model(entry, use_concise)
                if content is None:
                    # Excluded for small models or no content - add to menu
                    summary = self._get_entry_summary_for_model(entry, use_concise)
                    if summary is not None and entry_id not in entries_in_menu:
                        menu_items.append(self._format_menu_item_with_summary(entry, summary))
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
                        summary = self._get_entry_summary_for_model(entry, use_concise)
                        if summary is not None:
                            menu_items.append(self._format_menu_item_with_summary(entry, summary))
                            entries_in_menu.append(entry_id)

            if must_know_lines:
                must_know_section = "## Critical Knowledge\n\n" + "\n\n".join(must_know_lines)
                sections.append(must_know_section)

        # 3. Should-know - menu only (skip if already in menu or inlined)
        explicit_should_know = (knowledge_req.should_know or []) if knowledge_req else []
        archetype_should_know = matched_entries.get("should_know", [])
        all_should_know = _combine_entry_lists(explicit_should_know, archetype_should_know)
        for entry_id in all_should_know:
            if entry_id in entries_in_menu or entry_id in entries_inlined:
                continue
            entry = studio.knowledge.get(entry_id)
            if entry:
                summary = self._get_entry_summary_for_model(entry, use_concise)
                if summary is not None:  # None means excluded for small models
                    menu_items.append(self._format_menu_item_with_summary(entry, summary))
                    entries_in_menu.append(entry_id)

        # 4. Role-specific - menu only (skip if already in menu or inlined)
        explicit_role_specific = (knowledge_req.role_specific or []) if knowledge_req else []
        archetype_role_specific = matched_entries.get("role_specific", [])
        all_role_specific = _combine_entry_lists(explicit_role_specific, archetype_role_specific)
        for entry_id in all_role_specific:
            if entry_id in entries_in_menu or entry_id in entries_inlined:
                continue
            entry = studio.knowledge.get(entry_id)
            if entry:
                summary = self._get_entry_summary_for_model(entry, use_concise)
                if summary is not None:
                    menu_items.append(self._format_menu_item_with_summary(entry, summary))
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

    def _get_entry_summary_for_model(self, entry: KnowledgeEntry, use_concise: bool) -> str | None:
        """Get appropriate summary based on model class.

        For small models, uses concise_summary if available.
        Empty string means exclude entirely for small models.

        Returns None if entry should be excluded.
        """
        if use_concise:
            concise = entry.concise_summary
            if concise is not None:
                if concise == "":
                    return None  # Exclude from menu
                return concise
        return entry.summary or "(No summary)"

    def _get_entry_content_for_model(self, entry: KnowledgeEntry, use_concise: bool) -> str | None:
        """Get appropriate content based on model class.

        For small models, uses concise_description if available.
        Empty string means exclude entirely for small models.

        Returns None if entry should be excluded or has no content.
        """
        if use_concise:
            concise = entry.concise_description
            if concise is not None:
                if concise == "":
                    return None  # Exclude from consult
                return concise
        # Fall back to normal content extraction
        return self._get_entry_content(entry)

    def _format_menu_item_with_summary(self, entry: KnowledgeEntry, summary: str) -> dict[str, str]:
        """Format a knowledge entry for the menu with explicit summary."""
        return {
            "id": entry.id,
            "name": entry.name or entry.id,
            "summary": summary,
        }

    def _get_constitution_text(self, studio: Studio) -> str | None:
        """Get constitution text from studio.

        Constitution is loaded from the governance/constitution.json file
        referenced by studio.constitution_ref.
        """
        # Try loading from file if domain_path is available
        if self._domain_path and studio.constitution_ref:
            const_path = self._domain_path / studio.constitution_ref
            if const_path.exists():
                try:
                    data = json.loads(const_path.read_text())
                    return self._format_constitution_data(data)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load constitution: {e}")

        # Fall back to knowledge entry (legacy support)
        entry = studio.knowledge.get("constitution")
        if entry:
            return self._get_entry_content(entry)

        return None

    def _format_constitution_data(self, data: dict[str, Any]) -> str:
        """Format constitution JSON data into readable text.

        Extracts preamble and principle statements for agent consumption.
        """
        lines = []

        # Preamble
        preamble = data.get("preamble")
        if preamble:
            lines.append(preamble)
            lines.append("")

        # Principles as bullet list
        principles = data.get("principles", [])
        for p in principles:
            if isinstance(p, dict):
                statement = p.get("statement", "")
                if statement:
                    lines.append(f"- {statement}")

        return "\n".join(lines) if lines else ""

    def _get_entry_content(self, entry: KnowledgeEntry) -> str | None:
        """Extract text content from a knowledge entry.

        Delegates to shared content_utils for consistent handling across
        the codebase. Supports structured, file_ref, and corpus content types.
        """
        return extract_knowledge_content(entry, self._domain_path)

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

    def _find_entries_by_applicability(self, agent: Agent, studio: Studio) -> dict[str, list[str]]:
        """
        Find knowledge entries that apply to this agent.

        Checks each knowledge entry's applicable_to field for matches against:
        1. Agent's archetypes (via applicable_to.archetypes)
        2. Agent's ID (via applicable_to.agents)

        Returns entries grouped by their layer.

        Args:
            agent: Agent definition with archetypes and id
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
        agent_id = agent.id

        for entry_id, entry in studio.knowledge.items():
            applicable_to = entry.applicable_to
            if not applicable_to:
                continue

            # Handle both dict and model forms
            if isinstance(applicable_to, dict):
                entry_archetypes = set(applicable_to.get("archetypes", []))
                entry_agents = set(applicable_to.get("agents", []))
            else:
                entry_archetypes = set(applicable_to.archetypes or [])
                entry_agents = set(applicable_to.agents or [])

            # Check for archetype match OR agent ID match
            archetype_match = bool(entry_archetypes.intersection(agent_archetypes))
            agent_match = agent_id in entry_agents

            if not (archetype_match or agent_match):
                continue

            # Add to appropriate layer bucket
            layer = entry.layer
            if layer in VALID_LAYERS:
                result[layer].append(entry_id)
            elif layer not in {"constitution", "lookup"}:
                # Warn about unknown layers (constitution and lookup are handled separately)
                logger.warning(
                    "Knowledge entry '%s' has unknown layer '%s', skipping applicability inclusion",
                    entry_id,
                    layer,
                )

        return result


def build_knowledge_context(
    agent: Agent,
    studio: Studio,
    config: KnowledgeBudgetConfig | None = None,
    domain_path: Path | str | None = None,
    model_class: ModelClass = ModelClass.LARGE,
) -> KnowledgeContext:
    """
    Convenience function to build knowledge context for an agent.

    Args:
        agent: Agent definition
        studio: Studio with knowledge entries
        config: Optional budget configuration
        domain_path: Path to domain directory for loading constitution
        model_class: Model size class (ModelClass.SMALL, MEDIUM, LARGE)

    Returns:
        KnowledgeContext with formatted text
    """
    builder = KnowledgeContextBuilder(config, domain_path=domain_path)
    return builder.build_context(agent, studio, model_class=model_class)
