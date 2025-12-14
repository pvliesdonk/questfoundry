"""
Prompt builder for agent system prompts.

Constructs the system prompt from agent definition and injected knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent


@dataclass
class PromptSection:
    """A section of the system prompt."""

    name: str
    content: str
    priority: int = 0  # Higher = more important, included first


@dataclass
class BuiltPrompt:
    """Result of prompt building."""

    text: str
    sections: list[PromptSection]
    token_estimate: int  # Rough estimate based on character count


class PromptBuilder:
    """
    Builds system prompts for agent activation.

    The prompt structure:
    1. Agent Identity
    2. Knowledge - Constitution
    3. Knowledge - Must Know
    4. Constraints
    5. Capabilities
    6. Available Knowledge (menu)
    """

    # Rough estimate: 4 chars per token
    CHARS_PER_TOKEN = 4

    def __init__(self) -> None:
        """Initialize prompt builder."""
        self._sections: list[PromptSection] = []

    def reset(self) -> None:
        """Clear all sections."""
        self._sections = []

    def add_section(
        self,
        name: str,
        content: str,
        priority: int = 0,
    ) -> None:
        """
        Add a section to the prompt.

        Args:
            name: Section name (for debugging)
            content: Section text
            priority: Higher = more important
        """
        self._sections.append(PromptSection(name=name, content=content, priority=priority))

    def build_for_agent(
        self,
        agent: Agent,
        constitution_text: str | None = None,
        must_know_entries: list[dict[str, str]] | None = None,
        role_specific_menu: list[dict[str, str]] | None = None,
    ) -> BuiltPrompt:
        """
        Build the system prompt for an agent.

        Args:
            agent: The agent definition
            constitution_text: Full constitution text to inject
            must_know_entries: List of {id, name, content} for must_know knowledge
            role_specific_menu: List of {id, name, summary} for available knowledge

        Returns:
            Built prompt with text and metadata
        """
        self.reset()

        # 1. Agent Identity (highest priority)
        identity = self._build_identity_section(agent)
        self.add_section("identity", identity, priority=100)

        # 2. Constitution
        if constitution_text:
            self.add_section(
                "constitution", self._format_constitution(constitution_text), priority=90
            )

        # 3. Must Know
        if must_know_entries:
            must_know = self._build_must_know_section(must_know_entries)
            self.add_section("must_know", must_know, priority=80)

        # 4. Constraints
        if agent.constraints:
            constraints = self._build_constraints_section(agent)
            self.add_section("constraints", constraints, priority=70)

        # 5. Capabilities
        if agent.capabilities:
            capabilities = self._build_capabilities_section(agent)
            self.add_section("capabilities", capabilities, priority=60)

        # 6. Available Knowledge Menu
        if role_specific_menu:
            menu = self._build_knowledge_menu(role_specific_menu)
            self.add_section("knowledge_menu", menu, priority=50)

        return self._assemble()

    def _build_identity_section(self, agent: Agent) -> str:
        """Build the agent identity section."""
        lines = []
        lines.append(f"You are {agent.name}.")

        if agent.description:
            lines.append(agent.description)

        if agent.archetypes:
            archetypes_str = ", ".join(agent.archetypes)
            lines.append(f"Your role archetype(s): {archetypes_str}")

        return "\n".join(lines)

    def _format_constitution(self, text: str) -> str:
        """Format constitution for injection."""
        return f"## Constitution (Inviolable Principles)\n\n{text}"

    def _build_must_know_section(self, entries: list[dict[str, str]]) -> str:
        """Build the must-know knowledge section."""
        lines = ["## Critical Knowledge\n"]

        for entry in entries:
            name = entry.get("name", entry.get("id", "Unknown"))
            content = entry.get("content", "")

            lines.append(f"### {name}\n")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def _build_constraints_section(self, agent: Agent) -> str:
        """Build the constraints section."""
        lines = ["## Constraints\n"]

        # Group by severity
        critical = []
        warnings = []

        for c in agent.constraints:
            if c.severity == "critical":
                critical.append(c)
            else:
                warnings.append(c)

        if critical:
            lines.append("**Critical (Never Violate):**")
            for c in critical:
                lines.append(f"- {c.rule}")
            lines.append("")

        if warnings:
            lines.append("**Guidelines:**")
            for c in warnings:
                lines.append(f"- {c.rule}")

        return "\n".join(lines)

    def _build_capabilities_section(self, agent: Agent) -> str:
        """Build the capabilities section."""
        lines = ["## Your Capabilities\n"]

        # Group by category
        by_category: dict[str, list[str]] = {}

        for cap in agent.capabilities:
            category = cap.category or "general"
            if category not in by_category:
                by_category[category] = []

            desc = cap.description or cap.name
            by_category[category].append(desc)

        for category, caps in by_category.items():
            lines.append(f"**{category.replace('_', ' ').title()}:**")
            for c in caps:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)

    def _build_knowledge_menu(self, entries: list[dict[str, str]]) -> str:
        """Build the available knowledge menu."""
        lines = ["## Available Knowledge\n"]
        lines.append("You can consult the following knowledge entries:\n")

        for entry in entries:
            name = entry.get("name", entry.get("id", "Unknown"))
            summary = entry.get("summary", "")
            entry_id = entry.get("id", "")

            lines.append(f"- **{name}** (`{entry_id}`): {summary}")

        return "\n".join(lines)

    def _assemble(self) -> BuiltPrompt:
        """Assemble sections into final prompt."""
        # Sort by priority (descending)
        sorted_sections = sorted(self._sections, key=lambda s: -s.priority)

        # Join with double newlines
        text = "\n\n".join(s.content for s in sorted_sections)

        # Estimate tokens
        token_estimate = len(text) // self.CHARS_PER_TOKEN

        return BuiltPrompt(
            text=text,
            sections=sorted_sections,
            token_estimate=token_estimate,
        )


# Convenience function
def build_prompt(
    agent: Agent,
    constitution_text: str | None = None,
    must_know_entries: list[dict[str, str]] | None = None,
    role_specific_menu: list[dict[str, str]] | None = None,
) -> BuiltPrompt:
    """Build a prompt for an agent."""
    builder = PromptBuilder()
    return builder.build_for_agent(
        agent=agent,
        constitution_text=constitution_text,
        must_know_entries=must_know_entries,
        role_specific_menu=role_specific_menu,
    )
