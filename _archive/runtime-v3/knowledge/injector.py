"""Knowledge injector - builds agent prompts with injected knowledge.

The injector assembles system prompts for agents using:
1. Constitution (always, if agent requires it)
2. Must-know entries (full text injected)
3. Role-specific entries (summary menu + consult tool)
4. Agent identity (name, description, constraints)
5. Runtime nudges (tool guidance, etc.)
6. Playbook nudges (from tracker, when active)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from questfoundry.runtime.domain.models import (
    Agent,
    KnowledgeEntry,
    Studio,
)

if TYPE_CHECKING:
    from questfoundry.runtime.playbook_tracker import PlaybookTracker

logger = logging.getLogger(__name__)


def build_agent_prompt(agent: Agent, studio: Studio) -> str:
    """Build complete system prompt for an agent.

    Assembles the prompt from:
    - Constitution (if agent.knowledge_requirements.constitution is True)
    - Must-know entries (full content injected)
    - Role-specific menu (summaries only, with consult instruction)
    - Agent identity section
    - Constraints section
    - Runtime nudges

    Args:
        agent: The agent to build the prompt for
        studio: The loaded studio containing knowledge entries

    Returns:
        Complete system prompt string
    """
    sections: list[str] = []

    # 1. Constitution (if required)
    if agent.knowledge_requirements.constitution:
        constitution_section = _build_constitution_section(studio)
        if constitution_section:
            sections.append(constitution_section)

    # 2. Must-know entries (always inject full text)
    must_know_section = _build_must_know_section(agent, studio)
    if must_know_section:
        sections.append(must_know_section)

    # 3. Role-specific menu (summaries only)
    role_specific_section = _build_role_specific_section(agent, studio)
    if role_specific_section:
        sections.append(role_specific_section)

    # 4. Agent identity and constraints
    identity_section = _build_identity_section(agent)
    sections.append(identity_section)

    constraints_section = _build_constraints_section(agent)
    if constraints_section:
        sections.append(constraints_section)

    # 5. Runtime nudges
    nudges_section = _build_runtime_nudges(agent, studio)
    if nudges_section:
        sections.append(nudges_section)

    return "\n\n".join(sections)


def _build_constitution_section(studio: Studio) -> str | None:
    """Build constitution section from studio constitution."""
    if not studio.constitution:
        return None

    lines = ["# Constitution", "", studio.constitution.preamble, "", "## Principles"]

    for principle in studio.constitution.principles:
        lines.append(f"\n### {principle.id}: {principle.statement}")
        lines.append(f"\n*Rationale*: {principle.rationale}")
        if principle.enforcement == "absolute":
            lines.append("\n**Enforcement**: Absolute - no exceptions.")

    return "\n".join(lines)


def _build_must_know_section(agent: Agent, studio: Studio) -> str | None:
    """Build must-know section with full content injection."""
    kr = agent.knowledge_requirements
    if not kr.must_know:
        return None

    entries: list[str] = []

    for entry_id in kr.must_know:
        entry = studio.knowledge_entries.get(entry_id)
        if not entry:
            logger.debug(f"Must-know entry '{entry_id}' not found for agent {agent.id}")
            continue

        if not _agent_can_access(agent, entry):
            logger.debug(
                f"Agent {agent.id} cannot access must-know entry '{entry_id}'"
            )
            continue

        content = _get_entry_content(entry)
        if content:
            entries.append(f"## {entry.name}\n\n{content}")

    if not entries:
        return None

    return "# Critical Knowledge\n\n" + "\n\n---\n\n".join(entries)


def _build_role_specific_section(agent: Agent, studio: Studio) -> str | None:
    """Build role-specific menu with summaries."""
    kr = agent.knowledge_requirements
    if not kr.role_specific:
        return None

    menu_items: list[str] = []

    for entry_id in kr.role_specific:
        entry = studio.knowledge_entries.get(entry_id)
        if not entry:
            logger.debug(
                f"Role-specific entry '{entry_id}' not found for agent {agent.id}"
            )
            continue

        if not _agent_can_access(agent, entry):
            continue

        summary = entry.summary or f"Reference material: {entry.name}"
        menu_items.append(f"- **{entry.name}** (`{entry.id}`): {summary}")

    if not menu_items:
        return None

    section = "# Available Reference Material\n\n"
    section += "\n".join(menu_items)
    section += "\n\nUse `consult_knowledge(id)` to retrieve full details."

    return section


def _build_identity_section(agent: Agent) -> str:
    """Build agent identity section."""
    lines = [
        f"# Your Role: {agent.name}",
        "",
        agent.description,
        "",
    ]

    if agent.archetypes:
        archetypes_str = ", ".join(agent.archetypes)
        lines.append(f"**Archetypes**: {archetypes_str}")

    if agent.is_entry_agent:
        lines.append(
            "**Role**: You are an entry agent - you receive requests directly "
            "from users and coordinate work across the studio."
        )

    return "\n".join(lines)


def _build_constraints_section(agent: Agent) -> str | None:
    """Build constraints section."""
    if not agent.constraints:
        return None

    lines = ["# Constraints", ""]

    # Group by severity
    critical = [c for c in agent.constraints if c.severity == "critical"]
    errors = [c for c in agent.constraints if c.severity == "error"]
    warnings = [c for c in agent.constraints if c.severity == "warning"]

    if critical:
        lines.append("## CRITICAL (Never violate)")
        for c in critical:
            lines.append(f"- **{c.name}**: {c.rule}")
        lines.append("")

    if errors:
        lines.append("## Required")
        for c in errors:
            lines.append(f"- **{c.name}**: {c.rule}")
        lines.append("")

    if warnings:
        lines.append("## Guidance")
        for c in warnings:
            lines.append(f"- **{c.name}**: {c.rule}")

    return "\n".join(lines)


def _build_runtime_nudges(agent: Agent, studio: Studio) -> str | None:
    """Build runtime nudges section."""
    nudges: list[str] = []

    # Tool usage guidance
    tool_caps = [c for c in agent.capabilities if c.category == "tool"]
    if tool_caps:
        tool_names = [c.tool_ref for c in tool_caps if c.tool_ref]
        if tool_names:
            nudges.append(
                f"**Available tools**: {', '.join(tool_names)}. "
                "Use `consult_schema(artifact_type)` before creating artifacts."
            )

    # Store access guidance
    store_caps = [c for c in agent.capabilities if c.category == "store_access"]
    if store_caps:
        readable = set()
        writable = set()
        for cap in store_caps:
            if cap.stores:
                if cap.access_level in ("write", "admin"):
                    writable.update(cap.stores)
                readable.update(cap.stores)

        if readable:
            nudges.append(f"**Readable stores**: {', '.join(sorted(readable))}")
        if writable:
            nudges.append(f"**Writable stores**: {', '.join(sorted(writable))}")

    # Delegation guidance for orchestrators
    if "orchestrator" in agent.archetypes:
        nudges.append(
            "**Delegation**: Use `delegate(to_agent, task, context)` to assign work. "
            "Include relevant artifact_refs and expected_outputs."
        )

    # Can-lookup hint
    kr = agent.knowledge_requirements
    if kr.can_lookup:
        nudges.append(
            "**Lookup available**: Use `query_knowledge(search_term)` "
            "to search the knowledge base for detailed information."
        )

    if not nudges:
        return None

    return "# Runtime Guidance\n\n" + "\n".join(nudges)


def _agent_can_access(agent: Agent, entry: KnowledgeEntry) -> bool:
    """Check if an agent can access a knowledge entry based on applicable_to."""
    if not entry.applicable_to:
        # No restrictions - everyone can access
        return True

    # Check agent ID
    if entry.applicable_to.agents and agent.id in entry.applicable_to.agents:
        return True

    # Check archetypes
    if entry.applicable_to.archetypes:
        for archetype in agent.archetypes:
            if archetype in entry.applicable_to.archetypes:
                return True

    # If applicable_to is set but agent doesn't match, deny
    if entry.applicable_to.agents or entry.applicable_to.archetypes:
        return False

    # Empty applicable_to lists means everyone can access
    return True


def _get_entry_content(entry: KnowledgeEntry) -> str | None:
    """Extract content from a knowledge entry."""
    content = entry.content

    if content.type == "inline" and content.text:
        return content.text

    if content.type == "file_ref" and content.path:
        # File references would need to be resolved at load time
        # For now, log warning and return None
        logger.warning(
            f"File ref content not supported yet for entry {entry.id}: {content.path}"
        )
        return None

    return None


def build_playbook_nudge(tracker: "PlaybookTracker | None") -> str | None:
    """Build playbook nudge section from tracker state.

    This function generates runtime nudges based on playbook progress.
    Call this periodically (e.g., before each agent turn) to inject
    contextual guidance.

    Args:
        tracker: The playbook tracker with current state

    Returns:
        Nudge section if nudges are needed, None otherwise
    """
    if not tracker:
        return None

    nudges: list[str] = []

    # Get playbook progress nudge
    playbook_nudge = tracker.get_nudge()
    if playbook_nudge:
        nudges.append(playbook_nudge)

    # Get phase guidance
    phase_guidance = tracker.get_phase_guidance()
    if phase_guidance:
        nudges.append(phase_guidance)

    if not nudges:
        return None

    return "# Playbook Status\n\n" + "\n\n".join(nudges)


def inject_playbook_context(
    base_prompt: str,
    tracker: "PlaybookTracker | None",
) -> str:
    """Inject playbook context into an agent prompt.

    This is a convenience function that appends playbook nudges
    to an existing prompt.

    Args:
        base_prompt: The base agent prompt
        tracker: The playbook tracker

    Returns:
        Updated prompt with playbook context if applicable
    """
    nudge = build_playbook_nudge(tracker)
    if nudge:
        return f"{base_prompt}\n\n{nudge}"
    return base_prompt
