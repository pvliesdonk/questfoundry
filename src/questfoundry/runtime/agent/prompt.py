"""
Prompt builder for agent system prompts.

Constructs the system prompt from agent definition and injected knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
        tool_schemas: list[dict[str, Any]] | None = None,
        playbooks_menu: list[dict[str, Any]] | None = None,
        stores_menu: list[dict[str, Any]] | None = None,
        artifact_types_menu: list[dict[str, Any]] | None = None,
        agents_menu: list[dict[str, Any]] | None = None,
    ) -> BuiltPrompt:
        """
        Build the system prompt for an agent.

        Args:
            agent: The agent definition
            constitution_text: Full constitution text to inject
            must_know_entries: List of {id, name, content} for must_know knowledge
            role_specific_menu: List of {id, name, summary} for available knowledge
            tool_schemas: List of tool schemas from ToolRegistry.get_langchain_tools()
            playbooks_menu: List of {id, name, purpose, triggers} for available playbooks
            stores_menu: List of {id, name, description, semantics, access} for accessible stores
            artifact_types_menu: List of {id, name, description, category, actions} for workable types
            agents_menu: List of {id, name, description, archetypes, specialties} for delegation

        Returns:
            Built prompt with text and metadata
        """
        self.reset()

        # Priority ordering: actionable content first, reference material later
        # Higher priority = appears earlier in prompt
        #
        # Identity & Behavior (who you are, how to behave):
        #   identity (100), behavioral (98), agents (95)
        # Actionable (what to do NOW):
        #   tools (90), playbooks (85)
        # Operational (how to behave):
        #   capabilities (80), constraints (75)
        # Reference (background knowledge):
        #   constitution (65), must_know (60), stores (55), artifact_types (50)

        # 1. Agent Identity (highest priority - who you are)
        identity = self._build_identity_section(agent)
        self.add_section("identity", identity, priority=100)

        # 2. Available Agents for Delegation (actionable - who to delegate to)
        if agents_menu:
            agents_section = self._build_agents_section(agents_menu)
            self.add_section("agents", agents_section, priority=95)

        # 3. Behavioral Guidance (operational - WRONG/CORRECT examples by archetype)
        # V3 pattern: This was critical for keeping models on track
        archetypes = [a.value if hasattr(a, "value") else str(a) for a in (agent.archetypes or [])]
        has_persistence_tool = any(
            schema.get("name") == "save_artifact" for schema in (tool_schemas or [])
        )
        behavioral = self._build_behavioral_guidance(archetypes, has_persistence_tool)
        self.add_section("behavioral", behavioral, priority=98)

        # 3.5. Language Enforcement (operational - how to communicate)
        # Critical for multilingual models that may default to other languages
        language = self._build_language_section()
        self.add_section("language", language, priority=96)

        # 4. Available Tools (actionable - how to act)
        if tool_schemas:
            tools_section = self._build_tools_section(tool_schemas)
            self.add_section("tools", tools_section, priority=90)

        # 5. Available Playbooks (actionable - workflow guidance)
        if playbooks_menu:
            playbooks_section = self._build_playbooks_section(playbooks_menu)
            self.add_section("playbooks", playbooks_section, priority=85)

        # 6. Capabilities (operational - what you can do)
        if agent.capabilities:
            capabilities = self._build_capabilities_section(agent)
            self.add_section("capabilities", capabilities, priority=80)

        # 7. Constraints (operational - what not to do)
        if agent.constraints:
            constraints = self._build_constraints_section(agent)
            self.add_section("constraints", constraints, priority=75)

        # 8. Constitution (reference - principles)
        if constitution_text:
            self.add_section(
                "constitution", self._format_constitution(constitution_text), priority=65
            )

        # 9. Must Know (reference - knowledge including operational guidelines)
        if must_know_entries:
            must_know = self._build_must_know_section(must_know_entries)
            self.add_section("must_know", must_know, priority=60)

        # 10. Store Access (reference - what stores you can read/write)
        if stores_menu:
            stores_section = self._build_stores_section(stores_menu)
            self.add_section("stores", stores_section, priority=55)

        # 11. Artifact Types (reference - what types you can create)
        if artifact_types_menu:
            artifact_types_section = self._build_artifact_types_section(artifact_types_menu)
            self.add_section("artifact_types", artifact_types_section, priority=50)

        # 12. Available Knowledge Menu
        if role_specific_menu:
            menu = self._build_knowledge_menu(role_specific_menu)
            self.add_section("knowledge_menu", menu, priority=45)

        # 13. Critical Reminder (sandwich pattern - LOWEST priority = appears LAST)
        # Repeats critical knowledge at end to combat "Lost in the Middle" effect
        if must_know_entries:
            reminder = self._build_critical_reminder_section(must_know_entries)
            if reminder:
                self.add_section("critical_reminder", reminder, priority=0)

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

    def _build_language_section(self) -> str:
        """Build the language enforcement section.

        Critical for multilingual models (e.g., Qwen3) that may default
        to Chinese or other languages during generation.
        """
        return """## Language

ALL communication MUST be in English:
- Tool calls and their parameters
- Responses to other agents
- Status updates and reasoning
- Questions and clarifications

Story content language may vary based on project settings, but system
communication is always English."""

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

    def _build_critical_reminder_section(self, entries: list[dict[str, str]]) -> str | None:
        """Build the critical reminder section (sandwich pattern).

        Extracts entries with injection_priority="critical" and repeats
        them at the END of the prompt in condensed form. This combats
        the "Lost in the Middle" effect where models forget instructions
        that appear in the middle of long prompts.

        Args:
            entries: List of must_know entries with injection_priority and concise_summary

        Returns:
            Formatted reminder section, or None if no critical entries with summaries
        """
        # Collect reminder lines first, then check if any exist
        reminder_lines = []
        for entry in entries:
            if entry.get("injection_priority") == "critical":
                summary = entry.get("concise_summary", "")
                if summary:
                    reminder_lines.append(f"**{entry.get('name', entry.get('id'))}**: {summary}")

        if not reminder_lines:
            return None

        lines = ["## REMEMBER (Critical Rules)", ""]
        lines.extend(reminder_lines)
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

            # Include tool_ref when available so agent knows which tool to call
            if cap.tool_ref:
                desc = f"{cap.description or cap.name} (`{cap.tool_ref}`)"
            else:
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

    def _build_playbooks_section(self, playbooks: list[dict[str, Any]]) -> str:
        """Build compact playbooks menu.

        Args:
            playbooks: List of playbook menu items with id, name, purpose, triggers, workflow

        Returns:
            Compact formatted playbooks section for the prompt
        """
        lines = ["## Available Playbooks\n"]
        lines.append("Use `consult_playbook(id)` to get full workflow steps and outputs.\n")

        for pb in playbooks:
            name = pb.get("name", pb.get("id", "Unknown"))
            # Use summary (preferred) or fall back to purpose
            purpose = pb.get("summary") or pb.get("purpose", "")
            pb_id = pb.get("id", "")
            lines.append(f"- **{name}** (`{pb_id}`): {purpose}")

        return "\n".join(lines)

    def _build_agents_section(self, agents: list[dict[str, Any]]) -> str:
        """Build compact delegation agents table.

        V3 pattern: Compact table format instead of full descriptions.
        Details available via list_agents() or consulting agent charters.

        Args:
            agents: List of agent menu items with id, name, summary, description, archetypes, specialties

        Returns:
            Compact formatted agents section for the prompt
        """
        lines = ["## Available Agents for Delegation\n"]
        lines.append("| Agent | Archetype | Role |")
        lines.append("|-------|-----------|------|")

        for ag in agents:
            name = ag.get("name", ag.get("id", "Unknown"))
            archetypes = ag.get("archetypes", [])
            archetypes_str = ", ".join(archetypes[:2]) if archetypes else "general"
            # Use summary for role description
            role = ag.get("summary", "")
            lines.append(f"| {name} | {archetypes_str} | {role} |")

        lines.append("")
        lines.append("Use `delegate(to_agent, task, expected_outputs)` to assign work.")
        lines.append("Use `list_agents()` for full agent details.")

        return "\n".join(lines)

    def _build_tools_section(self, tool_schemas: list[dict[str, Any]]) -> str:
        """Build compact tools list.

        V3 pattern: Name + first line of description only.
        LangChain already provides full parameter schemas to the model.

        Args:
            tool_schemas: List of tool schemas in LangChain format
                          Each has: name, description, parameters

        Returns:
            Compact formatted tools section for the prompt
        """
        lines = ["## Your Tools\n"]

        # Chain-of-Thought: Require reasoning for every tool call
        lines.append("**IMPORTANT:** Before calling any tool, provide a `reasoning` parameter:")
        lines.append("- Explain WHY you are using this specific tool")
        lines.append("- State WHAT you expect to achieve")
        lines.append("- Describe HOW it advances your current task")
        lines.append("")

        for tool in tool_schemas:
            name = tool.get("name", "unknown")
            # Use summary if available, otherwise first line of description
            raw_desc = tool.get("description", "") or ""
            desc = tool.get("summary") or raw_desc.split("\n")[0]
            lines.append(f"- **{name}**: {desc}")

        return "\n".join(lines)

    def _build_stores_section(self, stores: list[dict[str, Any]]) -> str:
        """Build compact stores menu.

        V3 pattern: Compact list with semantics and access level.

        Args:
            stores: List of store menu items with id, name, description, semantics, access

        Returns:
            Compact formatted stores section for the prompt
        """
        lines = ["## Your Store Access\n"]

        for store in stores:
            name = store.get("name", store.get("id", "Unknown"))
            semantics = store.get("semantics", "unknown")
            access = store.get("access", "read")
            lines.append(f"- **{name}**: {access} ({semantics})")

        return "\n".join(lines)

    def _build_artifact_types_section(self, artifact_types: list[dict[str, Any]]) -> str:
        """Build compact artifact types menu.

        V3 pattern: Compact list with consult hint for full schemas.
        Types are already role-specific (filtered in context.py).

        Args:
            artifact_types: List of artifact type menu items with id, name, description, category, actions

        Returns:
            Compact formatted artifact types section for the prompt
        """
        lines = ["## Artifact Types\n"]
        lines.append("Use `consult_schema(id)` for full field definitions.\n")

        for at in artifact_types:
            name = at.get("name", at.get("id", "Unknown"))
            at_id = at.get("id", "")
            category = at.get("category", "document")
            actions = at.get("actions", ["read"])
            actions_str = ", ".join(actions)
            lines.append(f"- **{name}** (`{at_id}`): [{actions_str}] ({category})")

        return "\n".join(lines)

    def _build_behavioral_guidance(
        self, archetypes: list[str], include_persistence_guidance: bool
    ) -> str:
        """Build archetype-specific behavioral guidance with WRONG/CORRECT examples.

        V3 pattern: Explicit examples of what NOT to do and what TO do.
        This was critical for keeping models on track.

        Args:
            archetypes: List of archetype strings for the agent

        Returns:
            Formatted behavioral guidance section
        """
        lines = ["## Output Guidelines\n"]

        # Normalize archetypes to lowercase for matching
        archetypes_lower = [a.lower() for a in archetypes]

        if "orchestrator" in archetypes_lower:
            lines.extend(
                [
                    "## CRITICAL: Assess Before Acting",
                    "",
                    "**Before selecting a playbook**, assess the user's request:",
                    "1. Is the goal clear and specific? → Select appropriate playbook",
                    "2. Is the request ambiguous? → Use communicate(type='question') to clarify",
                    "",
                    "## Playbook Selection Guide",
                    "",
                    "Match user goal to playbook:",
                    "- Story structure/creation → `story_spark`",
                    "- Prose from existing briefs → `scene_weave`",
                    "- World lore expansion → `lore_deepening`",
                    "- Export for publication → `binding_run`",
                    "- Visual planning → `art_planning`",
                    "- Audio planning → `audio_planning`",
                    "- Hook/idea triage → `hook_harvest`",
                    "",
                    "Use `consult_playbook(playbook_id='...')` once you know which workflow fits.",
                    "",
                    "## Workflow Pattern",
                    "",
                    "1. User request arrives",
                    "2. Assess clarity - clarify if ambiguous",
                    "3. Select and consult appropriate playbook",
                    "4. Follow playbook steps, delegating to specialists",
                    "5. Pass artifact IDs between delegations",
                    "6. Delegate to gatekeeper for quality validation",
                    "",
                    "## Output Format",
                    "",
                    "Your responses should consist of:",
                    "1. **Tool calls** - communicate, consult_playbook, delegate",
                    "2. **Brief status updates** - 1-2 sentences",
                    "",
                    "**NEVER generate story content yourself.** If your response contains more",
                    "than 2-3 sentences that aren't a tool call, you're doing it wrong.",
                    "",
                    "## Anti-Pattern Reminder",
                    "",
                    "WRONG: The detective entered the dimly lit room, her footsteps echoing...",
                    "RIGHT: communicate(type='question', message='What genre and tone for this story?')",
                ]
            )
        elif any(a in archetypes_lower for a in ["creator", "author", "writer"]):
            if include_persistence_guidance:
                lines.extend(
                    [
                        "CRITICAL: Persist artifacts to storage BEFORE returning.",
                        "",
                        "WRONG: Returning without saving work",
                        "CORRECT: save_artifact(artifact_type='section', data={...}) then report the artifact ID",
                        "",
                        "Use save_artifact for every artifact you create and list those IDs in your response.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "CRITICAL: Return every turn with concrete artifact references.",
                        "",
                        "WRONG: “Draft complete” with no artifact noted",
                        "CORRECT: “Draft complete — ready to persist as section anchor005.”",
                    ]
                )
        elif any(a in archetypes_lower for a in ["validator", "auditor", "guardian"]):
            lines.extend(
                [
                    "CRITICAL: Read artifacts before approving.",
                    "",
                    "WRONG: Approving without inspection",
                    "CORRECT: read_artifact('scene_1') -> check criteria -> report findings",
                    "",
                    "Always read each artifact, evaluate against quality criteria,",
                    "then return with detailed validation results.",
                ]
            )
        elif any(a in archetypes_lower for a in ["architect", "planner", "designer"]):
            lines.extend(
                [
                    "CRITICAL: Design structure, leave content empty for creators.",
                    "",
                    "WRONG: Writing prose content in structural artifacts",
                    "CORRECT: Create skeleton with content='' for creators to fill",
                    "",
                    "Your role is topology - the 'bones' that prose hangs on.",
                    "Scene Smith fills the actual prose content.",
                ]
            )
        elif any(a in archetypes_lower for a in ["librarian", "curator", "archivist"]):
            lines.extend(
                [
                    "CRITICAL: List existing artifacts before promoting.",
                    "",
                    "WRONG: Promoting only what was explicitly mentioned",
                    "CORRECT: list_hot_store_keys() -> promote ALL matching artifacts",
                    "",
                    "Always discover what exists, then process everything.",
                ]
            )
        else:
            # Generic guidance for any role
            generic_lines = ["1. Read relevant artifacts before acting"]
            if include_persistence_guidance:
                generic_lines.append("2. Persist your work with save_artifact before returning")
            else:
                generic_lines.append("2. Document every artifact you touched before returning")
            generic_lines.append("3. Return with status, artifacts list, and recommendation")
            lines.extend(generic_lines)

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
    tool_schemas: list[dict[str, Any]] | None = None,
    playbooks_menu: list[dict[str, Any]] | None = None,
    stores_menu: list[dict[str, Any]] | None = None,
    artifact_types_menu: list[dict[str, Any]] | None = None,
    agents_menu: list[dict[str, Any]] | None = None,
) -> BuiltPrompt:
    """Build a prompt for an agent."""
    builder = PromptBuilder()
    return builder.build_for_agent(
        agent=agent,
        constitution_text=constitution_text,
        must_know_entries=must_know_entries,
        role_specific_menu=role_specific_menu,
        tool_schemas=tool_schemas,
        playbooks_menu=playbooks_menu,
        stores_menu=stores_menu,
        artifact_types_menu=artifact_types_menu,
        agents_menu=agents_menu,
    )
