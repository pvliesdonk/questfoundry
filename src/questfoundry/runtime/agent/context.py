"""
Context builder for agent activation.

Gathers and prepares all context needed for an agent, including
knowledge injection and conversation history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.agent.content_utils import extract_knowledge_content

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent, KnowledgeEntry, Playbook, Studio


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

    # Playbooks menu (for orchestrator agents)
    playbooks_menu: list[dict[str, Any]] = field(default_factory=list)

    # Stores menu (shows which stores agent can access)
    stores_menu: list[dict[str, Any]] = field(default_factory=list)

    # Artifact types menu (shows what types agent can create/read/update)
    artifact_types_menu: list[dict[str, Any]] = field(default_factory=list)

    # Agents menu (for orchestrators - shows agents available for delegation)
    agents_menu: list[dict[str, Any]] = field(default_factory=list)

    # Token estimates for context budget
    constitution_tokens: int = 0
    must_know_tokens: int = 0
    menu_tokens: int = 0
    playbooks_tokens: int = 0
    stores_tokens: int = 0
    artifact_types_tokens: int = 0
    agents_tokens: int = 0

    # Rough estimate: 4 chars per token
    CHARS_PER_TOKEN: int = 4

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens for all context."""
        return (
            self.constitution_tokens
            + self.must_know_tokens
            + self.menu_tokens
            + self.playbooks_tokens
            + self.stores_tokens
            + self.artifact_types_tokens
            + self.agents_tokens
        )


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

        # 4. Playbooks menu (for orchestrator agents)
        if self._is_orchestrator(agent) and studio.playbooks:
            for playbook in studio.playbooks:
                menu_item = self._format_playbook_for_menu(playbook)
                context.playbooks_menu.append(menu_item)
                context.playbooks_tokens += (
                    len(menu_item.get("purpose", "")) // self.CHARS_PER_TOKEN
                )

        # 4b. Agents menu (for orchestrator agents - who can they delegate to?)
        if self._is_orchestrator(agent) and studio.agents:
            # Build list of agents this orchestrator can delegate to
            delegatable_agents = self._extract_delegatable_agents(agent, studio)
            for agent_item in delegatable_agents:
                context.agents_menu.append(agent_item)
                context.agents_tokens += (
                    len(agent_item.get("description", "")) // self.CHARS_PER_TOKEN
                )

        # 5. Stores menu (from agent capabilities)
        stores_access = self._extract_stores_access(agent, studio)
        for store_item in stores_access:
            context.stores_menu.append(store_item)
            context.stores_tokens += len(store_item.get("description", "")) // self.CHARS_PER_TOKEN

        # 6. Artifact types menu (from agent capabilities)
        artifact_types_access = self._extract_artifact_types_access(agent, studio)
        for at_item in artifact_types_access:
            context.artifact_types_menu.append(at_item)
            context.artifact_types_tokens += (
                len(at_item.get("description", "")) // self.CHARS_PER_TOKEN
            )

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
        content = extract_knowledge_content(entry, self._domain_path) or ""

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

    def _is_orchestrator(self, agent: Agent) -> bool:
        """Check if agent has orchestrator archetype."""
        if not agent.archetypes:
            return False
        # Handle both string and Archetype enum values
        for a in agent.archetypes:
            arch_str = a.value if hasattr(a, "value") else str(a)
            if arch_str == "orchestrator":
                return True
        return False

    def _format_playbook_for_menu(self, playbook: Playbook) -> dict[str, Any]:
        """Format a playbook for the playbooks menu.

        Returns a dict with:
        - id: Playbook identifier
        - name: Human-readable name
        - purpose: What this playbook accomplishes
        - triggers: When to use this playbook
        - workflow: Summary of phases and key agents (for orchestrator guidance)
        """
        triggers = []
        if playbook.triggers:
            for t in playbook.triggers:
                if isinstance(t, dict):
                    triggers.append(t.get("condition", ""))
                else:
                    triggers.append(str(t))

        # Build workflow summary from phases
        workflow_summary = self._build_workflow_summary(playbook)

        return {
            "id": playbook.id,
            "name": playbook.name or playbook.id,
            "purpose": playbook.purpose or "",
            "triggers": triggers,
            "workflow": workflow_summary,
        }

    def _build_workflow_summary(self, playbook: Playbook) -> str:
        """Build a condensed workflow summary from playbook phases.

        Returns a string like:
        "topology_design (plotwright) -> brief_creation (plotwright) -> preview_gate (gatekeeper)"
        """
        if not playbook.phases:
            return ""

        # Extract phase info
        phase_summaries = []
        for phase_id, phase in playbook.phases.items():
            # Get the primary agent for this phase from steps
            agents = set()
            if isinstance(phase, dict):
                steps = phase.get("steps", {})
                for step in steps.values():
                    if isinstance(step, dict):
                        if step.get("specific_agent"):
                            agents.add(step["specific_agent"])
                        elif step.get("agent_archetype"):
                            agents.add(f"[{step['agent_archetype']}]")

                phase_name = phase.get("name", phase_id)
            else:
                phase_name = phase_id

            if agents:
                agents_str = "/".join(sorted(agents))
                phase_summaries.append(f"{phase_name} ({agents_str})")
            else:
                phase_summaries.append(phase_name)

        # Order by dependency if possible, otherwise use dict order
        # For now, just join in definition order
        return " -> ".join(phase_summaries)

    def _extract_stores_access(self, agent: Agent, studio: Studio) -> list[dict[str, Any]]:
        """Extract stores the agent can access from capabilities.

        Returns a list of dicts with:
        - id: Store identifier
        - name: Human-readable name
        - description: What this store contains
        - semantics: Storage behavior (hot/cold/versioned/ephemeral)
        - access: 'read', 'write', or 'read/write'
        """
        # Build a lookup of stores by ID
        stores_by_id = {s.id: s for s in (studio.stores or [])}

        # Collect store access from capabilities
        stores_access: dict[str, set[str]] = {}  # store_id -> set of access levels

        if agent.capabilities:
            for cap in agent.capabilities:
                if cap.category == "store_access" and cap.stores:
                    access = cap.access_level or "read"
                    for store_id in cap.stores:
                        if store_id not in stores_access:
                            stores_access[store_id] = set()
                        stores_access[store_id].add(access)

        # Build menu items
        result = []
        for store_id, access_levels in stores_access.items():
            store = stores_by_id.get(store_id)
            if not store:
                continue

            # Combine access levels
            if "read" in access_levels and "write" in access_levels:
                access_str = "read/write"
            elif "write" in access_levels:
                access_str = "write"
            else:
                access_str = "read"

            result.append(
                {
                    "id": store.id,
                    "name": store.name,
                    "description": store.description or "",
                    "semantics": store.semantics.value if store.semantics else "unknown",
                    "access": access_str,
                }
            )

        return result

    def _extract_artifact_types_access(self, agent: Agent, studio: Studio) -> list[dict[str, Any]]:
        """Extract artifact types the agent can work with from capabilities.

        Returns a list of dicts with:
        - id: Artifact type identifier
        - name: Human-readable name
        - description: What this type represents
        - category: Type category (document, record, etc.)
        - actions: List of allowed actions (create, read, update, delete)
        """
        # Build a lookup of artifact types by ID
        types_by_id = {at.id: at for at in (studio.artifact_types or [])}

        # Collect artifact type access from capabilities
        types_access: dict[str, set[str]] = {}  # type_id -> set of actions

        if agent.capabilities:
            for cap in agent.capabilities:
                if cap.category == "artifact_action" and cap.artifact_types:
                    actions = cap.actions or ["read"]
                    for type_id in cap.artifact_types:
                        if type_id not in types_access:
                            types_access[type_id] = set()
                        types_access[type_id].update(actions)

        # Build menu items
        result = []
        for type_id, action_set in types_access.items():
            at = types_by_id.get(type_id)
            if not at:
                continue

            result.append(
                {
                    "id": at.id,
                    "name": at.name,
                    "description": at.description or "",
                    "category": at.category or "document",
                    "actions": sorted(action_set),
                }
            )

        return result

    def _extract_delegatable_agents(self, agent: Agent, studio: Studio) -> list[dict[str, Any]]:
        """Extract agents this orchestrator can delegate to.

        Checks the agent's capabilities for delegation permissions and
        returns a menu of available agents with their roles and responsibilities.

        Returns a list of dicts with:
        - id: Agent identifier
        - name: Human-readable name
        - description: What this agent does
        - archetypes: Agent archetypes (orchestrator, creator, validator, etc.)
        - specialties: Key capabilities/domains
        """
        result: list[dict[str, Any]] = []

        # Check if agent has delegation capability
        # For now, any capability with category="delegation" enables delegation to all agents
        # TODO: Add fine-grained delegation control when meta/ schemas support it
        has_delegation_capability = False

        if agent.capabilities:
            for cap in agent.capabilities:
                if cap.category == "delegation":
                    has_delegation_capability = True
                    break

        if not has_delegation_capability:
            return result

        # Build menu of delegatable agents (all agents except self)
        for target_agent in studio.agents or []:
            # Don't include self
            if target_agent.id == agent.id:
                continue

            # Extract key specialties from capabilities
            specialties = []
            if target_agent.capabilities:
                for cap in target_agent.capabilities[:3]:  # First 3 capabilities
                    if cap.description:
                        specialties.append(cap.description)
                    elif cap.name:
                        specialties.append(cap.name)

            archetypes = []
            if target_agent.archetypes:
                # Handle both string and Archetype enum values
                archetypes = [
                    a.value if hasattr(a, "value") else str(a) for a in target_agent.archetypes
                ]

            result.append(
                {
                    "id": target_agent.id,
                    "name": target_agent.name,
                    "summary": target_agent.summary or "",
                    "description": target_agent.description or "",
                    "archetypes": archetypes,
                    "specialties": specialties,
                }
            )

        return result


# Convenience function
def build_context(
    agent: Agent,
    studio: Studio,
    domain_path: Path | None = None,
) -> AgentContext:
    """Build context for an agent."""
    builder = ContextBuilder(domain_path=domain_path)
    return builder.build(agent, studio)
