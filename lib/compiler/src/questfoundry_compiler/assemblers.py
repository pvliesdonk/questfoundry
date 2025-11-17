"""Assemblers for composing prompts and resolving references."""

import logging
import re
from pathlib import Path
from typing import Any, Literal, cast

from questfoundry_compiler.types import BehaviorPrimitive, CompilationError

logger = logging.getLogger(__name__)

ProfileType = Literal["walkthrough", "reference", "brief"]


class ReferenceResolver:
    """Resolve references and assemble content."""

    def __init__(self, primitives: dict[str, BehaviorPrimitive], spec_root: Path):
        """Initialize resolver.

        Args:
            primitives: Dictionary of loaded primitives
            spec_root: Root directory of spec/
        """
        self.primitives = primitives
        self.spec_root = Path(spec_root)
        self.reference_pattern = re.compile(r"@(\w+):([a-z_0-9]+)(?:#([a-z_0-9-]+))?")
        # Pre-compile section extraction pattern (## heading only, not deeper levels)
        self._section_pattern_template = (
            r"## [^#\n]*{section_id}[^#\n]*\n(.*?)(?=\n## |\Z)"
        )
        logger.debug(f"Initialized ReferenceResolver with {len(primitives)} primitives")

    def resolve_reference(self, ref: str, inline_content: bool = True) -> str:
        """Resolve a single reference.

        Args:
            ref: Reference string like '@expertise:lore_weaver_expertise'
            inline_content: Whether to inline the content or just create a link

        Returns:
            Resolved content or link

        Raises:
            CompilationError: If reference cannot be resolved
        """
        logger.debug(f"Resolving reference: {ref} (inline={inline_content})")
        match = self.reference_pattern.match(ref)
        if not match:
            logger.error(f"Invalid reference format: {ref}")
            raise CompilationError(f"Invalid reference format: {ref}")

        ref_type, ref_id, section = match.groups()

        # Handle schema and role references (always links, never inline)
        if ref_type == "schema":
            schema_path = self.spec_root / "03-schemas" / ref_id
            if schema_path.exists():
                return f"[`{ref_id}`](../../../03-schemas/{ref_id})"
            raise CompilationError(f"Schema not found: {ref_id}")

        if ref_type == "role":
            role_path = self.spec_root / "01-roles" / "charters" / f"{ref_id}.md"
            if role_path.exists():
                return f"[{ref_id}](../../../01-roles/charters/{ref_id}.md)"
            raise CompilationError(f"Role not found: {ref_id}")

        # Handle behavior primitive references
        prim_key = f"{ref_type}:{ref_id}"
        primitive = self.primitives.get(prim_key)

        if not primitive:
            raise CompilationError(f"Primitive not found: {ref_type}:{ref_id}")

        # For playbook/adapter references, always create links
        if ref_type in ["playbook", "adapter"] or not inline_content:
            ext = self._get_extension(ref_type)
            return f"[{ref_id}](../05-behavior/{ref_type}s/{ref_id}.{ext})"

        # Inline the content
        content = primitive.content

        # If section anchor specified, extract that section
        if section:
            content = self._extract_section(content, section)

        return content

    def _get_extension(self, ref_type: str) -> str:
        """Get file extension for primitive type."""
        if ref_type in ["playbook", "adapter"]:
            return "yaml"
        return "md"

    def _extract_section(self, content: str, section_id: str) -> str:
        """Extract a specific section from markdown content.

        Args:
            content: Full markdown content
            section_id: Section identifier (e.g., 'step1')

        Returns:
            Extracted section content

        Raises:
            CompilationError: If section not found
        """
        # Look for heading with matching ID (## level only for precision)
        # Support both "## Step 1" and "## step1" formats
        pattern = self._section_pattern_template.format(section_id=section_id)
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if not match:
            raise CompilationError(f"Section '{section_id}' not found in content")

        return match.group(1).strip()

    def assemble_primitive_content(self, prim_type: str, prim_id: str) -> str:
        """Assemble complete primitive markdown by resolving references.

        Args:
            prim_type: Type of primitive ('expertise', 'procedure', etc.)
            prim_id: ID of primitive to assemble

        Returns:
            Assembled markdown content

        Raises:
            CompilationError: If primitive not found
        """
        primitive = self.primitives.get(f"{prim_type}:{prim_id}")
        if not primitive:
            raise CompilationError(f"{prim_type.capitalize()} not found: {prim_id}")

        content = primitive.content

        # Resolve all embedded references
        def replace_ref(match: re.Match[str]) -> str:
            ref = match.group(0)
            try:
                return self.resolve_reference(ref, inline_content=True)
            except CompilationError as e:
                logger.error(f"Compilation Error: {e}")
                # If resolution fails, keep the reference as-is
                return ref

        assembled = self.reference_pattern.sub(replace_ref, content)
        return assembled


class StandalonePromptAssembler:
    """Assemble standalone prompts for roles."""

    def __init__(
        self,
        primitives: dict[str, BehaviorPrimitive],
        resolver: ReferenceResolver,
        spec_root: Path,
    ):
        """Initialize assembler.

        Args:
            primitives: Dictionary of loaded primitives
            resolver: Reference resolver
            spec_root: Root directory of spec/
        """
        self.primitives = primitives
        self.resolver = resolver
        self.spec_root = Path(spec_root)

    def assemble_role_prompt(self, adapter_id: str) -> str:
        """Assemble complete standalone prompt for a role.

        Args:
            adapter_id: ID of adapter (role)

        Returns:
            Complete assembled markdown prompt

        Raises:
            CompilationError: If assembly fails
        """
        adapter = self.primitives.get(f"adapter:{adapter_id}")
        if not adapter:
            raise CompilationError(f"Adapter not found: {adapter_id}")

        data = adapter.metadata
        role_name = data.get("role_name", adapter_id)

        # Build prompt sections
        sections = []

        # Header
        sections.append(f"# {role_name} — System Prompt")
        sections.append("")
        sections.append("Target: GPT-5, Claude Sonnet 4.5+")
        sections.append("")

        # Mission
        if "mission" in data:
            sections.append("## Mission")
            sections.append("")
            sections.append(data["mission"])
            sections.append("")

        # References
        sections.append("## References")
        sections.append("")
        if "references" in data:
            if "layer_1" in data["references"]:
                role_ref = self.resolver.resolve_reference(
                    data["references"]["layer_1"], inline_content=False
                )
                sections.append(f"- {role_ref}")
        sections.append(
            f"- Compiled from: spec/05-behavior/adapters/{adapter_id}.adapter.yaml"
        )
        sections.append("")
        sections.append("---")
        sections.append("")

        # Core Expertise
        if "expertise" in data:
            expertise_ref = data["expertise"]
            try:
                expertise_content = self.resolver.resolve_reference(
                    expertise_ref, inline_content=True
                )
                sections.append("## Core Expertise")
                sections.append("")
                sections.append(expertise_content)
                sections.append("")
                sections.append("---")
                sections.append("")
            except CompilationError as e:
                logger.error(f"Compilation Error: {e}")
                sections.append(f"<!-- Error loading expertise: {e} -->")
                sections.append("")

        # Primary Procedures
        if "procedures" in data and "primary" in data["procedures"]:
            sections.append("## Primary Procedures")
            sections.append("")
            for proc_ref in data["procedures"]["primary"]:
                try:
                    proc_content = self.resolver.resolve_reference(
                        proc_ref, inline_content=True
                    )
                    sections.append(proc_content)
                    sections.append("")
                except CompilationError as e:
                    logger.error(f"Compilation Error: {e}")
                    sections.append(f"<!-- Error loading procedure {proc_ref}: {e} -->")
                    sections.append("")
            sections.append("---")
            sections.append("")

        # Safety & Validation
        if "safety_protocols" in data:
            sections.append("## Safety & Validation")
            sections.append("")
            for snippet_ref in data["safety_protocols"]:
                try:
                    snippet_content = self.resolver.resolve_reference(
                        snippet_ref, inline_content=True
                    )
                    sections.append(snippet_content)
                    sections.append("")
                except CompilationError as e:
                    logger.error(f"Compilation Error: {e}")
                    sections.append(
                        f"<!-- Error loading snippet {snippet_ref}: {e} -->"
                    )
                    sections.append("")
            sections.append("---")
            sections.append("")

        # Protocol Intents
        if "protocol_intents" in data:
            sections.append("## Protocol Intents")
            sections.append("")
            intents = data["protocol_intents"]
            if "receives" in intents:
                sections.append("**Receives:**")
                for intent in intents["receives"]:
                    sections.append(f"- `{intent}`")
                sections.append("")
            if "sends" in intents:
                sections.append("**Sends:**")
                for intent in intents["sends"]:
                    sections.append(f"- `{intent}`")
                sections.append("")
            sections.append("---")
            sections.append("")

        # Loop Participation
        if "loops" in data:
            sections.append("## Loop Participation")
            sections.append("")
            for loop in data["loops"]:
                playbook = loop.get("playbook", "")
                raci = loop.get("raci", "")
                desc = loop.get("description", "")
                sections.append(f"**{playbook}** ({raci})")
                if desc:
                    sections.append(f": {desc}")
                sections.append("")
            sections.append("---")
            sections.append("")

        # Escalation Rules
        if "escalation" in data:
            sections.append("## Escalation Rules")
            sections.append("")
            escalation = data["escalation"]
            if "ask_human" in escalation:
                sections.append("**Ask Human:**")
                for rule in escalation["ask_human"]:
                    sections.append(f"- {rule}")
                sections.append("")
            if "wake_showrunner" in escalation:
                sections.append("**Wake Showrunner:**")
                for rule in escalation["wake_showrunner"]:
                    sections.append(f"- {rule}")
                sections.append("")
            sections.append("---")
            sections.append("")

        return "\n".join(sections)


class PromptAssembler:
    """Assemble monolithic web prompts for loops or roles."""

    def __init__(
        self,
        primitives: dict[str, BehaviorPrimitive],
        resolver: ReferenceResolver,
        spec_root: Path,
    ):
        """Initialize assembler.

        Args:
            primitives: Dictionary of loaded primitives
            resolver: Reference resolver
            spec_root: Root directory of spec/
        """
        self.primitives = primitives
        self.resolver = resolver
        self.spec_root = Path(spec_root)

    def _parse_raci_from_markdown(self, loop_name: str) -> dict[str, list[str]]:
        """Parse RACI assignments from spec/01-roles/raci/by_loop.md.

        Args:
            loop_name: Name of the loop (e.g., "Lore Deepening", "Story Spark")

        Returns:
            Dict with keys 'R', 'A', 'C', 'I' containing role abbreviations

        Raises:
            CompilationError: If RACI file not found or loop not found
        """
        raci_file = self.spec_root / "01-roles" / "raci" / "by_loop.md"
        if not raci_file.exists():
            raise CompilationError(f"RACI file not found: {raci_file}")

        # Read and parse the markdown
        content = raci_file.read_text(encoding="utf-8")

        # Map of role abbreviations to adapter IDs
        role_map = {
            "SR": "showrunner",
            "GK": "gatekeeper",
            "PW": "plotwright",
            "SS": "scene_smith",
            "ST": "style_lead",
            "LW": "lore_weaver",
            "CC": "codex_curator",
            "RS": "researcher",
            "AD": "art_director",
            "IL": "illustrator",
            "AuD": "audio_director",
            "AuP": "audio_producer",
            "TR": "translator",
            "BB": "book_binder",
            "PN": "player_narrator",
        }

        # Find the section for this loop
        # Pattern: ## N) Loop Name
        loop_pattern = rf"## \d+\)\s+{re.escape(loop_name)}\s*(?:\(.*?\))?\s*\n"
        match = re.search(loop_pattern, content, re.IGNORECASE)
        if not match:
            raise CompilationError(f"Loop '{loop_name}' not found in RACI by_loop.md")

        # Extract section until next heading
        start_pos = match.end()
        next_section = re.search(r"\n##\s", content[start_pos:])
        if next_section:
            section = content[start_pos : start_pos + next_section.start()]
        else:
            section = content[start_pos:]

        # Parse RACI lines
        raci: dict[str, list[str]] = {"R": [], "A": [], "C": [], "I": []}

        # Look for lines like: - **R:** PW
        # or: - **R (plan):** AD
        for line in section.split("\n"):
            for key in ["R", "A", "C", "I"]:
                pattern = rf"-\s+\*\*{key}(?:\s*\([^)]+\))?\s*:\*\*\s+(.+)"
                match = re.search(pattern, line)
                if match:
                    roles_str = match.group(1)
                    # Handle "All creation roles (...)" specially
                    if "All creation roles" in roles_str:
                        # Extract roles from parentheses
                        paren_match = re.search(r"\(([^)]+)\)", roles_str)
                        if paren_match:
                            roles_str = paren_match.group(1)

                    # Split on comma and resolve abbreviations
                    for abbrev in re.findall(r"\b[A-Z]{2,3}\b", roles_str):
                        if abbrev in role_map:
                            adapter_id = role_map[abbrev]
                            if adapter_id not in raci[key]:
                                raci[key].append(adapter_id)

        return raci

    def _get_adapter_id_from_ref(self, ref: str) -> str | None:
        """Extract adapter ID from a reference string.

        Args:
            ref: Reference like '@adapter:lore_weaver'

        Returns:
            Adapter ID or None if not a valid adapter reference
        """
        match = self.resolver.reference_pattern.match(ref)
        if match and match.group(1) == "adapter":
            return match.group(2)
        return None

    def _load_role_categories(self) -> dict[str, str]:
        """Load role categories from ROLE_INDEX.md (Layer 0)."""

        cache = getattr(self, "_role_category_cache", None)
        if cache is not None:
            return cache

        role_index = self.spec_root / "00-north-star" / "ROLE_INDEX.md"
        categories: dict[str, str] = {}
        if role_index.exists():
            current_category: str | None = None
            for raw_line in role_index.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line.startswith("## "):
                    current_category = line[3:].strip()
                    continue
                if line.startswith("### ") and current_category:
                    role_name = line[4:].strip()
                    if not role_name:
                        continue
                    role_name = role_name.split("(")[0].strip()
                    categories[role_name] = current_category

        self._role_category_cache = categories
        return categories

    def _get_role_category(self, role_name: str) -> str:
        return self._load_role_categories().get(role_name, "Uncategorized")

    def _collect_role_safety_snippets(self, role_ids: set[str]) -> list[str]:
        """Gather safety snippets referenced by participating roles."""

        snippets: list[str] = []
        seen: set[str] = set()
        for role_id in role_ids:
            adapter = self.primitives.get(f"adapter:{role_id}")
            if not adapter:
                continue
            for snippet_ref in adapter.metadata.get("safety_protocols", []) or []:
                if snippet_ref not in seen:
                    seen.add(snippet_ref)
                    snippets.append(snippet_ref)
        return snippets

    def _normalize_profile(self, profile: ProfileType | str | None) -> ProfileType:
        if isinstance(profile, str):
            lowered = profile.lower()
            if lowered in {"walkthrough", "brief"}:
                return cast(ProfileType, lowered)
        return "reference"

    def _build_loop_controller_block(
        self,
        playbook_name: str,
        data: dict[str, Any],
        raci_roles: dict[str, list[str]],
        profile: ProfileType,
    ) -> list[str]:
        heading = {
            "walkthrough": "Controller Checklist",
            "reference": "Controller Notes",
            "brief": "Quick Controller",
        }[profile]

        lines: list[str] = [f"## {heading}", ""]

        responsible = raci_roles.get("R") or []
        if responsible:
            lines.append(f"**Primary Owners:** {', '.join(responsible)}")
            lines.append("")

        activation = data.get("activation_criteria") or {}
        triggers = activation.get("triggers") or []
        if triggers:
            lines.append("**Activation Triggers:**")
            for trig in triggers:
                lines.append(f"- {trig}")
            lines.append("")

        showrunner_decision = activation.get("showrunner_decision")
        if showrunner_decision:
            lines.append("> ➡️ Decision Gate: " + showrunner_decision)
            lines.append("")

        inputs = data.get("inputs") or []
        if inputs:
            lines.append("> ➡️ Gather these inputs from the user:")
            for item in inputs:
                lines.append(f"> - {item}")
            lines.append("")

        deliverables = data.get("deliverables") or {}
        if deliverables:
            lines.append("**Deliverables to return:**")
            for name in deliverables.keys():
                pretty = name.replace("_", " ").title()
                lines.append(f"- {pretty}")
            lines.append("")

        if profile == "walkthrough":
            lines.extend(
                [
                    (
                        "1. Confirm the triggers above are satisfied before "
                        "opening the TU."
                    ),
                    "2. Ask the user for every input listed before Step 1.",
                    "3. Narrate each step, pausing whenever a ➡️ callout appears.",
                    (
                        "4. Close the "
                        f"{playbook_name} loop with a summary + deliverables."
                    ),
                    "",
                ]
            )
        elif profile == "brief":
            lines.extend(
                [
                    "- Confirm scope → gather inputs → act.",
                    "- Keep narration lightweight; expand only on request.",
                    "",
                ]
            )

        return lines

    def _format_reference_label(self, ref: str) -> str:
        if ref.startswith("@") and ":" in ref:
            label = ref.split(":", 1)[1]
            label = label.replace("@", "").replace("_", " ")
            return label.replace(".schema.json", " schema").title()
        return ref

    def _build_step_callouts(self, step_data: dict[str, Any]) -> list[str]:
        prompts: list[str] = []
        intents = step_data.get("protocol_intents")
        if not intents:
            intents = step_data.get("protocol_intent")
        if isinstance(intents, str):
            intents = [intents]
        if intents:
            prompts.append("Prepare to send intents: " + ", ".join(intents))

        owner = step_data.get("owner")
        if isinstance(owner, str):
            owner_label = owner.split(":")[-1].replace("@adapter", "").replace("_", " ")
            prompts.append(f"Check in with {owner_label} before continuing.")

        step_name = step_data.get("name")
        if step_name:
            prompts.append(f"Confirm '{step_name}' is complete with the user.")

        if not prompts:
            return []

        return ["> ➡️ " + " ".join(prompts), ""]

    def _build_role_controller_block(
        self,
        role_names: list[str],
        context_requirements: list[str],
        profile: ProfileType,
    ) -> list[str]:
        if not role_names:
            return []

        heading = {
            "walkthrough": "Role Controller",
            "reference": "Controller Notes",
            "brief": "Quick Controller",
        }[profile]

        lines = [f"## {heading}", ""]
        lines.append(f"**Roles Active:** {', '.join(role_names)}")
        lines.append("")

        if context_requirements:
            lines.append("> ➡️ Gather or confirm these before acting:")
            for item in context_requirements:
                lines.append(f"> - {item}")
            lines.append("")

        if profile == "walkthrough":
            lines.extend(
                [
                    "1. Ask the user which role should take point first.",
                    "2. Before switching roles, summarize current findings.",
                    "3. Escalate any blockers immediately with a chat ping.",
                    "",
                ]
            )
        elif profile == "brief":
            lines.extend(
                [
                    "- Stay high level; only dive when asked.",
                    "- Remind the user which role is speaking when tone changes.",
                    "",
                ]
            )

        return lines

    def assemble_web_prompt_for_loop(
        self, playbook_id: str, profile: ProfileType = "reference"
    ) -> str:
        """Assemble complete web prompt for a specific loop.

        This creates a monolithic prompt suitable for web agent simulation,
        including the Showrunner expertise, playbook procedures, and all
        participating role expertises.

        Args:
            playbook_id: ID of the playbook (e.g., 'lore_deepening')

        Returns:
            Complete assembled markdown prompt

        Raises:
            CompilationError: If assembly fails
        """
        profile = self._normalize_profile(profile)

        playbook = self.primitives.get(f"playbook:{playbook_id}")
        if not playbook:
            raise CompilationError(f"Playbook not found: id='{playbook_id}'")

        data = playbook.metadata
        playbook_name = data.get("playbook_name", playbook_id)

        # Get RACI from by_loop.md
        try:
            raci_roles = self._parse_raci_from_markdown(playbook_name)
        except CompilationError as e:
            logger.error(f"Compilation Error: {e}")
            raci_roles = {"R": [], "A": [], "C": [], "I": []}
            if "raci" in data:
                for role_type_key, role_type_val in [
                    ("responsible", "R"),
                    ("accountable", "A"),
                    ("consulted", "C"),
                    ("informed", "I"),
                ]:
                    if role_type_key in data["raci"]:
                        for item in data["raci"][role_type_key]:
                            if isinstance(item, dict) and "role" in item:
                                adapter_id = self._get_adapter_id_from_ref(item["role"])
                                if adapter_id:
                                    raci_roles[role_type_val].append(adapter_id)

        # Collect all unique roles
        all_roles = set()
        for role_list in raci_roles.values():
            all_roles.update(role_list)

        sections = []

        # Header
        sections.append(f"# QuestFoundry Web Agent Prompt: {playbook_name}")
        sections.append("")
        sections.append(
            "**Purpose:** This prompt enables a web agent (LLM) to simulate "
            "the entire QuestFoundry studio executing the "
            f"{playbook_name} loop."
        )
        sections.append("")
        controller = self._build_loop_controller_block(
            playbook_name, data, raci_roles, profile
        )
        if controller:
            sections.extend(controller)
            sections.append("")
        sections.append("**Target:** GPT-4+, Claude Sonnet 3.5+, Gemini 1.5 Pro+")
        sections.append("")
        sections.append("---")
        sections.append("")

        # Overview
        sections.append("## Loop Overview")
        sections.append("")
        sections.append(f"**Loop:** {playbook_name}")
        if "category" in data:
            sections.append(f"**Category:** {data['category']}")
        if "purpose" in data:
            sections.append(f"**Purpose:** {data['purpose']}")
        if "outcome" in data:
            sections.append(f"**Expected Outcome:** {data['outcome']}")
        sections.append("")
        sections.append("---")
        sections.append("")

        # RACI Matrix
        sections.append("## RACI Matrix")
        sections.append("")
        if raci_roles["R"]:
            sections.append(f"**Responsible:** {', '.join(raci_roles['R'])}")
        if raci_roles["A"]:
            sections.append(f"**Accountable:** {', '.join(raci_roles['A'])}")
        if raci_roles["C"]:
            sections.append(f"**Consulted:** {', '.join(raci_roles['C'])}")
        if raci_roles["I"]:
            sections.append(f"**Informed:** {', '.join(raci_roles['I'])}")
        sections.append("")
        sections.append("---")
        sections.append("")

        include_role_details = profile != "brief"
        if include_role_details:
            sections.append("## Showrunner Expertise")
            sections.append("")
            sections.append(
                "*The Showrunner orchestrates the loop and coordinates all roles.*"
            )
            sections.append("")
            try:
                showrunner_orchestration = self.resolver.resolve_reference(
                    "@expertise:showrunner_orchestration", inline_content=True
                )
                sections.append(showrunner_orchestration)
            except CompilationError as e:
                logger.exception(f"Compilation Error: {e}", stack_info=True)
                sections.append("<!-- Showrunner expertise not found -->")
            sections.append("")
            sections.append("---")
            sections.append("")

            sections.append("## Role Expertises")
            sections.append("")
            for role_id in sorted(all_roles):
                if role_id == "showrunner":
                    continue  # Already included above

                adapter = self.primitives.get(f"adapter:{role_id}")
                if not adapter:
                    continue

                adapter_data = adapter.metadata
                role_name = adapter_data.get("role_name", role_id)

                sections.append(f"### {role_name}")
                sections.append("")

                if "mission" in adapter_data:
                    sections.append(f"**Mission:** {adapter_data['mission']}")
                    sections.append("")

                if "expertise" in adapter_data:
                    try:
                        expertise_content = self.resolver.resolve_reference(
                            adapter_data["expertise"], inline_content=True
                        )
                        sections.append(expertise_content)
                        sections.append("")
                    except CompilationError as e:
                        logger.error(f"Compilation Error: {e}")
                        sections.append("<!-- Expertise not found -->")
                        sections.append("")

            sections.append("---")
            sections.append("")
        else:
            sections.append("## Key Roles")
            sections.append("")
            for role_id in sorted(all_roles):
                adapter = self.primitives.get(f"adapter:{role_id}")
                if not adapter:
                    continue
                role_name = adapter.metadata.get("role_name", role_id)
                category = self._get_role_category(role_name)
                sections.append(f"- {role_name} — {category}")
            sections.append("")
            sections.append("---")
            sections.append("")

        # Primary Procedures
        if "procedures" in data and "primary" in data["procedures"]:
            sections.append("## Primary Procedures")
            sections.append("")
            sections.append(
                f"*These are the core procedures for the {playbook_name} loop.*"
            )
            sections.append("")

            for proc_ref in data["procedures"]["primary"]:
                if profile == "brief":
                    label = self._format_reference_label(proc_ref)
                    sections.append(f"- {label}")
                else:
                    try:
                        proc_content = self.resolver.resolve_reference(
                            proc_ref, inline_content=True
                        )
                        sections.append(proc_content)
                        sections.append("")
                    except CompilationError as e:
                        logger.error(f"Compilation Error: {e}")
                        sections.append(f"<!-- Error loading procedure {proc_ref} -->")
                        sections.append("")

            if profile == "brief":
                sections.append("")
            sections.append("---")
            sections.append("")

        # Supporting Procedures
        if "procedures" in data and "supporting" in data["procedures"]:
            sections.append("## Supporting Procedures")
            sections.append("")
            for proc_ref in data["procedures"]["supporting"]:
                if profile == "brief":
                    sections.append(f"- {self._format_reference_label(proc_ref)}")
                else:
                    try:
                        proc_content = self.resolver.resolve_reference(
                            proc_ref, inline_content=True
                        )
                        sections.append(proc_content)
                        sections.append("")
                    except CompilationError as e:
                        logger.error(f"Compilation Error: {e}")
                        sections.append(f"<!-- Error loading procedure {proc_ref} -->")
                        sections.append("")

            if profile == "brief":
                sections.append("")
            sections.append("---")
            sections.append("")

        # Safety Protocols
        sections.append("## Safety & Validation Protocols")
        sections.append("")

        # Add validation reminder from playbook
        if (
            "validation_requirements" in data
            and "reference" in data["validation_requirements"]
        ):
            try:
                validation_ref = data["validation_requirements"]["reference"]
                validation_content = self.resolver.resolve_reference(
                    validation_ref, inline_content=True
                )
                sections.append(validation_content)
                sections.append("")
            except CompilationError as e:
                logger.error(f"Compilation Error: {e}")
                sections.append(f"<!-- Error loading validation requirements: {e} -->")
                sections.append("")

        snippet_refs = [
            "@snippet:spoiler_hygiene_check",
            "@snippet:pn_safety_warning",
        ]
        snippet_refs.extend(self._collect_role_safety_snippets(all_roles))

        deduped = list(dict.fromkeys(ref for ref in snippet_refs if ref))

        for snippet_ref in deduped:
            if profile == "brief":
                sections.append(f"- {self._format_reference_label(snippet_ref)}")
            else:
                try:
                    snippet_content = self.resolver.resolve_reference(
                        snippet_ref, inline_content=True
                    )
                    sections.append(snippet_content)
                    sections.append("")
                except CompilationError as e:
                    logger.error(f"Compilation Error: {e}")
                    pass

        if profile == "brief" and deduped:
            sections.append("")

        sections.append("---")
        sections.append("")

        # Procedure Steps
        if "procedure_steps" in data:
            sections.append("## Execution Steps")
            sections.append("")
            for step_data in data["procedure_steps"]:
                step_num = step_data.get("step", "?")
                step_name = step_data.get("name", "Unnamed Step")
                owner = step_data.get("owner", "?")
                action = step_data.get("action", "")

                if profile == "brief":
                    summary = f"- Step {step_num}: {step_name} (Owner: {owner})"
                    sections.append(summary)
                else:
                    sections.append(f"### Step {step_num}: {step_name}")
                    sections.append("")
                    sections.append(f"**Owner:** {owner}")
                    if action:
                        sections.append(f"**Action:** {action}")
                    sections.append("")
                    if profile == "walkthrough":
                        callouts = self._build_step_callouts(step_data)
                        sections.extend(callouts)

            sections.append("---")
            sections.append("")

        # Artifacts
        if "artifacts" in data:
            sections.append("## Artifacts")
            sections.append("")
            sections.append("*Schemas referenced by this loop:*")
            sections.append("")
            for artifact in data["artifacts"]:
                sections.append(f"- `{artifact}`")
            sections.append("")
            sections.append("---")
            sections.append("")

        # Footer
        sections.append("## Usage Instructions")
        sections.append("")
        if profile == "walkthrough":
            sections.append(
                "1. Confirm activation triggers and gather inputs listed in the"
                " controller section."
            )
            sections.append("2. Narrate each execution step and wait for user cues.")
            sections.append(
                "3. Pause at every ➡️ callout to collect data or confirm decisions."
            )
            sections.append(
                "4. Summarize loop outcomes and deliverables before closing the TU."
            )
        elif profile == "brief":
            sections.append("- Run this loop quickly: confirm scope → act → report.")
            sections.append(
                "- Reference full procedures only if the user asks for detail."
            )
            sections.append("- Always restate risks and safety notes at handoff.")
        else:
            sections.append(
                "1. You are simulating the entire QuestFoundry studio in this chat."
            )
            sections.append(
                "2. The Customer will provide inputs to start the "
                f"{playbook_name} loop."
            )
            sections.append(
                "3. Act as the Showrunner to coordinate all roles (LW, SR, etc.)."
            )
            sections.append(
                "4. Follow the procedures and steps outlined above in sequence."
            )
            sections.append(
                "5. Produce artifacts matching the referenced schemas where applicable."
            )
            sections.append("6. Respect all safety protocols and quality bars.")
        sections.append("")

        return "\n".join(sections)

    def assemble_web_prompt_for_roles(
        self,
        role_ids: list[str],
        standalone: bool = False,
        profile: ProfileType = "reference",
    ) -> str:
        """Assemble complete web prompt for specific roles.

        Args:
            role_ids: Adapter IDs (e.g., `['lore_weaver', 'plotwright']`).
            standalone: Include playbook procedures referenced by the roles.
            profile: Interaction profile to shape the controller output.

        Returns:
            Markdown prompt ready for delivery to a chat agent.

        Raises:
            CompilationError: If any required primitive cannot be resolved.
        """

        profile = self._normalize_profile(profile)
        sections: list[str] = []

        role_names: list[str] = []
        adapter_entries: list[tuple[str, BehaviorPrimitive | None]] = []
        aggregated_context: list[str] = []
        loop_participation: dict[str, list[str]] = {}
        for role_id in role_ids:
            adapter = self.primitives.get(f"adapter:{role_id}")
            adapter_entries.append((role_id, adapter))
            if adapter:
                metadata = adapter.metadata
                role_names.append(metadata.get("role_name", role_id))
                aggregated_context.extend(
                    metadata.get("context_requirements", []) or []
                )
                for loop_entry in metadata.get("loops", []) or []:
                    ref = loop_entry.get("playbook")
                    if isinstance(ref, str) and ":" in ref:
                        loop_id = ref.split(":", 1)[1]
                        loop_participation.setdefault(loop_id, []).append(role_id)
            else:
                role_names.append(role_id)

        sections.append(f"# QuestFoundry Web Agent Prompt: {', '.join(role_names)}")
        sections.append("")
        sections.append(
            "**Purpose:** This prompt enables a web agent (LLM) to act as "
            f"the following QuestFoundry roles: {', '.join(role_names)}."
        )
        sections.append("")
        sections.append("**Target:** GPT-4+, Claude Sonnet 3.5+, Gemini 1.5 Pro+")
        sections.append("")
        controller = self._build_role_controller_block(
            role_names, aggregated_context, profile
        )
        if controller:
            sections.extend(controller)
            sections.append("")
        sections.append("---")
        sections.append("")

        if profile == "brief":
            sections.append("## Role Summaries")
            sections.append("")
            for role_id, adapter in adapter_entries:
                if not adapter:
                    sections.append(f"- {role_id}: unavailable")
                    continue
                adapter_data = adapter.metadata
                role_name = adapter_data.get("role_name", role_id)
                mission = adapter_data.get("mission", "See expertise notes.")
                sections.append(f"- **{role_name}:** {mission}")
            sections.append("")
            sections.append("---")
            sections.append("")
        else:
            seen_snippets: set[str] = set()
            for role_id, adapter in adapter_entries:
                if not adapter:
                    sections.append(f"<!-- Adapter not found: {role_id} -->")
                    sections.append("")
                    continue

                adapter_data = adapter.metadata
                role_name = adapter_data.get("role_name", role_id)

                sections.append(f"## {role_name}")
                sections.append("")

                if "mission" in adapter_data:
                    sections.append(f"**Mission:** {adapter_data['mission']}")
                    sections.append("")

                if "expertise" in adapter_data:
                    sections.append("### Expertise")
                    sections.append("")
                    try:
                        expertise_content = self.resolver.resolve_reference(
                            adapter_data["expertise"], inline_content=True
                        )
                        sections.append(expertise_content)
                        sections.append("")
                    except CompilationError as e:
                        logger.error(f"Compilation Error: {e}")
                        sections.append("<!-- Expertise not found -->")
                        sections.append("")

                if "procedures" in adapter_data:
                    if "primary" in adapter_data["procedures"]:
                        sections.append("### Primary Procedures")
                        sections.append("")
                        for proc_ref in adapter_data["procedures"]["primary"]:
                            try:
                                proc_content = self.resolver.resolve_reference(
                                    proc_ref, inline_content=True
                                )
                                sections.append(proc_content)
                                sections.append("")
                            except CompilationError as e:
                                logger.error(f"Compilation Error: {e}")
                                sections.append(
                                    f"<!-- Error loading procedure {proc_ref} -->"
                                )
                                sections.append("")
                    if "supporting" in adapter_data["procedures"]:
                        sections.append("### Supporting Procedures")
                        sections.append("")
                        for proc_ref in adapter_data["procedures"]["supporting"]:
                            try:
                                proc_content = self.resolver.resolve_reference(
                                    proc_ref, inline_content=True
                                )
                                sections.append(proc_content)
                                sections.append("")
                            except CompilationError as e:
                                logger.error(f"Compilation Error: {e}")
                                sections.append(
                                    f"<!-- Error loading procedure {proc_ref} -->"
                                )
                                sections.append("")

                if "protocol_intents" in adapter_data:
                    sections.append("### Protocol Intents")
                    sections.append("")
                    if "receives" in adapter_data["protocol_intents"]:
                        sections.append("**Receives:**")
                        for intent in adapter_data["protocol_intents"]["receives"]:
                            sections.append(f"- {intent}")
                        sections.append("")
                    if "sends" in adapter_data["protocol_intents"]:
                        sections.append("**Sends:**")
                        for intent in adapter_data["protocol_intents"]["sends"]:
                            sections.append(f"- {intent}")
                        sections.append("")

                if "loops" in adapter_data:
                    sections.append("### Loop Participation")
                    sections.append("")
                    for loop_info in adapter_data["loops"]:
                        playbook_ref = loop_info.get("playbook")
                        raci = loop_info.get("raci", "?")
                        desc = loop_info.get("description", "")
                        playbook_id = None
                        if isinstance(playbook_ref, str):
                            playbook_id = playbook_ref.split(":", 1)[1]
                        sections.append(f"- **{playbook_id}** ({raci})")
                        if desc:
                            sections.append(f": {desc}")
                        sections.append("")

                        if standalone and playbook_id:
                            playbook = self.primitives.get(f"playbook:{playbook_id}")
                            if playbook:
                                playbook_data = playbook.metadata
                                if (
                                    "procedures" in playbook_data
                                    and "primary" in playbook_data["procedures"]
                                ):
                                    sections.append(f"*Procedures from {playbook_id}:*")
                                    sections.append("")
                                    for proc_ref in playbook_data["procedures"][
                                        "primary"
                                    ]:
                                        try:
                                            proc_content = (
                                                self.resolver.resolve_reference(
                                                    proc_ref, inline_content=True
                                                )
                                            )
                                            sections.append(proc_content)
                                            sections.append("")
                                        except CompilationError as e:
                                            logger.error(f"Compilation Error: {e}")
                                            pass

                if "safety_protocols" in adapter_data:
                    sections.append("### Safety Protocols")
                    sections.append("")
                    for snippet_ref in adapter_data["safety_protocols"]:
                        if snippet_ref in seen_snippets:
                            continue
                        seen_snippets.add(snippet_ref)
                        try:
                            snippet_content = self.resolver.resolve_reference(
                                snippet_ref, inline_content=True
                            )
                            sections.append(snippet_content)
                            sections.append("")
                        except CompilationError as e:
                            logger.error(f"Compilation Error: {e}")
                            pass

                sections.append("---")
                sections.append("")

        shared_loops = {
            loop_id: members
            for loop_id, members in loop_participation.items()
            if len(members) > 1
        }
        if shared_loops:
            sections.append("## Cross-Role Coordination")
            sections.append("")
            for loop_id, members in sorted(shared_loops.items()):
                playbook = self.primitives.get(f"playbook:{loop_id}")
                loop_name = loop_id
                if playbook:
                    loop_name = playbook.metadata.get("playbook_name", loop_id)
                member_names = []
                for member in members:
                    adapter = self.primitives.get(f"adapter:{member}")
                    member_names.append(
                        adapter.metadata.get("role_name", member) if adapter else member
                    )
                sections.append(
                    f"- **{loop_name}:** coordinate {', '.join(member_names)}"
                )
            sections.append("")
            sections.append("---")
            sections.append("")

        sections.append("## Usage Instructions")
        sections.append("")
        if profile == "walkthrough":
            sections.append("1. Act in-order for each role, pausing for user input.")
            sections.append(
                "2. Use the controller callouts to request live context "
                "before handoffs."
            )
            sections.append(
                "3. Highlight shared loops when coordinating multiple roles."
            )
        elif profile == "brief":
            sections.append("- Summarize missions and ask when to dive deeper.")
            sections.append("- Keep replies short but enforce safety reminders.")
        else:
            sections.append(
                "1. You are acting as the specified QuestFoundry role(s) in this chat."
            )
            sections.append(
                "2. Follow the procedures and expertise guidelines outlined above."
            )
            sections.append("3. Respect all safety protocols and quality bars.")
            sections.append(
                "4. Coordinate with other roles as indicated in loop participation."
            )
        sections.append("")

        return "\n".join(sections)
