"""Assemblers for composing prompts and resolving references."""

import re
from pathlib import Path

from questfoundry.compiler.types import BehaviorPrimitive, CompilationError


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
        match = self.reference_pattern.match(ref)
        if not match:
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
            except CompilationError:
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
