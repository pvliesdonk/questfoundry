"""
Knowledge Tools - Allow agents to consult the Cartridge (spec) for knowledge.

These tools implement the "Read-Only Cartridge" pattern from the mesh architecture:
- Agents don't hardcode behavior
- They consult the spec via tools to know how to act
- The spec (Layer 0-5) is the agent's knowledge base
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _find_spec_root() -> Path | None:
    """Find the spec directory by searching upward from this file."""
    current = Path(__file__).parent
    for _ in range(10):
        spec_candidate = current / "spec"
        if spec_candidate.exists() and (spec_candidate / "05-definitions").exists():
            return spec_candidate
        current = current.parent
    return None


SPEC_ROOT = _find_spec_root()


class ConsultPlaybook(BaseTool):
    """
    Consult a loop playbook to understand workflow steps.

    The Showrunner uses this to know which roles to wake and
    what sequence of work to coordinate.
    """

    name: str = "consult_playbook"
    description: str = (
        "Look up a loop/playbook definition to understand its purpose, "
        "participating roles, quality gates, and workflow. "
        "Input: loop_id (e.g., 'hook_harvest', 'story_spark')"
    )

    def _run(self, loop_id: str) -> str:
        """Look up playbook/loop definition."""
        if not SPEC_ROOT:
            return "Error: Spec root not found"

        try:
            # Try loop definition first
            loop_path = SPEC_ROOT / "05-definitions" / "loops" / f"{loop_id}.yaml"
            if loop_path.exists():
                with open(loop_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return self._format_loop(data)

            # Try playbook (Layer 0)
            playbook_path = SPEC_ROOT / "00-north-star" / "PLAYBOOKS" / f"{loop_id}.md"
            if playbook_path.exists():
                return playbook_path.read_text(encoding="utf-8")

            # Try loop guide (Layer 0)
            loop_guide_path = SPEC_ROOT / "00-north-star" / "LOOPS" / f"{loop_id}.md"
            if loop_guide_path.exists():
                return loop_guide_path.read_text(encoding="utf-8")

            return f"Loop/playbook '{loop_id}' not found"

        except Exception as e:
            logger.error(f"Error consulting playbook {loop_id}: {e}")
            return f"Error: {e}"

    def _format_loop(self, data: dict[str, Any]) -> str:
        """Format loop definition for agent consumption."""
        lines = [
            f"# Loop: {data.get('metadata', {}).get('name', data.get('id', 'Unknown'))}",
            "",
            f"**Type**: {data.get('metadata', {}).get('type', 'Unknown')}",
            "",
            "## Description",
            data.get("metadata", {}).get("description", "No description"),
            "",
        ]

        # Topology info
        topology = data.get("topology", {})
        if topology:
            lines.append("## Workflow Nodes")
            for node in topology.get("nodes", []):
                if isinstance(node, dict):
                    role = node.get("role_id", "Unknown")
                    node_id = node.get("node_id", role)
                    desc = node.get("description", "")
                    lines.append(f"- **{node_id}** ({role}): {desc}")
            lines.append("")

        # Quality gates
        gates = data.get("gates", {})
        if gates:
            bars = gates.get("quality_bars", [])
            if bars:
                lines.append("## Quality Gates")
                lines.append(f"Required bars: {', '.join(bars)}")
                lines.append("")

        # Handoffs
        handoffs = data.get("handoffs", [])
        if handoffs:
            lines.append("## Handoffs to Other Loops")
            for h in handoffs:
                target = h.get("target", "")
                condition = h.get("condition", "")
                lines.append(f"- → **{target}**: {condition}")
            lines.append("")

        return "\n".join(lines)


class ConsultQualityGate(BaseTool):
    """
    Consult a quality gate definition to understand validation criteria.

    The Gatekeeper uses this to know what to validate and how.
    """

    name: str = "consult_quality_gate"
    description: str = (
        "Look up a quality gate/bar definition to understand its validation criteria, "
        "pass conditions, and remediation guidance. "
        "Input: bar_name (e.g., 'integrity', 'reachability', 'style')"
    )

    def _run(self, bar_name: str) -> str:
        """Look up quality gate definition."""
        if not SPEC_ROOT:
            return "Error: Spec root not found"

        try:
            # Normalize bar name
            normalized = bar_name.lower().replace(" ", "_")

            # Try quality gate definition
            gate_path = SPEC_ROOT / "05-definitions" / "quality_gates" / f"{normalized}.yaml"
            if gate_path.exists():
                with open(gate_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return self._format_gate(data)

            # Try quality bars overview (Layer 0)
            bars_path = SPEC_ROOT / "00-north-star" / "QUALITY_BARS.md"
            if bars_path.exists():
                content = bars_path.read_text(encoding="utf-8")
                # Extract relevant section
                return f"Quality bar '{bar_name}' definition:\n\n{content}"

            return f"Quality gate '{bar_name}' not found"

        except Exception as e:
            logger.error(f"Error consulting quality gate {bar_name}: {e}")
            return f"Error: {e}"

    def _format_gate(self, data: dict[str, Any]) -> str:
        """Format quality gate for agent consumption."""
        bar_def = data.get("bar_definition", {})
        lines = [
            f"# Quality Bar: {data.get('bar_name', 'Unknown')}",
            "",
            "## What It Means",
            bar_def.get("what_it_means", "No definition"),
            "",
        ]

        # Validation checks
        checks = data.get("validation_checks", [])
        if checks:
            lines.append("## Validation Checks")
            for check in checks:
                check_id = check.get("check_id", "")
                severity = check.get("severity", "")
                fail_msg = check.get("failure_message", "")
                lines.append(f"- **{check_id}** [{severity}]: {fail_msg}")
            lines.append("")

        # Status thresholds
        thresholds = data.get("status_thresholds", {})
        if thresholds:
            lines.append("## Pass Thresholds")
            green = thresholds.get("green", {})
            lines.append(
                f"- Green: max {green.get('critical_failures', 0)} critical, {green.get('warning_failures', 0)} warnings"
            )
            lines.append("")

        # Remediation
        remediation = data.get("remediation", {})
        if remediation:
            common = remediation.get("common_failures", [])
            if common:
                lines.append("## Common Failures & Fixes")
                for failure in common:
                    issue = failure.get("issue", "")
                    fix = failure.get("fix", "")
                    lines.append(f"- **Issue**: {issue}")
                    lines.append(f"  **Fix**: {fix}")
                lines.append("")

        return "\n".join(lines)


class ConsultProtocol(BaseTool):
    """
    Consult the protocol specification for valid message formats.

    Any role can use this to look up valid intents and envelope structure.
    """

    name: str = "consult_protocol"
    description: str = (
        "Look up protocol information: valid intents, envelope structure, "
        "or message flow patterns. "
        "Input: query (e.g., 'tu.open', 'envelope', 'intents')"
    )

    def _run(self, query: str) -> str:
        """Look up protocol information."""
        if not SPEC_ROOT:
            return "Error: Spec root not found"

        try:
            query_lower = query.lower()

            # Intent lookup
            if "." in query or query_lower in ["intents", "intent"]:
                intents_path = SPEC_ROOT / "04-protocol" / "INTENTS.md"
                if intents_path.exists():
                    content = intents_path.read_text(encoding="utf-8")
                    if "." in query:
                        # Extract specific intent section
                        return f"Protocol intent '{query}':\n\n{content}"
                    return content

            # Envelope lookup
            if query_lower in ["envelope", "envelopes"]:
                envelope_path = SPEC_ROOT / "04-protocol" / "ENVELOPE.md"
                if envelope_path.exists():
                    return envelope_path.read_text(encoding="utf-8")

            # Lifecycle lookup
            if "lifecycle" in query_lower:
                lifecycles_dir = SPEC_ROOT / "04-protocol" / "LIFECYCLES"
                if lifecycles_dir.exists():
                    files = list(lifecycles_dir.glob("*.md"))
                    if files:
                        combined = ["# Protocol Lifecycles\n"]
                        for f in files:
                            combined.append(f"## {f.stem}\n")
                            combined.append(f.read_text(encoding="utf-8"))
                            combined.append("\n---\n")
                        return "\n".join(combined)

            # Flow lookup
            if "flow" in query_lower:
                flows_dir = SPEC_ROOT / "04-protocol" / "FLOWS"
                if flows_dir.exists():
                    files = list(flows_dir.glob("*.md"))
                    if files:
                        combined = ["# Protocol Flows\n"]
                        for f in files[:5]:  # Limit to 5 flows
                            combined.append(f"## {f.stem}\n")
                            combined.append(f.read_text(encoding="utf-8")[:2000])
                            combined.append("\n---\n")
                        return "\n".join(combined)

            # Example messages
            if "example" in query_lower:
                examples_dir = SPEC_ROOT / "04-protocol" / "EXAMPLES"
                if examples_dir.exists():
                    files = list(examples_dir.glob("*.json"))
                    if files:
                        combined = ["# Protocol Message Examples\n"]
                        for f in files[:5]:
                            combined.append(f"## {f.stem}\n```json\n")
                            combined.append(f.read_text(encoding="utf-8"))
                            combined.append("\n```\n")
                        return "\n".join(combined)

            return f"Protocol query '{query}' not found. Try: 'intents', 'envelope', 'lifecycle', 'flow', 'example'"

        except Exception as e:
            logger.error(f"Error consulting protocol {query}: {e}")
            return f"Error: {e}"


class ConsultRoleCharter(BaseTool):
    """
    Consult a role's charter to understand its responsibilities.

    Any role can use this to understand another role's domain and capabilities.
    """

    name: str = "consult_role_charter"
    description: str = (
        "Look up a role's charter to understand its mandate, capabilities, "
        "and what it's responsible for. "
        "Input: role_id (e.g., 'plotwright', 'gatekeeper', 'scene_smith')"
    )

    def _run(self, role_id: str) -> str:
        """Look up role charter."""
        if not SPEC_ROOT:
            return "Error: Spec root not found"

        try:
            # Normalize role_id
            normalized = role_id.lower().replace(" ", "_")

            # Try role definition (Layer 5)
            role_path = SPEC_ROOT / "05-definitions" / "roles" / f"{normalized}.yaml"
            if role_path.exists():
                with open(role_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return self._format_role(data)

            # Try charter (Layer 1)
            charter_path = SPEC_ROOT / "01-roles" / "charters" / f"{normalized}.md"
            if charter_path.exists():
                return charter_path.read_text(encoding="utf-8")

            return f"Role '{role_id}' not found"

        except Exception as e:
            logger.error(f"Error consulting role charter {role_id}: {e}")
            return f"Error: {e}"

    def _format_role(self, data: dict[str, Any]) -> str:
        """Format role definition for agent consumption."""
        identity = data.get("identity", {})
        prompt_content = data.get("prompt_content", {})

        lines = [
            f"# Role: {identity.get('name', data.get('id', 'Unknown'))}",
            f"**Abbreviation**: {identity.get('abbreviation', 'N/A')}",
            f"**Type**: {identity.get('role_type', 'reasoning_agent')}",
            f"**Dormancy**: {identity.get('dormancy_policy', 'default_on')}",
            "",
            "## Core Mandate",
            prompt_content.get("core_mandate", "No mandate defined"),
            "",
        ]

        # Operating principles
        principles = prompt_content.get("operating_principles", [])
        if principles:
            lines.append("## Operating Principles")
            for p in principles:
                name = p.get("name", "")
                desc = p.get("description", "")
                lines.append(f"- **{name}**: {desc}")
            lines.append("")

        # Quality bars owned
        bars = prompt_content.get("quality_bars_owned", [])
        if bars:
            lines.append("## Quality Bars Owned")
            lines.append(", ".join(bars))
            lines.append("")

        # Protocol intents
        protocol = data.get("protocol", {})
        intents = protocol.get("intents", {})
        if intents:
            can_send = intents.get("can_send", [])
            can_receive = intents.get("can_receive", [])
            lines.append("## Protocol Capabilities")
            lines.append(
                f"**Can Send**: {', '.join(can_send[:10])}{'...' if len(can_send) > 10 else ''}"
            )
            lines.append(
                f"**Can Receive**: {', '.join(can_receive[:10])}{'...' if len(can_receive) > 10 else ''}"
            )
            lines.append("")

        return "\n".join(lines)


class ConsultGlossary(BaseTool):
    """
    Consult the glossary for term definitions.

    Any role can use this to look up terminology and conventions.
    """

    name: str = "consult_glossary"
    description: str = (
        "Look up terminology, artifact types, or conventions. "
        "Input: term or 'all' for full glossary"
    )

    def _run(self, term: str) -> str:
        """Look up glossary term."""
        if not SPEC_ROOT:
            return "Error: Spec root not found"

        try:
            glossary_path = SPEC_ROOT / "02-dictionary" / "glossary.md"
            if glossary_path.exists():
                content = glossary_path.read_text(encoding="utf-8")
                if term.lower() == "all":
                    return content
                # Simple search for term
                return f"Glossary lookup for '{term}':\n\n{content}"

            # Try conventions
            conventions_dir = SPEC_ROOT / "02-dictionary" / "conventions"
            if conventions_dir.exists():
                files = list(conventions_dir.glob("*.md"))
                if files:
                    combined = [f"# Conventions (searching for '{term}')\n"]
                    for f in files:
                        content = f.read_text(encoding="utf-8")
                        if term.lower() in content.lower():
                            combined.append(f"## {f.stem}\n")
                            combined.append(content[:2000])
                            combined.append("\n---\n")
                    if len(combined) > 1:
                        return "\n".join(combined)

            return f"Term '{term}' not found in glossary or conventions"

        except Exception as e:
            logger.error(f"Error consulting glossary {term}: {e}")
            return f"Error: {e}"


class ConsultSchema(BaseTool):
    """
    Consult artifact schema definitions to understand structure and validation requirements.

    Use this when validation fails or before creating artifacts to ensure compliance.
    """

    name: str = "consult_schema"
    description: str = (
        "Look up artifact schema definition to understand required/optional fields, "
        "validation patterns, and see examples. "
        "Input: artifact_type (e.g., 'section', 'section_brief', 'hook_card')"
    )

    def _run(self, artifact_type: str) -> str:
        """Look up schema definition for an artifact type."""
        if not SPEC_ROOT:
            return "Error: Spec root not found"

        try:
            # Normalize artifact_type (add .schema.json if not present)
            schema_filename = artifact_type
            if not schema_filename.endswith(".schema.json"):
                schema_filename = f"{artifact_type}.schema.json"

            # Try loading schema from 03-schemas directory
            schema_path = SPEC_ROOT / "03-schemas" / schema_filename
            if not schema_path.exists():
                # List available schemas to help agent
                schemas_dir = SPEC_ROOT / "03-schemas"
                if schemas_dir.exists():
                    available = [
                        f.stem.replace(".schema", "")
                        for f in schemas_dir.glob("*.schema.json")
                    ]
                    return (
                        f"Schema '{artifact_type}' not found.\n\n"
                        f"Available artifact schemas: {', '.join(sorted(available[:20]))}"
                        + ("..." if len(available) > 20 else "")
                    )
                return f"Schema '{artifact_type}' not found (schemas directory missing)"

            # Load and parse schema
            with open(schema_path, encoding="utf-8") as f:
                import json

                schema = json.load(f)

            # Format schema for agent consumption
            return self._format_schema(artifact_type.replace(".schema", ""), schema)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema {artifact_type}: {e}")
            return f"Error: Invalid JSON in schema {artifact_type}"
        except Exception as e:
            logger.error(f"Error consulting schema {artifact_type}: {e}")
            return f"Error: {e}"

    def _format_schema(self, artifact_type: str, schema: dict[str, Any]) -> str:
        """Format JSON schema as readable markdown for agents."""
        lines = [
            f"# Artifact Schema: {artifact_type}",
            "",
            f"**Title**: {schema.get('title', artifact_type)}",
            f"**Description**: {schema.get('description', 'No description')}",
            "",
        ]

        # Extract properties and required fields
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        if not properties:
            lines.append("⚠️  No properties defined in schema")
            return "\n".join(lines)

        # Required fields
        required_props = {k: v for k, v in properties.items() if k in required_fields}
        if required_props:
            lines.append("## Required Fields")
            lines.append("")
            for field_name, field_schema in required_props.items():
                lines.extend(self._format_field(field_name, field_schema, required=True))
            lines.append("")

        # Optional fields
        optional_props = {k: v for k, v in properties.items() if k not in required_fields}
        if optional_props:
            lines.append("## Optional Fields")
            lines.append("")
            for field_name, field_schema in optional_props.items():
                lines.extend(self._format_field(field_name, field_schema, required=False))
            lines.append("")

        # Add minimal valid example hint
        lines.append("## Creating This Artifact")
        lines.append("")
        lines.append("Minimal valid example must include:")
        for req_field in required_fields:
            field_type = properties.get(req_field, {}).get("type", "any")
            lines.append(f"- `{req_field}` ({field_type})")
        lines.append("")
        lines.append("Use write_hot_sot or specific typed tool to create this artifact.")
        lines.append("")

        return "\n".join(lines)

    def _format_field(self, field_name: str, field_schema: dict[str, Any], required: bool) -> list[str]:
        """Format a single field definition."""
        lines = []

        # Field header
        field_type = field_schema.get("type", "any")
        description = field_schema.get("description", "")

        header = f"### `{field_name}`"
        if required:
            header += " ✅ REQUIRED"

        lines.append(header)
        lines.append(f"- **Type**: `{field_type}`")

        if description:
            lines.append(f"- **Description**: {description}")

        # Type-specific constraints
        if field_type == "string":
            if "pattern" in field_schema:
                pattern = field_schema["pattern"]
                lines.append(f"- **Pattern**: `{pattern}`")
                # Add human-readable pattern explanation
                pattern_hint = self._explain_pattern(pattern)
                if pattern_hint:
                    lines.append(f"  - {pattern_hint}")

            if "enum" in field_schema:
                enum_values = field_schema["enum"]
                lines.append(f"- **Allowed values**: {', '.join(map(str, enum_values))}")

            if "minLength" in field_schema or "maxLength" in field_schema:
                min_len = field_schema.get("minLength", 0)
                max_len = field_schema.get("maxLength", "∞")
                lines.append(f"- **Length**: {min_len} to {max_len} characters")

        elif field_type == "array":
            items = field_schema.get("items", {})
            item_type = items.get("type", "any")
            lines.append(f"- **Items**: `{item_type}`")

            if "minItems" in field_schema:
                lines.append(f"- **Min items**: {field_schema['minItems']}")
            if "maxItems" in field_schema:
                lines.append(f"- **Max items**: {field_schema['maxItems']}")

        elif field_type == "number" or field_type == "integer":
            if "minimum" in field_schema:
                lines.append(f"- **Minimum**: {field_schema['minimum']}")
            if "maximum" in field_schema:
                lines.append(f"- **Maximum**: {field_schema['maximum']}")

        # Default value
        if "default" in field_schema:
            lines.append(f"- **Default**: `{field_schema['default']}`")

        lines.append("")
        return lines

    def _explain_pattern(self, pattern: str) -> str:
        """Provide human-readable explanation of common regex patterns."""
        explanations = {
            r"^TU-\d{4}-\d{2}-\d{2}-[A-Z]{2,4}\d{2}$": "Format: TU-YYYY-MM-DD-ROLESEQ (e.g., TU-2025-12-04-PW01)",
            r"^SB-\d{3,}$": "Format: SB-### (e.g., SB-001, SB-042)",
            r"^anchor\d{3,}$": "Format: anchor### (e.g., anchor001, anchor042)",
            r"^anchor[0-9]{3,}$": "Format: anchor### (e.g., anchor001, anchor042)",
            r"^gate_[a-z_]+$": "Format: gate_name (lowercase with underscores, e.g., gate_has_key)",
        }

        for known_pattern, explanation in explanations.items():
            if pattern == known_pattern:
                return explanation

        # Generic hints
        if pattern.startswith("^") and pattern.endswith("$"):
            return "Exact match required (no extra characters)"
        if r"\d" in pattern:
            return "Must contain digits"
        if "[A-Z]" in pattern or "[a-z]" in pattern:
            return "Must contain letters"

        return ""
