"""Consult tools - allow agents to look up compiled resources for guidance.

These tools implement the "menu + consult" pattern:
- SR gets a menu of available roles/loops in system prompt
- Agents use consult_* tools to get details when needed

Available tools:
- consult_role_charter: Look up role responsibilities, tools, constraints
- consult_playbook: Look up loop/workflow guidance
- consult_schema: Look up artifact schema definitions
- consult_tool: Look up tool documentation (parameters, valid values, examples)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

from questfoundry.runtime.resources import get_resource_loader

logger = logging.getLogger(__name__)


class ConsultRoleCharter(BaseTool):
    """Look up a role's charter (responsibilities, tools, constraints).

    Use this to understand what a role can do before delegating work,
    or to understand your own capabilities and constraints.
    """

    name: str = "consult_role_charter"
    description: str = (
        "Look up a role's charter to understand its archetype, mandate, "
        "available tools, and constraints. "
        "Input: role_id (e.g., 'plotwright', 'scene_smith', 'lorekeeper')"
    )

    def _run(self, role_id: str) -> str:
        """Look up role charter."""
        loader = get_resource_loader()

        # Normalize role_id
        role_id = role_id.lower().replace(" ", "_").replace("-", "_")

        # Try to load role
        role = loader.load_role(role_id)
        if role is None:
            # List available roles to help agent
            available = loader.list_roles()
            if not available:
                # Fallback to generated.roles module
                try:
                    from questfoundry.generated.roles import ALL_ROLES

                    available = list(ALL_ROLES.keys())
                except ImportError:
                    pass

            if available:
                return (
                    f"Role '{role_id}' not found.\n\n"
                    f"Available roles: {', '.join(sorted(available))}"
                )
            return f"Role '{role_id}' not found and no roles available."

        return self._format_role(role_id, role)

    def _format_role(self, role_id: str, role: dict[str, Any]) -> str:
        """Format role definition for agent consumption."""
        # Handle both dict and RoleIR object
        if hasattr(role, "model_dump"):
            role = role.model_dump()

        lines = [
            f"# Role: {role_id}",
            "",
            f"**Archetype**: {role.get('archetype', 'Unknown')}",
            f"**Abbreviation**: {role.get('abbr', role_id[:2].upper())}",
            f"**Agency**: {role.get('agency', 'medium')}",
            f"**Mandate**: {role.get('mandate', 'No mandate defined')}",
            "",
        ]

        # Tools
        tools = role.get("tools", [])
        if tools:
            lines.append("## Available Tools")
            for tool in tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "")
                else:
                    # RoleToolIR object
                    name = getattr(tool, "name", "unknown")
                    desc = getattr(tool, "description", "")
                lines.append(f"- **{name}**: {desc}")
            lines.append("")

        # Constraints
        constraints = role.get("constraints", [])
        if constraints:
            lines.append("## Constraints")
            for c in constraints:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)


class ConsultPlaybook(BaseTool):
    """Look up a loop/playbook definition to understand workflow guidance.

    Use this to understand:
    - What roles participate in a workflow
    - What the expected sequence of work is
    - What quality gates apply
    """

    name: str = "consult_playbook"
    description: str = (
        "Look up a loop/playbook definition to understand its workflow, "
        "participating roles, and quality gates. "
        "Input: loop_id (e.g., 'story_spark')"
    )

    def _run(self, loop_id: str) -> str:
        """Look up playbook/loop definition."""
        loader = get_resource_loader()

        # Normalize loop_id
        loop_id = loop_id.lower().replace(" ", "_").replace("-", "_")

        # Try to load loop from JSON files first
        loop = loader.load_loop(loop_id)
        if loop is not None:
            return self._format_loop(loop_id, loop)

        # Fallback: try generated loops module
        loop = self._get_loop_from_generated(loop_id)
        if loop is not None:
            return self._format_loop(loop_id, loop)

        # List available loops
        available = self._list_available_loops()
        if available:
            return (
                f"Loop '{loop_id}' not found.\n\n"
                f"Available loops: {', '.join(sorted(available))}"
            )
        return f"Loop '{loop_id}' not found and no loops available."

    def _get_loop_from_generated(self, loop_id: str) -> dict[str, Any] | None:
        """Get loop definition from generated loops module."""
        try:
            from questfoundry.generated.loops import ALL_LOOPS

            loop_ir = ALL_LOOPS.get(loop_id)
            if loop_ir is not None:
                # Convert LoopIR to dict for formatting
                return loop_ir.model_dump()
        except ImportError:
            logger.debug("generated.loops module not available")
        return None

    def _list_available_loops(self) -> list[str]:
        """List available loops from both JSON files and generated module."""
        available = set()

        # Check resource loader
        loader = get_resource_loader()
        available.update(loader.list_loops())

        # Add loops from generated module
        try:
            from questfoundry.generated.loops import ALL_LOOPS

            available.update(ALL_LOOPS.keys())
        except ImportError:
            pass

        return list(available)

    def _format_loop(self, loop_id: str, loop: dict[str, Any]) -> str:
        """Format loop definition for agent consumption."""
        lines = [
            f"# Loop: {loop_id}",
            "",
        ]

        # Name and trigger info
        if "name" in loop:
            lines.append(f"**Name**: {loop['name']}")
        if "trigger" in loop:
            lines.append(f"**Trigger**: {loop['trigger']}")
        if "entry_point" in loop:
            lines.append(f"**Entry Point**: {loop['entry_point']}")
        if lines[-1] != "":
            lines.append("")

        # Description
        if "description" in loop:
            lines.append("## Description")
            lines.append(loop["description"])
            lines.append("")

        # Nodes (roles/steps)
        nodes = loop.get("nodes", [])
        if nodes:
            lines.append("## Workflow Steps (Nodes)")
            for node in nodes:
                if isinstance(node, dict):
                    node_id = node.get("id", node.get("node_id", "unknown"))
                    role = node.get("role", node.get("role_id", ""))
                    timeout = node.get("timeout", 300)
                    lines.append(f"- **{node_id}** (role: {role}, timeout: {timeout}s)")
                else:
                    lines.append(f"- {node}")
            lines.append("")

        # Edges (transitions)
        edges = loop.get("edges", [])
        if edges:
            lines.append("## Transitions (Edges)")
            for edge in edges:
                if isinstance(edge, dict):
                    src = edge.get("source", "?")
                    tgt = edge.get("target", "?")
                    cond = edge.get("condition", "")
                    # Format target nicely
                    tgt_display = "END" if tgt == "__end__" else tgt
                    lines.append(f"- {src} → {tgt_display}: `{cond}`")
            lines.append("")

        # Quality gates (handle both "gates" and "quality_gates" keys)
        gates = loop.get("quality_gates", loop.get("gates", []))
        if gates:
            lines.append("## Quality Gates")
            for gate in gates:
                if isinstance(gate, dict):
                    before = gate.get("before", "")
                    role = gate.get("role", "gatekeeper")
                    bars = gate.get("bars", [])
                    blocking = gate.get("blocking", True)
                    blocking_str = "blocking" if blocking else "non-blocking"
                    lines.append(
                        f"- Before **{before}** ({role}, {blocking_str}): {', '.join(bars)}"
                    )
            lines.append("")

        return "\n".join(lines)


class ConsultSchema(BaseTool):
    """Look up artifact schema to understand required/optional fields.

    Use this when:
    - Validation fails and you need to understand field requirements
    - Before creating an artifact to ensure compliance
    - To understand field types and allowed values
    """

    name: str = "consult_schema"
    description: str = (
        "Look up artifact schema definition to understand required/optional fields, "
        "types, and validation patterns. "
        "Input: artifact_type (e.g., 'hook_card', 'scene', 'act', 'chapter')"
    )

    def _run(self, artifact_type: str) -> str:
        """Look up schema definition."""
        loader = get_resource_loader()

        # Normalize artifact_type
        artifact_type = artifact_type.lower().replace(" ", "_").replace("-", "_")
        artifact_type = artifact_type.replace(".schema.json", "").replace(".schema", "")

        # Try to load schema from JSON files first
        schema = loader.load_schema(artifact_type)
        if schema is not None:
            return self._format_schema(artifact_type, schema)

        # Fallback: generate schema from Pydantic models
        schema = self._get_schema_from_pydantic(artifact_type)
        if schema is not None:
            return self._format_schema(artifact_type, schema)

        # List available schemas
        available = self._list_available_schemas()
        if available:
            return (
                f"Schema '{artifact_type}' not found.\n\n"
                f"Available schemas: {', '.join(sorted(available))}"
            )
        return f"Schema '{artifact_type}' not found and no schemas available."

    def _get_schema_from_pydantic(self, artifact_type: str) -> dict[str, Any] | None:
        """Get JSON schema from Pydantic model."""
        try:
            from questfoundry.runtime.validation import get_artifact_model

            model = get_artifact_model(artifact_type)
            if model is not None:
                return model.model_json_schema()
        except ImportError:
            logger.debug("validation module not available for schema generation")
        return None

    def _list_available_schemas(self) -> list[str]:
        """List available schema types from both JSON files and Pydantic models."""
        available = set()

        # Check resource loader
        loader = get_resource_loader()
        available.update(loader.list_schemas())

        # Add schemas from Pydantic models
        try:
            from questfoundry.runtime.validation import _get_artifact_models

            models = _get_artifact_models()
            # Deduplicate by getting unique model classes
            seen_models: set[str] = set()
            for key, model in models.items():
                if model.__name__ not in seen_models:
                    seen_models.add(model.__name__)
                    available.add(key)
        except ImportError:
            pass

        return list(available)

    def _format_schema(self, artifact_type: str, schema: dict[str, Any]) -> str:
        """Format JSON schema as readable markdown for agents."""
        lines = [
            f"# Schema: {artifact_type}",
            "",
        ]

        # Description
        if "description" in schema:
            lines.append(schema["description"])
            lines.append("")

        # Required fields
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})

        if properties:
            lines.append("## Fields")
            lines.append("")

            # Required fields first
            if required:
                lines.append("### Required Fields")
                for name in sorted(required):
                    if name in properties:
                        prop = properties[name]
                        lines.append(self._format_property(name, prop, required=True))
                lines.append("")

            # Optional fields
            optional = [n for n in properties if n not in required]
            if optional:
                lines.append("### Optional Fields")
                for name in sorted(optional):
                    prop = properties[name]
                    lines.append(self._format_property(name, prop, required=False))
                lines.append("")

        return "\n".join(lines)

    def _format_property(self, name: str, prop: dict[str, Any], required: bool) -> str:
        """Format a single property."""
        prop_type = prop.get("type", "any")
        if isinstance(prop_type, list):
            prop_type = " | ".join(prop_type)

        desc = prop.get("description", "")
        marker = "*" if required else ""

        parts = [f"- **{name}**{marker} ({prop_type})"]

        if desc:
            parts.append(f": {desc}")

        # Add enum values if present
        if "enum" in prop:
            enum_vals = ", ".join(f"`{v}`" for v in prop["enum"])
            parts.append(f" Allowed values: {enum_vals}")

        # Add pattern if present
        if "pattern" in prop:
            parts.append(f" Pattern: `{prop['pattern']}`")

        # Add min/max length
        if "minLength" in prop or "maxLength" in prop:
            length_parts = []
            if "minLength" in prop:
                length_parts.append(f"min: {prop['minLength']}")
            if "maxLength" in prop:
                length_parts.append(f"max: {prop['maxLength']}")
            parts.append(f" Length: {', '.join(length_parts)}")

        return "".join(parts)


class ConsultTool(BaseTool):
    """Look up tool documentation (parameters, valid values, examples).

    Use this when:
    - You need to know what parameters a tool accepts
    - Validation failed and you need to see valid values for an enum
    - You want to understand what a tool does before calling it
    """

    name: str = "consult_tool"
    description: str = (
        "Look up tool documentation to understand its parameters, types, "
        "valid enum values, and usage. "
        "Input: tool_name (e.g., 'return_to_sr', 'write_hot_sot', 'delegate_to')"
    )

    # Tool registry is injected by executor
    tool_registry: dict[str, BaseTool] | None = None

    def _run(self, tool_name: str) -> str:
        """Look up tool documentation."""
        # Normalize tool_name
        tool_name = tool_name.lower().replace(" ", "_").replace("-", "_")

        # Try to find tool in registry
        tool = None
        if self.tool_registry:
            tool = self.tool_registry.get(tool_name)

        if tool is None:
            # List available tools
            available = sorted(self.tool_registry.keys()) if self.tool_registry else []
            if available:
                return (
                    f"Tool '{tool_name}' not found.\n\n"
                    f"Available tools: {', '.join(available)}"
                )
            return f"Tool '{tool_name}' not found and no tools available."

        return self._format_tool(tool)

    def _format_tool(self, tool: BaseTool) -> str:
        """Format tool documentation as man page."""
        lines = [
            f"# Tool: {tool.name}",
            "",
            "## Description",
            tool.description,
            "",
        ]

        # Get input schema
        try:
            schema = tool.get_input_schema()
            if hasattr(schema, "model_fields"):
                lines.append("## Parameters")
                lines.append("")

                # Determine required fields
                required_fields: set[str] = set()
                if hasattr(schema, "model_json_schema"):
                    json_schema = schema.model_json_schema()
                    required_fields = set(json_schema.get("required", []))

                for name, field_info in schema.model_fields.items():
                    is_required = name in required_fields or field_info.is_required()
                    lines.append(self._format_param(name, field_info, is_required))

                lines.append("")
        except Exception as e:
            logger.debug(f"Could not get schema for {tool.name}: {e}")

        # Add usage hint
        lines.append("## Hints")
        lines.append("")
        lines.append(
            "- If validation fails, check the error message for valid values."
        )
        lines.append(
            "- Use consult_schema for artifact field requirements."
        )
        lines.append(
            "- Use consult_role_charter to see which tools a role can use."
        )

        return "\n".join(lines)

    def _format_param(self, name: str, field_info: Any, is_required: bool) -> str:
        """Format a single parameter."""
        # Get type annotation
        type_str = "any"
        if field_info.annotation is not None:
            type_str = self._format_type(field_info.annotation)

        # Required marker
        marker = " **(required)**" if is_required else " (optional)"

        # Description
        desc = field_info.description or ""

        parts = [f"- **{name}**{marker}: {type_str}"]

        if desc:
            parts.append(f"\n  {desc}")

        # Default value
        if not is_required and field_info.default is not None:
            default_str = repr(field_info.default)
            if len(default_str) < 50:  # Don't show huge defaults
                parts.append(f"\n  Default: `{default_str}`")

        # Enum values from Literal type
        if hasattr(field_info.annotation, "__args__"):
            origin = getattr(field_info.annotation, "__origin__", None)
            if origin is not None:
                origin_name = getattr(origin, "__name__", str(origin))
                if origin_name == "Literal":
                    enum_vals = ", ".join(f"`{v}`" for v in field_info.annotation.__args__)
                    parts.append(f"\n  Valid values: {enum_vals}")

        return "".join(parts)

    def _format_type(self, annotation: Any) -> str:
        """Format type annotation as readable string."""
        if annotation is None:
            return "any"

        # Handle basic types
        if hasattr(annotation, "__name__"):
            return str(annotation.__name__)

        # Handle generic types (Optional, List, etc.)
        origin = getattr(annotation, "__origin__", None)
        if origin is not None:
            origin_name = getattr(origin, "__name__", str(origin))
            args = getattr(annotation, "__args__", ())

            if origin_name == "Union":
                # Handle Optional (Union[X, None])
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1 and len(args) == 2:
                    return f"{self._format_type(non_none[0])} | None"
                return " | ".join(self._format_type(a) for a in args)

            if origin_name == "Literal":
                vals = ", ".join(repr(a) for a in args)
                return f"Literal[{vals}]"

            if args:
                args_str = ", ".join(self._format_type(a) for a in args)
                return f"{origin_name}[{args_str}]"

            return origin_name

        # Fallback
        return str(annotation)
