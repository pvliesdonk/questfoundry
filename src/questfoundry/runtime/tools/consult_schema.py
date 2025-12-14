"""
Consult Schema tool implementation.

Returns artifact type schema with field definitions, validation rules,
and lifecycle states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

if TYPE_CHECKING:
    from questfoundry.runtime.models import ArtifactType


@register_tool("consult_schema")
class ConsultSchemaTool(BaseTool):
    """
    Retrieve schema definition for an artifact type.

    Returns field definitions, validation rules, and lifecycle states.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute schema lookup."""
        artifact_type_id = args.get("artifact_type_id")
        _include_examples = args.get("include_examples", True)  # TODO: add examples support
        include_validation_rules = args.get("include_validation_rules", True)
        _ = _include_examples  # Silence unused variable warning until implemented

        # Find artifact type in studio
        artifact_type = None
        for at in self._context.studio.artifact_types:
            if at.id == artifact_type_id:
                artifact_type = at
                break

        if not artifact_type:
            return ToolResult(
                success=False,
                data={},
                error=f"Artifact type not found: {artifact_type_id}",
            )

        # Build response
        result_data: dict[str, Any] = {
            "artifact_type": {
                "id": artifact_type.id,
                "name": artifact_type.name,
                "description": artifact_type.description,
                "category": artifact_type.category,
            },
        }

        # Build field summary
        field_summary = self._build_field_summary(artifact_type)
        result_data["field_summary"] = field_summary

        # Include full field definitions
        if artifact_type.fields:
            result_data["artifact_type"]["fields"] = [
                {
                    "name": f.name,
                    "type": f.type.value if f.type else "string",
                    "required": f.required,
                    "description": f.description,
                }
                for f in artifact_type.fields
            ]

        # Build lifecycle summary
        if artifact_type.lifecycle:
            lifecycle_summary = self._build_lifecycle_summary(artifact_type)
            result_data["lifecycle_summary"] = lifecycle_summary

            result_data["artifact_type"]["lifecycle"] = {
                "initial_state": artifact_type.lifecycle.initial_state,
                "states": [
                    {"id": s.id, "name": s.name, "terminal": s.terminal}
                    for s in artifact_type.lifecycle.states
                ],
                "transitions": [
                    {
                        "from": t.from_state,
                        "to": t.to_state,
                        "allowed_agents": t.allowed_agents,
                    }
                    for t in artifact_type.lifecycle.transitions
                ],
            }

        # Include validation rules
        if include_validation_rules and artifact_type.validation:
            result_data["artifact_type"]["validation"] = {
                "required_together": artifact_type.validation.required_together,
                "mutually_exclusive": artifact_type.validation.mutually_exclusive,
            }

        # Include default store
        if artifact_type.default_store:
            result_data["artifact_type"]["default_store"] = artifact_type.default_store

        return ToolResult(success=True, data=result_data)

    def _build_field_summary(self, artifact_type: ArtifactType) -> str:
        """Build human-readable field summary."""
        if not artifact_type.fields:
            return "No fields defined."

        required = []
        optional = []

        for field in artifact_type.fields:
            field_desc = f"{field.name} ({field.type.value if field.type else 'string'})"
            if field.description:
                field_desc += f": {field.description}"

            if field.required:
                required.append(field_desc)
            else:
                optional.append(field_desc)

        parts = []
        if required:
            parts.append(f"Required: {', '.join(required)}")
        if optional:
            parts.append(f"Optional: {', '.join(optional)}")

        return "\n".join(parts) if parts else "No fields defined."

    def _build_lifecycle_summary(self, artifact_type: ArtifactType) -> str:
        """Build human-readable lifecycle summary."""
        if not artifact_type.lifecycle:
            return "No lifecycle defined."

        lc = artifact_type.lifecycle
        parts = []

        if lc.initial_state:
            parts.append(f"Initial state: {lc.initial_state}")

        if lc.states:
            state_names = [s.id for s in lc.states]
            parts.append(f"States: {' -> '.join(state_names)}")

        if lc.transitions:
            trans_strs = []
            for t in lc.transitions:
                agents = ", ".join(t.allowed_agents) if t.allowed_agents else "any"
                trans_strs.append(f"{t.from_state} -> {t.to_state} (by {agents})")
            parts.append("Transitions:\n  " + "\n  ".join(trans_strs))

        return "\n".join(parts) if parts else "No lifecycle defined."
