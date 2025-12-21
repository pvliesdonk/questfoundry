"""
List Artifact Types tool implementation.

Returns compact list of available artifact types for discovery.
Use consult_schema for full field definitions.
"""

from __future__ import annotations

from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool


@register_tool("list_artifact_types")
class ListArtifactTypesTool(BaseTool):
    """
    List all available artifact types with brief descriptions.

    Use this to discover what artifact types exist.
    Then call consult_schema(artifact_type_id) for full field definitions.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute artifact type listing."""
        _ = args  # No args needed for this tool

        types = []
        for at in self._context.studio.artifact_types:
            types.append(
                {
                    "id": at.id,
                    "name": at.name,
                    "category": at.category or "document",
                    "description": (at.description or "")[:80],
                }
            )

        return ToolResult(
            success=True,
            data={
                "artifact_types": types,
                "count": len(types),
                "recommended_action": "Call consult_schema(artifact_type_id) to get full field definitions for the type you need.",
            },
        )
