"""
List Agents tool implementation.

Returns compact list of available agents for delegation.
"""

from __future__ import annotations

from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool


@register_tool("list_agents")
class ListAgentsTool(BaseTool):
    """
    List all agents available for delegation.

    Use this to discover what specialist agents exist and their capabilities.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute agent listing."""
        _ = args  # No args needed for this tool

        agents = []
        for agent in self._context.studio.agents:
            archetypes = [
                a.value if hasattr(a, "value") else str(a) for a in (agent.archetypes or [])
            ]
            agents.append(
                {
                    "id": agent.id,
                    "name": agent.name,
                    "archetypes": archetypes,
                    "description": (agent.description or "")[:80],
                }
            )

        return ToolResult(
            success=True,
            data={
                "agents": agents,
                "count": len(agents),
            },
        )
