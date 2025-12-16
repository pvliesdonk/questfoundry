"""
List Stores tool implementation.

Returns compact list of available stores with semantics.
"""

from __future__ import annotations

from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool


@register_tool("list_stores")
class ListStoresTool(BaseTool):
    """
    List all available stores with their semantics.

    Use this to discover what stores exist and their access patterns.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute store listing."""
        _ = args  # No args needed for this tool

        stores = []
        for store in self._context.studio.stores:
            semantics = store.semantics.value if store.semantics else "unknown"
            stores.append(
                {
                    "id": store.id,
                    "name": store.name,
                    "semantics": semantics,
                    "description": (store.description or "")[:80],
                }
            )

        return ToolResult(
            success=True,
            data={
                "stores": stores,
                "count": len(stores),
            },
        )
