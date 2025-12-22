"""
Search Workspace tool implementation.

Searches for artifacts in the project storage with filtering by:
- Artifact type(s)
- Lifecycle state(s)
- Creator agent
- Creation date
- Text query
"""

from __future__ import annotations

import json
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolExecutionError, ToolResult
from questfoundry.runtime.tools.registry import register_tool


@register_tool("search_workspace")
class SearchWorkspaceTool(BaseTool):
    """
    Search for artifacts in the workspace.

    Supports filtering by type, lifecycle state, creator, and text search.
    """

    def check_availability(self) -> bool:
        """Check if project storage is available."""
        return self._context.project is not None

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute workspace search."""
        if not self._context.project:
            return ToolResult(
                success=False,
                data={"results": [], "total_count": 0},
                error="No project storage available",
            )

        query_input = args.get("query")
        artifact_types = args.get("artifact_types", [])
        lifecycle_states = args.get("lifecycle_states", [])
        created_by = args.get("created_by")
        created_after = args.get("created_after")
        related_to = args.get("related_to")
        limit = args.get("limit", 20)

        # Normalize query to list (supports single string or array input)
        if query_input is None:
            queries = None
        elif isinstance(query_input, list):
            queries = [q.strip() for q in query_input if q and q.strip()]
        else:
            queries = [query_input.strip()] if query_input.strip() else None

        try:
            results, warning = self._search_artifacts(
                queries=queries,
                artifact_types=artifact_types,
                lifecycle_states=lifecycle_states,
                created_by=created_by,
                created_after=created_after,
                related_to=related_to,
                limit=limit,
            )

            data: dict[str, Any] = {
                "action_outcome": f"found {len(results)} artifacts",
                "results": results,
                "total_count": len(results),
            }
            if warning:
                data["warning"] = warning

            return ToolResult(
                success=True,
                data=data,
            )

        except Exception as e:
            raise ToolExecutionError(f"Search failed: {e}") from e

    def _search_artifacts(
        self,
        queries: list[str] | None = None,
        artifact_types: list[str] | None = None,
        lifecycle_states: list[str] | None = None,
        created_by: str | None = None,
        created_after: str | None = None,
        related_to: str | None = None,
        limit: int = 20,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Search artifacts with filters. Returns (results, warning)."""
        if not self._context.project:
            return [], None
        conn = self._context.project._get_connection()
        warning: str | None = None

        # Build query dynamically
        sql = """
            SELECT _id, _type, _lifecycle_state, _created_by, _created_at, _updated_at, data
            FROM artifacts
            WHERE 1=1
        """
        params: list[Any] = []

        # Filter by artifact types
        if artifact_types:
            placeholders = ",".join("?" for _ in artifact_types)
            sql += f" AND _type IN ({placeholders})"
            params.extend(artifact_types)

        # Filter by lifecycle states
        if lifecycle_states:
            placeholders = ",".join("?" for _ in lifecycle_states)
            sql += f" AND _lifecycle_state IN ({placeholders})"
            params.extend(lifecycle_states)

        # Filter by creator
        if created_by:
            sql += " AND _created_by = ?"
            params.append(created_by)

        # Filter by creation date
        if created_after:
            sql += " AND _created_at > ?"
            params.append(created_after)

        # Text search in JSON data
        # TODO: LIKE on JSON data is inefficient for large datasets.
        # Consider implementing FTS5 (Full-Text Search) for better performance.
        # See: https://www.sqlite.org/fts5.html
        if queries:
            if len(queries) > 1:
                # Multiple queries: search for each with OR
                or_clauses = " OR ".join("(_id LIKE ? OR data LIKE ?)" for _ in queries)
                sql += f" AND ({or_clauses})"
                for term in queries:
                    params.extend([f"%{term}%", f"%{term}%"])
            else:
                # Single term - original behavior
                # Search both artifact ID and JSON payload for the query fragment
                sql += " AND (_id LIKE ? OR data LIKE ?)"
                params.extend([f"%{queries[0]}%", f"%{queries[0]}%"])

        # Order and limit
        sql += " ORDER BY _updated_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            data = json.loads(row["data"]) if row["data"] else {}

            # Build summary from data
            summary = self._build_summary(row["_type"], data)

            # Skip if related_to filter doesn't match
            # TODO: Filtering in Python is inefficient - consider using SQLite's
            # json_each() or json_extract() for SQL-level filtering of relationships.
            # This would also allow proper LIMIT application.
            if related_to and not self._is_related(data, related_to):
                continue

            results.append(
                {
                    "artifact_id": row["_id"],
                    "artifact_type": row["_type"],
                    "lifecycle_state": row["_lifecycle_state"],
                    "summary": summary,
                    "created_by": row["_created_by"],
                    "created_at": row["_created_at"],
                }
            )

        return results, warning

    def _build_summary(self, artifact_type: str, data: dict[str, Any]) -> str:
        """Build a human-readable summary of an artifact."""
        # Try common summary fields
        summary_fields = ["title", "name", "summary", "description", "goal", "brief_id"]

        for field in summary_fields:
            if field in data and data[field]:
                value = data[field]
                if isinstance(value, str):
                    # Truncate long values
                    if len(value) > 100:
                        return value[:97] + "..."
                    return value

        # Fallback: show type and field count
        return f"{artifact_type} with {len(data)} fields"

    def _is_related(self, data: dict[str, Any], related_to: str) -> bool:
        """Check if artifact data references another artifact."""
        # Check all string values for the reference
        for value in data.values():
            if isinstance(value, str) and related_to in value:
                return True
            elif isinstance(value, list):
                for item in value:
                    if (
                        isinstance(item, str)
                        and related_to in item
                        or isinstance(item, dict)
                        and self._is_related(item, related_to)
                    ):
                        return True
            elif isinstance(value, dict) and self._is_related(value, related_to):
                return True

        return False
