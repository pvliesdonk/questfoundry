"""
Unified Consult tool implementation.

Dispatches to consult_playbook, consult_knowledge, or consult_schema
based on the lookup_type parameter. Provides a single tool for small
models to reduce schema token overhead.
"""

from __future__ import annotations

from typing import Any

from questfoundry.runtime.models import Tool
from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.consult_knowledge import ConsultKnowledgeTool
from questfoundry.runtime.tools.consult_playbook import ConsultPlaybookTool
from questfoundry.runtime.tools.consult_schema import ConsultSchemaTool
from questfoundry.runtime.tools.registry import register_tool

# Stub definitions for delegate tools (required for BaseTool constructor)
_PLAYBOOK_TOOL_DEF = Tool(
    id="consult_playbook",
    name="Consult Playbook",
    description="Get full details for a playbook/workflow.",
)
_KNOWLEDGE_TOOL_DEF = Tool(
    id="consult_knowledge",
    name="Consult Knowledge",
    description="Retrieve full content for a knowledge entry.",
)
_SCHEMA_TOOL_DEF = Tool(
    id="consult_schema",
    name="Consult Schema",
    description="Get artifact type definition with schema and validation rules.",
)


@register_tool("consult")
class ConsultTool(BaseTool):
    """
    Unified lookup tool for reference material.

    Dispatches to the appropriate specialized tool based on lookup_type:
    - playbook: Get workflow phases, steps, agents, and completion criteria
    - knowledge: Get full content of knowledge entries from Knowledge Menu
    - schema: Get artifact type definitions with fields and validation rules

    This tool is designed for small models to reduce schema token overhead.
    Large models should use the specialized tools directly for better clarity.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Lazily initialize delegate tools
        self._playbook_tool: ConsultPlaybookTool | None = None
        self._knowledge_tool: ConsultKnowledgeTool | None = None
        self._schema_tool: ConsultSchemaTool | None = None

    def _get_playbook_tool(self) -> ConsultPlaybookTool:
        """Get or create the playbook tool delegate."""
        if self._playbook_tool is None:
            self._playbook_tool = ConsultPlaybookTool(_PLAYBOOK_TOOL_DEF, self._context)
        return self._playbook_tool

    def _get_knowledge_tool(self) -> ConsultKnowledgeTool:
        """Get or create the knowledge tool delegate."""
        if self._knowledge_tool is None:
            self._knowledge_tool = ConsultKnowledgeTool(_KNOWLEDGE_TOOL_DEF, self._context)
        return self._knowledge_tool

    def _get_schema_tool(self) -> ConsultSchemaTool:
        """Get or create the schema tool delegate."""
        if self._schema_tool is None:
            self._schema_tool = ConsultSchemaTool(_SCHEMA_TOOL_DEF, self._context)
        return self._schema_tool

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute lookup by dispatching to the appropriate tool."""
        lookup_type = args.get("lookup_type")
        lookup_id = args.get("id")

        if not lookup_type:
            return ToolResult(
                success=False,
                data={},
                error="lookup_type is required. Use 'playbook', 'knowledge', or 'schema'.",
            )

        if not lookup_id:
            return ToolResult(
                success=False,
                data={},
                error="id is required. Specify the ID of the playbook, knowledge entry, or artifact type.",
            )

        if lookup_type == "playbook":
            return await self._consult_playbook(lookup_id)
        elif lookup_type == "knowledge":
            section = args.get("section")
            return await self._consult_knowledge(lookup_id, section)
        elif lookup_type == "schema":
            include_examples = args.get("include_examples", True)
            include_validation_rules = args.get("include_validation_rules", True)
            return await self._consult_schema(lookup_id, include_examples, include_validation_rules)
        else:
            return ToolResult(
                success=False,
                data={},
                error=f"Unknown lookup_type: {lookup_type}. Use 'playbook', 'knowledge', or 'schema'.",
            )

    async def _consult_playbook(self, playbook_id: str) -> ToolResult:
        """Delegate to consult_playbook."""
        tool = self._get_playbook_tool()
        result = await tool.execute({"playbook_id": playbook_id})

        # Wrap result to indicate unified tool was used
        if result.success:
            result.data["lookup_type"] = "playbook"
        return result

    async def _consult_knowledge(self, entry_id: str, section: str | None = None) -> ToolResult:
        """Delegate to consult_knowledge."""
        tool = self._get_knowledge_tool()
        args: dict[str, Any] = {"entry_id": entry_id}
        if section:
            args["section"] = section
        result = await tool.execute(args)

        # Wrap result to indicate unified tool was used
        if result.success:
            result.data["lookup_type"] = "knowledge"
        return result

    async def _consult_schema(
        self,
        artifact_type_id: str,
        include_examples: bool = True,
        include_validation_rules: bool = True,
    ) -> ToolResult:
        """Delegate to consult_schema."""
        tool = self._get_schema_tool()
        result = await tool.execute(
            {
                "artifact_type_id": artifact_type_id,
                "include_examples": include_examples,
                "include_validation_rules": include_validation_rules,
            }
        )

        # Wrap result to indicate unified tool was used
        if result.success:
            result.data["lookup_type"] = "schema"
        return result

    def validate_input(self, args: dict[str, Any]) -> None:
        """Validate input arguments."""
        super().validate_input(args)

        lookup_type = args.get("lookup_type")
        if lookup_type is not None:
            if not isinstance(lookup_type, str):
                raise ToolValidationError("lookup_type must be a string")
            if lookup_type not in ("playbook", "knowledge", "schema"):
                raise ToolValidationError(
                    f"lookup_type must be 'playbook', 'knowledge', or 'schema', got '{lookup_type}'"
                )

        lookup_id = args.get("id")
        if lookup_id is not None and not isinstance(lookup_id, str):
            raise ToolValidationError("id must be a string")

        section = args.get("section")
        if section is not None and not isinstance(section, str):
            raise ToolValidationError("section must be a string")
