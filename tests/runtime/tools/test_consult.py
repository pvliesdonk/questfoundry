"""Tests for the unified consult tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.runtime.tools.base import ToolContext, ToolResult, ToolValidationError
from questfoundry.runtime.tools.consult import ConsultTool


@pytest.fixture
def mock_tool_definition():
    """Create a mock tool definition."""
    definition = MagicMock()
    definition.id = "consult"
    definition.name = "Consult Reference"
    definition.description = "Look up reference material"
    definition.timeout_ms = 30000
    definition.input_schema = MagicMock()
    definition.input_schema.required = ["lookup_type", "id"]
    definition.input_schema.properties = {
        "lookup_type": {"type": "string"},
        "id": {"type": "string"},
    }
    definition.concise_description = "Look up references"
    return definition


@pytest.fixture
def mock_context():
    """Create a mock tool context."""
    studio = MagicMock()
    studio.playbooks = []
    context = ToolContext(studio=studio)
    return context


class TestConsultTool:
    """Tests for ConsultTool."""

    def test_validate_input_rejects_invalid_lookup_type(self, mock_tool_definition, mock_context):
        """validate_input rejects invalid lookup_type values."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        with pytest.raises(
            ToolValidationError, match="must be 'playbook', 'knowledge', or 'schema'"
        ):
            tool.validate_input({"lookup_type": "invalid", "id": "test"})

    def test_validate_input_accepts_valid_lookup_types(self, mock_tool_definition, mock_context):
        """validate_input accepts valid lookup_type values."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        # These should not raise
        tool.validate_input({"lookup_type": "playbook", "id": "test"})
        tool.validate_input({"lookup_type": "knowledge", "id": "test"})
        tool.validate_input({"lookup_type": "schema", "id": "test"})

    @pytest.mark.asyncio
    async def test_dispatches_to_playbook(self, mock_tool_definition, mock_context):
        """lookup_type=playbook dispatches to ConsultPlaybookTool."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        # Mock the playbook tool
        mock_result = ToolResult(
            success=True,
            data={"playbook_id": "story_spark", "phases": []},
        )

        with patch.object(tool, "_get_playbook_tool") as mock_get:
            mock_playbook_tool = AsyncMock()
            mock_playbook_tool.execute.return_value = mock_result
            mock_get.return_value = mock_playbook_tool

            result = await tool.execute({"lookup_type": "playbook", "id": "story_spark"})

            mock_playbook_tool.execute.assert_called_once_with({"playbook_id": "story_spark"})
            assert result.success
            assert result.data["lookup_type"] == "playbook"

    @pytest.mark.asyncio
    async def test_dispatches_to_knowledge(self, mock_tool_definition, mock_context):
        """lookup_type=knowledge dispatches to ConsultKnowledgeTool."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        mock_result = ToolResult(
            success=True,
            data={"entry_id": "runtime_guidelines", "content": "..."},
        )

        with patch.object(tool, "_get_knowledge_tool") as mock_get:
            mock_knowledge_tool = AsyncMock()
            mock_knowledge_tool.execute.return_value = mock_result
            mock_get.return_value = mock_knowledge_tool

            result = await tool.execute({"lookup_type": "knowledge", "id": "runtime_guidelines"})

            mock_knowledge_tool.execute.assert_called_once_with({"entry_id": "runtime_guidelines"})
            assert result.success
            assert result.data["lookup_type"] == "knowledge"

    @pytest.mark.asyncio
    async def test_dispatches_to_knowledge_with_section(self, mock_tool_definition, mock_context):
        """lookup_type=knowledge with section passes section to tool."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        mock_result = ToolResult(
            success=True,
            data={"entry_id": "runtime_guidelines", "section": "tools"},
        )

        with patch.object(tool, "_get_knowledge_tool") as mock_get:
            mock_knowledge_tool = AsyncMock()
            mock_knowledge_tool.execute.return_value = mock_result
            mock_get.return_value = mock_knowledge_tool

            result = await tool.execute(
                {"lookup_type": "knowledge", "id": "runtime_guidelines", "section": "tools"}
            )

            mock_knowledge_tool.execute.assert_called_once_with(
                {"entry_id": "runtime_guidelines", "section": "tools"}
            )
            assert result.success

    @pytest.mark.asyncio
    async def test_dispatches_to_schema(self, mock_tool_definition, mock_context):
        """lookup_type=schema dispatches to ConsultSchemaTool."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        mock_result = ToolResult(
            success=True,
            data={"artifact_type": "passage", "fields": []},
        )

        with patch.object(tool, "_get_schema_tool") as mock_get:
            mock_schema_tool = AsyncMock()
            mock_schema_tool.execute.return_value = mock_result
            mock_get.return_value = mock_schema_tool

            result = await tool.execute({"lookup_type": "schema", "id": "passage"})

            mock_schema_tool.execute.assert_called_once_with(
                {
                    "artifact_type_id": "passage",
                    "include_examples": True,
                    "include_validation_rules": True,
                }
            )
            assert result.success
            assert result.data["lookup_type"] == "schema"

    @pytest.mark.asyncio
    async def test_dispatches_to_schema_with_options(self, mock_tool_definition, mock_context):
        """lookup_type=schema respects include_examples and include_validation_rules."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        mock_result = ToolResult(
            success=True,
            data={"artifact_type": "passage"},
        )

        with patch.object(tool, "_get_schema_tool") as mock_get:
            mock_schema_tool = AsyncMock()
            mock_schema_tool.execute.return_value = mock_result
            mock_get.return_value = mock_schema_tool

            result = await tool.execute(
                {
                    "lookup_type": "schema",
                    "id": "passage",
                    "include_examples": False,
                    "include_validation_rules": False,
                }
            )

            mock_schema_tool.execute.assert_called_once_with(
                {
                    "artifact_type_id": "passage",
                    "include_examples": False,
                    "include_validation_rules": False,
                }
            )
            assert result.success
