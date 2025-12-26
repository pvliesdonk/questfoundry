"""Tests for the unified consult tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.runtime.tools.base import ToolContext, ToolResult
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

    def test_requires_lookup_type(self, mock_tool_definition, mock_context):
        """lookup_type is required."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        # Execute without lookup_type
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(tool.execute({"id": "some_id"}))

        assert not result.success
        assert "lookup_type is required" in result.error

    def test_requires_id(self, mock_tool_definition, mock_context):
        """id is required."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            tool.execute({"lookup_type": "playbook"})
        )

        assert not result.success
        assert "id is required" in result.error

    def test_unknown_lookup_type(self, mock_tool_definition, mock_context):
        """Unknown lookup_type returns error."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            tool.execute({"lookup_type": "invalid", "id": "some_id"})
        )

        assert not result.success
        assert "Unknown lookup_type" in result.error

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

    def test_validate_input_lookup_type_must_be_string(self, mock_tool_definition, mock_context):
        """lookup_type must be a string."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        from questfoundry.runtime.tools.base import ToolValidationError

        with pytest.raises(ToolValidationError, match="'lookup_type'.*must be a string"):
            tool.validate_input({"lookup_type": 123, "id": "test"})

    def test_validate_input_lookup_type_must_be_valid(self, mock_tool_definition, mock_context):
        """lookup_type must be playbook, knowledge, or schema."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        from questfoundry.runtime.tools.base import ToolValidationError

        with pytest.raises(
            ToolValidationError, match="must be 'playbook', 'knowledge', or 'schema'"
        ):
            tool.validate_input({"lookup_type": "invalid", "id": "test"})

    def test_validate_input_id_must_be_string(self, mock_tool_definition, mock_context):
        """id must be a string."""
        tool = ConsultTool(mock_tool_definition, mock_context)

        from questfoundry.runtime.tools.base import ToolValidationError

        with pytest.raises(ToolValidationError, match="'id'.*must be a string"):
            tool.validate_input({"lookup_type": "playbook", "id": 123})
