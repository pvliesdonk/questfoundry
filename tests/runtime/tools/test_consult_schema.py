"""Tests for consult_schema tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.models.enums import FieldType
from questfoundry.runtime.tools.base import ToolContext
from questfoundry.runtime.tools.consult_schema import ConsultSchemaTool


def make_mock_artifact_type():
    """Create a mock artifact type."""
    artifact_type = MagicMock()
    artifact_type.id = "section"
    artifact_type.name = "Section"
    artifact_type.description = "A story section"
    artifact_type.category = "document"

    # Fields (use lowercase names to match typical field identifiers)
    field1 = MagicMock()
    field1.name = "title"
    field1.type = FieldType.STRING
    field1.required = True
    field1.description = "Section title"

    field2 = MagicMock()
    field2.name = "body"
    field2.type = FieldType.TEXT
    field2.required = True
    field2.description = "Section content"

    field3 = MagicMock()
    field3.name = "tags"
    field3.type = FieldType.ARRAY
    field3.required = False
    field3.description = "Optional tags"

    artifact_type.fields = [field1, field2, field3]

    # Lifecycle
    lifecycle = MagicMock()
    lifecycle.initial_state = "draft"

    state1 = MagicMock()
    state1.id = "draft"
    state1.name = "Draft"
    state1.terminal = False

    state2 = MagicMock()
    state2.id = "approved"
    state2.name = "Approved"
    state2.terminal = True

    lifecycle.states = [state1, state2]

    trans = MagicMock()
    trans.from_state = "draft"
    trans.to_state = "approved"
    trans.allowed_agents = ["gatekeeper"]
    lifecycle.transitions = [trans]

    artifact_type.lifecycle = lifecycle

    # Validation
    validation = MagicMock()
    validation.required_together = []
    validation.mutually_exclusive = []
    artifact_type.validation = validation

    artifact_type.default_store = "workspace"

    return artifact_type


def make_mock_definition():
    """Create mock tool definition."""
    definition = MagicMock()
    definition.id = "consult_schema"
    definition.name = "Consult Schema"
    definition.description = "Get schema info"
    definition.timeout_ms = 30000
    definition.input_schema = None
    return definition


class TestConsultSchemaTool:
    """Tests for ConsultSchemaTool."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ConsultSchemaTool(definition, context)

        result = await tool.execute({"artifact_type_id": "section"})

        assert result.success is True
        assert "artifact_type" in result.data
        assert result.data["artifact_type"]["id"] == "section"
        assert "field_summary" in result.data
        assert "lifecycle_summary" in result.data

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ConsultSchemaTool(definition, context)

        result = await tool.execute({"artifact_type_id": "nonexistent"})

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_field_summary_includes_required(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ConsultSchemaTool(definition, context)

        result = await tool.execute({"artifact_type_id": "section"})

        field_summary = result.data["field_summary"]
        assert "Required" in field_summary
        assert "title" in field_summary
        assert "body" in field_summary

    @pytest.mark.asyncio
    async def test_field_summary_includes_optional(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ConsultSchemaTool(definition, context)

        result = await tool.execute({"artifact_type_id": "section"})

        field_summary = result.data["field_summary"]
        assert "Optional" in field_summary
        assert "tags" in field_summary

    @pytest.mark.asyncio
    async def test_lifecycle_summary(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ConsultSchemaTool(definition, context)

        result = await tool.execute({"artifact_type_id": "section"})

        lifecycle_summary = result.data["lifecycle_summary"]
        assert "draft" in lifecycle_summary
        assert "approved" in lifecycle_summary
        assert "gatekeeper" in lifecycle_summary

    @pytest.mark.asyncio
    async def test_includes_fields_in_response(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ConsultSchemaTool(definition, context)

        result = await tool.execute({"artifact_type_id": "section"})

        fields = result.data["artifact_type"]["fields"]
        assert len(fields) == 3
        assert fields[0]["name"] == "title"
        assert fields[0]["required"] is True
