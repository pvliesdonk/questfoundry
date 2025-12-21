"""Tests for validate_artifact tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.models.enums import FieldType
from questfoundry.runtime.tools.base import ToolContext
from questfoundry.runtime.tools.validate_artifact import ValidateArtifactTool


def make_mock_artifact_type():
    """Create a mock artifact type for testing."""
    artifact_type = MagicMock()
    artifact_type.id = "section"
    artifact_type.name = "Section"

    # Fields (use 'name' attribute to match FieldDefinition model)
    field1 = MagicMock()
    field1.name = "title"
    field1.type = FieldType.STRING
    field1.required = True

    field2 = MagicMock()
    field2.name = "body"
    field2.type = FieldType.TEXT
    field2.required = True

    field3 = MagicMock()
    field3.name = "word_count"
    field3.type = FieldType.INTEGER
    field3.required = False

    artifact_type.fields = [field1, field2, field3]

    # Validation rules
    validation = MagicMock()
    validation.required_together = []
    validation.mutually_exclusive = []
    artifact_type.validation = validation

    return artifact_type


def make_mock_definition():
    """Create mock tool definition."""
    definition = MagicMock()
    definition.id = "validate_artifact"
    definition.name = "Validate Artifact"
    definition.description = "Validate artifact"
    definition.timeout_ms = 30000
    definition.input_schema = None
    return definition


class TestValidateArtifactTool:
    """Tests for ValidateArtifactTool."""

    @pytest.mark.asyncio
    async def test_schema_validation_valid(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {"title": "Test", "body": "Content"},
                "validation_mode": "schema_only",
            }
        )

        assert result.success is True
        assert result.data["valid"] is True
        assert len(result.data["schema_errors"]) == 0

    @pytest.mark.asyncio
    async def test_schema_validation_missing_required(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {"title": "Test"},  # Missing body
                "validation_mode": "schema_only",
            }
        )

        assert result.success is True
        assert result.data["valid"] is False
        assert len(result.data["schema_errors"]) > 0
        assert any("body" in e["field"] for e in result.data["schema_errors"])

    @pytest.mark.asyncio
    async def test_schema_validation_wrong_type(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {"title": 123, "body": "Content"},  # Wrong type
                "validation_mode": "schema_only",
            }
        )

        assert result.success is True
        assert result.data["valid"] is False
        assert any("title" in e["field"] for e in result.data["schema_errors"])

    @pytest.mark.asyncio
    async def test_bars_validation(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {"title": "Test", "body": "Content"},
                "validation_mode": "bars_only",
            }
        )

        assert result.success is True
        assert "bar_results" in result.data
        # All 8 bars should be checked
        assert len(result.data["bar_results"]) == 8

    @pytest.mark.asyncio
    async def test_bars_subset(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {"title": "Test", "body": "Content"},
                "validation_mode": "bars_only",
                "bars_to_check": ["integrity", "style"],
            }
        )

        assert result.success is True
        bar_names = {br["bar"] for br in result.data["bar_results"]}
        assert bar_names == {"integrity", "style"}

    @pytest.mark.asyncio
    async def test_full_validation(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {"title": "Test", "body": "Content"},
                "validation_mode": "full",
            }
        )

        assert result.success is True
        assert "schema_errors" in result.data
        assert "bar_results" in result.data

    @pytest.mark.asyncio
    async def test_artifact_type_not_found(self):
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "nonexistent",
                "artifact_data": {},
            }
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_no_artifact_data(self):
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                # No artifact_data
            }
        )

        assert result.success is False
        assert "No artifact data" in result.error


class TestQualityBars:
    """Tests for individual quality bar checks."""

    @pytest.mark.asyncio
    async def test_nonlinearity_with_choices(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {
                    "title": "Test",
                    "body": "Content",
                    "choices": [
                        {"text": "Go left"},
                        {"text": "Go right"},
                        {"text": "Stay here"},
                    ],
                },
                "validation_mode": "bars_only",
                "bars_to_check": ["nonlinearity"],
            }
        )

        bar_result = result.data["bar_results"][0]
        assert bar_result["bar"] == "nonlinearity"
        assert bar_result["status"] == "pass"
        assert "3 choices" in bar_result["evidence"]

    @pytest.mark.asyncio
    async def test_accessibility_short_choice_text(self):
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition()
        context = ToolContext(studio=studio)
        tool = ValidateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type_id": "section",
                "artifact_data": {
                    "title": "Test",
                    "body": "Content",
                    "choices": [
                        {"text": "Go"},  # Too short
                        {"text": "Stay and explore"},
                    ],
                },
                "validation_mode": "bars_only",
                "bars_to_check": ["accessibility"],
            }
        )

        bar_result = result.data["bar_results"][0]
        assert bar_result["bar"] == "accessibility"
        assert bar_result["status"] == "warn"
        assert "too short" in bar_result["evidence"]
