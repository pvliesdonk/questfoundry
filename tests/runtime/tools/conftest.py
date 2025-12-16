"""Shared fixtures for runtime tools tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.models.enums import FieldType


def _create_mock_field(
    name: str,
    field_type: FieldType,
    required: bool = False,
    description: str | None = None,
) -> MagicMock:
    """Create a mock field definition."""
    field = MagicMock()
    field.name = name
    field.type = field_type
    field.required = required
    field.description = description
    return field


@pytest.fixture
def mock_artifact_type_factory():
    """
    Factory fixture for creating mock artifact types.

    Usage:
        def test_something(mock_artifact_type_factory):
            artifact_type = mock_artifact_type_factory()
            artifact_type = mock_artifact_type_factory(
                type_id="custom",
                default_store="workspace",
                has_lifecycle=True,
                extra_fields=[("tags", FieldType.ARRAY, False)],
            )
    """

    def _factory(
        type_id: str = "section",
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        default_store: str | None = None,
        has_lifecycle: bool = False,
        initial_state: str = "draft",
        fields: list[tuple[str, FieldType, bool]] | None = None,
        extra_fields: list[tuple[str, FieldType, bool]] | None = None,
        has_validation: bool = False,
    ) -> MagicMock:
        """
        Create a mock artifact type for testing.

        Args:
            type_id: Artifact type ID
            name: Display name (defaults to type_id.title())
            description: Type description
            category: Type category
            default_store: Default store ID
            has_lifecycle: Whether to include lifecycle config
            initial_state: Initial lifecycle state
            fields: Override default fields as [(name, type, required), ...]
            extra_fields: Additional fields to append to defaults
            has_validation: Whether to include validation rules
        """
        artifact_type = MagicMock()
        artifact_type.id = type_id
        artifact_type.name = name or type_id.title()
        artifact_type.description = description
        artifact_type.category = category
        artifact_type.default_store = default_store

        # Default fields: title (required string), body (optional text)
        if fields is not None:
            artifact_type.fields = [_create_mock_field(f[0], f[1], f[2]) for f in fields]
        else:
            artifact_type.fields = [
                _create_mock_field("title", FieldType.STRING, required=True),
                _create_mock_field("body", FieldType.TEXT, required=False),
            ]

        # Add extra fields if specified
        if extra_fields:
            for f in extra_fields:
                artifact_type.fields.append(_create_mock_field(f[0], f[1], f[2]))

        # Lifecycle
        if has_lifecycle:
            lifecycle = MagicMock()
            lifecycle.initial_state = initial_state
            artifact_type.lifecycle = lifecycle
        else:
            artifact_type.lifecycle = None

        # Validation rules
        if has_validation:
            validation = MagicMock()
            validation.required_together = []
            validation.mutually_exclusive = []
            artifact_type.validation = validation
        else:
            artifact_type.validation = None

        return artifact_type

    return _factory


@pytest.fixture
def mock_tool_definition_factory():
    """Factory fixture for creating mock tool definitions."""

    def _factory(tool_id: str, timeout_ms: int = 30000) -> MagicMock:
        definition = MagicMock()
        definition.id = tool_id
        definition.timeout_ms = timeout_ms
        return definition

    return _factory


@pytest.fixture
def mock_artifact_type(mock_artifact_type_factory: Any) -> MagicMock:
    """Default mock artifact type for simple tests."""
    return mock_artifact_type_factory()
