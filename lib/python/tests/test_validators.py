"""Tests for schema validation utilities."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from questfoundry.validators.schema import (
    validate_instance,
    validate_instance_detailed,
    validate_schema,
)


@patch("questfoundry.validators.schema.get_schema")
def test_validate_instance_valid(mock_get_schema):
    """Test validation of a valid instance."""
    mock_get_schema.return_value = {
        "type": "object",
        "properties": {"data": {"type": "object"}},
    }
    instance = {
        "type": "object",
        "data": {"foo": "bar"},
    }
    assert validate_instance(instance, "any_schema") is True


@patch("questfoundry.validators.schema.get_schema")
def test_validate_instance_invalid(mock_get_schema):
    """Test validation returns False for invalid instance."""
    mock_get_schema.return_value = {"type": "object", "required": ["foo"]}
    instance = {}
    assert validate_instance(instance, "any_schema") is False


@patch("questfoundry.validators.schema.get_schema")
def test_validate_instance_detailed_valid(mock_get_schema):
    """Test detailed validation of valid instance."""
    mock_get_schema.return_value = {
        "type": "object",
        "properties": {"test": {"type": "string"}},
    }
    instance = {"test": "value"}
    result = validate_instance_detailed(instance, "any_schema")
    assert result["valid"] is True
    assert not result["errors"]


@patch("questfoundry.validators.schema.get_schema")
def test_validate_instance_detailed_invalid(mock_get_schema):
    """Test detailed validation returns error info for invalid instance."""
    mock_get_schema.return_value = {"type": "object", "required": ["foo"]}
    instance = {}
    result = validate_instance_detailed(instance, "any_schema")
    assert result["valid"] is False
    assert len(result["errors"]) > 0
    assert "is a required property" in result["message"]


def test_validate_schema_valid(tmp_path: Path):
    """Test validation of a valid schema file."""
    # Create a valid JSON Schema Draft 2020-12 schema
    valid_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/test.schema.json",
        "title": "Test Schema",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(valid_schema))

    result = validate_schema(schema_path)
    assert result is True


def test_validate_schema_missing_required_fields(tmp_path: Path):
    """Test schema validation fails for missing required fields."""
    # Schema missing $schema and $id
    invalid_schema = {
        "title": "Test Schema",
        "type": "object",
    }

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(invalid_schema))

    result = validate_schema(schema_path)
    assert result is False


def test_validate_schema_file_not_found():
    """Test schema validation raises error for missing file."""
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        validate_schema("/nonexistent/path/to/schema.json")


def test_validate_schema_invalid_json(tmp_path: Path):
    """Test schema validation raises error for invalid JSON."""
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{ invalid json }")

    with pytest.raises(ValueError, match="Invalid JSON in schema file"):
        validate_schema(schema_path)


def test_validate_schema_invalid_structure(tmp_path: Path):
    """Test schema validation fails for invalid schema structure."""
    # Schema with required fields but invalid structure
    invalid_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/test.schema.json",
        "type": "invalid_type",  # Invalid type
    }

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(invalid_schema))

    result = validate_schema(schema_path)
    assert result is False


def test_validate_schema_string_path(tmp_path: Path):
    """Test schema validation works with string path."""
    valid_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/test.schema.json",
        "type": "object",
    }

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(valid_schema))

    result = validate_schema(str(schema_path))  # Pass string, not Path
    assert result is True
