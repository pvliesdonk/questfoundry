"""Tests for schema-first code generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def dream_schema() -> dict[str, Any]:
    """Load the DREAM artifact JSON schema."""
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "dream.schema.json"
    with schema_path.open() as f:
        return json.load(f)


@pytest.fixture
def simple_schema(tmp_path: Path) -> Path:
    """Create a minimal valid schema for testing."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Test Artifact",
        "description": "A test artifact for generation",
        "type": "object",
        "required": ["name", "count"],
        "properties": {
            "type": {"type": "string", "const": "test"},
            "version": {"type": "integer", "minimum": 1},
            "name": {"type": "string", "minLength": 1, "description": "Item name"},
            "count": {"type": "integer", "minimum": 0, "description": "Item count"},
            "tags": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "description": "Item tags",
            },
            "optional_note": {
                "type": "string",
                "description": "Optional note",
            },
        },
    }
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    schema_path = schema_dir / "test.schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    return tmp_path


@pytest.fixture
def nested_schema(tmp_path: Path) -> Path:
    """Create a schema with nested objects for testing."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Nested Artifact",
        "type": "object",
        "required": ["title"],
        "properties": {
            "type": {"type": "string", "const": "nested"},
            "version": {"type": "integer", "minimum": 1},
            "title": {"type": "string", "minLength": 1},
            "metadata": {
                "type": "object",
                "description": "Nested metadata object",
                "required": ["author"],
                "properties": {
                    "author": {"type": "string", "minLength": 1},
                    "created_year": {"type": "integer", "minimum": 1900},
                },
            },
        },
    }
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    schema_path = schema_dir / "nested.schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    return tmp_path


class TestGeneratorProducesValidPython:
    """Tests that generated code is valid Python."""

    def test_generated_file_is_importable(self) -> None:
        """Generated models file should be importable."""
        # Import should not raise
        from questfoundry.artifacts.generated import DreamArtifact, Scope

        assert DreamArtifact is not None
        assert Scope is not None

    def test_generated_models_can_be_instantiated(self) -> None:
        """Generated models should be instantiable with valid data."""
        from questfoundry.artifacts.generated import DreamArtifact

        artifact = DreamArtifact(
            genre="mystery",
            tone=["dark"],
            audience="adult",
            themes=["betrayal"],
        )
        assert artifact.genre == "mystery"
        assert artifact.type == "dream"

    def test_generated_nested_model_works(self) -> None:
        """Nested models (Scope) should work correctly."""
        from questfoundry.artifacts.generated import DreamArtifact, Scope

        artifact = DreamArtifact(
            genre="mystery",
            tone=["dark"],
            audience="adult",
            themes=["betrayal"],
            scope=Scope(target_word_count=10000, estimated_passages=20),
        )
        assert artifact.scope is not None
        assert artifact.scope.target_word_count == 10000


class TestGeneratedModelsMatchSchema:
    """Tests that generated models align with schema definitions."""

    def test_required_fields_match_schema(self, dream_schema: dict[str, Any]) -> None:
        """Generated model required fields should match schema."""
        from questfoundry.artifacts.generated import DreamArtifact

        schema_required = set(dream_schema.get("required", []))
        model_required = {
            name for name, field in DreamArtifact.model_fields.items() if field.is_required()
        }

        # Model should require at least what schema requires
        missing = schema_required - model_required
        # Exclude type and version since they have defaults
        missing = missing - {"type", "version"}
        assert not missing, f"Schema requires {missing} but model doesn't"

    def test_scope_required_fields_match(self, dream_schema: dict[str, Any]) -> None:
        """Scope model required fields should match schema."""
        from questfoundry.artifacts.generated import Scope

        scope_schema = dream_schema.get("properties", {}).get("scope", {})
        schema_required = set(scope_schema.get("required", []))
        model_required = {name for name, field in Scope.model_fields.items() if field.is_required()}

        assert schema_required == model_required, (
            f"Schema scope.required={schema_required}, model required={model_required}"
        )


class TestGeneratedModelsHaveConstraints:
    """Tests that Pydantic Field constraints are preserved."""

    def test_string_min_length_constraint(self) -> None:
        """String fields with minLength should have min_length constraint."""
        from pydantic import ValidationError

        from questfoundry.artifacts.generated import DreamArtifact

        with pytest.raises(ValidationError) as exc_info:
            DreamArtifact(
                genre="",  # Empty string violates minLength=1
                tone=["dark"],
                audience="adult",
                themes=["test"],
            )
        assert "genre" in str(exc_info.value)

    def test_integer_minimum_constraint(self) -> None:
        """Integer fields with minimum should have ge constraint."""
        from pydantic import ValidationError

        from questfoundry.artifacts.generated import Scope

        with pytest.raises(ValidationError) as exc_info:
            Scope(
                target_word_count=500,  # Below 1000 minimum
                estimated_passages=10,
            )
        assert "target_word_count" in str(exc_info.value)

    def test_array_min_items_constraint(self) -> None:
        """Array fields with minItems should have min_length constraint."""
        from pydantic import ValidationError

        from questfoundry.artifacts.generated import DreamArtifact

        with pytest.raises(ValidationError) as exc_info:
            DreamArtifact(
                genre="mystery",
                tone=[],  # Empty array violates minItems=1
                audience="adult",
                themes=["test"],
            )
        assert "tone" in str(exc_info.value)

    def test_array_item_min_length_constraint(self) -> None:
        """Array items with minLength should validate each item."""
        from pydantic import ValidationError

        from questfoundry.artifacts.generated import DreamArtifact

        with pytest.raises(ValidationError) as exc_info:
            DreamArtifact(
                genre="mystery",
                tone=[""],  # Empty string in array
                audience="adult",
                themes=["test"],
            )
        assert "tone" in str(exc_info.value)


class TestGeneratorHandlesNestedObjects:
    """Tests for nested object handling."""

    def test_nested_object_generates_separate_class(self) -> None:
        """Nested objects should generate separate Pydantic classes."""
        from questfoundry.artifacts.generated import ContentNotes, Scope

        # Both should be importable as separate classes
        assert Scope.__name__ == "Scope"
        assert ContentNotes.__name__ == "ContentNotes"

    def test_nested_object_default_none(self) -> None:
        """Optional nested objects should default to None."""
        from questfoundry.artifacts.generated import DreamArtifact

        artifact = DreamArtifact(
            genre="test",
            tone=["dark"],
            audience="adult",
            themes=["test"],
        )
        assert artifact.scope is None
        assert artifact.content_notes is None


class TestGeneratorIsDeterministic:
    """Tests that generation is deterministic."""

    def test_same_input_produces_same_output(self) -> None:
        """Running generator twice should produce identical output."""
        # Run generator
        result1 = subprocess.run(
            [sys.executable, "scripts/generate_models.py"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result1.returncode == 0

        # Read generated file
        generated_path = (
            Path(__file__).parent.parent.parent / "src/questfoundry/artifacts/generated.py"
        )
        content1 = generated_path.read_text()

        # Run generator again
        result2 = subprocess.run(
            [sys.executable, "scripts/generate_models.py"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result2.returncode == 0

        # Read generated file again
        content2 = generated_path.read_text()

        assert content1 == content2, "Generator output is not deterministic"


class TestGeneratorErrorHandling:
    """Tests for error handling in generator."""

    def test_generator_fails_on_invalid_json(self, tmp_path: Path) -> None:
        """Generator should fail gracefully on invalid JSON."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema_path = schema_dir / "bad.schema.json"
        schema_path.write_text("{invalid json")
        output_file = tmp_path / "generated.py"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_models.py",
                "--schemas-dir",
                str(schema_dir),
                "--output-file",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 1, "Script should fail on invalid JSON"
        assert "Error processing" in result.stderr
        assert "bad.schema.json" in result.stderr

    def test_generator_fails_on_unsupported_type(self, tmp_path: Path) -> None:
        """Generator should fail on unsupported JSON Schema types."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["enabled"],
            "properties": {
                "enabled": {"type": "boolean", "description": "Flag"},  # Unsupported
            },
        }
        schema_path = schema_dir / "test.schema.json"
        schema_path.write_text(json.dumps(schema))
        output_file = tmp_path / "generated.py"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_models.py",
                "--schemas-dir",
                str(schema_dir),
                "--output-file",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 1, "Script should fail on unsupported type"
        assert "Unsupported JSON Schema type" in result.stderr
        assert "boolean" in result.stderr


class TestGeneratorHandlesDefaultValues:
    """Tests for default value handling in generator."""

    def test_integer_default_from_schema(self) -> None:
        """Integer fields with default should use that default."""
        from questfoundry.artifacts.generated import DreamArtifact

        artifact = DreamArtifact(
            genre="mystery",
            tone=["dark"],
            audience="adult",
            themes=["betrayal"],
        )
        # version should default to 1 per schema
        assert artifact.version == 1

    def test_string_default_from_schema(self) -> None:
        """String fields with default should use that default."""
        from questfoundry.artifacts.generated import Scope

        scope = Scope(
            target_word_count=10000,
            estimated_passages=20,
        )
        # branching_depth should default to "moderate" per schema
        assert scope.branching_depth == "moderate"

    def test_default_values_are_overridable(self) -> None:
        """Default values should be overridable by explicit values."""
        from questfoundry.artifacts.generated import DreamArtifact, Scope

        artifact = DreamArtifact(
            genre="mystery",
            tone=["dark"],
            audience="adult",
            themes=["betrayal"],
            version=2,
        )
        assert artifact.version == 2

        scope = Scope(
            target_word_count=10000,
            estimated_passages=20,
            branching_depth="heavy",
        )
        assert scope.branching_depth == "heavy"

    def test_generator_produces_correct_default_syntax(self, tmp_path: Path) -> None:
        """Generator should produce correct Python default syntax."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "default": 42},
                "label": {"type": "string", "default": "default_label"},
            },
        }
        schema_path = schema_dir / "defaults.schema.json"
        schema_path.write_text(json.dumps(schema))
        output_file = tmp_path / "generated.py"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_models.py",
                "--schemas-dir",
                str(schema_dir),
                "--output-file",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, f"Generator failed: {result.stderr}"
        content = output_file.read_text()
        # Check that defaults are generated correctly
        assert "default=42" in content
        assert 'default="default_label"' in content
