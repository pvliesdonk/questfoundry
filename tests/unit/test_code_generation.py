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


class TestGeneratorHandlesConstValues:
    """Tests for const value handling in generator."""

    def test_string_const_generates_literal(self) -> None:
        """String const should generate Literal type."""
        from questfoundry.artifacts.generated import DreamArtifact

        # type field is a const="dream"
        artifact = DreamArtifact(
            genre="test",
            tone=["dark"],
            audience="adult",
            themes=["test"],
        )
        assert artifact.type == "dream"

    def test_integer_const_generates_correctly(self, tmp_path: Path) -> None:
        """Integer const should generate Literal[123] not Literal['123']."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["version"],
            "properties": {
                "version": {"const": 42},
            },
        }
        schema_path = schema_dir / "intconst.schema.json"
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
        # Should be Literal[42] not Literal["42"]
        assert "Literal[42]" in content
        assert "= 42" in content


class TestGeneratorHandlesPythonKeywords:
    """Tests for Python keyword handling in generator."""

    def test_keyword_field_gets_alias(self, tmp_path: Path) -> None:
        """Fields named after Python keywords should get aliases."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["from", "class"],
            "properties": {
                "from": {"type": "string", "minLength": 1, "description": "Source"},
                "class": {"type": "string", "minLength": 1, "description": "Category"},
            },
        }
        schema_path = schema_dir / "keywords.schema.json"
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
        # Should have from_ and class_ with aliases
        assert "from_:" in content
        assert 'alias="from"' in content
        assert "class_:" in content
        assert 'alias="class"' in content


class TestEscapeDescription:
    """Tests for escape_description function."""

    def test_escapes_special_characters(self, tmp_path: Path) -> None:
        """Description with special chars should be escaped."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "description": 'Has "quotes" and\nnewlines and\ttabs',
                },
            },
        }
        schema_path = schema_dir / "escape.schema.json"
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
        # Should have escaped newlines and tabs (ruff may use single quotes for strings)
        assert "\\n" in content
        assert "\\t" in content
        # The quotes should be present (either escaped or in single-quoted string)
        assert "quotes" in content


class TestGeneratorArrayValidation:
    """Tests for array field validation in generator."""

    def test_array_without_items_type_fails(self, tmp_path: Path) -> None:
        """Array fields missing items.type should raise an error."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["tags"],
            "properties": {
                "tags": {
                    "type": "array",
                    "description": "Tags without type",
                    # Missing "items" entirely
                },
            },
        }
        schema_path = schema_dir / "noitems.schema.json"
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

        assert result.returncode == 1, "Generator should fail on missing items.type"
        assert "items.type" in result.stderr
        assert "tags" in result.stderr

    def test_array_with_empty_items_fails(self, tmp_path: Path) -> None:
        """Array fields with empty items object should raise an error."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["tags"],
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {},  # Empty items - no type specified
                    "description": "Tags with empty items",
                },
            },
        }
        schema_path = schema_dir / "emptyitems.schema.json"
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

        assert result.returncode == 1, "Generator should fail on empty items"
        assert "items.type" in result.stderr


class TestGeneratorEmptyProperties:
    """Tests for empty properties handling in generator."""

    def test_empty_properties_generates_valid_class(self, tmp_path: Path) -> None:
        """Schema with empty properties should generate valid Python class with pass."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Empty Artifact",
            "description": "An artifact with no properties",
            "type": "object",
            "properties": {},  # Empty properties
        }
        schema_path = schema_dir / "empty.schema.json"
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
        # Should have pass statement for empty class body
        assert "pass" in content
        # Verify it's valid Python by trying to import it
        assert "class EmptyArtifact" in content

    def test_nested_empty_properties_generates_valid_class(self, tmp_path: Path) -> None:
        """Nested object with empty properties should generate valid Python class."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Parent Artifact",
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "metadata": {
                    "type": "object",
                    "description": "Empty metadata object",
                    "properties": {},  # Empty nested properties
                },
            },
        }
        schema_path = schema_dir / "nestedempty.schema.json"
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
        # Nested empty class should have pass
        assert "class Metadata" in content
        assert "pass" in content


class TestSnakeToPascal:
    """Tests for snake_to_pascal helper function."""

    def test_simple_conversion(self) -> None:
        """Simple snake_case should convert to PascalCase."""
        from scripts.generate_models import snake_to_pascal

        assert snake_to_pascal("content_notes") == "ContentNotes"
        assert snake_to_pascal("target_word_count") == "TargetWordCount"

    def test_single_word(self) -> None:
        """Single word should capitalize first letter."""
        from scripts.generate_models import snake_to_pascal

        assert snake_to_pascal("scope") == "Scope"
        assert snake_to_pascal("metadata") == "Metadata"

    def test_multiple_underscores(self) -> None:
        """Multiple underscores should work correctly."""
        from scripts.generate_models import snake_to_pascal

        assert snake_to_pascal("very_long_field_name") == "VeryLongFieldName"


class TestArtifactTypeNaming:
    """Tests for artifact type class naming."""

    def test_multi_word_artifact_type_uses_pascal_case(self, tmp_path: Path) -> None:
        """Multi-word artifact types should generate PascalCase class names."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Brain Storm Artifact",
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
            },
        }
        # Use underscore in filename to test snake_to_pascal
        schema_path = schema_dir / "brain_storm.schema.json"
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
        # Should be BrainStormArtifact, not Brain_stormArtifact
        assert "class BrainStormArtifact" in content
        assert "Brain_storm" not in content


class TestGeneratorArrayItemTypes:
    """Tests for array item type handling in generator."""

    def test_integer_array_items_generate_list_int(self, tmp_path: Path) -> None:
        """Array with integer items should generate list[int], not list[integer]."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["scores"],
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of scores",
                },
            },
        }
        schema_path = schema_dir / "intarray.schema.json"
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
        # Should generate list[int], not list[integer]
        assert "list[int]" in content
        assert "list[integer]" not in content

    def test_unsupported_array_item_type_fails(self, tmp_path: Path) -> None:
        """Array with unsupported item type should fail."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test",
            "type": "object",
            "required": ["flags"],
            "properties": {
                "flags": {
                    "type": "array",
                    "items": {"type": "boolean"},  # boolean not supported in array items
                    "description": "List of flags",
                },
            },
        }
        schema_path = schema_dir / "boolarray.schema.json"
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

        assert result.returncode == 1, "Generator should fail on unsupported array item type"
        assert "Unsupported array item type" in result.stderr
        assert "boolean" in result.stderr
