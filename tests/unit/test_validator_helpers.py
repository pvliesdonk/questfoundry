"""Tests for validator helper functions."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from questfoundry.artifacts.validator import (
    _generate_requirement_text,
    _get_nested_value,
    _path_to_field_name,
    get_all_field_paths,
    pydantic_errors_to_details,
    strip_null_values,
)
from questfoundry.conversation import ValidationErrorDetail

# --- strip_null_values Tests ---


class TestStripNullValues:
    """Tests for strip_null_values function."""

    def test_strips_top_level_nulls(self) -> None:
        """Null values at top level are removed."""
        data = {"genre": "fantasy", "subgenre": None, "audience": "adult"}
        result = strip_null_values(data)
        assert result == {"genre": "fantasy", "audience": "adult"}
        assert "subgenre" not in result

    def test_preserves_non_null_values(self) -> None:
        """Non-null values are preserved."""
        data = {"genre": "fantasy", "age": 0, "active": False, "items": []}
        result = strip_null_values(data)
        assert result == data

    def test_strips_nested_nulls(self) -> None:
        """Null values in nested dicts are removed."""
        data = {
            "scope": {
                "target_word_count": 5000,
                "estimated_playtime_minutes": None,
            }
        }
        result = strip_null_values(data)
        assert result == {"scope": {"target_word_count": 5000}}

    def test_preserves_empty_nested_dicts(self) -> None:
        """Empty nested dicts are preserved for schema validation to handle."""
        data = {
            "genre": "fantasy",
            "scope": {"optional_field": None},
        }
        result = strip_null_values(data)
        # Empty dicts preserved - let schema validation report "missing required field"
        assert result == {"genre": "fantasy", "scope": {}}

    def test_strips_nulls_in_list_items(self) -> None:
        """Nulls in dicts inside lists are stripped."""
        data = {"items": [{"name": "A", "value": None}, {"name": "B", "value": 5}]}
        result = strip_null_values(data)
        assert result == {"items": [{"name": "A"}, {"name": "B", "value": 5}]}

    def test_preserves_empty_lists(self) -> None:
        """Empty lists are preserved (not considered null)."""
        data = {"themes": [], "genre": "fantasy"}
        result = strip_null_values(data)
        assert result == {"themes": [], "genre": "fantasy"}

    def test_preserves_zero_and_false(self) -> None:
        """Zero and False are preserved (falsy but not null)."""
        data = {"count": 0, "active": False, "name": ""}
        result = strip_null_values(data)
        assert result == {"count": 0, "active": False, "name": ""}

    def test_deeply_nested_nulls(self) -> None:
        """Nulls in deeply nested structures are stripped."""
        data = {
            "level1": {
                "level2": {
                    "keep": "value",
                    "strip": None,
                }
            }
        }
        result = strip_null_values(data)
        assert result == {"level1": {"level2": {"keep": "value"}}}

    def test_all_nulls_returns_empty(self) -> None:
        """Dict with only null values returns empty dict."""
        data = {"a": None, "b": None}
        result = strip_null_values(data)
        assert result == {}

    def test_empty_dict_returns_empty(self) -> None:
        """Empty input returns empty output."""
        result = strip_null_values({})
        assert result == {}


# --- Helper Function Tests ---


class TestPathToFieldName:
    """Tests for _path_to_field_name helper."""

    def test_single_field(self) -> None:
        """Single field path."""
        assert _path_to_field_name(("genre",)) == "genre"

    def test_nested_field(self) -> None:
        """Nested field path with dot notation."""
        assert _path_to_field_name(("scope", "target_word_count")) == "scope.target_word_count"

    def test_strips_list_indices(self) -> None:
        """List indices are stripped from path."""
        assert _path_to_field_name(("themes", 0)) == "themes"
        assert _path_to_field_name(("items", 2, "name")) == "items.name"

    def test_empty_path(self) -> None:
        """Empty path returns root indicator."""
        assert _path_to_field_name(()) == "(root)"

    def test_only_indices(self) -> None:
        """Path with only indices returns root."""
        assert _path_to_field_name((0, 1)) == "(root)"


class TestGetNestedValue:
    """Tests for _get_nested_value helper."""

    def test_simple_field(self) -> None:
        """Get simple field value."""
        data = {"genre": "fantasy"}
        assert _get_nested_value(data, ("genre",)) == "fantasy"

    def test_nested_field(self) -> None:
        """Get nested field value."""
        data = {"scope": {"target_word_count": 5000}}
        assert _get_nested_value(data, ("scope", "target_word_count")) == 5000

    def test_list_index(self) -> None:
        """Get list item by index."""
        data = {"themes": ["adventure", "friendship"]}
        assert _get_nested_value(data, ("themes", 0)) == "adventure"
        assert _get_nested_value(data, ("themes", 1)) == "friendship"

    def test_missing_field(self) -> None:
        """Missing field returns None."""
        data = {"genre": "fantasy"}
        assert _get_nested_value(data, ("missing",)) is None

    def test_missing_nested_field(self) -> None:
        """Missing nested field returns None."""
        data = {"scope": {}}
        assert _get_nested_value(data, ("scope", "missing")) is None

    def test_none_in_path(self) -> None:
        """None value in path returns None."""
        data = {"scope": None}
        assert _get_nested_value(data, ("scope", "target_word_count")) is None

    def test_list_index_out_of_bounds(self) -> None:
        """Out of bounds list index returns None."""
        data = {"themes": ["one"]}
        assert _get_nested_value(data, ("themes", 5)) is None

    def test_empty_path(self) -> None:
        """Empty path returns the data itself."""
        data = {"genre": "fantasy"}
        assert _get_nested_value(data, ()) == data


# --- pydantic_errors_to_details Tests ---


class SampleModel(BaseModel):
    """Sample model for testing error conversion."""

    name: str = Field(min_length=1)
    age: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)


class NestedModel(BaseModel):
    """Model with nested objects for testing."""

    meta: SampleModel


class TestPydanticErrorsToDetails:
    """Tests for pydantic_errors_to_details function."""

    def test_missing_required_field(self) -> None:
        """Missing required field produces correct error detail with error_type."""
        data = {"age": 25}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].field == "name"
        assert details[0].provided is None
        # Verify error_type is captured for reliable categorization
        assert details[0].error_type == "missing"

    def test_invalid_value(self) -> None:
        """Invalid value produces correct error detail with provided value."""
        data = {"name": "", "age": 25}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].field == "name"
        assert details[0].provided == ""

    def test_constraint_violation(self) -> None:
        """Constraint violation includes provided value and error_type."""
        data = {"name": "Test", "age": -5}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].field == "age"
        assert details[0].provided == -5
        # Constraint violations have specific error types (not "missing")
        assert details[0].error_type == "greater_than_equal"

    def test_multiple_errors(self) -> None:
        """Multiple errors produce multiple details."""
        data = {"name": "", "age": -5}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 2
        fields = {d.field for d in details}
        assert "name" in fields
        assert "age" in fields

    def test_nested_error(self) -> None:
        """Nested field error produces dot-notation field path."""
        data = {"meta": {"name": "", "age": 25}}
        try:
            NestedModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].field == "meta.name"
        assert details[0].provided == ""

    def test_empty_errors_list(self) -> None:
        """Empty errors list produces empty details list."""
        details = pydantic_errors_to_details([], {})
        assert details == []

    def test_returns_validation_error_detail_instances(self) -> None:
        """Results are ValidationErrorDetail instances."""
        data = {"age": 25}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert all(isinstance(d, ValidationErrorDetail) for d in details)


# --- Integration with DreamArtifact ---


class TestDreamArtifactErrors:
    """Test error conversion with actual DreamArtifact model."""

    def test_dream_missing_required_fields(self) -> None:
        """Missing required DreamArtifact fields."""
        from questfoundry.artifacts import DreamArtifact

        data = {}
        try:
            DreamArtifact.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        fields = {d.field for d in details}
        # Required fields: genre, tone, audience, themes
        assert "genre" in fields
        assert "tone" in fields
        assert "audience" in fields
        assert "themes" in fields

    def test_dream_nested_scope_error(self) -> None:
        """Nested scope field errors have correct path."""
        from questfoundry.artifacts import DreamArtifact

        data = {
            "genre": "fantasy",
            "tone": ["epic"],
            "audience": "adult",
            "themes": ["heroism"],
            "scope": {"target_word_count": 100},  # Below minimum of 1000
        }
        try:
            DreamArtifact.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        # Should have error for scope.target_word_count and missing estimated_passages
        fields = {d.field for d in details}
        assert "scope.target_word_count" in fields or "scope.estimated_passages" in fields

    def test_dream_list_item_error(self) -> None:
        """List item validation errors reference parent field without indices."""
        from questfoundry.artifacts import DreamArtifact

        data = {
            "genre": "fantasy",
            "tone": ["epic", ""],  # Empty string not allowed
            "audience": "adult",
            "themes": ["heroism"],
        }
        try:
            DreamArtifact.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        # Get all errors related to "tone"
        tone_errors = [d for d in details if "tone" in d.field]
        assert len(tone_errors) >= 1, "Expected at least one tone error"

        # Verify indices are stripped (field should be exactly "tone", not "tone.1")
        for err in tone_errors:
            assert err.field == "tone", (
                f"Expected 'tone' but got '{err.field}' - indices should be stripped"
            )

        # Verify the provided value is captured (the empty string at index 1)
        assert any(err.provided == "" for err in tone_errors)


# --- _generate_requirement_text Tests ---


class TestGenerateRequirementText:
    """Tests for _generate_requirement_text helper."""

    def test_missing_error(self) -> None:
        """Missing field error produces 'required field' requirement."""
        error = {"type": "missing", "msg": "Field required", "loc": ("name",)}
        assert _generate_requirement_text(error) == "required field"

    def test_string_too_short(self) -> None:
        """String too short error includes min_length."""
        error = {
            "type": "string_too_short",
            "msg": "String should have at least 1 character",
            "loc": ("name",),
            "ctx": {"min_length": 1},
        }
        assert _generate_requirement_text(error) == "string with at least 1 character(s)"

    def test_string_too_short_custom_min(self) -> None:
        """String too short with custom min_length."""
        error = {
            "type": "string_too_short",
            "msg": "String should have at least 5 characters",
            "loc": ("name",),
            "ctx": {"min_length": 5},
        }
        assert _generate_requirement_text(error) == "string with at least 5 character(s)"

    def test_greater_than_equal(self) -> None:
        """Greater than or equal constraint."""
        error = {
            "type": "greater_than_equal",
            "msg": "Input should be greater than or equal to 1000",
            "loc": ("count",),
            "ctx": {"ge": 1000},
        }
        assert _generate_requirement_text(error) == "integer >= 1000"

    def test_less_than_equal(self) -> None:
        """Less than or equal constraint."""
        error = {
            "type": "less_than_equal",
            "msg": "Input should be less than or equal to 100",
            "loc": ("count",),
            "ctx": {"le": 100},
        }
        assert _generate_requirement_text(error) == "integer <= 100"

    def test_list_too_short(self) -> None:
        """List too short error."""
        error = {
            "type": "too_short",
            "msg": "List should have at least 1 item",
            "loc": ("items",),
            "ctx": {"min_length": 1},
        }
        assert _generate_requirement_text(error) == "array with at least 1 item(s)"

    def test_type_errors(self) -> None:
        """Type mismatch errors."""
        assert _generate_requirement_text({"type": "string_type", "msg": "x"}) == "must be a string"
        assert _generate_requirement_text({"type": "int_type", "msg": "x"}) == "must be an integer"
        assert _generate_requirement_text({"type": "list_type", "msg": "x"}) == "must be an array"
        assert _generate_requirement_text({"type": "dict_type", "msg": "x"}) == "must be an object"

    def test_fallback_to_message(self) -> None:
        """Unknown error type falls back to error message."""
        error = {"type": "unknown_type", "msg": "Some custom error message", "loc": ("x",)}
        assert _generate_requirement_text(error) == "Some custom error message"

    def test_fallback_when_no_message(self) -> None:
        """Falls back to 'see tool definition' when no message."""
        error = {"type": "unknown_type", "loc": ("x",)}
        assert _generate_requirement_text(error) == "see tool definition"


# --- get_all_field_paths Tests ---


class TestGetAllFieldPaths:
    """Tests for get_all_field_paths function."""

    def test_simple_model(self) -> None:
        """Simple model with flat fields."""
        paths = get_all_field_paths(SampleModel)
        assert paths == {"name", "age", "tags"}

    def test_nested_model(self) -> None:
        """Nested model includes parent and nested paths."""
        paths = get_all_field_paths(NestedModel)
        # Should include: meta, meta.name, meta.age, meta.tags
        assert "meta" in paths
        assert "meta.name" in paths
        assert "meta.age" in paths
        assert "meta.tags" in paths

    def test_returns_set(self) -> None:
        """Returns a set, not a list."""
        paths = get_all_field_paths(SampleModel)
        assert isinstance(paths, set)

    def test_dream_artifact_fields(self) -> None:
        """DreamArtifact includes expected fields."""
        from questfoundry.artifacts import DreamArtifact

        paths = get_all_field_paths(DreamArtifact)

        # Top-level required fields
        assert "genre" in paths
        assert "tone" in paths
        assert "audience" in paths
        assert "themes" in paths

        # Optional fields
        assert "subgenre" in paths
        assert "style_notes" in paths
        assert "scope" in paths
        assert "content_notes" in paths

        # Nested scope fields
        assert "scope.target_word_count" in paths
        assert "scope.estimated_passages" in paths
        assert "scope.branching_depth" in paths

        # Nested content_notes fields
        assert "content_notes.includes" in paths
        assert "content_notes.excludes" in paths

    def test_with_prefix(self) -> None:
        """Prefix is applied to all paths."""
        paths = get_all_field_paths(SampleModel, prefix="nested")
        assert "nested.name" in paths
        assert "nested.age" in paths
        assert "nested.tags" in paths


# --- pydantic_errors_to_details requirement field Tests ---


class TestPydanticErrorsRequirement:
    """Tests that pydantic_errors_to_details includes requirement field."""

    def test_missing_field_has_requirement(self) -> None:
        """Missing field error includes requirement text."""
        data = {"age": 25}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].requirement == "required field"

    def test_constraint_has_requirement(self) -> None:
        """Constraint violation includes requirement text."""
        data = {"name": "Test", "age": -5}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].requirement == "integer >= 0"

    def test_string_too_short_has_requirement(self) -> None:
        """String too short includes min_length in requirement."""
        data = {"name": "", "age": 25}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].requirement == "string with at least 1 character(s)"
