"""Tests for validator helper functions."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from questfoundry.artifacts.validator import (
    _get_nested_value,
    _path_to_field_name,
    pydantic_errors_to_details,
)
from questfoundry.conversation import ValidationErrorDetail

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
        """Missing required field produces correct error detail."""
        data = {"age": 25}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].field == "name"
        assert "required" in details[0].issue.lower() or "missing" in details[0].issue.lower()
        assert details[0].provided is None

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
        """Constraint violation includes provided value."""
        data = {"name": "Test", "age": -5}
        try:
            SampleModel.model_validate(data)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            details = pydantic_errors_to_details(e.errors(), data)

        assert len(details) == 1
        assert details[0].field == "age"
        assert details[0].provided == -5

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
        """List item validation errors reference parent field."""
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

        # Should reference "tone" without index
        assert any(d.field == "tone" for d in details)
