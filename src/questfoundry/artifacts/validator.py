"""Artifact validation using JSON Schema and Pydantic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
from pydantic import BaseModel, ValidationError

from questfoundry.artifacts import DreamArtifact

if TYPE_CHECKING:
    from pydantic_core import ErrorDetails

    from questfoundry.conversation import ValidationErrorDetail


class SchemaNotFoundError(Exception):
    """Raised when a JSON schema file doesn't exist."""

    def __init__(self, schema_name: str, path: Path) -> None:
        self.schema_name = schema_name
        self.path = path
        super().__init__(f"Schema not found: {schema_name} at {path}")


class ArtifactValidationError(Exception):
    """Raised when artifact validation fails."""

    def __init__(self, stage_name: str, errors: list[str]) -> None:
        self.stage_name = stage_name
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(f"Validation failed for {stage_name}:\n  - {error_list}")


# Mapping from stage names to Pydantic models
STAGE_MODELS: dict[str, type[BaseModel]] = {
    "dream": DreamArtifact,
}


class ArtifactValidator:
    """Validate artifacts using JSON Schema and Pydantic models."""

    def __init__(self, schemas_path: Path | None = None) -> None:
        """Initialize validator.

        Args:
            schemas_path: Path to the schemas directory. If None, uses
                the schemas directory in the package root.
        """
        if schemas_path is None:
            # Default to project root schemas directory
            self.schemas_path = Path(__file__).parent.parent.parent.parent / "schemas"
        else:
            self.schemas_path = schemas_path

        self._schema_cache: dict[str, dict[str, object]] = {}

    def _load_schema(self, stage_name: str) -> dict[str, object]:
        """Load a JSON schema from disk.

        Args:
            stage_name: Name of the stage.

        Returns:
            The JSON schema as a dictionary.

        Raises:
            SchemaNotFoundError: If the schema file doesn't exist.
        """
        if stage_name in self._schema_cache:
            return self._schema_cache[stage_name]

        schema_path = self.schemas_path / f"{stage_name}.schema.json"

        if not schema_path.exists():
            raise SchemaNotFoundError(stage_name, schema_path)

        with schema_path.open("r", encoding="utf-8") as f:
            schema: dict[str, object] = json.load(f)

        self._schema_cache[stage_name] = schema
        return schema

    def validate_with_schema(self, data: dict[str, Any], stage_name: str) -> list[str]:
        """Validate data against a JSON schema.

        Args:
            data: The artifact data to validate.
            stage_name: Name of the stage.

        Returns:
            List of validation error messages (empty if valid).
        """
        try:
            schema = self._load_schema(stage_name)
        except SchemaNotFoundError:
            # No schema file, skip JSON Schema validation
            return []

        errors: list[str] = []
        validator = jsonschema.Draft7Validator(schema)

        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.absolute_path)
            if path:
                errors.append(f"{path}: {error.message}")
            else:
                errors.append(error.message)

        return errors

    def validate_with_model(self, data: dict[str, Any], stage_name: str) -> list[str]:
        """Validate data against a Pydantic model.

        Args:
            data: The artifact data to validate.
            stage_name: Name of the stage.

        Returns:
            List of validation error messages (empty if valid).
        """
        model = STAGE_MODELS.get(stage_name)
        if model is None:
            return []

        errors: list[str] = []

        try:
            model.model_validate(data)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(part) for part in error["loc"])
                msg = error["msg"]
                if loc:
                    errors.append(f"{loc}: {msg}")
                else:
                    errors.append(msg)

        return errors

    def validate(
        self, data: dict[str, Any], stage_name: str, *, raise_on_error: bool = False
    ) -> list[str]:
        """Validate artifact data using both JSON Schema and Pydantic.

        Args:
            data: The artifact data to validate.
            stage_name: Name of the stage.
            raise_on_error: If True, raise ArtifactValidationError on failure.

        Returns:
            List of all validation error messages (empty if valid).

        Raises:
            ArtifactValidationError: If raise_on_error is True and validation fails.
        """
        errors: list[str] = []

        # Validate with JSON Schema
        errors.extend(self.validate_with_schema(data, stage_name))

        # Validate with Pydantic model
        errors.extend(self.validate_with_model(data, stage_name))

        if errors and raise_on_error:
            raise ArtifactValidationError(stage_name, errors)

        return errors

    def is_valid(self, data: dict[str, Any], stage_name: str) -> bool:
        """Check if artifact data is valid.

        More efficient than validate() as it stops at the first error.

        Args:
            data: The artifact data to validate.
            stage_name: Name of the stage.

        Returns:
            True if the artifact is valid.
        """
        # Validate with JSON Schema first
        try:
            schema = self._load_schema(stage_name)
            validator = jsonschema.Draft7Validator(schema)
            if not validator.is_valid(data):
                return False
        except SchemaNotFoundError:
            pass  # No schema file, skip JSON Schema validation

        # Validate with Pydantic model
        model = STAGE_MODELS.get(stage_name)
        if model:
            try:
                model.model_validate(data)
            except ValidationError:
                return False

        return True


def _get_nested_value(data: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    """Get a value from nested data using a path tuple.

    Traverses the data structure following the path. Returns None if any
    key is missing or if a None value is encountered mid-path.

    Note:
        This function cannot distinguish between a missing key and an
        explicitly provided None value - both return None. For LLM
        feedback purposes, this is acceptable since both cases indicate
        "no valid value was provided".

    Args:
        data: The data dictionary to traverse.
        path: Tuple of keys/indices from Pydantic error location.

    Returns:
        The value at the path, or None if path doesn't exist or
        contains None values.

    Example:
        >>> _get_nested_value({"scope": {"count": 5}}, ("scope", "count"))
        5
        >>> _get_nested_value({"items": ["a", "b"]}, ("items", 1))
        'b'
        >>> _get_nested_value({"x": None}, ("x", "y"))
        None
    """
    current: Any = data
    for key in path:
        if current is None:
            return None
        if isinstance(key, int):
            # Traverse into list by index
            if isinstance(current, list) and key < len(current):
                current = current[key]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _path_to_field_name(path: tuple[str | int, ...]) -> str:
    """Convert Pydantic error path to field name string.

    Strips list indices to produce cleaner field names for LLM feedback.
    When multiple items in a list have errors, they will all reference
    the parent field name (e.g., "themes") with different `provided` values,
    rather than showing indices like "themes.0", "themes.1".

    This design choice prioritizes LLM comprehension - the model should
    focus on fixing the field content, not navigating array indices.

    Args:
        path: Tuple of keys/indices from Pydantic error location.

    Returns:
        Dot-separated field path string, or "(root)" for empty paths.

    Example:
        >>> _path_to_field_name(("scope", "target_word_count"))
        'scope.target_word_count'
        >>> _path_to_field_name(("themes", 0))
        'themes'
        >>> _path_to_field_name(("items", 2, "name"))
        'items.name'
        >>> _path_to_field_name(())
        '(root)'
    """
    # Filter out integer indices for cleaner field names
    str_parts = [str(p) for p in path if not isinstance(p, int)]
    return ".".join(str_parts) if str_parts else "(root)"


def pydantic_errors_to_details(
    errors: list[ErrorDetails],
    data: dict[str, Any],
) -> list[ValidationErrorDetail]:
    """Convert Pydantic ValidationError details to structured ValidationErrorDetail list.

    Args:
        errors: List of error dicts from ValidationError.errors().
        data: The original data that was validated (for extracting provided values).

    Returns:
        List of ValidationErrorDetail with field, issue, and provided value.

    Example:
        >>> from pydantic import ValidationError
        >>> try:
        ...     DreamArtifact.model_validate({"genre": ""})
        ... except ValidationError as e:
        ...     details = pydantic_errors_to_details(e.errors(), {"genre": ""})
        ...     print(details[0].field, details[0].provided)
        genre
    """
    from questfoundry.conversation import ValidationErrorDetail

    result: list[ValidationErrorDetail] = []

    for error in errors:
        path = error["loc"]
        field = _path_to_field_name(path)
        issue = error["msg"]
        provided = _get_nested_value(data, path)

        result.append(ValidationErrorDetail(field=field, issue=issue, provided=provided))

    return result
