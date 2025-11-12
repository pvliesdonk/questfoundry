"""Enhanced schema validation with detailed error reporting"""

from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from ..utils.resources import get_schema


@dataclass
class ValidationError:
    """Represents a single validation error"""

    message: str
    path: list[str | int]
    schema_path: list[str]
    validator: str
    validator_value: Any = None

    def __str__(self) -> str:
        """Human-readable error message"""
        path_str = ".".join(str(p) for p in self.path) if self.path else "(root)"
        return f"At {path_str}: {self.message}"


@dataclass
class ValidationWarning:
    """Represents a validation warning (non-fatal)"""

    message: str
    path: list[str | int]

    def __str__(self) -> str:
        """Human-readable warning message"""
        path_str = ".".join(str(p) for p in self.path) if self.path else "(root)"
        return f"At {path_str}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating an instance against a schema"""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    schema_name: str = ""

    @property
    def error_count(self) -> int:
        """Number of errors"""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings"""
        return len(self.warnings)

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context"""
        return self.valid

    def format_errors(self) -> str:
        """Format all errors as a multi-line string"""
        if not self.errors:
            return "No errors"
        return "\n".join(f"  - {error}" for error in self.errors)

    def format_warnings(self) -> str:
        """Format all warnings as a multi-line string"""
        if not self.warnings:
            return "No warnings"
        return "\n".join(f"  - {warning}" for warning in self.warnings)


def validate_artifact(instance: dict[str, Any], schema_name: str) -> ValidationResult:
    """
    Validate an artifact instance against its schema with detailed error reporting.

    Args:
        instance: The artifact data to validate
        schema_name: Name of the schema (e.g., "hook_card", "tu_brief")

    Returns:
        ValidationResult with detailed error and warning information

    Example:
        >>> result = validate_artifact({"type": "hook_card"}, "hook_card")
        >>> if not result.valid:
        ...     print(result.format_errors())
    """
    schema = get_schema(schema_name)
    validator = Draft202012Validator(schema)

    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    # Collect all validation errors
    for error in validator.iter_errors(instance):
        validation_error = ValidationError(
            message=error.message,
            path=list(error.path),
            schema_path=list(error.schema_path),
            validator=error.validator,
            validator_value=error.validator_value,
        )
        errors.append(validation_error)

    # Check for additional warnings (non-standard fields, etc.)
    warnings.extend(_check_warnings(instance, schema))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        schema_name=schema_name,
    )


def _check_warnings(
    instance: dict[str, Any], schema: dict[str, Any]
) -> list[ValidationWarning]:
    """
    Check for potential issues that are warnings, not errors.

    Note: This performs basic property checking on top-level properties only.
    It does not handle complex schema keywords like allOf, anyOf, oneOf, or
    patternProperties. For comprehensive validation, rely on the error-level
    checks which use Draft202012Validator.
    """
    warnings: list[ValidationWarning] = []

    # Check for unknown properties if additionalProperties is false
    if schema.get("additionalProperties") is False:
        properties = schema.get("properties", {})
        for key in instance:
            if key not in properties:
                warnings.append(
                    ValidationWarning(
                        message=f"Unknown property '{key}' (will be ignored)",
                        path=[key],
                    )
                )

    return warnings


def validate_artifact_type(artifact: dict[str, Any]) -> ValidationResult:
    """
    Validate an artifact by auto-detecting its type.

    Args:
        artifact: Artifact with a "type" field

    Returns:
        ValidationResult after validating against the appropriate schema.
        If the artifact is missing a "type" field, returns an invalid result
        with an appropriate error.

    Raises:
        FileNotFoundError: If schema for artifact type doesn't exist
    """
    if "type" not in artifact:
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError(
                    message="Missing required field 'type'",
                    path=["type"],
                    schema_path=[],
                    validator="required",
                )
            ],
        )

    artifact_type = artifact["type"]
    return validate_artifact(artifact, artifact_type)
