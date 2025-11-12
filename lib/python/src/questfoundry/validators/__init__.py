"""
Schema and artifact validation utilities.

This module provides validation functions for QuestFoundry artifacts and schemas,
including JSON schema validation, artifact type validation, and structured
validation results with errors and warnings.

Typical usage:
    >>> from questfoundry.validators import validate_artifact
    >>> result = validate_artifact(artifact_data, artifact_type="manuscript_section")
    >>> if not result.is_valid:
    ...     print(result.errors)
"""

from .schema import validate_instance, validate_schema
from .validation import (
    ValidationError,
    ValidationResult,
    ValidationWarning,
    validate_artifact,
    validate_artifact_type,
)

__all__ = [
    "validate_instance",
    "validate_schema",
    "validate_artifact",
    "validate_artifact_type",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
]
