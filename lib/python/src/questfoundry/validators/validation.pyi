"""Type stubs for validation module"""

from dataclasses import dataclass
from typing import Any

@dataclass
class ValidationError:
    """Represents a single validation error"""

    message: str
    path: list[str | int]
    schema_path: list[str]
    validator: str
    validator_value: Any
    def __str__(self) -> str: ...

@dataclass
class ValidationWarning:
    """Represents a validation warning (non-fatal)"""

    message: str
    path: list[str | int]
    def __str__(self) -> str: ...

@dataclass
class ValidationResult:
    """Result of validating an instance against a schema"""

    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationWarning]
    schema_name: str
    @property
    def error_count(self) -> int: ...
    @property
    def warning_count(self) -> int: ...
    def __bool__(self) -> bool: ...
    def format_errors(self) -> str: ...
    def format_warnings(self) -> str: ...

def validate_artifact(
    instance: dict[str, Any], schema_name: str
) -> ValidationResult: ...
def validate_artifact_type(artifact: dict[str, Any]) -> ValidationResult: ...
