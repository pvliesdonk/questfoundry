"""Artifact reading, writing, and validation."""

from questfoundry.artifacts.generated import (
    ArtifactType,
    ContentNotes,
    DreamArtifact,
    Scope,
)
from questfoundry.artifacts.reader import (
    ArtifactNotFoundError,
    ArtifactParseError,
    ArtifactReader,
)
from questfoundry.artifacts.validator import (
    ArtifactValidationError,
    ArtifactValidator,
    SchemaNotFoundError,
    ValidationErrorDetail,
    get_all_field_paths,
    pydantic_errors_to_details,
)
from questfoundry.artifacts.writer import ArtifactWriteError, ArtifactWriter

__all__ = [
    "ArtifactNotFoundError",
    "ArtifactParseError",
    "ArtifactReader",
    "ArtifactType",
    "ArtifactValidationError",
    "ArtifactValidator",
    "ArtifactWriteError",
    "ArtifactWriter",
    "ContentNotes",
    "DreamArtifact",
    "SchemaNotFoundError",
    "Scope",
    "ValidationErrorDetail",
    "get_all_field_paths",
    "pydantic_errors_to_details",
]
