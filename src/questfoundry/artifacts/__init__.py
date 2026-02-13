"""Artifact validation and legacy reading."""

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
from questfoundry.models.dream import (
    ContentNotes,
    DreamArtifact,
    Scope,
)

# Legacy alias - DreamArtifact is the only artifact type currently
ArtifactType = DreamArtifact

__all__ = [
    "ArtifactNotFoundError",
    "ArtifactParseError",
    "ArtifactReader",
    "ArtifactType",
    "ArtifactValidationError",
    "ArtifactValidator",
    "ContentNotes",
    "DreamArtifact",
    "SchemaNotFoundError",
    "Scope",
    "ValidationErrorDetail",
    "get_all_field_paths",
    "pydantic_errors_to_details",
]
