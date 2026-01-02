"""Pydantic models for pipeline artifacts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# Non-empty string type for list items
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class Scope(BaseModel):
    """Story scope constraints."""

    target_word_count: int = Field(ge=1000, description="Approximate final length")
    estimated_passages: int = Field(ge=5, description="Target scene count")
    branching_depth: Literal["light", "moderate", "heavy"] = Field(
        default="moderate", description="Branching complexity"
    )
    estimated_playtime_minutes: int | None = Field(
        default=None, ge=1, description="Target reading time"
    )


class ContentNotes(BaseModel):
    """Content boundaries for the story."""

    includes: list[NonEmptyStr] = Field(default_factory=list, description="Content included")
    excludes: list[NonEmptyStr] = Field(default_factory=list, description="Content excluded")


class DreamArtifact(BaseModel):
    """DREAM stage artifact - creative vision for the story."""

    type: Literal["dream"] = "dream"
    version: int = Field(default=1, ge=1)

    # Core creative direction
    genre: str = Field(min_length=1, description="Primary genre")
    subgenre: str | None = Field(default=None, min_length=1, description="Optional refinement")
    tone: list[NonEmptyStr] = Field(min_length=1, description="Tone descriptors")
    audience: Literal["adult", "young_adult", "all_ages"] = Field(description="Target audience")
    themes: list[NonEmptyStr] = Field(min_length=1, description="Thematic elements")

    # Optional fields
    style_notes: str | None = Field(default=None, min_length=1, description="Style guidance")
    scope: Scope | None = Field(default=None, description="Scope constraints")
    content_notes: ContentNotes | None = Field(default=None, description="Content boundaries")


# Type alias for artifact types
ArtifactType = DreamArtifact
