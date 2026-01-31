"""Pydantic models for DREAM stage output.

These models define the structured output the LLM produces during the
DREAM serialize phase. The ontology is defined in docs/design/00-spec.md;
these models are the implementation used for validation.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class ContentNotes(BaseModel):
    """Content advisory notes for inclusions and exclusions."""

    excludes: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        default_factory=list, description="Content excluded"
    )
    includes: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        default_factory=list, description="Content included"
    )


class Scope(BaseModel):
    """Story scope constraints."""

    story_size: Literal["vignette", "short", "standard", "long"] = Field(
        default="standard",
        description=(
            'Story size preset: "vignette" (5-15 passages, tight single-thread), '
            '"short" (15-30 passages, modest branching), '
            '"standard" (30-60 passages, full branching), '
            '"long" (60-120 passages, extensive branching)'
        ),
    )
    branching_depth: str = Field(
        default="moderate",
        description="Branching complexity (e.g., light, moderate, heavy, extensive)",
        min_length=1,
    )
    estimated_passages: int = Field(default=45, description="Target scene count", ge=5)
    estimated_playtime_minutes: int | None = Field(
        default=None, description="Target reading time", ge=1
    )
    target_word_count: int = Field(default=20000, description="Approximate final length", ge=1000)


class DreamArtifact(BaseModel):
    """DREAM stage output - creative vision and constraints for the story."""

    audience: str = Field(
        description="Target audience (e.g., adult, young adult, all ages, mature)", min_length=1
    )
    content_notes: ContentNotes | None = Field(
        default=None, description="Content advisory notes for inclusions and exclusions"
    )
    genre: str = Field(description="Primary genre", min_length=1)
    scope: Scope = Field(default_factory=Scope)
    style_notes: str | None = Field(default=None, description="Style guidance", min_length=1)
    subgenre: str | None = Field(
        default=None, description="Optional genre refinement", min_length=1
    )
    themes: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        description="Thematic elements", min_length=1
    )
    tone: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        description="Tone descriptors", min_length=1
    )
    type: Literal["dream"] = "dream"
    version: int = Field(default=1, description="Schema version number", ge=1)
