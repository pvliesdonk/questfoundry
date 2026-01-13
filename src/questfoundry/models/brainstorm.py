"""Pydantic models for BRAINSTORM stage output.

BRAINSTORM is the expansive exploration phase that generates raw creative
material: entities (characters, locations, objects, factions) and tensions
(binary dramatic questions with two alternatives each).

See docs/design/00-spec.md and docs/design/procedures/brainstorm.md for details.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

EntityType = Literal["character", "location", "object", "faction"]


class Entity(BaseModel):
    """A story entity: character, location, object, or faction.

    Entities are the raw building blocks generated during BRAINSTORM.
    SEED will later decide which to retain, cut, or modify.

    Attributes:
        id: Short identifier (e.g., "kay", "mentor", "archive").
        type: Entity category.
        concept: One-line essence capturing what makes it interesting.
        notes: Freeform context from discussion (rich detail allowed).
    """

    id: str = Field(min_length=1, description="Short identifier for the entity")
    type: EntityType = Field(description="Entity category")
    concept: str = Field(min_length=1, description="One-line essence of the entity")
    notes: str | None = Field(default=None, description="Freeform notes from discussion")


class Alternative(BaseModel):
    """One possible answer to a tension's binary question.

    Each tension has exactly two alternatives. One is marked canonical
    (the default story path), and one is the alternate (becomes a branch
    only if explicitly explored in SEED).

    Attributes:
        id: Short identifier (e.g., "mentor_protector").
        description: Full description of this answer/path.
        canonical: True if this is the default story path.
    """

    id: str = Field(min_length=1, description="Short identifier for the alternative")
    description: str = Field(min_length=1, description="Full description of this path")
    canonical: bool = Field(description="True if this is the default/spine path")


class Tension(BaseModel):
    """A binary dramatic question with two alternative answers.

    Tensions represent meaningful story choices. The binary constraint
    keeps contrasts crisp and meaningful. For nuanced concepts, use
    multiple binary tensions instead of a single multi-way choice.

    Attributes:
        id: Short identifier (e.g., "mentor_trust").
        question: The dramatic question (must end with "?").
        alternatives: Exactly two possible answers.
        involves: Entity IDs central to this tension.
        why_it_matters: Thematic stakes and consequences.
    """

    id: str = Field(min_length=1, description="Short identifier for the tension")
    question: str = Field(min_length=1, description="Dramatic question (should end with ?)")
    alternatives: list[Alternative] = Field(
        min_length=2,
        max_length=2,
        description="Exactly two alternative answers",
    )
    involves: list[str] = Field(
        default_factory=list,
        description="Entity IDs central to this tension",
    )
    why_it_matters: str = Field(
        min_length=1,
        description="Thematic stakes and narrative consequences",
    )

    @model_validator(mode="after")
    def validate_exactly_one_canonical(self) -> Tension:
        """Ensure exactly one alternative is marked canonical."""
        canonical_count = sum(1 for alt in self.alternatives if alt.canonical)
        if canonical_count != 1:
            msg = f"Tension '{self.id}' must have exactly one canonical alternative, found {canonical_count}"
            raise ValueError(msg)
        return self


class BrainstormOutput(BaseModel):
    """Complete output of the BRAINSTORM stage.

    This structured output is produced by the LLM after the Discuss phase.
    It contains all generated entities and tensions that will be triaged
    by SEED into committed story structure.

    Good BRAINSTORM produces 15-25 entities and 4-8 tensions.

    Attributes:
        entities: All generated story entities.
        tensions: All generated dramatic tensions.
    """

    entities: list[Entity] = Field(
        default_factory=list,
        description="Generated story entities",
    )
    tensions: list[Tension] = Field(
        default_factory=list,
        description="Generated dramatic tensions",
    )
