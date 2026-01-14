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
        entity_id: Short identifier (e.g., "kay", "mentor", "archive").
        entity_category: Entity category (character, location, object, faction).
        concept: One-line essence capturing what makes it interesting.
        notes: Freeform context from discussion (rich detail allowed).
    """

    entity_id: str = Field(
        min_length=1,
        description="Unique identifier for this entity (e.g., 'lady_beatrice', 'vane_manor')",
    )
    entity_category: EntityType = Field(
        description="Entity category: character, location, object, or faction"
    )
    concept: str = Field(min_length=1, description="One-line essence of the entity")
    notes: str | None = Field(default=None, description="Freeform notes from discussion")


class Alternative(BaseModel):
    """One possible answer to a tension's binary question.

    Each tension has exactly two alternatives. One is marked as the default
    path (spine), and one is the alternate (becomes a branch only if
    explicitly explored in SEED).

    Attributes:
        alternative_id: Unique identifier for this alternative (e.g., "guilty", "framed").
        description: Full description of this answer/path.
        is_default_path: True if this is the default story path (spine).
    """

    alternative_id: str = Field(
        min_length=1,
        description="Unique identifier for this alternative path (e.g., 'guilty', 'framed', 'betrayed')",
    )
    description: str = Field(min_length=1, description="Full description of this path")
    is_default_path: bool = Field(
        description="True if this is the default story path (spine). Exactly one per tension."
    )


class Tension(BaseModel):
    """A binary dramatic question with two alternative answers.

    Tensions represent meaningful story choices. The binary constraint
    keeps contrasts crisp and meaningful. For nuanced concepts, use
    multiple binary tensions instead of a single multi-way choice.

    Attributes:
        tension_id: Short identifier (e.g., "mentor_trust").
        question: The dramatic question (must end with "?").
        alternatives: Exactly two possible answers.
        central_entity_ids: Entity IDs central to this tension.
        why_it_matters: Thematic stakes and consequences.
    """

    tension_id: str = Field(
        min_length=1,
        description="Unique identifier for this tension (e.g., 'mentor_trust', 'murder_weapon')",
    )
    question: str = Field(min_length=1, description="Dramatic question (should end with ?)")
    alternatives: list[Alternative] = Field(
        min_length=2,
        max_length=2,
        description="Exactly two alternative answers",
    )
    central_entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity IDs central to this tension (references entity_id values)",
    )
    why_it_matters: str = Field(
        min_length=1,
        description="Thematic stakes and narrative consequences",
    )

    @model_validator(mode="after")
    def validate_exactly_one_default_path(self) -> Tension:
        """Ensure exactly one alternative is marked as the default path."""
        default_count = sum(1 for alt in self.alternatives if alt.is_default_path)
        if default_count != 1:
            msg = f"Tension '{self.tension_id}' must have exactly one default path alternative, found {default_count}"
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
