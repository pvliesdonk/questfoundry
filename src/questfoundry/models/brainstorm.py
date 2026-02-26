"""Pydantic models for BRAINSTORM stage output.

BRAINSTORM is the expansive exploration phase that generates raw creative
material: entities (characters, locations, objects, factions) and dilemmas
(binary dramatic questions with two answers each).

See docs/design/00-spec.md and docs/design/procedures/brainstorm.md for details.

Terminology (v5):
- dilemma: Binary dramatic questions
- answer: Possible resolutions to dilemmas
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

EntityType = Literal["character", "location", "object", "faction"]


class Entity(BaseModel):
    """A story entity: character, location, object, or faction.

    Entities are the raw building blocks generated during BRAINSTORM.
    SEED will later decide which to retain, cut, or modify.

    Attributes:
        entity_id: Short identifier (e.g., "kay", "mentor", "archive").
        entity_category: Entity category (character, location, object, faction).
        name: Canonical display name if it emerges naturally (e.g., "Dr. Aris Chen").
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
    name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Canonical display name if it emerges naturally during brainstorming "
            "(e.g., 'Dr. Aris Chen', 'Maya's Bakery'). SEED will generate if missing."
        ),
    )
    concept: str = Field(min_length=1, description="One-line essence of the entity")
    notes: str | None = Field(default=None, description="Freeform notes from discussion")


class Answer(BaseModel):
    """One possible answer to a dilemma's binary question.

    Each dilemma has exactly two answers. One is marked as the default
    path (spine), and one is the alternate (becomes a branch only if
    explicitly explored in SEED).

    Attributes:
        answer_id: Unique identifier for this answer (e.g., "guilty", "framed").
        description: Full description of this answer/path.
        is_canonical: True if this is the default story path (spine).
    """

    answer_id: str = Field(
        min_length=1,
        description="Unique identifier for this answer (e.g., 'guilty', 'framed', 'betrayed')",
    )
    description: str = Field(min_length=1, description="Full description of this path")
    is_canonical: bool = Field(
        description="True if this is the default story path (spine). Exactly one per dilemma."
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_alternative_id(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old 'alternative_id' field to 'answer_id'."""
        if isinstance(data, dict) and "alternative_id" in data and "answer_id" not in data:
            data = dict(data)
            data["answer_id"] = data.pop("alternative_id")
        return data

    # Backward compatibility property
    @property
    def alternative_id(self) -> str:
        """Deprecated: Use 'answer_id' instead."""
        return self.answer_id


class Dilemma(BaseModel):
    """A binary dramatic question with two possible answers.

    Dilemmas represent meaningful story choices. The binary constraint
    keeps contrasts crisp and meaningful. For nuanced concepts, use
    multiple binary dilemmas instead of a single multi-way choice.

    Attributes:
        dilemma_id: Scoped identifier with dilemma:: prefix (e.g., "dilemma::mentor_trust").
        question: The dramatic question (must end with "?").
        answers: Exactly two possible answers.
        central_entity_ids: Entity IDs central to this dilemma.
        why_it_matters: Thematic stakes and consequences.
    """

    dilemma_id: str = Field(
        min_length=1,
        description=(
            "Unique identifier for this dilemma (e.g., 'dilemma::mentor_trust', "
            "'dilemma::murder_weapon')"
        ),
    )
    question: str = Field(min_length=1, description="Dramatic question (should end with ?)")

    @field_validator("dilemma_id")
    @classmethod
    def validate_dilemma_id_no_trailing_or(cls, v: str) -> str:
        """Reject dilemma IDs ending with '_or_' (common LLM generation error)."""
        raw = v.removeprefix("dilemma::")
        if raw.endswith("_or_") or raw.endswith("_or"):
            msg = (
                f"dilemma_id '{v}' ends with '_or_' — "
                "the ID must end with the second option word "
                "(e.g., 'host_benevolent_or_selfish', not 'host_benevolent_or_selfish_or_')"
            )
            raise ValueError(msg)
        return v

    answers: list[Answer] = Field(
        min_length=2,
        max_length=2,
        description="Exactly two possible answers",
    )
    central_entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity IDs central to this dilemma — stored as anchored_to edges, not node properties",
    )
    why_it_matters: str = Field(
        min_length=1,
        description="Thematic stakes and narrative consequences",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_alternatives_field(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old 'alternatives' field to 'answers'."""
        if isinstance(data, dict) and "alternatives" in data and "answers" not in data:
            data = dict(data)
            data["answers"] = data.pop("alternatives")
        return data

    @model_validator(mode="after")
    def validate_exactly_one_default_path(self) -> Dilemma:
        """Ensure exactly one answer is marked as the default path."""
        default_count = sum(1 for ans in self.answers if ans.is_canonical)
        if default_count != 1:
            msg = f"Dilemma '{self.dilemma_id}' must have exactly one default path answer, found {default_count}"
            raise ValueError(msg)
        return self

    @property
    def alternatives(self) -> list[Answer]:
        """Deprecated: Use 'answers' instead."""
        return self.answers


class BrainstormOutput(BaseModel):
    """Complete output of the BRAINSTORM stage.

    This structured output is produced by the LLM after the Discuss phase.
    It contains all generated entities and dilemmas that will be triaged
    by SEED into committed story structure.

    Good BRAINSTORM produces 15-25 entities and 4-8 dilemmas.

    Attributes:
        entities: All generated story entities.
        dilemmas: All generated dramatic dilemmas.
    """

    entities: list[Entity] = Field(
        default_factory=list,
        description="Generated story entities",
    )
    dilemmas: list[Dilemma] = Field(
        default_factory=list,
        description="Generated dramatic dilemmas",
    )
