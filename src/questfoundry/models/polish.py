"""POLISH stage models.

These models define LLM output schemas for POLISH's three pre-freeze
phases (beat reordering, pacing, character arc synthesis) and the
stage result container.

Node types created by POLISH:
- micro_beat: Brief transition beats inserted for pacing (Phase 2)
- character_arc_metadata: Arc data for entities (Phase 3)

See docs/design/procedures/polish.md for algorithm details.

Terminology (v5):
- passage: A rendered scene grouping one or more beats
- variant: An alternative passage gated by state flags
- residue beat: A mood-setting beat preceding shared passages
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Phase 1: Beat Reordering — LLM output schema
# ---------------------------------------------------------------------------


class ReorderedSection(BaseModel):
    """A reordered linear section of the beat DAG.

    The LLM proposes a new order for beats within a linear section,
    optimizing for scene-sequel rhythm, entity continuity, and
    emotional arc.
    """

    section_id: str = Field(min_length=1, description="Identifier for the linear section")
    beat_ids: list[str] = Field(
        min_length=1,
        description="Beat IDs in proposed new order (must be same set as input)",
    )
    rationale: str = Field(
        min_length=1,
        description="Brief explanation of why this order is better",
    )


class Phase1Output(BaseModel):
    """Output of Phase 1: Beat Reordering.

    Contains reordered sections. Sections not listed are kept in
    original order.
    """

    reordered_sections: list[ReorderedSection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 2: Pacing & Micro-beat Injection — LLM output schema
# ---------------------------------------------------------------------------


class MicroBeatProposal(BaseModel):
    """A micro-beat proposed for insertion to improve pacing.

    Micro-beats are brief transitions that don't advance any dilemma.
    They provide breathing room between major scenes.
    """

    after_beat_id: str = Field(
        min_length=1,
        description="Insert after this beat in the DAG",
    )
    summary: str = Field(
        min_length=1,
        description="One-sentence summary of the micro-beat",
    )
    entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity IDs referenced in this micro-beat (subset of surrounding beats)",
    )


class Phase2Output(BaseModel):
    """Output of Phase 2: Pacing & Micro-beat Injection.

    Contains micro-beat proposals. Empty list means no pacing issues found.
    """

    micro_beats: list[MicroBeatProposal] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 3: Character Arc Synthesis — LLM output schema
# ---------------------------------------------------------------------------


class ArcPivot(BaseModel):
    """A key moment where an entity's trajectory changes."""

    path_id: str = Field(min_length=1, description="Path where this pivot occurs")
    beat_id: str = Field(min_length=1, description="Beat where the pivot happens")
    description: str = Field(
        min_length=1,
        description="What changes at this moment for this entity",
    )


class CharacterArcMetadata(BaseModel):
    """Arc description for a single entity across the story.

    Consumed by FILL when writing prose to ensure character
    consistency across passages.
    """

    entity_id: str = Field(min_length=1, description="Entity this arc describes")
    start: str = Field(
        min_length=1,
        description="How the entity is introduced",
    )
    pivots: list[ArcPivot] = Field(
        default_factory=list,
        description="Key trajectory changes per path",
    )
    end_per_path: dict[str, str] = Field(
        default_factory=dict,
        description="Where the entity ends up on each path (path_id → description)",
    )


class Phase3Output(BaseModel):
    """Output of Phase 3: Character Arc Synthesis.

    Contains arc metadata for entities appearing in 2+ beats.
    """

    character_arcs: list[CharacterArcMetadata] = Field(default_factory=list)


# Note: POLISH phases return the shared PhaseResult from models.pipeline.
# Stage-specific result models (PolishResult) are added in later PRs
# when passage/choice counts are available.
