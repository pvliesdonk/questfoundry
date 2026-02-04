"""GROW stage models.

These models define the graph node types created during GROW,
LLM sub-phase output schemas for future LLM phases, and the
stage result container.

Node types created by GROW:
- Arc: A route through the story (spine or branch)
- Passage: A rendered scene corresponding to a beat
- Codeword: A state flag tracking consequence commitment

See docs/design/00-spec.md for ontology details.

Terminology (v5):
- path: Routes exploring specific answers to dilemmas
- intersection: Beats serving multiple paths
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from questfoundry.models.pipeline import PhaseResult

# ---------------------------------------------------------------------------
# Graph node types
# ---------------------------------------------------------------------------


class Arc(BaseModel):
    """A route through the story graph.

    Arcs are enumerated from path combinations across dilemmas.
    The spine arc contains all canonical (default-path) paths.
    Branch arcs diverge from the spine at specific beats.
    """

    arc_id: str = Field(min_length=1)
    arc_type: Literal["spine", "branch"]
    paths: list[str] = Field(min_length=1)
    sequence: list[str] = Field(default_factory=list)
    diverges_from: str | None = None
    diverges_at: str | None = None
    converges_to: str | None = None
    converges_at: str | None = None


class Passage(BaseModel):
    """A rendered scene corresponding to a beat.

    Each beat in the graph gets exactly one passage node
    during Phase 8a.
    """

    passage_id: str = Field(min_length=1)
    from_beat: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    entities: list[str] = Field(default_factory=list)


class Codeword(BaseModel):
    """A state flag tracking consequence commitment.

    Codewords are granted at commits beats and track which
    consequences have been locked in by the player's path.
    """

    codeword_id: str = Field(min_length=1)
    tracks: str = Field(min_length=1)
    codeword_type: Literal["granted"] = "granted"


class Choice(BaseModel):
    """A player choice linking two passages (future Phase 9)."""

    from_passage: str = Field(min_length=1)
    to_passage: str = Field(min_length=1)
    label: str = Field(min_length=1)
    requires: list[str] = Field(default_factory=list)
    grants: list[str] = Field(default_factory=list)


class EntityOverlay(BaseModel):
    """Conditional entity details activated by codewords (future Phase 8c)."""

    entity_id: str = Field(min_length=1)
    when: list[str] = Field(min_length=1)
    details: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM sub-phase output models
# ---------------------------------------------------------------------------


class PathAgnosticAssessment(BaseModel):
    """Phase 2: Marks beats that are path-agnostic for specific dilemmas."""

    beat_id: str = Field(min_length=1)
    agnostic_for: list[str] = Field(default_factory=list)


class Phase2Output(BaseModel):
    """Wrapper for Phase 2 structured output (path-agnostic assessment)."""

    assessments: list[PathAgnosticAssessment] = Field(default_factory=list)


class IntersectionProposal(BaseModel):
    """Phase 3: Proposes beats that form structural intersections."""

    beat_ids: list[str] = Field(min_length=2)
    resolved_location: str | None = None
    rationale: str = Field(min_length=1)


class Phase3Output(BaseModel):
    """Wrapper for Phase 3 structured output (intersection proposals)."""

    intersections: list[IntersectionProposal] = Field(default_factory=list)


class SceneTypeTag(BaseModel):
    """Phase 4a: Tags beats with scene type, narrative function, and exit mood."""

    beat_id: str = Field(min_length=1)
    scene_type: Literal["scene", "sequel", "micro_beat"]
    narrative_function: Literal["introduce", "develop", "complicate", "confront", "resolve"] = (
        Field(
            description=(
                "What dramatic role this beat plays: introduce (establish new elements), "
                "develop (deepen understanding), complicate (raise stakes/obstacles), "
                "confront (direct engagement with tension), resolve (settle a thread)"
            ),
        )
    )
    exit_mood: str = Field(
        min_length=2,
        max_length=40,
        description="2-3 word emotional descriptor for how reader feels leaving this beat",
    )


class Phase4aOutput(BaseModel):
    """Wrapper for Phase 4a structured output (scene-type tags)."""

    tags: list[SceneTypeTag] = Field(default_factory=list)


class AtmosphericDetail(BaseModel):
    """Phase 4d: Sensory environment detail for a beat."""

    beat_id: str = Field(min_length=1)
    atmospheric_detail: str = Field(
        min_length=10,
        max_length=200,
        description="Recurring sensory detail for this beat's setting (sight, sound, smell, texture)",
    )


class EntryMood(BaseModel):
    """Phase 4d: Per-path entry mood for shared beats."""

    path_id: str = Field(min_length=1)
    mood: str = Field(
        min_length=2,
        max_length=50,
        description="Emotional quality arriving from this path (2-3 words)",
    )


class EntryStateBeat(BaseModel):
    """Phase 4d: Entry mood assignments for a single shared beat."""

    beat_id: str = Field(min_length=1)
    moods: list[EntryMood] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_mood_paths(self) -> EntryStateBeat:
        path_ids = [mood.path_id for mood in self.moods]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("path_id in moods list must be unique for a single beat")
        return self


class Phase4dOutput(BaseModel):
    """Wrapper for Phase 4d structured output (atmospheric details + entry states)."""

    details: list[AtmosphericDetail] = Field(default_factory=list)
    entry_states: list[EntryStateBeat] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_beat_ids(self) -> Phase4dOutput:
        if self.details:
            detail_ids = [d.beat_id for d in self.details]
            if len(detail_ids) != len(set(detail_ids)):
                raise ValueError("beat_id in details list must be unique")
        if self.entry_states:
            entry_ids = [es.beat_id for es in self.entry_states]
            if len(entry_ids) != len(set(entry_ids)):
                raise ValueError("beat_id in entry_states list must be unique")
        return self


class PathMiniArc(BaseModel):
    """Phase 4e: Path-level narrative metadata."""

    path_id: str = Field(min_length=1)
    path_theme: str = Field(
        min_length=10,
        max_length=200,
        description="Emotional through-line for this path",
    )
    path_mood: str = Field(
        min_length=2,
        max_length=50,
        description="Overall quality/tone descriptor (2-3 words)",
    )


class Phase4eOutput(BaseModel):
    """Wrapper for Phase 4e structured output (per-path mini-arcs)."""

    arcs: list[PathMiniArc] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_path_ids(self) -> Phase4eOutput:
        if self.arcs:
            path_ids = [arc.path_id for arc in self.arcs]
            if len(path_ids) != len(set(path_ids)):
                raise ValueError("path_id in arcs list must be unique")
        return self


class EntityArcDescriptor(BaseModel):
    """Phase 4f: Per-entity arc trajectory on a single path.

    Describes how an entity progresses through a path's beat sequence.
    The arc_type is NOT stored here — it is computed deterministically
    from the entity's category (character→transformation, object→significance,
    location→atmosphere, faction→relationship).
    """

    entity_id: str = Field(min_length=1)
    arc_line: str = Field(
        min_length=10,
        max_length=200,
        description='Trajectory in "A → B → C" format',
    )
    pivot_beat: str = Field(min_length=1)


class Phase4fOutput(BaseModel):
    """Wrapper for Phase 4f structured output (entity arcs per path)."""

    arcs: list[EntityArcDescriptor] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_entity_ids(self) -> Phase4fOutput:
        if self.arcs:
            eids = [a.entity_id for a in self.arcs]
            if len(eids) != len(set(eids)):
                raise ValueError("entity_id in arcs list must be unique")
        return self


class GapProposal(BaseModel):
    """Phase 4b/4c: Proposes new beats to fill structural gaps."""

    path_id: str = Field(min_length=1)
    after_beat: str | None = None
    before_beat: str | None = None
    summary: str = Field(min_length=1)
    scene_type: Literal["scene", "sequel", "micro_beat"] = "sequel"


class Phase4bOutput(BaseModel):
    """Wrapper for Phase 4b/4c structured output (gap proposals)."""

    gaps: list[GapProposal] = Field(default_factory=list)


class OverlayProposal(BaseModel):
    """Phase 8c: Proposes entity overlay conditions."""

    entity_id: str = Field(min_length=1)
    when: list[str] = Field(min_length=1)
    details: dict[str, str] = Field(default_factory=dict)


class Phase8cOutput(BaseModel):
    """Wrapper for Phase 8c structured output (entity overlay proposals)."""

    overlays: list[OverlayProposal] = Field(default_factory=list)


class ChoiceLabel(BaseModel):
    """Phase 9: Labels for player choices between passages."""

    from_passage: str = Field(min_length=1)
    to_passage: str = Field(min_length=1)
    label: str = Field(min_length=1)


class Phase9Output(BaseModel):
    """Wrapper for Phase 9 structured output (choice labels)."""

    labels: list[ChoiceLabel] = Field(default_factory=list)


class ForkProposal(BaseModel):
    """Phase 9b: A proposed fork at a linear stretch that reconverges."""

    fork_at: str = Field(min_length=1, description="Passage ID where the fork occurs")
    reconverge_at: str = Field(min_length=1, description="Passage ID where both options rejoin")
    option_a_summary: str = Field(min_length=1, description="Summary of option A")
    option_b_summary: str = Field(min_length=1, description="Summary of option B")
    label_a: str = Field(min_length=1, description="Choice label for option A")
    label_b: str = Field(min_length=1, description="Choice label for option B")


class Phase9bOutput(BaseModel):
    """Wrapper for Phase 9b structured output (fork proposals)."""

    proposals: list[ForkProposal] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------


class GrowPhaseResult(PhaseResult):
    """Result of a single GROW phase execution.

    Inherits all fields from PhaseResult. Currently identical,
    but allows GROW-specific fields to be added without affecting
    other stages.
    """


class GrowResult(BaseModel):
    """Overall GROW stage result."""

    arc_count: int = 0
    passage_count: int = 0
    codeword_count: int = 0
    choice_count: int = 0
    overlay_count: int = 0
    phases_completed: list[GrowPhaseResult] = Field(default_factory=list)
    spine_arc_id: str | None = None
