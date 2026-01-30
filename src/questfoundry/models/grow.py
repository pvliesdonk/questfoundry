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

from pydantic import BaseModel, Field

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
    """Phase 4a: Tags beats with scene type classification."""

    beat_id: str = Field(min_length=1)
    scene_type: Literal["scene", "sequel", "micro_beat"]


class Phase4aOutput(BaseModel):
    """Wrapper for Phase 4a structured output (scene-type tags)."""

    tags: list[SceneTypeTag] = Field(default_factory=list)


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
