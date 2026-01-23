"""GROW stage models.

These models define the graph node types created during GROW,
LLM sub-phase output schemas for future LLM phases, and the
stage result container.

Node types created by GROW:
- Arc: A path through the story (spine or branch)
- Passage: A rendered scene corresponding to a beat
- Codeword: A state flag tracking consequence commitment

See docs/design/00-spec.md for ontology details.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Graph node types
# ---------------------------------------------------------------------------


class Arc(BaseModel):
    """A path through the story graph.

    Arcs are enumerated from thread combinations across tensions.
    The spine arc contains all canonical (default-path) threads.
    Branch arcs diverge from the spine at specific beats.
    """

    arc_id: str = Field(min_length=1)
    arc_type: Literal["spine", "branch"]
    threads: list[str] = Field(min_length=1)
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


class ThreadAgnosticAssessment(BaseModel):
    """Phase 2: Marks beats that are thread-agnostic for specific tensions."""

    beat_id: str = Field(min_length=1)
    agnostic_for: list[str] = Field(default_factory=list)


class Phase2Output(BaseModel):
    """Wrapper for Phase 2 structured output (thread-agnostic assessment)."""

    assessments: list[ThreadAgnosticAssessment] = Field(default_factory=list)


class KnotProposal(BaseModel):
    """Phase 3: Proposes beats that form structural knots."""

    beat_ids: list[str] = Field(min_length=2)
    resolved_location: str | None = None
    rationale: str = ""


class Phase3Output(BaseModel):
    """Wrapper for Phase 3 structured output (knot proposals)."""

    knots: list[KnotProposal] = Field(default_factory=list)


class SceneTypeTag(BaseModel):
    """Phase 4a: Tags beats with scene type classification."""

    beat_id: str = Field(min_length=1)
    scene_type: Literal["scene", "sequel", "micro_beat"]


class GapProposal(BaseModel):
    """Phase 4b: Proposes new beats to fill structural gaps."""

    thread_id: str = Field(min_length=1)
    after_beat: str | None = None
    before_beat: str | None = None
    summary: str = Field(min_length=1)
    scene_type: Literal["scene", "sequel", "micro_beat"] = "sequel"


class OverlayProposal(BaseModel):
    """Phase 8c: Proposes entity overlay conditions."""

    entity_id: str = Field(min_length=1)
    when: list[str] = Field(min_length=1)
    details: dict[str, str] = Field(default_factory=dict)


class ChoiceLabel(BaseModel):
    """Phase 9: Labels for player choices between passages."""

    from_passage: str = Field(min_length=1)
    to_passage: str = Field(min_length=1)
    label: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------


class GrowPhaseResult(BaseModel):
    """Result of a single GROW phase execution."""

    phase: str = Field(min_length=1)
    status: Literal["completed", "skipped", "failed"]
    detail: str = ""
    llm_calls: int = 0
    tokens_used: int = 0


class GrowResult(BaseModel):
    """Overall GROW stage result."""

    arc_count: int = 0
    passage_count: int = 0
    codeword_count: int = 0
    choice_count: int = 0
    overlay_count: int = 0
    phases_completed: list[GrowPhaseResult] = Field(default_factory=list)
    spine_arc_id: str | None = None
