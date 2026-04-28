"""GROW stage models.

These models define the graph node types created during GROW,
LLM sub-phase output schemas for future LLM phases, and the
stage result container.

Node types created by GROW:
- Arc: A route through the story (spine or branch)
- Passage: A rendered scene corresponding to a beat
- StateFlag: A state flag tracking consequence commitment

See docs/design/00-spec.md for ontology details.

Terminology (v5):
- path: Routes exploring specific answers to dilemmas
- intersection: Beats serving multiple paths
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from questfoundry.models.pipeline import PhaseResult
from questfoundry.models.seed import DilemmaImpact  # noqa: TC001

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


class StateFlag(BaseModel):
    """A state flag tracking consequence commitment.

    State flags are granted at commits beats and track which
    consequences have been locked in by the player's path.
    """

    flag_id: str = Field(min_length=1)
    derived_from: str = Field(min_length=1)
    flag_type: Literal["granted"] = "granted"


class Choice(BaseModel):
    """A player choice linking two passages (future Phase 9)."""

    from_passage: str = Field(min_length=1)
    to_passage: str = Field(min_length=1)
    label: str = Field(min_length=1)
    requires_state_flags: list[str] = Field(default_factory=list)
    grants: list[str] = Field(default_factory=list)
    is_return: bool = Field(default=False, description="True for spoke→hub return links")


class EntityOverlay(BaseModel):
    """Conditional entity details activated by state flags (future Phase 8c)."""

    entity_id: str = Field(min_length=1)
    when: list[str] = Field(min_length=1)
    details: dict[str, str] = Field(
        description="Entity state changes when state flags are active (must have at least one key)"
    )

    @model_validator(mode="after")
    def details_not_empty(self) -> EntityOverlay:
        """Reject empty details dict - must have at least one key."""
        if not self.details:
            msg = "details must contain at least one key-value pair"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# LLM sub-phase output models
# ---------------------------------------------------------------------------


class IntersectionProposal(BaseModel):
    """Phase 3: Proposes beats that form structural intersections."""

    beat_ids: list[str] = Field(min_length=2)
    resolved_location: str = Field(
        min_length=1,
        description=(
            "Specific location where this intersection scene occurs. "
            "Must be a non-empty string. When multiple locations are mentioned "
            "in the candidate group, pick the one that fits most beats. "
            "When genuinely uncertain, use the most specific location from "
            "any of the beat descriptions rather than a placeholder."
        ),
    )
    shared_entities: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class Phase3Output(BaseModel):
    """Wrapper for Phase 3 structured output (intersection proposals)."""

    intersections: list[IntersectionProposal] = Field(default_factory=list)


class ConflictGroupResolution(BaseModel):
    """LLM decision for a single swap pair in the resolve_temporal_hints phase.

    The LLM receives a group ID (e.g. 'P1') and must choose which of the two
    beats in the swap pair to drop.
    """

    group_id: str = Field(description="The group ID from the conflict list (e.g. 'P1', 'P2')")
    drop_beat_id: str = Field(description="The beat ID whose temporal hint should be dropped")
    reason: str = Field(
        min_length=1,
        description="One-sentence narrative justification for the choice",
    )


class TemporalResolutionOutput(BaseModel):
    """Structured output for the resolve_temporal_hints phase (#1123, #1140).

    Mandatory drops are applied before the LLM call and are not listed here.
    The LLM only resolves swap pairs — beats where either hint can survive
    but not both.
    """

    resolutions: list[ConflictGroupResolution] = Field(
        default_factory=list,
        description=(
            "One resolution per swap pair. Mandatory drops are pre-applied and not listed here."
        ),
    )


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
    """Wrapper for Phase 4a structured output (scene-type tags).

    Spec R-4b.1 distinguishes partial coverage (1 ≤ tagged < total → WARNING +
    fallback) from zero coverage (= LLM failure under R-4b.4 → halt + retry).
    `min_length=1` enforces the zero-coverage halt at Pydantic time so the
    retry loop fires; partial coverage is handled downstream as a WARNING.
    """

    tags: list[SceneTypeTag] = Field(
        min_length=1,
        description=(
            "Scene-type annotations for beats. Zero tags is treated as LLM "
            "failure (R-4b.4) and triggers retry; partial coverage is allowed "
            "with a downstream WARNING (R-4b.1)."
        ),
    )


class AtmosphericDetail(BaseModel):
    """Phase 4d: Sensory environment detail for a beat."""

    beat_id: str = Field(min_length=1)
    atmospheric_detail: str = Field(
        min_length=10,
        max_length=200,
        description="Recurring sensory detail for this beat's setting (sight, sound, smell, texture)",
    )


class Phase4dOutput(BaseModel):
    """Wrapper for Phase 4d structured output (atmospheric details).

    Like Phase 4a, zero details = full coverage gap = LLM failure that should
    trigger a retry. Partial coverage emits a WARNING per POLISH R-5e.1 and
    FILL falls back without explicit guidance.
    """

    details: list[AtmosphericDetail] = Field(
        min_length=1,
        description=(
            "Atmospheric details per beat. Zero details is treated as LLM "
            "failure and triggers retry; partial coverage emits WARNING."
        ),
    )

    @model_validator(mode="after")
    def _validate_unique_beat_ids(self) -> Phase4dOutput:
        if self.details:
            detail_ids = [d.beat_id for d in self.details]
            if len(detail_ids) != len(set(detail_ids)):
                raise ValueError("beat_id in details list must be unique")
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
    """POLISH Phase 1a (Narrative Gap Insertion): proposes new beats to fill
    structural transitions between adjacent beats. Per polish.md R-1a.2, gap
    beats are STRUCTURAL ONLY — they MUST NOT carry `dilemma_impacts`. The
    field below is retained as an empty default for back-compat with the
    prior GROW Phase 4b/4c shape; a model_validator rejects any non-empty
    value so an LLM that ignores the prompt fails validation immediately
    rather than producing a beat that violates R-1a.2 downstream.
    """

    path_id: str = Field(min_length=1)
    after_beat: str | None = Field(
        default=None,
        description="The EARLIER beat — new beat is inserted AFTER this one",
    )
    before_beat: str | None = Field(
        default=None,
        description="The LATER beat — new beat is inserted BEFORE this one",
    )
    summary: str = Field(min_length=1)
    scene_type: Literal["scene", "sequel", "micro_beat"] = "sequel"
    dilemma_impacts: list[DilemmaImpact] = Field(
        default_factory=list,
        description=(
            "MUST be empty per polish.md R-1a.2 — gap beats are structural "
            "transition beats and cannot advance, reveal, commit, or "
            "complicate any dilemma."
        ),
    )

    @model_validator(mode="after")
    def _no_dilemma_impacts(self) -> GapProposal:
        if self.dilemma_impacts:
            raise ValueError(
                "Gap beats MUST NOT carry dilemma_impacts (polish.md R-1a.2). "
                "Gap beats are structural transition beats only — they cannot "
                "advance, reveal, commit, or complicate any dilemma. Remove the "
                "dilemma_impacts entries from this gap proposal."
            )
        return self

    @model_validator(mode="after")
    def _require_placement(self) -> GapProposal:
        """POLISH R-1a.3: gap beats record `bridges_from` / `bridges_to`.

        Both `after_beat` and `before_beat` being null makes the gap
        unplaceable in the beat sequence. POLISH has no semantic_validator
        hook (#1498), so any retry-bypass in a POLISH-shaped output relies
        on Pydantic enforcement to fire the retry loop.
        """
        if self.after_beat is None and self.before_beat is None:
            raise ValueError(
                "GapProposal must have at least one of `after_beat` or "
                "`before_beat` set to be placeable in the beat sequence "
                "(POLISH R-1a.3). A gap beat with neither anchor cannot be "
                "inserted — provide the earlier beat (`after_beat`) or the "
                "later beat (`before_beat`) or both."
            )
        return self


class Phase4bOutput(BaseModel):
    """Wrapper for Phase 4b/4c structured output (gap proposals)."""

    gaps: list[GapProposal] = Field(default_factory=list)


class OverlayDetailItem(BaseModel):
    """Single key-value pair for an entity overlay.

    OpenAI strict mode forbids ``dict[str, str]`` (additionalProperties).
    A list of explicit key/value objects is fully compatible.
    """

    key: str = Field(min_length=1, description="Attribute name being overridden")
    value: str = Field(min_length=1, description="New value when state flags are active")


class OverlayProposal(BaseModel):
    """Phase 8c: Proposes entity overlay conditions."""

    entity_id: str = Field(min_length=1)
    when: list[str] = Field(min_length=1)
    details: list[OverlayDetailItem] = Field(
        min_length=1,
        description="Entity state changes when state flags are active",
    )

    @model_validator(mode="after")
    def _check_no_duplicate_keys(self) -> OverlayProposal:
        """Reject duplicate keys — prevents silent data loss in details_as_dict()."""
        keys = [d.key for d in self.details]
        if len(keys) != len(set(keys)):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            msg = f"Duplicate keys in details: {dupes}"
            raise ValueError(msg)
        return self

    def details_as_dict(self) -> dict[str, str]:
        """Convert details list to dict for graph storage."""
        return {d.key: d.value for d in self.details}


class Phase8cOutput(BaseModel):
    """Wrapper for Phase 8c structured output (entity overlay proposals)."""

    overlays: list[OverlayProposal] = Field(default_factory=list)


class TransitionBridge(BaseModel):
    """A single transition bridge between two beats."""

    transition_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    entities: list[str] = Field(default_factory=list)
    location: str = ""


class TransitionGapsOutput(BaseModel):
    """Output of the transition gap detection LLM call."""

    bridges: list[TransitionBridge] = Field(default_factory=list)


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
    state_flag_count: int = 0
    overlay_count: int = 0
    phases_completed: list[GrowPhaseResult] = Field(default_factory=list)
    spine_arc_id: str | None = None
