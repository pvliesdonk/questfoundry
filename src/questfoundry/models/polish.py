"""POLISH stage models.

These models define LLM output schemas for POLISH's three pre-freeze
phases (beat reordering, pacing, character arc synthesis) and the
stage result container.

Node types created by POLISH:
- micro_beat: Brief transition beats inserted for pacing (Phase 2)

Note: Phase 3 writes character_arc annotations directly onto Entity nodes
(data-dict field) — no separate node type.  See docs/design/procedures/polish.md
§R-3.3 and ontology Part 1 "Character Arc Metadata".

See docs/design/procedures/polish.md for algorithm details.

Terminology (v5):
- passage: A rendered scene grouping one or more beats
- variant: An alternative passage gated by state flags
- residue beat: A mood-setting beat preceding shared passages
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

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


class PerPathArc(BaseModel):
    """Per-path arc trajectory entry for one entity on one path (R-3.6).

    Produced by POLISH Phase 3 in the same LLM call that synthesizes
    ``start``/``pivots``/``end_per_path``, ensuring internal consistency
    by construction (single LLM context).

    ``arc_type`` is set deterministically by POLISH from the entity's
    category per R-3.7 (NOT chosen by the LLM); the LLM-supplied value
    is validated against the derived value. ``pivot_beat`` MUST equal
    ``CharacterArcMetadata.pivots[path_id]`` for the same path per R-3.8.
    """

    path_id: str = Field(min_length=1, description="Path this entry describes")
    arc_type: str = Field(
        min_length=1,
        description=(
            "Entity-category-derived arc type. The LLM emits the placeholder "
            '``"_set_by_code"`` (or any value); POLISH overwrites this with '
            "the value derived from the entity's category per R-3.7 — "
            "character → transformation, location → atmosphere, "
            "object → significance, faction → relationship."
        ),
    )
    arc_line: str = Field(
        min_length=1,
        description="Concise A → B → C trajectory of the entity on this path",
    )
    pivot_beat: str = Field(
        min_length=1,
        description="Beat where the entity's trajectory turns (same path's pivots[path_id])",
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
    arcs_per_path: list[PerPathArc] = Field(
        default_factory=list,
        description=(
            "R-3.6: per-path positional trajectories (path_id, arc_type, "
            "arc_line, pivot_beat). One entry per path on which the entity "
            "is arc-worthy. Same-LLM-call consistency: arcs_per_path[*].pivot_beat "
            "MUST equal pivots[path_id] for the same path (R-3.8)."
        ),
    )


class Phase3Output(BaseModel):
    """Output of Phase 3: Character Arc Synthesis.

    Contains arc metadata for entities appearing in 2+ beats.
    """

    character_arcs: list[CharacterArcMetadata] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 4: Plan Computation — deterministic plan models
# ---------------------------------------------------------------------------


class PassageSpec(BaseModel):
    """Specification for a passage grouping beats.

    Created during Phase 4a (beat grouping). Each passage groups one
    or more beats into a single scene.
    """

    passage_id: str = Field(min_length=1)
    beat_ids: list[str] = Field(min_length=1)
    summary: str = Field(
        min_length=1,
        description=(
            "Human-readable scene summary derived from the passage's beats. "
            "Stage Output Contract item 2 requires every passage to have a "
            "non-empty summary; FILL has no prose context without it."
        ),
    )
    entities: list[str] = Field(default_factory=list)
    grouping_type: str = Field(
        default="singleton",
        description="How beats were grouped: intersection, collapse, or singleton",
    )
    transition_guidance: list[str] = Field(
        default_factory=list,
        description="Transition instructions between beats (N-1 for N beats; Phase 5f)",
    )


class VariantSpec(BaseModel):
    """Specification for a variant passage.

    Created during Phase 4b (feasibility audit) for passages with
    heavy residue flags that need separate prose per state.
    """

    base_passage_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    requires: list[str] = Field(
        default_factory=list,
        description="State flags that must be active for this variant",
    )
    summary: str = Field(default="")


class ResidueSpec(BaseModel):
    """Specification for a residue beat.

    Created during Phase 4b for passages with light/cosmetic residue
    flags. Residue beats are mood-setting moments before shared passages.
    """

    target_passage_id: str = Field(min_length=1)
    residue_id: str = Field(min_length=1)
    flag: str = Field(min_length=1, description="State flag this residue addresses")
    path_id: str = Field(default="")
    content_hint: str = Field(
        default="", description="Mood-setting prose hint (populated by Phase 5)"
    )
    mapping_strategy: Literal["residue_passage_with_variants", "parallel_passages"] = Field(
        default="residue_passage_with_variants",
        description=(
            "Passage-layer mapping for this residue spec.  "
            "'residue_passage_with_variants' creates a single residue passage "
            "with variant children; 'parallel_passages' creates sibling "
            "passages that branch and rejoin.  Chosen per-residue by the "
            "Phase 5b LLM call per spec R-5.7/R-5.8.  Defaults to "
            "'residue_passage_with_variants' for back-compat; Phase 5b's "
            "LLM overrides per residue."
        ),
    )


class ChoiceSpec(BaseModel):
    """Specification for a choice edge between passages.

    Created during Phase 4c (choice edge derivation). Labels are
    populated by Phase 5 (LLM enrichment).
    """

    from_passage: str = Field(min_length=1)
    to_passage: str = Field(min_length=1)
    requires: list[str] = Field(default_factory=list)
    grants: list[str] = Field(default_factory=list)
    label: str = Field(default="")


class FalseBranchCandidate(BaseModel):
    """A stretch of passages that could benefit from false branching.

    Identified in Phase 4d. The decision (skip/diamond/sidetrack)
    is made by the LLM in Phase 5.
    """

    passage_ids: list[str] = Field(min_length=1)
    context_summary: str = Field(default="")


class FalseBranchSpec(BaseModel):
    """Specification for a false branch decision.

    Populated by Phase 5 (LLM enrichment) based on candidates
    from Phase 4d.
    """

    candidate_passage_ids: list[str] = Field(min_length=1)
    branch_type: str = Field(
        min_length=1,
        description="skip, diamond, or sidetrack",
    )
    details: str = Field(default="")
    diamond_summary_a: str = Field(default="", description="First alternative passage (diamond)")
    diamond_summary_b: str = Field(default="", description="Second alternative passage (diamond)")
    sidetrack_summary: str = Field(default="", description="Detour beat summary (sidetrack)")
    sidetrack_entities: list[str] = Field(
        default_factory=list,
        description="Entity IDs for sidetrack beat",
    )
    choice_label_enter: str = Field(default="", description="Label for entering sidetrack")
    choice_label_return: str = Field(default="", description="Label for returning from sidetrack")


# ---------------------------------------------------------------------------
# Phase 5: LLM Enrichment — output schemas
# ---------------------------------------------------------------------------


class ChoiceLabelItem(BaseModel):
    """A single choice label produced by the LLM."""

    from_passage: str = Field(min_length=1)
    to_passage: str = Field(min_length=1)
    label: str = Field(min_length=1, description="Diegetic, concise label for this choice")


class Phase5aOutput(BaseModel):
    """Output of Phase 5a: Choice Label Generation.

    POLISH has no semantic_validator hook (#1498), so Pydantic is the only
    in-retry enforcement point. An empty list silently leaves all
    `ChoiceSpec.label` fields at their default `""`, which Phase 7 R-7.7
    catches at exit AFTER Phase 6 has committed all passages — too late
    for repair. `min_length=1` fires the retry loop instead.
    """

    choice_labels: list[ChoiceLabelItem] = Field(
        min_length=1,
        description=(
            "Diegetic, concise labels for choices. MUST contain ≥1 entry; "
            "empty list is treated as LLM failure and triggers retry."
        ),
    )


class ResidueContentItem(BaseModel):
    """Mood-setting prose hint for a residue beat."""

    residue_id: str = Field(min_length=1)
    content_hint: str = Field(min_length=1, description="Brief mood-setting prose hint")
    mapping_strategy: Literal["residue_passage_with_variants", "parallel_passages"] = Field(
        default="residue_passage_with_variants",
        description=(
            "Passage-layer mapping chosen by the LLM for this residue.  "
            "'residue_passage_with_variants': one residue passage shared across flag "
            "combinations (use for subtle mood shifts in the same scene).  "
            "'parallel_passages': sibling passages branching from the predecessor "
            "and rejoining at the target (use for meaningfully different scenes)."
        ),
    )


class Phase5bOutput(BaseModel):
    """Output of Phase 5b: Residue Beat Content."""

    residue_content: list[ResidueContentItem] = Field(default_factory=list)


class FalseBranchDecisionItem(BaseModel):
    """LLM decision for a single false branch candidate."""

    candidate_index: int = Field(ge=0, description="Index into the candidates list")
    decision: str = Field(min_length=1, description="skip, diamond, or sidetrack")
    details: str = Field(default="", description="Brief rationale")
    diamond_summary_a: str = Field(default="")
    diamond_summary_b: str = Field(default="")
    sidetrack_summary: str = Field(default="")
    sidetrack_entities: list[str] = Field(default_factory=list)
    choice_label_enter: str = Field(default="")
    choice_label_return: str = Field(default="")

    @model_validator(mode="after")
    def _require_decision_specific_fields(self) -> FalseBranchDecisionItem:
        """R-5.9 / R-5.10: decision-specific fields MUST be populated.

        - sidetrack: sidetrack_summary MUST be non-empty (R-5.9)
        - diamond: diamond_summary_a AND diamond_summary_b MUST be non-empty (R-5.10)
        - skip: no additional content required

        POLISH has no semantic_validator hook (#1498); this Pydantic check is
        the only in-retry enforcement point. Phase 6 would otherwise create
        false branch beats with empty summaries that FILL has nothing to
        write from.
        """
        if self.decision == "sidetrack" and not self.sidetrack_summary:
            raise ValueError(
                "FalseBranchDecisionItem with decision='sidetrack' MUST have "
                "non-empty `sidetrack_summary` (R-5.9). FILL has no prose "
                "context to write the sidetrack beat without it."
            )
        if self.decision == "diamond" and (
            not self.diamond_summary_a or not self.diamond_summary_b
        ):
            raise ValueError(
                "FalseBranchDecisionItem with decision='diamond' MUST have "
                "non-empty `diamond_summary_a` AND `diamond_summary_b` "
                "(R-5.10). Both alternative passages need prose hints."
            )
        return self


class Phase5cOutput(BaseModel):
    """Output of Phase 5c: False Branch Decisions."""

    decisions: list[FalseBranchDecisionItem] = Field(default_factory=list)


class VariantSummaryItem(BaseModel):
    """Summary for a variant passage."""

    variant_id: str = Field(min_length=1)
    summary: str = Field(
        min_length=1, description="Variant passage summary reflecting active flags"
    )


class Phase5dOutput(BaseModel):
    """Output of Phase 5d: Variant Passage Summaries."""

    variant_summaries: list[VariantSummaryItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 5e: Ambiguous Feasibility — LLM output schema
# ---------------------------------------------------------------------------


class AmbiguousFeasibilityCase(BaseModel):
    """One passage with ambiguous feasibility flags for LLM review.

    Ambiguous means 2+ relevant flags with mixed residue weights
    (at least one heavy AND at least one light dilemma).
    """

    passage_id: str = Field(min_length=1)
    passage_summary: str = Field(default="")
    entities: list[str] = Field(default_factory=list)
    flags: list[str] = Field(
        min_length=1, description="Flag IDs that are ambiguous (mixed weights)"
    )


class FeasibilityDecisionItem(BaseModel):
    """LLM decision for one flag in an ambiguous passage."""

    passage_id: str = Field(min_length=1)
    flag_index: int = Field(ge=0, description="Index into the case's flags list")
    decision: str = Field(description='"variant" | "residue" | "irrelevant"')


class Phase5eOutput(BaseModel):
    """Output of Phase 5e: Ambiguous Feasibility Resolution."""

    feasibility_decisions: list[FeasibilityDecisionItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 5f: Transition Guidance — LLM output schema
# ---------------------------------------------------------------------------


class TransitionGuidanceItem(BaseModel):
    """Transition guidance for a collapsed passage."""

    passage_id: str = Field(min_length=1)
    transitions: list[str] = Field(
        min_length=1,
        description="One transition instruction per beat boundary (N-1 for N beats)",
    )


class Phase5fOutput(BaseModel):
    """Output of Phase 5f: Transition Guidance for Collapsed Passages."""

    transition_guidance: list[TransitionGuidanceItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

# Note: POLISH phases return the shared PhaseResult from models.pipeline.


class PolishResult(BaseModel):
    """Summary result of the complete POLISH stage execution.

    Captures key metrics from the passage graph created by POLISH.
    """

    passage_count: int = Field(default=0, description="Total passages created")
    choice_count: int = Field(default=0, description="Total choice edges created")
    variant_count: int = Field(default=0, description="Variant passages created")
    residue_count: int = Field(default=0, description="Residue beats created")
    sidetrack_count: int = Field(default=0, description="Sidetrack beats created")
    false_branch_count: int = Field(default=0, description="False branches applied")
