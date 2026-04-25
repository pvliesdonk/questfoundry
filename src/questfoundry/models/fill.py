"""FILL stage models.

These models define the voice document, prose output schemas,
review flags, and stage result container for the FILL stage.

FILL transforms passage summaries into prose. It takes a validated
story graph from GROW and produces playable content with consistent
voice and style.

See docs/design/procedures/fill.md for algorithm details.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from questfoundry.models.pipeline import PhaseResult

# ---------------------------------------------------------------------------
# Voice document
# ---------------------------------------------------------------------------


class VoiceDocument(BaseModel):
    """Stylistic contract governing all prose generation.

    Created in Phase 0 and referenced by every subsequent FILL call.
    Captures concrete prose decisions (POV, tense, register) that
    DREAM's high-level vision doesn't specify.
    """

    pov: Literal[
        "first_person", "second_person", "third_person_limited", "third_person_omniscient"
    ] = Field(description="Narrative point of view")
    pov_character: str = Field(
        default="",
        description=(
            "POV character (required for first_person and third_person_limited; "
            "empty for second_person and third_person_omniscient). See fill.md R-1.3."
        ),
    )
    tense: Literal["past", "present"] = Field(description="Narrative tense")
    voice_register: Literal["formal", "conversational", "literary", "sparse"] = Field(
        description="Formality and style"
    )
    sentence_rhythm: Literal["varied", "punchy", "flowing"] = Field(description="Pacing pattern")
    tone_words: list[str] = Field(
        min_length=1,
        description="Adjectives describing the voice (e.g. terse, wry, melancholic)",
    )
    avoid_words: list[str] = Field(
        default_factory=list,
        description="Words/phrases to never use",
    )
    avoid_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns to avoid (e.g. adverb-heavy, said-bookisms)",
    )

    @model_validator(mode="after")
    def _check_pov_character_required(self) -> VoiceDocument:
        if self.pov in ("first_person", "third_person_limited") and not self.pov_character:
            raise ValueError(
                f"pov_character is required when pov is {self.pov!r}; "
                "set it to the entity ID whose perspective the prose follows. See fill.md R-1.3."
            )
        return self


# ---------------------------------------------------------------------------
# Per-passage output
# ---------------------------------------------------------------------------


class EntityUpdate(BaseModel):
    """Micro-detail discovered during prose generation.

    FILL cannot create new entities — only update existing ones
    with additive details (appearance, mannerisms, voice).
    """

    entity_id: str = Field(min_length=1)
    field: str = Field(min_length=1, description="Field to update (e.g. 'appearance')")
    value: str = Field(min_length=1, description="New detail to add")


class SpokeLabelUpdate(BaseModel):
    """Final label for a hub-to-spoke choice.

    When GROW creates spokes with label=None, FILL generates labels
    alongside prose to ensure the choice text matches passage content.
    """

    choice_id: str = Field(min_length=1, description="ID of the hub-to-spoke choice")
    label: str = Field(
        min_length=1,
        max_length=80,  # Allows some flexibility for verbose styles and translations
        description="Final choice label, 3-60 chars (e.g., 'Examine the sketch')",
    )


class FillPassageOutput(BaseModel):
    """LLM output for a single passage prose generation."""

    passage_id: str = Field(min_length=1)
    prose: str = Field(default="", description="Generated prose text")
    entity_updates: list[EntityUpdate] = Field(
        default_factory=list,
        description="Micro-details discovered during generation",
    )
    spoke_labels: list[SpokeLabelUpdate] = Field(
        default_factory=list,
        description="Labels for hub-to-spoke choices (when hub passage generates prose)",
    )


# ---------------------------------------------------------------------------
# Expand blueprint (Phase 1a)
# ---------------------------------------------------------------------------


class ExpandBlueprint(BaseModel):
    """Scene blueprint for a single passage.

    Generated in Phase 1a (expand) and consumed by Phase 1 (prose).
    Separates creative planning from prose rendering so each LLM call
    has lower cognitive load.
    """

    passage_id: str = Field(min_length=1)
    sensory_palette: list[str] = Field(
        min_length=3,
        max_length=8,
        description="Sense-tagged details (e.g. 'sight: guttering torchlight')",
    )
    character_gestures: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Physical mannerisms or body language beats",
    )
    opening_move: Literal["dialogue", "action", "sensory_image", "internal_thought"] = Field(
        description="How the passage should begin"
    )
    craft_constraint: str = Field(
        default="",
        description="Structural or stylistic constraint (empty when probabilistically omitted)",
    )
    emotional_arc_word: str = Field(
        min_length=1,
        description="Single word capturing the emotional trajectory (e.g. 'dread', 'resolve')",
    )


class BatchedExpandOutput(BaseModel):
    """Phase 1a structured output: blueprints for a batch of passages."""

    blueprints: list[ExpandBlueprint] = Field(
        min_length=1,
        description="One blueprint per passage in the batch",
    )


class FillExtractOutput(BaseModel):
    """Analytical extraction of entity updates from generated prose.

    Used in two-step FILL mode: after prose is generated as plain text,
    a separate low-temperature call extracts entity micro-details. The
    prose itself is already stored on the passage node; this schema
    captures only the structured metadata.
    """

    entity_updates: list[EntityUpdate] = Field(
        default_factory=list,
        description="Micro-details discovered in the prose",
    )


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


class ReviewFlag(BaseModel):
    """Issue identified during Phase 2 review."""

    passage_id: str = Field(min_length=1)
    issue: str = Field(min_length=1, description="Description of the problem")
    issue_type: Literal[
        "voice_drift",
        "scene_type_mismatch",
        "summary_deviation",
        "continuity_break",
        "convergence_awkwardness",
        "flat_prose",
        "blueprint_bleed",
        "near_duplicate",
        "opening_trigram",
        "low_vocabulary",
    ] = Field(description="Category of the review issue")


# ---------------------------------------------------------------------------
# LLM phase output wrappers
# ---------------------------------------------------------------------------


class FillPhase0Output(BaseModel):
    """Phase 0 structured output: voice determination and story title."""

    voice: VoiceDocument
    story_title: str = Field(
        min_length=1,
        description="A compelling title for the story (2-8 words)",
        json_schema_extra={"strip_whitespace": True},
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_title_whitespace(cls, data: Any) -> Any:
        """Strip whitespace from story_title before min_length check."""
        if isinstance(data, dict) and isinstance(data.get("story_title"), str):
            data["story_title"] = data["story_title"].strip()
        return data


class FillPhase1Output(BaseModel):
    """Phase 1 structured output: single passage prose."""

    passage: FillPassageOutput


class FillPhase2Output(BaseModel):
    """Phase 2 structured output: review flags for a batch of passages."""

    flags: list[ReviewFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------


class FillPhaseResult(PhaseResult):
    """Result of a single FILL phase execution.

    Inherits all fields from PhaseResult. Allows FILL-specific
    fields to be added without affecting other stages.
    """


class FillEscalation(BaseModel):
    """A FILL escalation event collected during the run.

    Escalations represent contract violations that the stage cannot
    resolve locally — a missing entity (R-2.14) needs SEED, persistent
    quality issues after the final cycle (R-5.2) need POLISH or upstream
    fixes. The stage collects them rather than failing fast on the
    first one (so the user sees the full set), then raises
    ``FillContractError`` at stage exit if any were recorded.
    """

    kind: Literal[
        "missing_entity",
        "unresolved_review_flags",
        "voice_research_failed",
        "blueprint_validation_failed",
        "entity_extract_failed",
    ] = Field(description="What kind of escalation this is.")
    passage_id: str = Field(
        description="Passage where the escalation was raised. Empty string if not passage-scoped."
    )
    detail: str = Field(
        description="Human-readable description of the specific violation.",
    )
    upstream_stage: Literal["SEED", "GROW", "POLISH", "FILL"] = Field(
        description="Which stage owns the fix. ``FILL`` for self-owned failures "
        "(LLM call failures during voice research, blueprint validation, or "
        "entity extraction — rerun FILL or adjust provider settings).",
    )


class FillResult(BaseModel):
    """Overall FILL stage result."""

    passages_filled: int = 0
    passages_flagged: int = 0
    entity_updates_applied: int = 0
    review_cycles: int = 0
    phases_completed: list[FillPhaseResult] = Field(default_factory=list)
