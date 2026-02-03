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

    pov: Literal["first", "second", "third_limited", "third_omniscient"] = Field(
        description="Narrative point of view"
    )
    pov_character: str = Field(
        default="",
        description="Whose perspective (for limited POVs). Empty for omniscient.",
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
    exemplar_passages: list[str] = Field(
        default_factory=list,
        description="Optional examples of the target voice",
    )


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


class FillPassageOutput(BaseModel):
    """LLM output for a single passage prose generation.

    When the LLM cannot write poly-state prose for a shared beat,
    it sets ``flag`` to ``incompatible_states`` and leaves ``prose`` empty.
    """

    passage_id: str = Field(min_length=1)
    prose: str = Field(default="", description="Generated prose text")
    flag: Literal["ok", "incompatible_states"] = Field(
        default="ok",
        description="Signal for poly-state failures requiring beat splitting",
    )
    flag_reason: str = Field(
        default="",
        description="Explanation when flag is incompatible_states",
    )
    entity_updates: list[EntityUpdate] = Field(
        default_factory=list,
        description="Micro-details discovered during generation",
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


class FillResult(BaseModel):
    """Overall FILL stage result."""

    passages_filled: int = 0
    passages_flagged: int = 0
    entity_updates_applied: int = 0
    review_cycles: int = 0
    phases_completed: list[FillPhaseResult] = Field(default_factory=list)
