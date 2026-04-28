"""DRESS stage models.

These models define the art direction, entity visuals, illustration
briefs, codex entries, and stage result containers for the DRESS stage.

DRESS generates the presentation layer — art direction, illustrations,
and codex — for a completed story. It operates on finished prose and
entities, adding visual and encyclopedic content without modifying
narrative structure.

See docs/design/procedures/dress.md for algorithm details.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from questfoundry.models.pipeline import PhaseResult

IllustrationCategory = Literal["scene", "portrait", "vista", "item_detail", "cover"]

# ---------------------------------------------------------------------------
# Persistent nodes (exported by SHIP)
# ---------------------------------------------------------------------------


class ArtDirection(BaseModel):
    """Global visual identity document for the story's presentation layer.

    Analogous to FILL's VoiceDocument but for visual style. Created
    through the standard discuss/summarize/serialize pattern in Phase 0.
    """

    style: str = Field(min_length=1, description="Art style (e.g. watercolor, digital painting)")
    medium: str = Field(min_length=1, description="What it looks like it was made with")
    palette: list[str] = Field(
        min_length=1,
        description="Dominant colors / mood",
    )
    composition_notes: str = Field(
        min_length=1,
        description="Framing preferences",
    )
    style_exclusions: str = Field(
        min_length=1,
        description=(
            "Visual styles to exclude across all story images — story-tone "
            "prohibitions only (e.g. 'no photorealism, no modern clothing for "
            "a Victorian setting'). Do NOT include renderer quality fillers "
            "(blurry, watermark, etc.) — those are auto-injected by the "
            "render pipeline."
        ),
    )
    aspect_ratio: str = Field(
        min_length=1,
        description="Default image dimensions (e.g. 16:9, 1:1)",
    )


class Illustration(BaseModel):
    """Art asset with descriptive caption for accessibility.

    Generated from an IllustrationBrief and linked to a passage
    via a Depicts edge.
    """

    asset: str = Field(min_length=1, description="Path to image file (e.g. assets/<hash>.png)")
    caption: str = Field(
        min_length=1,
        max_length=100,  # Defensive max; prompt targets 10-60 chars
        description="Short descriptive caption for alt-text (10-60 chars ideal)",
    )
    category: IllustrationCategory = Field(
        description="Image category",
    )
    quality: Literal["placeholder", "low", "high"] = Field(
        default="high",
        description="Image quality tier: placeholder, low, or high",
    )


class CodexEntry(BaseModel):
    """Player-facing encyclopedia entry for an entity.

    Multiple entries per entity enable spoiler graduation: players see
    more as they accumulate state flags through gameplay. Linked to
    entity via HasEntry edge. (SHIP later projects a subset of state
    flags as player-visible "codewords" for the gamebook export, but
    DRESS gates internally on state flag IDs — see dress.md R-3.7.)
    """

    title: str = Field(
        min_length=1,
        description="Short display title for this entry (e.g., character name, location name)",
    )
    rank: int = Field(ge=1, description="Display order (1 = base knowledge, higher = deeper)")
    visible_when: list[str] = Field(
        default_factory=list,
        description="State flag IDs that must all be present to unlock this tier (see dress.md R-3.7)",
    )
    content: str = Field(min_length=1, description="Diegetic content — in-world voice, no spoilers")


# ---------------------------------------------------------------------------
# Working nodes (not exported by SHIP)
# ---------------------------------------------------------------------------


class EntityVisual(BaseModel):
    """Per-entity visual identity profile.

    Ensures consistent appearance across all illustrations featuring
    this entity. The reference_prompt_fragment is injected into every
    image prompt where the entity appears.
    """

    description: str = Field(min_length=1, description="Prose description of appearance")
    distinguishing_features: list[str] = Field(
        min_length=1,
        description="Key visual identifiers",
    )
    color_associations: list[str] = Field(
        default_factory=list,
        description="Colors tied to this entity",
    )
    reference_prompt_fragment: str = Field(
        min_length=1,
        description="Injected into every image prompt featuring this entity",
    )


class EntityVisualWithId(EntityVisual):
    """EntityVisual with the entity ID it describes.

    Used in DressPhase0Output to pair each visual profile with its entity.
    """

    entity_id: str = Field(min_length=1, description="Entity this visual describes")


class IllustrationBrief(BaseModel):
    """Structured image prompt with priority scoring.

    One brief per passage. Only selected briefs are rendered into
    Illustration nodes during Phase 4.

    All fields are required (no defaults) so that JSON_MODE (json_schema)
    enforces the complete schema. The TOOL strategy (function_calling) on
    OpenAI models allowed caption to be omitted ~50% of the time.
    """

    priority: int = Field(ge=1, le=3, description="1=must-have, 2=important, 3=nice-to-have")
    category: IllustrationCategory = Field(
        description="Image category",
    )
    subject: str = Field(min_length=1, description="What the image depicts")
    entities: list[str] = Field(
        description="Entity IDs present in scene",
    )
    caption: str = Field(
        min_length=1,
        max_length=100,  # Defensive max; prompt targets 10-60 chars
        description="Short descriptive caption for alt-text (10-60 chars ideal)",
    )
    mood: str = Field(min_length=1, description="Emotional tone")
    composition: str = Field(min_length=1, description="Framing / camera notes")
    style_overrides: str = Field(
        description="Deviations from global art direction (usually empty string)",
    )
    negative: str = Field(description="Things to avoid in this image")


# ---------------------------------------------------------------------------
# LLM phase output wrappers
# ---------------------------------------------------------------------------


class DressPhase0Output(BaseModel):
    """Phase 0 structured output: art direction + entity visuals."""

    art_direction: ArtDirection
    entity_visuals: list[EntityVisualWithId] = Field(min_length=1)


class DressPhase1Output(BaseModel):
    """Phase 1 structured output: illustration brief for a single passage.

    The llm_adjustment field captures the LLM's narrative judgment
    about visual priority, which is combined with the structural
    base score to determine final priority.
    """

    brief: IllustrationBrief
    llm_adjustment: int = Field(
        ge=-2,
        le=2,
        description="LLM priority adjustment (-2 to +2)",
    )


class DressPhase2Output(BaseModel):
    """Phase 2 structured output: codex entries for a single entity."""

    entries: list[CodexEntry] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Batched LLM output wrappers
# ---------------------------------------------------------------------------


class BatchedBriefItem(BaseModel):
    """One brief within a batched response."""

    passage_id: str = Field(min_length=1, description="Passage ID (must match exactly)")
    brief: IllustrationBrief
    llm_adjustment: int = Field(
        ge=-2,
        le=2,
        description="LLM priority adjustment (-2 to +2)",
    )


class BatchedBriefOutput(BaseModel):
    """Phase 1 batched output: briefs for multiple passages."""

    briefs: list[BatchedBriefItem] = Field(min_length=1)


class BatchedCodexItem(BaseModel):
    """Codex entries for one entity within a batch."""

    entity_id: str = Field(min_length=1, description="Entity ID (must match exactly)")
    entries: list[CodexEntry] = Field(min_length=1)


class BatchedCodexOutput(BaseModel):
    """Phase 2 batched output: codex entries for multiple entities."""

    entities: list[BatchedCodexItem] = Field(min_length=1)


class SpoilerLeak(BaseModel):
    """One spoiler-direction violation between two ranks of one entity."""

    lower_rank: int = Field(
        ge=1,
        description="Rank of the entry that prematurely discloses information",
    )
    higher_rank: int = Field(
        ge=2,
        description="Rank of the entry whose reveal was leaked",
    )
    leaked_content: str = Field(
        min_length=1,
        description="Short quote or paraphrase of the leaked information",
    )

    @model_validator(mode="after")
    def _check_rank_ordering(self) -> SpoilerLeak:
        # R-3.6's spoiler direction is strictly low → high. An LLM that
        # returns lower_rank ≥ higher_rank has either inverted the
        # arguments or invented a self-referential leak; either way the
        # downstream retry feedback would be nonsensical, so reject at
        # validation time so the LLM repair loop fixes it.
        if self.lower_rank >= self.higher_rank:
            msg = (
                f"SpoilerLeak: lower_rank ({self.lower_rank}) must be "
                f"strictly less than higher_rank ({self.higher_rank})"
            )
            raise ValueError(msg)
        return self


class SpoilerCheckResult(BaseModel):
    """Result of an LLM spoiler check on one entity's codex entries (R-3.6)."""

    has_leak: bool = Field(
        description="True if any lower-ranked entry leaks higher-ranked content",
    )
    leaks: list[SpoilerLeak] = Field(
        default_factory=list,
        description="Detected spoiler violations (empty if has_leak is False)",
    )
    reason: str = Field(
        default="",
        description="Brief LLM explanation; populated when has_leak is True",
    )


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------


class DressPhaseResult(PhaseResult):
    """Result of a single DRESS phase execution.

    Inherits all fields from PhaseResult. Allows DRESS-specific
    fields to be added without affecting other stages.
    """


class DressEscalation(BaseModel):
    """A DRESS escalation event collected during the run.

    Mirrors ``FillEscalation``: when a ``batch_llm_calls`` invocation
    exhausts retries on an item, the stage records a per-item
    escalation rather than silently dropping the item via the
    ``_errors`` underscore-discard convention. ``DressStage.execute``
    folds these into the ``DressStageError`` raised at exit so the
    failure surface points at the batch responses that misbehaved
    (not just the missing artifacts caught by the output contract).
    """

    kind: Literal[
        "briefs_batch_failed",
        "codex_batch_failed",
        "distill_batch_failed",
    ] = Field(description="Which DRESS phase produced this escalation.")
    item_id: str = Field(
        description=(
            "Affected item ID. ``passage::*`` for briefs, prefixed entity ID "
            "(e.g. ``character::clara_yu``) for codex, ``brief::*`` for distill. "
            "Empty string only when the failure is not item-scoped."
        ),
    )
    detail: str = Field(
        description="Human-readable description of the specific failure (exception type + message).",
    )
    upstream_stage: Literal["DRESS", "DREAM", "BRAINSTORM", "GROW"] = Field(
        description=(
            "Which stage owns the fix. ``DRESS`` for self-owned LLM-call "
            "failures (rerun DRESS or adjust provider settings)."
        ),
    )


class DressResult(BaseModel):
    """Overall DRESS stage result."""

    art_direction_created: bool = False
    entity_visuals_created: int = 0
    briefs_created: int = 0
    codex_entries_created: int = 0
    illustrations_generated: int = 0
    illustrations_failed: int = 0
    phases_completed: list[DressPhaseResult] = Field(default_factory=list)
