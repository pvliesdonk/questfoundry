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

from pydantic import BaseModel, Field

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
    negative_defaults: str = Field(
        min_length=1,
        description="Global things to avoid in image generation",
    )
    aspect_ratio: str = Field(
        default="16:9",
        min_length=1,
        description="Default image dimensions (e.g. 16:9, 1:1)",
    )


class Illustration(BaseModel):
    """Art asset with diegetic caption.

    Generated from an IllustrationBrief and linked to a passage
    via a Depicts edge.
    """

    asset: str = Field(min_length=1, description="Path to image file (e.g. assets/<hash>.png)")
    caption: str = Field(
        min_length=1,
        max_length=100,
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
    more as they unlock codewords. Linked to entity via HasEntry edge.
    """

    title: str = Field(
        min_length=1,
        description="Short display title for this entry (e.g., character name, location name)",
    )
    rank: int = Field(ge=1, description="Display order (1 = base knowledge, higher = deeper)")
    visible_when: list[str] = Field(
        default_factory=list,
        description="Codeword IDs that must all be present to unlock this tier",
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
        max_length=100,
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
# Stage result
# ---------------------------------------------------------------------------


class DressPhaseResult(PhaseResult):
    """Result of a single DRESS phase execution.

    Inherits all fields from PhaseResult. Allows DRESS-specific
    fields to be added without affecting other stages.
    """


class DressResult(BaseModel):
    """Overall DRESS stage result."""

    art_direction_created: bool = False
    entity_visuals_created: int = 0
    briefs_created: int = 0
    codex_entries_created: int = 0
    illustrations_generated: int = 0
    illustrations_failed: int = 0
    phases_completed: list[DressPhaseResult] = Field(default_factory=list)
