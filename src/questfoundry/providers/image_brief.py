"""Structured image brief for provider-agnostic prompt assembly.

The ``ImageBrief`` dataclass is the typed contract between the DRESS stage
(which gathers story data) and image providers (which know how to format
prompts for their backend).  See also :func:`flatten_brief_to_prompt` for
the default comma-join flattener.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageBrief:
    """Provider-agnostic structured brief for image generation.

    Built by :func:`build_image_brief` from graph nodes; consumed by
    :class:`PromptDistiller` implementations or flattened via
    :func:`flatten_brief_to_prompt`.
    """

    subject: str
    composition: str
    mood: str
    negative: str | None = None
    style_overrides: str | None = None
    entity_fragments: list[str] = field(default_factory=list)
    art_style: str | None = None
    art_medium: str | None = None
    palette: list[str] = field(default_factory=list)
    negative_defaults: str | None = None
    aspect_ratio: str = "16:9"
    category: str = "scene"


def flatten_brief_to_prompt(brief: ImageBrief) -> tuple[str, str | None]:
    """Flatten an ``ImageBrief`` into (positive, negative) prompt strings.

    This is the default flattener used when a provider does not implement
    :class:`PromptDistiller`.  It reproduces the original
    ``assemble_image_prompt`` behaviour, with the addition of
    ``style_overrides`` (previously stored but never included).

    Args:
        brief: Structured image brief.

    Returns:
        Tuple of (positive_prompt, negative_prompt_or_None).
    """
    palette_str = ", ".join(brief.palette) + " palette" if brief.palette else ""

    style_medium = f"{brief.art_style or ''}, {brief.art_medium or ''} style".strip(", ")

    positive_parts = [
        " and ".join(brief.entity_fragments) if brief.entity_fragments else "",
        brief.subject,
        brief.composition,
        brief.mood,
        style_medium if style_medium != "style" else "",
        palette_str,
        brief.style_overrides or "",
    ]
    positive = ", ".join(p for p in positive_parts if p)

    negative_parts = [
        brief.negative or "",
        brief.negative_defaults or "",
    ]
    negative = ", ".join(n for n in negative_parts if n)

    return positive, negative or None
