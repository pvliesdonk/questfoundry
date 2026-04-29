# src/questfoundry/providers/checkpoint_styles.py
"""Static checkpoint style metadata for image-generation renderers.

Used by DRESS Phase 0 (prevention — when `--image-provider` is set, biases
ArtDirection toward checkpoint-compatible styles) and the A1111 distiller
(recovery — adapts CLIP-tag selection to the active checkpoint).

The A1111 distiller also reads `prompt_format` (via
`prompt_format_for_checkpoint`) to choose between LLM-driven CLIP-tag
distillation (CLIP-encoder checkpoints) and direct prose flattening
(T5-encoder checkpoints like Flux on Forge Neo) — see PR #1559.

Pattern matching:
    Patterns are matched against the lowercased checkpoint filename via
    `re.Pattern.search`. First match wins. Specific patterns MUST precede
    generic ones — the dreamshaper family is the canonical example
    (Lightning/Alpha specific → XL generic → SD1.5 fallback).
    The empty-pattern default-fallback entry is always last.

Adding a new entry:
    Insert above the default-fallback entry, in priority order.
    Test: add a parametrize row to `tests/unit/test_checkpoint_styles.py`
    asserting the expected label substring.
"""

from __future__ import annotations

import re

_CHECKPOINT_STYLE_MAP: tuple[tuple[re.Pattern[str], dict[str, str]], ...] = (
    # ----- Flux (NF4 quantised; both v1 and v2 quants share identity) -----
    (
        re.compile(r"flux"),
        {
            "label": "Flux (photorealistic / highly-detailed)",
            "style_hints": (
                "photorealistic imagery, extreme fine detail, architectural "
                "photography, natural lighting, product shots, documentary "
                "portraiture, coherent text in scene"
            ),
            "incompatible_styles": (
                "anime, manga, cel-shading, watercolor washes, heavy "
                "painterly texture, low-detail illustration styles; "
                "also: negative-prompt weighting is weak on Flux — "
                "do not rely on negative prompts for strong style exclusion"
            ),
            "good_example": (
                'style="cinematic urban photography", medium="digital photograph with shallow DOF"'
            ),
            "bad_example": (
                'style="watercolor wash", medium="hand-painted ink" '
                "(Flux is tuned for photorealism; painterly media will fight the model)"
            ),
            "prompt_format": "natural_language",
        },
    ),
    # ----- Coloring-book fine-tune (SD1.5 base) -----
    (
        re.compile(r"coloring.?book"),
        {
            "label": "Coloring Book (line-art SD1.5)",
            "style_hints": (
                "clean outlines on white background, no fill colors, "
                "strong linework, simple shapes, children's-book-friendly "
                "compositions, decorative borders"
            ),
            "incompatible_styles": (
                "photorealism, color renders, painterly textures, "
                "complex shading, dark backgrounds, photographic lighting"
            ),
            "good_example": ('style="bold ink linework", medium="black-and-white outline drawing"'),
            "bad_example": (
                'style="photorealistic portrait", '
                'medium="oil paint with rich color" '
                "(this checkpoint is fine-tuned for line-art only; "
                "color renders will fail)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- Juggernaut XL (photorealistic SDXL) -----
    (
        re.compile(r"juggernaut"),
        {
            "label": "Juggernaut XL (photorealistic SDXL)",
            "style_hints": (
                "photorealistic portraits, cinematic lighting, "
                "sharp textural detail, skin pores, fabric weave, "
                "dramatic rim lighting, environmental storytelling"
            ),
            "incompatible_styles": (
                "anime, cartoon, flat illustration, watercolor, comic-book ink outlines, chibi"
            ),
            "good_example": ('style="gritty photorealistic urban", medium="digital photo"'),
            "bad_example": (
                'style="watercolor wash", medium="traditional ink" '
                "(Juggernaut is tuned for photorealism; "
                "stylised media will fight the checkpoint)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- Animagine XL (anime-focused SDXL) -----
    (
        re.compile(r"animagine"),
        {
            "label": "Animagine XL (anime SDXL)",
            "style_hints": (
                "anime illustration, Danbooru-style tag vocabulary, "
                "clean cell shading, expressive character art, "
                "vivid saturated palette, manga panel compositions"
            ),
            "incompatible_styles": (
                "photorealism, photography-style lighting, "
                "gritty texture, oil painting, detailed backgrounds "
                "without anime stylisation"
            ),
            "good_example": (
                'style="anime illustration with cel shading", medium="digital anime art"'
            ),
            "bad_example": (
                'style="documentary photograph", medium="35mm film" '
                "(Animagine is anime-specialised; "
                "photographic styles produce off-distribution outputs)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- DreamShaperXL Lightning / Alpha (must precede generic dreamshaperxl) -----
    (
        re.compile(r"dreamshaperxl.*lightning|dreamshaperxl.*alpha"),
        {
            "label": "DreamShaperXL Lightning / Alpha (fast fantasy SDXL)",
            "style_hints": (
                "fantasy concept art, painterly illustration, "
                "vibrant color, dramatic character portraits, "
                "acceptable in 4-8 steps"
            ),
            "incompatible_styles": (
                "photorealism (style is stylised by design), "
                "highly detailed textures at very low step counts, "
                "strict architectural accuracy"
            ),
            "good_example": (
                'style="dramatic fantasy concept art", medium="painterly digital illustration"'
            ),
            "bad_example": (
                'style="hyperrealistic skin detail at 4K", '
                'medium="macro photograph" '
                "(Lightning checkpoints sacrifice fine detail for speed)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- DreamShaperXL standard -----
    (
        re.compile(r"dreamshaperxl|dreamshaper.*xl"),
        {
            "label": "DreamShaperXL (versatile fantasy SDXL)",
            "style_hints": (
                "fantasy illustration, painterly portraits, "
                "concept-art style, stylised environments, "
                "strong use of negative space"
            ),
            "incompatible_styles": (
                "strict photorealism, clinical document photography, flat-color infographic styles"
            ),
            "good_example": (
                'style="painterly fantasy illustration", medium="digital concept art"'
            ),
            "bad_example": (
                'style="clinical product photography", '
                'medium="catalog studio shot" '
                "(DreamShaperXL is stylised by design; "
                "strict photo-real fights the model)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- DreamShaper SD1.5 (generic, must come after XL variants) -----
    (
        re.compile(r"dreamshaper"),
        {
            "label": "DreamShaper (versatile SD1.5)",
            "style_hints": (
                "general-purpose stylised illustration, "
                "fantasy character art, soft painterly lighting, "
                "portrait and environmental compositions; "
                "notably versatile — adapt style tags rather than "
                "leaning on a single category"
            ),
            "incompatible_styles": (
                "extreme photorealism (slightly stylised by design), "
                "Danbooru/anime tag grammar (use natural descriptors instead)"
            ),
            "good_example": (
                'style="painterly fantasy character portrait", medium="soft digital illustration"'
            ),
            "bad_example": (
                'style="Danbooru anime tags", medium="cel-shading" '
                "(DreamShaper SD1.5 expects natural descriptors, "
                "not anime tag grammar)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- SDXL base -----
    (
        re.compile(r"sd_xl_base|sdxl_base|sdxl-base"),
        {
            "label": "SDXL Base (general-purpose SDXL)",
            "style_hints": (
                "broad style range, photography, illustration, concept art; "
                "best results with refiner pass or ControlNet; "
                "responds well to explicit style tokens"
            ),
            "incompatible_styles": (
                "anime-specific Danbooru vocabulary without style priming, "
                "very low step counts (needs ≥30 steps for coherence)"
            ),
            "good_example": (
                'style="cinematic illustration with explicit style tokens", medium="digital art"'
            ),
            "bad_example": (
                'style="anime without style priming", '
                'medium="bare Danbooru tags" '
                "(SDXL base needs explicit style direction; "
                "bare anime grammar underperforms)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- SD 1.5 base / pruned -----
    (
        re.compile(r"v1[-_]5|sd[-_]?1[-._]?5"),
        {
            "label": "SD 1.5 (general-purpose base)",
            "style_hints": (
                "broad style range at 512-768px, watercolor, "
                "ink illustration, painterly portraiture; "
                "well-supported by community LoRAs"
            ),
            "incompatible_styles": (
                "photorealistic skin detail at high resolution "
                "(768px ceiling limits fine detail), "
                "SDXL-native aspect ratios"
            ),
            "good_example": ('style="watercolor portraiture", medium="ink illustration"'),
            "bad_example": (
                'style="hyperrealistic skin at 1024px", '
                'medium="macro studio photograph" '
                "(SD 1.5 caps at ~768px; high-detail photoreal won't render)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
    # ----- Default — always matches; must be last -----
    (
        re.compile(r""),
        {
            "label": "Unknown checkpoint (SD general-purpose defaults)",
            "style_hints": (
                "broad range: illustration, painterly, concept art; "
                "Stable Diffusion generally excels at stylised imagery, "
                "fantasy environments, and character portraiture; "
                "use explicit style tokens (e.g. 'watercolor painting', "
                "'cinematic photograph') for best results"
            ),
            "incompatible_styles": (
                "coherent embedded text, photographic product catalogs "
                "without specialised fine-tuning"
            ),
            "good_example": (
                'style="painterly fantasy illustration with explicit style tokens", '
                'medium="digital concept art"'
            ),
            "bad_example": (
                'style="coherent embedded text", '
                'medium="document scan with readable signage" '
                "(Stable Diffusion generally cannot render legible text)"
            ),
            "prompt_format": "clip_tags",
        },
    ),
)


def resolve_checkpoint_style(model: str) -> dict[str, str]:
    """Look up style metadata for a checkpoint name.

    Always returns a populated dict — the empty-pattern default-fallback
    entry guarantees a match.

    Args:
        model: Checkpoint filename (with or without extension). Case-insensitive.

    Returns:
        Dict with keys ``label``, ``style_hints``, ``incompatible_styles``,
        ``good_example``, ``bad_example``.
    """
    lowered = model.lower()
    for pattern, info in _CHECKPOINT_STYLE_MAP:
        if pattern.search(lowered):
            return info
    raise AssertionError(  # pragma: no cover — default fallback always matches
        "default-fallback entry must always match"
    )


def prompt_format_for_checkpoint(model: str | None) -> str:
    """Return the prompt format the active checkpoint expects.

    Returns ``"clip_tags"`` when no model is set — preserves the LLM-distill
    default path through ``A1111ImageProvider.distill_prompt``. CLIP-encoder
    checkpoints (SDXL, SD1.5, etc.) take ``"clip_tags"``; T5-encoder
    checkpoints (Flux) take ``"natural_language"``. See PR #1559.

    Args:
        model: Checkpoint filename (with or without extension), or None / empty
            string when no checkpoint is selected.

    Returns:
        ``"clip_tags"`` or ``"natural_language"``.
    """
    if not model:
        return "clip_tags"
    return resolve_checkpoint_style(model)["prompt_format"]
