# src/questfoundry/providers/checkpoint_styles.py
"""Static checkpoint style metadata for image-generation renderers.

Used by DRESS Phase 0 (prevention — when `--image-provider` is set, biases
ArtDirection toward checkpoint-compatible styles) and the A1111 distiller
(recovery — adapts CLIP-tag selection to the active checkpoint).

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
        Dict with keys ``label``, ``style_hints``, ``incompatible_styles``.
    """
    lowered = model.lower()
    for pattern, info in _CHECKPOINT_STYLE_MAP:
        if pattern.search(lowered):
            return info
    raise AssertionError(  # pragma: no cover — default fallback always matches
        "default-fallback entry must always match"
    )
