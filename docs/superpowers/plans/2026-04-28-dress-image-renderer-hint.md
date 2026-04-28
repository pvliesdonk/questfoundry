# DRESS Image-Renderer Hint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ArtDirection` (DRESS Phase 0) renderer-aware when the image provider is selected at orchestrator init, and rename `negative_defaults` → `style_exclusions` to remove the small-model field-name confusion.

**Architecture:** Two layers share one static checkpoint style map. (1) DRESS Phase 0 prevention: when `--image-provider` is set, `dress_discuss.yaml` gets a `{image_renderer_section}` populated from the map; otherwise empty. (2) Distiller recovery: replace the existing crude `checkpoint_hint` in `image_a1111.py` with structured map data. Bundled rename gives the field one unambiguous semantic ("story-tone visual prohibitions").

**Tech Stack:** Python 3.11+, `uv`, `pydantic`, `ruamel.yaml`, `pytest`, `ruff`, `mypy`. Existing patterns: PromptLoader → `template.system.format(**kwargs)`; `re.compile` patterns matched against lowercased filenames in `_resolve_preset` (precedent for the new `_CHECKPOINT_STYLE_MAP`).

**Spec:** `docs/superpowers/specs/2026-04-28-dress-image-renderer-hint-design.md`. Closes #1557.

---

## File Structure

**New files (1):**
- `src/questfoundry/providers/checkpoint_styles.py` — `_CHECKPOINT_STYLE_MAP` regex table + `resolve_checkpoint_style()` lookup
- `tests/unit/test_checkpoint_styles.py` — pattern resolution tests

**Modified files (12):**
- `docs/design/procedures/dress.md` — R-1.2 field-list + worked-example
- `docs/design/procedures/ship.md` — R-3.9 partial-DRESS warning enumeration
- `src/questfoundry/models/dress.py` — `ArtDirection.negative_defaults` → `style_exclusions` + new description
- `src/questfoundry/providers/image_brief.py` — `ImageBrief.negative_defaults` → `style_exclusions` + flattener concat
- `src/questfoundry/providers/image_a1111.py` — distiller `checkpoint_hint` body replaced with map lookup; concat-site rename
- `src/questfoundry/providers/image_openai.py` — concat-site rename
- `src/questfoundry/pipeline/stages/dress.py` — `_build_image_renderer_section()` helper + threading; ArtDirection→ImageBrief construction site rename
- `src/questfoundry/export/context.py` — `_REQUIRED_ART_DIRECTION_FIELDS` tuple
- `prompts/templates/dress_discuss.yaml` — insert `{image_renderer_section}` placeholder before `## Guidelines`
- `prompts/templates/dress_serialize.yaml` — schema field rename + GOOD/BAD instruction (story-tone vs renderer fillers)
- `prompts/templates/dress_brief.yaml` + `dress_brief_batch.yaml` — field-name references
- Tests: any fixture or assertion using the old name

---

## Task 1: Spec docs (R-1.2 + R-3.9 rename)

Specs precede code per CLAUDE.md "Spec-first fix order." This task touches only docs, no code, no tests.

**Files:**
- Modify: `docs/design/procedures/dress.md`
- Modify: `docs/design/procedures/ship.md`

- [ ] **Step 1: Open `dress.md` and update R-1.2 field enumeration (line ~40 and ~286)**

Current R-1.2 lists: "ArtDirection fields include `style`, `medium`, `palette`, `composition_notes`, `negative_defaults`, `aspect_ratio`."

Change every occurrence of `negative_defaults` to `style_exclusions` in that rule. Both the prose at line ~40 and the consolidated table around line ~286.

- [ ] **Step 2: Update the worked-example YAML block in `dress.md` (around line ~382)**

Find the YAML showing `negative_defaults: "no photorealism, no cartoon styling, no modern clothing"`. Rename the field to `style_exclusions` and update the line above it (the field's commentary or description) to: `# Story-tone visual prohibitions only — renderer-quality fillers (blurry, watermark, etc.) are auto-injected at render time.`

- [ ] **Step 3: Open `ship.md` and update R-3.9**

R-3.9 lists DRESS-required visual fields: "`style`, `medium`, `palette`, `composition_notes`, `negative_defaults`, `aspect_ratio`". Change `negative_defaults` → `style_exclusions`. Rest of R-3.9 unchanged.

- [ ] **Step 4: Verify no other doc references the old name**

```bash
rg "negative_defaults" docs/
```
Expected: zero hits in `docs/`.

- [ ] **Step 5: Commit**

```bash
git add docs/design/procedures/dress.md docs/design/procedures/ship.md
git commit -m "docs(dress,ship): rename negative_defaults → style_exclusions in R-1.2 + R-3.9 (#1557)"
```

---

## Task 2: Checkpoint style map module — failing test

TDD pure — the new module is testable in isolation. Write tests first.

**Files:**
- Create: `tests/unit/test_checkpoint_styles.py`

- [ ] **Step 1: Create the failing test file**

```python
# tests/unit/test_checkpoint_styles.py
"""Tests for checkpoint style metadata lookup (#1557)."""

from __future__ import annotations

import pytest

from questfoundry.providers.checkpoint_styles import resolve_checkpoint_style


class TestResolveCheckpointStyle:
    """`resolve_checkpoint_style()` returns label/style_hints/incompatible_styles
    for a checkpoint name. Always returns a populated dict (default fallback
    if no specific pattern matches)."""

    @pytest.mark.parametrize(
        ("model", "label_substring"),
        [
            # User's loaded checkpoints at A1111 (2026-04-28)
            ("flux1-dev-bnb-nf4-v2.safetensors", "Flux"),
            ("flux1-dev-bnb-nf4.safetensors", "Flux"),
            ("coloring_book.ckpt", "Coloring Book"),
            ("v1-5-pruned-emaonly.safetensors", "SD 1.5"),
            ("animagine-xl.safetensors", "Animagine"),
            ("Dreamshaper.safetensors", "DreamShaper"),
            ("sd_xl_base_1.0.safetensors", "SDXL Base"),
            ("dreamshaperXL_lightningDPMSDE.safetensors", "DreamShaperXL Lightning"),
            ("juggernautXL_ragnarokBy.safetensors", "Juggernaut"),
            ("dreamshaperXL_alpha2Xl10.safetensors", "DreamShaperXL Lightning"),
        ],
    )
    def test_known_checkpoint_resolves_to_specific_label(
        self, model: str, label_substring: str
    ) -> None:
        info = resolve_checkpoint_style(model)
        assert label_substring in info["label"]
        assert info["style_hints"]
        assert info["incompatible_styles"]

    def test_unknown_checkpoint_falls_back_to_default(self) -> None:
        info = resolve_checkpoint_style("totally-made-up-model.safetensors")
        assert "Unknown" in info["label"] or "general-purpose" in info["label"].lower()
        assert info["style_hints"]
        assert info["incompatible_styles"]

    def test_pattern_ordering_dreamshaperxl_lightning_wins_over_generic(self) -> None:
        # The dreamshaper family has three patterns; the most specific must win.
        # dreamshaperxl_lightning|alpha → "DreamShaperXL Lightning / Alpha"
        # dreamshaperxl|dreamshaper.*xl → "DreamShaperXL"
        # dreamshaper                    → "DreamShaper" (SD1.5)
        lightning = resolve_checkpoint_style("dreamshaperXL_lightningDPMSDE.safetensors")
        assert "Lightning" in lightning["label"] or "Alpha" in lightning["label"]

        alpha = resolve_checkpoint_style("dreamshaperXL_alpha2Xl10.safetensors")
        assert "Lightning" in alpha["label"] or "Alpha" in alpha["label"]

        sd15 = resolve_checkpoint_style("Dreamshaper.safetensors")
        assert "Lightning" not in sd15["label"]
        assert "Alpha" not in sd15["label"]
        assert "XL" not in sd15["label"]

    def test_returns_required_keys(self) -> None:
        info = resolve_checkpoint_style("anything.safetensors")
        assert set(info.keys()) >= {"label", "style_hints", "incompatible_styles"}

    def test_case_insensitive_matching(self) -> None:
        # Patterns match against lowercased filename.
        upper = resolve_checkpoint_style("FLUX1-DEV.safetensors")
        lower = resolve_checkpoint_style("flux1-dev.safetensors")
        assert upper == lower
```

- [ ] **Step 2: Run test to verify it fails (module doesn't exist yet)**

```bash
uv run --frozen pytest tests/unit/test_checkpoint_styles.py -v
```
Expected: collection error or `ModuleNotFoundError: questfoundry.providers.checkpoint_styles`.

---

## Task 3: Checkpoint style map module — implementation

**Files:**
- Create: `src/questfoundry/providers/checkpoint_styles.py`

- [ ] **Step 1: Create the module**

```python
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
                "anime, cartoon, flat illustration, "
                "watercolor, comic-book ink outlines, chibi"
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
                "strict photorealism, clinical document photography, "
                "flat-color infographic styles"
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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run --frozen pytest tests/unit/test_checkpoint_styles.py -v
```
Expected: 14 passed (10 parametrize rows + 4 standalone tests).

- [ ] **Step 3: Run mypy + ruff on the new module**

```bash
uv run --frozen mypy src/questfoundry/providers/checkpoint_styles.py
uv run --frozen ruff check src/questfoundry/providers/checkpoint_styles.py tests/unit/test_checkpoint_styles.py
```
Expected: both clean.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/providers/checkpoint_styles.py tests/unit/test_checkpoint_styles.py
git commit -m "feat(providers): add _CHECKPOINT_STYLE_MAP for renderer-aware art direction (#1557)"
```

---

## Task 4: Atomic rename `negative_defaults` → `style_exclusions`

A global symbol rename across model + dataclass + 3 providers + DRESS stage + 3 prompt templates + export-context tuple + tests. Doing it atomically keeps the test suite green between commits — splitting causes intermediate red states. Per CLAUDE.md "Replace directly, no compat shim."

**Files:**
- Modify: `src/questfoundry/models/dress.py:46`
- Modify: `src/questfoundry/providers/image_brief.py:32,68`
- Modify: `src/questfoundry/providers/image_a1111.py:354`
- Modify: `src/questfoundry/providers/image_openai.py:145,147`
- Modify: `src/questfoundry/pipeline/stages/dress.py:2142`
- Modify: `src/questfoundry/export/context.py:44`
- Modify: `prompts/templates/dress_serialize.yaml:21`
- Modify: `prompts/templates/dress_brief.yaml:85`
- Modify: `prompts/templates/dress_brief_batch.yaml:74`
- Modify: any test/fixture using the old name

- [ ] **Step 1: Inventory all sites**

```bash
rg -n "negative_defaults" src/ tests/ prompts/
```
Expected: ~15-20 hits across the listed files. If any new site appears post-merge, include it.

- [ ] **Step 2: Update the Pydantic model + description**

In `src/questfoundry/models/dress.py` around line 46, replace:

```python
    negative_defaults: str = Field(
        min_length=1,
        description="Global things to avoid in image generation",
    )
```

with:

```python
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
```

- [ ] **Step 3: Update `ImageBrief` dataclass**

In `src/questfoundry/providers/image_brief.py`, rename the field at line 32 (`negative_defaults: str | None = None` → `style_exclusions: str | None = None`) and at line ~68 (the flattener concat). Keep semantics identical.

- [ ] **Step 4: Update provider concatenation sites**

`src/questfoundry/providers/image_a1111.py:354`:

```python
# Before
n for n in [brief.negative or "", brief.negative_defaults or ""] if n
# After
n for n in [brief.negative or "", brief.style_exclusions or ""] if n
```

`src/questfoundry/providers/image_openai.py:145-147`:

```python
# Before
if brief.negative_defaults:
    negative = (
        f"{negative}, {brief.negative_defaults}" if negative else brief.negative_defaults
    )
# After
if brief.style_exclusions:
    negative = (
        f"{negative}, {brief.style_exclusions}" if negative else brief.style_exclusions
    )
```

- [ ] **Step 5: Update DRESS stage's ArtDirection→ImageBrief construction**

In `src/questfoundry/pipeline/stages/dress.py:2142`:

```python
# Before
negative_defaults=art_dir.get("negative_defaults") or None,
# After
style_exclusions=art_dir.get("style_exclusions") or None,
```

- [ ] **Step 6: Update `_REQUIRED_ART_DIRECTION_FIELDS` tuple**

In `src/questfoundry/export/context.py:44`:

```python
# Before
"negative_defaults",
# After
"style_exclusions",
```

- [ ] **Step 7: Update DRESS prompt templates (schema + GOOD/BAD)**

In `prompts/templates/dress_serialize.yaml:21`, rename the field in the schema description and prepend the GOOD/BAD instruction:

```yaml
  - style_exclusions: Visual styles to exclude across all story images — story-tone
    prohibitions only.
    GOOD: "no photorealism, no anachronistic technology" (story-tone)
    BAD:  "blurry, watermark, deformed hands" (renderer-quality fillers — auto-injected at render time, never put them here)
```

In `prompts/templates/dress_brief.yaml:85` and `prompts/templates/dress_brief_batch.yaml:74`, rename `negative_defaults` → `style_exclusions` in any reference.

- [ ] **Step 8: Update tests + fixtures (sweep)**

```bash
rg -l "negative_defaults" tests/
```

For every file listed: replace `negative_defaults` with `style_exclusions` in fixture dicts, kwargs, and assertions. Common patterns:
- `ArtDirection(... negative_defaults=...)` → `style_exclusions=`
- `ImageBrief(... negative_defaults=...)` → `style_exclusions=`
- Fixture YAML/dict literals: same rename
- Assertions reading the field: same rename

- [ ] **Step 9: Verify no survivors in src/, tests/, prompts/**

```bash
rg "negative_defaults" src/ tests/ prompts/ docs/
```
Expected: zero hits. If any surface, treat as missed sites and update.

- [ ] **Step 10: Run targeted test suite**

```bash
uv run --frozen pytest tests/unit/test_dress_stage.py tests/unit/test_image_brief.py tests/unit/test_export_context.py tests/unit/test_dress_models.py tests/unit/test_a1111_provider.py -x -q
```
Adjust file list to match actual filenames in `tests/unit/`. Expected: all green.

- [ ] **Step 11: Run mypy + ruff on changed source**

```bash
uv run --frozen mypy src/questfoundry/models/dress.py src/questfoundry/providers/image_brief.py src/questfoundry/providers/image_a1111.py src/questfoundry/providers/image_openai.py src/questfoundry/pipeline/stages/dress.py src/questfoundry/export/context.py
uv run --frozen ruff check src/ tests/
```
Expected: clean.

- [ ] **Step 12: Commit**

```bash
git add -u src/ tests/ prompts/
git commit -m "refactor(dress): rename ArtDirection.negative_defaults → style_exclusions (#1557)"
```

---

## Task 5: DRESS Phase 0 hint — failing context-builder test

The renderer hint section is built by a new helper in `pipeline/stages/dress.py`. Test it in isolation first.

**Files:**
- Test: `tests/unit/test_dress_stage.py` (add a new test class)

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/test_dress_stage.py` (or create a new test file `tests/unit/test_dress_image_renderer_section.py` if the existing dress test file is large):

```python
class TestImageRendererSection:
    """`_build_image_renderer_section()` produces the hint block injected
    into dress_discuss when an image provider is selected at orchestrator
    init (#1557)."""

    def test_no_provider_returns_empty_string(self) -> None:
        from questfoundry.pipeline.stages.dress import _build_image_renderer_section

        assert _build_image_renderer_section(None) == ""

    def test_provider_with_checkpoint_includes_label(self) -> None:
        from questfoundry.pipeline.stages.dress import _build_image_renderer_section

        section = _build_image_renderer_section("a1111/juggernautXL_ragnarokBy")
        assert "Image Renderer Constraint" in section
        assert "Juggernaut" in section
        assert "photorealistic" in section.lower()
        # Must include both positive and negative guidance
        assert "best with" in section.lower() or "excels" in section.lower() or "works best" in section.lower()
        assert "cannot" in section.lower() or "incompatible" in section.lower() or "struggles" in section.lower()

    def test_provider_without_checkpoint_uses_default(self) -> None:
        from questfoundry.pipeline.stages.dress import _build_image_renderer_section

        section = _build_image_renderer_section("a1111")
        assert "Image Renderer Constraint" in section
        # Default fallback label
        assert "general-purpose" in section.lower() or "unknown" in section.lower()

    def test_anime_checkpoint_warns_against_photorealism(self) -> None:
        from questfoundry.pipeline.stages.dress import _build_image_renderer_section

        section = _build_image_renderer_section("a1111/animagine-xl")
        assert "anime" in section.lower()
        # Should signal incompatible vocabulary somewhere
        assert "photoreal" in section.lower() or "photo" in section.lower()

    def test_section_contains_tiebreaker_for_collisions(self) -> None:
        from questfoundry.pipeline.stages.dress import _build_image_renderer_section

        section = _build_image_renderer_section("a1111/juggernaut")
        # The hint must instruct conflict resolution into composition_notes
        assert "composition_notes" in section
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run --frozen pytest tests/unit/test_dress_stage.py::TestImageRendererSection -v
```
Expected: `ImportError` on `_build_image_renderer_section` (helper doesn't exist yet).

---

## Task 6: DRESS Phase 0 hint — implement helper + thread into render call

**Files:**
- Modify: `src/questfoundry/pipeline/stages/dress.py`

- [ ] **Step 1: Add the helper near the top of the module (above the stage class)**

Insert after the imports block in `src/questfoundry/pipeline/stages/dress.py`:

```python
def _build_image_renderer_section(provider_spec: str | None) -> str:
    """Build the renderer-aware hint section for dress_discuss (#1557).

    When `provider_spec` is None (no `--image-provider` set), returns "" so
    the `{image_renderer_section}` placeholder collapses cleanly. Otherwise
    looks up checkpoint style metadata via `resolve_checkpoint_style()` and
    formats a hint block that biases DRESS Phase 0 toward
    renderer-compatible art direction.

    Args:
        provider_spec: e.g. ``"a1111/juggernautXL_ragnarokBy"`` or ``"a1111"``
            or ``None`` when no image provider is selected at DRESS time.
    """
    if not provider_spec:
        return ""

    from questfoundry.providers.checkpoint_styles import resolve_checkpoint_style

    _, _, checkpoint = provider_spec.partition("/")
    style_info = resolve_checkpoint_style(checkpoint or provider_spec)

    return (
        "## Image Renderer Constraint (CRITICAL)\n"
        f"This story's images will be rendered by: {style_info['label']}\n\n"
        f"This renderer works best with these visual styles: "
        f"{style_info['style_hints']}\n"
        f"It CANNOT faithfully produce: {style_info['incompatible_styles']}\n\n"
        "GOOD art direction given this renderer: "
        'style="gritty photorealistic urban", medium="digital photo"\n'
        "BAD art direction given this renderer: "
        'style="watercolor wash", medium="traditional ink" '
        "(this renderer is tuned for photorealism; stylised media will "
        "fight the checkpoint and degrade image quality)\n\n"
        "Your art direction MUST be compatible with the renderer. If the "
        "story's tone strongly suggests a medium the renderer cannot "
        "produce well, choose the closest compatible style and note the "
        "compromise in `composition_notes`.\n"
    )
```

- [ ] **Step 2: Thread it into the `_phase_0_art_direction` render call**

In `src/questfoundry/pipeline/stages/dress.py`, find the `system_prompt = discuss_template.system.format(...)` call (around line 509). Add the new kwarg:

```python
system_prompt = discuss_template.system.format(
    vision_context=vision_context,
    entity_list=entity_list,
    research_tools_section=research_section,
    sandbox_section=load_sandbox_section(),
    mode_section=mode_section,
    image_renderer_section=_build_image_renderer_section(self._image_provider_spec),
)
```

`self._image_provider_spec` already exists on the DressStage (set at construction time, see line 199).

- [ ] **Step 3: Run the helper tests**

```bash
uv run --frozen pytest tests/unit/test_dress_stage.py::TestImageRendererSection -v
```
Expected: 5 passed.

- [ ] **Step 4: Run mypy on the modified file**

```bash
uv run --frozen mypy src/questfoundry/pipeline/stages/dress.py
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/pipeline/stages/dress.py tests/unit/test_dress_stage.py
git commit -m "feat(dress): _build_image_renderer_section helper for renderer-aware Phase 0 (#1557)"
```

---

## Task 7: Add `{image_renderer_section}` placeholder to dress_discuss.yaml

**Files:**
- Modify: `prompts/templates/dress_discuss.yaml`

- [ ] **Step 1: Open `prompts/templates/dress_discuss.yaml` and locate the `## Guidelines` section**

Currently the system prompt has section blocks like `{vision_context}`, `{entity_list}`, `{research_tools_section}`, `{sandbox_section}`, `{mode_section}`. The `## Guidelines` heading appears later in the file.

- [ ] **Step 2: Insert the placeholder immediately before `## Guidelines`**

Add a blank line followed by `{image_renderer_section}` followed by another blank line. The placeholder is at top-level (no indentation other than the YAML system block's existing indent). Example:

```yaml
  ... (existing sections) ...

  {image_renderer_section}

  ## Guidelines
  ... (existing guidelines) ...
```

The two blank lines around the placeholder ensure that when the placeholder collapses to empty string, no orphan whitespace remains visually awkward; when it's populated, the section is offset cleanly from neighbours.

- [ ] **Step 3: Run a smoke test that the template still loads + renders**

```bash
uv run --frozen pytest tests/unit/test_dress_stage.py -k "phase_0 or art_direction or discuss" -x -q
```
Expected: pass. If any test was checking exact prompt text, update its fixture to include the new placeholder OR include `image_renderer_section=""` in the test's render call.

- [ ] **Step 4: Smoke-test end-to-end with a no-provider scenario (the placeholder MUST resolve to empty string)**

```bash
uv run --frozen python -c "
from questfoundry.pipeline.stages.dress import _build_image_renderer_section
from questfoundry.prompts.loader import PromptLoader
from pathlib import Path
loader = PromptLoader(Path('prompts'))
template = loader.load('dress_discuss')
out = template.system.format(
    vision_context='V',
    entity_list='E',
    research_tools_section='',
    sandbox_section='',
    mode_section='',
    image_renderer_section=_build_image_renderer_section(None),
)
assert '{image_renderer_section}' not in out, 'placeholder did not resolve'
print('OK: placeholder resolves cleanly when provider is None')
"
```
Expected: `OK: placeholder resolves cleanly when provider is None`.

- [ ] **Step 5: Commit**

```bash
git add prompts/templates/dress_discuss.yaml
git commit -m "feat(prompts): add {image_renderer_section} placeholder to dress_discuss.yaml (#1557)"
```

---

## Task 8: Distiller hint replacement — failing test

The A1111 distiller currently builds `checkpoint_hint` from `self._model` plus a hand-written generic example. Replace its body with `resolve_checkpoint_style()` data. The behaviour change is observable in the system prompt the distiller builds.

**Files:**
- Test: `tests/unit/test_a1111_provider.py` (or whatever the existing A1111 test file is; create one if absent)

- [ ] **Step 1: Find the existing A1111 distiller tests**

```bash
ls tests/unit/ | grep -i a1111
```
Use the existing test file. If none exists, create `tests/unit/test_image_a1111.py`.

- [ ] **Step 2: Add the failing test**

```python
class TestDistillerCheckpointHint:
    """The distiller's `checkpoint_hint` must come from `resolve_checkpoint_style()`
    (#1557), not from a generic LLM-self-classifying prompt."""

    def _make_provider(self, model: str | None) -> "A1111ImageProvider":
        from questfoundry.providers.image_a1111 import A1111ImageProvider
        # Use a minimal mock for the LLM — distiller test doesn't actually call it.
        from unittest.mock import MagicMock
        llm = MagicMock()
        return A1111ImageProvider(host="http://x", model=model, llm=llm)

    def test_no_model_omits_checkpoint_hint(self) -> None:
        # Internal helper / system_msg should not include "TARGET CHECKPOINT" when no model
        # NOTE: build the same brief the production code uses; access the system prompt
        # construction by the same path. If the existing test suite already has an
        # introspection helper, reuse it. Otherwise, mirror the assembly inline here.
        provider = self._make_provider(None)
        # The current implementation gates on `if self._model:`. Verify that the
        # hint block is empty when self._model is None.
        # (This test relies on the implementation exposing checkpoint_hint or building it.
        #  If implementation hides it, switch to asserting absence of "TARGET CHECKPOINT"
        #  in the assembled system_msg via a public seam.)
        assert provider._model is None  # sanity

    def test_juggernaut_hint_uses_map_label(self) -> None:
        provider = self._make_provider("juggernautXL_ragnarokBy.safetensors")
        from questfoundry.providers.checkpoint_styles import resolve_checkpoint_style
        info = resolve_checkpoint_style(provider._model or "")
        # The distiller's hint MUST surface the structured label and style fields,
        # not the bare model name.
        assert "Juggernaut" in info["label"]
        assert "photorealistic" in info["style_hints"].lower()

    def test_distiller_system_prompt_includes_label_for_known_checkpoint(self) -> None:
        # Integration: distiller's assembled system message contains the resolved label.
        # Implementation hint: factor out a small helper or test against a substring of
        # the assembled message. If the existing distiller code is monolithic, add a
        # tiny `_format_checkpoint_hint(self) -> str` method during this task and test it.
        provider = self._make_provider("juggernautXL_ragnarokBy.safetensors")
        # Once the helper exists:
        from questfoundry.providers.image_a1111 import A1111ImageProvider
        if hasattr(provider, "_format_checkpoint_hint"):
            hint = provider._format_checkpoint_hint()
            assert "Juggernaut" in hint
            assert "TARGET CHECKPOINT" in hint
            assert "photorealistic" in hint.lower()
        else:
            # Fallback: read source for the substring (less ideal; remove once helper lands)
            import inspect
            src = inspect.getsource(A1111ImageProvider._distill_with_llm)
            assert "resolve_checkpoint_style" in src
```

- [ ] **Step 3: Run the failing test**

```bash
uv run --frozen pytest tests/unit/test_image_a1111.py::TestDistillerCheckpointHint -v
```
Expected: `test_distiller_system_prompt_includes_label_for_known_checkpoint` fails (no `_format_checkpoint_hint` method, no `resolve_checkpoint_style` import in `_distill_with_llm`).

---

## Task 9: Distiller hint replacement — implementation

**Files:**
- Modify: `src/questfoundry/providers/image_a1111.py`

- [ ] **Step 1: Extract a `_format_checkpoint_hint()` method on the provider**

In `src/questfoundry/providers/image_a1111.py`, find the existing `checkpoint_hint = ""` block inside `_distill_with_llm` (around line 392). Replace it by calling a new method:

```python
        checkpoint_hint = self._format_checkpoint_hint()
```

Then add the method to the class (above `_distill_with_llm`):

```python
    def _format_checkpoint_hint(self) -> str:
        """Build the TARGET CHECKPOINT hint block injected into the
        distiller's system prompt. Sources data from
        `resolve_checkpoint_style()` (single map, shared with DRESS
        Phase 0) so the distiller's guidance and the upstream Phase 0
        guidance cannot drift apart (#1557)."""
        if not self._model:
            return ""

        from questfoundry.providers.checkpoint_styles import resolve_checkpoint_style

        info = resolve_checkpoint_style(self._model)
        return (
            f"\nTARGET CHECKPOINT: {info['label']}\n"
            f"This checkpoint excels at: {info['style_hints']}\n"
            f"This checkpoint struggles with: {info['incompatible_styles']}\n"
            "Adapt your CLIP tags accordingly. If the brief asks for a "
            "style this checkpoint can't render well, translate to the "
            "closest compatible vocabulary (e.g. brief says 'watercolor "
            "wash' on a photoreal checkpoint → render as 'soft natural "
            "light, painterly atmosphere').\n"
        )
```

The old inline checkpoint_hint construction (the `f"\nTARGET CHECKPOINT: {self._model}\n"...` block plus the generic "anime/illustration models...photorealistic models prefer..." prose) is removed.

- [ ] **Step 2: Run the distiller tests**

```bash
uv run --frozen pytest tests/unit/test_image_a1111.py::TestDistillerCheckpointHint -v
```
Expected: 3 passed.

- [ ] **Step 3: Run any wider distiller tests to confirm no regression**

```bash
uv run --frozen pytest tests/unit/test_image_a1111.py -x -q
```
Expected: all green. If a test was asserting on the old generic "anime/illustration models" wording, update its assertion to use the new structured form.

- [ ] **Step 4: Run mypy + ruff**

```bash
uv run --frozen mypy src/questfoundry/providers/image_a1111.py
uv run --frozen ruff check src/questfoundry/providers/image_a1111.py
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/providers/image_a1111.py tests/unit/test_image_a1111.py
git commit -m "feat(providers): A1111 distiller uses _CHECKPOINT_STYLE_MAP for checkpoint hint (#1557)"
```

---

## Task 10: Final verification + draft PR

**Files:**
- N/A (verification + git only)

- [ ] **Step 1: Verify zero `negative_defaults` survivors anywhere**

```bash
rg "negative_defaults" src/ tests/ docs/ prompts/
```
Expected: zero hits.

- [ ] **Step 2: Run unit test suite**

```bash
uv run --frozen pytest tests/unit/ -x -q
```
Expected: all green. If anything fails, treat as a missed rename site or a stale test fixture.

- [ ] **Step 3: Run mypy on changed source files**

```bash
uv run --frozen mypy src/questfoundry/
```
Expected: clean.

- [ ] **Step 4: Run ruff**

```bash
uv run --frozen ruff check src/ tests/
uv run --frozen ruff format --check src/ tests/
```
Expected: clean. If format issues, run without `--check` to fix and commit.

- [ ] **Step 5: Push branch + open draft PR**

```bash
git push -u origin feat/1554-followup-image-renderer-hint
gh pr create --draft --title "feat(dress): renderer-aware art direction (checkpoint hint + style_exclusions rename)" --body "$(cat <<'EOF'
## Summary

Two-layer renderer-aware art direction backed by a single static checkpoint style map:

- **Prevention (DRESS Phase 0)**: when \`--image-provider\` is set at orchestrator init, \`dress_discuss.yaml\` injects an Image-Renderer-Constraint section so DRESS picks compatible art direction. Empty when provider is deferred.
- **Recovery (A1111 distiller)**: existing crude \`checkpoint_hint\` replaced with structured map lookup; the two layers cannot drift.

Bundled rename: \`ArtDirection.negative_defaults\` → \`style_exclusions\`. The original name activated LLM training associations with SD-style negative-prompt fillers; small models (qwen3:4b) read field names stronger than descriptions. \`style_exclusions\` gives the field one unambiguous semantic (story-tone visual prohibitions); renderer-quality fillers move to per-provider code-side constants.

Designed via two rounds of \`@prompt-engineer\` advisory.

## Spec

\`docs/superpowers/specs/2026-04-28-dress-image-renderer-hint-design.md\`

## Cascade (13 sites)

Specs first per CLAUDE.md "Spec-first fix order": dress.md R-1.2 + ship.md R-3.9 → checkpoint_styles.py module → atomic rename across 11 sites → DRESS context-builder + dress_discuss.yaml → distiller hint replacement.

Closes #1557.

## Test plan

- [x] \`uv run pytest tests/unit/test_checkpoint_styles.py\` — 14 passed
- [x] \`uv run pytest tests/unit/test_dress_stage.py::TestImageRendererSection\` — 5 passed
- [x] \`uv run pytest tests/unit/test_image_a1111.py::TestDistillerCheckpointHint\` — 3 passed
- [x] \`uv run pytest tests/unit/\` — full suite green
- [x] \`uv run mypy src/questfoundry/\` — clean
- [x] \`uv run ruff check src/ tests/\` — clean
- [x] \`rg "negative_defaults" src/ tests/ docs/ prompts/\` — zero hits

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Wait for bot review loop**

Per CLAUDE.md, address findings in-PR. Gemini is at daily quota; claude-review approval is sufficient. Flip ready when claude-review LGTM + all CI green + bot's review body explicitly says "Ready to merge" (read the body, not just the check status).

---

## Self-Review Notes

Ran the spec-coverage check:

- ✅ Spec §1 (DRESS Phase 0 prevention) → Tasks 5-7
- ✅ Spec §2 (Distiller recovery) → Tasks 8-9
- ✅ Spec §`_CHECKPOINT_STYLE_MAP` → Tasks 2-3 (full table inlined in code)
- ✅ Spec §`negative_defaults` rename → Tasks 1 (specs) + 4 (atomic code rename)
- ✅ Spec §file cascade (13 sites) → Task 4 (rename) + Task 1 (specs) + Task 6-7 (DRESS) + Task 9 (distiller); 12 of 13 mapped — `prompts/templates/dress_serialize.yaml` GOOD/BAD instruction lives in Task 4 step 7
- ✅ Spec §verification → Task 10

No placeholders in the plan body. No "implement later" / "similar to Task N" / TBD. Type consistency: `_build_image_renderer_section`, `_format_checkpoint_hint`, `resolve_checkpoint_style` referenced consistently across tasks.
