# DRESS image-renderer hint + `style_exclusions` rename

**Status:** Approved 2026-04-28
**Owner:** @pvliesdonk
**Designed with:** @prompt-engineer (advisory, two rounds)

## Problem

`ArtDirection` (created in DRESS Phase 0) is currently produced blind to which image generator will render it. When an author runs the pipeline with `--image-provider a1111/juggernautXL_ragnarok` (a photorealistic SDXL checkpoint), DRESS may still pick `style="watercolor"` and the renderer fights the checkpoint at generation time. The author wants to prevent that mismatch when the renderer is known up-front.

A second, related issue surfaces: the `negative_defaults` field on `ArtDirection` is named in a way that activates the LLM's training association with SD-style negative-prompt fillers (`blurry, watermark, mutated hands, ...`) — quality concerns that should be code-injected per-provider, not author-curated. Small models (`qwen3:4b`) read field names as a stronger signal than descriptions, and the current name pulls them toward the wrong category.

## Non-goals

- Restructuring DRESS's three-pass discuss/summarize/serialize architecture.
- Adding new image providers.
- Auto-detecting the renderer post-DRESS (the deferred-image-gen workflow stays valid; this design respects it).
- Replacing the existing free-text → CLIP-tag distillation step.

## Design

### Two layers, one source of truth

Renderer-aware adaptation happens at **two layers**, sharing a single static map of checkpoint style metadata:

1. **Prevention (DRESS Phase 0)** — when `--image-provider` is set at orchestrator init, inject a renderer hint into the `dress_discuss.yaml` system prompt so DRESS picks compatible art direction in the first place. When the provider is unset (deferred image-gen), the hint section is omitted entirely (empty-string substitution); DRESS proceeds renderer-agnostic as today.
2. **Recovery (distiller)** — the existing `checkpoint_hint` block in `image_a1111.py` (renderer-time, free-text → CLIP-tag conversion) already has crude awareness via the bare model name and a generic anime-vs-photoreal example. Replace its body with structured data from the same static map. This is the safety net when DRESS was renderer-blind, and a stronger steer when DRESS already had the hint.

Both layers consume the **same** `_CHECKPOINT_STYLE_MAP`, lookup function, and label/style/incompatible triplet. Single source of truth.

### `_CHECKPOINT_STYLE_MAP` — checkpoint style metadata

A new module `src/questfoundry/providers/checkpoint_styles.py` exposes:

**Pattern ordering rule:** patterns are matched against the lowercased checkpoint filename, first-match-wins. Specific patterns must precede generic ones. The dreamshaper family demonstrates this: `dreamshaperxl.*lightning|dreamshaperxl.*alpha` precedes `dreamshaperxl|dreamshaper.*xl`, which in turn precedes plain `dreamshaper` (SD1.5). The default-fallback (empty pattern) is always last.

Full map content (one entry per loaded checkpoint at A1111 + generic family fallbacks + default):

| Pattern | Label | Style hints (positive) | Incompatible styles (negative) |
|---|---|---|---|
| `flux` | Flux (photorealistic / highly-detailed) | photorealistic imagery, extreme fine detail, architectural photography, natural lighting, product shots, documentary portraiture, coherent text in scene | anime, manga, cel-shading, watercolor washes, heavy painterly texture, low-detail illustration styles; **also flagged**: negative-prompt weighting is weak on Flux |
| `coloring.?book` | Coloring Book (line-art SD1.5) | clean outlines on white background, no fill colors, strong linework, simple shapes, children's-book-friendly compositions, decorative borders | photorealism, color renders, painterly textures, complex shading, dark backgrounds, photographic lighting |
| `juggernaut` | Juggernaut XL (photorealistic SDXL) | photorealistic portraits, cinematic lighting, sharp textural detail, skin pores, fabric weave, dramatic rim lighting, environmental storytelling | anime, cartoon, flat illustration, watercolor, comic-book ink outlines, chibi |
| `animagine` | Animagine XL (anime SDXL) | anime illustration, Danbooru-style tag vocabulary, clean cell shading, expressive character art, vivid saturated palette, manga panel compositions | photorealism, photography-style lighting, gritty texture, oil painting, detailed backgrounds without anime stylisation |
| `dreamshaperxl.*lightning\|dreamshaperxl.*alpha` | DreamShaperXL Lightning / Alpha (fast fantasy SDXL) | fantasy concept art, painterly illustration, vibrant color, dramatic character portraits, acceptable in 4-8 steps | photorealism (style is stylised by design), highly detailed textures at very low step counts, strict architectural accuracy |
| `dreamshaperxl\|dreamshaper.*xl` | DreamShaperXL (versatile fantasy SDXL) | fantasy illustration, painterly portraits, concept-art style, stylised environments, strong use of negative space | strict photorealism, clinical document photography, flat-color infographic styles |
| `dreamshaper` | DreamShaper (versatile SD1.5) | general-purpose stylised illustration, fantasy character art, soft painterly lighting, portrait and environmental compositions; **notably versatile — adapt style tags rather than leaning on a single category** | extreme photorealism (slightly stylised by design), Danbooru/anime tag grammar (use natural descriptors instead) |
| `sd_xl_base\|sdxl_base\|sdxl-base` | SDXL Base (general-purpose SDXL) | broad style range, photography, illustration, concept art; best results with refiner pass or ControlNet; responds well to explicit style tokens | anime-specific Danbooru vocabulary without style priming, very low step counts (needs ≥30 steps for coherence) |
| `v1[-_]5\|sd[-_]?1[-._]?5` | SD 1.5 (general-purpose base) | broad style range at 512-768px, watercolor, ink illustration, painterly portraiture; well-supported by community LoRAs | photorealistic skin detail at high resolution (768px ceiling limits fine detail), SDXL-native aspect ratios |
| `""` *(default)* | Unknown checkpoint (SD general-purpose defaults) | broad range: illustration, painterly, concept art; Stable Diffusion generally excels at stylised imagery, fantasy environments, and character portraiture; use explicit style tokens (e.g. 'watercolor painting', 'cinematic photograph') for best results | coherent embedded text, photographic product catalogs without specialised fine-tuning |

```python
def resolve_checkpoint_style(model: str) -> dict[str, str]:
    """Look up style metadata for a checkpoint name. Always returns a
    populated dict (default fallback if no specific pattern matches).
    """
    lowered = model.lower()
    for pattern, info in _CHECKPOINT_STYLE_MAP:
        if pattern.search(lowered):
            return info
    raise AssertionError("default fallback should always match")
```

### DRESS Phase 0 hint section

`dress_discuss.yaml` adds a placeholder immediately before the `## Guidelines` block:

```yaml
{image_renderer_section}
```

The context builder in `pipeline/stages/dress.py` populates it via:

```python
def _build_image_renderer_section(provider_spec: str | None) -> str:
    """Build the renderer-aware hint section for dress_discuss.

    Returns "" when no provider is selected (deferred image-gen).
    Returns a fully-populated section when provider/checkpoint is known.
    """
    if not provider_spec:
        return ""

    provider_name, _, checkpoint = provider_spec.partition("/")
    style_info = resolve_checkpoint_style(checkpoint or provider_name)

    return (
        "## Image Renderer Constraint (CRITICAL)\n"
        f"This story's images will be rendered by: {style_info['label']}\n\n"
        f"This renderer works best with these visual styles: "
        f"{style_info['style_hints']}\n"
        f"It CANNOT faithfully produce: {style_info['incompatible_styles']}\n\n"
        "GOOD art direction given this renderer: style=\"gritty "
        "photorealistic urban\", medium=\"digital photo\"\n"
        "BAD art direction given this renderer: style=\"watercolor wash\", "
        "medium=\"traditional ink\" (this renderer is tuned for "
        "photorealism; stylised media will fight the checkpoint and "
        "degrade image quality)\n\n"
        "Your art direction MUST be compatible with the renderer. If the "
        "story's tone strongly suggests a medium the renderer cannot "
        "produce well, choose the closest compatible style and note the "
        "compromise in `composition_notes`.\n"
    )
```

The `GOOD`/`BAD` examples in the section above are template-level and stay verbatim; the renderer-specific details come from the resolved `style_hints` / `incompatible_styles`. (A future enhancement could template the GOOD/BAD bodies too, but the current shape gives small models the structural anchor without per-checkpoint divergence.)

The section is injected only into `dress_discuss.yaml` (creative phase), **not** into `dress_summarize.yaml` (R-1.2: summarize must not add new ideas) or `dress_serialize.yaml` (pure format conversion). If the hint isn't in the discussion, the schema-driven serialize phase cannot recover it.

### Distiller hint (`image_a1111.py`)

The current `checkpoint_hint` block (lines 392-400) is replaced with a map lookup:

```python
checkpoint_hint = ""
if self._model:
    style_info = resolve_checkpoint_style(self._model)
    checkpoint_hint = (
        f"\nTARGET CHECKPOINT: {style_info['label']}\n"
        f"This checkpoint excels at: {style_info['style_hints']}\n"
        f"This checkpoint struggles with: {style_info['incompatible_styles']}\n"
        "Adapt your CLIP tags accordingly. If the brief asks for a style "
        "this checkpoint can't render well, translate to the closest "
        "compatible vocabulary (e.g. brief says 'watercolor wash' on a "
        "photoreal checkpoint → render as 'soft natural light, painterly "
        "atmosphere').\n"
    )
```

This is a strict improvement over the current generic block: structured guidance, deterministic, and shares its data with DRESS so the two layers cannot drift.

`image_openai.py` keeps its current shape — no checkpoint indirection there, since the OpenAI image model is itself the renderer (`gpt-image-1`, `dall-e-3`). Its concatenation site only needs the field rename (see below).

### `negative_defaults` → `style_exclusions` rename

Renaming the field is the load-bearing change for output quality. The word "negative" in `negative_defaults` activates the LLM's training association with SD-style negative-prompt fillers; even with an improved description, small models will dump quality terms there. After the rename, the field's single, unambiguous meaning becomes:

> **Visual styles to exclude across all story images** — story-tone prohibitions only (e.g. *no photorealism, no modern clothing for a Victorian setting*). Renderer-quality fillers (`blurry, watermark, mutated hands, jpeg artifacts`) are auto-injected by the render pipeline at distill time and must not appear here.

The renderer-quality fillers become a **per-provider code-side constant**, appended at distill time. Authors and the LLM never see or write them.

### File cascade (13 sites, ordered)

| # | File | Change |
|---|------|--------|
| 1 | `docs/design/procedures/dress.md` | R-1.2 enumerated field list: `negative_defaults` → `style_exclusions`. Worked-example YAML block field rename + clarifying description. |
| 2 | `docs/design/procedures/ship.md` | R-3.9 partial-DRESS warning enumeration: rename. |
| 3 | `src/questfoundry/providers/checkpoint_styles.py` *(new)* | `_CHECKPOINT_STYLE_MAP` + `resolve_checkpoint_style()`. |
| 4 | `src/questfoundry/providers/image_a1111.py` | Replace `checkpoint_hint` body with map lookup; rename concatenation site (`brief.negative_defaults` → `brief.style_exclusions`). |
| 5 | `src/questfoundry/pipeline/stages/dress.py` | New `_build_image_renderer_section()` helper; thread `image_renderer_section` into the discuss prompt render call. Rename concatenation site for ArtDirection→ImageBrief construction. |
| 6 | `prompts/templates/dress_discuss.yaml` | Insert `{image_renderer_section}` placeholder immediately before `## Guidelines`. |
| 7 | `src/questfoundry/models/dress.py` | `ArtDirection.negative_defaults` → `style_exclusions` + new description. |
| 8 | `src/questfoundry/providers/image_brief.py` | `ImageBrief.negative_defaults` → `style_exclusions`; flattener field. |
| 9 | `src/questfoundry/providers/image_openai.py` | Rename concatenation site (no checkpoint map needed — model is the renderer). |
| 10 | `prompts/templates/dress_serialize.yaml` | Schema block field rename + GOOD/BAD instruction (story-tone vs renderer fillers). |
| 11 | `prompts/templates/dress_brief.yaml` + `dress_brief_batch.yaml` | Field-name references. |
| 12 | `src/questfoundry/export/context.py` | `_REQUIRED_ART_DIRECTION_FIELDS` tuple at line 39: rename. |
| 13 | Tests / fixtures | Replace directly per CLAUDE.md (no compat shim). Affected: `tests/unit/test_dress*.py`, `tests/unit/test_a1111*.py`, `tests/unit/test_image_brief*.py`, `tests/unit/test_export_context.py`, any DRESS YAML fixtures. |

Specs precede code per `CLAUDE.md` "Spec-first fix order."

### Verification

`rg "negative_defaults" src/ tests/ docs/ prompts/` returns 0 after the cascade lands. `rg "_CHECKPOINT_STYLE_MAP" src/` returns one definition + two consumers (DRESS context-builder, distiller).

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| DRESS over-anchors on renderer, ignores story tone. | Explicit tiebreaker in the hint: "choose closest compatible style, note compromise in `composition_notes`." Style collisions surface in Phase 0 human gate (R-1.4). |
| LLM dumps quality fillers into `style_exclusions` after rename. | GOOD/BAD examples in `dress_serialize.yaml` instruction explicitly warn: `BAD: blurry, watermark, deformed hands` with rationale "those are auto-injected by the render pipeline." |
| Pattern shadowing in `_CHECKPOINT_STYLE_MAP`. | Module-level docstring documents first-match-wins; specific patterns must precede generic. Test asserts ordering for the dreamshaper/dreamshaperxl/dreamshaperxl-lightning chain. |
| Provider known but checkpoint absent (e.g. `--image-provider a1111`). | `resolve_checkpoint_style()` falls through to the default entry, returning the generic SD-defaults label. The hint still injects, just with broader guidance. |
| Style collision irreconcilable (story = dreamlike surreal, checkpoint = photoreal). | Phase 0 human gate is the resolution point. The hint should not adjudicate; it declares the constraint and lets the human override. |
| Field-rename ripples through tests. | One PR; replace directly. No backward-compatibility shim per CLAUDE.md "Refactoring & Removal Discipline." |

## PR shape

**Single PR.** The static map is the load-bearing asset and both layers (DRESS context-builder + distiller) ride on it; splitting forces either a temporary import path or a duplicate map. The 13 sites are cohesive — spec → model → providers → templates → tests — and a reviewer can validate the entire feature in one diff.

## Out of scope (explicitly deferred)

- **OpenAI / Gemini / placeholder providers** getting a checkpoint-style entry. They map to specific image-generation models with much narrower style ranges; if author benefit emerges, a follow-up issue can extend the map.
- **Distiller-time renderer-quality filler constant.** The design says these become auto-injected at distill time, but the implementation can keep the existing inline filler in the EXAMPLE block until a clean factoring opportunity arises. Filing as a follow-up tracker keeps this PR focused.
- **Phase 0 sample-image review (R-5.3)** integration. The renderer hint should improve sample quality, but no measurement / feedback loop is added in this PR.

## References

- Round-1 advisory from `@prompt-engineer` (DRESS Phase 0 hint placement, GOOD/BAD pattern, conditional rendering, negative-space risks).
- Round-2 advisory from `@prompt-engineer` (concrete checkpoint map for the user's loaded models, `style_exclusions` rename rationale and cascade).
- A1111 instance: `http://athena.int.liesdonk.nl:7860`, 10 checkpoints loaded as of 2026-04-28.
- Authoritative spec docs: `docs/design/procedures/dress.md`, `docs/design/procedures/ship.md`.
