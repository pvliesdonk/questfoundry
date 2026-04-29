# Flux natural-language prompt distillation on Forge Neo

**Status:** Approved 2026-04-28
**Owner:** @pvliesdonk
**Tracking:** #1559
**Builds on:** #1558 (merged) — checkpoint-style map

## Problem

PR #1558 added a static `_CHECKPOINT_STYLE_MAP` and an A1111 distiller `_format_checkpoint_hint` that adapt CLIP-tag-format prompts to the active checkpoint. The distiller's output format is hardcoded: comma-separated CLIP tags with a `tag_limit = 75 if is_xl else 40` budget and a `BREAK`-separated scene/style chunking pattern.

The user has replaced their A1111 instance with **Forge Neo**, which is API-compatible with A1111 (same HTTP endpoints, same WebUI) but hosts **Flux** alongside standard SD checkpoints. Flux uses a **T5 text encoder** with a ~512-token window, trained on natural-language prose — the opposite end of the spectrum from CLIP's ~77-token tag-soup. Distilling to CLIP tags on Flux:

- discards narrative cues T5 was trained on;
- produces the wrong prompt shape (comma-tag form is not how T5 expects input);
- defeats the point of the larger token window.

The user's Forge Neo instance hosts both Flux *and* standard SDXL/SD1.5 checkpoints. The format toggle must therefore be **per-checkpoint**, not per-provider.

## Non-goals

- Replacing the A1111 provider class. Forge Neo is API-compatible; the HTTP layer in `image_a1111.py` works unchanged.
- LLM-driven natural-language distillation. This spec picks the simpler path: skip LLM distillation entirely for NL-mode checkpoints, flatten the brief's prose fields directly. If the flatten approach underperforms on Flux output quality, an LLM-distill-to-prose mode can be added later.
- Touching `image_openai.py`. The OpenAI image providers already produce NL prompts; no changes needed.
- Renaming the `a1111` provider identifier. Forge Neo presents the same API; users keep `--image-provider a1111/<checkpoint>`.

## Design

### `prompt_format` field on each map entry

`_CHECKPOINT_STYLE_MAP` entries gain a `prompt_format` key with values:

- `"clip_tags"` — current behavior. CLIP-encoder checkpoints (SDXL, SD1.5, Animagine, Juggernaut, DreamShaper variants, SDXL Base, Coloring Book). LLM-distilled to comma-tag format.
- `"natural_language"` — T5-encoder checkpoints (Flux). Direct flatten via `flatten_brief_to_prompt()`; no LLM call.

Existing entries: every entry except Flux gets `"clip_tags"`. Flux gets `"natural_language"`. Default-fallback entry stays on `"clip_tags"` (SD-family is the long tail; the safer assumption is CLIP).

### `prompt_format_for_checkpoint()` helper

A new sibling of `resolve_checkpoint_style()`:

```python
def prompt_format_for_checkpoint(model: str | None) -> Literal["clip_tags", "natural_language"]:
    """Return the prompt format the active checkpoint expects.

    Returns ``"clip_tags"`` when no model is set — preserves the LLM-distill
    default path through ``A1111ImageProvider.distill_prompt``.
    """
    if not model:
        return "clip_tags"
    return resolve_checkpoint_style(model)["prompt_format"]
```

The function is a thin wrapper around `resolve_checkpoint_style()` so the static map remains the single source of truth.

### `A1111ImageProvider.distill_prompt` branch

Currently:

```python
async def distill_prompt(self, brief: ImageBrief) -> tuple[str, str | None]:
    if self._llm is None:
        raise RuntimeError("A1111 prompt distillation requires an LLM. ...")
    return await self._distill_with_llm(brief)
```

After:

```python
async def distill_prompt(self, brief: ImageBrief) -> tuple[str, str | None]:
    if prompt_format_for_checkpoint(self._model) == "natural_language":
        # T5-encoder checkpoints (Flux): the structured brief already encodes
        # subject/composition/mood/entities/style/medium/palette as comma-joined
        # prose; T5's larger token window and natural-language training make
        # CLIP-tag distillation counterproductive (#1559).
        return flatten_brief_to_prompt(brief)

    if self._llm is None:
        raise RuntimeError("A1111 prompt distillation requires an LLM. ...")
    return await self._distill_with_llm(brief)
```

The LLM-required assertion narrows to the `clip_tags` branch. NL mode runs without an LLM, since `flatten_brief_to_prompt()` is pure.

### What flows through to T5

`flatten_brief_to_prompt()` (already in `image_brief.py`) produces a comma-joined positive prompt assembled from:

```
[entity_fragments] and [entity_fragments], [subject], [composition], [mood], [art_style], [art_medium] style, [palette] palette, [style_overrides]
```

This is human-readable, narrative-flavored prose with light comma structure — exactly the shape T5 expects. Negative prompt joins `negative` + `style_exclusions`.

The DRESS Phase 0 renderer-hint section (PR #1558) already biases the brief itself toward Flux-friendly art direction when the renderer is known at orchestrator init. The distiller's only remaining job in NL mode is to format the brief's existing prose; no further LLM-driven adaptation is needed.

## File cascade (4 sites)

| # | File | Change |
|---|---|---|
| 1 | `src/questfoundry/providers/checkpoint_styles.py` | Add `prompt_format` key to all 10 `_CHECKPOINT_STYLE_MAP` entries (Flux: `"natural_language"`; everything else: `"clip_tags"`). Add `prompt_format_for_checkpoint()` helper. Update module docstring. |
| 2 | `tests/unit/test_checkpoint_styles.py` | Failing test asserting Flux returns `"natural_language"` and 9 other entries return `"clip_tags"`; default fallback returns `"clip_tags"`; `None` model returns `"clip_tags"`. |
| 3 | `src/questfoundry/providers/image_a1111.py` | `distill_prompt` branches on `prompt_format_for_checkpoint(self._model)`. NL mode calls `flatten_brief_to_prompt(brief)`. LLM assertion narrows to clip_tags branch. Import `flatten_brief_to_prompt` and `prompt_format_for_checkpoint`. |
| 4 | `tests/unit/test_image_a1111.py` | Failing test: `distill_prompt` on a Flux model returns `flatten_brief_to_prompt(brief)` output verbatim, with the LLM mock receiving zero calls; clip_tags model still goes through the LLM path. |

## Verification

```sh
$ uv run pytest tests/unit/test_checkpoint_styles.py tests/unit/test_image_a1111.py -v
# green

$ rg "prompt_format" src/questfoundry/providers/
# 10+ map-entry hits + 1 helper definition + 1 distiller branch
```

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `flatten_brief_to_prompt()` output too sparse for Flux's 512-token window. | If output quality on Flux underperforms, add an LLM-distill-to-prose mode (option (a) from brainstorm) as a follow-up. The static `prompt_format` field can grow a third value `"natural_language_distilled"` later. |
| Comma-joined prose isn't what T5 wants — T5 may prefer flowing sentences. | The brief's structured layout (subject + composition + mood as separate fields) is already prose-friendly; commas separate semantic chunks. If observation in practice shows it underperforms, restructure `flatten_brief_to_prompt()` to produce flowing sentences for NL mode. Same surface, different formatter. |
| User sets `--image-provider a1111/flux1-dev-bnb-nf4` but the orchestrator path still requires `_llm` for some other reason. | The `_llm` parameter on `A1111ImageProvider.__init__` stays `Optional`; only `distill_prompt`'s `clip_tags` branch asserts it's not None. NL mode initializes cleanly without an LLM. |
| Adding `prompt_format` to existing entries is a contract change for any consumer that does `set(info.keys()) == {...}` exact-equality. | The existing test `test_returns_required_keys` uses `>=` (subset check), not equality. No regression. |

## Out of scope

- The OpenAI image provider (`image_openai.py`) and the placeholder provider (`image_placeholder.py`) — both already produce NL prompts via their own paths; no `prompt_format` consumption needed there.
- Spec-doc updates to `dress.md` / `ship.md` — `ArtDirection` schema is unchanged; renderer behavior change is internal to the providers.
- DRESS Phase 0 prompt updates — the renderer-hint section (PR #1558) already biases the brief toward checkpoint-compatible styles; no additional Phase 0 changes needed for this PR.

## References

- PR #1558 (merged) — checkpoint-style map and DRESS Phase 0 renderer hint.
- `_CHECKPOINT_STYLE_MAP` Flux entry: "negative-prompt weighting is weak on Flux — do not rely on negative prompts for strong style exclusion." (Already shipped in #1558; complementary to this PR's natural-language distillation.)
- `flatten_brief_to_prompt()` in `src/questfoundry/providers/image_brief.py:37` — the existing default flattener that NL mode reuses.
- Forge Neo: A1111-API-compatible Stable Diffusion WebUI fork that hosts Flux (T5) alongside standard SD checkpoints (CLIP).
- Closes #1559.
