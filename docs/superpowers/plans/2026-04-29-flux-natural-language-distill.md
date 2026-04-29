# Flux natural-language prompt distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-checkpoint `prompt_format` toggle (`"clip_tags"` vs `"natural_language"`) so the A1111 distiller skips LLM-driven CLIP-tag distillation for Flux on Forge Neo and flattens the brief's prose fields directly via the existing `flatten_brief_to_prompt()`.

**Architecture:** Single source of truth: `_CHECKPOINT_STYLE_MAP` gains a `prompt_format` key on every entry. New helper `prompt_format_for_checkpoint()` is a thin wrapper. `A1111ImageProvider.distill_prompt` branches on the format — NL mode calls `flatten_brief_to_prompt(brief)` directly (no LLM call); clip_tags mode keeps the current `_distill_with_llm` path. Existing tests for clip_tags behavior stay green; the LLM-required assertion narrows to the clip_tags branch.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `mypy`, `ruff`. Existing patterns: `re.compile` regex map (`_CHECKPOINT_STYLE_MAP`), `dataclasses.dataclass(frozen=True)` for `ImageBrief`, async `distill_prompt` returning `tuple[str, str | None]`.

**Spec:** `docs/superpowers/specs/2026-04-28-flux-natural-language-distill-design.md`. Closes #1559. Builds on #1558 (merged).

---

## File Structure

**Modified files (4):**
- `src/questfoundry/providers/checkpoint_styles.py` — add `prompt_format` key to all 10 map entries; new `prompt_format_for_checkpoint()` helper; update module docstring.
- `tests/unit/test_checkpoint_styles.py` — add `prompt_format` assertions to existing test class + parametrize coverage for the new helper.
- `src/questfoundry/providers/image_a1111.py` — `distill_prompt` branches on `prompt_format_for_checkpoint(self._model)`; LLM assertion moves inside the clip_tags branch; new imports.
- `tests/unit/test_image_a1111.py` — failing test asserting NL mode skips the LLM and returns `flatten_brief_to_prompt(brief)` verbatim; existing clip_tags tests stay green.

**No new files. No prompt template changes. No spec doc changes (those are in the spec file already committed).**

---

## Task 1: Add `prompt_format` field + helper — failing tests

TDD strict: write the test before any production change.

**Files:**
- Test: `tests/unit/test_checkpoint_styles.py`

- [ ] **Step 1: Append failing tests for `prompt_format` field + helper**

Append to `tests/unit/test_checkpoint_styles.py` (end of file, after the existing `TestResolveCheckpointStyle` class):

```python
class TestPromptFormat:
    """`prompt_format` field on each map entry + `prompt_format_for_checkpoint()`
    helper. Drives the A1111 distiller's clip_tags vs natural_language
    branching for Flux on Forge Neo (#1559)."""

    def test_resolve_checkpoint_style_returns_prompt_format(self) -> None:
        # Every entry must carry a prompt_format key.
        info = resolve_checkpoint_style("anything.safetensors")
        assert "prompt_format" in info
        assert info["prompt_format"] in {"clip_tags", "natural_language"}

    def test_flux_uses_natural_language(self) -> None:
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        assert prompt_format_for_checkpoint("flux1-dev-bnb-nf4-v2.safetensors") == "natural_language"
        assert prompt_format_for_checkpoint("flux1-dev-bnb-nf4.safetensors") == "natural_language"

    @pytest.mark.parametrize(
        "model",
        [
            "coloring_book.ckpt",
            "v1-5-pruned-emaonly.safetensors",
            "animagine-xl.safetensors",
            "Dreamshaper.safetensors",
            "sd_xl_base_1.0.safetensors",
            "dreamshaperXL_lightningDPMSDE.safetensors",
            "juggernautXL_ragnarokBy.safetensors",
            "dreamshaperXL_alpha2Xl10.safetensors",
            "dreamshaperXL_v1.safetensors",
        ],
    )
    def test_non_flux_checkpoints_use_clip_tags(self, model: str) -> None:
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        assert prompt_format_for_checkpoint(model) == "clip_tags"

    def test_unknown_checkpoint_uses_clip_tags(self) -> None:
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        # Default fallback: SD-family is the long tail; safer assumption is CLIP.
        assert prompt_format_for_checkpoint("totally-made-up-model.safetensors") == "clip_tags"

    def test_no_model_uses_clip_tags(self) -> None:
        # When no model is set, preserve the LLM-distill default path.
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        assert prompt_format_for_checkpoint(None) == "clip_tags"
        assert prompt_format_for_checkpoint("") == "clip_tags"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --frozen pytest tests/unit/test_checkpoint_styles.py::TestPromptFormat -v
```

Expected: failures. The first test (`test_resolve_checkpoint_style_returns_prompt_format`) fails with `AssertionError: 'prompt_format' not in info`. The other tests fail with `ImportError: cannot import name 'prompt_format_for_checkpoint'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/unit/test_checkpoint_styles.py
git commit -m "test(providers): failing tests for prompt_format field + helper (#1559)"
```

If pre-commit hooks block the failing-test commit, leave the file uncommitted in the working tree and report DONE_WITH_CONCERNS — the controller will commit it together with Task 2's implementation. Do NOT use `--no-verify`.

---

## Task 2: Implement `prompt_format` field + helper

**Files:**
- Modify: `src/questfoundry/providers/checkpoint_styles.py`

- [ ] **Step 1: Add `prompt_format` to all 10 map entries**

Open `src/questfoundry/providers/checkpoint_styles.py`. For each of the 10 entries in `_CHECKPOINT_STYLE_MAP`, add `"prompt_format"` as a new key. Flux gets `"natural_language"`; everything else gets `"clip_tags"`. Place `"prompt_format"` consistently at the end of each entry's dict (after `bad_example`).

Concretely:

```python
# Flux entry — natural_language
(
    re.compile(r"flux"),
    {
        "label": "Flux (photorealistic / highly-detailed)",
        # ... existing keys: style_hints, incompatible_styles, good_example, bad_example ...
        "prompt_format": "natural_language",
    },
),
```

```python
# All other 9 entries — clip_tags
(
    re.compile(r"coloring.?book"),
    {
        # ... existing keys ...
        "prompt_format": "clip_tags",
    },
),
# ... and so on for juggernaut, animagine, dreamshaperxl_lightning|alpha,
#     dreamshaperxl|dreamshaper.*xl, dreamshaper, sd_xl_base, v1[-_]5, default ...
```

The default-fallback entry (empty pattern, last in tuple) gets `"prompt_format": "clip_tags"`.

- [ ] **Step 2: Add the helper function below `resolve_checkpoint_style()`**

Append to `src/questfoundry/providers/checkpoint_styles.py`, after the existing `resolve_checkpoint_style()` definition:

```python
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
```

- [ ] **Step 3: Update module docstring**

In the module docstring at the top of `checkpoint_styles.py`, add a short sentence to the "Used by" paragraph noting the new prompt-format usage. The current docstring says:

```
Used by DRESS Phase 0 (prevention — when `--image-provider` is set, biases
ArtDirection toward checkpoint-compatible styles) and the A1111 distiller
(recovery — adapts CLIP-tag selection to the active checkpoint).
```

Append:

```
The A1111 distiller also reads `prompt_format` (via
`prompt_format_for_checkpoint`) to choose between LLM-driven CLIP-tag
distillation (CLIP-encoder checkpoints) and direct prose flattening
(T5-encoder checkpoints like Flux on Forge Neo) — see PR #1559.
```

- [ ] **Step 4: Run the new tests and confirm they pass**

```bash
uv run --frozen pytest tests/unit/test_checkpoint_styles.py -v
```

Expected: all `TestResolveCheckpointStyle` tests still green (no regression) AND all new `TestPromptFormat` tests pass.

- [ ] **Step 5: Run mypy + ruff**

```bash
uv run --frozen mypy src/questfoundry/providers/checkpoint_styles.py
uv run --frozen ruff check src/questfoundry/providers/checkpoint_styles.py tests/unit/test_checkpoint_styles.py
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/providers/checkpoint_styles.py
git commit -m "feat(providers): add prompt_format field + helper for Flux NL mode (#1559)"
```

---

## Task 3: A1111 distiller branch — failing test

**Files:**
- Test: `tests/unit/test_image_a1111.py`

- [ ] **Step 1: Append failing tests for the distill_prompt branch**

Append to `tests/unit/test_image_a1111.py` (end of file, after the existing `TestDistillerCheckpointHint` class):

```python
class TestDistillPromptFormatBranch:
    """`distill_prompt` branches on `prompt_format_for_checkpoint(self._model)`.
    Flux (T5 encoder) skips the LLM distill entirely and flattens the brief's
    prose fields via `flatten_brief_to_prompt()`. Standard SD/SDXL checkpoints
    keep the existing LLM-distill path (#1559)."""

    def _make_brief(self) -> "ImageBrief":
        from questfoundry.providers.image_brief import ImageBrief

        return ImageBrief(
            subject="warrior on a stone bridge",
            composition="wide shot, low angle",
            mood="tense, golden hour",
            entity_fragments=["scarred warrior with leather armor"],
            art_style="cinematic photography",
            art_medium="digital photo",
            palette=["amber", "slate"],
            style_exclusions="no anime, no anachronistic technology",
            aspect_ratio="16:9",
            category="scene",
        )

    @pytest.mark.asyncio
    async def test_flux_skips_llm_and_uses_flattener(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from questfoundry.providers.image_a1111 import A1111ImageProvider
        from questfoundry.providers.image_brief import flatten_brief_to_prompt

        # MagicMock LLM whose method usage we can audit.
        llm = MagicMock()
        llm.ainvoke = AsyncMock()  # would be called by _distill_with_llm
        provider = A1111ImageProvider(
            host="http://x", model="flux1-dev-bnb-nf4-v2.safetensors", llm=llm
        )

        brief = self._make_brief()
        result = await provider.distill_prompt(brief)

        # NL mode: result equals flatten_brief_to_prompt() output exactly.
        expected = flatten_brief_to_prompt(brief)
        assert result == expected

        # NL mode: LLM is never invoked.
        assert llm.ainvoke.await_count == 0

    @pytest.mark.asyncio
    async def test_clip_tags_checkpoint_still_uses_llm_distill(self) -> None:
        # Sanity: a non-Flux checkpoint must still go through _distill_with_llm.
        # We verify by stubbing _distill_with_llm and checking it was called.
        from unittest.mock import AsyncMock, MagicMock, patch

        from questfoundry.providers.image_a1111 import A1111ImageProvider

        llm = MagicMock()
        provider = A1111ImageProvider(
            host="http://x", model="juggernautXL_ragnarokBy.safetensors", llm=llm
        )

        brief = self._make_brief()
        with patch.object(
            provider, "_distill_with_llm", new=AsyncMock(return_value=("pos", "neg"))
        ) as mock_distill:
            result = await provider.distill_prompt(brief)

        assert result == ("pos", "neg")
        mock_distill.assert_awaited_once_with(brief)

    @pytest.mark.asyncio
    async def test_flux_works_without_llm(self) -> None:
        # NL mode should not require an LLM at all — `_llm=None` is OK.
        from questfoundry.providers.image_a1111 import A1111ImageProvider
        from questfoundry.providers.image_brief import flatten_brief_to_prompt

        provider = A1111ImageProvider(
            host="http://x", model="flux1-dev-bnb-nf4-v2.safetensors", llm=None
        )

        brief = self._make_brief()
        result = await provider.distill_prompt(brief)

        assert result == flatten_brief_to_prompt(brief)

    @pytest.mark.asyncio
    async def test_clip_tags_without_llm_raises(self) -> None:
        # Sanity: clip_tags branch still raises when no LLM is provided.
        from questfoundry.providers.image_a1111 import A1111ImageProvider
        from questfoundry.providers.base import ImageProviderError

        provider = A1111ImageProvider(
            host="http://x", model="juggernautXL_ragnarokBy.safetensors", llm=None
        )

        brief = self._make_brief()
        with pytest.raises(ImageProviderError):
            await provider.distill_prompt(brief)
```

If `pytest_asyncio` isn't already configured for this test file, the `@pytest.mark.asyncio` decorator may need a different invocation pattern. Check how the existing async tests in `tests/unit/test_image_a1111.py` are structured — copy their pattern. (If existing tests use plain `async def` without the marker because of project-wide `asyncio_mode = "auto"` config, drop the decorators here too.)

- [ ] **Step 2: Run the failing tests**

```bash
uv run --frozen pytest tests/unit/test_image_a1111.py::TestDistillPromptFormatBranch -v
```

Expected:
- `test_flux_skips_llm_and_uses_flattener` fails — Flux currently goes through `_distill_with_llm` which would either invoke the LLM mock or fail.
- `test_flux_works_without_llm` fails — current code raises `ImageProviderError` when `_llm is None` regardless of checkpoint.
- `test_clip_tags_checkpoint_still_uses_llm_distill` may pass already (it tests existing behavior) — that's OK.
- `test_clip_tags_without_llm_raises` may pass already — also OK.

The two failing tests are the load-bearing ones for the branch.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/unit/test_image_a1111.py
git commit -m "test(providers): failing tests for distill_prompt prompt_format branch (#1559)"
```

If pre-commit hooks block, report DONE_WITH_CONCERNS and leave uncommitted. Do NOT use `--no-verify`.

---

## Task 4: A1111 distiller branch — implementation

**Files:**
- Modify: `src/questfoundry/providers/image_a1111.py`

- [ ] **Step 1: Add the import for `prompt_format_for_checkpoint` + `flatten_brief_to_prompt`**

In `src/questfoundry/providers/image_a1111.py`, find the existing imports block. Add:

```python
from questfoundry.providers.checkpoint_styles import (
    prompt_format_for_checkpoint,
    resolve_checkpoint_style,  # may already be imported via _format_checkpoint_hint;
                                # if so, add prompt_format_for_checkpoint to the same line
)
from questfoundry.providers.image_brief import (
    ImageBrief,                    # may already be imported
    flatten_brief_to_prompt,
)
```

Match the existing import style — if `image_brief.ImageBrief` is already imported, just add `flatten_brief_to_prompt` to the same import statement. Same for `checkpoint_styles.resolve_checkpoint_style` (added in PR #1558).

- [ ] **Step 2: Branch in `distill_prompt`**

Replace the existing `distill_prompt` method body (around line 227-239):

```python
    async def distill_prompt(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Transform a structured brief into SD-optimised prompts via LLM.

        Raises:
            ImageProviderError: If no LLM was provided at construction.
        """
        if self._llm is None:
            raise ImageProviderError(
                "a1111",
                "A1111 prompt distillation requires an LLM. "
                "Pass --provider to generate-images or set QF_PROVIDER.",
            )
        return await self._distill_with_llm(brief)
```

with:

```python
    async def distill_prompt(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Transform a structured brief into a renderer-shaped prompt.

        Branches on the active checkpoint's prompt format (#1559):

        - ``"natural_language"`` (Flux on Forge Neo, T5 encoder, ~512-token
          window): skip LLM distillation entirely. The structured brief
          already encodes subject/composition/mood/entities/style/medium/
          palette as comma-joined prose via :func:`flatten_brief_to_prompt`,
          which is the shape T5 was trained on.
        - ``"clip_tags"`` (SDXL, SD1.5, etc.): LLM-distill into the comma-tag
          form CLIP expects, using the existing :meth:`_distill_with_llm`.

        Raises:
            ImageProviderError: If the active checkpoint requires
                ``clip_tags`` distillation but no LLM was provided at
                construction. ``natural_language`` mode runs without an LLM.
        """
        if prompt_format_for_checkpoint(self._model) == "natural_language":
            return flatten_brief_to_prompt(brief)

        if self._llm is None:
            raise ImageProviderError(
                "a1111",
                "A1111 prompt distillation requires an LLM. "
                "Pass --provider to generate-images or set QF_PROVIDER.",
            )
        return await self._distill_with_llm(brief)
```

- [ ] **Step 3: Run the failing tests; confirm green**

```bash
uv run --frozen pytest tests/unit/test_image_a1111.py::TestDistillPromptFormatBranch -v
```

Expected: 4 passed.

- [ ] **Step 4: Run the wider distiller test file to confirm no regression**

```bash
uv run --frozen pytest tests/unit/test_image_a1111.py -x -q
```

Expected: all green. The existing `TestDistillerCheckpointHint` class (PR #1558) covers `_format_checkpoint_hint` independently and is unaffected by the new branch.

- [ ] **Step 5: Run mypy + ruff**

```bash
uv run --frozen mypy src/questfoundry/providers/image_a1111.py
uv run --frozen ruff check src/questfoundry/providers/image_a1111.py
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/providers/image_a1111.py tests/unit/test_image_a1111.py
git commit -m "feat(providers): A1111 distill_prompt branches on prompt_format for Flux (#1559)"
```

If Task 3's failing-test commit was held back due to pre-commit, this commit picks up both the test additions and the implementation in one shot.

---

## Task 5: Final verification + draft PR

**Files:**
- N/A (verification + git only)

- [ ] **Step 1: Run the unit test suite**

```bash
uv run --frozen pytest tests/unit/ -x -q
```

Expected: all green (the same pre-existing flaky `test_provider_factory.py::test_create_chat_model_ollama_success` may surface — confirm it passes in isolation if so).

- [ ] **Step 2: Run mypy on changed source files**

```bash
uv run --frozen mypy src/questfoundry/providers/checkpoint_styles.py src/questfoundry/providers/image_a1111.py src/questfoundry/providers/image_brief.py
```

Expected: clean.

- [ ] **Step 3: Run ruff**

```bash
uv run --frozen ruff check src/ tests/
uv run --frozen ruff format --check src/ tests/
```

Expected: clean. If format issues, run without `--check` to fix and re-commit.

- [ ] **Step 4: Verify the new fields and helper exist where expected**

```bash
rg "prompt_format" src/questfoundry/providers/checkpoint_styles.py
```

Expected: 10 entry-key occurrences (one per map entry) + the helper signature + at least one docstring mention. Roughly 13-15 hits.

```bash
rg "prompt_format_for_checkpoint" src/questfoundry/
```

Expected: 1 definition in `checkpoint_styles.py` + 1 use in `image_a1111.py`.

- [ ] **Step 5: Push branch + open draft PR**

```bash
git push -u origin feat/forge-flux-natural-language-distill
gh pr create --draft --title "feat(providers): natural-language prompt distillation for Flux on Forge Neo (#1559)" --body "$(cat <<'EOF'
## Summary

Adds a per-checkpoint \`prompt_format\` toggle to \`_CHECKPOINT_STYLE_MAP\` (\`"clip_tags"\` default; \`"natural_language"\` for Flux). \`A1111ImageProvider.distill_prompt\` branches on the format: Flux (T5 encoder, 512-token window) skips LLM distillation entirely and flattens the brief's prose fields via the existing \`flatten_brief_to_prompt()\`. Standard SDXL/SD1.5/Animagine/Juggernaut/etc. keep the LLM-driven CLIP-tag distillation unchanged.

The user's A1111 instance has been replaced with **Forge Neo** (API-compatible with A1111), which hosts Flux alongside standard SD checkpoints. The toggle is per-checkpoint, not per-provider — provider identifier stays \`a1111\`.

## Spec + plan

- Design: \`docs/superpowers/specs/2026-04-28-flux-natural-language-distill-design.md\`
- Implementation plan: \`docs/superpowers/plans/2026-04-29-flux-natural-language-distill.md\`

Builds on PR #1558 (merged 2026-04-28).

Closes #1559.

## Cascade (4 files)

1. \`src/questfoundry/providers/checkpoint_styles.py\` — \`prompt_format\` field on all 10 map entries + new \`prompt_format_for_checkpoint()\` helper
2. \`src/questfoundry/providers/image_a1111.py\` — \`distill_prompt\` branches on format; LLM assertion narrows to \`clip_tags\` branch
3. \`tests/unit/test_checkpoint_styles.py\` — \`TestPromptFormat\` class (5 tests covering field presence, Flux NL mode, 9 non-Flux checkpoints in clip_tags mode, default fallback, no-model fallback)
4. \`tests/unit/test_image_a1111.py\` — \`TestDistillPromptFormatBranch\` class (4 tests covering Flux→flatten, clip_tags→LLM, Flux without LLM, clip_tags without LLM raises)

## Test plan

- [x] \`uv run pytest tests/unit/test_checkpoint_styles.py\` — green
- [x] \`uv run pytest tests/unit/test_image_a1111.py\` — green
- [x] \`uv run pytest tests/unit/\` — full suite green (modulo pre-existing flaky network test)
- [x] \`uv run mypy src/questfoundry/\` — clean
- [x] \`uv run ruff check src/ tests/\` — clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Wait for bot review loop**

Per CLAUDE.md, address findings in-PR. Gemini may be at daily quota; claude-review approval is sufficient. Flip ready when claude-review LGTM + all CI green + bot's review body explicitly says "Ready to merge" (read the body, not the check status).

---

## Self-Review Notes

Spec coverage check:

- ✅ Spec §`prompt_format` field on each map entry → Task 2 step 1
- ✅ Spec §`prompt_format_for_checkpoint()` helper → Task 2 step 2
- ✅ Spec §`A1111ImageProvider.distill_prompt` branch → Task 4 step 2
- ✅ Spec §LLM assertion narrowing → Task 4 step 2 (the `if self._llm is None` check moves below the NL-mode early return)
- ✅ Spec §What flows through to T5 (`flatten_brief_to_prompt` reuse) → Task 4 step 2 + Task 3 test asserting equality with the flattener output
- ✅ Spec §file cascade (4 sites) → Tasks 1-4 (one task per file pair: test + impl)
- ✅ Spec §verification (`pytest`, `rg "prompt_format" src/questfoundry/providers/`) → Task 5
- ✅ Spec §risks: contract change for consumers expecting strict-equal key sets → covered by Task 1 step 1 (`test_returns_required_keys` already uses `>=` per spec; no regression)

No placeholders. No "implement later" / "similar to Task N" / TBD. Type consistency: `prompt_format`, `prompt_format_for_checkpoint`, `flatten_brief_to_prompt` referenced consistently across tasks.
