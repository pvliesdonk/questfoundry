# `belongs_to: list[str]` rename — Implementation PR Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementation PR (2 of 2) for #1564. Rename `InitialBeat.path_id: str` + `also_belongs_to: str | None` to `belongs_to: list[str]` (`min_length=1, max_length=2`) across the SEED schema, mutation layer, all SEED prompt templates, all production consumers, and all test fixtures. Atomic rename per CLAUDE.md "Replace directly, no compat shim."

**Architecture:** Schema-first refactor. The `InitialBeat` Pydantic model gains a single symmetric `belongs_to: list[str]` field replacing the asymmetric pair. The mutation layer (`graph/mutations.py::_get_path_ids_from_beat`) reads the list directly and creates one `belongs_to` graph edge per element (it already supports 2-edge creation; only the read pattern changes). All prompt templates that describe the SEED beat schema rewrite their schema descriptions and GOOD/BAD examples to use the list shape. Test fixtures replace `path_id=`/`also_belongs_to=` kwargs and YAML keys with `belongs_to=[...]`.

The schema's own `path_id` field is the only one being renamed. `Path.path_id` (the Path node's own identifier) and `Consequence.path_id` (which path a consequence belongs to) are NOT touched — they're the Path node's identity, not an InitialBeat field.

**Tech Stack:** Python 3.11+, `pydantic`, `pytest`, `ruamel.yaml`, `ruff`, `mypy`. Existing patterns: `_migrate_paths_to_path_id` is a `@model_validator(mode="before")` that originally migrated `paths: list[str]` to the asymmetric form — this rename effectively reverses that migration with a new field name.

**Spec:** `docs/superpowers/specs/2026-04-29-belongs-to-list-rename-design.md`. Closes #1564 (part 2/2 — depends on PR #1565 having merged, which it has).

---

## File Structure

**Modified files (~17-25 sites across these groups):**

- **Schema** (1 file): `src/questfoundry/models/seed.py` — `InitialBeat.path_id` and `also_belongs_to` → `belongs_to: list[str]`. Drop `_migrate_paths_to_path_id` (no longer needed; the field name comes back to the original `paths` shape, just renamed). Replace `_also_belongs_to_differs_from_path_id` with a list-uniqueness validator.
- **Mutation layer** (1 file): `src/questfoundry/graph/mutations.py` — `_get_path_ids_from_beat()` reads `beat["belongs_to"]` (list) instead of the asymmetric pair. Creates one `belongs_to` graph edge per element (already does this; only the read pattern changes).
- **Production consumers** (a few files): `src/questfoundry/pipeline/stages/seed.py`, `src/questfoundry/agents/serialize.py`, `src/questfoundry/graph/seed_validation.py` — anywhere code reads `beat.path_id` / `beat.also_belongs_to` from an `InitialBeat`. Reads switch to `beat.belongs_to[0]` (primary) and `beat.belongs_to[1]` (sibling, if present).
- **Prompt templates** (5 files): `prompts/templates/serialize_seed_sections.yaml`, `prompts/templates/summarize_seed_sections.yaml`, `prompts/templates/serialize_seed.yaml`, `prompts/templates/summarize_seed.yaml`, `prompts/templates/discuss_seed.yaml` — schema descriptions, GOOD/BAD examples, FINAL CHECK assertions. Every reference to `path_id` and/or `also_belongs_to` for a SEED beat rewrites to `belongs_to: [...]`.
- **Tests** (~12-15 files): `tests/unit/test_seed_models.py`, `test_seed_stage.py`, `test_seed_validation.py`, `test_serialize.py`, `test_mutations.py`, `test_grow_*.py`, `test_polish_*.py`, `test_inspection.py`, `tests/integration/test_y_shape_end_to_end.py`, `tests/fixtures/grow_fixtures.py` — every `InitialBeat(path_id=..., also_belongs_to=...)` constructor or YAML/dict fixture using those keys.

**No new files. No new tests beyond what already exists. Test count stays the same; assertions and fixtures are mechanically rewritten.**

---

## Task 1: Inventory + branch confirmation

**Files:** N/A (read-only)

- [ ] **Step 1: Confirm we're on the impl branch and the spec PR (#1565) is merged**

```bash
git rev-parse --abbrev-ref HEAD
# expected: refactor/1564-belongs-to-list-impl

git log --oneline origin/main -5 | head -3
# expected: top commit references #1565 merge (069c1ef9 or later)
```

- [ ] **Step 2: Inventory `also_belongs_to` sites — these are the high-confidence rename targets**

```bash
rg -l "also_belongs_to" src/ tests/ prompts/
# expected: ~17 files
```

Save this list mentally — every file here MUST be touched.

- [ ] **Step 3: Inventory `InitialBeat`-related `path_id` sites**

```bash
rg "InitialBeat\(" tests/ src/ | head -30
```

Every fixture constructor matching `InitialBeat(...path_id=..., ...)` needs updating. Some test files use only `path_id=` (no `also_belongs_to`) for post-commit beats — those still need conversion to `belongs_to=[<path>]`.

```bash
# Find candidates: path_id appearances NOT inside Path() or Consequence() constructors.
# Heuristic: lines with `path_id=` in fixtures, broadly.
rg -n "path_id=" tests/ | head -30
```

For each hit, check context — is it constructing an `InitialBeat`/initial beat dict, or a `Path`/`Consequence`? Rule: if the surrounding fixture is for a beat (variable/key contains `beat` or `initial_beats`), rename. If it's for a path or consequence, leave alone.

- [ ] **Step 4: Note: this task does not commit anything**

Inventory only. The rewrite happens in Task 2.

---

## Task 2: Schema rename (`InitialBeat`)

**Files:**
- Modify: `src/questfoundry/models/seed.py` lines ~240-330 (the `InitialBeat` class + `_migrate_paths_to_path_id` + `_also_belongs_to_differs_from_path_id`)

- [ ] **Step 1: Replace the `InitialBeat` class field declarations**

Find (lines ~273-285):

```python
    path_id: str = Field(
        min_length=1,
        description="Primary path this beat belongs to (first belongs_to edge)",
    )
    also_belongs_to: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Sibling path for pre-commit (Y-shape) beats: creates a second "
            "belongs_to edge. Must be null for post-commit beats. Must "
            "reference a path with the same parent dilemma as path_id."
        ),
    )
```

Replace with:

```python
    belongs_to: list[str] = Field(
        min_length=1,
        max_length=2,
        description=(
            "Path(s) this beat belongs to. List of length 2 means dual "
            "membership (pre-commit shared beat — both paths of the same "
            "dilemma). List of length 1 means singular membership (commit, "
            "post-commit, or gap beat). Cross-dilemma dual membership is "
            "rejected by the mutation layer; same-dilemma sibling-pair check "
            "happens there as well, where path→dilemma resolution is available."
        ),
    )
```

- [ ] **Step 2: Update the class docstring**

Find the existing class docstring (lines ~240-269) that references `path_id` and `also_belongs_to`. Rewrite the relevant paragraphs:

The current docstring says (excerpt):
```
* **Post-commit beats** (the default) belong to exactly one path via
  ``path_id``; ``also_belongs_to`` is ``None``. These beats prove one
  answer and are exclusive to the path they belong to.
* **Pre-commit beats** (shared dilemma setup) belong to *both* paths of
  their dilemma via ``path_id`` and ``also_belongs_to``. This is the
  Y-shape ratified in #1206/#1208: every player experiences pre-commit
  beats regardless of which answer they later choose.
```

Replace with:
```
* **Post-commit beats** (the default) belong to exactly one path —
  ``belongs_to`` is a single-element list. These beats prove one answer
  and are exclusive to the path they belong to.
* **Pre-commit beats** (shared dilemma setup) belong to *both* paths of
  their dilemma — ``belongs_to`` is a two-element list, both referencing
  paths of the same parent dilemma. This is the Y-shape ratified in
  #1206/#1208 (and refined to symmetric list form in #1564): every
  player experiences pre-commit beats regardless of which answer they
  later choose.
```

Also replace the `Attributes:` block entries for `path_id` and `also_belongs_to` with a single entry for `belongs_to`:

```
Attributes:
    beat_id: Unique identifier for the beat.
    summary: What happens in this beat.
    belongs_to: Path(s) this beat belongs to. Single-element list for
        commit/post-commit/gap beats; two-element list (both paths of
        the same dilemma) for pre-commit shared beats. Cross-dilemma
        dual membership is rejected by the mutation layer.
    dilemma_impacts: How this beat affects dilemmas.
    entities: Entity IDs present in this beat.
    location: Primary location entity ID.
    location_alternatives: Other valid locations (enables intersection flexibility).
    temporal_hint: Advisory placement relative to another dilemma (consumed by GROW).
```

- [ ] **Step 3: Replace `_migrate_paths_to_path_id` with a noop / drop it**

The current `_migrate_paths_to_path_id` (lines ~287-323) migrated legacy `paths: list[str]` input data into the asymmetric `path_id` + `also_belongs_to` form. Now that the field IS a list (renamed to `belongs_to`), the migration is no longer needed. Two options:

**Option A (cleaner): drop the validator entirely.** The new `belongs_to: list[str]` field directly accepts list input.

**Option B: keep a slim migrator that maps legacy `paths` → `belongs_to`.** Only useful if there are on-disk artifacts with the old `paths` field. Project memory says #1206 already migrated those, so any pre-#1206 snapshots are dead.

Take **Option A**. Delete the `@model_validator(mode="before")` decorator and the `_migrate_paths_to_path_id` method body in full. Remove the `import warnings` at the top of the file IF nothing else in the file uses it (`grep "warnings\." src/questfoundry/models/seed.py` after deletion — if only the migrator used it, drop the import).

- [ ] **Step 4: Replace `_also_belongs_to_differs_from_path_id` validator with a list-uniqueness check**

Find (lines ~325-330):

```python
    @model_validator(mode="after")
    def _also_belongs_to_differs_from_path_id(self) -> InitialBeat:
        """Dual membership requires two distinct path IDs."""
        if self.also_belongs_to is not None and self.also_belongs_to == self.path_id:
            msg = "also_belongs_to must differ from path_id — dual membership needs two paths."
            raise ValueError(msg)
        return self
```

Replace with:

```python
    @model_validator(mode="after")
    def _belongs_to_elements_unique(self) -> InitialBeat:
        """Dual membership requires two distinct path IDs."""
        if len(self.belongs_to) == 2 and self.belongs_to[0] == self.belongs_to[1]:
            msg = "belongs_to elements must be distinct — dual membership needs two paths."
            raise ValueError(msg)
        return self
```

The same-dilemma sibling-path check stays in the mutation layer where path→dilemma resolution is available.

- [ ] **Step 5: Run tests just for `models/seed.py` to surface the cascade**

```bash
uv run --frozen pytest tests/unit/test_seed_models.py -x 2>&1 | tail -20
```

Expected: many failures. Every test using `InitialBeat(path_id=..., ...)` will fail. That's fine — Task 4 fixes them. Don't try to fix tests yet; this is a sanity check that the model layer compiles.

- [ ] **Step 6: Commit (production code only — tests follow)**

```bash
git add src/questfoundry/models/seed.py
git commit -m "refactor(seed): InitialBeat path_id+also_belongs_to → belongs_to: list[str] (#1564)"
```

Pre-commit hooks (mypy, ruff) will scan the file. mypy may fail because downstream consumers in `pipeline/stages/seed.py`, `agents/serialize.py`, `graph/mutations.py`, `graph/seed_validation.py` still reference `beat.path_id` and `beat.also_belongs_to`. **If mypy fails here, do not bypass.** Move on to Tasks 3 + 4 to fix the consumers, then come back and amend (or do them in one larger commit).

If pre-commit blocks this commit due to mypy errors in unrelated production files, take Option B: skip this commit and bundle it with Tasks 3 and 4 in a single atomic commit. The committer's choice depends on whether the schema change alone passes mypy in isolation.

---

## Task 3: Mutation layer + production consumers

**Files:**
- Modify: `src/questfoundry/graph/mutations.py` lines ~855-880 (the `_get_path_ids_from_beat()` helper)
- Modify: `src/questfoundry/graph/seed_validation.py` (every read of `beat.path_id` / `beat.also_belongs_to`)
- Modify: `src/questfoundry/pipeline/stages/seed.py` (every read)
- Modify: `src/questfoundry/agents/serialize.py` (every read)
- Modify: `src/questfoundry/pipeline/stages/brainstorm.py` (1 reference per the count)

- [ ] **Step 1: Rewrite `_get_path_ids_from_beat()` in `mutations.py`**

Find (lines ~855-880):

```python
def _get_path_ids_from_beat(beat: dict[str, Any]) -> tuple[str, ...]:
    """...
    - Y-shape current: ``path_id`` + optional ``also_belongs_to``.
    - Legacy single: ``path_id`` alone.
    ...
    Returns:
        Tuple of 0, 1, or 2 raw path IDs in declaration order (``path_id``
        first, then ``also_belongs_to``). A return of 2 always represents a
        ...
    """
    if beat.get("path_id"):
        primary = str(beat["path_id"])
        also = beat.get("also_belongs_to")
        ...
```

Rewrite the body to read `belongs_to` (a list):

```python
def _get_path_ids_from_beat(beat: dict[str, Any]) -> tuple[str, ...]:
    """Extract path IDs from a SEED beat dict, in declaration order.

    Reads the symmetric ``belongs_to`` list (#1564). Returns 0 (no
    membership — structural beats), 1 (singular — commit / post-commit /
    gap), or 2 (dual — pre-commit shared) raw path IDs.

    Args:
        beat: Beat dict (typically from an ``InitialBeat.model_dump()``
            or graph node data).

    Returns:
        Tuple of 0, 1, or 2 raw path IDs in declaration order. A return
        of 2 always represents a shared pre-commit beat with dual
        ``belongs_to`` to two paths of the same dilemma.
    """
    belongs_to = beat.get("belongs_to") or []
    return tuple(str(p) for p in belongs_to)
```

The function previously also handled a "legacy single `path_id`" path — drop it. The schema is now uniform.

- [ ] **Step 2: Sweep production code for `beat.path_id` and `beat.also_belongs_to` reads**

```bash
rg -n "\.path_id\b|\.also_belongs_to\b" src/questfoundry/graph/seed_validation.py src/questfoundry/pipeline/stages/seed.py src/questfoundry/agents/serialize.py src/questfoundry/pipeline/stages/brainstorm.py
```

For each hit:
- `beat.path_id` → `beat.belongs_to[0]` (the primary path; first element of the list).
- `beat.also_belongs_to` → `beat.belongs_to[1] if len(beat.belongs_to) >= 2 else None`.

If the existing code branched on `if beat.also_belongs_to is not None:`, rewrite as `if len(beat.belongs_to) >= 2:`.

If existing code constructed a list of paths from `[beat.path_id, beat.also_belongs_to]` (filtering None), simplify to `beat.belongs_to`.

For dict-based reads (`beat["path_id"]`, `beat["also_belongs_to"]`), apply the same pattern with dict indexing.

- [ ] **Step 3: Run mypy on the production source tree**

```bash
uv run --frozen mypy src/questfoundry/
```

Expected: clean. If errors surface, fix them — these are real type issues from incomplete sweep.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/graph/mutations.py src/questfoundry/graph/seed_validation.py src/questfoundry/pipeline/stages/seed.py src/questfoundry/agents/serialize.py src/questfoundry/pipeline/stages/brainstorm.py
git commit -m "refactor(seed): mutation + consumers read belongs_to: list[str] (#1564)"
```

If pre-commit hooks fail because tests are still using the old field names, that's expected — Task 4 fixes the tests. mypy on `src/` should pass; pytest on `tests/` will fail until Task 4 completes.

---

## Task 4: Prompt templates

**Files:**
- Modify: `prompts/templates/serialize_seed_sections.yaml`
- Modify: `prompts/templates/summarize_seed_sections.yaml`
- Modify: `prompts/templates/serialize_seed.yaml`
- Modify: `prompts/templates/summarize_seed.yaml`
- Modify: `prompts/templates/discuss_seed.yaml`

- [ ] **Step 1: Inventory hits per template**

```bash
for f in prompts/templates/serialize_seed_sections.yaml prompts/templates/summarize_seed_sections.yaml prompts/templates/serialize_seed.yaml prompts/templates/summarize_seed.yaml prompts/templates/discuss_seed.yaml; do
  echo "=== $f ==="
  rg -n "path_id|also_belongs_to" "$f"
done
```

Expected: every YAML schema description, GOOD/BAD example, and FINAL CHECK assertion that mentions either field.

- [ ] **Step 2: Rewrite each occurrence**

For each YAML template:

- **Schema description block** (typically: "fields are `beat_id`, `summary`, `path_id`, `also_belongs_to`, ..."): replace with "...`beat_id`, `summary`, `belongs_to` (list of 1 or 2 path IDs), ...".
- **Field-by-field rule sections** for `path_id` and `also_belongs_to`: collapse into a single `belongs_to` section. The combined section should explain: list shape, length 1 for commit/post-commit/gap beats, length 2 for shared pre-commit beats with both paths from the same dilemma.
- **GOOD examples** that show `path_id: path::foo, also_belongs_to: path::bar` (or similar JSON): rewrite to `"belongs_to": ["path::foo", "path::bar"]`. For singular GOOD examples (post-commit), use `"belongs_to": ["path::foo"]`.
- **BAD examples** that target the old asymmetric field bugs: rewrite to target list-shape mistakes (e.g. `"belongs_to": []` is REJECTED — every beat must belong to at least one path; `"belongs_to": ["path::foo", "path::foo"]` is REJECTED — elements must be distinct; `"belongs_to": ["path::foo", "path::bar", "path::baz"]` is REJECTED — at most 2 elements).
- **FINAL CHECK assertions** that say "every shared beat has `also_belongs_to` set" → rewrite to "every shared beat has `belongs_to` of length 2".
- **WHAT NOT TO DO entries** mentioning `also_belongs_to` → rewrite for the list shape.

For `discuss_seed.yaml`: it's the creative-LLM brief, not the structured schema, so most occurrences are prose ("decide which beats are shared between paths"). Where the prose explicitly names `path_id` / `also_belongs_to`, switch to `belongs_to` with the list-shape framing.

- [ ] **Step 3: Verify zero survivors per template**

```bash
for f in prompts/templates/*.yaml; do
  if rg -q "also_belongs_to|path_id" "$f"; then
    echo "$f: $(rg -c 'also_belongs_to|path_id' "$f") hits remaining"
  fi
done
```

Expected: zero hits for `also_belongs_to`. Some `path_id` hits may remain if the YAML legitimately references Path's `path_id` field (e.g. in a "valid_path_ids" enumeration) — those are fine. The rename is `InitialBeat.path_id` only.

- [ ] **Step 4: Commit**

```bash
git add prompts/templates/serialize_seed_sections.yaml prompts/templates/summarize_seed_sections.yaml prompts/templates/serialize_seed.yaml prompts/templates/summarize_seed.yaml prompts/templates/discuss_seed.yaml
git commit -m "refactor(prompts): SEED prompts describe belongs_to: list[str] (#1564)"
```

---

## Task 5: Test fixtures + assertion sweep

**Files:**
- Modify: every `.py` file under `tests/` matching `rg -l "InitialBeat\b" tests/` (production code already done in Tasks 2-3).
- Specifically (from inventory): `tests/unit/test_seed_models.py`, `test_seed_stage.py`, `test_seed_validation.py`, `test_serialize.py`, `test_mutations.py`, `test_grow_models.py`, `test_grow_stage.py`, `test_grow_algorithms.py`, `test_grow_validation_contract.py`, `test_polish_phases.py`, `test_polish_phase5_models.py`, `test_polish_apply.py`, `test_polish_deterministic.py`, `test_polish_phase5_context.py`, `test_polish_llm_phases.py`, `test_inspection.py`, `test_context_compact.py`, `test_entity_naming.py`, `test_ontology_explored.py`, `test_graph_context.py`, `tests/integration/test_y_shape_end_to_end.py`, `tests/integration/test_grow_e2e.py`, `tests/fixtures/grow_fixtures.py`.

- [ ] **Step 1: Sweep every test file for `InitialBeat` constructions**

For each file in the list above, find every `InitialBeat(...)` constructor call and every dict-fixture / YAML fixture that uses `path_id`/`also_belongs_to` keys for an InitialBeat.

Conversion rules:
- `InitialBeat(path_id="path::foo", also_belongs_to="path::bar", ...)` → `InitialBeat(belongs_to=["path::foo", "path::bar"], ...)`
- `InitialBeat(path_id="path::foo", ...)` (no `also_belongs_to`) → `InitialBeat(belongs_to=["path::foo"], ...)`
- `InitialBeat(path_id="path::foo", also_belongs_to=None, ...)` → `InitialBeat(belongs_to=["path::foo"], ...)`
- Dict fixtures: `{"path_id": "path::foo", "also_belongs_to": "path::bar", ...}` → `{"belongs_to": ["path::foo", "path::bar"], ...}` (and the singular variants analogously).
- Direct attribute reads in assertions: `assert beat.path_id == "path::foo"` → `assert beat.belongs_to[0] == "path::foo"` (or `assert beat.belongs_to == ["path::foo"]` for the singular case).
- `assert beat.also_belongs_to == "path::bar"` → `assert beat.belongs_to[1] == "path::bar"` (or `assert beat.belongs_to == ["path::foo", "path::bar"]`).
- `assert beat.also_belongs_to is None` → `assert len(beat.belongs_to) == 1`.

Distinguish from `Path.path_id` and `Consequence.path_id` — those are NOT changing. The rule of thumb: if the constructor or assertion is on a beat object, rename. If on a Path or Consequence, leave alone.

- [ ] **Step 2: Run test_seed_models.py first — fastest signal**

```bash
uv run --frozen pytest tests/unit/test_seed_models.py -x 2>&1 | tail -20
```

Expected: all green after rewrite. If anything fails, fix the surrounding fixture or assertion.

- [ ] **Step 3: Sweep targeted unit tests**

```bash
uv run --frozen pytest tests/unit/test_seed_models.py tests/unit/test_seed_stage.py tests/unit/test_seed_validation.py tests/unit/test_serialize.py tests/unit/test_mutations.py tests/unit/test_grow_models.py tests/unit/test_grow_stage.py tests/unit/test_grow_algorithms.py tests/unit/test_grow_validation_contract.py tests/unit/test_polish_phases.py tests/unit/test_polish_phase5_models.py tests/unit/test_polish_apply.py tests/unit/test_polish_deterministic.py tests/unit/test_polish_phase5_context.py tests/unit/test_polish_llm_phases.py tests/unit/test_inspection.py tests/unit/test_context_compact.py tests/unit/test_entity_naming.py tests/unit/test_ontology_explored.py tests/unit/test_graph_context.py -x 2>&1 | tail -10
```

Expected: all green. If failures, fix the offending fixture and re-run.

- [ ] **Step 4: Verify zero `also_belongs_to` survivors anywhere**

```bash
rg "also_belongs_to" src/ tests/ prompts/
```

Expected: zero hits.

```bash
rg "InitialBeat\(.*path_id=" tests/
```

Expected: zero hits (every `InitialBeat(path_id=...)` should now use `belongs_to=[...]`).

```bash
rg "InitialBeat\(.*paths=" tests/
```

Expected: zero hits (the legacy `paths` keyword is gone from fixtures too — the migrator that handled it was dropped).

- [ ] **Step 5: Run mypy and ruff**

```bash
uv run --frozen mypy src/
uv run --frozen ruff check src/ tests/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "refactor(tests): InitialBeat fixtures use belongs_to: list[str] (#1564)"
```

---

## Task 6: Final verification + draft PR

**Files:** N/A (verification + git only)

- [ ] **Step 1: Full unit test sweep**

```bash
uv run --frozen pytest tests/unit/ -x -q 2>&1 | tail -10
```

Expected: all green. (The pre-existing flaky `test_provider_factory.py::test_create_chat_model_ollama_success` may surface — confirm in isolation if so.)

- [ ] **Step 2: Final survivor sweep**

```bash
rg "also_belongs_to" src/ tests/ prompts/ docs/design/
```

Expected: ZERO hits. (`docs/design/story-graph-ontology.md` mentions `also_belongs_to` once in the Impact paragraph as historical reference — that's the only allowed survivor and it's intentional. After this PR merges, that reference is the entire historical record of the deprecated form.)

```bash
rg "InitialBeat\b" src/ tests/ | grep "path_id="
```

Expected: zero. No InitialBeat construction uses `path_id=` anymore.

- [ ] **Step 3: mypy + ruff**

```bash
uv run --frozen mypy src/questfoundry/
uv run --frozen ruff check src/ tests/
uv run --frozen ruff format --check src/ tests/
```

Expected: clean.

- [ ] **Step 4: Push branch + open draft PR**

```bash
git push -u origin refactor/1564-belongs-to-list-impl
gh pr create --draft --title "refactor: InitialBeat belongs_to: list[str] — schema + mutations + prompts + tests (#1564 part 2/2)" --body "$(cat <<'EOF'
## Summary

Implementation PR (2 of 2) for #1564. Replaces the asymmetric \`InitialBeat.path_id: str\` + \`also_belongs_to: str | None\` schema with the symmetric \`belongs_to: list[str]\` (\`min_length=1, max_length=2\`) shape that matches the graph layer. Atomic rename per CLAUDE.md \"Replace directly, no compat shim.\"

The asymmetric form was the direct cause of the murder6 production failure: qwen3:4b emitted \`path_id\` set with \`also_belongs_to\` omitted entirely, failing after 3 retry attempts. The list shape is structurally unambiguous — \`min_length=1, max_length=2\` is enforceable at the schema layer, and the model emits a list or doesn't.

This PR depends on PR #1565 (spec PR — the ontology + seed.md doc updates) which has merged.

## Spec + plan

- Design: \`docs/superpowers/specs/2026-04-29-belongs-to-list-rename-design.md\`
- Plan: \`docs/superpowers/plans/2026-04-29-belongs-to-list-rename-impl.md\`
- Spec PR (1/2, merged): #1565

## Cascade

- **Schema** (1 file): \`src/questfoundry/models/seed.py\` — \`InitialBeat\` field rename, drop migrator, replace validator.
- **Mutation layer** (1 file): \`src/questfoundry/graph/mutations.py\` — \`_get_path_ids_from_beat()\` reads list directly.
- **Production consumers** (~4 files): \`pipeline/stages/seed.py\`, \`agents/serialize.py\`, \`graph/seed_validation.py\`, \`pipeline/stages/brainstorm.py\` — read \`belongs_to[0]\` / \`belongs_to[1]\`.
- **Prompt templates** (5 files): \`serialize_seed_sections.yaml\`, \`summarize_seed_sections.yaml\`, \`serialize_seed.yaml\`, \`summarize_seed.yaml\`, \`discuss_seed.yaml\` — schema descriptions, GOOD/BAD examples, FINAL CHECK assertions.
- **Tests** (~15-20 files): every \`InitialBeat\` fixture / assertion in unit + integration tests + \`tests/fixtures/\`.

Closes #1564.

## Test plan

- [x] \`uv run pytest tests/unit/\` — all green
- [x] \`uv run mypy src/\` — clean
- [x] \`uv run ruff check src/ tests/\` — clean
- [x] \`rg \"also_belongs_to\" src/ tests/ prompts/\` — zero hits
- [x] \`rg \"InitialBeat\\(.*path_id=\" tests/\` — zero hits

## End-to-end smoke (optional, requires LLM access)

- [ ] Re-run \`projects/murder6\` against qwen3:4b — confirm SEED \`shared_beats\` succeeds without retry. The list shape is unambiguous to small models, structurally eliminating the murder6 failure mode.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Wait for bot review loop**

Per CLAUDE.md, address findings in-PR. Gemini may be at quota; claude-review approval is sufficient. Flip ready when claude-review LGTM + all CI green + bot's review body explicitly says "Ready to merge."

---

## Self-Review Notes

Spec coverage check (against `docs/superpowers/specs/2026-04-29-belongs-to-list-rename-design.md`):

- ✅ Spec § Schema → Task 2 (InitialBeat field rewrite, validator replacement, migrator drop)
- ✅ Spec § Code-side ripples → Task 3 (mutation layer, production consumers)
- ✅ Spec § Code-side ripples — prompts → Task 4 (5 SEED templates)
- ✅ Spec § Code-side ripples — tests → Task 5 (~15-20 test files)
- ✅ Spec § Verification → Task 6

No placeholders. No "implement later" / "similar to Task N" / TBD. Type consistency: `belongs_to: list[str]`, `len(self.belongs_to)`, `belongs_to[0]`, `belongs_to[1]` referenced consistently across tasks.

The plan structures Tasks 2-5 as separate commits but acknowledges they form a single atomic PR. If pre-commit blocks intermediate commits because mypy fails on a half-renamed file, the implementer is told to bundle into one commit.
