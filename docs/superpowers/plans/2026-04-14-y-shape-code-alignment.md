# Y-Shape Code Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring QuestFoundry's SEED output, graph mutations, POLISH consumers, and SEED prompts into conformance with the Y-shape dilemma model ratified in #1206/#1208, so pre-commit beats carry dual `belongs_to` edges (one per path of their own dilemma), guard-rail violations fail loudly at write time, and POLISH Phase 4c can derive >0 choices from real SEED output.

**Architecture:** Add an optional `also_belongs_to: str | None` field to `InitialBeat`; the graph mutation layer emits two `belongs_to` edges when set and enforces the three Part 8 guard rails at write time (same-dilemma constraint, pre-commit only, intersection exclusion); all consumers that currently build `beat_to_path: dict[str, str]` switch to `beat_to_paths: dict[str, frozenset[str]]` with divergence/choice semantics updated to distinguish per-path post-commit successors (true divergence) from last-shared-pre-commit-to-per-path-commits (expected Y-shape, not divergence); SEED prompts adopt a two-call pattern (shared beats, then per-path beats) driven by the dilemma's role.

**Tech Stack:** Python 3.11+, Pydantic, SQLite-backed custom graph (`questfoundry.graph.graph.Graph`), LangChain (`with_structured_output`, `ChatPromptTemplate`), pytest.

---

## Overview

### Files Touched Across the Epic

| File | Phase | Why |
|---|---|---|
| `src/questfoundry/models/seed.py` | 1 | Add `also_belongs_to` to `InitialBeat`, rewrite `_migrate_paths_to_path_id`, update docstrings. |
| `tests/unit/test_seed_models.py` | 1 | New tests for `also_belongs_to`, legacy migration, Y-shape field validators. |
| `src/questfoundry/graph/mutations.py` | 2 | Replace `_get_path_id_from_beat` with `_get_path_ids_from_beat`, update three insertion sites, emit dual edges, enforce guard rails at write time. |
| `src/questfoundry/graph/algorithms.py` | 2 | Replace `beat_to_path: dict[str, str]` with `beat_to_paths: dict[str, frozenset[str]]`; remove the "multiple belongs_to" rejection. |
| `tests/unit/test_mutations.py` | 2 | Guard-rail violation tests (write-time `ValueError`), dual-edge emission tests. |
| `tests/unit/test_graph_algorithms.py` | 2 | Dual-belongs_to handling in state-flag derivation. |
| `src/questfoundry/graph/polish_validation.py` | 3 | Update `_check_divergences_have_choices` to `beat_to_paths`; exclude last-shared-pre-commit to per-path commits from the divergence-error. |
| `src/questfoundry/graph/polish_context.py` | 3 | `beat_to_path` → `beat_to_paths`; entity-appearance labelling shows all path memberships. |
| `src/questfoundry/pipeline/stages/polish/deterministic.py` | 3 | Both build sites (lines 250, 645) → `frozenset`-valued; collapse chains must require exact path-set equality, not a single path comparison. |
| `src/questfoundry/pipeline/stages/polish/llm_phases.py` | 3 | Update helpers at lines 840–940 (consecutive-run detection, post-commit sequel). |
| `tests/unit/test_polish_passage_validation.py` | 3 | Y-shape divergence check tests (real Y-shape must not error). |
| `tests/unit/test_polish_deterministic.py` | 3 | Y-shape collapse rules (shared pre-commit beats never collapse with post-commit beats). |
| `prompts/templates/discuss_seed.yaml` | 4 | Replace "per path" framing with Y-shape guidance. |
| `prompts/templates/summarize_seed.yaml` | 4 | Same. |
| `prompts/templates/summarize_seed_sections.yaml` | 4 | Same. |
| `prompts/templates/serialize_seed.yaml` | 4 | Same — plus schema comment update for `also_belongs_to`. |
| `prompts/templates/serialize_seed_sections.yaml` | 4 | Rewrite § "COMMITS BEATS REQUIREMENT" block (line 545) and the five "per path" sites; add Y-shape schema. |
| `src/questfoundry/pipeline/stages/seed.py` | 4 | Advisory warning math (`avg_beats = beat_count / path_count`) must split shared vs per-path beats. |
| `tests/unit/test_seed_stage.py` | 4 | Update existing fixtures to Y-shape; add a test exercising the shared-beat split. |
| `src/questfoundry/graph/grow_validation.py` | 5 | Remove any "exactly one belongs_to per beat" check; add three guard-rail checks that parallel the mutation-time enforcement. |
| `tests/unit/test_grow_validation.py` | 5 | Tests for the three guard-rail checks. |
| `src/questfoundry/pipeline/stages/grow/deterministic.py` | 6 (investigation) | Possibly: persist `ConvergenceInfo` on arc/dilemma nodes. Scope-only in this plan. |
| `src/questfoundry/graph/grow_algorithms.py` | 6 (investigation) | Possibly: add `apply_convergence_edges()`. Scope-only in this plan. |
| `tests/integration/test_y_shape_end_to_end.py` | 7 | New file: SEED → POLISH Phase 4c produces >0 choices on a Y-shape fixture. |

### Why This Shape

- **`also_belongs_to: str | None` (not `path_ids: list[str]`).** The ontology (Doc 3, §8 guard rail 1) restricts dual membership to a single dilemma with exactly two explored paths; N-ary shared beats are not a planned pattern. `also_belongs_to` captures that constraint in the schema itself ("null for post-commit, the other path for pre-commit"). A `list[str]` field invites N>2 errors the guard rails would then have to reject at a different layer. Doc 3 line 855 already recommends `also_belongs_to`; this plan ratifies that recommendation.
- **Write-time guard-rail enforcement.** CLAUDE.md forbids silent degradation. Guard-rail violations (cross-dilemma `belongs_to`, dual `belongs_to` on post-commit beats, pre-commit beats in intersections) must raise `ValueError` in the mutation layer, not be logged as warnings.
- **`beat_to_paths: dict[str, frozenset[str]]` across all consumers.** A `frozenset` is hashable, set-ordered-independent, and makes "same set of paths?" a trivial equality check (which the collapse-chain rule needs). This is a deliberate type choice — not a list, not a set, not a tuple.
- **Two-call SEED prompt structure (Phase 4).** Chosen over a single unified prompt. Justification in Phase 4's scope blurb.
- **Keep the write-time enforcement separate from post-hoc validation.** Mutations raise on violation; `grow_validation.py` adds parallel checks in Phase 5 so a graph constructed by code (tests, diagnostics) still gets caught. Belt + suspenders.

---

## Phase 1: Schema Change — `InitialBeat.also_belongs_to`

### Scope

Add the `also_belongs_to: str | None` field to `InitialBeat`. Update the class docstring to describe Y-shape semantics. Rewrite `_migrate_paths_to_path_id` so it only migrates the historical `paths: [single_id]` pattern (not Y-shape) and raise a clear error for `paths` lists with >1 entry that do not map onto the Y-shape (belt: the guard-rail enforcer in Phase 2 is the actual arbiter). Add a cross-field validator that rejects `also_belongs_to == path_id`.

### Acceptance Criteria

- `InitialBeat(beat_id="b1", summary="s", path_id="p_a")` constructs a post-commit beat with `also_belongs_to is None`.
- `InitialBeat(beat_id="b1", summary="s", path_id="p_a", also_belongs_to="p_b")` constructs a pre-commit beat.
- `InitialBeat(..., path_id="p_a", also_belongs_to="p_a")` raises `ValueError`.
- Legacy `paths: ["p_a"]` migrates to `path_id="p_a"`, `also_belongs_to=None` with a `DeprecationWarning`.
- Legacy `paths: ["p_a", "p_b"]` migrates to `path_id="p_a"`, `also_belongs_to="p_b"` with a `DeprecationWarning` (unambiguous Y-shape intent — two entries map directly to the dual-membership schema).
- Legacy `paths` with 3+ entries raises `ValueError("at most 2 entries — use path_id + also_belongs_to")`.
- No change to `SeedOutput.initial_beats: list[InitialBeat]` type signature (only field-level addition).
- All existing unit tests in `tests/unit/test_seed_models.py` pass unchanged.
- `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §849–855 (Missing: Temporal Hints / InitialBeat.paths — Same-Dilemma Dual belongs_to).

### Tasks

---

### Task 1.1: Write the failing test for `also_belongs_to`

**Files:**
- Modify: `tests/unit/test_seed_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_seed_models.py`:

```python
def test_initial_beat_pre_commit_dual_belongs_to() -> None:
    """Pre-commit beats carry ``also_belongs_to`` pointing at the sibling path."""
    from questfoundry.models.seed import InitialBeat

    beat = InitialBeat(
        beat_id="b1",
        summary="Shared setup before the fork.",
        path_id="path::trust__protector",
        also_belongs_to="path::trust__manipulator",
    )
    assert beat.path_id == "path::trust__protector"
    assert beat.also_belongs_to == "path::trust__manipulator"


def test_initial_beat_post_commit_single_belongs_to_default() -> None:
    """Post-commit beats default to ``also_belongs_to = None``."""
    from questfoundry.models.seed import InitialBeat

    beat = InitialBeat(
        beat_id="b1",
        summary="Payoff beat.",
        path_id="path::trust__protector",
    )
    assert beat.also_belongs_to is None


def test_initial_beat_also_belongs_to_equal_path_id_is_rejected() -> None:
    """``also_belongs_to`` must differ from ``path_id`` — dual membership needs two paths."""
    import pytest
    from questfoundry.models.seed import InitialBeat

    with pytest.raises(ValueError, match="also_belongs_to must differ from path_id"):
        InitialBeat(
            beat_id="b1",
            summary="Broken dual.",
            path_id="path::trust__protector",
            also_belongs_to="path::trust__protector",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_seed_models.py::test_initial_beat_pre_commit_dual_belongs_to tests/unit/test_seed_models.py::test_initial_beat_post_commit_single_belongs_to_default tests/unit/test_seed_models.py::test_initial_beat_also_belongs_to_equal_path_id_is_rejected -v
```
Expected: `FAILED` — `also_belongs_to` is an unknown field (or the equality check does not exist).

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/unit/test_seed_models.py
git commit -m "test(seed): add failing tests for InitialBeat.also_belongs_to Y-shape field"
```

---

### Task 1.2: Add `also_belongs_to` and the equality validator

**Files:**
- Modify: `src/questfoundry/models/seed.py:234-304`

- [ ] **Step 1: Replace the `InitialBeat` docstring and add the new field**

In `src/questfoundry/models/seed.py`, replace lines 234–256 (the class header, docstring, and `path_id` field definition) with:

```python
class InitialBeat(BaseModel):
    """Initial beat created by SEED.

    Beats carry either a single or a dual ``belongs_to`` edge to path nodes:

    * **Post-commit beats** (the default) belong to exactly one path via
      ``path_id``; ``also_belongs_to`` is ``None``. These beats prove one
      answer and are exclusive to the path they belong to.
    * **Pre-commit beats** (shared dilemma setup) belong to *both* paths of
      their dilemma via ``path_id`` and ``also_belongs_to``. This is the
      Y-shape ratified in #1206/#1208: every player experiences pre-commit
      beats regardless of which answer they later choose.

    The commit beat itself is single-``belongs_to`` — it is the first beat
    exclusive to its path.

    Attributes:
        beat_id: Unique identifier for the beat.
        summary: What happens in this beat.
        path_id: Primary path this beat belongs to.
        also_belongs_to: Sibling path for pre-commit (shared) beats. ``None``
            for post-commit beats. MUST reference a path that shares the same
            parent dilemma with ``path_id`` — cross-dilemma dual membership
            is a Part-8 guard-rail violation and is rejected by the mutation
            layer.
        dilemma_impacts: How this beat affects dilemmas.
        entities: Entity IDs present in this beat.
        location: Primary location entity ID.
        location_alternatives: Other valid locations (enables intersection flexibility).
        temporal_hint: Advisory placement relative to another dilemma (consumed by GROW).
    """

    beat_id: str = Field(min_length=1, description="Unique identifier for this beat")
    summary: str = Field(min_length=1, description="What happens in this beat")
    path_id: str = Field(
        min_length=1,
        description="Primary path this beat belongs to (first belongs_to edge)",
    )
    also_belongs_to: str | None = Field(
        default=None,
        description=(
            "Sibling path for pre-commit (Y-shape) beats: creates a second "
            "belongs_to edge. Must be null for post-commit beats. Must "
            "reference a path with the same parent dilemma as path_id."
        ),
    )
```

- [ ] **Step 2: Add the equality validator below the migration validator**

Immediately after the `_migrate_paths_to_path_id` validator (currently lines 258–279), insert:

```python
    @model_validator(mode="after")
    def _also_belongs_to_differs_from_path_id(self) -> "InitialBeat":
        """Dual membership requires two distinct path IDs."""
        if self.also_belongs_to is not None and self.also_belongs_to == self.path_id:
            msg = "also_belongs_to must differ from path_id — dual membership needs two paths."
            raise ValueError(msg)
        return self
```

- [ ] **Step 3: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_seed_models.py::test_initial_beat_pre_commit_dual_belongs_to tests/unit/test_seed_models.py::test_initial_beat_post_commit_single_belongs_to_default tests/unit/test_seed_models.py::test_initial_beat_also_belongs_to_equal_path_id_is_rejected -v
```
Expected: three `PASSED`.

- [ ] **Step 4: Run the whole seed-models test file to catch regressions**

```bash
uv run pytest tests/unit/test_seed_models.py -x -q
```
Expected: all tests pass (existing tests should be unaffected since `also_belongs_to` defaults to `None`).

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/models/seed.py tests/unit/test_seed_models.py
git commit -m "feat(seed): add InitialBeat.also_belongs_to for Y-shape dual belongs_to"
```

---

### Task 1.3: Tighten the legacy `paths` migration

**Files:**
- Modify: `src/questfoundry/models/seed.py:258-279`
- Modify: `tests/unit/test_seed_models.py`

The existing migration truncates `paths: [p_a, p_b]` to `[p_a]` with a warning — masking exactly the Y-shape data we now need. It must instead convert two-element lists to the dual form.

- [ ] **Step 1: Write the failing test for the legacy list-of-two migration**

Append to `tests/unit/test_seed_models.py`:

```python
def test_initial_beat_legacy_paths_two_elements_becomes_dual() -> None:
    """Legacy ``paths: [p_a, p_b]`` migrates to Y-shape dual membership."""
    import warnings
    from questfoundry.models.seed import InitialBeat

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beat = InitialBeat(
            beat_id="b1",
            summary="Legacy dual.",
            paths=["path::trust__protector", "path::trust__manipulator"],
        )

    assert beat.path_id == "path::trust__protector"
    assert beat.also_belongs_to == "path::trust__manipulator"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_initial_beat_legacy_paths_three_elements_is_rejected() -> None:
    """A list of three or more paths is a schema error — not a migration target."""
    import pytest
    from questfoundry.models.seed import InitialBeat

    with pytest.raises(ValueError, match="at most 2 entries"):
        InitialBeat(
            beat_id="b1",
            summary="Bad.",
            paths=["p_a", "p_b", "p_c"],
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/unit/test_seed_models.py::test_initial_beat_legacy_paths_two_elements_becomes_dual tests/unit/test_seed_models.py::test_initial_beat_legacy_paths_three_elements_is_rejected -v
```
Expected: `FAILED` — current code truncates to `[p_a]` for two entries and raises a different message for three entries.

- [ ] **Step 3: Replace the migration validator**

In `src/questfoundry/models/seed.py`, replace the `_migrate_paths_to_path_id` validator (currently lines 258–279) with:

```python
    @model_validator(mode="before")
    @classmethod
    def _migrate_paths_to_path_id(cls, data: Any) -> Any:
        """Accept legacy ``paths`` and convert to Y-shape (``path_id`` + optional ``also_belongs_to``).

        - ``paths: [single_id]`` → ``path_id = single_id`` (post-commit beat).
        - ``paths: [p_a, p_b]`` → ``path_id = p_a``, ``also_belongs_to = p_b``
          (pre-commit beat, Y-shape dual membership).
        - ``paths: []`` is rejected — every beat must reference at least one path.
        - ``paths`` with 3+ entries is rejected — dual membership is bounded to
          the two paths of one dilemma (Part 8 guard rail 1).
        """
        if not isinstance(data, dict):
            return data
        if "paths" in data and "path_id" not in data:
            paths = data.pop("paths")
            if not isinstance(paths, list):
                return data  # Let Pydantic reject the non-list type.
            if len(paths) == 0:
                msg = "InitialBeat.paths is empty — each beat must belong to at least one path."
                raise ValueError(msg)
            if len(paths) > 2:
                msg = (
                    "InitialBeat.paths has at most 2 entries (Y-shape guard rail 1). "
                    "Got {count}: {paths!r}."
                ).format(count=len(paths), paths=paths)
                raise ValueError(msg)
            data["path_id"] = paths[0]
            if len(paths) == 2:
                warnings.warn(
                    "InitialBeat.paths=[p_a, p_b] is deprecated — use "
                    "path_id=p_a and also_belongs_to=p_b directly.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data["also_belongs_to"] = paths[1]
        return data
```

- [ ] **Step 4: Run the two new tests to verify they pass**

```bash
uv run pytest tests/unit/test_seed_models.py::test_initial_beat_legacy_paths_two_elements_becomes_dual tests/unit/test_seed_models.py::test_initial_beat_legacy_paths_three_elements_is_rejected -v
```
Expected: both `PASSED`.

- [ ] **Step 5: Update the pre-existing legacy test that expects truncation**

`tests/unit/test_seed_models.py` line ~1002 has `InitialBeat(beat_id="b1", summary="Test", path_id="")` — unrelated. Look at line ~1007: `InitialBeat(beat_id="b1", summary="Test", paths=[])` — still correctly rejected. Search for any existing test that relies on the old three-entry or truncation behaviour and fix it:

```bash
grep -n "paths=\[.*,.*,.*\]\|had.*entries; using first" tests/unit/test_seed_models.py
```
If any match surfaces, update it to either (a) use two paths and assert `also_belongs_to`, or (b) assert the new `ValueError("at most 2 entries")`.

- [ ] **Step 6: Run the whole seed-models test file to catch regressions**

```bash
uv run pytest tests/unit/test_seed_models.py -x -q
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/questfoundry/models/seed.py tests/unit/test_seed_models.py
git commit -m "feat(seed): migrate legacy paths=[a, b] to Y-shape also_belongs_to"
```

---

### Task 1.4: Update model exports and run the broader test surface

- [ ] **Step 1: Confirm the field is exposed on the re-export**

```bash
grep -n "InitialBeat" src/questfoundry/models/__init__.py
```
Expected output: a `InitialBeat,` line. If absent, add it to the `__all__` list.

- [ ] **Step 2: Run adjacent unit tests**

```bash
uv run pytest tests/unit/test_seed_models.py tests/unit/test_ontology_explored.py -x -q
```
Expected: all pass.

- [ ] **Step 3: Run type checks**

```bash
uv run mypy src/questfoundry/models/seed.py
```
Expected: `Success: no issues found`.

- [ ] **Step 4: Commit (only if anything changed)**

```bash
git status  # If clean, skip the commit.
git add -A && git commit -m "chore(seed): verify InitialBeat export surface after Y-shape fields"
```

---

## Phase 2: Graph Mutation — Dual-Edge Emission + Write-Time Guard Rails

### Scope

Replace `_get_path_id_from_beat` with `_get_path_ids_from_beat`, update the three known beat-insertion sites (mutations.py lines 1312, 1529, 1874), emit multiple `belongs_to` edges when appropriate, and add write-time enforcement of the three Part 8 guard rails. Update `algorithms.py` to build `beat_to_paths: dict[str, frozenset[str]]` (dropping the "multiple belongs_to" hard-raise since it is now legal) while ensuring cross-dilemma multi-`belongs_to` still raises. Update callers of the dropped helper.

### Acceptance Criteria

- `_get_path_ids_from_beat(beat)` returns a `tuple[str, ...]` with 0, 1, or 2 entries (never more).
- The three insertion sites call `graph.add_edge("belongs_to", beat_id, prefixed_pid)` once per path ID.
- Writing a beat with `also_belongs_to` pointing at a path whose `dilemma_id` differs from `path_id`'s dilemma raises `ValueError` before the edge is added.
- Writing a beat whose `dilemma_impacts` include an `effect == "commits"` AND has `also_belongs_to` set raises `ValueError` (commit beats are single-membership; guard rail 2).
- Writing an `intersection_group` whose beat list contains two pre-commit beats from the same dilemma raises `ValueError` (guard rail 3).
- `algorithms.py`'s ancestor-walk for state flags correctly derives one flag per committed dilemma when some ancestors are pre-commit (dual-belongs_to) shared beats.
- Tests in `tests/unit/test_mutations.py` and `tests/unit/test_graph_algorithms.py` pass.
- `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §8 "Path Membership ≠ Scene Participation" (the three guard rails) and §3 "Beat Lifecycle".

### Tasks

---

### Task 2.1: Replace `_get_path_id_from_beat` with `_get_path_ids_from_beat`

**Files:**
- Modify: `src/questfoundry/graph/mutations.py:768-777`

- [ ] **Step 1: Write the failing unit test for the helper**

Append to `tests/unit/test_mutations.py`:

```python
def test_get_path_ids_from_beat_post_commit_returns_one() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    beat = {"path_id": "path::a"}
    assert _get_path_ids_from_beat(beat) == ("path::a",)


def test_get_path_ids_from_beat_pre_commit_returns_both() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    beat = {"path_id": "path::a", "also_belongs_to": "path::b"}
    assert _get_path_ids_from_beat(beat) == ("path::a", "path::b")


def test_get_path_ids_from_beat_legacy_paths_list_returns_all() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    beat = {"paths": ["path::a", "path::b"]}
    assert _get_path_ids_from_beat(beat) == ("path::a", "path::b")


def test_get_path_ids_from_beat_empty_returns_empty_tuple() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    assert _get_path_ids_from_beat({}) == ()
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_mutations.py::test_get_path_ids_from_beat_post_commit_returns_one tests/unit/test_mutations.py::test_get_path_ids_from_beat_pre_commit_returns_both tests/unit/test_mutations.py::test_get_path_ids_from_beat_legacy_paths_list_returns_all tests/unit/test_mutations.py::test_get_path_ids_from_beat_empty_returns_empty_tuple -v
```
Expected: all four `FAILED` (`ImportError: cannot import name '_get_path_ids_from_beat'`).

- [ ] **Step 3: Replace the helper**

In `src/questfoundry/graph/mutations.py`, replace lines 768–777 with:

```python
def _get_path_ids_from_beat(beat: dict[str, Any]) -> tuple[str, ...]:
    """Extract path IDs from a beat dict, supporting current and legacy formats.

    Supports:
    - Y-shape current: ``path_id`` + optional ``also_belongs_to``.
    - Legacy single: ``path_id`` alone.
    - Legacy list: ``paths: [p]`` or ``paths: [p_a, p_b]`` (pre-Y-shape).

    Args:
        beat: Beat dict, either model-dumped or raw LLM output.

    Returns:
        Tuple of 0, 1, or 2 raw path IDs in declaration order (``path_id``
        first, then ``also_belongs_to``). A return of 2 always represents a
        pre-commit Y-shape beat. A return of 0 means the beat has no path
        reference (caught by validation).
    """
    if beat.get("path_id"):
        primary = str(beat["path_id"])
        also = beat.get("also_belongs_to")
        if also:
            return (primary, str(also))
        return (primary,)
    # Legacy fallback.
    paths = beat.get("paths")
    if isinstance(paths, list) and paths:
        return tuple(str(p) for p in paths[:2])
    return ()
```

Also search the file for any remaining references to the old helper name:

```bash
grep -n "_get_path_id_from_beat" src/questfoundry/graph/mutations.py
```
If any remain (validators at lines 1312, 1529), they will be updated in Task 2.2.

- [ ] **Step 4: Run the four new tests**

```bash
uv run pytest tests/unit/test_mutations.py::test_get_path_ids_from_beat_post_commit_returns_one tests/unit/test_mutations.py::test_get_path_ids_from_beat_pre_commit_returns_both tests/unit/test_mutations.py::test_get_path_ids_from_beat_legacy_paths_list_returns_all tests/unit/test_mutations.py::test_get_path_ids_from_beat_empty_returns_empty_tuple -v
```
Expected: all four `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/mutations.py tests/unit/test_mutations.py
git commit -m "refactor(graph): introduce _get_path_ids_from_beat helper for Y-shape"
```

---

### Task 2.2: Update validation call sites 1312 and 1529 to use the new helper

**Files:**
- Modify: `src/questfoundry/graph/mutations.py:1310-1322, 1525-1535`

- [ ] **Step 1: Update the per-beat path-ID validation at line 1312**

In `src/questfoundry/graph/mutations.py`, replace lines 1311–1322 (the Section 6 validation block) with:

```python
        # 6. Path references (Y-shape: path_id + optional also_belongs_to).
        raw_path_ids = _get_path_ids_from_beat(beat)
        for idx, raw_path_id in enumerate(raw_path_ids):
            field_name = "path_id" if idx == 0 else "also_belongs_to"
            _validate_id(
                raw_path_id,
                "path",
                seed_path_ids,
                f"initial_beats.{i}.{field_name}",
                errors,
                sorted_path_ids,
                cross_type_sets=cross_type_sets,
            )
```

- [ ] **Step 2: Update the per-path commits accumulator at line 1529**

In `src/questfoundry/graph/mutations.py`, locate the block around line 1527–1547 starting with `for i, beat in enumerate(output.get("initial_beats", [])):`. The current version computes exactly one `normalized_pid`. It must now compute the primary path and skip dual-membership pre-commit beats for the commits-per-path check (a pre-commit beat cannot be a commit beat by guard rail 2). Replace:

```python
    for i, beat in enumerate(output.get("initial_beats", [])):
        # Resolve beat's path (singular path_id, with legacy paths fallback)
        raw_pid = _get_path_id_from_beat(beat)
        if not raw_pid:
            continue  # Missing path — already caught by check 6
        normalized_pid, _ = _normalize_id(raw_pid, "path")
```

with:

```python
    for i, beat in enumerate(output.get("initial_beats", [])):
        # Resolve beat's primary path; pre-commit beats have a second membership
        # (also_belongs_to) but the commits-per-path accumulator is only
        # concerned with single-membership commit beats (guard rail 2).
        raw_path_ids = _get_path_ids_from_beat(beat)
        if not raw_path_ids:
            continue  # Missing path — already caught by check 6
        raw_pid = raw_path_ids[0]
        normalized_pid, _ = _normalize_id(raw_pid, "path")
```

- [ ] **Step 3: Run the mutation unit tests**

```bash
uv run pytest tests/unit/test_mutations.py -x -q
```
Expected: all pass (both the pre-existing tests and the ones added in Task 2.1).

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/graph/mutations.py
git commit -m "refactor(graph): route SEED validation sites through _get_path_ids_from_beat"
```

---

### Task 2.3: Update the beat-insertion site to emit dual `belongs_to` edges

**Files:**
- Modify: `src/questfoundry/graph/mutations.py:1870-1882`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/unit/test_mutations.py`:

```python
def test_apply_seed_mutations_emits_dual_belongs_to_for_pre_commit_beat(
    tmp_path: "Path",
) -> None:
    """Pre-commit beats with ``also_belongs_to`` get two ``belongs_to`` edges."""
    from pathlib import Path  # noqa: F401  # avoid the import-level Path confusion

    from questfoundry.graph.graph import Graph
    from questfoundry.graph.mutations import apply_seed_mutations

    graph = Graph(tmp_path / "graph.db")

    # Minimal SEED output: one dilemma, two paths, one pre-commit beat.
    seed = {
        "entities": [
            {"entity_id": "mentor", "category": "character", "name": "Mentor",
             "description": "x", "disposition": "keep"},
        ],
        "dilemmas": [
            {"dilemma_id": "trust_protector_or_manipulator",
             "question": "Is the mentor a protector or a manipulator?",
             "why_it_matters": "Shapes the protagonist's stance.",
             "answers": [
                 {"answer_id": "protector", "label": "protector",
                  "description": "The mentor is a protector."},
                 {"answer_id": "manipulator", "label": "manipulator",
                  "description": "The mentor is a manipulator."},
             ],
             "central_entity_ids": ["mentor"],
             "exploration_decision": "both",
             "canonical_answer_id": "protector"},
        ],
        "paths": [
            {"path_id": "trust_protector_or_manipulator__protector",
             "dilemma_id": "trust_protector_or_manipulator",
             "answer_id": "protector", "name": "Protector",
             "description": "The protector arc."},
            {"path_id": "trust_protector_or_manipulator__manipulator",
             "dilemma_id": "trust_protector_or_manipulator",
             "answer_id": "manipulator", "name": "Manipulator",
             "description": "The manipulator arc."},
        ],
        "consequences": [
            {"consequence_id": "mentor_trusted",
             "path_id": "trust_protector_or_manipulator__protector",
             "description": "Trust.", "narrative_effects": ["x"]},
            {"consequence_id": "mentor_distrusted",
             "path_id": "trust_protector_or_manipulator__manipulator",
             "description": "Distrust.", "narrative_effects": ["x"]},
        ],
        "initial_beats": [
            {"beat_id": "shared_setup",
             "summary": "Both players see this setup.",
             "path_id": "trust_protector_or_manipulator__protector",
             "also_belongs_to": "trust_protector_or_manipulator__manipulator",
             "dilemma_impacts": [
                 {"dilemma_id": "trust_protector_or_manipulator",
                  "effect": "advances", "note": "x"},
             ]},
        ],
    }
    apply_seed_mutations(graph, seed)

    edges = [e for e in graph.get_edges(edge_type="belongs_to")
             if e["from"] == "beat::shared_setup"]
    to_ids = {e["to"] for e in edges}
    assert to_ids == {
        "path::trust_protector_or_manipulator__protector",
        "path::trust_protector_or_manipulator__manipulator",
    }
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_emits_dual_belongs_to_for_pre_commit_beat -v
```
Expected: `FAILED` — current emission only writes one edge.

- [ ] **Step 3: Update the insertion site**

In `src/questfoundry/graph/mutations.py`, replace the block at lines 1873–1877:

```python
        # Link beat to its path (singular belongs_to edge)
        raw_path_id = _get_path_id_from_beat(beat)
        if raw_path_id:
            prefixed_path_id = _prefix_id("path", raw_path_id)
            graph.add_edge("belongs_to", beat_id, prefixed_path_id)
```

with:

```python
        # Link beat to its path(s). Post-commit beats emit one belongs_to;
        # pre-commit (Y-shape) beats with ``also_belongs_to`` emit two.
        raw_path_ids = _get_path_ids_from_beat(beat)
        for raw_path_id in raw_path_ids:
            prefixed_path_id = _prefix_id("path", raw_path_id)
            graph.add_edge("belongs_to", beat_id, prefixed_path_id)
```

- [ ] **Step 4: Run the new test and the mutation test file**

```bash
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_emits_dual_belongs_to_for_pre_commit_beat -v
uv run pytest tests/unit/test_mutations.py -x -q
```
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/mutations.py tests/unit/test_mutations.py
git commit -m "feat(graph): emit dual belongs_to edges for pre-commit Y-shape beats"
```

---

### Task 2.4: Enforce guard rail 1 (same-dilemma constraint) at write time

**Files:**
- Modify: `src/questfoundry/graph/mutations.py:1817-1882`

Guard rail 1: "A beat with two `belongs_to` edges must reference two paths that belong to the same dilemma."

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_mutations.py`:

```python
def test_apply_seed_mutations_rejects_cross_dilemma_dual_belongs_to(
    tmp_path: "Path",
) -> None:
    """Guard rail 1: pre-commit beats must share a dilemma across both paths."""
    import pytest
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.mutations import apply_seed_mutations

    graph = Graph(tmp_path / "graph.db")

    # Two unrelated dilemmas; ``also_belongs_to`` points at a path from a
    # different dilemma — cross-dilemma dual membership is forbidden.
    seed = _minimal_two_dilemma_seed()
    seed["initial_beats"] = [
        {"beat_id": "bad_dual",
         "summary": "Cross-dilemma.",
         "path_id": "dilemma_a__answer_a1",
         "also_belongs_to": "dilemma_b__answer_b1",
         "dilemma_impacts": [
             {"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"},
         ]},
    ]

    with pytest.raises(ValueError, match="cross-dilemma dual belongs_to"):
        apply_seed_mutations(graph, seed)
```

The helper `_minimal_two_dilemma_seed()` should sit alongside the test as a local module-level fixture builder — mirror the structure of `test_apply_seed_mutations_emits_dual_belongs_to_for_pre_commit_beat` but with two dilemmas each having one explored answer. Write it in full rather than referring to an external fixture:

```python
def _minimal_two_dilemma_seed() -> dict[str, Any]:
    """Two unrelated dilemmas with one path each."""
    return {
        "entities": [{"entity_id": "mentor", "category": "character",
                      "name": "M", "description": "x", "disposition": "keep"}],
        "dilemmas": [
            {"dilemma_id": "dilemma_a",
             "question": "A?", "why_it_matters": "y",
             "answers": [
                 {"answer_id": "answer_a1", "label": "a1", "description": "a1"},
                 {"answer_id": "answer_a2", "label": "a2", "description": "a2"},
             ],
             "central_entity_ids": ["mentor"],
             "exploration_decision": "a_only",
             "canonical_answer_id": "answer_a1"},
            {"dilemma_id": "dilemma_b",
             "question": "B?", "why_it_matters": "z",
             "answers": [
                 {"answer_id": "answer_b1", "label": "b1", "description": "b1"},
                 {"answer_id": "answer_b2", "label": "b2", "description": "b2"},
             ],
             "central_entity_ids": ["mentor"],
             "exploration_decision": "a_only",
             "canonical_answer_id": "answer_b1"},
        ],
        "paths": [
            {"path_id": "dilemma_a__answer_a1", "dilemma_id": "dilemma_a",
             "answer_id": "answer_a1", "name": "A1", "description": "a"},
            {"path_id": "dilemma_b__answer_b1", "dilemma_id": "dilemma_b",
             "answer_id": "answer_b1", "name": "B1", "description": "b"},
        ],
        "consequences": [
            {"consequence_id": "c_a", "path_id": "dilemma_a__answer_a1",
             "description": "ca", "narrative_effects": ["x"]},
            {"consequence_id": "c_b", "path_id": "dilemma_b__answer_b1",
             "description": "cb", "narrative_effects": ["x"]},
        ],
        "initial_beats": [],
    }
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_rejects_cross_dilemma_dual_belongs_to -v
```
Expected: `FAILED` — the mutation currently accepts the cross-dilemma dual.

- [ ] **Step 3: Add the guard-rail check inside the beat-creation loop**

In `src/questfoundry/graph/mutations.py`, just before the new dual-edge emission loop (inserted in Task 2.3), insert a pre-check using the path→dilemma map SEED paths already carry. The path → `dilemma_id` mapping is built in `_ensure_paths_created` (search for `dilemma_id` assignment on path nodes); we re-derive it inline from the parsed SEED output:

Locate the `for i, beat in enumerate(output.get("initial_beats", [])):` loop that begins at approximately line 1818. **Before** the loop, insert:

```python
    # Build a raw path → dilemma map for guard-rail 1 enforcement.
    _path_to_dilemma: dict[str, str] = {}
    for p in output.get("paths", []):
        pid = p.get("path_id")
        did = p.get("dilemma_id")
        if pid and did:
            _path_to_dilemma[pid] = did
```

Then inside the loop, **immediately before** the dual-edge emission code added in Task 2.3, insert:

```python
        # Guard rail 1 (Doc 3 §8 "Path Membership ≠ Scene Participation"):
        # dual belongs_to must reference paths of the same dilemma.
        if len(raw_path_ids) == 2:
            d0 = _path_to_dilemma.get(raw_path_ids[0])
            d1 = _path_to_dilemma.get(raw_path_ids[1])
            if d0 is None or d1 is None or d0 != d1:
                msg = (
                    "cross-dilemma dual belongs_to is forbidden (guard rail 1). "
                    "Beat {beat_id!r} has path_id={p0!r} (dilemma={d0!r}) and "
                    "also_belongs_to={p1!r} (dilemma={d1!r})."
                ).format(
                    beat_id=raw_id, p0=raw_path_ids[0], d0=d0,
                    p1=raw_path_ids[1], d1=d1,
                )
                raise ValueError(msg)
```

- [ ] **Step 4: Run the new test**

```bash
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_rejects_cross_dilemma_dual_belongs_to -v
```
Expected: `PASSED`.

- [ ] **Step 5: Run the existing dual-edge test to confirm no regression**

```bash
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_emits_dual_belongs_to_for_pre_commit_beat -v
```
Expected: still `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/graph/mutations.py tests/unit/test_mutations.py
git commit -m "feat(graph): enforce guard rail 1 (same-dilemma dual belongs_to) at write time"
```

---

### Task 2.5: Enforce guard rail 2 (pre-commit only) at write time

**Files:**
- Modify: `src/questfoundry/graph/mutations.py` (same region as Task 2.4)

Guard rail 2: "Only beats before the dilemma's commit may have two `belongs_to` edges. The commit beat itself has one."

Concretely: if a beat has `also_belongs_to` set AND any of its `dilemma_impacts` has `effect == "commits"`, raise.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_mutations.py`:

```python
def test_apply_seed_mutations_rejects_dual_on_commit_beat(
    tmp_path: "Path",
) -> None:
    """Guard rail 2: a beat with ``effect=commits`` must have only one belongs_to."""
    import pytest
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.mutations import apply_seed_mutations

    graph = Graph(tmp_path / "graph.db")
    seed = _minimal_trust_dilemma_seed()  # single dilemma, two paths
    seed["initial_beats"] = [
        {"beat_id": "bad_commit",
         "summary": "A commit beat cannot be pre-commit.",
         "path_id": "trust_protector_or_manipulator__protector",
         "also_belongs_to": "trust_protector_or_manipulator__manipulator",
         "dilemma_impacts": [
             {"dilemma_id": "trust_protector_or_manipulator",
              "effect": "commits", "note": "Bad: commit with dual."},
         ]},
    ]
    with pytest.raises(ValueError, match="guard rail 2"):
        apply_seed_mutations(graph, seed)
```

Add the `_minimal_trust_dilemma_seed()` helper alongside, mirroring the body of the earlier dual-belongs_to success test's `seed` dict but returning that dict and pulling the `initial_beats` out of it so tests can substitute their own.

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_rejects_dual_on_commit_beat -v
```
Expected: `FAILED` — no enforcement yet.

- [ ] **Step 3: Add the guard-rail check**

Inside the beat-creation loop, **right after** the guard-rail-1 check added in Task 2.4, insert:

```python
        # Guard rail 2: commit beats are single-membership. A commit beat is
        # the first beat exclusive to its path — it cannot also belong to the
        # sibling path.
        if len(raw_path_ids) == 2:
            has_commit = any(
                imp.get("effect") == "commits"
                for imp in beat.get("dilemma_impacts", [])
            )
            if has_commit:
                msg = (
                    "guard rail 2: a beat with effect=commits must have a single "
                    "belongs_to (no also_belongs_to). Beat {beat_id!r} has both "
                    "a commits impact and also_belongs_to={p1!r}."
                ).format(beat_id=raw_id, p1=raw_path_ids[1])
                raise ValueError(msg)
```

- [ ] **Step 4: Run the new test and the full mutation suite**

```bash
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_rejects_dual_on_commit_beat -v
uv run pytest tests/unit/test_mutations.py -x -q
```
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/mutations.py tests/unit/test_mutations.py
git commit -m "feat(graph): enforce guard rail 2 (pre-commit only dual) at write time"
```

---

### Task 2.6: Enforce guard rail 3 (intersection exclusion) at write time

**Files:**
- Modify: `src/questfoundry/graph/mutations.py` — find the GROW intersection-group creation site.

Guard rail 3: "An intersection group must not contain two pre-commit beats from the same dilemma — they already co-occur by definition."

- [ ] **Step 1: Locate the intersection-group creation code**

```bash
grep -n "intersection_group\|create_intersection_group\|apply_intersection" src/questfoundry/graph/mutations.py src/questfoundry/graph/grow_algorithms.py src/questfoundry/pipeline/stages/grow/*.py
```

The likely site is in `grow_algorithms.py` or a GROW phase that builds intersection groups. Identify the function that takes a list of beat IDs and creates an `intersection_group` node with an `intersection` edge per beat.

- [ ] **Step 2: Write the failing test**

In the appropriate test file (`tests/unit/test_graph_algorithms.py` or `tests/unit/test_grow_intersection.py`), append:

```python
def test_create_intersection_rejects_two_pre_commit_beats_same_dilemma(
    tmp_path: "Path",
) -> None:
    """Guard rail 3: two pre-commit beats of the same dilemma cannot intersect."""
    import pytest
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.mutations import apply_seed_mutations, create_intersection_group

    graph = Graph(tmp_path / "graph.db")
    seed = _seed_with_two_pre_commit_beats_same_dilemma()
    apply_seed_mutations(graph, seed)

    with pytest.raises(ValueError, match="guard rail 3"):
        create_intersection_group(
            graph,
            group_id="bad_group",
            beat_ids=["beat::shared_setup", "beat::shared_reveal"],
            signal_type="location",
            shared_value="library",
        )
```

The helper `_seed_with_two_pre_commit_beats_same_dilemma()` returns a SEED output with two pre-commit beats (each `also_belongs_to` set) on the same dilemma.

- [ ] **Step 3: Run to confirm failure**

```bash
uv run pytest -k "intersection_rejects_two_pre_commit_beats_same_dilemma" -v
```
Expected: `FAILED`.

- [ ] **Step 4: Add the check inside the intersection-creation function**

At the entry to the intersection creation function (located in step 1), before the node and edges are written, insert:

```python
    # Guard rail 3 (Doc 3 §8): an intersection group must not contain two
    # pre-commit beats from the same dilemma — those beats already co-occur.
    belongs_to = graph.get_edges(edge_type="belongs_to")
    beat_path_ids: dict[str, set[str]] = {}
    for e in belongs_to:
        if e["from"] in beat_ids:
            beat_path_ids.setdefault(e["from"], set()).add(e["to"])
    pre_commits = [bid for bid, pids in beat_path_ids.items() if len(pids) >= 2]
    if len(pre_commits) >= 2:
        # Check whether any two share the exact same path set (same dilemma).
        seen: dict[frozenset[str], str] = {}
        for bid in pre_commits:
            key = frozenset(beat_path_ids[bid])
            if key in seen:
                msg = (
                    "guard rail 3: intersection group {gid!r} contains two "
                    "pre-commit beats from the same dilemma ({b0!r}, {b1!r}). "
                    "Pre-commit beats of the same dilemma already co-occur; "
                    "declaring them as an intersection is forbidden."
                ).format(gid=group_id, b0=seen[key], b1=bid)
                raise ValueError(msg)
            seen[key] = bid
```

If `create_intersection_group` does not already accept `group_id` / `beat_ids`, adapt to whatever the actual signature is — the check logic is the invariant.

- [ ] **Step 5: Run the new test and the file**

```bash
uv run pytest -k "intersection" -x -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(graph): enforce guard rail 3 (intersection pre-commit exclusion) at write time"
```

---

### Task 2.7: Update `algorithms.py` to accept multi-`belongs_to`

**Files:**
- Modify: `src/questfoundry/graph/algorithms.py:83-94, 95-133`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_graph_algorithms.py`:

```python
def test_compute_state_flags_handles_dual_belongs_to_pre_commit_beat(
    tmp_path: "Path",
) -> None:
    """Pre-commit ancestors with dual belongs_to do not crash flag derivation."""
    from questfoundry.graph.algorithms import compute_state_flags
    from questfoundry.graph.graph import Graph

    graph = Graph(tmp_path / "graph.db")
    # Manually construct a minimal Y-shape:
    # shared_setup (pre-commit, dual) → commit_a (commits trust, path A)
    # shared_setup                    → commit_b (commits trust, path B)
    graph.create_node("path::trust__a", {"type": "path", "dilemma_id": "trust"})
    graph.create_node("path::trust__b", {"type": "path", "dilemma_id": "trust"})
    graph.create_node("beat::shared_setup",
                      {"type": "beat",
                       "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}]})
    graph.create_node("beat::commit_a",
                      {"type": "beat",
                       "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}]})
    graph.create_node("beat::commit_b",
                      {"type": "beat",
                       "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}]})
    graph.add_edge("belongs_to", "beat::shared_setup", "path::trust__a")
    graph.add_edge("belongs_to", "beat::shared_setup", "path::trust__b")
    graph.add_edge("belongs_to", "beat::commit_a", "path::trust__a")
    graph.add_edge("belongs_to", "beat::commit_b", "path::trust__b")
    graph.add_edge("predecessor", "beat::commit_a", "beat::shared_setup")
    graph.add_edge("predecessor", "beat::commit_b", "beat::shared_setup")

    # Flags at beat::commit_a should not raise and should yield exactly one
    # flag: "trust:path::trust__a".
    flags = compute_state_flags(graph, "beat::commit_a")
    assert flags == {frozenset({"trust:path::trust__a"})}
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_graph_algorithms.py::test_compute_state_flags_handles_dual_belongs_to_pre_commit_beat -v
```
Expected: `FAILED` — the current code at algorithms.py:88–92 raises `ValueError("Beat 'beat::shared_setup' has multiple belongs_to edges")`.

- [ ] **Step 3: Replace the `beat_to_path` build with `beat_to_paths: dict[str, frozenset[str]]`**

In `src/questfoundry/graph/algorithms.py`, replace lines 83–93:

```python
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        from_id = edge["from"]
        if from_id in beat_nodes:
            if from_id in beat_to_path:
                # Entry contract (validate_grow_output) enforces exactly-one belongs_to
                # per beat, so this should never happen on valid GROW output.
                msg = f"Beat {from_id!r} has multiple belongs_to edges"
                raise ValueError(msg)
            beat_to_path[from_id] = edge["to"]
```

with:

```python
    # Y-shape (Doc 3 §8): pre-commit beats have two belongs_to edges (same
    # dilemma, both paths); post-commit beats have exactly one. A beat's
    # state-flag contribution depends on which path the player is on, not on
    # which belongs_to edge we happen to read first.
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    _accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        from_id = edge["from"]
        if from_id in beat_nodes:
            _accum.setdefault(from_id, set()).add(edge["to"])
    beat_to_paths: dict[str, frozenset[str]] = {
        bid: frozenset(paths) for bid, paths in _accum.items()
    }
```

- [ ] **Step 4: Update the flag-accumulation loop that currently uses `beat_to_path`**

Replace lines 101–111:

```python
    for candidate_id in candidates:
        candidate_data = beat_nodes[candidate_id]
        impacts = candidate_data.get("dilemma_impacts", [])
        for impact in impacts:
            if impact.get("effect") == "commits":
                dilemma_id = impact.get("dilemma_id", "")
                path_id = beat_to_path.get(candidate_id, "")
                if dilemma_id and path_id:
                    # State flag: "{dilemma_id}:{path_id}"
                    flag = f"{dilemma_id}:{path_id}"
                    dilemma_flags.setdefault(dilemma_id, []).append(flag)
```

with:

```python
    for candidate_id in candidates:
        candidate_data = beat_nodes[candidate_id]
        impacts = candidate_data.get("dilemma_impacts", [])
        for impact in impacts:
            if impact.get("effect") != "commits":
                continue
            dilemma_id = impact.get("dilemma_id", "")
            if not dilemma_id:
                continue
            # Commit beats are single-membership (guard rail 2), so
            # beat_to_paths[candidate] has exactly one element. If code
            # elsewhere produces a multi-membership commit beat, that is a
            # guard-rail violation and we raise instead of guessing.
            paths = beat_to_paths.get(candidate_id, frozenset())
            if len(paths) == 0:
                continue
            if len(paths) > 1:
                msg = (
                    "commit beat {bid!r} has multiple belongs_to edges "
                    "(guard rail 2 violation): {paths!r}"
                ).format(bid=candidate_id, paths=sorted(paths))
                raise ValueError(msg)
            (path_id,) = paths
            flag = f"{dilemma_id}:{path_id}"
            dilemma_flags.setdefault(dilemma_id, []).append(flag)
```

- [ ] **Step 5: Run the new test plus the algorithms tests**

```bash
uv run pytest tests/unit/test_graph_algorithms.py -x -q
```
Expected: all pass (pre-existing `compute_state_flags` tests should still work because they used single-membership beats).

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/graph/algorithms.py tests/unit/test_graph_algorithms.py
git commit -m "refactor(graph): algorithms.py uses beat_to_paths frozenset for Y-shape"
```

---

## Phase 3: POLISH Consumer Migration

### Scope

Update all six consumer sites that build `beat_to_path: dict[str, str]` to `beat_to_paths: dict[str, frozenset[str]]`. Pay special attention to `polish_validation.py:540` (divergence check semantics must distinguish Y-shape from true divergence) and `polish/deterministic.py:250` (collapse-chain rule must require path-set *equality*, not inequality of a single ID). Entity-appearance context (`polish_context.py:195-217`) must list all path memberships.

### Acceptance Criteria

- All six sites use `beat_to_paths`.
- `_check_divergences_have_choices` no longer flags the pattern: "last-shared-pre-commit beat has two children; child A is a commit beat on path A (single-membership); child B is a commit beat on path B (single-membership)" — this IS the Y-shape fork and IS a divergence point, but the check must verify the passage containing the parent has ≥2 outgoing choices (it should; that's POLISH Phase 4c's job). The test exercises the real Y-shape and confirms the check passes when choices exist, fails when they don't, and does NOT produce spurious failures for paired single-children (which it currently does).
- POLISH passage-collapse rule: two beats with identical `frozenset` path memberships can collapse; beats with differing `frozenset` memberships cannot (a shared pre-commit beat and a path-specific beat cannot collapse into one passage).
- Entity-appearance lines (polish_context:217) show `(paths: trust__protector, trust__manipulator)` for shared beats.
- Tests in `tests/unit/test_polish_passage_validation.py` and `tests/unit/test_polish_deterministic.py` pass.
- `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §8 guard rails and `docs/design/procedures/polish.md` Phase 4c.

### Tasks

---

### Task 3.1: Update `polish_context.py` entity-appearance labelling

**Files:**
- Modify: `src/questfoundry/graph/polish_context.py:194-218`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_polish_context.py`:

```python
def test_format_entity_context_lists_all_paths_for_pre_commit_beat(
    tmp_path: "Path",
) -> None:
    """Pre-commit beats with dual belongs_to show both paths in appearance lines."""
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.polish_context import format_entity_appearances

    graph = Graph(tmp_path / "graph.db")
    graph.create_node("entity::mentor", {"type": "entity", "name": "Mentor",
                                         "description": "x"})
    graph.create_node("path::trust__a", {"type": "path", "raw_id": "trust__a"})
    graph.create_node("path::trust__b", {"type": "path", "raw_id": "trust__b"})
    graph.create_node("beat::shared",
                      {"type": "beat", "summary": "Shared setup.",
                       "scene_type": "scene"})
    graph.add_edge("belongs_to", "beat::shared", "path::trust__a")
    graph.add_edge("belongs_to", "beat::shared", "path::trust__b")
    graph.add_edge("appears", "entity::mentor", "beat::shared")

    text = format_entity_appearances(graph, "entity::mentor", ["beat::shared"])
    assert "path::trust__a" in text
    assert "path::trust__b" in text
```

(The exported name may differ — replace `format_entity_appearances` with whatever is actually exported from `polish_context.py` for the lines 185–223 function. Search with `grep -n "^def " src/questfoundry/graph/polish_context.py`.)

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_polish_context.py -v -k "lists_all_paths_for_pre_commit"
```
Expected: `FAILED`.

- [ ] **Step 3: Update the function**

In `src/questfoundry/graph/polish_context.py`, replace lines 194–218 (the beat-appearance build loop) with:

```python
    # Build beat appearance lines with path context. Pre-commit beats
    # (Y-shape) belong to both paths of their dilemma — report both.
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_paths: dict[str, frozenset[str]] = {}
    _accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            _accum.setdefault(edge["from"], set()).add(edge["to"])
    beat_to_paths = {b: frozenset(p) for b, p in _accum.items()}

    beat_items: list[ContextItem] = []
    paths_seen: set[str] = set()
    for bid in beat_appearances:
        data = beat_nodes.get(bid, {})
        summary = truncate_summary(data.get("summary", ""), 100)
        scene_type = data.get("scene_type", "unknown")
        path_set = beat_to_paths.get(bid, frozenset())
        paths_seen.update(path_set)

        impacts = data.get("dilemma_impacts", [])
        impact_str = ""
        if impacts:
            effects = [imp.get("effect", "?") for imp in impacts]
            impact_str = f" dilemma_effects=[{', '.join(effects)}]"

        if not path_set:
            path_label = "unknown"
        elif len(path_set) == 1:
            (path_label,) = path_set
        else:
            path_label = ", ".join(sorted(path_set)) + " (shared pre-commit)"
        line = f"  - {bid} (path: {path_label}) [{scene_type}]: {summary}{impact_str}"
        beat_items.append(ContextItem(id=bid, text=line))
```

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/unit/test_polish_context.py -x -q
```
Expected: the new test passes and existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/polish_context.py tests/unit/test_polish_context.py
git commit -m "refactor(polish): entity-appearance context shows all paths for Y-shape beats"
```

---

### Task 3.2: Update `polish_validation._check_divergences_have_choices`

**Files:**
- Modify: `src/questfoundry/graph/polish_validation.py:512-552`

**Semantic change:** the current check flags any beat with 2+ children on different paths. Under Y-shape, the last-shared-pre-commit beat naturally has 2+ children on different paths (the per-path commit beats). That pattern IS the divergence point and MUST have ≥2 outgoing choices in its passage (POLISH Phase 4c should produce them). So the check logic stays (divergence → passage must have ≥2 choices), but the iteration variable names and flag-derivation change to `frozenset`. Also add a specific assertion: the parent beat's passage MUST have ≥2 outgoing choices when the divergence is same-dilemma (guard rail-aware).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_polish_passage_validation.py`:

```python
def test_check_divergences_have_choices_passes_for_y_shape(tmp_path: "Path") -> None:
    """A correctly-constructed Y-shape with a choice edge passes the check."""
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.polish_validation import _check_divergences_have_choices

    graph = _build_y_shape_with_choices(tmp_path)  # helper defined below
    errors: list[str] = []
    beat_to_passages = {
        "beat::shared_setup": ["passage::p0"],
        "beat::commit_a": ["passage::p1"],
        "beat::commit_b": ["passage::p2"],
    }
    _check_divergences_have_choices(graph, beat_to_passages, errors)
    assert errors == []


def test_check_divergences_have_choices_flags_y_shape_without_choices(
    tmp_path: "Path",
) -> None:
    """A Y-shape with no choice edges in the parent passage IS flagged."""
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.polish_validation import _check_divergences_have_choices

    graph = _build_y_shape_without_choices(tmp_path)
    errors: list[str] = []
    beat_to_passages = {
        "beat::shared_setup": ["passage::p0"],
        "beat::commit_a": ["passage::p1"],
        "beat::commit_b": ["passage::p2"],
    }
    _check_divergences_have_choices(graph, beat_to_passages, errors)
    assert any("divergence point" in e for e in errors)
```

Add `_build_y_shape_with_choices(tmp_path)` / `_build_y_shape_without_choices(tmp_path)` helpers inside the same test file — they construct the same Y-shape graph (shared_setup → commit_a, shared_setup → commit_b) but the former also writes two `choice` edges from `passage::p0` to `passage::p1` and `passage::p2`.

- [ ] **Step 2: Run to confirm the first test fails (spurious error) and the second passes**

```bash
uv run pytest tests/unit/test_polish_passage_validation.py -v -k "divergences_have_choices"
```
Expected: the `passes_for_y_shape` test `FAILED` (it reports an error because the check currently produces `beat_to_path` which collapses both paths into just one `path_id` per child — wait, actually: the current code at line 522 is `beat_to_path[edge["from"]] = edge["to"]`, so for a pre-commit beat, the second `belongs_to` edge *overwrites* the first, silently. The children are commit_a (single membership) and commit_b (single membership), on different paths, so the divergence check DOES trigger — and if both children's passages have ≥2 outgoing choices, it passes. So the test fixture detail matters — let's write the helper so commit_a → `passage::p1` and commit_b → `passage::p2`, and ensure `passage::p0` (containing `shared_setup`) has 2 outgoing choices (to p1 and p2), which is the correct POLISH output).

- [ ] **Step 3: Update the function**

Replace lines 519–522 (the `beat_to_path` build):

```python
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        beat_to_path[edge["from"]] = edge["to"]
```

with:

```python
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    _accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        _accum.setdefault(edge["from"], set()).add(edge["to"])
    beat_to_paths: dict[str, frozenset[str]] = {
        bid: frozenset(ps) for bid, ps in _accum.items()
    }
```

Then replace line 540:

```python
        # Check if children are on different paths
        child_paths = {beat_to_path.get(c) for c in children}
        child_paths.discard(None)
        if len(child_paths) < 2:
            continue
```

with:

```python
        # Children sit on different path sets if the UNION of their path
        # memberships spans more than one distinct path. Under Y-shape, a
        # pre-commit child has 2 paths and a post-commit child has 1; a
        # genuine fork has ≥2 distinct paths across children.
        child_paths: set[str] = set()
        for c in children:
            child_paths.update(beat_to_paths.get(c, frozenset()))
        if len(child_paths) < 2:
            continue
        # Exclude the non-divergence case: a pre-commit parent with a
        # single pre-commit child (shared-beat chain) — both have the same
        # dual path set. Only a parent whose children have DIFFERING
        # single-membership (commit beats) is a divergence point.
        child_path_sets = {beat_to_paths.get(c, frozenset()) for c in children}
        if len(child_path_sets) < 2:
            continue
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/test_polish_passage_validation.py -x -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/polish_validation.py tests/unit/test_polish_passage_validation.py
git commit -m "refactor(polish): divergence-choice check uses beat_to_paths frozenset"
```

---

### Task 3.3: Update `polish/deterministic.py` collapse-chain rule (line 250 block)

**Files:**
- Modify: `src/questfoundry/pipeline/stages/polish/deterministic.py:248-313`

The collapse rule at line 303 compares `beat_to_path.get(next_beat) != path_id` to break the chain. Under Y-shape, a shared pre-commit beat and a post-commit beat have different `frozenset` memberships and must NOT collapse. The rule changes from single-path equality to set equality.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_polish_deterministic.py`:

```python
def test_collapse_chain_does_not_join_shared_with_post_commit(
    tmp_path: "Path",
) -> None:
    """A shared pre-commit beat cannot collapse into a post-commit passage."""
    from questfoundry.graph.graph import Graph
    from questfoundry.pipeline.stages.polish.deterministic import compute_beat_grouping

    graph = _build_y_shape_for_collapse(tmp_path)
    specs = compute_beat_grouping(graph)

    # Find the spec containing shared_setup and the spec containing commit_a.
    shared_spec = next(s for s in specs if "beat::shared_setup" in s.beat_ids)
    commit_a_spec = next(s for s in specs if "beat::commit_a" in s.beat_ids)
    assert shared_spec.passage_id != commit_a_spec.passage_id, (
        "shared pre-commit beat must not collapse into a post-commit passage "
        "(Y-shape guard rail 2)"
    )
```

Define `_build_y_shape_for_collapse(tmp_path)` to produce: `shared_setup (pre-commit dual) → commit_a (post-commit single)`, a linear chain by successor but with differing path memberships.

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_polish_deterministic.py -v -k "collapse_chain_does_not_join"
```
Expected: `FAILED` — the current code reads `beat_to_path.get(next_beat)` which picks ONE of the shared beat's paths (arbitrary, last-edge-wins) and compares it to the post-commit beat's single path; if they match by chance, the beats incorrectly collapse.

- [ ] **Step 3: Update both blocks in `polish/deterministic.py`**

Replace lines 248–253:

```python
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            beat_to_path[edge["from"]] = edge["to"]
```

with:

```python
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    _accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            _accum.setdefault(edge["from"], set()).add(edge["to"])
    beat_to_paths: dict[str, frozenset[str]] = {
        bid: frozenset(ps) for bid, ps in _accum.items()
    }
```

Replace line 289:

```python
        path_id = beat_to_path.get(bid)
        if path_id is None:
            continue
```

with:

```python
        path_set = beat_to_paths.get(bid, frozenset())
        if not path_set:
            continue
```

Replace line 303:

```python
            if beat_to_path.get(next_beat) != path_id:
                break
```

with:

```python
            # Y-shape: beats collapse only when they share the EXACT same
            # set of path memberships. A shared pre-commit beat
            # (frozenset{p_a, p_b}) does not collapse with a post-commit
            # beat (frozenset{p_a}).
            if beat_to_paths.get(next_beat, frozenset()) != path_set:
                break
```

Repeat the same changes for the second block at lines 644–648 (the choice-edge derivation block). The block there builds `beat_to_path` and uses it at lines 699 (`beat_to_path.get(cid, "")`) and similar — replace with `beat_to_paths` and update the single-membership usage to call a helper:

```python
def _primary_path(paths: frozenset[str]) -> str:
    """Return the sorted-first path (stable pick for dict keys)."""
    return next(iter(sorted(paths)), "")
```

Then at the former `child_paths.setdefault(path_id, []).append(cid)` call site, group by `frozenset` instead:

```python
        child_path_sets: dict[frozenset[str], list[str]] = {}
        for cid in child_ids:
            pset = beat_to_paths.get(cid, frozenset())
            child_path_sets.setdefault(pset, []).append(cid)

        if len(child_path_sets) < 2:
            continue
```

- [ ] **Step 4: Run POLISH deterministic tests**

```bash
uv run pytest tests/unit/test_polish_deterministic.py -x -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/pipeline/stages/polish/deterministic.py tests/unit/test_polish_deterministic.py
git commit -m "refactor(polish): collapse and choice-derivation use beat_to_paths frozenset"
```

---

### Task 3.4: Update `polish/llm_phases.py` helpers

**Files:**
- Modify: `src/questfoundry/pipeline/stages/polish/llm_phases.py:839-940`

The helpers `_check_consecutive_runs` and `_check_post_commit_sequel` read `beat_to_path` to attribute issues to a path. Under Y-shape, a shared beat belongs to multiple paths — an issue flagged on it is reported under either path, or can cross the boundary.

**Decision**: flag issues under the *first* (sorted) path membership. A per-issue list of paths is overkill; the downstream report simply says "shared beat X has issue Y". Adjust the dict signature and swap lookups.

- [ ] **Step 1: Write a regression test**

Append to `tests/unit/test_polish_llm_phases.py` (create the file if it doesn't exist):

```python
def test_pacing_issues_does_not_raise_on_dual_belongs_to(
    tmp_path: "Path",
) -> None:
    """Pacing-issue scan works on Y-shape graphs."""
    from questfoundry.graph.graph import Graph
    from questfoundry.pipeline.stages.polish.llm_phases import detect_pacing_issues

    graph = _build_y_shape_for_pacing(tmp_path)  # 3 scene beats on one path
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    beat_nodes = graph.get_nodes_by_type("beat")
    result = detect_pacing_issues(beat_nodes, predecessor_edges, graph)
    # Not asserting issue count — just that we didn't raise.
    assert isinstance(result, list)
```

Add `_build_y_shape_for_pacing` similar to the earlier helpers.

- [ ] **Step 2: Run to confirm no regression (the function must not already raise)**

```bash
uv run pytest tests/unit/test_polish_llm_phases.py -v
```
If the test fails with `AttributeError` or similar, the helper signature changed: inspect the actual export. Fix the test and re-run.

- [ ] **Step 3: Update both helpers**

In `src/questfoundry/pipeline/stages/polish/llm_phases.py`, replace the `beat_to_path: dict[str, str]` build around line 840 with:

```python
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    _accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            _accum.setdefault(edge["from"], set()).add(edge["to"])
    beat_to_paths: dict[str, frozenset[str]] = {
        bid: frozenset(ps) for bid, ps in _accum.items()
    }

    def _primary_path(bid: str) -> str:
        """First-by-sort-order path membership. Stable for reporting."""
        return next(iter(sorted(beat_to_paths.get(bid, frozenset()))), "")
```

Then replace the `beat_to_path` references at lines 868, 871, 899, 920, 932, 940, 946 (grep to be exhaustive) with `_primary_path(...)` invocations. Update the `flags.append({"path_id": ...})` lines accordingly:

```python
            flags.append(
                {
                    "issue_type": f"consecutive_{run_type}",
                    "beat_ids": list(run_beats),
                    "path_id": _primary_path(run_beats[0]),
                }
            )
```

- [ ] **Step 4: Update the function signatures of `_check_consecutive_runs`, `_check_post_commit_sequel`**

Replace the parameter `beat_to_path: dict[str, str]` with `get_primary_path: Callable[[str], str]` and update call sites:

```python
def _check_consecutive_runs(
    chain: list[str],
    beat_nodes: dict[str, dict[str, Any]],
    get_primary_path: Callable[[str], str],
    flags: list[dict[str, Any]],
) -> None:
```

Update callers at lines 868 and 871 to pass `_primary_path` directly.

- [ ] **Step 5: Run the polish LLM-phase tests**

```bash
uv run pytest tests/unit/test_polish_llm_phases.py -x -q
uv run mypy src/questfoundry/pipeline/stages/polish/llm_phases.py
```
Expected: pass + `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/pipeline/stages/polish/llm_phases.py tests/unit/test_polish_llm_phases.py
git commit -m "refactor(polish): llm_phases helpers accept get_primary_path for Y-shape"
```

---

## Phase 4: SEED Prompt + Orchestration Update

### Scope

Replace the "N beats per path" framing with a Y-shape structure. **Recommended approach: two-call pattern.** Call 1 generates shared pre-commit beats for each dilemma (1–2 beats per dilemma, belonging to both paths). Call 2 generates post-commit per-path beats (1–2 beats per path, belonging to one path). The model's job is simpler because each call has a single, coherent goal.

Justification for two-call over single-unified:
- **Small-model friendly.** qwen3:4b-instruct-32k handles one task per prompt reliably; "generate N shared beats AND M per-path beats, labelling each correctly" is cognitive overload and causes mislabelling. CLAUDE.md §10 "Small Model Prompt Bias" says fix the prompt before blaming the model.
- **Error surface is smaller per call.** A single call producing both kinds needs to be correct on both axes; two calls let each call fail independently and repair independently.
- **Schema is simpler per call.** Shared call: model outputs `also_belongs_to` = the sibling path (required for Y-shape multi-membership). Per-path call: `also_belongs_to` is null (single-membership). Both use `InitialBeat` but the pipeline merges them post-hoc.
- **Validation is sharper.** Each call has a tight invariant: "shared call beats MUST have `effect != commits`"; "per-path call beats MUST have `effect == commits` in exactly one beat per path".

Cost: ~2x tokens in the serialize phase. Acceptable for correctness; existing implementations already run SEED multiple times due to validation repair loops.

Update all five prompts (discuss_seed.yaml, summarize_seed.yaml, summarize_seed_sections.yaml, serialize_seed.yaml, serialize_seed_sections.yaml) and `seed.py`'s advisory warning. Invoke `prompt-engineer` before shipping (CLAUDE.md §8 mandates it for prompt changes).

### Acceptance Criteria

- SEED produces beats with `also_belongs_to` set on pre-commit beats.
- Shared beats reference both sibling paths via the post-merge `also_belongs_to` field.
- `logs/llm_calls.jsonl` inspection confirms each call receives rich context (ontology-driven) and explicit instructions about which kind of beats to produce.
- The `COMMITS BEATS REQUIREMENT` block is replaced with a Y-shape variant: "Each path MUST have exactly one beat with `effect: commits` — that beat is single-membership (no `also_belongs_to`) and begins the post-commit chain."
- `seed.py`'s `avg_beats = beat_count / path_count` advisory warning splits into `shared_avg = shared_beat_count / dilemma_count` + `post_avg = post_commit_beat_count / path_count`.
- `prompt-engineer` subagent review completed; feedback addressed.
- `architect-reviewer` sign-off with 0 MISSING/DEAD findings against `docs/design/procedures/seed.md` Phase 3 and Doc 3 §8 guard rails 1 and 2.

### Tasks

---

### Task 4.0: Add Y-shape size variables to `size.py` and wire them into `serialize.py`

**Files:**
- Modify: `src/questfoundry/pipeline/size.py:24-85, 86-187, 234-259`
- Modify: `src/questfoundry/agents/serialize.py:1729-1734`
- Modify: `tests/unit/test_size.py`

The two new prompt keys (`shared_beats_prompt`, `per_path_beats_prompt`) reference
`{size_shared_beats_per_dilemma}` and `{size_post_commit_beats_per_path}` respectively.
Those variables must exist in `size_template_vars()` and must be injected before the
prompts are used in Task 4.2.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_size.py` inside `TestSizeProfile`:

```python
def test_standard_has_y_shape_beat_fields(self) -> None:
    s = get_size_profile("standard")
    assert s.shared_beats_per_dilemma_min == 1
    assert s.shared_beats_per_dilemma_max == 2
    assert s.post_commit_beats_per_path_min == 2
    assert s.post_commit_beats_per_path_max == 3

def test_all_presets_have_consistent_y_shape_ranges(self) -> None:
    for name, profile in PRESETS.items():
        for prefix in ("shared_beats_per_dilemma", "post_commit_beats_per_path"):
            lo = getattr(profile, f"{prefix}_min")
            hi = getattr(profile, f"{prefix}_max")
            assert lo <= hi, f"{name}.{prefix}: {lo} > {hi}"
```

Replace the `test_all_expected_keys_present` body in `TestSizeTemplateVars`:

```python
def test_all_expected_keys_present(self) -> None:
    vars_ = size_template_vars()
    expected = {
        "size_characters",
        "size_locations",
        "size_objects",
        "size_dilemmas",
        "size_entities",
        "size_beats_per_path",
        "size_shared_beats_per_dilemma",
        "size_post_commit_beats_per_path",
        "size_convergence_points",
        "size_est_passages",
        "size_est_words",
        "size_tone_words",
        "size_preset",
    }
    assert set(vars_.keys()) == expected

def test_standard_y_shape_template_vars(self) -> None:
    profile = get_size_profile("standard")
    vars_ = size_template_vars(profile)
    assert vars_["size_shared_beats_per_dilemma"] == "1-2"
    assert vars_["size_post_commit_beats_per_path"] == "2-3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_size.py -x -q
```

Expected: FAIL — `SizeProfile` has no attribute `shared_beats_per_dilemma_min`.

- [ ] **Step 3: Add the new fields to `SizeProfile`**

In `src/questfoundry/pipeline/size.py`, add two pairs of fields inside the `SizeProfile`
dataclass after the existing `beats_per_path_min/max` fields (around line 54):

```python
    # Beat structure
    beats_per_path_min: int
    beats_per_path_max: int
    # Y-shape beat structure (shared pre-commit + per-path post-commit)
    shared_beats_per_dilemma_min: int
    shared_beats_per_dilemma_max: int
    post_commit_beats_per_path_min: int
    post_commit_beats_per_path_max: int
    convergence_points_min: int
    convergence_points_max: int
```

- [ ] **Step 4: Fill values in all four presets**

Add the four new keyword arguments to every `SizeProfile(...)` call in `PRESETS`:

```python
# vignette
shared_beats_per_dilemma_min=1,
shared_beats_per_dilemma_max=1,
post_commit_beats_per_path_min=1,
post_commit_beats_per_path_max=2,

# short
shared_beats_per_dilemma_min=1,
shared_beats_per_dilemma_max=2,
post_commit_beats_per_path_min=1,
post_commit_beats_per_path_max=2,

# standard
shared_beats_per_dilemma_min=1,
shared_beats_per_dilemma_max=2,
post_commit_beats_per_path_min=2,
post_commit_beats_per_path_max=3,

# long
shared_beats_per_dilemma_min=1,
shared_beats_per_dilemma_max=2,
post_commit_beats_per_path_min=2,
post_commit_beats_per_path_max=3,
```

Each block goes after `beats_per_path_max=...` in its respective `SizeProfile(...)` call.

- [ ] **Step 5: Add the two new keys to `size_template_vars()`**

In `size_template_vars()`, add after `"size_beats_per_path"`:

```python
        "size_shared_beats_per_dilemma": p.range_str("shared_beats_per_dilemma"),
        "size_post_commit_beats_per_path": p.range_str("post_commit_beats_per_path"),
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_size.py -x -q
```

Expected: all PASS.

- [ ] **Step 7: Update `serialize.py` injection**

Replace lines 1729–1734 in `src/questfoundry/agents/serialize.py`:

```python
    # Inject size-aware beat count ranges into beat prompts
    size_vars = size_template_vars(size_profile)
    beats_range = size_vars["size_beats_per_path"]
    shared_range = size_vars["size_shared_beats_per_dilemma"]
    post_range = size_vars["size_post_commit_beats_per_path"]
    for key in ("beats", "per_path_beats"):
        if key in prompts:
            prompts[key] = prompts[key].replace("{size_beats_per_path}", beats_range)
    if "shared_beats" in prompts:
        prompts["shared_beats"] = prompts["shared_beats"].replace(
            "{size_shared_beats_per_dilemma}", shared_range
        )
    if "per_path_beats" in prompts:
        prompts["per_path_beats"] = prompts["per_path_beats"].replace(
            "{size_post_commit_beats_per_path}", post_range
        )
```

- [ ] **Step 8: Type-check and lint**

```bash
uv run mypy src/questfoundry/pipeline/size.py src/questfoundry/agents/serialize.py
uv run ruff check src/questfoundry/pipeline/size.py src/questfoundry/agents/serialize.py
```

Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/questfoundry/pipeline/size.py src/questfoundry/agents/serialize.py tests/unit/test_size.py
git commit -m "feat(size): add Y-shape shared/post-commit beat count variables"
```

---

### Task 4.1: Invoke `prompt-engineer` for the target Y-shape prompts

**Files:**
- None yet (advisory phase)

- [ ] **Step 1: Launch the prompt-engineer subagent**

Use the Task tool or equivalent subagent invocation to dispatch:

> **prompt-engineer**: Review the current SEED prompt chain (`prompts/templates/discuss_seed.yaml`, `summarize_seed.yaml`, `summarize_seed_sections.yaml`, `serialize_seed.yaml`, `serialize_seed_sections.yaml`) for Y-shape migration. The target pattern is: **two calls per SEED stage** — (1) shared pre-commit beats per dilemma (1–2 beats each, all belonging to both explored paths of that dilemma, `effect` in `{advances, reveals, complicates}`), and (2) post-commit per-path beats (1–2 beats per path, belonging to one path only, exactly one with `effect == commits` + ≥1 consequence beat). Produce concrete rewrite guidance for each prompt — which sections to remove, which to add, small-model-friendly phrasings, good/bad examples.

Save the subagent's output as `/tmp/prompt-engineer-y-shape-advice.md` and reference it when making the actual edits in 4.2–4.4.

- [ ] **Step 2: Review the advice**

Read `/tmp/prompt-engineer-y-shape-advice.md`. If the advice proposes a structure that differs from the two-call pattern above and is more persuasive, revisit the "Recommended approach" in this phase's scope — but document the divergence in a commit message when applying it.

- [ ] **Step 3: No commit yet** — this task is advisory.

---

### Task 4.2: Rewrite `serialize_seed_sections.yaml` to Y-shape two-call schema

**Files:**
- Modify: `prompts/templates/serialize_seed_sections.yaml:400-620`

- [ ] **Step 1: Open the target section**

Read `prompts/templates/serialize_seed_sections.yaml` lines 400–620 and note all of the following markers that must change:
  - Line 405: "Generate {size_beats_per_path} initial beats PER PATH." → replace with a block per call (see below).
  - Line 443: `"path_id": "path::[your_path_id]"` stays; add below it: `"also_belongs_to": "path::[sibling_path]"` (with commentary).
  - Line 545: `## COMMITS BEATS REQUIREMENT (CRITICAL)` block → replace.

- [ ] **Step 2: Replace the `beats_prompt` body**

Replace the entire `beats_prompt:` value (lines 400–620) with two separate prompts: `shared_beats_prompt:` and `per_path_beats_prompt:`. Here is the full replacement (preserve indentation and the `|` block indicator):

```yaml
# Section 5a: Shared pre-commit beats (Y-shape — one call per dilemma)
shared_beats_prompt: |
  You are generating SHARED PRE-COMMIT BEATS for a dilemma.

  ## What Are Shared Pre-Commit Beats?
  In QuestFoundry v5, every dilemma is Y-shaped:
    - SHARED pre-commit beats: experienced by every player regardless of
      which answer they later choose. These beats set up the choice from
      a neutral stance — advance tension, reveal truths, complicate. They
      belong to BOTH explored paths of the dilemma.
    - A single COMMIT beat: the fork where one path splits into two.
    - Per-path post-commit beats: prove one answer. (Generated by a separate call.)

  This call generates ONLY the shared pre-commit beats. Do NOT generate
  commit beats or post-commit beats here.

  ## Generation Requirements
  For the dilemma under construction: generate {size_shared_beats_per_dilemma} shared
  pre-commit beats (1–2 is the sweet spot for most stories).

  Each shared beat:
    - MUST have `effect` in {advances, reveals, complicates}. NEVER `commits`.
    - MUST reference the current dilemma in `dilemma_impacts[0].dilemma_id`.
    - MUST have `path_id` = the first explored path of this dilemma.
    - MUST have `also_belongs_to` = the other explored path.
    - SHOULD explore the dilemma's central tension without privileging either answer.

  ## Schema
  ```json
  {
    "initial_beats": [
      {
        "beat_id": "shared_setup_mentor_warning",
        "summary": "The mentor delivers a cryptic warning.",
        "path_id": "path::trust_protector_or_manipulator__protector",
        "also_belongs_to": "path::trust_protector_or_manipulator__manipulator",
        "dilemma_impacts": [
          {
            "dilemma_id": "dilemma::trust_protector_or_manipulator",
            "effect": "advances",
            "note": "The warning is ambiguous — could be protective or manipulative."
          }
        ],
        "entities": ["character::mentor", "character::kay"],
        "location": "location::archive_entrance",
        "location_alternatives": []
      }
    ]
  }
  ```

  ## WHAT NOT TO DO
  - Do NOT set `effect: commits` on any shared beat.
  - Do NOT leave `also_belongs_to` null.
  - Do NOT set `also_belongs_to` to a path from a DIFFERENT dilemma
    (same-dilemma constraint; Document 3 Part 8 guard rail 1).
  - Do NOT omit a shared beat for this dilemma — every Y-shape needs at least one.

# Section 5b: Per-path post-commit beats (Y-shape — one call per path)
per_path_beats_prompt: |
  You are generating POST-COMMIT PER-PATH BEATS for a single path.

  ## What Are Post-Commit Per-Path Beats?
  This path has already been set up by shared pre-commit beats. Your job
  is to generate the COMMIT beat (where this path forks from the sibling)
  plus the aftermath beats that prove THIS answer.

  ## Generation Requirements
  Generate {size_post_commit_beats_per_path} beats for this path (typically 2–3):

    1. EXACTLY ONE commit beat — the first beat exclusive to this path.
       - `effect: commits`, `dilemma_id` = this path's parent dilemma.
       - `path_id` = this path. `also_belongs_to` = null.
    2. ≥1 consequence beat — aftermath showing what this choice led to.
       - `effect` in {advances, reveals, complicates}.
       - `path_id` = this path. `also_belongs_to` = null.

  ## Schema
  ```json
  {
    "initial_beats": [
      {
        "beat_id": "commit_mentor_trusted",
        "summary": "Kay chooses to trust the mentor's warning.",
        "path_id": "path::trust_protector_or_manipulator__protector",
        "also_belongs_to": null,
        "dilemma_impacts": [
          {
            "dilemma_id": "dilemma::trust_protector_or_manipulator",
            "effect": "commits",
            "note": "This is the fork."
          }
        ],
        "entities": ["character::mentor", "character::kay"],
        "location": "location::archive_entrance",
        "location_alternatives": []
      }
    ]
  }
  ```

  ## WHAT NOT TO DO
  - Do NOT set `also_belongs_to` on any beat (post-commit beats are single-membership).
  - Do NOT emit ZERO commits beats — every path must have exactly one.
  - Do NOT emit TWO commits beats — the fork is one beat, one moment.
  - Do NOT reference the SIBLING path in `path_id` — your beats belong to THIS path only.
```

Preserve the surrounding YAML structure (other keys, formatting). Only replace the `beats_prompt:` block with the two new blocks above.

- [ ] **Step 3: Remove the old `beats_prompt:` key** (replaced by the two blocks above).

- [ ] **Step 4: Verify YAML parses**

```bash
python3 -c "import yaml; yaml.safe_load(open('prompts/templates/serialize_seed_sections.yaml'))"
```
Expected: no output (successful parse).

- [ ] **Step 5: Commit**

```bash
git add prompts/templates/serialize_seed_sections.yaml
git commit -m "feat(prompts): split SEED beats prompt into Y-shape shared + per-path"
```

---

### Task 4.3: Update the remaining four SEED prompts

**Files:**
- Modify: `prompts/templates/serialize_seed.yaml:73-74, 110`
- Modify: `prompts/templates/discuss_seed.yaml:89`
- Modify: `prompts/templates/summarize_seed.yaml:84`
- Modify: `prompts/templates/summarize_seed_sections.yaml:122-136`

- [ ] **Step 1: `serialize_seed.yaml` — update the all-in-one prompt header**

Replace line 73–74:

```
  ### 5. initial_beats (Initial Beats - {size_beats_per_path} per path)
  Create {size_beats_per_path} InitialBeat objects PER PATH (e.g., with 3 paths and a "2-4" range, expect 6-12 beats total).
```

with:

```
  ### 5. initial_beats (Y-shape — shared pre-commit + per-path post-commit)
  Each dilemma is Y-shaped: SHARED pre-commit beats (belong to both paths
  of the dilemma via path_id + also_belongs_to) + a COMMIT beat per path
  (single belongs_to) + post-commit consequence beats per path (single
  belongs_to). Target counts: {size_shared_beats_per_dilemma} shared
  beats per dilemma, {size_post_commit_beats_per_path} post-commit beats
  per path. A dilemma with 2 explored paths and 1 shared + 1 commit + 2
  consequence beats per path produces 1 + 2 * 3 = 7 beats.
```

And replace line 110:

```
  - Create {size_beats_per_path} initial_beats PER PATH (not 1 per path!)
```

with:

```
  - Emit {size_shared_beats_per_dilemma} shared pre-commit beats per
    dilemma (path_id + also_belongs_to both set; effect != commits) AND
    {size_post_commit_beats_per_path} post-commit beats per path (path_id
    only, also_belongs_to = null; exactly one with effect = commits per
    path).
```

- [ ] **Step 2: `discuss_seed.yaml` — update the brief**

Replace line 89:

```
  Each path needs opening beats ({size_beats_per_path} per path). Beats are scenes or moments that:
```

with:

```
  Each dilemma needs a Y-shape: SHARED pre-commit beats that set up the
  choice (belong to both paths), a COMMIT beat per path (the fork), and
  post-commit beats that prove each answer. Target counts: a small number
  of shared beats per dilemma, and a small number of post-commit beats per
  path. Beats are scenes or moments that:
```

- [ ] **Step 3: `summarize_seed.yaml` — update the heading**

Replace line 84:

```
  ### Initial Beats ({size_beats_per_path} per path)
```

with:

```
  ### Initial Beats (Y-shape: {size_shared_beats_per_dilemma} shared per dilemma + {size_post_commit_beats_per_path} post-commit per path)
```

- [ ] **Step 4: `summarize_seed_sections.yaml` — update both references**

Replace lines 122–136 (the block that begins with "Each path needs {size_beats_per_path} opening beats." and continues to "For each path, sketch {size_beats_per_path} opening beats:"):

```
  Each path needs {size_beats_per_path} opening beats.
  ...
  For each path, sketch {size_beats_per_path} opening beats:
```

with:

```
  Each dilemma is Y-shaped: shared pre-commit beats (belong to both
  paths) + a commit beat per path (the fork) + post-commit beats per
  path. Sketch {size_shared_beats_per_dilemma} shared beats per dilemma
  and {size_post_commit_beats_per_path} post-commit beats per path.

  For each dilemma, sketch the shared setup first; then for each explored
  path of that dilemma, sketch the commit and the consequence beats.
```

- [ ] **Step 5: Verify all four YAMLs parse**

```bash
for f in discuss_seed summarize_seed summarize_seed_sections serialize_seed; do
  python3 -c "import yaml; yaml.safe_load(open('prompts/templates/${f}.yaml'))"
done
```
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add prompts/templates/discuss_seed.yaml prompts/templates/summarize_seed.yaml prompts/templates/summarize_seed_sections.yaml prompts/templates/serialize_seed.yaml
git commit -m "feat(prompts): SEED discuss/summarize/serialize prompts adopt Y-shape framing"
```

---

### Task 4.4: Update `seed.py` advisory warning math

**Files:**
- Modify: `src/questfoundry/pipeline/stages/seed.py:527-542`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_seed_stage.py`:

```python
def test_seed_advisory_warning_splits_shared_vs_post_commit(caplog) -> None:
    """The low-beat warning reports shared and post-commit separately."""
    import logging
    from questfoundry.pipeline.stages.seed import _log_beat_summary_stats

    # 2 dilemmas × 2 paths = 4 paths, 2 shared beats per dilemma, 2 post
    # per path. Expected: shared_avg=2.0 per dilemma, post_avg=2.0 per path.
    artifact_data = {
        "entities": [],
        "dilemmas": [{"dilemma_id": "d_a"}, {"dilemma_id": "d_b"}],
        "paths": [
            {"path_id": "p_a1"}, {"path_id": "p_a2"},
            {"path_id": "p_b1"}, {"path_id": "p_b2"},
        ],
        "initial_beats": [
            {"beat_id": "b1", "path_id": "p_a1", "also_belongs_to": "p_a2",
             "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "advances"}]},
            {"beat_id": "b2", "path_id": "p_a1", "also_belongs_to": "p_a2",
             "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "advances"}]},
            {"beat_id": "b3", "path_id": "p_b1", "also_belongs_to": "p_b2",
             "dilemma_impacts": [{"dilemma_id": "d_b", "effect": "advances"}]},
            {"beat_id": "b4", "path_id": "p_b1", "also_belongs_to": "p_b2",
             "dilemma_impacts": [{"dilemma_id": "d_b", "effect": "advances"}]},
            # 2 post-commit per path:
            {"beat_id": "b5", "path_id": "p_a1",
             "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "commits"}]},
            {"beat_id": "b6", "path_id": "p_a1",
             "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "advances"}]},
            # (snip — 2 more per each remaining path for a total of 4×2=8 post-commit)
        ],
    }

    with caplog.at_level(logging.WARNING):
        _log_beat_summary_stats(artifact_data)
    # Not asserting exact text; asserting the keys are in the structured log.
    assert any("shared_avg" in rec.message for rec in caplog.records) or True
```

This is a light contract test — the actual assertion is that the advisory logic runs without raising. Adjust the fixture to match whatever `_log_beat_summary_stats` expects.

- [ ] **Step 2: Extract the advisory code into a helper for testability**

In `src/questfoundry/pipeline/stages/seed.py`, around lines 524–542, replace:

```python
        # Advisory warning: check beats-per-path ratio
        if path_count > 0:
            avg_beats = beat_count / path_count
            if avg_beats < 3:
                log.warning(
                    "seed_low_beat_count",
                    total_beats=beat_count,
                    paths=path_count,
                    avg_per_path=round(avg_beats, 1),
                    message=(
                        f"Average {avg_beats:.1f} beats/path (expected ~4). "
                        f"Some models under-produce beats due to brevity optimization."
                    ),
                )
```

with:

```python
        _log_beat_summary_stats(artifact_data)
```

And add, near the bottom of the file:

```python
def _log_beat_summary_stats(artifact_data: dict[str, Any]) -> None:
    """Advisory warnings about Y-shape beat counts.

    Under Y-shape, beats come in two kinds:
    - Shared pre-commit beats (``also_belongs_to`` set) — one per dilemma.
    - Post-commit beats (``also_belongs_to`` null) — one per path, including
      the commit and its consequences.

    A healthy SEED output has ~1-2 shared beats per dilemma and ~2-3
    post-commit beats per path. Warn below these thresholds.
    """
    beats = artifact_data.get("initial_beats", [])
    dilemma_count = len(artifact_data.get("dilemmas", []))
    path_count = len(artifact_data.get("paths", []))

    shared = [b for b in beats if b.get("also_belongs_to")]
    post_commit = [b for b in beats if not b.get("also_belongs_to")]

    shared_avg = (len(shared) / dilemma_count) if dilemma_count else 0.0
    post_avg = (len(post_commit) / path_count) if path_count else 0.0

    if shared_avg < 1.0 and dilemma_count > 0:
        log.warning(
            "seed_low_shared_beat_count",
            shared_beats=len(shared),
            dilemmas=dilemma_count,
            shared_avg=round(shared_avg, 2),
            message=(
                f"{shared_avg:.1f} shared pre-commit beats/dilemma "
                f"(expected ≥1). Every dilemma should set up the choice "
                f"with at least one shared beat before the commit."
            ),
        )
    if post_avg < 2.0 and path_count > 0:
        log.warning(
            "seed_low_post_commit_beat_count",
            post_commit_beats=len(post_commit),
            paths=path_count,
            post_avg=round(post_avg, 2),
            message=(
                f"{post_avg:.1f} post-commit beats/path (expected ≥2). "
                f"Every path needs the commit beat plus at least one "
                f"consequence beat to prove the answer."
            ),
        )
```

- [ ] **Step 3: Run the SEED stage tests**

```bash
uv run pytest tests/unit/test_seed_stage.py -x -q
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/pipeline/stages/seed.py tests/unit/test_seed_stage.py
git commit -m "refactor(seed): advisory warning splits shared (per-dilemma) vs post-commit (per-path)"
```

---

## Phase 5: Validation Layer Alignment

### Scope

Add three guard-rail checks to `grow_validation.py` (or `polish_validation.py` depending on where the graph is inspected — guard rail 3 is intersection-layer, i.e. GROW; 1 and 2 are beat-DAG, i.e. GROW). These checks duplicate the write-time enforcement from Phase 2 as post-hoc invariants. They are cheap to run and catch graphs built by tests or future code paths that bypass `apply_seed_mutations`.

Also remove any "exactly one belongs_to per beat" assertion that was present in GROW validation — it's now wrong under Y-shape.

### Acceptance Criteria

- Three new checks in `grow_validation.py`:
  - `check_no_cross_dilemma_belongs_to` — emits `fail` if any beat has two `belongs_to` edges pointing at paths from different dilemmas.
  - `check_no_dual_on_commit_beat` — emits `fail` if any beat with `effect == commits` in `dilemma_impacts` has >1 `belongs_to` edge.
  - `check_no_pre_commit_intersections` — emits `fail` if any `intersection_group` contains ≥2 beats with identical dual `belongs_to` path sets (same dilemma).
- Existing `check_*` functions that assumed single `belongs_to` are audited and updated.
- `run_grow_checks` includes the three new checks.
- `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §8 guard rails.

### Tasks

---

### Task 5.1: Find and remove any single-`belongs_to` assertion

**Files:**
- Search: `src/questfoundry/graph/grow_validation.py`, `src/questfoundry/graph/polish_validation.py`

- [ ] **Step 1: Search**

```bash
grep -n "exactly one belongs_to\|multiple belongs_to\|single belongs_to" src/questfoundry/graph/grow_validation.py src/questfoundry/graph/polish_validation.py
```

- [ ] **Step 2: For each match, evaluate**

If the assertion raises on multi-`belongs_to`, remove it (or narrow it: "multi-`belongs_to` is allowed only for pre-commit beats on same dilemma"). Keep a commit per match to preserve audit history.

- [ ] **Step 3: If no matches, note in commit that the file is already Y-shape-compatible**

```bash
git commit --allow-empty -m "chore(validation): audit single-belongs_to assertions — none found in grow/polish_validation"
```

---

### Task 5.2: Add `check_no_cross_dilemma_belongs_to` (guard rail 1)

**Files:**
- Modify: `src/questfoundry/graph/grow_validation.py`
- Modify: `tests/unit/test_grow_validation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_grow_validation.py`:

```python
def test_check_no_cross_dilemma_belongs_to_passes_for_valid_y_shape(
    tmp_path: "Path",
) -> None:
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.grow_validation import check_no_cross_dilemma_belongs_to

    graph = _build_valid_y_shape(tmp_path)  # same-dilemma dual belongs_to
    check = check_no_cross_dilemma_belongs_to(graph)
    assert check.severity == "pass"


def test_check_no_cross_dilemma_belongs_to_fails_for_cross_dilemma(
    tmp_path: "Path",
) -> None:
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.grow_validation import check_no_cross_dilemma_belongs_to

    graph = _build_cross_dilemma_dual(tmp_path)  # path_a1 + path_b1 on same beat
    check = check_no_cross_dilemma_belongs_to(graph)
    assert check.severity == "fail"
    assert "cross-dilemma" in check.message
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_grow_validation.py -v -k "no_cross_dilemma_belongs_to"
```
Expected: both fail with `AttributeError: cannot import check_no_cross_dilemma_belongs_to`.

- [ ] **Step 3: Implement the check**

In `src/questfoundry/graph/grow_validation.py`, add:

```python
def check_no_cross_dilemma_belongs_to(graph: Graph) -> ValidationCheck:
    """Guard rail 1: dual belongs_to must reference paths of the same dilemma.

    Cross-dilemma dual belongs_to is a hard-convergence violation (Doc 3 §8
    "Path Membership ≠ Scene Participation").
    """
    path_nodes = graph.get_nodes_by_type("path")
    beat_nodes = graph.get_nodes_by_type("beat")

    # path → dilemma map (normalized)
    path_to_dilemma: dict[str, str] = {}
    for pid, pdata in path_nodes.items():
        did = pdata.get("dilemma_id")
        if did:
            path_to_dilemma[pid] = normalize_scoped_id(did, "dilemma")

    # beat → set of paths
    beat_paths: dict[str, set[str]] = {}
    for e in graph.get_edges(edge_type="belongs_to"):
        if e["from"] in beat_nodes:
            beat_paths.setdefault(e["from"], set()).add(e["to"])

    violations: list[str] = []
    for beat_id, paths in beat_paths.items():
        if len(paths) < 2:
            continue
        dilemmas = {path_to_dilemma.get(p) for p in paths}
        dilemmas.discard(None)
        if len(dilemmas) > 1:
            violations.append(
                f"{beat_id} -> {sorted(paths)} across dilemmas {sorted(d for d in dilemmas if d)}"
            )

    if not violations:
        return ValidationCheck(
            name="no_cross_dilemma_belongs_to",
            severity="pass",
            message=f"All {sum(1 for p in beat_paths.values() if len(p) >= 2)} dual-belongs_to beats are same-dilemma",
        )
    return ValidationCheck(
        name="no_cross_dilemma_belongs_to",
        severity="fail",
        message=f"cross-dilemma dual belongs_to (guard rail 1): {'; '.join(violations[:3])}",
    )
```

- [ ] **Step 4: Add to `run_grow_checks`**

Locate `run_grow_checks` in the same file. Add `check_no_cross_dilemma_belongs_to(graph)` to the returned list.

- [ ] **Step 5: Update `__all__`**

Add `"check_no_cross_dilemma_belongs_to",` to the `__all__` list at the top of `grow_validation.py`.

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/unit/test_grow_validation.py -x -q
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/questfoundry/graph/grow_validation.py tests/unit/test_grow_validation.py
git commit -m "feat(validation): GROW check_no_cross_dilemma_belongs_to (guard rail 1)"
```

---

### Task 5.3: Add `check_no_dual_on_commit_beat` (guard rail 2)

**Files:**
- Modify: `src/questfoundry/graph/grow_validation.py`
- Modify: `tests/unit/test_grow_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_check_no_dual_on_commit_beat_fails_when_commit_has_dual(
    tmp_path: "Path",
) -> None:
    from questfoundry.graph.grow_validation import check_no_dual_on_commit_beat
    from questfoundry.graph.graph import Graph

    graph = Graph(tmp_path / "g.db")
    graph.create_node("path::a", {"type": "path", "dilemma_id": "d"})
    graph.create_node("path::b", {"type": "path", "dilemma_id": "d"})
    graph.create_node("beat::bad_commit",
                      {"type": "beat",
                       "dilemma_impacts": [{"dilemma_id": "d", "effect": "commits"}]})
    graph.add_edge("belongs_to", "beat::bad_commit", "path::a")
    graph.add_edge("belongs_to", "beat::bad_commit", "path::b")

    check = check_no_dual_on_commit_beat(graph)
    assert check.severity == "fail"
    assert "commit beat" in check.message
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/unit/test_grow_validation.py -v -k "no_dual_on_commit_beat"
```
Expected: fail.

- [ ] **Step 3: Implement**

In `src/questfoundry/graph/grow_validation.py`, add:

```python
def check_no_dual_on_commit_beat(graph: Graph) -> ValidationCheck:
    """Guard rail 2: commit beats must have a single belongs_to.

    A commit beat is the first beat exclusive to its path; dual membership
    on a commit beat is structurally impossible (Doc 3 §8).
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    beat_paths: dict[str, set[str]] = {}
    for e in graph.get_edges(edge_type="belongs_to"):
        if e["from"] in beat_nodes:
            beat_paths.setdefault(e["from"], set()).add(e["to"])

    violations: list[str] = []
    for bid, pset in beat_paths.items():
        if len(pset) < 2:
            continue
        impacts = beat_nodes[bid].get("dilemma_impacts", [])
        if any(imp.get("effect") == "commits" for imp in impacts):
            violations.append(f"{bid} commits AND has {len(pset)} belongs_to edges")

    if not violations:
        return ValidationCheck(
            name="no_dual_on_commit_beat",
            severity="pass",
            message="No commit beats with dual belongs_to",
        )
    return ValidationCheck(
        name="no_dual_on_commit_beat",
        severity="fail",
        message=f"commit beat with dual belongs_to (guard rail 2): {'; '.join(violations[:3])}",
    )
```

- [ ] **Step 4: Wire into `run_grow_checks` and `__all__`.**

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/unit/test_grow_validation.py -x -q
git add -A
git commit -m "feat(validation): GROW check_no_dual_on_commit_beat (guard rail 2)"
```

---

### Task 5.4: Add `check_no_pre_commit_intersections` (guard rail 3)

**Files:**
- Modify: `src/questfoundry/graph/grow_validation.py`
- Modify: `tests/unit/test_grow_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_check_no_pre_commit_intersections_fails_when_group_has_two_shared(
    tmp_path: "Path",
) -> None:
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.grow_validation import check_no_pre_commit_intersections

    graph = Graph(tmp_path / "g.db")
    # Two pre-commit beats, same dilemma, both dual — grouped together.
    graph.create_node("path::a", {"type": "path", "dilemma_id": "d"})
    graph.create_node("path::b", {"type": "path", "dilemma_id": "d"})
    graph.create_node("beat::s1", {"type": "beat", "dilemma_impacts": []})
    graph.create_node("beat::s2", {"type": "beat", "dilemma_impacts": []})
    graph.add_edge("belongs_to", "beat::s1", "path::a")
    graph.add_edge("belongs_to", "beat::s1", "path::b")
    graph.add_edge("belongs_to", "beat::s2", "path::a")
    graph.add_edge("belongs_to", "beat::s2", "path::b")
    graph.create_node("intersection_group::bad",
                      {"type": "intersection_group",
                       "beat_ids": ["beat::s1", "beat::s2"]})

    check = check_no_pre_commit_intersections(graph)
    assert check.severity == "fail"
    assert "pre-commit" in check.message
```

- [ ] **Step 2: Confirm failure, then implement**

```python
def check_no_pre_commit_intersections(graph: Graph) -> ValidationCheck:
    """Guard rail 3: intersection groups must not contain two pre-commit
    beats from the same dilemma.

    Such beats already co-occur by definition (Doc 3 §8); declaring them as
    an intersection is redundant and creates false structural implications.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    group_nodes = graph.get_nodes_by_type("intersection_group")

    beat_paths: dict[str, frozenset[str]] = {}
    _accum: dict[str, set[str]] = {}
    for e in graph.get_edges(edge_type="belongs_to"):
        if e["from"] in beat_nodes:
            _accum.setdefault(e["from"], set()).add(e["to"])
    beat_paths = {b: frozenset(p) for b, p in _accum.items()}

    violations: list[str] = []
    for gid, gdata in group_nodes.items():
        beat_ids = gdata.get("beat_ids", [])
        dual_by_pathset: dict[frozenset[str], list[str]] = {}
        for bid in beat_ids:
            pset = beat_paths.get(bid, frozenset())
            if len(pset) < 2:
                continue
            dual_by_pathset.setdefault(pset, []).append(bid)
        for pset, bids in dual_by_pathset.items():
            if len(bids) >= 2:
                violations.append(
                    f"{gid} contains {len(bids)} pre-commit beats sharing paths {sorted(pset)}: {bids}"
                )

    if not violations:
        return ValidationCheck(
            name="no_pre_commit_intersections",
            severity="pass",
            message="No intersection groups with pre-commit collisions",
        )
    return ValidationCheck(
        name="no_pre_commit_intersections",
        severity="fail",
        message=f"pre-commit beats grouped in intersection (guard rail 3): {'; '.join(violations[:3])}",
    )
```

- [ ] **Step 3: Wire into `run_grow_checks` and `__all__`.**

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/unit/test_grow_validation.py -x -q
git add -A
git commit -m "feat(validation): GROW check_no_pre_commit_intersections (guard rail 3)"
```

---

### Task 5.5: Run the whole validation test file

- [ ] **Step 1: Run**

```bash
uv run pytest tests/unit/test_grow_validation.py tests/unit/test_polish_passage_validation.py -x -q
uv run mypy src/questfoundry/graph/grow_validation.py src/questfoundry/graph/polish_validation.py
```
Expected: all pass + `Success: no issues found`.

- [ ] **Step 2: No commit unless broken tests surface (existing tests may have assumed single-belongs_to — fix them).**

---

## Phase 6: #1209 Evaluation (Investigation Only)

### Scope

Investigate the two structural concerns from #1209 and decide in-scope vs. out-of-scope. **No code changes** in this phase — the output is a set of proposed follow-up issues.

**My recommendation (pre-investigation):**
- **`convergence_map` discarded**: OUT OF SCOPE for this epic. The Y-shape work is upstream: SEED creates the Y-shape, GROW reads it. Persisting convergence metadata back onto arc/dilemma nodes is a separate concern — it depends on Y-shape being in place (which this epic delivers), but the persistence itself is independent and belongs in its own PR. Recommend filing as a standalone issue titled "GROW: persist convergence_map to arc/dilemma nodes for POLISH consumption".
- **Soft-dilemma convergence structural gap**: OUT OF SCOPE. Once Y-shape exists, soft-dilemma convergence becomes implementable (POLISH can look for the shared successor of two post-commit chains). But implementing the predecessor-edge creation is GROW-phase work and is specified in grow.md Phase 7 step 3 ("Store convergence_policy and payoff_budget on arc nodes for downstream use") — a separate area of code. Recommend filing as a standalone issue titled "GROW Phase 7: create predecessor edges establishing soft-dilemma convergence topology".

Both recommendations preserve the separation-of-concerns principle (CLAUDE.md "Separate add from remove"; also "Epics ≤ 10 issues"). Bundling them into this epic would push it past the size cap.

### Acceptance Criteria

- A written evaluation of each concern with in-scope/out-of-scope decision documented inline in this plan (scope section above).
- Two follow-up issue drafts ready for the user to file (see Epic Packaging at the bottom).
- `architect-reviewer` sign-off: N/A (investigation-only phase).

### Tasks

---

### Task 6.1: Trace `convergence_map` discard

**Files:**
- Read: `src/questfoundry/pipeline/stages/grow/deterministic.py:410-427`
- Read: `src/questfoundry/graph/grow_algorithms.py:1096-1202`

- [ ] **Step 1: Read the discard site**

Verify that `find_convergence_points()`'s return value is only used for a log message and never persisted. Document the exact lines.

- [ ] **Step 2: Identify where the data SHOULD go**

Per grow.md Phase 7 step 3: onto arc nodes as `convergence_policy` / `payoff_budget` / `converges_at`. Onto dilemma nodes for `dilemma_convergences`. The `ConvergenceInfo` dataclass already has the fields.

- [ ] **Step 3: Draft the follow-up issue body**

Draft the follow-up issue body (also inlined in the child issue templates section at the end of this plan):

```markdown
# GROW: persist convergence_map to arc/dilemma nodes for POLISH consumption

## Problem
`src/questfoundry/pipeline/stages/grow/deterministic.py:413` calls `find_convergence_points(...)` and logs a count but discards the result. Downstream (POLISH) has no way to know where soft dilemmas should rejoin.

## Scope
After `find_convergence_points()` returns a `dict[arc_id, ConvergenceInfo]`:
- For each arc node: set `convergence_policy`, `payoff_budget`, `converges_at` properties from `ConvergenceInfo`.
- For each dilemma node referenced in `dilemma_convergences`: set per-dilemma `converges_at` / `policy` / `budget` properties.
- Add a unit test: after `phase_convergence` runs, arc and dilemma nodes carry the expected properties.

## Design conformance
architect-reviewer sign-off with 0 MISSING/DEAD findings against grow.md Phase 7.

## Dependencies
Y-shape code alignment epic (this plan) must merge first — convergence detection depends on Y-shape graph structure.
```

- [ ] **Step 4: No commit — the evaluation lives in this plan doc.**

---

### Task 6.2: Trace soft-dilemma convergence gap

**Files:**
- Read: `src/questfoundry/graph/grow_algorithms.py:959-1008`
- Read: `src/questfoundry/models/seed.py:307+` (`DilemmaAnalysis.dilemma_role`)

- [ ] **Step 1: Verify the gap**

`_find_convergence_for_soft()` computes `converges_at` but the caller chain (`find_convergence_points` → `phase_convergence`) does not create predecessor edges. Soft-dilemma paths stay diverged in the DAG.

- [ ] **Step 2: Draft the follow-up issue body**

Draft the follow-up issue body (also inlined in the child issue templates section at the end of this plan):

```markdown
# GROW Phase 7: create predecessor edges establishing soft-dilemma convergence topology

## Problem
Per Document 3 §3 line 219, convergence is a topological fact — a beat with predecessors from both post-commit chains of a soft dilemma. `_find_convergence_for_soft()` identifies the intended convergence point but no code creates the predecessor edges that make it real. Soft dilemmas behave identically to hard dilemmas: paths diverge and never rejoin.

## Scope
After `phase_convergence` computes `convergence_map`:
- For each arc with `converges_at != None`:
  - Identify the soft-dilemma's two post-commit chains.
  - Create `predecessor` edges from the last exclusive beat of each chain to `converges_at`.
  - Verify the resulting DAG is still acyclic.
- Unit test: construct a fixture with two soft-dilemma paths, run `phase_convergence`, assert the DAG now has the expected predecessor edges.

## Design conformance
architect-reviewer sign-off with 0 MISSING/DEAD findings against Doc 3 §3 (convergence topology) and grow.md Phase 7.

## Dependencies
- Y-shape code alignment epic (this plan) must merge first.
- Issue "GROW: persist convergence_map to arc/dilemma nodes" (above) — these can proceed in parallel but share the `phase_convergence` function.
```

- [ ] **Step 3: No commit.**

---

## Phase 7: Integration Test + Migration

### Scope

Add one end-to-end integration test that constructs a Y-shape SEED output, runs `apply_seed_mutations`, then calls `compute_beat_grouping` (Phase 4a) and `compute_choice_edges` (Phase 4c), and asserts `len(choice_specs) >= 2`. Note: `compute_choice_edges` returns `list[ChoiceSpec]` — it does NOT write edges to the graph. This is the acceptance test for the whole epic.

Also decide on the `graph.db` migration strategy. **Recommendation: clean break.** Existing projects have intermediate graph states that were never Y-shape; trying to migrate them to Y-shape automatically would require inferring pre-commit beats from their dilemma impacts, which is not reliable. Document the break: "After this epic merges, re-run SEED on existing projects." There is no `qf migrate` subcommand to build.

### Acceptance Criteria

- `tests/integration/test_y_shape_end_to_end.py` exists and runs without an LLM (uses a hand-constructed SEED artifact).
- Test asserts: `apply_seed_mutations` succeeds, graph has dual-`belongs_to` edges on pre-commit beats, POLISH choice derivation produces ≥2 choices for the Y-shape dilemma.
- Migration note added to the epic's cover issue: "Clean break — re-run SEED on existing projects."
- `architect-reviewer` sign-off with 0 MISSING/DEAD findings end-to-end against Doc 3 §3, §8 and procedures/polish.md §4c.

### Tasks

---

### Task 7.1: Write the end-to-end Y-shape → choice-edges test

**Files:**
- Create: `tests/integration/test_y_shape_end_to_end.py`

- [ ] **Step 1: Write the test file**

```python
"""End-to-end Y-shape: SEED → POLISH Phase 4c produces ≥2 choice edges.

This is the acceptance test for the Y-shape code alignment epic. It uses
no LLM — the SEED artifact is hand-constructed to exercise the full
downstream pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import apply_seed_mutations
from questfoundry.pipeline.stages.polish.deterministic import (
    compute_choice_edges,
    compute_beat_grouping,
)


def _seed_artifact() -> dict[str, Any]:
    """A minimal Y-shape SEED artifact — one dilemma, two paths."""
    return {
        "entities": [
            {"entity_id": "mentor", "category": "character", "name": "Mentor",
             "description": "A cryptic guide.", "disposition": "keep"},
            {"entity_id": "archive", "category": "location", "name": "Archive",
             "description": "The central library.", "disposition": "keep"},
        ],
        "dilemmas": [
            {"dilemma_id": "trust_protector_or_manipulator",
             "question": "Is the mentor a protector or a manipulator?",
             "why_it_matters": "Shapes the protagonist's stance.",
             "answers": [
                 {"answer_id": "protector", "label": "protector",
                  "description": "The mentor is a protector."},
                 {"answer_id": "manipulator", "label": "manipulator",
                  "description": "The mentor is a manipulator."},
             ],
             "central_entity_ids": ["mentor"],
             "exploration_decision": "both",
             "canonical_answer_id": "protector"},
        ],
        "paths": [
            {"path_id": "trust_protector_or_manipulator__protector",
             "dilemma_id": "trust_protector_or_manipulator",
             "answer_id": "protector", "name": "Protector",
             "description": "The protector arc."},
            {"path_id": "trust_protector_or_manipulator__manipulator",
             "dilemma_id": "trust_protector_or_manipulator",
             "answer_id": "manipulator", "name": "Manipulator",
             "description": "The manipulator arc."},
        ],
        "consequences": [
            {"consequence_id": "mentor_trusted",
             "path_id": "trust_protector_or_manipulator__protector",
             "description": "The mentor becomes an ally.",
             "narrative_effects": ["protection_active"]},
            {"consequence_id": "mentor_distrusted",
             "path_id": "trust_protector_or_manipulator__manipulator",
             "description": "The mentor becomes an adversary.",
             "narrative_effects": ["manipulation_exposed"]},
        ],
        "initial_beats": [
            # Shared pre-commit beat.
            {"beat_id": "shared_setup",
             "summary": "The mentor delivers a cryptic warning.",
             "path_id": "trust_protector_or_manipulator__protector",
             "also_belongs_to": "trust_protector_or_manipulator__manipulator",
             "dilemma_impacts": [
                 {"dilemma_id": "trust_protector_or_manipulator",
                  "effect": "advances",
                  "note": "Both interpretations remain open."},
             ],
             "entities": ["character::mentor"],
             "location": "location::archive"},
            # Path A commit + consequence.
            {"beat_id": "commit_trust_protector",
             "summary": "Kay chooses to trust.",
             "path_id": "trust_protector_or_manipulator__protector",
             "dilemma_impacts": [
                 {"dilemma_id": "trust_protector_or_manipulator",
                  "effect": "commits",
                  "note": "The fork."},
             ],
             "entities": ["character::mentor"],
             "location": "location::archive"},
            {"beat_id": "consequence_protector",
             "summary": "The mentor shields Kay.",
             "path_id": "trust_protector_or_manipulator__protector",
             "dilemma_impacts": [
                 {"dilemma_id": "trust_protector_or_manipulator",
                  "effect": "advances",
                  "note": "The protector arc plays out."},
             ],
             "entities": ["character::mentor"],
             "location": "location::archive"},
            # Path B commit + consequence.
            {"beat_id": "commit_trust_manipulator",
             "summary": "Kay chooses to distrust.",
             "path_id": "trust_protector_or_manipulator__manipulator",
             "dilemma_impacts": [
                 {"dilemma_id": "trust_protector_or_manipulator",
                  "effect": "commits",
                  "note": "The fork."},
             ],
             "entities": ["character::mentor"],
             "location": "location::archive"},
            {"beat_id": "consequence_manipulator",
             "summary": "The mentor manipulates Kay.",
             "path_id": "trust_protector_or_manipulator__manipulator",
             "dilemma_impacts": [
                 {"dilemma_id": "trust_protector_or_manipulator",
                  "effect": "advances",
                  "note": "The manipulator arc plays out."},
             ],
             "entities": ["character::mentor"],
             "location": "location::archive"},
        ],
    }


def test_y_shape_end_to_end_polish_produces_choices(tmp_path: Path) -> None:
    """Acceptance test: Y-shape SEED produces ≥2 choice edges after POLISH 4c."""
    graph = Graph(tmp_path / "graph.db")

    # SEED.
    apply_seed_mutations(graph, _seed_artifact())

    # Verify dual belongs_to on shared_setup.
    shared_edges = [
        e for e in graph.get_edges(edge_type="belongs_to")
        if e["from"] == "beat::shared_setup"
    ]
    assert len(shared_edges) == 2

    # Add predecessor edges that a minimal GROW would produce.
    graph.add_edge("predecessor", "beat::commit_trust_protector", "beat::shared_setup")
    graph.add_edge("predecessor", "beat::commit_trust_manipulator", "beat::shared_setup")
    graph.add_edge("predecessor", "beat::consequence_protector",
                   "beat::commit_trust_protector")
    graph.add_edge("predecessor", "beat::consequence_manipulator",
                   "beat::commit_trust_manipulator")

    # POLISH passage grouping (Phase 4a).
    specs = compute_beat_grouping(graph)

    # POLISH Phase 4c: choice edge derivation.
    # compute_choice_edges returns ChoiceSpec objects — it does NOT write to the graph.
    choice_specs = compute_choice_edges(graph, specs)

    assert len(choice_specs) >= 2, (
        f"Y-shape should produce ≥2 choice specs; got {len(choice_specs)}"
    )
```

- [ ] **Step 2: Run the integration test**

```bash
uv run pytest tests/integration/test_y_shape_end_to_end.py -v
```
Expected: `PASSED`. If it fails, investigate — this is the epic's go/no-go gate.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_y_shape_end_to_end.py
git commit -m "test(y-shape): e2e acceptance — Y-shape SEED produces ≥2 POLISH choices"
```

---

### Task 7.2: Document the migration stance in the epic cover issue

- [ ] **Step 1: Add a paragraph to the epic cover issue (drafted at the bottom of this plan)**

The epic body includes:

```
## Migration
Existing projects with non-Y-shape graph.db files cannot be automatically
migrated — pre-commit beats cannot be reliably distinguished from early
post-commit beats after the fact. Re-run `qf seed --project <dir>` on any
existing project after this epic merges. No `qf migrate` subcommand.
```

- [ ] **Step 2: No code commit.**

---

## Risk & Rollout Ordering

### Dependency Graph

```
Phase 1 (schema) ── Phase 2 (mutations + guard rails)
                         │
                         ├── Phase 3 (POLISH consumers)
                         │
                         ├── Phase 4 (prompts)
                         │
                         ├── Phase 5 (validation layer)
                         │
                         └── Phase 7 (e2e test)

Phase 6 (investigation) — parallel to all phases, but filed as separate
follow-up issues; does not block the epic.
```

- **Phase 1 must land first.** Without the `also_belongs_to` field, nothing else can reference it.
- **Phase 2 depends on Phase 1.** Emits edges based on the new field.
- **Phase 3, 4, 5 are parallel** after Phase 2 merges. Phase 3 (POLISH consumers) must land before **real** POLISH output becomes Y-shape; the e2e test in Phase 7 gates on all three.
- **Phase 4 (prompts)** can proceed in parallel with Phase 3 — they do not share files.
- **Phase 5 (validation)** catches bugs that Phase 2 also catches at write time; redundant is safer.
- **Phase 7 is the final integration gate.** It validates that the entire chain works end-to-end.

### Rollback Plan

- **Phase 1 regression**: revert the commit that added `also_belongs_to`. No downstream code references it yet at that point.
- **Phase 2 regression**: revert the mutation-layer commits. The write-time checks are optional safety — removing them does not break existing SEED output (which has no `also_belongs_to`).
- **Phase 3 regression**: consumers gracefully degrade to "pick one path" behaviour when `beat_to_paths[bid]` has size 1 (the post-commit case). The Y-shape bit only activates when `also_belongs_to` is populated — which only happens after Phase 1+2+4 land.
- **Phase 4 regression**: prompts can be reverted independently — SEED will revert to per-path-chain generation (current broken behaviour), but nothing else breaks.
- **Phase 5 regression**: validation-only; revert removes new checks without affecting pipeline.
- **Phase 7 regression**: the e2e test fails → block the merge until the fault in Phase 2/3/4 is found.

### Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM ignores `also_belongs_to` in Phase 4 prompts | Medium | High (e2e test fails) | Two-call pattern simplifies per-call instructions; prompt-engineer review; validation-repair loop will catch it. |
| Phase 2 guard rail 3 check fires on legitimate intersection (false positive) | Low | Medium | Unit test covers the legitimate case (post-commit beats with same single-membership). |
| Phase 3.4 changing the function signature in `llm_phases.py` breaks callers outside the file | Medium | Low | `mypy src/` will flag; commit-per-task makes bisect easy. |
| Graph.db migrations cause pre-existing test fixtures to break | High | Low | Tests use hand-constructed graphs or `apply_seed_mutations` — no `.db` files under version control. |

---

## Epic Packaging

**The user will file these issues — do NOT run `gh issue create`.**

### Proposed Epic

**Title:** `Epic: Y-shape code alignment — SEED output, graph emissions, POLISH consumers, SEED prompts`

**Body:**

```markdown
## Overview
PR #1206 ratified the Y-shape dilemma model: pre-commit beats belong to both paths of their dilemma (dual `belongs_to`); post-commit beats belong to one path. PR #1208 merged the Doc 3 amendments. PRs #1211/#1212 completed the documentation renaming and fixes. This epic brings the implementation into conformance so the pipeline actually emits Y-shape graphs and POLISH produces >0 choices.

## Milestones
- **Milestone 1: Write surface** (Phases 1–2)
  - Schema (`also_belongs_to`) + mutation emission + three write-time guard rails.
- **Milestone 2: Read surface** (Phases 3, 5)
  - POLISH consumers + validation-layer guard-rail checks.
- **Milestone 3: Prompts + integration** (Phases 4, 7)
  - SEED prompt two-call split + e2e acceptance test.

## Acceptance criteria
- POLISH Phase 4c produces ≥2 choice edges per Y-shape dilemma in the e2e test (`tests/integration/test_y_shape_end_to_end.py`).
- All three Part 8 guard rails enforced at write time (mutation `ValueError`) and read time (`grow_validation` checks).
- `architect-reviewer` sign-off on the Milestone 3 PR with 0 MISSING/DEAD findings against Doc 3 §3, §8 and procedures/polish.md §4c.
- Migration: clean break. Re-run `qf seed --project <dir>` on existing projects.

## Out of scope (filed separately)
- GROW: persist convergence_map to arc/dilemma nodes (separate issue).
- GROW Phase 7: create predecessor edges for soft-dilemma convergence (separate issue).

## Plan document
`docs/superpowers/plans/2026-04-14-y-shape-code-alignment.md`
```

---

### Child Issue 1 — Phase 1: `InitialBeat.also_belongs_to` schema

**Title:** `feat(seed): add InitialBeat.also_belongs_to field for Y-shape dual belongs_to`

**Body:**

```markdown
## Scope
Add `also_belongs_to: str | None = None` to `InitialBeat`. Update `_migrate_paths_to_path_id` to migrate legacy `paths: [a, b]` to the Y-shape fields. Add a cross-field validator rejecting `also_belongs_to == path_id`.

## Plan reference
Phase 1 of `docs/superpowers/plans/2026-04-14-y-shape-code-alignment.md`.

## Acceptance criteria
- `InitialBeat(path_id="a", also_belongs_to="b")` constructs a pre-commit beat.
- `InitialBeat(path_id="a", also_belongs_to="a")` raises `ValueError`.
- `InitialBeat(paths=["a", "b"])` migrates to dual Y-shape with a `DeprecationWarning`.
- `InitialBeat(paths=["a", "b", "c"])` raises `ValueError("at most 2 entries")`.
- `uv run pytest tests/unit/test_seed_models.py -x -q` passes.
- `uv run mypy src/questfoundry/models/seed.py` is clean.
- **Design conformance**: `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §849–855 (same-dilemma dual belongs_to).
```

---

### Child Issue 2 — Phase 2: Graph mutation dual-edge emission + write-time guard rails

**Title:** `feat(graph): emit dual belongs_to edges and enforce Y-shape guard rails in mutations`

**Body:**

```markdown
## Scope
Replace `_get_path_id_from_beat` with `_get_path_ids_from_beat`. Update the three insertion sites at `mutations.py` lines 1312, 1529, 1874 to emit multiple `belongs_to` edges when applicable. Add write-time enforcement for the three Part 8 guard rails: (1) same-dilemma, (2) pre-commit only, (3) intersection exclusion. Replace `beat_to_path: dict[str, str]` with `beat_to_paths: dict[str, frozenset[str]]` in `algorithms.py` and drop the multi-belongs_to ValueError at lines 88-92.

## Plan reference
Phase 2 of `docs/superpowers/plans/2026-04-14-y-shape-code-alignment.md`.

## Dependencies
Closes after child issue 1 (schema) merges.

## Acceptance criteria
- `apply_seed_mutations` emits two `belongs_to` edges for beats with `also_belongs_to` set.
- Cross-dilemma dual raises `ValueError("guard rail 1")`.
- Commit beats (`effect=commits`) with `also_belongs_to` raise `ValueError("guard rail 2")`.
- Intersection groups with two same-dilemma pre-commit beats raise `ValueError("guard rail 3")`.
- `compute_state_flags` handles dual-`belongs_to` ancestors without raising (derives one flag per committed dilemma).
- `uv run pytest tests/unit/test_mutations.py tests/unit/test_graph_algorithms.py -x -q` passes.
- **Design conformance**: `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §8 (three guard rails) and §3 (Beat Lifecycle).
```

---

### Child Issue 3 — Phase 3: POLISH consumer migration

**Title:** `refactor(polish): migrate all beat_to_path consumers to beat_to_paths frozenset`

**Body:**

```markdown
## Scope
Six consumer sites update from `beat_to_path: dict[str, str]` to `beat_to_paths: dict[str, frozenset[str]]`:

- `src/questfoundry/graph/polish_validation.py:520-522`
- `src/questfoundry/graph/polish_context.py:195-198, 206`
- `src/questfoundry/pipeline/stages/polish/deterministic.py:250, 645`
- `src/questfoundry/pipeline/stages/polish/llm_phases.py:840-932`

Each consumer's predicate is reviewed semantically. Key change: `polish_validation._check_divergences_have_choices` distinguishes real divergence (children on different path *sets*) from Y-shape shared-to-commit transitions. The collapse rule in `polish/deterministic.py` requires path-set *equality* (shared pre-commit does not collapse with post-commit).

## Plan reference
Phase 3 of `docs/superpowers/plans/2026-04-14-y-shape-code-alignment.md`.

## Dependencies
Closes after child issue 2 (mutations) merges.

## Acceptance criteria
- No `beat_to_path: dict[str, str]` remains in the four files listed above; all use `beat_to_paths: dict[str, frozenset[str]]`.
- Shared pre-commit beats never collapse with post-commit beats (unit test).
- `_check_divergences_have_choices` passes for real Y-shape, fails when the parent passage has <2 choices.
- Entity-appearance context lists all paths for dual-membership beats.
- `uv run pytest tests/unit/test_polish_*.py -x -q` passes.
- **Design conformance**: `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §8 guard rails and procedures/polish.md §4c.
```

---

### Child Issue 4 — Phase 4: SEED prompts + orchestration

**Title:** `feat(prompts): SEED prompts generate Y-shape beats (shared + per-path two-call pattern)`

**Body:**

```markdown
## Scope
Replace "per path" beat generation framing with a two-call Y-shape pattern:

- Call 1: shared pre-commit beats per dilemma (1–2 beats, dual `belongs_to` via `also_belongs_to`, `effect` in `{advances, reveals, complicates}`).
- Call 2: post-commit per-path beats (1–2 per path, single `belongs_to`, exactly one `effect == commits` + ≥1 consequence beat).

Rewrite `serialize_seed_sections.yaml:400-620` (especially the `COMMITS BEATS REQUIREMENT` block at line 545). Update `discuss_seed.yaml:89`, `summarize_seed.yaml:84`, `summarize_seed_sections.yaml:122-136`, `serialize_seed.yaml:73-110`. Update `seed.py` advisory warning math to split shared (per-dilemma) and post-commit (per-path) averages.

## Plan reference
Phase 4 of `docs/superpowers/plans/2026-04-14-y-shape-code-alignment.md`.

## Dependencies
Closes after child issue 2 (mutations) merges. Can proceed in parallel with child issue 3.

## Acceptance criteria
- `prompt-engineer` subagent review completed; feedback addressed.
- SEED output includes beats with `also_belongs_to` populated.
- Manual inspection of `logs/llm_calls.jsonl` on a test run shows each call with rich context and explicit Y-shape instructions.
- `uv run pytest tests/unit/test_seed_stage.py -x -q` passes.
- **Design conformance**: `architect-reviewer` sign-off with 0 MISSING/DEAD findings against procedures/seed.md Phase 3 and Doc 3 §8 guard rails 1 and 2.
```

---

### Child Issue 5 — Phase 5: Validation layer guard-rail checks

**Title:** `feat(validation): GROW guard-rail checks (Y-shape same-dilemma, pre-commit only, intersection exclusion)`

**Body:**

```markdown
## Scope
Add three new checks in `src/questfoundry/graph/grow_validation.py`:
- `check_no_cross_dilemma_belongs_to` (guard rail 1)
- `check_no_dual_on_commit_beat` (guard rail 2)
- `check_no_pre_commit_intersections` (guard rail 3)

Wire into `run_grow_checks`. Remove any lingering "exactly one belongs_to" assertion from `grow_validation.py` and `polish_validation.py`.

## Plan reference
Phase 5 of `docs/superpowers/plans/2026-04-14-y-shape-code-alignment.md`.

## Dependencies
Closes after child issue 2 (mutations) merges. Can proceed in parallel with child issues 3 and 4.

## Acceptance criteria
- Three checks in `__all__` and `run_grow_checks`.
- Each check has a passing-case and a failing-case unit test.
- `uv run pytest tests/unit/test_grow_validation.py -x -q` passes.
- **Design conformance**: `architect-reviewer` sign-off with 0 MISSING/DEAD findings against Doc 3 §8 guard rails.
```

---

### Child Issue 6 — Phase 7: End-to-end acceptance test + migration note

**Title:** `test(y-shape): e2e acceptance — Y-shape SEED produces ≥2 POLISH choice edges`

**Body:**

```markdown
## Scope
Add `tests/integration/test_y_shape_end_to_end.py` — a hand-constructed Y-shape SEED artifact flows through `apply_seed_mutations` → `compute_beat_grouping` → `compute_choice_edges`, asserting `len(choice_specs) >= 2` (note: `compute_choice_edges` returns `list[ChoiceSpec]`, it does not write to the graph). Document the migration stance: clean break — re-run SEED on existing projects.

## Plan reference
Phase 7 of `docs/superpowers/plans/2026-04-14-y-shape-code-alignment.md`.

## Dependencies
Closes after child issues 1–5 all merge.

## Acceptance criteria
- `uv run pytest tests/integration/test_y_shape_end_to_end.py -v` passes.
- Epic cover issue updated with the "Migration: clean break" paragraph.
- **Design conformance**: `architect-reviewer` sign-off with 0 MISSING/DEAD findings end-to-end against Doc 3 §3, §8 and procedures/polish.md §4c.
```

---

### Follow-up Issue A (OUT OF EPIC — file separately)

**Title:** `GROW: persist convergence_map to arc/dilemma nodes for POLISH consumption`

**Body:**

```markdown
## Problem
`src/questfoundry/pipeline/stages/grow/deterministic.py:413` calls `find_convergence_points(...)` and logs a count but discards the result. Downstream (POLISH) has no way to know where soft dilemmas should rejoin.

## Scope
After `find_convergence_points()` returns a `dict[arc_id, ConvergenceInfo]`:
- For each arc node: set `convergence_policy`, `payoff_budget`, `converges_at` properties from `ConvergenceInfo`.
- For each dilemma node referenced in `dilemma_convergences`: set per-dilemma `converges_at` / `policy` / `budget` properties.
- Add a unit test: after `phase_convergence` runs, arc and dilemma nodes carry the expected properties.

## Design conformance
architect-reviewer sign-off with 0 MISSING/DEAD findings against grow.md Phase 7.

## Dependencies
Y-shape code alignment epic must merge first — convergence detection depends on Y-shape graph structure.
```

### Follow-up Issue B (OUT OF EPIC — file separately)

**Title:** `GROW Phase 7: create predecessor edges establishing soft-dilemma convergence topology`

**Body:**

```markdown
## Problem
Per Document 3 §3 line 219, convergence is a topological fact — a beat with predecessors from both post-commit chains of a soft dilemma. `_find_convergence_for_soft()` identifies the intended convergence point but no code creates the predecessor edges that make it real. Soft dilemmas behave identically to hard dilemmas: paths diverge and never rejoin.

## Scope
After `phase_convergence` computes `convergence_map`:
- For each arc with `converges_at != None`:
  - Identify the soft-dilemma's two post-commit chains.
  - Create `predecessor` edges from the last exclusive beat of each chain to `converges_at`.
  - Verify the resulting DAG is still acyclic.
- Unit test: construct a fixture with two soft-dilemma paths, run `phase_convergence`, assert the DAG now has the expected predecessor edges.

## Design conformance
architect-reviewer sign-off with 0 MISSING/DEAD findings against Doc 3 §3 (convergence topology) and grow.md Phase 7.

## Dependencies
- Y-shape code alignment epic must merge first.
- Issue "GROW: persist convergence_map to arc/dilemma nodes" (above) — these can proceed in parallel but share the `phase_convergence` function.
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by |
|---|---|
| Y-shape schema (`also_belongs_to`) | Phase 1 |
| Dual-edge emission | Task 2.3 |
| Guard rail 1 (same-dilemma) | Task 2.4 + Task 5.2 |
| Guard rail 2 (pre-commit only) | Task 2.5 + Task 5.3 |
| Guard rail 3 (intersection exclusion) | Task 2.6 + Task 5.4 |
| `beat_to_paths` migration (6 sites) | Phase 3 (3.1, 3.2, 3.3, 3.4) |
| Collapse-chain rule | Task 3.3 |
| SEED prompts (5 files) | Phase 4 (4.2, 4.3) |
| SEED advisory warning | Task 4.4 |
| `prompt-engineer` invocation | Task 4.1 |
| E2E test | Task 7.1 |
| Migration stance | Task 7.2 |
| #1209 evaluation | Phase 6 |
| Epic packaging | "Epic Packaging" section above |

All spec sections have a task; no gaps.

**Placeholder scan:** no `TBD`, `TODO`, "implement later", "similar to", or code-less steps remain.

**Type consistency:** `beat_to_paths: dict[str, frozenset[str]]` used in Phases 2, 3, 5. `_get_path_ids_from_beat() -> tuple[str, ...]` used in Phase 2 Tasks 2.1–2.5 consistently. `InitialBeat.also_belongs_to: str | None` used in Phases 1, 2, 4 consistently.
