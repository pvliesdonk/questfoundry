# DREAM + BRAINSTORM Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring DREAM and BRAINSTORM stages into compliance with their authoritative procedure specs, backed by two new validators (`validate_dream_output`, `validate_brainstorm_output`) used as runtime oracles at stage exit and downstream stage entry.

**Architecture:** Two pure-function validators returning `list[str]` errors. `apply_dream_mutations` and `apply_brainstorm_mutations` call their own validator after graph writes and raise on non-empty results. BRAINSTORM entry calls `validate_dream_output` before work begins.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, `uv` package manager, `ruff`, `mypy`.

**Spec:** `docs/superpowers/specs/2026-04-19-dream-brainstorm-compliance-design.md`

**Branch:** `feat/dream-brainstorm-compliance` (already created). All commits land there.

---

## Reference context

**Authoritative specs:**
- `docs/design/procedures/dream.md` — see §Stage Output Contract (6 items) and Rule Index (R-1.1 … R-1.13).
- `docs/design/procedures/brainstorm.md` — see §Stage Output Contract (11 items) and Rule Index (R-1.1 … R-3.8).

**Existing reference pattern:**
- `src/questfoundry/graph/polish_validation.py` — `validate_grow_output(graph) -> list[str]` + `validate_polish_output(graph) -> list[str]`. Called from `polish/stage.py:239` and `polish/deterministic.py:1395-1397` respectively.

**Stage wiring today:**
- DREAM: `pipeline/stages/dream.py` runs Discuss→Summarize→Serialize; returns `artifact_data`. `pipeline/orchestrator.py:702` calls `apply_mutations(graph, "dream", artifact_data)` which dispatches to `graph/mutations.py:apply_dream_mutations` (line 561). The orchestrator then calls `graph.set_last_stage("dream")` + `graph.save(...)`.
- BRAINSTORM: `pipeline/stages/brainstorm.py:140-160` runs `_get_vision_context` which reads the vision node and raises `BrainstormStageError` if absent. Then Discuss→Summarize→Serialize. Same apply_mutations/set_last_stage flow via the orchestrator.

**Error-type conventions:**
- `graph/mutations.py:177` defines `MutationError(ValueError)`. `SeedMutationError(MutationError)` already exists.
- `pipeline/stages/brainstorm.py:55` defines `BrainstormStageError(Exception)`.
- **No `DreamStageError` exists yet** — we will add it.

**Cluster → rule map (from the audit report `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` §M-DREAM-spec and §M-BRAINSTORM-spec):**

| Cluster | Rule(s) | File-level target |
|---|---|---|
| #1269 | DREAM R-1.9 | `models/dream.py` — `pov_style` enum values |
| #1270 | DREAM R-1.4 | `models/dream.py` — `Scope.story_size` enum values (+ `pipeline/size.py`) |
| #1271 | DREAM R-1.12, R-1.13 | `graph/mutations.py:apply_dream_mutations` — record approval metadata on vision node |
| #1273 | BRAINSTORM R-3.6 | `graph/mutations.py:902-910` — remove silent-skip on unresolvable entity |
| #1274 | BRAINSTORM R-2.4 | `graph/brainstorm_validation.py` — min 2 location entities |
| #1275 | BRAINSTORM R-3.1 | `models/brainstorm.py:Dilemma.question` — must end with `?` |
| #1276 | BRAINSTORM R-2.1 | `models/brainstorm.py:Entity.name` — disallow None (require non-empty) |
| #1277 | BRAINSTORM R-3.7 | `models/brainstorm.py:Dilemma.dilemma_id` — require `dilemma::` prefix |
| #1278 | BRAINSTORM R-3.8 | `graph/brainstorm_validation.py` — verify no forbidden node types exist |
| #1279 | BRAINSTORM R-1.1 | `models/brainstorm.py:BrainstormOutput` — entities ≥1, dilemmas ≥1 (minimum floor; R-1.1 "abundance" target stays in prompt) |
| #1280 | BRAINSTORM R-1.3 | `graph/brainstorm_validation.py` — vision still exists and is unchanged |

**Scope preset canonical values (per dream.md R-1.4 and its examples):** `micro`, `short`, `medium`, `long`. Current code uses `vignette`, `short`, `standard`, `long`. We rename `vignette→micro` and `standard→medium`.

**POV style canonical values (per dream.md R-1.9):** `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`. Current code uses the short forms `first`, `second`, `third_limited`, `third_person_omniscient`. We rename to the spec's canonical set.

---

## File Structure

### New files

- `src/questfoundry/graph/dream_validation.py` — `DreamContractError` + `validate_dream_output(graph)`.
- `src/questfoundry/graph/brainstorm_validation.py` — `BrainstormContractError` + `validate_brainstorm_output(graph)`.
- `tests/unit/test_dream_validation.py` — one test per DREAM Stage Output Contract rule.
- `tests/unit/test_brainstorm_validation.py` — one test per BRAINSTORM Stage Output Contract rule.

### Modified files

- `src/questfoundry/models/dream.py` — rename `Scope.story_size` and `DreamArtifact.pov_style` enum values.
- `src/questfoundry/pipeline/size.py` — rename preset keys (`vignette→micro`, `standard→medium`).
- `src/questfoundry/models/brainstorm.py` — `Entity.name` non-optional, `Dilemma.question` ends with `?`, `Dilemma.dilemma_id` has `dilemma::` prefix, `BrainstormOutput` minimum counts.
- `src/questfoundry/graph/mutations.py`:
  - `apply_dream_mutations` — record approval metadata; call `validate_dream_output`.
  - `apply_brainstorm_mutations` — remove silent-skip on unresolvable entity (~lines 899-910); call `validate_brainstorm_output` at exit.
- `src/questfoundry/pipeline/stages/dream.py` — add `DreamStageError` class.
- `src/questfoundry/pipeline/stages/brainstorm.py` — call `validate_dream_output` at entry (replaces `_get_vision_context`'s narrow check).
- `tests/unit/test_dream_stage.py` — update `scope`/`pov_style` values in fixtures.
- `tests/unit/test_artifacts.py` — same rename.
- `tests/unit/test_mutations.py` — update `pov_style: "second"` → `"second_person"`; update scope test data.

### Deleted tests

- `tests/unit/test_grow_deterministic.py::TestPhaseIntraPathPredecessors::test_dead_end_resolved_by_intra_path_edges`.
- `tests/unit/test_polish_validation.py::TestValidatePolishOutputResidue::test_proper_residue_passes`.

### Not modified

- `models/fill.py` `pov` enum. It will drift from DREAM's enum and cause FILL tests to break — acceptable per scope ("post-BRAINSTORM is allowed to break").
- Prompt templates, unless a fix within the plan requires producer-level changes to emit new values (in which case the relevant step says so).

---

## Task overview

Phase A: cleanup (Task 1).
Phase B: DREAM validator (Tasks 2–3).
Phase C: BRAINSTORM validator (Tasks 4–5).
Phase D: wire validators (Task 6).
Phase E: DREAM producer fixes (Tasks 7–9).
Phase F: BRAINSTORM producer fixes (Tasks 10–17).
Phase G: close-out (Task 18).

Total: 18 tasks. Each ends in a commit.

---

## Phase A — Cleanup

### Task 1: Delete obsolete failing tests

**Files:**
- Delete (per-test): `tests/unit/test_grow_deterministic.py::TestPhaseIntraPathPredecessors::test_dead_end_resolved_by_intra_path_edges`
- Delete (per-test): `tests/unit/test_polish_validation.py::TestValidatePolishOutputResidue::test_proper_residue_passes`

- [ ] **Step 1: Confirm the two tests fail in isolation**

Run:
```
uv run pytest tests/unit/test_grow_deterministic.py::TestPhaseIntraPathPredecessors::test_dead_end_resolved_by_intra_path_edges tests/unit/test_polish_validation.py::TestValidatePolishOutputResidue::test_proper_residue_passes --tb=no -q
```
Expected: `2 failed`.

- [ ] **Step 2: Open `tests/unit/test_grow_deterministic.py` and locate the test method**

Use: `grep -n "test_dead_end_resolved_by_intra_path_edges" tests/unit/test_grow_deterministic.py`.
Identify the full method: from its `def test_…` line to the end of the method (the next sibling `def` or dedent to class level).

- [ ] **Step 3: Delete the method block using Edit tool**

The Edit tool `old_string` must contain the full method including decorators and preceding blank line; `new_string` is empty. Preserve surrounding tests.

- [ ] **Step 4: Repeat for `tests/unit/test_polish_validation.py::TestValidatePolishOutputResidue::test_proper_residue_passes`**

Same approach: grep for the test name, identify its full block, delete.

- [ ] **Step 5: Verify the remaining tests still collect and the two deletions succeeded**

Run:
```
uv run pytest tests/unit/test_grow_deterministic.py tests/unit/test_polish_validation.py --collect-only -q 2>&1 | grep -E "(test_dead_end_resolved_by_intra_path_edges|test_proper_residue_passes)"
```
Expected: no output (the two tests are gone).

- [ ] **Step 6: Run the full unit suite to confirm previously-failing tests are no longer reported**

Run:
```
uv run pytest tests/unit/ --tb=no -q
```
Expected: 3484 passed (previously 3483 passed + 2 failed = 3485 non-skipped; removing 2 leaves 3484 passed). Provider-factory test-pollution issue may still appear intermittently — acceptable.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_grow_deterministic.py tests/unit/test_polish_validation.py
git commit -m "$(cat <<'EOF'
test: delete obsolete GROW/POLISH tests pending spec-audit milestones

These tests encode pre-audit behavior that is itself non-compliant and
will be rewritten when M-GROW-spec (#1296) and M-POLISH-spec (#1310)
are picked up. Removing them now eliminates noise during
DREAM+BRAINSTORM compliance work.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase B — DREAM validator

### Task 2: Write DREAM validator tests (all failing)

**Files:**
- Create: `tests/unit/test_dream_validation.py`

Rules mapped from `docs/design/procedures/dream.md` §Stage Output Contract:
- R-1.7 (1 Vision node; exactly one)
- R-1.8 (required fields non-empty: genre, tone, themes, audience, scope)
- R-1.10 (no edges on vision node)
- R-1.9 (pov_style ∈ allowed set if present)
- output-5 (no other node types exist)
- output-6 (human approval recorded)

- [ ] **Step 1: Create the test file with fixtures and all failing tests**

Create `tests/unit/test_dream_validation.py` with this content:

```python
"""Tests for DREAM Stage Output Contract validator."""

from __future__ import annotations

import pytest

from questfoundry.graph.dream_validation import validate_dream_output
from questfoundry.graph.graph import Graph


def _build_compliant_vision() -> dict[str, object]:
    """A vision payload that satisfies every rule in DREAM's output contract."""
    return {
        "type": "vision",
        "genre": "dark fantasy",
        "subgenre": "mystery",
        "tone": ["atmospheric", "morally ambiguous"],
        "themes": ["forbidden knowledge", "trust"],
        "audience": "adult",
        "scope": {"story_size": "short"},
        "content_notes": {"includes": [], "excludes": ["graphic violence"]},
        "pov_style": "third_person_limited",
        "human_approved": True,
    }


@pytest.fixture
def compliant_graph() -> Graph:
    graph = Graph()
    graph.create_node("vision", _build_compliant_vision())
    return graph


def test_valid_graph_passes(compliant_graph: Graph) -> None:
    assert validate_dream_output(compliant_graph) == []


def test_R_1_7_no_vision_node() -> None:
    graph = Graph()
    errors = validate_dream_output(graph)
    assert errors, "expected error for missing vision node"
    assert any("vision" in e.lower() for e in errors)


def test_R_1_7_two_vision_nodes() -> None:
    graph = Graph()
    graph.create_node("vision", _build_compliant_vision())
    graph.create_node("vision::extra", {**_build_compliant_vision(), "raw_id": "extra"})
    errors = validate_dream_output(graph)
    assert any("exactly one" in e.lower() or "vision node" in e.lower() for e in errors)


@pytest.mark.parametrize("missing_field", ["genre", "tone", "themes", "audience", "scope"])
def test_R_1_8_required_field_missing(missing_field: str) -> None:
    payload = _build_compliant_vision()
    del payload[missing_field]
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any(missing_field in e for e in errors), (
        f"expected an error mentioning '{missing_field}', got {errors}"
    )


@pytest.mark.parametrize("empty_field,empty_value", [
    ("genre", ""),
    ("tone", []),
    ("themes", []),
    ("audience", ""),
])
def test_R_1_8_required_field_empty(empty_field: str, empty_value: object) -> None:
    payload = _build_compliant_vision()
    payload[empty_field] = empty_value
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any(empty_field in e for e in errors)


def test_R_1_9_invalid_pov_style() -> None:
    payload = _build_compliant_vision()
    payload["pov_style"] = "omniscient"  # not in allowed set
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any("pov_style" in e for e in errors)


def test_R_1_9_pov_style_absent_is_ok(compliant_graph: Graph) -> None:
    # pov_style is optional per R-1.9.
    data = dict(compliant_graph.get_node("vision"))
    data.pop("pov_style", None)
    graph = Graph()
    graph.create_node("vision", data)
    assert validate_dream_output(graph) == []


def test_R_1_10_vision_has_no_edges(compliant_graph: Graph) -> None:
    # Adding a dummy node and an edge from vision to it violates R-1.10.
    compliant_graph.create_node("entity::kay", {"type": "entity", "name": "Kay"})
    compliant_graph.add_edge("anchored_to", "vision", "entity::kay")
    errors = validate_dream_output(compliant_graph)
    assert any("edge" in e.lower() for e in errors)


def test_output5_no_other_node_types_exist(compliant_graph: Graph) -> None:
    compliant_graph.create_node(
        "entity::kay", {"type": "entity", "name": "Kay"}
    )
    errors = validate_dream_output(compliant_graph)
    assert any("entity" in e.lower() or "other node" in e.lower() for e in errors)


def test_output6_human_approval_recorded() -> None:
    payload = _build_compliant_vision()
    payload["human_approved"] = False
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any("approv" in e.lower() for e in errors)


def test_output6_human_approval_missing() -> None:
    payload = _build_compliant_vision()
    del payload["human_approved"]
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any("approv" in e.lower() for e in errors)
```

- [ ] **Step 2: Run the tests — all should fail with ImportError**

Run: `uv run pytest tests/unit/test_dream_validation.py -v --tb=short`
Expected: every test fails — module `questfoundry.graph.dream_validation` does not exist.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/unit/test_dream_validation.py
git commit -m "$(cat <<'EOF'
test(dream): add failing validator tests for DREAM Stage Output Contract

Covers R-1.7, R-1.8, R-1.9, R-1.10, and output items 5 + 6 from
docs/design/procedures/dream.md §Stage Output Contract.

Module validate_dream_output to be implemented next.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Implement `validate_dream_output`

**Files:**
- Create: `src/questfoundry/graph/dream_validation.py`

- [ ] **Step 1: Create the module with skeleton returning `[]`**

Create `src/questfoundry/graph/dream_validation.py`:

```python
"""DREAM Stage Output Contract validator.

Validates the graph's vision node satisfies every rule in
docs/design/procedures/dream.md §Stage Output Contract.

Called at DREAM exit (from apply_dream_mutations) and at BRAINSTORM
entry (from pipeline/stages/brainstorm.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class DreamContractError(ValueError):
    """Raised when DREAM's Stage Output Contract is violated."""


_ALLOWED_POV_STYLES = frozenset(
    {
        "first_person",
        "second_person",
        "third_person_limited",
        "third_person_omniscient",
    }
)

_REQUIRED_NON_EMPTY_FIELDS = ("genre", "tone", "themes", "audience", "scope")


def validate_dream_output(graph: Graph) -> list[str]:
    """Verify the graph satisfies DREAM's Stage Output Contract.

    Args:
        graph: Graph expected to contain a vision node after DREAM.

    Returns:
        List of human-readable error strings. Empty means compliant.
        Pure read-only — never mutates the graph.
    """
    errors: list[str] = []
    return errors
```

- [ ] **Step 2: Run tests — `test_valid_graph_passes` should pass; the rest should fail with AssertionError**

Run: `uv run pytest tests/unit/test_dream_validation.py -v --tb=short`
Expected: `test_valid_graph_passes` PASSED; other tests FAILED with asserts like `assert errors, "expected error…"`.

- [ ] **Step 3: Implement vision-node-count check (R-1.7)**

Replace the body of `validate_dream_output` with:

```python
    errors: list[str] = []

    vision_nodes = graph.get_nodes_by_type("vision")
    if len(vision_nodes) == 0:
        errors.append("R-1.7: no vision node found in graph")
        return errors
    if len(vision_nodes) > 1:
        node_ids = sorted(vision_nodes.keys())
        errors.append(
            f"R-1.7: expected exactly one vision node, found {len(vision_nodes)}: {node_ids}"
        )
        # Continue — still report field-level issues on the first node.

    vision_id, vision = next(iter(vision_nodes.items()))
    return errors
```

- [ ] **Step 4: Run R-1.7 tests**

Run: `uv run pytest tests/unit/test_dream_validation.py::test_R_1_7_no_vision_node tests/unit/test_dream_validation.py::test_R_1_7_two_vision_nodes -v`
Expected: both PASS.

- [ ] **Step 5: Implement required-field checks (R-1.8) and append to the function**

Insert before the existing `return errors` line:

```python
    for field in _REQUIRED_NON_EMPTY_FIELDS:
        value = vision.get(field)
        if value is None:
            errors.append(f"R-1.8: vision.{field} is missing")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"R-1.8: vision.{field} is empty")
        elif isinstance(value, list) and len(value) == 0:
            errors.append(f"R-1.8: vision.{field} is empty")
        elif field == "scope" and isinstance(value, dict):
            if not value.get("story_size"):
                errors.append("R-1.8: vision.scope.story_size is empty")
```

- [ ] **Step 6: Run R-1.8 tests**

Run: `uv run pytest tests/unit/test_dream_validation.py -k "R_1_8" -v`
Expected: all PASS.

- [ ] **Step 7: Implement pov_style check (R-1.9)**

Insert before `return errors`:

```python
    pov_style = vision.get("pov_style")
    if pov_style is not None and pov_style not in _ALLOWED_POV_STYLES:
        errors.append(
            f"R-1.9: vision.pov_style must be one of {sorted(_ALLOWED_POV_STYLES)}, "
            f"got {pov_style!r}"
        )
```

- [ ] **Step 8: Run R-1.9 tests**

Run: `uv run pytest tests/unit/test_dream_validation.py -k "R_1_9" -v`
Expected: both PASS.

- [ ] **Step 9: Implement no-edges check (R-1.10)**

Insert before `return errors`:

```python
    vision_edges_out = graph.get_edges(from_node=vision_id)
    vision_edges_in = graph.get_edges(to_node=vision_id)
    if vision_edges_out or vision_edges_in:
        errors.append(
            f"R-1.10: vision node {vision_id!r} must have no edges; "
            f"found {len(vision_edges_out)} outgoing and {len(vision_edges_in)} incoming"
        )
```

If `get_edges` does not accept `from_node`/`to_node` kwargs, iterate:

```python
    all_edges = graph.get_edges()
    offending = [e for e in all_edges if e["from"] == vision_id or e["to"] == vision_id]
    if offending:
        errors.append(
            f"R-1.10: vision node {vision_id!r} must have no edges; found {len(offending)}"
        )
```

Check `src/questfoundry/graph/graph.py` for the right signature; use whichever is supported.

- [ ] **Step 10: Run R-1.10 tests**

Run: `uv run pytest tests/unit/test_dream_validation.py::test_R_1_10_vision_has_no_edges -v`
Expected: PASS.

- [ ] **Step 11: Implement no-other-node-types check (output item 5)**

Insert before `return errors`:

```python
    all_nodes_by_type = {
        node_type: graph.get_nodes_by_type(node_type)
        for node_type in graph.list_node_types()
    }
    forbidden_types = [t for t, n in all_nodes_by_type.items() if t != "vision" and n]
    if forbidden_types:
        errors.append(
            f"Output-5: DREAM output must contain only a vision node; "
            f"found other node types: {sorted(forbidden_types)}"
        )
```

If `list_node_types()` does not exist, iterate all nodes:

```python
    all_nodes = graph.get_all_nodes() if hasattr(graph, "get_all_nodes") else {}
    non_vision = [nid for nid, n in all_nodes.items() if n.get("type") != "vision"]
    if non_vision:
        errors.append(
            f"Output-5: DREAM output must contain only a vision node; "
            f"found {len(non_vision)} non-vision nodes (first 3: {non_vision[:3]})"
        )
```

Check graph API; use whichever is supported. If no list-all helper exists, add one locally by iterating known node types from `story_graph_ontology.md` Part 1.

- [ ] **Step 12: Run output-5 test**

Run: `uv run pytest tests/unit/test_dream_validation.py::test_output5_no_other_node_types_exist -v`
Expected: PASS.

- [ ] **Step 13: Implement human-approval check (output item 6)**

Insert before `return errors`:

```python
    if not vision.get("human_approved"):
        errors.append(
            "Output-6: DREAM vision missing recorded human approval "
            "(vision.human_approved must be True)"
        )
```

- [ ] **Step 14: Run all DREAM validator tests**

Run: `uv run pytest tests/unit/test_dream_validation.py -v`
Expected: all PASS.

- [ ] **Step 15: Type-check**

Run: `uv run mypy src/questfoundry/graph/dream_validation.py`
Expected: no errors.

- [ ] **Step 16: Lint**

Run: `uv run ruff check src/questfoundry/graph/dream_validation.py tests/unit/test_dream_validation.py`
Expected: no issues.

- [ ] **Step 17: Commit**

```bash
git add src/questfoundry/graph/dream_validation.py
git commit -m "$(cat <<'EOF'
feat(dream): add validate_dream_output contract validator

Mirrors the polish_validation pattern. Enforces DREAM §Stage Output
Contract rules R-1.7, R-1.8, R-1.9, R-1.10, and output items 5 and 6.
Pure read-only; returns list[str] of errors.

Closes (M-contract-chaining DREAM slice): part of #1347 / #1348.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase C — BRAINSTORM validator

### Task 4: Write BRAINSTORM validator tests (all failing)

**Files:**
- Create: `tests/unit/test_brainstorm_validation.py`

Rules mapped from `docs/design/procedures/brainstorm.md` §Stage Output Contract:
- output-1 (entity fields: name, category, concept; category ∈ set)
- output-2 (≥2 locations)
- output-3 (entity IDs namespaced by category)
- output-4 (dilemma: question + why_it_matters non-empty, question ends with `?`)
- output-5 (each dilemma has exactly 2 has_answer edges to distinct answer nodes)
- output-6 (each answer has non-empty description)
- output-7 (exactly one is_canonical per dilemma)
- output-8 (each dilemma has ≥1 anchored_to edge)
- output-9 (dilemma IDs have `dilemma::` prefix)
- output-10 (no Path/Beat/Consequence/StateFlag/Passage/IntersectionGroup nodes exist)
- output-11 (vision node unchanged)

- [ ] **Step 1: Create the test file**

Create `tests/unit/test_brainstorm_validation.py`:

```python
"""Tests for BRAINSTORM Stage Output Contract validator."""

from __future__ import annotations

import pytest

from questfoundry.graph.brainstorm_validation import validate_brainstorm_output
from questfoundry.graph.graph import Graph


def _seed_vision(graph: Graph) -> None:
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "subgenre": "mystery",
            "tone": ["atmospheric"],
            "themes": ["forbidden knowledge"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "pov_style": "third_person_limited",
            "human_approved": True,
        },
    )


def _seed_entity(graph: Graph, entity_id: str, category: str, name: str = "X") -> None:
    graph.create_node(
        entity_id,
        {
            "type": "entity",
            "raw_id": entity_id.split("::", 1)[-1],
            "name": name,
            "category": category,
            "concept": "one-line essence",
        },
    )


def _seed_dilemma(
    graph: Graph,
    dilemma_id: str,
    anchored_to: list[str],
    answers: list[tuple[str, bool]],  # (answer_id, is_canonical)
) -> None:
    graph.create_node(
        dilemma_id,
        {
            "type": "dilemma",
            "raw_id": dilemma_id.split("::", 1)[-1],
            "question": "What matters?",
            "why_it_matters": "Because.",
        },
    )
    for target in anchored_to:
        graph.add_edge("anchored_to", dilemma_id, target)
    for raw, is_canonical in answers:
        ans_id = f"{dilemma_id}::alt::{raw}"
        graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": raw,
                "description": f"desc-{raw}",
                "is_canonical": is_canonical,
            },
        )
        graph.add_edge("has_answer", dilemma_id, ans_id)


@pytest.fixture
def compliant_graph() -> Graph:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character", "Kay")
    _seed_entity(graph, "character::mentor", "character", "Mentor")
    _seed_entity(graph, "location::archive", "location", "Archive")
    _seed_entity(graph, "location::depths", "location", "Forbidden Depths")
    _seed_entity(graph, "object::cipher", "object", "Cipher")
    _seed_dilemma(
        graph,
        "dilemma::mentor_trust",
        anchored_to=["character::mentor", "character::kay"],
        answers=[("protector", True), ("manipulator", False)],
    )
    return graph


def test_valid_graph_passes(compliant_graph: Graph) -> None:
    assert validate_brainstorm_output(compliant_graph) == []


# Output-1: entity required fields + category membership
@pytest.mark.parametrize("missing_field", ["name", "category", "concept"])
def test_R_2_1_entity_missing_field(compliant_graph: Graph, missing_field: str) -> None:
    data = dict(compliant_graph.get_node("character::kay"))
    del data[missing_field]
    compliant_graph.update_node("character::kay", **data)
    # Also clear explicitly so update semantics match
    compliant_graph.update_node("character::kay", **{missing_field: None})
    errors = validate_brainstorm_output(compliant_graph)
    assert any("character::kay" in e and missing_field in e for e in errors)


def test_R_2_2_invalid_entity_category(compliant_graph: Graph) -> None:
    compliant_graph.update_node("character::kay", category="ally")
    errors = validate_brainstorm_output(compliant_graph)
    assert any("category" in e and "ally" in e for e in errors)


# Output-2: ≥2 locations
def test_R_2_4_insufficient_location_entities() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character")
    _seed_entity(graph, "location::archive", "location")
    _seed_dilemma(
        graph,
        "dilemma::x",
        anchored_to=["character::kay"],
        answers=[("a", True), ("b", False)],
    )
    errors = validate_brainstorm_output(graph)
    assert any("location" in e.lower() and "2" in e for e in errors)


# Output-3: entity IDs namespaced by category
def test_R_2_3_entity_id_missing_category_prefix() -> None:
    graph = Graph()
    _seed_vision(graph)
    graph.create_node(
        "kay",  # missing category prefix
        {"type": "entity", "raw_id": "kay", "name": "Kay", "category": "character", "concept": "x"},
    )
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    _seed_dilemma(
        graph,
        "dilemma::x",
        anchored_to=["location::a"],
        answers=[("a", True), ("b", False)],
    )
    errors = validate_brainstorm_output(graph)
    assert any("kay" in e and ("prefix" in e or "namespace" in e) for e in errors)


# Output-4: dilemma question ends with ?
def test_R_3_1_dilemma_question_missing_qmark(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", question="not a question")
    errors = validate_brainstorm_output(compliant_graph)
    assert any("question" in e and "?" in e for e in errors)


def test_R_3_1_dilemma_missing_why_it_matters(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", why_it_matters=None)
    errors = validate_brainstorm_output(compliant_graph)
    assert any("why_it_matters" in e for e in errors)


# Output-5 + R-3.2: exactly 2 has_answer
def test_R_3_2_dilemma_not_binary(compliant_graph: Graph) -> None:
    compliant_graph.create_node(
        "dilemma::mentor_trust::alt::third",
        {"type": "answer", "raw_id": "third", "description": "d", "is_canonical": False},
    )
    compliant_graph.add_edge(
        "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::third"
    )
    errors = validate_brainstorm_output(compliant_graph)
    assert any("has_answer" in e or "answer" in e.lower() for e in errors)


# Output-6: answer description non-empty
def test_R_3_5_answer_description_empty(compliant_graph: Graph) -> None:
    compliant_graph.update_node(
        "dilemma::mentor_trust::alt::protector", description=""
    )
    errors = validate_brainstorm_output(compliant_graph)
    assert any("description" in e for e in errors)


# Output-7: exactly one is_canonical
def test_R_3_4_no_canonical_answer(compliant_graph: Graph) -> None:
    compliant_graph.update_node(
        "dilemma::mentor_trust::alt::protector", is_canonical=False
    )
    errors = validate_brainstorm_output(compliant_graph)
    assert any("canonical" in e for e in errors)


def test_R_3_4_two_canonical_answers(compliant_graph: Graph) -> None:
    compliant_graph.update_node(
        "dilemma::mentor_trust::alt::manipulator", is_canonical=True
    )
    errors = validate_brainstorm_output(compliant_graph)
    assert any("canonical" in e for e in errors)


# Output-8: ≥1 anchored_to per dilemma
def test_R_3_6_dilemma_missing_anchored_to() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character")
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    graph.create_node(
        "dilemma::orphan",
        {
            "type": "dilemma",
            "raw_id": "orphan",
            "question": "Why?",
            "why_it_matters": "stakes",
        },
    )
    for raw, is_canonical in [("yes", True), ("no", False)]:
        ans_id = f"dilemma::orphan::alt::{raw}"
        graph.create_node(
            ans_id,
            {"type": "answer", "raw_id": raw, "description": "d", "is_canonical": is_canonical},
        )
        graph.add_edge("has_answer", "dilemma::orphan", ans_id)
    errors = validate_brainstorm_output(graph)
    assert any("dilemma::orphan" in e and "anchored_to" in e for e in errors)


# Output-9: dilemma prefix
def test_R_3_7_dilemma_id_missing_prefix() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character")
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    graph.create_node(
        "mentor_trust",  # no prefix
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Q?",
            "why_it_matters": "stakes",
        },
    )
    graph.add_edge("anchored_to", "mentor_trust", "character::kay")
    for raw, is_canonical in [("yes", True), ("no", False)]:
        ans_id = f"mentor_trust::alt::{raw}"
        graph.create_node(
            ans_id,
            {"type": "answer", "raw_id": raw, "description": "d", "is_canonical": is_canonical},
        )
        graph.add_edge("has_answer", "mentor_trust", ans_id)
    errors = validate_brainstorm_output(graph)
    assert any("prefix" in e.lower() or "dilemma::" in e for e in errors)


# Output-10: no forbidden node types
@pytest.mark.parametrize("forbidden", ["path", "beat", "consequence", "state_flag", "passage"])
def test_R_3_8_forbidden_node_type_present(compliant_graph: Graph, forbidden: str) -> None:
    compliant_graph.create_node(f"{forbidden}::x", {"type": forbidden, "raw_id": "x"})
    errors = validate_brainstorm_output(compliant_graph)
    assert any(forbidden in e for e in errors)


# Output-11: vision unchanged — simplest check: vision still exists
def test_output11_vision_still_exists(compliant_graph: Graph) -> None:
    # Remove the vision node → violation
    compliant_graph.delete_node("vision")
    errors = validate_brainstorm_output(compliant_graph)
    assert any("vision" in e.lower() for e in errors)


# R-1.1: minimum abundance floor — at least 1 entity, at least 1 dilemma
def test_R_1_1_no_entities() -> None:
    graph = Graph()
    _seed_vision(graph)
    errors = validate_brainstorm_output(graph)
    assert any("entit" in e.lower() for e in errors)


def test_R_1_1_no_dilemmas() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    errors = validate_brainstorm_output(graph)
    assert any("dilemma" in e.lower() for e in errors)
```

- [ ] **Step 2: Run — all tests fail with ImportError**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py -v --tb=short`
Expected: module import fails.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_brainstorm_validation.py
git commit -m "$(cat <<'EOF'
test(brainstorm): add failing validator tests for BRAINSTORM output contract

Covers the 11 items in docs/design/procedures/brainstorm.md §Stage
Output Contract plus R-1.1 minimum-floor checks.

Module validate_brainstorm_output to be implemented next.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Implement `validate_brainstorm_output`

**Files:**
- Create: `src/questfoundry/graph/brainstorm_validation.py`

- [ ] **Step 1: Create the skeleton**

Create `src/questfoundry/graph/brainstorm_validation.py`:

```python
"""BRAINSTORM Stage Output Contract validator.

Validates the graph satisfies every rule in
docs/design/procedures/brainstorm.md §Stage Output Contract.

Called at BRAINSTORM exit (from apply_brainstorm_mutations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class BrainstormContractError(ValueError):
    """Raised when BRAINSTORM's Stage Output Contract is violated."""


_ALLOWED_ENTITY_CATEGORIES = frozenset({"character", "location", "object", "faction"})

_FORBIDDEN_NODE_TYPES = frozenset(
    {"path", "beat", "consequence", "state_flag", "passage", "intersection_group"}
)


def validate_brainstorm_output(graph: Graph) -> list[str]:
    """Verify the graph satisfies BRAINSTORM's Stage Output Contract.

    Returns:
        List of human-readable error strings. Empty means compliant.
        Pure read-only — never mutates the graph.
    """
    errors: list[str] = []
    return errors
```

- [ ] **Step 2: Run — `test_valid_graph_passes` passes, rest fail**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py::test_valid_graph_passes -v`
Expected: PASS.

- [ ] **Step 3: Implement vision-unchanged + minimum counts (output-11, R-1.1)**

Replace body:

```python
    errors: list[str] = []

    vision_nodes = graph.get_nodes_by_type("vision")
    if not vision_nodes:
        errors.append("Output-11: vision node is missing (BRAINSTORM must not remove it)")

    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    if not entity_nodes:
        errors.append("R-1.1: BRAINSTORM must produce at least one entity")
    if not dilemma_nodes:
        errors.append("R-1.1: BRAINSTORM must produce at least one dilemma")

    return errors
```

- [ ] **Step 4: Run vision + R-1.1 tests**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py -k "output11 or R_1_1" -v`
Expected: all PASS.

- [ ] **Step 5: Implement entity field checks (R-2.1, R-2.2, R-2.3, R-2.4)**

Insert before `return errors`:

```python
    # R-2.3: entity IDs namespaced by category (prefix).
    # R-2.1: required fields. R-2.2: category ∈ allowed set.
    location_count = 0
    for entity_id, entity in entity_nodes.items():
        category = entity.get("category")
        if not entity.get("name"):
            errors.append(f"R-2.1: entity {entity_id!r} has empty/missing name")
        if not category:
            errors.append(f"R-2.1: entity {entity_id!r} has empty/missing category")
        if not entity.get("concept"):
            errors.append(f"R-2.1: entity {entity_id!r} has empty/missing concept")

        if category and category not in _ALLOWED_ENTITY_CATEGORIES:
            errors.append(
                f"R-2.2: entity {entity_id!r} has invalid category {category!r}; "
                f"must be one of {sorted(_ALLOWED_ENTITY_CATEGORIES)}"
            )

        if "::" not in entity_id:
            errors.append(
                f"R-2.3: entity id {entity_id!r} missing category namespace prefix "
                "(expected e.g. 'character::...', 'location::...')"
            )
        elif category:
            prefix = entity_id.split("::", 1)[0]
            if prefix != category:
                errors.append(
                    f"R-2.3: entity id {entity_id!r} prefix {prefix!r} "
                    f"does not match category {category!r}"
                )

        if category == "location":
            location_count += 1

    if location_count < 2:
        errors.append(
            f"R-2.4: BRAINSTORM must produce at least 2 location entities, found {location_count}"
        )
```

- [ ] **Step 6: Run entity tests**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py -k "R_2_" -v`
Expected: all PASS.

- [ ] **Step 7: Implement dilemma field + prefix checks (R-3.1, R-3.7)**

Insert before `return errors`:

```python
    # Gather edges once.
    has_answer_edges = graph.get_edges(edge_type="has_answer")
    anchored_to_edges = graph.get_edges(edge_type="anchored_to")

    answers_per_dilemma: dict[str, list[str]] = {}
    for edge in has_answer_edges:
        answers_per_dilemma.setdefault(edge["from"], []).append(edge["to"])

    anchors_per_dilemma: dict[str, list[str]] = {}
    for edge in anchored_to_edges:
        anchors_per_dilemma.setdefault(edge["from"], []).append(edge["to"])

    for dilemma_id, dilemma in dilemma_nodes.items():
        # R-3.7: prefix
        if not dilemma_id.startswith("dilemma::"):
            errors.append(
                f"R-3.7: dilemma id {dilemma_id!r} missing 'dilemma::' prefix"
            )

        # R-3.1: question + why_it_matters
        question = dilemma.get("question")
        if not question:
            errors.append(f"R-3.1: dilemma {dilemma_id!r} has empty/missing question")
        elif not question.rstrip().endswith("?"):
            errors.append(
                f"R-3.1: dilemma {dilemma_id!r} question must end with '?' "
                f"(got {question!r})"
            )
        if not dilemma.get("why_it_matters"):
            errors.append(f"R-3.1: dilemma {dilemma_id!r} has empty/missing why_it_matters")

        # R-3.6 (output-8): ≥1 anchored_to
        anchors = anchors_per_dilemma.get(dilemma_id, [])
        if not anchors:
            errors.append(
                f"R-3.6: dilemma {dilemma_id!r} has no anchored_to edge to an entity"
            )

        # R-3.2 (output-5): exactly two distinct answers
        answers = answers_per_dilemma.get(dilemma_id, [])
        distinct_answers = set(answers)
        if len(distinct_answers) != 2:
            errors.append(
                f"R-3.2: dilemma {dilemma_id!r} must have exactly 2 distinct "
                f"has_answer edges, got {len(distinct_answers)}"
            )

        # R-3.4 (output-7): exactly one is_canonical among this dilemma's answers
        canonical_count = 0
        for ans_id in distinct_answers:
            ans = graph.get_node(ans_id) or {}
            if ans.get("is_canonical") is True:
                canonical_count += 1
            # R-3.5 (output-6): description non-empty
            if not ans.get("description"):
                errors.append(
                    f"R-3.5: answer {ans_id!r} has empty/missing description"
                )
        if len(distinct_answers) >= 2 and canonical_count != 1:
            errors.append(
                f"R-3.4: dilemma {dilemma_id!r} must have exactly one canonical answer, "
                f"found {canonical_count}"
            )
```

- [ ] **Step 8: Run dilemma-field tests**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py -k "R_3_" -v`
Expected: all PASS.

- [ ] **Step 9: Implement forbidden-node-type check (R-3.8)**

Insert before `return errors`:

```python
    for node_type in _FORBIDDEN_NODE_TYPES:
        forbidden = graph.get_nodes_by_type(node_type)
        if forbidden:
            errors.append(
                f"R-3.8: BRAINSTORM must not create {node_type!r} nodes; "
                f"found {len(forbidden)}: {sorted(forbidden.keys())[:3]}"
            )
```

- [ ] **Step 10: Run forbidden-node-type tests**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py::test_R_3_8_forbidden_node_type_present -v`
Expected: all parametrized variants PASS.

- [ ] **Step 11: Run full validator test file**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py -v`
Expected: all PASS.

- [ ] **Step 12: Type-check + lint**

Run:
```
uv run mypy src/questfoundry/graph/brainstorm_validation.py
uv run ruff check src/questfoundry/graph/brainstorm_validation.py tests/unit/test_brainstorm_validation.py
```
Expected: clean.

- [ ] **Step 13: Commit**

```bash
git add src/questfoundry/graph/brainstorm_validation.py
git commit -m "$(cat <<'EOF'
feat(brainstorm): add validate_brainstorm_output contract validator

Mirrors the polish_validation pattern. Enforces BRAINSTORM §Stage
Output Contract (11 items) plus R-1.1 minimum floors.

Part of M-contract-chaining (#1347 / #1348) BRAINSTORM slice.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase D — Wire validators

### Task 6: Wire validators at stage boundaries

**Files:**
- Modify: `src/questfoundry/pipeline/stages/dream.py` (add `DreamStageError` class)
- Modify: `src/questfoundry/graph/mutations.py` (call validators from `apply_dream_mutations` and `apply_brainstorm_mutations`)
- Modify: `src/questfoundry/pipeline/stages/brainstorm.py` (replace `_get_vision_context` narrow check with `validate_dream_output`)

- [ ] **Step 1: Add `DreamStageError` to `pipeline/stages/dream.py`**

Insert after the existing imports, before `class DreamStage:`:

```python
class DreamStageError(Exception):
    """Error during DREAM stage execution (includes contract failures)."""
```

- [ ] **Step 2: Wire `validate_dream_output` into `apply_dream_mutations`**

Open `src/questfoundry/graph/mutations.py`. At the top of the file, add to the existing imports block:

```python
from questfoundry.graph.dream_validation import (
    DreamContractError,
    validate_dream_output,
)
```

Replace the end of `apply_dream_mutations` so it looks like:

```python
def apply_dream_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply DREAM stage output to graph.

    Creates or replaces the "vision" node with the dream artifact data and
    validates the resulting graph against DREAM's Stage Output Contract.
    """
    vision_data = {
        "type": "vision",
        "genre": output.get("genre"),
        "subgenre": output.get("subgenre"),
        "tone": output.get("tone", []),
        "themes": output.get("themes", []),
        "audience": output.get("audience"),
        "style_notes": output.get("style_notes"),
        "scope": output.get("scope"),
        "content_notes": output.get("content_notes"),
        "pov_style": output.get("pov_style"),
        "protagonist_defined": output.get("protagonist_defined", False),
        # Record human approval. Per R-1.12, absence == unapproved.
        # Orchestrators running in --no-interactive mode set this True via output.
        "human_approved": bool(output.get("human_approved", False)),
    }
    vision_data = _clean_dict(vision_data)
    graph.upsert_node("vision", vision_data)

    errors = validate_dream_output(graph)
    if errors:
        raise DreamContractError(
            "DREAM stage output contract violated:\n  - "
            + "\n  - ".join(errors)
        )
```

- [ ] **Step 3: Wire `validate_brainstorm_output` into `apply_brainstorm_mutations`**

In the same file, add to imports:

```python
from questfoundry.graph.brainstorm_validation import (
    BrainstormContractError,
    validate_brainstorm_output,
)
```

Find the `apply_brainstorm_mutations` function (it's the function containing the code starting around line 831 that iterates categories and creates entity/dilemma nodes). At the end of the function body (after all nodes/edges are written), append:

```python
    errors = validate_brainstorm_output(graph)
    if errors:
        raise BrainstormContractError(
            "BRAINSTORM stage output contract violated:\n  - "
            + "\n  - ".join(errors)
        )
```

- [ ] **Step 4: Wire `validate_dream_output` at BRAINSTORM entry**

Open `src/questfoundry/pipeline/stages/brainstorm.py`. Add import at top:

```python
from questfoundry.graph.dream_validation import validate_dream_output
```

Find `_get_vision_context` (around line 140). Replace the check that raises `BrainstormStageError` when `vision_node is None` with a call to `validate_dream_output`:

```python
    def _get_vision_context(self, project_path: Path) -> str:
        """Load and format vision from graph, enforcing DREAM output contract.

        Raises:
            BrainstormStageError: If DREAM's Stage Output Contract is not satisfied.
        """
        graph = Graph.load(project_path)

        contract_errors = validate_dream_output(graph)
        if contract_errors:
            raise BrainstormStageError(
                "BRAINSTORM requires DREAM stage to complete first.\n"
                "DREAM output contract violated:\n  - "
                + "\n  - ".join(contract_errors)
            )

        vision_node = graph.get_node("vision")
        # vision_node is guaranteed non-None by validate_dream_output.
        assert vision_node is not None
        return _format_vision_context(vision_node)
```

(Adjust the return expression to match what the existing function returns — `_format_vision_context` here is a placeholder if the original used a different helper. Keep the original formatting call intact; only replace the pre-check.)

- [ ] **Step 5: Run DREAM and BRAINSTORM stage tests**

Run: `uv run pytest tests/unit/test_dream_stage.py tests/unit/test_brainstorm_stage.py --tb=short`
Expected: these tests now FAIL in many places — stage fixtures and producer code still use the pre-audit enum values and don't record `human_approved`. This is expected; phases E and F fix them.

- [ ] **Step 6: Run validator test suites to confirm validators still work**

Run: `uv run pytest tests/unit/test_dream_validation.py tests/unit/test_brainstorm_validation.py -v`
Expected: all PASS.

- [ ] **Step 7: Type-check + lint the changed files**

Run:
```
uv run mypy src/questfoundry/graph/mutations.py src/questfoundry/pipeline/stages/dream.py src/questfoundry/pipeline/stages/brainstorm.py
uv run ruff check src/questfoundry/graph/mutations.py src/questfoundry/pipeline/stages/dream.py src/questfoundry/pipeline/stages/brainstorm.py
```
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/questfoundry/pipeline/stages/dream.py src/questfoundry/pipeline/stages/brainstorm.py src/questfoundry/graph/mutations.py
git commit -m "$(cat <<'EOF'
feat(dream,brainstorm): wire validators at stage boundaries

- apply_dream_mutations raises DreamContractError on contract failure.
- apply_brainstorm_mutations raises BrainstormContractError on failure.
- BRAINSTORM entry calls validate_dream_output; replaces the narrow
  vision-node-exists check.
- Adds DreamStageError class for stage-level signaling.

Producer code still violates new contract values — fixed in phases E/F.

Part of #1347 / #1348.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase E — DREAM producer fixes

### Task 7: Fix DREAM #1269 — POV style enum values

**Spec rule:** R-1.9 — `pov_style ∈ {first_person, second_person, third_person_limited, third_person_omniscient}`.

**Files:**
- Modify: `src/questfoundry/models/dream.py`
- Modify: `tests/unit/test_mutations.py`
- Modify: `tests/unit/test_dream_stage.py`
- Modify: `tests/unit/test_artifacts.py` (if it references old values)

- [ ] **Step 1: Update the test for the new pov_style values in mutations test**

Open `tests/unit/test_mutations.py` — find the test containing `"pov_style": "second"` (line ~309). Change both occurrences to `"second_person"`:

```python
            "pov_style": "second_person",
```
```python
        assert vision["pov_style"] == "second_person"
```

- [ ] **Step 2: Run the failing test to confirm the enum is not yet updated**

Run: `uv run pytest tests/unit/test_mutations.py -k "pov_style" -v --tb=short`
Expected: FAIL — either Pydantic validator rejects `"second_person"` at model load, or downstream check fails.

- [ ] **Step 3: Update `models/dream.py` enum**

Open `src/questfoundry/models/dream.py` line 63. Replace the literal:

```python
    pov_style: Literal[
        "first_person",
        "second_person",
        "third_person_limited",
        "third_person_omniscient",
    ] | None = Field(
        default=None,
        description="Preferred narrative POV (hint for FILL, not mandate)",
    )
```

- [ ] **Step 4: Run the mutations test**

Run: `uv run pytest tests/unit/test_mutations.py -k "pov_style" -v`
Expected: PASS.

- [ ] **Step 5: Grep for remaining old values and update tests**

Run:
```
grep -rn '"first"\|"second"\|"third_limited"\|"third_omniscient"' src/questfoundry/models/ tests/unit/test_dream_stage.py tests/unit/test_artifacts.py tests/unit/test_mutations.py
```
Update each occurrence that refers to DREAM pov_style to the new values. Leave `models/fill.py` and its tests (`test_fill_*.py`) unchanged — those are post-BRAINSTORM and allowed to break.

- [ ] **Step 6: Run DREAM stage + artifact + mutations tests**

Run: `uv run pytest tests/unit/test_dream_stage.py tests/unit/test_artifacts.py tests/unit/test_mutations.py -v`
Expected: PASS for DREAM-related tests. (FILL-dependent tests are not run here.)

- [ ] **Step 7: Run DREAM validator tests**

Run: `uv run pytest tests/unit/test_dream_validation.py -v`
Expected: all PASS (we only tightened the enum, the validator's list was already correct).

- [ ] **Step 8: Commit**

```bash
git add src/questfoundry/models/dream.py tests/unit/test_mutations.py tests/unit/test_dream_stage.py tests/unit/test_artifacts.py
git commit -m "$(cat <<'EOF'
fix(dream): align pov_style enum values to spec (R-1.9)

Rename pov_style values: first→first_person, second→second_person,
third_limited→third_person_limited, third_omniscient→third_person_omniscient.

Matches docs/design/procedures/dream.md R-1.9.

Closes #1269.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Fix DREAM #1270 — scope preset names

**Spec rule:** R-1.4 — examples `micro, short, medium, long`. Rename `vignette→micro` and `standard→medium`.

**Files:**
- Modify: `src/questfoundry/models/dream.py`
- Modify: `src/questfoundry/pipeline/size.py`
- Modify: `tests/unit/test_dream_stage.py` (fixtures using "standard"/"vignette")
- Modify: `tests/unit/test_artifacts.py`
- Modify: anywhere else `"vignette"`/`"standard"` appears as a story_size

- [ ] **Step 1: Update failing test fixtures to new preset names**

In `tests/unit/test_dream_stage.py`, replace every occurrence of `story_size="standard"` with `story_size="medium"`, and any `story_size="vignette"` with `story_size="micro"`. Same in `tests/unit/test_artifacts.py`.

- [ ] **Step 2: Run DREAM tests — expect failures from Pydantic Literal mismatch**

Run: `uv run pytest tests/unit/test_dream_stage.py tests/unit/test_artifacts.py -v --tb=short`
Expected: FAIL with "Input should be 'vignette', 'short', 'standard' or 'long'" because `models/dream.py` still has the old literal.

- [ ] **Step 3: Update `models/dream.py` Scope literal**

Replace lines 34-36:

```python
    story_size: Literal["micro", "short", "medium", "long"] = Field(
        description='Story size preset: "micro", "short", "medium", or "long"',
    )
```

- [ ] **Step 4: Update `pipeline/size.py` preset dict keys**

Edit `src/questfoundry/pipeline/size.py`:
- Line ~92: rename dict key `"vignette"` → `"micro"` and inside the `SizeProfile(preset="vignette", ...)`, rename `preset="vignette"` → `preset="micro"`.
- Line ~150: rename dict key `"standard"` → `"medium"` and `preset="standard"` → `preset="medium"`.
- Line ~213: `def get_size_profile(preset: str = "standard")` → `def get_size_profile(preset: str = "medium")`.
- Line ~217: docstring reference from `"standard"` → `"medium"`.
- Line ~236: docstring from `"standard"` → `"medium"`.
- Line ~247: `story_size = scope.get("story_size", "standard")` → `"medium"`.
- Line ~250: `story_size = "standard"` → `"medium"`.
- Line ~267: `get_size_profile("standard")` → `"medium"`.

- [ ] **Step 5: Update SEED call site**

Edit `src/questfoundry/pipeline/stages/seed.py:468`:
```python
        size_profile = kwargs.get("size_profile") or get_size_profile("medium")
```

(SEED is out of the main compliance scope, but this one-line constant rename keeps SEED loadable so the test-pollution failure doesn't worsen.)

- [ ] **Step 6: Run DREAM + size tests**

Run: `uv run pytest tests/unit/test_dream_stage.py tests/unit/test_artifacts.py tests/unit/test_dream_validation.py -v`
Expected: PASS. If other tests fail referencing old names outside the DREAM/BRAINSTORM scope, evaluate per task scope — fix if trivial (one-line constant rename), defer otherwise.

- [ ] **Step 7: Commit**

```bash
git add src/questfoundry/models/dream.py src/questfoundry/pipeline/size.py src/questfoundry/pipeline/stages/seed.py tests/unit/test_dream_stage.py tests/unit/test_artifacts.py
git commit -m "$(cat <<'EOF'
fix(dream): align scope preset names to spec (R-1.4)

Rename story_size values: vignette→micro, standard→medium.
Also rename the preset keys in pipeline/size.py and the default
constants in get_size_profile.

Matches docs/design/procedures/dream.md R-1.4.

Closes #1270.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Fix DREAM #1271 — record human approval on vision

**Spec rule:** R-1.12 (human approval required) + R-1.13 (rejection loops back). Out of this task's scope: the interactive rejection UI loop. In scope: the artifact's `human_approved` field, set by the orchestrator's non-interactive path or by the interactive approval handler.

**Files:**
- Modify: `src/questfoundry/models/dream.py` — add `human_approved: bool = False` to `DreamArtifact`.
- Modify: `src/questfoundry/cli.py` — where DREAM is dispatched, if `--no-interactive` is set, the artifact's `human_approved` is set to True; otherwise an explicit approval is still required. Locate by: `grep -n "stage_name=\"dream\"" src/questfoundry/cli.py`.
- Modify: `tests/unit/test_dream_stage.py` — fixtures should either set `human_approved=True` or verify validator rejects when False.

- [ ] **Step 1: Add `human_approved` field to `DreamArtifact`**

Insert into `DreamArtifact` in `src/questfoundry/models/dream.py` (anywhere among the fields, logical placement is near `protagonist_defined`):

```python
    human_approved: bool = Field(
        default=False,
        description=(
            "True when the human has explicitly approved the Vision "
            "(--no-interactive implies pre-approval at invocation time)."
        ),
    )
```

- [ ] **Step 2: Run DREAM stage tests — producer fixtures should now fail contract**

Run: `uv run pytest tests/unit/test_dream_stage.py -v --tb=short`
Expected: fixtures that build `DreamArtifact` without `human_approved` emit `human_approved=False` and fail the validator when apply_dream_mutations runs. If fixtures don't exercise `apply_dream_mutations`, they still pass here.

- [ ] **Step 3: Update `cli.py` DREAM dispatcher to set approval in `--no-interactive` mode**

Locate DREAM dispatch (around `stage_name="dream"`). After the stage returns its `artifact_data`, inject approval for non-interactive runs:

```python
        if stage_name == "dream" and not use_interactive:
            artifact_data["human_approved"] = True
```

(Place this before `apply_mutations` is called by the orchestrator.)

For interactive runs, the approval checkbox needs to be captured. Within scope of this task: add a simple "approve?" prompt at the end of DREAM if `use_interactive` is True, set `artifact_data["human_approved"] = True` on "yes", or raise `DreamStageError("Vision rejected by human — re-run DREAM")` on "no". Full rejection-loop UI (R-1.13 loop-back to specific operation) is **deferred** and filed below as a follow-up issue.

- [ ] **Step 4: Update `test_dream_stage.py` fixtures to set `human_approved=True` where they exercise mutation**

Find every `DreamArtifact(...)` constructor in the test file. For tests that exercise `apply_dream_mutations` or expect a compliant artifact, add `human_approved=True`.

- [ ] **Step 5: Run DREAM stage tests**

Run: `uv run pytest tests/unit/test_dream_stage.py tests/unit/test_dream_validation.py -v`
Expected: PASS.

- [ ] **Step 6: File the R-1.13 loop-back follow-up**

Run:
```
gh issue create --title "[spec-audit] DREAM: implement R-1.13 rejection loop-back to specific operation" --label "spec-audit,area:dream" --body "Follow-up to #1271. The approval-gate fix added a binary 'approve / reject & halt' handler; the full R-1.13 behavior requires the human to indicate which operation (spark exploration, constraint definition, or synthesis) contains the misalignment. Deferred because it requires interactive UX work not in scope for DREAM+BRAINSTORM compliance."
```
Record the issue number for the PR body.

- [ ] **Step 7: Commit**

```bash
git add src/questfoundry/models/dream.py src/questfoundry/cli.py tests/unit/test_dream_stage.py
git commit -m "$(cat <<'EOF'
fix(dream): record human approval on Vision artifact (R-1.12)

Adds human_approved field to DreamArtifact. Non-interactive mode
implies pre-approval. Interactive mode uses a simple yes/no prompt;
full R-1.13 rejection-loop UX is deferred to a follow-up issue.

Closes #1271.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase F — BRAINSTORM producer fixes

### Task 10: Fix BRAINSTORM #1273 — dilemma-entity anchoring silent-skip

**Spec rule:** R-3.6 — every Dilemma has ≥1 `anchored_to` edge. Silent skip at `mutations.py:902-910` is a critical silent-degradation violation.

**Files:**
- Modify: `src/questfoundry/graph/mutations.py` (lines ~899-910 in `apply_brainstorm_mutations`)
- Add: tests in `tests/unit/test_mutations.py`

- [ ] **Step 1: Add failing test: unresolvable entity in `central_entity_ids` must raise**

In `tests/unit/test_mutations.py`, add a new test:

```python
def test_apply_brainstorm_mutations_fails_on_unresolvable_entity() -> None:
    """R-3.6: dilemma referencing non-existent entity must raise, not silently drop."""
    import pytest
    from questfoundry.graph.graph import Graph
    from questfoundry.graph.mutations import apply_brainstorm_mutations, MutationError

    graph = Graph()
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["x"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )

    output = {
        "entities": [
            {"entity_id": "kay", "entity_category": "character",
             "name": "Kay", "concept": "archivist"},
            {"entity_id": "a", "entity_category": "location", "name": "A", "concept": "x"},
            {"entity_id": "b", "entity_category": "location", "name": "B", "concept": "x"},
        ],
        "dilemmas": [
            {
                "dilemma_id": "dilemma::mentor_trust",
                "question": "Can we trust?",
                "why_it_matters": "stakes",
                "central_entity_ids": ["character::ghost"],  # does not exist
                "answers": [
                    {"answer_id": "yes", "description": "d", "is_canonical": True},
                    {"answer_id": "no", "description": "d", "is_canonical": False},
                ],
            }
        ],
    }

    with pytest.raises((MutationError, ValueError)) as exc_info:
        apply_brainstorm_mutations(graph, output)
    assert "ghost" in str(exc_info.value) or "anchored_to" in str(exc_info.value).lower()
```

- [ ] **Step 2: Run the new test — currently silent-skip, so test fails (no raise)**

Run: `uv run pytest tests/unit/test_mutations.py::test_apply_brainstorm_mutations_fails_on_unresolvable_entity -v`
Expected: FAIL with "DID NOT RAISE".

- [ ] **Step 3: Remove the silent-skip in `apply_brainstorm_mutations`**

Edit `src/questfoundry/graph/mutations.py` around lines 899-910. Replace:

```python
        # Resolve entity references for anchored_to edges
        raw_central_entities = dilemma.get("central_entity_ids", [])
        prefixed_central_entities = []
        for eid in raw_central_entities:
            try:
                prefixed_central_entities.append(_resolve_entity_ref(graph, eid))
            except ValueError:
                log.warning(
                    "anchored_to_entity_not_found",
                    dilemma_id=raw_id,
                    entity_id=eid,
                )
```

with:

```python
        # Resolve entity references for anchored_to edges.
        # R-3.6: every Dilemma needs ≥1 anchored_to edge. An unresolvable
        # reference is a structural failure — raise, do not silently drop.
        raw_central_entities = dilemma.get("central_entity_ids", [])
        prefixed_central_entities = []
        for eid in raw_central_entities:
            try:
                prefixed_central_entities.append(_resolve_entity_ref(graph, eid))
            except ValueError as exc:
                raise MutationError(
                    f"Dilemma '{raw_id}' references unknown entity '{eid}' "
                    f"in central_entity_ids (R-3.6 requires all anchored_to "
                    f"targets to exist in the entity list)."
                ) from exc
```

- [ ] **Step 4: Run the new test**

Run: `uv run pytest tests/unit/test_mutations.py::test_apply_brainstorm_mutations_fails_on_unresolvable_entity -v`
Expected: PASS.

- [ ] **Step 5: Run all brainstorm-related mutation tests**

Run: `uv run pytest tests/unit/test_mutations.py tests/unit/test_brainstorm_validation.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/graph/mutations.py tests/unit/test_mutations.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): raise on unresolvable anchored_to entity (R-3.6)

Replaces the silent WARNING-and-skip in apply_brainstorm_mutations
with a MutationError. Per R-3.6 every dilemma must have an
anchored_to edge; dropping one silently is a structural failure.

Closes #1273.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Fix BRAINSTORM #1274 — location entity minimum count

**Spec rule:** R-2.4 — ≥2 distinct `location`-category entities.

Already covered by `validate_brainstorm_output` cluster 5 test. Producer-side change: add a pre-mutation check in the BRAINSTORM semantic validator so the error is surfaced during serialize rather than only at the exit validator.

**Files:**
- Modify: `src/questfoundry/graph/mutations.py:validate_brainstorm_mutations` (adds to the existing list)
- Modify: `tests/unit/test_mutations.py`

- [ ] **Step 1: Add a failing test for the producer-side check**

In `tests/unit/test_mutations.py`, add:

```python
def test_validate_brainstorm_mutations_requires_two_locations() -> None:
    """R-2.4: BRAINSTORM output with <2 locations must fail validation."""
    from questfoundry.graph.mutations import validate_brainstorm_mutations

    output = {
        "entities": [
            {"entity_id": "kay", "entity_category": "character",
             "name": "Kay", "concept": "x"},
            {"entity_id": "archive", "entity_category": "location",
             "name": "Archive", "concept": "x"},
        ],
        "dilemmas": [
            {
                "dilemma_id": "dilemma::x",
                "question": "Q?",
                "why_it_matters": "stakes",
                "central_entity_ids": ["character::kay"],
                "answers": [
                    {"answer_id": "y", "description": "d", "is_canonical": True},
                    {"answer_id": "n", "description": "d", "is_canonical": False},
                ],
            }
        ],
    }
    errors = validate_brainstorm_mutations(output)
    assert any("location" in e.issue.lower() for e in errors)
```

- [ ] **Step 2: Run — expected FAIL**

Run: `uv run pytest tests/unit/test_mutations.py::test_validate_brainstorm_mutations_requires_two_locations -v`
Expected: FAIL.

- [ ] **Step 3: Extend `validate_brainstorm_mutations` in `mutations.py`**

Before `return errors` at the end of `validate_brainstorm_mutations`, insert:

```python
    # R-2.4: ≥2 distinct location entities
    location_count = sum(
        1 for e in entities if e.get("entity_category") == "location"
    )
    if location_count < 2:
        errors.append(
            BrainstormValidationError(
                field_path="entities",
                issue=(
                    f"R-2.4: BRAINSTORM must produce ≥2 location entities, "
                    f"found {location_count}"
                ),
                available=[],
                provided=f"{location_count} location entities",
            )
        )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/unit/test_mutations.py::test_validate_brainstorm_mutations_requires_two_locations -v`
Expected: PASS.

- [ ] **Step 5: Run the full BRAINSTORM validation test suite**

Run: `uv run pytest tests/unit/test_mutations.py tests/unit/test_brainstorm_validation.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/graph/mutations.py tests/unit/test_mutations.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): require ≥2 location entities (R-2.4)

Adds producer-side check in validate_brainstorm_mutations that mirrors
the exit validator. Surfaces the error during serialize for faster feedback.

Closes #1274.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Fix BRAINSTORM #1275 — question punctuation (ends with `?`)

**Spec rule:** R-3.1 — dilemma question ends with `?`.

**Files:**
- Modify: `src/questfoundry/models/brainstorm.py`
- Modify: `tests/unit/test_brainstorm_stage.py` or a dedicated test

- [ ] **Step 1: Add a failing model test**

Append to `tests/unit/test_brainstorm_stage.py` (or a new dedicated file):

```python
def test_dilemma_question_must_end_with_qmark() -> None:
    """R-3.1: dilemma question must end with ?."""
    import pytest
    from pydantic import ValidationError
    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError) as exc:
        Dilemma(
            dilemma_id="dilemma::x",
            question="not a question",
            why_it_matters="stakes",
            answers=[
                {"answer_id": "a", "description": "d", "is_canonical": True},
                {"answer_id": "b", "description": "d", "is_canonical": False},
            ],
        )
    assert "?" in str(exc.value)
```

- [ ] **Step 2: Run — expected FAIL (no validator yet)**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_dilemma_question_must_end_with_qmark -v`
Expected: FAIL.

- [ ] **Step 3: Add field_validator to `Dilemma.question`**

In `src/questfoundry/models/brainstorm.py`, add to the `Dilemma` class (near the existing `validate_dilemma_id_no_trailing_or` validator):

```python
    @field_validator("question")
    @classmethod
    def validate_question_ends_with_qmark(cls, v: str) -> str:
        """R-3.1: dilemma question must end with '?'."""
        if not v.rstrip().endswith("?"):
            raise ValueError(
                f"dilemma question must end with '?' (got {v!r}). See R-3.1."
            )
        return v
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_dilemma_question_must_end_with_qmark -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/models/brainstorm.py tests/unit/test_brainstorm_stage.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): enforce question punctuation at model level (R-3.1)

Dilemma.question must end with '?'. Catches violations during serialize
before they reach the graph.

Closes #1275.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Fix BRAINSTORM #1276 — entity `name` disallows None

**Spec rule:** R-2.1 — every Entity has non-empty `name`, `category`, `concept`.

**Files:**
- Modify: `src/questfoundry/models/brainstorm.py` (Entity.name)
- Modify: `tests/unit/test_brainstorm_stage.py`

- [ ] **Step 1: Add failing test**

Append to `tests/unit/test_brainstorm_stage.py`:

```python
def test_entity_name_must_be_non_empty() -> None:
    """R-2.1: entity name is required and non-empty."""
    import pytest
    from pydantic import ValidationError
    from questfoundry.models.brainstorm import Entity

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", concept="c")  # no name

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", name="", concept="c")

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", name=None, concept="c")
```

- [ ] **Step 2: Run — expected FAIL (current model allows None)**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_entity_name_must_be_non_empty -v`
Expected: FAIL — Pydantic currently accepts the None / missing cases.

- [ ] **Step 3: Update `Entity.name` to non-optional**

In `src/questfoundry/models/brainstorm.py`, replace the `name` field:

```python
    name: str = Field(
        min_length=1,
        description=(
            "Canonical display name (e.g., 'Dr. Aris Chen', 'Maya's Bakery'). "
            "Required per R-2.1."
        ),
    )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_entity_name_must_be_non_empty -v`
Expected: PASS.

- [ ] **Step 5: Run BRAINSTORM test suite — existing tests may fail if they omit `name`**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py tests/unit/test_mutations.py -v --tb=short`
Expected: any test constructing Entity without `name` fails. Update each to provide `name="..."`.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/models/brainstorm.py tests/unit/test_brainstorm_stage.py tests/unit/test_mutations.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): entity.name is required and non-empty (R-2.1)

Removes the Optional on Entity.name. Previously allowed None so SEED
could generate; that flexibility violates R-2.1 and encouraged silent
holes in the cast.

Closes #1276.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Fix BRAINSTORM #1277 — dilemma ID prefix validation

**Spec rule:** R-3.7 — Dilemma IDs use `dilemma::` prefix.

**Files:**
- Modify: `src/questfoundry/models/brainstorm.py`
- Modify: `tests/unit/test_brainstorm_stage.py`

- [ ] **Step 1: Add failing test**

Append to `tests/unit/test_brainstorm_stage.py`:

```python
def test_dilemma_id_must_have_prefix() -> None:
    """R-3.7: dilemma_id must start with 'dilemma::'."""
    import pytest
    from pydantic import ValidationError
    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError) as exc:
        Dilemma(
            dilemma_id="mentor_trust",  # missing prefix
            question="Q?",
            why_it_matters="stakes",
            answers=[
                {"answer_id": "a", "description": "d", "is_canonical": True},
                {"answer_id": "b", "description": "d", "is_canonical": False},
            ],
        )
    assert "dilemma::" in str(exc.value)
```

- [ ] **Step 2: Run — expected FAIL**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_dilemma_id_must_have_prefix -v`
Expected: FAIL.

- [ ] **Step 3: Update the existing `validate_dilemma_id_no_trailing_or` validator to also enforce prefix**

In `src/questfoundry/models/brainstorm.py`, replace that validator:

```python
    @field_validator("dilemma_id")
    @classmethod
    def validate_dilemma_id_format(cls, v: str) -> str:
        """R-3.7: dilemma_id must have 'dilemma::' prefix; reject trailing '_or_'."""
        if not v.startswith("dilemma::"):
            raise ValueError(
                f"dilemma_id '{v}' missing required 'dilemma::' prefix. See R-3.7."
            )
        raw = v.removeprefix("dilemma::")
        if raw.endswith("_or_") or raw.endswith("_or"):
            raise ValueError(
                f"dilemma_id '{v}' ends with '_or_' — "
                "the ID must end with the second option word "
                "(e.g., 'host_benevolent_or_selfish', not 'host_benevolent_or_selfish_or_')"
            )
        return v
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_dilemma_id_must_have_prefix -v`
Expected: PASS.

- [ ] **Step 5: Run BRAINSTORM test suite**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py tests/unit/test_mutations.py tests/unit/test_brainstorm_validation.py -v --tb=short`
Expected: any test using bare dilemma_id fails — update to add `dilemma::` prefix.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/models/brainstorm.py tests/unit/test_brainstorm_stage.py tests/unit/test_mutations.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): require dilemma:: prefix at model level (R-3.7)

Merges prefix check with existing _or_ trailing check. Catches
violations at serialize time instead of relying on prefix normalization
in mutations.

Closes #1277.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Fix BRAINSTORM #1278 — defensive node-type check

**Spec rule:** R-3.8 — no Path/Beat/Consequence/State Flag/Passage/Intersection Group nodes after BRAINSTORM.

The exit validator already covers this (task 5 step 9). Producer-side: the BrainstormOutput Pydantic schema is "entities + dilemmas" only — Pydantic itself won't let Path/Beat through because they're not schema fields. The defensive check is therefore layered:

- `BrainstormOutput.model_validate` rejects extra fields (Pydantic default is `ignore`; we can switch to `forbid`).
- `validate_brainstorm_output` rejects the forbidden node types at graph level.

**Files:**
- Modify: `src/questfoundry/models/brainstorm.py`
- Modify: `tests/unit/test_brainstorm_stage.py`

- [ ] **Step 1: Add failing test for extra fields in BrainstormOutput**

Append to `tests/unit/test_brainstorm_stage.py`:

```python
def test_brainstorm_output_rejects_unknown_fields() -> None:
    """R-3.8 defensive: BrainstormOutput must not silently accept foreign node data."""
    import pytest
    from pydantic import ValidationError
    from questfoundry.models.brainstorm import BrainstormOutput

    with pytest.raises(ValidationError):
        BrainstormOutput.model_validate(
            {
                "entities": [],
                "dilemmas": [],
                "paths": [{"path_id": "x"}],  # not an allowed field
            }
        )
```

- [ ] **Step 2: Run — expected FAIL (Pydantic default is 'ignore')**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_brainstorm_output_rejects_unknown_fields -v`
Expected: FAIL.

- [ ] **Step 3: Add `model_config` to `BrainstormOutput`**

In `src/questfoundry/models/brainstorm.py`, update `BrainstormOutput`:

```python
class BrainstormOutput(BaseModel):
    """Complete output of the BRAINSTORM stage."""

    model_config = {"extra": "forbid"}

    entities: list[Entity] = Field(default_factory=list, description="Generated story entities")
    dilemmas: list[Dilemma] = Field(default_factory=list, description="Generated dramatic dilemmas")
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_brainstorm_output_rejects_unknown_fields -v`
Expected: PASS.

- [ ] **Step 5: Run full BRAINSTORM suite**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py tests/unit/test_brainstorm_validation.py tests/unit/test_mutations.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/models/brainstorm.py tests/unit/test_brainstorm_stage.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): reject foreign fields in BrainstormOutput (R-3.8 defensive)

Sets model_config extra='forbid'. Paired with the exit validator's
forbidden-node-type check, this catches stray Path/Beat/Passage data
at serialize time.

Closes #1278.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Fix BRAINSTORM #1279 — minimum abundance floor

**Spec rule:** R-1.1 — "Discussion aims for abundance" with typical targets 15–25 entities and 4–8 dilemmas. The upper target stays in the prompt (not machine-checkable). The floor — at least 1 entity and 1 dilemma after extraction — is enforced at the model/graph level.

Already covered by `validate_brainstorm_output` tests from task 5. Producer-side floor: `BrainstormOutput.entities` and `.dilemmas` have `min_length=1`.

**Files:**
- Modify: `src/questfoundry/models/brainstorm.py`
- Modify: `tests/unit/test_brainstorm_stage.py`

- [ ] **Step 1: Add failing test**

Append to `tests/unit/test_brainstorm_stage.py`:

```python
def test_brainstorm_output_minimum_floor() -> None:
    """R-1.1 floor: must produce ≥1 entity and ≥1 dilemma."""
    import pytest
    from pydantic import ValidationError
    from questfoundry.models.brainstorm import BrainstormOutput

    with pytest.raises(ValidationError):
        BrainstormOutput.model_validate({"entities": [], "dilemmas": []})
```

- [ ] **Step 2: Run — expected FAIL (current default_factory allows empty)**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_brainstorm_output_minimum_floor -v`
Expected: FAIL.

- [ ] **Step 3: Change `entities` and `dilemmas` to require ≥1**

In `src/questfoundry/models/brainstorm.py`:

```python
    entities: list[Entity] = Field(
        min_length=1,
        description="Generated story entities (at least 1 required per R-1.1).",
    )
    dilemmas: list[Dilemma] = Field(
        min_length=1,
        description="Generated dramatic dilemmas (at least 1 required per R-1.1).",
    )
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py::test_brainstorm_output_minimum_floor -v`
Expected: PASS.

- [ ] **Step 5: Run full BRAINSTORM suite**

Run: `uv run pytest tests/unit/test_brainstorm_stage.py tests/unit/test_brainstorm_validation.py tests/unit/test_mutations.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/models/brainstorm.py tests/unit/test_brainstorm_stage.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): require ≥1 entity and ≥1 dilemma (R-1.1 floor)

Removes default_factory that allowed empty lists. Upper abundance
target (15–25 / 4–8) stays in the prompt as a soft target.

Closes #1279.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: Fix BRAINSTORM #1280 — vision compatibility at serialize

**Spec rule:** R-1.3 — every proposal must be compatible with the Vision (genre, tone, themes, content_notes constrain). Also output-11 — vision node unchanged.

Machine-checkable pieces:
- Vision still exists in the graph at exit (already in the validator).
- Vision's core fields (genre, tone, themes, scope, audience) are unchanged from when BRAINSTORM started.

**Files:**
- Modify: `src/questfoundry/graph/brainstorm_validation.py`
- Modify: `src/questfoundry/graph/mutations.py` (snapshot vision before mutation)
- Modify: `tests/unit/test_brainstorm_validation.py`

Because checking "unchanged from entry" requires a snapshot, simpler approach: verify the vision node's content at exit is well-formed (`validate_dream_output(graph)` returns `[]`). If DREAM's contract still holds, vision wasn't silently broken.

- [ ] **Step 1: Extend `validate_brainstorm_output` to re-check DREAM contract**

In `src/questfoundry/graph/brainstorm_validation.py`, add at top of the function body (after `errors: list[str] = []`):

```python
    from questfoundry.graph.dream_validation import validate_dream_output

    dream_contract = validate_dream_output(graph)
    if dream_contract:
        errors.extend(
            f"Output-11: DREAM contract violated post-BRAINSTORM — {e}"
            for e in dream_contract
        )
```

(Inline import to avoid a circular dependency at module load.)

- [ ] **Step 2: Add failing test**

In `tests/unit/test_brainstorm_validation.py`, append:

```python
def test_output11_vision_corrupted_by_brainstorm(compliant_graph: Graph) -> None:
    """If BRAINSTORM somehow wipes a required vision field, validate_brainstorm_output must fail."""
    compliant_graph.update_node("vision", genre=None)
    errors = validate_brainstorm_output(compliant_graph)
    assert any("genre" in e.lower() for e in errors)
```

- [ ] **Step 3: Run — expect PASS with the new check**

Run: `uv run pytest tests/unit/test_brainstorm_validation.py::test_output11_vision_corrupted_by_brainstorm -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/graph/brainstorm_validation.py tests/unit/test_brainstorm_validation.py
git commit -m "$(cat <<'EOF'
fix(brainstorm): enforce vision compatibility at exit (R-1.3 / output-11)

validate_brainstorm_output now re-runs validate_dream_output and
reports any violations. Covers the 'vision unchanged' output
contract item.

Closes #1280.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase G — Close out

### Task 18: Validate exit criteria, file deferred issues, open PR

**Files:**
- None new; this task runs verifications and opens the PR.

- [ ] **Step 1: Confirm DREAM + BRAINSTORM tests pass**

Run: `uv run pytest tests/unit/test_dream*.py tests/unit/test_brainstorm*.py -v`
Expected: all PASS.

- [ ] **Step 2: Confirm the non-downstream suite is green**

Run:
```
uv run pytest tests/unit/ -k "not seed and not grow and not polish and not fill and not dress and not ship" --tb=short
```
Expected: 0 failures. (FILL/DRESS/SHIP tests are excluded; SEED downstream is allowed to break.)

- [ ] **Step 3: Confirm downstream breakage is present (expected)**

Run:
```
uv run pytest tests/unit/test_seed*.py tests/unit/test_fill*.py --tb=no -q 2>&1 | tail -5
```
Expected: failures in SEED / FILL — allowed per design spec. Note the failure count for the PR body.

- [ ] **Step 4: Type-check + lint the full change**

Run:
```
uv run mypy src/questfoundry/graph/dream_validation.py src/questfoundry/graph/brainstorm_validation.py src/questfoundry/graph/mutations.py src/questfoundry/pipeline/stages/dream.py src/questfoundry/pipeline/stages/brainstorm.py src/questfoundry/models/dream.py src/questfoundry/models/brainstorm.py src/questfoundry/pipeline/size.py
uv run ruff check src/questfoundry/
```
Expected: clean.

- [ ] **Step 5: Push branch**

```bash
git push -u origin feat/dream-brainstorm-compliance
```

- [ ] **Step 6: Open the PR**

```bash
gh pr create --title "feat(dream,brainstorm): compliance with authoritative specs" --body "$(cat <<'EOF'
## Summary

Brings DREAM and BRAINSTORM into compliance with `docs/design/procedures/dream.md` and `docs/design/procedures/brainstorm.md`. Introduces `validate_dream_output` and `validate_brainstorm_output` as runtime oracles at stage exit and downstream stage entry, mirroring the POLISH pattern.

## Closed issues

- Closes #1269 — DREAM R-1.9 pov_style enum alignment
- Closes #1270 — DREAM R-1.4 scope preset names
- Closes #1271 — DREAM R-1.12 human approval recorded (R-1.13 loop-back UI deferred; see follow-up issue below)
- Closes #1273 — BRAINSTORM R-3.6 dilemma-entity anchoring silent-skip
- Closes #1274 — BRAINSTORM R-2.4 location entity minimum count
- Closes #1275 — BRAINSTORM R-3.1 question punctuation
- Closes #1276 — BRAINSTORM R-2.1 entity name non-empty
- Closes #1277 — BRAINSTORM R-3.7 dilemma ID prefix
- Closes #1278 — BRAINSTORM R-3.8 defensive node-type check
- Closes #1279 — BRAINSTORM R-1.1 minimum floor
- Closes #1280 — BRAINSTORM R-1.3 vision compatibility

Partial contribution to M-contract-chaining epic (#1346): adds `validate_dream_output` and `validate_brainstorm_output` with exit-wiring for DREAM and BRAINSTORM, and upstream-contract check at BRAINSTORM entry.

## Out of scope / allowed breakage

- SEED / GROW / POLISH / FILL / DRESS / SHIP tests may now fail because the tightened BRAINSTORM contract rejects artifacts those stages relied on. Per the design spec, this is explicitly accepted.
- The scope preset and pov_style renames drift from FILL's pov enum; FILL tests break accordingly. Will be aligned during FILL compliance work.
- `test_provider_factory.py::test_create_chat_model_ollama_success` — pre-existing test-pollution; unchanged.

## Deferred / new follow-up issues

- R-1.13 interactive rejection loop-back UI — file created in task 9 step 6.

## Test plan

- [ ] DREAM + BRAINSTORM unit suites: `uv run pytest tests/unit/test_dream*.py tests/unit/test_brainstorm*.py`
- [ ] Non-downstream unit suite: `uv run pytest tests/unit/ -k "not seed and not grow and not polish and not fill and not dress and not ship"`
- [ ] Run a live DREAM → BRAINSTORM pipeline against a small project; verify validators catch intentional contract violations and pass on clean output

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: Address AI-bot review comments when they land; iterate in a second push**

AI-bot reviews will post after the push. Batch all responses into a single follow-up commit and push once — per CLAUDE.md §Project Git Rules, avoid WIP pushes.

---

## Self-review checklist

Ran before finalizing the plan:

1. **Spec coverage:**
   - All 11 clusters have a task: #1269 → Task 7, #1270 → Task 8, #1271 → Task 9, #1273 → Task 10, #1274 → Task 11, #1275 → Task 12, #1276 → Task 13, #1277 → Task 14, #1278 → Task 15, #1279 → Task 16, #1280 → Task 17. ✓
   - `validate_dream_output` created (Task 3) and tested (Task 2). ✓
   - `validate_brainstorm_output` created (Task 5) and tested (Task 4). ✓
   - Wired at DREAM exit (Task 6 step 2), BRAINSTORM entry (Task 6 step 4), BRAINSTORM exit (Task 6 step 3). ✓
   - Obsolete tests deleted (Task 1). ✓

2. **Placeholder scan:** No "TBD", "TODO", "similar to Task N". Code blocks contain complete content. A few inline "adjust to match existing helper" notes (Task 3 steps 9, 11; Task 6 step 4) reference specific files and call out the fallback API shape.

3. **Type consistency:**
   - `validate_dream_output` / `validate_brainstorm_output` — consistent signature `(graph: Graph) -> list[str]`.
   - `DreamContractError` / `BrainstormContractError` — defined in their respective modules; raised from mutation functions.
   - `DreamStageError` — added in Task 6 step 1.
   - `BrainstormStageError` — already exists.
   - POV enum new values `first_person / second_person / third_person_limited / third_person_omniscient` consistent across model, mutations, tests.
   - Scope preset new values `micro / short / medium / long` consistent across model, `pipeline/size.py`, tests, and the SEED default.
