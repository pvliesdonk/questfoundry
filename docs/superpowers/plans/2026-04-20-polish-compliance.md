# POLISH Spec-Compliance (Hot-Path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the POLISH stage into compliance with its authoritative spec for the 4 hot-path clusters (#1311, #1312, #1313, #1314), using a stage-exit contract validator with a dedicated `PolishContractError` raised from Phase 7.

**Architecture:** Validator-first TDD. Add `PolishContractError` and wire Phase 7 to raise it. Extend `validate_polish_output` with hot-path helpers. Write failing contract tests covering all 4 rules. Fix clusters one at a time, flipping contract tests green. Rewrite fixtures that still describe valid spec intent; delete those whose premise is pre-audit. Downstream (FILL/DRESS/SHIP/post-POLISH) is allowed to break.

**Tech Stack:** Python 3.11+, `uv` package manager, `pydantic`, `pytest`, `ruff`, `mypy`, `pyright`. No new libraries.

**Spec:** `docs/superpowers/specs/2026-04-20-polish-compliance-design.md`

**Prereqs:** PR #1357 (GROW compliance) and PR #1358 (R-4a.3 spec update + ontology Part 8 alignment) must be merged. Both landed on `main` 2026-04-20.

**Branch:** `feat/polish-compliance` (already exists locally on top of main; holds the design doc commit).

---

## File Structure

### Files modified

- `src/questfoundry/graph/polish_validation.py` — add `PolishContractError` class; extend `validate_polish_output` with 4 new `_check_*` helpers.
- `src/questfoundry/pipeline/stages/polish/deterministic.py` — rewire `phase_validation` to raise `PolishContractError`; rewrite `compute_beat_grouping` (Cluster #1311); add zero-choice halt in Phase 4c (Cluster #1312); branch residue creation on `mapping_strategy` (Cluster #1313).
- `src/questfoundry/pipeline/stages/polish/llm_phases.py` — replace `character_arc_metadata` node creation with entity-data annotation (Cluster #1314); add `mapping_strategy` to Phase 5b prompt + schema (Cluster #1313).
- `src/questfoundry/models/polish.py` — add `mapping_strategy` field to `ResidueSpec` (Cluster #1313).

### Files created

- `tests/unit/test_polish_contract.py` — rule-by-rule contract tests for the 4 hot-path rules.

### Test files modified

- `tests/unit/test_polish_deterministic.py` — Phase 4a grouping tests rewritten to assert maximal-linear-collapse output.
- `tests/unit/test_polish_phases.py` — delete `test_has_arc_metadata_edge_created`; add `test_character_arc_field_on_entity`.
- `tests/unit/test_polish_llm_phases.py` — Phase 3 tests updated; Phase 5b tests add `mapping_strategy` assertions.
- `tests/unit/test_polish_apply.py` — residue mapping tests parameterized over both `mapping_strategy` values.

### Files NOT touched

- `docs/design/procedures/polish.md` — already updated in PR #1358.
- `docs/design/story-graph-ontology.md` — already updated in PR #1358.
- FILL/DRESS/SHIP code and tests — downstream allowed to break per policy.
- `# pyright: reportArgumentType=false` suppression on `polish/stage.py:16` — saved for final cluster PR, not this hot-path PR.

---

## Naming conventions used in this plan

- `PolishContractError(ValueError)` — raised from Phase 7 when `validate_polish_output` reports errors.
- `mapping_strategy` — `Literal["residue_passage_with_variants", "parallel_passages"]` on `ResidueSpec`.
- `character_arc` — dict field on entity `data` dicts with keys `start: str`, `pivots: dict[str, str]` (path_id → pivot beat_id), `end_per_path: dict[str, str]`.
- `_check_no_character_arc_metadata_nodes`, `_check_passage_maximal_linear_collapse`, `_check_residue_mapping_strategy`, `_check_has_choice_edges` — new validator helpers.

---

## Phase 0 — Prereqs

### Task 1: Baseline sanity

**Files:** None (branch hygiene).

- [ ] **Step 1: Confirm main has both prereq PRs merged**

Run:
```bash
git log --oneline origin/main | head -5
```
Expected: recent commits include `feat(grow): compliance with authoritative spec (#1357)` and `docs(polish): rewrite R-4a.3 as maximal-linear-collapse rule (#1358)`.

- [ ] **Step 2: Confirm branch is rebased on main**

Run:
```bash
git fetch origin main
git log --oneline origin/main..feat/polish-compliance
```
Expected: exactly one commit on the branch — the design doc commit (`docs(polish): spec-compliance design for hot-path clusters`).

If the branch is not on top of main, rebase:
```bash
git rebase origin/main
```

- [ ] **Step 3: Confirm POLISH-local tests pass on baseline**

Run:
```bash
uv run pytest tests/unit/test_polish_validation.py tests/unit/test_polish_deterministic.py tests/unit/test_polish_phases.py tests/unit/test_polish_apply.py tests/unit/test_polish_llm_phases.py -x -q
```
Expected: all pass. Record the count for later comparison. If any fail on `main`, stop and investigate before modifying.

- [ ] **Step 4: No commit — this is a checkpoint**

Skip. Proceed to Phase 1.

---

## Phase 1 — Validator scaffolding

### Task 2: Add `PolishContractError` class

**Files:**
- Modify: `src/questfoundry/graph/polish_validation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_polish_contract.py` with only this initial content (more tests land in Task 4):

```python
"""Rule-by-rule POLISH Stage Output Contract validator tests.

Layered over DREAM + BRAINSTORM + SEED + GROW + POLISH compliant baseline.
Mirrors the pattern of tests/unit/test_grow_validation_contract.py.
"""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_validation import (
    PolishContractError,
    validate_polish_output,
)


def test_polish_contract_error_is_value_error() -> None:
    """PolishContractError is a ValueError subclass (same convention as GrowContractError)."""
    assert issubclass(PolishContractError, ValueError)


def test_polish_contract_error_carries_message() -> None:
    """PolishContractError preserves the error message for callers."""
    err = PolishContractError("R-4a.4: intersection groups consumed")
    assert "R-4a.4" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -x -q
```
Expected: ImportError — `PolishContractError` does not exist.

- [ ] **Step 3: Add `PolishContractError` to `polish_validation.py`**

Insert this block in `src/questfoundry/graph/polish_validation.py` immediately after the imports section (around line 40, before `def validate_polish_output`):

```python
class PolishContractError(ValueError):
    """Raised when POLISH Phase 7 validation reports contract errors.

    Mirrors ``GrowContractError`` — a dedicated exception type for
    stage-exit contract failures so callers can distinguish them from
    generic ``ValueError`` noise.  Callers receive the formatted error
    list in the exception message.
    """
```

Add `"PolishContractError"` to the `__all__` list in the same file.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -x -q
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/polish_validation.py tests/unit/test_polish_contract.py
git commit -m "feat(polish): add PolishContractError for Phase 7 exit contract

ValueError subclass; mirrors GrowContractError.  Will be raised from
phase_validation when validate_polish_output reports errors (next
task).  Part of epic #1310."
```

---

### Task 3: Wire Phase 7 to raise `PolishContractError`

**Files:**
- Modify: `src/questfoundry/pipeline/stages/polish/deterministic.py:1385-1408`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_polish_contract.py`:

```python
import asyncio
from unittest.mock import MagicMock


def test_phase_validation_raises_contract_error_on_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_validation raises PolishContractError (not PhaseResult) when
    validate_polish_output returns errors."""
    from questfoundry.pipeline.stages.polish import deterministic

    graph = Graph.empty()

    def _mock_validate(g: Graph) -> list[str]:
        return ["R-4a.4: intersection groups consumed (test)"]

    monkeypatch.setattr(
        "questfoundry.graph.polish_validation.validate_polish_output",
        _mock_validate,
    )

    with pytest.raises(PolishContractError, match=r"R-4a\.4"):
        asyncio.run(deterministic.phase_validation(graph, MagicMock()))


def test_phase_validation_passes_on_clean_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_validation returns completed PhaseResult when no errors."""
    from questfoundry.pipeline.stages.polish import deterministic

    graph = Graph.empty()
    monkeypatch.setattr(
        "questfoundry.graph.polish_validation.validate_polish_output",
        lambda g: [],
    )

    result = asyncio.run(deterministic.phase_validation(graph, MagicMock()))
    assert result.status == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py::test_phase_validation_raises_contract_error_on_errors -x -q
```
Expected: FAIL — current `phase_validation` returns a `PhaseResult(status="failed", ...)`, does not raise.

- [ ] **Step 3: Update `phase_validation` in `polish/deterministic.py`**

Replace the block at `deterministic.py:1385-1408` with:

```python
async def phase_validation(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> PhaseResult:
    """Phase 7: Validate the complete passage graph.

    Runs structural, variant, choice, and feasibility checks on the
    passage layer created by Phase 6.  On any error, raises
    ``PolishContractError`` after logging a structured ERROR event —
    failures at this seam indicate bugs in Phases 4-6 or insufficient
    GROW output and should halt the pipeline loudly.
    """
    from questfoundry.graph.polish_validation import (
        PolishContractError,
        validate_polish_output,
    )

    errors = validate_polish_output(graph)

    if errors:
        log.error(
            "polish_contract_failed",
            error_count=len(errors),
            errors=errors[:10],  # cap for log readability
        )
        raise PolishContractError(
            f"POLISH stage output contract violated ({len(errors)} "
            f"error(s)):\n" + "\n".join(f"  - {e}" for e in errors)
        )
```

Keep the existing summary-stats block at the end of the function (lines 1410+) unchanged.

- [ ] **Step 4: Run all tests to verify they pass**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -x -q
```
Expected: 4 passed.

Run the broader POLISH validation suite to catch regressions:
```bash
uv run pytest tests/unit/test_polish_validation.py -x -q
```
Expected: all pass (no behavior change for the structural checks).

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/pipeline/stages/polish/deterministic.py tests/unit/test_polish_contract.py
git commit -m "feat(polish): raise PolishContractError from phase_validation

phase_validation now raises PolishContractError when
validate_polish_output reports errors, with a structured
polish_contract_failed log.error event.  Previously wrapped as a
failed PhaseResult which the stage loop translated to
PolishStageError — the dedicated contract error type is clearer at
the seam and mirrors the GROW pattern.  Part of epic #1310."
```

---

## Phase 2 — Validator extensions (failing contract tests + helpers)

### Task 4: Compliant-baseline fixture + failing hot-path tests

**Files:**
- Modify: `tests/unit/test_polish_contract.py`

- [ ] **Step 1: Append the compliant-baseline fixture**

Append to `tests/unit/test_polish_contract.py` (top-level, before the existing class/tests):

```python
# --------------------------------------------------------------------------
# Compliant POLISH-output baseline (DREAM + BRAINSTORM + SEED + GROW + POLISH)
# --------------------------------------------------------------------------


def _polish_upstream_baseline(graph: Graph) -> None:
    """Layer a compliant DREAM+BRAINSTORM+SEED+GROW baseline.

    Produces the same graph shape used by test_grow_validation_contract.py
    so validate_polish_output's upstream-contract delegation passes.
    Single soft dilemma `mentor_trust` with 2 paths, Y-shape beats,
    state flags, convergence metadata.
    """
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["forbidden knowledge"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )
    for eid, cat, name in [
        ("character::kay", "character", "Kay"),
        ("character::mentor", "character", "Mentor"),
        ("location::archive", "location", "Archive"),
        ("location::depths", "location", "Forbidden Depths"),
    ]:
        graph.create_node(
            eid,
            {
                "type": "entity",
                "raw_id": eid.split("::", 1)[-1],
                "name": name,
                "category": cat,
                "concept": "x",
                "disposition": "retained",
            },
        )
    graph.create_node(
        "dilemma::mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Trust?",
            "why_it_matters": "stakes",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "low",
        },
    )
    for ans, is_canon in [("protector", True), ("manipulator", False)]:
        ans_id = f"dilemma::mentor_trust::alt::{ans}"
        graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": ans,
                "description": f"d-{ans}",
                "is_canonical": is_canon,
                "explored": True,
            },
        )
        graph.add_edge("has_answer", "dilemma::mentor_trust", ans_id)
    graph.add_edge("anchored_to", "dilemma::mentor_trust", "character::mentor")

    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        graph.create_node(
            path_id,
            {
                "type": "path",
                "raw_id": f"mentor_trust__{ans}",
                "dilemma_id": "dilemma::mentor_trust",
                "is_canonical": ans == "protector",
            },
        )
        graph.add_edge("explores", path_id, f"dilemma::mentor_trust::alt::{ans}")
        conseq_id = f"consequence::mentor_trust__{ans}"
        graph.create_node(
            conseq_id,
            {
                "type": "consequence",
                "raw_id": f"mentor_trust__{ans}",
                "description": "mentor becomes hostile",
                "ripples": ["faction mistrust rises"],
            },
        )
        graph.add_edge("has_consequence", path_id, conseq_id)

    graph.create_node(
        "beat::pre_mentor_01",
        {
            "type": "beat",
            "raw_id": "pre_mentor_01",
            "summary": "Mentor delivers warning",
            "entities": ["character::mentor", "character::kay"],
            "dilemma_impacts": [
                {"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}
            ],
        },
    )
    graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__protector")
    graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__manipulator")

    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        commit_id = f"beat::commit_{ans}"
        graph.create_node(
            commit_id,
            {
                "type": "beat",
                "raw_id": f"commit_{ans}",
                "summary": f"Mentor reveals {ans} motive",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}
                ],
            },
        )
        graph.add_edge("belongs_to", commit_id, path_id)
        graph.add_edge("predecessor", commit_id, "beat::pre_mentor_01")
        for i in range(1, 3):
            post_id = f"beat::post_{ans}_{i:02d}"
            graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": f"Post-commit {i} on {ans}",
                    "entities": ["character::mentor"],
                    "dilemma_impacts": [],
                },
            )
            graph.add_edge("belongs_to", post_id, path_id)
            prev = commit_id if i == 1 else f"beat::post_{ans}_{i - 1:02d}"
            graph.add_edge("predecessor", post_id, prev)

    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": True})

    for ans in ["protector", "manipulator"]:
        flag_id = f"state_flag::mentor_{ans}"
        graph.create_node(
            flag_id,
            {
                "type": "state_flag",
                "raw_id": f"mentor_{ans}",
                "name": f"mentor_is_{ans}",
            },
        )
        graph.add_edge("derived_from", flag_id, f"consequence::mentor_trust__{ans}")

    graph.update_node(
        "dilemma::mentor_trust",
        converges_at="beat::post_protector_02",
        convergence_payoff=2,
    )


def _polish_passage_baseline(graph: Graph) -> None:
    """Add a spec-compliant POLISH passage layer on top of the upstream baseline.

    5 passages using maximal-linear-collapse over the Y-shape beat DAG:
      P_pre   = [pre_mentor_01]              — shared pre-commit, closes at Y-fork
      P_prot  = [commit_protector, post_protector_01, post_protector_02]
      P_mani  = [commit_manipulator, post_manipulator_01, post_manipulator_02]
    One choice edge from P_pre to each of P_prot / P_mani.
    Each entity with ≥2 appearances carries a `character_arc` annotation.
    """
    passage_specs = [
        ("passage::pre", ["beat::pre_mentor_01"], False),
        (
            "passage::prot",
            ["beat::commit_protector", "beat::post_protector_01", "beat::post_protector_02"],
            False,
        ),
        (
            "passage::mani",
            ["beat::commit_manipulator", "beat::post_manipulator_01", "beat::post_manipulator_02"],
            False,
        ),
    ]
    for passage_id, beat_ids, is_variant in passage_specs:
        graph.create_node(
            passage_id,
            {
                "type": "passage",
                "raw_id": passage_id.split("::", 1)[-1],
                "from_beat": beat_ids[0],
                "summary": f"Passage at {beat_ids[0]}",
                "is_variant": is_variant,
            },
        )
        for bid in beat_ids:
            graph.add_edge("grouped_in", bid, passage_id)

    for idx, to_id in enumerate(("passage::prot", "passage::mani")):
        choice_id = f"choice::pre_to_{to_id.rsplit('::', 1)[-1]}"
        graph.create_node(
            choice_id,
            {
                "type": "choice",
                "raw_id": choice_id.split("::", 1)[-1],
                "from_passage": "passage::pre",
                "to_passage": to_id,
                "label": f"Choice {idx + 1}",
                "requires": [],
            },
        )
        graph.add_edge("choice_from", choice_id, "passage::pre")
        graph.add_edge("choice_to", choice_id, to_id)

    # Character arc on the recurring entity.
    graph.update_node(
        "character::mentor",
        character_arc={
            "start": "warning delivered",
            "pivots": {
                "path::mentor_trust__protector": "beat::commit_protector",
                "path::mentor_trust__manipulator": "beat::commit_manipulator",
            },
            "end_per_path": {
                "path::mentor_trust__protector": "beat::post_protector_02",
                "path::mentor_trust__manipulator": "beat::post_manipulator_02",
            },
        },
    )


@pytest.fixture
def compliant_polish_graph() -> Graph:
    graph = Graph()
    _polish_upstream_baseline(graph)
    _polish_passage_baseline(graph)
    return graph


# --------------------------------------------------------------------------
# Positive baseline
# --------------------------------------------------------------------------


def test_valid_polish_graph_passes(compliant_polish_graph: Graph) -> None:
    errors = validate_polish_output(compliant_polish_graph)
    assert errors == [], f"expected no errors, got: {errors}"


# --------------------------------------------------------------------------
# Upstream delegation
# --------------------------------------------------------------------------


def test_upstream_grow_contract_violation_surfaces(compliant_polish_graph: Graph) -> None:
    compliant_polish_graph.update_node("seed_freeze", human_approved=False)
    errors = validate_polish_output(compliant_polish_graph)
    assert any("SEED" in e or "seed_freeze" in e for e in errors), (
        f"expected upstream contract error, got {errors}"
    )


# --------------------------------------------------------------------------
# R-3.3: arc metadata as entity annotation (Cluster #1314)
# --------------------------------------------------------------------------


def test_R_3_3_character_arc_metadata_node_forbidden(compliant_polish_graph: Graph) -> None:
    compliant_polish_graph.create_node(
        "character_arc_metadata::mentor",
        {"type": "character_arc_metadata", "raw_id": "mentor"},
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-3.3" in e or "character_arc_metadata" in e for e in errors), (
        f"expected R-3.3 error, got {errors}"
    )


def test_R_3_3_has_arc_metadata_edge_forbidden(compliant_polish_graph: Graph) -> None:
    compliant_polish_graph.create_node(
        "character_arc_metadata::mentor",
        {"type": "character_arc_metadata", "raw_id": "mentor"},
    )
    compliant_polish_graph.add_edge(
        "has_arc_metadata", "character::mentor", "character_arc_metadata::mentor"
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-3.3" in e or "has_arc_metadata" in e for e in errors), (
        f"expected R-3.3 edge error, got {errors}"
    )


def test_R_3_3_arc_worthy_entity_missing_annotation(compliant_polish_graph: Graph) -> None:
    compliant_polish_graph.update_node("character::mentor", character_arc=None)
    errors = validate_polish_output(compliant_polish_graph)
    assert any(
        "R-3.3" in e or ("character::mentor" in e and "character_arc" in e)
        for e in errors
    ), f"expected missing-annotation error, got {errors}"


# --------------------------------------------------------------------------
# R-4a.4: maximal-linear-collapse (Cluster #1311)
# --------------------------------------------------------------------------


def test_R_4a_4_passage_spans_divergence_forbidden(compliant_polish_graph: Graph) -> None:
    """A passage whose member beats straddle a Y-fork divergence is a grouping error."""
    # Move commit_protector into passage::pre — now the passage spans the Y-fork.
    compliant_polish_graph.remove_edge("grouped_in", "beat::commit_protector", "passage::prot")
    compliant_polish_graph.add_edge("grouped_in", "beat::commit_protector", "passage::pre")
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-4a.4" in e or "divergence" in e.lower() or "linear" in e.lower() for e in errors), (
        f"expected R-4a.4 error, got {errors}"
    )


def test_R_4a_4_passage_stops_mid_linear_run(compliant_polish_graph: Graph) -> None:
    """Splitting a linear run into two passages is a grouping error."""
    # Move post_protector_02 out of passage::prot into a new singleton.
    compliant_polish_graph.remove_edge(
        "grouped_in", "beat::post_protector_02", "passage::prot"
    )
    compliant_polish_graph.create_node(
        "passage::prot_tail",
        {
            "type": "passage",
            "raw_id": "prot_tail",
            "from_beat": "beat::post_protector_02",
            "summary": "Orphan tail",
        },
    )
    compliant_polish_graph.add_edge(
        "grouped_in", "beat::post_protector_02", "passage::prot_tail"
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-4a.4" in e or "linear" in e.lower() for e in errors), (
        f"expected R-4a.4 mid-run split error, got {errors}"
    )


# --------------------------------------------------------------------------
# R-5.7 / R-5.8: residue mapping_strategy (Cluster #1313)
# --------------------------------------------------------------------------


def test_R_5_7_residue_passage_missing_mapping_strategy(
    compliant_polish_graph: Graph,
) -> None:
    compliant_polish_graph.create_node(
        "passage::residue_01",
        {
            "type": "passage",
            "raw_id": "residue_01",
            "from_beat": "beat::post_protector_01",
            "summary": "Residue",
            "residue_for": "passage::prot",
            # mapping_strategy intentionally absent
        },
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-5.7" in e or "R-5.8" in e or "mapping_strategy" in e for e in errors), (
        f"expected missing mapping_strategy error, got {errors}"
    )


def test_R_5_8_residue_passage_bad_mapping_strategy(
    compliant_polish_graph: Graph,
) -> None:
    compliant_polish_graph.create_node(
        "passage::residue_01",
        {
            "type": "passage",
            "raw_id": "residue_01",
            "from_beat": "beat::post_protector_01",
            "summary": "Residue",
            "residue_for": "passage::prot",
            "mapping_strategy": "not_a_valid_value",
        },
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-5.8" in e or "mapping_strategy" in e for e in errors), (
        f"expected invalid mapping_strategy error, got {errors}"
    )


# --------------------------------------------------------------------------
# R-4c.2: zero-choice ERROR halt (Cluster #1312, belt-and-suspenders)
# --------------------------------------------------------------------------


def test_R_4c_2_zero_choice_edges_fails(compliant_polish_graph: Graph) -> None:
    for cid in list(compliant_polish_graph.get_nodes_by_type("choice")):
        compliant_polish_graph.delete_node(cid, cascade=True)
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-4c.2" in e or "zero choice" in e.lower() for e in errors), (
        f"expected zero-choice error, got {errors}"
    )
```

- [ ] **Step 2: Run all new tests to verify they fail**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -x -q
```
Expected: `test_valid_polish_graph_passes` may pass or fail depending on existing validator coverage of the baseline; each of the new R-3.3 / R-4a.4 / R-5.7-8 / R-4c.2 tests FAILS because the helpers don't exist yet. Record which tests pass/fail — tasks 5–8 will flip each failing test green.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/unit/test_polish_contract.py
git commit -m "test(polish): layered baseline + failing hot-path contract tests

Adds _polish_upstream_baseline + _polish_passage_baseline helpers and
rule-by-rule tests for R-3.3 (Cluster #1314), R-4a.4 (Cluster #1311),
R-5.7/R-5.8 (Cluster #1313), R-4c.2 (Cluster #1312).

Tests fail today because the validator helpers don't exist yet;
each is flipped green by a subsequent _check_* helper (tasks 5-8)
or cluster fix (tasks 9-12)."
```

---

### Task 5: `_check_no_character_arc_metadata_nodes` (R-3.3)

**Files:**
- Modify: `src/questfoundry/graph/polish_validation.py`

- [ ] **Step 1: Add the helper and wire it into `validate_polish_output`**

Insert this helper in `polish_validation.py` (place it with the other `_check_*` helpers, after `_check_arc_metadata_edges` near line 419):

```python
def _check_no_character_arc_metadata_nodes(graph: Graph, errors: list[str]) -> None:
    """R-3.3: arc metadata is stored as annotation on Entity nodes, never as separate nodes.

    - No node may have type ``character_arc_metadata``.
    - No ``has_arc_metadata`` edge may exist.
    - Every Entity with ≥2 ``appears`` edges (arc-worthy) must carry a
      ``character_arc`` data dict with ``start``, ``pivots``, ``end_per_path``.
    """
    for nid, ndata in graph.get_nodes_by_type("character_arc_metadata").items():
        errors.append(
            f"R-3.3: node {nid!r} has forbidden type 'character_arc_metadata'; "
            "arc metadata must be stored on the Entity node itself"
        )
    for edge in graph.get_edges(edge_type="has_arc_metadata"):
        errors.append(
            f"R-3.3: forbidden 'has_arc_metadata' edge {edge['from']!r} → "
            f"{edge['to']!r}; arc metadata is an entity annotation, not a "
            "separate node"
        )

    # Arc-worthy entities need a character_arc annotation.
    appears_edges = graph.get_edges(edge_type="appears")
    appearance_count: dict[str, int] = {}
    for edge in appears_edges:
        appearance_count[edge["from"]] = appearance_count.get(edge["from"], 0) + 1

    entity_nodes = graph.get_nodes_by_type("entity")
    for entity_id, entity_data in sorted(entity_nodes.items()):
        if appearance_count.get(entity_id, 0) < 2:
            continue
        arc = entity_data.get("character_arc")
        if not isinstance(arc, dict):
            errors.append(
                f"R-3.3: entity {entity_id!r} has {appearance_count[entity_id]} beat "
                "appearances but no 'character_arc' annotation on its data dict"
            )
            continue
        for required in ("start", "pivots", "end_per_path"):
            if required not in arc:
                errors.append(
                    f"R-3.3: entity {entity_id!r} 'character_arc' annotation "
                    f"missing required field {required!r}"
                )
```

Then add `_check_no_character_arc_metadata_nodes(graph, errors)` to `validate_polish_output`'s helper-call block.

- [ ] **Step 2: Run R-3.3 tests to verify pass**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -k R_3_3 -x -q
```
Expected: 3 passed (test_R_3_3_character_arc_metadata_node_forbidden, test_R_3_3_has_arc_metadata_edge_forbidden, test_R_3_3_arc_worthy_entity_missing_annotation).

Note: the `appears` edges are not present in the baseline yet — Cluster #1314 will wire those up. For now, `test_R_3_3_arc_worthy_entity_missing_annotation` may not trigger the "arc-worthy" branch. That's fine — the test constructs the positive case (entity with 2+ appearances) by other means or documents the edge pre-requisite. If the test fails because of missing `appears` edges, adjust the baseline to add them (see Step 3 below) or the test to manufacture them.

- [ ] **Step 3: Add `appears` edges to `_polish_upstream_baseline`**

If `test_R_3_3_arc_worthy_entity_missing_annotation` still fails because no `appears` edges exist, add them to `_polish_upstream_baseline` after beat creation (match beats to their `entities` list):

```python
# Wire appears(entity, beat) for arc-worthy-entity detection.
for bid, bdata in graph.get_nodes_by_type("beat").items():
    for eid in bdata.get("entities", []) or []:
        graph.add_edge("appears", eid, bid)
```

Re-run the R-3.3 suite.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/graph/polish_validation.py tests/unit/test_polish_contract.py
git commit -m "feat(polish): _check_no_character_arc_metadata_nodes (R-3.3)

Validates that arc metadata is an Entity annotation, not a separate
node.  Flipps the three R-3.3 contract tests green.  Code producing
such nodes/edges today is fixed in Cluster #1314."
```

---

### Task 6: `_check_passage_maximal_linear_collapse` (R-4a.4)

**Files:**
- Modify: `src/questfoundry/graph/polish_validation.py`

- [ ] **Step 1: Add the helper**

Insert this helper in `polish_validation.py` (after the helper from Task 5):

```python
def _check_passage_maximal_linear_collapse(graph: Graph, errors: list[str]) -> None:
    """R-4a.4 (maximal-linear-collapse): a passage's beats form a maximal linear run.

    For each passage:
      - Member beats form a linear run: each interior beat has exactly one
        in-passage predecessor and one in-passage successor.
      - Passage boundaries sit at DAG divergences/convergences or terminals:
        the first beat's in-degree ≠ 1 or its predecessor has out-degree ≠ 1;
        the last beat's out-degree ≠ 1 or its successor has in-degree ≠ 1.

    See docs/design/procedures/polish.md §R-4a.3 (maximal-linear-collapse).
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    passage_nodes = graph.get_nodes_by_type("passage")
    pred_edges = graph.get_edges(edge_type="predecessor")
    grouped_in = graph.get_edges(edge_type="grouped_in")

    successors: dict[str, set[str]] = {}
    predecessors: dict[str, set[str]] = {}
    for edge in pred_edges:
        # Convention: predecessor edges point successor → predecessor
        successor, predecessor = edge["from"], edge["to"]
        successors.setdefault(predecessor, set()).add(successor)
        predecessors.setdefault(successor, set()).add(predecessor)

    beat_to_passage: dict[str, str] = {}
    passage_beats: dict[str, set[str]] = {}
    for edge in grouped_in:
        beat_id, passage_id = edge["from"], edge["to"]
        if beat_id in beat_nodes and passage_id in passage_nodes:
            beat_to_passage[beat_id] = passage_id
            passage_beats.setdefault(passage_id, set()).add(beat_id)

    for passage_id, beats in sorted(passage_beats.items()):
        # Interior linearity: each beat's in-passage predecessors and
        # successors are ≤ 1.
        for bid in beats:
            in_passage_preds = predecessors.get(bid, set()) & beats
            in_passage_succs = successors.get(bid, set()) & beats
            if len(in_passage_preds) > 1:
                errors.append(
                    f"R-4a.4: passage {passage_id!r} contains beat {bid!r} with "
                    f"{len(in_passage_preds)} in-passage predecessors — not a "
                    "linear run"
                )
            if len(in_passage_succs) > 1:
                errors.append(
                    f"R-4a.4: passage {passage_id!r} contains beat {bid!r} with "
                    f"{len(in_passage_succs)} in-passage successors — not a "
                    "linear run"
                )

        # Boundary check: the passage's first beat starts at a real boundary
        # (in-degree ≠ 1 OR its predecessor has out-degree ≠ 1).
        starts = [b for b in beats if not (predecessors.get(b, set()) & beats)]
        ends = [b for b in beats if not (successors.get(b, set()) & beats)]
        if len(starts) != 1 or len(ends) != 1:
            errors.append(
                f"R-4a.4: passage {passage_id!r} is not a single linear run "
                f"(starts={len(starts)}, ends={len(ends)})"
            )
            continue

        first, last = starts[0], ends[0]
        # First beat: if it has exactly one predecessor and that predecessor
        # has out-degree 1, the run should have started earlier.
        first_preds = predecessors.get(first, set())
        if len(first_preds) == 1:
            only_pred = next(iter(first_preds))
            if len(successors.get(only_pred, set())) == 1 and only_pred in beat_to_passage:
                errors.append(
                    f"R-4a.4: passage {passage_id!r} starts at {first!r} but its "
                    f"predecessor {only_pred!r} has out-degree 1 — grouping "
                    "stopped mid-linear-run"
                )

        # Last beat: mirror.
        last_succs = successors.get(last, set())
        if len(last_succs) == 1:
            only_succ = next(iter(last_succs))
            if len(predecessors.get(only_succ, set())) == 1 and only_succ in beat_to_passage:
                errors.append(
                    f"R-4a.4: passage {passage_id!r} ends at {last!r} but its "
                    f"successor {only_succ!r} has in-degree 1 — grouping "
                    "stopped mid-linear-run"
                )
```

Add `_check_passage_maximal_linear_collapse(graph, errors)` to `validate_polish_output`'s helper-call block.

- [ ] **Step 2: Run R-4a.4 tests**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -k R_4a_4 -x -q
```
Expected: 2 passed (test_R_4a_4_passage_spans_divergence_forbidden, test_R_4a_4_passage_stops_mid_linear_run).

- [ ] **Step 3: Commit**

```bash
git add src/questfoundry/graph/polish_validation.py
git commit -m "feat(polish): _check_passage_maximal_linear_collapse (R-4a.4)

Validates passage grouping conforms to the maximal-linear-collapse
rule: interior beats have single in-passage predecessor/successor;
passage boundaries sit at DAG divergences/convergences.  Flips
R-4a.4 contract tests green.  Code producing non-compliant groupings
(intersection-driven) is fixed in Cluster #1311."
```

---

### Task 7: `_check_residue_mapping_strategy` (R-5.7, R-5.8)

**Files:**
- Modify: `src/questfoundry/graph/polish_validation.py`

- [ ] **Step 1: Add the helper**

Insert in `polish_validation.py` (after the Task 6 helper):

```python
_VALID_MAPPING_STRATEGIES = frozenset({"residue_passage_with_variants", "parallel_passages"})


def _check_residue_mapping_strategy(graph: Graph, errors: list[str]) -> None:
    """R-5.7, R-5.8: every residue passage records its chosen mapping strategy.

    Residue passages are identified by a ``residue_for`` field pointing to
    their target shared passage.  Each must have ``mapping_strategy`` set
    to one of the two legal values.  See docs/design/procedures/polish.md
    §R-5.7/R-5.8.
    """
    for pid, pdata in sorted(graph.get_nodes_by_type("passage").items()):
        if not pdata.get("residue_for"):
            continue
        strategy = pdata.get("mapping_strategy")
        if strategy is None:
            errors.append(
                f"R-5.8: residue passage {pid!r} missing required "
                "'mapping_strategy' field"
            )
            continue
        if strategy not in _VALID_MAPPING_STRATEGIES:
            errors.append(
                f"R-5.8: residue passage {pid!r} has invalid mapping_strategy "
                f"{strategy!r} (expected one of "
                f"{sorted(_VALID_MAPPING_STRATEGIES)})"
            )
```

Add `_check_residue_mapping_strategy(graph, errors)` to `validate_polish_output`.

- [ ] **Step 2: Run R-5.7/R-5.8 tests**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -k "R_5_7 or R_5_8" -x -q
```
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add src/questfoundry/graph/polish_validation.py
git commit -m "feat(polish): _check_residue_mapping_strategy (R-5.7, R-5.8)

Validates residue passages carry a mapping_strategy attribute in
{residue_passage_with_variants, parallel_passages}.  Flips R-5.7/R-5.8
contract tests green.  The Phase 5b LLM call and Phase 6 applier are
updated in Cluster #1313 to set this field and branch on it."
```

---

### Task 8: `_check_has_choice_edges` (R-4c.2)

**Files:**
- Modify: `src/questfoundry/graph/polish_validation.py`

- [ ] **Step 1: Add the helper**

Insert in `polish_validation.py`:

```python
def _check_has_choice_edges(graph: Graph, errors: list[str]) -> None:
    """R-4c.2 (belt-and-suspenders): zero choice edges in the passage graph
    indicates a SEED/GROW bug — Phase 4c should already have raised.  This
    check catches silent regressions where Phase 4c produced zero choices
    but did not halt.
    """
    if not graph.get_nodes_by_type("choice"):
        errors.append(
            "R-4c.2: zero choice edges in passage graph — SEED/GROW DAG "
            "has no Y-forks; Phase 4c should have halted"
        )
```

Add `_check_has_choice_edges(graph, errors)` to `validate_polish_output`.

- [ ] **Step 2: Run R-4c.2 test**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py -k R_4c_2 -x -q
```
Expected: 1 passed (test_R_4c_2_zero_choice_edges_fails).

- [ ] **Step 3: Commit**

```bash
git add src/questfoundry/graph/polish_validation.py
git commit -m "feat(polish): _check_has_choice_edges (R-4c.2, belt-and-suspenders)

Validates that the passage graph contains at least one choice edge.
Phase 4c itself halts when compute_choice_edges returns empty (added
in Cluster #1312); this postcondition catches silent regressions
where Phase 4c produced zero choices but did not halt."
```

---

## Phase 3 — Cluster fixes

### Task 9: Cluster #1314 (R-3.3) — arc metadata as entity annotation

**Files:**
- Modify: `src/questfoundry/pipeline/stages/polish/llm_phases.py:286-299`

- [ ] **Step 1: Locate current code**

Read `src/questfoundry/pipeline/stages/polish/llm_phases.py` around lines 280-310.  Current code creates a separate `character_arc_metadata::{id}` node and a `has_arc_metadata` edge per arc-worthy entity.

- [ ] **Step 2: Replace with entity-data annotation**

Replace the block at lines 286-299 (the `graph.create_node("character_arc_metadata::...")` and `graph.add_edge("has_arc_metadata", ...)` calls) with:

```python
# R-3.3: arc metadata is an annotation on the Entity node itself,
# never a separate node.  Mutate the entity's data dict in-place.
graph.update_node(
    entity_id,
    character_arc={
        "start": arc_data.start,
        "pivots": dict(arc_data.pivots),
        "end_per_path": dict(arc_data.end_per_path),
    },
)
```

(Adjust the `arc_data` attribute accesses to match the actual Pydantic model field names used locally — confirm by reading the surrounding loop.)

- [ ] **Step 3: Run Phase 3 tests**

Run:
```bash
uv run pytest tests/unit/test_polish_phases.py -k "phase3 or arc" -x -q
```
Expected: the test `test_has_arc_metadata_edge_created` now fails (the edge is no longer created — correct behavior). Leave it failing; it's deleted in Task 13.

Run the contract tests to confirm Cluster #1314 is now green against compliant-baseline construction:
```bash
uv run pytest tests/unit/test_polish_contract.py -k R_3_3 -x -q
```
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/pipeline/stages/polish/llm_phases.py
git commit -m "fix(polish): store character_arc on Entity nodes, not as separate nodes (R-3.3)

Closes cluster #1314 of epic #1310.  Phase 3 arc synthesis now
annotates the Entity node's data dict with 'character_arc' (start,
pivots per path, end_per_path) rather than creating a separate
'character_arc_metadata' node linked by 'has_arc_metadata'.  Matches
spec R-3.3 and ontology Part 1 Character Arc Metadata.

The old test_has_arc_metadata_edge_created is now correctly failing
and is deleted in the fixture-cleanup task."
```

---

### Task 10: Cluster #1312 (R-4c.2) — zero-choice ERROR halt

**Files:**
- Modify: `src/questfoundry/pipeline/stages/polish/deterministic.py:146` (inside `phase_plan_computation`)

- [ ] **Step 1: Locate current code**

Read `deterministic.py` around lines 103-150.  After the line `plan.choice_specs = compute_choice_edges(graph, plan.passage_specs)` there is no zero-choice check.

- [ ] **Step 2: Write a failing test**

Append to `tests/unit/test_polish_contract.py`:

```python
def test_phase_4c_zero_choices_raises_contract_error() -> None:
    """Phase 4c raises PolishContractError when compute_choice_edges returns empty."""
    import asyncio
    from unittest.mock import MagicMock

    from questfoundry.pipeline.stages.polish import deterministic

    # A graph with passages but no Y-forks → zero choice edges.  Simplest
    # case: an empty graph.  Phase 4a will also fail on this, but the
    # assertion is specifically about the zero-choice halt once reached.
    graph = Graph.empty()

    with pytest.raises(PolishContractError, match=r"R-4c\.2|zero choice"):
        # phase_plan_computation delegates Phase 4c internally.
        asyncio.run(deterministic.phase_plan_computation(graph, MagicMock()))
```

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py::test_phase_4c_zero_choices_raises_contract_error -x -q
```
Expected: FAIL — current code does not raise.

- [ ] **Step 3: Add the zero-choice halt**

Modify `phase_plan_computation` in `deterministic.py` around line 146 (after `plan.choice_specs = compute_choice_edges(...)`) to insert:

```python
if not plan.choice_specs:
    from questfoundry.graph.polish_validation import PolishContractError

    log.error(
        "polish_zero_choice_halt",
        upstream="SEED/GROW",
        passage_count=len(plan.passage_specs),
        detail="Phase 4c produced zero choice edges — upstream DAG has no Y-forks",
    )
    raise PolishContractError(
        "R-4c.2: Phase 4c produced zero choice edges — SEED/GROW DAG has "
        "no Y-forks.  Upstream bug — halting POLISH."
    )
```

- [ ] **Step 4: Run the test**

Run:
```bash
uv run pytest tests/unit/test_polish_contract.py::test_phase_4c_zero_choices_raises_contract_error -x -q
```
Expected: PASS.

Also run broader POLISH-local suites to check nothing else regressed:
```bash
uv run pytest tests/unit/test_polish_deterministic.py tests/unit/test_polish_validation.py -x -q
```
Expected: pre-existing pass counts unchanged (or any new failures are already-pre-audit fixtures that Task 14 will rewrite).

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/pipeline/stages/polish/deterministic.py tests/unit/test_polish_contract.py
git commit -m "fix(polish): halt Phase 4c with ERROR on zero choice edges (R-4c.2)

Closes cluster #1312 of epic #1310.  If compute_choice_edges returns
empty, Phase 4c now logs polish_zero_choice_halt at ERROR level and
raises PolishContractError identifying SEED/GROW as the upstream
source of the failure.  Previously this silently passed through to
Phase 6, violating Silent Degradation policy (CLAUDE.md §Anti-Patterns)."
```

---

### Task 11: Cluster #1311 (R-4a.4) — maximal-linear-collapse grouping

**Files:**
- Modify: `src/questfoundry/pipeline/stages/polish/deterministic.py:178-608` (the `compute_beat_grouping` function)

This task is the largest. Delete the intersection-group iteration branch entirely and replace the algorithm with a single-pass DAG walk that partitions beats into maximal linear runs.

- [ ] **Step 1: Locate current code**

Read `deterministic.py:178-608` (the `compute_beat_grouping` function and its helpers). Note in particular:
- Lines 213-234: iterates intersection groups and creates `grouping_type="intersection"` passages.
- Subsequent blocks handle other grouping mechanisms.

- [ ] **Step 2: Rewrite `compute_beat_grouping`**

Replace the function body with the maximal-linear-collapse algorithm:

```python
def compute_beat_grouping(graph: Graph) -> list[PassageSpec]:
    """Phase 4a: group beats into passages using the maximal-linear-collapse rule.

    Walk the finalized beat DAG; partition beats into maximal linear runs
    (no internal divergence or convergence).  Applies uniformly to
    narrative and structural beats.  Passage boundaries sit at DAG
    divergences or convergences; every passage ends at a choice point
    (divergence → choice edges in Phase 4c) or a convergence/terminal.

    POLISH does NOT consume intersection groups — those are GROW-internal.
    See docs/design/procedures/polish.md §R-4a.3, R-4a.4.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return []

    pred_edges = graph.get_edges(edge_type="predecessor")
    successors: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    predecessors: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    for edge in pred_edges:
        # predecessor edges point successor → predecessor
        succ, pred = edge["from"], edge["to"]
        if succ in beat_nodes and pred in beat_nodes:
            successors[pred].add(succ)
            predecessors[succ].add(pred)

    def _is_run_start(bid: str) -> bool:
        preds = predecessors[bid]
        if len(preds) != 1:
            return True  # root or convergence
        only_pred = next(iter(preds))
        return len(successors[only_pred]) != 1  # predecessor forks

    specs: list[PassageSpec] = []
    assigned: set[str] = set()

    # Order start beats deterministically so fixture tests are stable.
    start_beats = sorted(bid for bid in beat_nodes if _is_run_start(bid))

    for start in start_beats:
        if start in assigned:
            continue
        run = [start]
        assigned.add(start)
        current = start
        while True:
            succs = successors[current]
            if len(succs) != 1:
                break  # divergence or terminal
            nxt = next(iter(succs))
            if len(predecessors[nxt]) != 1 or nxt in assigned:
                break  # convergence or already taken
            run.append(nxt)
            assigned.add(nxt)
            current = nxt

        specs.append(
            PassageSpec(
                passage_id=f"passage::{run[0].split('::', 1)[-1]}",
                beat_ids=list(run),
                from_beat=run[0],
                summary=beat_nodes[run[0]].get("summary", ""),
            )
        )

    # Safety: every beat must be assigned to exactly one PassageSpec.  Any
    # leftover indicates a graph-topology anomaly; surface loudly rather
    # than silently skip — matches Silent Degradation policy.
    leftover = sorted(set(beat_nodes) - assigned)
    if leftover:
        raise ValueError(
            f"compute_beat_grouping: {len(leftover)} beats not assigned to "
            f"any passage — graph may have disconnected components or "
            f"malformed predecessor edges: {leftover[:5]}"
        )

    return specs
```

Remove any now-unused helpers (e.g. `_group_by_intersection_signal`, or similar — find via grep inside `deterministic.py` after the replacement).

- [ ] **Step 3: Run Phase 4a tests**

Run:
```bash
uv run pytest tests/unit/test_polish_deterministic.py -x -q
```
Expected: previously-passing grouping tests that assume intersection-driven grouping will fail. Those are rewritten in Task 14. Tests that check general structural properties (every beat in one passage, passage has ≥1 beat) should still pass.

Run the R-4a.4 contract tests:
```bash
uv run pytest tests/unit/test_polish_contract.py -k R_4a_4 -x -q
```
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/pipeline/stages/polish/deterministic.py
git commit -m "fix(polish): rewrite compute_beat_grouping with maximal-linear-collapse (R-4a.4)

Closes cluster #1311 of epic #1310.  Phase 4a no longer consumes
Intersection Group nodes — those are GROW-internal per the track-1
architectural correction.  Replaces intersection-driven grouping with
a single-pass DAG walk: partition beats into maximal linear runs
ending at divergences/convergences.  Applies uniformly to narrative
and structural beats per updated R-4a.3 (in docs PR #1358).

Tests that asserted intersection-driven grouping are rewritten in
the fixture-cleanup task."
```

---

### Task 12: Cluster #1313 (R-5.7, R-5.8) — residue `mapping_strategy`

**Files:**
- Modify: `src/questfoundry/models/polish.py:181-194` (add field)
- Modify: `src/questfoundry/pipeline/stages/polish/llm_phases.py` (Phase 5b prompt + schema + collection)
- Modify: `src/questfoundry/pipeline/stages/polish/deterministic.py:1176-1241` (Phase 6 branching)

- [ ] **Step 1: Extend `ResidueSpec`**

Modify `ResidueSpec` in `src/questfoundry/models/polish.py:181-194`:

```python
class ResidueSpec(BaseModel):
    """Specification for a residue beat.

    Created during Phase 4b for passages with light/cosmetic residue
    flags.  Residue beats are mood-setting moments before shared passages.
    """

    target_passage_id: str = Field(min_length=1)
    residue_id: str = Field(min_length=1)
    flag: str = Field(min_length=1, description="State flag this residue addresses")
    path_id: str = Field(default="")
    content_hint: str = Field(
        default="", description="Mood-setting prose hint (populated by Phase 5)"
    )
    mapping_strategy: Literal["residue_passage_with_variants", "parallel_passages"] = Field(
        description=(
            "Passage-layer mapping for this residue spec.  "
            "'residue_passage_with_variants' creates a single residue passage "
            "with variant children; 'parallel_passages' creates sibling "
            "passages that branch and rejoin.  Chosen per-residue by the "
            "Phase 5b LLM call per spec R-5.7/R-5.8."
        ),
    )
```

Add the `Literal` import at the top of the file if not already present:

```python
from typing import Literal
```

- [ ] **Step 2: Update Phase 5b LLM call**

In `src/questfoundry/pipeline/stages/polish/llm_phases.py`, locate the Phase 5b residue content generation call (search for `ResidueSpec` construction or `residue_content`). Add `mapping_strategy` to the structured-output schema and the prompt:

- Extend the Pydantic schema passed to `with_structured_output` so the LLM returns a `mapping_strategy` alongside `content_hint`.
- In the prompt template (inline or in `prompts/templates/`), add a paragraph describing both options:
  - `residue_passage_with_variants` — one passage, variants differentiate prose by flag combo (compact).
  - `parallel_passages` — sibling passages that branch and rejoin (clearer for distinct moods).
- Store the LLM's choice on the resulting `ResidueSpec`.

(Specific line numbers and template names depend on the current Phase 5b layout. The implementer identifies them by reading around the existing residue-content-generation block and mirrors the structure.)

- [ ] **Step 3: Update Phase 6 applier**

In `src/questfoundry/pipeline/stages/polish/deterministic.py:1176-1241` (`_create_residue_beat_and_passage`), branch on `spec.mapping_strategy`:

```python
def _create_residue_beat_and_passage(graph: Graph, rspec: ResidueSpec) -> None:
    """Phase 6: materialize a residue spec into the passage layer.

    Branches on ``rspec.mapping_strategy`` per R-5.7/R-5.8:
      - ``residue_passage_with_variants`` — one residue passage with
        variant children carrying flag-gated prose.
      - ``parallel_passages`` — sibling passages that branch from the
        target's predecessor and rejoin at the target.
    """
    if rspec.mapping_strategy == "residue_passage_with_variants":
        _apply_residue_with_variants(graph, rspec)
    elif rspec.mapping_strategy == "parallel_passages":
        _apply_residue_parallel_passages(graph, rspec)
    else:  # defensive — _check_residue_mapping_strategy catches this too
        raise ValueError(
            f"R-5.8: unknown mapping_strategy {rspec.mapping_strategy!r} for "
            f"residue {rspec.residue_id!r}"
        )
```

Extract the existing body of `_create_residue_beat_and_passage` into a new function `_apply_residue_with_variants` (preserves current behavior — that was the single shape POLISH used). Add a new `_apply_residue_parallel_passages` that creates N sibling passages branching from the target's predecessor and rejoining at the target (each sibling's prose is gated by the per-flag-combo requirements):

```python
def _apply_residue_parallel_passages(graph: Graph, rspec: ResidueSpec) -> None:
    """Alternative residue mapping: parallel passages that branch and rejoin.

    For a residue spec with N distinct flag combinations, create N sibling
    passages.  Each sibling's content is gated by its flag combo and its
    choice edges lead from the shared predecessor to the shared target.
    """
    # TODO(implementer): collect flag-combos from the surrounding plan, create
    # N passages, wire choice edges from predecessor to each and choice edges
    # from each to rspec.target_passage_id.  Record mapping_strategy on each
    # new passage node for Phase 7 validation.
    raise NotImplementedError(
        "Parallel-passages residue mapping is implemented here; see plan "
        "task 12 step 3 for the algorithm."
    )
```

Also set `mapping_strategy` on the created passage node's data dict in both branches:

```python
graph.create_node(
    passage_id,
    {
        "type": "passage",
        # ... existing fields ...
        "residue_for": rspec.target_passage_id,
        "mapping_strategy": rspec.mapping_strategy,
    },
)
```

Note: if `parallel_passages` is not yet exercised by any LLM-output fixture, its implementation may be a stub that raises `NotImplementedError` — provided no current test exercises it. Track any stub in a follow-on issue filed before merge (see Task 18).

- [ ] **Step 4: Run Phase 5/6 and contract tests**

Run:
```bash
uv run pytest tests/unit/test_polish_apply.py tests/unit/test_polish_llm_phases.py tests/unit/test_polish_contract.py -k "R_5_7 or R_5_8 or residue" -x -q
```
Expected: R-5.7/R-5.8 contract tests pass. Existing residue tests may now fail because the new required field is missing from fixtures — Task 15 rewrites them.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/models/polish.py src/questfoundry/pipeline/stages/polish/llm_phases.py src/questfoundry/pipeline/stages/polish/deterministic.py
git commit -m "feat(polish): residue mapping_strategy field + Phase 6 branching (R-5.7, R-5.8)

Closes cluster #1313 of epic #1310.  ResidueSpec gains a required
mapping_strategy Literal field.  Phase 5b LLM call asks the model
to choose between 'residue_passage_with_variants' (compact, single
passage with variant children) and 'parallel_passages' (sibling
passages branching then rejoining).  Phase 6 branches on the chosen
strategy and records it on the residue passage node's data dict so
Phase 7 can validate (R-5.8 enforcement via
_check_residue_mapping_strategy)."
```

---

## Phase 4 — Fixture cleanup + sweep

### Task 13: Delete `test_has_arc_metadata_edge_created`; add replacement

**Files:**
- Modify: `tests/unit/test_polish_phases.py:398-436`

- [ ] **Step 1: Locate the old test**

Read lines 398-436 of `tests/unit/test_polish_phases.py`. The test asserts that after Phase 3, a `character_arc_metadata::{id}` node exists and a `has_arc_metadata` edge from the entity to the arc node exists. Its premise is invalid post-audit.

- [ ] **Step 2: Delete the old test**

Remove the entire `def test_has_arc_metadata_edge_created(...)` function and its docstring (lines 398-436).

- [ ] **Step 3: Add the replacement**

Insert in the same location:

```python
def test_character_arc_field_on_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    """R-3.3: Phase 3 annotates the Entity node's data dict with
    'character_arc'; no separate 'character_arc_metadata' node and no
    'has_arc_metadata' edge are created."""
    # (Implementer: reconstruct the existing test harness from the deleted
    # test — mock LLM, run Phase 3, then assert:
    #   - The entity node has a 'character_arc' key in its data dict.
    #   - No node has type 'character_arc_metadata'.
    #   - No edge has type 'has_arc_metadata'.
    # Concrete implementation mirrors the pattern of the removed test.)
    ...
```

Replace the `...` placeholder with the actual test code — the setup parallels the removed test. The assertion block:

```python
    assert "character_arc" in entity_data
    assert graph.get_nodes_by_type("character_arc_metadata") == {}
    assert graph.get_edges(edge_type="has_arc_metadata") == []
```

- [ ] **Step 4: Run**

```bash
uv run pytest tests/unit/test_polish_phases.py -x -q
```
Expected: new test passes. Other Phase 3 tests pass (arc synthesis still runs; only storage shape changed).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_polish_phases.py
git commit -m "test(polish): replace test_has_arc_metadata_edge_created with character_arc-field test

Old test asserted pre-audit behavior (separate character_arc_metadata
node + has_arc_metadata edge) which R-3.3 and ontology Part 1 forbid.
Replacement tests the new annotation-on-Entity shape (cluster #1314)."
```

---

### Task 14: Rewrite Phase 4a grouping tests

**Files:**
- Modify: `tests/unit/test_polish_deterministic.py`

- [ ] **Step 1: Identify intersection-driven assertions**

Grep for references to intersection-driven grouping:

```bash
grep -n "grouping_type.*intersection\|intersection_group\|compute_beat_grouping" tests/unit/test_polish_deterministic.py | head -20
```

Record the tests that depend on intersection-driven grouping. Each needs its assertions rewritten to express maximal-linear-collapse outputs.

- [ ] **Step 2: Rewrite each affected test**

For each identified test, update the assertions. Pattern:

- **Before:** "Phase 4a creates an intersection passage for beats in the same intersection group."
- **After:** "Phase 4a collapses beats into a linear run and closes the passage at the next DAG divergence/convergence."

Example rewrite (skeleton):

```python
def test_phase_4a_collapses_linear_chain(sample_graph_with_linear_chain: Graph) -> None:
    """Phase 4a groups a linear chain of 3 beats into one passage (R-4a.3)."""
    from questfoundry.pipeline.stages.polish.deterministic import compute_beat_grouping

    specs = compute_beat_grouping(sample_graph_with_linear_chain)

    assert len(specs) == 1
    assert specs[0].beat_ids == ["beat::a", "beat::b", "beat::c"]


def test_phase_4a_closes_at_divergence(sample_graph_with_yfork: Graph) -> None:
    """Phase 4a closes a passage at a Y-fork; each branch starts a new passage."""
    from questfoundry.pipeline.stages.polish.deterministic import compute_beat_grouping

    specs = compute_beat_grouping(sample_graph_with_yfork)

    beat_sets = [set(s.beat_ids) for s in specs]
    # Shared pre-commit is one passage; each commit + post-commit chain is its own.
    assert {"beat::pre_01"} in beat_sets
```

Concrete rewrites are guided by each test's original fixture — the implementer reads the pre-audit assertion and translates it into maximal-linear-collapse terms.

- [ ] **Step 3: Run the full Phase 4a test file**

```bash
uv run pytest tests/unit/test_polish_deterministic.py -x -q
```
Expected: all tests pass. If any test's premise is pre-audit-only (cannot be rephrased under the new rule), delete it and file a follow-on issue (see Task 18) if coverage is lost.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_polish_deterministic.py
git commit -m "test(polish): rewrite Phase 4a grouping tests for maximal-linear-collapse

Intersection-driven grouping assertions replaced with DAG-topology
assertions.  Closes fixture-rewrite coverage gap for cluster #1311."
```

---

### Task 15: Rewrite residue tests

**Files:**
- Modify: `tests/unit/test_polish_apply.py`

- [ ] **Step 1: Identify residue tests**

```bash
grep -n "residue\|ResidueSpec\|_create_residue" tests/unit/test_polish_apply.py | head -20
```

- [ ] **Step 2: Parameterize over `mapping_strategy`**

For each test that creates a `ResidueSpec` or exercises residue Phase 6 application, add a `mapping_strategy` argument. Where feasible, parameterize:

```python
@pytest.mark.parametrize(
    "mapping_strategy",
    ["residue_passage_with_variants", "parallel_passages"],
)
def test_residue_applies_correctly(mapping_strategy: str, sample_graph: Graph) -> None:
    spec = ResidueSpec(
        target_passage_id="passage::target",
        residue_id="res_01",
        flag="flag_x",
        mapping_strategy=mapping_strategy,
    )
    # ... rest of test parameterized where shape differs ...
```

For tests that only cover one strategy, add the `mapping_strategy` argument with a single explicit value.

- [ ] **Step 3: Run**

```bash
uv run pytest tests/unit/test_polish_apply.py -x -q
```
Expected: all pass. If `parallel_passages` implementation in Task 12 was stubbed with `NotImplementedError`, mark those parameter cases `xfail` with a reference to the tracking issue filed in Task 18.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_polish_apply.py
git commit -m "test(polish): parameterize residue tests over mapping_strategy

Covers both residue_passage_with_variants and parallel_passages
values of the new field (cluster #1313).  Any stubbed branch is
marked xfail with a tracking-issue reference."
```

---

### Task 16: Phase 5b + Phase 3 LLM tests update

**Files:**
- Modify: `tests/unit/test_polish_llm_phases.py`

- [ ] **Step 1: Phase 3 tests — drop expectation of separate arc nodes**

Find any test that asserts `graph.get_nodes_by_type("character_arc_metadata")` is non-empty or that `has_arc_metadata` edges exist. Replace with assertions on the entity's `character_arc` data field (same shape as Task 13).

- [ ] **Step 2: Phase 5b tests — add `mapping_strategy` to expected structured output**

Find the Phase 5b residue-content test(s). The mock LLM now returns a dict with `mapping_strategy`; update the expected values:

```python
# Mock output for Phase 5b:
{
    "residues": [
        {
            "residue_id": "res_01",
            "content_hint": "A quiet moment",
            "mapping_strategy": "residue_passage_with_variants",
        },
    ],
}
```

Update any assertions that inspect the returned `ResidueSpec` to verify `mapping_strategy` is populated.

- [ ] **Step 3: Run**

```bash
uv run pytest tests/unit/test_polish_llm_phases.py -x -q
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_polish_llm_phases.py
git commit -m "test(polish): update Phase 3 + Phase 5b LLM tests for new field shapes

Phase 3 tests assert on the character_arc entity annotation (R-3.3).
Phase 5b tests include mapping_strategy in expected LLM output and
assert it lands on the returned ResidueSpec (R-5.7/R-5.8)."
```

---

### Task 17: Non-downstream sweep

**Files:** None (verification only).

- [ ] **Step 1: Static checks**

```bash
uv run mypy src/questfoundry/
uv run ruff check src/
uv run pyright src/
```
Expected: all clean (zero errors).

- [ ] **Step 2: POLISH-local suite**

```bash
uv run pytest \
  tests/unit/test_polish_contract.py \
  tests/unit/test_polish_validation.py \
  tests/unit/test_polish_phases.py \
  tests/unit/test_polish_deterministic.py \
  tests/unit/test_polish_apply.py \
  tests/unit/test_polish_llm_phases.py \
  -q
```
Expected: all pass.

- [ ] **Step 3: Upstream regression check**

Run the stages upstream of POLISH to confirm no leakage:

```bash
uv run pytest \
  tests/unit/test_dream_stage.py \
  tests/unit/test_brainstorm_stage.py \
  tests/unit/test_seed_stage.py tests/unit/test_seed_validation.py \
  tests/unit/test_grow_validation.py tests/unit/test_grow_validation_contract.py \
  tests/unit/test_grow_algorithms.py \
  tests/integration/test_grow_e2e.py \
  -q
```
Expected: all pass.

- [ ] **Step 4: Full non-downstream unit sweep**

```bash
uv run pytest tests/unit/ -q \
  --ignore=tests/unit/test_fill_context.py \
  --ignore=tests/unit/test_fill_models.py \
  --ignore=tests/unit/test_fill_stage.py \
  --ignore=tests/unit/test_fill_validation.py \
  --ignore=tests/unit/test_fill_continuity_warning.py \
  --ignore=tests/unit/test_dress_context.py \
  --ignore=tests/unit/test_dress_models.py \
  --ignore=tests/unit/test_dress_mutations.py \
  --ignore=tests/unit/test_dress_stage.py \
  --ignore=tests/unit/test_ship_*.py \
  --ignore=tests/unit/test_polish_passage_validation.py
```
Expected: all pass (the specific ignore list is confirmed empirically — add any post-POLISH test files whose premise is genuinely pre-audit).

- [ ] **Step 5: Record any surprises**

If a test outside the ignore list fails, investigate root cause before committing. Either rewrite it if its intent is still valid post-audit, or add to the ignore list + file a follow-on issue.

- [ ] **Step 6: No commit — verification only**

Skip.

---

### Task 18: Downstream-break catalog + follow-on issues

**Files:** None (GitHub issues only).

- [ ] **Step 1: Enumerate broken downstream tests**

Identify tests that break because of the POLISH contract tightening:

```bash
uv run pytest tests/unit/test_fill_*.py tests/unit/test_dress_*.py tests/unit/test_ship_*.py tests/unit/test_polish_passage_validation.py -q --no-header 2>&1 | grep -E "FAIL|ERROR" | head -40
```
Expected: list of failing tests.

- [ ] **Step 2: Group and file follow-on issues**

For each distinct group (e.g., "FILL context reads character_arc_metadata nodes", "DRESS iteration depends on intersection-group-derived passages"), file a follow-on issue via `gh issue create` with:

- Title: `[polish-compliance follow-on] <short description>`
- Body: list the broken test(s), the root cause (which cluster's contract they conflict with), and a pointer to this PR.

- [ ] **Step 3: Also file stubs for Task 12's `parallel_passages` if stubbed**

If Task 12 left `_apply_residue_parallel_passages` as a `NotImplementedError` stub, file an issue: `[polish-compliance follow-on] implement parallel-passages residue mapping`.

- [ ] **Step 4: No commit — GitHub state only**

Skip.

---

## Phase 5 — Ship

### Task 19: Push branch + open PR

**Files:** None (git/gh operations).

- [ ] **Step 1: Final status check**

```bash
git status
git log --oneline origin/main..HEAD | head -20
```
Expected: clean tree; commits cover design doc + Phase 1 + Phase 2 + Phase 3 + Phase 4.

- [ ] **Step 2: Push**

```bash
git push -u origin feat/polish-compliance
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --base main --title 'feat(polish): compliance with authoritative spec (hot-path clusters)' --body "$(cat <<'EOF'
## Summary

Brings the POLISH stage into compliance with the authoritative spec for the 4 hot-path clusters of epic #1310. Establishes a stage-exit contract validator with a dedicated \`PolishContractError\` raised from Phase 7, mirroring the DREAM/BRAINSTORM/SEED/GROW compliance pattern.

Closes #1311, #1312, #1313, #1314.

Does NOT close the epic (#1310) — clusters #1315, #1316, #1317, #1318 are deferred to follow-on PRs.

## What changed

**Validator contract (new):**
- \`PolishContractError(ValueError)\` raised from Phase 7 when \`validate_polish_output\` reports errors.
- Phase 7 emits a structured \`log.error("polish_contract_failed", …)\` event before raising.
- 4 new \`_check_*\` helpers: \`_check_no_character_arc_metadata_nodes\` (R-3.3), \`_check_passage_maximal_linear_collapse\` (R-4a.4), \`_check_residue_mapping_strategy\` (R-5.7/R-5.8), \`_check_has_choice_edges\` (R-4c.2 belt-and-suspenders).

**Per-cluster fixes:**
- #1311 (R-4a.4) — \`compute_beat_grouping\` rewritten as maximal-linear-collapse; intersection-group iteration removed entirely. Prereq spec change landed in #1358.
- #1312 (R-4c.2) — Phase 4c halts with \`PolishContractError\` when \`compute_choice_edges\` returns empty.
- #1313 (R-5.7, R-5.8) — \`ResidueSpec.mapping_strategy\` required field; Phase 5b prompt asks LLM to choose; Phase 6 branches on it.
- #1314 (R-3.3) — Phase 3 arc metadata stored on Entity data dict as \`character_arc\`; no more \`character_arc_metadata\` nodes or \`has_arc_metadata\` edges.

## Allowed-to-break

POLISH, FILL, DRESS, SHIP tests that depend on pre-compliance POLISH output are allowed to fail. Follow-on issues filed (see below).

## Test plan

- [ ] POLISH contract tests: all pass (\`tests/unit/test_polish_contract.py\`).
- [ ] POLISH-local suites all pass (validation, phases, deterministic, apply, llm_phases).
- [ ] Non-downstream sweep passes with documented ignore list.
- [ ] \`uv run mypy src/\`, \`ruff check src/\`, \`pyright src/\` — all clean.
- [ ] CI must verify across Python 3.11 / 3.12 / 3.13.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Record PR URL**

Report the URL returned by `gh pr create` to the user as the terminal signal for this plan.

---

## Self-Review Notes

**Spec coverage:** Design doc sections 1–7 each have tasks:
- §1 (sequencing) — prereq verified in Task 1.
- §3 (validator extensions) — Tasks 2–8 (class, wiring, 4 helpers with failing-first tests).
- §4 (per-cluster fixes) — Tasks 9–12 (one task per cluster, ordered small-to-large).
- §5 (fixtures + downstream break) — Tasks 13–16 rewrite fixtures; Task 18 files follow-on issues.
- §6 (work sequence) — matches the 19-task layout exactly.
- §7 (exit criteria) — Task 17 sweeps; Task 19's PR body enumerates.

**Type consistency:** `PolishContractError`, `mapping_strategy`, `character_arc`, and the four `_check_*` helper names are used identically across tasks. `Literal["residue_passage_with_variants", "parallel_passages"]` appears the same way in model, validator, prompt, and Phase 6 code.

**Placeholder scan:** Two placeholder-adjacent spots were intentionally left for the implementer because they depend on reading the current surrounding code:
- Task 12 Step 2: Phase 5b prompt template name and line numbers — the implementer locates them by reading around the existing residue-content-generation block.
- Task 13 Step 3: the replacement test reconstructs the mock/setup harness from the deleted test.

Both are guided by concrete surrounding code, not left as "TBD" — they tell the implementer exactly what to look for.
