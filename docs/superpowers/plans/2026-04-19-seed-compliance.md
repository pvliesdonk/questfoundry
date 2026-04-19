# SEED Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring SEED into compliance with `docs/design/procedures/seed.md`, close all 14 clusters of epic #1281, add `validate_seed_output` as the runtime oracle for SEED's Stage Output Contract, and remove the `models/seed.py` pyright suppression.

**Architecture:** Pure-function validator returning `list[str]` (matching DREAM/BRAINSTORM pattern in PR #1351). `apply_seed_mutations` in `graph/mutations.py` calls the validator after graph writes and raises `SeedContractError` on non-empty results. Decomposed into 7 private `_check_*` helpers to keep each concern small and testable.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, `uv`, `ruff`, `mypy`, `pyright` (standard mode).

**Spec:** `docs/superpowers/specs/2026-04-19-seed-compliance-design.md`
**Branch:** `feat/seed-compliance` (already created; the spec commit is `84d12bd4`).

---

## Reference context

### Authoritative spec — key rule groupings

From `docs/design/procedures/seed.md`. Only rules directly exercised in this plan are listed; see the spec for full wording.

- **Phase 1 Entity Triage:** R-1.1 disposition set, R-1.2 no-cut-while-anchored, R-1.3 no-new-entities, R-1.4 two-location-minimum survives.
- **Phase 2 Answer Selection:** R-2.1 canonical is explored, R-2.2 each non-canonical decided, R-2.3 `explored` immutable, R-2.4 decisions for all answers.
- **Phase 3 Path Construction:** R-3.1 one Path per explored Answer, R-3.2 path-id naming, R-3.3 ≥1 Consequence, R-3.4 ≥1 ripple with description, R-3.5 world-state not player-action, **R-3.6 pre-commit dual `belongs_to` same dilemma**, R-3.7 commit single + `effect: commits`, R-3.8 post-commit single + no commits, **R-3.9 no cross-dilemma dual `belongs_to`**, **R-3.10 ≥1 pre-commit per explored dilemma**, R-3.11 one commit beat per path, R-3.12 2–4 post-commit per path, R-3.13 non-empty summary + entities, R-3.14 setup/epilogue structural, R-3.15 setup/epilogue optional but non-empty when present.
- **Phase 3b Flexibility:** R-3b.1 preserves dramatic function, R-3b.2 `role` property, R-3b.3 any category, R-3b.4 advisory.
- **Phase 4 Convergence:** R-4.1 hint not binding, R-4.2 hard dilemmas no intent, R-4.3 if-can't-rejoin-then-hard, R-4.4 loop-back allowed.
- **Phase 5 Viability:** R-5.1 arc count ≤16, R-5.2 pruning doesn't mutate `explored`, R-5.3 logged at INFO, R-5.4 no orphans after prune.
- **Phase 6 Path Freeze:** R-6.1 edge endpoints exist, R-6.2 beat entities are retained, R-6.3 Path has full Y-shape, R-6.4 human approval recorded, R-6.5 downstream doesn't create new Path/Entity/Dilemma/Consequence.
- **Phase 7 Dilemma Analysis:** R-7.1 `dilemma_role` ∈ {hard, soft}, R-7.2 `residue_weight` ∈ {heavy, light, cosmetic}, R-7.3 `ending_salience` ∈ {high, low, none}, R-7.4 independent axes, **R-7.5 LLM failure logged at WARNING — no silent defaults**.
- **Phase 8 Ordering:** R-8.1 relationships ∈ {wraps, concurrent, serial}, R-8.2 sparse only, R-8.3 `concurrent` symmetric + lex-smaller as dilemma_a, R-8.4 `shared_entity` derived not declared, **R-8.5 LLM failure logged at WARNING**.

### Stage Output Contract (§seed.md end)

16 items. Anything in the contract must be enforceable by `validate_seed_output`.

### Cluster → rule / file map (from audit report + code scouting)

| Cluster | Spec rule(s) | Primary location | Secondary |
|---|---|---|---|
| #1282 | R-3.6, R-3.10 | `graph/mutations.py::apply_seed_mutations` (belongs_to write) | `graph/seed_validation.py` validator |
| #1283 | R-3.9 | `graph/mutations.py::apply_seed_mutations` | `graph/seed_validation.py` |
| #1284 | R-3.14, R-3.15 | `graph/seed_validation.py::_check_beats` | `models/seed.py::InitialBeat` |
| #1285 | R-2.3, R-5.2 | `models/seed.py` (model-validator immutability) | `graph/mutations.py` pruning |
| #1286 | R-7.5 | `agents/serialize.py` convergence-analysis call | validator cross-check |
| #1287 | R-8.5 | `agents/serialize.py` dilemma-ordering call | validator cross-check |
| #1288 | R-7.1 | `models/seed.py::DilemmaAnalysis` (remove `flavor`) | serialize path too |
| #1289 | R-8.3 | `graph/mutations.py` ordering-edge writer | `models/seed.py::DilemmaRelationship` |
| #1290 | R-8.4 | `graph/mutations.py` ordering-edge writer | validator |
| #1291 | R-5.1 | `graph/seed_validation.py::_check_paths` | `pipeline/stages/seed.py` |
| #1292 | R-3.13 | `models/seed.py::InitialBeat` | `graph/seed_validation.py` |
| #1293 | spec-vs-code | spec update (if path_importance stays) OR model update (if removed) | decide during Task 25 |
| #1294 | R-3.4 | `models/seed.py::Consequence` | validator |
| #1295 | R-6.4 | `pipeline/stages/seed.py` approval gate | `graph/mutations.py` artifact write |

### Existing code — key shapes to match

- **Validator module pattern:** `src/questfoundry/graph/dream_validation.py` and `src/questfoundry/graph/brainstorm_validation.py` (created in PR #1351). Copy their idioms: `from __future__ import annotations`, `TYPE_CHECKING` guard for `Graph`, pure `list[str]` return, error messages start with `"R-X.Y: "` or `"Output-N: "`.
- **Wiring pattern:** `src/questfoundry/graph/mutations.py` has `apply_dream_mutations` and `apply_brainstorm_mutations` that each end with:
  ```python
  errors = validate_<stage>_output(graph)
  if errors:
      log.error("<stage>_contract_violated", errors=errors)
      raise <Stage>ContractError(
          "<STAGE> stage output contract violated:\n  - "
          + "\n  - ".join(errors)
      )
  ```
- **Graph API** (see `src/questfoundry/graph/graph.py`): `Graph()` constructor, `graph.create_node(id, data)`, `graph.update_node(id, **fields)`, `graph.get_node(id)`, `graph.get_nodes_by_type(type)`, `graph.add_edge(type, from, to)`, `graph.get_edges(edge_type=..., from_id=..., to_id=...)`, `graph.all_node_ids()`, `graph.delete_node(id)`.
- **`apply_seed_mutations` currently lives at `src/questfoundry/graph/mutations.py:1735-2062`** (328 lines). Deep enough that this plan touches specific regions by name, not by line number.
- **`SeedMutationError`** is defined at `src/questfoundry/graph/mutations.py:312`. Keep for mutation-time errors. `SeedContractError` (new) is defined in the new validator module for contract-time errors post-write.
- **`log` object** is already set up at `src/questfoundry/graph/mutations.py` top via `log = get_logger(__name__)`.
- **Pyright suppression to remove** (in `models/seed.py`):
  ```
  # pyright: reportInvalidTypeForm=false
  # TODO(#1281): cleanup during M-SEED-spec compliance work
  ```
  Removed in Task 27.

### Pattern for follow-up issue filing

When a cluster fix defers richer UX (e.g. R-6.4 interactive rejection loop-back), file a follow-up issue similar to DREAM #1350. Template:

```
gh issue create --title "[spec-audit] SEED: <short>" --label "spec-audit,area:seed" --body "Follow-up to #<cluster>. …"
```

Capture the issue number and reference it in the PR body + the commit message.

---

## File Structure

### New files

- `src/questfoundry/graph/seed_validation.py` — `SeedContractError` + `validate_seed_output` + 7 `_check_*` helpers.
- `tests/unit/test_seed_validation.py` — rule-by-rule validator tests (one test per rule + parametrized families + compliant-baseline fixture).

### Modified files

- `src/questfoundry/graph/mutations.py` — wire validator at `apply_seed_mutations` exit; structural enforcement for #1282 / #1283 / #1289 / #1290.
- `src/questfoundry/models/seed.py` — tighten validators for #1285, #1288, #1292, #1293, #1294. Remove pyright suppression in Task 27.
- `src/questfoundry/pipeline/stages/seed.py` — #1295 Path Freeze approval gate.
- `src/questfoundry/agents/serialize.py` — #1286 / #1287 raise-on-LLM-failure.
- `tests/unit/test_seed_models.py`, `tests/unit/test_seed_stage.py`, `tests/unit/test_mutations.py`, `tests/unit/test_serialize.py` — fixtures and assertions updated per the rewrite-or-delete policy.

### Not modified

- GROW / POLISH / FILL / DRESS / SHIP stage code or tests.
- DREAM / BRAINSTORM validators (they stayed compliant after PR #1351).
- Prompt templates, unless a cluster fix specifically requires a producer-level change.

---

## Task overview

- **Phase A — Baseline (Task 1).** Confirm baseline green; delete obvious obsolete SEED tests (if any).
- **Phase B — Validator tests (Task 2).** Failing tests written first.
- **Phase C — Validator implementation (Tasks 3–9).** One commit per `_check_*` helper.
- **Phase D — Wire validator (Task 10).**
- **Phase E — Critical / silent-degradation fixes (Tasks 11–15).** #1282 → #1283 → #1286 → #1287 → #1295.
- **Phase F — Moderate fixes (Tasks 16–26).** Remaining 9 clusters.
- **Phase G — Close-out (Tasks 27–28).** Lift pyright suppression; push and open PR.

28 tasks total. Each ends with a commit per CLAUDE.md §Project Git Rules.

---

## Phase A — Baseline

### Task 1: Baseline sanity + cleanup

**Files:**
- No new files. Possibly delete individual tests in `tests/unit/test_seed_*.py` or `tests/unit/test_mutations.py` if any show a pre-audit premise on the current baseline.

- [ ] **Step 1: Confirm non-downstream suite green on `feat/seed-compliance`**

Run:
```
uv run pytest tests/unit/ -k "not grow and not polish and not fill and not dress and not ship" --tb=no -q 2>&1 | tail -5
```
Expected: 2060 passed, 1 pre-existing pollution failure (`test_provider_factory::test_create_chat_model_ollama_success`). No SEED-related failures.

- [ ] **Step 2: Scan for pre-existing SEED-related failures**

Run:
```
uv run pytest tests/unit/test_seed_models.py tests/unit/test_seed_stage.py tests/unit/test_serialize.py -k "seed" --tb=short -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 3: If any SEED test fails above, decide delete vs rewrite-later**

If a failure reflects a pre-audit premise that the spec no longer supports, delete the test with a short rationale in the commit. If it fails for some other reason, leave it — Phase F's cluster-specific tasks will address it.

- [ ] **Step 4: Commit only if anything was deleted**

```bash
git add tests/unit/test_seed_*.py
git commit -m "$(cat <<'EOF'
test(seed): remove pre-audit test premises

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

If no test was deleted, skip the commit — the task exits without changes. Note in the subagent report: "no deletions needed; baseline clean."

---

## Phase B — Validator tests

### Task 2: Failing validator tests

**Files:**
- Create: `tests/unit/test_seed_validation.py`

The test file must fail at collection with `ModuleNotFoundError: No module named 'questfoundry.graph.seed_validation'` — the validator module is created in Task 3. Each test asserts a specific rule produces a specific error substring.

- [ ] **Step 1: Create `tests/unit/test_seed_validation.py`**

Write the full file with this content:

```python
"""Tests for SEED Stage Output Contract validator."""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.seed_validation import validate_seed_output


# --------------------------------------------------------------------------
# Compliant-baseline fixture
# --------------------------------------------------------------------------


def _seed_dream_baseline(graph: Graph) -> None:
    """Minimal DREAM-compliant vision node (upstream pre-condition)."""
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


def _seed_brainstorm_baseline(graph: Graph) -> None:
    """Minimal BRAINSTORM-compliant entities + 1 dilemma with 2 answers."""
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


def _seed_paths_and_beats(graph: Graph) -> None:
    """SEED Y-shape scaffold for the mentor_trust dilemma."""
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
        graph.add_edge(
            "explores", path_id, f"dilemma::mentor_trust::alt::{ans}"
        )

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

    # Pre-commit beat (dual belongs_to, same dilemma)
    graph.create_node(
        "beat::pre_mentor_01",
        {
            "type": "beat",
            "raw_id": "pre_mentor_01",
            "summary": "Mentor delivers warning",
            "entities": ["character::mentor", "character::kay"],
            "dilemma_impacts": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "effect": "advances",
                }
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
                "entities": ["character::mentor", "character::kay"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "dilemma::mentor_trust",
                        "effect": "commits",
                    }
                ],
            },
        )
        graph.add_edge("belongs_to", commit_id, path_id)

        for i in range(1, 4):  # 3 post-commit beats
            post_id = f"beat::post_{ans}_{i:02d}"
            graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": f"Post-commit beat {i} on {ans}",
                    "entities": ["character::mentor", "character::kay"],
                    "dilemma_impacts": [],
                },
            )
            graph.add_edge("belongs_to", post_id, path_id)


def _seed_freeze_approval(graph: Graph) -> None:
    """Mark SEED Path Freeze approved."""
    graph.create_node(
        "seed_freeze",
        {
            "type": "seed_freeze",
            "human_approved": True,
        },
    )


@pytest.fixture
def compliant_graph() -> Graph:
    graph = Graph()
    _seed_dream_baseline(graph)
    _seed_brainstorm_baseline(graph)
    _seed_paths_and_beats(graph)
    _seed_freeze_approval(graph)
    return graph


# --------------------------------------------------------------------------
# Positive baseline
# --------------------------------------------------------------------------


def test_valid_graph_passes(compliant_graph: Graph) -> None:
    assert validate_seed_output(compliant_graph) == []


# --------------------------------------------------------------------------
# Upstream-contract delegation
# --------------------------------------------------------------------------


def test_upstream_brainstorm_contract_violation_surfaces(compliant_graph: Graph) -> None:
    # Wipe a BRAINSTORM-required field (R-2.1 entity name) to force upstream failure.
    compliant_graph.update_node("character::kay", name=None)
    errors = validate_seed_output(compliant_graph)
    assert any("BRAINSTORM" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 1 — disposition
# --------------------------------------------------------------------------


def test_R_1_1_entity_missing_disposition(compliant_graph: Graph) -> None:
    compliant_graph.update_node("character::kay", disposition=None)
    errors = validate_seed_output(compliant_graph)
    assert any("disposition" in e for e in errors)


def test_R_1_2_cut_entity_still_anchored(compliant_graph: Graph) -> None:
    compliant_graph.update_node("character::mentor", disposition="cut")
    errors = validate_seed_output(compliant_graph)
    assert any("mentor" in e.lower() and "cut" in e.lower() for e in errors)


def test_R_1_4_two_location_minimum_survives(compliant_graph: Graph) -> None:
    compliant_graph.update_node("location::depths", disposition="cut")
    errors = validate_seed_output(compliant_graph)
    assert any("location" in e.lower() and "2" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 3 — Y-shape (hot path)
# --------------------------------------------------------------------------


def test_R_3_1_explored_answer_missing_path(compliant_graph: Graph) -> None:
    compliant_graph.delete_node("path::mentor_trust__manipulator")
    errors = validate_seed_output(compliant_graph)
    assert any("path" in e.lower() and "manipulator" in e.lower() for e in errors)


def test_R_3_6_precommit_missing_dual_belongs_to(compliant_graph: Graph) -> None:
    # Remove one of the two belongs_to edges on the pre-commit beat.
    for edge in list(compliant_graph.get_edges(edge_type="belongs_to")):
        if (
            edge["from"] == "beat::pre_mentor_01"
            and edge["to"] == "path::mentor_trust__manipulator"
        ):
            # Graph has no public remove_edge in the tests; simulate by
            # reconstructing the graph minus that edge.
            pass
    # Use a different approach — rebuild graph without the second belongs_to
    # so the pre-commit beat has only ONE belongs_to edge.
    new_graph = Graph()
    _seed_dream_baseline(new_graph)
    _seed_brainstorm_baseline(new_graph)
    # Rebuild paths/beats without the second belongs_to on pre_mentor_01
    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        new_graph.create_node(
            path_id,
            {
                "type": "path",
                "raw_id": f"mentor_trust__{ans}",
                "dilemma_id": "dilemma::mentor_trust",
                "is_canonical": ans == "protector",
            },
        )
        new_graph.add_edge(
            "explores", path_id, f"dilemma::mentor_trust::alt::{ans}"
        )
        conseq_id = f"consequence::mentor_trust__{ans}"
        new_graph.create_node(
            conseq_id,
            {
                "type": "consequence",
                "raw_id": f"mentor_trust__{ans}",
                "description": "d",
                "ripples": ["r"],
            },
        )
        new_graph.add_edge("has_consequence", path_id, conseq_id)
    new_graph.create_node(
        "beat::pre_mentor_01",
        {
            "type": "beat",
            "raw_id": "pre_mentor_01",
            "summary": "warning",
            "entities": ["character::mentor"],
            "dilemma_impacts": [
                {"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}
            ],
        },
    )
    # only ONE belongs_to — violation of R-3.6 (pre-commit must have two)
    new_graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__protector")
    for ans in ["protector", "manipulator"]:
        commit_id = f"beat::commit_{ans}"
        new_graph.create_node(
            commit_id,
            {
                "type": "beat",
                "raw_id": f"commit_{ans}",
                "summary": "commit",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}
                ],
            },
        )
        new_graph.add_edge("belongs_to", commit_id, f"path::mentor_trust__{ans}")
        for i in range(1, 4):
            post_id = f"beat::post_{ans}_{i:02d}"
            new_graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": "post",
                    "entities": ["character::mentor"],
                    "dilemma_impacts": [],
                },
            )
            new_graph.add_edge("belongs_to", post_id, f"path::mentor_trust__{ans}")
    _seed_freeze_approval(new_graph)
    errors = validate_seed_output(new_graph)
    assert any(
        "pre" in e.lower() and "belongs_to" in e for e in errors
    ), f"expected a pre-commit dual-belongs_to error, got {errors}"


def test_R_3_9_cross_dilemma_belongs_to_forbidden(compliant_graph: Graph) -> None:
    # Create a second dilemma + path; then add a cross-dilemma belongs_to.
    compliant_graph.create_node(
        "dilemma::other",
        {
            "type": "dilemma",
            "raw_id": "other",
            "question": "Why?",
            "why_it_matters": "x",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "low",
        },
    )
    compliant_graph.add_edge("anchored_to", "dilemma::other", "character::kay")
    for ans, is_canon in [("a", True), ("b", False)]:
        ans_id = f"dilemma::other::alt::{ans}"
        compliant_graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": ans,
                "description": f"d-{ans}",
                "is_canonical": is_canon,
                "explored": True,
            },
        )
        compliant_graph.add_edge("has_answer", "dilemma::other", ans_id)
        compliant_graph.create_node(
            f"path::other__{ans}",
            {
                "type": "path",
                "raw_id": f"other__{ans}",
                "dilemma_id": "dilemma::other",
                "is_canonical": is_canon,
            },
        )
        compliant_graph.add_edge(
            "explores", f"path::other__{ans}", ans_id
        )
    # Cross-dilemma belongs_to: the pre_mentor_01 beat now also belongs to a path of OTHER dilemma.
    compliant_graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::other__a")
    errors = validate_seed_output(compliant_graph)
    assert any("cross" in e.lower() or "R-3.9" in e for e in errors)


def test_R_3_10_explored_dilemma_missing_precommit(compliant_graph: Graph) -> None:
    # Remove the pre-commit beat entirely.
    compliant_graph.delete_node("beat::pre_mentor_01")
    errors = validate_seed_output(compliant_graph)
    assert any("pre-commit" in e.lower() or "R-3.10" in e for e in errors)


def test_R_3_11_path_needs_exactly_one_commit_beat(compliant_graph: Graph) -> None:
    # Duplicate a commit beat — one path now has two commit beats.
    compliant_graph.create_node(
        "beat::commit_protector_dup",
        {
            "type": "beat",
            "raw_id": "commit_protector_dup",
            "summary": "duplicate commit",
            "entities": ["character::mentor"],
            "dilemma_impacts": [
                {"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}
            ],
        },
    )
    compliant_graph.add_edge(
        "belongs_to", "beat::commit_protector_dup", "path::mentor_trust__protector"
    )
    errors = validate_seed_output(compliant_graph)
    assert any("commit" in e.lower() for e in errors)


def test_R_3_12_post_commit_count_below_min(compliant_graph: Graph) -> None:
    # Remove two post-commit beats on protector path.
    compliant_graph.delete_node("beat::post_protector_02")
    compliant_graph.delete_node("beat::post_protector_03")
    errors = validate_seed_output(compliant_graph)
    assert any("post-commit" in e.lower() or "2" in e for e in errors)


def test_R_3_13_beat_missing_summary(compliant_graph: Graph) -> None:
    compliant_graph.update_node("beat::pre_mentor_01", summary="")
    errors = validate_seed_output(compliant_graph)
    assert any("summary" in e for e in errors)


def test_R_3_13_beat_missing_entities(compliant_graph: Graph) -> None:
    compliant_graph.update_node("beat::pre_mentor_01", entities=[])
    errors = validate_seed_output(compliant_graph)
    assert any("entities" in e for e in errors)


def test_R_3_14_setup_beat_must_not_belong_to_path(compliant_graph: Graph) -> None:
    compliant_graph.create_node(
        "beat::setup_intro",
        {
            "type": "beat",
            "raw_id": "setup_intro",
            "role": "setup",
            "summary": "opener",
            "entities": ["location::archive"],
            "dilemma_impacts": [],
        },
    )
    compliant_graph.add_edge(
        "belongs_to", "beat::setup_intro", "path::mentor_trust__protector"
    )
    errors = validate_seed_output(compliant_graph)
    assert any("setup" in e.lower() and "belongs_to" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 3 — Consequence + Path
# --------------------------------------------------------------------------


def test_R_3_3_path_without_consequence(compliant_graph: Graph) -> None:
    compliant_graph.delete_node("consequence::mentor_trust__protector")
    errors = validate_seed_output(compliant_graph)
    assert any("consequence" in e.lower() for e in errors)


def test_R_3_4_consequence_without_ripples(compliant_graph: Graph) -> None:
    compliant_graph.update_node("consequence::mentor_trust__protector", ripples=[])
    errors = validate_seed_output(compliant_graph)
    assert any("ripple" in e.lower() for e in errors)


# --------------------------------------------------------------------------
# Phase 5 — arc-count
# --------------------------------------------------------------------------


def test_R_5_1_arc_count_over_16(compliant_graph: Graph) -> None:
    # Add 16 more dilemmas, each with 2 explored answers → arc count explodes.
    for i in range(5):
        did = f"dilemma::d_{i}"
        compliant_graph.create_node(
            did,
            {
                "type": "dilemma",
                "raw_id": f"d_{i}",
                "question": "Q?",
                "why_it_matters": "x",
                "dilemma_role": "soft",
                "residue_weight": "light",
                "ending_salience": "low",
            },
        )
        compliant_graph.add_edge("anchored_to", did, "character::kay")
        for ans, is_canon in [("a", True), ("b", False)]:
            ans_id = f"{did}::alt::{ans}"
            compliant_graph.create_node(
                ans_id,
                {
                    "type": "answer",
                    "raw_id": ans,
                    "description": "d",
                    "is_canonical": is_canon,
                    "explored": True,
                },
            )
            compliant_graph.add_edge("has_answer", did, ans_id)
            compliant_graph.create_node(
                f"path::d_{i}__{ans}",
                {
                    "type": "path",
                    "raw_id": f"d_{i}__{ans}",
                    "dilemma_id": did,
                    "is_canonical": is_canon,
                },
            )
            compliant_graph.add_edge(
                "explores", f"path::d_{i}__{ans}", ans_id
            )
    errors = validate_seed_output(compliant_graph)
    assert any("arc" in e.lower() or "R-5.1" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 6 — approval
# --------------------------------------------------------------------------


def test_R_6_4_missing_path_freeze_approval() -> None:
    graph = Graph()
    _seed_dream_baseline(graph)
    _seed_brainstorm_baseline(graph)
    _seed_paths_and_beats(graph)
    # Do NOT call _seed_freeze_approval.
    errors = validate_seed_output(graph)
    assert any("approv" in e.lower() or "R-6.4" in e for e in errors)


def test_R_6_4_path_freeze_explicitly_unapproved() -> None:
    graph = Graph()
    _seed_dream_baseline(graph)
    _seed_brainstorm_baseline(graph)
    _seed_paths_and_beats(graph)
    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": False})
    errors = validate_seed_output(graph)
    assert any("approv" in e.lower() for e in errors)


# --------------------------------------------------------------------------
# Phase 7 — dilemma analysis
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", ["dilemma_role", "residue_weight", "ending_salience"])
def test_R_7_x_dilemma_analysis_field_missing(
    compliant_graph: Graph, missing_field: str
) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", **{missing_field: None})
    errors = validate_seed_output(compliant_graph)
    assert any(missing_field in e for e in errors)


def test_R_7_1_dilemma_role_flavor_forbidden(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", dilemma_role="flavor")
    errors = validate_seed_output(compliant_graph)
    assert any("flavor" in e for e in errors) or any(
        "dilemma_role" in e for e in errors
    )


# --------------------------------------------------------------------------
# Phase 8 — ordering relationships
# --------------------------------------------------------------------------


def test_R_8_3_concurrent_non_lex_order_forbidden(compliant_graph: Graph) -> None:
    # Create a concurrent edge with non-lex order: dilemma_a > dilemma_b alphabetically.
    compliant_graph.create_node(
        "dilemma::z_later",
        {
            "type": "dilemma",
            "raw_id": "z_later",
            "question": "Q?",
            "why_it_matters": "x",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "low",
        },
    )
    compliant_graph.add_edge("anchored_to", "dilemma::z_later", "character::kay")
    # Add a pair: dilemma_a should be mentor_trust (lex-smaller), NOT z_later.
    compliant_graph.create_node(
        "ordering::bad",
        {
            "type": "ordering",
            "relationship": "concurrent",
            "dilemma_a": "dilemma::z_later",  # WRONG — should be mentor_trust
            "dilemma_b": "dilemma::mentor_trust",
        },
    )
    errors = validate_seed_output(compliant_graph)
    assert any("lex" in e.lower() or "concurrent" in e.lower() for e in errors)


def test_R_8_4_shared_entity_edge_forbidden(compliant_graph: Graph) -> None:
    compliant_graph.add_edge(
        "shared_entity", "dilemma::mentor_trust", "character::kay"
    )
    errors = validate_seed_output(compliant_graph)
    assert any("shared_entity" in e for e in errors)


# --------------------------------------------------------------------------
# Forbidden node types (Output-16)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["passage", "state_flag", "intersection_group"])
def test_output16_forbidden_node_type_present(
    compliant_graph: Graph, forbidden: str
) -> None:
    compliant_graph.create_node(
        f"{forbidden}::x",
        {"type": forbidden, "raw_id": "x"},
    )
    errors = validate_seed_output(compliant_graph)
    assert any(forbidden in e for e in errors)
```

- [ ] **Step 2: Run — expect collection failure (ModuleNotFoundError)**

Run:
```
uv run pytest tests/unit/test_seed_validation.py --collect-only -q 2>&1 | tail -5
```
Expected: `ModuleNotFoundError: No module named 'questfoundry.graph.seed_validation'`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_seed_validation.py
git commit -m "$(cat <<'EOF'
test(seed): add failing validator tests for SEED output contract

Covers 16 items in docs/design/procedures/seed.md §Stage Output
Contract plus key rules from phases 1, 3, 5, 6, 7, 8. Module
validate_seed_output implemented in following tasks.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase C — Validator implementation

Each check helper gets its own commit.

### Task 3: Validator skeleton + upstream-contract check

**Files:**
- Create: `src/questfoundry/graph/seed_validation.py`

- [ ] **Step 1: Create skeleton**

```python
"""SEED Stage Output Contract validator.

Validates the graph satisfies every rule in
docs/design/procedures/seed.md §Stage Output Contract.

Called at SEED exit (from apply_seed_mutations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class SeedContractError(ValueError):
    """Raised when SEED's Stage Output Contract is violated."""


_VALID_DILEMMA_ROLES = frozenset({"hard", "soft"})
_VALID_RESIDUE_WEIGHTS = frozenset({"heavy", "light", "cosmetic"})
_VALID_ENDING_SALIENCES = frozenset({"high", "low", "none"})
_VALID_ORDERING_RELATIONSHIPS = frozenset({"wraps", "concurrent", "serial"})
_VALID_DISPOSITIONS = frozenset({"retained", "cut"})
_FORBIDDEN_NODE_TYPES = frozenset(
    {"passage", "state_flag", "intersection_group", "transition_beat", "choice"}
)
_MAX_ARC_COUNT = 16


def validate_seed_output(graph: Graph) -> list[str]:
    """Verify the graph satisfies SEED's Stage Output Contract.

    Args:
        graph: Graph expected to contain SEED output.

    Returns:
        List of human-readable error strings. Empty means compliant.
        Pure read-only — never mutates the graph.
    """
    errors: list[str] = []
    _check_upstream_contract(graph, errors)
    return errors


def _check_upstream_contract(graph: Graph, errors: list[str]) -> None:
    """Delegate to BRAINSTORM validator (with downstream-node types allowed).

    At SEED time, the graph legitimately contains beat/path/consequence nodes
    that BRAINSTORM's R-3.8 would forbid — skip_forbidden_types=True relaxes
    that one check while preserving all other upstream invariants.
    """
    # Inline import avoids module-load circular dependencies.
    from questfoundry.graph.brainstorm_validation import validate_brainstorm_output

    # NOTE: `skip_forbidden_types=True` bypasses BRAINSTORM R-3.8 (forbidden
    # node types), which prohibits beat/path/consequence nodes — SEED
    # legitimately creates those, so BRAINSTORM's forbidden-types check
    # must not apply at SEED exit.
    upstream = validate_brainstorm_output(graph, skip_forbidden_types=True)
    for e in upstream:
        errors.append(f"Output-0: BRAINSTORM contract violated post-SEED — {e}")
```

- [ ] **Step 2: Run — baseline + upstream-contract tests pass**

Run:
```
uv run pytest tests/unit/test_seed_validation.py::test_valid_graph_passes tests/unit/test_seed_validation.py::test_upstream_brainstorm_contract_violation_surfaces -v
```
Expected: both PASS.

- [ ] **Step 3: mypy + ruff**

```
uv run mypy src/questfoundry/graph/seed_validation.py
uv run ruff check src/questfoundry/graph/seed_validation.py
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/graph/seed_validation.py
git commit -m "$(cat <<'EOF'
feat(seed): skeleton validator with upstream-contract delegation

Initial shape of validate_seed_output mirrors dream_validation and
brainstorm_validation (PR #1351). Delegates to validate_brainstorm_output
to guard against SEED silently corrupting upstream state.

Subsequent commits add _check_paths, _check_beats,
_check_belongs_to_yshape, _check_convergence_and_ordering,
_check_state_flags_and_consequences, _check_approval_and_forbidden_nodes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `_check_entities` (Phase 1 dispositions)

**Files:**
- Modify: `src/questfoundry/graph/seed_validation.py`

- [ ] **Step 1: Add helper + wire it**

Append this helper below `_check_upstream_contract`:

```python
def _check_entities(graph: Graph, errors: list[str]) -> None:
    """Phase 1 entity-triage checks (R-1.1, R-1.2, R-1.4)."""
    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    retained_location_count = 0
    for entity_id, entity in sorted(entity_nodes.items()):
        disposition = entity.get("disposition")
        if disposition is None:
            errors.append(
                f"R-1.1: entity {entity_id!r} has no disposition (must be 'retained' or 'cut')"
            )
            continue
        if disposition not in _VALID_DISPOSITIONS:
            errors.append(
                f"R-1.1: entity {entity_id!r} has invalid disposition {disposition!r}; "
                f"must be one of {sorted(_VALID_DISPOSITIONS)}"
            )
        if disposition == "retained" and entity.get("category") == "location":
            retained_location_count += 1

    # R-1.2: anchored_to from surviving dilemmas must not point to cut entities.
    for edge in sorted(
        graph.get_edges(edge_type="anchored_to"),
        key=lambda e: (e["from"], e["to"]),
    ):
        dilemma = dilemma_nodes.get(edge["from"])
        if dilemma is None:
            continue
        entity = entity_nodes.get(edge["to"])
        if entity is None:
            continue
        if entity.get("disposition") == "cut":
            errors.append(
                f"R-1.2: dilemma {edge['from']!r} is anchored to cut entity "
                f"{edge['to']!r}; re-anchor or cut the dilemma first"
            )

    if retained_location_count < 2:
        errors.append(
            f"R-1.4: SEED must retain ≥2 location entities, found {retained_location_count}"
        )
```

Update `validate_seed_output` to call it:
```python
    errors: list[str] = []
    _check_upstream_contract(graph, errors)
    _check_entities(graph, errors)
    return errors
```

- [ ] **Step 2: Run Phase-1 tests**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -k "R_1_" -v
```
Expected: all PASS.

- [ ] **Step 3: mypy + ruff clean, commit**

```bash
git add src/questfoundry/graph/seed_validation.py
git commit -m "$(cat <<'EOF'
feat(seed): _check_entities (Phase 1 dispositions, R-1.1/R-1.2/R-1.4)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `_check_paths_and_consequences` (Phase 3 structural — paths, consequences, ripples)

**Files:**
- Modify: `src/questfoundry/graph/seed_validation.py`

- [ ] **Step 1: Add helper**

```python
def _check_paths_and_consequences(graph: Graph, errors: list[str]) -> None:
    """Phase 3 path structure (R-3.1, R-3.2, R-3.3, R-3.4, R-3.5)."""
    path_nodes = graph.get_nodes_by_type("path")
    answer_nodes = graph.get_nodes_by_type("answer")
    consequence_nodes = graph.get_nodes_by_type("consequence")

    # R-3.1 + R-3.2: each explored answer has exactly one Path via `explores`,
    # with a well-formed path id.
    explores_edges = graph.get_edges(edge_type="explores")
    path_by_answer: dict[str, list[str]] = {}
    for edge in sorted(explores_edges, key=lambda e: (e["from"], e["to"])):
        path_by_answer.setdefault(edge["to"], []).append(edge["from"])

    for answer_id, answer in sorted(answer_nodes.items()):
        if not answer.get("explored"):
            continue
        paths = path_by_answer.get(answer_id, [])
        if len(paths) == 0:
            errors.append(
                f"R-3.1: explored answer {answer_id!r} has no path (expected exactly one)"
            )
        elif len(paths) > 1:
            errors.append(
                f"R-3.1: explored answer {answer_id!r} has {len(paths)} paths; "
                f"expected exactly one: {sorted(paths)}"
            )

    for path_id in sorted(path_nodes.keys()):
        if not path_id.startswith("path::"):
            errors.append(f"R-3.2: path id {path_id!r} missing 'path::' prefix")

    # R-3.3: every Path has ≥1 has_consequence edge.
    has_consequence_edges = graph.get_edges(edge_type="has_consequence")
    consequences_per_path: dict[str, list[str]] = {}
    for edge in has_consequence_edges:
        consequences_per_path.setdefault(edge["from"], []).append(edge["to"])

    for path_id in sorted(path_nodes.keys()):
        if not consequences_per_path.get(path_id):
            errors.append(f"R-3.3: path {path_id!r} has no has_consequence edge")

    # R-3.4: every Consequence has non-empty description + ≥1 ripple.
    for conseq_id, conseq in sorted(consequence_nodes.items()):
        if not conseq.get("description"):
            errors.append(f"R-3.4: consequence {conseq_id!r} has empty description")
        ripples = conseq.get("ripples", [])
        if not ripples:
            errors.append(f"R-3.4: consequence {conseq_id!r} has no ripples")
```

Wire in `validate_seed_output`:
```python
    _check_paths_and_consequences(graph, errors)
```

- [ ] **Step 2: Run tests**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -k "R_3_1 or R_3_3 or R_3_4" -v
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/questfoundry/graph/seed_validation.py
git commit -m "$(cat <<'EOF'
feat(seed): _check_paths_and_consequences (Phase 3 paths + consequences)

R-3.1 one path per explored answer, R-3.2 path id prefix, R-3.3 path
has consequence, R-3.4 consequence has description + ripples.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `_check_beats` (Phase 3 beat structure)

**Files:**
- Modify: `src/questfoundry/graph/seed_validation.py`

- [ ] **Step 1: Add helper**

```python
def _check_beats(graph: Graph, errors: list[str]) -> None:
    """Phase 3 beat structural rules (R-3.13, R-3.14, R-3.15)."""
    beat_nodes = graph.get_nodes_by_type("beat")
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beats_with_path: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beats_with_path.setdefault(edge["from"], []).append(edge["to"])

    for beat_id, beat in sorted(beat_nodes.items()):
        role = beat.get("role")
        is_structural = role in {"setup", "epilogue"}

        # R-3.13 + R-3.15: every beat has non-empty summary + entities.
        if not beat.get("summary"):
            errors.append(f"R-3.13: beat {beat_id!r} has empty summary")
        if not beat.get("entities"):
            errors.append(f"R-3.13: beat {beat_id!r} has empty entities list")

        # R-3.14: structural beats must have zero belongs_to + zero commits impact.
        if is_structural:
            paths = beats_with_path.get(beat_id, [])
            if paths:
                errors.append(
                    f"R-3.14: {role} beat {beat_id!r} must have zero belongs_to "
                    f"edges, found {len(paths)}"
                )
            if any(
                impact.get("effect") == "commits"
                for impact in beat.get("dilemma_impacts", [])
            ):
                errors.append(
                    f"R-3.14: {role} beat {beat_id!r} must not contain "
                    f"dilemma_impacts.effect: commits"
                )
```

Wire call in `validate_seed_output`.

- [ ] **Step 2: Run tests**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -k "R_3_13 or R_3_14" -v
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/questfoundry/graph/seed_validation.py
git commit -m "$(cat <<'EOF'
feat(seed): _check_beats (Phase 3 beat structural rules)

R-3.13 non-empty summary + entities, R-3.14/R-3.15 setup/epilogue
structural beats have zero belongs_to + zero commits impact.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `_check_belongs_to_yshape` (R-3.6, R-3.7, R-3.8, R-3.9, R-3.10, R-3.11, R-3.12)

**This is the hot-path Y-shape enforcement.** The most complex helper.

**Files:**
- Modify: `src/questfoundry/graph/seed_validation.py`

- [ ] **Step 1: Add helper**

```python
def _check_belongs_to_yshape(graph: Graph, errors: list[str]) -> None:
    """Y-shape guard rails and commit/post-commit counts (R-3.6–R-3.12)."""
    beat_nodes = graph.get_nodes_by_type("beat")
    path_nodes = graph.get_nodes_by_type("path")
    answer_nodes = graph.get_nodes_by_type("answer")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_paths: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beat_to_paths.setdefault(edge["from"], []).append(edge["to"])

    path_dilemma: dict[str, str] = {
        path_id: path.get("dilemma_id", "")
        for path_id, path in path_nodes.items()
    }

    # Track per-path commit and post-commit beat counts (R-3.11, R-3.12)
    commit_beats_per_path: dict[str, list[str]] = {}
    post_beats_per_path: dict[str, list[str]] = {}
    pre_commit_by_dilemma: dict[str, list[str]] = {}

    for beat_id in sorted(beat_nodes.keys()):
        beat = beat_nodes[beat_id]
        role = beat.get("role")
        if role in {"setup", "epilogue"}:
            continue  # structural — handled by _check_beats

        paths = beat_to_paths.get(beat_id, [])
        impacts = beat.get("dilemma_impacts", [])
        has_commits_impact = any(imp.get("effect") == "commits" for imp in impacts)

        # R-3.9 cross-dilemma belongs_to prohibition
        dilemmas_of_this_beat = {
            path_dilemma.get(p, "") for p in paths if p in path_nodes
        }
        dilemmas_of_this_beat.discard("")
        if len(dilemmas_of_this_beat) > 1:
            errors.append(
                f"R-3.9: beat {beat_id!r} has cross-dilemma belongs_to — "
                f"references paths of dilemmas {sorted(dilemmas_of_this_beat)}"
            )

        if has_commits_impact:
            # Commit beat: R-3.7
            if len(paths) != 1:
                errors.append(
                    f"R-3.7: commit beat {beat_id!r} must have exactly one "
                    f"belongs_to edge, found {len(paths)}"
                )
            for p in paths:
                commit_beats_per_path.setdefault(p, []).append(beat_id)
        elif len(paths) >= 2:
            # Pre-commit beat: R-3.6
            if len(dilemmas_of_this_beat) != 1:
                errors.append(
                    f"R-3.6: pre-commit beat {beat_id!r} belongs_to edges must "
                    f"reference paths of the same dilemma, got "
                    f"{sorted(dilemmas_of_this_beat)}"
                )
            # Record per-dilemma pre-commit presence
            for d in dilemmas_of_this_beat:
                pre_commit_by_dilemma.setdefault(d, []).append(beat_id)
        elif len(paths) == 1:
            # Post-commit beat: R-3.8 — no commits impact + single belongs_to
            post_beats_per_path.setdefault(paths[0], []).append(beat_id)

    # R-3.11: every explored path has exactly one commit beat
    for path_id in sorted(path_nodes.keys()):
        commits = commit_beats_per_path.get(path_id, [])
        if len(commits) != 1:
            errors.append(
                f"R-3.11: path {path_id!r} must have exactly one commit beat, "
                f"found {len(commits)}: {sorted(commits)}"
            )

    # R-3.12: every explored path has 2–4 post-commit beats
    for path_id in sorted(path_nodes.keys()):
        post = post_beats_per_path.get(path_id, [])
        if len(post) < 2 or len(post) > 4:
            errors.append(
                f"R-3.12: path {path_id!r} must have 2–4 post-commit beats, "
                f"found {len(post)}"
            )

    # R-3.10: every dilemma with 2 explored answers has ≥1 pre-commit beat
    for dilemma_id, dilemma in sorted(dilemma_nodes.items()):
        explored_answers = [
            a_id
            for a_id, a in answer_nodes.items()
            if a.get("explored")
            and any(
                e["from"] == dilemma_id and e["to"] == a_id
                for e in graph.get_edges(edge_type="has_answer")
            )
        ]
        if len(explored_answers) >= 2:
            if not pre_commit_by_dilemma.get(dilemma_id):
                errors.append(
                    f"R-3.10: dilemma {dilemma_id!r} has {len(explored_answers)} "
                    f"explored answers but no pre-commit beats — Y-shape fork missing"
                )
```

Wire the call in `validate_seed_output`.

- [ ] **Step 2: Run tests**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -k "R_3_6 or R_3_9 or R_3_10 or R_3_11 or R_3_12" -v
```
Expected: all PASS.

- [ ] **Step 3: Full validator test sweep**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -v 2>&1 | tail -10
```
Expected: every test implemented so far passes; later-rule tests (R-6, R-7, R-8) still fail — that's expected.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/graph/seed_validation.py
git commit -m "$(cat <<'EOF'
feat(seed): _check_belongs_to_yshape (hot-path Y-shape invariants)

Covers R-3.6 pre-commit dual belongs_to, R-3.7 commit single + commits
impact, R-3.8 post-commit single + no commits, R-3.9 no cross-dilemma
dual belongs_to, R-3.10 ≥1 pre-commit per explored dilemma,
R-3.11 exactly-one commit beat per path, R-3.12 2–4 post-commit.

Owns the validator-side of clusters #1282 and #1283.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `_check_convergence_and_ordering` (Phase 7 + 8)

**Files:**
- Modify: `src/questfoundry/graph/seed_validation.py`

- [ ] **Step 1: Add helper**

```python
def _check_convergence_and_ordering(graph: Graph, errors: list[str]) -> None:
    """Phase 7 dilemma analysis + Phase 8 ordering (R-7.1–R-7.3, R-8.3, R-8.4)."""
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    # R-7.1 / R-7.2 / R-7.3
    for dilemma_id, dilemma in sorted(dilemma_nodes.items()):
        role = dilemma.get("dilemma_role")
        weight = dilemma.get("residue_weight")
        salience = dilemma.get("ending_salience")
        if role is None:
            errors.append(f"R-7.1: dilemma {dilemma_id!r} missing dilemma_role")
        elif role not in _VALID_DILEMMA_ROLES:
            errors.append(
                f"R-7.1: dilemma {dilemma_id!r} has invalid dilemma_role {role!r}; "
                f"must be one of {sorted(_VALID_DILEMMA_ROLES)}"
            )
        if weight is None:
            errors.append(
                f"R-7.2: dilemma {dilemma_id!r} missing residue_weight"
            )
        elif weight not in _VALID_RESIDUE_WEIGHTS:
            errors.append(
                f"R-7.2: dilemma {dilemma_id!r} has invalid residue_weight {weight!r}; "
                f"must be one of {sorted(_VALID_RESIDUE_WEIGHTS)}"
            )
        if salience is None:
            errors.append(
                f"R-7.3: dilemma {dilemma_id!r} missing ending_salience"
            )
        elif salience not in _VALID_ENDING_SALIENCES:
            errors.append(
                f"R-7.3: dilemma {dilemma_id!r} has invalid ending_salience "
                f"{salience!r}; must be one of {sorted(_VALID_ENDING_SALIENCES)}"
            )

    # R-8.3 concurrent lex-smaller-first; R-8.1 valid relationship set.
    ordering_nodes = graph.get_nodes_by_type("ordering")
    for ord_id, ord_node in sorted(ordering_nodes.items()):
        rel = ord_node.get("relationship")
        if rel not in _VALID_ORDERING_RELATIONSHIPS:
            errors.append(
                f"R-8.1: ordering {ord_id!r} has invalid relationship {rel!r}"
            )
        a, b = ord_node.get("dilemma_a"), ord_node.get("dilemma_b")
        if rel == "concurrent" and a and b and a > b:
            errors.append(
                f"R-8.3: concurrent ordering {ord_id!r} must have lex-smaller "
                f"dilemma as dilemma_a (got {a!r} > {b!r})"
            )

    # R-8.4 shared_entity edges forbidden.
    shared_entity_edges = graph.get_edges(edge_type="shared_entity")
    if shared_entity_edges:
        errors.append(
            f"R-8.4: shared_entity edges are forbidden (derived from anchored_to, "
            f"not declared); found {len(shared_entity_edges)}"
        )
```

Wire call in `validate_seed_output`.

- [ ] **Step 2: Run Phase-7 / Phase-8 tests**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -k "R_7_ or R_8_" -v
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/questfoundry/graph/seed_validation.py
git commit -m "$(cat <<'EOF'
feat(seed): _check_convergence_and_ordering (Phases 7 + 8)

R-7.1/R-7.2/R-7.3 dilemma analysis field sets, R-8.1 valid
relationships, R-8.3 concurrent lex-smaller-first, R-8.4 shared_entity
edges forbidden.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `_check_arc_count_and_approval` + forbidden node types (R-5.1, R-6.4, Output-16)

**Files:**
- Modify: `src/questfoundry/graph/seed_validation.py`

- [ ] **Step 1: Add helper**

```python
def _check_arc_count_and_approval(graph: Graph, errors: list[str]) -> None:
    """Phase 5 arc-count guardrail (R-5.1), Phase 6 approval (R-6.4),
    and Stage Output Contract item 16 (forbidden node types)."""
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    answer_nodes = graph.get_nodes_by_type("answer")

    # R-5.1: arc count = 2 ^ (# dilemmas with both answers explored)
    fully_explored_dilemmas = 0
    has_answer_edges = graph.get_edges(edge_type="has_answer")
    answers_by_dilemma: dict[str, list[str]] = {}
    for edge in has_answer_edges:
        answers_by_dilemma.setdefault(edge["from"], []).append(edge["to"])
    for dilemma_id in dilemma_nodes:
        ans_ids = answers_by_dilemma.get(dilemma_id, [])
        explored = [
            a_id for a_id in ans_ids if answer_nodes.get(a_id, {}).get("explored")
        ]
        if len(explored) >= 2:
            fully_explored_dilemmas += 1
    arc_count = 2 ** fully_explored_dilemmas if fully_explored_dilemmas else 1
    if arc_count > _MAX_ARC_COUNT:
        errors.append(
            f"R-5.1: arc count {arc_count} exceeds maximum {_MAX_ARC_COUNT} "
            f"({fully_explored_dilemmas} fully explored dilemmas)"
        )

    # R-6.4: path freeze human approval recorded.
    freeze = graph.get_node("seed_freeze")
    if freeze is None:
        errors.append(
            "R-6.4: SEED Path Freeze approval is not recorded "
            "(expected seed_freeze node with human_approved: True)"
        )
    elif not freeze.get("human_approved"):
        errors.append(
            "R-6.4: SEED Path Freeze is not approved "
            "(seed_freeze.human_approved is not True)"
        )

    # Output-16: no forbidden node types.
    for node_type in sorted(_FORBIDDEN_NODE_TYPES):
        forbidden = graph.get_nodes_by_type(node_type)
        if forbidden:
            errors.append(
                f"Output-16: SEED must not create {node_type!r} nodes; "
                f"found {len(forbidden)}: {sorted(forbidden.keys())[:3]}"
            )
```

Wire call in `validate_seed_output`.

- [ ] **Step 2: Run R-5.1, R-6.4, Output-16 tests**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -k "R_5_1 or R_6_4 or output16" -v
```
Expected: all PASS.

- [ ] **Step 3: Full validator sweep — all tests pass**

Run:
```
uv run pytest tests/unit/test_seed_validation.py -v 2>&1 | tail -10
```
Expected: every test in `test_seed_validation.py` PASSES.

- [ ] **Step 4: mypy + ruff clean**

```
uv run mypy src/questfoundry/graph/seed_validation.py
uv run ruff check src/questfoundry/graph/seed_validation.py
uv run pyright src/questfoundry/graph/seed_validation.py
```
Expected: all clean.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/seed_validation.py
git commit -m "$(cat <<'EOF'
feat(seed): _check_arc_count_and_approval + forbidden node types

R-5.1 arc count ≤16, R-6.4 Path Freeze approval recorded,
Output-16 no passage/state_flag/intersection_group/transition_beat/
choice nodes.

Completes validate_seed_output; all validator tests pass.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase D — Wire validator

### Task 10: Wire `validate_seed_output` at SEED exit

**Files:**
- Modify: `src/questfoundry/graph/mutations.py`

- [ ] **Step 1: Add imports**

Near the top of `src/questfoundry/graph/mutations.py` (alongside the existing `dream_validation` / `brainstorm_validation` imports that PR #1351 added):

```python
from questfoundry.graph.seed_validation import (
    SeedContractError,
    validate_seed_output,
)
```

- [ ] **Step 2: Append validator call at end of `apply_seed_mutations`**

Find the function `apply_seed_mutations` in `src/questfoundry/graph/mutations.py` (starts around line 1735). At the very end of the function body — after all node and edge creation, before any `return` — add:

```python
    errors = validate_seed_output(graph)
    if errors:
        log.error("seed_contract_violated", errors=errors)
        raise SeedContractError(
            "SEED stage output contract violated:\n  - "
            + "\n  - ".join(errors)
        )
```

- [ ] **Step 3: Run tests — expected mix of pass/fail**

Run:
```
uv run pytest tests/unit/test_seed_models.py tests/unit/test_seed_stage.py tests/unit/test_mutations.py -k "seed" --tb=short -q 2>&1 | tail -20
```
Expected: many tests fail because fixtures don't satisfy the new contract (Y-shape, dilemma_role, seed_freeze, etc.). This is the TDD signal Phase E/F tasks will resolve. Do NOT fix these tests here.

Run the validator suite:
```
uv run pytest tests/unit/test_seed_validation.py --tb=no -q
```
Expected: all pass.

- [ ] **Step 4: mypy + ruff + pyright**

```
uv run mypy src/questfoundry/graph/mutations.py
uv run ruff check src/questfoundry/graph/mutations.py
uv run pyright src/
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/graph/mutations.py
git commit -m "$(cat <<'EOF'
feat(seed): wire validate_seed_output at SEED exit

apply_seed_mutations now runs the contract validator after all writes
and raises SeedContractError on violations (with structured log event).
Matches the DREAM/BRAINSTORM pattern from PR #1351.

Existing SEED-consuming tests will red until Phase E/F cluster fixes
land. This is the expected TDD signal.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase E — Critical / silent-degradation fixes (hot-path priority)

### Task 11: #1282 — Y-shape shared beat enforcement (producer side)

**Files:**
- Modify: `src/questfoundry/graph/mutations.py` (producer-time enforcement in `apply_seed_mutations` beat-write block)
- Modify: `tests/unit/test_mutations.py` (add targeted test)

- [ ] **Step 1: Add failing test**

Append to `tests/unit/test_mutations.py` (near the SEED mutation tests):

```python
def test_apply_seed_mutations_rejects_precommit_with_mismatched_dilemmas() -> None:
    """R-3.6 / R-3.9: pre-commit dual belongs_to must reference paths of the
    same dilemma. Mismatched dilemmas must raise at write time, not slip
    through to the exit validator."""
    import pytest

    from questfoundry.graph.graph import Graph
    from questfoundry.graph.mutations import apply_seed_mutations, MutationError

    graph = Graph()
    # Minimal BRAINSTORM-compliant seed (borrowed from other SEED mutation tests).
    # The specific fixture shape here is less important than the mutation input:
    # paths from two different dilemmas referenced via path_id + also_belongs_to.
    # If the fixture-building is nontrivial, reuse a helper from the existing
    # test file's shared fixtures.
    # ...
    # Expected: apply_seed_mutations raises (MutationError or SeedMutationError
    # or SeedContractError) when a pre-commit beat declares two paths from
    # different dilemmas.
    with pytest.raises((MutationError, ValueError)):
        apply_seed_mutations(graph, {
            # mutation output that produces a cross-dilemma pre-commit beat
            ...
        })
```

The exact fixture depends on existing mutation test helpers. Read `tests/unit/test_mutations.py` for the SEED fixture pattern, and compose the smallest graph + output that triggers the violation. Key: the beat's `path_id` and `also_belongs_to` point to paths of different dilemmas.

- [ ] **Step 2: Run — expected FAIL**

Run:
```
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_rejects_precommit_with_mismatched_dilemmas -v
```
Expected: FAIL (current mutation code does not check this invariant at write time).

- [ ] **Step 3: Add producer-time guard**

In `apply_seed_mutations` (around the beat-write section where `path_id` / `also_belongs_to` are resolved into `belongs_to` edges), before the `graph.add_edge("belongs_to", …)` calls, verify dual-path dilemma agreement:

```python
            if also_belongs_to:
                primary_path = graph.get_node(primary_path_id)
                sibling_path = graph.get_node(also_belongs_to_id)
                if primary_path and sibling_path:
                    primary_dilemma = primary_path.get("dilemma_id")
                    sibling_dilemma = sibling_path.get("dilemma_id")
                    if primary_dilemma != sibling_dilemma:
                        raise MutationError(
                            f"Beat {beat_id!r} has cross-dilemma dual "
                            f"belongs_to: path_id ({primary_path_id!r}, dilemma "
                            f"{primary_dilemma!r}) and also_belongs_to "
                            f"({also_belongs_to_id!r}, dilemma {sibling_dilemma!r}). "
                            "Dual belongs_to must reference paths of the SAME dilemma "
                            "(R-3.6 / R-3.9)."
                        )
```

Exact variable names / control flow depends on the existing `apply_seed_mutations` code. Locate the block that currently reads `also_belongs_to` from the InitialBeat and creates the two edges; insert the guard just before the `add_edge` calls.

- [ ] **Step 4: Run test — expect PASS**

```
uv run pytest tests/unit/test_mutations.py::test_apply_seed_mutations_rejects_precommit_with_mismatched_dilemmas -v
```

- [ ] **Step 5: Run full validator + mutations sweep**

```
uv run pytest tests/unit/test_seed_validation.py tests/unit/test_mutations.py -k "seed" --tb=short -q 2>&1 | tail -15
```
Expected: new test passes; existing SEED-related tests have some failures (Phase F will address fixtures). Validator suite still green.

- [ ] **Step 6: Commit**

```bash
git add src/questfoundry/graph/mutations.py tests/unit/test_mutations.py
git commit -m "$(cat <<'EOF'
fix(seed): enforce Y-shape shared-beat dilemma agreement at write time (R-3.6, R-3.9)

Pre-commit beats with dual belongs_to must reference two paths of the
SAME dilemma. Cross-dilemma dual membership is a structural failure —
raise MutationError immediately rather than relying on the exit
validator.

Closes #1282.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: #1283 — Cross-dilemma `belongs_to` prohibition (broader than #1282)

Task 11 handled the dual-belongs_to-on-one-beat case. #1283 extends this: any beat with multiple `belongs_to` edges anywhere in the apply path must check all target dilemmas, not just the primary + sibling case.

**Files:**
- Modify: `src/questfoundry/graph/mutations.py`
- Modify: `tests/unit/test_mutations.py`

- [ ] **Step 1: Audit the beat-write code for any path where multi-edge belongs_to can be created**

Search `apply_seed_mutations` for any code that adds `belongs_to` edges outside the `path_id` + `also_belongs_to` block. Likely there is none today; if so, Task 11's guard covers #1283 fully — the remaining work is a targeted test.

- [ ] **Step 2: Add defensive test covering the broader invariant**

Append to `tests/unit/test_mutations.py`:

```python
def test_apply_seed_mutations_never_produces_cross_dilemma_belongs_to() -> None:
    """R-3.9: after apply_seed_mutations, no beat in the graph has
    belongs_to edges to paths of more than one dilemma."""
    # Build a realistic SEED output (reuse existing helper) and apply.
    # Then inspect belongs_to edges and assert no beat has multi-dilemma membership.
    ...
```

- [ ] **Step 3: Run — should PASS given Task 11's enforcement**

If it fails, locate the missing guard and add it.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_mutations.py src/questfoundry/graph/mutations.py
git commit -m "$(cat <<'EOF'
fix(seed): defensive coverage for no-cross-dilemma-belongs_to (R-3.9)

Extends the Task 11 write-time guard with a property-style test that
inspects post-apply graph state directly. Closes the gap that write-
time guards could miss if future beat-write paths are added.

Closes #1283.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: #1286 — Convergence analysis LLM failure surfacing (R-7.5)

**Files:**
- Modify: `src/questfoundry/agents/serialize.py` (convergence analysis call site)
- Modify: `tests/unit/test_serialize.py`

- [ ] **Step 1: Find the convergence-analysis call**

Run: `grep -n "dilemma_role\|convergence\|DilemmaAnalysis\|Phase 7" src/questfoundry/agents/serialize.py | head -20`. Locate the function that invokes the LLM and assigns `dilemma_role` / `residue_weight` / `ending_salience`.

- [ ] **Step 2: Add failing test**

Append to `tests/unit/test_serialize.py`:

```python
def test_convergence_analysis_llm_failure_logs_warning_not_silent() -> None:
    """R-7.5: on LLM failure, default analysis may be applied but the
    failure MUST be logged at WARNING with affected dilemma IDs. Silent
    default application is forbidden."""
    import logging
    # Mock an LLM call that raises or returns None.
    # Call the convergence-analysis wrapper.
    # Assert either a WARNING log entry contains the affected dilemma IDs,
    # OR the function raised explicitly — NOT silently defaulting.
    ...
```

- [ ] **Step 3: Replace silent-default path with warning-log-and-default (or raise)**

Per the spec R-7.5, defaults MAY be applied on LLM failure but the failure MUST be logged at WARNING with the affected dilemma IDs. Rewrite the silent except/pass pattern (if present) to use `log.warning("convergence_analysis_llm_failure", dilemma_ids=…, reason=…)` before applying defaults.

- [ ] **Step 4: Run test + commit**

```bash
git add src/questfoundry/agents/serialize.py tests/unit/test_serialize.py
git commit -m "$(cat <<'EOF'
fix(seed): log convergence-analysis LLM failure at WARNING (R-7.5)

Silent default application is forbidden. On LLM failure, defaults
(role: soft, weight: light, salience: low) may be applied — but the
failure is now logged at WARNING with the affected dilemma IDs.

Closes #1286.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: #1287 — Dilemma ordering LLM failure surfacing (R-8.5)

**Files:**
- Modify: `src/questfoundry/agents/serialize.py` (dilemma-ordering call site)
- Modify: `tests/unit/test_serialize.py`

Same structure as Task 13 but for the Phase 8 ordering call.

- [ ] **Step 1: Add failing test asserting WARNING on LLM failure**

Append to `tests/unit/test_serialize.py`. Pattern matches Task 13.

- [ ] **Step 2: Replace silent-empty-list path with log.warning-and-empty**

Per R-8.5: on LLM failure, zero relationships may be produced — but log at WARNING.

- [ ] **Step 3: Commit**

```bash
git add src/questfoundry/agents/serialize.py tests/unit/test_serialize.py
git commit -m "$(cat <<'EOF'
fix(seed): log dilemma-ordering LLM failure at WARNING (R-8.5)

Silent empty-relationships on LLM failure is forbidden. Log warning
with the affected dilemma pair count before proceeding.

Closes #1287.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: #1295 — Path Freeze approval gate

**Files:**
- Modify: `src/questfoundry/pipeline/stages/seed.py`
- Modify: `src/questfoundry/models/seed.py` (SeedOutput adds `human_approved_paths: bool`)
- Modify: `src/questfoundry/graph/mutations.py` (apply_seed_mutations writes seed_freeze node)
- Modify: `tests/unit/test_seed_stage.py`

Mirrors DREAM's Task 9 (#1271) pattern. Full R-6.4 rejection-loop UX is deferred to a follow-up issue (filed below).

- [ ] **Step 1: Add `human_approved_paths` field to `SeedOutput`**

In `src/questfoundry/models/seed.py`, find `SeedOutput` (line ~514). Add:

```python
    human_approved_paths: bool = Field(
        default=False,
        description=(
            "True when the human has explicitly approved the Path Freeze "
            "(--no-interactive implies pre-approval at invocation time)."
        ),
    )
```

- [ ] **Step 2: Write `seed_freeze` node in `apply_seed_mutations`**

In `apply_seed_mutations`, early in the function (after BRAINSTORM context is loaded), or at the very end alongside other SEED nodes:

```python
    graph.upsert_node(
        "seed_freeze",
        {
            "type": "seed_freeze",
            "human_approved": bool(output.get("human_approved_paths", False)),
        },
    )
```

- [ ] **Step 3: SEED stage dispatcher sets the field**

In `src/questfoundry/pipeline/stages/seed.py`, in `SeedStage.execute` after `artifact.model_dump()`:

```python
        artifact_data = artifact.model_dump()
        if not interactive:
            artifact_data["human_approved_paths"] = True
        else:
            # Minimal approval gate — full R-6.4 loop-back UX deferred.
            if user_input_fn is not None:
                if on_assistant_message is not None:
                    await on_assistant_message(
                        "Path Freeze: approve and continue to GROW? (y/n): "
                    )
                response = (await user_input_fn()) or ""
                if response.strip().lower().startswith("y"):
                    artifact_data["human_approved_paths"] = True
                else:
                    log.info("seed_paths_rejected_by_human")
                    raise SeedStageError(
                        "Path Freeze rejected by human — re-run SEED to revise."
                    )
            else:
                log.warning(
                    "seed_approval_fallback_no_input_fn",
                    reason="interactive=True but no user_input_fn; auto-approving",
                )
                artifact_data["human_approved_paths"] = True
```

Also add `class SeedStageError(Exception)` near the top of `pipeline/stages/seed.py` if it doesn't already exist (check with `grep -n "class SeedStageError" src/questfoundry/pipeline/stages/seed.py`).

- [ ] **Step 4: Update SEED test fixtures exercising apply_seed_mutations**

For every SEED fixture in `tests/unit/test_seed_stage.py` and `tests/unit/test_mutations.py` that exercises `apply_seed_mutations`, add `"human_approved_paths": True` to the SeedOutput / raw dict.

- [ ] **Step 5: Add new approval-behavior tests**

Append to `tests/unit/test_seed_stage.py`: tests analogous to the DREAM approval tests (non-interactive pre-approve, interactive y sets True, interactive n raises SeedStageError).

- [ ] **Step 6: File follow-up issue for R-6.4 rejection loop-back**

```
gh issue create --title "[spec-audit] SEED: implement R-6.4 rejection loop-back UX" --label "spec-audit,area:seed" --body "Follow-up to #1295. The Path Freeze gate added in this PR implements binary approve/reject-and-halt. Full R-6.4 behavior requires the human to indicate which phase (1–5) contains the misalignment and loop back there specifically. Deferred because it requires richer interactive UX not in scope for SEED compliance.

Spec reference: docs/design/procedures/seed.md R-6.4 + §Iteration Control backward-loops table.
Related: epic #1281, cluster #1295."
```

Capture the issue number for the PR body.

- [ ] **Step 7: Commit**

```bash
git add src/questfoundry/models/seed.py src/questfoundry/pipeline/stages/seed.py src/questfoundry/graph/mutations.py tests/unit/test_seed_stage.py tests/unit/test_mutations.py
git commit -m "$(cat <<'EOF'
fix(seed): record Path Freeze approval on seed_freeze node (R-6.4)

Adds human_approved_paths field to SeedOutput. Non-interactive mode
implies pre-approval. Interactive mode uses a simple y/n prompt;
full R-6.4 loop-back-to-phase UX is deferred to a follow-up issue.

Closes #1295.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase F — Moderate cluster fixes

Each task follows the same small-TDD pattern: add failing test, make it pass, commit.

### Task 16: #1284 — Setup/epilogue beat semantics validation

**Files:**
- Modify: `src/questfoundry/models/seed.py` (InitialBeat `role` literal if absent)
- Modify: `tests/unit/test_seed_models.py`

- [ ] **Step 1: If `InitialBeat.role` is not yet constrained, add a Literal**

Check `grep -n "role" src/questfoundry/models/seed.py`. If `role` exists as a free string or is absent, add:

```python
    role: Literal["setup", "epilogue", None] | None = Field(
        default=None,
        description="Structural role: 'setup' (story opener), 'epilogue' (story closer), or None for dilemma-owned beats.",
    )
```

- [ ] **Step 2: Add failing test in `test_seed_models.py` (R-3.14 semantics)**

- [ ] **Step 3: Run, fix, commit**

```bash
git commit -m "fix(seed): constrain InitialBeat.role to setup/epilogue literal (R-3.14)

Closes #1284.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 17: #1285 — `explored` immutability at model-validator level (R-2.3, R-5.2)

**Files:**
- Modify: `src/questfoundry/models/seed.py` (add `model_validator(mode="after")` to the model that owns `explored`)
- Modify: `tests/unit/test_seed_models.py`

The `explored` field is immutable after Phase 2. Pruning (Phase 5) drops Paths but doesn't mutate `explored`. The model-validator level ensures any code that tries to modify `explored` after construction fails loudly.

- [ ] **Step 1: Find which model holds `explored`** (likely `DilemmaDecision` at line ~79 or an AnswerDecision)

- [ ] **Step 2: Add validator guaranteeing `explored` cannot be mutated**

- [ ] **Step 3: Write test + commit**

```bash
git commit -m "fix(seed): enforce explored-field immutability post-Phase-2 (R-2.3, R-5.2)

Closes #1285.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 18: #1288 — Remove `flavor` from dilemma_role (R-7.1)

**Files:**
- Modify: `src/questfoundry/models/seed.py` (`DilemmaAnalysis.dilemma_role` Literal)
- Modify: `tests/unit/test_seed_models.py` (or wherever DilemmaAnalysis is tested)

- [ ] **Step 1: Grep for `flavor` references**

`grep -rn "\"flavor\"" src/questfoundry/models/seed.py`

- [ ] **Step 2: Change Literal from `{hard, soft, flavor}` to `{hard, soft}`**

- [ ] **Step 3: Failing test on instantiation with `flavor`, then passing test, commit**

```bash
git commit -m "fix(seed): remove 'flavor' from dilemma_role enum (R-7.1)

Flavor-level choices are POLISH false-branch concerns, not dilemma
roles. Deprecation completed.

Closes #1288.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 19: #1289 — Concurrent ordering normalization (R-8.3)

**Files:**
- Modify: `src/questfoundry/graph/mutations.py` (ordering-edge writer)
- Modify: `tests/unit/test_mutations.py`

Ensure that when `apply_seed_mutations` writes a `concurrent` ordering, it always normalizes `dilemma_a` to the lex-smaller ID — regardless of the order the LLM emits.

- [ ] **Step 1: Locate the ordering-edge write block**

`grep -n "relationship\|concurrent\|dilemma_a" src/questfoundry/graph/mutations.py`

- [ ] **Step 2: Add test — apply with non-normalized concurrent, assert post-write the node stores lex-smaller as dilemma_a**

- [ ] **Step 3: Add `if relationship == "concurrent" and a > b: a, b = b, a` before write**

- [ ] **Step 4: Commit**

```bash
git commit -m "fix(seed): normalize concurrent ordering to lex-smaller dilemma_a (R-8.3)

Closes #1289.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 20: #1290 — `shared_entity` derivation guard (R-8.4)

**Files:**
- Modify: `src/questfoundry/graph/mutations.py` (ordering-edge writer rejects `shared_entity` relationship)
- Modify: `tests/unit/test_mutations.py`

- [ ] **Step 1: Add test — apply SEED output with an ordering relationship of `shared_entity` raises**

- [ ] **Step 2: Add rejection in `apply_seed_mutations`**

- [ ] **Step 3: Commit**

```bash
git commit -m "fix(seed): reject shared_entity as declared ordering (R-8.4)

shared_entity is derived from anchored_to edges, not declared. Any
LLM output that declares one raises at write time.

Closes #1290.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 21: #1291 — Arc-count guardrail through Phase 7/8 (R-5.1)

The audit flagged that Phase 7/8 can weaken the arc-count invariant. `_check_arc_count_and_approval` already enforces R-5.1 at the exit validator level. This task adds a test that exercises the post-Phase-7/8 state, and if Phase 7/8 code touches path/answer state, audits that code for any path that could push arc count above 16.

**Files:**
- Modify: `tests/unit/test_seed_validation.py` (or `test_seed_stage.py`)
- Possibly modify: `src/questfoundry/pipeline/stages/seed.py` if Phase 7/8 has a bug

- [ ] **Step 1: Write a test that runs Phase 7/8 on a graph at the 16-arc threshold and asserts arc count stays ≤16**

- [ ] **Step 2: If the test fails, find the offending code path and fix**

- [ ] **Step 3: Commit**

```bash
git commit -m "fix(seed): preserve arc-count guardrail through Phase 7/8 (R-5.1)

Closes #1291.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 22: #1292 — Beat `entities` validated for narrative beats (R-3.13)

**Files:**
- Modify: `src/questfoundry/models/seed.py` (`InitialBeat.entities` validation)
- Modify: `tests/unit/test_seed_models.py`

The spec requires every beat to have non-empty `entities`. Tighten the Pydantic model to enforce `min_length=1` when `role` is not `setup`/`epilogue` (or universally if the spec's R-3.15 applies to setup/epilogue too).

Per spec R-3.13 + R-3.15, every beat (narrative and structural) must have non-empty `summary` and `entities`. So the constraint is universal.

- [ ] **Step 1: Change `entities: list[str] = Field(default_factory=list, …)` to `entities: list[str] = Field(min_length=1, …)`**

- [ ] **Step 2: Test fails on empty entities list, passes after change**

- [ ] **Step 3: Commit**

```bash
git commit -m "fix(seed): require non-empty entities on all beats (R-3.13, R-3.15)

Closes #1292.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 23: #1293 — `path_importance` spec-vs-code reconciliation

**Files:**
- Either modify: `docs/design/procedures/seed.md` (add the field to the spec)
- Or modify: `src/questfoundry/models/seed.py` (remove `path_importance`)

Before writing code, **decide which side is wrong**:

- [ ] **Step 1: Grep the codebase for every `path_importance` reference**

`grep -rn "path_importance" src/ tests/ docs/`

- [ ] **Step 2: Read the spec to confirm it's truly absent**

`grep -n "path_importance\|importance" docs/design/procedures/seed.md`

- [ ] **Step 3: Ask the user in the PR body which side to keep**

If the field has downstream consumers (GROW/POLISH use it), spec should gain it — per CLAUDE.md §Spec-gap policy, this is a spec update commit first, then code stays. If no consumers, remove from the model.

Default stance: **keep the field; add it to the spec** unless clearly unused. Implement as a spec-update commit:

```bash
git commit -m "docs(spec): add path_importance field to SEED (clarification per #1293)

Reconciles the existing models/seed.py path_importance field with the
procedure doc. No code change; spec now documents what the code
already supports.

Closes #1293.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 4: If chosen side is "remove from code"**, do a separate code-removal commit with appropriate tests.

---

### Task 24: #1294 — Consequence ripples validation (R-3.4)

**Files:**
- Modify: `src/questfoundry/models/seed.py` (`Consequence.ripples`)
- Modify: `tests/unit/test_seed_models.py`

- [ ] **Step 1: Change `ripples: list[str] = Field(default_factory=list, …)` to `ripples: list[str] = Field(min_length=1, …)`**

- [ ] **Step 2: Tests**

- [ ] **Step 3: Commit**

```bash
git commit -m "fix(seed): require ≥1 ripple per Consequence (R-3.4)

Closes #1294.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 25: Test-fixture cleanup — SEED mutation tests

All Phase E and Phase F code changes tighten validators. SEED mutation tests and SEED stage tests need their fixtures updated to satisfy the new contract.

**Files:**
- Modify: `tests/unit/test_mutations.py`
- Modify: `tests/unit/test_seed_stage.py`
- Modify: `tests/unit/test_seed_models.py`

- [ ] **Step 1: Run the SEED test suites and collect failures**

```
uv run pytest tests/unit/test_mutations.py tests/unit/test_seed_stage.py tests/unit/test_seed_models.py -k "seed" --tb=short -q 2>&1 | tail -40
```

- [ ] **Step 2: Triage each failure**

For each failing test, apply the rewrite-or-delete policy:
- **Rewrite:** the test's intent is still valid but the fixture needs updating (add `human_approved_paths: True`, add `dilemma_role: "soft"`, add Y-shape beats, add `ripples: ["x"]` to consequences, etc.).
- **Delete:** the test's premise is fundamentally pre-audit (e.g., asserts SEED produces output the new spec forbids). Note rationale in the commit.

- [ ] **Step 3: Iterate until the SEED test files go green**

Run the triage + fix loop until:
```
uv run pytest tests/unit/test_seed*.py tests/unit/test_mutations.py -k "seed" --tb=no -q
```
reports 0 failures.

- [ ] **Step 4: Run the non-downstream suite**

```
uv run pytest tests/unit/ -k "not grow and not polish and not fill and not dress and not ship" --tb=no -q 2>&1 | tail -5
```
Expected: 0 failures (modulo the pre-existing `test_provider_factory` pollution).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_mutations.py tests/unit/test_seed_stage.py tests/unit/test_seed_models.py
git commit -m "$(cat <<'EOF'
test(seed): update fixtures to satisfy tightened SEED output contract

Rewrite fixtures where the spec intent is unchanged (add Y-shape
scaffold, dilemma_role, seed_freeze approval, ripples). Delete tests
whose pre-audit premise is incompatible with the new spec.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 26: Downstream-break notice

After all Phase E/F fixes, downstream SEED→GROW transition tests and integration tests may be broken. Per the spec's "post-SEED is allowed to break" rule, no fix here. But document the break in the PR body and optionally file a tracking issue if the break is unusually large.

- [ ] **Step 1: Run GROW+ tests to catalog the break**

```
uv run pytest tests/unit/test_grow*.py --tb=no -q 2>&1 | tail -10
```

- [ ] **Step 2: Record the failure count and pattern for the PR body**

- [ ] **Step 3: Commit nothing (this is a documentation step only)**

---

## Phase G — Close-out

### Task 27: Remove `models/seed.py` pyright suppression

**Files:**
- Modify: `src/questfoundry/models/seed.py`

- [ ] **Step 1: Delete the suppression**

Remove from `src/questfoundry/models/seed.py`:
```python
# pyright: reportInvalidTypeForm=false
# TODO(#1281): cleanup during M-SEED-spec compliance work
```

- [ ] **Step 2: Run pyright**

```
uv run pyright src/
```
Expected: **0 errors**. If errors appear, they are newly-visible `reportInvalidTypeForm` issues that need narrower fixes (individual `# pyright: ignore[reportInvalidTypeForm]` on the problem line(s), or refactoring the problem type annotations).

- [ ] **Step 3: mypy + ruff**

```
uv run mypy src/
uv run ruff check src/
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/models/seed.py
git commit -m "$(cat <<'EOF'
chore(seed): remove pyright suppression from models/seed.py (#1281)

SEED compliance work complete. The file-wide reportInvalidTypeForm
suppression from PR #1352 is no longer needed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 28: Push and open PR

**Files:**
- None new.

- [ ] **Step 1: Final exit-criteria check**

```
uv run pytest tests/unit/test_seed*.py tests/unit/test_seed_validation.py --tb=no -q
uv run pytest tests/unit/ -k "not grow and not polish and not fill and not dress and not ship" --tb=no -q
uv run mypy src/
uv run pyright src/
uv run ruff check src/
```

All must pass (modulo the pre-existing pollution).

- [ ] **Step 2: Push**

```bash
git push -u origin feat/seed-compliance
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --title "feat(seed): compliance with authoritative spec" --body "$(cat <<'EOF'
## Summary

Brings SEED into compliance with \`docs/design/procedures/seed.md\`. Introduces \`validate_seed_output\` as the runtime oracle for SEED's Stage Output Contract, wires it at \`apply_seed_mutations\` exit, and closes all 14 audit clusters under epic #1281.

Spec: \`docs/superpowers/specs/2026-04-19-seed-compliance-design.md\`
Plan: \`docs/superpowers/plans/2026-04-19-seed-compliance.md\`

## Closed issues

- Closes #1282 — Y-shape shared beat enforcement (R-3.6, R-3.10)
- Closes #1283 — Cross-dilemma \`belongs_to\` prohibition (R-3.9)
- Closes #1284 — Setup/epilogue beat semantics (R-3.14, R-3.15)
- Closes #1285 — \`explored\` field immutability (R-2.3, R-5.2)
- Closes #1286 — Convergence analysis LLM failure logged at WARNING (R-7.5)
- Closes #1287 — Dilemma ordering LLM failure logged at WARNING (R-8.5)
- Closes #1288 — \`flavor\` removed from dilemma_role (R-7.1)
- Closes #1289 — Concurrent ordering normalization (R-8.3)
- Closes #1290 — \`shared_entity\` derivation guard (R-8.4)
- Closes #1291 — Arc count guardrail through Phase 7/8 (R-5.1)
- Closes #1292 — Beat \`entities\` validated (R-3.13)
- Closes #1293 — \`path_importance\` spec-vs-code reconciliation
- Closes #1294 — Consequence ripples required (R-3.4)
- Closes #1295 — Path Freeze approval gate (R-6.4)

Partial contribution to M-contract-chaining (#1346): adds \`validate_seed_output\` and wires SEED exit. GROW entry-side wiring deferred to epic #1296.

## New / removed

- New: \`src/questfoundry/graph/seed_validation.py\` (\`SeedContractError\`, \`validate_seed_output\`, 7 check helpers).
- New: \`tests/unit/test_seed_validation.py\` (rule-by-rule coverage).
- Removed: the \`# pyright: reportInvalidTypeForm=false\` suppression on \`src/questfoundry/models/seed.py\` (from PR #1352).

## Allowed breakage (per design spec)

- GROW / POLISH / FILL / DRESS / SHIP unit tests may fail because the tightened SEED output contract rejects artifacts those stages relied on. Accepted; each downstream stage's own compliance PR will align fixtures.
- Integration / e2e tests may break.
- \`test_provider_factory::test_create_chat_model_ollama_success\` — pre-existing pollution; unchanged.

## Deferred follow-up

- R-6.4 full rejection-loop UX (interactive "loop back to phase N" prompt) — see follow-up issue filed during Task 15.

## Test plan

- [x] \`validate_seed_output\` tests: 0 failures
- [x] SEED unit suites: 0 failures
- [x] Non-downstream unit suite: 0 failures (modulo pre-existing pollution)
- [x] mypy / pyright / ruff clean
- [ ] Manual: run DREAM → BRAINSTORM → SEED against a small project; verify the validator catches intentional contract violations and passes on a clean run

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Capture the PR URL in the final report.

- [ ] **Step 4: AI-bot reviews will arrive**

Per CLAUDE.md §Project Git Rules, batch any review responses locally before pushing the next round. Do not push incremental WIP commits.

---

## Self-review checklist

Ran before finalizing the plan:

1. **Spec coverage:**
   - 14 clusters covered: #1282 → Task 11, #1283 → Task 12, #1284 → Task 16, #1285 → Task 17, #1286 → Task 13, #1287 → Task 14, #1288 → Task 18, #1289 → Task 19, #1290 → Task 20, #1291 → Task 21, #1292 → Task 22, #1293 → Task 23, #1294 → Task 24, #1295 → Task 15. ✓
   - `validate_seed_output` created in Tasks 3–9 (1 per helper). ✓
   - Wired at exit in Task 10. ✓
   - Pyright suppression removed in Task 27. ✓
   - Baseline + rewrite-or-delete policy in Tasks 1 and 25. ✓
   - Downstream-break documentation in Task 26. ✓

2. **Placeholder scan:**
   - Tasks 11 / 12 / 13 / 14 / 15 / 21 / 23 / 25 reference existing helpers that the implementer must locate via `grep` rather than specifying exact line numbers. Each task includes the grep command and the fixture-building guidance. This is intentional — those locations will shift as the codebase grows; brittle line numbers are worse than "read surrounding code."
   - Task 13 and Task 14 test stubs end with `...` — these are mock-structure placeholders; the implementer fills in the actual pytest mock pattern based on `test_serialize.py`'s existing mock style. Acceptable given the test is small and pattern-matched.
   - No "TBD" / "TODO in-place". No "similar to Task N" without showing the code.

3. **Type consistency:**
   - `validate_seed_output(graph: Graph) -> list[str]` — consistent across spec, design, and all tasks. ✓
   - `SeedContractError(ValueError)` — defined in Task 3, referenced in Task 10. ✓
   - Helpers `_check_upstream_contract`, `_check_entities`, `_check_paths_and_consequences`, `_check_beats`, `_check_belongs_to_yshape`, `_check_convergence_and_ordering`, `_check_arc_count_and_approval` — consistent names in Tasks 3–9 and the validator body. ✓
   - `human_approved_paths` — used consistently in Task 15. ✓
   - `seed_freeze` node type — consistent in Task 2's fixture, Task 9's validator, Task 15's producer, Task 25's fixture updates. ✓

### Risk notes (not fixed — just flagged)

- Task 15 assumes `SeedStageError` exists or can be added. If `pipeline/stages/seed.py` has a different error-class convention, adapt. Plan says "check with grep" — good.
- Task 21's arc-count-through-Phase-7/8 may not need a code fix if Phase 7/8 doesn't touch path state. Plan allows "if the test fails, find and fix" which is honest but might mean the task degenerates to just a test addition.
- Task 25 (fixture cleanup) is the highest-risk task — it rewrites many tests. Budget more review time for this one during subagent-driven execution.
