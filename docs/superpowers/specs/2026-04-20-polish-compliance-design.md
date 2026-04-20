# POLISH Spec-Compliance — Design

**Date:** 2026-04-20
**Epic:** #1310 (M-POLISH-spec)
**Scope:** Hot-path 4 clusters (#1311, #1312, #1313, #1314) — follow-on PRs handle #1315–#1318.
**Policy:** POLISH never worked correctly before; post-POLISH breakage (FILL, DRESS, SHIP) is allowed.

---

## 1. Sequencing and dependencies

Two sequential PRs:

**PR 1 — Spec update (blocks PR 2).** Doc-only change to `docs/design/procedures/polish.md`:
- Rewrite **R-4a.3** to a maximal-linear-collapse rule that applies uniformly to narrative and structural beats.
- Remove **R-4a.2** entirely (identical path membership is not a necessary property of a topology-based grouping rule; linear runs may span different `belongs_to` sets). No Phase 7 check is needed for this property.
- Rewrite the Phase 4a "Operations" subsection to match the new rule.
- No rule-count change. No other procedure docs touched.

**PR 2 — POLISH compliance (this spec's target).** Branch `feat/polish-compliance` off `main` after PR 1 merges. Closes 4 hot-path cluster issues.

Deferred to follow-on PRs (one each, same branching policy): #1315, #1316, #1317, #1318.

---

## 2. New R-4a.3 (for PR 1)

> **R-4a.3.** All beats in the finalized DAG — narrative or structural — are grouped by the maximal-linear-collapse rule. A passage is a maximal run of beats with no internal divergence or convergence. A new passage starts at any beat whose in-degree ≠ 1 or whose predecessor has out-degree ≠ 1, and ends at any beat whose out-degree ≠ 1 or whose successor has in-degree ≠ 1.

**Implications:**

- Micro-beats (Phase 2) sit in linear sections by construction → absorbed into their parent passage; no special rule.
- Transition beats (GROW) land at path-segment boundaries → DAG topology naturally gives them their own passage when they sit at divergences/convergences, or folds them into a neighbor when linear.
- Residue and false-branch beats are created in Phase 6 *after* grouping is fixed. Their passage placement is governed by Phase 6 rules (R-6.x), not by Phase 4a.
- R-4a.2 is removed: a linear DAG run may legitimately span different `belongs_to` sets (e.g., post-commit of dilemma A → transition beat → pre-commit of dilemma B). Phase 7 does not check path-membership uniformity within a passage.

---

## 3. Phase 7 validator extensions (PR 2)

**New class.** `PolishContractError(ValueError)` in `src/questfoundry/graph/polish_validation.py`. Mirrors `GrowContractError`.

**Phase 7 wiring.** `phase_validation()` in `polish/deterministic.py:1395-1397` currently returns `PhaseResult(status="failed", detail=...)` on errors; the stage then wraps as `PolishStageError`. Replace with: Phase 7 emits `log.error("polish_contract_failed", error_count=N, errors=[...])` then raises `PolishContractError` directly.

**New `_check_*` helpers added to `validate_polish_output`:**

1. **`_check_no_character_arc_metadata_nodes(graph)`** — R-3.3.
   - Error if any node has type `character_arc_metadata`.
   - Error if any `has_arc_metadata` edge exists.
   - For each entity with ≥2 `appears` edges, error if its data dict lacks a `character_arc` field with `start`, `pivots`, `end_per_path`.

2. **`_check_passage_maximal_linear_collapse(graph)`** — R-4a.4 (and the new R-4a.3 from PR 1).
   - Each passage's member beats form a linear run in the finalized beat DAG (in-passage predecessors/successors unique).
   - Passage boundaries sit at DAG divergences or convergences: first beat's in-degree ≠ 1 or predecessor has out-degree ≠ 1; mirror for last beat.
   - Error catches any passage that spans a divergence/convergence or stops mid-linear-run.

3. **`_check_residue_mapping_strategy(graph)`** — R-5.7, R-5.8.
   - Every residue passage node has a `mapping_strategy` attribute in `{residue_passage_with_variants, parallel_passages}`.
   - Topological check: `residue_passage_with_variants` → one residue passage with variant children; `parallel_passages` → N sibling passages branching then rejoining.

4. **`_check_has_choice_edges(graph)`** — R-4c.2 (belt-and-suspenders).
   - Error if passage graph has zero `choice` edges. Phase 4c is expected to halt before Phase 7, but this catches silent regressions.

**Existing checks preserved.** Current R-7.x structural checks stay as-is. New checks are additive.

**Not in this PR's validator.** R-2.5, R-4c.3/4, R-5.2, R-5.10 — added with their respective follow-on cluster PRs.

---

## 4. Per-cluster fixes (PR 2)

### Cluster #1311 — R-4a.4: remove intersection-group consumption

**Current state:** `compute_beat_grouping()` in `polish/deterministic.py:213-234` iterates intersection groups and creates passages with `grouping_type="intersection"`.

**Fix:**
- Delete the intersection iteration branch entirely.
- Replace with a single-pass algorithm: walk the finalized beat DAG; seed queue with boundary beats (in-degree ≠ 1 or predecessor out-degree ≠ 1); extend each into a linear run; close at next divergence/convergence. Applies uniformly to narrative and structural beats.
- Passage nodes written without any `intersection` provenance field.
- POLISH may still read `intersection` edges for context in LLM prompts (e.g., variant generation), but they cannot constrain grouping.

### Cluster #1312 — R-4c.2: zero-choice ERROR halt

**Current state:** Phase 4c computes choice edges; zero result passes through silently.

**Fix:**
- After `plan.choice_specs = compute_choice_edges(...)` in `polish/deterministic.py:146`:
  - `if not plan.choice_specs:`
    - `log.error("polish_zero_choice_halt", upstream="SEED/GROW", detail="...")`
    - `raise PolishContractError("Phase 4c produced zero choice edges — SEED/GROW DAG has no Y-forks. Upstream bug.")`
- Not a warning, not a fallback. Pipeline halts.

### Cluster #1313 — R-5.7, R-5.8: residue mapping_strategy

**Current state:** `ResidueSpec` has no `mapping_strategy` field; Phase 6 uses a single hardcoded shape.

**Fix:**
- Add `mapping_strategy: Literal["residue_passage_with_variants", "parallel_passages"]` to `ResidueSpec` in `models/polish.py:181-194`. Required field, no default.
- Phase 5b LLM prompt: describe both options with narrative-appropriate criteria; structured output schema includes `mapping_strategy`.
- Phase 6 `_create_residue_beat_and_passage()` in `deterministic.py:1176-1241`: branch on `spec.mapping_strategy`.
  - `residue_passage_with_variants` → one residue passage with variant children (existing shape).
  - `parallel_passages` → N sibling passages branching then rejoining.
- Phase 6 writes `mapping_strategy` onto the residue passage node's data dict for Phase 7 validation.

### Cluster #1314 — R-3.3: arc metadata as entity annotation

**Current state:** Phase 3 creates `character_arc_metadata` nodes and `has_arc_metadata` edges in `polish/llm_phases.py:286-299`.

**Fix:**
- Remove `character_arc_metadata` node creation.
- Remove `has_arc_metadata` edge creation.
- Replace with entity-data-dict mutation: `entity.data["character_arc"] = {"start": ..., "pivots": {...}, "end_per_path": {...}}` via the existing entity-mutation pathway.
- Downstream readers that currently fetch `character_arc_metadata` nodes must read `entity.data["character_arc"]` — but that's downstream breakage, deferred.

---

## 5. Test fixtures and downstream breakage

**Fixture policy (same as GROW):**
- **Rewrite** if the test's spec-intent is still valid post-audit (just update the assertions).
- **Delete + track** if the premise is pre-audit. Deletions get a follow-on issue if rewritten coverage doesn't subsume the old test.

**Known fixture rewrites:**
- `tests/unit/test_polish_deterministic.py` — Phase 4a grouping: assertions from intersection-driven → DAG-topology-driven.
- `tests/unit/test_polish_phases.py:398-436` — `test_has_arc_metadata_edge_created` deleted; replaced by `test_character_arc_field_on_entity`.
- `tests/unit/test_polish_apply.py` — residue mapping: parameterize over both `mapping_strategy` values.
- `tests/unit/test_polish_llm_phases.py` — Phase 5b prompt/response tests add `mapping_strategy`; Phase 3 tests stop expecting `character_arc_metadata` nodes.

**Known downstream-breaking tests (allowed to fail; tracked, not fixed here):**
- `tests/unit/test_polish_passage_validation.py` — pre-audit passage-validation suite; some assertions tied to intersection-driven grouping will fail.
- Any `tests/unit/test_fill_*.py` / `test_dress_*.py` / `test_ship_*.py` reading `character_arc_metadata` nodes or intersection-derived structure.
- `tests/integration/test_polish_e2e.py` and beyond.

**Verification sweeps:**
- POLISH-local: `uv run pytest tests/unit/test_polish_{contract,validation,phases,deterministic,apply,llm_phases}.py -x -q` — must pass.
- Non-downstream sweep: `uv run pytest tests/unit/ -x -q` with concrete ignore list for FILL/DRESS/SHIP/post-POLISH consumers — all green before merge.
- `uv run mypy src/ && uv run ruff check src/ && uv run pyright src/` — all clean.
- Full suite: CI only.

**Upstream check.** POLISH's entry contract is `validate_grow_output`, already called at `stage.py:242`. No kwarg additions needed. GROW PR (#1357 — spec-compliance for GROW) must be merged before POLISH branches.

---

## 6. Work sequence

19 tasks across 5 phases. Subagent-driven execution (fresh implementer per task + two-stage review).

**Phase 0 — Prereqs (1 task)**
- T1. Baseline sanity: spec PR merged; main clean; branch `feat/polish-compliance` off `main`.

**Phase 1 — Validator scaffolding (2 tasks)**
- T2. Add `PolishContractError(ValueError)` to `polish_validation.py`.
- T3. Rewire `phase_validation()` to raise `PolishContractError` with structured `log.error("polish_contract_failed", …)` event instead of wrapping in `PolishStageError`.

**Phase 2 — Validator extensions (5 tasks)**
- T4. Write failing contract tests for all 4 hot-path rules in new `tests/unit/test_polish_contract.py`. Layered DREAM+BRAINSTORM+SEED+GROW compliant baseline fixture (mirroring GROW contract-test pattern).
- T5. `_check_no_character_arc_metadata_nodes` (R-3.3).
- T6. `_check_passage_maximal_linear_collapse` (R-4a.4).
- T7. `_check_residue_mapping_strategy` (R-5.7, R-5.8).
- T8. `_check_has_choice_edges` (R-4c.2, belt-and-suspenders).

**Phase 3 — Cluster fixes (4 tasks, small-to-large)**
- T9. #1314 / R-3.3: arc metadata as entity annotation.
- T10. #1312 / R-4c.2: zero-choice ERROR halt.
- T11. #1311 / R-4a.4: maximal-linear-collapse grouping.
- T12. #1313 / R-5.7, R-5.8: `mapping_strategy` field, Phase 5b prompt, Phase 6 branching.

**Phase 4 — Fixture cleanup + sweep (6 tasks)**
- T13. Rewrite `test_polish_phases.py` Phase-3 arc tests; delete `test_has_arc_metadata_edge_created`.
- T14. Rewrite `test_polish_deterministic.py` Phase-4a grouping tests.
- T15. Rewrite `test_polish_apply.py` residue tests (parameterize over both `mapping_strategy`).
- T16. Rewrite `test_polish_llm_phases.py` Phase-5b + Phase-3 tests.
- T17. Non-downstream sweep: mypy/ruff/pyright clean; targeted pytest green on POLISH-local + upstream regression tests.
- T18. Catalog downstream-breaking tests; file follow-on issues (notice only).

**Phase 5 — Ship (1 task)**
- T19. Push `feat/polish-compliance`; open PR with pre-written body (`Closes #1311, #1312, #1313, #1314`; downstream-break notice; CI expectations).

---

## 7. Exit criteria

PR is mergeable when all are true:

1. `tests/unit/test_polish_contract.py` — all 4 hot-path rule tests green.
2. `validate_polish_output` errors → `phase_validation()` raises `PolishContractError` with structured `log.error("polish_contract_failed", …)` event.
3. No `character_arc_metadata` nodes in any POLISH-produced graph. Grep: `grep -rn 'character_arc_metadata\|has_arc_metadata' src/` returns zero hits outside historical removal comments.
4. No intersection-group consumption in Phase 4a: no reference to `intersection` edges or intersection-group iteration in `compute_beat_grouping()`.
5. Residue specs carry `mapping_strategy`; model validation rejects absence; Phase 6 branches on both values; validator errors on absent.
6. Phase 4c raises `PolishContractError` with upstream-identifying message when `compute_choice_edges()` returns empty.
7. All `tests/unit/test_polish_{contract,validation,phases,deterministic,apply,llm_phases}.py` pass.
8. Non-downstream sweep green with documented ignore list.
9. `uv run mypy src/` / `ruff check src/` / `pyright src/` — zero errors.
10. Downstream-break notice filed; PR body links follow-on issues.
11. PR body includes `Closes #1311, #1312, #1313, #1314`.

**Explicit non-goals:**
- Full POLISH test suite green end-to-end (downstream allowed to break).
- Removal of `# pyright: reportArgumentType=false` on `polish/stage.py:16` — saved for final cluster PR.
- Integration test green (CI-only decision).
- Spec PR merged *by* this PR — prereq, not exit criterion.
