# GROW Compliance — Design

**Date:** 2026-04-20
**Status:** Approved — ready for implementation plan
**Authoritative specs being brought into compliance with:**
- `docs/design/procedures/grow.md`
- `docs/design/how-branching-stories-work.md`
- `docs/design/story-graph-ontology.md` (Parts 5–9: beat DAG, intersections, state flags, ordering)

Plus CLAUDE.md §Design Doc Authority, §Silent Degradation, and §Logging.

## Problem

The 2026-04-19 spec-compliance audit (report at `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` §M-GROW-spec) surfaced 13 compliance gaps across GROW (epic #1296, clusters #1297–#1309). GROW owns the beat-DAG interleaving, intersection-group formation, state-flag derivation, transition-beat insertion, and pairwise-dilemma-ordering edges — structural output that POLISH converts to passages. The audit flagged six clusters as hot-path priority (#1297 Y-fork postcondition, #1298 no-conditional-prerequisites, #1299 temporal hint acyclicity, #1300 transition beat zero-overlap seam detection, #1301 soft-dilemma no-convergence halt, #1303 all-intersections-rejected ERROR escalation) plus one removal task (#1302 dead passage/choice counting from the pre-POLISH-split era).

Per CLAUDE.md §Design Doc Authority, the authoritative procedure doc supersedes code and tests. This project brings GROW into compliance with `grow.md`, consolidates `validate_grow_output` into `grow_validation.py` as the runtime oracle for GROW's Stage Output Contract (currently the function lives in `polish_validation.py` for historical reasons), and removes GROW's file-level pyright suppressions (tagged `TODO(#1296)` in commit `425eebb5`).

## Goal

Produce a GROW pipeline stage that:
1. Closes all 13 audit clusters against the authoritative spec.
2. Emits artifacts that pass a consolidated `validate_grow_output(graph)` oracle at stage exit.
3. Surfaces every structural violation loudly (no silent skips, no silent fallbacks, no WARNING-as-ERROR).
4. Leaves `grow/stage.py` and `grow/llm_phases.py` passing pyright standard mode without suppressions.

No changes to POLISH/FILL/DRESS/SHIP. Downstream breakage caused by the tightened GROW output contract is explicitly accepted — this project stops at GROW's exit boundary. POLISH's entry call to `validate_grow_output` continues to work unchanged (only the import path moves).

## Scope

**In scope:**
- Close audit clusters: #1297, #1298, #1299, #1300, #1301, #1302, #1303, #1304, #1305, #1306, #1307, #1308, #1309.
- #1302 is pure deletion (dead passage/choice counting from pre-POLISH-split era).
- Consolidate `validate_grow_output` into `src/questfoundry/graph/grow_validation.py`; add `GrowContractError`.
- Wire `validate_grow_output` at GROW exit in `src/questfoundry/pipeline/stages/grow/stage.py` (after the final phase, before `graph.set_last_stage("grow")`).
- POLISH entry continues to call `validate_grow_output` via import from the new location (closes GROW's slice of #1347 / #1348 contract-chaining).
- Add `skip_forbidden_types: bool = False` kwarg to `validate_seed_output` (mirrors the BRAINSTORM pattern from PR #1356) so GROW's upstream check can include seed-level invariants without tripping R-Output-16's "no state_flag / intersection_group / transition_beat" clause (GROW legitimately creates those).
- Remove the two `# pyright: …=false` + `TODO(#1296)` suppressions on `src/questfoundry/pipeline/stages/grow/stage.py` and `src/questfoundry/pipeline/stages/grow/llm_phases.py`.
- Test updates per the rewrite-or-delete policy.
- Drift discovered during implementation: blocking → fix inline, note in PR body; non-blocking → file new cluster issue under #1296, defer.

**Out of scope:**
- POLISH/FILL/DRESS/SHIP compliance — allowed to break.
- Integration / e2e tests that break from the new contract.
- `run_grow_checks` / `check_*` legacy infrastructure in `grow_validation.py` — stays for `qf inspect`; we only add the new contract validator alongside.
- 2 uncheckable rules in SEED's audit — stay deferred.
- `test_provider_factory::test_create_chat_model_ollama_success` pre-existing pollution.

## Architecture

Reference pattern: the prior per-stage validators (`dream_validation.py`, `brainstorm_validation.py`, `seed_validation.py`) introduced in PRs #1351 / #1356. GROW mirrors their shape, with one twist: the validator function already exists (in the wrong module) and is consumed by POLISH's entry check. Consolidation is a move + extend.

**Function contract:**

```python
def validate_grow_output(graph: Graph) -> list[str]:
    """Check graph satisfies GROW's Stage Output Contract.

    Returns a list of human-readable error strings; empty list means compliant.
    Pure read-only — never mutates the graph.
    """
```

**Error-type discipline:**

- Validator returns strings; never raises; never logs.
- GROW stage tail: if list non-empty, `log.error("grow_contract_violated", errors=errors)` then `raise GrowContractError(...)`. Matches the SEED pattern.
- `GrowContractError(ValueError)` lives in `grow_validation.py`. `GrowMutationError` (already defined in `mutations.py`) stays for mutation-time failures.
- POLISH entry continues to call `validate_grow_output` at `polish/stage.py:239` (unchanged) but imports from the new location. If POLISH's call produces errors, POLISH raises `PolishStageError` (existing behavior).

**Internal decomposition into private helpers:**

- `_check_upstream_contract(graph, errors)` — delegates to `validate_seed_output(graph, skip_forbidden_types=True)`; prefixes as "Output-0: SEED contract violated post-GROW — …". Guards against GROW silently corrupting upstream state.
- `_check_beat_dag(graph, errors)` — predecessor cycles, Y-fork postcondition (#1297), no-conditional-prerequisites (#1298), temporal hint acyclicity validator side (#1299).
- `_check_intersections(graph, errors)` — intersection candidates + groups, all-intersections-rejected raise (#1303 validator side), deterministic signal derivation (#1308 validator side).
- `_check_state_flags(graph, errors)` — derivation edges, one-per-consequence, world-state phrasing (#1305, #1306 validator side; R-6.1, R-6.2, R-6.4).
- `_check_entity_overlays(graph, errors)` — composition rules (#1307 validator side; R-6.7, R-6.8).
- `_check_transition_beats(graph, errors)` — seam detection, zero-overlap (#1300 validator side; R-5.2).
- `_check_arc_enumeration(graph, errors)` — materialized arc data prefix (#1309 validator side; R-8.2).
- `_check_convergence_and_ordering_exit(graph, errors)` — soft-dilemma no-convergence halt (#1301 validator side; R-7.4).

**Silent-skip rewrites:**

- #1299 temporal hint acyclicity on failure → `log.error` + raise at call site (not silent rethrow).
- #1301 soft-dilemma no-convergence → raise `GrowContractError` at detection; no "continue with partial convergence."
- #1303 all-intersections-rejected → `log.error` + raise (was `log.warning` + return empty).
- #1304 logging-level misuse at validation failure sites — reclassify per CLAUDE.md §Logging litmus test.

**Wiring:**

- `src/questfoundry/pipeline/stages/grow/stage.py` — in the stage's phase-loop tail, after the final registered phase completes and before `graph.set_last_stage("grow")`, call `validate_grow_output(graph)`; log + raise on non-empty.
- `src/questfoundry/graph/polish_validation.py` — `validate_grow_output` local definition deleted; replaced with `from questfoundry.graph.grow_validation import validate_grow_output`. POLISH's call site at `polish/stage.py:239` is functionally unchanged.

**`validate_seed_output(graph, skip_forbidden_types=True)` requirement:**

GROW legitimately creates `state_flag`, `intersection_group`, and `transition_beat` nodes; SEED's output contract item 16 forbids those. Mirroring the PR #1356 approach for BRAINSTORM→SEED, we add `skip_forbidden_types: bool = False` kwarg to `validate_seed_output`. GROW's `_check_upstream_contract` passes `True`. Default behavior unchanged for SEED's own exit check.

**What is deliberately NOT changed:**

- GROW's phase-based stage architecture (phases are registered, executed in dependency order). The validator runs after all phases complete.
- The existing `run_grow_checks` / `check_*` / `ValidationReport` infrastructure in `grow_validation.py` — it serves `qf inspect` and stays untouched.
- Prompt templates, unless a cluster fix requires a producer-level prompt change to emit compliant output.
- Models (`src/questfoundry/models/grow.py`) structure overall; isolated field-level tightening for #1306 state flag naming or #1307 overlay composition is in scope but model-wide refactor is not.

## Components

### New files

| Path | Responsibility |
|---|---|
| `tests/unit/test_grow_validation_contract.py` | Rule-by-rule contract tests with compliant-baseline fixture + parametrized negatives. Separate from the existing `test_grow_validation.py` to avoid mixing two test patterns (legacy `ValidationCheck` vs new `list[str]`). |

### Modified files (code)

| Path | Change |
|---|---|
| `src/questfoundry/graph/grow_validation.py` | Add `GrowContractError`. Move `validate_grow_output` in from `polish_validation.py`. Extend with 8 private `_check_*` helpers. |
| `src/questfoundry/graph/polish_validation.py` | Delete local `validate_grow_output`; replace with import from `grow_validation.py`. |
| `src/questfoundry/graph/seed_validation.py` | Add `skip_forbidden_types: bool = False` kwarg to `validate_seed_output`. |
| `src/questfoundry/pipeline/stages/grow/stage.py` | Wire `validate_grow_output` at stage tail; raise `GrowContractError` with structured log event. Fix #1299 temporal hint acyclicity; fix #1301 soft-dilemma convergence halt. Remove file-level pyright suppression after verifying clean. |
| `src/questfoundry/pipeline/stages/grow/llm_phases.py` | Fix #1303 all-intersections-rejected ERROR escalation. Fix #1304 logging-level misuse. Remove file-level pyright suppression after verifying clean. |
| `src/questfoundry/graph/grow_algorithms.py` | Fix #1297 Y-fork postcondition check, #1298 no-conditional-prerequisites write-time guard, #1300 transition beat zero-overlap seam check, #1308 deterministic intersection candidate signals. Remove dead passage/choice counting code per #1302. |
| `src/questfoundry/graph/mutations.py` | Potentially #1305 state flag derivation edge validation, #1306 state flag name enforcement, #1307 entity overlay composition — decisions resolved during plan-writing based on where the code lives. |
| `src/questfoundry/models/grow.py` | Potentially #1306 state flag naming Literal, #1307 overlay composition model-level checks — plan-writing resolves. |

### Modified files (tests)

| Path | Change |
|---|---|
| `tests/unit/test_grow_algorithms.py` | Update fixtures for new contracts. Delete tests that exercise deleted #1302 code. |
| `tests/unit/test_grow_stage.py` | Update fixtures for contract-satisfaction. |
| `tests/unit/test_grow_deterministic.py` | Update fixtures for tightened beat-DAG invariants. |
| `tests/unit/test_grow_models.py` | Update fixtures for tightened state-flag / overlay models (if applicable). |
| `tests/unit/test_grow_validators.py` | Update fixtures; no pattern change. |
| `tests/unit/test_polish_entry_contract.py` | May break if fixtures relied on GROW-output shapes the new contract forbids — rewrite per policy. |
| `tests/unit/test_grow_validation.py` | Largely unchanged (tests legacy `check_*` functions, not the new contract validator). |

### Potentially modified — owner resolved during plan writing

- #1305 state flag derivation edge validation: `mutations.py` write-time guard vs validator-only. Lean toward validator-only unless the invariant can be checked cheaply at write time.
- #1306 state flag naming: validator-only (naming is semantic and cross-cutting).
- #1307 entity overlay composition: model-level (Pydantic) + validator. The model catches LLM-level errors; validator catches graph-state errors.

### Deleted files

- None anticipated. Individual tests may be deleted under the rewrite-or-delete policy; whole-file deletions are not planned.

### Not modified

- POLISH/FILL/DRESS/SHIP stage code (except the one-line import swap in `polish_validation.py`).
- `grow_validation.py`'s existing `check_*` / `run_grow_checks` infrastructure (consumed by `qf inspect`).
- GROW's phase-based orchestration.
- Prompt templates unless a specific cluster demands it.

## Work sequence (TDD order)

Approach 1 (validator-first). One commit per numbered step. Targets ~32 commits.

**Phase A — Baseline + dead-code removal.**
1. Confirm non-downstream suite green modulo pre-existing `test_provider_factory` pollution.
2. #1302 — identify and remove dead passage/choice counting code from `grow_algorithms.py` (+ related files + tests that exercise it).

**Phase B — Consolidate `validate_grow_output` + failing contract tests.**
3. Move `validate_grow_output` from `polish_validation.py` into `grow_validation.py`; add `GrowContractError`. POLISH imports from the new location. All existing tests still pass.
4. Create `tests/unit/test_grow_validation_contract.py` with compliant-baseline fixture plus one failing test per GROW Stage Output Contract rule + per silent-degradation invariant.

**Phase C — Extend SEED validator.**
5. Add `skip_forbidden_types: bool = False` kwarg to `validate_seed_output` (mirrors BRAINSTORM pattern from PR #1356).

**Phase D — Implement new check helpers (one commit per helper).**
6. `_check_upstream_contract` — delegates to `validate_seed_output(graph, skip_forbidden_types=True)`.
7. `_check_beat_dag`.
8. `_check_intersections`.
9. `_check_state_flags`.
10. `_check_entity_overlays`.
11. `_check_transition_beats`.
12. `_check_arc_enumeration`.
13. `_check_convergence_and_ordering_exit`.

**Phase E — Wire validator at GROW exit.**
14. `grow/stage.py` calls `validate_grow_output` at end of phase loop; raises `GrowContractError` with structured log.

**Phase F — Hot-path cluster fixes (producer side).**
15. #1297 Y-fork postcondition enforcement.
16. #1298 No-conditional-prerequisites invariant.
17. #1299 Temporal hint acyclicity ERROR-logged.
18. #1300 Transition beat zero-overlap seam detection.
19. #1301 Soft-dilemma no-convergence halts loudly.
20. #1303 All-intersections-rejected raises at ERROR.

**Phase G — Moderate cluster fixes.**
21. #1304 Logging-level misuse at validation failure sites.
22. #1305 State flag derivation edge validation.
23. #1306 State flag name world-state phrasing.
24. #1307 Entity overlay composition.
25. #1308 Intersection candidate signals deterministic.
26. #1309 Materialized arc data prefix.

**Phase H — Fixture cleanup + close-out.**
27. Full GROW test sweep; rewrite / delete fixtures until `test_grow*` is green.
28. Non-downstream suite sweep; fix any upstream fixture regressions (expected: none).
29. Remove pyright suppression from `grow/stage.py`.
30. Remove pyright suppression from `grow/llm_phases.py`.
31. Downstream-break notice (documentation only) — catalog POLISH+ failures for PR body.
32. Push branch + open PR.

## Testing strategy

### Contract validator tests

- File: `tests/unit/test_grow_validation_contract.py`.
- Naming: `test_<rule_id>_<short_description>` — e.g. `test_R_1_4_yfork_postcondition`, `test_R_2_7_no_conditional_prerequisites`, `test_R_3_7_temporal_hint_acyclicity_raises`.
- Each rule test builds a minimal graph violating exactly that rule + asserts substring match on the error list.
- Compliant baseline: `compliant_graph` fixture layered as `_seed_dream_baseline` → `_seed_brainstorm_baseline` → `_seed_seed_baseline` → `_seed_grow_baseline` (matches the SEED test file's fixture-layering pattern).
- `pytest.mark.parametrize` for rule families.

### Legacy validator tests

- `tests/unit/test_grow_validation.py` is unchanged unless the legacy `check_*` functions themselves need adjustment. They test `run_grow_checks` / `ValidationReport` — different pattern, different consumer.

### Producer tests

- `tests/unit/test_grow_algorithms.py`, `test_grow_stage.py`, `test_grow_deterministic.py`, `test_grow_models.py`, `test_grow_validators.py` — existing structure kept; rewrite assertions that encode pre-audit behavior; delete tests whose premise is pre-audit.
- Add targeted tests for new raise-behavior (#1299, #1301, #1303).
- One end-to-end pathway that runs `producer → apply X → validate_grow_output(graph) == []`.

### POLISH entry-contract tests

- `tests/unit/test_polish_entry_contract.py` — continues to pass. Import path for `validate_grow_output` changes but call semantics are unchanged.
- May surface new failures from fixtures that relied on a GROW shape the tighter contract rejects — rewrite per policy.

### Integration tests

- Out of scope. Allowed to break.

### Coverage target

- 85% for the new contract validator code (matches CLAUDE.md).
- No target on modified code.

### What is NOT tested

- LLM prompt quality (2 uncheckable rules stay deferred).
- Live LLM-call behavior — mocks only.
- Downstream stage integration.

## Error handling

- Validator: pure, returns `list[str]`, never raises, never logs.
- Caller (GROW stage tail): structured log event before raise, per PR #1351 / PR #1356 pattern.
- Silent-skip sites rewritten at source for #1299, #1301, #1303. No structural failure may be silently absorbed.
- `GrowContractError(ValueError)` in `grow_validation.py`; `GrowMutationError` retained in `mutations.py` for mutation-time failures.

## Spec-gap policy

Per CLAUDE.md §Instruction Hierarchy. If a spec rule is silent, ambiguous, or self-contradictory during implementation:

1. Stop. Do not guess.
2. Raise the question in the current session.
3. On alignment: update the spec in a dedicated `docs(spec): clarify …` commit before any code change.
4. Then update code and tests to match.
5. Never flip the order.

Audit clusters are implementation guidelines, not the authoritative spec. Where a cluster description conflicts with `grow.md`, the spec wins and the PR body calls out the clarification.

## Exit criteria

1. All 13 cluster issues closed via `Closes #…` in PR body: #1297–#1309.
2. `validate_grow_output` lives in `grow_validation.py`; `tests/unit/test_grow_validation_contract.py` covers every rule in GROW Stage Output Contract with a named test.
3. `grow/stage.py` calls `validate_grow_output` at exit.
4. POLISH entry continues to call `validate_grow_output` (import path moved, behavior unchanged).
5. `uv run pytest tests/unit/test_grow*.py` — 0 failures.
6. `uv run pytest tests/unit/ -k "not polish and not fill and not dress and not ship"` — 0 failures, modulo pre-existing `test_provider_factory` pollution.
7. `uv run mypy src/` — clean.
8. `uv run ruff check src/ tests/` — clean on modified files.
9. `uv run pyright src/` — 0 errors. Both `grow/stage.py` and `grow/llm_phases.py` suppressions removed and the files pass standard mode cleanly.
10. Downstream breakage (POLISH+ unit tests, integration / e2e tests) is allowed and not blocking.
11. PR body lists cluster closures, inline-fixed drift, deferred drift with new issue numbers.

## What does not change

- No changes to POLISH/FILL/DRESS/SHIP code (except the one-line import swap in `polish_validation.py`).
- No changes to authoritative specs unless a spec gap is found (per §Spec-gap policy).
- No new agents, LLM providers, CLI commands, or pipeline phases.
- No architectural refactor of GROW's phase-based orchestration.
- DREAM / BRAINSTORM / SEED validators and their stage code remain untouched (beyond the additive `skip_forbidden_types` kwarg on `validate_seed_output`).
