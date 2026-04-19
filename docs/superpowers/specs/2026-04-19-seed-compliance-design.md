# SEED Compliance — Design

**Date:** 2026-04-19
**Status:** Approved — ready for implementation plan
**Authoritative specs being brought into compliance with:**
- `docs/design/procedures/seed.md`
- `docs/design/how-branching-stories-work.md`
- `docs/design/story-graph-ontology.md` (Part 8 Y-shape guard rails in particular)

Plus CLAUDE.md §Design Doc Authority, §Silent Degradation, and §Logging.

## Problem

The 2026-04-19 spec-compliance audit (report at `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` §M-SEED-spec) surfaced 14 compliance gaps across SEED (epic #1281, clusters #1282–#1295). SEED is the stage that builds the Y-shape beat scaffold every downstream stage depends on; structural correctness here is a prerequisite for POLISH, FILL, DRESS, and SHIP compliance. The audit flagged four clusters as hot-path priority (#1282 Y-shape shared beat enforcement, #1283 cross-dilemma `belongs_to` prohibition, #1286 convergence LLM failure silent degradation, #1287 dilemma ordering LLM failure silent degradation) and one missing approval gate (#1295 Path Freeze).

Per CLAUDE.md §Design Doc Authority, the authoritative procedure doc supersedes code and tests. This project brings SEED into compliance with `seed.md`, introduces `validate_seed_output` as the runtime oracle for the Stage Output Contract, and removes SEED's file-level pyright suppression (tagged `TODO(#1281)` in commit `425eebb5`).

## Goal

Produce a SEED pipeline stage that:
1. Closes all 14 audit clusters against the authoritative spec.
2. Emits artifacts that pass a new `validate_seed_output(graph)` oracle at stage exit.
3. Surfaces every structural violation loudly (no silent skips, no silent defaults).
4. Leaves `models/seed.py` passing pyright standard mode without suppressions.

No changes to GROW/POLISH/FILL/DRESS/SHIP. Downstream breakage caused by the new SEED output contract is explicitly accepted — this project stops at SEED's exit boundary. The entry-side wiring of `validate_seed_output` at GROW start is deferred to the GROW compliance PR, mirroring the DREAM→BRAINSTORM split in PR #1351.

## Scope

**In scope:**
- Close audit clusters: #1282, #1283, #1284, #1285, #1286, #1287, #1288, #1289, #1290, #1291, #1292, #1293, #1294, #1295.
- New `validate_seed_output(graph) -> list[str]` helper in `src/questfoundry/graph/seed_validation.py` + `SeedContractError`.
- Wire `validate_seed_output` at SEED exit via `apply_seed_mutations` in `src/questfoundry/graph/mutations.py`. Closes SEED's slice of #1348.
- Remove the `# pyright: reportInvalidTypeForm=false` + `TODO(#1281)` suppression on `src/questfoundry/models/seed.py`.
- Test updates per the rewrite-or-delete policy: rewrite assertions that encode pre-audit behavior where the spec intent is similar; delete tests whose entire premise is pre-audit.
- Drift discovered during implementation: blocking drift fixed inline and noted in PR body; non-blocking drift filed as a new issue under #1281 and deferred.
- Prompt template changes: only where a cluster specifically requires a producer-level change to emit compliant output.

**Out of scope:**
- GROW entry-side wiring of `validate_seed_output` — deferred to GROW compliance (epic #1296).
- GROW, POLISH, FILL, DRESS, SHIP compliance.
- Runtime-verification of the 1 uncheckable rule in SEED — stays in the long-term deferred list.
- `test_provider_factory::test_create_chat_model_ollama_success` — pre-existing test-pollution; unchanged.
- Downstream SEED→GROW transition tests that break because the tightened Stage Output Contract rejects fixtures GROW relied on.
- Integration / e2e tests. Existing ones may break from the new contract enforcement; that is accepted.

## Architecture

Reference pattern: `src/questfoundry/graph/polish_validation.py` (`validate_grow_output` / `validate_polish_output`), and the DREAM/BRAINSTORM validators introduced in PR #1351 (`dream_validation.py`, `brainstorm_validation.py`). SEED mirrors their shape.

**Function contract:**

```python
def validate_seed_output(graph: Graph) -> list[str]:
    """Check graph satisfies SEED's Stage Output Contract.

    Returns a list of human-readable error strings; empty list means compliant.
    Pure read-only — never mutates the graph.
    """
```

**Decomposition into private helpers** (matches BRAINSTORM validator's decomposition pattern after PR #1351's review round):

- `_check_upstream_contract(graph, errors)` — delegates to `validate_brainstorm_output(graph)`; reports violations with `"Output-N: BRAINSTORM contract violated post-SEED — …"` prefix. Guards against SEED silently corrupting upstream state.
- `_check_paths(graph, errors)` — path-node structure, path-to-dilemma linking, canonical marking, path importance.
- `_check_beats(graph, errors)` — beat node structure, role set (`setup`, `epilogue`, `commit_beat`, `pre_commit`, `post_commit`, `transition_beat`, `gap_beat`), `entities` for narrative beats, consequence ripples.
- `_check_belongs_to_yshape(graph, errors)` — Y-shape guard rails per Story Graph Ontology Part 8: multi-`belongs_to` only on pre-commit beats of the same dilemma; no cross-dilemma `belongs_to` edges; commit and post-commit beats have single `belongs_to`. This helper owns the #1282 and #1283 structural invariants.
- `_check_convergence_and_ordering(graph, errors)` — dilemma role (`hard` / `soft`), convergence analysis artifacts, ordering relationships including concurrent normalization and `shared_entity` derivation.
- `_check_state_flags_and_consequences(graph, errors)` — state flag derivation from consequences, one `derived_from` edge per state flag.
- `_check_approval_and_forbidden_nodes(graph, errors)` — Path Freeze approval recorded; forbidden node types (e.g., passages — those are POLISH's).

**Error-type discipline:**

- Validator returns `list[str]`; never raises; never logs.
- Stage code calling it: if the list is non-empty, emit `log.error("seed_contract_violated", errors=errors)` and raise `SeedContractError(...)` with the errors joined. Matches the `apply_dream_mutations` / `apply_brainstorm_mutations` pattern in PR #1351.
- `SeedContractError(ValueError)` lives in `seed_validation.py`. `SeedMutationError(MutationError)` already exists in `mutations.py`; keep both — contract errors (post-write) vs mutation errors (reference resolution, duplicate IDs) remain distinct.

**Silent-skip rewrites:**

- #1286 convergence LLM failure (currently silent default): on failure, either raise at call site or surface the missing artifact via the exit validator. No `try/except/pass`. No fallback-to-empty.
- #1287 dilemma ordering LLM failure: same treatment. `serialize_dilemma_relationships` (or whichever call produces the artifact) no longer returns silently on failure.
- #1285 `explored` field immutability: runtime guard via model validator so pruning can't mutate it without being caught.

**Wiring:**

- `apply_seed_mutations` in `src/questfoundry/graph/mutations.py` — at end of function body, after all node/edge creation, before return: call `validate_seed_output(graph)`; log and raise on non-empty.
- No GROW entry-side wiring (out of scope).

**What is deliberately NOT changed:**

- SEED's three-phase Discuss → Summarize → Serialize structure.
- Prompt templates, unless a cluster fix requires a producer-level change to emit compliant output.
- The orchestrator's `apply_mutations(...)` dispatch flow — it already calls `apply_seed_mutations` via the stage dispatch table.
- The existing `SeedMutationError` hierarchy — `SeedContractError` is added alongside.

## Components

### New files

| Path | Responsibility |
|---|---|
| `src/questfoundry/graph/seed_validation.py` | `SeedContractError` + `validate_seed_output` + 7 `_check_*` helpers. |
| `tests/unit/test_seed_validation.py` | One test per Stage Output Contract rule + Y-shape guard rail + silent-degradation invariant. Compliant-baseline fixture plus parametrized negatives. |

### Modified files

| Path | Change |
|---|---|
| `src/questfoundry/graph/mutations.py` | `apply_seed_mutations` calls `validate_seed_output` at exit; raises `SeedContractError` with structured log event. Fixes for #1286 and #1287 silent-degradation violations. |
| `src/questfoundry/models/seed.py` | Remove the `# pyright: reportInvalidTypeForm=false` + `TODO(#1281)` suppression once standard-mode passes. Address #1285 `explored` immutability, #1288 dilemma role "flavor" deprecation, #1292 beat `entities` validation, #1293 `path_importance` spec-vs-code mismatch, #1294 consequence ripples. |
| `src/questfoundry/pipeline/stages/seed.py` | #1295 Path Freeze approval gate: record approval on the seed artifact; `--no-interactive` pre-approves; minimal interactive yes/no prompt; full loop-back UX deferred to a follow-up issue (mirrors DREAM #1271 + follow-up #1350 pattern). |
| `src/questfoundry/agents/serialize.py` | Convergence / dilemma-ordering call sites: raise on LLM failure instead of silent default (supports #1286 and #1287). |
| `tests/unit/test_seed_models.py` | Update fixtures for tightened model validators. |
| `tests/unit/test_seed_stage.py` | Update fixtures for approval gate + tightened output contract. |
| `tests/unit/test_mutations.py` | SEED mutation tests: rewrite or delete fixtures per the rewrite-or-delete policy. |
| `tests/unit/test_serialize.py` | Convergence / dilemma-ordering tests assert raise-behavior on LLM failure, not silent-default behavior. |

### Potentially modified — owner resolved during plan writing

- #1289 concurrent ordering normalization post-check: likely `graph/mutations.py` ordering-edge creation, possibly `agents/serialize.py`. Implementation plan resolves.
- #1290 `shared_entity` derivation guard: likely `graph/mutations.py` in the beat/entity apply path.
- #1291 arc-count guardrail through Phase 7/8: likely `models/seed.py` or SEED stage-level invariant. Resolve during plan.

### Deleted files

- None anticipated. Test files may have individual tests deleted under the rewrite-or-delete policy; whole-file deletions are not planned.

### Not modified

- GROW, POLISH, FILL, DRESS, SHIP stage code.
- SEED stage three-phase structure.
- Prompt templates unless a specific cluster demands it.

## Work sequence (TDD order)

Approach 1 (validator-first). One commit per numbered step. Targets ~28 commits total — comparable to DREAM+BRAINSTORM (PR #1351 landed with 27).

**Phase A — Baseline.**
1. Confirm non-downstream suite green modulo the pre-existing `test_provider_factory` pollution. Delete any SEED tests already failing on main with pre-audit premises (expected: none).

**Phase B — SEED validator tests.**
2. Create `tests/unit/test_seed_validation.py` with compliant-baseline fixture plus one test per rule in SEED's Stage Output Contract, per Y-shape guard rail, and per silent-degradation invariant. All fail at collection with `ModuleNotFoundError`. Commit.

**Phase C — SEED validator implementation (one check-helper per commit).**
3. Skeleton `src/questfoundry/graph/seed_validation.py` returning `[]`. Compliant-baseline passes.
4. Implement `_check_upstream_contract` (delegates to `validate_brainstorm_output`).
5. Implement `_check_paths`.
6. Implement `_check_beats`.
7. Implement `_check_belongs_to_yshape` — hot-path Y-shape invariants (owns #1282 / #1283 validator side).
8. Implement `_check_convergence_and_ordering`.
9. Implement `_check_state_flags_and_consequences`.
10. Implement `_check_approval_and_forbidden_nodes`.

**Phase D — Wire validator.**
11. Wire `validate_seed_output` at `apply_seed_mutations` exit. Raises `SeedContractError` with structured log event. Existing SEED tests likely go red — expected TDD signal.

**Phase E — Critical / silent-degradation fixes (hot-path priority).**
12. #1282 — Y-shape shared beat enforcement at write time.
13. #1283 — Cross-dilemma `belongs_to` prohibition at write time.
14. #1286 — Convergence analysis LLM failure: raise-or-surface.
15. #1287 — Dilemma ordering LLM failure: raise-or-surface.
16. #1295 — Path Freeze approval gate (`human_approved_paths` or equivalent; `--no-interactive` pre-approves; minimal interactive prompt; file follow-up issue for full R-6.4 loop-back UX).

**Phase F — Moderate cluster fixes.**
17. #1284 — Setup/epilogue beat semantics.
18. #1285 — `explored` field immutability at model-validator level + runtime check.
19. #1288 — Dilemma role "flavor" deprecation complete.
20. #1289 — Concurrent ordering normalization post-check.
21. #1290 — `shared_entity` derivation guard.
22. #1291 — Arc count guardrail through Phase 7/8.
23. #1292 — Beat `entities` validated for narrative beats.
24. #1293 — `path_importance` field spec-vs-code mismatch (may require a spec-update commit first per the spec-gap policy).
25. #1294 — Consequence ripples validation.

**Phase G — Close-out.**
26. Remove `# pyright: reportInvalidTypeForm=false` + `TODO(#1281)` from `src/questfoundry/models/seed.py`. Verify `uv run pyright src/` still reports 0 errors.
27. PR body assembled with `Closes #1282 … #1295`, inline-drift notes, deferred-drift new issue numbers, impacted-test summary, and Path Freeze follow-up issue number.
28. Push; open PR; respond to AI-bot reviews.

## Testing strategy

### Validator tests

- File: `tests/unit/test_seed_validation.py`.
- Naming: `test_<rule_id>_<short_description>` — e.g. `test_R_3_6_yshape_shared_beat_enforcement`, `test_R_3_9_cross_dilemma_belongs_to_forbidden`, `test_R_7_5_convergence_llm_failure_surfaced`.
- Each rule test builds a minimal graph violating exactly that rule and asserts the error list contains a substring identifying it. Assertions use substring matching for resilience.
- Shared `compliant_graph` fixture builds a complete valid SEED graph (with a preceding compliant BRAINSTORM baseline for the upstream check).
- `pytest.mark.parametrize` for rule families with uniform structure.
- Single positive test `test_valid_graph_passes` asserts `validate_seed_output(compliant_graph) == []`.

### Producer tests

- `tests/unit/test_seed_models.py`, `tests/unit/test_seed_stage.py`, `tests/unit/test_mutations.py`, `tests/unit/test_serialize.py` retain their current structure.
- Rewrite assertions that encode pre-audit behavior. Delete tests whose entire premise is pre-audit.
- Add targeted tests for new behavior introduced by cluster fixes — notably the raise-behavior assertions for #1286 and #1287, and the approval-gate recording for #1295.
- One end-to-end pathway per applicable test file that runs `producer → apply_seed_mutations → validate_seed_output(graph) == []`.

### Integration tests

- Out of scope. Existing integration tests may break from tightened contract; accepted.

### Coverage target

- 85% for `seed_validation.py` (matches CLAUDE.md).
- No target imposed on modified code.

### What is NOT tested

- LLM prompt quality (the 1 uncheckable rule remains deferred).
- Live LLM-call behavior — mocks only.
- Downstream stage integration.

## Error handling

- Validator: pure, returns `list[str]`, never raises, never logs.
- Caller (`apply_seed_mutations`): structured log event before raise, per PR #1351 pattern.
- Silent-skip sites rewritten at source for #1285, #1286, #1287. No structural failure may be silently absorbed.
- `SeedContractError(ValueError)` in `seed_validation.py`; `SeedMutationError` in `mutations.py` retained for mutation-time failures.

## Spec-gap policy

Per CLAUDE.md §Instruction Hierarchy. If a spec rule is silent, ambiguous, or self-contradictory during implementation:

1. Stop. Do not guess.
2. Raise the question in the current session.
3. On alignment: update the spec in a dedicated `docs(spec): clarify …` commit before any code change.
4. Then update code and tests to match.
5. Never flip the order.

#1293 `path_importance` is a likely spec-vs-code mismatch case. The audit flagged "field not defined in spec" — implementation will surface which side is wrong and take the spec-first path if the spec needs updating.

Audit clusters are implementation guidelines, not the authoritative spec. Where a cluster description conflicts with the actual text of `seed.md`, the spec wins and the PR body calls out the clarification.

## Exit criteria

1. All 14 cluster issues closed via `Closes #…` in PR body: #1282–#1295.
2. `validate_seed_output` exists; `tests/unit/test_seed_validation.py` covers every rule in SEED's Stage Output Contract, every Y-shape guard rail, and every silent-degradation invariant with a named test.
3. SEED exit (via `apply_seed_mutations`) calls the validator.
4. `uv run pytest tests/unit/test_seed*.py` — 0 failures.
5. `uv run pytest tests/unit/ -k "not grow and not polish and not fill and not dress and not ship"` — 0 failures, modulo the pre-existing `test_provider_factory` pollution (unchanged).
6. `uv run mypy src/questfoundry/graph/seed_validation.py` — clean.
7. `uv run ruff check` — clean on all modified files.
8. `uv run pyright src/` — 0 errors. The `models/seed.py` suppression is removed and the file passes standard mode cleanly.
9. Downstream breakage (GROW+ unit tests, integration / e2e tests) is allowed and not a blocker.
10. PR body lists cluster closures, inline-fixed drift, deferred drift with new issue numbers, and the Path Freeze follow-up issue number.

## What does not change

- No changes to GROW/POLISH/FILL/DRESS/SHIP code.
- No changes to authoritative specs unless a spec gap is found (per §Spec-gap policy).
- No new agents, LLM providers, CLI commands, or pipeline phases.
- No architectural refactor of SEED's three-phase structure.
- The DREAM / BRAINSTORM validators and their suppressions are untouched — those stages stayed compliant after PR #1351.
