# DREAM + BRAINSTORM Compliance — Design

**Date:** 2026-04-19
**Status:** Approved — ready for implementation plan
**Authoritative specs being brought into compliance with:**
- `docs/design/procedures/dream.md`
- `docs/design/procedures/brainstorm.md`
- `docs/design/how-branching-stories-work.md`
- `docs/design/story-graph-ontology.md`

Plus CLAUDE.md §Design Doc Authority and §Silent Degradation.

## Problem

The 2026-04-19 spec-compliance audit (report at `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md`) surfaced 11 compliance gaps across DREAM (3 clusters, #1269–#1271) and BRAINSTORM (8 clusters, #1273–#1280), including one critical silent-degradation violation (#1273, dilemma `anchored_to` edges silently dropped). The audit also proposed two new helpers — `validate_dream_output` and `validate_brainstorm_output` — as the runtime oracles for each stage's Stage Output Contract (M-contract-chaining epic #1346, clusters #1347 entry and #1348 exit).

Per CLAUDE.md §Design Doc Authority, the authoritative procedure docs supersede code and tests. This project brings DREAM and BRAINSTORM into compliance with those docs and introduces the corresponding validators, using audit clusters as guidelines rather than the sole authoritative scope.

## Goal

Produce a DREAM pipeline stage and a BRAINSTORM pipeline stage that:
1. Close all 11 audit clusters against their authoritative specs.
2. Emit artifacts that pass a new `validate_<stage>_output(graph)` oracle at stage exit.
3. Verify the upstream stage's contract at entry (BRAINSTORM calls `validate_dream_output` before running).
4. Surface every violation loudly (no silent skips, no silent defaults).

No changes to SEED/GROW/POLISH/FILL/DRESS/SHIP. Downstream breakage caused by the new BRAINSTORM output contract is explicitly accepted — this project stops at BRAINSTORM's exit boundary.

## Scope

**In scope:**
- Close audit clusters #1269, #1270, #1271, #1273, #1274, #1275, #1276, #1277, #1278, #1279, #1280.
- New `validate_dream_output(graph) -> list[str]` helper in `src/questfoundry/graph/dream_validation.py`.
- New `validate_brainstorm_output(graph) -> list[str]` helper in `src/questfoundry/graph/brainstorm_validation.py`.
- Wire `validate_dream_output` at DREAM exit and at BRAINSTORM entry.
- Wire `validate_brainstorm_output` at BRAINSTORM exit.
- Closes DREAM+BRAINSTORM slices of #1347 and #1348 (4 of 12 seam-wirings total).
- Delete two obsolete tests that fail today:
  - `tests/unit/test_grow_deterministic.py::TestPhaseIntraPathPredecessors::test_dead_end_resolved_by_intra_path_edges`
  - `tests/unit/test_polish_validation.py::TestValidatePolishOutputResidue::test_proper_residue_passes`
- Drift discovered during implementation that blocks TDD: fix inline, note in PR body. Non-blocking drift: file a new cluster issue under #1268 / #1272 and defer.
- Prompt-template changes: fix inline if blocking, else file an issue under the owning epic. Not modified speculatively.

**Out of scope:**
- BRAINSTORM → SEED entry-side wiring of `validate_brainstorm_output` (SEED is out of scope; deferred to when SEED compliance work begins).
- SEED, GROW, POLISH, FILL, DRESS, SHIP compliance.
- `test_provider_factory.py::test_create_chat_model_ollama_success` — passes in isolation, fails only via test pollution. Not in scope; will get its own investigation.
- Runtime-verification (uncheckable) rules from the audit — those remain deferred to a future LLM-live track.
- Integration / e2e tests. Existing ones may break from the new contract enforcement; that is allowed per the "post-BRAINSTORM breakage is acceptable" scope decision.
- Downstream SEED tests that break because the BRAINSTORM Stage Output Contract now rejects artifacts they relied on.

## Architecture

The existing `graph/polish_validation.py` is the reference pattern for this work:
- `validate_grow_output(graph) -> list[str]` is called at POLISH entry (`polish/stage.py:239`) and raises `PolishStageError` on non-empty result.
- `validate_polish_output(graph) -> list[str]` is called at POLISH exit (`polish/deterministic.py:1395-1397`).

DREAM and BRAINSTORM follow the same shape.

**Function contract (identical for both validators):**

```python
def validate_<stage>_output(graph: Graph) -> list[str]:
    """Check graph satisfies <stage>'s Stage Output Contract.

    Returns a list of human-readable error strings; empty list means compliant.
    Pure read-only — never mutates the graph.
    """
```

**Error-type discipline:**

- Validators return strings and never raise.
- Stage code that calls a validator is responsible for raising the stage-local error type if the list is non-empty; it joins errors with newlines for the message.
- Upstream-contract failure at entry uses the downstream stage's error type (e.g., `BrainstormStageError` on DREAM contract failure) so that the failure localizes to the stage that was about to start.
- Own-contract failure at exit uses the same stage's error type.

**Silent-skip rewrites:**

- Sites currently using `except ValueError: pass` or similar to absorb reference-resolution failures (e.g. `graph/mutations.py:905` for `anchored_to` edge creation) are rewritten to either raise or surface the issue via the stage-exit validator. Nothing may be silently dropped from structural operations.

**What is deliberately NOT changed:**

- Prompt templates for DREAM / BRAINSTORM, unless a template change is required to make the producer emit output the validator accepts (e.g. if the model must be told about a new enum).
- The shape of the three-phase (Discuss → Summarize → Serialize) pipeline; only the exit/entry hooks are added.
- How the `last_stage` sentinel works; the new validators are additive, not replacement.

## Components

### New files

| Path | Responsibility |
|---|---|
| `src/questfoundry/graph/dream_validation.py` | `validate_dream_output(graph)` — enforces DREAM Stage Output Contract. |
| `src/questfoundry/graph/brainstorm_validation.py` | `validate_brainstorm_output(graph)` — enforces BRAINSTORM Stage Output Contract. |
| `tests/unit/test_dream_validation.py` | One test per DREAM Stage Output Contract rule plus a compliant-baseline test. |
| `tests/unit/test_brainstorm_validation.py` | One test per BRAINSTORM Stage Output Contract rule plus a compliant-baseline test. |

### Modified files

| Path | Change |
|---|---|
| `src/questfoundry/pipeline/stages/dream.py` | Add exit-validator call before `set_last_stage("dream")`; raise `DreamStageError` on failure. |
| `src/questfoundry/pipeline/stages/brainstorm.py` | Replace / augment the existing "vision node exists" check with `validate_dream_output(graph)` at entry; add `validate_brainstorm_output(graph)` at exit; raise `BrainstormStageError` on failure. |
| `src/questfoundry/models/dream.py` (and related) | Fix #1269 (POV style enum values) and #1270 (scope preset names) to match `dream.md`. |
| `src/questfoundry/models/brainstorm.py` (and related) | Fix #1275 (question punctuation), #1276 (entity `name` allows None), #1277 (dilemma ID prefix), #1278 (defensive node-type check), #1279 (discussion abundance), #1280 (vision compatibility). |
| `src/questfoundry/graph/mutations.py` | Fix #1273 — replace the silent skip around `anchored_to` edge creation (approx. line 905) so the failure is either raised immediately or captured by the exit validator. Touches #1274 semantics if location-min-count enforcement lives here. |
| `tests/unit/test_dream_stage.py` | Update tests whose assertions encode pre-audit DREAM behavior (POV values, scope names). |
| `tests/unit/test_brainstorm_stage.py` | Update tests whose assertions encode pre-audit BRAINSTORM behavior. |

### Deleted tests

- `tests/unit/test_grow_deterministic.py::TestPhaseIntraPathPredecessors::test_dead_end_resolved_by_intra_path_edges` — depends on GROW/POLISH behavior that is itself non-compliant; will be rewritten when M-GROW-spec / M-POLISH-spec work begins.
- `tests/unit/test_polish_validation.py::TestValidatePolishOutputResidue::test_proper_residue_passes` — same rationale.

### Open ownership question (resolved during plan-writing)

Cluster #1274 (location entity minimum count) could live in `models/brainstorm.py` (Pydantic validator) or `graph/mutations.py` (semantic validator). The implementation plan step that implements #1274 will read the spec and the surrounding code to choose; this design does not prescribe.

## Work sequence (TDD order)

Phase-by-phase. One commit per numbered step unless noted.

**Phase A — Cleanup.**
1. Delete the two obsolete tests listed above.

**Phase B — DREAM validator.**
2. Add `tests/unit/test_dream_validation.py` with one test per DREAM Stage Output Contract rule (failing: violations must produce the expected error string) plus one compliant-baseline test (passing: empty list).
3. Add `src/questfoundry/graph/dream_validation.py` with a stub that returns `[]`. Confirm compliant-baseline test passes and all rule tests fail.
4. Implement rules incrementally until all DREAM validator tests pass.

**Phase C — BRAINSTORM validator.**
5. Add `tests/unit/test_brainstorm_validation.py` with one test per BRAINSTORM Stage Output Contract rule plus a compliant-baseline test.
6. Add `src/questfoundry/graph/brainstorm_validation.py` with a stub that returns `[]`.
7. Implement rules incrementally until all BRAINSTORM validator tests pass.

**Phase D — Wire validators.**
8. Wire `validate_dream_output` at DREAM exit in `pipeline/stages/dream.py`.
9. Wire `validate_dream_output` at BRAINSTORM entry in `pipeline/stages/brainstorm.py`, replacing the narrow "vision node exists" check.
10. Wire `validate_brainstorm_output` at BRAINSTORM exit.

After phase D, existing DREAM + BRAINSTORM tests may go red — expected, the producers still violate rules.

**Phase E — Fix DREAM producers (one cluster per commit).**
11. #1269 — POV style enum values.
12. #1270 — scope preset names.
13. #1271 — human approval gate (record approval on the Vision artifact for the `--no-interactive` path).

**Phase F — Fix BRAINSTORM producers.**
14. #1273 — dilemma-entity anchoring silent-skip (priority 1; silent-degradation critical).
15. #1274 — location entity minimum count.
16. #1275 — question punctuation validation.
17. #1276 — entity `name` field no longer allows None.
18. #1277 — dilemma ID prefix validated at model level.
19. #1278 — defensive check for brainstorm-only node types.
20. #1279 — discussion abundance target.
21. #1280 — vision compatibility enforced at serialize.

**Phase G — Close out.**
22. PR body assembled with: `Closes #1269 … #1280`, inline-drift notes, deferred-drift issue numbers, impacted-test summary.
23. Push branch, open PR, respond to AI-bot reviews.

Expected total commits: ~20 (phases A–F) plus review-iteration commits.

## Testing strategy

### Validator tests

- File naming: `tests/unit/test_dream_validation.py`, `tests/unit/test_brainstorm_validation.py`.
- Test naming: one test per rule, named `test_<rule_id>_<short_description>` — e.g. `test_R_3_6_dilemma_missing_anchored_to_edge`.
- Each rule test builds a minimal graph that violates exactly that rule and asserts the rule's error string appears in the return value. Avoid compound violations; we want rule-to-error traceability.
- Shared `compliant_graph` fixture builds a complete valid stage output; the single `test_valid_graph_passes` test uses it and asserts `== []`.
- Use `pytest.mark.parametrize` for rule families with uniform structure (e.g. "required field missing" across entity fields).

### Producer tests

- `tests/unit/test_dream_stage.py` and `tests/unit/test_brainstorm_stage.py` retain their current structure. Update assertions that encode pre-audit behavior.
- Where a cluster fix adds new semantic behavior, add targeted tests alongside the existing ones.
- Add one end-to-end test per stage that runs producer → validator and asserts the validator returns `[]`. Catches regressions that narrower tests miss.

### Coverage

- Target: 85% for the two new validator modules (matches CLAUDE.md's 85% for new code).
- No coverage target imposed on modified code.

### Not tested here

- LLM prompt quality or diegetic-voice concerns. The 5+1 uncheckable rules from the audit remain uncheckable.
- Actual LLM calls — mocks only.
- Downstream stage integration — SEED+ is out of scope; integration tests are allowed to break.

## Error handling

- Validators return `list[str]`. Never raise. Never log.
- Stage callers: if the list is non-empty, log once at ERROR with a structured event name (`dream_contract_failed`, `brainstorm_contract_failed`, `upstream_dream_contract_failed`) and raise the stage's error type with the errors joined by newlines.
- Silent-skip sites rewritten per §Architecture. Every structural failure must either raise at the site of detection or be captured by the stage-exit validator. No `except … pass`, no fallback-to-empty-output.

## Spec-gap policy

Per CLAUDE.md §Instruction Hierarchy. If during implementation a spec rule is silent, ambiguous, or self-contradictory:

1. Stop. Do not guess.
2. Raise the question in the current session.
3. If the user aligns on the intended behavior: update the spec in a dedicated commit (`docs(spec): clarify …`) before touching code.
4. Then update code and tests to match the updated spec.
5. Never flip this order. Never "document what broken code happens to do."

The audit clusters are implementation guidelines, not the authoritative spec. If a cluster description conflicts with the actual text of `dream.md` / `brainstorm.md`, the spec wins; the PR body calls out the clarification.

## Exit criteria

1. All 11 cluster issues closed via `Closes #…` in the PR body: #1269, #1270, #1271, #1273, #1274, #1275, #1276, #1277, #1278, #1279, #1280.
2. `validate_dream_output` and `validate_brainstorm_output` exist; validator test files cover every rule in each Stage Output Contract with one named test each.
3. DREAM exit, BRAINSTORM entry, BRAINSTORM exit each call the appropriate validator.
4. `uv run pytest tests/unit/test_dream*.py tests/unit/test_brainstorm*.py` — 0 failures.
5. `uv run pytest tests/unit/ -k "not seed and not grow and not polish and not fill and not dress and not ship"` — 0 failures.
6. `uv run mypy src/questfoundry/graph/dream_validation.py src/questfoundry/graph/brainstorm_validation.py` — clean.
7. `uv run ruff check` — clean on all modified files.
8. Post-BRAINSTORM breakage (SEED+ unit tests, integration / e2e tests) is allowed and not a blocker.
9. PR body lists cluster closures, inline-fixed drift, and deferred drift with new issue numbers.

## What does not change

- No changes to SEED/GROW/POLISH/FILL/DRESS/SHIP code.
- No changes to authoritative specs unless a spec gap is found (per §Spec-gap policy).
- No new agents, LLM providers, CLI commands, or pipeline phases.
- No architectural refactor of the three-phase Discuss/Summarize/Serialize pipeline.
