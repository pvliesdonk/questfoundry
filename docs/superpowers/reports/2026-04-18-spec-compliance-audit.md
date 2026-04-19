# Spec-Compliance Audit Report

**Date:** 2026-04-18 (audit begins)
**Spec:** `docs/superpowers/specs/2026-04-18-spec-compliance-audit-design.md`
**Plan:** `docs/superpowers/plans/2026-04-19-spec-compliance-audit.md`

This report records every gap found between the authoritative design
specs and current code + tests. Per CLAUDE.md §Design Doc Authority,
specs supersede code and tests; findings below document where code/tests
diverge from the spec and need to be brought into compliance.

Each section corresponds to one milestone. Cluster issues filed in GitHub
reference the section anchors in this document.

---

## Status

| Milestone | Status | Clusters | Epic | Filed |
|---|---|---|---|---|
| M-DREAM-spec | drafted | 3 | — | — |
| M-BRAINSTORM-spec | pending | — | — | — |
| M-SEED-spec | pending | — | — | — |
| M-GROW-spec | pending | — | — | — |
| M-POLISH-spec | pending | — | — | — |
| M-FILL-spec | pending | — | — | — |
| M-DRESS-spec | pending | — | — | — |
| M-SHIP-spec | pending | — | — | — |
| M-logging-compliance | pending | — | — | — |
| M-silent-degradation | pending | — | — | — |
| M-contract-chaining | pending | — | — | — |

---

## Runtime-verification deferred list

Rules that can only be verified against a live LLM run (prompt quality,
diegetic voice, etc.). These accumulate here and are NOT filed as
compliance issues — they are a follow-on track.

(none yet)

---

## M-DREAM-spec

### Summary
- Rules checked: 13
- Compliant: 4 | Drift: 2 | Missing: 2 | Uncheckable: 5

### Cluster: POV Style Enum Values Don't Match Spec

**Rules covered:** R-1.9

**Current state:** The Pydantic model defines `pov_style` with values `"first"`, `"second"`, `"third_limited"`, `"third_omniscient"`, stripped of the `_person` suffix.

**Gap:** The authoritative spec (dream.md, line 75) requires the values to be `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient` — with the `_person` suffix included. The implementation omits this suffix, causing a mismatch between the spec and code.

**Recommended fix:** Update the Pydantic model's `pov_style` Literal to include the `_person` suffix on all four values to match the authoritative spec exactly.

**Code refs:** `src/questfoundry/models/dream.py:63`, `src/questfoundry/prompts/templates/dream.yaml:25-27`

**Test refs:** `tests/unit/test_dream_stage.py` (no tests validate pov_style values against spec)

### Cluster: Scope Preset Names Diverge from Spec Examples

**Rules covered:** R-1.4

**Current state:** The Scope model defines `story_size` as one of `["vignette", "short", "standard", "long"]`.

**Gap:** The authoritative spec (dream.md, line 51) states scope presets should be "e.g., `micro`, `short`, `medium`, `long`, or their equivalent implementation values." The spec examples include `micro` and `medium`, but the code uses `vignette` and `standard` instead. While the spec does say "or their equivalent implementation values," using entirely different names (`vignette` vs. `micro`, `standard` vs. `medium`) creates ambiguity about whether these are truly equivalent or a silent divergence.

**Recommended fix:** Either update the scope preset names to match the spec examples (`micro`, `short`, `medium`, `long`), or update the spec to formally document the equivalence (`micro` ↔ `vignette`, `medium` ↔ `standard`) with clear justification. If keeping the current names, add a comment to the model explaining the mapping.

**Code refs:** `src/questfoundry/models/dream.py:34`, `src/questfoundry/prompts/templates/dream.yaml:16`

**Test refs:** `tests/unit/test_dream_stage.py` (tests use `standard` directly), `tests/integration/test_dream_pipeline.py:59` (tests validate artifact structure but not scope values against spec)

### Cluster: Missing Human Approval Gate for Vision Node

**Rules covered:** R-1.12, R-1.13

**Current state:** The DREAM stage executes (discuss → summarize → serialize) and writes the Vision node to the graph via `apply_dream_mutations()`, but there is no explicit human approval step before the stage marks itself complete. The CLI command `qf dream` runs the stage and saves the artifact without waiting for user confirmation. There is no rejection loop mechanism for the user to request changes and loop back.

**Gap:** The authoritative spec (dream.md, R-1.12 and R-1.13) requires DREAM to be "not complete until the human explicitly approves the Vision node" and rejection must "loop back to the operation that contains the misalignment." The current implementation skips both. The Vision node is upserted to the graph immediately after serialization with no approval gate. If the user is unsatisfied, there is no in-pipeline mechanism to request changes and re-run a specific operation (spark exploration, constraint definition, or synthesis).

**Recommended fix:** Introduce an approval gate after the serialize phase: display the serialized Vision to the user and require an explicit approve/reject response before marking DREAM complete. On rejection, support user indication of which operation failed (spark exploration, constraint definition, or synthesis) and re-enter that operation rather than re-serializing. This may require changes to the orchestrator's phase-progression model and the CLI's interactive flow.

**Code refs:** `src/questfoundry/pipeline/stages/dream.py:170-191` (no approval check), `src/questfoundry/graph/mutations.py:561-591` (upsert without approval), `src/questfoundry/cli.py:784-869` (no approval prompt)

**Test refs:** `tests/unit/test_dream_stage.py` (no tests for approval gate), `tests/integration/test_dream_pipeline.py` (no approval simulation)

### Uncheckable rules

- R-1.1: Discussion produces a single coherent vision — requires live LLM dialogue to verify convergence (is one vision picked, or does the LLM offer a menu?)
- R-1.3: Themes are abstract ideas, not plot points — requires semantic theme validation beyond schema constraints
- R-1.5: Content notes shape creative direction — BRAINSTORM enforcement is external to DREAM; DREAM stores notes correctly but validation of downstream respect happens in BRAINSTORM stage
- R-1.6: Constraints are firm — enforcement depends on BRAINSTORM respecting content notes; DREAM has no mechanism to validate this
- R-1.11: Synthesis captures only what was discussed — requires monitoring LLM behavior against discussion context

---

<!-- Milestone sections appended below as each audit stage completes. -->
