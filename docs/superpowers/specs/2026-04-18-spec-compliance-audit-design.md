# Spec-Compliance Audit — Design

**Date:** 2026-04-18
**Status:** Approved — ready for implementation plan
**Authoritative specs being audited against:**
- `docs/design/how-branching-stories-work.md`
- `docs/design/story-graph-ontology.md`
- `docs/design/procedures/*.md` (all 8 stages)

Plus the logging-level policy in `CLAUDE.md` §Logging.

---

## Problem

The authoritative design specs were just finalized across PRs #1263, #1264, #1265, #1266. The codebase and tests were written earlier — in many places by an LLM with a wrong mental model — and are expected to have significant drift, particularly in SEED, GROW, and POLISH. Per CLAUDE.md §Design Doc Authority, specs supersede code and tests: where the two conflict, the code/tests are wrong. We need to systematically identify every gap so it can be tracked and fixed.

The goal of this design is **an audit plan**, not the audit itself and not the fixes. The audit plan produces:

1. A living report that enumerates findings
2. GitHub epic issues per milestone
3. GitHub cluster issues under each epic

The actual fixes happen later, milestone by milestone, each with its own TDD implementation plan.

## Goal

A systematic audit that identifies every gap between the authoritative specs and current code + tests, across all 8 pipeline stages plus three cross-cutting concerns (logging, silent degradation, contract chaining). Findings are packaged as GitHub issues grouped by milestone so they can be tracked and addressed as coherent units of work.

## Scope

**In scope:**
- All 8 stages: DREAM, BRAINSTORM, SEED, GROW, POLISH, FILL, DRESS, SHIP.
- Every numbered rule in every procedure doc's Rule Index (exhaustive, not priority-filtered).
- Code and tests, audited together per stage.
- Logging-level compliance across all stages against CLAUDE.md §Logging.
- Silent Degradation policy compliance across all stages.
- Stage Input / Output Contract chaining between adjacent stages.

**Out of scope:**
- Any code change. The audit identifies gaps only.
- Any spec change. If the audit surfaces a spec gap, it is flagged `spec-gap` in the report and becomes a separate task; the audit does not cross the spec-update boundary mid-flight.
- Runtime verification that requires live LLM calls. Rules that can only be verified this way are marked "uncheckable" and deferred to a follow-up track.
- TDD implementation plans for the fixes. Those are written later, one per milestone.

## Architecture

The audit produces three kinds of artifact:

1. **Audit report** — one living markdown document at `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md`, organized by milestone. Each milestone section lists the semantic clusters of findings; each cluster lists the spec rules it covers, the current-code state, the gap, a recommended fix, and code/test references.
2. **Epic issues** — one GitHub issue per milestone, whose body references the report's milestone section and enumerates child cluster issues.
3. **Cluster issues** — one GitHub issue per semantic cluster, whose body is short (rules covered, report section link, recommended fix, touch points). Tagged with the appropriate milestone and labels.

### Milestones (11 total)

| Milestone | Scope | Kind |
|---|---|---|
| M-DREAM-spec | DREAM code + tests vs `procedures/dream.md` | Stage |
| M-BRAINSTORM-spec | BRAINSTORM code + tests vs `procedures/brainstorm.md` | Stage |
| M-SEED-spec | SEED code + tests vs `procedures/seed.md` (expected large) | Stage |
| M-GROW-spec | GROW code + tests vs `procedures/grow.md` (expected large) | Stage |
| M-POLISH-spec | POLISH code + tests vs `procedures/polish.md` (expected large) | Stage |
| M-FILL-spec | FILL code + tests vs `procedures/fill.md` | Stage |
| M-DRESS-spec | DRESS code + tests vs `procedures/dress.md` | Stage |
| M-SHIP-spec | SHIP code + tests vs `procedures/ship.md` | Stage |
| M-logging-compliance | CLAUDE.md logging-level policy across all stages | Cross-cutting |
| M-silent-degradation | Silent Degradation policy across all stages | Cross-cutting |
| M-contract-chaining | Stage N input ≡ Stage N-1 output enforcement at seams | Cross-cutting |

## Audit methodology (per stage)

Each stage audit follows a fixed unit of work:

1. **Dispatch a stage-audit subagent** with:
   - The procedure doc path
   - Relevant narrative and ontology references
   - Code entry points (e.g., `src/questfoundry/pipeline/stages/<stage>/`, `src/questfoundry/graph/<stage>_*.py`, shared `src/questfoundry/models/`)
   - Test entry points (e.g., `tests/unit/test_<stage>_*.py`, `tests/integration/test_<stage>_e2e.py`)
   - Instruction: walk the Rule Index exhaustively; for each rule locate enforcing code + test; classify **compliant / drift / missing / uncheckable**; group findings into semantic clusters; output a report section matching the fixed template below.

2. **Subagent output template** (identical for all 8 stages):

   ```markdown
   ## M-<STAGE>-spec

   ### Summary
   - Rules checked: N
   - Compliant: X | Drift: Y | Missing: Z | Uncheckable: W

   ### Cluster: <human-readable name>
   **Rules covered:** R-1.1, R-1.2, …
   **Current state:** <what the code does today>
   **Gap:** <what's missing or wrong>
   **Recommended fix:** <approach, not full implementation plan>
   **Code refs:** `path/to/file.py:lines`, …
   **Test refs:** `tests/path/test_file.py`, …

   ### Cluster: …
   ```

3. **Review by main agent (me)**: resolve ambiguity where a rule's code coverage is unclear; reconcile overlapping clusters; merge the subagent's section into the master report.

4. **File issues**: one epic for the milestone; one child cluster issue per cluster. Issue bodies are short and reference the report section.

5. **User checkpoint**: after each stage's merged section is ready, user reviews and confirms before the next stage begins.

### Sequencing

Strict pipeline order for stage audits: DREAM → BRAINSTORM → SEED → GROW → POLISH → FILL → DRESS → SHIP. Each stage is one checkpoint.

After all 8 stage audits: the three cross-cutting passes, in order: logging-compliance, silent-degradation, contract-chaining. Each cross-cutting pass is one checkpoint.

Final: a summary section in the report tallying findings per milestone, noting priority hotspots, and listing epic/issue URLs.

### Uncheckable rules

Rules that describe LLM prompt quality or behavior that only a live run can verify (e.g., "the LLM generates diegetic captions") are flagged `uncheckable` in the report. They are not bundled into this audit's findings; they accumulate into a single "runtime-verification" deferred list at the end of the report.

## Cross-cutting methodology

**M-logging-compliance.** Walk every stage's code and shared utilities against CLAUDE.md §Logging. Litmus test: *if the system detected a problem AND handled it correctly, that's `INFO` or `DEBUG` — not `WARNING`.*

Cluster by misuse pattern:
- Warnings used for successful fallbacks
- Info used for per-beat filtering noise (should be debug)
- Errors swallowed at log level instead of raised
- Missing logs at phase/stage transitions

Findings include stage-local code references so they can be cross-linked back to the stage milestones where appropriate.

**M-silent-degradation.** Walk every stage for violations of the Silent Degradation policy. Known signatures to search for:
- `interleave_cycle_skipped`
- "all intersections rejected" outcomes
- `None`-returning error paths that should raise
- `try/except/pass` around structural operations
- Fallback-to-empty-output patterns
- LLM failures that apply defaults silently (no WARNING)

Cluster by pattern. Findings may overlap with stage milestones; the cross-cutting milestone focuses on the universal policy.

**M-contract-chaining.** For each adjacent stage pair (7 seams: DREAM→BRAINSTORM, BRAINSTORM→SEED, SEED→GROW, GROW→POLISH, POLISH→FILL, FILL→DRESS, DRESS→SHIP):
- Does the downstream stage explicitly enforce its Stage Input Contract at entry?
- Does the upstream stage's exit validation produce the documented Stage Output Contract?
- Does the "verbatim match" claim in the procedure docs hold in practice?

Expected finding: most stages trust graph state without explicit contract validation. Cluster by seam or by pattern, depending on how widespread the drift is.

All three cross-cutting passes are done by the main agent directly, not delegated to subagents, because they require the accumulated per-stage findings as context.

## Issue format and labels

**Epic issue body** (one per milestone):

```markdown
Spec-compliance audit for <stage or cross-cutting concern>.

Findings live in: docs/superpowers/reports/2026-04-18-spec-compliance-audit.md §M-<milestone>

Child cluster issues: #xxxx, #xxxx, …

This epic tracks bringing <stage or concern> into compliance with the
authoritative spec. Per CLAUDE.md §Design Doc Authority, specs supersede
code and tests — these issues document drift found in the audit.
```

**Cluster issue body** (one per cluster):

```markdown
**Spec rules:** R-<phase>.<n>, R-<phase>.<n>, …
**Spec reference:** docs/design/procedures/<stage>.md §<phase>
**Report section:** docs/superpowers/reports/2026-04-18-spec-compliance-audit.md §M-<milestone>#<cluster>

**Current state:** <1–2 sentences from report>
**Gap:** <1–2 sentences from report>
**Recommended fix:** <1–2 sentences from report>

**Code touch points:** src/questfoundry/…:lines
**Test touch points:** tests/…
```

**Labels:** `spec-audit`, plus one `area:*` label matching the milestone (`area:seed`, `area:grow`, `area:polish`, `area:cross-cutting`, etc.).

**Issue titles:** `[spec-audit] <stage>: <cluster name>` — the `[spec-audit]` prefix makes the full set filterable.

**Priority:** not encoded as labels at audit time. Priority becomes a scheduling decision when the milestones are later picked up for implementation.

## Deliverable order

1. Stage audits in pipeline order, with a user checkpoint after each. Epic + cluster issues for that stage are filed after the user confirms.
2. Three cross-cutting passes, with a user checkpoint after each. Epic + cluster issues filed after each confirmation.
3. Final summary section in the report; epic/issue URLs listed; audit complete.

## Success criteria

The audit is complete when:

1. Every rule in every procedure doc's Rule Index has been classified as `compliant`, `drift`, `missing`, or `uncheckable` in the report.
2. Every `drift` or `missing` finding is covered by a semantic cluster in the report.
3. Every cluster is represented by a GitHub issue, assigned to the correct milestone.
4. Every milestone has an epic issue linking to its child cluster issues.
5. All 11 milestones exist in the GitHub issue tracker.
6. The three cross-cutting audits are done and their findings linked back to the stage milestones where the code lives.
7. An `uncheckable` deferred list exists at the end of the report for follow-up runtime verification.

## What does not change

Nothing in code or tests. No spec edits. No implementation plans for the fixes yet. This audit's only outputs are the report, the issues, and the milestones.
