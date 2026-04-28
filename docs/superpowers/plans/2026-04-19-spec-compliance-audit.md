# Spec-Compliance Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the spec-compliance audit across all 8 pipeline stages plus three cross-cutting concerns, producing a living audit report and a set of GitHub epic + cluster issues grouped by 11 milestones. No code changes; no spec changes.

**Architecture:** Per-stage audits delegated to subagents (one subagent per stage, stage-first in pipeline order, code + tests together). Each stage's output is a report section following a fixed template; the main agent merges sections into the master report, files epic + cluster issues after user approval, and checkpoints before moving to the next stage. The three cross-cutting passes (logging, silent-degradation, contract-chaining) are executed directly by the main agent because they need the accumulated per-stage findings as context.

**Tech Stack:** Markdown (report), `gh` CLI (milestones + issues), Agent tool with `Explore` subagent type (stage audits), Read/Grep/Glob/Bash (cross-cutting passes). No application code or tests touched.

**Spec:** `docs/superpowers/specs/2026-04-18-spec-compliance-audit-design.md`

**Branch:** `docs/spec-audit-design` (already created by brainstorming step). All report commits land there; GitHub issues reference paths that will exist after this PR merges.

---

## Reference: Subagent Prompt Template

Every stage-audit subagent receives a prompt following this template. Per-task steps below only override the stage-specific fields.

```
You are auditing the <STAGE> stage for compliance with its authoritative
procedure spec. Output only the Markdown report section described below —
no implementation suggestions beyond the 'Recommended fix' field per
cluster, no file edits, no issue-filing.

AUTHORITATIVE SPECS (do not modify these):
- docs/design/procedures/<stage>.md  ← stage procedure (primary)
- docs/design/how-branching-stories-work.md  ← narrative model
- docs/design/story-graph-ontology.md  ← graph/data model
- CLAUDE.md §Logging (for logging-level expectations)
- CLAUDE.md §Design Doc Authority (for the overall precedence rule)

CODE TO AUDIT:
<list of exact paths per stage — filled per task below>

TESTS TO AUDIT:
<list of exact paths per stage — filled per task below>

METHODOLOGY:
1. Read procedures/<stage>.md end to end.
2. For each rule R-*.* in the Rule Index (exhaustively):
   a. Locate the code that should enforce it.
   b. Locate the test that should verify it.
   c. Classify: compliant | drift | missing | uncheckable.
      - 'uncheckable' = requires live LLM runtime to verify
        (e.g., prompt quality, diegetic voice). Flag and defer.
3. Group related drift/missing findings into semantic clusters.
   A cluster covers rules that share a root cause or fix.
   Aim for coarse clusters — one per concern, not one per rule.
4. For each cluster produce exactly the template below.
5. Emit a per-stage summary count at the top.

OUTPUT TEMPLATE (exact Markdown, no deviations):

## M-<STAGE>-spec

### Summary
- Rules checked: <N>
- Compliant: <X> | Drift: <Y> | Missing: <Z> | Uncheckable: <W>

### Cluster: <human-readable name>
**Rules covered:** R-<phase>.<n>, R-<phase>.<n>, …
**Current state:** <what the code does today, 1–2 sentences>
**Gap:** <what's missing or wrong, 1–2 sentences>
**Recommended fix:** <approach, not code, 1–2 sentences>
**Code refs:** `path/to/file.py:lines`, …
**Test refs:** `tests/path/test_file.py`, …

### Cluster: …

(Repeat per cluster.)

### Uncheckable rules
- R-<phase>.<n>: <why uncheckable>
- …

CONSTRAINTS:
- Output only the Markdown section. No preamble, no summary commentary
  outside the template, no code blocks other than what the template
  allows.
- Every drift/missing rule must appear in exactly one cluster.
- Recommended fix is an approach, not a TDD plan.
- Do not edit any files.
- Do not create issues.
- If you cannot locate a code or test reference for a rule, mark the
  rule 'missing' with a note '(no enforcing code found)'.
```

---

## Files Created / Modified by This Plan

- **Created:** `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` — living audit report, grows through all stages.
- **Created:** 11 GitHub milestones (`M-DREAM-spec`, …, `M-contract-chaining`).
- **Created:** ≈11 epic issues + N cluster issues (N depends on findings).
- **No other files touched.**

---

## Task 1: Create the Report Skeleton and Milestones

**Files:**
- Create: `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md`

- [ ] **Step 1: Verify parent directory exists**

Run:
```bash
ls -d docs/superpowers/reports/ 2>/dev/null || mkdir -p docs/superpowers/reports/
ls docs/superpowers/reports/
```
Expected: directory exists (empty or pre-existing).

- [ ] **Step 2: Write the report skeleton**

Create `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` with exactly this content:

```markdown
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
| M-DREAM-spec | pending | — | — | — |
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

(Milestone sections appended below as each audit stage completes.)
```

- [ ] **Step 3: Create the 11 GitHub milestones**

Run each command individually and confirm the response includes `"state":"open"`. If a milestone already exists (409 Conflict), skip it.

```bash
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-DREAM-spec' -f description='Spec-compliance audit for DREAM stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-BRAINSTORM-spec' -f description='Spec-compliance audit for BRAINSTORM stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-SEED-spec' -f description='Spec-compliance audit for SEED stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-GROW-spec' -f description='Spec-compliance audit for GROW stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-POLISH-spec' -f description='Spec-compliance audit for POLISH stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-FILL-spec' -f description='Spec-compliance audit for FILL stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-DRESS-spec' -f description='Spec-compliance audit for DRESS stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-SHIP-spec' -f description='Spec-compliance audit for SHIP stage — see docs/superpowers/reports/2026-04-18-spec-compliance-audit.md'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-logging-compliance' -f description='Spec-compliance audit: logging-level policy (CLAUDE.md §Logging) across all stages'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-silent-degradation' -f description='Spec-compliance audit: Silent Degradation policy across all stages'
gh api repos/pvliesdonk/questfoundry/milestones -f title='M-contract-chaining' -f description='Spec-compliance audit: Stage N input ≡ Stage N-1 output enforcement at seams'
```

- [ ] **Step 4: Capture milestone numbers for later**

Run:
```bash
gh api 'repos/pvliesdonk/questfoundry/milestones?state=open&per_page=50' --jq '.[] | select(.title | startswith("M-")) | "\(.number)\t\(.title)"'
```
Expected output: 11 lines of form `<number>\t<title>`. Copy this output into the Status table's `Status` column replacement later (Task 14).

- [ ] **Step 5: Commit the report skeleton**

```bash
git add docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
git commit -m "docs(audit): create spec-compliance audit report skeleton

Skeleton for the living audit report. 11 milestones created in
GitHub. Per-stage sections will be appended as each audit stage
completes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
Expected: commit succeeds; pre-commit hooks skip (no code files).

---

## Task Template: Stage Audit

Tasks 2–9 follow the same shape. Each audits one stage. The only things that change per task are the stage name, the code paths, the test paths, and the milestone title. Each task is a complete unit: dispatch subagent, merge report, checkpoint with user, file issues.

**Steps repeated in every stage-audit task (Tasks 2–9):**

1. Dispatch the `Explore` subagent with the prompt from the Reference section above, substituting the stage-specific fields.
2. Wait for subagent output.
3. Sanity-check the subagent output: does it match the template? Are the counts internally consistent (Compliant + Drift + Missing + Uncheckable = Rules checked)?
4. Append the subagent's section to `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md`.
5. Update the Status table: set the row's Status to `drafted`, fill Clusters count.
6. Commit the appended section.
7. **Checkpoint with user** — show the merged section; wait for approval before filing issues.
8. On approval: file the epic issue and child cluster issues against the stage's milestone.
9. Update the Status table: Status = `issues-filed`, fill Epic column with the epic issue number, Filed = count of cluster issues.
10. Commit the Status table update.

**Subagent dispatch command template:**

```
Agent tool invocation:
  subagent_type: "Explore"
  description: "Audit <STAGE> stage for spec compliance"
  prompt: <the Reference template above, with stage-specific fields filled in>
```

**Epic issue creation command template:**

```bash
gh issue create \
  --milestone 'M-<STAGE>-spec' \
  --label 'spec-audit' \
  --label 'area:<stage>' \
  --title '[spec-audit] <stage>: epic' \
  --body "$(cat <<'EOF'
Spec-compliance audit for the <STAGE> stage.

Findings live in: docs/superpowers/reports/2026-04-18-spec-compliance-audit.md §M-<STAGE>-spec

Child cluster issues: (listed below as they are filed)

This epic tracks bringing <STAGE> into compliance with the authoritative
spec at docs/design/procedures/<stage>.md. Per CLAUDE.md §Design Doc
Authority, specs supersede code and tests — these issues document drift
found in the audit.
EOF
)"
```

After filing cluster children, edit the epic body to list their numbers:
```bash
gh issue edit <epic_number> --body "$(cat <<'EOF'
Spec-compliance audit for the <STAGE> stage.

Findings live in: docs/superpowers/reports/2026-04-18-spec-compliance-audit.md §M-<STAGE>-spec

Child cluster issues: #NNNN, #NNNN, #NNNN

This epic tracks bringing <STAGE> into compliance with the authoritative
spec at docs/design/procedures/<stage>.md. Per CLAUDE.md §Design Doc
Authority, specs supersede code and tests — these issues document drift
found in the audit.
EOF
)"
```

**Cluster issue creation command template (one per cluster):**

```bash
gh issue create \
  --milestone 'M-<STAGE>-spec' \
  --label 'spec-audit' \
  --label 'area:<stage>' \
  --title '[spec-audit] <stage>: <cluster name>' \
  --body "$(cat <<'EOF'
**Spec rules:** R-<phase>.<n>, R-<phase>.<n>, …
**Spec reference:** docs/design/procedures/<stage>.md §<phase>
**Report section:** docs/superpowers/reports/2026-04-18-spec-compliance-audit.md §M-<STAGE>-spec §Cluster: <cluster name>

**Current state:** <copy from report>
**Gap:** <copy from report>
**Recommended fix:** <copy from report>

**Code touch points:**
- <copy from report>

**Test touch points:**
- <copy from report>

**Parent epic:** #<epic_number>
EOF
)"
```

---

## Task 2: Audit DREAM

**Files:**
- Modify: `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` (append M-DREAM-spec section)

- [ ] **Step 1: Dispatch the stage-audit subagent**

Invoke the `Explore` subagent with the Reference template prompt, substituting:

- `<STAGE>`: `DREAM`
- `<stage>`: `dream`
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/dream.py
  src/questfoundry/pipeline/stages/base.py (shared with all stages)
  src/questfoundry/models/ (any dream-related Pydantic models)
  src/questfoundry/prompts/templates/dream.md (if present)
  src/questfoundry/graph/mutations.py (Vision node mutations)
  src/questfoundry/graph/graph.py (Vision singleton retrieval)
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_dream_*.py (glob for all)
  tests/integration/test_dream_*.py (if any)
  tests/e2e/*dream* (if any)
  ```

- [ ] **Step 2: Sanity-check the subagent output**

Verify:
- Output begins with `## M-DREAM-spec`.
- Summary line shows `Rules checked: N` with N ≥ 13 (DREAM has 13 rules: R-1.1 through R-1.13).
- Compliant + Drift + Missing + Uncheckable = Rules checked.
- Every cluster has the five bold fields (Rules covered / Current state / Gap / Recommended fix / Code refs / Test refs).
- No orphan text outside the template.

If any check fails, re-dispatch the subagent with a targeted correction prompt.

- [ ] **Step 3: Append the section to the report**

Use the Edit tool to append the subagent's output (preceded by a `---` separator) to `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md`, right before the trailing `(Milestone sections appended below as each audit stage completes.)` comment line. Replace that comment with an empty line if it's the last section; otherwise leave the comment in place for subsequent appends.

- [ ] **Step 4: Update the Status table for DREAM**

Edit the Status table row for `M-DREAM-spec`: `Status: drafted`, `Clusters: <count>`.

- [ ] **Step 5: Commit the drafted DREAM section**

```bash
git add docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
git commit -m "docs(audit): draft DREAM stage findings

Per-stage audit of DREAM against docs/design/procedures/dream.md.
Issue filing follows after user approval.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 6: Checkpoint with user**

Show the DREAM section to the user. Wait for explicit approval before filing issues. If the user requests changes, edit the section in the report and re-commit, then re-checkpoint.

- [ ] **Step 7: File the DREAM epic issue**

Use the Epic issue creation template from Task Template. Substitute `<STAGE>` = `DREAM`, `<stage>` = `dream`. Record the returned issue number as `$EPIC_DREAM`.

- [ ] **Step 8: File cluster issues for DREAM**

For each cluster in the DREAM section, use the Cluster issue creation template. Substitute `<STAGE>` = `DREAM`, `<stage>` = `dream`, copy the five fields verbatim from the report, and use `$EPIC_DREAM` for the Parent epic. Record each returned issue number.

- [ ] **Step 9: Edit the DREAM epic body to list children**

Use the `gh issue edit` command from the Task Template with the collected cluster issue numbers.

- [ ] **Step 10: Update Status table for DREAM**

Edit the Status table row for `M-DREAM-spec`: `Status: issues-filed`, `Epic: #<epic>`, `Filed: <cluster_count>`.

- [ ] **Step 11: Commit the Status table update**

```bash
git add docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
git commit -m "docs(audit): file DREAM issues and update status table

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Audit BRAINSTORM

Follow Task 2's exact structure, substituting:

- `<STAGE>`: `BRAINSTORM` / `<stage>`: `brainstorm`
- Rule count: ≥ 17 (BRAINSTORM has 17 rules: R-1.1 through R-3.8 across 3 phases)
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/brainstorm.py
  src/questfoundry/models/ (Entity, Dilemma, Answer models)
  src/questfoundry/prompts/templates/brainstorm.md (if present)
  src/questfoundry/graph/mutations.py (Entity/Dilemma/Answer mutations)
  src/questfoundry/graph/context.py (format_valid_ids_context for entities)
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_brainstorm_*.py
  tests/integration/test_brainstorm_*.py (if any)
  ```

Execute Steps 1–11 of Task 2 with these substitutions.

---

## Task 4: Audit SEED

Follow Task 2's structure with:

- `<STAGE>`: `SEED` / `<stage>`: `seed`
- Rule count: ≥ 45 (SEED has the largest Rule Index; R-1.x through R-8.x across 8 phases including 3b)
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/seed.py
  src/questfoundry/agents/serialize.py (if SEED uses it)
  src/questfoundry/models/seed.py (Path, Consequence, beat models)
  src/questfoundry/prompts/templates/seed.md (and any sub-prompts)
  src/questfoundry/graph/context.py (format_valid_ids_context,
    format_path_ids_context, SEED context builders)
  src/questfoundry/graph/mutations.py (Path, Consequence, beat mutations)
  src/questfoundry/graph/invariants.py (Y-shape invariants)
  src/questfoundry/graph/dilemma_scoring.py (over-generate-and-select pruning)
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_seed_*.py (stage, models, prompts)
  tests/unit/test_mutations.py (SEED-related mutations)
  tests/integration/test_seed_e2e.py (if present)
  ```

**Expected to be large.** The subagent may need to return clusters in the double digits. That is the design.

Execute Steps 1–11.

---

## Task 5: Audit GROW

Follow Task 2's structure with:

- `<STAGE>`: `GROW` / `<stage>`: `grow`
- Rule count: ≥ 48 (GROW has rules across 9 phases: R-1.x through R-9.x)
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/grow/stage.py
  src/questfoundry/pipeline/stages/grow/deterministic.py
  src/questfoundry/pipeline/stages/grow/llm_phases.py
  src/questfoundry/pipeline/stages/grow/llm_helper.py
  src/questfoundry/pipeline/stages/grow/registry.py
  src/questfoundry/pipeline/stages/grow/_helpers.py
  src/questfoundry/graph/grow_algorithms.py
  src/questfoundry/graph/grow_context.py
  src/questfoundry/graph/grow_validation.py
  src/questfoundry/graph/grow_validators.py
  src/questfoundry/graph/algorithms.py (shared — intersection, arc, state-flag derivation)
  src/questfoundry/graph/invariants.py (shared invariants)
  src/questfoundry/models/grow.py
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_grow_*.py (deterministic, validation, validators, gates, registry, models)
  tests/integration/test_grow_e2e.py
  ```

**Expected to be large.** GROW is the stage with the most known drift (intersection lifecycle, `interleave_cycle_skipped`, `belongs_to` mutation during intersections).

Execute Steps 1–11.

---

## Task 6: Audit POLISH

Follow Task 2's structure with:

- `<STAGE>`: `POLISH` / `<stage>`: `polish`
- Rule count: ≥ 46 (POLISH has rules across 7 phases plus sub-phases 4a–4d)
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/polish/stage.py
  src/questfoundry/pipeline/stages/polish/deterministic.py
  src/questfoundry/pipeline/stages/polish/llm_phases.py
  src/questfoundry/pipeline/stages/polish/llm_helper.py
  src/questfoundry/pipeline/stages/polish/registry.py
  src/questfoundry/pipeline/stages/polish/_helpers.py
  src/questfoundry/graph/polish_context.py
  src/questfoundry/graph/polish_validation.py
  src/questfoundry/graph/algorithms.py (shared — compute_active_flags_at_beat, passage grouping)
  src/questfoundry/models/polish.py
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_polish_*.py (deterministic, phases, llm_phases, registry, apply, phase5_*, entry_contract, passage_validation, cli, models)
  ```

**Expected to be large.** POLISH is the stage whose passage-layer plan (Phase 4) was most affected by the intersection-lifecycle correction.

Execute Steps 1–11.

---

## Task 7: Audit FILL

Follow Task 2's structure with:

- `<STAGE>`: `FILL` / `<stage>`: `fill`
- Rule count: ≥ 28 (FILL has rules across 5 phases)
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/fill.py
  src/questfoundry/graph/fill_context.py
  src/questfoundry/graph/fill_validation.py
  src/questfoundry/models/fill.py (if present)
  src/questfoundry/prompts/templates/fill*.md
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_fill_*.py
  tests/integration/test_fill_*.py (if any)
  ```

Execute Steps 1–11.

---

## Task 8: Audit DRESS

Follow Task 2's structure with:

- `<STAGE>`: `DRESS` / `<stage>`: `dress`
- Rule count: ≥ 27 (DRESS has rules across 5 phases)
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/dress.py
  src/questfoundry/graph/dress_context.py
  src/questfoundry/graph/dress_mutations.py
  src/questfoundry/models/dress.py (if present)
  src/questfoundry/prompts/templates/dress*.md
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_dress_*.py
  ```

Execute Steps 1–11.

---

## Task 9: Audit SHIP

Follow Task 2's structure with:

- `<STAGE>`: `SHIP` / `<stage>`: `ship`
- Rule count: ≥ 18 (SHIP has rules across 4 phases)
- CODE TO AUDIT:
  ```
  src/questfoundry/pipeline/stages/ship.py
  src/questfoundry/export/ (all files — Twee, HTML, JSON, gamebook exporters)
  src/questfoundry/models/ship.py (if present)
  ```
- TESTS TO AUDIT:
  ```
  tests/unit/test_ship_*.py
  tests/unit/test_export_*.py (if separate)
  ```

Execute Steps 1–11.

---

## Task 10: Cross-Cutting Audit — Logging Compliance

**Files:**
- Modify: `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` (append M-logging-compliance section)

This task is executed directly by the main agent, not a subagent. It needs the accumulated stage findings as context.

- [ ] **Step 1: Enumerate every logging call site in the code**

Run:
```bash
grep -rn 'log\.\(debug\|info\|warning\|error\|critical\)\|logger\.\(debug\|info\|warning\|error\|critical\)' src/questfoundry/ --include='*.py' | wc -l
grep -rn 'log\.\(debug\|info\|warning\|error\|critical\)\|logger\.\(debug\|info\|warning\|error\|critical\)' src/questfoundry/ --include='*.py' > /tmp/qf_log_sites.txt
```
Expected: a count, and a file listing every `log.*` / `logger.*` call site with file:line.

- [ ] **Step 2: Classify each call site against CLAUDE.md §Logging**

For each line in `/tmp/qf_log_sites.txt`, read the surrounding code context (Read tool with the file path and a ±5-line window). Classify:
- **Compliant:** level matches the situation per the CLAUDE.md litmus test.
- **Misuse (over-level):** warning/error used where the system handled the problem correctly (should be info/debug).
- **Misuse (under-level):** info/debug used where a failure occurred (should be warning/error).
- **Misuse (missing):** no log at a phase/stage transition where one is required (this surfaces from the stage audits, not the grep).

Record each misuse in a scratch document (`/tmp/qf_logging_findings.md`) with file:line, current level, recommended level, rationale.

- [ ] **Step 3: Cluster the misuses**

Group by pattern. Typical clusters:
- Warnings used for successful fallbacks
- Info used for per-beat filtering noise (should be debug)
- Errors swallowed at log level instead of raised
- Missing logs at phase/stage transitions

Each cluster covers multiple file:line sites. Aim for 3–8 clusters total.

- [ ] **Step 4: Write the M-logging-compliance section**

Follow the report template. For each cluster, list the affected file:line sites under Code refs (may span multiple stages). Use the cross-cutting area label (`area:cross-cutting`) and cross-reference the stage milestones where the misuses are concentrated.

- [ ] **Step 5: Append to the report and commit**

Use Edit to append the section to `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md`. Update the Status table row: Status = `drafted`, Clusters = count.

```bash
git add docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
git commit -m "docs(audit): draft cross-cutting logging-compliance findings

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 6: Checkpoint with user**

Show the section. Wait for approval before filing issues.

- [ ] **Step 7: File epic + cluster issues**

Use the Epic and Cluster templates from the Stage Task Template, with:
- `--milestone 'M-logging-compliance'`
- `--label 'spec-audit' --label 'area:cross-cutting'`
- Epic title: `[spec-audit] logging-compliance: epic`
- Cluster titles: `[spec-audit] logging: <cluster name>`

Record the epic number and cluster numbers. Edit the epic to list children.

- [ ] **Step 8: Update Status table and commit**

```bash
git add docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
git commit -m "docs(audit): file logging-compliance issues and update status table

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 11: Cross-Cutting Audit — Silent Degradation

**Files:**
- Modify: `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` (append M-silent-degradation section)

- [ ] **Step 1: Enumerate known violation signatures**

Run:
```bash
grep -rn 'interleave_cycle_skipped\|all_intersections_rejected\|all-intersections-rejected' src/questfoundry/ tests/ --include='*.py' > /tmp/qf_silent_signatures.txt
grep -rn 'except[^:]*:\s*pass' src/questfoundry/ --include='*.py' > /tmp/qf_silent_except_pass.txt
grep -rn 'return None.*#.*fallback\|# fallback\|# silent' src/questfoundry/ --include='*.py' > /tmp/qf_silent_fallbacks.txt
```
Expected: three files listing potential silent-degradation sites.

- [ ] **Step 2: Read each site and classify**

For each entry in the three files, read surrounding context. Classify:
- **Genuine violation:** a structural constraint failure is silently absorbed (pipeline continues with wrong state).
- **Acceptable:** a narrow defensive except-pass that handles a benign case (log-and-continue with documented rationale).
- **Unclear:** needs human review.

Record in `/tmp/qf_silent_findings.md`.

- [ ] **Step 3: Also review stage-audit Recommended fix fields**

Re-read each completed stage section's clusters for mentions of silent-degradation patterns (cycle skips, empty-choice fallbacks, default-on-failure without logging). These augment the grep-derived findings.

- [ ] **Step 4: Cluster the violations**

Typical clusters:
- Cycle-skip suppression (interleave / temporal-hint)
- LLM-failure defaults applied without WARNING
- Validation fallbacks that mask structural bugs
- Empty-output fallbacks that should halt

Aim for 3–6 clusters.

- [ ] **Step 5: Write the M-silent-degradation section**

Follow the report template. For each cluster, list affected file:line sites. Cross-reference stage milestones.

- [ ] **Step 6: Append, commit, checkpoint, file issues, update Status table, commit**

Same flow as Task 10 steps 5–8, substituting:
- Commit message: `docs(audit): draft cross-cutting silent-degradation findings`
- Issue milestone: `M-silent-degradation`
- Epic title: `[spec-audit] silent-degradation: epic`
- Cluster titles: `[spec-audit] silent-degradation: <cluster name>`

---

## Task 12: Cross-Cutting Audit — Contract Chaining

**Files:**
- Modify: `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` (append M-contract-chaining section)

- [ ] **Step 1: For each of the 7 adjacent stage pairs, locate entry and exit validation**

The 7 pairs: DREAM→BRAINSTORM, BRAINSTORM→SEED, SEED→GROW, GROW→POLISH, POLISH→FILL, FILL→DRESS, DRESS→SHIP.

For each pair, answer:
- Does the downstream stage's code have an explicit entry-validation function/check that mirrors its Stage Input Contract?
- Does the upstream stage's code have an explicit exit-validation that produces the documented Stage Output Contract?
- If both exist: do they agree item-for-item with the spec's verbatim claim?

Use grep to locate validation code:
```bash
grep -rn 'validate\|entry_contract\|input_contract\|output_contract\|stage_input\|stage_output' src/questfoundry/pipeline/ src/questfoundry/graph/ --include='*.py'
```

Record findings per seam in `/tmp/qf_contract_findings.md`.

- [ ] **Step 2: Classify each seam**

- **Both sides validate and agree:** compliant.
- **One side validates:** drift — partial enforcement.
- **Neither side validates:** missing — seam is implicit.
- **Both validate but disagree with spec:** drift — contract divergence.

- [ ] **Step 3: Cluster by pattern**

Typical clusters:
- Seams with no explicit entry validation
- Seams with drift between code and spec contract
- Seams where validation exists but is not called by the stage pipeline

Aim for 2–5 clusters (7 seams usually cluster by pattern, not per seam).

- [ ] **Step 4: Write the M-contract-chaining section**

Follow the report template.

- [ ] **Step 5: Append, commit, checkpoint, file issues, update Status table, commit**

Same flow as Task 10 steps 5–8, substituting:
- Commit message: `docs(audit): draft cross-cutting contract-chaining findings`
- Issue milestone: `M-contract-chaining`
- Epic title: `[spec-audit] contract-chaining: epic`
- Cluster titles: `[spec-audit] contract-chaining: <cluster name>`

---

## Task 13: Finalize the Report

**Files:**
- Modify: `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` (add summary section; update Status table)

- [ ] **Step 1: Append a summary section**

Use Edit to append the following block to the end of the report, before any trailing comment:

```markdown
---

## Final Summary

**Audit completed:** <YYYY-MM-DD>
**Spec reference:** docs/superpowers/specs/2026-04-18-spec-compliance-audit-design.md

### Findings by milestone

| Milestone | Rules checked | Compliant | Drift | Missing | Uncheckable | Clusters | Epic |
|---|---|---|---|---|---|---|---|
| M-DREAM-spec | <N> | <X> | <Y> | <Z> | <W> | <count> | #<epic> |
| M-BRAINSTORM-spec | … | … | … | … | … | … | … |
| M-SEED-spec | … | … | … | … | … | … | … |
| M-GROW-spec | … | … | … | … | … | … | … |
| M-POLISH-spec | … | … | … | … | … | … | … |
| M-FILL-spec | … | … | … | … | … | … | … |
| M-DRESS-spec | … | … | … | … | … | … | … |
| M-SHIP-spec | … | … | … | … | … | … | … |
| M-logging-compliance | — | — | — | — | — | <count> | #<epic> |
| M-silent-degradation | — | — | — | — | — | <count> | #<epic> |
| M-contract-chaining | — | — | — | — | — | <count> | #<epic> |
| **Total** | **<sum>** | **<sum>** | **<sum>** | **<sum>** | **<sum>** | **<sum>** | — |

### Priority hotspots

Stages with the highest Drift+Missing count: <top 3 stages, names only>.

These should be scheduled first when implementation begins, because
they are most at risk of producing incorrect output.

### Uncheckable deferred list

See the Uncheckable deferred list section at the top of this report
for rules that need live-LLM runtime verification.

### Issue index

- M-DREAM-spec epic: #<n> (children: #<n>, #<n>, …)
- M-BRAINSTORM-spec epic: #<n> (children: …)
- …
- M-contract-chaining epic: #<n> (children: …)
```

Fill in all `<...>` placeholders from the Status table and the per-section data.

- [ ] **Step 2: Verify the Status table is fully populated**

Every row should have Status = `issues-filed`, Clusters = a number, Epic = `#<n>`, Filed = a number. No `pending` entries should remain.

- [ ] **Step 3: Commit the final summary**

```bash
git add docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
git commit -m "docs(audit): add final summary and issue index

The spec-compliance audit is complete. 11 milestones filed; all
drift/missing findings tracked as cluster issues under epic parents.
Priority hotspots identified for implementation scheduling.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 14: Push the Branch and Open the PR

**Files:**
- None (git operations only)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin docs/spec-audit-design
```
Expected: branch pushed; GitHub URL printed.

- [ ] **Step 2: Create the PR**

```bash
gh pr create --base main --title 'docs(audit): spec-compliance audit design, plan, and report' --body "$(cat <<'EOF'
## Summary

Spec-compliance audit across all 8 pipeline stages plus three cross-cutting concerns (logging, silent-degradation, contract-chaining). Output is a living audit report, 11 GitHub milestones, and epic + cluster issues per milestone.

**No code or test changes.** This PR delivers:
- `docs/superpowers/specs/2026-04-18-spec-compliance-audit-design.md` — audit design
- `docs/superpowers/plans/2026-04-19-spec-compliance-audit.md` — execution plan
- `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` — findings report

Issues filed (see the report's Final Summary §Issue index for a full list).

## Findings at a glance

See the Final Summary section of the report for per-milestone counts and priority hotspots.

## Next steps

Each milestone becomes its own implementation track with a dedicated TDD plan written via `superpowers:writing-plans`. Priority hotspots identified in the report's Final Summary guide the scheduling.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR URL returned.

- [ ] **Step 3: Report completion to user**

Post the PR URL and Final Summary snippet to the user as the terminal signal for this plan.

---

## Verification Checklist (to run after Task 13)

Before Task 14 (pushing), verify these end-to-end conditions. Each should be checkable via `gh`/`grep`; if any fails, fix the specific gap before pushing.

- [ ] 11 milestones exist in GitHub with expected titles:
  ```bash
  gh api 'repos/pvliesdonk/questfoundry/milestones?state=open&per_page=50' --jq '.[] | select(.title | startswith("M-")) | .title' | sort
  ```
  Expected: 11 lines, alphabetical by title.

- [ ] 11 epic issues exist (one per milestone):
  ```bash
  gh issue list --label 'spec-audit' --search 'epic in:title' --limit 50 --state open --json number,title,milestone
  ```
  Expected: 11 entries.

- [ ] Every cluster issue has a parent epic referenced in its body:
  ```bash
  gh issue list --label 'spec-audit' --limit 200 --state open --json number,body --jq '.[] | select(.body | contains("Parent epic") | not) | .number'
  ```
  Expected: empty output (no issues missing Parent epic reference).

- [ ] The report's Status table has no `pending` rows:
  ```bash
  grep '| pending |' docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
  ```
  Expected: no output.

- [ ] The report references every epic issue in the Final Summary:
  ```bash
  grep -c '^- M-.*epic: #' docs/superpowers/reports/2026-04-18-spec-compliance-audit.md
  ```
  Expected: 11.

---

## Self-Review Notes

**Spec coverage:** Every milestone from the spec's architecture table has exactly one audit task. Stage audits (Tasks 2–9) cover all 8 stages in pipeline order per the spec's sequencing. Cross-cutting audits (Tasks 10–12) cover the three cross-cutting milestones. Task 1 creates the foundation; Task 13 finalizes; Task 14 closes out with PR creation. The "out of scope" items from the spec (code changes, spec changes, runtime verification, TDD implementation plans) are explicitly absent from the task list.

**Placeholder scan:** No TBDs, no "add appropriate", no "similar to task N". The Stage Task Template is repeated structure with explicit substitutions documented per task.

**Type / naming consistency:** Milestone titles (`M-DREAM-spec`, etc.) are consistent across Task 1 creation, per-task dispatch, and Task 13 summary. Report path `docs/superpowers/reports/2026-04-18-spec-compliance-audit.md` is consistent. Issue title prefix `[spec-audit]` and label `spec-audit` are consistent.

**Known limitation:** Rule counts in each stage task are lower-bound estimates (from the Rule Index counts as of 2026-04-18). If a procedure doc has been updated post-audit-start, the subagent will detect more rules; the `≥` phrasing accommodates this.
