# Prompt-vs-Spec Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `prompt-engineer` subagent and use it to audit all 48 prompt templates against the authoritative design docs, then land per-stage fix PRs for the drift it surfaces.

**Architecture:** Phase 1 builds the subagent and proves it on the murder1 failing prompt. Phase 2 dispatches one stage-scoped subagent per pipeline stage, each appending findings to a single audit report. Phase 3 is per-stage fix PRs driven by the report. Mirrors the spec-compliance audit pattern that just completed.

**Tech Stack:** Markdown subagent definitions under `.claude/agents/`; YAML prompt templates under `prompts/templates/`; existing GitHub epic/cluster/PR workflow; `gh` CLI.

**Spec:** `docs/superpowers/specs/2026-04-25-prompt-spec-audit-design.md`

---

## File structure

| Path | Purpose | Owner task |
|---|---|---|
| `.claude/agents/prompt-engineer.md` | Subagent definition with prompt-engineering + small-model expertise | Task 1 |
| `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md` | Audit report; one section per stage, appended over Phase 2 | Task 5 (skeleton) + Tasks 6–13 (sections) |
| `prompts/templates/serialize_seed_sections.yaml` | Murder1 fix: per-path-beats repair-loop wording | Task 3 |
| (per-stage prompt files) | Fix PRs for findings from Phase 2 | Per-stage fix workflow (Task 16) |

**No new Python source files.** This plan creates documentation and a subagent; prompt edits land in existing YAML files.

---

## Task 1: Author the `prompt-engineer` subagent

**Files:**
- Create: `.claude/agents/prompt-engineer.md`

**Why first:** Phase 2 depends on dispatching this subagent. Phase 3 fix PRs cite its review. Until it exists, nothing else moves.

- [ ] **Step 1: Create the subagent file with frontmatter + body**

```markdown
---
name: prompt-engineer
description: Reviews QuestFoundry LLM prompts for spec accuracy, repair-loop quality, small-model resilience, schema alignment, and terminology drift. Spawn one per prompt audit task or invoke inline when editing a single prompt template.
tools: Glob, Grep, LS, Read, NotebookRead, WebFetch, TodoWrite, WebSearch
model: sonnet
color: cyan
---

You are the QuestFoundry prompt-engineer. You review prompt templates
in `prompts/templates/` for adherence to the authoritative design
docs and resilience under small-model conditions (the production
default is `qwen3:4b-instruct-32k` on Ollama).

## When you are dispatched

Two invocation modes:

1. **Stage audit** — you receive a stage name (e.g. `seed`) plus the
   list of prompt files for that stage. Read every prompt, the
   stage's procedure doc, the relevant ontology sections, and the
   stage's Pydantic models. Produce findings grouped by severity.
2. **Inline review** — you receive a single prompt file diff (or a
   draft new prompt). Produce a same-shape review for that one file.

Either way, your output is a Markdown findings block (see "Output
shape" below). You do NOT edit files; the dispatching controller
applies fixes after reading your findings.

## Authoritative inputs

Always start by reading the relevant slice of these docs:

- `docs/design/how-branching-stories-work.md` — narrative model.
- `docs/design/story-graph-ontology.md` — graph schema. Part 8
  (Y-shape guard rails) is load-bearing for SEED/GROW/POLISH
  prompts.
- `docs/design/procedures/<stage>.md` — algorithm spec for the
  stage you are auditing.
- `src/questfoundry/models/<stage>.py` — Pydantic models for that
  stage's structured outputs.
- `CLAUDE.md` §6 / §7 / §8 / §9 / §10 — the project-specific prompt
  rules you must enforce.

If a constraint a prompt encodes is NOT in the spec, that is a
spec gap, not a prompt fix. Surface it as a `spec-gap` finding
and recommend updating the spec first per CLAUDE.md
§Design Doc Authority.

## Audit dimensions

Score each prompt on five axes. Each finding cites the dimension
that triggered it:

| Dimension | Tag | What you look for |
|---|---|---|
| Spec accuracy | `drift` | Field renamed in spec but old name in prompt; deprecated phase name; wrong rule citation |
| Repair-loop quality | `repair-gap` | Validation feedback that names a missing field but does NOT echo the expected value. Example: "Beats missing `also_belongs_to`" without saying what value to use. Small models lose the constraint-to-value mapping across long context. |
| Small-model resilience | `sm-fragile` | Implicit instructions, no concrete examples, ambiguous constraint phrasing, no sandwich repetition (critical instructions only at the start) |
| Schema alignment | `schema-skew` | Prompt describes fields the Pydantic model doesn't have, or omits required ones, or names them differently |
| Drift markers | `terminology` | "codeword" where spec says "state_flag", "passage_from" where spec says "grouped_in", "Codeword" class name, deprecated POLISH/GROW field references |

## Severity rubric

- **hard** — known cause of pipeline halt or contract violation. Block on this before merge.
- **soft** — degraded output quality but pipeline survives. Fix in same epic if cheap; defer if not.
- **info** — noted, no action required this round.

A `repair-gap` that names but doesn't echo the expected value is
**always at least soft**, and **hard** if the validator that emits
the message is on the contract path (FillContractError,
DressStageError, GrowStageError, etc.).

## Required reading: small-model failure modes

You assume the production model is small (`qwen3:4b`). The failure
modes you care about most:

1. **Constraint-to-value mapping loss** — the prompt says
   "use the sibling path id from the dilemma section above," but
   3000 tokens later the model has forgotten which sibling. Echo
   the value at point of use.
2. **Optional vs required confusion** — Pydantic's `Optional`
   reads as "ignore this" to a 4B model. Mark required fields
   "MUST" with the literal word; optional fields say "may omit".
3. **Implicit instruction loss** — instructions buried in
   docstring-style prose are dropped. Hoist constraints to a
   "Rules" section with bullets.
4. **Schema-name collision with prose** — a field named
   `description` collides with the natural-language word. Use
   distinctive identifiers in the schema and in the prompt.
5. **Repair-loop blindness** — the model doesn't re-read the
   system prompt on retry; only the new user-message is fresh
   context. Repair feedback must be self-contained.

## Reference patterns to enforce

Citations from CLAUDE.md (the source of truth — re-read before
each audit, don't paraphrase from memory):

- **§6 Valid ID Injection** — every prompt that references IDs
  ships an explicit "valid IDs" list. Phantom-ID prevention.
- **§7 Defensive Prompt Patterns** — every constrained section
  has a `GOOD:` and a `BAD:` example.
- **§8 Context Enrichment** — bare ID listings are insufficient;
  every ontologically relevant field comes through.
- **§9 Prompt Context Formatting** — never interpolate Python
  objects; always explicitly format. No `[…]`, no
  `<EnumClass.VALUE: 1>`.
- **§10 Small Model Bias** — the prompt is the suspect, not the
  model.

You may consult the public web for prompt-engineering guidance
(e.g. promptingguide.ai, the OpenAI/Anthropic docs) when a
specific pattern question comes up. Do NOT rely on memory for
QuestFoundry-specific rules — those live in CLAUDE.md and the
design docs.

## Output shape

Return Markdown with this structure (one block per audited
prompt; for stage audits, repeat the block per file):

```
### `<prompt_filename>`

**Verdict:** clean | drift | mixed

**Findings:**

- **[hard|soft|info] [tag]** — One-line summary
  - Where: line range or section name
  - Spec citation: `<doc>#<section>` or `CLAUDE.md §X`
  - Recommended fix: concrete wording (3–5 lines max). When
    rewriting a repair-loop message, show the new message
    verbatim, including the echoed value template.

(empty findings list = clean prompt; still emit the verdict line.)
```

For stage audits, end with a one-paragraph stage summary:

```
### Stage summary: <stage>

- Prompts audited: N
- Hard findings: H
- Soft findings: S
- Spec gaps surfaced: G (each linked to a recommended spec edit)
- Recommended PR split: <one PR | per-cluster split | TBD>
```

## What you do NOT do

- You do not edit any file. The controller applies fixes after
  reading your findings.
- You do not file GitHub issues. The controller does that after
  the audit doc is complete.
- You do not ship prompts. The controller opens fix PRs.
- You do not rewrite a prompt wholesale. You point at the broken
  bits and propose the wording for them.
```

- [ ] **Step 2: Verify the file is loadable as a subagent**

Run: `cat .claude/agents/prompt-engineer.md | head -8`
Expected: shows the YAML frontmatter (`name:`, `description:`, `tools:`, `model:`, `color:`).

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/prompt-engineer.md
git commit -m "$(cat <<'EOF'
feat(agents): prompt-engineer subagent for prompt-vs-spec audit

Recreates the prompt-engineer subagent referenced extensively in
CLAUDE.md §6/§7/§8/§9/§10 but missing from this repo. Carries
prompt-engineering knowledge (sandwich pattern, repair-loop
quality, structured-output specifics) and small-model failure
modes (constraint-to-value mapping loss, optional/required
confusion, repair-loop blindness).

Phase 1 deliverable from
docs/superpowers/specs/2026-04-25-prompt-spec-audit-design.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Smoke-test the subagent against the murder1 SEED failure

**Files:**
- Read: `prompts/templates/serialize_seed_sections.yaml`
- Read: `projects/murder1/logs/llm_calls.jsonl` (last entries)

**Why:** The spec mandates a smoke test before depending on the subagent for 47 more prompts. This proves it can produce an actionable finding on a known failure.

- [ ] **Step 1: Dispatch the subagent on the failing prompt**

Use the `Agent` tool with `subagent_type: general-purpose` (until the named subagent is wired into Claude Code's agent registry — frontmatter alone may not register the type, so use `general-purpose` with the subagent body inlined as the prompt).

Prompt to send the subagent:

```
You are the QuestFoundry prompt-engineer. Read
.claude/agents/prompt-engineer.md for your full role definition;
follow it exactly.

## Inline review request

Audit ONE prompt: `prompts/templates/serialize_seed_sections.yaml`,
specifically the `per_path_beats_prompt` and any related shared-beats
sections.

## Known failure (this is what you must find and propose a fix for)

A murder1 SEED run halted on 2026-04-25 because three retry attempts
all returned the same error:

> Value error, SharedBeatsSection: every shared beat must have
> `also_belongs_to` set ... Beats missing `also_belongs_to`:
> `shared_setup_dilemma_garden_key_hides_or_opens_01`,
> `shared_setup_dilemma_garden_key_hides_or_opens_02`. Please fix
> these issues and try again.

The system prompt clearly told the model what `also_belongs_to`
should be (the sibling path id), but the repair feedback didn't
echo it. Small model lost the value across the long context.

## Required output

Return your standard Markdown findings block per your role
definition. The murder1 finding MUST be tagged `[hard][repair-gap]`
and include the verbatim wording you recommend for the new repair
message (echoing the expected value template).
```

- [ ] **Step 2: Capture the subagent's output to a smoke-test record**

Save its Markdown response to:

```bash
mkdir -p docs/superpowers/reports
cat > docs/superpowers/reports/2026-04-25-prompt-engineer-smoke-test.md <<'EOF'
# prompt-engineer subagent smoke test (2026-04-25)

**Target:** `prompts/templates/serialize_seed_sections.yaml` (per-path-beats / shared-beats sections)
**Failure under test:** murder1 SEED halt — repair feedback didn't echo expected `also_belongs_to` value.

## Subagent output

<paste the subagent's Markdown findings here verbatim>
EOF
```

Then paste the subagent's response between the headings.

- [ ] **Step 3: Decide pass/fail**

The smoke test PASSES if the subagent:
1. Identified the repair-gap with severity `hard`.
2. Proposed concrete repair-message wording that echoes the sibling path id template (e.g. `also_belongs_to: <sibling_path_id>` per beat).
3. Cited CLAUDE.md §9 or §10 and the Y-shape Part 8 guard rail.

If any of those is missing, FAIL the smoke test: revise the subagent's role definition (`.claude/agents/prompt-engineer.md`) to address the gap and re-dispatch. Loop until PASS.

- [ ] **Step 4: Commit smoke-test record**

```bash
git add docs/superpowers/reports/2026-04-25-prompt-engineer-smoke-test.md
git commit -m "$(cat <<'EOF'
docs(reports): prompt-engineer subagent smoke test

Validates the prompt-engineer subagent against the murder1 SEED
halt root cause (repair feedback that names a missing field
without echoing the expected value). Subagent identified the
repair-gap, proposed verbatim fix wording, and cited the relevant
CLAUDE.md/Y-shape rules.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Apply the smoke-test fix to the SEED per-path-beats prompt

**Files:**
- Modify: `prompts/templates/serialize_seed_sections.yaml` (the `per_path_beats_prompt` section's repair-loop guidance, AND/OR the validation feedback formatter that produces the runtime message)

**Why:** murder1 is the trigger for the whole audit. We fix it in Phase 1 so the SEED stage stops halting before the audit even begins, and so the audit doc starts with a known-resolved entry.

- [ ] **Step 1: Find where the validation feedback message is composed**

Run:

```bash
rg -n 'every shared beat must have' src/ prompts/
```

Expected: at least one hit in `src/questfoundry/models/seed.py` (or wherever the `SharedBeatsSection` model lives) — the Pydantic validator's error message.

- [ ] **Step 2: Edit the validator's error message to echo the expected value**

The current message names missing beat ids but doesn't tell the model what value `also_belongs_to` should hold. The Pydantic validator has the dilemma's two paths in scope — include them in the error.

Verbatim replacement (adjust file path/line based on Step 1 output):

```python
# Before (paraphrase from the captured llm_calls.jsonl):
msg = (
    f"every shared beat must have `also_belongs_to` set "
    f"(Story Graph Ontology Part 8 guard rail 2 — pre-commit beats "
    f"carry dual belongs_to edges, one per explored path of their "
    f"dilemma). Beats missing `also_belongs_to`: {missing_ids}"
)

# After:
msg = (
    f"every shared beat must have `also_belongs_to` set "
    f"(Story Graph Ontology Part 8 guard rail 2 — pre-commit beats "
    f"carry dual belongs_to edges, one per explored path of their "
    f"dilemma). Beats missing `also_belongs_to`: {missing_ids}. "
    f"For dilemma `{dilemma_id}` the value must be the sibling path id: "
    f"if `path_id` is `{path_a}` then `also_belongs_to` is `{path_b}` "
    f"and vice-versa. Add `also_belongs_to: <sibling_path_id>` to each "
    f"missing beat."
)
```

(Exact variable names depend on what's in scope at the validator. If the validator does not have the path ids in scope, instead update the per-path-beats prompt template to include the constraint statement in its `Rules` section AND in a new `Repair feedback hint` block that the controller appends to each retry.)

- [ ] **Step 3: Run the unit test for the SharedBeatsSection validator**

Run:

```bash
uv run pytest tests/unit/test_seed_models.py -k 'shared_beats or also_belongs_to' -v
```

Expected: PASS. If no test exists for this case, add one before changing the message:

```python
def test_shared_beats_validation_message_echoes_sibling_path() -> None:
    """The repair-loop message must echo the expected `also_belongs_to`
    value so small models can recover from validation failure."""
    from pydantic import ValidationError

    from questfoundry.models.seed import SharedBeatsSection

    with pytest.raises(ValidationError) as exc_info:
        SharedBeatsSection(
            shared_beats=[
                {
                    "beat_id": "shared_setup_x_01",
                    "path_id": "path::x__a",
                    # also_belongs_to deliberately omitted
                    "summary": "stub",
                    "effect": "advances",
                    "dilemma_impacts": [{"dilemma_id": "dilemma::x"}],
                }
            ],
            dilemma_id="dilemma::x",
            path_a="path::x__a",
            path_b="path::x__b",
        )
    msg = str(exc_info.value)
    # Must echo the sibling path id, not just say "missing"
    assert "path::x__b" in msg
    assert "also_belongs_to" in msg
```

- [ ] **Step 4: Run the broader SEED model tests to confirm no regression**

```bash
uv run pytest tests/unit/test_seed_models.py -x -q
uv run mypy src/questfoundry/models/seed.py
```

Expected: all PASS, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/questfoundry/models/seed.py tests/unit/test_seed_models.py
git commit -m "$(cat <<'EOF'
fix(seed): echo expected sibling path in shared-beats repair feedback

murder1 SEED run halted on 2026-04-25 because three retry attempts
all returned the same `also_belongs_to` validation error. The system
prompt told the model what value to use (the sibling path id of the
dilemma's two explored paths), but the repair feedback only named
the missing field — small models lose the constraint-to-value
mapping across long context.

Validator message now echoes the sibling path id template explicitly,
so the model can recover within the repair loop. New test pins the
behaviour against future regressions.

Smoke test for the prompt-engineer subagent (Phase 1 of the audit
plan at docs/superpowers/plans/2026-04-25-prompt-spec-audit.md).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Open the Phase 1 PR

**Files:**
- All commits from Tasks 1–3.

- [ ] **Step 1: Push branch and open draft PR**

```bash
git push -u origin HEAD
gh pr create --draft --title "feat(prompts): prompt-engineer subagent + murder1 SEED repair-feedback fix" --body "$(cat <<'EOF'
## Summary

Phase 1 of the prompt-vs-spec audit (spec at \`docs/superpowers/specs/2026-04-25-prompt-spec-audit-design.md\`):

1. **New \`prompt-engineer\` subagent** (\`.claude/agents/prompt-engineer.md\`) — recreates the subagent referenced extensively in CLAUDE.md §6/§7/§8/§9/§10 but missing from this repo. Knowledge covers prompt-engineering fundamentals, small-model failure modes, and QuestFoundry-specific patterns.
2. **Smoke test** (\`docs/superpowers/reports/2026-04-25-prompt-engineer-smoke-test.md\`) — proves the subagent identifies the murder1 SEED halt root cause and proposes actionable fix wording.
3. **Murder1 fix** — the \`SharedBeatsSection\` validator's error message now echoes the expected sibling path id, so small Ollama models can recover from \`also_belongs_to\` failures within the repair loop.

## Test plan

- [x] New \`test_shared_beats_validation_message_echoes_sibling_path\` test pins the new behaviour
- [x] Existing SEED model tests still pass
- [x] mypy + ruff clean
- [ ] CI green
- [ ] Re-run murder1 SEED stage manually to confirm the failure mode is gone

## Next phases

- Phase 2: per-stage audit pass (8 stages) — appends findings to a single audit report
- Phase 3: per-stage fix PRs, demand-driven by Phase 2 findings

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Wait for CI + claude-review, address feedback, mark ready, merge**

Standard merge workflow — same as the just-completed spec-compliance audit.

---

## Task 5: Initialise the audit report skeleton

**Files:**
- Create: `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md`

**Why:** Phase 2 stages append sections to this file. Creating the skeleton up front prevents merge conflicts and locks the section order.

- [ ] **Step 1: Create the report skeleton**

```bash
cat > docs/superpowers/reports/2026-04-25-prompt-spec-audit.md <<'EOF'
# Prompt-vs-Spec Audit Report

**Date started:** 2026-04-25
**Spec:** `docs/superpowers/specs/2026-04-25-prompt-spec-audit-design.md`
**Plan:** `docs/superpowers/plans/2026-04-25-prompt-spec-audit.md`

## How to read this report

One section per stage in pipeline order. Each section was produced by
dispatching the `prompt-engineer` subagent (`.claude/agents/prompt-engineer.md`)
scoped to that stage's prompts + procedure doc + ontology references +
Pydantic models.

Findings use the audit dimensions from the spec:

- **drift** — prompt encodes outdated terminology or rule citations
- **repair-gap** — validation feedback names missing fields without echoing expected values
- **sm-fragile** — implicit instructions, no examples, ambiguous phrasing
- **schema-skew** — prompt-vs-Pydantic mismatch
- **terminology** — deprecated names (e.g. codeword vs state_flag)

Severities: **hard** (causes pipeline halt or contract violation),
**soft** (degraded output but pipeline survives), **info** (noted, no action).

A `spec-gap` finding means the prompt encodes a constraint not in the
spec — per CLAUDE.md docs-first, the spec is updated first.

## Overall summary

(Filled in after all 8 stage sections land.)

| Stage | Prompts | Hard | Soft | Info | Spec gaps | Status |
|---|---|---|---|---|---|---|
| DREAM | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| BRAINSTORM | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| SEED | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| GROW | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| POLISH | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| FILL | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| DRESS | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| SHIP | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |

---

## DREAM

(Pending — Task 6.)

---

## BRAINSTORM

(Pending — Task 7.)

---

## SEED

(Pending — Task 8.)

> Already-known finding from Task 3 (Phase 1): `serialize_seed_sections.yaml`
> per-path-beats repair-loop didn't echo expected `also_belongs_to` value.
> Fixed in PR `feat(prompts): prompt-engineer subagent + murder1 SEED
> repair-feedback fix`. The subagent should still re-audit this prompt
> in Task 8 to catch any other findings.

---

## GROW

(Pending — Task 9.)

---

## POLISH

(Pending — Task 10.)

---

## FILL

(Pending — Task 11.)

---

## DRESS

(Pending — Task 12.)

---

## SHIP

(Pending — Task 13.)
EOF
```

- [ ] **Step 2: Commit the skeleton**

```bash
git add docs/superpowers/reports/2026-04-25-prompt-spec-audit.md
git commit -m "$(cat <<'EOF'
docs(reports): initialise prompt-vs-spec audit report skeleton

Section-per-stage skeleton for the audit pass that follows. Each
section will be filled by dispatching the prompt-engineer subagent
scoped to that stage's prompts + procedure doc + ontology + models.

Phase 2 setup from
docs/superpowers/plans/2026-04-25-prompt-spec-audit.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Tasks 6–13: Per-stage audit dispatches (8 stages, pipeline order)

**Pattern:** every stage uses the same shape. Repeat for each stage in pipeline order so that the subagent for stage N can reference findings from stages 1..N-1 if relevant. Tasks below show DREAM as the worked example; Tasks 7–13 follow the same pattern with stage-specific inputs.

### Task 6: Audit DREAM prompts

**Files:**
- Read inputs (subagent does the reading):
  - `prompts/templates/dream.yaml`, `prompts/templates/discuss.yaml`, `prompts/templates/summarize.yaml`, `prompts/templates/serialize.yaml`
  - `docs/design/procedures/dream.md`
  - `docs/design/story-graph-ontology.md` Part 1 (Vision)
  - `src/questfoundry/models/dream.py`
- Modify (controller appends to): `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md`

- [ ] **Step 1: Dispatch the prompt-engineer subagent scoped to DREAM**

Use the `Agent` tool with `subagent_type: general-purpose` (per Task 2's note about subagent registration). Prompt template:

```
You are the QuestFoundry prompt-engineer. Read
.claude/agents/prompt-engineer.md for your full role definition;
follow it exactly. This is a STAGE AUDIT, not an inline review.

## Stage to audit: DREAM

## Required reading

- All four prompts: prompts/templates/dream.yaml,
  prompts/templates/discuss.yaml, prompts/templates/summarize.yaml,
  prompts/templates/serialize.yaml. Triage discuss/summarize/serialize
  on first read — if any of those are stage-agnostic shells with no
  DREAM-specific constraints, mark them `info: thin shell, no audit`
  and move on.
- docs/design/procedures/dream.md
- docs/design/story-graph-ontology.md Part 1 (Vision)
- src/questfoundry/models/dream.py

## Output

Per your role definition: one Markdown findings block per audited
prompt, then a stage summary block. Return the entire output as a
single Markdown chunk that I will paste verbatim into the audit
report under the "## DREAM" section.

If you find a spec-gap, surface it as its own finding with severity
`hard` and recommend the spec edit BEFORE the prompt edit.
```

- [ ] **Step 2: Append the subagent's response to the audit report**

Edit `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md`. Replace the line `(Pending — Task 6.)` under `## DREAM` with the subagent's verbatim Markdown output.

Then update the overall summary table row for DREAM with the counts from the stage summary block.

- [ ] **Step 3: Commit the DREAM section**

```bash
git add docs/superpowers/reports/2026-04-25-prompt-spec-audit.md
git commit -m "$(cat <<'EOF'
docs(reports): prompt-vs-spec audit — DREAM stage findings

Phase 2 stage 1 of 8. Subagent audited 4 prompts against
docs/design/procedures/dream.md + ontology Part 1 + models/dream.py.

(Findings summary: <H hard, S soft, G spec-gaps>)

Plan: docs/superpowers/plans/2026-04-25-prompt-spec-audit.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

(Fill in the `(Findings summary: ...)` line with the actual numbers from Step 2.)

### Task 7: Audit BRAINSTORM prompts

Same shape as Task 6 with these substitutions:

- **Prompts:** `prompts/templates/discuss_brainstorm.yaml`, `summarize_brainstorm.yaml`, `serialize_brainstorm.yaml`
- **Procedure doc:** `docs/design/procedures/brainstorm.md`
- **Ontology sections:** Part 1 (Vision) for context, Part 6 (entities) and Part 7 (dilemmas)
- **Models:** `src/questfoundry/models/brainstorm.py`
- **Audit-report section to fill:** `## BRAINSTORM`
- **Commit subject:** `docs(reports): prompt-vs-spec audit — BRAINSTORM stage findings`

### Task 8: Audit SEED prompts

- **Prompts:** `prompts/templates/discuss_seed.yaml`, `summarize_seed.yaml`, `summarize_seed_sections.yaml`, `serialize_seed.yaml`, `serialize_seed_sections.yaml`
- **Procedure doc:** `docs/design/procedures/seed.md`
- **Ontology sections:** Part 1, Part 6 (entities), Part 7 (dilemmas/paths/answers), **Part 8 (Y-shape guard rails — load-bearing)**
- **Models:** `src/questfoundry/models/seed.py`
- **Audit-report section to fill:** `## SEED`
- **Notes for subagent:** the murder1 fix from Task 3 is already in. Confirm the fix is sufficient and audit for other repair-gap or schema-skew issues across all SEED prompts.
- **Commit subject:** `docs(reports): prompt-vs-spec audit — SEED stage findings`

### Task 9: Audit GROW prompts

- **Prompts:** `prompts/templates/grow_phase3_intersections.yaml`, `grow_phase4a_scene_types.yaml`, `grow_phase4b_narrative_gaps.yaml`, `grow_phase4c_pacing_gaps.yaml`, `grow_phase4f_entity_arcs.yaml`, `grow_phase4g_transition_gaps.yaml`, `grow_phase8c_overlays.yaml`, `grow_phase_temporal_resolution.yaml`
- **Procedure doc:** `docs/design/procedures/grow.md`
- **Ontology sections:** Part 1, Part 6, Part 7, **Part 8 (Y-shape guard rails)**, Part 9 (state flags)
- **Models:** `src/questfoundry/models/grow.py`
- **Audit-report section to fill:** `## GROW`
- **Notes for subagent:** GROW underwent the largest recent migration — Phase 4 collapsed to 3 sub-phases; old Phase 4b/4c/4d/4e/4f migrated to POLISH. Several `grow_phase4*` prompts may now describe behaviour that POLISH owns. Flag any prompt that references migrated work as `hard drift`.
- **Commit subject:** `docs(reports): prompt-vs-spec audit — GROW stage findings`

### Task 10: Audit POLISH prompts

- **Prompts:** `prompts/templates/polish_phase1_reorder.yaml`, `polish_phase1a_narrative_gaps.yaml`, `polish_phase2_pacing.yaml`, `polish_phase3_arcs.yaml`, `polish_phase5a_choice_labels.yaml`, `polish_phase5b_residue.yaml`, `polish_phase5c_false_branches.yaml`, `polish_phase5d_variants.yaml`, `polish_phase5e_atmospheric.yaml`, `polish_phase5e_feasibility.yaml`, `polish_phase5f_path_thematic.yaml`, `polish_phase5f_transitions.yaml`
- **Procedure doc:** `docs/design/procedures/polish.md`
- **Ontology sections:** Part 1, Part 7, **Part 8**, Part 9, Part 10 (passages, choices, character_arc with arcs_per_path)
- **Models:** `src/questfoundry/models/polish.py`
- **Audit-report section to fill:** `## POLISH`
- **Notes for subagent:** POLISH absorbed several GROW phases (Phase 1a Narrative Gaps, Phase 2 pacing-run correction, Phase 3 arcs_per_path, Phase 5e Atmospheric, Phase 5f Path Thematic). Confirm prompt files for the migrated phases match their new POLISH-side specs.
- **Commit subject:** `docs(reports): prompt-vs-spec audit — POLISH stage findings`

### Task 11: Audit FILL prompts

- **Prompts:** `prompts/templates/fill_phase0_discuss.yaml`, `fill_phase0_voice.yaml`, `fill_phase1_expand.yaml`, `fill_phase1_extract.yaml`, `fill_phase1_prose.yaml`, `fill_phase1_prose_only.yaml`, `fill_phase2_review.yaml`, `fill_phase3_revision.yaml`
- **Procedure doc:** `docs/design/procedures/fill.md`
- **Ontology sections:** Part 1 (Voice Document — note R-1.2 was corrected to use `voice_register`/`sentence_rhythm` not `register`/`rhythm`), Part 9, Part 10
- **Models:** `src/questfoundry/models/fill.py`
- **Audit-report section to fill:** `## FILL`
- **Notes for subagent:** Voice Document field names were corrected from `register`/`rhythm` to `voice_register`/`sentence_rhythm` in PR #1379. Confirm FILL prompts use the new names.
- **Commit subject:** `docs(reports): prompt-vs-spec audit — FILL stage findings`

### Task 12: Audit DRESS prompts

- **Prompts:** `prompts/templates/dress_discuss.yaml`, `dress_summarize.yaml`, `dress_serialize.yaml`, `dress_brief.yaml`, `dress_brief_batch.yaml`, `dress_codex.yaml`, `dress_codex_batch.yaml`, `dress_codex_spoiler_check.yaml`
- **Procedure doc:** `docs/design/procedures/dress.md`
- **Ontology sections:** Part 1, Part 9, Part 11 (DRESS art direction, EntityVisuals, IllustrationBrief, CodexEntry)
- **Models:** `src/questfoundry/models/dress.py`
- **Audit-report section to fill:** `## DRESS`
- **Notes for subagent:** The DRESS spec gained R-3.9 (partial-DRESS as graceful degradation, distinct from R-3.8's "DRESS skipped") in PR #1377. Confirm DRESS prompts that mention skipped DRESS reflect both rules.
- **Commit subject:** `docs(reports): prompt-vs-spec audit — DRESS stage findings`

### Task 13: Audit SHIP

- **Prompts:** SHIP has no LLM prompts (it's deterministic). Audit SHIP-adjacent text:
  - export-format error messages in `src/questfoundry/export/validation.py` for repair-gap-style improvements
  - the codeword playability warning text in `src/questfoundry/export/context.py`
  - the partial-DRESS warning text in the same file
- **Procedure doc:** `docs/design/procedures/ship.md`
- **Models:** none (SHIP has no LLM-facing schemas)
- **Audit-report section to fill:** `## SHIP`
- **Notes for subagent:** SHIP is deterministic — there's no model to fail. The audit dimension that applies is `drift` (do user-facing error messages cite current rule numbers?) and `terminology`. If the section ends up empty, write a one-line "SHIP has no LLM prompts; no audit findings" entry and update the summary row to 0/0/0/0.
- **Commit subject:** `docs(reports): prompt-vs-spec audit — SHIP stage findings`

---

## Task 14: Fill the overall summary table and stage close-out

**Files:**
- Modify: `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md`

- [ ] **Step 1: Fill the summary table from each stage's stage-summary block**

Replace the `_TBD_` cells with actual counts. Status column: `clean` (no findings), `pending fixes` (findings exist), or `done` (after fix PRs land).

- [ ] **Step 2: Add an executive summary paragraph at the top of the report**

Write 3–5 sentences naming: total prompts audited, total hard findings, which stages had the most drift, which spec gaps surfaced, and the recommended PR sequencing.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/reports/2026-04-25-prompt-spec-audit.md
git commit -m "$(cat <<'EOF'
docs(reports): prompt-vs-spec audit — close out Phase 2

Filled the overall summary table and added an executive summary.
All 8 stages audited. Phase 3 fix PRs follow per
docs/superpowers/plans/2026-04-25-prompt-spec-audit.md Task 16.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: File Phase 3 tracking issues per stage

**Files:**
- No file changes — uses `gh` CLI.

**Why:** Per the spec-compliance audit pattern, fix work is tracked as one epic per stage with cluster sub-issues for logically grouped findings. The audit report drives this.

- [ ] **Step 1: For each stage in the audit report with one or more `hard` or `soft` findings, file an epic**

```bash
gh issue create --title "[prompts] <stage>: epic" --label "area:prompts,area:<stage>,prompt-audit" --milestone "M-prompts-<stage>" --body "$(cat <<'EOF'
Bringing <stage> prompts in line with the authoritative specs.

Findings live in `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md` §<stage>.

**Child cluster issues:** (filled in as cluster issues are filed)

This epic tracks bringing <stage> prompts into compliance. Per CLAUDE.md §Design Doc Authority, specs supersede prompts.
EOF
)"
```

Repeat per stage. SHIP epic only opens if Task 13 surfaced findings.

- [ ] **Step 2: For each cluster of related findings within a stage, file a cluster issue**

A "cluster" = findings that share a fix (e.g. all FILL prompts referencing `register` instead of `voice_register` are one cluster). The audit subagent's stage summary suggests cluster splits.

```bash
gh issue create --title "[prompts] <stage>: <cluster summary>" --label "area:prompts,area:<stage>,prompt-audit" --milestone "M-prompts-<stage>" --body "$(cat <<'EOF'
**Spec rules:** <citations>
**Spec reference:** \`docs/design/procedures/<stage>.md\` §<section>
**Report section:** \`docs/superpowers/reports/2026-04-25-prompt-spec-audit.md\` §<stage>

**Current state:** <quote from audit>

**Gap:** <what's wrong>

**Recommended fix:** <verbatim wording from audit>

**Prompt files:** <list>

**Parent epic:** #<epic_number>
EOF
)"
```

- [ ] **Step 3: Commit (no file changes — this is a `git commit --allow-empty` to create a checkpoint)**

```bash
git commit --allow-empty -m "$(cat <<'EOF'
chore(prompts): file Phase 3 tracking issues per stage

Per-stage epics + cluster issues filed for findings from the
prompt-vs-spec audit report. Phase 3 fix PRs follow.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Per-stage fix PRs (demand-driven workflow)

**Files:** stage-specific prompt YAMLs under `prompts/templates/<stage>_*.yaml`, plus any related Pydantic validator messages.

**Why:** This is one logical task per stage that has findings. The actual edits are dictated by the audit report and the cluster issues. The plan documents the workflow but doesn't enumerate specific edits.

For each stage in pipeline order with open cluster issues:

- [ ] **Step 1: Branch + open draft PR**

```bash
git checkout main && git pull --ff-only
git checkout -b prompts/<stage>-audit-fixes
gh pr create --draft --title "fix(prompts): <stage> — bring prompts in line with spec" --body "<links to epic + cluster issues + audit report section>"
```

- [ ] **Step 2: For each cluster issue in the stage's epic, edit the named prompt file(s)**

Apply the verbatim fix wording from the audit report. Re-dispatch the prompt-engineer subagent on each modified prompt as an inline review before commit:

```
You are the QuestFoundry prompt-engineer. Inline review of:
prompts/templates/<file>.yaml (current diff: <paste git diff>).

Confirm the fix addresses cluster issue #<id>'s finding without
introducing new ones. Return your standard findings block.
```

If the inline review finds new issues, fix them in the same commit.

- [ ] **Step 3: Commit per cluster (one cluster = one commit)**

```bash
git add prompts/templates/<file>.yaml
git commit -m "fix(prompts): <stage> — <cluster summary> (closes #<cluster_issue>)"
```

- [ ] **Step 4: Push, ready for review, merge, repeat for the next stage**

After each stage's PR merges, smoke re-run that stage on a small project (or `projects/murder1/` if applicable) to confirm no halt. Log the outcome in the epic close-out.

```bash
gh issue close <epic_number> --comment "All cluster issues closed via PR #<pr_number>. Smoke re-run on <project> passed: <stage> completed without halt."
```

---

## Self-review notes

This plan was reviewed against the spec at
`docs/superpowers/specs/2026-04-25-prompt-spec-audit-design.md`:

- **Spec coverage:** every spec section maps to a task. Phase 1 → Tasks 1–4. Phase 2 → Tasks 5–14. Phase 3 → Tasks 15–16. Phase 4 (smoke re-run) → folded into Task 16 Step 4 per stage.
- **Bite-sized check:** Tasks 1, 3, 5 are small (one file each). Tasks 2 and 6–13 each dispatch a subagent — that's a single logical unit even though the subagent itself does substantial reading. Task 16 is intentionally a meta-workflow (one logical task per stage), not a single 5-minute step.
- **Type/name consistency:** the audit-report section headings (`## DREAM`, `## SEED`, etc.) are referenced consistently across Tasks 5, 6–13, and 15. The subagent file path (`.claude/agents/prompt-engineer.md`) is consistent across Tasks 1, 2, 6–13, and 16.
- **Subagent registration caveat:** Tasks 2 and 6 note that `subagent_type: general-purpose` may need to be used with the role inlined as the prompt, until the named subagent is registered with Claude Code. This is realistic for a fresh subagent file.
- **Murder1 fix scope:** Task 3 fixes the validator's error message rather than the prompt template, because the validator is what composes the runtime feedback. The prompt template is also worth a mention in the audit (Task 8) for completeness, but the runtime fix is in the Pydantic code.
