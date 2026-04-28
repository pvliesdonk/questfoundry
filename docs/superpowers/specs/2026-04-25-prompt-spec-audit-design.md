# Prompt-vs-Spec Audit Design

**Date:** 2026-04-25
**Status:** Design approved, awaiting implementation plan
**Trigger:** murder1 SEED run halted because the per-path-beats serialize prompt's repair feedback didn't echo the expected `also_belongs_to` value, so a small Ollama model lost the constraint across three retry attempts. The spec-compliance audit (completed 2026-04-25) brought design docs and pipeline code into alignment but explicitly postponed prompt updates. Likely many other prompts have similar drift.

## Goal

1. Build a reusable **`prompt-engineer` subagent** that encodes prompt-engineering expertise (sandwich pattern, repair-loop quality, structured-output specifics, defensive examples, valid-ID injection) and small-model failure modes.
2. Run a **full audit** of every YAML in `prompts/templates/` against the authoritative specs, producing a per-stage audit document.
3. Land **fix PRs per stage** for the drift the audit surfaces.

## Authoritative inputs

- `docs/design/how-branching-stories-work.md` — narrative model
- `docs/design/story-graph-ontology.md` — graph / data model (Y-shape Part 8 guard rails are particularly load-bearing)
- `docs/design/procedures/{dream,brainstorm,seed,grow,polish,fill,dress,ship}.md` — per-stage algorithm specs

CLAUDE.md §Design Doc Authority is the tie-breaker: spec wins; if a prompt encodes a constraint the spec doesn't, the spec is updated first.

## Non-goals

- Audit prompts outside `prompts/templates/` (provider-specific image prompts in `providers/image_*` are not in scope).
- Audit thin discuss/summarize wrappers that carry no spec-derived constraints — the subagent triages this on first read of each file.
- Build prompts for stages or features that don't exist yet.

## Architecture

### `prompt-engineer` subagent

A Markdown agent file under `.claude/agents/prompt-engineer.md`. Knowledge body covers:

- **Prompt-engineering fundamentals** — based on promptingguide.ai patterns: zero/few-shot, chain-of-thought, sandwich (repeat critical instructions at start AND end), persona/role framing.
- **Small-model failure modes** — per CLAUDE.md §10:
  - Ambiguity loss across long context
  - Optional-vs-required field confusion
  - Implicit instructions that get dropped
  - Loss of constraint-to-value mapping (the murder1 root cause)
  - Schema-name collisions with prose
- **QuestFoundry-specific patterns** — direct citations of CLAUDE.md sections:
  - §6 Valid ID Injection Principle
  - §7 Defensive Prompt Patterns (good/bad examples)
  - §8 Context Enrichment Principle (every relevant ontology field)
  - §9 Prompt Context Formatting (no Python repr in LLM-facing text)
  - §10 Small Model Prompt Bias (fix the prompt before blaming the model)
- **Repair-loop authoring** — feedback messages must echo the expected value, not just name the missing field. Example library covering common cases (ID prefix wrong, missing required field, wrong literal value).
- **Structured-output specifics** — JSON_MODE vs TOOL strategy implications, Pydantic model alignment, how to phrase constraints the schema can't express.

The subagent has read access to all repo files and is invoked either:
- Inline by the controller (me) when editing a single prompt
- As a per-stage audit dispatcher (the audit pass uses 8 instances, one per stage)

### Audit dimensions

Each prompt is checked along five axes. The subagent produces structured findings with severity:

| Dimension | Severity tag | Examples |
|---|---|---|
| **Spec accuracy** | `drift` | Field renamed in spec but old name in prompt; deprecated phase name |
| **Repair-loop quality** | `repair-gap` | Validation feedback names missing field but not the expected value (murder1 pattern) |
| **Small-model resilience** | `sm-fragile` | Implicit instructions, no examples, ambiguous constraint phrasing, no sandwich repetition |
| **Schema alignment** | `schema-skew` | Prompt describes fields the Pydantic model doesn't have, or omits required ones |
| **Drift markers** | `terminology` | "codeword" where spec now uses "state_flag", "passage_from" where spec uses "grouped_in" |

Severity rubric:
- **hard**: known cause of pipeline halt or contract violation. Fix this PR.
- **soft**: degraded output quality but pipeline survives. Fix in same epic if cheap.
- **info**: noted, no action required.

### Per-stage organisation

The audit doc has one section per stage (DREAM, BRAINSTORM, SEED, GROW, POLISH, FILL, DRESS, SHIP). Each section lists:

- Prompts in scope for that stage
- Per-prompt findings, grouped by severity
- A consolidated **Spec drift** sub-section (if the audit found cases where the spec needs updating before the prompt)
- Cross-references to the murder1 failure when relevant (already known finding)

Tracking-issue structure mirrors the spec-compliance audit: one epic per stage (`area:prompts area:<stage>`), one cluster issue per logical group of related findings within a stage.

## Process

### Phase 1 — Build the subagent

1. Write `.claude/agents/prompt-engineer.md` with the knowledge body above.
2. Exercise it against the murder1 failing prompt (`prompts/templates/serialize_seed_sections.yaml` per-path-beats section) as the smoke test. The subagent must:
   - Identify the repair-loop gap (`repair-gap` finding, severity `hard`)
   - Propose the specific fix wording
   - Cite CLAUDE.md §9 / §10 and the spec's Y-shape guard rail
3. Commit subagent + smoke-test record. No prompt changes yet.

### Phase 2 — Per-stage audit pass

For each stage in pipeline order:

1. Dispatch one `prompt-engineer` subagent scoped to that stage.
2. Subagent reads:
   - All `prompts/templates/<stage>_*.yaml`
   - `docs/design/procedures/<stage>.md`
   - Relevant ontology sections (Part 1, 6, 8 etc.)
   - The Pydantic models for that stage's structured outputs (`src/questfoundry/models/<stage>.py`)
3. Subagent produces a findings report appended to `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md`.
4. Controller reviews and files tracking issues.

Stage order matches pipeline order so each subagent can reference upstream stages' decisions when relevant.

### Phase 3 — Fix PRs

Per stage:

1. Open per-stage epic referencing the audit doc section.
2. File cluster issues per logical group of findings.
3. Implement fixes — prompt edits, repair-loop wording, defensive examples — using the subagent itself for review before each commit (per existing CLAUDE.md guidance).
4. PRs follow the established merge workflow (draft → `/pr-review-toolkit:review-pr` → ready).

If the audit reveals a spec gap, that gap is fixed in its own commit before any prompt change in the same PR.

### Phase 4 — Smoke re-run

Once a stage's prompt fixes land, re-run the murder1 project (or a smaller stand-in) for that stage to confirm the failure mode no longer triggers. This is per-stage, not end-of-audit, so feedback lands fast.

## Deliverables

| # | Artifact | Owner |
|---|---|---|
| 1 | `.claude/agents/prompt-engineer.md` | Phase 1 |
| 2 | `docs/superpowers/reports/2026-04-25-prompt-spec-audit.md` | Phase 2 (one section per stage, appended over multiple sub-runs) |
| 3 | Per-stage epic + cluster issues on GitHub | Phase 2 close-out |
| 4 | Per-stage fix PRs | Phase 3 |
| 5 | Per-stage smoke-rerun confirmation | Phase 4 (logged in epic close-out) |

## Risks

- **Subagent quality drift**: if the subagent gives shallow advice, the audit findings won't be actionable. Smoke test in Phase 1 (murder1 prompt) proves it can produce an actionable finding before we depend on it for 47 more.
- **Spec turns out to be wrong**: per docs-first, that's expected. The audit doc is allowed to surface spec gaps and pause prompt edits until the spec catches up. (Same pattern as the spec-compliance audit's GROW Phase 4 sidestep.)
- **Token spend**: 8 stage-scope subagents reading procedure doc + ontology + N prompts is non-trivial. Cap each stage subagent's scope to its own stage and only reference the ontology sections relevant to its node types.
- **Audit-doc churn**: single combined doc could conflict if multiple stages are audited concurrently. Process audits sequentially in pipeline order to avoid this; each appends a section to the doc and commits before the next starts.

## Success criteria

1. The murder1 SEED failure mode is fixed and no longer halts the pipeline.
2. Every prompt in `prompts/templates/` has either a clean-bill audit entry OR a fix PR landed.
3. The `prompt-engineer` subagent is wired into the existing PR-review pipeline so future prompt edits get the same treatment.
4. CLAUDE.md's references to the prompt-engineer subagent (§6 / §7 / §8 / §9 / §10) point to a real, in-repo asset.
