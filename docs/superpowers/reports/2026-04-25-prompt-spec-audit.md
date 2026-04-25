# Prompt-vs-Spec Audit Report

**Date started:** 2026-04-25
**Spec:** `docs/superpowers/specs/2026-04-25-prompt-spec-audit-design.md`
**Plan:** `docs/superpowers/plans/2026-04-25-prompt-spec-audit.md`

## How to read this report

One section per stage in pipeline order. Each section was produced by
dispatching the `prompt-engineer` subagent
(`.claude/agents/prompt-engineer.md`) scoped to that stage's prompts +
procedure doc + ontology references + Pydantic models.

Findings use the audit dimensions from the spec:

- **drift** — prompt encodes outdated terminology or rule citations
- **repair-gap** — validation feedback names missing fields without
  echoing expected values (the murder1 failure shape)
- **sm-fragile** — implicit instructions, no examples, ambiguous
  phrasing, no sandwich repetition
- **schema-skew** — prompt-vs-Pydantic mismatch
- **terminology** — deprecated names (e.g. codeword vs state_flag)

Severities: **hard** (causes pipeline halt or contract violation),
**soft** (degraded output but pipeline survives), **info** (noted, no
action).

A `spec-gap` finding means the prompt encodes a constraint not in the
spec — per CLAUDE.md docs-first, the spec is updated first.

## Overall summary

(Executive summary and totals filled in by Task 14 after all 8 stage
sections land.)

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

> **Already-known finding from Phase 1:** `serialize_seed_sections.yaml`
> per-path-beats repair-loop didn't echo expected `also_belongs_to`
> value. Fixed in PR #1384. The subagent should still re-audit this
> prompt in Task 8 to catch any other findings (the smoke test
> already surfaced 5 bonus items).

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
