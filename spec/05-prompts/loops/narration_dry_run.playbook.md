# Narration Dry-Run — Executable Loop Playbook

**Category:** Export **Abbreviation:** NDR **Schema:**
<https://questfoundry.liesdonk.nl/schemas/view_log.schema.json>

## Purpose

Have the Player-Narrator (PN) play through the current Cold snapshot exactly as a player would
experience it—in-world, spoiler-safe—to surface UX issues in comprehension, choice clarity, pacing,
and diegetic gate enforcement. Outcome: A set of PN playtest notes and handoff tickets (Trace Units
or edit notes) that improve player-facing surfaces without rewriting canon or topology during the
dry-run itself.

## Activation Criteria (Showrunner)

- After a Binding Run exports a view on Cold (chapter/act/book)
- Before user playtests, recordings, or live demos
- When PN phrasing patterns or recaps were recently updated

Showrunner selects the same Cold snapshot used by the Binding Run and opens a TU:
`tu-pn-dryrun-<date|milestone>`.

## RACI Matrix

| Role            | Assignment | Responsibilities                                                                                           |
| --------------- | ---------- | ---------------------------------------------------------------------------------------------------------- |
| Player-Narrator | R          | Perform the book in-voice; enforce gates diegetically; log UX issues. No creative rewrites during the pass |
| Showrunner      | A          | Scope routes to test; decide depth (single path vs sample of branches)                                     |
| Gatekeeper      | C          | Classify issues against Presentation Safety, Integrity, Style bars                                         |
| Style Lead      | C          | Advise on voice/register adjustments; review PN phrasing patterns                                          |
| Book Binder     | C          | Investigate navigation issues (anchors, labels, layout)                                                    |
| Codex Curator   | C          | Address codex coverage gaps or cross-ref problems                                                          |
| Translator      | C          | If testing a localized slice, flag mistransfers in tone/links                                              |

## Inputs

- Exported bundle from the Binding Run (Markdown/HTML/EPUB/PDF)
- PN Principles (in-world, spoiler-safe, diegetic gates)
- Style guardrails (voice/register/motifs)
- Any Language Pack if testing a non-source language

PN reads Cold only—no Hot drafts, no canon notes.

## Validation Requirements (All Artifacts)

**CRITICAL:** All roles producing JSON artifacts in this loop MUST validate before handoff.

**Refer to:** `_shared/validation_contract.md` (file #1 in your kit)

**For every artifact-producing step:**

1. **Producer role** looks up artifact schema in `SCHEMA_INDEX.json`
2. **Producer role** runs preflight protocol (echo schema metadata + examples)
3. **Producer role** produces artifact with `"$schema"` field
4. **Producer role** validates and emits `validation_report.json`
5. **If validation fails:** STOP loop immediately, escalate to Showrunner
6. **Showrunner** verifies both files exist before accepting handoff:
   - `/out/artifact.json` (with valid `"$schema"` field)
   - `/out/artifact_validation_report.json` (with `"valid": true`)
7. **Only then** proceed to next step

**Hard gate:** Invalid artifacts STOP the loop. No exceptions.

**Showrunner responsibility:**

Before allowing any role-to-role handoff, verify:

- Artifact file exists and has `"$schema"` field
- validation_report.json exists
- validation_report.json shows `"valid": true` with empty `"errors": []`

If any validation fails, STOP the loop and escalate to human for guidance.

## Procedure (Message Sequences)

### Step 1: Route Plan (Showrunner + PN)

Choose representative paths: a hub route, a loop return, a gated branch, and at least one terminal.
If time-boxed, pick high-traffic sections and one "weird" edge case.

### Step 2: Live Read (PN)

Narrate in-voice; present choices clearly; perform recaps when natural. Enforce gateways without
exposing internals (see PN Principles). Do not improvise new content that changes meaning; note
issues instead.

### Step 3: Issue Logging (PN)

Tag each note with an issue type and a target (manuscript/codex/PN phrasing/layout). Use short,
actionable phrasing.

**Issue types (common):**

- `choice-ambiguity` — options unclear or indistinct
- `gate-friction` — diegetic enforcement feels clunky; suggests rephrase
- `recap-needed` — reader likely lost; suggest recap placement
- `codex-invite` — codex reference could aid comprehension here
- `leak-risk` — phrasing brushes against internals/spoilers
- `nav-bug` — link/anchor mismatch, breadcrumb confusion
- `tone-wobble` — register slips; motif absent where expected
- `accessibility` — missing alt text, risky audio cue, low contrast

**Example findings:**

- **`gate-friction`**: Found: "Access denied without CODEWORD: ASH." Fix: "The foreman scans your
  lapel. No union token—he shakes his head."
- **`recap-needed`**: Found: After three detours, stakes feel vague. Fix: Add a two-sentence recap
  before the hub: where we are, what changed, what's at risk.
- **`codex-invite`**: Found: Players stumble on salvage law nuance. Fix: PN offer: "You can recall
  station salvage rules" → link to codex summary.

### Step 4: Spot Checks (Gatekeeper + Binder + Curator + Style)

- Gatekeeper classifies severity and bar mapping
- Binder verifies navigation/format issues
- Curator checks codex links and coverage
- Style Lead proposes phrasing patterns or motif nudges (no structural edits)

### Step 5: Package & Handoff (Showrunner)

Convert notes into TUs or edit tasks routed to the right loop:

- Style Tune-up for tone/phrasing systems
- Codex Expansion for coverage
- Binding fixes for navigation/format
- Story Spark only if the issue actually reveals a topology flaw (rare)

## Deliverables

- **PN Playtest Notes** (human text), each item with:
  - Context: section/choice label (player-safe ref), path snippet
  - Issue type: from the list above
  - Suggested remedy: phrasing pattern, recap cue, codex invite, or fix target
- **Summary Sheet:**
  - Count by issue type and severity
  - Recommended follow-up loops
  - Any languages tested and findings

Do not alter Cold during the dry-run; changes happen in follow-up loops.

## Success Criteria

- Gates enforced diegetically without confusion or leaks
- Choices read as distinct and fair; recaps land where helpful
- Codex invites clarify, not spoil
- Navigation is smooth across all export formats
- A small, prioritized list of actionable follow-ups exists

## Failure Modes & Remedies

- **PN improvises fixes live** → Log issues; route to Style Tune-up or Binder; don't unilaterally
  rewrite
- **Recurring choice ambiguity** → Add micro-context in surface text; run a focused Style Tune-up
- **Gate phrasing leaks plumbing** → Replace with diegetic token/reputation/knowledge checks
- **Codex rabbit-holing** → Cap invites; keep the main path legible
- **Format-specific nav bugs** → Binder repairs anchors and re-exports all formats

## Quality Bars Pressed

**Primary:** Presentation (diegetic gates), Style (cadence), Accessibility

**Secondary:** Integrity (navigation coherence)

## Handoffs

- **To Style Tune-up:** Tone/phrasing patterns that need adjustment
- **To Codex Expansion:** Coverage gaps or cross-ref issues identified by PN
- **To Book Binder:** Navigation/format bugs requiring re-export
- **To Showrunner:** Summary of findings with prioritized follow-up TUs
