# Translation Pass — Executable Loop Playbook

**Category:** Asset **Abbreviation:** TP **Schema:**
<https://questfoundry.liesdonk.nl/schemas/language_pack.schema.json>

## Purpose

Create or update a player-safe translation of the manuscript/codex surfaces while preserving PN
boundaries, style intent, and navigation. This loop supports plan-only merges (glossary/style kit)
when the full slice isn't ready. Outcome: A Language Pack (glossary, style notes, localized
surfaces, open issues) ready for Gatekeeper checks and merge to Cold. Exports can include complete
or partial translations flagged accordingly.

## Activation Criteria (Showrunner)

- New target language requested
- Significant style/canon changes warrant a translation refresh
- Accessibility or market goals require multilingual exports

Showrunner opens/attaches a Trace Unit (TU): `tu-translation-<lang>-<scope>`. Confirm Translator
activation (others may be dormant).

## RACI Matrix

| Role            | Assignment | Responsibilities                                                                                                       |
| --------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| Translator      | R          | Produce the Language Pack: glossary, style transfer notes, localized surfaces; flag untranslatables and cultural risks |
| Showrunner      | A          | Scope completeness (chapters only, full book, codex subset) and merge timing                                           |
| Style Lead      | C          | Approve register mapping (formal/informal "you"), motif equivalence, idiom strategy                                    |
| Player-Narrator | C          | Validate diegetic gate phrasing patterns in target language; no internals exposed                                      |
| Codex Curator   | C          | Ensure cross-refs and titles localize consistently; avoid spoiler drift                                                |
| Gatekeeper      | C          | Check Presentation Safety, Integrity (links resolve), Style adherence                                                  |
| Book Binder     | I          | Includes translated slices in exports with coverage flags                                                              |

## Inputs

- Cold snapshot (manuscript/codex surfaces), PN Principles, Style guardrails
- Current glossary/terminology (if any); motif kit
- Cultural notes from Lore/Style (tone, registers to preserve/avoid)
- Known untranslatable idioms list (if exists)

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

### Step 1: Scope & Coverage Plan (Translator + Showrunner)

Decide slice (full book, acts, codex subset). Set coverage target and time box; mark partial outputs
as `incomplete`.

### Step 2: Glossary First

Create/refresh glossary with Style Lead. Decide register (T/V distinction, dialect); lock decisions
here.

### Step 3: Segment & Localize

Translate player surfaces: no internal labels, no spoilers, preserve PN diegesis. Keep hyperlinks
and anchors intact; choice labels stay distinct and clear.

### Step 4: Motif & Idiom Pass (Style Lead)

Validate motif resonance; solve idioms with functionally equivalent phrases.

### Step 5: PN Phrasing Check

Confirm gate enforcement phrasing is diegetic and natural in target language.

### Step 6: Link Audit (Curator)

Ensure cross-refs resolve to localized targets; add stubs if scoped.

### Step 7: Pre-Gate (Gatekeeper)

Presentation Safety (no leaks), Integrity (links), Style (tone/voice).

### Step 8: Package

Assemble Language Pack; compute coverage and list open issues.

## Deliverables

- **Language Pack** for `<lang>`:
  - Glossary (term → approved translation; part-of-speech; usage notes; do-not-translate list;
    examples)
  - Register Map (pronoun system choice; honorifics; tone equivalents; swear policy)
  - Motif Equivalence (how house motifs render)
  - Idiom Strategy (list of idioms → literal/functional equivalents or rewrites)
  - Localized Surfaces (manuscript sections and choice labels; codex titles/summaries; captions and
    alt text)
  - Open Issues (untranslatables needing rewrite upstream; glossary gaps; cultural cautions)
  - Traceability (TU-ID, snapshot ID, coverage % by section count and codex entries)
- **Pre-gate note:** Gatekeeper's findings and remediation

## Success Criteria

- Register and tone feel native; PN voice preserved
- Links/cross-refs resolve; choice labels remain distinct
- Glossary stable; idioms handled; motifs resonate
- No spoilers or internal labels leaked; accessibility text localized

## Failure Modes & Remedies

- **Literalism breaks tone** → Recast with Style Lead; add glossary usage examples
- **Untranslatable idiom** → Provide functional equivalent or upstream rewrite suggestion
- **Broken anchors** → Sync IDs; re-run link audit
- **PN leaks plumbing** → Replace with diegetic phrasing; Gatekeeper blocks until fixed
- **Inconsistent terms** → Lock glossary; batch-fix before merge

## Quality Bars Pressed

**Primary:** Presentation (labels, anchors), Accessibility, Style (register)

**Secondary:** Integrity (link resolution)

## Handoffs

- **To Book Binder:** Language Pack with coverage flags (`complete`/`incomplete`); Binder includes
  in exports with front matter recording snapshot ID, languages included, and coverage %
- **To Gatecheck:** After Language Pack packaged, full gatecheck before merge to Cold

**Example (EN → NL):**

**Source choice:** "Slip through maintenance, or face the foreman." **Glossary:**
`foreman → voorman` (register: colloquial), `maintenance → onderhoudsdek` **Localized:** "Glip door
het onderhoudsdek, of sta de voorman te woord." **PN diegesis:** gate checks phrased as "De voorman
knikt naar het token op je borst."
