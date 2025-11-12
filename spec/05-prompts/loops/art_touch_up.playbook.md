# Art Touch-up — Executable Loop Playbook

**Category:** Asset **Abbreviation:** AT **Schema:**
<https://questfoundry.liesdonk.nl/schemas/art_plan.schema.json>

## Purpose

Decide what to illustrate and why, then (optionally) produce illustrations that match style and
narrative intent—without leaking spoilers. This loop supports plan-only merges when the Illustrator
is dormant. Outcome: An Art Plan (subjects, purpose, composition intent, captions, constraints) and,
if active, art renders with determinism notes. Ready for Gatekeeper checks and merge to Cold (plans
may merge as deferred; renders merge only on full pass).

## Activation Criteria (Showrunner)

- New chapter/act needs anchoring visuals
- Scene gained iconic imagery (after Story Spark / Scene Smith pass)
- Style Lead requests motif reinforcement
- Replacement or upgrade of an existing illustration
- Export goal includes art plan/renders

Showrunner opens/attaches a Trace Unit (TU): `tu-art-<scope>`. Confirm Art Director/Illustrator
activation; they may be dormant individually.

## RACI Matrix

| Role            | Assignment | Responsibilities                                                                                               |
| --------------- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| Art Director    | R          | Select scenes/subjects; state purpose; write composition intent and spoiler-safe captions; specify constraints |
| Showrunner      | A          | Sets scope; decides plan-only vs render; sequences merge                                                       |
| Illustrator     | R          | If active: produce renders; iterate to intent; log determinism parameters if promised                          |
| Style Lead      | C          | Ensure visual language matches register/motifs; flag drift                                                     |
| Gatekeeper      | C          | Check Presentation Safety, Style, and Determinism (when applicable)                                            |
| Codex Curator   | I          | Ensure captions stay player-safe                                                                               |
| Player-Narrator | I          | Receives art for reference during narration (surface wording stays safe)                                       |

## Inputs

- Current Cold snapshot (canon, codex, style guardrails)
- Target sections/scenes (Hot drafts acceptable for planning)
- Lore Deepening notes (spoiler flags), Codex entries (player-safe wording)
- Accessibility and PN considerations (alt text patterns, diegetic references)

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

### Step 1: Select & Justify (Art Director)

For each proposed image:

- Subject (who/what), Scene anchor (section), Purpose (clarify, foreshadow, mood), Spoiler risk
  (low/med/high)
- If risk > low, consider alternate subject or move to plan-only

### Step 2: Write the Art Plan (Art Director)

- Composition intent (framing, focal points, motion cues)
- Caption (player-safe; no twist reveals)
- Constraints (aspect, palette/motif hooks, negative constraints to avoid clichés)
- Accessibility notes (alt text guidance)

### Step 3: Style Alignment (Style Lead)

Tune plan to guardrails; add motif tie-ins; veto style drift.

### Step 4: Render (Illustrator, if active)

Produce candidate renders; refine to intent. Record determinism log: seed, prompt/version,
model/build, aspect, steps/CFG/parameters, post-process chain. Provide alt text (succinct,
descriptive, spoiler-safe).

### Step 5: Pre-Gate (Gatekeeper)

- Style: guardrails and tone
- Presentation Safety: no spoilers; captions and alt text safe; PN can reference diegetically
- Determinism: logs complete when promised; else mark non-deterministic explicitly

### Step 6: Package & Handoff

Attach Art Plan and (if produced) renders + logs to the TU. Notify Binder about inclusion options
for the next export.

## Deliverables

- **Art Plan** (per image):
  - Subject, Section anchor, Purpose, Composition intent, Caption (player-safe), Constraints,
    Accessibility notes, Spoiler risk
- **Renders** (optional):
  - Image files + Determinism log (if promised) + alt text
- **Pre-gate note** (Gatekeeper): pass/fail + remediations

## Success Criteria

- Each image has a clear narrative purpose (clarity, recall anchor, mood, or signposting)
- Captions are spoiler-safe, readable, and consistent with Style Lead
- If renders are included, determinism logs are sufficient (when promised) and alt text is present
- Gatekeeper reports green on Style/Presentation (and Determinism if applicable)

## Failure Modes & Remedies

- **Caption spoils twist** → Move reveal to canon notes; rewrite caption atmospheric
- **Style drift** → Adjust plan with Style Lead; iterate render
- **No determinism log (promised)** → Either re-render with logging or declare non-deterministic
  explicitly and do not promise reproducibility
- **Over-detailed caption** → Trim to mood/purpose; avoid internal labels or gate logic
- **Inaccessible surface** → Add/repair alt text; ensure contrast and motion-safety notes for
  animated media (if any)

## Quality Bars Pressed

**Primary:** Presentation (captions/alt), Accessibility, Style

**Secondary:** Determinism (when renders provided), Integrity (navigation signposting)

## Handoffs

- **To Book Binder:** Art plans and/or renders ready for inclusion in exports with specified options
  (include art plans, include art renders, both, or neither)
- **To Gatecheck:** After plan/renders packaged, full gatecheck before merge to Cold
