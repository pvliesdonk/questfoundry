# Audio Pass — Executable Loop Playbook

**Category:** Asset **Abbreviation:** AP **Schema:**
<https://questfoundry.liesdonk.nl/schemas/audio_plan.schema.json>

## Purpose

Decide what the audience should hear and why—ambience, foley, stingers, or voice—then (optionally)
produce audio that fits style and narrative intent without leaking spoilers. This loop supports
plan-only merges when the Audio Producer is dormant. Outcome: An Audio Plan (cues, purpose,
timing/placement, captions/text-equivalents, safety notes) and, if active, audio assets with
reproducibility notes. Ready for Gatekeeper checks and merge to Cold (plans may merge as deferred;
assets merge only on full pass).

## Activation Criteria (Showrunner)

- A chapter/scene needs mood scaffolding or clarifying sound cues
- Style Lead requests motif reinforcement via sound
- Replacement/upgrade of existing sounds or VO
- Export targets include audio plan/assets

Showrunner opens/attaches a Trace Unit (TU): `tu-audio-<scope>`. Confirm Audio Director/Audio
Producer activation; either may be dormant.

## RACI Matrix

| Role            | Assignment | Responsibilities                                                                                                                     |
| --------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Audio Director  | R          | Choose cue targets and why; specify placement, duration, intensity; write player-safe cue descriptions and captions/text-equivalents |
| Showrunner      | A          | Scope the pass; decide plan-only vs asset production; sequence merge                                                                 |
| Audio Producer  | R          | If active: create/arrange/mix assets; export masters; log reproducibility notes                                                      |
| Style Lead      | C          | Ensure audio language aligns with register/motifs; veto drift                                                                        |
| Player-Narrator | C          | Confirm cues can be referenced diegetically when appropriate; never expose technique                                                 |
| Translator      | C          | If active: flag VO/linguistic content and localization needs                                                                         |
| Gatekeeper      | C          | Check Presentation Safety (no spoilers; safe levels), Style, and Determinism/Reproducibility (when promised)                         |

## Inputs

- Cold snapshot (canon, codex, style guardrails)
- Target scenes/sections (Hot drafts acceptable for planning)
- PN Principles (diegetic references; no plumbing)
- Accessibility requirements (text equivalents, loudness safety)
- Localization posture (Translator active or dormant)

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

### Step 1: Select & Justify (Audio Director)

For each proposed cue:

- Cue ID and Scene anchor (section or moment)
- Purpose (clarify affordance, intensify stakes, transition, recall motif)
- Type (ambience, foley, stinger, VO)
- Spoiler risk (low/med/high) and mitigation (alternate cue or plan-only)

### Step 2: Write the Audio Plan (Audio Director)

- Description (player-safe): what the listener perceives, not how it was made
- Placement: entry/exit, loop or one-shot, suggested duration
- Intensity curve: low/med/high, ramp/fade guidance
- Motif ties: how the cue threads house motifs
- Captions/text-equivalents for accessibility
- Safety notes: avoid sudden peaks; caution tags for harsh sounds
- Localization notes (if VO): dialect, register, terms to preserve

### Step 3: Style Alignment (Style Lead)

Tune language and motif ties; approve or request revisions.

### Step 4: Produce Assets (Audio Producer, if active)

Create cues; export masters; provide stems when relevant. Record reproducibility: DAW name/version,
plugin list/versions, session sample rate/bit depth, key settings or presets, normalization target.
Provide text equivalents and any lyrics avoidance if copyrighted texts would otherwise appear.

### Step 5: Pre-Gate (Gatekeeper)

- Style: cohesive with book register
- Presentation Safety: spoiler-safe cue descriptions; reasonable loudness; caption coverage
- Determinism/Reproducibility: logs sufficient when promised; otherwise mark non-deterministic
  explicitly

### Step 6: Package & Handoff

Attach Audio Plan and (if produced) assets + logs to the TU. Notify Binder about inclusion options
for the next export.

## Deliverables

- **Audio Plan** (per cue):
  - Cue ID, Scene anchor, Purpose, Type, Player-safe description, Placement, Intensity curve, Motif
    ties, Captions/text-equivalents, Safety and Localization notes, Spoiler risk
- **Audio Assets** (optional):
  - Files (masters) + stems (if applicable) + reproducibility notes
- **Pre-gate note** (Gatekeeper): pass/fail + remediations

## Success Criteria

- Each cue has a clear narrative purpose (clarity, mood, or signposting)
- Descriptions and captions are player-safe and consistent with Style Lead
- Assets (if included) have reproducibility notes; loudness is reasonable; text equivalents exist
- Gatekeeper reports green on Style/Presentation (and Reproducibility if applicable)

## Failure Modes & Remedies

- **Cue telegraphs a twist** → Move detail to canon notes; keep description atmospheric
- **Technique on surface** → Remove DAW/plugin talk; keep diegetic
- **Loudness shocks** → Tame transients; add fade/ramp; include safety note
- **Missing repro notes (promised)** → Re-export with logs or mark non-deterministic and do not
  promise reproducibility
- **Untranslatable VO idioms** → Coordinate with Translator; adjust script; provide alt phrasing

## Quality Bars Pressed

**Primary:** Presentation (captions/text-equiv), Accessibility, Style

**Secondary:** Determinism (when assets provided), Pace

## Handoffs

- **To Book Binder:** Audio plans and/or assets ready for inclusion in exports with specified
  options (include audio plans, include audio assets, both, or neither)
- **To Gatecheck:** After plan/assets packaged, full gatecheck before merge to Cold
