# Audio Pass — Plan and/or Add Sound

**Purpose**  
Decide _what the audience should hear and why_—ambience, foley, stingers, or voice—then (optionally)
produce audio that fits style and narrative intent without leaking spoilers. This loop supports
**plan-only** merges when the **Audio Producer** is dormant.

**Outcome**  
An **Audio Plan** (cues, purpose, timing/placement, captions/text-equivalents, safety notes) and, if
active, **audio assets** with reproducibility notes. Ready for Gatekeeper checks and merge to
**Cold** (plans may merge as **deferred**; assets merge only on full pass).

---

## 1) Triggers (Showrunner)

- A chapter/scene needs mood scaffolding or clarifying sound cues.
- Style Lead requests motif reinforcement via sound.
- Replacement/upgrade of existing sounds or VO.
- Export targets include audio plan/assets.

**Activation**  
Open/attach a **Trace Unit (TU)**: `tu-audio-<scope>`. Confirm **Audio Director**/**Audio Producer**
activation; either may be **dormant**.

---

## 2) Inputs

- **Cold** snapshot (canon, codex, style guardrails).
- Target scenes/sections (Hot drafts acceptable for planning).
- PN Principles (diegetic references; no plumbing).
- Accessibility requirements (text equivalents, loudness safety).
- Localization posture (Translator active or dormant).

---

## 3) Roles & Responsibilities

- **Audio Director (R)**
  - Choose cue targets and **why** (clarity, mood, signposting); specify **placement** (entry/exit),
    **duration**, and **intensity**; write **player-safe cue descriptions** and
    captions/text-equivalents.
- **Style Lead (C)**
  - Ensure audio language aligns with register/motifs; veto drift.
- **Audio Producer (R, optional)**
  - Create/arrange/mix assets; export masters; log **reproducibility notes** (DAW/session, key
    plugins, versions, sample rate/bit depth, stems).
- **PN (C)**
  - Confirm cues can be referenced **diegetically** when appropriate; never expose technique.
- **Translator (C, optional)**
  - Flag VO/linguistic content and localization needs.
- **Gatekeeper (C)**
  - Check **Presentation Safety** (no spoilers; safe levels), **Style**, and
    **Determinism**/**Reproducibility** (when promised).
- **Showrunner (A)**
  - Scope the pass; decide plan-only vs asset production; sequence merge.

---

## 4) Procedure

1. **Select & Justify (Audio Director)**  
   For each proposed cue:
   - **Cue ID & Scene anchor** (section or moment).
   - **Purpose** (clarify affordance, intensify stakes, transition, recall motif).
   - **Type** (ambience, foley, stinger, VO).
   - **Spoiler risk** (low/med/high) and mitigation (alternate cue or plan-only).

2. **Write the Audio Plan (Audio Director)**
   - **Description (player-safe)**: what the listener perceives, not how it was made.
   - **Placement**: entry/exit, loop or one-shot, suggested duration.
   - **Intensity curve**: low/med/high, ramp/fade guidance.
   - **Motif ties**: how the cue threads house motifs.
   - **Captions/text-equivalents** for accessibility.
   - **Safety notes**: avoid sudden peaks; caution tags for harsh sounds.
   - **Localization notes** (if VO): dialect, register, terms to preserve.

3. **Style Alignment (Style Lead)**
   - Tune language and motif ties; approve or request revisions.

4. **Produce Assets (Audio Producer, if active)**
   - Create cues; export masters; provide **stems** when relevant.
   - Record **reproducibility**: DAW name/version, plugin list/versions, session sample rate/bit
     depth, key settings or presets, normalization target.
   - Provide **text equivalents** and any **lyrics avoidance** if copyrighted texts would otherwise
     appear.

5. **Pre-Gate (Gatekeeper)**
   - **Style**: cohesive with book register.
   - **Presentation Safety**: spoiler-safe cue descriptions; reasonable loudness; caption coverage.
   - **Determinism/Reproducibility**: logs sufficient when promised; otherwise mark
     **non-deterministic** explicitly.

6. **Package & Handoff**
   - Attach **Audio Plan** and (if produced) **assets + logs** to the TU.
   - Notify **Binder** about inclusion options for the next export.

---

## 5) Deliverables (Hot)

- **Audio Plan** (per cue):
  - Cue ID, Scene anchor, Purpose, Type, Player-safe description, Placement, Intensity curve, Motif
    ties, Captions/text-equivalents, Safety & Localization notes, Spoiler risk.
- **Audio Assets** (optional):
  - Files (masters) + **stems** (if applicable) + **reproducibility notes**.
- **Pre-gate note** (Gatekeeper): pass/fail + remediations.

---

## 6) Merge Path (summary)

- **Plan-only**: May merge to **Cold** as **deferred:audio** if **Style** and **Presentation
  Safety** pass.
- **With assets**: Merge to **Cold** only if **Style + Presentation** pass and **Reproducibility**
  info is adequate (when promised).
- **Showrunner** stamps Cold snapshot; TU updated.

---

## 7) Success Criteria

- Each cue has a clear **narrative purpose** (clarity, mood, or signposting).
- Descriptions and captions are **player-safe** and consistent with Style Lead.
- Assets (if included) have **reproducibility notes**; loudness is reasonable; text equivalents
  exist.
- Gatekeeper reports **green** on Style/Presentation (and Reproducibility if applicable).

---

## 8) Failure Modes & Remedies

- **Cue telegraphs a twist** → Move detail to canon notes; keep description atmospheric.
- **Technique on surface** → Remove DAW/plugin talk; keep diegetic.
- **Loudness shocks** → Tame transients; add fade/ramp; include safety note.
- **Missing repro notes (promised)** → Re-export with logs or mark non-deterministic and do not
  promise reproducibility.
- **Untranslatable VO idioms** → Coordinate with Translator; adjust script; provide alt phrasing.

---

## 9) Inclusion in Exports (Binder)

- **Options**: include **audio plans**, **assets**, both, or neither.
- Plans appear as **cue descriptions** and captions; assets shipped with text equivalents.
- The **View Log** records snapshot ID and inclusion switches.

---

## 10) RACI (quick)

| Task                     | R              | A          | C                          | I              |
| ------------------------ | -------------- | ---------- | -------------------------- | -------------- |
| Choose cues & write plan | Audio Director | Showrunner | Style Lead, PN             | Gatekeeper     |
| Style alignment          | Style Lead     | Showrunner | Audio Director             | Gatekeeper     |
| Produce & log            | Audio Producer | Showrunner | Audio Director, Style Lead | Gatekeeper     |
| Pre-gate                 | Gatekeeper     | Showrunner | Style Lead                 | All            |
| Merge & export opts      | Showrunner     | Showrunner | Gatekeeper, Binder         | PN, Translator |

---

**TL;DR**  
Score the book with intention: say _why_ the sound exists, keep surfaces spoiler-safe, and either
ship the **plan** now or the **plan + assets** with proper logs and captions. Mood, not machinery.
