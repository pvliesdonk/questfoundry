# Art Touch-up — Plan and/or Add Illustrations

**Purpose**  
Decide _what_ to illustrate and _why_, then (optionally) produce illustrations that match style and
narrative intent—without leaking spoilers. This loop supports **plan-only** merges when the
**Illustrator** is dormant.

**Outcome**  
An **Art Plan** (subjects, purpose, composition intent, captions, constraints) and, if active, **art
renders** with determinism notes. Ready for Gatekeeper checks and merge to **Cold** (plans may merge
as **deferred**; renders merge only on full pass).

---

## 1) Triggers (Showrunner)

- New chapter/act needs anchoring visuals.
- Scene gained iconic imagery (after **Story Spark** / **Scene Smith** pass).
- Style Lead requests motif reinforcement.
- Replacement or upgrade of an existing illustration.
- Export goal includes art plan/renders.

**Activation**  
Open/attach a **Trace Unit (TU)**: `tu-art-<scope>`. Confirm **Art Director**/**Illustrator**
activation; they may be **dormant** individually.

---

## 2) Inputs

- Current **Cold** snapshot (canon, codex, style guardrails).
- Target sections/scenes (Hot drafts acceptable for planning).
- Lore Deepening notes (spoiler flags), Codex entries (player-safe wording).
- Accessibility and PN considerations (alt text patterns, diegetic references).

---

## 3) Roles & Responsibilities

- **Art Director (R)**
  - Select scenes/subjects; state purpose (narrative/UX); write **composition intent** and
    **spoiler-safe captions**; specify constraints (aspect, framing, palette cues).
- **Style Lead (C)**
  - Ensure visual language matches register/motifs; flag drift.
- **Illustrator (R, optional)**
  - Produce renders; iterate to intent; log **determinism parameters** (if promised).
- **Gatekeeper (C)**
  - Check **Presentation Safety**, **Style**, and **Determinism** (when applicable).
- **Showrunner (A)**
  - Sets scope; decides plan-only vs render; sequences merge.

_(Codex Curator and PN are informed to ensure surface wording stays player-safe.)_

---

## 4) Procedure

1. **Select & Justify (Art Director)**
   - For each proposed image:
     - **Subject** (who/what), **Scene anchor** (section), **Purpose** (clarify, foreshadow, mood),
       **Spoiler risk** (low/med/high).
     - If risk > low, consider **alternate subject** or move to **plan-only**.

2. **Write the Art Plan (Art Director)**
   - **Composition intent** (framing, focal points, motion cues).
   - **Caption** (player-safe; no twist reveals).
   - **Constraints** (aspect, palette/motif hooks, negative constraints to avoid clichés).
   - **Accessibility notes** (alt text guidance).

3. **Style Alignment (Style Lead)**
   - Tune plan to guardrails; add motif tie-ins; veto style drift.

4. **Render (Illustrator, if active)**
   - Produce candidate renders; refine to intent.
   - Record **determinism log**: seed, prompt/version, model/build, aspect, steps/CFG/parameters,
     post-process chain.
   - Provide **alt text** (succinct, descriptive, spoiler-safe).

5. **Pre-Gate (Gatekeeper)**
   - **Style**: guardrails & tone.
   - **Presentation Safety**: no spoilers; captions and alt text safe; PN can reference
     diegetically.
   - **Determinism**: logs complete when promised; else mark **non-deterministic** explicitly.

6. **Package & Handoff**
   - Attach **Art Plan** and (if produced) **renders + logs** to the TU.
   - Notify **Binder** about inclusion options for the next export.

---

## 5) Deliverables (Hot)

- **Art Plan** (per image):
  - Subject, Section anchor, Purpose, Composition intent, Caption (player-safe), Constraints,
    Accessibility notes, Spoiler risk.
- **Renders** (optional):
  - Image files + **Determinism log** (if promised) + **alt text**.
- **Pre-gate note** (Gatekeeper): pass/fail + remediations.

---

## 6) Merge Path (summary)

- **Plan-only**: May merge to **Cold** as **deferred:art** if Style/Presentation pass and captions
  are safe.
- **With renders**: Merge to **Cold** only if **Style + Presentation** pass; **Determinism** pass
  when promised.
- **Showrunner** stamps Cold snapshot; TU updated.

---

## 7) Success Criteria

- Each image has a clear **narrative purpose** (clarity, recall anchor, mood, or signposting).
- Captions are **spoiler-safe**, readable, and consistent with Style Lead.
- If renders are included, **determinism logs** are sufficient (when promised) and **alt text** is
  present.
- Gatekeeper reports **green** on Style/Presentation (and Determinism if applicable).

---

## 8) Failure Modes & Remedies

- **Caption spoils twist** → Move reveal to canon notes; rewrite caption atmospheric.
- **Style drift** → Adjust plan with Style Lead; iterate render.
- **No determinism log (promised)** → Either re-render with logging or declare non-deterministic
  explicitly and **do not** promise reproducibility.
- **Over-detailed caption** → Trim to mood/purpose; avoid internal labels or gate logic.
- **Inaccessible surface** → Add/repair alt text; ensure contrast and motion-safety notes for
  animated media (if any).

---

## 9) Inclusion in Exports (Binder)

- **Options**: include **art plans**, **renders**, both, or neither.
- Plans appear as **captions + intent**; renders with alt text.
- Snapshot ID and inclusion switches recorded in the **View Log**.

---

## 10) RACI (quick)

| Task                         | R            | A          | C                        | I           |
| ---------------------------- | ------------ | ---------- | ------------------------ | ----------- |
| Choose subjects & write plan | Art Director | Showrunner | Style Lead               | Curator, PN |
| Style alignment              | Style Lead   | Showrunner | Art Director             | Gatekeeper  |
| Render & log                 | Illustrator  | Showrunner | Art Director, Style Lead | Gatekeeper  |
| Pre-gate                     | Gatekeeper   | Showrunner | Style Lead               | All         |
| Merge & export opts          | Showrunner   | Showrunner | Gatekeeper, Binder       | PN          |

---

**TL;DR**  
Pick the right scenes, say _why_ the image exists, keep captions spoiler-safe, and either ship the
**plan** now or the **plan + render** with proper determinism and alt text. Style stays tight;
players see mood, not machinery.
