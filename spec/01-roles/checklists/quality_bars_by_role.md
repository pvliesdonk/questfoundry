# Quality Bars by Role — Quick Checklists (Layer 1)

> **Use:** Run these before asking Gatekeeper to pre-gate or gatecheck. Keep it light, keep it
> moving.  
> **Bars (see Layer-0 `QUALITY_BARS.md`):** **Integrity**, **Reachability**, **Nonlinearity**,
> **Gateways**, **Style**, **Presentation**, **Accessibility**.

---

## How to read this

- **Primary bars** = the ones you’re expected to _move_ for your slice.
- **Self-check (≤3 min)** = do these before handoff.
- **Typical fails** = what trips this role most often.
- **If red, route to…** = where to escalate (see `../interfaces/escalation_rules.md`).

---

## Showrunner (SR)

**Primary bars:** Integrity (process), Presentation (front-matter hygiene)  
**Self-check:**

- TU is _small_ and roles awakened match the slice.
- Cold snapshot/View identified; _no_ Hot in exports.
- Dormancy noted: `deferred:*` tags set or cleared.  
  **Typical fails:** Scope creep; mixing Hot/Cold.  
  **If red, route to…** Gatekeeper (bar mapping), owners per lane.

---

## Gatekeeper (GK)

**Primary bars:** Integrity, Presentation, Accessibility, Gateways sanity  
**Self-check:**

- Anchors/links resolve; no orphans.
- No spoilers/technique on surfaces.
- Alt/captions present where required; labels localizable.
- Diegetic gate phrasing present; no meta.  
  **Typical fails:** “Option locked” text; caption technique leaks; untranslated labels breaking
  anchors.  
  **If red, route to…** Owner role (Style/Plot/Binder/Translator/Art/Audio), SR to coordinate.

---

## Plotwright (PW)

**Primary bars:** Reachability, Nonlinearity, Gateways  
**Self-check:**

- Choices are **contrastive**; verbs signal intent.
- Loops return **with difference** (state/affordance/stakes).
- Gates have **world reasons** and ≥1 **fair path**.  
  **Typical fails:** Near-synonym options; unfair gates; retcon pressure.  
  **If red, route to…** Lore (reasons/constraints), Style (labels), Researcher (plausibility).

---

## Scene Smith (SS)

**Primary bars:** Style (at line level), Presentation, Gateways phrasing  
**Self-check:**

- Brief honored (goal, stakes, choice intents).
- **Diegetic** refusals; no meta/mechanics.
- Choice list isolated; add one micro-context line if needed.  
  **Typical fails:** Expository codex in prose; “Go / Proceed” options.  
  **If red, route to…** Plot (structure), Style (pattern), Curator (entry), PN (cadence).

---

## Style Lead (ST)

**Primary bars:** Style, Presentation, Accessibility (readability)  
**Self-check:**

- Choice labels sharpened; PN patterns updated.
- Banned phrases avoided; examples recorded in Addendum.
- Captions one line, atmospheric; alt concrete nouns/relations.  
  **Typical fails:** Fixing structure with prose; idioms that won’t localize.  
  **If red, route to…** Plot (structure), Translator (idiom/anchors), PN (performance).

---

## Lore Weaver (LW)

**Primary bars:** Integrity (canon consistency), Gateways (world reasons)  
**Self-check:**

- Hooks answered in **Canon Pack (Hot)**; invariants noted.
- **Player-safe summaries** produced for neighbors.
- Knowledge ledger updated (who knows what, when).  
  **Typical fails:** Canon bleeding to surfaces; topology edits from lore.  
  **If red, route to…** Plot (structure), Curator (taxonomy), Researcher (evidence posture).

---

## Codex Curator (CC)

**Primary bars:** Presentation (player-safe), Integrity (anchors), Accessibility  
**Self-check:**

- Entry skeleton: Overview → Usage → Context → See also → Notes → Lineage.
- Crosslinks resolve; no orphans; slugs stable.
- Terms match Style register; Translator notes added if needed.  
  **Typical fails:** Spoiler backstory; meta hints about gates.  
  **If red, route to…** Lore (summary), Style (register), Binder (anchors).

---

## Researcher (RS)

**Primary bars:** Integrity (evidence posture), Presentation (neutral phrasing)  
**Self-check:**

- **Short answer + 2–5 citations**; posture labeled.
- **Neutral surface lines** provided if certainty < corroborated.
- Risks/mitigations named (safety/culture/legal).  
  **Typical fails:** Over-claiming; jargon that needs codex without warning.  
  **If red, route to…** Lore/Plot (implications), Curator (anchor), Style (wording).

---

## Art Director (AD)

**Primary bars:** Presentation (captions), Accessibility, Navigation signposting  
**Self-check:**

- Each slot has **purpose** (clarify/recall/mood/signpost).
- Caption one line, atmospheric; **no technique**; alt guidance concrete.
- Inclusion criteria clear; determinism logging requirements **off-surface**.  
  **Typical fails:** Spectacle without purpose; captions that hint at twists.  
  **If red, route to…** Style (phrasing), Gatekeeper (hygiene), Curator (terms).

---

## Illustrator (IL)

**Primary bars:** Presentation, Accessibility  
**Self-check:**

- Readable at target size; subject/relation clear.
- Alt text one sentence; **no technique on surfaces**.
- Determinism logs complete **off-surface** (if promised).  
  **Typical fails:** Technique leakage; illegible affordances.  
  **If red, route to…** Art Director (plan tweak), Style (caption), Gatekeeper.

---

## Audio Director (AuD)

**Primary bars:** Presentation (captions/text equivalents), Accessibility, Pace  
**Self-check:**

- Purpose per cue (**clarify/recall/mood/signpost/pace**).
- Caption bracketed, one line, portable; **safety notes** present.
- Placement won’t mask prose; plan-only allowed (`deferred:audio`).  
  **Typical fails:** Meta stingers; startle onsets; idioms that won’t travel.  
  **If red, route to…** Style/Translator (phrasing), Producer (feasibility), Gatekeeper.

---

## Audio Producer (AuP)

**Primary bars:** Presentation (caption parity), Accessibility (safety), Repro (off-surface)  
**Self-check:**

- Matches plan **purpose/placement/intensity/duration**.
- Loudness normalized; fades before choice lists; no clipping/startle.
- Determinism logs **off-surface**; captions align with what’s heard.  
  **Typical fails:** Technique in captions; overlong beds masking prose.  
  **If red, route to…** Audio Director (plan adjust), Style/Translator (caption), Gatekeeper.

---

## Translator (TR)

**Primary bars:** Presentation (labels/anchors), Style (register), Accessibility  
**Self-check:**

- **Register map** written; PN patterns localized.
- Choices remain **contrastive**; gates **diegetic**.
- Anchors/labels survive binding; coverage % reported; safe fallbacks set.  
  **Typical fails:** Anchor collisions; over-domestication; meta leakage.  
  **If red, route to…** Style (register), Curator (taxonomy), Binder (anchors).

---

## Book Binder (BB)

**Primary bars:** Integrity (anchors/links), Presentation (front-matter), Accessibility  
**Self-check:**

- Single **Cold** snapshot; no Hot bleed.
- Front-matter states snapshot, options, coverage, accessibility.
- Crosslinks Manuscript ↔ Codex ↔ Captions resolve.  
  **Typical fails:** Silent text edits; mixed snapshots.  
  **If red, route to…** Owners (fix upstream), SR (options), GK (export hygiene).

---

## Player-Narrator (PN)

**Primary bars:** Presentation (diegetic gates), Style (cadence), Accessibility  
**Self-check (dry-run):**

- Reads from View; **no improvisation**.
- Logs `choice-ambiguity`, `gate-friction`, `nav-bug`, `tone-wobble`, `translation-glitch`,
  `accessibility`.
- Micro-recaps ≤2 lines; in-voice.  
  **Typical fails:** Meta explanations; over-recap; inventing text.  
  **If red, route to…** Style (patterns), Plot/Scene (labels/structure), Binder (nav).

---

## Bar→Owner cheat (where to aim first)

- **Integrity:** Binder (anchors), Gatekeeper (checks), Curator (links), Plot (topology)
- **Reachability/Nonlinearity:** Plotwright → Scene (execution)
- **Gateways:** Plotwright/Lore (reason), Style (phrasing), Researcher (constraints)
- **Style:** Style Lead → Scene/PN/Translator (apply)
- **Presentation:** Gatekeeper (hygiene), Style (surface wording), Art/Audio (captions)
- **Accessibility:** Gatekeeper (policy), Art/Audio (alt/text-equiv), Translator (readability)

---

## One-minute green pass (any role, before handoff)

- [ ] **Player-safe**: no spoilers, no internals (seeds/models/DAW/flags).
- [ ] **Contrastive**: choices read as different intents.
- [ ] **Diegetic**: gates phrased in-world.
- [ ] **Anchors**: links/ids stable; no “see here”.
- [ ] **Accessibility**: alt/captions present if applicable; sentences readable.
- [ ] **Traceability**: TU noted; deferred tags set if plan-only/partial.

---
