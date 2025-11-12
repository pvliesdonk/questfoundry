# Playbook — Audio Pass

**Use when:** You want to **plan and/or add sound** (ambience, foley, stingers, VO) that clarifies,
anchors recall, or sets mood—without leaking spoilers. Works even if the **Audio Producer** is
dormant (plan-only merge).

**Outcome:** An **Audio Plan** (cues, purpose, placement, captions/text-equivalents, safety notes)
and, if active, **assets** with reproducibility notes. Ready for gatecheck and Cold merge (plans may
land as **deferred:audio**).

---

## 1) One-minute scope (Showrunner)

- [ ] Define **slice** (which chapters/sections/moments).
- [ ] Decide **plan-only** vs **plan+assets**.
- [ ] Mark roles **active/dormant** (Audio Director, Audio Producer, Translator).
- [ ] Open TU: `tu-audio-<scope>` and state **why** these cues exist
      (clarity/recall/mood/signposting).

---

## 2) Inputs you need on screen

- **Cold snapshot** (canon, codex, style guardrails).
- Target **sections/moments** (Hot drafts acceptable for planning).
- **PN Principles** (diegetic references; no plumbing).
- Accessibility policy (captions/text equivalents, loudness safety).
- Localization posture (Translator on/off).

---

## 3) Do the thing (compact procedure)

**Audio Director (R)**

1. **Select cues** (per moment): scene anchor, **purpose** (clarify/recall/mood/transition),
   **type** (ambience/foley/stinger/VO), **spoiler risk** (low/med/high).
2. Write **Player-safe Description** (what the listener perceives, not technique).
3. Specify **Placement** (entry/exit), **duration/loop**, **intensity curve** (low/med/high;
   ramp/fade).
4. Add **Motif ties** and **Safety notes** (sudden peaks, harsh sounds).
5. Provide **Captions/Text-equivalents** (concise, descriptive).
6. Add **Localization notes** (if VO: register, dialect, terms to preserve).

**Style Lead (C)** 7. Tune plan language to register; ensure motif coherence.

**Audio Producer (R, optional)** 8. If active, **produce** cues; export masters (and stems if
useful).  
9. Create **Reproducibility Notes**: DAW/version, plugin list/versions, session sample rate/bit
depth, key presets/settings, normalization target.  
10. Ensure **text equivalents** for accessibility.

**PN (C)** 11. Confirm cues can be referenced **diegetically** when appropriate; never expose
technique.

**Gatekeeper (C)** 12. **Pre-gate**: Style, Presentation Safety (no spoilers, reasonable loudness,
caption coverage), Reproducibility (when promised).

**Audio Director (R)** 13. **Package** plan (+ assets/logs if any) into the TU.

---

## 4) Deliverables (Hot)

- **Audio Plan** (per cue)
  - Cue ID • Section/moment • **Purpose** • **Type** • **Player-safe description**
  - **Placement** • **Intensity curve** • **Motif ties**
  - **Captions/Text-equivalents** • **Safety & Localization notes** • **Spoiler risk**
- **Assets** (optional): files (masters) + stems (if any) + **Reproducibility notes**
- **Pre-gate note** (Gatekeeper): pass/fail + remediations

---

## 5) Hand-offs

- → **Binding Run**: inclusion options (plan, assets, both).
- → **Style Tune-up**: if cue wording clashes with register.
- → **Translation Pass**: VO or caption localization details.
- → **PN**: confirm diegetic references where cues are mentioned in-world.

---

## 6) Definition of “done” (for this play)

- [ ] Every cue has a **clear narrative purpose**.
- [ ] Descriptions & captions are **player-safe** and in register.
- [ ] **Safety** addressed (no shock peaks; warnings where needed).
- [ ] If assets included and reproducibility is promised, **notes are complete**.
- [ ] Gatekeeper pre-gate **green** (Style/Presentation; Reproducibility if applicable).
- [ ] TU updated; plan can merge as **deferred:audio** if Producer dormant.

---

## 7) Fast heuristics

- Prefer cues that **clarify affordances**, **anchor place**, or **telegraph transitions**.
- If **spoiler risk ≥ med**, ship **plan-only** now; produce later.
- Describe **what** is heard, not **how** it was made (no DAW/plugin talk on surfaces).
- Keep captions **short and specific**; avoid onomatopoeia spam.

---

## 8) RACI (micro)

| Task                | R              | A          | C                          | I              |
| ------------------- | -------------- | ---------- | -------------------------- | -------------- |
| Choose cues & plan  | Audio Director | Showrunner | Style Lead, PN             | Gatekeeper     |
| Style alignment     | Style Lead     | Showrunner | Audio Director             | Gatekeeper     |
| Produce & log       | Audio Producer | Showrunner | Audio Director, Style Lead | Gatekeeper     |
| Pre-gate            | Gatekeeper     | Showrunner | Style Lead                 | All            |
| Merge / export opts | Showrunner     | Showrunner | Gatekeeper, Binder         | PN, Translator |

---

## 9) Templates

**Audio Plan (per cue)**

```

Cue: <label>
Section/Moment: <anchor>
Purpose: <clarify | recall | mood | transition>
Type: <ambience | foley | stinger | VO>
Description (player-safe): <1–2 lines>
Placement: <enter/exit; loop/one-shot; duration>
Intensity: <low | med | high>  (ramp/fade: <notes>)
Motif ties: <house motifs it threads>
Captions/Text-equivalents: <concise, descriptive>
Safety: <peaks? harsh? warnings?>
Localization: <VO register/dialect; terms to preserve>
Spoiler risk: <low | med | high>  (if med/high: consider plan-only)
Lineage: TU <id>

```

**Reproducibility Notes (if promised)**

```

DAW/Version: <name vX.Y>   Sample Rate/Bit Depth: <48k/24b>
Plugins (versions): <list>  Key Presets/Settings: <notes>
Normalization: <target LUFS>  Exports: <formats>
Stems: <if provided>

```

---

## 10) Anti-patterns

- Cue **telegraphs a twist** via title/description.
- Technique on surface (“bitcrusher sweep”) instead of perception.
- **Startle peaks** or fatiguing loops without safety notes.
- VO that locks culture/idiom without Translator review.

---

**Cheat line (TU note):**  
“Audio Pass (Act I): 6 cues (shadow-side ambience, relay-hum transition, Dock 7 crowd, foreman gate
foley, medbay monitors, wormhole stinger). Captions & safety notes in place; 3 assets w/ repro
notes; 3 **deferred:audio** plans; Gatekeeper pre-gate green.”
