# Role Readiness — Pre-Flight Checklists (Layer 1)

> **Purpose:** Prevent doomed laps. Before opening a TU or accepting one, run the _pre-flight_ for
> your role. Start only when inputs are real, neighbors are awake (or explicitly dormant), and you
> can finish a small slice without guessing.

**Normative refs:**  
`../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SOURCES_OF_TRUTH.md` ·
`../../00-north-star/SPOILER_HYGIENE.md` · `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
· `../interfaces/pair_guides.md` · `../interfaces/dormancy_signals.md` ·
`../interfaces/escalation_rules.md`

**Conventions:** **Hot** (private, spoilers) · **Cold** (player-safe) · **Bars** (Integrity,
Reachability, Nonlinearity, Gateways, Style, Presentation, Accessibility) · **TU** (timeboxed task
unit)

---

## 0) Universal pre-flight (any role)

- [ ] **TU slice named** (small enough for one sitting); loop chosen (Story Spark, Style Tune-up,
      …).
- [ ] **Owner/A count clear:** who is **R** and who is **A** per `../raci/by_loop.md`.
- [ ] **Dormancy set:** optional roles marked `deferred:*` or woken with rubric
      (`dormancy_signals.md`).
- [ ] **Snapshot context known:** which **Cold** snapshot your work must align to (no Hot in Views).
- [ ] **Pairing plan:** handshakes you’ll need are noted (`pair_guides.md`).
- [ ] **Exit criteria stated:** which Bars must go green for this slice.
- [ ] **Traceability ready:** TU id created / to be created; where notes land (Addendum, Pack, View
      Log, ADR).

---

## 1) Showrunner (SR)

**Inputs you must have**

- [ ] A concrete slice + loop; roles to wake vs keep dormant (with rubric score).
- [ ] Last **Cold** snapshot and any View goal/options.
- [ ] Known Bar pressure (what you expect to go red/yellow).

**Neighbors to ping**

- [ ] Gatekeeper (pre-gate), owners of impacted lanes.

**Ready to start if**

- [ ] Work can complete without inventing canon/structure in the SR seat.
- [ ] You can ship a merge or a clear follow-up TU in ≤90 minutes.

**Red flags → escalate**

- [ ] Scope creeping across ≥3 domains → call cross-domain per `escalation_rules.md`.

---

## 2) Gatekeeper (GK)

**Inputs you must have**

- [ ] TU deliverables or a representative sample.
- [ ] Bar map from SR (which Bars to check first).
- [ ] Access to **Cold** snapshot / draft View if export.

**Neighbors to ping**

- [ ] Owners of the slice; SR for timing.

**Ready to start if**

- [ ] You can sample the slice and name **smallest viable fixes** per Bar.

**Red flags → escalate**

- [ ] Systemic failures (anchors, labels) → SR + Binder; consider ADR if policy-level.

---

## 3) Plotwright (PW)

**Inputs you must have**

- [ ] Clear **Story Spark** scope: hub/loop/gateways.
- [ ] Style register constraints; any Lore **player-safe** summaries; known Research posture.

**Neighbors to ping**

- [ ] Scene, Style, Gatekeeper (pre-gate), Lore, Curator.

**Ready to start if**

- [ ] You can express choices as **intent labels** without prose drafting.
- [ ] At least one **fair path** exists (or you know which hook to file).

**Red flags → escalate**

- [ ] Gate fairness needs canon → pair with Lore; if blocked, escalate via SR.

---

## 4) Scene Smith (SS)

**Inputs you must have**

- [ ] Plot briefs (goal • beats • choice intents • outcomes).
- [ ] Style addendum; Curator anchors; Lore summaries.

**Neighbors to ping**

- [ ] Style (phrasing pressure), Gatekeeper (pre-gate), Curator (terms).

**Ready to start if**

- [ ] You can draft 1–3 sections **without guessing structure or canon**.

**Red flags → escalate**

- [ ] Ambiguity that would require spoilers; ask Style for micro-context or Lore summary.

---

## 5) Style Lead (ST)

**Inputs you must have**

- [ ] Concrete text needing **pattern** fixes (not structural rewrites).
- [ ] PN friction/Translator notes if relevant.

**Neighbors to ping**

- [ ] PN, Translator, Gatekeeper.

**Ready to start if**

- [ ] You can supply **patterns + exemplars** owners can apply themselves.

**Red flags → escalate**

- [ ] Pattern requires topology or canon change → SR huddle.

---

## 6) Lore Weaver (LW)

**Inputs you must have**

- [ ] A cluster of accepted **hooks**; current knowledge ledger.
- [ ] Research posture if facts matter.

**Neighbors to ping**

- [ ] Plotwright, Curator, Style; Researcher if posture unclear.

**Ready to start if**

- [ ] You can answer hooks **in Hot** and extract **player-safe** summaries.

**Red flags → escalate**

- [ ] Fix implies major topology change → SR + PW.

---

## 7) Codex Curator (CC)

**Inputs you must have**

- [ ] Lore **player-safe** summaries; Style register; prior entries.
- [ ] Anchor policy (slugs) from Binder.

**Neighbors to ping**

- [ ] Translator (glossary), Gatekeeper (Presentation), Binder (anchors).

**Ready to start if**

- [ ] You can write entries with **Overview → Usage → Context → See also → Notes → Lineage**.

**Red flags → escalate**

- [ ] Explaining term requires spoilers → ask Lore or defer.

---

## 8) Researcher (RS)

**Inputs you must have**

- [ ] The **smallest useful question** + stakes + where it appears.
- [ ] Owner tags for handoff.

**Neighbors to ping**

- [ ] Lore/Plot/Style/Curator/Translator as needed.

**Ready to start if**

- [ ] You can deliver a **short answer + 2–5 citations + posture + neutral phrasing**.

**Red flags → escalate**

- [ ] Safety/legal risks without neutral fallback → SR immediate.

---

## 9) Art Director (AD)

**Inputs you must have**

- [ ] Scene slices; Style motifs; Curator terms; Translator caption constraints.
- [ ] Decision: **plan-only** or **plan+render** this loop.

**Neighbors to ping**

- [ ] Illustrator, Style, Gatekeeper, Translator.

**Ready to start if**

- [ ] You can name **purpose** for each slot (clarify/recall/mood/signpost) and write **caption/alt
      guidance**.

**Red flags → escalate**

- [ ] Caption requires spoilers or technique → ST/CC fix or defer (`deferred:art`).

---

## 10) Illustrator (IL)

**Inputs you must have**

- [ ] Approved **Art Plans** + shotlist; Style motifs; target sizes.
- [ ] Caption/alt guidance.

**Neighbors to ping**

- [ ] Art Director first; then Style/Translator; Gatekeeper for alt check.

**Ready to start if**

- [ ] You can render without inventing composition intent; determinism logging path exists
      **off-surface**.

**Red flags → escalate**

- [ ] Legibility/safety fights the plan → propose tweak before rendering.

---

## 11) Audio Director (AuD)

**Inputs you must have**

- [ ] Scene slices; PN rhythm concerns; Researcher safety notes; Style register.
- [ ] Decision: **plan-only** or **plan+production** this loop.

**Neighbors to ping**

- [ ] Audio Producer, Style, Translator, Gatekeeper, PN.

**Ready to start if**

- [ ] You can specify **purpose • placement • duration • captions • safety** per cue.

**Red flags → escalate**

- [ ] Caption idiom won’t travel; startle risk can’t be mitigated → ST/TR/SR.

---

## 12) Audio Producer (AuP)

**Inputs you must have**

- [ ] Approved **Audio Plans**; loudness target; file specs; caption text.
- [ ] Off-surface determinism log location.

**Neighbors to ping**

- [ ] Audio Director; Style/Translator if captions change; Gatekeeper for spot-check.

**Ready to start if**

- [ ] You can render 3–5 cues **without altering intent** and log reproducibility.

**Red flags → escalate**

- [ ] Plan conflicts with PN cadence/safety → Director + SR.

---

## 13) Translator (TR)

**Inputs you must have**

- [ ] Stable source slice (Cold); Style addendum; Curator glossary.
- [ ] Target coverage goal.

**Neighbors to ping**

- [ ] Binder (anchors), Style (register), PN (patterns), Gatekeeper (labels).

**Ready to start if**

- [ ] You can ship **Register Map + glossary slice**, and localize a **bounded** set
      (choices/headings/captions).

**Red flags → escalate**

- [ ] Anchor collisions or taxonomy shifts required → Binder/Curator + SR.

---

## 14) Book Binder (BB)

**Inputs you must have**

- [ ] A single **Cold** snapshot; SR export options (art/audio/locale coverage).
- [ ] Label/slug policies; Translator anchor diffs.

**Neighbors to ping**

- [ ] Translator (multilingual binds), Gatekeeper (export hygiene), SR (options).

**Ready to start if**

- [ ] You can **dry bind** without editing text; front matter fields are known.

**Red flags → escalate**

- [ ] Integrity red (anchors/links) → owners upstream; no quick edits here.

---

## 15) Player-Narrator (PN)

**Inputs you must have**

- [ ] A bound **View**; Style PN patterns; Translator slice if localized.
- [ ] Clarification on _dry-run_ vs _audience_ mode.

**Neighbors to ping**

- [ ] Style (cadence), Translator (phrasing), Binder (nav), SR (logging path).

**Ready to start if**

- [ ] You can perform **without improvising**; know how to tag notes (`choice-ambiguity`,
      `gate-friction`, …).

**Red flags → escalate**

- [ ] Repeated gate/choice friction or nav breaks → SR + owners per `escalation_rules.md`.

---

## 16) One-liner TU openers (copy/paste)

```

TU: <loop> — <slice> · Roles awake: <codes> · Bars: <...>
Exit: <bars green + artifact(s)> · Dormant: deferred:<...>
Pre-gate: @gatekeeper · Handshakes: <pair guides>

```

---
