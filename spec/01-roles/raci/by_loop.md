# RACI by Micro-Loop (Layer 1)

> **Purpose:** Clear ownership per loop so small passes stay fast and player-safe. _R_ = Responsible
> (does the work). _A_ = Accountable (final decision/merge). _C_ = Consulted (two-way). _I_ =
> Informed (one-way).

**Normative refs:**  
`../../00-north-star/LOOPS/README.md` · individual loop guides in `../../00-north-star/LOOPS/*.md`  
Layer-0 bars: `QUALITY_BARS.md` · surfaces: `SOURCES_OF_TRUTH.md` · hygiene: `SPOILER_HYGIENE.md` ·
access: `ACCESSIBILITY_AND_CONTENT_NOTES.md`

**Roles (short codes used below):**  
SR = Showrunner · GK = Gatekeeper · PW = Plotwright · SS = Scene Smith · ST = Style Lead · LW = Lore
Weaver · CC = Codex Curator · RS = Researcher · AD = Art Director · IL = Illustrator · AuD = Audio
Director · AuP = Audio Producer · TR = Translator · BB = Book Binder · PN = Player-Narrator

---

## 1) Story Spark (topology slice)

**Bar focus:** Reachability, Nonlinearity, Gateways, Presentation (choice labels)

- **R:** PW
- **A:** SR
- **C:** SS, ST, GK, LW, CC
- **I:** PN, BB, RS

**Notes:** Pre-gate by GK before SS drafts. Hooks to LW/CC as needed.

---

## 2) Style Tune-up (voice, register, PN patterns)

**Bar focus:** Style, Presentation, Accessibility (readability)

- **R:** ST
- **A:** SR
- **C:** PN, TR, GK
- **I:** SS, PW, CC

**Notes:** ST supplies patterns + exemplars; owners apply edits in their lanes.

---

## 3) Hook Harvest (collect/triage)

**Bar focus:** Integrity (traceability), Planning hygiene

- **R:** SR
- **A:** SR
- **C:** All creation roles (PW, SS, ST, LW, CC, AD, AuD, TR, RS)
- **I:** GK, BB, PN

**Notes:** Triage to TUs; set dormancy tags (`deferred:*`) per rubric.

---

## 4) Lore Deepening (answer hooks → canon in Hot)

**Bar focus:** Integrity (no contradictions), Gateways (world reasons)

- **R:** LW
- **A:** SR
- **C:** PW, RS, ST
- **I:** CC, GK

**Notes:** Output **Canon Pack (Hot)** + **player-safe summaries** to CC.

---

## 5) Codex Expansion (player-safe entries)

**Bar focus:** Presentation (spoiler-safe), Integrity (anchors), Accessibility

- **R:** CC
- **A:** SR
- **C:** LW, ST, TR, GK
- **I:** BB, PN

**Notes:** Entries: Overview → Usage → Context → See also → Notes → Lineage.

---

## 6) Art Touch-up (plan-first; optional render)

**Bar focus:** Presentation (captions/alt), Accessibility, Navigation signposting

- **R (plan):** AD
- **R (render, if any):** IL
- **A:** SR
- **C:** ST, GK, TR, CC
- **I:** BB, PN

**Notes:** Plan-only allowed → tag `deferred:art`. No technique on surfaces.

---

## 7) Audio Pass (plan-first; optional production)

**Bar focus:** Presentation (captions/text-equiv), Accessibility, Pace

- **R (plan):** AuD
- **R (produce, if any):** AuP
- **A:** SR
- **C:** ST, GK, TR, PN
- **I:** BB

**Notes:** Plan-only allowed → tag `deferred:audio`. Startle risk documented.

---

## 8) Translation Pass (localize slice)

**Bar focus:** Presentation (labels, anchors), Accessibility, Style (register)

- **R:** TR
- **A:** SR
- **C:** ST, CC, GK, PN
- **I:** BB

**Notes:** Partial delivery OK: **Register Map + Glossary slice** → `deferred:translation`.

---

## 9) Binding Run (assemble a View from Cold)

**Bar focus:** Integrity (anchors/links), Presentation (no internals), Accessibility (front-matter)

- **R:** BB
- **A:** SR
- **C:** TR, GK
- **I:** PN, ST, CC, AD, AuD

**Notes:** Cold-only. Front matter states snapshot, options, coverage, accessibility.

---

## 10) Narration Dry-Run (performance QA)

**Bar focus:** Presentation (diegetic gates), Style (cadence), Accessibility

- **R:** PN
- **A:** SR
- **C:** ST, TR, BB
- **I:** GK

**Notes:** Log **PN Playtest Notes** (player-safe): `choice-ambiguity`, `gate-friction`, etc.

---

## 11) Full Production Run (coordinated multi-slice pass)

**Bar focus:** All (Integrity, Reachability, Nonlinearity, Gateways, Style, Presentation,
Accessibility)

- **R:** Domain owners per slice (PW, SS, ST, LW, CC, AD/IL, AuD/AuP, TR, BB)
- **A:** SR
- **C:** GK (pre-gate + gatecheck)
- **I:** PN, RS

**Notes:** Break into sub-TUs; don’t run omnibus changes without pre-gates.

---

## Appendix — Quick Matrix (who is usually A/R)

| Role | Typical A                              | Typical R (primary loops)         |
| ---- | -------------------------------------- | --------------------------------- |
| SR   | Most micro-loops; merge/view decisions | Hook Harvest, loop setup          |
| GK   | —                                      | Pre-gate notes; Gatecheck reports |
| PW   | —                                      | Story Spark                       |
| SS   | —                                      | Drafts/rewrites from PW briefs    |
| ST   | —                                      | Style Tune-up; PN patterns        |
| LW   | —                                      | Lore Deepening                    |
| CC   | —                                      | Codex Expansion                   |
| RS   | —                                      | Research memos (by TU)            |
| AD   | —                                      | Art Touch-up (plan)               |
| IL   | —                                      | Art Touch-up (renders)            |
| AuD  | —                                      | Audio Pass (plan)                 |
| AuP  | —                                      | Audio Pass (production)           |
| TR   | —                                      | Translation Pass                  |
| BB   | —                                      | Binding Run                       |
| PN   | —                                      | Narration Dry-Run                 |

**Reminder:** Optional tracks (Art, Audio, Translation, Research) may be **dormant** per
`../interfaces/dormancy_signals.md`. Use `deferred:*` tags and front-matter notes when shipping
without them.
