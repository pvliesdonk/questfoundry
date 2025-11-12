# Escalation Rules — How to Resolve Cross-Domain Conflicts (Layer 1, Human-Level)

> **Purpose:** Keep creative friction productive. This guide defines **when to escalate, to whom,
> with what bundle,** and the **decision records** we leave behind—without leaking spoilers or
> internals to player surfaces.

---

## 0) Normative references (Layer 0)

- Sources of Truth — `../../00-north-star/SOURCES_OF_TRUTH.md`
- Quality Bars — `../../00-north-star/QUALITY_BARS.md`
- PN Principles — `../../00-north-star/PN_PRINCIPLES.md`
- Spoiler Hygiene — `../../00-north-star/SPOILER_HYGIENE.md`
- Accessibility & Content Notes — `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Traceability — `../../00-north-star/TRACEABILITY.md`

**Layer-1 companions:**  
`./pair_guides.md` · `./dormancy_signals.md`

**Conventions:**  
**Hot** = private working truth; **Cold** = player-safe surfaces; **TU** = timeboxed task unit;
**Bars** = Integrity, Reachability, Nonlinearity, Gateways, Style, Presentation, Accessibility.

---

## 1) Core principles

1. **Escalate the smallest thing.** Name the _one decision_ blocking progress; avoid omnibus “fix
   everything” escalations.
2. **Route by ownership.** Structure → **Plotwright**, Voice/Register → **Style**, Canon → **Lore**,
   Terminology → **Curator**, Export/Nav → **Binder**, Presentation checks → **Gatekeeper**.
3. **Showrunner chairs cross-domain calls.** When a decision touches multiple owners, the
   **Showrunner** convenes and records the outcome.
4. **Bars beat opinions.** Tie disagreements to the **Quality Bars**; propose the _smallest viable
   fix_ to turn a bar green.
5. **ADR for policy, TU for work.** If a decision sets precedent across books, write an **ADR**
   (Architecture Decision Record) in `/DECISIONS` and keep the TU focused on the current slice.

---

## 2) When to escalate (triggers)

**Immediate (stop work & escalate):**

- **Integrity red:** broken anchors/IDs affecting navigation or exports.
- **Presentation red:** spoilers/meta/technique leaking to Cold.
- **Accessibility red:** missing alt/captions where required; untranslatable labels blocking a
  locale.
- **Safety/legal risk:** Researcher posture high and no neutral phrasing fallback.

**Timed (escalate within a TU):**

- **Gateway fairness** disputed (Plot ↔ Gatekeeper ↔ Lore/Research).
- **Register conflict** between Style and PN/Translator.
- **Taxonomy collision** (Curator ↔ Translator) that breaks anchors or meanings.
- **Art/Audio feasibility** contradicts plan intent repeatedly.

**Deferred (note, then ADR later):**

- Export pipeline standards (multilingual layout, caption conventions).
- Determinism promises and where logs live.
- Global choice-labeling patterns.

---

## 3) Lanes & owners (who decides what)

| Topic                                      | Primary Owner  | Must Consult                   | May Consult  | Decision Record            |
| ------------------------------------------ | -------------- | ------------------------------ | ------------ | -------------------------- |
| Section topology, hubs/loops/gates         | Plotwright     | Gatekeeper, Lore, Style        | PN, Curator  | TU note (+ ADR if policy)  |
| Gate _reasons_ (world causality)           | Lore Weaver    | Plotwright, Researcher         | Style        | TU note                    |
| Voice, register, PN phrasing patterns      | Style Lead     | PN, Translator                 | Curator      | Style Addendum (+ TU note) |
| Terminology, taxonomy, crosslinks          | Codex Curator  | Translator, Style              | Binder, Lore | Codex Pack note            |
| Research posture & neutral phrasing        | Researcher     | Owners of affected slice       | Gatekeeper   | Memo note                  |
| Captions/alt phrasing                      | Style Lead     | Art/Audio Director, Translator | Gatekeeper   | Style Addendum (+ TU note) |
| Image selection/intent                     | Art Director   | Style, Curator                 | Illustrator  | Shotlist note              |
| Audio cue selection/intent                 | Audio Director | Style, PN                      | Producer     | Cue list note              |
| Export binding (labels, anchors, coverage) | Book Binder    | Translator, Gatekeeper         | Showrunner   | View Log                   |
| Merge/view readiness by bar                | Gatekeeper     | Showrunner                     | Owners       | Gatecheck report           |
| Cross-domain deadlock                      | Showrunner     | All relevant                   | —            | ADR in `/DECISIONS`        |

---

## 4) Escalation severity & timeboxes

- **Level 1 — Micro:** owner-to-owner huddle (≤ **10 min**). Output: one-line decision in the TU.
- **Level 2 — Cross-domain:** Showrunner chairs (≤ **25 min**). Output: TU decision + pointer to any
  edited addenda.
- **Level 3 — Policy:** Showrunner drafts **ADR** (≤ **45 min** writing), owners review async.
  Output: `DECISIONS/ADR-<id>-<slug>.md`.

> **Default timebox:** If the clock runs out, pick the **lowest-risk** option that keeps **Cold**
> clean and record a follow-up TU.

---

## 5) Minimal escalation bundle (what to bring)

Copy/paste this into Hot and fill it before you ping anyone:

```

Escalation — <topic> · TU <id> · Slice <name>
Bar(s) pressed: <Integrity|Reachability|...> (state: red/yellow)
Smallest decision needed (one sentence):
Options:
A) <short> — pros/cons
B) <short> — pros/cons
Owner(s): <primary> · Consult: <roles>
Player-surface impact: <1 line; spoilers/technique? none>
Proposed record: <TU note | Style addendum | Codex note | View Log | ADR>
Deadline: <today HH:MM>

```

---

## 6) Decision records (where to write it down)

- **TU Note (inline):** small, local choices (phrasing variant; one gate tweak).
- **Addendum/Pack note:** Style Addendum, Codex Pack, Canon Pack, Shotlist, Cue list—when the
  handshake creates reusable guidance.
- **Gatecheck report:** pass/fail per bar with the chosen fix.
- **View Log:** export decisions (snapshot, options, coverage).
- **ADR in `/DECISIONS`:** policy, standards, or reusable patterns beyond this book.

**ADR skeleton:**

```

# ADR-<seq>-<slug>

## Context

<what forced the decision; bars pressed; prior attempts>

## Decision

<the choice; scope; owner>

## Consequences

<positive/negative; migration notes>

## Alternatives

<A/B/C considered; why not chosen>

## Lineage

TU <id> · Edited <YYYY-MM-DD> · Roles: <list>

```

---

## 7) Worked examples

**A) Meta gate vs diegetic gate (Style ↔ PN ↔ Gatekeeper)**

- Trigger: Presentation **red** (“Option locked: missing CODEWORD”).
- Owner: **Style**. Decision: adopt PN pattern “world refuses” line.
- Record: Style Addendum example + TU note; Gatecheck **green**.

**B) Unfair gateway path (Plot ↔ Lore ↔ Research ↔ Gatekeeper)**

- Trigger: Gateways **yellow/red**; no discoverable path.
- Owner: **Plotwright**. Lore supplies world reason; Research posture `plausible`.
- Decision: add discoverable insider route.
- Record: TU note + Gateway Map update; Gatecheck **green**.

**C) Translation anchor collisions (Translator ↔ Binder ↔ Gatekeeper)**

- Trigger: Integrity **red** in NL bind (diacritics in slugs).
- Owner: **Translator** with **Binder**.
- Decision: kebab-case ASCII slugs; glossary note.
- Record: View Log + Language Pack note; ADR if standardizing globally.

**D) Caption idiom doesn’t travel (Style ↔ Translator ↔ Audio Director)**

- Trigger: Presentation **yellow** (idiom opaque in target).
- Owner: **Style** + **Translator**.
- Decision: neutral caption variant; Audio Director confirms parity.
- Record: Style Addendum + Language Pack; Gatecheck **green**.

---

## 8) Anti-patterns (don’t)

- **Escalate vibes.** Always tie to a bar; propose options.
- **Rewrite by committee.** Owners implement; escalations decide _direction_, not every line.
- **Ship from Hot.** Snapshots or it didn’t happen.
- **Policy by stealth.** If it affects multiple books, write an ADR.

---

## 9) Done checklist (for the Showrunner)

- [ ] Smallest decision named; bar(s) cited
- [ ] Correct lane chosen; owners/consults tagged
- [ ] Timebox set; decision made within level target
- [ ] Record written (TU / Addendum / Pack / View Log / ADR)
- [ ] Bars re-checked (Gatekeeper) → **green** or follow-up TU filed
- [ ] Tracelog updated (who/what/when/why)

---

## 10) Metadata

**Document:** Escalation Rules (Layer 1)  
**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `./pair_guides.md` · `./dormancy_signals.md` · `/DECISIONS/README.md`
