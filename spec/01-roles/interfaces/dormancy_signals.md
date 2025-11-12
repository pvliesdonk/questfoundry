# Dormancy Signals — When to Sleep, When to Wake (Layer 1, Human-Level)

> **Purpose:** Make optional roles truly optional—without guessing. This guide defines _sleep/wake_
> signals, deferral tags, and safety rules so the Showrunner can run **small, focused loops** while
> keeping player surfaces clean.

---

## 0) Normative references (Layer 0)

- Sources of Truth — `../../00-north-star/SOURCES_OF_TRUTH.md`
- Quality Bars — `../../00-north-star/QUALITY_BARS.md`
- PN Principles — `../../00-north-star/PN_PRINCIPLES.md`
- Spoiler Hygiene — `../../00-north-star/SPOILER_HYGIENE.md`
- Accessibility & Content Notes — `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Traceability — `../../00-north-star/TRACEABILITY.md`

**Conventions:**  
**Hot** = private working truth (spoilers allowed). **Cold** = player-safe surfaces (no
internals/spoilers). **TU** = timeboxed task unit. **Bars** = Integrity, Reachability, Nonlinearity,
Gateways, Style, Presentation, Accessibility.

---

## 1) Roles that may be dormant

- **Art Director**, **Illustrator**
- **Audio Director**, **Audio Producer**
- **Translator (Localization Lead)**
- **Researcher**

> Others _can_ be paused between loops, but these four tracks are explicitly optional per run. Books
> can ship **prose-only**; plans can ship **plan-only**.

---

## 2) Deferral tags & front-matter signals

Use these **player-safe** deferral tags in Hot and for Binder notes (never in prose):

- `deferred:art` — Art Plans exist or not; no renders included.
- `deferred:audio` — Audio Plans exist or not; no cues included.
- `deferred:translation` — Register map/glossary may exist; localized slices partial or absent.
- `deferred:research` — Claims marked with **posture**; neutral phrasing in place; deep
  corroboration pending.

**Binder front matter (example):**

```

Options: art — plans only; audio — none; languages — EN 100%, NL 74%
Accessibility: alt present; captions n/a
Notes: deferred:art · deferred:audio

```

---

## 3) Wake rubric (Showrunner quick triage)

Score each criterion **0–2**; wake when total ≥ **4** or any **hard wake** triggers.

**A. Player benefit (0–2)**  
0 = marginal; 1 = noticeable clarity/mood; 2 = unlocks stalled comprehension or choice clarity.

**B. Bar pressure (0–2)**  
0 = none; 1 = Presentation/Accessibility yellow; 2 = Gatekeeper red without that role.

**C. Scope fit (0–2)**  
0 = sprawling; 1 = fits a TU slice; 2 = fits a 45–90 min loop.

**D. Reuse leverage (0–2)**  
0 = bespoke; 1 = reusable pattern; 2 = becomes house pattern (PN, Style, Codex).

**Hard wake triggers (wake immediately)**

- **Gatekeeper red** on Presentation/Accessibility that the dormant role owns.
- Export goal **requires** the asset (e.g., multilingual release).
- Safety/legal risk identified (Researcher posture high with player-harm potential).

---

## 4) Sleep rules (when _not_ to wake)

- If the fix is **topology/voice/terminology**, wake Plot/Scene/Style/Curator instead.
- If captions/alt can be written from existing plans, **Illustrator/Producer may stay dormant**.
- If localization is <30% coverage and not a release goal, ship **register map + glossary slice**
  and keep Translator dormant (`deferred:translation`).
- If Researcher posture is `plausible` and **neutral phrasing** is safe, keep Researcher dormant and
  schedule a research TU later.

---

## 5) Per-role signals

### 5.1 Art (Director/Illustrator)

**Wake when:**

- **Signposting gaps** at hubs/gates (PN/Gatekeeper flags)
- Style requests a **motif anchor**; Curator wants a **visual entry point**
- A release promises **artful** presentation or a cover

**Sleep when:**

- Choices are clear without images; captions/alt would duplicate text
- Style/Curator can solve with phrasing/glossary alone

**Plan-only allowed:** yes → tag `deferred:art`

---

### 5.2 Audio (Director/Producer)

**Wake when:**

- PN cadence suffers; **pace cues** would unstick reading
- Accessibility favors **text equivalents** for implied soundscapes
- Export targets audio-inclusive experiences

**Sleep when:**

- Dialogue density is high; added cues risk masking prose
- Accessibility goals are satisfied via text equivalents alone

**Plan-only allowed:** yes → tag `deferred:audio`

---

### 5.3 Translation (Localization Lead)

**Wake when:**

- Multilingual view is a release goal
- Gate phrasing or choice contrast **depends on register/idiom** in target language
- Curator introduces taxonomy with localization consequences

**Sleep when:**

- Source still volatile; anchors/labels unstable
- Target coverage would fall below stated threshold for release

**Partial deliverable allowed:** **Register Map + Glossary slice** → tag `deferred:translation`

---

### 5.4 Research

**Wake when:**

- Gate fairness or safety depends on **real-world constraints**
- Medical/legal/cultural sensitivity risk exists
- A disputed claim drives **keystone** structure or tone

**Sleep when:**

- Neutral phrasing removes over-claiming and risk is low
- Posture can be marked `plausible` without blocking a View

**Partial deliverable allowed:** **Posture + neutral phrasing** → tag `deferred:research`

---

## 6) Gatekeeper policy under dormancy

- **Block (red)** only when a dormant role is the **owner** of a failing bar and no safe fallback
  exists (e.g., missing alt → Accessibility red; un-localized labels break anchors → Integrity red).
- **Merge-safe (yellow)** when neutral fallbacks exist (e.g., text equivalents present, art
  optional, translation partial with coverage stated). Note `deferred:*` tags in Binder notes.
- Always cite **bar + smallest viable fix** and whether waking is required or optional.

---

## 7) TU scaffolds (copy/paste)

**Wake a dormant role (Showrunner TU):**

```

TU: wake-<role>-<slice>
Loop: <Style Tune-up | Art Touch-up | Audio Pass | Translation Pass | Story Spark | Hook Harvest>
Slice: <short description>
Why now (rubric): A=<..> B=<..> C=<..> D=<..> → total=<N> [hard wake? yes/no]
Deliverables: <list>
Bars pressed: <Presentation/Accessibility/...>
Exit: bars green + <artifact(s)>
Tags: remove deferred:<art|audio|translation|research> if exit met

```

**Keep dormant with safe fallback:**

```

Decision: keep <role> dormant
Fallbacks: <neutral phrasing | register map only | plan-only | posture labels>
Deferral tags: deferred:<...>
Risk notes: <1–2 lines>
Schedule: revisit in release-RC loop

```

---

## 8) Worked micro-examples

**A) Keep art dormant; plan later**

- PN reports minor hesitation at **Foreman Gate**; Style proposes a micro-context line.
- Gatekeeper: Presentation **green** after fix.
- Decision: keep Art dormant; file **Art Plan** hook for signpost image; tag `deferred:art`.

**B) Wake translation**

- Binder shows anchor collisions in NL due to diacritics; Gatekeeper Integrity **red**.
- Wake Translator for **label slug policy** + register map; re-bind; remove `deferred:translation`.

**C) Research posture with neutral phrasing**

- Claim: “Badge cloning trivial.” Researcher dormant; risk uncertain.
- Style swaps to **neutral surface**: “The scanner hesitates.”
- Tag `deferred:research`; schedule memo later; View ships safely.

**D) Audio plan-only**

- PN cadence fine; Style requests optional **pace cue** post-inspection.
- Audio Director writes plan + caption; Producer dormant; tag `deferred:audio`.

---

## 9) Done checklist (for the Showrunner)

- [ ] Dormant roles reviewed with **wake rubric** (score or hard trigger)
- [ ] Decision recorded: wake vs keep dormant (with fallback)
- [ ] **Deferral tags** set (`deferred:*`) or removed upon completion
- [ ] Binder front matter updated (options, coverage, accessibility)
- [ ] Gatekeeper status: **red/green/yellow** with rationale by bar
- [ ] Tracelog updated (TU ID, snapshot/view impacts)

---

## 10) Notes on traceability

- Always record dormancy decisions in the **TU** and **Tracelog** so future loops can reason without
  chat memory.
- When a deferred track completes, note the **removal** of the corresponding `deferred:*` tag in the
  **View Log** and **front matter**.

---
