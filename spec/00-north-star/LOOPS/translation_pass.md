# Translation Pass — Add or Maintain a Target-Language Slice

**Purpose**  
Create or update a **player-safe** translation of the manuscript/codex surfaces while preserving PN
boundaries, style intent, and navigation. This loop supports plan-only merges (glossary/style kit)
when the full slice isn’t ready.

**Outcome**  
A **Language Pack** (glossary, style notes, localized surfaces, open issues) ready for Gatekeeper
checks and merge to **Cold**. Exports can include complete or partial translations flagged
accordingly.

---

## 1) Triggers (Showrunner)

- New target language requested.
- Significant style/canon changes warrant a translation refresh.
- Accessibility or market goals require multilingual exports.

**Activation**  
Open/attach a **Trace Unit (TU)**: `tu-translation-<lang>-<scope>`. Confirm **Translator**
activation (others may be dormant).

---

## 2) Inputs

- **Cold snapshot** (manuscript/codex surfaces), PN Principles, Style guardrails.
- Current glossary/terminology (if any); motif kit.
- Cultural notes from Lore/Style (tone, registers to preserve/avoid).
- Known untranslatable idioms list (if exists).

---

## 3) Roles & Responsibilities

- **Translator (R)**
  - Produce the **Language Pack**: glossary, style transfer notes, localized surfaces; flag
    untranslatables and cultural risks.
- **Style Lead (C)**
  - Approve register mapping (e.g., formal/informal “you”), motif equivalence, idiom strategy.
- **PN (C)**
  - Validate diegetic gate phrasing patterns in target language; no internals exposed.
- **Codex Curator (C)**
  - Ensure cross-refs and titles localize consistently; avoid spoiler drift.
- **Gatekeeper (C)**
  - Check **Presentation Safety**, **Integrity** (links resolve), **Style** adherence.
- **Showrunner (A)**
  - Scope completeness (chapters only, full book, codex subset) and merge timing.

_(Lore Weaver/Researcher may advise on cultural specifics or factual terms when needed.)_

---

## 4) Language Pack Contents (human-level, not a schema)

- **Glossary**
  - Term → approved translation; part-of-speech; usage notes; do-not-translate list (proper nouns,
    codeword _names_ never shown on surface anyway); examples.
- **Register Map**
  - Pronoun system choice; honorifics; tone equivalents; swear policy.
- **Motif Equivalence**
  - How house motifs render (e.g., “low-G dust” image choices).
- **Idiom Strategy**
  - List of idioms → literal/functional equivalents or rewrites.
- **Localized Surfaces**
  - Manuscript sections and choice labels; codex titles/summaries; captions and alt text.
- **Open Issues**
  - Untranslatables needing rewrite upstream; glossary gaps; cultural cautions.
- **Traceability**
  - TU-ID, snapshot ID, coverage % (by section count and codex entries).

---

## 5) Procedure

1. **Scope & Coverage Plan (Translator + Showrunner)**
   - Decide **slice** (full book, acts, codex subset).
   - Set coverage target and time box; mark partial outputs as `incomplete`.

2. **Glossary First**
   - Create/refresh glossary with Style Lead.
   - Decide **register** (T/V distinction, dialect); lock decisions here.

3. **Segment & Localize**
   - Translate player surfaces: **no internal labels**, no spoilers, preserve PN diegesis.
   - Keep hyperlinks and anchors intact; choice labels stay distinct and clear.

4. **Motif & Idiom Pass (Style Lead)**
   - Validate motif resonance; solve idioms with functionally equivalent phrases.

5. **PN Phrasing Check**
   - Confirm gate enforcement phrasing is diegetic and natural in target language.

6. **Link Audit (Curator)**
   - Ensure cross-refs resolve to localized targets; add stubs if scoped.

7. **Pre-Gate (Gatekeeper)**
   - **Presentation Safety** (no leaks), **Integrity** (links), **Style** (tone/voice).

8. **Package**
   - Assemble **Language Pack**; compute coverage and list open issues.

---

## 6) Deliverables (Hot)

- **Language Pack** for `<lang>`: glossary, register map, motif equivalence, idiom strategy,
  localized surfaces, open issues, coverage %.
- **Pre-gate note**: Gatekeeper’s findings and remediation.

---

## 7) Merge Path (summary)

- If **glossary/kit only**: may merge to **Cold** as **deferred:translation** with coverage `0%`.
- If localized surfaces present: Gatekeeper passes **Presentation/Integrity/Style** → **Showrunner**
  merges to **Cold**, stamping snapshot and coverage %.
- Binder may now include the slice (marked `complete` / `incomplete`).

---

## 8) Success Criteria

- Register and tone feel native; PN voice preserved.
- Links/cross-refs resolve; choice labels remain distinct.
- Glossary stable; idioms handled; motifs resonate.
- No spoilers or internal labels leaked; accessibility text localized.

---

## 9) Failure Modes & Remedies

- **Literalism breaks tone** → Recast with Style Lead; add glossary usage examples.
- **Untranslatable idiom** → Provide functional equivalent or upstream rewrite suggestion.
- **Broken anchors** → Sync IDs; re-run link audit.
- **PN leaks plumbing** → Replace with diegetic phrasing; Gatekeeper blocks until fixed.
- **Inconsistent terms** → Lock glossary; batch-fix before merge.

---

## 10) Inclusion in Exports (Binder)

- Export options: include **translations** per language with coverage flags:
  `complete`/`incomplete`.
- Front matter records **snapshot ID**, languages included, and coverage %.
- If incomplete, Binder lists missing sections so readers know scope.

---

## 11) RACI (quick)

| Task                          | R          | A          | C                  | I          |
| ----------------------------- | ---------- | ---------- | ------------------ | ---------- |
| Build glossary & register map | Translator | Showrunner | Style Lead         | Gatekeeper |
| Localize surfaces             | Translator | Showrunner | PN, Curator        | Gatekeeper |
| Motif/idiom pass              | Style Lead | Showrunner | Translator         | Gatekeeper |
| Pre-gate                      | Gatekeeper | Showrunner | Style Lead         | All        |
| Merge & export opts           | Showrunner | Showrunner | Gatekeeper, Binder | PN         |

---

## 12) Mini Example (EN → NL)

**Source choice**: “Slip through maintenance, or face the foreman.”  
**Glossary**: `foreman → voorman` (register: colloquial), `maintenance → onderhoudsdek`  
**Localized**: “Glip door het onderhoudsdek, of sta de voorman te woord.”  
**PN diegesis**: gate checks phrased as “De voorman knikt naar het token op je borst.”

---

**TL;DR**  
Lock terms and tone, translate the player surface without exposing the gears, keep links alive, and
ship a clearly labeled language slice—even if it’s just the glossary this round.
