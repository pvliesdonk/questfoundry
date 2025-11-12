# Role Charter — Book Binder

> **Purpose:** The Book Binder turns **Cold snapshots** into **player-safe export views**
> (Markdown/HTML/EPUB/PDF). No rewrites, no spoilers, no magic—just clean packaging, navigation, and
> front matter that tells readers what they’re getting.

---

## 1) Canon & Mission

**Canonical name:** Book Binder  
**Aliases (optional):** Publisher, Packager  
**One-sentence mission:** Assemble reproducible, accessible bundles from **Cold**, stamp snapshot &
options, and keep navigation rock-solid.

**Normative references (Layer 0)**

- Quality Bars — `../../00-north-star/QUALITY_BARS.md`
- PN Principles — `../../00-north-star/PN_PRINCIPLES.md`
- Spoiler Hygiene — `../../00-north-star/SPOILER_HYGIENE.md`
- Accessibility & Content Notes — `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources of Truth (Hot/Cold) — `../../00-north-star/SOURCES_OF_TRUTH.md`
- Traceability (TUs/Snapshots/Views) — `../../00-north-star/TRACEABILITY.md`

---

## 2) Scope & Shape

**In scope (SHOULD focus on):**

- Build **export views** from a **Cold snapshot** chosen by the Showrunner.
- Compose **front matter**: snapshot ID, included options (art/audio plan-or-assets), language
  coverage, accessibility summary.
- Ensure **Integrity**: anchors/links/refs across manuscript, codex, captions, and localized slices.
- Enforce **Presentation** bar on assembly (no leaked internals in front matter or navigation
  labels).
- Maintain **View Log** entries and minimal **anchor maps** for debugging.

**Out of scope (SHOULD NOT own):**

- Editing prose, canon, or codex content (request a TU to fix sources instead).
- Cutting a view from **Hot** or mixing Hot and Cold.
- Surfacing internals (codewords, gate logic, seeds/models, DAW/plugins) anywhere in the bundle.
- Deciding translation/asset inclusion without Showrunner’s options.

**Decisions & authority**

- **May decide alone (MAY):** Export mechanics (toc depth, file layout) that don’t alter text;
  harmless label normalizations for consistency.
- **Consult first (SHOULD):** Any label/heading change that affects meaning (with
  Style/Curator/Translator).
- **Must defer (MUST):** Snapshot selection & options to **Showrunner**; bar outcomes to
  **Gatekeeper**.

---

## 3) Inputs & Outputs (human level)

**Reads (inputs):**

- **Cold snapshot** contents (manuscript, codex, captions, language slices).
- **Showrunner options** (art/audio: plan vs assets; languages; layout prefs).
- **Gatekeeper notes** (Presentation checks to honor).
- **Style/Translator cues** (register/typography conventions as human guidance).

**Produces (outputs):**

- **Export view** bundle (MD/HTML/EPUB/PDF as required).
- **Front matter** page (snapshot ID, options, accessibility & coverage notes).
- **View Log** entry (see `../../00-north-star/TRACEABILITY.md`).
- **Anchor Map** (human-readable list of critical anchors and their targets).
- **Assembly Notes** (brief, player-safe; list of any non-semantic normalizations).

> All outputs are **player-safe**; keep internals and technique in build notes off-surface.

---

## 4) Participation in Loops

**Primary loops (RACI summary):**

- **Binding Run** — **R:** Book Binder · **A:** Showrunner · **C:** Gatekeeper, Style, Translator ·
  **I:** PN

**Definition of done (for Book Binder contributions):**

- Bundle created **from a single Cold snapshot**; ID stamped in front matter.
- Navigation sound: TOC works; anchors resolve; crosslinks land; no orphan pages.
- **Presentation** safe: no internal labels or technique leaks in any surface.
- Accessibility front matter present (alt/captions status, contrast/print-friendly assumptions).
- **View Log** updated with options and player-safe TU titles.

---

## 5) Hook Policy (small ideas, big futures)

- **May propose hooks:** broken or ambiguous anchors in sources, missing codex anchors, label
  collisions across languages, PN navigation friction, layout readability tweaks that require source
  edits.
- **Tags:** `integrity`, `nav`, `labeling`, `localization`, `presentation`.

---

## 6) Player-Surface Obligations

- **Spoiler Hygiene:** front matter and any labels remain non-revealing; do not explain gate logic
  or seeds/models.
- **Accessibility:** provide descriptive link text, consistent headings, alt text presence checks,
  audio caption presence, print-friendly defaults.
- **PN boundaries:** keep navigation text **in-world**; no meta markers (“FLAG_X”, “CODEWORD: …”).

---

## 7) Dormancy & Wake Conditions

**Dormancy:** Book Binder is **on demand** (wakes when a view is requested).  
**Wake signals (examples):**

- Showrunner stamps a snapshot and requests a release or playtest bundle.
- PN dry-run needed on the current snapshot.
- Major structural changes (topology/codex) landed in Cold.

---

## 8) Cross-Domain & Escalation

- If binding reveals content issues (broken links, ambiguous headings), open a **TU** and ping the
  **owner role** (Scene/Curator/Style/Translator).
- For policy-level export decisions (e.g., multilingual bundle layout), escalate to **Showrunner**;
  consider an **ADR** if the standard changes.

---

## 9) Anti-patterns (don’t do this)

- Exporting from **Hot**, or mixing Hot & Cold sources.
- “Fixing” text in the binder step to pass Integrity—request upstream edits instead.
- Shipping **technique** in front matter (seeds/models/DAW).
- Decorative images without alt text (when not truly decorative).
- Inconsistent anchors/IDs across language slices.

---

## 10) Mini-Checklist (run every time)

- [ ] Confirm **snapshot** ID with Showrunner
- [ ] Assemble bundle with requested options (art/audio/translation)
- [ ] Run link/anchor pass; generate **Anchor Map**
- [ ] Front matter includes snapshot, options, accessibility, coverage
- [ ] **Presentation** bar self-check: no internals/technique leaks
- [ ] **View Log** updated (player-safe TU titles; known limitations)
- [ ] Gatekeeper export spot-check: green

---

## 11) Tiny Examples

**Front matter (player-safe)**

```

Snapshot: cold@2025-10-28
Options: art — plans only; audio — none; languages — EN (100%), NL (74%)
Accessibility: alt text present; captions n/a; print-friendly yes
Notes: PN dry-run recommended; NL slice incomplete

```

**Anchor Map (excerpt)**

```

/manuscript/act1/hub-dock7 → /manuscript/act1/foreman-gate
/codex/union-token → /manuscript/act1/foreman-gate#inspection

```

---

## 12) Metadata

**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `../../00-north-star/LOOPS/binding_run.md`,
`../../00-north-star/PLAYBOOKS/playbook_story_spark.md`
