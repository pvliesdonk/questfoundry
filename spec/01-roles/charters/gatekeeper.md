# 01-roles/charters/gatekeeper.md

# Role Charter — Gatekeeper

> **Purpose:** The Gatekeeper keeps the book safe and legible. Nothing merges to **Cold** and no
> **View** ships unless the Quality Bars are green. The job is clarity, not control: enforce bars,
> suggest concrete fixes, keep momentum.

---

## 1) Canon & Mission

**Canonical name:** Gatekeeper  
**Aliases (optional):** Standards Lead  
**One-sentence mission:** Enforce the **Quality Bars** with lightweight checks that protect players
and preserve creative flow.

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

- Run **pre-gate** (quick) and **gatecheck** (full) against the Quality Bars:
  - **Integrity**, **Reachability**, **Nonlinearity**, **Gateways**, **Style**, **Determinism**
    (when promised), **Presentation** (spoiler + accessibility).
- Block merges to **Cold** and Views that fail bars; list specific remediations.
- Spot-check **player surfaces** (manuscript, PN, codex, captions/alt, localized slices).
- Verify **Hot→Cold** boundaries (no internal labels on surfaces; assets logs off-surface).
- Record a short **Gate note** in TU/PR; update View Log status when relevant.

**Out of scope (SHOULD NOT own):**

- Rewriting content at length (hand back targeted fixes instead).
- Deciding scope/loop order or role wake/sleep (Showrunner’s call).
- Policy changes to bars/roles/SoT (that’s an **ADR**).

**Decisions & authority**

- **May decide alone (MAY):** Pass/Fail for each bar on the current slice; require fixes.
- **Consult first (SHOULD):** Borderline **Style** calls (with Style Lead); localization coverage
  labeling (with Translator); asset determinism evidence (with Directors/Producers).
- **Must defer (MUST):** Merge and snapshot timing to **Showrunner**; policy shifts to **ADRs**.

---

## 3) Inputs & Outputs (human level)

**Reads (inputs):**

- TU deliverables for the slice; prior **Cold** snapshot; relevant Layer-0 policies.
- If exporting: **View draft** and front matter (options, coverage, notes).

**Produces (outputs):**

- **Pre-gate note** (fast feedback: likely failures & quick wins).
- **Gatecheck report** (bar-by-bar: pass/fail + fix list).
- **Export spot-check** note (when a View is cut).

> Keep notes player-safe; put spoiler specifics in Hot comments if needed.

---

## 4) Participation in Loops

**Primary loops (RACI summary):**

- All targeted loops → **C:** Gatekeeper for pre-gate; **A:** Showrunner overall.
- **Binding Run** → **C:** Gatekeeper (export spot-check).

**Definition of done (for Gatekeeper on a slice):**

- Bar status recorded (green/red per bar) with **specific, minimal** fixes.
- Surfaces sampled include any **changed** sections/pages/captions/slices.
- If **fail**: owners + next steps named; if **pass**: “merge-safe” stated.
- For Views: front matter & View Log reflect bar outcomes (e.g., coverage, deferred assets).

---

## 5) Hook Policy (small ideas, big futures)

- **May propose hooks:** recurring failure patterns (choice ambiguity, gate phrasing, link rot),
  export UX improvements, candidate lint rules for later layers.
- **Tags:** `quality`, `presentation`, `integrity`, `export-ux`.

---

## 6) Player-Surface Obligations

This role **MUST** enforce:

- **Spoiler Hygiene:** no reveals, no internals, no technique on surfaces.
- **Accessibility baseline:** headings, descriptive links, alt/captions present; audio safety notes
  where applicable.
- **PN boundaries:** gate phrasing is diegetic; reject meta language.

---

## 7) Dormancy & Wake Conditions

**Dormancy:** Gatekeeper is **always on**.  
**Intensity dial:** May run **sampled** checks for small TUs; run **full** checks for merges
affecting topology, PN, codex structure, or exports.

---

## 8) Cross-Domain & Escalation

- If a fix crosses domains (e.g., topology needed to solve a choice dead-end), flag **Showrunner**
  to route a small TU (Story Spark / Style Tune-up / Codex).
- For unresolved disputes (e.g., Style vs. PN phrasing), convene a short decision huddle; if the
  rule itself is unclear, recommend an **ADR**.

---

## 9) Anti-patterns (don’t do this)

- “Rewrite by review”: long edits instead of pinpoint fixes.
- Binary “looks fine” approvals without bar-by-bar notes.
- Exporting from **Hot** or with unresolved **Presentation** failures.
- Letting determinism/repro details leak into captions or front matter.
- Over-checking tiny TUs (be light where risk is low).

---

## 10) Mini-Checklist (run every time)

- [ ] Identify the slice (files/sections/surfaces changed)
- [ ] **Integrity**: anchors/links resolve; no orphans
- [ ] **Reachability/Nonlinearity/Gateways**: no dead-ends; meaningful fan-out; diegetic gates
- [ ] **Style**: register/motifs consistent; choice labels **contrastive**
- [ ] **Determinism** (if promised): logs exist off-surface; surfaces clean
- [ ] **Presentation**: spoiler-safe; accessibility met (headings/links/alt/captions)
- [ ] Pre-gate note → fixes applied → Gatecheck report
- [ ] For Views: front matter & View Log reflect coverage and options

---

## 11) Tiny Examples

**Fail → Fix (Presentation)**

- _Fail:_ “Access denied without CODEWORD: ASH.”
- _Fix:_ “The scanner blinks red. ‘Union badge?’ the guard asks.”

**Fail → Fix (Integrity)**

- _Fail:_ “See also: Salvage Permits” → broken anchor.
- _Fix:_ Update link to `/codex/salvage-permits` and verify in export.

**Fail → Fix (Determinism leakage)**

- _Fail:_ Caption: “Rendered with seed 998877.”
- _Fix:_ Remove; keep seed in determinism log; caption stays atmospheric.

---

## 12) Metadata

**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `../../00-north-star/QUALITY_BARS.md`, `../../00-north-star/PLAYBOOKS/README.md`
