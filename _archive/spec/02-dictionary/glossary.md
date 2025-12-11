# Glossary — Common Language (Layer 2)

> **Status:** ⚠️ **REVIEW NEEDED — Outdated enumerations, update to defer to taxonomies.md**
>
> This file provides prose definitions of core system terminology. Some enumerations (hook types,
> hook status, quality bars) are outdated and should defer to `taxonomies.md` as the single source
> of truth. See LAYER2_AUDIT_PHASE4.md for details.

Clear words prevent weird bugs. This glossary defines how we use key terms across roles. These are
**human** definitions for Layer 2; technical codification (schemas/protocol) comes in Layers 3–4.

---

## A. Narrative topology

**Section** — A unit of prose that ends in one or more **choices**.

**Choice** — A player-facing action at the end of a section that links to another section.

**Hub** — A section (or cluster) with meaningful fan-out to multiple distinct routes.

**Loop** — A design that returns the reader to a prior location **with difference** (new context,
options, or consequences).

**Gateway** — A diegetic condition controlling access (e.g., a token, reputation, knowledge, or
tool). PN enforces gates **in-world**; internal labels never surface.

**Codeword** — A hidden state flag used by the system to alter future outcomes. **Names of codewords
never appear on player surfaces.** Surfaces speak in-world (“the foreman spots your union token”).

---

## B. Sources of Truth (SoT)

**Hot** — Discovery space: drafts, hooks, canon work-in-progress, plan-only art/audio/translation.
Spoilers allowed.

**Cold** — Curated canon and **player-safe surfaces** (codex summaries, captions, localized text).
No spoilers, no internal labels.

**Snapshot** — An immutable tag of **Cold** (e.g., `cold@2025-10-28`). Exports are cut from
snapshots.

**View** — A player-safe export bundle (MD/HTML/EPUB/PDF) built from a snapshot via **Binding Run**.

---

## C. Traceability

**Trace Unit (TU)** — A small, scoped change record (who/why/what/risks/hand-offs). Lives in Hot;
referenced by PRs and merge notes.

**Merge Train** — The sequence: Loop in Hot → Gatecheck → Merge to Cold → Snapshot → View.

**View Log** — A record inside the export noting snapshot ID, included options
(art/audio/translations), and notable TUs since last view.

---

## D. Roles (canon names)

**Showrunner** — Coordinates loops, wakes dormant roles, approves merges and exports.

**Gatekeeper** — Enforces **Quality Bars** before Cold merges and in exports.

**Plotwright** — Designs topology: hubs/loops/gateways; drafts section briefs.

**Scene Smith** — Writes/rewrites section prose to briefs and style.

**Lore Weaver** — Turns accepted hooks into spoiler-level **canon** (backstory, timeline,
causality).

**Codex Curator** — Publishes **player-safe** entries and cross-refs; never invents deep lore.

**Style Lead** — Maintains voice/register/motifs and phrasing patterns.

**Researcher** _(optional/dormant)_ — Verifies factual claims; assigns uncertainty levels.

**Art Director / Illustrator** _(optional/dormant)_ — Plan and/or render illustrations.

**Audio Director / Audio Producer** _(optional/dormant)_ — Plan and/or produce sound.

**Translator** _(optional/dormant)_ — Builds language packs and localized surfaces.

**Book Binder** — Assembles exports (views) from Cold.

**Player-Narrator (PN)** — Performs the book in-world; enforces gates diegetically; reports UX
issues.

---

## E. Work units

**Hook** — A compact proposal (idea/uncertainty/tell) small enough to triage. For hook types (13
total) and status values (7 total), see `taxonomies.md` §1–2.

**Canon** — Spoiler-level truth authored by Lore Weaver (causal, time-anchored). Player-safe
summaries live in the **codex**, not here.

**Plan-only** — A merged plan without assets (e.g., art captions/intent, audio cue sheets,
glossary). Marked **deferred** until assets land.

---

## F. Safety & Quality

**Player-safe surface** — Text a player can see: manuscript, codex, captions, PN lines. No spoilers,
no internal labels, no technique talk.

**Quality Bars** — Merge/export criteria (8 mandatory checks): **Integrity**, **Reachability**,
**Nonlinearity**, **Gateways**, **Style**, **Determinism** (when promised), **Presentation**
(spoiler hygiene), **Accessibility** (navigation, alt text, sensory considerations). See
`taxonomies.md` §5 and `../00-north-star/QUALITY_BARS.md` for details.

**Choice Integrity** — How to converge branches without “fake agency.” Key terms:

- **Immediate reflection** — When sibling choices converge, the next scene’s first paragraph
  reflects the chosen path (lexical, behavioral, or situational). Not necessarily a literal echo,
  but perceivable to the player.
- **Diegetic bridge** — A short, in-world transit line that bridges anchors and preserves causality.
- **State-aware affordances** — At least one option reads differently based on the entering state.
- **Micro-beat between** — A short beat between scenes (when convergence is necessary) that sets a
  visible state flag and can grow into a larger scene later.

See also: `conventions/choice_integrity.md`.

**Determinism (assets)** — If promised, renders/cues carry reproducibility notes (seed/model or
DAW/session). Never surface technique to players.

---

## G. Localization

**Language Pack** — Glossary, register map, motif equivalence, idiom strategy, localized surfaces,
coverage %, open issues.

**Coverage %** — Portion of sections/codex entries localized in a slice; exports label slices
`complete` / `incomplete`.

---

## H. Naming & references

**Player-safe ref** — A human ref that avoids internal IDs (e.g., “Dock 7 — Foreman’s Gate” rather
than `s41` or `CODEWORD: ASH`).

**Lineage** — A one-line pointer in a doc noting the TU/snapshot that produced it, without spoilers.

---

### Notes

- These are **definitions**, not schema fields. Technical codification lives later (Layers 3–4).
- When a term risks spoiler leakage, prefer **player-safe phrasing** on surfaces and keep labels in
  Hot.
