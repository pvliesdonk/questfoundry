# Evergreen Manuscript — Exporting Views, Not Finals

A QuestFoundry manuscript is **never “finished.”** We export **views on Cold SoT** at points in
time. New loops can always reshape canon, prose, or presentation. This document defines what a
“view” is, how we assemble it, and how we keep it safe, clear, and traceable.

---

## 1) What is a manuscript “view”?

An **exported bundle** assembled by the **Book Binder** from a specific **Cold snapshot**. It is
player-safe and ready for reading/playing, but it’s not the last word—just the current one.

Each view records:

- **Cold snapshot ID** (tag/commit, e.g., `cold@2025-10-28T14:05Z`)
- **Export options** (include/exclude: art plan, art renders, audio plan, audio assets, localization
  slices)
- **Surface checks** passed (Gatekeeper report id)

---

## 2) What a view contains (player-safe surfaces)

- **Hyperlinked manuscript** (sections + choices; terminals clearly marked)
- **Codeword & gateway checklists** _(player-safe wording; no internal names)_
- **Codex** (player-safe entries with cross-refs)
- **Front matter** (title, credits, snapshot ID, content warnings if applicable)
- **Back matter** (appendices, acknowledgements, export options used)

**Optional inclusions**

- **Art plan** (captions + intent, spoiler-safe)
- **Art renders** (with alt text; parameters kept off-surface)
- **Audio plan** (cue sheets, spoiler-safe)
- **Audio assets** (with captions/text-equivalents)
- **Localization slices** (language packs flagged complete/incomplete)

---

## 3) Binder responsibilities (human-level)

- Assemble from **Cold** only (never from Hot).
- Ensure **navigation clarity** (links resolve; breadcrumbs/headers readable).
- Maintain **spoiler hygiene** (no hidden gates or codeword labels on surface).
- Provide **accessibility** basics: alt text, descriptive link text, sensible headings, motion/audio
  cautions.
- Record **snapshot ID** and **export options** inside the bundle.

Binder does not rewrite content; issues get logged as TUs or loop triggers via Showrunner.

---

## 4) Export options (mix-and-match)

| Option                | Default | Notes                                                           |
| --------------------- | ------- | --------------------------------------------------------------- |
| Include art plan      | off     | Plans may ship without renders; must be spoiler-safe.           |
| Include art renders   | off     | Requires determinism log in Cold if promised.                   |
| Include audio plan    | off     | Spoiler-safe cue descriptions only.                             |
| Include audio assets  | off     | Provide text equivalents and safety notes.                      |
| Include translations  | off     | Flag each slice `complete` / `incomplete`.                      |
| Print-friendly layout | on      | Single-column, high contrast, no color-only cues.               |
| PN script             | off     | Usually reserved for PN tools; if included, stays spoiler-safe. |

---

## 5) Player safety & codex boundary

- **Manuscript**: no internal labels (“CODEWORD: ASH”), no schema lingo, no RNG talk.
- **Codex**: player-safe summaries; spoilers remain in **canon notes** (not exported).
- **PN**: same snapshot as the view; enforces gates diegetically (see PN Principles).

---

## 6) Versioning & changelogs

- Each view includes a **View Log** page:
  - Snapshot ID and date
  - **TU-IDs** merged since previous view (titles only, spoiler-safe)
  - Notable additions (new hub/loop/gateway), high-level only
  - Known limitations (e.g., “art plan present; renders deferred”)

This keeps readers/QA oriented without revealing internals.

---

## 7) Formats

Minimum supported:

- **Markdown** (source of truth for the bundle)
- **HTML** (hyperlinked reading)
- **EPUB** (e-readers)
- **PDF** (print-ready, single column)

All formats must retain: links, alt text, front/back matter, and the snapshot/option metadata.

---

## 8) Checklists (pre-export)

**Binder preflight**

- [ ] All links resolve; terminals marked.
- [ ] Player-safe wording (no codeword names, no gate internals).
- [ ] Codex entries exist for key terms referenced by surface text.
- [ ] Alt text present; link text descriptive; headings structured.
- [ ] Export options chosen; art/audio inclusions are spoiler-safe.
- [ ] Snapshot ID embedded in front matter and metadata.
- [ ] Gatekeeper report ID attached.

**Gatekeeper spot checks**

- [ ] Integrity / Reachability / Nonlinearity / Gateways
- [ ] Style adherence (voice, captions)
- [ ] Determinism metadata present for any promised assets
- [ ] Presentation safety (no leaks)

---

## 9) Deferred assets & plans

Art/Audio/Translation may be **plan-only** in Cold:

- Mark as `deferred:<asset>`; include **why** it adds value and **how** it will be safe.
- Plans in a view must **not** spoil reveals (caption wording reviewed by Style Lead & Gatekeeper).

---

## 10) “Views never end” policy

- A view does not freeze canon; it **reflects** canon at a moment in time.
- New loops can run after publication; the next export cites a new snapshot ID and View Log.
- Readers and playtesters can always request a refreshed view; the Showrunner schedules Binding
  Runs.

---

## 11) Failure modes & remedies

- **Leaked internal labels** → Rewrite with diegetic phrasing; recheck PN & codex surfaces.
- **Dead links in export** → Fix section anchors; re-export.
- **Spoiler-y captions** → Move detail into canon notes; keep caption atmospheric, not revealing.
- **Accessibility gaps** → Add alt text; ensure contrast; provide audio text-equivalents.

---

**TL;DR**  
We don’t ship “finals.” We ship **clean, traceable views** on Cold SoT. The Book Binder assembles
them player-safe, accessible, and clearly labeled with what they include and when they were cut.
