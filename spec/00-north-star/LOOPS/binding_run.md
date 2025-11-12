# Binding Run — Assemble an Export View on Cold

**Purpose**  
Assemble a **player-safe export view** of the book from a specific **Cold snapshot**. Package
manuscript, codex, and checklists—optionally including art/audio plans or assets and translation
slices—without leaking spoilers or internal plumbing.

**Outcome**  
A stamped **export bundle** (Markdown/HTML/EPUB/PDF) with front-matter metadata (snapshot ID,
options used) and a **View Log**. No new canon is created; this loop does **not** write to Cold.

---

## 1) Triggers (Showrunner)

- Milestone release (chapter/act/book).
- Playtest build needed for PN **Narration Dry-Run**.
- External review or print/export request.

**Activation**  
Showrunner chooses a **Cold snapshot** and export **options**, opens/updates a TU:
`tu-binding-run-<date|milestone>` (trace only; content remains unchanged).

---

## 2) Inputs

- **Cold snapshot** (manuscript, codex, style addenda, optional plans/assets, translations).
- Gatekeeper’s latest pass notes (for presentation safety).
- Inclusion options from Showrunner (see §5).

---

## 3) Roles & Responsibilities

- **Book Binder (R)**
  - Assemble bundle from **Cold** only; ensure navigation, accessibility, and spoiler hygiene; stamp
    metadata.
- **Showrunner (A)**
  - Select snapshot & options; approve distribution.
- **Gatekeeper (C)**
  - Spot-check **Presentation Safety**, **Integrity**, **Style** on the built bundle.
- **PN (C)**
  - Confirms the bundle is suitable for **Narration Dry-Run** (same snapshot).
- **Style Lead (C)**
  - Sanity on tone consistency in visible surfaces (titles/captions).
- **Translator (C, optional)**
  - If translations included, verify coverage flags and link integrity.

---

## 4) Procedure

1. **Pick Snapshot & Options (Showrunner)**
   - Freeze a Cold snapshot (`cold@YYYY-MM-DDThh:mmZ` or tag).
   - Choose inclusions: art plan/renders, audio plan/assets, translations, print layout, PN script.

2. **Assemble Manuscript (Binder)**
   - Compile hyperlinked sections; ensure terminals clearly marked.
   - Generate **codeword/gateway checklists** in **player-safe wording** (no internal labels).

3. **Attach Codex**
   - Include player-safe entries; build **See also** crosslinks; add glossary if present.
   - Ensure all manuscript terms with codex references resolve.

4. **Include Optional Surfaces**
   - **Art**: plan captions (spoiler-safe) and/or renders with alt text; no technique/seed info on
     surface.
   - **Audio**: plan cue descriptions and/or assets with text equivalents; loudness safety notes.
   - **Translations**: add language slices; mark `complete` / `incomplete`; keep anchors intact.

5. **Accessibility & Navigation Pass**
   - Alt text, descriptive link text, proper headings; high-contrast print layout if enabled.
   - No motion/flash surprises; audio warnings where needed.

6. **Stamp Metadata**
   - Front matter fields: title, credits, **Cold snapshot ID**, options used, coverage %, Gatekeeper
     report ID.
   - Generate a **View Log** page (see §6).

7. **Export Formats**
   - Emit Markdown (source), HTML, EPUB, PDF (single-column, print-friendly).
   - Verify links/anchors across all formats.

8. **Gatekeeper Spot-Check**
   - Presentation Safety, Integrity (links), Style; note any fixes before distribution.

9. **Handoff**
   - Provide bundle to **PN** for **Narration Dry-Run**; archive the export with its metadata.

---

## 5) Export Options

| Option                | Default | Notes                                                         |
| --------------------- | ------- | ------------------------------------------------------------- |
| Include art plan      | off     | Captions/intent only; spoiler-safe.                           |
| Include art renders   | off     | Requires alt text; determinism logged in Cold (not surfaced). |
| Include audio plan    | off     | Cue descriptions & text equivalents.                          |
| Include audio assets  | off     | Provide captions/text equivalents; loudness safety.           |
| Include translations  | off     | Per-language coverage flags (`complete`/`incomplete`).        |
| Print-friendly layout | on      | Single column, high contrast, no color-only cues.             |
| PN script             | off     | If included, remains player-safe and in-voice.                |

---

## 6) Deliverables

- **Export bundle**:
  - `manuscript/` (Markdown source)
  - `codex/` (Markdown source)
  - `bundle.html`, `bundle.epub`, `bundle.pdf`
- **Front matter** in every format with: snapshot ID, options, coverage flags.
- **View Log** page:
  - Snapshot ID, date/time
  - Included languages & coverage %
  - Included plans/assets (art/audio)
  - **TU-IDs** merged since prior view (titles only, spoiler-safe)
  - Known limitations (e.g., “art plan present; renders deferred”)

---

## 7) Checklists

**Binder Preflight**

- [ ] Built from **Cold** snapshot (not Hot).
- [ ] All links/anchors resolve; terminals marked.
- [ ] No internal labels (codeword names, gate logic) on surfaces.
- [ ] Codex coverage sufficient for high-frequency terms; See also links resolve.
- [ ] Alt text present; descriptive link text; heading structure valid.
- [ ] Inclusion options correctly reflected; translations flagged `complete`/`incomplete`.
- [ ] Front matter & View Log stamped with snapshot ID and options.

**Gatekeeper Spot-Check**

- [ ] **Presentation Safety** (no spoilers, no technique talk).
- [ ] **Integrity** (no dead links across formats).
- [ ] **Style** (titles/captions/choice labels aligned to register).
- [ ] **Accessibility** basics satisfied.

---

## 8) Success Criteria

- The bundle is **player-safe**, navigable, and accessible.
- Snapshot & options are **clear and consistent** across formats.
- PN can run a **Narration Dry-Run** without encountering leaks.
- Known limitations are disclosed in the View Log.

---

## 9) Failure Modes & Remedies

- **Built from Hot** → Rebuild from Cold snapshot; restamp metadata.
- **Spoiler in captions/codex** → Move detail to canon notes; rewrite surface text.
- **Broken anchors** → Fix IDs; re-export all formats.
- **Accessibility gaps** → Add alt text; fix contrast; add audio text equivalents.
- **Mismatched options** → Ensure options table matches actual inclusions; update View Log.

---

## 10) RACI (quick)

| Task                      | R           | A          | C                             | I   |
| ------------------------- | ----------- | ---------- | ----------------------------- | --- |
| Choose snapshot & options | Showrunner  | Showrunner | Gatekeeper, Binder            | PN  |
| Assemble & export         | Book Binder | Showrunner | Gatekeeper, Style, Translator | PN  |
| Spot-check                | Gatekeeper  | Showrunner | Style Lead                    | All |
| Narration Dry-Run handoff | Binder      | Showrunner | PN                            | All |

---

**TL;DR**  
Cut a clean **view** from Cold: safe, linked, accessible, and clearly labeled. The Binder ships the
bundle; PN takes it for a spin; canon remains untouched.
