# Playbook — Binding Run

**Use when:** You need a **player-safe export** from a specific **Cold snapshot** for release,
playtest, or review.

**Outcome:** Export bundle (Markdown/HTML/EPUB/PDF) with front matter, View Log, and all player
surfaces stamped with snapshot ID. No new content created; this reads Cold only.

---

## 1) One-minute scope (Showrunner)

- [ ] Pick **Cold snapshot** (tag or timestamp).
- [ ] Choose **inclusions**: art (plan/renders), audio (plan/assets), translations, print layout, PN
      script.
- [ ] Open/update TU: `tu-binding-run-<date|milestone>` (trace only, no Hot changes).

---

## 2) Inputs you need on screen

- **Cold snapshot** (manuscript, codex, style, optional plans/assets/translations).
- Gatekeeper's latest pass notes (Presentation Safety reference).
- Export option decisions from Showrunner (see §5 in full loop doc).

---

## 3) Do the thing (compact procedure)

**Book Binder (R)**

1. Assemble hyperlinked manuscript; mark terminals clearly.
2. Attach codex with **See also** crosslinks; ensure anchors resolve.
3. Include optional surfaces (art/audio plans or renders, translations) with **player-safe**
   captions/text equivalents.
4. Accessibility & navigation pass: alt text, descriptive links, proper headings, high contrast.
5. Stamp **front matter** with snapshot ID, options, coverage flags.
6. Generate **View Log** page: snapshot ID, TU titles merged since prior view, limitations noted.
7. Export formats: Markdown, HTML, EPUB, PDF; verify links/anchors across all.

**Gatekeeper (C)** 8. Spot-check **Presentation Safety**, **Integrity** (links), **Style**, and
**Accessibility**.

**PN (C)** 9. Confirms bundle is suitable for **Narration Dry-Run**.

---

## 4) Deliverables (archive, not merged)

- **Export bundle**: manuscript + codex + front matter + View Log in multiple formats.
- **View Log** with snapshot ID, included languages/coverage, art/audio status, TU titles
  (spoiler-safe).
- **Gatekeeper spot-check note** on Presentation/Integrity/Style/Accessibility.
- Archived with snapshot reference for reproducibility.

---

## 5) Hand-off map

- → **Narration Dry-Run** (PN playtests this exact bundle).
- → Distribution (release, review, playtest).
- → Archive (snapshots + exports preserved).

---

## 6) Definition of "done" (for this loop)

- [ ] Built from **Cold** only (not Hot).
- [ ] All links/anchors resolve; no internal labels on surfaces.
- [ ] Codex coverage sufficient; See also links work.
- [ ] Alt text present; navigation clear; accessibility basics satisfied.
- [ ] Front matter & View Log stamped correctly.
- [ ] Gatekeeper green on Presentation/Integrity/Style/Accessibility.
- [ ] PN confirms readiness for dry-run.

---

## 7) Fast triage rubric (Showrunner)

- **Ship now**: milestone met; bars passed; PN-ready; snapshot stable.
- **Defer**: waiting for deferred art/audio assets; translation coverage incomplete but disclosed.
- **Block**: links broken, spoilers visible, accessibility failures require fixes.

---

## 8) RACI (micro)

| Task               | R           | A          | C                     | I   |
| ------------------ | ----------- | ---------- | --------------------- | --- |
| Snapshot & options | Showrunner  | Showrunner | Gatekeeper, PN        | All |
| Assemble bundle    | Book Binder | Showrunner | Gatekeeper, Style     | PN  |
| Spot-check         | Gatekeeper  | Showrunner | Binder, Style, PN     | All |
| Distribution       | Showrunner  | Showrunner | Binder (archival), PN | All |
