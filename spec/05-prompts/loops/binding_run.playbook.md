# Binding Run — Executable Loop Playbook

**Category:** Export **Abbreviation:** BR **Schema:**
<https://questfoundry.liesdonk.nl/schemas/view_log.schema.json>

## Purpose

Assemble a player-safe export view of the book from a specific Cold snapshot. Package manuscript,
codex, and checklists—optionally including art/audio plans or assets and translation slices—without
leaking spoilers or internal plumbing. Outcome: A stamped export bundle (Markdown/HTML/EPUB/PDF)
with front-matter metadata (snapshot ID, options used) and a View Log. No new canon is created; this
loop does not write to Cold.

## Activation Criteria (Showrunner)

- Milestone release (chapter/act/book)
- Playtest build needed for PN Narration Dry-Run
- External review or print/export request

Showrunner chooses a Cold snapshot and export options, opens/updates a TU:
`tu-binding-run-<date|milestone>` (trace only; content remains unchanged).

## RACI Matrix

| Role            | Assignment | Responsibilities                                                                                      |
| --------------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| Book Binder     | R          | Assemble bundle from Cold only; ensure navigation, accessibility, and spoiler hygiene; stamp metadata |
| Showrunner      | A          | Select snapshot and options; approve distribution                                                     |
| Gatekeeper      | C          | Spot-check Presentation Safety, Integrity, Style on the built bundle                                  |
| Player-Narrator | C          | Confirms the bundle is suitable for Narration Dry-Run (same snapshot)                                 |
| Style Lead      | C          | Sanity on tone consistency in visible surfaces (titles/captions)                                      |
| Translator      | C          | If translations included, verify coverage flags and link integrity                                    |

## Inputs

- Cold snapshot (manuscript, codex, style addenda, optional plans/assets, translations)
- Gatekeeper's latest pass notes (for presentation safety)
- Inclusion options from Showrunner

## Validation Requirements (All Artifacts)

**CRITICAL:** All roles producing JSON artifacts in this loop MUST validate before handoff.

**Refer to:** `_shared/validation_contract.md` (file #1 in your kit)

**For every artifact-producing step:**

1. **Producer role** looks up artifact schema in `SCHEMA_INDEX.json`
2. **Producer role** runs preflight protocol (echo schema metadata + examples)
3. **Producer role** produces artifact with `"$schema"` field
4. **Producer role** validates and emits `validation_report.json`
5. **If validation fails:** STOP loop immediately, escalate to Showrunner
6. **Showrunner** verifies both files exist before accepting handoff:
   - `/out/artifact.json` (with valid `"$schema"` field)
   - `/out/artifact_validation_report.json` (with `"valid": true`)
7. **Only then** proceed to next step

**Hard gate:** Invalid artifacts STOP the loop. No exceptions.

**Showrunner responsibility:**

Before allowing any role-to-role handoff, verify:

- Artifact file exists and has `"$schema"` field
- validation_report.json exists
- validation_report.json shows `"valid": true` with empty `"errors": []`

If any validation fails, STOP the loop and escalate to human for guidance.

## Procedure (Message Sequences)

### Step 1: Pick Snapshot & Options (Showrunner)

Freeze a Cold snapshot (`cold@YYYY-MM-DDThh:mmZ` or tag). Choose inclusions: art plan/renders, audio
plan/assets, translations, print layout, PN script.

```json
{
  "intent": "view.export.request",
  "sender": "SR",
  "receiver": "BB",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Binding Run"
  },
  "payload": {
    "type": "view_log",
    "data": {
      "view_name": "milestone-chapter-3-export",
      "snapshot": "Cold @ 2025-10-28",
      "export_options": {
        "formats": ["markdown", "html", "epub", "pdf"],
        "include_art_plans": false,
        "include_art_renders": false,
        "include_audio_plans": false,
        "include_audio_assets": false,
        "include_translations": false,
        "print_friendly_layout": true,
        "include_pn_script": false
      }
    }
  }
}
```

### Step 2: Assemble Manuscript (Binder)

Compile hyperlinked sections; ensure terminals clearly marked. Generate codeword/gateway checklists
in player-safe wording (no internal labels).

### Step 3: Attach Codex

Include player-safe entries; build See also crosslinks; add glossary if present. Ensure all
manuscript terms with codex references resolve.

### Step 4: Include Optional Surfaces

- Art: plan captions (spoiler-safe) and/or renders with alt text; no technique/seed info on surface
- Audio: plan cue descriptions and/or assets with text equivalents; loudness safety notes
- Translations: add language slices; mark `complete` / `incomplete`; keep anchors intact

### Step 5: Accessibility & Navigation Pass

Alt text, descriptive link text, proper headings; high-contrast print layout if enabled. No
motion/flash surprises; audio warnings where needed.

### Step 6: Stamp Metadata

Front matter fields: title, credits, Cold snapshot ID, options used, coverage %, Gatekeeper report
ID. Generate a View Log page.

### Step 7: Export Formats

Emit Markdown (source), HTML, EPUB, PDF (single-column, print-friendly). Verify links/anchors across
all formats.

### Step 8: Gatekeeper Spot-Check

Presentation Safety, Integrity (links), Style; note any fixes before distribution.

```json
{
  "intent": "view.export.result",
  "sender": "BB",
  "receiver": "SR",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Binding Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "forbidden"
  },
  "payload": {
    "type": "view_log",
    "data": {
      "view_name": "milestone-chapter-3-export",
      "snapshot": "Cold @ 2025-10-28",
      "manifest": ["manuscript/section-01.md", "manuscript/section-02.md", "codex/factions.md"],
      "formats_generated": ["markdown", "html", "epub", "pdf"],
      "coverage": {
        "manuscript_sections": 12,
        "codex_entries": 24,
        "art_plans": 0,
        "translations_languages": []
      }
    }
  }
}
```

### Step 9: Handoff

Provide bundle to PN for Narration Dry-Run; archive the export with its metadata.

## Deliverables

- **Export bundle:**
  - `manuscript/` (Markdown source)
  - `codex/` (Markdown source)
  - `bundle.html`, `bundle.epub`, `bundle.pdf`
- **Front matter** in every format with: snapshot ID, options, coverage flags
- **View Log** page:
  - Snapshot ID, date/time
  - Included languages and coverage %
  - Included plans/assets (art/audio)
  - TU-IDs merged since prior view (titles only, spoiler-safe)
  - Known limitations (e.g., "art plan present; renders deferred")

## Success Criteria

- The bundle is player-safe, navigable, and accessible
- Snapshot and options are clear and consistent across formats
- PN can run a Narration Dry-Run without encountering leaks
- Known limitations are disclosed in the View Log

## Failure Modes & Remedies

- **Built from Hot** → Rebuild from Cold snapshot; restamp metadata
- **Spoiler in captions/codex** → Move detail to canon notes; rewrite surface text
- **Broken anchors** → Fix IDs; re-export all formats
- **Accessibility gaps** → Add alt text; fix contrast; add audio text equivalents
- **Mismatched options** → Ensure options table matches actual inclusions; update View Log

## Quality Bars Pressed

**Primary:** Integrity (anchors/links), Presentation (no internals), Accessibility (front-matter)

**Secondary:** Style (titles/captions consistency)

## Handoffs

- **To Player-Narrator:** Bundle for Narration Dry-Run (same snapshot)
- **To Gatekeeper:** Spot-check results for any presentation/integrity issues before wide
  distribution
