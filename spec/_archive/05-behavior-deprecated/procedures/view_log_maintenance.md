---
procedure_id: view_log_maintenance
name: View Log Maintenance
description: Maintain View Log entries and minimal anchor maps; record player-safe TU titles and known limitations
roles: [book_binder]
references_schemas:
  - view_log.schema.json
references_expertises:
  - book_binder_export
quality_bars: [integrity, presentation]
---

# View Log Maintenance

## Purpose

Maintain a history of exported views with enough detail to support debugging, reproducibility, and understanding of what shipped when—while keeping entries player-safe.

## Core Principles

- **Historical Record**: Track all view exports over time
- **Reproducibility**: Enable recreating any prior view from snapshot + options
- **Player-Safe Entries**: Log descriptions contain no spoilers or internals
- **Debug Support**: Enough detail to diagnose issues post-export
- **Minimal Overhead**: Lightweight tracking, not comprehensive documentation

## Steps

1. **Create View Log Entry**: For each export, record:
   - **View ID**: Unique identifier for this export
   - **Export Timestamp**: When view was generated
   - **Snapshot ID**: Which Cold snapshot was source
   - **Export Options**: What was included
     - Art status (assets/plans/none)
     - Audio status (assets/plans/none)
     - Languages included
     - Format(s) generated (MD/HTML/EPUB/PDF)
   - **Known Limitations**: Documented gaps or constraints
     - Partial art coverage
     - Audio plan-only (not rendered)
     - Partial translation coverage
     - Format-specific issues

2. **Record Player-Safe TU Titles**: Link to source work
   - TU IDs that contributed to this snapshot
   - Player-safe TU titles (no spoilers or internals)
   - General themes (e.g., "Plotwright topology work", "Style tune-up")

3. **Attach Minimal Anchor Map**: For debugging
   - Critical anchors (sections, keystones, major codex entries)
   - Not exhaustive—just key navigation points
   - Human-readable format
   - Player-safe labels

4. **Document Assembly Notes**: Record non-semantic changes
   - Format-specific normalizations applied
   - Known rendering quirks (e.g., "EPUB doesn't support audio embeds")
   - Accessibility adaptations
   - Keep brief and player-safe

5. **Note Coverage Status**: Track what's included
   - Section count (e.g., "42 manuscript sections")
   - Codex entry count
   - Art asset count (if included)
   - Audio cue count (if included)
   - Translation coverage percentage (if applicable)

6. **Link to Prior Views**: Maintain history
   - Reference to previous view (if iterative exports)
   - Note major differences (e.g., "added 5 sections, 12 new codex entries")

7. **Store Log Entry**: Save to View Log
   - Indexed by View ID
   - Searchable by snapshot ID, timestamp, options
   - Accessible for debugging and audits

## Outputs

- **View Log Entry**: Complete record containing:
  - View ID and timestamp
  - Snapshot ID and export options
  - Known limitations and gaps
  - Player-safe TU titles
  - Coverage status
  - Assembly notes (non-semantic normalizations)
  - Link to prior views
- **Minimal Anchor Map**: Critical navigation anchors for debugging
- **Historical View Index**: Updated log of all exports

## Quality Checks

- Every export has a View Log entry
- Entries contain snapshot ID for reproducibility
- Export options accurately documented
- Known limitations transparently stated
- TU titles player-safe (no spoilers or internals)
- Anchor map minimal but useful for debugging
- Assembly notes brief and player-safe
- Coverage status accurate
- Log searchable and accessible
- Historical continuity maintained (prior view links)
