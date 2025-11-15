---
procedure_id: front_matter_composition
name: Front Matter Composition
description: Compose front matter with snapshot ID, included options (art/audio plan-or-assets), language coverage, accessibility summary
roles: [book_binder]
references_schemas:
  - view_export.schema.json
  - snapshot.schema.json
references_expertises:
  - book_binder_export
  - style_lead_voice
quality_bars: [presentation, accessibility]
---

# Front Matter Composition

## Purpose

Provide clear, player-safe front matter that documents what's included in the view, accessibility features, and build details without exposing internals or spoilers.

## Core Principles

- **Player-Safe Content**: No spoilers, internal labels, or technique exposure
- **Transparency**: Clear about what's included and what isn't
- **Accessibility First**: Summarize accessibility features and known limitations
- **Reproducibility**: Document snapshot ID and options for traceability

## Steps

1. **Document Snapshot Source**: Record build provenance
   - Snapshot ID (for reproducibility)
   - Build date/timestamp
   - Version identifier (if applicable)

2. **List Included Options**: Specify what's in this view
   - Art status: assets included, plans only, or none
   - Audio status: assets included, plans only, or none
   - Language coverage: which languages/slices included
   - Format: MD/HTML/EPUB/PDF

3. **Compose Accessibility Summary**: Document accessibility features
   - Alt text presence/coverage
   - Audio caption/text equivalent presence
   - Contrast and print-friendly status
   - Navigation features (TOC, headings, links)
   - Known limitations or gaps

4. **Add Usage Guidance**: Brief, player-safe usage notes (if needed)
   - How to navigate (if not obvious)
   - Recommended reading order (if non-linear)
   - How to access multilingual content (if applicable)

5. **Coordinate Phrasing with Style Lead**: Ensure front matter tone aligns
   - Register consistent with project voice
   - Labels and headings clear and in-world (where applicable)
   - No meta jargon or internals

6. **Verify No Spoilers**: Ensure front matter is plot-safe
   - No twist telegraph in descriptions
   - No revealing section titles or summaries
   - Keep technique (seeds/models) off-surface

## Outputs

- **Front Matter Package**: Containing:
  - Snapshot ID and build information
  - Included options (art/audio/language status)
  - Accessibility summary (features and limitations)
  - Usage guidance (if needed)
  - Any necessary credits or attribution
  - Player-safe labels and navigation cues

## Quality Checks

- Front matter contains no spoilers or plot reveals
- No internal labels, codewords, or technique exposure
- Accessibility features clearly documented
- Known gaps or limitations transparently stated
- Snapshot ID present for reproducibility
- Included options accurately reflect view content
- Phrasing aligned with Style Lead's register
- Navigation and usage guidance clear and helpful
- All headings and labels player-safe and in-voice
