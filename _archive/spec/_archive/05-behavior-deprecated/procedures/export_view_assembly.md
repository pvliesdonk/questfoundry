---
procedure_id: export_view_assembly
name: Export View Assembly
description: Build export views from Cold snapshot chosen by Showrunner; no Hot/Cold mixing; single snapshot source only
roles: [book_binder]
references_schemas:
  - view_export.schema.json
  - snapshot.schema.json
references_expertises:
  - book_binder_export
quality_bars: [integrity, presentation, accessibility]
---

# Export View Assembly

## Purpose

Assemble reproducible, accessible bundles from Cold snapshot content that players can experience through various export formats (MD/HTML/EPUB/PDF).

## Core Principles

- **Cold-Only Rule**: NEVER export from Hot; NEVER mix Hot & Cold sources
- **Single Snapshot Source**: Entire view built from one snapshot for consistency
- **Format Compliance**: Support multiple export formats while maintaining integrity
- **Reproducibility**: Same snapshot + options = identical view

## Steps

1. **Receive Export Request**: Get from Showrunner:
   - Cold snapshot ID (authoritative source)
   - Export options (art/audio: plan vs assets; languages; layout preferences)
   - Format targets (MD/HTML/EPUB/PDF)

2. **Verify Snapshot Integrity**: Validate snapshot before assembly
   - Confirm snapshot exists and is complete
   - Check file existence for all referenced content
   - Verify SHA-256 hashes match manifest
   - Validate asset approval metadata
   - Check section ordering

3. **Load Source Content**: Pull from specified Cold snapshot only
   - Manuscript sections in correct order
   - Codex entries with crosslinks
   - Localized slices (if applicable)
   - Art assets or plans (per options)
   - Audio assets or plans (per options)

4. **Assemble View Structure**: Build export structure
   - Table of contents and navigation
   - Section ordering per snapshot
   - Codex integration points
   - Asset placements

5. **Resolve References**: Ensure all links work
   - Internal anchors resolve
   - Crosslinks between manuscript and codex land correctly
   - Navigation elements functional
   - No orphan pages or broken references

6. **Apply Format-Specific Processing**: Convert to target format(s)
   - Maintain semantic structure
   - Preserve accessibility features
   - Apply layout per format requirements
   - Keep presentation boundaries clean

7. **Generate View Bundle**: Package complete export
   - All requested formats
   - Front matter (see Front Matter Composition)
   - Asset files (if included)
   - Necessary metadata

## Outputs

- **View Export Result**: Complete bundle in requested format(s)
- **View Anchor Map**: Human-readable list of critical anchors and targets (for debugging)
- **View Assembly Notes**: Brief, player-safe list of non-semantic normalizations applied

## Quality Checks

- Entire view sourced from single Cold snapshot (no Hot content)
- Snapshot integrity validated before assembly
- All references resolve (no broken links or orphan pages)
- TOC functional across all formats
- Navigation enables reaching all content
- Format conversions preserve semantic structure
- No Hot/Cold mixing anywhere in view
- Reproducible: same snapshot + options = identical view
