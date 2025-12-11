---
procedure_id: binder_presentation_enforcement
name: Binder Presentation Bar Enforcement
description: Enforce Presentation bar on assembly; no leaked internals in front matter or navigation labels
roles: [book_binder]
references_schemas:
  - view_export.schema.json
references_expertises:
  - book_binder_export
  - gatekeeper_presentation
quality_bars: [presentation]
---

# Binder Presentation Bar Enforcement

## Purpose

Ensure the assembled view meets Presentation bar standards with no leaked internals, spoilers, or technique in any player-visible surface including front matter and navigation.

## Core Principles

- **No Internals**: No codewords, flags, internal labels, or build metadata on surfaces
- **No Technique**: No seeds, models, DAW/tool names in visible content
- **No Spoilers**: Front matter and labels remain non-revealing
- **Diegetic Navigation**: Navigation text stays in-world (no meta markers)
- **Clean Assembly**: Surface-level formatting only, no content "fixes"

## Steps

1. **Review Front Matter**: Check for Presentation violations
   - No internal labels or codeword names
   - No spoiler hints in descriptions or summaries
   - No build technique (seeds, models, tools) mentioned
   - Accessibility notes player-safe (no internals)
   - Credits/attribution don't reveal spoilers

2. **Audit Navigation Labels**: Ensure in-world phrasing
   - Section titles player-safe (no twist reveals)
   - TOC entries non-spoiling
   - Breadcrumb text diegetic
   - Link text descriptive, not meta ("See Salvage Permits" not "click here")
   - No flags or internal markers (no "FLAG_X", "CODEWORD: ...")

3. **Check Asset Metadata**: Verify no technique leakage
   - Image metadata/EXIF cleaned (no seeds/models)
   - Audio file metadata clean (no DAW/plugin names)
   - Caption/alt text free of technique
   - File names player-safe (not "scene_42_TWIST_reveal.jpg")

4. **Validate Codex Integration**: Check encyclopedia surfaces
   - Entry titles and headings player-safe
   - Crosslink labels descriptive
   - No spoiler hints in "See also" lists
   - Lineage/notes sections non-revealing

5. **Verify Localized Content**: If multilingual
   - Translation maintains Presentation bar
   - No meta leakage in translated labels
   - Cultural adaptations remain player-safe

6. **Spot-Check Sample Sections**: Random audit for issues
   - Read through sample manuscript sections
   - Check a few codex entries
   - Review captions and alt text
   - Test navigation flow

7. **Flag Issues (Don't Fix)**: If Presentation violations found
   - Create hooks to owning roles for upstream fixes
   - Don't rewrite content in Binder step
   - Document issues for Gatekeeper export spot-check

## Outputs

- **Presentation Validation Report**: Confirmation that:
  - Front matter contains no internals, spoilers, or technique
  - Navigation labels are diegetic and player-safe
  - Asset metadata clean
  - Codex integration maintains bar
  - Localized content (if any) clean
- **Violation Hooks**: Requests for upstream fixes (if issues found)
- **Export Spot-Check Request**: Submission to Gatekeeper before view ships

## Quality Checks

- No codewords, flags, or internal labels visible anywhere
- No seeds, models, or tool names in any surface
- Front matter and navigation labels non-revealing
- Section titles don't telegraph twists
- Link text descriptive and in-world
- Asset metadata clean (no EXIF/file leakage)
- Codex surfaces maintain spoiler hygiene
- Translation (if any) preserves Presentation bar
- Issues reported to owners for upstream fixes
- Gatekeeper export spot-check requested before ship
