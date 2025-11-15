---
procedure_id: binder_integrity_enforcement
name: Binder Integrity Enforcement
description: Ensure anchors/links/refs across manuscript, codex, captions, localized slices; generate anchor maps for debugging
roles: [book_binder]
references_schemas:
  - view_export.schema.json
references_expertises:
  - book_binder_export
quality_bars: [integrity]
---

# Binder Integrity Enforcement

## Purpose

Ensure all references, links, and anchors within the exported view resolve correctly across manuscript, codex, captions, and localized content.

## Core Principles

- **Complete Resolution**: Every link must have a valid target
- **Cross-Surface Integrity**: Links work across manuscript ↔ codex ↔ captions
- **Multilingual Consistency**: Localized slices maintain reference integrity
- **No Dead Ends**: No orphan pages or unreachable content
- **Debuggability**: Generate anchor maps for issue diagnosis

## Steps

1. **Collect All Anchors**: Build inventory of link targets
   - Section anchors in manuscript
   - Codex entry anchors
   - Heading IDs
   - Figure/image anchors
   - Audio cue references
   - Custom anchor points

2. **Collect All References**: Build inventory of links
   - Internal manuscript links (section → section)
   - Codex crosslinks (entry → entry)
   - Manuscript → codex links
   - Codex → manuscript references
   - Caption/alt text references
   - Navigation links (TOC, breadcrumbs)

3. **Validate Resolution**: Check every reference
   - Each link target exists as an anchor
   - No broken references (link without target)
   - No orphaned anchors (unreferenced content is OK, but log it)
   - Anchor IDs unique (no collisions)

4. **Check Navigation Integrity**: Verify navigation works
   - TOC links resolve correctly
   - Section ordering makes sense
   - Breadcrumbs (if any) functional
   - Next/previous links work (if applicable)

5. **Verify Multilingual Consistency**: If localized content present
   - Anchors consistent across language slices
   - References resolve within each language
   - Cross-language references handled (if applicable)

6. **Generate Anchor Map**: Create debugging resource
   - List of critical anchors (sections, codex entries, keystones)
   - Reference counts per anchor
   - Orphaned content (anchors with zero incoming links)
   - Format: human-readable, player-safe labels

7. **Create Hooks for Issues**: If problems found
   - Broken references → hook to owning role (Scene/Curator)
   - Label collisions → hook to Style/Scene
   - Navigation friction → hook to Plotwright

## Outputs

- **Integrity Validation Report**: Confirmation that:
  - All links resolve to valid targets
  - No broken references
  - No anchor ID collisions
  - Navigation functional
  - Multilingual consistency maintained (if applicable)
- **Anchor Map**: Human-readable list of critical anchors and targets
- **Issue Hooks**: Requests for upstream fixes (if problems found)

## Quality Checks

- Every internal link resolves to a valid anchor
- No broken references in any surface (manuscript, codex, captions)
- Anchor IDs are unique across entire view
- TOC and navigation links functional
- No orphan pages (content unreachable via navigation)
- Multilingual slices maintain reference integrity
- Anchor map generated for debugging
- Issues reported to owning roles (don't fix in Binder)
