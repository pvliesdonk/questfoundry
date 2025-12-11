---
procedure_id: integrity_enforcement
description: Validate anchors, links, and cross-references resolve correctly across all export surfaces
version: 1.0.0
references_expertises:
  - book_binder_export
references_schemas:
  - view_bundle.schema.json
  - anchor_map.schema.json
references_roles:
  - book_binder
tags:
  - integrity
  - validation
  - navigation
---

# Integrity Enforcement Procedure

## Overview

Ensure all references, links, and anchors within the exported view resolve correctly across manuscript, codex, captions, and localized content before distribution.

## Source

Extracted from v1 `spec/05-prompts/book_binder/system_prompt.md` anchor validation and crosslink checking

## Steps

### Step 1: Collect All Anchors

Build inventory of link targets:

- Section anchors in manuscript
- Codex entry anchors
- Heading IDs and custom anchor points
- Figure/image anchors
- Audio cue references

### Step 2: Collect All References

Build inventory of links:

- Internal manuscript links (section → section)
- Codex crosslinks (entry → entry)
- Manuscript ↔ codex bidirectional references
- Caption/alt text references
- Navigation links (TOC, breadcrumbs)

### Step 3: Validate Resolution

Check every reference resolves:

- Each link target exists as an anchor
- No broken references (link without target)
- Anchor IDs unique (no collisions)
- Orphaned content logged (unreferenced anchors OK but noted)

### Step 4: Check Navigation Integrity

Verify navigation functional:

- TOC links resolve correctly
- Section ordering logical
- Breadcrumbs work (if applicable)
- Next/previous links functional

### Step 5: Normalize Anchor IDs

Apply Cold SoT anchor conventions:

- Convert to lowercase-dash format (e.g., `s1-return`, `dock-seven`)
- Replace underscores with dashes
- Remove apostrophes/primes
- Create alias map for legacy IDs
- Rewrite all href="#OldID" to href="#canonical-id"

### Step 6: Verify Multilingual Consistency

If localized content present:

- Anchors consistent across language slices
- References resolve within each language
- Cross-language references handled

### Step 7: Generate Anchor Map

Create debugging resource:

- List of critical anchors (sections, codex entries)
- Reference counts per anchor
- Orphaned content (zero incoming links)
- Format: human-readable, player-safe labels

### Step 8: Report Issues

For problems found:

- Broken references → flag to owning role (Scene Smith, Codex Curator)
- Label collisions → flag to Style Lead
- Navigation friction → flag to Plotwright
- Log in view_log for traceability

## Output

Integrity validation confirming all links resolve, anchor map for debugging, and issue reports for upstream fixes.

## Quality Criteria

- All internal links resolve to valid anchors
- No broken references in any surface
- Anchor IDs unique and normalized
- TOC and navigation functional
- No orphan pages (unreachable content)
- Multilingual slices maintain integrity
- Anchor map generated and logged
- Issues reported to owning roles (not silently fixed)
