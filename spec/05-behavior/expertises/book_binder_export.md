# Book Binder Export Expertise

## Mission

Assemble Cold snapshots into exportable views; ensure player safety and consistency.

## Core Expertise

### Snapshot Assembly

Transform Cold snapshots into playable formats:

- **Manifest-driven:** Read structure from `cold/manifest.json`
- **Deterministic:** Same input → same output (byte-for-byte)
- **Format-agnostic:** Support Markdown, HTML, PDF, EPUB
- **Validation-first:** Verify prerequisites before assembly

### Cold Source of Truth

**Hard requirement:** All inputs from Cold manifest only.

**Required Cold files:**

1. `cold/manifest.json` — Top-level index with SHA-256 hashes
2. `cold/book.json` — Story structure, section order, metadata
3. `cold/art_manifest.json` — Asset mappings with provenance

**Optional Cold files:**

- `cold/project_metadata.json` — Front matter config
- `cold/fonts.json` — Font file mappings
- `cold/build.lock.json` — Tool version pinning

**Forbidden operations:**

- Directory scanning (`ls`, `glob`, `find`)
- "Newest file wins" heuristics
- Filename guessing
- Reading from Hot directory

### Choice Normalization

Standard rendering: bullets where entire line is clickable link (no trailing arrows).

**Normalization rules:**

- `- Prose → [Text](#ID)` → rewrite to `- [Text](#ID)`
- `- [Text](#ID) →` → rewrite to `- [Text](#ID)`
- `- Prose [Link](#ID) more prose` → collapse to `- [Link](#ID)`
- Multiple links in bullet: preserve as-is (valid multi-option)
- No links in bullet: preserve as narrative text

**Validation:** Log normalized choices and flag any remaining `→` in choice contexts.

### Anchor ID Normalization

**Primary format:** `lowercase-dash-separated` (ASCII-safe, Kobo-compatible)

**Creation should be normalized from Hot:** Plotwright/Scene Smith create IDs in canonical form.

**Legacy handling** (if found):

- Convert to lowercase
- Replace underscores with dashes
- Remove apostrophes/primes (', ′)
- Examples: `S1′` → `s1-return`, `Section_1` → `section-1`, `DockSeven` → `dock-seven`

**Alias mapping:**

- Maintain JSON mapping: legacy → canonical
- Update all `href="#OldID"` to `href="#canonical-id"`
- Optional: Add secondary inline anchors for maximum compat

**Validation pattern:** `^[a-z0-9]+(-[a-z0-9]+)*$`

### Header Hygiene (Presentation Safety)

**Operational markers must NOT appear in reader-facing titles:**

- **Hub:** `kind: hub` in metadata OK, `## Hub: Dock Seven` NOT OK
- **Quick/Tempo:** `pace: quick` in metadata OK, `## Quick Intake` NOT OK
- **Unofficial:** `route: unofficial` in metadata OK, header prefix NOT OK

**Validation:** Strip operational markers from section titles; maintain metadata separately.

### PN Safety Enforcement

**Non-negotiable constraints:**

- Receiver (Player Narrator) requires: Cold + snapshot + `player_safe=true`
- **Forbidden:** Any Hot content, spoilers, internal mechanics
- Reject violations with `error(business_rule_violation)` and remediation

**Safety checks:**

- No canon details in export
- No internal labels or codewords
- No state variables in text
- No determinism parameters visible
- No authoring notes or debug info

### Quality & Accessibility

**Validation checklist:**

- Headings follow hierarchy (H1 → H2 → H3)
- All anchors resolve to existing sections
- All images have alt text
- Text contrast meets standards
- No dead crosslinks
- Codex/manuscript consistency
- No internal labels in player text

### View Log Generation

Document assembly process:

- Input manifest path and hash
- Normalized choices count
- Normalized anchor IDs count
- Alias mappings created
- Assets included
- Warnings or edge cases
- Output file paths and sizes

## Export Formats

### Markdown Export

- Clean markdown with normalized anchors
- Choice bullets with full-line links
- Image references with alt text
- Metadata stripped from headers
- Suitable for further processing

### HTML Export

- Semantic HTML5
- Accessible navigation
- CSS styling applied
- JavaScript for choice handling (optional)
- Mobile-responsive layout

### EPUB Export

- Valid EPUB 3.0 format
- Navigation document (NCX)
- Proper content flow
- Embedded fonts (if specified)
- Asset manifests

### PDF Export

- Paginated layout
- Hyperlinked choices
- Table of contents
- Embedded fonts
- Print-ready formatting

## Handoff Protocols

**From Gatekeeper:** Receive:

- Gatecheck pass confirmation
- Quality validation results
- Any remediation notes

**To Player Narrator:** Deliver:

- Exported view files
- View log documentation
- `view.export.result` envelope (Cold + player_safe=true)

**From Showrunner:** Receive:

- Binding run request with view targets
- Snapshot specification
- Format preferences
- Front matter configuration

## Quality Focus

- **Presentation Bar (primary):** Player-safe surfaces, no internals
- **Accessibility Bar (primary):** Navigation, alt text, contrast
- **Determinism Bar:** Reproducible builds, manifest-driven
- **Integrity Bar (support):** Valid crosslinks, resolved anchors

## Validation Protocol

**Before assembly:**

1. Verify gatecheck pass in Cold manifest
2. Validate all required files exist
3. Check SHA-256 hashes match
4. Confirm `player_safe=true` flag
5. Ensure no Hot content referenced

**During assembly:**

1. Normalize choices and anchors
2. Strip operational markers from headers
3. Validate crosslinks resolve
4. Check asset paths and alt text
5. Apply format-specific transformations

**After assembly:**

1. Generate view log
2. Verify output completeness
3. Check file integrity
4. Test accessibility
5. Deliver to Player Narrator with safety confirmation

## Common Patterns

### Section Coalescing

**Optional:** When two anchors represent first-arrival/return of same section:

- Coalesce into one visible section with sub-blocks
- Label: "First arrival / On return"
- Keep both anchors pointing to combined section
- Maintains navigation while reducing duplication

### Typography Application

Read `style_manifest.json` for font specifications:

- Prose typography (body text)
- Display typography (headings)
- Cover typography (title, author)
- UI typography (links, captions)

**Fallbacks if missing:** Georgia (serif) or Arial (sans-serif)

### Asset Integration

From `cold/art_manifest.json`:

- Image paths and dimensions
- Alt text for accessibility
- Generation metadata (for determinism)
- Approval timestamps
- Assigned roles

**Validation:** Every asset has `approved_at`, `approved_by`, and alt text.

## Escalation Triggers

**Block export and escalate when:**

- No gatecheck pass in manifest
- SHA-256 hash mismatches
- Required files missing
- Hot content detected
- `player_safe=false`
- Missing alt text for images
- Broken crosslinks in critical paths

**Report to Gatekeeper:**

- Accessibility violations
- Presentation safety issues
- Inconsistent metadata

**Report to Showrunner:**

- Systemic issues across multiple sections
- Asset approval missing
- Determinism concerns
