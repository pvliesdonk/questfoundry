---
snippet_id: cold_manifest_validation
name: Cold Manifest Validation
description: Preflight check - block Binder if manifest fails (missing files, SHA mismatch, missing assets, missing approval, order gaps)
applies_to_roles: [gatekeeper, book_binder]
quality_bars: [integrity, presentation]
phase: preflight
---

# Cold Manifest Validation

## Core Principle

Before any view export, validate the Cold Manifest completely. No heuristic fixes allowed—manifest must be corrected at source.

## Preflight Checks

### File Completeness

- [ ] All referenced sections exist
- [ ] No missing dependencies
- [ ] All crosslinks resolve
- [ ] TOC entries have targets

**Block if:**

- Any referenced file missing
- Any section ID unresolved
- Any dependency not found

### SHA-256 Integrity

- [ ] All files have SHA-256 checksums
- [ ] Checksums match current file state
- [ ] No silent modifications since approval

**Block if:**

- SHA-256 mismatch detected
- File modified after approval
- Checksum missing for included file

### Asset Coverage

- [ ] All images referenced exist
- [ ] All audio files referenced exist
- [ ] All assets have required metadata
- [ ] Alt text present for images
- [ ] Text equivalents present for audio

**Block if:**

- Missing image file
- Missing audio file
- Missing alt text
- Missing text equivalent
- Asset metadata incomplete

### Approval Metadata

- [ ] All sections have approval timestamps
- [ ] All sections have gatecheck pass
- [ ] All quality bars satisfied
- [ ] Approval chain complete

**Block if:**

- Section not gatechecked
- Missing approval metadata
- Quality bar failures unresolved
- Approval older than content

### Section Order

- [ ] No gaps in section sequence
- [ ] Navigation paths complete
- [ ] Hub connections valid
- [ ] Gateway references exist

**Block if:**

- Section order has gaps
- Missing navigation targets
- Broken gateway references
- Hub without outbound paths

## No Heuristic Fixes

**CRITICAL:** Gatekeeper does NOT attempt to fix manifest issues.

❌ Do NOT:

- Generate missing files
- Guess at checksums
- Skip missing assets
- Assume approval
- Fill gaps with placeholders

✓ Instead:

- BLOCK export
- Report specific failures
- Assign owner for fixes
- Wait for corrected manifest
- Re-validate after fixes

## Failure Reporting

When manifest validation fails, report:

```yaml
gate_result: BLOCK
reason: cold_manifest_validation_failure
failures:
  - type: missing_file
    file: sections/cargo_bay_12.md
    referenced_by: manifest.json
    owner: scene_smith
  - type: sha_mismatch
    file: sections/airlock_03.md
    expected: abc123...
    actual: def456...
    owner: book_binder
  - type: missing_asset
    asset: images/frost_viewport.png
    referenced_by: sections/observation_01.md
    owner: illustrator
```

## Ownership Assignment

**Missing Files:**

- Scene Smith: manuscript sections
- Lore Weaver: canon packs (should not be in Cold manifest)
- Codex Curator: codex entries

**SHA Mismatches:**

- Book Binder: investigate modification source
- Scene Smith: if prose changed post-approval
- Gatekeeper: if quality bar checks altered

**Missing Assets:**

- Illustrator: images
- Audio Producer: audio files
- Book Binder: asset metadata

**Missing Approval:**

- Showrunner: coordinate re-gatecheck
- Original author: address quality bar issues

**Section Order Gaps:**

- Plotwright: fix topology
- Book Binder: correct manifest sequence

## Validation Workflow

1. **Receive Export Request** (from Showrunner via Binder)
2. **Load Cold Manifest** (snapshot ID specified)
3. **Run Preflight Checks** (all validation rules)
4. **Collect Failures** (if any)
5. **Block or Pass:**
   - If failures: BLOCK with detailed report
   - If clean: Proceed to content validation
6. **Route Fixes** (if blocked, assign to owners)
7. **Re-validate** (after fixes applied)

## Common Failures

**Post-Approval Edits:**

- File modified after gatecheck
- SHA-256 no longer matches
- Re-gatecheck required

**Incomplete Merges:**

- Hot content merged to Cold without full approval
- Missing quality bar checks
- Orphaned references

**Asset Pipeline Lag:**

- Images rendered but not committed
- Audio files not synced to Cold
- Alt text not yet written

**Topology Changes:**

- Sections added/removed after manifest created
- Gateway references not updated
- Navigation paths broken
