# Archive Restoration Guide

## Purpose

Instructions for restoring QuestFoundry project state from an archive snapshot.

## Prerequisites

- Archive manifest with complete file listing
- Schema versions matching archive
- QuestFoundry tooling compatible with archive version

## Restoration Steps

### Step 1: Verify Archive Integrity

- Check SHA-256 checksums for all files
- Validate manifest completeness
- Confirm schema version compatibility

### Step 2: Restore Directory Structure

- Create Cold and Hot directories
- Restore file hierarchy from manifest
- Preserve permissions and timestamps

### Step 3: Restore Artifacts

- Copy all manuscript sections
- Restore canon packs and codex entries
- Restore TU records and gatecheck reports
- Restore assets (art, audio, translations)

### Step 4: Rebuild Indexes

- Regenerate SCHEMA_INDEX.json if needed
- Rebuild manifest references
- Update tool configuration

### Step 5: Validate Restoration

- Run spec compiler validation
- Check cross-references resolve
- Verify assets accessible
- Confirm no missing dependencies

## Recovery Scenarios

### Partial Restoration

Restore subset of project (e.g., specific acts, translations):

- Use manifest to identify required files
- Restore dependencies recursively
- Validate subset completeness

### Version Migration

Restore from older archive version:

- Check schema migration path
- Apply transformation scripts
- Validate migrated artifacts
- Update version tags

## Troubleshooting

- **Missing files**: Check manifest SHA-256, verify archive integrity
- **Schema mismatch**: Consult migration guides, consider version upgrade
- **Broken references**: Run integrity validation, identify missing dependencies
