---
procedure_id: manifest_generation
description: Generate comprehensive archive manifest with checksums and metadata
version: 1.0.0
references_expertises:
  - book_binder_export
references_schemas:
  - archive_manifest.schema.json
references_roles:
  - book_binder
  - showrunner
tags:
  - archive
  - manifest
  - provenance
---

# Manifest Generation Procedure

## Overview

Create comprehensive listing of all archived artifacts with checksums, metadata, and version information for long-term provenance and recovery.

## Source

Extracted from v1 `spec/05-prompts/loops/archive_snapshot.playbook.md` Step 3: "Generate Manifest"

## Steps

### Step 1: Enumerate Artifacts

List all artifacts being archived:

- Manuscript sections (all versions in Cold/Hot)
- Canon Packs (spoiler-level lore)
- Codex entries (player-safe surfaces)
- Hook Cards (all statuses: proposed/accepted/deferred/rejected)
- TU Briefs (complete lifecycle history)
- Gatecheck Reports (all decisions and bar statuses)

### Step 2: Include Assets and Plans

Add generated assets and planning artifacts:

- Style Addenda and motif kits
- Art Plans and renders (with determinism logs)
- Audio Plans and assets (with reproducibility notes)
- Language Packs (all translation slices)
- View Logs (all exports)

### Step 3: Record Schema Versions

Document schema specifications:

- Schema files and versions
- SCHEMA_INDEX.json snapshot
- Configuration files
- Tool versions used

### Step 4: Compute Checksums

Generate integrity verification:

- SHA-256 hash for each artifact file
- File sizes and timestamps
- Directory structure checksums

### Step 5: Add Snapshot Metadata

Include snapshot identifiers and provenance:

- Cold snapshot ID with merge date
- Hot snapshot ID with active TU states
- Archive timestamp and version tag
- Milestone or trigger reason

### Step 6: Assemble Manifest

Create archive_manifest.json with comprehensive listing and validate against schema.

## Output

Archive manifest with complete artifact listing, checksums, versions, and metadata.

## Quality Criteria

- All artifacts enumerated
- SHA-256 checksums present for all files
- Schema versions documented
- Snapshot IDs traceable
- Manifest validates against schema
