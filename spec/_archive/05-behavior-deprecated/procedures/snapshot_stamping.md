---
procedure_id: snapshot_stamping
description: Timestamp and version tag snapshots for archival provenance
version: 1.0.0
references_expertises:
  - book_binder_export
references_schemas:
  - archive_manifest.schema.json
  - snapshot.schema.json
references_roles:
  - showrunner
  - book_binder
tags:
  - archive
  - versioning
  - provenance
---

# Snapshot Stamping Procedure

## Overview

Freeze current project state with timestamp and version tag for archival, reproducibility, and provenance tracking.

## Source

Extracted from v1 `spec/05-prompts/loops/archive_snapshot.playbook.md` Step 1: "Prepare Snapshot"

## Steps

### Step 1: Freeze Current State

Capture snapshot identifiers:

- Cold snapshot ID with merge date (e.g., "Cold @ 2025-11-15")
- Hot snapshot ID with active TU states (e.g., "Hot @ 2025-11-15")
- List of active TUs and their states

### Step 2: Generate Timestamp

Create ISO 8601 timestamp for archival:

- Archive creation time (UTC)
- Milestone or trigger description
- Archive sequence number if periodic

### Step 3: Create Version Tag

Assign version identifier:

- Semantic version if release milestone (e.g., v1.0.0)
- Date-based tag if periodic snapshot (e.g., snapshot-2025-11-15)
- Milestone tag if event-driven (e.g., chapter-3-complete)

### Step 4: Record Provenance

Document snapshot context:

- Trigger reason (milestone, periodic, incident, transition)
- Participating roles and active TUs
- Snapshot scope (full project, subset, branch)

### Step 5: Tag Repository

Create git tag if applicable:

- Tag name with version or date
- Tag message with milestone description
- Push tag to remote for preservation

### Step 6: Update Snapshot Metadata

Include stamping information in archive manifest and snapshot artifact.

## Output

Timestamped and versioned snapshot with provenance metadata ready for archival.

## Quality Criteria

- Timestamp in ISO 8601 UTC format
- Version tag follows project conventions
- Provenance clearly documented
- Git tag created if applicable
- Snapshot ID traceable
