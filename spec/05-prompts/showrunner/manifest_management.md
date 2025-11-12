# Showrunner — Manifest Management

**Module:** Hot/Cold manifest operations and snapshot management

## Overview

Showrunner maintains Hot and Cold manifests as master indices of project state. All file operations,
merges, and snapshots flow through manifest updates.

## Schema Reference

All schemas available at: `https://questfoundry.liesdonk.nl/schemas/`

## Master Manifests

### hot_manifest.json — Discovery State Tracking

**Schema:** <https://questfoundry.liesdonk.nl/schemas/hot_manifest.schema.json>

**Contents:**

- In-progress TUs, hooks, research memos, canon packs, style addenda
- Draft sections and proposed assets awaiting approval
- Gatecheck reports and view logs
- References Cold snapshot via `cold_reference` field

**Purpose:** Tracks all active work and proposed changes before they become canonical.

### cold/manifest.json — Canonical State Tracking

**Schema:** <https://questfoundry.liesdonk.nl/schemas/cold_manifest.schema.json>

**Contents:**

- All approved files with SHA-256 hashes for deterministic builds
- References: `cold/book.json`, `cold/art_manifest.json`, `cold/fonts.json`, `cold/build.lock.json`
- Optionally includes `cold/project_metadata.json` for Binder

**Purpose:** Single source of truth for approved, player-ready content.

## Orchestration Responsibilities

Showrunner enforces manifest integrity at key workflow points:

### Before Opening TU

- Verify Hot manifest exists and is readable
- Verify Cold reference is set in Hot manifest
- If manifests missing: Error and block TU open

### Before Binding Run

- Ensure Gatekeeper validates Cold manifest (preflight checks)
- Verify all referenced files in manifest are present
- Confirm SHA-256 hashes match for deterministic build

**Cross-Reference:** See `../loops/binding_run.playbook.md` for complete binding workflow.

### On Hot → Cold Merge

1. Gatekeeper approves Hot changes (see `../gatekeeper/` and `../loops/gatecheck.playbook.md`)
2. Update `cold/manifest.json` with new file hashes
3. Recompute manifest checksum
4. Update Hot manifest's `cold_reference` to new snapshot

### On Snapshot Creation

1. Generate new `snapshot_id` (timestamp-based or semantic versioning)
2. Update Cold manifest with snapshot metadata
3. Update Hot manifest to reference new snapshot
4. Archive previous snapshot per retention policy

**Cross-Reference:** See `../loops/archive_snapshot.playbook.md` for snapshot procedures.

## Layer 6 Implementation Note

**Storage Agnostic:** Manifests are logical structures; implementations may use:

- JSON files (reference implementation)
- SQLite (for query performance)
- Redis (for distributed systems)
- Other backends (as needed)

**Schema Contract:** Regardless of storage backend, all implementations must validate against
canonical schemas at `https://questfoundry.liesdonk.nl/schemas/`.

## Manifest Validation

Showrunner should validate manifests before critical operations:

**Hot Manifest Checks:**

- Schema compliance (use hot_manifest.schema.json)
- Valid `cold_reference` points to existing Cold snapshot
- All TU references are well-formed

**Cold Manifest Checks:**

- Schema compliance (use cold_manifest.schema.json)
- All referenced files exist and hashes match
- No orphaned references

**Error Handling:** If validation fails, use `error` intent with `validation_error` type and
specific remediation steps. See `protocol_handlers.md` for error taxonomy.

## Traceability

Every manifest update should preserve:

- **Upstream TU** - Which TU drove the change
- **Correlation ID** - Link to original request/gatecheck
- **Snapshot ID** - Which snapshot was active during change
- **Timestamp** - When change occurred

**Cross-Reference:** See `protocol_handlers.md` for complete traceability requirements.
