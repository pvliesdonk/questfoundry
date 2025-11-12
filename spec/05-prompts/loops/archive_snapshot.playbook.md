# Archive Snapshot — Executable Loop Playbook

**Category:** Export **Abbreviation:** AS **Schema:**
<https://questfoundry.liesdonk.nl/schemas/tu_brief.schema.json>

## Purpose

Create a comprehensive, versioned snapshot of the entire project state (Hot and Cold) at a
significant milestone for long-term archival, reproducibility, and provenance. Capture all
artifacts, TU history, gatecheck reports, schema versions, and metadata to enable future recovery,
audit, or analysis. Outcome: A timestamped archive package with complete project state, manifest,
and restoration instructions.

## Activation Criteria (Showrunner)

- Major milestone completion (chapter/act/book release)
- Before significant refactoring or architectural changes
- Periodic archival schedule (e.g., monthly, quarterly)
- Before team transitions or long breaks
- Legal/compliance requirements for version preservation

Showrunner opens a Trace Unit (TU): `tu-archive-snapshot-<milestone>` and coordinates archival
process.

## RACI Matrix

| Role        | Assignment | Responsibilities                                              |
| ----------- | ---------- | ------------------------------------------------------------- |
| Showrunner  | R/A        | Coordinates snapshot; verifies completeness; archives package |
| Book Binder | C          | Assists with artifact collection and manifest generation      |
| Gatekeeper  | C          | Verifies integrity of archived artifacts                      |
| All Roles   | I          | Informed of snapshot creation and archive location            |

## Inputs

- Current Cold snapshot with all merged artifacts
- Current Hot snapshot with work-in-progress
- All TU records (opened, stabilizing, gatecheck, cold-merged, closed)
- All Gatecheck Reports
- All Hook Cards
- Schema versions and specifications
- Configuration files and tool versions
- Post-Mortem Reports (if any)
- View Logs from Binding Runs

## Validation Requirements (All Artifacts)

**CRITICAL:** All roles producing JSON artifacts in this loop MUST validate before handoff.

**Refer to:** `_shared/validation_contract.md` (file #1 in your kit)

**For every artifact-producing step:**

1. **Producer role** looks up artifact schema in `SCHEMA_INDEX.json`
2. **Producer role** runs preflight protocol (echo schema metadata + examples)
3. **Producer role** produces artifact with `"$schema"` field
4. **Producer role** validates and emits `validation_report.json`
5. **If validation fails:** STOP loop immediately, escalate to Showrunner
6. **Showrunner** verifies both files exist before accepting handoff:
   - `/out/artifact.json` (with valid `"$schema"` field)
   - `/out/artifact_validation_report.json` (with `"valid": true`)
7. **Only then** proceed to next step

**Hard gate:** Invalid artifacts STOP the loop. No exceptions.

**Showrunner responsibility:**

Before allowing any role-to-role handoff, verify:

- Artifact file exists and has `"$schema"` field
- validation_report.json exists
- validation_report.json shows `"valid": true` with empty `"errors": []`

If any validation fails, STOP the loop and escalate to human for guidance.

## Procedure (Message Sequences)

### Step 1: Prepare Snapshot

**Showrunner** opens TU with `tu.open` intent, broadcasting to all roles:

```json
{
  "intent": "tu.open",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-YYYY-MM-DD-SR##",
    "snapshot": "Cold @ YYYY-MM-DD",
    "loop": "Archive Snapshot"
  },
  "payload": {
    "type": "tu_brief",
    "data": {
      "loop": "Archive Snapshot",
      "slice": "Create comprehensive snapshot at [milestone]",
      "awake": ["SR", "BB", "GK"],
      "press": ["Integrity", "Determinism"],
      "inputs": [
        "Cold snapshot with all merged artifacts",
        "Hot snapshot with active TUs",
        "All TU records, Gatecheck reports, Hook cards"
      ],
      "deliverables": [
        "Archive package with complete project state",
        "Manifest with checksums",
        "Archive index entry"
      ]
    }
  }
}
```

Freeze current state:

- Cold snapshot with merge date
- Hot snapshot with active TU states
- Timestamp and version tag

### Step 2: Collect Artifacts

**Book Binder** gathers all artifacts and metadata:

- Manuscript sections (all versions in Cold/Hot)
- Canon Packs (spoiler-level lore)
- Codex entries (player-safe surfaces)
- Hook Cards (all statuses: proposed/accepted/deferred/rejected)
- TU Briefs (complete lifecycle history)
- Gatecheck Reports (all decisions and bar statuses)
- Style Addenda and motif kits
- Art Plans and renders (with determinism logs)
- Audio Plans and assets (with reproducibility notes)
- Language Packs (all translation slices)
- View Logs (all exports)
- Schema files and versions

### Step 3: Generate Manifest

**Book Binder** sends `archive.manifest` intent to Showrunner with comprehensive listing:

```json
{
  "intent": "archive.manifest",
  "payload": {
    "type": "tu_brief",
    "data": {
      "manifest": {
        "snapshot_ids": {
          "cold": "Cold @ YYYY-MM-DD",
          "hot": "Hot @ YYYY-MM-DD"
        },
        "artifacts_collected": {
          "manuscript_sections": ##,
          "canon_packs": ##,
          "codex_entries": ##,
          "hook_cards": ##,
          "tu_briefs": ##,
          "gatecheck_reports": ##
        },
        "schema_versions": {
          "hook_card": "0.2.0",
          "tu_brief": "0.2.0"
        },
        "checksums": "SHA-256 for all files",
        "total_files": ##
      }
    }
  }
}
```

Manifest includes:

- All files with checksums (SHA-256)
- Directory structure
- Schema versions used
- Tool versions and dependencies
- Snapshot IDs (Cold and Hot)
- Archive creation date and creator
- Restoration instructions

### Step 4: Verify Integrity

**Gatekeeper** sends `archive.verify` intent to Showrunner after spot-checking:

```json
{
  "intent": "archive.verify",
  "payload": {
    "type": "ack",
    "data": {
      "message": "Archive integrity verified; all checksums valid; critical artifacts present (## TUs, ## gatechecks, ## hooks); manifest completeness confirmed; restoration instructions testable"
    }
  }
}
```

Verification checks:

- All checksums valid
- Critical artifacts present
- Manifest completeness
- Restoration instructions testable

### Step 5: Package and Store

**Book Binder** sends `archive.package` intent to Showrunner:

```json
{
  "intent": "archive.package",
  "payload": {
    "type": "tu_brief",
    "data": {
      "archive_package": {
        "filename": "questfoundry-[milestone]-YYYY-MM-DD.tar.gz",
        "size_mb": ##,
        "checksum_sha256": "[64-char hash]",
        "storage_locations": [
          "/archives/local/...",
          "s3://questfoundry-archives/...",
          "/backup/offline/..."
        ],
        "restoration_tested": "yes"
      }
    }
  }
}
```

Package creation:

- Compressed archive (tar.gz or zip)
- Signed with checksum/signature
- Stored in multiple locations (local, cloud, offline backup)
- Cataloged in archive index

### Step 6: Document Snapshot

**Showrunner** broadcasts `tu.update` intent with final cataloging:

```json
{
  "intent": "tu.update",
  "payload": {
    "type": "tu_brief",
    "data": {
      "deliverables": [
        "Archive package: [filename] ([size] MB, [count] files)",
        "Archive stored in 3 locations: local, cloud, offline",
        "Archive index entry cataloged"
      ],
      "linkage": "[Milestone] project state archived; restoration tested; team notified"
    }
  }
}
```

Record in project history:

- Archive creation date
- Snapshot IDs captured
- Archive location(s)
- Restoration tested (yes/no/partial)
- Notes on significant changes since last archive

## Deliverables

- **Archive Package:**
  - Complete project state (Hot and Cold)
  - All artifacts and metadata
  - Manifest with checksums
  - Restoration instructions
  - Schema files and tool versions
- **Archive Index Entry:**
  - Snapshot metadata
  - Archive location(s)
  - Creation date and creator
  - Restoration status

## Success Criteria

- All critical artifacts included in archive
- Manifest checksums validate
- Archive stored in multiple locations
- Restoration instructions documented and testable
- Archive cataloged and findable
- Team informed of archive creation

## Failure Modes & Remedies

- **Incomplete artifact collection** → Use manifest checklist; verify against TU registry
- **Checksum failures** → Re-archive from source; verify file integrity
- **Untested restoration** → Test restoration on clean system; update instructions as needed
- **Single point of failure** → Store archives in at least 3 locations (local, cloud, offline)
- **Missing metadata** → Ensure snapshot IDs, schema versions, and tool versions recorded

## Quality Bars Pressed

**Primary:** Integrity (complete and verifiable archive)

**Secondary:** Determinism (reproducibility of project state), Presentation (clear documentation)

## Handoffs

- **To Showrunner:** Archive confirmation and location reference
- **To All Roles:** Archive creation notification with snapshot IDs and restoration contact
- **To Archive Index:** Cataloged entry for future retrieval
- **To Future Team:** Complete, recoverable project state for continuity
