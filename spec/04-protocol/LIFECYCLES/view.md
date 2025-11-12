# View Lifecycle — Export and PN Dry-Run

> **Status:** Normative — this document defines the canonical view/export lifecycle and PN boundary
> enforcement in Layer 4 protocol.

---

## 1. Overview

This specification defines the **View/Export lifecycle**: the state transitions from snapshot
selection through export binding to PN dry-run feedback, enforcing Cold-only and player-safe
constraints at every boundary.

### Purpose

View lifecycle ensures that exported artifacts and PN narration sessions are reproducible,
player-safe, and traceable to Cold snapshots. The lifecycle ensures:

- **Cold-only inputs** — PN and Binder consume only Cold snapshots, never Hot material
- **Player-safety enforced** — all exported content respects spoiler hygiene
- **Reproducibility** — snapshot references enable identical re-exports
- **PN boundary protection** — PN never sees internals, codewords, or author plumbing

### Design Principles

- **Explicit snapshot reference** — every view/export MUST cite Cold snapshot ID
- **PN sees Cold only** — PN consumes player-safe Cold surfaces, never Hot
- **Envelope validation** — messages without valid snapshot are rejected
- **Error clarity** — missing/invalid snapshots produce actionable errors

---

## 2. State Machine

### 2.1 States

The View/Export lifecycle has **5 states**:

```
snapshot-selected → export-binding → pn-dry-run → feedback-collected → view-published
                   ↘ export-failed (if binding errors occur)
```

**States:**

- `snapshot-selected` — Showrunner selects Cold snapshot for export
- `export-binding` — Book Binder generates player-safe export artifacts (PDF, HTML, EPUB, etc.)
- `pn-dry-run` — Player Narrator playtests the bound view for UX issues
- `feedback-collected` — PN playtest notes recorded and routed to owners
- `view-published` — View log finalized and archived (terminal)
- `export-failed` — Binding errors prevent export completion (terminal)

**Terminal states:** `view-published`, `export-failed`

---

## 3. State Transitions

### 3.1 Transition Matrix

| From State           | To State             | Allowed Sender | Intent             | Required Payload   | Notes                              |
| -------------------- | -------------------- | -------------- | ------------------ | ------------------ | ---------------------------------- |
| `snapshot-selected`  | `export-binding`     | BB             | `view.bind`        | view_log (partial) | Binder starts export generation    |
| `snapshot-selected`  | `export-failed`      | BB             | `view.bind_failed` | error              | Binding errors prevent export      |
| `export-binding`     | `pn-dry-run`         | BB             | `view.bound`       | view_log (full)    | Export artifacts ready for PN      |
| `export-binding`     | `export-failed`      | BB             | `view.bind_failed` | error              | Binding errors during generation   |
| `pn-dry-run`         | `feedback-collected` | PN             | `view.feedback`    | pn_playtest_notes  | PN dry-run complete, issues logged |
| `feedback-collected` | `view-published`     | SR or BB       | `view.publish`     | view_log (full)    | View archived and announced        |

**Role Abbreviations:**

- **BB** — Book Binder (export generation authority)
- **PN** — Player Narrator (dry-run playtest authority)
- **SR** — Showrunner (view publication authority)

---

## 4. Transition Details

### 4.1 `snapshot-selected` → `export-binding`

**Preconditions:**

- Showrunner selects Cold snapshot for export
- Gatecheck passed for snapshot content
- Export targets and options defined (PDF, HTML, EPUB, art/audio/locale coverage)

**Sender:** BB (Book Binder)

**Intent:** `view.bind`

**Payload Schema:** `view_log.schema.json` (partial)

**Required Fields:**

```json
{
  "title": "<View name>",
  "bound": "YYYY-MM-DD",
  "binder": "<BB name or agent>",
  "tu": "TU-YYYY-MM-DD-BBnn",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "targets": ["PDF", "HTML", "EPUB"],
  "options_and_coverage": "Art: plans-only; Audio: none; Locales: en-US 100%"
}
```

**Effects:**

- Binder begins export generation
- Snapshot locked for reproducibility
- Export artifacts in progress

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `snapshot-selected` state
- `NOT_AUTHORIZED` — if sender is not BB
- `VALIDATION_FAILED` — if snapshot reference missing or invalid
- `SNAPSHOT_NOT_FOUND` — if snapshot does not exist in Cold

---

### 4.2 `snapshot-selected` → `export-failed` OR `export-binding` → `export-failed`

**Preconditions:**

- Export binding attempted
- Critical errors prevent artifact generation (broken links, missing assets, schema violations)

**Sender:** BB (Book Binder)

**Intent:** `view.bind_failed`

**Payload Schema:** `error` (custom)

**Required Fields:**

```json
{
  "code": "EXPORT_BINDING_FAILED",
  "message": "Export binding failed due to errors",
  "details": {
    "snapshot": "Cold @ YYYY-MM-DD",
    "targets": ["PDF", "HTML"],
    "errors": [
      "Broken link in section 17: target 'hub_missing' does not exist",
      "Missing alt text for image in section 23"
    ],
    "remedy": "Fix broken link and add alt text; resubmit for binding"
  }
}
```

**Effects:**

- Export binding terminated
- Errors logged and routed to owners
- Remediation required before retry

**Error Cases:**

- `EXPORT_BINDING_FAILED` — binding process encountered critical errors
- `SNAPSHOT_CORRUPTED` — snapshot contains invalid or inconsistent data
- `ASSET_MISSING` — required asset (image, audio) not found

---

### 4.3 `export-binding` → `pn-dry-run`

**Preconditions:**

- Export artifacts successfully generated
- All targets (PDF, HTML, EPUB, etc.) ready
- Anchor map validated (no orphans, collisions resolved)
- Presentation and Accessibility bars validated

**Sender:** BB (Book Binder)

**Intent:** `view.bound`

**Payload Schema:** `view_log.schema.json` (full)

**Required Fields:**

```json
{
  "title": "<View name>",
  "bound": "YYYY-MM-DD",
  "binder": "<BB name or agent>",
  "tu": "TU-YYYY-MM-DD-BBnn",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "targets": ["PDF", "HTML", "EPUB"],
  "options_and_coverage": "Art: plans-only; Audio: none; Locales: en-US 100%; Research: uncorroborated:low; Accessibility: alt text 95%",
  "dormancy": ["deferred:art", "deferred:audio"],
  "anchor_map": "Manuscript: 120 sections; Codex: 45 entries; Crosslinks: 89; Locales: en-US; Orphans: 0; Collisions: 0",
  "presentation_status": "green",
  "accessibility_status": "yellow",
  "accessibility_notes": "5% images missing alt text; scheduled for Art Touch-up TU",
  "gatekeeper": "GK Bot v2.1",
  "gatecheck_id": "GK-2025-10-30-01",
  "export_artifacts": [
    {
      "kind": "PDF",
      "path": "exports/cold-2025-10-30/manuscript.pdf",
      "hash": "sha256:abc123..."
    },
    {
      "kind": "HTML",
      "path": "exports/cold-2025-10-30/index.html",
      "hash": "sha256:def456..."
    },
    {
      "kind": "EPUB",
      "path": "exports/cold-2025-10-30/manuscript.epub",
      "hash": "sha256:ghi789..."
    }
  ]
}
```

**Effects:**

- Export artifacts ready for PN consumption
- PN triggered for dry-run playtest
- View log published (partial state)

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `export-binding` state
- `NOT_AUTHORIZED` — if sender is not BB
- `VALIDATION_FAILED` — if view_log fields incomplete

---

### 4.4 `pn-dry-run` → `feedback-collected`

**Preconditions:**

- PN completed dry-run playtest of bound view
- UX issues logged in playtest notes
- All issues tagged, prioritized, and assigned to owners

**Sender:** PN (Player Narrator)

**Intent:** `view.feedback`

**Payload Schema:** `pn_playtest_notes.schema.json` (full)

**Required Fields:**

```json
{
  "title": "<View name>",
  "run": "YYYY-MM-DD HH:MM",
  "pn": "<PN name or agent>",
  "tu": "TU-YYYY-MM-DD-PNnn",
  "snapshot": "Cold @ YYYY-MM-DD",
  "targets": ["PDF", "HTML", "EPUB"],
  "mode": "dry-run (no improv)",
  "log": [
    {
      "when": "12:34",
      "location": "chapter_2/section_17.md#L34",
      "tags": ["choice-ambiguity"],
      "severity": "med",
      "snippet": "Choice text: 'Take the left path or right path' — both unclear",
      "smallest_viable_fix": "Add context: 'left path (toward the docks)' and 'right path (toward the markets)'",
      "owner": "SS"
    },
    {
      "when": "14:56",
      "location": "chapter_3/hub_18.md",
      "tags": ["tone-wobble"],
      "severity": "low",
      "snippet": "Hub 18 uses informal register; others formal",
      "smallest_viable_fix": "Adjust hub_18.md register to match formal tone",
      "owner": "ST"
    }
  ]
}
```

**Effects:**

- PN feedback logged
- Issues routed to responsible owners
- View feedback complete

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `pn-dry-run` state
- `NOT_AUTHORIZED` — if sender is not PN
- `VALIDATION_FAILED` — if playtest notes incomplete or missing required fields
- `SNAPSHOT_MISMATCH` — if PN snapshot does not match view log snapshot

---

### 4.5 `feedback-collected` → `view-published`

**Preconditions:**

- PN feedback reviewed and routed
- View log finalized with all artifacts, gatechecks, and feedback
- Showrunner approves view publication

**Sender:** SR (Showrunner) or BB (Book Binder)

**Intent:** `view.publish`

**Payload Schema:** `view_log.schema.json` (full, final)

**Required Fields:** (same as 4.3, with additional confirmation)

```json
{
  "title": "<View name>",
  "bound": "YYYY-MM-DD",
  "binder": "<BB name or agent>",
  "tu": "TU-YYYY-MM-DD-BBnn",
  "cold_snapshot": "Cold @ YYYY-MM-DD"
  // ... (all view_log fields from 4.3)
}
```

**Effects:**

- View log archived (terminal)
- View announced to stakeholders
- Export artifacts preserved for reproducibility

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `feedback-collected` state
- `NOT_AUTHORIZED` — if sender is not SR or BB
- `VALIDATION_FAILED` — if view log incomplete

---

## 5. PN Boundary Rules (Critical)

### 5.1 Cold-Only Constraint

**Rule:** PN MUST consume only Cold snapshots; PN MUST NEVER access Hot material

**Enforcement:**

- All messages with `receiver.role === "PN"` MUST have `context.hot_cold === "cold"`
- All PN messages MUST include valid `context.snapshot` reference (format: `"Cold @ YYYY-MM-DD"`)

**Error:** `PN_HOT_BOUNDARY_VIOLATION` if PN receives Hot context

**Example error:**

```json
{
  "code": "PN_HOT_BOUNDARY_VIOLATION",
  "message": "PN cannot access Hot material",
  "details": {
    "receiver": "PN",
    "context_hot_cold": "hot",
    "rule": "PN consumes only Cold snapshots",
    "reference": "00-north-star/PN_PRINCIPLES.md §3",
    "remedy": "Use Cold snapshot for PN dry-run"
  }
}
```

### 5.2 Player-Safe Constraint

**Rule:** All content consumed by PN MUST be player-safe (`safety.player_safe: true`)

**Enforcement:**

- All messages with `receiver.role === "PN"` MUST have `safety.player_safe === true`
- All PN outputs (narration, playtest notes) MUST be player-safe (no spoilers)

**Error:** `PN_PLAYER_SAFE_VIOLATION` if PN receives non-player-safe content

**Example error:**

```json
{
  "code": "PN_PLAYER_SAFE_VIOLATION",
  "message": "PN cannot access non-player-safe content",
  "details": {
    "receiver": "PN",
    "safety_player_safe": false,
    "rule": "PN consumes only player-safe surfaces",
    "reference": "00-north-star/PN_PRINCIPLES.md §2",
    "remedy": "Filter content for player-safety before sending to PN"
  }
}
```

### 5.3 Snapshot Reference Requirement

**Rule:** All PN messages MUST include `context.snapshot` with valid Cold snapshot ID

**Enforcement:**

- `context.snapshot` MUST match pattern `^Cold @ \d{4}-\d{2}-\d{2}$`
- Snapshot MUST exist in Cold SoT

**Error:** `SNAPSHOT_MISSING` or `SNAPSHOT_INVALID` if snapshot missing or malformed

**Example error:**

```json
{
  "code": "SNAPSHOT_MISSING",
  "message": "PN message missing required snapshot reference",
  "details": {
    "receiver": "PN",
    "context_snapshot": null,
    "rule": "All PN messages require context.snapshot",
    "reference": "04-protocol/LIFECYCLES/view.md §5.3",
    "remedy": "Include valid Cold snapshot reference in context.snapshot"
  }
}
```

### 5.4 What PN May See

**Allowed:**

- Hyperlinked manuscript view (sections, choices, surface text)
- Player-safe codex pages and cross-refs
- Style guidance (voice/register motifs)
- Export options (art plans, audio plans, locale coverage)

**Forbidden:**

- Codeword names or gate logic
- Hot topology or WIP drafts
- Author notes, hooks, or canon contradictions
- Internal plumbing (section numbers, RNG seeds, schema fields)

See `00-north-star/PN_PRINCIPLES.md` for complete boundaries.

---

## 6. Envelope Context Requirements

All view/export lifecycle messages MUST include proper envelope context per `ENVELOPE.md`:

### 6.1 Hot/Cold Boundaries

- **Cold context** (`hot_cold: "cold"`):
  - REQUIRED for all PN messages and export binding messages
  - MUST include `snapshot` reference (format: `"Cold @ YYYY-MM-DD"`)
  - `safety.player_safe: true` for PN-consumed content

- **Hot context** (`hot_cold: "hot"`):
  - Used for internal coordination (view planning, feedback routing)
  - NEVER used for PN messages

### 6.2 TU Linkage

All view/export messages MUST include:

- `context.tu` — the TU ID coordinating the view/export
- `context.snapshot` — the Cold snapshot being exported
- `refs` — array of related TU/gatecheck IDs for traceability

### 6.3 Loop Context

All view/export messages MUST include:

- `context.loop` — loop name from enum:
  - "Binding Run" — for export binding messages
  - "Narration Dry-Run" — for PN playtest messages

---

## 7. Validation Rules

### 7.1 Snapshot Reference Validation

**Rule:** All export/PN messages MUST include valid `context.snapshot`

- **Check:** `context.snapshot` matches pattern `^Cold @ \d{4}-\d{2}-\d{2}$`
- **Check:** Snapshot exists in Cold SoT
- **Error:** `SNAPSHOT_MISSING` if null, `SNAPSHOT_INVALID` if malformed, `SNAPSHOT_NOT_FOUND` if
  not in Cold

**Example error:**

```json
{
  "code": "SNAPSHOT_INVALID",
  "message": "Snapshot reference format invalid",
  "details": {
    "snapshot": "Cold@2025-10-30",
    "expected_format": "Cold @ YYYY-MM-DD",
    "remedy": "Use correct format: 'Cold @ 2025-10-30'"
  }
}
```

### 7.2 PN Boundary Validation

**Rule:** PN messages MUST have `hot_cold: "cold"` and `player_safe: true`

- **Check:** If `receiver.role === "PN"`, verify `context.hot_cold === "cold"` and
  `safety.player_safe === true`
- **Error:** `PN_HOT_BOUNDARY_VIOLATION` or `PN_PLAYER_SAFE_VIOLATION` if violated

**Example error:**

```json
{
  "code": "PN_HOT_BOUNDARY_VIOLATION",
  "message": "PN cannot access Hot material",
  "details": {
    "receiver": "PN",
    "context_hot_cold": "hot",
    "rule": "PN consumes only Cold snapshots",
    "remedy": "Change context.hot_cold to 'cold' and include snapshot"
  }
}
```

### 7.3 Snapshot Consistency Validation

**Rule:** All messages in a view lifecycle chain MUST reference the same snapshot

- **Check:** `context.snapshot` consistent across view.bind → view.bound → view.feedback →
  view.publish
- **Error:** `SNAPSHOT_MISMATCH` if snapshot differs

**Example error:**

```json
{
  "code": "SNAPSHOT_MISMATCH",
  "message": "PN playtest snapshot does not match view log snapshot",
  "details": {
    "view_snapshot": "Cold @ 2025-10-28",
    "pn_snapshot": "Cold @ 2025-10-30",
    "rule": "All messages in view lifecycle must use same snapshot",
    "remedy": "Ensure PN uses snapshot from view log"
  }
}
```

### 7.4 Export Artifact Validation

**Rule:** All export artifacts MUST be reproducible (hash recorded)

- **Check:** `export_artifacts[].hash` present for all artifacts
- **Error:** `VALIDATION_FAILED` if hash missing

**Example error:**

```json
{
  "code": "VALIDATION_FAILED",
  "message": "Export artifact missing hash for reproducibility",
  "details": {
    "artifact": "manuscript.pdf",
    "rule": "All artifacts require hash for reproducibility",
    "remedy": "Compute and record artifact hash (e.g., sha256)"
  }
}
```

---

## 8. Error Definitions

### 8.1 Snapshot Errors

| Error Code           | Trigger                                    | Remedy                               |
| -------------------- | ------------------------------------------ | ------------------------------------ |
| `SNAPSHOT_MISSING`   | `context.snapshot` is null or undefined    | Include valid snapshot reference     |
| `SNAPSHOT_INVALID`   | Snapshot format does not match pattern     | Use format `"Cold @ YYYY-MM-DD"`     |
| `SNAPSHOT_NOT_FOUND` | Snapshot does not exist in Cold SoT        | Verify snapshot exists before export |
| `SNAPSHOT_CORRUPTED` | Snapshot data is invalid or inconsistent   | Investigate snapshot integrity       |
| `SNAPSHOT_MISMATCH`  | Snapshot differs across lifecycle messages | Use consistent snapshot ID           |

### 8.2 PN Boundary Errors

| Error Code                  | Trigger                                        | Remedy                           |
| --------------------------- | ---------------------------------------------- | -------------------------------- |
| `PN_HOT_BOUNDARY_VIOLATION` | PN message has `hot_cold: "hot"`               | Change to `hot_cold: "cold"`     |
| `PN_PLAYER_SAFE_VIOLATION`  | PN message has `player_safe: false`            | Change to `player_safe: true`    |
| `PN_SPOILER_DETECTED`       | PN content contains spoilers (codewords, etc.) | Filter content for player-safety |

### 8.3 Export Binding Errors

| Error Code              | Trigger                                    | Remedy                             |
| ----------------------- | ------------------------------------------ | ---------------------------------- |
| `EXPORT_BINDING_FAILED` | Critical errors during artifact generation | Fix errors (broken links, etc.)    |
| `ASSET_MISSING`         | Required asset (image, audio) not found    | Add missing asset or mark deferred |
| `ANCHOR_COLLISION`      | Duplicate anchor IDs in manuscript         | Resolve anchor collision           |
| `BROKEN_LINK`           | Link target does not exist                 | Fix or remove broken link          |

---

## 9. Cross-References

### Layer 0/1 Policy

- `00-north-star/PN_PRINCIPLES.md` — PN boundaries, inputs/outputs, diegetic enforcement
- `00-north-star/SOURCES_OF_TRUTH.md` — Hot/Cold SoT boundaries and export policy
- `00-north-star/QUALITY_BARS.md` — Presentation and Accessibility bar definitions
- `01-roles/charters/book_binder.md` — BB export authority
- `01-roles/charters/player_narrator.md` — PN dry-run authority

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` §3 — Loop names (Binding Run, Narration Dry-Run)
- `02-dictionary/artifacts/view_log.md` — View Log structure
- `02-dictionary/artifacts/pn_playtest_notes.md` — PN Playtest Notes structure

### Layer 3 Schemas

- `03-schemas/view_log.schema.json` — View Log validation schema
- `03-schemas/pn_playtest_notes.schema.json` — PN Playtest Notes validation schema
- `03-schemas/gatecheck_report.schema.json` — Gatecheck Report schema (referenced)

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/LIFECYCLES/tu.md` — TU lifecycle (companion spec)
- `04-protocol/LIFECYCLES/gate.md` — Gate lifecycle (companion spec)

---

## 10. Examples

### 10.1 Example: View Bind (Snapshot Selected → Export Binding)

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:view-bind-001",
  "time": "2025-10-30T09:00:00Z",
  "sender": { "role": "BB", "agent": "bot:bb-v1.3" },
  "receiver": { "role": "SR" },
  "intent": "view.bind",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-BB01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Binding Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "none"
  },
  "payload": {
    "type": "view_log",
    "$schema": "../03-schemas/view_log.schema.json",
    "data": {
      "title": "Manuscript Export v2025-10-30",
      "bound": "2025-10-30",
      "binder": "BB Bot v1.3",
      "tu": "TU-2025-10-30-BB01",
      "cold_snapshot": "Cold @ 2025-10-30",
      "targets": ["PDF", "HTML", "EPUB"]
    }
  },
  "refs": ["TU-2025-10-30-BB01"],
  "correlation_id": "corr-binding-run-2025-10-30"
}
```

### 10.2 Example: View Bound (Export Binding Complete)

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:view-bound-001",
  "time": "2025-10-30T10:00:00Z",
  "sender": { "role": "BB", "agent": "bot:bb-v1.3" },
  "receiver": { "role": "PN" },
  "intent": "view.bound",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-BB01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Binding Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "none"
  },
  "payload": {
    "type": "view_log",
    "$schema": "../03-schemas/view_log.schema.json",
    "data": {
      "title": "Manuscript Export v2025-10-30",
      "bound": "2025-10-30",
      "binder": "BB Bot v1.3",
      "tu": "TU-2025-10-30-BB01",
      "cold_snapshot": "Cold @ 2025-10-30",
      "targets": ["PDF", "HTML", "EPUB"],
      "options_and_coverage": "Art: plans-only; Audio: none; Locales: en-US 100%; Research: uncorroborated:low; Accessibility: alt text 95%",
      "dormancy": ["deferred:art", "deferred:audio"],
      "anchor_map": "Manuscript: 120 sections; Codex: 45 entries; Crosslinks: 89; Locales: en-US; Orphans: 0; Collisions: 0",
      "presentation_status": "green",
      "accessibility_status": "yellow",
      "accessibility_notes": "5% images missing alt text; scheduled for Art Touch-up TU",
      "gatekeeper": "GK Bot v2.1",
      "gatecheck_id": "GK-2025-10-30-01",
      "export_artifacts": [
        {
          "kind": "PDF",
          "path": "exports/cold-2025-10-30/manuscript.pdf",
          "hash": "sha256:abc123def456"
        },
        {
          "kind": "HTML",
          "path": "exports/cold-2025-10-30/index.html",
          "hash": "sha256:def456ghi789"
        },
        {
          "kind": "EPUB",
          "path": "exports/cold-2025-10-30/manuscript.epub",
          "hash": "sha256:ghi789jkl012"
        }
      ]
    }
  },
  "refs": ["TU-2025-10-30-BB01", "GK-2025-10-30-01"],
  "correlation_id": "corr-binding-run-2025-10-30",
  "reply_to": "urn:uuid:view-bind-001"
}
```

### 10.3 Example: PN Dry-Run Feedback

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:pn-feedback-001",
  "time": "2025-10-30T12:00:00Z",
  "sender": { "role": "PN", "agent": "bot:pn-v3.2" },
  "receiver": { "role": "SR" },
  "intent": "view.feedback",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-PN01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Narration Dry-Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "none"
  },
  "payload": {
    "type": "pn_playtest_notes",
    "$schema": "../03-schemas/pn_playtest_notes.schema.json",
    "data": {
      "title": "Manuscript Export v2025-10-30",
      "run": "2025-10-30 11:00",
      "pn": "PN Bot v3.2",
      "tu": "TU-2025-10-30-PN01",
      "snapshot": "Cold @ 2025-10-30",
      "targets": ["PDF", "HTML", "EPUB"],
      "mode": "dry-run (no improv)",
      "log": [
        {
          "when": "12:34",
          "location": "chapter_2/section_17.md#L34",
          "tags": ["choice-ambiguity"],
          "severity": "med",
          "snippet": "Choice text: 'Take the left path or right path' — both unclear",
          "smallest_viable_fix": "Add context: 'left path (toward the docks)' and 'right path (toward the markets)'",
          "owner": "SS"
        },
        {
          "when": "14:56",
          "location": "chapter_3/hub_18.md",
          "tags": ["tone-wobble"],
          "severity": "low",
          "snippet": "Hub 18 uses informal register; others formal",
          "smallest_viable_fix": "Adjust hub_18.md register to match formal tone",
          "owner": "ST"
        }
      ]
    }
  },
  "refs": ["TU-2025-10-30-PN01", "TU-2025-10-30-BB01"],
  "correlation_id": "corr-narration-dry-run-2025-10-30",
  "reply_to": "urn:uuid:view-bound-001"
}
```

### 10.4 Example: View Publish

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:view-publish-001",
  "time": "2025-10-30T13:00:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "view.publish",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-BB01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Binding Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "none"
  },
  "payload": {
    "type": "view_log",
    "$schema": "../03-schemas/view_log.schema.json",
    "data": {
      "title": "Manuscript Export v2025-10-30",
      "bound": "2025-10-30",
      "binder": "BB Bot v1.3",
      "tu": "TU-2025-10-30-BB01",
      "cold_snapshot": "Cold @ 2025-10-30",
      "targets": ["PDF", "HTML", "EPUB"],
      "options_and_coverage": "Art: plans-only; Audio: none; Locales: en-US 100%; Research: uncorroborated:low; Accessibility: alt text 95%",
      "dormancy": ["deferred:art", "deferred:audio"],
      "anchor_map": "Manuscript: 120 sections; Codex: 45 entries; Crosslinks: 89; Locales: en-US; Orphans: 0; Collisions: 0",
      "presentation_status": "green",
      "accessibility_status": "yellow",
      "accessibility_notes": "5% images missing alt text; scheduled for Art Touch-up TU",
      "gatekeeper": "GK Bot v2.1",
      "gatecheck_id": "GK-2025-10-30-01",
      "export_artifacts": [
        {
          "kind": "PDF",
          "path": "exports/cold-2025-10-30/manuscript.pdf",
          "hash": "sha256:abc123def456"
        },
        {
          "kind": "HTML",
          "path": "exports/cold-2025-10-30/index.html",
          "hash": "sha256:def456ghi789"
        },
        {
          "kind": "EPUB",
          "path": "exports/cold-2025-10-30/manuscript.epub",
          "hash": "sha256:ghi789jkl012"
        }
      ]
    }
  },
  "refs": ["TU-2025-10-30-BB01", "TU-2025-10-30-PN01", "GK-2025-10-30-01"],
  "correlation_id": "corr-binding-run-2025-10-30",
  "reply_to": "urn:uuid:pn-feedback-001"
}
```

### 10.5 Example: PN Boundary Violation Error

**Error Response:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:pn-error-001",
  "time": "2025-10-30T11:00:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-validator" },
  "receiver": { "role": "BB" },
  "intent": "error.business_rule",
  "context": {
    "hot_cold": "hot",
    "loop": "Binding Run"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "error",
    "$schema": null,
    "data": {
      "code": "PN_HOT_BOUNDARY_VIOLATION",
      "message": "PN cannot access Hot material",
      "details": {
        "receiver": "PN",
        "context_hot_cold": "hot",
        "rule": "PN consumes only Cold snapshots",
        "reference": "00-north-star/PN_PRINCIPLES.md §3",
        "remedy": "Change context.hot_cold to 'cold' and include valid snapshot reference"
      }
    }
  },
  "refs": ["TU-2025-10-30-BB01"],
  "correlation_id": "corr-pn-error-2025-10-30",
  "reply_to": "urn:uuid:invalid-pn-msg"
}
```

### 10.6 Example: Snapshot Missing Error

**Error Response:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:snapshot-error-001",
  "time": "2025-10-30T11:30:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-validator" },
  "receiver": { "role": "PN" },
  "intent": "error.validation",
  "context": {
    "hot_cold": "cold",
    "loop": "Narration Dry-Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "none"
  },
  "payload": {
    "type": "error",
    "$schema": null,
    "data": {
      "code": "SNAPSHOT_MISSING",
      "message": "PN message missing required snapshot reference",
      "details": {
        "receiver": "PN",
        "context_snapshot": null,
        "rule": "All PN messages require context.snapshot",
        "reference": "04-protocol/LIFECYCLES/view.md §5.3",
        "remedy": "Include valid Cold snapshot reference in context.snapshot (format: 'Cold @ YYYY-MM-DD')"
      }
    }
  },
  "refs": ["TU-2025-10-30-PN01"],
  "correlation_id": "corr-snapshot-error-2025-10-30",
  "reply_to": "urn:uuid:invalid-snapshot-msg"
}
```

---

## 11. Implementation Checklist

For implementers of View/Export lifecycle systems:

- [ ] Validate incoming messages against `ENVELOPE.md` requirements
- [ ] Verify `payload.data` validates against `view_log.schema.json` or
      `pn_playtest_notes.schema.json`
- [ ] Check `context.snapshot` present and valid (format: `"Cold @ YYYY-MM-DD"`)
- [ ] Verify snapshot exists in Cold SoT before binding
- [ ] Enforce PN boundary rules: `hot_cold: "cold"` and `player_safe: true` for all PN messages
- [ ] Validate snapshot consistency across view lifecycle chain
- [ ] Check export artifacts include hash for reproducibility
- [ ] Generate appropriate error messages for violations (snapshot missing, PN boundary violations)
- [ ] Record view log with trace linkage
- [ ] Emit state change events for downstream systems (Showrunner, PN)

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group
