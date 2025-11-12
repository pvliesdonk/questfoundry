# TU Lifecycle — State Transitions and Protocol Rules

> **Status:** Normative — this document defines the canonical lifecycle and state transitions for
> Trace Units (TUs) in Layer 4 protocol.

---

## 1. Overview

This specification defines the **Trace Unit (TU) lifecycle**: the allowed state transitions, who can
trigger them, required message intents and payloads, and error conditions.

### Purpose

Trace Units track all meaningful changes targeting Cold SoT (Source of Truth). The lifecycle
ensures:

- **Traceability** — every Cold merge has documented rationale and lineage
- **Quality gates** — work passes quality bars before Cold merge
- **Clear ownership** — roles know who can transition states
- **Reproducibility** — Cold snapshots enable player-safe exports

### Design Principles

- **Explicit state machine** — no implicit transitions
- **Role-based authorization** — only specified roles can trigger transitions
- **Schema validation** — payloads must validate against Layer 3 schemas
- **Cold-bound requirement** — any artifact for Cold MUST have a TU

---

## 2. State Machine

### 2.1 States

The TU lifecycle has **6 states** (see `00-north-star/TRACEABILITY.md`):

```
hot-proposed → stabilizing → gatecheck → cold-merged
             ↘ deferred (parked for later)
             ↘ rejected (won't do, with reason)
```

**States:**

- `hot-proposed` — Initial state when TU is created in Hot
- `stabilizing` — Work in progress, artifacts being developed
- `gatecheck` — Ready for quality bar validation by Gatekeeper
- `cold-merged` — Successfully merged to Cold snapshot (terminal)
- `deferred` — Parked for future consideration with reason (can be reactivated)
- `rejected` — Won't pursue with reason (terminal)

**Terminal states:** `cold-merged`, `rejected` — once reached, TU cannot transition further

**Key Rule:** Any artifact destined for Cold MUST have an associated TU. No TU = No Cold merge.

---

## 3. State Transitions

### 3.1 Transition Matrix

| From State     | To State      | Allowed Sender  | Intent           | Required Payload   | Notes                               |
| -------------- | ------------- | --------------- | ---------------- | ------------------ | ----------------------------------- |
| `hot-proposed` | `stabilizing` | SR or Owner (A) | `tu.start`       | tu_brief (partial) | Work begins, roles activated        |
| `hot-proposed` | `deferred`    | SR              | `tu.defer`       | tu_brief (partial) | Requires deferral reason + revisit  |
| `hot-proposed` | `rejected`    | SR              | `tu.reject`      | tu_brief (partial) | Requires rejection reason           |
| `stabilizing`  | `gatecheck`   | Owner (A)       | `tu.submit_gate` | tu_brief (full)    | Deliverables complete, ready for GK |
| `stabilizing`  | `deferred`    | SR or Owner     | `tu.defer`       | tu_brief (partial) | If blocked during work              |
| `gatecheck`    | `cold-merged` | SR              | `tu.merge`       | tu_brief (full)    | GK passed, merged to Cold snapshot  |
| `gatecheck`    | `stabilizing` | GK or Owner     | `tu.rework`      | tu_brief (partial) | GK failed, remediation needed       |
| `deferred`     | `stabilizing` | SR              | `tu.reactivate`  | tu_brief (partial) | Reactivating deferred TU            |
| `deferred`     | `rejected`    | SR              | `tu.reject`      | tu_brief (partial) | Abandoning deferred TU              |

**Role Abbreviations:**

- **SR** — Showrunner (accountable for TU decisions and Cold merges)
- **Owner (A)** — The accountable role specified in `owner_a` (usually SR)
- **GK** — Gatekeeper (quality bar enforcement)

---

## 4. Transition Details

### 4.1 `hot-proposed` → `stabilizing`

**Preconditions:**

- TU status is `hot-proposed`
- Roles have been identified (awake/dormant)
- Inputs/prerequisites available

**Sender:** SR (Showrunner) or Owner (A role in `owner_a`)

**Intent:** `tu.start`

**Payload Schema:** `tu_brief.schema.json` (partial update)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "opened": "YYYY-MM-DD",
  "owner_a": "SR",
  "responsible_r": ["<roles>"],
  "loop": "<loop name>",
  "awake": ["<roles>"],
  "dormant": ["<roles>"],
  "timebox": "45 min" | "90 min"
}
```

**Effects:**

- TU status changes to `stabilizing`
- Work begins, roles activated per awake list
- Clock starts on timebox

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `hot-proposed`
- `NOT_AUTHORIZED` — if sender is not SR or owner_a
- `VALIDATION_FAILED` — if required fields missing (awake/dormant, timebox)

---

### 4.2 `hot-proposed` → `deferred`

**Preconditions:**

- TU status is `hot-proposed`
- Decision made to postpone (with reason)

**Sender:** SR (Showrunner)

**Intent:** `tu.defer`

**Payload Schema:** `tu_brief.schema.json` (partial update)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "deferral_tags": ["deferred:art", "deferred:audio", etc.],
  "linkage": "<reason for deferral and revisit condition>"
}
```

**Effects:**

- TU status changes to `deferred`
- Deferral reason and revisit condition recorded
- Work suspended

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `hot-proposed` or `stabilizing`
- `NOT_AUTHORIZED` — if sender is not SR
- `VALIDATION_FAILED` — if deferral_tags or linkage missing

---

### 4.3 `hot-proposed` → `rejected`

**Preconditions:**

- TU status is `hot-proposed`
- Decision made not to pursue (with reason)

**Sender:** SR (Showrunner)

**Intent:** `tu.reject`

**Payload Schema:** `tu_brief.schema.json` (partial update)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "linkage": "<rejection reason and audit note>"
}
```

**Effects:**

- TU status changes to `rejected` (terminal)
- TU removed from active work queue
- Rejection reason recorded for audit

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `hot-proposed` or `deferred`
- `NOT_AUTHORIZED` — if sender is not SR
- `VALIDATION_FAILED` — if rejection reason in linkage missing

---

### 4.4 `stabilizing` → `gatecheck`

**Preconditions:**

- TU status is `stabilizing`
- All deliverables completed
- Self-checks passed (owner believes bars are green)

**Sender:** Owner (A role in `owner_a`)

**Intent:** `tu.submit_gate`

**Payload Schema:** `tu_brief.schema.json` (full)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "deliverables": ["<artifact paths or descriptions>"],
  "bars_green": ["Integrity", "Reachability", etc.],
  "gatecheck": "<gatecheck plan with pass/fail criteria>",
  "linkage": "<hooks filed, snapshot impact, trace notes>"
}
```

**Effects:**

- TU status changes to `gatecheck`
- Gatekeeper triggered for quality bar validation
- No further artifact changes until GK decision

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `stabilizing`
- `NOT_AUTHORIZED` — if sender is not owner_a
- `VALIDATION_FAILED` — if deliverables, bars_green, or gatecheck plan missing
- `BUSINESS_RULE_VIOLATION` — if pre_gate_risks not addressed

---

### 4.5 `gatecheck` → `cold-merged`

**Preconditions:**

- TU status is `gatecheck`
- Gatekeeper validation passed (all bars green)
- No blocking issues identified

**Sender:** SR (Showrunner)

**Intent:** `tu.merge`

**Payload Schema:** `tu_brief.schema.json` (full)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "snapshot_context": "Cold @ YYYY-MM-DD",
  "linkage": "<final merge notes, snapshot reference, artifact locations>"
}
```

**Effects:**

- TU status changes to `cold-merged` (terminal)
- Artifacts merged to Cold snapshot
- Snapshot ID recorded for reproducibility
- Lifecycle complete

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `gatecheck`
- `NOT_AUTHORIZED` — if sender is not SR
- `VALIDATION_FAILED` — if snapshot_context missing
- `BUSINESS_RULE_VIOLATION` — if gatecheck not passed or bars not green

---

### 4.6 `gatecheck` → `stabilizing` (rework)

**Preconditions:**

- TU status is `gatecheck`
- Gatekeeper validation failed (one or more bars red/yellow)
- Remediation required

**Sender:** GK (Gatekeeper) or Owner (A)

**Intent:** `tu.rework`

**Payload Schema:** `tu_brief.schema.json` (partial update)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "pre_gate_risks": ["<remediation items from gatecheck report>"],
  "gatecheck": "<updated gatecheck plan>"
}
```

**Effects:**

- TU status reverts to `stabilizing`
- Owner must address failed bars
- Pre-gate risks updated with remediation tasks

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `gatecheck`
- `NOT_AUTHORIZED` — if sender is not GK or owner_a
- `VALIDATION_FAILED` — if remediation tasks not specified

---

### 4.7 `deferred` → `stabilizing` (reactivate)

**Preconditions:**

- TU status is `deferred`
- Decision made to reactivate
- Revisit condition met

**Sender:** SR (Showrunner)

**Intent:** `tu.reactivate`

**Payload Schema:** `tu_brief.schema.json` (partial update)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "opened": "YYYY-MM-DD",
  "linkage": "<reactivation reason and updated context>"
}
```

**Effects:**

- TU status changes to `stabilizing`
- Work resumes with updated context
- Deferral tags cleared or updated

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `deferred`
- `NOT_AUTHORIZED` — if sender is not SR

---

### 4.8 `deferred` → `rejected` (abandon)

**Preconditions:**

- TU status is `deferred`
- Decision made not to pursue

**Sender:** SR (Showrunner)

**Intent:** `tu.reject`

**Payload Schema:** `tu_brief.schema.json` (partial update)

**Required Fields:** (same as 4.3)

**Effects:**

- TU status changes to `rejected` (terminal)
- Rejection reason recorded

**Error Cases:** (same as 4.3)

---

## 5. Envelope Context Requirements

All TU lifecycle messages MUST include proper envelope context per `ENVELOPE.md`:

### 5.1 Hot/Cold Boundaries

- **Hot context** (`hot_cold: "hot"`):
  - Used for all TU lifecycle transitions except final merge
  - TUs coordinate Hot work, not player-facing
  - `safety.player_safe: false`, `safety.spoilers: "allowed"`

- **Cold context** (`hot_cold: "cold"`):
  - Used only for `cold-merged` state and subsequent references
  - MUST include `snapshot` reference (format: `"Cold @ YYYY-MM-DD"`)
  - Enables reproducibility for Binder/PN

### 5.2 TU Linkage (Self-Reference)

All TU lifecycle messages MUST include:

- `context.tu` — the TU ID being transitioned (self-reference)
- `context.snapshot` — the Cold snapshot context (updated on merge)
- `refs` — array of upstream hooks, TUs, ADRs, or artifacts

### 5.3 Loop Context

All TU messages MUST include:

- `context.loop` — the loop type from `tu_brief.loop` field

---

## 6. Validation Rules

### 6.1 State Transition Validation

**Rule:** Transitions MUST follow allowed paths from the state machine

- **Check:** Current TU status matches "From State" in transition matrix
- **Error:** `INVALID_STATE_TRANSITION` if not allowed

**Example error:**

```json
{
  "code": "INVALID_STATE_TRANSITION",
  "message": "Cannot transition from 'cold-merged' to 'stabilizing'",
  "details": {
    "tu_id": "TU-2025-10-28-SR01",
    "current_state": "cold-merged",
    "requested_state": "stabilizing",
    "reason": "Terminal state cannot be reopened. Create new TU if changes needed."
  }
}
```

### 6.2 Role Authorization Validation

**Rule:** Only specified roles can trigger transitions

- **Check:** `sender.role` matches "Allowed Sender" in transition matrix
- **Error:** `NOT_AUTHORIZED` if role mismatch

**Example error:**

```json
{
  "code": "NOT_AUTHORIZED",
  "message": "Role 'SS' not authorized to merge TUs to Cold",
  "details": {
    "sender_role": "SS",
    "required_role": "SR",
    "intent": "tu.merge",
    "tu_id": "TU-2025-10-28-SR01"
  }
}
```

### 6.3 Required Field Validation

**Rule:** Transitions requiring specific fields must validate

- **Check:** Required fields present and valid per schema
- **Error:** `VALIDATION_FAILED` if fields missing or invalid

**Example error:**

```json
{
  "code": "VALIDATION_FAILED",
  "message": "Gatecheck submission missing required fields",
  "details": {
    "tu_id": "TU-2025-10-28-LW01",
    "missing_fields": ["deliverables", "bars_green", "gatecheck"],
    "transition": "stabilizing → gatecheck"
  }
}
```

### 6.4 Terminal State Protection

**Rule:** Terminal states (`cold-merged`, `rejected`) cannot transition further

- **Check:** Current TU status is not terminal
- **Error:** `INVALID_STATE_TRANSITION` with terminal state reason

**Example error:**

```json
{
  "code": "INVALID_STATE_TRANSITION",
  "message": "Terminal state cannot be modified",
  "details": {
    "tu_id": "TU-2025-10-28-SR01",
    "current_state": "cold-merged",
    "reason": "Merged TUs are immutable. Create new TU if changes needed."
  }
}
```

### 6.5 Cold-Bound Requirement Validation

**Rule:** Any artifact targeting Cold MUST have an associated TU

- **Check:** Merge requests include valid TU ID
- **Error:** `BUSINESS_RULE_VIOLATION` if TU missing

**Example error:**

```json
{
  "code": "BUSINESS_RULE_VIOLATION",
  "message": "Cold merge attempted without TU",
  "details": {
    "rule": "COLD_BOUND_REQUIREMENT",
    "violation": "Artifact merge request missing TU linkage",
    "reference": "00-north-star/TRACEABILITY.md",
    "remedy": "Create TU before attempting Cold merge"
  }
}
```

---

## 7. Quality Bar Integration

### 7.1 Gatecheck Enforcement

TU transitions to `cold-merged` MUST verify:

- All bars in `bars_green` are actually green (validated by Gatekeeper)
- Gatecheck report shows PASS status
- No blocking pre-gate risks unresolved

### 7.2 Pre-Gate vs. Gatecheck

- **Pre-gate** (`pre_gate` field): Early checkpoint before heavy investment
  - Gatekeeper samples representative slice
  - Identifies likely failure points
  - Owner addresses before submitting full gatecheck

- **Gatecheck** (`gatecheck` field): Formal go/no-go decision
  - Gatekeeper validates all bars listed in `bars_green`
  - Produces formal gatecheck report (pass/fail per bar)
  - Blocks merge if any bar fails

### 7.3 Quality Bar Categories

All TU briefs MUST list bars being pressed:

- `press` — bars this TU commits to flipping green
- `monitor` — bars to watch without green commitment

See `02-dictionary/taxonomies.md` §5 for the 8 Quality Bars:

1. Integrity
2. Reachability
3. Nonlinearity
4. Gateways
5. Style
6. Determinism
7. Presentation
8. Accessibility

---

## 8. Dormancy and Deferral Integration

### 8.1 Role Wake/Sleep

TU briefs MUST explicitly list:

- `awake` — roles participating in this TU
- `dormant` — roles not participating (with deferral tags if optional roles)

### 8.2 Deferral Tags

If optional roles (Art, Audio, Translation, Research) are dormant:

- MUST include `deferral_tags` array
- Tags: `deferred:art`, `deferred:audio`, `deferred:translation`, `deferred:research`
- See `02-dictionary/taxonomies.md` §7 for deferral types

### 8.3 Wake Rubric

When deciding to wake optional roles, apply wake rubric:

- Score: Player benefit (0-2) + Bar pressure (0-2) + Scope fit (0-2) + Reuse leverage (0-2)
- Wake if total ≥ 4 OR hard trigger (Gatekeeper red, export requirement, safety risk)

---

## 9. Cross-References

### Layer 0/1 Policy

- `00-north-star/TRACEABILITY.md` — TU requirements and Cold-bound rule
- `00-north-star/QUALITY_BARS.md` — 8 quality bar definitions
- `01-roles/charters/showrunner.md` — SR merge authority
- `01-roles/charters/gatekeeper.md` — GK gatecheck authority

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` §3 — TU Types & Loop Alignment
- `02-dictionary/taxonomies.md` §5 — Quality Bar Categories
- `02-dictionary/taxonomies.md` §7 — Deferral Types
- `02-dictionary/artifacts/tu_brief.md` — TU Brief structure

### Layer 3 Schemas

- `03-schemas/tu_brief.schema.json` — Payload validation schema
- `03-schemas/gatecheck_report.schema.json` — Gatecheck result schema

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/INTENTS.md` — Intent catalog (future)
- `04-protocol/LIFECYCLES/hooks.md` — Hook lifecycle (companion spec)

---

## 10. Examples

### 10.1 Example: Start TU

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:tu-start-001",
  "time": "2025-10-30T09:00:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "tu.start",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Lore Deepening"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-SR01",
      "opened": "2025-10-30",
      "owner_a": "SR",
      "responsible_r": ["LW"],
      "loop": "Lore Deepening",
      "slice": "Canon entry for wormhole toll mechanism",
      "snapshot_context": "Cold @ 2025-10-28",
      "awake": ["SR", "GK", "LW"],
      "dormant": ["AD", "IL", "AuD", "AuP", "TR"],
      "deferral_tags": ["deferred:art", "deferred:audio", "deferred:translation"],
      "press": ["Integrity", "Nonlinearity"],
      "monitor": ["Gateways"],
      "pre_gate_risks": ["Canon contradicts existing faction lore"],
      "inputs": ["Hook HK-20251030-01", "Faction brief v1"],
      "deliverables": ["Canon pack for toll mechanism"],
      "bars_green": ["Integrity", "Nonlinearity"],
      "merge_view": "Merge canon to Cold, no view rebuild needed",
      "timebox": "60 min",
      "gatecheck": "GK validates Integrity and Nonlinearity bars",
      "linkage": "From Hook HK-20251030-01, impacts sections 8, 17, 33"
    }
  },
  "refs": ["HK-20251030-01"],
  "correlation_id": "corr-lore-deepening-2025-10-30"
}
```

### 10.2 Example: Submit for Gatecheck

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:tu-submit-gate-001",
  "time": "2025-10-30T10:45:00Z",
  "sender": { "role": "LW", "agent": "bot:lw-v1.5" },
  "receiver": { "role": "GK" },
  "intent": "tu.submit_gate",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Lore Deepening"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-SR01",
      "deliverables": [
        "Canon pack: manuscript/canon/toll_mechanism.md",
        "Updated faction relationships in canon/factions.md"
      ],
      "bars_green": ["Integrity", "Nonlinearity"],
      "gatecheck": "GK validates: Integrity (all IDs resolve, no contradictions), Nonlinearity (toll creates meaningful choice pressure)",
      "linkage": "Canon complete, ready for Cold merge. Impacts sections 8, 17, 33."
    }
  },
  "refs": ["TU-2025-10-30-SR01", "HK-20251030-01"],
  "correlation_id": "corr-lore-deepening-2025-10-30",
  "reply_to": "urn:uuid:tu-start-001"
}
```

### 10.3 Example: Merge to Cold

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:tu-merge-001",
  "time": "2025-10-30T11:30:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "tu.merge",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Lore Deepening"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-SR01",
      "snapshot_context": "Cold @ 2025-10-30",
      "linkage": "Merged: canon/toll_mechanism.md, canon/factions.md (updated). GK report: all bars green. Closes Hook HK-20251030-01."
    }
  },
  "refs": ["TU-2025-10-30-SR01", "HK-20251030-01", "GK-20251030-03"],
  "correlation_id": "corr-lore-deepening-2025-10-30",
  "reply_to": "urn:uuid:tu-submit-gate-001"
}
```

### 10.4 Example: Gatecheck Failure (Rework)

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:tu-rework-001",
  "time": "2025-10-30T11:00:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-validator" },
  "receiver": { "role": "LW" },
  "intent": "tu.rework",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Gatecheck"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-SR01",
      "pre_gate_risks": [
        "Integrity FAIL: Canon entry references 'Faction-X7' but no such faction exists in canon/factions.md",
        "Nonlinearity PASS: Toll creates meaningful choice pressure"
      ],
      "gatecheck": "Remediate: Add Faction-X7 to canon/factions.md OR update toll_mechanism.md to use existing faction. Resubmit for gatecheck."
    }
  },
  "refs": ["TU-2025-10-30-SR01", "GK-20251030-02"],
  "correlation_id": "corr-lore-deepening-2025-10-30",
  "reply_to": "urn:uuid:tu-submit-gate-001"
}
```

### 10.5 Example: Invalid Transition Error

**Error Response:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:tu-error-001",
  "time": "2025-10-30T12:00:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-validator" },
  "receiver": { "role": "SS" },
  "intent": "error.business_rule",
  "context": {
    "hot_cold": "hot",
    "loop": "Lore Deepening"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "error",
    "$schema": null,
    "data": {
      "code": "INVALID_STATE_TRANSITION",
      "message": "Cannot transition from 'cold-merged' to 'stabilizing'",
      "details": {
        "tu_id": "TU-2025-10-25-SR03",
        "current_state": "cold-merged",
        "requested_state": "stabilizing",
        "reason": "Terminal state cannot be reopened. Create new TU if changes needed.",
        "reference": "04-protocol/LIFECYCLES/tu.md §6.4"
      }
    }
  },
  "refs": ["TU-2025-10-25-SR03"],
  "correlation_id": "corr-tu-error-2025-10-30",
  "reply_to": "urn:uuid:invalid-reopen-attempt"
}
```

---

## 11. Implementation Checklist

For implementers of TU lifecycle systems:

- [ ] Validate incoming messages against `ENVELOPE.md` requirements
- [ ] Verify `payload.data` validates against `tu_brief.schema.json`
- [ ] Check current TU status against transition matrix
- [ ] Verify sender role matches allowed sender for transition
- [ ] Validate required fields present for specific transition
- [ ] Enforce terminal state protection (no transitions from `cold-merged`/`rejected`)
- [ ] Enforce Cold-bound requirement (no Cold merge without TU)
- [ ] Validate quality bar lists (`press`, `monitor`, `bars_green`)
- [ ] Verify gatecheck passed before allowing merge
- [ ] Record Cold snapshot ID on merge atomically
- [ ] Update TU status and audit trail with TU linkage
- [ ] Generate appropriate error messages for violations
- [ ] Emit state change events for downstream systems (Binder, PN)

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group
