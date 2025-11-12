# Hook Lifecycle — State Transitions and Protocol Rules

> **Status:** Normative — this document defines the canonical lifecycle and state transitions for
> Hook Cards in Layer 4 protocol.

---

## 1. Overview

This specification defines the **Hook Card lifecycle**: the allowed state transitions, who can
trigger them, required message intents and payloads, and error conditions.

### Purpose

Hook Cards track small, traceable follow-up ideas from proposal through resolution and canonization.
The lifecycle ensures:

- **Clear ownership** — roles know who can transition states
- **Traceability** — every transition references TU context
- **Quality gates** — transitions enforce quality bar checks
- **Safety** — terminal states prevent accidental reopening

### Design Principles

- **Explicit state machine** — no implicit transitions
- **Role-based authorization** — only specified roles can trigger transitions
- **Schema validation** — payloads must validate against Layer 3 schemas
- **Audit trail** — all transitions logged with TU linkage

---

## 2. State Machine

### 2.1 States

The Hook Card lifecycle has **7 states** (see `02-dictionary/taxonomies.md` §2):

```
proposed → accepted → in-progress → resolved → canonized
         ↘ deferred (parked for later)
         ↘ rejected (won't do, with reason)
```

**States:**

- `proposed` — Initial state when hook card is created
- `accepted` — Showrunner has triaged and approved for work
- `in-progress` — Work actively underway, assigned to owner
- `resolved` — Work completed, awaiting gatecheck/merge
- `canonized` — Merged to Cold, hook card links to Cold location (terminal)
- `deferred` — Parked for future consideration with reason (can be reactivated)
- `rejected` — Won't pursue with reason (terminal)

**Terminal states:** `canonized`, `rejected` — once reached, hook cannot transition further without
creating a new hook

---

## 3. State Transitions

### 3.1 Transition Matrix

| From State    | To State      | Allowed Sender | Intent               | Required Payload    | Notes                                           |
| ------------- | ------------- | -------------- | -------------------- | ------------------- | ----------------------------------------------- |
| `proposed`    | `accepted`    | SR             | `hook.update_status` | hook_card (partial) | SR triages; `status: "accepted"` in payload     |
| `proposed`    | `deferred`    | SR             | `hook.update_status` | hook_card (partial) | Requires deferral reason; `status: "deferred"`  |
| `proposed`    | `rejected`    | SR             | `hook.update_status` | hook_card (partial) | Requires rejection reason; `status: "rejected"` |
| `accepted`    | `in-progress` | Owner (R role) | `hook.update_status` | hook_card (partial) | Owner begins work; `status: "in-progress"`      |
| `accepted`    | `deferred`    | SR or Owner    | `hook.update_status` | hook_card (partial) | If blocked; `status: "deferred"`                |
| `in-progress` | `resolved`    | Owner (R role) | `hook.update_status` | hook_card (full)    | Work complete; `status: "resolved"`             |
| `in-progress` | `deferred`    | SR or Owner    | `hook.update_status` | hook_card (partial) | If blocked; `status: "deferred"`                |
| `resolved`    | `canonized`   | SR             | `hook.update_status` | hook_card (full)    | After Cold merge; `status: "canonized"`         |
| `resolved`    | `in-progress` | Owner (R role) | `hook.update_status` | hook_card (partial) | If rework needed; `status: "in-progress"`       |
| `deferred`    | `accepted`    | SR             | `hook.update_status` | hook_card (partial) | Reactivating; `status: "accepted"`              |
| `deferred`    | `rejected`    | SR             | `hook.update_status` | hook_card (partial) | Abandoning; `status: "rejected"`                |

**Role Abbreviations:**

- **SR** — Showrunner (accountable for lifecycle decisions)
- **Owner (R role)** — The responsible role specified in `proposed_next_step.owner_r`

---

## 4. Transition Details

### 4.1 `proposed` → `accepted`

**Preconditions:**

- Hook status is `proposed`
- Hook has been triaged by Showrunner
- Owner role and deliverables are clear

**Sender:** SR (Showrunner)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (partial update)

**Payload Field:** `header.status: "accepted"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "accepted",
    "edited": "YYYY-MM-DD"
  },
  "proposed_next_step": {
    "loop": "<loop name>",
    "owner_r": "<role>",
    "accountable_a": "SR"
  }
}
```

**Effects:**

- Hook status changes to `accepted`
- Owner role assigned (if not already set)
- Hook enters work backlog

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `proposed`
- `NOT_AUTHORIZED` — if sender is not SR
- `VALIDATION_FAILED` — if owner_r or loop not specified

---

### 4.2 `proposed` → `deferred`

**Preconditions:**

- Hook status is `proposed`
- Decision made to postpone (with reason)

**Sender:** SR (Showrunner)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (partial update)

**Payload Field:** `header.status: "deferred"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "deferred",
    "edited": "YYYY-MM-DD"
  },
  "dormancy_deferrals": {
    "deferral_tags": ["deferred:art", "deferred:audio", etc.],
    "fallback": "<what happens without deferred work>",
    "revisit": "<loop name or milestone/date to revisit>"
  }
}
```

**Effects:**

- Hook status changes to `deferred`
- Deferral reason and revisit condition recorded

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `proposed`, `accepted`, or `in-progress`
- `NOT_AUTHORIZED` — if sender is not SR or Owner
- `VALIDATION_FAILED` — if `dormancy_deferrals.fallback` or `revisit` missing

---

### 4.3 `proposed` → `rejected`

**Preconditions:**

- Hook status is `proposed`
- Decision made not to pursue (with reason)

**Sender:** SR (Showrunner)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (partial update)

**Payload Field:** `header.status: "rejected"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "rejected",
    "edited": "YYYY-MM-DD"
  },
  "resolution": {
    "decision": "<one sentence rejection reason>",
    "resolved_date": "YYYY-MM-DD",
    "resolved_by": "SR"
  }
}
```

**Effects:**

- Hook status changes to `rejected` (terminal)
- Hook removed from active backlog
- Rejection reason recorded for audit

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `proposed` or `deferred`
- `NOT_AUTHORIZED` — if sender is not SR
- `VALIDATION_FAILED` — if `resolution.decision` missing

---

### 4.4 `accepted` → `in-progress`

**Preconditions:**

- Hook status is `accepted`
- Owner role ready to begin work

**Sender:** Owner (R role specified in `proposed_next_step.owner_r`)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (partial update)

**Payload Field:** `header.status: "in-progress"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "in-progress",
    "edited": "YYYY-MM-DD"
  }
}
```

**Effects:**

- Hook status changes to `in-progress`
- Owner actively working on deliverables

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `accepted`
- `NOT_AUTHORIZED` — if sender is not the assigned owner role
- `VALIDATION_FAILED` — if hook has blocking issues unresolved

---

### 4.5 `in-progress` → `resolved`

**Preconditions:**

- Hook status is `in-progress`
- Work completed, deliverables ready
- Acceptance criteria met

**Sender:** Owner (R role)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (full)

**Payload Field:** `header.status: "resolved"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "resolved",
    "edited": "YYYY-MM-DD"
  },
  "resolution": {
    "decision": "<one sentence resolution summary>",
    "work_performed": ["TU-YYYY-MM-DD-RRnn", ...],
    "view_impact": "none" | "rebind needed",
    "gatekeeper_result": "<brief bars green summary>",
    "resolved_date": "YYYY-MM-DD",
    "resolved_by": "<role>"
  }
}
```

**Effects:**

- Hook status changes to `resolved`
- Ready for gatecheck and potential Cold merge
- TU IDs recorded for traceability

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `in-progress`
- `NOT_AUTHORIZED` — if sender is not the assigned owner role
- `VALIDATION_FAILED` — if acceptance criteria not met or resolution fields incomplete

---

### 4.6 `resolved` → `canonized`

**Preconditions:**

- Hook status is `resolved`
- Gatecheck passed
- Successfully merged to Cold

**Sender:** SR (Showrunner)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (full)

**Payload Field:** `header.status: "canonized"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "canonized",
    "edited": "YYYY-MM-DD",
    "snapshot_context": "Cold @ YYYY-MM-DD"
  },
  "locations_links": {
    "locations": ["<Cold paths where changes landed>", ...],
    "lineage": "<TU IDs>"
  }
}
```

**Effects:**

- Hook status changes to `canonized` (terminal)
- Hook card links to Cold snapshot and locations
- Lifecycle complete

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `resolved`
- `NOT_AUTHORIZED` — if sender is not SR
- `VALIDATION_FAILED` — if snapshot_context or locations missing
- `BUSINESS_RULE_VIOLATION` — if gatecheck not passed

---

### 4.7 `resolved` → `in-progress` (reopen)

**Preconditions:**

- Hook status is `resolved`
- Gatecheck failed or rework needed

**Sender:** Owner (R role)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (partial update)

**Payload Field:** `header.status: "in-progress"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "in-progress",
    "edited": "YYYY-MM-DD"
  }
}
```

**Effects:**

- Hook status reverts to `in-progress`
- Owner must address gatecheck issues

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `resolved`
- `NOT_AUTHORIZED` — if sender is not the assigned owner role or SR

---

### 4.8 `deferred` → `accepted` (reactivate)

**Preconditions:**

- Hook status is `deferred`
- Decision made to reactivate
- Revisit condition met

**Sender:** SR (Showrunner)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (partial update)

**Payload Field:** `header.status: "accepted"`

**Required Fields:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "accepted",
    "edited": "YYYY-MM-DD"
  }
}
```

**Effects:**

- Hook status changes to `accepted`
- Hook re-enters work backlog
- Deferral tags cleared or updated

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if current status is not `deferred`
- `NOT_AUTHORIZED` — if sender is not SR

---

### 4.9 `deferred` → `rejected` (abandon)

**Preconditions:**

- Hook status is `deferred`
- Decision made not to pursue

**Sender:** SR (Showrunner)

**Intent:** `hook.update_status`

**Payload Schema:** `hook_card.schema.json` (partial update)

**Payload Field:** `header.status: "rejected"`

**Required Fields:** (same as 4.3)

**Effects:**

- Hook status changes to `rejected` (terminal)
- Rejection reason recorded

**Error Cases:** (same as 4.3)

---

## 5. Envelope Context Requirements

All hook lifecycle messages MUST include proper envelope context per `ENVELOPE.md`:

### 5.1 Hot/Cold Boundaries

- **Hot context** (`hot_cold: "hot"`):
  - Used for all hook lifecycle transitions
  - Hooks are working artifacts, not player-facing
  - `safety.player_safe: false`, `safety.spoilers: "allowed"`

- **Cold context** (`hot_cold: "cold"`):
  - Only used when referencing canonized hooks in exports
  - Must include `snapshot` reference

### 5.2 TU Linkage

All transitions (except initial `proposed` creation) SHOULD include:

- `context.tu` — the trace unit driving the change
- `refs` — array of related hook/TU IDs for traceability

### 5.3 Loop Context

All hook messages MUST include:

- `context.loop` — the loop context (typically "Hook Harvest" for lifecycle transitions)

---

## 6. Validation Rules

### 6.1 State Transition Validation

**Rule:** Transitions MUST follow allowed paths from the state machine

- **Check:** Current status matches "From State" in transition matrix
- **Error:** `INVALID_STATE_TRANSITION` if not allowed

**Example error:**

```json
{
  "code": "INVALID_STATE_TRANSITION",
  "message": "Cannot transition from 'canonized' to 'in-progress'",
  "details": {
    "current_state": "canonized",
    "requested_state": "in-progress",
    "reason": "Terminal state cannot be reopened"
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
  "message": "Role 'SS' not authorized to accept hooks",
  "details": {
    "sender_role": "SS",
    "required_role": "SR",
    "intent": "hook.update_status"
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
  "message": "Deferral transition missing required fields",
  "details": {
    "missing_fields": ["dormancy_deferrals.fallback", "dormancy_deferrals.revisit"],
    "transition": "proposed → deferred"
  }
}
```

### 6.4 Terminal State Protection

**Rule:** Terminal states (`canonized`, `rejected`) cannot transition further

- **Check:** Current status is not terminal
- **Error:** `INVALID_STATE_TRANSITION` with terminal state reason

**Example error:**

```json
{
  "code": "INVALID_STATE_TRANSITION",
  "message": "Terminal state cannot be modified",
  "details": {
    "current_state": "canonized",
    "reason": "Canonized hooks are immutable. Create new hook if changes needed."
  }
}
```

---

## 7. Quality Bar Integration

### 7.1 Gatecheck Enforcement

Hook transitions to `canonized` MUST verify:

- All bars in `classification.bars_affected` are green
- Gatekeeper has approved via `resolution.gatekeeper_result`

### 7.2 Blocking Hooks

If `classification.blocking = "yes (<reason>)"`:

- Hook cannot be deferred without SR explicit decision
- Work on dependent artifacts should pause until resolved

---

## 8. Cross-References

### Layer 0/1 Policy

- `00-north-star/HOOKS.md` — Hook types and lifecycle overview
- `00-north-star/TRACEABILITY.md` — TU linkage requirements
- `01-roles/charters/showrunner.md` — SR lifecycle authority

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` §2 — Hook Status Lifecycle (canonical states)
- `02-dictionary/artifacts/hook_card.md` — Hook Card structure

### Layer 3 Schemas

- `03-schemas/hook_card.schema.json` — Payload validation schema

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/INTENTS.md` — Intent catalog (future)

---

## 9. Examples

### 9.1 Example: Accept Hook

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:a1b2c3d4-e5f6-4789-a012-3456789abcde",
  "time": "2025-10-30T12:30:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "LW" },
  "intent": "hook.update_status",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": null,
    "loop": "Hook Harvest"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "hook_card",
    "$schema": "../03-schemas/hook_card.schema.json",
    "data": {
      "header": {
        "id": "HK-20251030-01",
        "status": "accepted",
        "edited": "2025-10-30"
      },
      "proposed_next_step": {
        "loop": "Lore Deepening",
        "owner_r": "LW",
        "accountable_a": "SR"
      }
    }
  },
  "refs": ["TU-2025-10-30-SR01"],
  "correlation_id": "corr-hook-harvest-2025-10-30",
  "reply_to": "urn:uuid:original-hook-create-msg-id"
}
```

### 9.2 Example: Resolve Hook

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:f1e2d3c4-b5a6-4987-8765-43210fedcba0",
  "time": "2025-10-30T15:45:00Z",
  "sender": { "role": "LW", "agent": "bot:lw-v1.5" },
  "receiver": { "role": "SR" },
  "intent": "hook.update_status",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-LW01",
    "snapshot": null,
    "loop": "Lore Deepening"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "hook_card",
    "$schema": "../03-schemas/hook_card.schema.json",
    "data": {
      "header": {
        "id": "HK-20251030-01",
        "status": "resolved",
        "edited": "2025-10-30"
      },
      "resolution": {
        "decision": "Canon entry for toll mechanism completed",
        "work_performed": ["TU-2025-10-30-LW01"],
        "view_impact": "none",
        "gatekeeper_result": "All bars green: Integrity, Nonlinearity",
        "resolved_date": "2025-10-30",
        "resolved_by": "LW"
      }
    }
  },
  "refs": ["HK-20251030-01", "TU-2025-10-30-LW01"],
  "correlation_id": "corr-hook-harvest-2025-10-30",
  "reply_to": "urn:uuid:accept-msg-id"
}
```

### 9.3 Example: Invalid Transition Error

**Error Response:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:error-123-456-789",
  "time": "2025-10-30T16:00:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-validator" },
  "receiver": { "role": "SS" },
  "intent": "error.business_rule",
  "context": {
    "hot_cold": "hot",
    "loop": "Hook Harvest"
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
      "message": "Cannot transition from 'canonized' to 'in-progress'",
      "details": {
        "hook_id": "HK-20251025-03",
        "current_state": "canonized",
        "requested_state": "in-progress",
        "reason": "Terminal state cannot be reopened. Create new hook if changes needed."
      }
    }
  },
  "refs": ["HK-20251025-03"],
  "correlation_id": "corr-hook-error-2025-10-30",
  "reply_to": "urn:uuid:invalid-reopen-attempt-msg-id"
}
```

---

## 10. Implementation Checklist

For implementers of Hook lifecycle systems:

- [ ] Validate incoming messages against `ENVELOPE.md` requirements
- [ ] Verify `payload.data` validates against `hook_card.schema.json`
- [ ] Check current hook status against transition matrix
- [ ] Verify sender role matches allowed sender for transition
- [ ] Validate required fields present for specific transition
- [ ] Enforce terminal state protection (no transitions from `canonized`/`rejected`)
- [ ] Record audit trail with TU linkage
- [ ] Generate appropriate error messages for violations
- [ ] Emit state change events for downstream systems
- [ ] Update hook status in storage atomically

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group
