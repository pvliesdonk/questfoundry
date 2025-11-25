# Layer 4 Intents v1.0 — Core Message Catalog

> **Status:** Normative — this document defines the canonical message intents (verbs) for Layer 4
> protocol.

---

## 1. Overview

This specification defines the **message intents** (verbs) that drive QuestFoundry workflows: the
actions that trigger lifecycle state transitions, coordinate role handoffs, and enforce quality
gates.

All intents are validated against `04-protocol/envelope.schema.json`, which serves as the
**normative source** for envelope structure, required fields, and constraints. This document
provides the semantic meaning and usage patterns for each intent.

### Purpose

Intents provide:

1. **Explicit semantics** — every message declares its purpose via `intent` field
2. **Lifecycle coordination** — intents trigger state transitions in Hook, TU, Gate, View lifecycles
3. **Schema linkage** — intents specify which Layer 3 payload schemas are required
4. **Error taxonomy** — standardized error intents enable consistent error handling
5. **Role authorization** — intents document which roles can send/receive
6. **Envelope validation** — intents are constrained by envelope.schema.json

### Design Principles

- **Namespace organization** — intents use dot notation: `<domain>.<verb>[.<subverb>]`
- **Explicit authorization** — each intent specifies allowed sender/receiver roles
- **Schema enforcement** — intents reference Layer 3 schemas for payload validation
- **Envelope constraints** — special intents (ack, error, PN-bound) have envelope schema rules
- **Forward compatibility** — unknown intents should be rejected unless explicitly allowed

### Relationship to Envelope Schema

The `envelope.schema.json` defines several normative constraints that affect intents:

1. **PN Safety Invariant**: When `receiver.role = "PN"`, the envelope MUST enforce:
   - `context.hot_cold = "cold"`
   - `context.snapshot` present
   - `safety.player_safe = true`

2. **Ack Intent**: When `intent = "ack"`, the envelope enforces:
   - `payload.type = "none"`

3. **Error Intent**: When `intent = "error"`, the envelope enforces:
   - `payload.type = "none"`
   - `payload.data.code` must be one of the standard error codes
   - `payload.data.message` required (minLength: 5)

4. **Payload Type Mapping**: Each payload type is validated against its Layer 3 schema via `allOf`
   constraints

See `04-protocol/envelope.schema.json` for the complete normative definition.

---

## 2. Intent Naming Convention

### 2.1 Format

Intents follow the pattern: `<domain>.<verb>[.<subverb>]`

**Examples:**

- `hook.create` — create a new hook
- `tu.open` — open (start) a new TU
- `gate.report.submit` — submit a gatecheck report
- `error` — error (with `code` in payload specifying error type)

### 2.2 Domains

| Domain  | Purpose                                    | Examples                                                                 |
| ------- | ------------------------------------------ | ------------------------------------------------------------------------ |
| `hook`  | Hook Card lifecycle operations             | `hook.create`, `hook.update_status`                                      |
| `tu`    | Trace Unit lifecycle operations            | `tu.open`, `tu.update`, `tu.close`, `tu.checkpoint`                      |
| `gate`  | Quality gate lifecycle operations          | `gate.report.submit`, `gate.decision`                                    |
| `merge` | Cold merge operations                      | `merge.request`, `merge.approve`, `merge.reject`                         |
| `canon` | Canon workflow operations                  | `canon.transfer.export`, `canon.transfer.import`, `canon.genesis.create` |
| `view`  | View/export operations                     | `view.export.request`, `view.export.result`                              |
| `pn`    | Player Narrator operations                 | `pn.playtest.submit`                                                     |
| `human` | Human↔agent interactive prompts           | `human.question`, `human.response`                                       |
| `role`  | Wake/sleep control signals (orchestration) | `role.wake`, `role.dormant`                                              |
| (none)  | General-purpose operations                 | `ack`, `error`                                                           |

---

## 3. Error Taxonomy

All error intents follow the pattern `error.<error_type>` or simply `error` for general errors.
Error codes and structures are defined in `04-protocol/envelope.schema.json` and enforced by the
envelope schema validation.

### 3.1 Error Types

Error types are standardized and MUST match the `code` enum in the envelope schema:

| Error Intent             | Code                      | Trigger                                    | Remedy                                   |
| ------------------------ | ------------------------- | ------------------------------------------ | ---------------------------------------- |
| `error` (validation)     | `validation_error`        | Payload does not validate against schema   | Fix payload data to match schema         |
| `error` (business_rule)  | `business_rule_violation` | Policy violation (e.g., Hot→PN, PN Safety) | Respect business rules (PN safety, etc.) |
| `error` (not_authorized) | `not_authorized`          | Sender lacks permission for operation      | Use authorized role for operation        |
| `error` (not_found)      | `not_found`               | Referenced entity missing                  | Verify entity exists or create it        |
| `error` (conflict)       | `conflict`                | State conflict (e.g., TU already merged)   | Resolve state conflict before retrying   |

**Note:** Error intents use the `error` intent field with different `code` values in the payload,
rather than separate intent names like `error.validation`. The envelope schema enforces the valid
codes through the `allOf` constraint for `intent: "error"`.

### 3.2 Error Payload Structure (Envelope Schema Definition)

Error intents are special cases in `envelope.schema.json`. When `intent = "error"`, the envelope
schema enforces:

- `payload.type` MUST be `"none"`
- `payload.data` MUST contain:
  - `code` (string, required) — one of the error codes above
  - `message` (string, required, minLength: 5) — human-readable error message
  - `details` (object, optional) — additional context

**Envelope Requirements for Error Messages:**

- `intent` — MUST be `"error"`
- `reply_to` — MUST reference the failed message ID
- `correlation_id` — SHOULD match original message
- `payload.type` — MUST be `"none"`
- `payload.data.code` — MUST be one of: `validation_error`, `business_rule_violation`,
  `not_authorized`, `not_found`, `conflict`

**Example Error Envelope:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:error-123",
  "time": "2025-10-30T12:25:00Z",
  "sender": { "role": "GK" },
  "receiver": { "role": "LW" },
  "intent": "error",
  "context": { "hot_cold": "hot", "loop": "Gatecheck" },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "none",
    "data": {
      "code": "validation_error",
      "message": "Payload data does not validate against schema",
      "details": {
        "schema_path": "../03-schemas/hook_card.schema.json",
        "validation_errors": [
          "header.id: does not match pattern",
          "classification.bars_affected: must have at least 1 item"
        ]
      }
    }
  },
  "reply_to": "urn:uuid:original-message-id"
}
```

**Common fields in `details`:**

- `schema_path` — path to schema that failed validation
- `validation_errors` — array of specific validation failures
- `rule` — policy rule that was violated
- `violation` — specific violation description
- `reference` — link to relevant documentation
- `remedy` — suggested fix

---

## 4. General-Purpose Intents

### 4.1 `ack` — Acknowledge Receipt

**Purpose:** Confirm receipt of a message; no further action required.

**Direction:** Any role → Any role (typically receiver → sender)

**Envelope Requirements:**

When `intent = "ack"`, the envelope schema enforces:

- `payload.type` MUST be `"none"`
- `reply_to` MUST be present (references acknowledged message ID)
- `correlation_id` SHOULD match original message

**Context Fields:**

- `context.hot_cold` — inherits from original message context
- `context.loop` — inherits from original message
- Other context fields optional

**Safety Fields:**

- Typically matches original message safety context (Hot acks stay Hot, Cold acks stay Cold)

**Payload Schema:** None (payload.type = "none")

**Payload Example:**

```json
{
  "type": "none",
  "data": {
    "message": "Hook HK-20251030-01 received and queued for review"
  }
}
```

**Full Envelope Example:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:ack-abc123",
  "time": "2025-10-30T12:20:00Z",
  "sender": { "role": "LW" },
  "receiver": { "role": "SR" },
  "intent": "ack",
  "context": { "hot_cold": "hot", "loop": "Hook Harvest" },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "none",
    "data": { "message": "Hook received" }
  },
  "correlation_id": "corr-hook-harvest-2025-10-30",
  "reply_to": "urn:uuid:original-message-id"
}
```

**Expected Replies:** None (ack is terminal)

**Error Conditions:** None

**References:**

- `04-protocol/envelope.schema.json` — Schema constraint for ack intent
- `04-protocol/ENVELOPE.md` §6.1 (Example: Ack)
- `04-protocol/EXAMPLES/ack.json` — Example ack envelope

---

### 4.2 `error` — Error Response

**Purpose:** Report an error condition with standardized error codes.

**Direction:** Any role → sender of failed message

**Envelope Requirements:**

When `intent = "error"`, the envelope schema enforces:

- `payload.type` MUST be `"none"`
- `payload.data.code` MUST be one of: `validation_error`, `business_rule_violation`,
  `not_authorized`, `not_found`, `conflict`
- `payload.data.message` MUST be present (string, minLength: 5)
- `payload.data.details` MAY be present (object)
- `reply_to` MUST be present (references failed message ID)
- `correlation_id` SHOULD match original message

**Context Fields:**

- `context.hot_cold` — inherits from original message context
- `context.loop` — inherits from original message
- Other context fields optional

**Safety Fields:**

- Typically matches original message safety context

**Payload Schema:** None (payload.type = "none", enforced by envelope schema)

**Payload Structure (enforced by envelope.schema.json):**

```json
{
  "type": "none",
  "data": {
    "code": "<error_code>",
    "message": "Human-readable error message",
    "details": {
      "schema_path": "...",
      "validation_errors": [],
      "rule": "...",
      "violation": "...",
      "reference": "..."
    }
  }
}
```

**Full Envelope Example:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:error-def456",
  "time": "2025-10-30T12:25:00Z",
  "sender": { "role": "GK" },
  "receiver": { "role": "LW" },
  "intent": "error",
  "context": { "hot_cold": "hot", "loop": "Gatecheck" },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "none",
    "data": {
      "code": "validation_error",
      "message": "Payload data does not validate against schema",
      "details": {
        "schema_path": "../03-schemas/hook_card.schema.json",
        "validation_errors": ["header.id: invalid pattern"]
      }
    }
  },
  "reply_to": "urn:uuid:original-message-id"
}
```

**Expected Replies:** None (error is terminal)

**Error Conditions:** N/A (this is an error intent)

**References:**

- `04-protocol/envelope.schema.json` — Schema constraint for error intent
- `04-protocol/ENVELOPE.md` §5 (Error Envelopes), §6.2 (Example: Validation Error)
- `04-protocol/EXAMPLES/error.validation.json` — Example error envelope
- `04-protocol/CONFORMANCE.md` §4 (Error Taxonomy)

---

## 5. Human Interaction Intents

Interactive mode allows agents to ask a human collaborator questions mid‑loop. The Showrunner (SR)
proxies human I/O via the protocol; no direct human role exists in Layer 4. Payload type is `none`
for both intents; `payload.data` carries question/answer content.

### 5.1 `human.question` — Ask Human a Question

Purpose: Agent requests human input during an active TU/loop.

Direction: Any role → SR (SR proxies the question to UI/CLI)

Envelope Requirements:

- `context.hot_cold = "hot"`
- `context.loop` SHOULD be present
- `payload.type = "human_interaction"`
- `payload.data` validated against `human_interaction.schema.json`

Payload Schema: `human_interaction.schema.json`

Payload Requirements:

- `question` (string, required) — The question being asked
- `context` (object, optional) — Flexible context with additionalProperties allowed
- `suggested_answers` (array[string], optional) — Suggested answer choices

Example Envelope: `04-protocol/EXAMPLES/human.question.json`

### 5.2 `human.response` — Human Answer to Agent

Purpose: SR relays the human’s answer back to the requesting role.

Direction: SR → requesting role

Envelope Requirements:

- `reply_to` MUST reference the original `human.question` message id
- `correlation_id` SHOULD match original message
- `context` SHOULD mirror original
- `payload.type = "none"`
- `payload.data` SHOULD include:
  - `answer` (string, required) or `choice` (string) when suggestions used
  - `notes` (string, optional)

Example Envelope: `04-protocol/EXAMPLES/human.response.json`

---

## 6. TU Utility Intents

### 6.1 `tu.checkpoint` — Record a TU Checkpoint

Purpose: Record a mid‑loop checkpoint note without creating a new artifact.

Direction: Any role → SR

Envelope Requirements:

- `context.tu` SHOULD be present
- `context.loop` SHOULD be present
- `payload.type = "none"`
- `payload.data` SHOULD include:
  - `summary` (string, required)
  - `details` (object, optional)

Example Envelope: `04-protocol/EXAMPLES/tu.checkpoint.json`

---

## 7. Role Orchestration Intents

These are control signals used by the Showrunner to manage role dormancy. They carry no artifacts;
payload type is `none`.

### 7.1 `role.wake` — Wake a Dormant Role

Purpose: Instruct a role to become active for the current TU/loop.

Direction: SR → <role>

Envelope Requirements:

- `context.tu` SHOULD be present
- `payload.type = "none"`
- `payload.data` SHOULD include:
  - `reason` (string, required)

Example Envelope: `04-protocol/EXAMPLES/role.wake.json`

### 7.2 `role.dormant` — Return Role to Dormancy

Purpose: Instruct a role to go dormant after handoff.

Direction: SR → <role>

Envelope Requirements:

- `context.tu` SHOULD be present
- `payload.type = "none"`
- `payload.data` MAY include:
  - `notes` (string, optional)

Example Envelope: `04-protocol/EXAMPLES/role.dormant.json`

## 8. Hook Lifecycle Intents

### 8.1 `hook.create` — Create Hook

**Purpose:** Create a new Hook Card in `proposed` state.

**Direction:** Any role (typically SR) → LW or other owner role

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"` (hooks are Hot artifacts)
- `context.tu` — SHOULD reference the TU that prompted the hook
- `safety.player_safe` — MUST be `false` (hooks are internal)

**Payload Schema:** `hook_card.schema.json`

**Payload Requirements:**

- `header.status` — MUST be `"proposed"`
- `header.id` — Hook ID (format: `HK-YYYYMMDD-seq`)
- `classification` — Hook type, bars affected, blocking status
- `player_safe_summary` — One-line player-safe description

**Expected Replies:**

- `ack` — Hook received and queued
- `error` — Payload validation failed (with `code: "validation_error"`)

**Error Conditions:**

- `validation_error` — if payload does not match schema
- `conflict` — if hook ID already exists

**References:**

- `04-protocol/LIFECYCLES/hooks.md` — Hook lifecycle state machine
- `03-schemas/hook_card.schema.json` — Hook Card schema
- `04-protocol/ENVELOPE.md` §6.4 (Example: Hook Creation)

---

### 8.2 `hook.update_status` — Update Hook Status

**Purpose:** Transition a Hook Card to a new state (e.g., `proposed` → `accepted`).

**Direction:** Authorized role (per lifecycle) → broadcast or owner role

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"`
- `context.tu` — SHOULD reference the TU driving the transition
- `refs` — SHOULD include the hook ID being updated

**Payload Schema:** `hook_card.schema.json` (partial or full)

**Payload Requirements:**

- `header.id` — Hook ID being updated
- `header.status` — New status (must be valid transition per lifecycle)
- `header.edited` — Update timestamp

**Expected Replies:**

- `ack` — Status update accepted
- `error` — Invalid state transition (with `code: "business_rule_violation"`)
- `error` — Sender lacks permission (with `code: "not_authorized"`)

**Error Conditions:**

- `business_rule_violation` — transition not allowed per state machine
- `not_authorized` — sender role not allowed for transition
- `not_found` — hook ID does not exist

**Lifecycle Transition Mapping:**

All transitions use the `hook.update_status` intent with `header.status` field in payload specifying
the target state:

| From State    | To State      | Payload Status Field | Sender    |
| ------------- | ------------- | -------------------- | --------- |
| `proposed`    | `accepted`    | `"accepted"`         | SR        |
| `proposed`    | `deferred`    | `"deferred"`         | SR        |
| `proposed`    | `rejected`    | `"rejected"`         | SR        |
| `accepted`    | `in-progress` | `"in-progress"`      | Owner (R) |
| `in-progress` | `resolved`    | `"resolved"`         | Owner (R) |
| `resolved`    | `canonized`   | `"canonized"`        | SR        |
| `resolved`    | `in-progress` | `"in-progress"`      | Owner (R) |
| `deferred`    | `accepted`    | `"accepted"`         | SR        |
| `deferred`    | `rejected`    | `"rejected"`         | SR        |

**References:**

- `04-protocol/LIFECYCLES/hooks.md` §3 (State Transitions)
- `04-protocol/LIFECYCLES/hooks.md` §9 (Examples)

---

## 9. TU (Trace Unit) Lifecycle Intents

### 9.1 `tu.open` — Open Trace Unit

**Purpose:** Create and open a new Trace Unit in `hot-proposed` state, then transition to
`stabilizing`.

**Direction:** SR or Owner (A) → broadcast

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"` (TUs coordinate Hot work)
- `context.tu` — MUST reference the new TU ID
- `context.snapshot` — SHOULD reference Cold snapshot being modified
- `safety.player_safe` — MUST be `false` (TUs are internal)

**Payload Schema:** `tu_brief.schema.json`

**Payload Requirements:**

- `id` — TU ID (format: `TU-YYYY-MM-DD-RRnn`)
- `opened` — Date opened
- `owner_a` — Accountable role (usually SR)
- `responsible_r` — Responsible roles (array)
- `loop` — Loop name from taxonomy
- `awake` / `dormant` — Role wake/sleep lists
- `timebox` — Time allocation (e.g., `"60 min"`)

**Expected Replies:**

- `ack` — TU opened successfully
- `error` — Payload validation failed (with `code: "validation_error"`)

**Error Conditions:**

- `validation_error` —if payload does not match schema
- `CONFLICT` — if TU ID already exists

**References:**

- `04-protocol/LIFECYCLES/tu.md` §4.1 (hot-proposed → stabilizing)
- `03-schemas/tu_brief.schema.json` — TU Brief schema
- `04-protocol/LIFECYCLES/tu.md` §10.1 (Example: Start TU)

---

### 9.2 `tu.update` — Update Trace Unit

**Purpose:** Update a TU's state during its lifecycle (e.g., adding deliverables, submitting for
gatecheck).

**Direction:** Owner (A) or GK → broadcast or specific role

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"` (unless final merge, then `"cold"`)
- `context.tu` — MUST reference the TU ID being updated
- `context.snapshot` — SHOULD reference Cold snapshot context

**Payload Schema:** `tu_brief.schema.json` (partial or full)

**Payload Requirements:**

- `id` — TU ID being updated
- Additional fields depending on transition (e.g., `deliverables` for gatecheck submission)

**Expected Replies:**

- `ack` — Update accepted
- `error` —Invalid state transition
- `error.not_authorized` — Sender lacks permission

**Error Conditions:**

- `business_rule_violation` —transition not allowed per state machine
- `not_authorized` —sender role not allowed for transition
- `NOT_FOUND` — TU ID does not exist

**Lifecycle Transition Mapping:**

| From State     | To State      | Intent Subverb   | Sender      |
| -------------- | ------------- | ---------------- | ----------- |
| `hot-proposed` | `stabilizing` | `tu.start`       | SR or Owner |
| `hot-proposed` | `deferred`    | `tu.defer`       | SR          |
| `hot-proposed` | `rejected`    | `tu.reject`      | SR          |
| `stabilizing`  | `gatecheck`   | `tu.submit_gate` | Owner (A)   |
| `stabilizing`  | `deferred`    | `tu.defer`       | SR or Owner |
| `gatecheck`    | `stabilizing` | `tu.rework`      | GK or Owner |
| `deferred`     | `stabilizing` | `tu.reactivate`  | SR          |
| `deferred`     | `rejected`    | `tu.reject`      | SR          |

**References:**

- `04-protocol/LIFECYCLES/tu.md` §3 (State Transitions)
- `04-protocol/LIFECYCLES/tu.md` §10 (Examples)

---

### 9.3 `tu.close` — Close Trace Unit (Merge to Cold)

**Purpose:** Complete TU lifecycle by merging to Cold snapshot (transition to `cold-merged`).

**Direction:** SR → broadcast

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"cold"` (final merge)
- `context.tu` — MUST reference the TU ID being closed
- `context.snapshot` — MUST reference the new Cold snapshot ID
- `safety.player_safe` — MUST be `false` (TU briefs are internal)

**Payload Schema:** `tu_brief.schema.json` (full)

**Payload Requirements:**

- `id` — TU ID being closed
- `snapshot_context` — Cold snapshot ID (format: `"Cold @ YYYY-MM-DD"`)
- `linkage` — Final merge notes, artifact locations, trace references

**Expected Replies:**

- `ack` — Merge successful
- `error` —Gatecheck not passed or other violation

**Error Conditions:**

- `business_rule_violation` —TU not in `gatecheck` state
- `not_authorized` —sender is not SR
- `business_rule_violation` —gatecheck not passed or bars not green

**References:**

- `04-protocol/LIFECYCLES/tu.md` §4.5 (gatecheck → cold-merged)
- `04-protocol/LIFECYCLES/tu.md` §10.3 (Example: Merge to Cold)

---

## 10. Gate (Gatecheck) Lifecycle Intents

### 10.1 `gate.report.submit` — Submit Gatecheck Report

**Purpose:** Submit a gatecheck report (pre-gate or full gatecheck) for a TU or View.

**Direction:** GK → SR (and potentially owner roles)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"` (gatecheck coordination is Hot)
- `context.tu` — MUST reference the TU being gatechecked
- `context.snapshot` — MUST reference Cold snapshot being validated
- `safety.player_safe` — MUST be `false` (gatecheck reports are internal)

**Payload Schema:** `gatecheck_report.schema.json`

**Payload Requirements:**

- `title` — TU ID or View name
- `checked` — Date of gatecheck
- `gatekeeper` — GK name or agent
- `mode` — `"pre-gate"` or `"gatecheck"`
- `cold_snapshot` — Cold snapshot reference
- `bars` — Array of 8 quality bar evaluations (for full gatecheck)
- `decision` — `"pass"`, `"conditional pass"`, or `"block"`

**Guidance**

- Use `gatecheck_report.schema.json` and include all 8 bars.
- Tie `decision` to bar statuses (pass / conditional pass / block).
- When reporting Nonlinearity/Determinism issues on converged branches, use Choice Integrity terms
  (Immediate reflection, Diegetic bridge, State-aware affordances) and provide player-safe evidence
  (first-paragraph reflection is required; not necessarily a literal echo). See
  `02-dictionary/conventions/choice_integrity.md`.

**Expected Replies:**

- `ack` — Report received
- `error.validation` — Report validation failed

**Error Conditions:**

- `validation_error` —if report does not match schema
- `business_rule_violation` —if decision conflicts with bar status

**References:**

- `04-protocol/LIFECYCLES/gate.md` §4 (Transition Details)
- `03-schemas/gatecheck_report.schema.json` — Gatecheck Report schema
- `04-protocol/LIFECYCLES/gate.md` §10 (Examples)

---

### 10.2 `gate.decision` — Gatecheck Decision

**Purpose:** Final gatecheck decision: pass, conditional pass, or block. Centralized feedback flow.

**Direction:** GK → **Showrunner** (exclusively)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"`
- `context.tu` — MUST reference the TU being gatechecked
- `context.snapshot` — MUST reference Cold snapshot

**Payload Schema:** `gatecheck_report.schema.json` (full)

**Payload Requirements:**

- All fields from `gate.report.submit`
- `decision` — MUST be `"pass"`, `"conditional pass"`, or `"block"`
- `bars` — All 8 bars MUST be evaluated with status (green/yellow/red)
- `handoffs` — Remediation handoffs for yellow/red bars

**Expected Replies:**

- `ack` — Decision acknowledged
- `tu.merge` — if decision is `pass` (SR proceeds with merge)
- `tu.rework` — if decision is `block` (owner addresses red bars)

**Error Conditions:**

- `validation_error` — if decision conflicts with bar status
- `not_authorized` — sender is not GK

**Decision Mapping:**

All gate decisions use the `gate.decision` intent with the `decision` field in payload specifying
the outcome:

| Decision         | Bar Status Requirement     | Payload Decision Field |
| ---------------- | -------------------------- | ---------------------- |
| Pass             | All bars green             | `"pass"`               |
| Conditional Pass | Some bars yellow, none red | `"conditional_pass"`   |
| Block            | One or more bars red       | `"block"`              |

**References:**

- `04-protocol/LIFECYCLES/gate.md` §4.3-4.5 (Decision transitions)
- `04-protocol/LIFECYCLES/gate.md` §10.2-10.4 (Examples)

---

### 10.3 `gate.feedback` — Informal Gate Feedback

**Purpose:** Informal feedback loops for Yellow/Conditional Pass scenarios that do not change TU state. Allows iterative refinement without formal rework cycles.

**Direction:** GK → Owner

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"`
- `context.tu` — MUST reference the TU receiving feedback
- `context.loop` — SHOULD reference the active loop

**Payload Schema:** `gatecheck_report.schema.json`

**Payload Requirements:**

- `bars` — Specific bars with yellow status and remediation suggestions
- `feedback_notes` — Informal notes for owner (suggestions, not requirements)
- `urgency` — OPTIONAL: `"low"`, `"medium"`, `"high"` (default: `"low"`)

**Use Cases:**

- Yellow bar remediation suggestions (conditional pass with refinement suggestions)
- Style/presentation polish recommendations
- Non-blocking improvements

**Expected Replies:**

- `ack` — Feedback acknowledged
- (No formal TU state change; owner may address feedback in current TU or defer)

**Error Conditions:**

- `not_authorized` — sender is not GK
- `business_rule_violation` — feedback used for red bars (should use `gate.decision` with `block` instead)

**Note:** This intent enables lightweight iteration without triggering formal TU rework. It does NOT change TU state (unlike `gate.decision` with `block`).

**References:**

- `04-protocol/LIFECYCLES/gate.md` §4.4 (Conditional Pass handling)

---

## 11. Merge Lifecycle Intents

### 11.1 `merge.request` — Request Cold Merge

**Purpose:** Request approval to merge Hot changes to Cold snapshot.

**Direction:** Owner (A) → SR (via GK)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"` (merge request originates from Hot)
- `context.tu` — MUST reference the TU requesting merge
- `context.snapshot` — MUST reference target Cold snapshot
- `safety.player_safe` — MUST be `false`

**Payload Schema:** `tu_brief.schema.json` (full, ready for gatecheck)

**Payload Requirements:**

- `id` — TU ID requesting merge
- `deliverables` — List of artifacts to merge
- `bars_green` — Bars claimed to be green
- `gatecheck` — Gatecheck plan

**Expected Replies:**

- `gate.report.submit` — GK begins gatecheck
- `error.validation` — Payload incomplete or invalid

**Error Conditions:**

- `validation_error` —if deliverables or gatecheck plan missing
- `business_rule_violation` —if TU not in `gatecheck` state

**References:**

- `04-protocol/LIFECYCLES/tu.md` §4.4 (stabilizing → gatecheck)

---

### 11.2 `merge.approve` — Approve Cold Merge

**Purpose:** Approve merge to Cold after successful gatecheck.

**Direction:** SR → broadcast (after `gate.decision` with `decision: "pass"` or
`decision: "conditional_pass"`)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"cold"` (merge approved, now Cold)
- `context.tu` — MUST reference the TU being merged
- `context.snapshot` — MUST reference the new Cold snapshot ID
- `safety.player_safe` — MUST be `false`

**Payload Schema:** `tu_brief.schema.json` (full)

**Payload Requirements:**

- `id` — TU ID being merged
- `snapshot_context` — New Cold snapshot ID
- `linkage` — Merge notes, artifact locations

**Expected Replies:**

- `ack` — Merge acknowledged by downstream systems (Binder, PN)

**Error Conditions:**

- `business_rule_violation` —TU not in `gatecheck` state
- `not_authorized` —sender is not SR
- `business_rule_violation` —gatecheck not passed

**References:**

- `04-protocol/LIFECYCLES/tu.md` §4.5 (gatecheck → cold-merged)
- `04-protocol/LIFECYCLES/tu.md` §10.3 (Example: Merge to Cold)

---

### 11.3 `merge.reject` — Reject Cold Merge

**Purpose:** Reject merge to Cold after failed gatecheck; require rework.

**Direction:** GK or SR → Owner (A) (after `gate.decision` with `decision: "block"`)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"hot"` (merge rejected, stays Hot)
- `context.tu` — MUST reference the TU being rejected
- `context.snapshot` — SHOULD reference target Cold snapshot

**Payload Schema:** `gatecheck_report.schema.json` (with `decision: "block"`)

**Payload Requirements:**

- `decision` — MUST be `"block"`
- `bars` — One or more bars MUST be red
- `handoffs` — Remediation tasks for red bars

**Expected Replies:**

- `tu.rework` — Owner addresses red bars and resubmits

**Error Conditions:**

- `not_authorized` — sender is not GK or SR
- `validation_error` —if decision is not `block` or red bars lack remediation

**References:**

- `04-protocol/LIFECYCLES/tu.md` §4.6 (gatecheck → stabilizing / rework)
- `04-protocol/LIFECYCLES/gate.md` §4.5 (gatecheck → decision:block)

---

## 12. View/Export Lifecycle Intents

### 12.1 `view.export.request` — Request View Export

**Purpose:** Request export binding for a Cold snapshot.

**Direction:** SR → BB (Book Binder)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"cold"` (exports use Cold snapshots only)
- `context.snapshot` — MUST reference Cold snapshot to export
- `context.tu` — SHOULD reference the TU coordinating the export
- `safety.player_safe` — MUST be `true` (exports are player-facing)

**Payload Schema:** `view_log.schema.json` (partial)

**Payload Requirements:**

- `title` — View name
- `cold_snapshot` — Cold snapshot reference (format: `"Cold @ YYYY-MM-DD"`)
- `targets` — Export targets (e.g., `["PDF", "HTML", "EPUB"]`)
- `options_and_coverage` — Art/audio/locale coverage options

**Expected Replies:**

- `view.export.result` — BB delivers bound export artifacts
- `error` — Export binding failed

**Error Conditions:**

- `SNAPSHOT_NOT_FOUND` — if snapshot does not exist in Cold
- `SNAPSHOT_INVALID` — if snapshot format is malformed
- `validation_error` —if view_log fields incomplete

**References:**

- `04-protocol/LIFECYCLES/view.md` §4.1 (snapshot-selected → export-binding)
- `03-schemas/view_log.schema.json` — View Log schema

---

### 12.2 `view.export.result` — Export Result

**Purpose:** Deliver bound export artifacts after successful binding.

**Direction:** BB → PN (and SR)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"cold"` (exports are Cold)
- `context.snapshot` — MUST reference Cold snapshot exported
- `context.tu` — SHOULD reference the TU coordinating the export
- `safety.player_safe` — MUST be `true` (for PN consumption)
- `safety.spoilers` — MUST be `"forbidden"` (for PN consumption)

**Payload Schema:** `view_log.schema.json` (full)

**Payload Requirements:**

- All fields from `view.export.request`
- `bound` — Date bound
- `binder` — BB name or agent
- `anchor_map` — Anchor map summary (orphans, collisions)
- `export_artifacts` — Array of artifacts with paths and hashes
- `presentation_status` / `accessibility_status` — Bar status (green/yellow/red)

**Expected Replies:**

- `pn.playtest.submit` — PN delivers dry-run feedback
- `ack` — Export received

**Error Conditions:**

- `EXPORT_BINDING_FAILED` — if binding encountered errors
- `PN_HOT_BOUNDARY_VIOLATION` — if PN receives Hot content
- `PN_PLAYER_SAFE_VIOLATION` — if PN receives non-player-safe content

**References:**

- `04-protocol/LIFECYCLES/view.md` §4.3 (export-binding → pn-dry-run)
- `04-protocol/LIFECYCLES/view.md` §10.2 (Example: View Bound)
- `04-protocol/ENVELOPE.md` §6.5 (Example: Cold Content for PN)

---

## 13. PN (Player Narrator) Playtest Intent

### 13.1 `pn.playtest.submit` — Submit PN Playtest Notes

**Purpose:** Deliver PN dry-run playtest feedback after consuming a bound view.

**Direction:** PN → SR (and potentially owner roles)

**Required Envelope Fields:**

- `context.hot_cold` — MUST be `"cold"` (PN consumes only Cold)
- `context.snapshot` — MUST reference Cold snapshot playtested (MUST match view export snapshot)
- `context.tu` — SHOULD reference the TU coordinating the playtest
- `safety.player_safe` — MUST be `true` (PN output is player-safe)
- `safety.spoilers` — MUST be `"forbidden"` (no spoilers in playtest notes)

**Payload Schema:** `pn_playtest_notes.schema.json`

**Payload Requirements:**

- `title` — View name
- `run` — Playtest timestamp
- `pn` — PN name or agent
- `tu` — TU ID for playtest
- `snapshot` — Cold snapshot reference (MUST match view export)
- `log` — Array of playtest observations with tags, severity, smallest_viable_fix, owner

**Expected Replies:**

- `ack` — Feedback received and routed

**Error Conditions:**

- `SNAPSHOT_MISMATCH` — if PN snapshot does not match view export snapshot
- `PN_HOT_BOUNDARY_VIOLATION` — if PN snapshot is Hot (critical error)
- `validation_error` —if playtest notes incomplete

**References:**

- `04-protocol/LIFECYCLES/view.md` §4.4 (pn-dry-run → feedback-collected)
- `03-schemas/pn_playtest_notes.schema.json` — PN Playtest Notes schema
- `04-protocol/LIFECYCLES/view.md` §10.3 (Example: PN Dry-Run Feedback)

---

## 14. Intent Summary Table

### 14.1 All Intents by Domain

| Intent                  | Purpose                               | Sender    | Receiver  | Payload Schema                       |
| ----------------------- | ------------------------------------- | --------- | --------- | ------------------------------------ |
| `ack`                   | Acknowledge receipt                   | Any       | Any       | None                                 |
| `error`                 | Error (with `code` in payload)        | Any       | Sender    | None (error structure)               |
| `hook.create`           | Create hook                           | Any       | Owner     | `hook_card.schema.json`              |
| `hook.update_status`    | Update hook status                    | SR/Owner  | Owner     | `hook_card.schema.json`              |
| `tu.open`               | Open TU                               | SR/Owner  | Broadcast | `tu_brief.schema.json`               |
| `tu.start`              | Start TU work                         | SR/Owner  | Broadcast | `tu_brief.schema.json`               |
| `tu.defer`              | Defer TU                              | SR        | Broadcast | `tu_brief.schema.json`               |
| `tu.reject`             | Reject TU                             | SR        | Broadcast | `tu_brief.schema.json`               |
| `tu.submit_gate`        | Submit TU for gatecheck               | Owner (A) | GK        | `tu_brief.schema.json`               |
| `tu.rework`             | Rework TU after gatecheck failure     | GK/Owner  | Owner     | `tu_brief.schema.json`               |
| `tu.reactivate`         | Reactivate deferred TU                | SR        | Broadcast | `tu_brief.schema.json`               |
| `tu.close`              | Close TU (merge to Cold)              | SR        | Broadcast | `tu_brief.schema.json`               |
| `gate.report.submit`    | Submit gatecheck report (pre-gate)    | GK        | SR        | `gatecheck_report.schema.json`       |
| `gate.decision`         | Gate decision (with `decision` field) | GK        | SR/Owner  | `gatecheck_report.schema.json`       |
| `gate.defer`            | Defer gatecheck                       | SR/GK     | Broadcast | `tu_brief.schema.json`               |
| `merge.request`         | Request merge to Cold                 | Owner (A) | SR/GK     | `tu_brief.schema.json`               |
| `merge.approve`         | Approve merge to Cold                 | SR        | Broadcast | `tu_brief.schema.json`               |
| `merge.reject`          | Reject merge to Cold                  | GK/SR     | Owner     | `gatecheck_report.schema.json`       |
| `view.export.request`   | Request view export                   | SR        | BB        | `view_log.schema.json`               |
| `view.export.result`    | Export result                         | BB        | PN/SR     | `view_log.schema.json`               |
| `view.bind`             | Start export binding                  | BB        | SR        | `view_log.schema.json`               |
| `view.bound`            | Export binding complete               | BB        | PN        | `view_log.schema.json`               |
| `view.feedback`         | PN feedback                           | PN        | SR        | `pn_playtest_notes.schema.json`      |
| `view.publish`          | Publish view                          | SR/BB     | Broadcast | `view_log.schema.json`               |
| `pn.playtest.submit`    | Submit PN playtest notes              | PN        | SR        | `pn_playtest_notes.schema.json`      |
| `canon.transfer.export` | Export canon transfer package         | LW        | SR        | `canon_transfer_package.schema.json` |
| `canon.transfer.import` | Import canon transfer package         | LW        | SR        | `canon_transfer_package.schema.json` |
| `canon.genesis.create`  | Create World Genesis manifest         | LW        | SR        | `world_genesis_manifest.schema.json` |

---

## 15. Authorization Matrix

### 15.1 Role-Based Authorization

| Intent Domain | Allowed Senders                                              | Typical Receivers        |
| ------------- | ------------------------------------------------------------ | ------------------------ |
| `hook.*`      | SR, Owner (R), any role for `create`                         | Owner, SR, Broadcast     |
| `tu.*`        | SR, Owner (A), GK (for `rework`)                             | Broadcast, GK, Owner     |
| `gate.*`      | GK, SR (for `defer`)                                         | SR, Owner, Broadcast     |
| `merge.*`     | SR (for `approve`), Owner (for `request`), GK (for `reject`) | Broadcast, Owner         |
| `canon.*`     | LW (for `transfer.*`, `genesis.create`)                      | SR, Broadcast            |
| `view.*`      | BB, SR, PN (for `feedback`)                                  | BB, PN, SR, Broadcast    |
| `pn.*`        | PN                                                           | SR, Owner                |
| `ack`         | Any                                                          | Any                      |
| `error.*`     | Any                                                          | Sender of failed message |

**Role Abbreviations:**

- **SR** — Showrunner
- **GK** — Gatekeeper
- **Owner (R)** — Responsible role (from hook `proposed_next_step.owner_r`)
- **Owner (A)** — Accountable role (from TU `owner_a`)
- **BB** — Book Binder
- **PN** — Player Narrator

---

## 16. Cross-References

### Layer 0/1 Policy

- `00-north-star/TRACEABILITY.md` — TU requirements and Cold-bound rule
- `00-north-star/QUALITY_BARS.md` — 8 quality bar definitions
- `00-north-star/PN_PRINCIPLES.md` — PN boundaries and safety rules
- `01-roles/charters/*.md` — Role authorities and charters

### Layer 2 Dictionary

- `02-dictionary/role_abbreviations.md` — Role abbreviation reference
- `02-dictionary/loop_names.md` — Loop name reference
- `02-dictionary/taxonomies.md` — Hook status, TU types, quality bars

### Layer 3 Schemas

- `03-schemas/hook_card.schema.json` — Hook Card payload schema
- `03-schemas/tu_brief.schema.json` — TU Brief payload schema
- `03-schemas/gatecheck_report.schema.json` — Gatecheck Report payload schema
- `03-schemas/view_log.schema.json` — View Log payload schema
- `03-schemas/pn_playtest_notes.schema.json` — PN Playtest Notes payload schema

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/LIFECYCLES/hooks.md` — Hook lifecycle state machine
- `04-protocol/LIFECYCLES/tu.md` — TU lifecycle state machine
- `04-protocol/LIFECYCLES/gate.md` — Gate lifecycle state machine
- `04-protocol/LIFECYCLES/view.md` — View/export lifecycle state machine

---

## 17. Forward Compatibility

### 17.1 Unknown Intents

Receivers SHOULD reject unknown intents unless explicitly configured to accept them (e.g., for
testing or extensibility).

**Error response for unknown intent:**

```json
{
  "code": "UNKNOWN_INTENT",
  "message": "Intent not recognized",
  "details": {
    "intent": "hook.experimental_action",
    "known_intents": ["hook.create", "hook.accept", ...],
    "remedy": "Use a known intent from INTENTS.md or request protocol extension"
  }
}
```

### 17.2 Intent Versioning

Intents MAY be versioned via protocol version (e.g., `protocol.version = "2.0.0"` adds new intents).
Minor version changes MAY add intents; major version changes MAY remove or change intent semantics.

**Guidance:**

- New intents in minor versions should be optional (receivers can ignore if not supported)
- Breaking changes to existing intents require major version bump

---

## 18. Implementation Checklist

For implementers of intent handling systems:

- [ ] Validate `intent` field present in all messages
- [ ] Check intent against catalog of known intents
- [ ] Verify sender role authorized for intent
- [ ] Verify receiver role valid for intent
- [ ] Validate payload against schema specified by intent
- [ ] Enforce lifecycle transition rules (if applicable)
- [ ] Generate appropriate error responses for violations
- [ ] Support all error intents in error taxonomy
- [ ] Log intent usage for audit and debugging
- [ ] Handle unknown intents gracefully (reject with helpful error)

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group
