# Layer 4 Conformance Guide

> **Status:** Normative — defines conformance requirements, test matrix, and validation procedures
> for Layer 4 protocol implementations.

---

## 1. Overview

This document specifies conformance requirements for implementations of the QuestFoundry Layer 4
Protocol. Implementations MUST satisfy the normative requirements in this guide to be considered
conformant.

### Scope

- **Envelope validation** — structure, required fields, safety constraints
- **Payload validation** — Layer 3 schema compliance
- **PN Safety Invariant** — enforcement at multiple layers
- **Flow conformance** — message sequences per defined flows
- **Error handling** — standard error taxonomy and responses

---

## 2. Conformance Levels

### Level 1: Envelope Compliance

**MUST Requirements:**

1. **Protocol Metadata**
   - `protocol.name` MUST be `"qf-protocol"`
   - `protocol.version` MUST be valid semver (e.g., `"1.0.0"`)
   - Implementation MUST reject envelopes with unknown `protocol.name`

2. **Required Fields**
   - All envelopes MUST include: `protocol`, `id`, `time`, `sender`, `receiver`, `intent`,
     `context`, `safety`, `payload`
   - `id` MUST be globally unique (UUID/URN/ULID)
   - `time` MUST be RFC3339 format with timezone

3. **Routing**
   - `sender.role` and `receiver.role` MUST be valid role abbreviations from
     `02-dictionary/role_abbreviations.md`
   - `intent` MUST match defined intents in `04-protocol/INTENTS.md`

4. **Context**
   - `context.hot_cold` MUST be either `"hot"` or `"cold"`
   - `context.tu` MUST match pattern `^TU-\\d{4}-\\d{2}-\\d{2}-[A-Z]{2,4}\\d{2}$` when present
   - `context.snapshot` MUST match pattern `^Cold @ \\d{4}-\\d{2}-\\d{2}$` when present
   - `context.loop` MUST be a valid loop name from `02-dictionary/loop_names.md`

5. **Safety**
   - `safety.player_safe` MUST be boolean
   - `safety.spoilers` MUST be either `"allowed"` or `"forbidden"`

6. **Payload**
   - `payload.type` MUST be a valid payload type from envelope schema
   - `payload.data` MUST be an object

---

### Level 2: PN Safety Enforcement

**CRITICAL Requirements:**

When `receiver.role = "PN"`, the envelope MUST satisfy ALL of:

- `context.hot_cold = "cold"`
- `context.snapshot` MUST be present
- `safety.player_safe = true`
- `safety.spoilers = "forbidden"` (implicitly via player_safe)

**Enforcement Points:**

1. **Schema validation** — envelope.schema.json enforces via `allOf` constraint
2. **Sender validation** — sender MUST validate before transmission
3. **Transport routing** — router MUST reject Hot→PN messages
4. **PN ingestion** — PN MUST validate on receipt (defense in depth)

**Violation Handling:**

Violations MUST generate an `error` intent response with:

- `code`: `"business_rule_violation"`
- `details.rule`: `"PN_SAFETY_INVARIANT"`
- `details.reference`: `"00-north-star/PN_PRINCIPLES.md"`

---

### Level 3: Payload Validation

**MUST Requirements:**

1. **Schema Compliance**
   - `payload.data` MUST validate against the Layer 3 schema specified by `payload.type`
   - Schema path: `03-schemas/{payload.type}.schema.json`

2. **Validation Procedure**
   - Use the `qfspec-check-envelope` tool which automates two-pass validation:
     - **Pass 1:** Validates envelope structure against `envelope.schema.json`
     - **Pass 2:** Validates `payload.data` against the Layer 3 schema for `payload.type`
   - Examples:

     ```bash
     # Validate hook_card envelope and payload
     uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/hook.create.json

     # Validate gatecheck_report envelope and payload
     uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/gate.report.submit.json

     # Validate view_log envelope and payload
     uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/view.export.result.json

     # Validate pn_playtest_notes envelope and payload
     uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/pn.playtest.submit.json

     # Validate all examples at once
     uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/*.json
     ```

3. **Error Reporting**
   - Schema validation failures MUST return `error` intent with:
     - `code`: `"validation_error"`
     - `details.schema_path`: path to schema
     - `details.validation_errors`: array of specific errors

---

### Level 4: Flow Conformance

**MUST Requirements:**

1. **Binding Run Flow**
   - `view.export.request` MUST have `context.hot_cold = "cold"`
   - `view.export.result` to PN MUST satisfy PN Safety Invariant
   - `context.snapshot` MUST be consistent across flow

2. **Narration Dry-Run Flow**
   - Messages to PN MUST satisfy PN Safety Invariant
   - `pn.playtest.submit` MUST reference same `context.snapshot` as received view
   - Issue types SHOULD use standardized taxonomy

3. **Traceability**
   - Messages targeting Cold (merges, gatechecks) MUST include `context.tu`
   - `refs` array SHOULD include upstream TU/hook IDs

---

## 3. Test Matrix

### 3.1 Envelope Structure Tests

| Test Case                | Input                             | Expected Result |
| ------------------------ | --------------------------------- | --------------- |
| Valid envelope           | All required fields present       | Pass            |
| Missing protocol         | No `protocol` field               | Reject          |
| Invalid protocol.name    | `protocol.name = "other"`         | Reject          |
| Invalid protocol.version | `protocol.version = "not-semver"` | Reject          |
| Missing id               | No `id` field                     | Reject          |
| Invalid time format      | `time = "2025-10-30"`             | Reject          |
| Invalid sender.role      | `sender.role = "XX"`              | Reject          |
| Invalid hot_cold         | `context.hot_cold = "warm"`       | Reject          |
| Invalid tu format        | `context.tu = "TU-123"`           | Reject          |
| Invalid snapshot format  | `context.snapshot = "2025-10-28"` | Reject          |

### 3.2 PN Safety Tests

| Test Case           | Input                                             | Expected Result          |
| ------------------- | ------------------------------------------------- | ------------------------ |
| Valid PN message    | Cold + player_safe + snapshot                     | Pass                     |
| Hot to PN           | `hot_cold="hot"`, `receiver="PN"`                 | Reject with error intent |
| PN without snapshot | Cold + player_safe, no snapshot                   | Reject                   |
| PN not player_safe  | Cold + snapshot, player_safe=false                | Reject                   |
| PN with spoilers    | Cold + snapshot + player_safe, spoilers="allowed" | Reject                   |

### 3.3 Payload Validation Tests

| Test Case               | Payload Type            | Expected Result              |
| ----------------------- | ----------------------- | ---------------------------- |
| Valid hook_card         | Conforms to schema      | Pass                         |
| Invalid hook_card       | Missing required fields | Reject with validation_error |
| Valid tu_brief          | Conforms to schema      | Pass                         |
| Valid gatecheck_report  | Conforms to schema      | Pass                         |
| Valid view_log          | Conforms to schema      | Pass                         |
| Valid pn_playtest_notes | Conforms to schema      | Pass                         |
| Unknown payload type    | `type = "unknown"`      | Reject                       |

### 3.4 Flow Sequence Tests

| Test Case                             | Flow                      | Expected Result         |
| ------------------------------------- | ------------------------- | ----------------------- |
| Binding Run: Cold export              | SR→BB: cold request       | Pass                    |
| Binding Run: PN handoff               | BB→PN: cold + player_safe | Pass                    |
| Binding Run: Hot to PN                | BB→PN: hot content        | Reject                  |
| Narration Dry-Run: PN feedback        | PN→SR: same snapshot      | Pass                    |
| Narration Dry-Run: Different snapshot | PN→SR: different snapshot | Warning (inconsistency) |

---

## 4. Error Taxonomy

### Standard Error Codes

| Code                      | Intent  | Description               | Example                                      |
| ------------------------- | ------- | ------------------------- | -------------------------------------------- |
| `validation_error`        | `error` | Schema validation failure | Invalid field format, missing required field |
| `business_rule_violation` | `error` | Policy violation          | Hot→PN, missing TU for Cold merge            |
| `not_authorized`          | `error` | Permission denied         | Sender lacks role permission for action      |
| `not_found`               | `error` | Referenced entity missing | Hook ID, TU ID, snapshot not found           |
| `conflict`                | `error` | State conflict            | TU already merged, duplicate ID              |

### Error Response Template

```json
{
  "intent": "error",
  "payload": {
    "type": "none",
    "data": {
      "code": "<error_code>",
      "message": "<human-readable message>",
      "details": {
        "<context-specific fields>": "..."
      }
    }
  },
  "reply_to": "<original-message-id>"
}
```

---

## 5. Validation Tools

### 5.1 Schema Validation

**Envelope Schema:**

```bash
# Validate envelope structure
jsonschema -i example.json 04-protocol/envelope.schema.json
```

**Layer 3 Payload:**

```bash
# Validate envelope and payload using automated two-pass validation
uv run --directory tools qfspec-check-envelope example.json
```

### 5.2 PN Safety Validation

**Manual Check:**

```bash
# Check if message to PN is safe
jq 'select(.receiver.role == "PN") |
    {hot_cold: .context.hot_cold,
     snapshot: .context.snapshot,
     player_safe: .safety.player_safe}' example.json
```

**Expected output for valid PN message:**

```json
{
  "hot_cold": "cold",
  "snapshot": "Cold @ 2025-10-28",
  "player_safe": true
}
```

### 5.3 Automated Test Suite

**Run all examples with two-pass validation:**

```bash
# Validate all example envelopes and payloads using qfspec-check-envelope
# This performs both envelope structure validation and payload validation automatically
uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/*.json
```

**Validate individual examples:**

```bash
# Hook creation
uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/hook.create.json

# TU opening
uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/tu.open.lore.json

# Gate report
uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/gate.report.submit.json

# View export
uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/view.export.request.json

# PN playtest notes
uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/pn.playtest.submit.json
```

---

## 6. Conformance Checklist

### Implementation Checklist

- [ ] **Envelope Validation**
  - [ ] Validates all required fields
  - [ ] Rejects unknown protocol.name
  - [ ] Enforces field format constraints (id, time, tu, snapshot patterns)
  - [ ] Validates role abbreviations against dictionary

- [ ] **PN Safety Enforcement**
  - [ ] Schema validation enforces PN constraints
  - [ ] Sender validates before sending to PN
  - [ ] Transport layer rejects Hot→PN
  - [ ] PN validates safety on receipt
  - [ ] Generates error intent with code "business_rule_violation" for violations

- [ ] **Payload Validation**
  - [ ] Validates payload.data against Layer 3 schema
  - [ ] Generates error intent with code "validation_error" and details
  - [ ] Supports all payload types in envelope schema

- [ ] **Flow Conformance**
  - [ ] Implements Binding Run flow
  - [ ] Implements Narration Dry-Run flow
  - [ ] Maintains snapshot consistency
  - [ ] Includes TU linkage for Cold operations

- [ ] **Error Handling**
  - [ ] Uses standard error codes
  - [ ] Includes reply_to for errors
  - [ ] Provides actionable error details

- [ ] **Traceability**
  - [ ] Includes context.tu for Cold operations
  - [ ] Populates refs array for dependencies
  - [ ] Maintains correlation_id for workflows

---

## 7. Cross-References

- **Envelope Spec:** `04-protocol/ENVELOPE.md`
- **Flow Specs:** `04-protocol/FLOWS/*.md`
- **Intents:** `04-protocol/INTENTS.md`
- **Schemas:** `03-schemas/*.schema.json`, `04-protocol/envelope.schema.json`
- **PN Principles:** `00-north-star/PN_PRINCIPLES.md`
- **Tools:** `spec-tools/README.md`

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30
