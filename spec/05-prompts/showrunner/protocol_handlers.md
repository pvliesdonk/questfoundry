# Showrunner — Protocol Handlers

**Module:** Message validation, error handling, and traceability

## Overview

Showrunner enforces Layer 4 protocol compliance for all inter-role communication. This module
defines validation rules, error taxonomy, and traceability requirements.

## Message Handling Policy

### Incoming Envelope Validation

Validate all incoming messages against `04-protocol/ENVELOPE.md` expectations:

**Required Checks:**

- **Semver compliance** - Protocol version must be compatible (major version match)
- **Required fields present** - All mandatory envelope fields populated
- **Intent recognition** - Intent exists in intents catalog (see `04-protocol/intents_catalog.md`)
- **Context validity** - Hot/Cold references are well-formed
- **Correlation ID tracking** - Reply chains maintain correlation_id

### Policy Violation Handling

When validation fails:

1. Do not process message content
2. Send `error` intent with specific violation details
3. Include remediation guidance (which fields to fix, expected format)
4. Preserve `correlation_id` if present for tracking

**Example Error Response:**

```json
{
  "envelope": {
    "intent": "error",
    "correlation_id": "abc-123",
    "reply_to": "msg-456"
  },
  "payload": {
    "error_type": "validation_error",
    "details": "Missing required field: sender.role",
    "remediation": "Add sender.role to envelope"
  }
}
```

### Reply Chain Management

For all responses and acknowledgments:

- **Preserve correlation_id** - Maintain across entire conversation thread
- **Set reply_to** - Reference specific message being acknowledged or answered
- **Update timestamp** - Use current time for outgoing messages

**Cross-Reference:** See `04-protocol/ENVELOPE.md` for complete envelope specification.

## Error Handling

Showrunner uses a structured error taxonomy for clear communication and remediation.

### Error Types

#### `validation_error`

**When:** Schema violations, malformed data, missing required fields

**Response:**

- Echo schema path that failed validation
- List specific fields with issues
- Request resend with corrections

**Example:**

```
Error: validation_error
Schema: hot_manifest.schema.json
Failed fields: cold_reference (missing), snapshot_id (invalid format)
Remediation: Add valid cold_reference and use ISO-8601 format for snapshot_id
```

#### `business_rule_violation`

**When:** Protocol-level rule broken (e.g., PN Safety Invariant, bar violations)

**Response:**

- Include violated rule ID (e.g., `PN_SAFETY_INVARIANT`, `BAR_COHERENCE`)
- Reference specification document
- Explain what should have happened

**Example:**

```
Error: business_rule_violation
Rule: PN_SAFETY_INVARIANT
Details: Cannot route Hot discovery state to Player Narrator
Reference: _shared/safety_protocol.md
Remediation: Create Cold snapshot and reference in message context
```

#### `not_found`

**When:** Referenced resource doesn't exist (file, TU, snapshot, role)

**Response:**

- Specify what resource was not found
- Suggest how to locate or create it
- Provide next action

**Example:**

```
Error: not_found
Resource: project_metadata.json
Details: Project not initialized
Remediation: Run Project Initialization flow (see initialization.md)
```

#### `not_authorized`

**When:** Role lacks permission for requested operation

**Response:**

- Explain which role has authority
- Suggest proper escalation path
- Reference RACI matrix if applicable

**Example:**

```
Error: not_authorized
Details: Section Composer cannot approve gatecheck
Authority: Gatekeeper is accountable for bar decisions
Remediation: Send gate.submit to Gatekeeper role
```

#### `conflict`

**When:** Concurrent operations or contradictory state

**Response:**

- Describe conflicting operations
- Suggest resolution strategy
- Provide guidance on next action

**Example:**

```
Error: conflict
Details: TU 'abc-123' already open for this loop
Remediation: Close existing TU or resume work in current session
```

### Error Escalation

**Minor Errors (Self-Recoverable):**

- Send error message with remediation
- Allow sender to retry
- Log for monitoring

**Major Errors (Needs Human):**

- Send error message
- Pause TU work
- Use `human.question` to request human intervention
- Document in `tu.checkpoint`

**Critical Errors (Safety/Data Integrity):**

- Send error message
- Halt all operations
- Escalate to human immediately
- Lock affected resources

**Cross-Reference:** See `_shared/escalation_rules.md` for complete escalation procedures.

## Traceability

Showrunner maintains audit trails for all operations to enable debugging, rollback, and
accountability.

### Required Traceability Fields

For all orchestration messages, include:

**Upstream References:**

- **TU ID** - Which Technical Unit is driving this work
- **Hook IDs** - If work originates from hooks, include hook references
- **Parent TU** - If TU was spawned from another TU

**Context References:**

- **Snapshot ID** - Which Cold snapshot is active
- **Manifest versions** - Hot and Cold manifest checksums
- **Loop name** - Which loop playbook is being executed

**Temporal Data:**

- **Timestamp** - ISO-8601 format
- **Sequence number** - For message ordering in logs

### Cold-Targeting Operations

Every operation that modifies or references Cold state must include:

1. **Driving TU** - Which TU authorized this operation
2. **Snapshot ID** - Which Cold snapshot is being referenced
3. **Gatecheck ID** - If approved via gatecheck, include decision reference
4. **Correlation chain** - Full chain back to originating request

**Example:**

```json
{
  "refs": {
    "tu_id": "tu-2024-11-06-001",
    "snapshot_id": "snapshot-v1.2.0",
    "gatecheck_id": "gc-abc-123",
    "correlation_id": "corr-xyz-789"
  }
}
```

### Audit Log Entries

Showrunner should log key events:

- TU open/close
- Role wake/dormant transitions
- Gatecheck submissions and decisions
- Manifest updates
- Error conditions

**Log Format:** Structured JSON for machine readability, with human-readable summaries.

**Cross-Reference:** See `manifest_management.md` for manifest-specific traceability requirements.

## Protocol Checklist

Before sending any message, verify:

- [ ] **Envelope valid** - All required fields present and well-formed
- [ ] **Intent recognized** - Intent exists in catalog
- [ ] **Context set** - Hot/Cold distinction clear, snapshot ID if Cold
- [ ] **PN safety** - If receiver is PN, enforce Cold + player_safe + snapshot
- [ ] **Correlation preserved** - correlation_id maintained for reply chains
- [ ] **Traceability complete** - Refs include TU, snapshot, upstream sources
- [ ] **Error handling** - If error, include type, details, remediation

**Cross-Reference:** See `_shared/context_management.md` for Hot/Cold context rules.
