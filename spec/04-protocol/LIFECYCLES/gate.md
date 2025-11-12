# Gate Lifecycle — Gatecheck Decisioning and Quality Bar Enforcement

> **Status:** Normative — this document defines the canonical gatecheck lifecycle and decision flow
> in Layer 4 protocol.

---

## 1. Overview

This specification defines the **Gatecheck lifecycle**: the state transitions from pre-gate review
through final gate decision, the three possible outcomes (pass, conditional pass, block), and how
decisions map to Quality Bar status and remediation requirements.

### Purpose

Gatechecks ensure that all work merging to Cold SoT meets defined quality standards. The lifecycle
ensures:

- **Quality gates enforced** — no Cold merge without gatekeeper approval
- **Clear decisions** — pass/conditional pass/block with explicit rationale
- **Actionable remediation** — smallest viable fixes identified with owners
- **Traceability** — all decisions linked to TU context and bar evidence

### Design Principles

- **Explicit bar status** — every bar gets green/yellow/red evaluation
- **Evidence-based decisions** — rationale tied to concrete observations
- **Player-safety paramount** — all evidence must be player-safe (no spoilers)
- **Minimal fixes** — remediation focuses on smallest viable change

---

## 2. State Machine

### 2.1 States

The Gatecheck lifecycle has **4 states**:

```
pre-gate → gatecheck → decision (pass | conditional pass | block)
         ↘ deferred (if TU deferred before gatecheck)
```

**States:**

- `pre-gate` — Early checkpoint; gatekeeper samples representative slice to identify likely failure
  points
- `gatecheck` — Formal quality bar validation; all bars evaluated
- `decision:pass` — All bars green; approved for Cold merge (terminal)
- `decision:conditional-pass` — Some bars yellow; approved with time-boxed remediation plan
  (terminal)
- `decision:block` — One or more bars red; blocked from Cold merge pending fixes (terminal)
- `deferred` — TU deferred before completing gatecheck (lifecycle suspended)

**Terminal states:** `decision:pass`, `decision:conditional-pass`, `decision:block` — once reached,
gatecheck is complete and recorded

---

## 3. State Transitions

### 3.1 Transition Matrix

| From State  | To State                    | Allowed Sender | Intent          | Required Payload           | Notes                                                       |
| ----------- | --------------------------- | -------------- | --------------- | -------------------------- | ----------------------------------------------------------- |
| `pre-gate`  | `gatecheck`                 | GK             | `gate.submit`   | gatecheck_report (partial) | Pre-gate passed; ready for full check                       |
| `pre-gate`  | `deferred`                  | SR or GK       | `gate.defer`    | tu_brief (partial)         | TU deferred before gatecheck                                |
| `gatecheck` | `decision:pass`             | GK             | `gate.decision` | gatecheck_report (full)    | All bars green; `decision: "pass"` in payload               |
| `gatecheck` | `decision:conditional-pass` | GK             | `gate.decision` | gatecheck_report (full)    | Some bars yellow; `decision: "conditional_pass"` in payload |
| `gatecheck` | `decision:block`            | GK             | `gate.decision` | gatecheck_report (full)    | One or more bars red; `decision: "block"` in payload        |

**Role Abbreviations:**

- **GK** — Gatekeeper (quality bar enforcement authority)
- **SR** — Showrunner (can defer TU, triggering gate suspension)

---

## 4. Transition Details

### 4.1 `pre-gate` → `gatecheck`

**Preconditions:**

- Pre-gate review completed
- Representative slice sampled (not full deliverables)
- No critical red flags identified
- Owner addressed any early concerns

**Sender:** GK (Gatekeeper)

**Intent:** `gate.submit`

**Payload Schema:** `gatecheck_report.schema.json` (partial)

**Required Fields:**

```json
{
  "title": "<TU ID or View name>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "<GK name or agent>",
  "scope": "<slice or export target>",
  "mode": "pre-gate",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "artifacts_samples": ["<sampled artifact paths>"],
  "decision": "pass",
  "why": "Pre-gate sampling shows no blockers; ready for full gatecheck",
  "next_actions": "Owner submit full deliverables for formal gatecheck"
}
```

**Effects:**

- Pre-gate checkpoint passed
- Owner submits full deliverables
- Gatekeeper schedules full bar validation

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `pre-gate` state
- `NOT_AUTHORIZED` — if sender is not GK
- `VALIDATION_FAILED` — if required pre-gate fields missing

---

### 4.2 `pre-gate` → `deferred`

**Preconditions:**

- TU deferred by Showrunner before completing gatecheck
- Deferral reason and revisit condition recorded

**Sender:** SR (Showrunner) or GK (Gatekeeper)

**Intent:** `gate.defer`

**Payload Schema:** `tu_brief.schema.json` (partial update)

**Required Fields:**

```json
{
  "id": "TU-YYYY-MM-DD-RRnn",
  "deferral_tags": ["deferred:art", "deferred:audio", etc.],
  "linkage": "<deferral reason and revisit condition>"
}
```

**Effects:**

- Gatecheck lifecycle suspended
- TU enters deferred state
- Gatekeeper notes remain for future reference

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if already in terminal decision state
- `NOT_AUTHORIZED` — if sender is not SR or GK
- `VALIDATION_FAILED` — if deferral reason missing

---

### 4.3 `gatecheck` → `decision:pass`

**Preconditions:**

- All deliverables reviewed
- All 8 bars evaluated
- All bars status = `green`
- No blocking issues identified

**Sender:** GK (Gatekeeper)

**Intent:** `gate.decision`

**Payload Schema:** `gatecheck_report.schema.json` (full)

**Payload Field:** `decision: "pass"`

**Required Fields:**

```json
{
  "title": "<TU ID or View name>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "<GK name or agent>",
  "scope": "<slice or export target>",
  "mode": "gatecheck",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "artifacts_samples": ["<reviewed artifact paths>"],
  "decision": "pass",
  "why": "All 8 bars green; approved for Cold merge",
  "next_actions": "SR merge to Cold",
  "bars": [
    {
      "bar": "Integrity",
      "status": "green",
      "evidence": "<player-safe evidence>"
    }
    // ... (all 8 bars with status "green")
  ],
  "handoffs": ["None required; all bars green"],
  "checklist": [
    "Decision: pass",
    "All bars validated green",
    "Evidence player-safe",
    "No dormancy issues",
    "Handoffs none",
    "Trace updated"
  ]
}
```

**Effects:**

- Gatecheck decision = `pass` (terminal)
- TU approved for Cold merge
- Showrunner can proceed with merge

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `gatecheck` state
- `NOT_AUTHORIZED` — if sender is not GK
- `VALIDATION_FAILED` — if any bar is not green
- `BUSINESS_RULE_VIOLATION` — if decision:pass but bars contain yellow/red

---

### 4.4 `gatecheck` → `decision:conditional-pass`

**Preconditions:**

- All deliverables reviewed
- All 8 bars evaluated
- Some bars status = `yellow` (but none `red`)
- Time-boxed remediation plan accepted by Showrunner

**Sender:** GK (Gatekeeper)

**Intent:** `gate.decision`

**Payload Schema:** `gatecheck_report.schema.json` (full)

**Payload Field:** `decision: "conditional_pass"`

**Required Fields:**

```json
{
  "title": "<TU ID or View name>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "<GK name or agent>",
  "scope": "<slice or export target>",
  "mode": "gatecheck",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "artifacts_samples": ["<reviewed artifact paths>"],
  "decision": "conditional pass",
  "why": "Most bars green; yellow bars have time-boxed remediation plan",
  "next_actions": "SR merge to Cold; schedule follow-up TU for yellow bars",
  "bars": [
    {
      "bar": "Style",
      "status": "yellow",
      "evidence": "<player-safe evidence of style drift>",
      "smallest_viable_fix": "<specific remediation step>",
      "owner": "ST",
      "notes": "Schedule Style Tune-up TU within 1 week"
    },
    {
      "bar": "Integrity",
      "status": "green",
      "evidence": "<player-safe evidence>"
    }
    // ... (remaining bars)
  ],
  "handoffs": ["Style (yellow) → ST: Style Tune-up TU by 2025-11-07"],
  "checklist": [
    "Decision: conditional pass",
    "Yellow bars remediation plan accepted",
    "Evidence player-safe",
    "Dormancy acknowledged",
    "Handoffs documented",
    "Trace updated"
  ]
}
```

**Effects:**

- Gatecheck decision = `conditional pass` (terminal)
- TU approved for Cold merge with conditions
- Follow-up TU(s) scheduled for yellow bar remediation
- Handoffs documented for responsible owners

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `gatecheck` state
- `NOT_AUTHORIZED` — if sender is not GK
- `VALIDATION_FAILED` — if any bar is red, or yellow bars lack smallest_viable_fix/owner
- `BUSINESS_RULE_VIOLATION` — if decision:conditional-pass but no yellow bars present

---

### 4.5 `gatecheck` → `decision:block`

**Preconditions:**

- All deliverables reviewed
- All 8 bars evaluated
- One or more bars status = `red`
- Remediation required before Cold merge

**Sender:** GK (Gatekeeper)

**Intent:** `gate.decision`

**Payload Schema:** `gatecheck_report.schema.json` (full)

**Payload Field:** `decision: "block"`

**Required Fields:**

```json
{
  "title": "<TU ID or View name>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "<GK name or agent>",
  "scope": "<slice or export target>",
  "mode": "gatecheck",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "artifacts_samples": ["<reviewed artifact paths>"],
  "decision": "block",
  "why": "One or more bars red; remediation required before merge",
  "next_actions": "Owner address red bars; resubmit for gatecheck",
  "bars": [
    {
      "bar": "Presentation",
      "status": "red",
      "evidence": "<player-safe evidence of spoiler leak>",
      "smallest_viable_fix": "Remove codeword reference from section 17 caption",
      "owner": "SS",
      "notes": "Critical: spoiler hygiene violation"
    },
    {
      "bar": "Integrity",
      "status": "green",
      "evidence": "<player-safe evidence>"
    }
    // ... (remaining bars)
  ],
  "handoffs": [
    "Presentation (red) → SS: Remove spoiler from section 17 caption; resubmit within 24h"
  ],
  "checklist": [
    "Decision: block",
    "Red bars must be green before merge",
    "Evidence player-safe",
    "Dormancy acknowledged",
    "Handoffs documented",
    "Trace updated"
  ]
}
```

**Effects:**

- Gatecheck decision = `block` (terminal)
- TU rejected for Cold merge
- Owner must address red bars
- New gatecheck required after remediation

**Error Cases:**

- `INVALID_STATE_TRANSITION` — if not in `gatecheck` state
- `NOT_AUTHORIZED` — if sender is not GK
- `VALIDATION_FAILED` — if red bars lack smallest_viable_fix/owner
- `BUSINESS_RULE_VIOLATION` — if decision:block but no red bars present

---

## 5. Quality Bar Mapping

### 5.1 The 8 Quality Bars (from QUALITY_BARS.md)

Every gatecheck MUST evaluate all 8 bars:

1. **Integrity** — References resolve; no unintended dead ends
2. **Reachability** — Critical beats reachable via at least one viable path
3. **Nonlinearity** — Hubs/loops/gateways deliberate and meaningful
4. **Gateways** — Conditions consistent, enforceable, spoiler-safe
5. **Style** — Voice, register, motifs hold; PN in-voice
6. **Determinism** — Promised assets reproducible from parameters
7. **Presentation** — Player-safe surfaces; no spoilers or internals
8. **Accessibility** — Navigation clear; alt text present; sensory considerations respected

### 5.2 Bar Status Meanings

- **Green** — Bar requirements fully met; no remediation needed
- **Yellow** — Minor issues present; smallest viable fix identified; can proceed with follow-up
- **Red** — Critical failure; must remediate before Cold merge

### 5.3 Schema Field Mapping

Each bar entry in `gatecheck_report.schema.json` includes:

| Field                 | Required When | Purpose                                     |
| --------------------- | ------------- | ------------------------------------------- |
| `bar`                 | Always        | Bar name (enum: 8 bars)                     |
| `status`              | Always        | green / yellow / red                        |
| `evidence`            | Always        | Player-safe observation supporting status   |
| `smallest_viable_fix` | yellow or red | Minimal remediation step                    |
| `owner`               | yellow or red | Responsible role (enum: role abbreviations) |
| `notes`               | Optional      | Additional context or reminders             |

### 5.4 Decision Logic

**Pass:**

- ALL bars = green
- `decision` = "pass"
- `handoffs` = ["None required; all bars green"]

**Conditional Pass:**

- Some bars = yellow (none red)
- `decision` = "conditional pass"
- `handoffs` = explicit list of yellow bar remediation plans with owners and due dates

**Block:**

- One or more bars = red
- `decision` = "block"
- `handoffs` = explicit list of red bar remediation plans with owners and deadlines

---

## 6. Envelope Context Requirements

All gatecheck lifecycle messages MUST include proper envelope context per `ENVELOPE.md`:

### 6.1 Hot/Cold Boundaries

- **Hot context** (`hot_cold: "hot"`):
  - Used for all gatecheck transitions
  - Gatecheck reports coordinate Hot→Cold gate, not player-facing
  - `safety.player_safe: false`, `safety.spoilers: "allowed"` (for internal gatecheck coordination)
  - **However:** All evidence in `bars[].evidence` fields MUST be player-safe (no spoilers)

- **Cold context** (`hot_cold: "cold"`):
  - Used when referencing completed gatechecks in exports
  - MUST include `snapshot` reference

### 6.2 TU Linkage

All gatecheck messages MUST include:

- `context.tu` — the TU ID being gatechecked
- `context.snapshot` — the Cold snapshot being validated
- `refs` — array of related TU/artifact IDs for traceability

### 6.3 Loop Context

All gatecheck messages MUST include:

- `context.loop` — "Gatecheck" (from loop_name enum)

---

## 7. Validation Rules

### 7.1 Bar Count Validation

**Rule:** All 8 bars MUST be present in `bars` array

- **Check:** `bars.length === 8` and all bar names from enum present
- **Error:** `VALIDATION_FAILED` if bar count ≠ 8 or bar name invalid

**Example error:**

```json
{
  "code": "VALIDATION_FAILED",
  "message": "Gatecheck report missing required bars",
  "details": {
    "expected_bars": 8,
    "actual_bars": 7,
    "missing_bars": ["Accessibility"]
  }
}
```

### 7.2 Decision-Bar Status Consistency

**Rule:** Decision must match bar status distribution

- **Pass:** ALL bars green
- **Conditional Pass:** Some yellow, none red
- **Block:** One or more red

**Check:** Validate decision against bar statuses **Error:** `BUSINESS_RULE_VIOLATION` if mismatch

**Example error:**

```json
{
  "code": "BUSINESS_RULE_VIOLATION",
  "message": "Decision 'pass' conflicts with bar status",
  "details": {
    "decision": "pass",
    "bars_with_issues": [{ "bar": "Style", "status": "yellow" }],
    "rule": "Pass decision requires all bars green"
  }
}
```

### 7.3 Evidence Player-Safety Validation

**Rule:** All `bars[].evidence` fields MUST be player-safe

- **Check:** Evidence contains no codewords, gate logic, or hidden plot details
- **Error:** `VALIDATION_FAILED` if spoilers detected

**Example error:**

```json
{
  "code": "VALIDATION_FAILED",
  "message": "Gatecheck evidence contains spoilers",
  "details": {
    "bar": "Presentation",
    "evidence_excerpt": "...codeword ASH reveals...",
    "violation": "Codeword name exposed in evidence",
    "remedy": "Rephrase evidence in player-safe terms"
  }
}
```

### 7.4 Smallest Viable Fix Validation

**Rule:** Yellow/red bars MUST include `smallest_viable_fix` and `owner`

- **Check:** If status yellow or red, both fields present
- **Error:** `VALIDATION_FAILED` if missing

**Example error:**

```json
{
  "code": "VALIDATION_FAILED",
  "message": "Yellow bar missing remediation fields",
  "details": {
    "bar": "Style",
    "status": "yellow",
    "missing_fields": ["smallest_viable_fix", "owner"],
    "rule": "Yellow/red bars require remediation plan"
  }
}
```

---

## 8. Handoff Requirements

### 8.1 Handoff Format

Each handoff in `handoffs` array MUST include:

- Bar name and status (yellow or red)
- Responsible role (owner)
- Remediation action (smallest_viable_fix summary)
- Due date or TU reference

**Example:**

```
"Style (yellow) → ST: Style Tune-up TU by 2025-11-07"
"Presentation (red) → SS: Remove spoiler from section 17; resubmit within 24h"
```

### 8.2 Escalation (Optional)

If bar failure triggers escalation:

- Use `escalation` object in gatecheck_report
- Specify `topic`, `lane`, `level` (L1/L2/L3), `bundle_attached`

**Example:**

```json
{
  "escalation": {
    "topic": "Style drift across 3 consecutive TUs",
    "lane": "ST",
    "level": "L2",
    "bundle_attached": "yes"
  }
}
```

---

## 9. Cross-References

### Layer 0/1 Policy

- `00-north-star/QUALITY_BARS.md` — 8 quality bar definitions and checks
- `00-north-star/SOURCES_OF_TRUTH.md` — Hot/Cold SoT boundaries and merge rules
- `01-roles/charters/gatekeeper.md` — GK authority and decision-making

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` §5 — Quality Bar Categories
- `02-dictionary/artifacts/gatecheck_report.md` — Gatecheck Report structure

### Layer 3 Schemas

- `03-schemas/gatecheck_report.schema.json` — Payload validation schema
- `03-schemas/tu_brief.schema.json` — TU Brief schema (for gate.defer)

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/LIFECYCLES/tu.md` — TU lifecycle (companion spec)
- `04-protocol/LIFECYCLES/hooks.md` — Hook lifecycle (companion spec)

---

## 10. Examples

### 10.1 Example: Pre-gate Pass

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:pregate-pass-001",
  "time": "2025-10-30T10:00:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-v2.1" },
  "receiver": { "role": "SR" },
  "intent": "gate.submit",
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
    "type": "gatecheck_report",
    "$schema": "../03-schemas/gatecheck_report.schema.json",
    "data": {
      "title": "TU-2025-10-30-SR01",
      "checked": "2025-10-30",
      "gatekeeper": "GK Bot v2.1",
      "scope": "Canon entry for toll mechanism (sampled)",
      "mode": "pre-gate",
      "cold_snapshot": "Cold @ 2025-10-28",
      "artifacts_samples": ["canon/toll_mechanism.md (excerpt)"],
      "decision": "pass",
      "why": "Pre-gate sampling shows no blockers; ready for full gatecheck",
      "next_actions": "Owner submit full deliverables for formal gatecheck",
      "bars": [],
      "handoffs": ["None"],
      "checklist": [
        "Pre-gate sampling complete",
        "No red flags identified",
        "Ready for full gatecheck"
      ]
    }
  },
  "refs": ["TU-2025-10-30-SR01"],
  "correlation_id": "corr-gatecheck-2025-10-30"
}
```

### 10.2 Example: Gatecheck Pass (All Bars Green)

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:gate-pass-001",
  "time": "2025-10-30T11:00:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-v2.1" },
  "receiver": { "role": "SR" },
  "intent": "gate.decision",
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
    "type": "gatecheck_report",
    "$schema": "../03-schemas/gatecheck_report.schema.json",
    "data": {
      "title": "TU-2025-10-30-SR01",
      "checked": "2025-10-30",
      "gatekeeper": "GK Bot v2.1",
      "scope": "Canon entry for toll mechanism (full review)",
      "mode": "gatecheck",
      "cold_snapshot": "Cold @ 2025-10-28",
      "artifacts_samples": ["canon/toll_mechanism.md", "canon/factions.md (updated)"],
      "decision": "pass",
      "why": "All 8 bars green; approved for Cold merge",
      "next_actions": "SR merge to Cold",
      "bars": [
        {
          "bar": "Integrity",
          "status": "green",
          "evidence": "All faction references resolve; no dead links"
        },
        {
          "bar": "Reachability",
          "status": "green",
          "evidence": "Toll mechanism reachable from at least 2 paths"
        },
        {
          "bar": "Nonlinearity",
          "status": "green",
          "evidence": "Toll creates meaningful choice pressure"
        },
        {
          "bar": "Gateways",
          "status": "green",
          "evidence": "Toll conditions clear and enforceable"
        },
        {
          "bar": "Style",
          "status": "green",
          "evidence": "Voice consistent with established register"
        },
        {
          "bar": "Determinism",
          "status": "green",
          "evidence": "No asset promises made (text-only)"
        },
        {
          "bar": "Presentation",
          "status": "green",
          "evidence": "No spoilers; all text player-safe"
        },
        {
          "bar": "Accessibility",
          "status": "green",
          "evidence": "Navigation clear; no sensory overload"
        }
      ],
      "handoffs": ["None required; all bars green"],
      "checklist": [
        "Decision: pass",
        "All bars validated green",
        "Evidence player-safe",
        "No dormancy issues",
        "Handoffs none",
        "Trace updated"
      ]
    }
  },
  "refs": ["TU-2025-10-30-SR01"],
  "correlation_id": "corr-gatecheck-2025-10-30"
}
```

### 10.3 Example: Gatecheck Conditional Pass (Yellow Bars)

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:gate-conditional-001",
  "time": "2025-10-30T11:30:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-v2.1" },
  "receiver": { "role": "SR" },
  "intent": "gate.decision",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-PW02",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Gatecheck"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "gatecheck_report",
    "$schema": "../03-schemas/gatecheck_report.schema.json",
    "data": {
      "title": "TU-2025-10-30-PW02",
      "checked": "2025-10-30",
      "gatekeeper": "GK Bot v2.1",
      "scope": "Topology expansion: 3 new hub sections",
      "mode": "gatecheck",
      "cold_snapshot": "Cold @ 2025-10-28",
      "artifacts_samples": [
        "manuscript/chapter_3/hub_17.md",
        "manuscript/chapter_3/hub_18.md",
        "manuscript/chapter_3/hub_19.md"
      ],
      "decision": "conditional pass",
      "why": "Most bars green; Style shows minor drift; approved with Style Tune-up TU",
      "next_actions": "SR merge to Cold; schedule Style Tune-up TU within 1 week",
      "bars": [
        {
          "bar": "Integrity",
          "status": "green",
          "evidence": "All hub links resolve"
        },
        {
          "bar": "Reachability",
          "status": "green",
          "evidence": "All hubs reachable from chapter start"
        },
        {
          "bar": "Nonlinearity",
          "status": "green",
          "evidence": "Hubs create meaningful branching"
        },
        {
          "bar": "Gateways",
          "status": "green",
          "evidence": "Gateway conditions consistent"
        },
        {
          "bar": "Style",
          "status": "yellow",
          "evidence": "Hub 18 uses informal register; others formal",
          "smallest_viable_fix": "Adjust hub_18.md register to match formal tone",
          "owner": "ST",
          "notes": "Schedule Style Tune-up TU within 1 week"
        },
        {
          "bar": "Determinism",
          "status": "green",
          "evidence": "No asset promises made"
        },
        {
          "bar": "Presentation",
          "status": "green",
          "evidence": "No spoilers in hub text"
        },
        {
          "bar": "Accessibility",
          "status": "green",
          "evidence": "Navigation clear; breadcrumbs present"
        }
      ],
      "handoffs": ["Style (yellow) → ST: Style Tune-up TU by 2025-11-07"],
      "checklist": [
        "Decision: conditional pass",
        "Yellow bars remediation plan accepted",
        "Evidence player-safe",
        "Dormancy acknowledged",
        "Handoffs documented",
        "Trace updated"
      ]
    }
  },
  "refs": ["TU-2025-10-30-PW02"],
  "correlation_id": "corr-gatecheck-2025-10-30"
}
```

### 10.4 Example: Gatecheck Block (Red Bars)

**Message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:gate-block-001",
  "time": "2025-10-30T12:00:00Z",
  "sender": { "role": "GK", "agent": "bot:gk-v2.1" },
  "receiver": { "role": "SR" },
  "intent": "gate.decision",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SS03",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Gatecheck"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "gatecheck_report",
    "$schema": "../03-schemas/gatecheck_report.schema.json",
    "data": {
      "title": "TU-2025-10-30-SS03",
      "checked": "2025-10-30",
      "gatekeeper": "GK Bot v2.1",
      "scope": "Prose revision: section 17",
      "mode": "gatecheck",
      "cold_snapshot": "Cold @ 2025-10-28",
      "artifacts_samples": ["manuscript/chapter_2/section_17.md"],
      "decision": "block",
      "why": "Presentation bar red: spoiler hygiene violation in caption",
      "next_actions": "SS remove codeword reference; resubmit for gatecheck within 24h",
      "bars": [
        {
          "bar": "Integrity",
          "status": "green",
          "evidence": "All references resolve"
        },
        {
          "bar": "Reachability",
          "status": "green",
          "evidence": "Section reachable from hub"
        },
        {
          "bar": "Nonlinearity",
          "status": "green",
          "evidence": "Section contributes to branching"
        },
        {
          "bar": "Gateways",
          "status": "green",
          "evidence": "No new gateways introduced"
        },
        {
          "bar": "Style",
          "status": "green",
          "evidence": "Voice consistent"
        },
        {
          "bar": "Determinism",
          "status": "green",
          "evidence": "No asset promises made"
        },
        {
          "bar": "Presentation",
          "status": "red",
          "evidence": "Caption line 34 exposes codeword name 'ASH'",
          "smallest_viable_fix": "Replace 'ASH' with in-world phrasing: 'the foreman's token'",
          "owner": "SS",
          "notes": "Critical: spoiler hygiene violation"
        },
        {
          "bar": "Accessibility",
          "status": "green",
          "evidence": "Navigation clear"
        }
      ],
      "handoffs": [
        "Presentation (red) → SS: Remove 'ASH' from section 17 caption line 34; resubmit within 24h"
      ],
      "checklist": [
        "Decision: block",
        "Red bars must be green before merge",
        "Evidence player-safe",
        "Dormancy acknowledged",
        "Handoffs documented",
        "Trace updated"
      ]
    }
  },
  "refs": ["TU-2025-10-30-SS03"],
  "correlation_id": "corr-gatecheck-2025-10-30"
}
```

---

## 11. Implementation Checklist

For implementers of Gatecheck lifecycle systems:

- [ ] Validate incoming messages against `ENVELOPE.md` requirements
- [ ] Verify `payload.data` validates against `gatecheck_report.schema.json`
- [ ] Check all 8 bars present in `bars` array
- [ ] Validate bar status consistency with decision
- [ ] Verify evidence fields are player-safe (no spoilers)
- [ ] Check yellow/red bars include smallest_viable_fix and owner
- [ ] Validate handoffs list matches yellow/red bar owners
- [ ] Generate appropriate error messages for violations
- [ ] Record gatecheck decision and trace linkage
- [ ] Emit state change events for downstream systems (TU lifecycle, Showrunner)

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group
