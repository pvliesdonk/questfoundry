# Hook Harvest Flow — Message Sequences

> **Status:** Normative — this document defines end-to-end message sequences for the Hook Harvest
> loop.

---

## 1. Overview

This specification defines the **message sequences** for the **Hook Harvest** loop: the
protocol-level interactions that enable roles to collect, cluster, triage, and route hooks for
follow-on work.

### Purpose

Hook Harvest transforms proposed hooks into actionable work items by:

1. **Creating hooks** from any role during creative work
2. **Triaging** by Showrunner during a Hook Harvest session
3. **Routing** accepted hooks to appropriate loops with clear ownership

### Design Principles

- **Explicit state transitions** — hook status changes driven by intents
- **TU-bound** — Hook Harvest triage occurs within a TU context
- **Schema-validated** — all payloads validated against `hook_card.schema.json`
- **Role authorization** — only Showrunner can accept/defer/reject
- **Clear ownership** — accepted hooks MUST specify owner and next loop

---

## 2. Message Sequence: Hook Creation

### 2.1 Context

**Trigger:** Any role identifies a hook during creative work (Story Spark, Lore Deepening, Scene
writing, etc.)

**Participants:**

- **Originator** — Any role (SR, LW, PW, SS, etc.)
- **Receiver** — Typically broadcast or specific owner role

### 2.2 Sequence

```
1. Originator → [Broadcast/Owner]: hook.create
   Intent: hook.create
   Payload: hook_card (status: "proposed")
   Context: hot_cold="hot", tu=<origin-TU>, snapshot=<Cold snapshot>

2. [Broadcast/Owner] → Originator: ack
   Confirms receipt and hook queued for triage
```

### 2.3 Required Payload Fields (hook.create)

**Schema:** `03-schemas/hook_card.schema.json`

**Minimum required fields:**

```json
{
  "header": {
    "short_name": "Brief hook title (3-80 chars)",
    "id": "HK-YYYYMMDD-seq",
    "status": "proposed",
    "raised_by": "<role>",
    "tu": "TU-YYYY-MM-DD-<role><seq>",
    "edited": "YYYY-MM-DD",
    "slice": "Player-safe scope description (10-200 chars)",
    "snapshot_context": "Cold @ YYYY-MM-DD"
  },
  "classification": {
    "type_primary": "narrative|scene|factual|taxonomy|...",
    "bars_affected": ["Integrity", "..."],
    "blocking": "no|yes (<reason>)"
  },
  "player_safe_summary": "Brief need description (10-240 chars, no spoilers)",
  "proposed_next_step": {
    "loop": "Lore Deepening|Story Spark|...",
    "owner_r": "<role>",
    "accountable_a": "SR",
    "consult": ["<role>", "..."]
  },
  "acceptance_criteria": ["Exit criterion tied to Quality Bars"],
  "locations_links": {
    "locations": ["path/to/affected/content"],
    "related_hooks": ["HK-YYYYMMDD-seq"]
  }
}
```

**Notes:**

- `status` MUST be `"proposed"` for new hooks
- `tu` MUST reference the TU that prompted the hook (traceability)
- `snapshot_context` MUST reference current Cold snapshot
- `player_safe_summary` MUST NOT contain spoilers or internal details

### 2.4 Error Conditions

| Error Code                | Trigger                                            | Remedy                      |
| ------------------------- | -------------------------------------------------- | --------------------------- |
| `VALIDATION_FAILED`       | Payload missing required fields or violates schema | Fix payload to match schema |
| `CONFLICT`                | Hook ID already exists                             | Use unique hook ID          |
| `BUSINESS_RULE_VIOLATION` | Hot content leaks to PN                            | Keep hook Hot-only          |

---

## 3. Message Sequence: Hook Harvest Triage

### 3.1 Context

**Trigger:** Showrunner runs Hook Harvest session (TU opened with loop="Hook Harvest")

**Participants:**

- **Showrunner (SR)** — Accountable for triage decisions
- **Hook Owners** — Roles that will execute accepted hooks
- **Consulted Roles** — LW, PW, SS, CC, RS, ST, GK (provide input)

### 3.2 Sequence Overview

```
1. SR → Broadcast: tu.open
   Opens Hook Harvest TU, wakes required roles

2. [Triage Session — synchronous or asynchronous discussion]
   SR + consulted roles review proposed hooks
   Cluster by theme, annotate with triage tags
   SR makes accept/defer/reject decisions

3. For each hook decision:
   a) SR → [Owner/Broadcast]: hook.update_status
      Intent: hook.accept | hook.defer | hook.reject
      Payload: hook_card (partial update with new status)

   b) [Owner/Broadcast] → SR: ack
      Confirms status update received
```

### 3.3 TU Context (tu.open for Hook Harvest)

**Intent:** `tu.open`

**Payload:** `tu_brief.schema.json`

**Required fields:**

```json
{
  "id": "TU-YYYY-MM-DD-SR<seq>",
  "opened": "YYYY-MM-DD",
  "owner_a": "SR",
  "responsible_r": ["SR", "LW", "PW", "..."],
  "loop": "Hook Harvest",
  "slice": "Triage hooks from <source> for <theme>",
  "snapshot_context": "Cold @ YYYY-MM-DD",
  "awake": ["SR", "LW", "PW", "SS", "CC", "GK"],
  "dormant": ["AD", "IL", "AuD", "AuP", "TR", "BB", "PN"],
  "press": ["Integrity"],
  "inputs": ["Hook cards in proposed state", "Prior harvest decisions for consistency"],
  "deliverables": [
    "Harvest Sheet (summary of decisions)",
    "Updated hook cards with status and triage tags"
  ],
  "bars_green": ["Integrity"],
  "timebox": "60 min",
  "gatecheck": "Quick integrity check on triage decisions",
  "linkage": "Hooks accepted/deferred/rejected; handoffs to loops"
}
```

---

## 4. Message Sequence: Hook Status Updates

### 4.1 Accept Hook (`proposed` → `accepted`)

**Intent:** `hook.accept`

**Sender:** SR

**Payload:** `hook_card.schema.json` (partial update)

**Required fields for acceptance:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "accepted",
    "edited": "YYYY-MM-DD"
  },
  "proposed_next_step": {
    "loop": "Lore Deepening", // REQUIRED: next loop
    "owner_r": "LW", // REQUIRED: responsible owner
    "accountable_a": "SR"
  },
  "acceptance_criteria": ["Updated criteria if needed"]
}
```

**Critical requirements for acceptance:**

- `proposed_next_step.loop` — MUST specify which loop handles this hook
- `proposed_next_step.owner_r` — MUST specify responsible role
- Hook MUST NOT be blocking unless resolution plan exists

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:abc123...",
  "time": "2025-10-30T14:00:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "LW" },
  "intent": "hook.accept",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-25",
    "loop": "Hook Harvest"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "hook_card",
    "$schema": "../03-schemas/hook_card.schema.json",
    "data": {
      "header": {
        "id": "HK-20251030-03",
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
  "refs": ["HK-20251030-03"]
}
```

---

### 4.2 Defer Hook (`proposed` → `deferred`)

**Intent:** `hook.defer`

**Sender:** SR

**Payload:** `hook_card.schema.json` (partial update)

**Required fields for deferral:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "deferred",
    "edited": "YYYY-MM-DD"
  },
  "dormancy_deferrals": {
    "deferral_tags": ["deferred:research"], // REQUIRED: reason
    "fallback": "What happens without this work", // REQUIRED
    "revisit": "Loop name or milestone/date" // REQUIRED: when to reconsider
  }
}
```

**Critical requirements for deferral:**

- `dormancy_deferrals.deferral_tags` — MUST specify why deferred
- `dormancy_deferrals.fallback` — MUST describe impact if never done
- `dormancy_deferrals.revisit` — MUST specify wake condition

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:def456...",
  "time": "2025-10-30T14:05:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "RS" },
  "intent": "hook.defer",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-25",
    "loop": "Hook Harvest"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "hook_card",
    "$schema": "../03-schemas/hook_card.schema.json",
    "data": {
      "header": {
        "id": "HK-20251030-07",
        "status": "deferred",
        "edited": "2025-10-30"
      },
      "dormancy_deferrals": {
        "deferral_tags": ["deferred:research"],
        "fallback": "Neutral phrasing used; no hard claims",
        "revisit": "When Researcher role wakes or Q1 2026"
      }
    }
  },
  "refs": ["HK-20251030-07"]
}
```

---

### 4.3 Reject Hook (`proposed` → `rejected`)

**Intent:** `hook.reject`

**Sender:** SR

**Payload:** `hook_card.schema.json` (partial update)

**Required fields for rejection:**

```json
{
  "header": {
    "id": "HK-YYYYMMDD-seq",
    "status": "rejected",
    "edited": "YYYY-MM-DD"
  },
  "resolution": {
    "decision": "One sentence rejection reason", // REQUIRED
    "resolved_date": "YYYY-MM-DD",
    "resolved_by": "SR"
  }
}
```

**Critical requirements for rejection:**

- `resolution.decision` — MUST provide 1-line reason
- Common reasons: duplicate of another hook, violates PN/style boundaries, creates unwinnable
  states, no viable path forward

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:ghi789...",
  "time": "2025-10-30T14:10:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "hook.reject",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-25",
    "loop": "Hook Harvest"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "hook_card",
    "$schema": "../03-schemas/hook_card.schema.json",
    "data": {
      "header": {
        "id": "HK-20251030-09",
        "status": "rejected",
        "edited": "2025-10-30"
      },
      "resolution": {
        "decision": "Duplicate of HK-20251028-12; linked for provenance",
        "resolved_date": "2025-10-30",
        "resolved_by": "SR"
      }
    }
  },
  "refs": ["HK-20251030-09", "HK-20251028-12"]
}
```

---

## 5. Complete Flow Example

### 5.1 Scenario: Hook Harvest Session

**Context:** Showrunner runs Hook Harvest after Story Spark session generated 15 hooks. Triage goal:
accept 8, defer 5, reject 2.

**Sequence:**

```
Step 1: SR opens Hook Harvest TU
--------
SR → Broadcast: tu.open
  id: TU-2025-10-30-SR01
  loop: Hook Harvest
  awake: [SR, LW, PW, SS, CC, GK]
  deliverables: [Harvest Sheet, Updated hooks]

Step 2: SR + roles review proposed hooks (synchronous/async)
--------
[Internal triage discussion using Hook Harvest procedure]
- Cluster hooks by theme
- Annotate with triage tags
- Make accept/defer/reject decisions

Step 3: SR sends status updates for each hook
--------
SR → LW: hook.accept (HK-20251030-01)
  next_step: Lore Deepening, owner_r: LW

SR → PW: hook.accept (HK-20251030-02)
  next_step: Story Spark, owner_r: PW

SR → RS: hook.defer (HK-20251030-05)
  reason: deferred:research, revisit: Q1 2026

SR → Broadcast: hook.reject (HK-20251030-09)
  decision: Duplicate of HK-20251028-12

[...repeat for remaining hooks...]

Step 4: Owners acknowledge updates
--------
LW → SR: ack (received hook.accept)
PW → SR: ack (received hook.accept)
RS → SR: ack (received hook.defer)

Step 5: SR closes Hook Harvest TU
--------
SR → Broadcast: tu.update
  deliverables: [
    Harvest Sheet with 8 accepted, 5 deferred, 2 rejected,
    Updated hook cards in Hot
  ]
  linkage: Handoffs to Lore Deepening (LW), Story Spark (PW), ...
```

---

## 6. Handoffs from Hook Harvest

After Hook Harvest completes, accepted hooks are handed to appropriate loops:

### 6.1 To Lore Deepening

**Hooks:** narrative, scene, factual hooks requiring canon answers

**Handoff:** Accepted hooks with `proposed_next_step.loop = "Lore Deepening"` and `owner_r = "LW"`

**Next message:** LW opens Lore Deepening TU (see `lore_deepening.md`)

### 6.2 To Story Spark

**Hooks:** structure/topology hooks that reshape narrative flow

**Handoff:** Accepted hooks with `proposed_next_step.loop = "Story Spark"` and `owner_r = "PW"`

### 6.3 To Codex Expansion

**Hooks:** taxonomy/clarity hooks for player-facing codex

**Handoff:** Accepted hooks with `proposed_next_step.loop = "Codex Expansion"` and `owner_r = "CC"`

### 6.4 To Style Tune-up

**Hooks:** tone/voice/aesthetic hooks

**Handoff:** Accepted hooks with `proposed_next_step.loop = "Style Tune-up"` and `owner_r = "ST"`

---

## 7. Error Conditions

| Error Code                 | Trigger                             | Example                      | Remedy                     |
| -------------------------- | ----------------------------------- | ---------------------------- | -------------------------- |
| `VALIDATION_FAILED`        | hook.accept missing owner_r or loop | Accept without owner         | Add required fields        |
| `INVALID_STATE_TRANSITION` | hook.accept on non-proposed hook    | Accept already-accepted hook | Check current status first |
| `NOT_AUTHORIZED`           | Non-SR sends hook.accept            | LW tries to accept hook      | Only SR can triage         |
| `BUSINESS_RULE_VIOLATION`  | Deferral without revisit            | Defer without wake condition | Add revisit field          |

---

## 8. Success Criteria

A Hook Harvest flow is successful when:

- ✅ All proposed hooks have status updated (accepted/deferred/rejected)
- ✅ Accepted hooks have clear `owner_r` and `next loop`
- ✅ Deferred hooks have `reason` and `revisit` wake condition
- ✅ Rejected hooks have 1-line `decision` rationale
- ✅ TU linkage maintained (all hooks reference origin TU)
- ✅ Harvest Sheet delivered summarizing decisions
- ✅ Handoffs to downstream loops documented in TU

---

## 9. Cross-References

### Layer 0/1 Policy

- `00-north-star/LOOPS/hook_harvest.md` — Loop procedure and RACI
- `00-north-star/QUALITY_BARS.md` — Quality bar definitions
- `00-north-star/TRACEABILITY.md` — TU requirements

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` — Hook status taxonomy §2

### Layer 3 Schemas

- `03-schemas/hook_card.schema.json` — Hook Card payload schema
- `03-schemas/tu_brief.schema.json` — TU Brief payload schema

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/INTENTS.md` — Intent catalog (hook.create, hook.accept, hook.defer, hook.reject)
- `04-protocol/LIFECYCLES/hooks.md` — Hook lifecycle state machine
- `04-protocol/LIFECYCLES/tu.md` — TU lifecycle state machine

---

## 10. Implementation Notes

### 10.1 Hook Collection Phase

Hook creation can happen:

- **During other loops** (Story Spark, Lore Deepening, Scene writing) — roles file hooks as they
  surface
- **Batch at loop end** — roles review work and file hooks before closing TU
- **Ad-hoc** — any role can create hooks anytime

All hooks start in `proposed` state and await Hook Harvest triage.

### 10.2 Triage Session Format

Hook Harvest can be:

- **Synchronous** — real-time meeting with SR + consulted roles
- **Asynchronous** — SR posts hooks, roles comment, SR decides
- **Hybrid** — async review + sync decision meeting

All formats use same message protocol: `hook.create` → `hook.accept/defer/reject`

### 10.3 Harvest Sheet

The Harvest Sheet is a human-readable summary (not a structured message) that includes:

- Date & TU-ID
- Cluster headings (by theme or type)
- **Accepted** list (with next loop + owner + due window)
- **Deferred** list (reason + wake condition)
- **Rejected** list (reason + duplicate reference if applicable)
- **Risk notes** (dormant roles, style pressure, topology churn)
- **Activation requests** (which roles SR should wake for next loops)

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group
