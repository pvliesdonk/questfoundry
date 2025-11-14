---
procedure_id: role_wake_dormancy
description: Managing role activation states for resource optimization
version: 2.0.0
references_expertises:
  - showrunner_orchestration
references_schemas:
  - message_envelope.schema.json
references_roles:
  - showrunner
  - all
tags:
  - coordination
  - resource-management
  - protocol
---

# Role Wake & Dormancy Management Procedure

## Overview

Control role activation states to optimize resource usage and context window management. Showrunner wakes roles when needed and parks them when inactive.

## Role States

**Active:** Role is awake, participating in current loop, receiving messages
**Dormant:** Role is parked, not receiving messages, can be woken when criteria met
**Blocked:** Role is awake but waiting on dependency or human input

## Step 1: Assess Wake Criteria

Determine if role should be activated.

**Wake triggers per role:**

### Always-Active Roles

- **Showrunner:** Never dormant (orchestrator)
- **Gatekeeper:** Active for all gatechecks

### Content Roles (Wake on Loop Activation)

- **Lore Weaver:** Wake for Lore Deepening, Hook Harvest (consulted)
- **Plotwright:** Wake for Story Spark, topology changes
- **Scene Smith:** Wake for Story Spark, Scene Forge, Style Tune-up
- **Codex Curator:** Wake for Codex Expansion
- **Style Lead:** Wake for Style Tune-up, major register questions

### Support Roles (Wake on Demand)

- **Researcher:** Wake only for high-stakes fact checking (dormant by default)
- **Translator:** Wake only for Translation Pass (dormant by default)

### Asset Roles (Wake for Asset Loops)

- **Art Director:** Wake for Art Touch-up
- **Illustrator:** Wake when AD provides shotlist
- **Audio Director:** Wake for Audio Pass
- **Audio Producer:** Wake when AuD provides cuelist

### Runtime Roles (Wake for Export/Testing)

- **Book Binder:** Wake for Binding Run, Archive Snapshot
- **Player Narrator:** Wake for Narration Dry-Run

**Decision:**

- If role's wake criteria met → proceed to Step 2
- If not needed → keep dormant

## Step 2: Prepare Wake Context

Gather information needed by waking role.

**Context to provide:**

- **TU Brief:** What work is happening
- **Loop name:** Which playbook executing
- **Deliverables:** What role is responsible for
- **Inputs:** What artifacts/context available
- **Handoffs:** Who they'll coordinate with
- **Quality bars:** Which bars to focus on

**Example:**

```yaml
wake_context:
  tu: "TU-2025-11-06-LW01"
  loop: "lore_deepening"
  role_assignment: "responsible"
  deliverables:
    - "Canon Pack for accepted hooks"
    - "Player-safe summaries for Codex"
  inputs:
    - "Accepted hooks from Hook Harvest"
    - "Existing Cold canon for continuity check"
  coordination:
    - "Consult with Researcher if high-stakes claims"
    - "Coordinate with Plotwright on topology impacts"
  quality_focus: ["integrity", "presentation"]
```

## Step 3: Emit `role.wake` Intent

Send activation message to role.

**Protocol envelope:**

```json
{
  "protocol": "questfoundry/1.0.0",
  "id": "msg-20251106-103000-sr123",
  "time": "2025-11-06T10:30:00Z",
  "sender": "SR",
  "receiver": "LW",
  "intent": "role.wake",
  "context": {
    "tu": "TU-2025-11-06-LW01",
    "loop": "lore_deepening"
  },
  "payload": {
    "type": "wake_directive",
    "data": {
      "reason": "Lore Deepening loop activated, canon work needed",
      "tu_brief": { /* TU brief object */ },
      "deliverables": ["Canon Pack", "Player-safe summaries"],
      "estimated_duration": "2-3 hours",
      "handoff_roles": ["researcher", "codex_curator", "plotwright"]
    }
  }
}
```

**Required fields:**

- `intent = "role.wake"`
- `receiver` = role abbreviation to wake
- `payload.data.reason` - Why waking this role
- `payload.data.tu_brief` - Work context

**Role response:**

- Acknowledge with `ack` intent
- Begin work immediately
- Reference TU in all subsequent messages

## Step 4: Monitor Role Activity

Track role participation during loop.

**Activity indicators:**

- Emits `tu.checkpoint` regularly
- Produces artifacts on schedule
- Responds to coordination requests
- Escalates blockers promptly

**Inactivity indicators:**

- No checkpoints for extended period
- Stalled on blocker without escalation
- Deliverables overdue

**Actions:**

- If active: continue monitoring
- If inactive: check for blocker, offer help
- If complete: proceed to dormancy (Step 6)

## Step 5: Handle Mid-Loop Wake Requests

Sometimes roles request additional specialist wakes.

**Trigger:** Role sends `role.wake` request to SR

**Example:**

```json
{
  "sender": "LW",
  "receiver": "SR",
  "intent": "role.wake",
  "payload": {
    "type": "wake_request",
    "data": {
      "role_to_wake": "researcher",
      "reason": "High-stakes medical claim requires fact checking",
      "urgency": "blocking",
      "context": "Kestrel's injury recovery timeline needs validation"
    }
  }
}
```

**SR actions:**

1. Assess request validity
2. If approved, wake requested role (Step 3)
3. Provide context from requesting role
4. Coordinate handoff

## Step 6: Assess Dormancy Criteria

Determine when to park role.

**Dormancy triggers:**

### Work Complete

- All deliverables produced
- Artifacts validated and handed off
- No pending coordination

### Loop Ended

- TU closed
- No immediate follow-up work
- Graceful degradation acceptable

### Resource Optimization

- Context window pressure
- Role not needed for several loops
- Can be re-woken when needed

**Do NOT set dormant if:**

- Deliverables incomplete
- Blocking another role
- Human question pending answer
- Artifacts awaiting validation

## Step 7: Prepare Dormancy Handoff

Capture role contributions before parking.

**Actions:**

1. Request final `tu.checkpoint` from role
2. Verify all deliverables handed off
3. Document revisit criteria (when to wake again)
4. Archive role's session state

**Example checkpoint:**

```yaml
final_checkpoint:
  role: "lore_weaver"
  tu: "TU-2025-11-06-LW01"
  summary: "Canonized all accepted hooks. Delivered Canon Packs and player-safe summaries to CC. Topology impacts coordinated with PW."
  deliverables_complete: true
  handoffs:
    - to: "codex_curator"
      artifacts: ["player_safe_summaries.json"]
    - to: "plotwright"
      artifacts: ["topology_notes.md"]
  revisit_criteria: "Wake for next Lore Deepening or if canon conflicts arise"
```

## Step 8: Emit `role.dormant` Intent

Send dormancy message to role.

**Protocol envelope:**

```json
{
  "protocol": "questfoundry/1.0.0",
  "id": "msg-20251106-133000-sr456",
  "time": "2025-11-06T13:30:00Z",
  "sender": "SR",
  "receiver": "LW",
  "intent": "role.dormant",
  "context": {
    "tu": "TU-2025-11-06-LW01"
  },
  "payload": {
    "type": "dormancy_directive",
    "data": {
      "reason": "Lore Deepening complete, no immediate canon work",
      "session_summary": "Successfully canonized 4 hooks with full continuity checks",
      "revisit_criteria": "Next Lore Deepening TU or canon conflict resolution needed",
      "acknowledgment_required": true
    }
  }
}
```

**Role response:**

- Acknowledge with `ack`
- Stop monitoring for new work
- Archive session state
- Enter dormant mode

## Step 9: Manage Dormant Roles

Handle roles while parked.

**Do NOT:**

- Send routine messages to dormant roles
- Include dormant roles in broadcasts (except critical)
- Request work from dormant roles

**DO:**

- Monitor for wake criteria
- Keep dormancy reasons documented
- Re-wake promptly when needed

**If work arises for dormant role:**

1. Check if work meets wake criteria
2. If yes, wake role (Step 3)
3. If no, defer work or assign to active role

## Graceful Degradation

Some roles can remain dormant with acceptable impacts.

**Researcher dormant:**

- Mark claims `uncorroborated:<risk>`
- Use neutral phrasing
- Note revisit criteria
- **Impact:** Factual uncertainty documented but non-blocking

**Translator dormant:**

- Default to source language only
- Note localization deferred
- **Impact:** Single-language release, translation pass later

**Asset roles dormant:**

- Plan-only asset work
- Defer rendering to future
- **Impact:** Story complete, assets added later

## Common Patterns

### Standard Loop Wake

```
SR: Opens TU, identifies roles needed
SR: Wakes LW, SS, PW for Story Spark
Roles: Acknowledge and begin work
SR: Monitors progress via checkpoints
SR: Sets roles dormant after TU close
```

### Mid-Loop Specialist Wake

```
LW: Discovers high-stakes medical claim
LW: Requests Researcher wake
SR: Approves, wakes Researcher
RE: Provides fact-check memo
LW: Incorporates findings
SR: Sets Researcher dormant after handoff
```

### Context Pressure Dormancy

```
SR: Notes context window filling
SR: Identifies roles with completed work
SR: Requests final checkpoints
SR: Sets dormant, archives state
[Context freed for remaining roles]
```

## Coordination with TU Lifecycle

Role wake/dormancy aligns with TU lifecycle:

**TU Open:** Wake responsible and consulted roles
**TU Active:** Monitor, handle mid-loop wakes
**TU Close:** Set roles dormant, archive state
**Between TUs:** Most roles dormant except SR, GK

## Summary Checklist

**Waking a role:**

- [ ] Wake criteria met
- [ ] Context prepared (TU brief, deliverables)
- [ ] `role.wake` intent sent with reason
- [ ] Role acknowledges and begins work
- [ ] Activity monitored

**Setting dormant:**

- [ ] Work complete or loop ended
- [ ] Final checkpoint captured
- [ ] All deliverables handed off
- [ ] Revisit criteria documented
- [ ] `role.dormant` intent sent
- [ ] Role acknowledges

**Resource optimization through strategic wake/dormancy improves context management and reduces cognitive load.**
