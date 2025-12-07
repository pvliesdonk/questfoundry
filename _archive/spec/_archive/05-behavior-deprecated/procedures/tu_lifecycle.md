---
procedure_id: tu_lifecycle
description: Trace Unit lifecycle management from opening to closure
version: 2.0.0
references_expertises:
  - showrunner_orchestration
references_schemas:
  - tu_brief.schema.json
references_roles:
  - showrunner
  - all
tags:
  - traceability
  - workflow
  - coordination
---

# TU Lifecycle Management Procedure

## Overview

Trace Units (TUs) are the fundamental units of traceable work in QuestFoundry. This procedure defines the complete lifecycle from opening to closure.

## TU Naming Convention

**Format:** `TU-YYYY-MM-DD-<ROLE><NN>`

**Components:**

- `YYYY-MM-DD`: Date of TU creation
- `<ROLE>`: Two-letter abbreviation of responsible role
- `<NN>`: Sequential number for that role/date

**Examples:**

- `TU-2025-11-06-LW01`: First Lore Weaver TU on Nov 6, 2025
- `TU-2025-11-06-SS02`: Second Scene Smith TU on Nov 6, 2025
- `TU-2025-11-07-PW01`: First Plotwright TU on Nov 7, 2025

## Step 1: Open TU

Initiate a new trace unit for upcoming work.

**Trigger:**

- Loop activation (e.g., Lore Deepening starts)
- Customer directive requiring new work
- Discovery of work requiring traceability

**Responsible:** Showrunner (or responsible role with Showrunner approval)

**Actions:**

1. **Generate TU ID:** Using naming convention above

2. **Create TU Brief:**

   ```yaml
   id: "TU-2025-11-06-LW01"
   loop: "Lore Deepening"
   responsible_r: ["lore_weaver"]
   accountable_a: ["showrunner"]
   consulted_c: ["researcher", "plotwright"]
   informed_i: ["codex_curator"]

   inputs:
     - "HK-20251028-03 (Kestrel jaw scar)"
     - "HK-20251028-04 (Dock 7 fire history)"

   deliverables:
     - "Canon Pack: Kestrel backstory"
     - "Player-safe summary for Codex"
     - "Scene callbacks for downstream roles"

   scope:
     description: "Canonize accepted hooks related to Kestrel backstory and dock history"
     constraints:
       - "Maintain timeline consistency with existing canon"
       - "Coordinate with Plotwright on topology impacts"

   quality_bars_focus: ["integrity", "gateways", "presentation"]
   ```

3. **Broadcast `tu.open` Intent:**

   ```json
   {
     "intent": "tu.open",
     "sender": "SR",
     "receiver": "broadcast",
     "context": {
       "loop": "lore_deepening",
       "tu": "TU-2025-11-06-LW01"
     },
     "payload": {
       "type": "tu_brief",
       "data": { /* TU Brief from step 2 */ }
     }
   }
   ```

4. **Wake Required Roles:**
   - Send `role.wake` to responsible roles
   - Provide TU context and deliverables
   - Set expectations for checkpoints

**Output:** Active TU with all roles briefed

## Step 2: Track Progress with Checkpoints

Maintain visibility into work status during execution.

**Frequency:**

- After completing major sub-steps
- When blocked or needing coordination
- At natural workflow transitions
- Minimum once per session for long-running TUs

**Responsible:** Role performing the work

**Actions:**

1. **Emit `tu.checkpoint` Intent:**

   ```json
   {
     "intent": "tu.checkpoint",
     "sender": "LW",
     "receiver": "SR",
     "context": {
       "loop": "lore_deepening",
       "tu": "TU-2025-11-06-LW01",
       "correlation_id": "msg-original-tu-open"
     },
     "payload": {
       "summary": "Completed Steps 1-4 of canonization. Kestrel backstory drafted with timeline anchors. Identified topology impact: new guild location needed.",
       "completed": ["analyze_hooks", "draft_canon", "add_structure", "enumerate_impacts"],
       "next_actions": ["continuity_check", "coordinate_plotwright"],
       "blockers": [],
       "artifacts_produced": ["canon_pack_draft_v1.json"]
     }
   }
   ```

2. **Update TU State:**
   - Track which deliverables are in-progress vs complete
   - Note any scope changes or discoveries
   - Flag blockers requiring intervention

**Output:** Progress visibility for Showrunner and team

## Step 3: Handle Updates and Scope Changes

Adapt TU as work progresses and discoveries emerge.

**Trigger:**

- Scope expansion discovered
- New dependencies identified
- Blockers requiring replanning

**Responsible:** Showrunner (with responsible role input)

**Actions:**

1. **Emit `tu.update` Intent:**

   ```json
   {
     "intent": "tu.update",
     "sender": "SR",
     "receiver": "LW",
     "context": {
       "loop": "lore_deepening",
       "tu": "TU-2025-11-06-LW01"
     },
     "payload": {
       "updates": {
         "deliverables": [
           "Canon Pack: Kestrel backstory",
           "Player-safe summary for Codex",
           "Scene callbacks for downstream roles",
           "NEW: Topology notes for guild location (coordinate with PW)"
         ],
         "consulted_c": ["researcher", "plotwright"],
         "scope_change_reason": "Discovered guild location canonical detail needed for backstory coherence"
       }
     }
   }
   ```

2. **Coordinate Role Changes:**
   - Wake additional roles if needed
   - Update RACI assignments
   - Adjust timeline if necessary

**Output:** Updated TU Brief reflecting current scope

## Step 4: Pre-Close Review

Verify completeness before closing TU.

**Trigger:** Responsible role believes work is complete

**Responsible:** Showrunner

**Checklist:**

- [ ] All deliverables produced
- [ ] Artifacts validated (schema + quality)
- [ ] Gatecheck passed (if applicable)
- [ ] Downstream handoffs documented
- [ ] No unresolved blockers
- [ ] Traceability complete (lineage, sources)

**If incomplete:**

- Identify gaps
- Emit `tu.update` with remaining work
- Continue execution

**If complete:**

- Proceed to Step 5

## Step 5: Close TU

Formally complete the trace unit and archive results.

**Responsible:** Showrunner

**Actions:**

1. **Emit `tu.close` Intent:**

   ```json
   {
     "intent": "tu.close",
     "sender": "SR",
     "receiver": "broadcast",
     "context": {
       "loop": "lore_deepening",
       "tu": "TU-2025-11-06-LW01"
     },
     "payload": {
       "status": "completed",
       "summary": "Successfully canonized Kestrel backstory and dock history. All artifacts validated and gatechecked. Player-safe summaries delivered to Codex Curator. Topology impacts coordinated with Plotwright.",
       "deliverables_completed": [
         "canon_pack_kestrel_v1.json (validated, gatechecked, merged to Cold)",
         "canon_pack_dock_v1.json (validated, gatechecked, merged to Cold)",
         "player_safe_summaries.json (delivered to CC)",
         "topology_notes.md (delivered to PW)"
       ],
       "artifacts_merged_to_cold": [
         "cold/canon/kestrel_backstory.json",
         "cold/canon/dock_seven_history.json"
       ],
       "follow_up_work": [
         "Codex Expansion TU needed for publishing summaries",
         "Story Spark mini-TU if guild location requires new sections"
       ],
       "lessons_learned": "Coordinate with Plotwright earlier when canon implies new locations"
     }
   }
   ```

2. **Archive TU:**
   - Store TU Brief with final state
   - Link all produced artifacts
   - Capture checkpoint history
   - Document lessons learned

3. **Set Roles Dormant (if appropriate):**
   - If no immediate follow-up work
   - Emit `role.dormant` to roles no longer needed
   - Document revisit criteria

**Output:** Closed TU with complete traceability

## Step 6: Post-TU Actions

Handle follow-up work identified during TU.

**Actions:**

1. **Create Follow-Up TUs:**
   - For deferred work
   - For scope that expanded beyond original TU
   - For discovered opportunities

2. **Update Project State:**
   - Merge artifacts to Cold (if gatechecked)
   - Update manifests
   - Notify affected roles

3. **Feed Process Improvements:**
   - Capture lessons learned
   - Note coordination patterns that worked well
   - Identify pain points for future mitigation

**Output:** Clean handoff to next work phase

## TU States

**Active:** TU open, work in progress
**Checkpointed:** Partial progress reported
**Blocked:** Waiting on external input or decision
**Completed:** All deliverables done, gatecheck passed
**Closed:** Formally closed, archived
**Deferred:** Work postponed, TU paused

## Context Management

All messages during active TU should include:

```json
{
  "context": {
    "tu": "TU-2025-11-06-LW01",
    "loop": "lore_deepening",
    "hot_cold": "hot"  // or "cold" if delivering to PN
  }
}
```

This ensures traceability and proper routing.

## Memory Management

For long-running TUs approaching token limits:

1. **Summarize older turns:**
   - Create compact state note
   - Preserve objectives, constraints, decisions
   - Keep only critical raw quotes

2. **Emit frequent checkpoints:**
   - Offload history to checkpoint messages
   - Showrunner can reconstruct state if needed

3. **Break into sub-TUs if necessary:**
   - Large scope may need multiple TUs
   - Each TU manageable in context window

## Escalation Triggers

**Wake Showrunner:**

- TU blocked with no clear resolution
- Scope expansion requires approval
- Quality issues preventing closure

**Ask Human:**

- Ambiguous deliverable requirements
- Trade-offs affecting timeline or quality
- Creative decisions blocking progress

## Summary Checklist

- [ ] TU opened with clear brief and RACI
- [ ] Roles woke and briefed on context
- [ ] Regular checkpoints throughout work
- [ ] Scope updates handled systematically
- [ ] Pre-close review completed
- [ ] All deliverables validated
- [ ] TU closed with complete archive
- [ ] Follow-up work identified and planned

**TU lifecycle ensures complete traceability from customer intent to Cold artifact.**
