# Hook Harvest Loop

> **Goal:** Collect, cluster, and triage hooks into a prioritized set for downstream loops.

The **Hook Harvest** loop handles the triage phase after content creation or discovery. It transforms a backlog of proposed hooks into a prioritized, tagged set ready for canonization, topology changes, or style refinement.

:::{loop-meta}
id: hook_harvest
name: "Hook Harvest"
trigger: backlog_review
entry_point: showrunner
version: 1
:::

## Guidance

This section provides operational context for executing the Hook Harvest workflow.

### When to Trigger

Invoke the Hook Harvest loop when:

- **Post-creation cleanup**: After Story Spark or any drafting burst that produced hooks
- **Pre-stabilization**: Before a merge train or content freeze
- **Backlog drift**: Hook backlog looks fuzzy, duplicated, or uncategorized
- **Capacity planning**: Need to decide what to work on next

Do NOT invoke when:

- Hooks are already triaged and assigned (just execute them)
- Only one or two hooks exist (handle inline)
- Hooks require immediate action (skip triage, escalate)

### Success Criteria

The loop succeeds when:

- [ ] All proposed hooks are deduplicated and clustered by theme
- [ ] Each hook has a triage tag (quick-win, needs-research, structure-impact, etc.)
- [ ] Each hook is marked accepted, deferred, or rejected with reasoning
- [ ] Accepted hooks have assigned next loop and owner
- [ ] Harvest Sheet summarizes decisions for handoff

### Common Failure Modes

**Foggy clusters**

- Symptom: Clusters don't make sense or overlap heavily
- Fix: Recut by player value instead of source role
- Prevention: Start with user-facing themes, not internal categories

**Endless acceptance**

- Symptom: Everything gets accepted, backlog grows
- Fix: Enforce capacity limits; defer with explicit wake conditions
- Prevention: Set acceptance budget before starting

**Duplicate confusion**

- Symptom: Same idea appears multiple times with slight variations
- Fix: Link provenance and close duplicates
- Prevention: Check for similar hooks before creating new ones

**Missing uncertainty tags**

- Symptom: Factual hooks lack research posture
- Fix: Add `uncorroborated:<risk>` if Researcher dormant
- Prevention: Always tag uncertainty for factual claims

**Premature rejection**

- Symptom: Good ideas rejected without exploration
- Fix: Defer instead of reject; add wake conditions
- Prevention: Reject only for clear violations, not uncertainty

## Loop Participants

The roles that participate in this loop with their operational parameters.

:::{loop-participants}
showrunner:
  timeout: 600
  max_iterations: 10
lorekeeper:
  timeout: 300
  max_iterations: 5
plotwright:
  timeout: 300
  max_iterations: 5
gatekeeper:
  timeout: 300
  max_iterations: 3
:::

## Routing Rules

Decision table for Showrunner: after a role completes work, these rules
describe when to delegate to the next role.

### After Showrunner

:::{routing-rule}
after: showrunner
when: needs_canon_check
delegate_to: lorekeeper
description: Hooks have canon implications; LK assesses impact
:::

:::{routing-rule}
after: showrunner
when: needs_structure_check
delegate_to: plotwright
description: Hooks have topology implications; PW assesses impact
:::

:::{routing-rule}
after: showrunner
when: harvest_complete
delegate_to: gatekeeper
description: Triage done; GK validates harvest sheet
:::

:::{routing-rule}
after: showrunner
when: terminate
delegate_to: END
description: Harvest complete; end the loop
:::

### After Lorekeeper

:::{routing-rule}
after: lorekeeper
when: canon_assessed
delegate_to: showrunner
description: Canon impact assessed; SR continues triage
:::

:::{routing-rule}
after: lorekeeper
when: escalation
delegate_to: showrunner
description: LK encounters issue requiring SR decision
:::

### After Plotwright

:::{routing-rule}
after: plotwright
when: structure_assessed
delegate_to: showrunner
description: Structure impact assessed; SR continues triage
:::

:::{routing-rule}
after: plotwright
when: escalation
delegate_to: showrunner
description: PW encounters issue requiring SR decision
:::

### After Gatekeeper

:::{routing-rule}
after: gatekeeper
when: passed
delegate_to: showrunner
description: Harvest sheet validated; SR hands off to downstream
:::

:::{routing-rule}
after: gatekeeper
when: failed
delegate_to: showrunner
description: Validation issues found; SR reviews and adjusts
:::

## Quality Gates

### Post-Harvest Validation

:::{quality-gate}
before: END
role: gatekeeper
bars:

- integrity
blocking: false
:::

## Expected Flow

```text
Backlog Review Request
    |
[Showrunner] -> collects and clusters hooks
    | (if canon implications)
[Lorekeeper] -> assesses canon impact
    | (if structure implications)
[Plotwright] -> assesses topology impact
    |
[Showrunner] -> triages and decides
    |
[Gatekeeper] -> validates harvest sheet
    |
[Showrunner] -> hands off to downstream loops
```

## Artifacts Produced

- **HookCard** (updated): Status changed to accepted/deferred/rejected with tags
- **Harvest Sheet**: Summary of decisions with next loops and owners
- **Risk Notes**: Quality bar concerns for accepted hooks

## Handoffs

After Hook Harvest, accepted hooks flow to:

- **Lore Deepening**: Hooks requiring canonization (narrative, factual)
- **Story Spark**: Hooks that reshape topology
- **Codex Expansion**: Taxonomy and clarity hooks
- **Scene Weave**: Hooks that need prose work
