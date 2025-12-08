# Hook Harvest Loop

> **Goal:** Collect, cluster, and triage hooks into a prioritized set for downstream loops.

The **Hook Harvest** loop handles the triage phase after content creation or discovery. It transforms a backlog of proposed hooks into a prioritized, tagged set ready for canonization, topology changes, or style refinement.

:::{loop-meta}
id: hook_harvest
name: "Hook Harvest"
trigger: backlog_review
entry_point: showrunner
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

## Execution Graph

### Graph Nodes

#### Showrunner Node

The entry point that runs the harvest session and makes triage decisions.

:::{graph-node}
id: showrunner
role: showrunner
timeout: 600
max_iterations: 10
:::

#### Lorekeeper Node

Flags canon collisions and opportunities; suggests deepening order.

:::{graph-node}
id: lorekeeper
role: lorekeeper
timeout: 300
max_iterations: 5
:::

#### Plotwright Node

Judges structural impact and identifies gateway implications.

:::{graph-node}
id: plotwright
role: plotwright
timeout: 300
max_iterations: 5
:::

#### Gatekeeper Node

Points out quality bars likely to fail if a hook advances.

:::{graph-node}
id: gatekeeper
role: gatekeeper
timeout: 300
max_iterations: 3
:::

### Graph Edges

#### From Showrunner

:::{graph-edge}
source: showrunner
target: lorekeeper
condition: "intent.status == 'needs_canon_check'"
:::

:::{graph-edge}
source: showrunner
target: plotwright
condition: "intent.status == 'needs_structure_check'"
:::

:::{graph-edge}
source: showrunner
target: gatekeeper
condition: "intent.status == 'harvest_complete'"
:::

:::{graph-edge}
source: showrunner
target: END
condition: "intent.type == 'terminate'"
:::

#### From Lorekeeper

:::{graph-edge}
source: lorekeeper
target: showrunner
condition: "intent.status == 'canon_assessed'"
:::

:::{graph-edge}
source: lorekeeper
target: showrunner
condition: "intent.type == 'escalation'"
:::

#### From Plotwright

:::{graph-edge}
source: plotwright
target: showrunner
condition: "intent.status == 'structure_assessed'"
:::

:::{graph-edge}
source: plotwright
target: showrunner
condition: "intent.type == 'escalation'"
:::

#### From Gatekeeper

:::{graph-edge}
source: gatekeeper
target: showrunner
condition: "intent.status == 'passed'"
:::

:::{graph-edge}
source: gatekeeper
target: showrunner
condition: "intent.status == 'failed'"
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
    ↓
[Showrunner] → collects and clusters hooks
    ↓ (if canon implications)
[Lorekeeper] → assesses canon impact
    ↓ (if structure implications)
[Plotwright] → assesses topology impact
    ↓
[Showrunner] → triages and decides
    ↓
[Gatekeeper] → validates harvest sheet
    ↓
[Showrunner] → hands off to downstream loops
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
