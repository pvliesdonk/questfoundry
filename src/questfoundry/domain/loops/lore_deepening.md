# Lore Deepening Loop

> **Goal:** Transform accepted hooks into coherent, contradiction-aware canon.

The **Lore Deepening** loop handles the canonization phase where accepted hooks become verified world facts. It produces canonical entries with timeline anchors, invariants, and downstream handoff notes while keeping spoilers safely quarantined.

:::{loop-meta}
id: lore_deepening
name: "Lore Deepening"
trigger: hooks_accepted
entry_point: showrunner
version: 1
:::

## Guidance

This section provides operational context for executing the Lore Deepening workflow.

### When to Trigger

Invoke the Lore Deepening loop when:

- **Post-harvest**: Hook Harvest marked hooks as `accepted` requiring canonization
- **Causal backfill**: Plotwright or Scene Smith need world facts to proceed
- **Contradiction resolution**: Conflicting canon needs adjudication
- **World expansion**: New story areas need foundational lore

Do NOT invoke when:

- Hooks are purely structural (use Story Spark)
- Hooks are purely stylistic (use Scene Weave with Creative Director)
- Content is already canon (just reference it)
- Only player-safe surfaces needed (use Codex Expansion)

### Success Criteria

The loop succeeds when:

- [ ] Each accepted hook is canonized, deferred, or rejected with rationale
- [ ] Canon entries resolve prior contradictions or mark deliberate mysteries
- [ ] Timeline anchors are consistent with existing chronology
- [ ] Invariants don't contradict established world rules
- [ ] Spoilers are separated from player-safe summaries
- [ ] Downstream effects are enumerated for Plotwright, Scene Smith, and Codex

### Common Failure Modes

**Canon sprawl**

- Symptom: Too many interconnected changes, hard to validate
- Fix: Split into smaller themed entries; stage merges
- Prevention: Scope entries to single themes; defer tangents

**Hidden spoiler leak**

- Symptom: Sensitive details appear in player-safe summaries
- Fix: Move detail back to hot canon notes; rewrite summary
- Prevention: Write player-safe version first, then add spoilers separately

**Topology unacknowledged**

- Symptom: Canon changes affect structure but Plotwright wasn't consulted
- Fix: Route a Story Spark mini to adjust gateways/loops
- Prevention: Always assess topology impact before finalizing

**Circular references**

- Symptom: Canon A references B, B references A, neither is grounded
- Fix: Ground at least one in established facts; break the cycle
- Prevention: Trace reference chains during drafting

**Missing research posture**

- Symptom: Factual claims lack uncertainty markers
- Fix: Add `uncorroborated:<risk>` and neutral phrasing notes
- Prevention: Always tag factual claims with research status

## Loop Participants

The roles that participate in this loop with their operational parameters.

:::{loop-participants}
showrunner:
  timeout: 300
  max_iterations: 5
lorekeeper:
  timeout: 900
  max_iterations: 15
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
when: brief_created
delegate_to: lorekeeper
description: SR scopes deepening pass; LK drafts canon entries
:::

:::{routing-rule}
after: showrunner
when: terminate
delegate_to: END
description: Deepening complete or cancelled; end the loop
:::

### After Lorekeeper

:::{routing-rule}
after: lorekeeper
when: needs_topology_check
delegate_to: plotwright
description: Canon changes may affect structure; PW assesses
:::

:::{routing-rule}
after: lorekeeper
when: canon_drafted
delegate_to: gatekeeper
description: Canon entries ready; GK validates before approval
:::

:::{routing-rule}
after: lorekeeper
when: escalation
delegate_to: showrunner
description: LK encounters conflict requiring SR decision
:::

### After Plotwright

:::{routing-rule}
after: plotwright
when: topology_assessed
delegate_to: lorekeeper
description: Structure impact assessed; LK incorporates feedback
:::

:::{routing-rule}
after: plotwright
when: escalation
delegate_to: showrunner
description: PW finds major topology issue requiring SR decision
:::

### After Gatekeeper

:::{routing-rule}
after: gatekeeper
when: failed
delegate_to: lorekeeper
description: Validation failed; LK revises canon entries
:::

:::{routing-rule}
after: gatekeeper
when: passed
delegate_to: showrunner
description: Canon validated; SR approves for merge
:::

:::{routing-rule}
after: gatekeeper
when: waiver_requested
delegate_to: showrunner
description: GK requests waiver for quality bar; SR decides
:::

## Quality Gates

### Pre-Canonization Validation

:::{quality-gate}
before: gatekeeper
role: gatekeeper
bars:

- integrity
- gateways
blocking: true
:::

## Expected Flow

```text
Accepted Hooks
    |
[Showrunner] -> scopes deepening pass
    |
[Lorekeeper] -> drafts canon entries
    | (if topology affected)
[Plotwright] -> validates structure implications
    |
[Lorekeeper] -> finalizes with downstream notes
    |
[Gatekeeper] -> pre-gate validation
    | (if passed)
[Showrunner] -> approves for merge
```

## Artifacts Produced

- **CanonEntry**: Verified facts with timeline anchors and invariants
- **Canon Pack**: Collection of entries with downstream handoff notes
- **Pre-gate Note**: Anticipated quality bar risks

## Handoffs

After Lore Deepening, canonized content flows to:

- **Canon Commit**: Merge approved canon to cold_store
- **Codex Expansion**: Player-safe summaries for codex
- **Story Spark**: If topology needs adjustment (via hooks)
- **Scene Weave**: Prose callbacks and foreshadowing notes
