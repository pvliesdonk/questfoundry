# Story Spark Loop

> **Goal:** Create meaningful narrative content from a story seed.

The **Story Spark** loop handles full story creation from request to canon. It transforms user requests into structured topology (Plotwright), fills prose content (Scene Smith), validates quality (Gatekeeper), and promotes to cold_store (Lorekeeper).

:::{loop-meta}
id: story_spark
name: "Story Spark"
trigger: user_request
entry_point: showrunner
version: 1
:::

## Guidance

This section provides operational context for executing the Story Spark workflow.

### When to Trigger

Invoke the Story Spark loop when:

- **New content request**: User wants to create a new chapter, section, or story arc
- **Structural expansion**: Existing content needs additional scenes or branches
- **Reachability repair**: Gatekeeper flags unreachable content requiring new paths
- **Nonlinearity enhancement**: Linear content needs branching options added

Do NOT invoke when:

- Only prose revision is needed (use Scene Smith directly)
- Only style changes are needed (use Creative Director directly)
- Only canon verification is needed (use Lorekeeper directly)

### Success Criteria

The loop succeeds when:

- [ ] All created scenes are reachable via at least one valid path
- [ ] At least 2 meaningful choices exist per non-terminal scene
- [ ] No dead-end paths without explicit terminal markers
- [ ] All gates have defined, achievable unlock conditions
- [ ] Gatekeeper validates reachability, nonlinearity, and gateway bars

### Common Failure Modes

**Single linear path**

- Symptom: Only one route through content
- Fix: Return to Plotwright for branching additions
- Prevention: Specify nonlinearity requirement in Brief

**Orphaned scenes**

- Symptom: Scenes with no incoming edges
- Fix: Plotwright adds connections or removes orphan
- Prevention: Verify edges before marking topology complete

**Undefined gates**

- Symptom: Gates exist without unlock conditions
- Fix: Plotwright defines conditions in terms of player state
- Prevention: Gate checklist before handoff

**Circular dependencies**

- Symptom: Gate A requires B, gate B requires A
- Fix: Plotwright breaks cycle with alternative path
- Prevention: Dependency analysis in topology review

**Scope creep**

- Symptom: Brief keeps expanding during execution
- Fix: Complete current scope, create new Brief for additions
- Prevention: Showrunner enforces scope boundaries

## Loop Participants

The roles that participate in this loop with their operational parameters.

:::{loop-participants}
showrunner:
  timeout: 300
  max_iterations: 5
plotwright:
  timeout: 600
  max_iterations: 10
lorekeeper:
  timeout: 300
  max_iterations: 5
gatekeeper:
  timeout: 300
  max_iterations: 3
scene_smith:
  timeout: 900
  max_iterations: 15
:::

## Routing Rules

Decision table for Showrunner: after a role completes work, these rules
describe when to delegate to the next role.

### After Showrunner

:::{routing-rule}
after: showrunner
when: brief_created
delegate_to: plotwright
description: After SR creates the Brief, delegate to PW for topology design
:::

:::{routing-rule}
after: showrunner
when: terminate
delegate_to: END
description: User request complete or cancelled; end the loop
:::

### After Plotwright

:::{routing-rule}
after: plotwright
when: needs_lore
delegate_to: lorekeeper
description: PW needs canon facts to inform structural decisions
:::

:::{routing-rule}
after: plotwright
when: topology_complete
delegate_to: gatekeeper
description: Topology design done; GK validates structure quality
:::

:::{routing-rule}
after: plotwright
when: escalation
delegate_to: showrunner
description: PW encounters blocking issue requiring SR intervention
:::

### After Lorekeeper

:::{routing-rule}
after: lorekeeper
when: lore_verified
delegate_to: plotwright
description: Canon facts confirmed; PW continues topology work
:::

:::{routing-rule}
after: lorekeeper
when: canon_provided
delegate_to: scene_smith
description: Canon callbacks ready; SS can reference them in prose
:::

:::{routing-rule}
after: lorekeeper
when: promoted_to_canon
delegate_to: showrunner
description: Content promoted to cold_store; notify SR of completion
:::

:::{routing-rule}
after: lorekeeper
when: escalation
delegate_to: showrunner
description: LK encounters canon conflict requiring SR decision
:::

### After Gatekeeper

:::{routing-rule}
after: gatekeeper
when: topology_failed
delegate_to: plotwright
description: Structure validation failed; PW must fix issues
:::

:::{routing-rule}
after: gatekeeper
when: topology_passed
delegate_to: scene_smith
description: Structure validated; SS fills prose into scene shells
:::

:::{routing-rule}
after: gatekeeper
when: prose_failed
delegate_to: scene_smith
description: Prose validation failed; SS must revise content
:::

:::{routing-rule}
after: gatekeeper
when: prose_passed
delegate_to: lorekeeper
description: All quality bars passed; LK promotes to canon
:::

:::{routing-rule}
after: gatekeeper
when: waiver_requested
delegate_to: showrunner
description: GK requests waiver for quality bar; SR decides
:::

### After Scene Smith

:::{routing-rule}
after: scene_smith
when: prose_complete
delegate_to: gatekeeper
description: Prose written; GK validates content quality
:::

:::{routing-rule}
after: scene_smith
when: needs_canon
delegate_to: lorekeeper
description: SS needs canon facts for accurate prose
:::

:::{routing-rule}
after: scene_smith
when: escalation
delegate_to: showrunner
description: SS encounters blocking issue requiring SR decision
:::

## Quality Gates

### Pre-Gatekeeper Topology Validation

:::{quality-gate}
id: topology_check
before: gatekeeper
role: gatekeeper
bars:

- reachability
- nonlinearity
- gateways
blocking: true
:::

### Pre-Lorekeeper Prose Validation

After Scene Smith fills prose, validate content quality before promotion.

:::{quality-gate}
id: prose_check
before: lorekeeper
role: gatekeeper
bars:

- style
- presentation
blocking: true
:::

## Expected Flow

```text
User Request
    |
[Showrunner] -> creates Brief
    |
[Plotwright] -> designs topology (Acts, Chapters, Scenes)
    | (if needs canon check)
[Lorekeeper] -> verifies lore facts
    |
[Gatekeeper] -> validates topology
    | (if topology passed)
[Scene Smith] -> fills prose into Scene artifacts
    | (if needs canon)
[Lorekeeper] -> provides canon callbacks
    |
[Gatekeeper] -> validates prose
    | (if prose passed)
[Lorekeeper] -> promotes Scene artifacts to cold_store
    |
[Showrunner] -> terminates
```

## Artifacts Produced

- **Brief**: Defines the scope and goals of the work
- **Act**: Structural division (title, sequence, chapter references)
- **Chapter**: Structural grouping (title, sequence, scene references)
- **Scene**: Prose content (title, section_id, content, gates, choices)
- **GatecheckReport**: Validation results from Gatekeeper

**Key**: Scene Smith writes prose into the `content` field of Scene artifacts.
Acts and Chapters are structural containers that reference child IDs.
