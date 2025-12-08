# Story Spark Loop

> **Goal:** Create meaningful nonlinearity from a story seed.

The **Story Spark** loop handles the initial discovery phase when creating new story content. It transforms user requests into structured topology through delegation to specialist roles.

:::{loop-meta}
id: story_spark
name: "Story Spark"
trigger: user_request
entry_point: showrunner
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

## Execution Graph

### Graph Nodes

#### Showrunner Node

The entry point that receives user requests and delegates work.

:::{graph-node}
id: showrunner
role: showrunner
timeout: 300
max_iterations: 5
:::

#### Plotwright Node

Designs the narrative topology based on the brief.

:::{graph-node}
id: plotwright
role: plotwright
timeout: 600
max_iterations: 10
:::

#### Lorekeeper Node

Verifies canon consistency when structural decisions require it.

:::{graph-node}
id: lorekeeper
role: lorekeeper
timeout: 300
max_iterations: 5
:::

#### Gatekeeper Node

Validates the completed topology against quality bars.

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
target: plotwright
condition: "intent.status == 'brief_created'"
:::

:::{graph-edge}
source: showrunner
target: END
condition: "intent.type == 'terminate'"
:::

#### From Plotwright

:::{graph-edge}
source: plotwright
target: lorekeeper
condition: "intent.status == 'needs_lore'"
:::

:::{graph-edge}
source: plotwright
target: gatekeeper
condition: "intent.status == 'topology_complete'"
:::

:::{graph-edge}
source: plotwright
target: showrunner
condition: "intent.type == 'escalation'"
:::

#### From Lorekeeper

:::{graph-edge}
source: lorekeeper
target: plotwright
condition: "intent.status == 'verified'"
:::

:::{graph-edge}
source: lorekeeper
target: showrunner
condition: "intent.type == 'escalation'"
:::

#### From Gatekeeper

:::{graph-edge}
source: gatekeeper
target: plotwright
condition: "intent.status == 'failed'"
:::

:::{graph-edge}
source: gatekeeper
target: showrunner
condition: "intent.status == 'passed'"
:::

:::{graph-edge}
source: gatekeeper
target: showrunner
condition: "intent.status == 'waiver_requested'"
:::

## Quality Gates

### Pre-Gatekeeper Validation

:::{quality-gate}
before: gatekeeper
role: gatekeeper
bars:

- reachability
- nonlinearity
- gateways
blocking: true
:::

## Expected Flow

```
User Request
    ↓
[Showrunner] → creates Brief
    ↓
[Plotwright] → designs topology
    ↓ (if needs canon check)
[Lorekeeper] → verifies facts
    ↓
[Gatekeeper] → validates structure
    ↓ (if passed)
[Showrunner] → approves/terminates
```

## Artifacts Produced

- **Brief**: Defines the scope and goals of the work
- **Scene**: Structural shells with gates and choices (no prose yet)
- **GatecheckReport**: Validation results from Gatekeeper
