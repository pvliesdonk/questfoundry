# Story Spark Loop

The **Story Spark** loop handles the initial discovery phase when creating new story content. It transforms user requests into structured topology through delegation to specialist roles.

## Identity

:::{loop-meta}
id: story_spark
name: "Story Spark"
trigger: user_request
entry_point: showrunner
:::

## Overview

The Story Spark loop is the primary entry point for new content creation. It:

1. Receives user requests via the Showrunner
2. Creates a Brief artifact defining scope
3. Delegates to Plotwright for structural design
4. Optionally involves Lorekeeper for canon verification
5. Validates through Gatekeeper before completion

## Graph Nodes

### Showrunner Node

The entry point that receives user requests and delegates work.

:::{graph-node}
id: showrunner
role: showrunner
timeout: 300
max_iterations: 5
:::

### Plotwright Node

Designs the narrative topology based on the brief.

:::{graph-node}
id: plotwright
role: plotwright
timeout: 600
max_iterations: 10
:::

### Lorekeeper Node

Verifies canon consistency when structural decisions require it.

:::{graph-node}
id: lorekeeper
role: lorekeeper
timeout: 300
max_iterations: 5
:::

### Gatekeeper Node

Validates the completed topology against quality bars.

:::{graph-node}
id: gatekeeper
role: gatekeeper
timeout: 300
max_iterations: 3
:::

## Graph Edges

### From Showrunner

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

### From Plotwright

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

### From Lorekeeper

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

### From Gatekeeper

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
