# Graph Factory Component Specification

**Component Type**: STRICT (Core Mechanism)
**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Purpose

Transform loop pattern YAML definitions into executable LangGraph StateGraphs.

---

## Responsibilities

1. Load and parse loop YAML files
2. Create StateGraph with StudioState type annotation
3. Add nodes from loop.topology.nodes (delegate to NodeFactory)
4. Add edges from loop.topology.edges (direct and conditional)
5. Set entry point from loop.topology.entry_node
6. Add exit conditions from loop.topology.exit_conditions
7. Compile graph into executable CompiledGraph
8. Validate topology (no orphaned nodes, all edges valid)

---

## Input/Output Contract

### Input

```python
loop_id: str                    # e.g., "story_spark"
context: dict[str, Any]         # Initial context for state
```

### Output

```python
CompiledGraph                   # LangGraph CompiledGraph ready to invoke()
```

---

## Algorithm

### 1. Load Loop Definition

```python
def load_loop(loop_id: str) -> LoopPattern:
    """
    Load and validate loop YAML file.

    Steps:
    1. Construct path: spec/05-definitions/loops/{loop_id}.yaml
    2. Load YAML with PyYAML
    3. Validate against loop_pattern.schema.json using SchemaRegistry
    4. Parse into structured LoopPattern object
    5. Return LoopPattern

    Raises:
    - FileNotFoundError if loop YAML doesn't exist
    - ValidationError if YAML doesn't match schema
    """
```

**Key Fields to Extract**:

- `loop.id`
- `loop.metadata.name`
- `loop.topology.nodes` (list of node_ids)
- `loop.topology.entry_node` (string)
- `loop.topology.edges` (list of Edge objects)
- `loop.topology.exit_conditions` (list of ExitCondition objects)
- `loop.roles.required` (list of role_ids)

### 2. Create StateGraph

```python
def create_state_graph() -> StateGraph:
    """
    Create LangGraph StateGraph with StudioState annotation.

    from langgraph.graph import StateGraph
    from questfoundry.models.state import StudioState

    graph = StateGraph(StudioState)
    return graph
    """
```

**StudioState Structure** (see state_manager.md for full spec):

```python
class StudioState(TypedDict):
    tu_id: str                              # Trace Unit ID
    tu_lifecycle: str                       # hot-proposed, stabilizing, gatecheck, cold-merged
    current_node: str                       # Current role executing
    loop_context: dict[str, Any]            # Loop-specific context
    artifacts: dict[str, Artifact]          # Generated artifacts
    quality_bars: dict[str, BarStatus]      # 8 quality dimensions
    messages: list[Message]                 # Protocol messages
    snapshot_ref: str | None                # Read-only snapshot reference
    error: str | None                       # Error message if any
```

### 3. Add Nodes

```python
def add_nodes(graph: StateGraph, loop: LoopPattern, node_factory: NodeFactory) -> None:
    """
    Add all nodes from loop.topology.nodes.

    For each node_id in loop.topology.nodes:
        1. Load role definition (node_id corresponds to role_id)
        2. Create Runnable using NodeFactory.create_role_node()
        3. Add to graph: graph.add_node(node_id, runnable)

    Example:
        loop.topology.nodes = ["plotwright", "scene_smith", "gatekeeper"]

        # For each:
        plotwright_role = schema_registry.load_role("plotwright")
        plotwright_node = node_factory.create_role_node(plotwright_role)
        graph.add_node("plotwright", plotwright_node)
    """
```

**CRITICAL**: Node IDs in topology MUST match role IDs in definitions. This is enforced by schema validation.

### 4. Add Edges

Loop edges come in two types:

#### A. Direct Edges

```python
def add_direct_edge(graph: StateGraph, edge: Edge) -> None:
    """
    Add unconditional edge from source to target.

    Schema:
    {
      "source": "plotwright",
      "target": "scene_smith",
      "type": "direct"
    }

    Implementation:
    graph.add_edge(edge.source, edge.target)
    """
```

#### B. Conditional Edges

```python
def add_conditional_edge(graph: StateGraph, edge: Edge, edge_evaluator: EdgeEvaluator) -> None:
    """
    Add conditional edge with routing function.

    Schema:
    {
      "source": "gatekeeper",
      "target": "scene_smith",  # Default target if condition true
      "type": "conditional",
      "condition": {
        "evaluator": "bar_threshold",
        "bars_checked": ["Integrity", "Style"],
        "threshold": "all_green"
      }
    }

    Implementation:
    1. Create routing function that evaluates condition
    2. Add conditional edge: graph.add_conditional_edges(
         source=edge.source,
         condition=routing_function,
         path_map={
           "pass": edge.target,
           "fail": edge.source,  # Loop back for rework
           "__end__": "__end__"  # Exit if max retries
         }
       )

    Routing Function:
    def routing_function(state: StudioState) -> str:
        result = edge_evaluator.evaluate_condition(edge.condition, state)
        if result:
            return "pass"
        elif state.get("retry_count", 0) < max_retries:
            return "fail"
        else:
            return "__end__"
    """
```

**Edge Types from Schema**:

- `direct`: Unconditional transition
- `conditional`: Evaluated at runtime

**Condition Evaluators** (see edge_evaluator.md):

- `python_expression`: Evaluate Python expression against state
- `json_logic`: Evaluate JSON Logic rules
- `bar_threshold`: Check quality bars against threshold

### 5. Set Entry Point

```python
def set_entry_point(graph: StateGraph, loop: LoopPattern) -> None:
    """
    Set the entry node for the graph.

    graph.set_entry_point(loop.topology.entry_node)

    Example:
    loop.topology.entry_node = "plotwright"
    graph.set_entry_point("plotwright")
    """
```

**CRITICAL**: `entry_node` MUST be one of the nodes in `topology.nodes`.

### 6. Add Exit Conditions

```python
def add_exit_conditions(graph: StateGraph, loop: LoopPattern, edge_evaluator: EdgeEvaluator) -> None:
    """
    Add conditional edges to END for each exit condition.

    Schema:
    exit_conditions:
      - name: "success"
        condition:
          evaluator: "python_expression"
          expression: "state['tu_lifecycle'] == 'cold-merged'"
      - name: "failure"
        condition:
          evaluator: "python_expression"
          expression: "state.get('error') is not None"

    Implementation:
    For each exit_condition:
        1. Identify which nodes can trigger this exit
           (usually the last node in topology, or nodes with no outgoing edges)
        2. Create routing function that checks condition
        3. Add conditional edge from those nodes to END

    Example:
    def check_success_exit(state: StudioState) -> str:
        if edge_evaluator.evaluate_condition(exit_cond.condition, state):
            return "__end__"
        else:
            return "continue"  # Stay in loop

    # Add to final node (e.g., gatekeeper)
    graph.add_conditional_edges(
        source="gatekeeper",
        condition=check_success_exit,
        path_map={
            "__end__": "__end__",
            "continue": "plotwright"  # Loop back to start
        }
    )
    """
```

**Exit Condition Names** (from schema):

- `success`: Normal completion
- `failure`: Error occurred
- `deferred`: Postponed for later
- `max_iterations`: Safety limit reached

### 7. Compile Graph

```python
def compile_graph(graph: StateGraph) -> CompiledGraph:
    """
    Compile the graph into executable form.

    compiled = graph.compile()
    return compiled

    This performs:
    - Topology validation (no orphaned nodes, cycles allowed, entry point valid)
    - Type checking (StudioState compatibility)
    - Optimization (execution planning)
    """
```

### 8. Validate Topology

```python
def validate_topology(loop: LoopPattern) -> None:
    """
    Validate loop topology before building graph.

    Checks:
    1. entry_node is in topology.nodes
    2. All edge sources are in topology.nodes
    3. All edge targets are in topology.nodes or "__end__"
    4. All nodes are reachable from entry_node
    5. At least one exit condition exists

    Raises:
    - TopologyError with specific issue if validation fails
    """
```

---

## Complete Implementation Flow

```python
def create_loop_graph(loop_id: str, context: dict[str, Any]) -> CompiledGraph:
    """
    Complete graph creation flow.

    Args:
        loop_id: Loop pattern identifier (e.g., "story_spark")
        context: Initial context for StudioState

    Returns:
        Compiled LangGraph ready to invoke()

    Example:
        graph = create_loop_graph(
            loop_id="story_spark",
            context={"scene_text": "cargo bay confrontation"}
        )
        result = graph.invoke({
            "tu_id": "TU-2025-042",
            "tu_lifecycle": "hot-proposed",
            "loop_context": context,
            "artifacts": {},
            "quality_bars": {},
            "messages": []
        })
    """
    # 1. Load loop definition
    loop = load_loop(loop_id)

    # 2. Validate topology
    validate_topology(loop)

    # 3. Create StateGraph
    graph = create_state_graph()

    # 4. Add nodes
    node_factory = NodeFactory()
    add_nodes(graph, loop, node_factory)

    # 5. Add edges
    edge_evaluator = EdgeEvaluator()
    for edge in loop.topology.edges:
        if edge.type == "direct":
            add_direct_edge(graph, edge)
        elif edge.type == "conditional":
            add_conditional_edge(graph, edge, edge_evaluator)

    # 6. Set entry point
    set_entry_point(graph, loop)

    # 7. Add exit conditions
    add_exit_conditions(graph, loop, edge_evaluator)

    # 8. Compile
    compiled = compile_graph(graph)

    return compiled
```

---

## Error Handling

### FileNotFoundError

```python
raise FileNotFoundError(
    f"Loop definition not found: spec/05-definitions/loops/{loop_id}.yaml"
)
```

### ValidationError

```python
raise ValidationError(
    f"Loop {loop_id} failed schema validation:\n{error_details}"
)
```

### TopologyError

```python
raise TopologyError(
    f"Invalid topology in loop {loop_id}: {specific_issue}"
)
```

---

## Testing Requirements

1. **Test with all 10 loop patterns** from `spec/05-definitions/loops/`
2. **Test simple linear flow**: story_spark (plotwright → scene_smith → gatekeeper)
3. **Test conditional edges**: hook_harvest (gatekeeper can loop back)
4. **Test complex multi-role**: binding_run (7 roles, multiple branches)
5. **Test exit conditions**: All loops have success/failure exits
6. **Test topology validation**: Catch orphaned nodes, invalid entry points
7. **Test error propagation**: Invalid YAML, missing roles, broken edges

---

## Dependencies

- **SchemaRegistry**: Load and validate loop YAML
- **NodeFactory**: Create Runnable from role definitions
- **EdgeEvaluator**: Evaluate conditional edges
- **StateManager**: Initialize StudioState
- **LangGraph**: StateGraph, CompiledGraph classes

---

## Performance Considerations

1. **Cache compiled graphs**: Don't recompile same loop repeatedly
2. **Lazy node creation**: Only create Runnables when needed
3. **Validate once**: Don't re-validate YAML on every execution
4. **Optimize edge evaluation**: Use fast path for simple conditions

---

## Example Usage

```python
# Create graph factory
factory = GraphFactory(
    schema_registry=SchemaRegistry(),
    node_factory=NodeFactory(),
    edge_evaluator=EdgeEvaluator()
)

# Create story_spark loop
story_spark = factory.create_loop_graph(
    loop_id="story_spark",
    context={"scene_text": "tense cargo bay confrontation"}
)

# Initialize state
initial_state = {
    "tu_id": "TU-2025-042",
    "tu_lifecycle": "hot-proposed",
    "current_node": "plotwright",
    "loop_context": {"scene_text": "tense cargo bay confrontation"},
    "artifacts": {},
    "quality_bars": {},
    "messages": [],
    "snapshot_ref": None,
    "error": None
}

# Execute
final_state = story_spark.invoke(initial_state)

# Check result
if final_state["tu_lifecycle"] == "cold-merged":
    print("✓ Scene approved and merged to canon")
else:
    print(f"✗ Scene still in {final_state['tu_lifecycle']}")
```

---

## References

- **Loop Pattern Schema**: `spec/04-schemas/loop_pattern.schema.json`
- **Node Factory Spec**: `components/node_factory.md`
- **Edge Evaluator Spec**: `components/edge_evaluator.md`
- **State Manager Spec**: `components/state_manager.md`
- **LangGraph Docs**: <https://langchain-ai.github.io/langgraph/>

---

**IMPLEMENTATION NOTE**: This is a STRICT component. Every detail in this spec MUST be implemented exactly as described. Any deviation breaks the spec-to-runtime contract.
