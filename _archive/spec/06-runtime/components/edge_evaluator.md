# Edge Evaluator Component Specification

**Component Type**: STRICT (Core Mechanism)
**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Purpose

Evaluate conditional edges based on state to determine next node in loop execution.

---

## Responsibilities

1. Parse condition objects from edge definitions
2. Evaluate `python_expression` conditions safely
3. Evaluate `json_logic` conditions
4. Evaluate `bar_threshold` conditions (quality gates)
5. Return routing decision (next node ID or END)
6. Handle evaluation errors gracefully

---

## Input/Output Contract

### Input

```python
condition: Condition            # Condition object from edge
state: StudioState              # Current loop state
```

### Output

```python
bool                            # True if condition met, False otherwise
```

---

## Condition Schema

From `loop_pattern.schema.json`:

```yaml
condition:
  type: object
  properties:
    evaluator:
      type: string
      enum: [python_expression, json_logic, bar_threshold]
    expression:         # For python_expression
      type: string
    rules:              # For json_logic
      type: object
    bars_checked:       # For bar_threshold
      type: array
      items:
        type: string
        enum: [Integrity, Reachability, Nonlinearity, Gateways,
               Style, Determinism, Presentation, Accessibility]
    threshold:          # For bar_threshold
      type: string
      enum: [all_green, mostly_green, no_red, any_progress]
  required: [evaluator]
```

---

## Algorithm

### 1. Evaluate Condition (Main Entry Point)

```python
def evaluate_condition(
    condition: Condition,
    state: StudioState
) -> bool:
    """
    Evaluate condition based on evaluator type.

    Steps:
    1. Determine evaluator type
    2. Dispatch to appropriate evaluator
    3. Return result

    Example:
    condition = {
        "evaluator": "python_expression",
        "expression": "state['tu_lifecycle'] == 'cold-merged'"
    }

    result = evaluate_condition(condition, state)
    # Returns True if tu_lifecycle is 'cold-merged', False otherwise
    """
    evaluator = condition["evaluator"]

    if evaluator == "python_expression":
        return evaluate_python_expression(condition["expression"], state)
    elif evaluator == "json_logic":
        return evaluate_json_logic(condition["rules"], state)
    elif evaluator == "bar_threshold":
        return evaluate_bar_threshold(
            condition["bars_checked"],
            condition["threshold"],
            state
        )
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")
```

### 2. Evaluate Python Expression

```python
def evaluate_python_expression(
    expression: str,
    state: StudioState
) -> bool:
    """
    Safely evaluate Python expression against state.

    CRITICAL: Must use safe evaluation to prevent code injection.

    Steps:
    1. Prepare safe context (state variables only)
    2. Parse expression to AST
    3. Validate AST (no dangerous operations)
    4. Evaluate using ast.literal_eval or restricted eval
    5. Return boolean result

    Example:
    expression = "state['quality_bars']['Integrity']['status'] == 'green'"

    context = {
        'state': state,
        'hot_sot': state['hot_sot'],
        'cold_sot': state['cold_sot'],
        'quality_bars': state['quality_bars'],
        'tu_lifecycle': state['tu_lifecycle'],
        'error': state.get('error')
    }

    # Safe evaluation (option 1: use existing library)
    from asteval import Interpreter
    aeval = Interpreter()
    aeval.symtable.update(context)
    result = aeval(expression)
    return bool(result)

    # Safe evaluation (option 2: restricted eval)
    # Only allow whitelisted operations
    result = safe_eval(expression, context)
    return bool(result)
    """
```

**Common Expression Patterns**:

```python
# Lifecycle checks
"state['tu_lifecycle'] == 'cold-merged'"
"state['tu_lifecycle'] in ['stabilizing', 'gatecheck']"

# Quality bar checks
"state['quality_bars']['Integrity']['status'] == 'green'"
"all(bar['status'] != 'red' for bar in state['quality_bars'].values())"

# Artifact checks
"'current_hook' in state['hot_sot']"
"len(state['hot_sot']['hooks']) >= 3"

# Error checks
...
1. **Test python_expression**:
   - Simple equality: ""state['tu_lifecycle'] == 'cold-merged'""
   - Complex logic: ""all(bar['status'] != 'red' for bar in state['quality_bars'].values())""
   - Nested access: ""state['hot_sot']['current_hook']['header']['word_count'] > 200""
   - Edge cases: Empty `hot_sot`, missing keys, null values
```

**SECURITY**: Never use plain `eval()` or `exec()`. Use:

- `asteval` library (recommended)
- `RestrictedPython` library
- Custom AST validator + safe_eval

### 3. Evaluate JSON Logic

```python
def evaluate_json_logic(
    rules: dict,
    state: StudioState
) -> bool:
    """
    Evaluate JSON Logic rules against state.

    Uses json-logic-py library.

    Steps:
    1. Import json_logic
    2. Prepare state as data context
    3. Apply rules
    4. Return boolean result

    Example:
    rules = {
        "and": [
            {"==": [{"var": "tu_lifecycle"}, "cold-merged"]},
            {"==": [{"var": "error"}, None]}
        ]
    }

    from json_logic import jsonLogic

    data = {
        'tu_lifecycle': state['tu_lifecycle'],
        'hot_sot': state['hot_sot'],
        'cold_sot': state['cold_sot'],
        'quality_bars': state['quality_bars'],
        'error': state.get('error')
    }

    result = jsonLogic(rules, data)
    return bool(result)
    """
```

**Common JSON Logic Patterns**:

```json
{
  "==": [{"var": "tu_lifecycle"}, "cold-merged"]
}

{
  "and": [
    {">=": [{"var": "artifacts_count"}, 3]},
    {"==": [{"var": "error"}, null]}
  ]
}

{
  "or": [
    {"==": [{"var": "quality_bars.Integrity.status"}, "green"]},
    {"==": [{"var": "quality_bars.Integrity.status"}, "yellow"]}
  ]
}
```

### 4. Evaluate Bar Threshold

```python
def evaluate_bar_threshold(
    bars_checked: list[str],
    threshold: str,
    state: StudioState
) -> bool:
    """
    Evaluate quality bars against threshold.

    Thresholds:
    - all_green: All checked bars must be green
    - mostly_green: ≥75% green, rest yellow (no red)
    - no_red: No red bars allowed (green or yellow ok)
    - any_progress: At least one bar checked (not 'not_checked')

    Example:
    bars_checked = ["Integrity", "Style", "Presentation"]
    threshold = "all_green"

    # Get status for each bar
    statuses = [
        state['quality_bars'][bar]['status']
        for bar in bars_checked
    ]

    if threshold == "all_green":
        return all(s == "green" for s in statuses)

    elif threshold == "mostly_green":
        green_count = sum(1 for s in statuses if s == "green")
        yellow_count = sum(1 for s in statuses if s == "yellow")
        red_count = sum(1 for s in statuses if s == "red")

        return (
            red_count == 0 and
            green_count >= len(statuses) * 0.75
        )

    elif threshold == "no_red":
        return all(s != "red" for s in statuses)

    elif threshold == "any_progress":
        return any(s != "not_checked" for s in statuses)

    else:
        raise ValueError(f"Unknown threshold: {threshold}")
    """
```

**Threshold Decision Table**:

| Threshold     | Green | Yellow | Red | Not Checked | Result |
|---------------|-------|--------|-----|-------------|--------|
| all_green     | 3     | 0      | 0   | 0           | ✓ True |
| all_green     | 2     | 1      | 0   | 0           | ✗ False|
| mostly_green  | 3     | 1      | 0   | 0           | ✓ True |
| mostly_green  | 2     | 2      | 0   | 0           | ✗ False|
| no_red        | 2     | 1      | 0   | 0           | ✓ True |
| no_red        | 2     | 1      | 1   | 0           | ✗ False|
| any_progress  | 1     | 0      | 0   | 3           | ✓ True |
| any_progress  | 0     | 0      | 0   | 3           | ✗ False|

---

## Routing Functions (for LangGraph)

```python
def create_routing_function(
    edge: Edge,
    evaluator: EdgeEvaluator
) -> Callable[[StudioState], str]:
    """
    Create routing function for LangGraph conditional edge.

    Returns a function that:
    1. Evaluates condition
    2. Returns next node ID based on result

    Example:
    def routing_function(state: StudioState) -> str:
        if evaluator.evaluate_condition(edge.condition, state):
            return edge.target  # Condition met, go to target
        else:
            # Check retry count
            if state.get("retry_count", 0) < max_retries:
                return edge.source  # Loop back for rework
            else:
                return "__end__"  # Give up, exit loop

    return routing_function

    Usage in GraphFactory:
    graph.add_conditional_edges(
        source="gatekeeper",
        condition=create_routing_function(edge, evaluator),
        path_map={
            "scene_smith": "scene_smith",  # Rework
            "codex_curator": "codex_curator",  # Next stage
            "__end__": "__end__"  # Exit
        }
    )
    """
```

---

## Error Handling

### EvaluationError

```python
try:
    result = evaluate_condition(condition, state)
except Exception as e:
    # Log error
    logger.error(f"Condition evaluation failed: {e}")
    logger.error(f"Condition: {condition}")
    logger.error(f"State: {state}")

    # Return safe default (usually False to prevent unsafe progression)
    return False
```

### SecurityError

```python
# If unsafe expression detected
raise SecurityError(
    f"Unsafe expression detected: {expression}\n"
    f"Expressions must not use: import, exec, eval, __import__, etc."
)
```

---

## Testing Requirements

1. **Test python_expression**:
   - Simple equality: `"state['tu_lifecycle'] == 'cold-merged'"`
   - Complex logic: `"all(bar['status'] != 'red' for bar in state['quality_bars'].values())"`
   - Nested access: `"state['artifacts']['plotwright']['word_count'] > 200"`
   - Edge cases: Empty artifacts, missing keys, null values

2. **Test json_logic**:
   - Simple rules: `{"==": [{"var": "tu_lifecycle"}, "cold-merged"]}`
   - Complex rules: Nested and/or, multiple conditions
   - Edge cases: Null values, missing vars

3. **Test bar_threshold**:
   - all_green: All bars green → True, any non-green → False
   - mostly_green: 75% green → True, < 75% → False
   - no_red: No red → True, any red → False
   - any_progress: Any checked → True, all not_checked → False

4. **Test security**:
   - Reject unsafe expressions: `"import os; os.system('rm -rf /')"`
   - Reject dangerous functions: `"__import__('os').system('...')"`
   - Reject eval/exec: `"eval('...')"`

5. **Test error handling**:
   - Invalid expression syntax
   - Missing state keys
   - Type errors (comparing string to int)

---

## Dependencies

- **asteval** (recommended): Safe Python expression evaluation
- **json-logic-py**: JSON Logic evaluation
- **AST** (stdlib): Abstract Syntax Tree parsing for validation

---

## Performance Considerations

1. **Cache compiled expressions**: Don't reparse same expression repeatedly
2. **Optimize bar checks**: Precompute bar statuses before evaluation
3. **Fast path for simple conditions**: Direct comparison without parser overhead
4. **Limit expression complexity**: Set max AST depth to prevent abuse

---

## Example Usage

```python
# Create evaluator
evaluator = EdgeEvaluator()

# Test state
state = {
    "tu_id": "TU-2025-042",
    "tu_lifecycle": "gatecheck",
    "quality_bars": {
        "Integrity": {"status": "green", ...},
        "Style": {"status": "green", ...},
        "Presentation": {"status": "yellow", ...}
    }
}

# Evaluate python expression
condition_1 = {
    "evaluator": "python_expression",
    "expression": "state['tu_lifecycle'] == 'gatecheck'"
}
assert evaluator.evaluate_condition(condition_1, state) == True

# Evaluate bar threshold
condition_2 = {
    "evaluator": "bar_threshold",
    "bars_checked": ["Integrity", "Style"],
    "threshold": "all_green"
}
assert evaluator.evaluate_condition(condition_2, state) == True

# Evaluate mostly_green (should fail with 1 yellow)
condition_3 = {
    "evaluator": "bar_threshold",
    "bars_checked": ["Integrity", "Style", "Presentation"],
    "threshold": "all_green"
}
assert evaluator.evaluate_condition(condition_3, state) == False
```

---

## References

- **Loop Pattern Schema**: `spec/04-schemas/loop_pattern.schema.json`
- **Edge Definitions**: Loop topology.edges
- **Quality Bars**: spec/02-concepts/quality_bars.md
- **asteval**: <https://newville.github.io/asteval/>
- **json-logic-py**: <https://github.com/nadirizr/json-logic-py>

---

**IMPLEMENTATION NOTE**: This is a STRICT component. Condition evaluation correctness is critical for loop flow. Security is paramount - never execute untrusted code.
