# Tool Patterns

This document describes patterns for tool usage in the meta-model, with emphasis on orchestrator behavior.

## Tools-Only Orchestrators

**Problem**: Orchestrators (like Showrunner) should coordinate work, not generate content. When orchestrators produce prose directly, they bypass specialists and quality gates.

**Solution**: Enforce that orchestrator agents must produce ALL output via tool calls.

### The Pattern

```text
Human → Runtime → Orchestrator
                      ↓
                 tool call: communicate(type="status", message="...")
                      ↓
Runtime ← unwrap and format
   ↓
Human sees: formatted message
```

**Key principle**: The orchestrator never generates raw text that reaches the human. All human-facing communication goes through the `communicate` tool, which the runtime unwraps and formats.

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Enforcement is trivial** | No tool call = incomplete turn. Runtime rejects. |
| **Structured observability** | All communication is typed and logged |
| **UX flexibility** | Runtime controls presentation (CLI vs web vs API) |
| **Separation of concerns** | Orchestrator coordinates; specialists create |

### Implementation

Orchestrators must end every turn with a **terminating tool**:

- `delegate` — assign work to another agent
- `communicate` — send message to human

If an orchestrator turn produces text without a terminating tool, the runtime:

1. Sends a nudge: "Orchestrators must use tools for all output"
2. Retries the turn (up to N attempts)
3. Fails the turn if no tool call after retries

---

## The `terminates_turn` Property

**Problem**: The runtime needs to know which tools end an agent's turn.

**Solution**: Tools declare `terminates_turn: true` in their definition.

### Schema

In `tool-definition.schema.json`:

```json
"terminates_turn": {
  "type": "boolean",
  "default": false,
  "description": "If true, calling this tool ends the agent's turn."
}
```

### Usage

```json
{
  "id": "delegate",
  "name": "Delegate Work",
  "terminates_turn": true,
  ...
}
```

### Runtime Behavior

For **orchestrator** agents (archetype = `orchestrator`):

```python
def is_turn_complete(agent: Agent, response: LLMResponse) -> bool:
    if "orchestrator" not in agent.archetypes:
        return True  # Non-orchestrators can use prose

    for tool_call in response.tool_calls:
        tool = registry.get(tool_call.name)
        if tool.terminates_turn:
            return True

    return False  # Orchestrator must use terminating tool
```

For **non-orchestrator** agents: No enforcement. They may produce prose output directly (e.g., Scene Smith generates story text).

---

## The `communicate` Tool

**Problem**: Orchestrators need to communicate with humans (status updates, questions, notifications) but shouldn't generate raw prose.

**Solution**: A unified `communicate` tool for all human-facing messages.

### Message Types

| Type | Purpose | Blocks Turn? |
|------|---------|--------------|
| `status` | Progress update | No |
| `question` | Need human input | Yes (waits for response) |
| `notification` | Deliverable ready | No |
| `error` | Something failed | Depends on severity |

### Example Usage

**Status update** (non-blocking):

```json
{
  "type": "status",
  "message": "Delegating structure planning to Plotwright."
}
```

**Question** (blocking):

```json
{
  "type": "question",
  "message": "What tone would you like for this story?",
  "options": [
    {"id": "dark", "description": "Dark and serious"},
    {"id": "light", "description": "Light and adventurous"}
  ]
}
```

**Notification** (non-blocking):

```json
{
  "type": "notification",
  "message": "Chapter 1 draft is ready for review.",
  "artifacts": ["workspace:section:chapter_1_v1"]
}
```

### Runtime Handling

```python
async def handle_communicate(args: dict) -> dict:
    msg_type = args["type"]
    message = args["message"]

    # Format for display
    formatted = formatter.format(msg_type, message, args)
    await display_to_human(formatted)

    # Questions block for response
    if msg_type == "question":
        response = await prompt_human(args.get("options"))
        return {"delivered": True, "response": response}

    return {"delivered": True}
```

### Replaces `request_clarification`

The `communicate` tool supersedes `request_clarification`:

| Old | New |
|-----|-----|
| `request_clarification(question=...)` | `communicate(type="question", message=...)` |

The new tool adds status updates, notifications, and error reporting.

---

## Stop-Tool Inventory

Tools marked as `terminates_turn: true`:

| Tool | Purpose |
|------|---------|
| `delegate` | Assign work to another agent |
| `communicate` | Send message to human |

When an orchestrator calls one of these, its turn ends. The runtime processes the tool and either:

- Activates another agent (for `delegate`)
- Waits for/displays human response (for `communicate`)

---

## Agent Constraints

Orchestrator agents should include a constraint enforcing tools-only behavior:

```json
{
  "id": "tools_only_output",
  "name": "Tools-Only Output",
  "rule": "ALL output must be via tool calls (delegate or communicate). Never generate raw prose or text outside of tool arguments.",
  "category": "process",
  "enforcement": "runtime",
  "severity": "critical"
}
```

The `enforcement: "runtime"` signals that this constraint is enforced by the runtime, not just by LLM self-regulation.

---

## Retry Behavior

When an orchestrator produces output without a terminating tool:

### Nudge Message

```json
{
  "type": "nudge",
  "nudge_type": "missing_output",
  "observation": "Your response did not include a terminating tool call (delegate or communicate).",
  "expected": "Orchestrators must end every turn with either delegate() or communicate().",
  "question": "Please use one of these tools to complete your turn."
}
```

### Configuration

```python
# In runtime config
tools_only_enforcement = {
    "max_retries": 3,
    "fail_on_exceed": True
}
```

After `max_retries` attempts without a terminating tool, the runtime fails the turn with an error.

---

## Interaction with Delegation

The `delegate` tool both terminates the orchestrator's turn AND activates the delegatee:

```text
Turn N: Orchestrator
  └─ delegate(to="scene_smith", task="Write chapter 1")
      └─ terminates_turn=true → Orchestrator turn ends

Turn N+1: Scene Smith
  └─ (receives delegation, works on task)
  └─ (may produce prose — not an orchestrator)

Turn N+2: Orchestrator (when Scene Smith completes)
  └─ communicate(type="notification", message="Chapter 1 ready")
      └─ terminates_turn=true → Turn ends
```

---

## Summary

| Pattern | Applies To | Enforcement |
|---------|------------|-------------|
| Tools-only output | Orchestrators | Runtime validates terminating tool |
| `terminates_turn` | Tool definitions | Schema property |
| `communicate` | Human interaction | Replaces `request_clarification` |
| Retry on violation | Orchestrators | Nudge + retry up to N times |

This architecture ensures orchestrators remain coordinators, not content generators, while maintaining structured observability over all human-facing communication.
