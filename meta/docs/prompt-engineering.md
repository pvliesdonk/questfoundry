# Prompt Engineering Guide

Best practices for crafting effective agent prompts, derived from empirical testing with various model sizes.

## Sources

- Issue #242: Extensive testing of Qwen3:8b tool calling behavior
- Issue #256: Investigation of v3 vs v4 prompt size regression
- Issue #293: Tool schema bloat analysis
- Stanford research on "Lost in the Middle" attention patterns
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Context engineering for AI agents
- [Less is More: Optimizing Function Calling for LLM Execution on Edge Devices](https://arxiv.org/abs/2411.15399) (2024)
- [RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection](https://arxiv.org/html/2505.03275v1) (2025)
- [Anthropic Token-Efficient Tool Use](https://docs.claude.com/en/docs/agents-and-tools/tool-use/token-efficient-tool-use)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

---

## 1. Lost in the Middle

LLMs exhibit a U-shaped attention curve: information at the **beginning** and **end** of prompts receives stronger attention than content in the middle.

### The Problem

```
Position in prompt:  [START] -------- [MIDDLE] -------- [END]
Attention strength:   HIGH            LOW               HIGH
```

Critical instructions placed in the middle of a long prompt may be ignored, even by otherwise capable models.

### The Sandwich Pattern

For critical instructions, repeat them at the **start AND end** of the prompt:

```markdown
## CRITICAL: You are an orchestrator. NEVER write prose yourself.

[... 500+ lines of context ...]

## REMINDER: You are an orchestrator. NEVER write prose yourself.
```

### Implementation

Knowledge entries with `injection_priority: "critical"` are:

1. Placed at the start of the prompt (primacy)
2. Repeated in condensed form at the end (recency)

---

## 2. Tool Count Effects

Testing with Qwen3:8b revealed a strong correlation between tool count and compliance:

| Tool Count | Compliance Rate |
|------------|-----------------|
| 6 tools    | 100%            |
| 12 tools   | 85%             |
| 20 tools   | 70%             |

### Recommendations

- **Small models**: Limit to 6-8 tools per agent
- **Medium models**: Up to 12 tools acceptable
- **Large models**: Can handle 15+ tools but consider UX

---

## 3. Tool Schema Optimization

Tool schemas sent via API function calling are a major source of token overhead—often larger than the system prompt itself.

### The Hidden Cost

When using function calling, tool schemas are sent with every request:

| Component | Typical Size |
|-----------|--------------|
| Tool name | ~5 tokens |
| Description | 50-150 tokens |
| Parameter schema | 100-300 tokens |
| **Per tool total** | 150-450 tokens |
| **13 tools** | **2,000-6,000 tokens** |

This overhead exists *in addition to* the system prompt.

### Optimization Strategies

#### 1. Model-Class Tool Filtering

Define a reduced tool set for small models using `small_model_tools`:

```json
{
  "capabilities": [
    {"tool_ref": "delegate"},
    {"tool_ref": "communicate"},
    {"tool_ref": "search_workspace"},
    ...
  ],
  "small_model_tools": ["delegate", "communicate", "terminate_session"]
}
```

This mirrors the `small_model_must_know` pattern for knowledge.

#### 2. Two-Stage Tool Selection

For large tool libraries (20+), use a retrieval-based approach:

1. **Stage 1**: Show lightweight tool menu (name + summary only)
2. **Agent selects** which tool(s) are relevant
3. **Stage 2**: Load full schema only for selected tools

Research shows this can reduce token usage by 50%+ while improving accuracy 3x.

#### 3. Deferred Tool Loading

Mark tools as discoverable but not pre-loaded:

```json
{
  "id": "generate_image",
  "defer_loading": true
}
```

Deferred tools appear in a "tool search" interface rather than being sent to the API upfront. This approach achieved 85% token reduction in internal testing at Anthropic.

#### 4. Concise Tool Descriptions

Write descriptions that are 1-2 sentences max:

**Before** (~80 tokens):

```text
"Delegate work to another agent. This hands off control until the agent completes the task. Provide task description, context, expected outputs, and quality criteria. The receiving agent executes and returns via return_to_orchestrator with artifacts and assessment."
```

**After** (~20 tokens):

```text
"Hand off a task to another agent. Control returns when they complete."
```

The detailed usage guidance belongs in knowledge entries, not tool descriptions.

#### 5. Minimal Parameter Schemas

For small models, consider simplified schemas:

**Full schema** (~200 tokens):

```json
{
  "to_agent": {"type": "string", "description": "ID of agent"},
  "to_archetype": {"type": "string", "enum": [...]},
  "task": {"type": "string"},
  "context": {"type": "object", "properties": {...}},
  "expected_outputs": {"type": "array", "items": {...}},
  "quality_criteria": {"type": "array"},
  "priority": {"type": "string", "enum": [...]},
  "playbook_ref": {"type": "string"},
  "phase_ref": {"type": "string"}
}
```

**Minimal schema** (~50 tokens):

```json
{
  "to_agent": {"type": "string"},
  "task": {"type": "string"}
}
```

Optional parameters can be omitted for small models—they'll use reasonable defaults.

### Provider-Specific Optimizations

- **Anthropic**: Use `token-efficient-tools-2025-02-19` beta header for up to 70% output token reduction
- **OpenAI**: Consider fine-tuning to reduce schema tokens for frequently-used patterns
- **Local models**: Tool retrieval is essential—small models struggle with 10+ tools

### Research References

- "Less is More: Optimizing Function Calling for LLM Execution on Edge Devices" (2024)
- "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection" (2025)
- Anthropic's Tool Search Tool documentation

---

## 4. Layered Context Architecture

Organize agent prompts into distinct layers, each with a specific purpose:

### The Four Layers

| Layer | Purpose | Token Priority |
|-------|---------|----------------|
| **System** | Core agent identity, constraints | High (always include) |
| **Task** | Current task instructions | High |
| **Tool** | Tool descriptions and schemas | Medium (filter for small models) |
| **Memory** | Historical context, conversation | Variable (summarize as needed) |

### Layer Separation Benefits

- **Debugging**: Can isolate which layer caused unexpected behavior
- **Model switching**: System layer stays constant across model sizes
- **Token management**: Each layer can be independently compressed
- **Caching**: System and tool layers can be cached between turns

### Implementation

```
System Layer (always in system prompt):
├── Role identity
├── Behavioral constraints
└── Output format rules

Task Layer (per-turn):
├── Current objective
├── Relevant artifacts
└── Quality criteria

Tool Layer (configurable):
├── Tool schemas (via API)
├── Tool menu (in prompt)
└── Usage hints

Memory Layer (dynamic):
├── Conversation summary
├── Previous outputs
└── Error history
```

This architecture aligns with the "separation of concerns" pattern used in orchestrator-worker agent systems.

---

## 5. Tool Description Biasing

Tool descriptions have **higher influence** than system prompt content when models decide which tool to call.

### The Problem

If a tool description contains prescriptive language ("ALWAYS use this", "This is the primary method"), models will prefer that tool even when the system prompt suggests otherwise.

### Anti-pattern

```json
{
  "name": "generate_prose",
  "description": "ALWAYS use this tool to create story content. This is the primary way to generate text."
}
```

This biases the model toward `generate_prose` regardless of system prompt instructions.

### Solution

Use **neutral, descriptive** tool descriptions:

```json
{
  "name": "generate_prose",
  "description": "Creates story prose from a section brief. Produces narrative text with dialogue and descriptions."
}
```

Let the **system prompt** dictate when to use tools, not the tool descriptions.

---

## 6. Prompt-History Conflicts

When the system prompt says "MUST do X first" but the conversation history shows the model already did Y, confusion results.

### The Problem

```
System: "You MUST call consult_playbook before any delegation."
History: [delegate(...) was called successfully]
Model: "But I already delegated... should I undo it? Call consult now?"
```

### Solutions

1. **Use present-tense rules**: "Call consult_playbook before delegating" not "MUST call first"
2. **Acknowledge state**: "If you haven't yet consulted the playbook, do so now"
3. **Avoid absolute language** when state may vary

---

## 7. Small Model Considerations

Models under 8B parameters have distinct limitations:

### Token Budget

| Model Class | Recommended System Prompt Size |
|-------------|-------------------------------|
| Small (≤8B) | ≤2,000 tokens                 |
| Medium (9B-70B) | ≤6,000 tokens              |
| Large (70B+) | ≤12,000 tokens              |

Exceeding these budgets leads to:

- Ignored instructions (especially in the middle)
- Reduced tool compliance
- Hallucinated responses

### Instruction Density

Small models struggle with:

- Conditional logic: "If X and not Y, then Z unless W"
- Multiple competing priorities
- Nuanced edge cases

Simplify for small models:

- "Always call delegate" (not "call delegate unless validating")
- One instruction per topic
- Remove edge case handling (accept lower quality)

### The Concise Content Pattern

Knowledge entries support `concise_summary` and `concise_description` fields:

```json
{
  "summary": "Orchestrators delegate tasks to specialists. Before delegating, consult the relevant playbook to understand the workflow. Pass artifact IDs between steps. Monitor completion.",
  "concise_summary": "Delegate to specialists. Consult playbook first.",
  "content": { ... },
  "concise_description": "1. Consult playbook\n2. Delegate to specialist\n3. Pass artifact IDs"
}
```

Runtime selects the appropriate version based on model class.

---

## 8. Semantic Ambiguity

Avoid instructions that can be interpreted multiple ways.

### Anti-pattern

```
"Use your best judgment to determine when validation is needed."
```

Small models may interpret this as "never validate" or "always validate."

### Solution

Be explicit:

```
"Call validate_artifact after every save_artifact call."
```

---

## 9. Ordering for Attention

Given the U-shaped attention curve, structure prompts strategically:

### Recommended Order

1. **Critical behavioral constraints** (lines 1-20)
2. **Role identity and purpose** (lines 21-50)
3. **Tool descriptions** (injected by runtime)
4. **Reference material** (middle - lowest attention)
5. **Knowledge menu** (for consult pattern)
6. **Critical reminder** (last 10-20 lines)

### What Goes in the Middle

Lower-priority content that can be consulted on demand:

- Detailed procedures (use `consult_playbook`)
- Reference tables (use `consult_knowledge`)
- Quality criteria details (use `consult_schema`)

---

## 10. Menu + Consult Pattern

For knowledge that agents need access to but not always in context:

### Structure

```
System prompt contains:
- Summary/menu showing what knowledge exists
- Tool to consult for full details

System prompt does NOT contain:
- Full knowledge content
- Detailed procedures
- Reference material
```

### Benefits

- Smaller initial prompt
- Agent can "pull" knowledge when needed
- Works well with small models

### When to Inject vs. Consult

| Content Type | Small Model | Large Model |
|--------------|-------------|-------------|
| Role identity | Inject | Inject |
| Behavioral constraints | Inject | Inject |
| Workflow procedures | Consult | Inject or Consult |
| Quality criteria | Consult | Inject |
| Reference material | Consult | Consult |

---

## 11. Testing Prompts

### With Small Models

Before deploying prompts:

1. Test with smallest target model (e.g., Qwen3:8b)
2. Verify first-turn tool calls work
3. Check for prose generation (should not happen for orchestrators)
4. Measure token count of system prompt

### Metrics to Track

- **Tool compliance rate**: % of turns with correct tool calls
- **First-turn success**: Does the model call a tool on turn 1?
- **Prose leakage**: Does an orchestrator generate story content?
- **Instruction following**: Are critical constraints obeyed?

---

## Summary Checklist

- [ ] Critical instructions at start AND end (sandwich pattern)
- [ ] Tool descriptions are neutral, not prescriptive
- [ ] Tool descriptions concise (1-2 sentences max)
- [ ] Tool count appropriate for model size (≤8 for small models)
- [ ] `small_model_tools` defined for agents used with small models
- [ ] System prompt within token budget
- [ ] Context organized into layers (system, task, tool, memory)
- [ ] No prompt-history conflicts
- [ ] Complex logic simplified for small models
- [ ] Menu + consult pattern for reference material
- [ ] `injection_priority: "critical"` on must-follow rules
- [ ] Tested with smallest target model
