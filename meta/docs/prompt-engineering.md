# Prompt Engineering Guide

Best practices for crafting effective agent prompts, derived from empirical testing with various model sizes.

## Sources

- Issue #242: Extensive testing of Qwen3:8b tool calling behavior
- Issue #256: Investigation of v3 vs v4 prompt size regression
- Stanford research on "Lost in the Middle" attention patterns

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

## 3. Tool Description Biasing

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

## 4. Prompt-History Conflicts

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

## 5. Small Model Considerations

Models under 8B parameters have distinct limitations:

### Token Budget

| Model Class | Recommended System Prompt Size |
|-------------|-------------------------------|
| Small (≤8B) | ≤2,000 tokens                 |
| Medium (8-70B) | ≤6,000 tokens              |
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

## 6. Semantic Ambiguity

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

## 7. Ordering for Attention

Given the U-shaped attention curve, structure prompts strategically:

### Recommended Order

1. **Critical behavioral constraints** (lines 1-20)
2. **Role identity and purpose** (lines 21-50)
3. **Tool descriptions** (injected by LangChain)
4. **Reference material** (middle - lowest attention)
5. **Knowledge menu** (for consult pattern)
6. **Critical reminder** (last 10-20 lines)

### What Goes in the Middle

Lower-priority content that can be consulted on demand:

- Detailed procedures (use `consult_playbook`)
- Reference tables (use `consult_knowledge`)
- Quality criteria details (use `consult_schema`)

---

## 8. Menu + Consult Pattern

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

## 9. Testing Prompts

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
- [ ] Tool count appropriate for model size
- [ ] System prompt within token budget
- [ ] No prompt-history conflicts
- [ ] Complex logic simplified for small models
- [ ] Menu + consult pattern for reference material
- [ ] `injection_priority: "critical"` on must-follow rules
- [ ] Tested with smallest target model
