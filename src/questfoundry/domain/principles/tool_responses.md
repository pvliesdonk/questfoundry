# Tool Response Clarity

Tools must return clear, actionable results—not guidance for the LLM to interpret.

## Purpose

When tools return ambiguous responses (like criteria lists or instructions asking the LLM to "analyze"), the LLM may retry the same tool repeatedly, hoping for a definitive answer. This causes infinite loops and wasted compute.

## The Pattern

### Anti-Pattern: Guidance Response

```json
{
  "bar": "style",
  "artifact_content": {...},
  "evaluation_criteria": [
    "Check voice consistency",
    "Verify tone matches genre"
  ],
  "instruction": "Analyze the artifact and determine if it passes."
}
```

**Problem**: The LLM interprets this as "the tool needs more input" and retries.

### Correct Pattern: Verdict Response

```json
{
  "bar": "style",
  "artifact_id": "scene_001",
  "passed": true,
  "issues": [],
  "notes": "Style evaluation passed - voice and tone consistent.",
  "next_step": "Record this result and proceed to create_gatecheck_report."
}
```

**Why it works**: Clear verdict (`passed: true/false`), explicit issues list, guidance on what to do next.

## Rules

1. **Always return a verdict**: `passed`, `failed`, `error`, or similar boolean/enum
2. **List specific issues**: Empty list means no issues, not "check yourself"
3. **Provide next_step**: Tell the LLM what to do with this result
4. **Never ask the LLM to interpret**: The tool does the work, not the caller

## Applies To

- All evaluation tools (quality bars, validation)
- Query tools that can succeed or fail
- Any tool where the LLM might retry expecting different output

## Consequences of Violation

- LLM retry loops (5+ calls to same tool)
- Wasted tokens and compute
- Workflow stalls or timeouts
- Poor user experience
