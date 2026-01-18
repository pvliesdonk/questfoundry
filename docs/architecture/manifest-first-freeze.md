# Manifest-First Freeze Architecture

This document describes the architecture for ensuring completeness in structured output pipelines, specifically the SEED stage's Discuss → Summarize → Serialize pattern.

## Problem: The Information Bottleneck

The SEED stage's three-phase pattern had a critical flaw in its original design:

```
[Discuss] → [Summarize] → [Serialize]
    ↓           ↓             ↓
 Rich tool   Prose brief    JSON output
 context     (lossy)        (incomplete)
```

**The failure mode:**
1. **Discuss phase** explores via tools, producing rich context in message history
2. **Summarize phase** converts messages to prose, losing tool call details
3. **Serialize phase** extracts from prose, missing items not prominently mentioned
4. **Validation** catches missing items post-hoc, but recovery is structurally impossible

The prose brief became a lossy compression—entities discussed via `query_graph` but not explicitly mentioned in summaries were lost forever.

## Solution: Manifest-First Freeze

### Core Principle

> **Manifest drives generation, not validation.**

Completeness is enforced by construction, not post-hoc parsing. The manifest (list of all required IDs) appears at every phase boundary, making omission visible and recovery possible.

### Three Gates Architecture

```
[Discuss] → Gate 1 → [Summarize] → Gate 2 → [Serialize] → Gate 3 → [Graph]
               ↓                       ↓                      ↓
         Manifest-aware          Manifest-driven        Count-based
         (must list ALL)         (generate for ALL)     (fast check)
```

**Gate 1 (Summarize Prompt):** Manifest-aware
- Prompt includes explicit list of ALL entity and tension IDs
- Summarize must include decisions for each listed ID
- Format: `entity_id: [retain|cut] - justification`

**Gate 2 (Serialize Prompt):** Manifest-driven
- Prompt language changed from "extraction" to "generation"
- Before: "Do NOT include entities not listed in brief"
- After: "You MUST generate a decision for EVERY ID below"
- Counts explicit: "Generate EXACTLY 5 entity decisions"

**Gate 3 (Validation):** Count-based structural check
- Fast pre-check: `len(output.entities) == expected.entities`
- No string parsing—just count comparison
- Semantic validation follows only if counts match

### Tool Call Preservation

A prerequisite for Gate 1: the summarize phase must see tool research results.

```python
# In _format_messages_for_summary:
for msg in messages:
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(f"[TOOL CALL: {tc['name']}]")
    elif isinstance(msg, ToolMessage):
        parts.append(f"[TOOL RESULT: {msg.name}]\n{msg.content}")
```

Without this, entities discovered via `query_graph` but not echoed in assistant text are invisible to summarize.

## Error Classification

Different error types require different recovery strategies. Treating all errors the same wastes retries and delays failure.

### Categories

| Category | Trigger | Recovery Strategy |
|----------|---------|-------------------|
| INNER | Schema/type error | Retry with Pydantic feedback |
| SEMANTIC | Invalid ID reference | Retry with valid ID list |
| COMPLETENESS | Count mismatch | Retry with manifest counts |
| FATAL | Unrecoverable | Fail immediately |

### Implementation

```python
class SeedErrorCategory(Enum):
    INNER = auto()       # Schema/type error in a section
    SEMANTIC = auto()    # Invalid ID reference (phantom IDs)
    COMPLETENESS = auto() # Missing entity/tension decisions
    FATAL = auto()       # Reserved for future use

def categorize_error(error: SeedValidationError) -> SeedErrorCategory:
    # Uses pattern constants for testability
    if _PATTERN_SEMANTIC_BRAINSTORM in issue or _PATTERN_SEMANTIC_SEED in issue:
        return SeedErrorCategory.SEMANTIC
    if _PATTERN_COMPLETENESS in issue:
        return SeedErrorCategory.COMPLETENESS
    return SeedErrorCategory.INNER
```

Targeted retry strategies can then:
- SEMANTIC errors: Re-inject valid ID list
- COMPLETENESS errors: Re-inject manifest with counts
- INNER errors: Re-inject Pydantic validation details

## Implementation References

| Component | Location | Purpose |
|-----------|----------|---------|
| Tool call preservation | `agents/summarize.py` | Preserves research context |
| Summarize manifest | `graph/context.py:format_summarize_manifest()` | Gate 1 ID lists |
| Serialize manifest | `graph/context.py:format_valid_ids_context()` | Gate 2 ID lists |
| Structural check | `graph/context.py:check_structural_completeness()` | Gate 3 fast validation |
| Error classification | `graph/mutations.py:categorize_error()` | Retry strategy selection |
| Serialize prompts | `prompts/templates/serialize_seed_sections.yaml` | Generation language |
| Summarize prompts | `prompts/templates/summarize_seed.yaml` | Manifest-aware format |

## Key Lessons

1. **Extraction mindset fails for completeness**: "Do NOT include items not in brief" makes omission irrecoverable

2. **Counts are more reliable than parsing**: `len(entities) == 5` is unambiguous; parsing "all entities mentioned above" is not

3. **Gates prevent, validation detects**: Prevention at phase boundaries is cheaper than detection at the end

4. **String matching is fragile**: Error categorization via patterns works but should migrate to structured error codes (see issue #216)

## Related Documents

- [ADR-011: Manifest-First Freeze](decisions.md#adr-011-manifest-first-freeze-for-seed-stage) - Decision record
- [Interactive Stages](interactive-stages.md) - The Discuss → Summarize → Serialize pattern
- [Graph Storage](graph-storage.md) - Graph mutations and validation
