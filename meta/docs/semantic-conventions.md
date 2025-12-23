# Semantic Conventions for LLM-Facing Interfaces

> **Purpose**: Define canonical vocabulary for tool interfaces, message payloads, and artifact schemas to prevent semantic ambiguity that confuses LLM agents.

## Background

LLM agents are particularly sensitive to semantic ambiguity. Vague or conflated terms lead to:

- Agents misinterpreting tool results
- Agents ignoring actionable feedback
- Inconsistent behavior across similar interfaces

This document establishes naming conventions and reserved vocabulary to ensure clarity.

## Core Principle: Separate the Axes

Every response surface should separate these three conceptual axes:

| Axis | Question Answered | Reserved Name(s) |
|------|-------------------|------------------|
| **Execution outcome** | Did the operation apply? | `action_outcome` |
| **Quality assessment** | Is the data acceptable? | `validation_result`, `overall_assessment` |
| **Next-action recommendation** | What should happen next? | `recommendation`, `recovery_action` |

### Anti-Pattern: Conflating Axes

```json
// BAD: "status" conflates all three axes
{ "status": "failed" }  // Does this mean: operation failed? data invalid? should retry?

// GOOD: Each axis is explicit
{
  "action_outcome": "rejected",      // Operation was not applied
  "validation_result": "fail",       // Data didn't pass checks
  "recovery_action": "Fix field X, then retry"  // What to do
}
```

## Reserved Vocabulary

### Execution Outcomes (`action_outcome`)

The result of attempting an operation. **Transport-level, not quality-related.**

| Value | Meaning |
|-------|---------|
| `saved` | Artifact was persisted successfully |
| `rejected` | Operation was refused (validation, permissions, etc.) |
| `deferred` | Operation queued for later (async processing) |
| `queued` | Added to a processing queue |

**Note**: `success: boolean` at the transport/tool level means "the tool ran without crashing" — it does NOT indicate domain-level outcomes.

### Quality Assessment (`validation_result` or `overall_assessment`)

Assessment of data quality or compliance.

| Value | Meaning |
|-------|---------|
| `pass` | Meets all requirements |
| `fail` | Does not meet requirements |
| `warn` | Passes with concerns (formerly `yellow`) |
| `skip` | Check was not applicable |
| `info` | Informational only, no pass/fail judgment |

**Banned values**: `green`, `yellow`, `red` — color codes are ambiguous for APIs.

### Next-Action Recommendations

| Field | Scope | Values |
|-------|-------|--------|
| `recommendation` | Orchestrator-level decision | `proceed`, `rework`, `escalate`, `hold` |
| `recovery_action` | Specific corrective instruction | Free-form directive string |

**Key rule**: `recovery_action` should be **directive and first** in the response, not buried as a passive "hint" at the end.

### Lifecycle States (`lifecycle_state`)

| Value | Meaning |
|-------|---------|
| `draft` | Initial creation, mutable |
| `review` | Submitted for validation |
| `approved` | Passed quality gates |
| `cold` | Committed to canon, immutable |

**For lifecycle transition results** (`transition_result`): `committed`, `rejected`, `deferred`

### Progress Phases (`progress_phase`)

| Value | Meaning |
|-------|---------|
| `started` | Work has begun |
| `in_progress` | Actively working |
| `blocked` | Cannot proceed, needs input |

**Banned values**: `completing` — use `in_progress` with high percent_complete instead.

### Relationship Kinds (`kind` in relationship.schema.json)

Defines the semantic nature of artifact-to-artifact relationships:

| Value | Meaning | Typical Cascade |
|-------|---------|-----------------|
| `derived_from` | Child is implementation of parent (e.g., section from brief) | Parent edit → demote children |
| `depends_on` | Child requires parent to function (hard dependency) | Parent delete → block or cascade |
| `supersedes` | Child replaces parent (versioning, retcon) | New version → archive old |
| `references` | Informational link only | No automatic cascade |

### Impact Policies (`impact_policy` in relationship.schema.json)

Defines runtime behavior when parent artifact changes:

**`on_parent_edit`** — What happens to children when parent is edited:

| Value | Meaning |
|-------|---------|
| `none` | No effect on children (default) |
| `flag_stale` | Set `_stale_parent=true` on children (advisory, no state change) |
| `demote` | Demote children to their `initial_state` (enforced) |

**`on_parent_delete`** — What happens to children when parent is deleted:

| Value | Meaning |
|-------|---------|
| `none` | No effect on children |
| `orphan` | Clear `link_field` on children (default) |
| `cascade_delete` | Delete children |
| `block` | Prevent parent deletion while children exist |

### Edit Policies (`edit_policy` in artifact-type.schema.json)

Defines runtime behavior when artifacts are edited in non-initial states:

| Value | Meaning |
|-------|---------|
| `allow` | Edit proceeds normally (default) |
| `demote` | Edit succeeds, artifact auto-demoted to `demote_target_state` |
| `disallow` | Edit rejected with error |

See [lifecycle-policy.md](lifecycle-policy.md) for detailed documentation.

## Content Field Naming

Different text fields serve different purposes. Use consistent names:

| Role | Preferred Name | Description |
|------|---------------|-------------|
| Player-facing narrative | `prose` | Canon-eligible story text |
| Internal reasoning/notes | `notes`, `internal_notes` | Not player-visible |
| Generic text blob | `text` | When role is unspecified |
| Container/wrapper | `content` | Object containing other fields, not a synonym for prose |
| User-visible entry | `full_entry` | Complete formatted text for codex-style artifacts |

**Anti-pattern**: Using `content`, `body`, `text`, `prose` interchangeably. Pick one per purpose.

## Tool Description Guidelines

Tool descriptions should explain **what** a tool does, not **who** uses it or **when**.

### Anti-Pattern: Role-Specific Biasing

```json
// BAD: Steers LLM tool selection based on role, not capability
"description": "The Showrunner uses this extensively to route work..."

// GOOD: Capability-focused
"description": "Route a task to a specialized agent for execution..."
```

### Anti-Pattern: Over-Emphatic Language

```json
// BAD: Competes with system prompt for attention
"description": "This is the ONLY way to communicate with humans..."

// GOOD: Factual
"description": "Send a message to the human user. Use for status updates, questions, and results."
```

### When Workflow Ordering Matters

If a tool should typically be called first or in a specific order, that guidance belongs in:

1. The agent's system prompt (preferred)
2. Knowledge base entries the agent consults
3. As a last resort, a brief factual note in the tool description

## Feedback Field Conventions

### Directive vs. Passive Language

| Passive (avoid) | Directive (preferred) |
|-----------------|----------------------|
| `hint` | `recovery_action`, `recommended_action` |
| `suggestion` | `next_step`, `required_action` |
| `you might want to` | `Do X` |

### Feedback Structure Priority

When returning validation feedback, structure it so the most important information comes first:

```json
{
  "action_outcome": "rejected",           // 1. What happened
  "rejection_reason": "validation_failed", // 2. Why
  "recovery_action": "Rename 'content' to 'prose', then retry", // 3. What to do
  "errors": [...]                          // 4. Details
}
```

## Enum Values

### Be Specific, Not Generic

| Generic (avoid) | Specific (preferred) |
|-----------------|---------------------|
| `any` | `all_time`, `all_sources` |
| `other` | Define the actual category |
| `misc` | Name the actual grouping |

### Use Enums for Finite Sets

If a field has a known, finite set of valid values, use an enum rather than an open string.

```json
// BAD: Open string allows invalid values
"layer": { "type": "string" }

// GOOD: Enum constrains to valid values
"layer": {
  "type": "string",
  "enum": ["constitution", "must_know", "should_know", "role_specific", "lookup"]
}
```

## Date/Time Formats

Always specify the format in the field description:

```json
{
  "published_date": {
    "type": "string",
    "description": "Publication date in ISO 8601 format (YYYY-MM-DD)"
  }
}
```

## Checklist for New Interfaces

When designing a new tool or message type:

- [ ] Does any field conflate multiple axes (execution/quality/action)?
- [ ] Are enum values specific, not generic (`all_time` vs `any`)?
- [ ] Is feedback directive, not passive (`recovery_action` vs `hint`)?
- [ ] Does the description explain capability, not role?
- [ ] Are date/time fields format-specified?
- [ ] Do finite sets use enums, not open strings?

## References

- Issue #228: Semantic ambiguity audit
- PR #220: Separated `status` into `task_completion`, `result_assessment`, `action_recommendation`
- PR #227: Changed `success`/`hint` to `action_outcome`/`recovery_action` in save_artifact
- `meta/schemas/core/_definitions.schema.json`: Canonical enum definitions
