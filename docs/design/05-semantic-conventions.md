# Semantic Conventions

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

These conventions prevent semantic ambiguity that confuses LLMs and humans. Use these exact terms in prompts, schemas, and code.

---

## Core Principle: Separate the Axes

Every response surface should separate three conceptual axes:

| Axis | Question | Reserved Name |
|------|----------|---------------|
| **Execution outcome** | Did the operation apply? | `action_outcome` |
| **Quality assessment** | Is the data acceptable? | `validation_result` |
| **Next-action** | What should happen? | `recommendation` |

### Anti-Pattern: Conflating Axes

```yaml
# WRONG: "status" conflates all three
status: failed  # Operation failed? Data invalid? Should retry?

# CORRECT: Each axis explicit
action_outcome: rejected      # Operation not applied
validation_result: fail       # Data didn't pass checks
recommendation: rework        # Fix and retry
```

---

## Reserved Vocabulary

### Execution Outcomes (`action_outcome`)

Result of attempting an operation. Transport-level, not quality-related.

| Value | Meaning |
|-------|---------|
| `saved` | Artifact persisted successfully |
| `rejected` | Operation refused |
| `deferred` | Queued for later |
| `skipped` | Intentionally not performed |

### Quality Assessment (`validation_result`)

Assessment of data quality.

| Value | Meaning |
|-------|---------|
| `pass` | Meets all requirements |
| `fail` | Does not meet requirements |
| `warn` | Passes with concerns |
| `skip` | Check not applicable |

**Banned values**: `green`, `yellow`, `red`, `success`, `error`

### Recommendations (`recommendation`)

What should happen next.

| Value | Meaning |
|-------|---------|
| `proceed` | Continue to next stage |
| `rework` | Revise and retry |
| `escalate` | Needs human decision |
| `hold` | Wait for external input |

---

## Lifecycle States

Use these exact terms for artifact lifecycle:

| State | Meaning |
|-------|---------|
| `draft` | Initial creation, mutable |
| `review` | Awaiting validation |
| `approved` | Passed quality gates |
| `shipped` | Exported for distribution |

---

## Content Field Naming

Use consistent names for text fields:

| Purpose | Name | Description |
|---------|------|-------------|
| Player narrative | `prose` | Story text shown to player |
| Internal notes | `notes` | Author/system notes, not player-visible |
| Short summary | `summary` | Brief overview (< 100 chars) |
| Full description | `description` | Complete explanation |
| Implementation | `details` | Technical specifics |

### Anti-Pattern

```yaml
# WRONG: Inconsistent naming
content: "The rain fell..."  # Is this prose? Notes? Description?
body: "Maria walked..."      # Ambiguous
text: "..."                  # Too generic

# CORRECT: Purpose-specific
prose: "The rain fell..."           # Player-facing story text
notes: "Foreshadows later scene"    # Internal note
summary: "Maria investigates"       # Brief summary
```

---

## ID Naming

### Artifact IDs

Format: `<type>_<descriptor>`

```yaml
# Passages
opening_001
morgue_discovery
chinatown_003

# Characters
detective_maria
captain_chen

# Locations
harbor_district
city_morgue
```

**Rules**:
- Lowercase with underscores
- No spaces or special characters
- Descriptive but concise
- Unique within type

### State IDs

Format: `<category>_<descriptor>` or `<verb>_<object>`

```yaml
# Codewords
found_body
has_badge_favor
knows_jimmy_alive
trusted_lily

# Stats
investigation
chinatown_trust
approach
```

**Rules**:
- Codewords: past tense or has/is prefix
- Stats: noun or noun phrase
- Hidden variables: same as stats

---

## Choice Text

### Format

Choices should be:
- First person (what Maria does)
- Action-oriented
- Distinct in intent

```yaml
# CORRECT
choices:
  - text: "Ask about Tommy directly"
  - text: "Ease into conversation first"

# WRONG
choices:
  - text: "Tommy question"        # Not a sentence
  - text: "Maybe ask about it"    # Vague
  - text: "Click here to continue" # Meta, not diegetic
```

### Conditional Choices

When choices have conditions, provide unavailable text:

```yaml
choices:
  - text: "Flash your badge"
    condition: has_badge_favor
    unavailable_text: "You'd need credentials for that"
```

---

## Gate Descriptions

Gates must use diegetic language:

| Mechanical (Wrong) | Diegetic (Correct) |
|-------------------|-------------------|
| "Requires investigation >= 30" | "You don't know enough yet" |
| "Missing codeword: has_key" | "The door is locked" |
| "Stat check failed" | "Your hands are shaking too much" |

---

## Prompt Language

### Instructions

Use directive language, not passive:

| Passive (Avoid) | Directive (Use) |
|-----------------|-----------------|
| "You might want to..." | "Do X" |
| "Consider perhaps..." | "Include X" |
| "It would be nice if..." | "Ensure X" |

### Constraints

Be explicit about requirements:

```yaml
# WRONG: Vague
prose_guidance: "Write something atmospheric"

# CORRECT: Specific
prose_guidance:
  length: medium           # 200-400 words
  pov: first_person
  tense: past
  sensory_focus:
    - smell of rain
    - sound of traffic
  tone: melancholic
```

---

## Error Messages

### Format

```yaml
error:
  action_outcome: rejected
  reason: "Invalid passage reference"
  location: "grow/branches/chinatown_path.yaml:23"
  recovery_action: "Change target to valid passage ID"
```

### Required Fields

| Field | Purpose |
|-------|---------|
| `action_outcome` | What happened |
| `reason` | Why it failed |
| `location` | Where the error is |
| `recovery_action` | How to fix it |

---

## Validation Output

### Per-Check Format

```yaml
checks:
  - name: reference_resolution
    status: pass  # pass | fail | warn | skip
    details: "All 47 references resolve"

  - name: gate_satisfiability
    status: warn
    location: grow/anchors.yaml:45
    details: "Gate 'tong_meeting' has only 2 viable paths"
    suggestion: "Add alternative path to increase chinatown_trust"
```

### Summary Format

```yaml
summary:
  passed: 6
  warnings: 2
  failed: 0
  overall: warn  # pass | warn | fail
```

---

## Artifact Headers

Every artifact must include:

```yaml
type: <artifact_type>    # Required
version: 1               # Required: schema version

# Optional metadata
_meta:
  created_by: <stage>
  created_at: <ISO8601>
  derived_from: [<source_refs>]
```

---

## Enum Values

### Be Specific

| Generic (Avoid) | Specific (Use) |
|-----------------|----------------|
| `any` | `all_time`, `all_sources` |
| `other` | Name the actual category |
| `misc` | Name the actual grouping |
| `none` | `not_applicable`, `excluded` |

### Use Enums for Finite Sets

```yaml
# WRONG: Open string
narrative_function: string

# CORRECT: Enum (set by GROW Phase 4a)
narrative_function:
  enum:
    - introduce
    - develop
    - complicate
    - confront
    - resolve
```

---

## Date/Time Format

Always use ISO 8601:

```yaml
created_at: 2026-01-01T10:30:00Z
export_date: 2026-01-01
```

---

## Checklist for New Interfaces

When designing new prompts, schemas, or outputs:

- [ ] Does any field conflate multiple axes?
- [ ] Are enum values specific, not generic?
- [ ] Is feedback directive, not passive?
- [ ] Are error messages actionable?
- [ ] Do finite sets use enums?
- [ ] Are dates ISO 8601?
- [ ] Are IDs lowercase with underscores?
- [ ] Is player-facing text in `prose`, not `content`?

---

## See Also

- [02-artifact-schemas.md](./02-artifact-schemas.md) — Schema definitions
- [05-prompt-compiler.md](./05-prompt-compiler.md) — Prompt assembly
- [07-design-principles.md](./07-design-principles.md) — Spoiler hygiene
