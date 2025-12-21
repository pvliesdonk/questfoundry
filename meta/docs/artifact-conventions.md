# Artifact Schema Conventions

> **Purpose**: Document content field and lifecycle state naming conventions across artifact types.

## Content Field Conventions

Artifact types use different field names for text content based on their role:

| Field Name | Purpose | Used By |
|------------|---------|---------|
| `prose` | Player-facing story narrative | `section` |
| `full_entry` | Player-safe glossary/codex text | `codex_entry` |
| `summary` | Brief overview of the artifact | Many types |
| `description` | Explanatory text, often internal | `audio_plan`, `style_guide` |
| `short_answer` | Concise research finding | `research_memo` |
| `goal`, `key_beats` | Planning/structural content | `section_brief` |

### Why Different Names?

The distinction between `prose` and `full_entry` is intentional:

- **`prose`**: Story narrative that goes through full gatecheck (8 bars) before becoming `cold`
- **`full_entry`**: Player-safe reference text that goes through spoiler validation before being `published`

Both are player-facing, but they have different validation paths and terminal states.

## Lifecycle State Conventions

### Standard Content Lifecycle (Narrative)

```text
draft → review → gatecheck → approved → cold
```

Used by: `section`

- `draft`: Being created
- `review`: Style review in progress
- `gatecheck`: Awaiting 8-bar validation
- `approved`: Passed gatecheck
- `cold`: Merged to canon (terminal)

### Codex Lifecycle (Reference Content)

```text
draft → review → validated → published
```

Used by: `codex_entry`

- `draft`: Being written
- `review`: Spoiler/style check
- `validated`: Passed checks
- `published`: Exported to player (terminal)

### Asset Plan Lifecycle

```text
draft → ready → generating → rendered
              ↘ deferred
```

Used by: `art_plan`, `audio_plan`

- `draft`: Plan being created
- `ready`: Ready for generation
- `generating`: Generation in progress
- `rendered`: Asset created (terminal)
- `deferred`: Generation postponed

### Support Artifact Lifecycle

```text
draft → active/complete/ready → superseded/archived
```

Used by: `style_guide`, `canon_pack`, `research_memo`, etc.

- `draft`: Being created
- `active`/`complete`/`ready`: In use
- `superseded`: Replaced by newer version (terminal)
- `archived`: Work complete, no longer active (terminal)

## Terminal States

All terminal states are marked with `"terminal": true` in the schema.

| State | Meaning |
|-------|---------|
| `cold` | Merged to canon, immutable |
| `published` | Exported to player |
| `rendered` | Asset generated |
| `archived` | Work complete |
| `superseded` | Replaced by newer version |
| `rejected` | Will not be pursued |

## Common Initial State

Almost all artifact types use `draft` as the initial state, providing consistency for:

- Workspace queries (`lifecycle_state: draft`)
- Agent prompts ("work on draft artifacts")
- Tooling assumptions

## Semantic Alignment

These conventions align with the semantic axes defined in `semantic-conventions.md`:

- Lifecycle states describe **artifact state**, not action outcomes
- The `cold` state corresponds to the `lifecycle_state: cold` defined in `_definitions.schema.json`
- Transitions are requested via `request_lifecycle_transition` and return `transition_result` (committed/rejected/deferred)
