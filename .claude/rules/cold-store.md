# Cold Store: Design Principles

> **Note**: Runtime is being rebuilt. These are design principles from `meta/`, not current implementation.

## Storage Model (from meta/schemas/core/store.schema.json)

Stores have `semantics`:

- `hot` — Working memory, mutable, ephemeral
- `cold` — Committed canon, append-only, permanent
- `versioned` — Immutable snapshots for export
- `ephemeral` — Session-only scratch space

## Lifecycle States (from meta/)

Artifacts have `_lifecycle_state`:

- `draft` — Initial creation
- `review` — Pending validation
- `approved` — Passed quality gates
- `cold` — Committed to canon

**Key insight**: "Cold" is a lifecycle state, not a separate type. Any artifact can become cold.

## Write Permissions

From `domain-v4/stores/*.json`:

| Store | Exclusive Writer | Semantics |
|-------|-----------------|-----------|
| canon | Lorekeeper | cold |
| codex | Codex Curator | cold |
| workspace | (all agents) | hot |
| scratch | (all agents) | ephemeral |
| exports | Book Binder | versioned |

## Critical Rule

**Lorekeeper owns lifecycle transitions to `cold`.**

The runtime must enforce this via the `request_lifecycle_transition` tool.

## Domain Reference

See `domain-v4/stores/` for store definitions.
See `meta/schemas/core/store.schema.json` for the schema.
