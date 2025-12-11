# Cold Store: Who Writes Where

## Three-Tier Storage Model

```
hot_store (drafts) -> cold_store (canon) -> Views (filtered exports)
```

| Store | Writers | Readers | Persistence |
|-------|---------|---------|-------------|
| hot_store | All roles | All roles | Session only |
| cold_store | **ONLY Lorekeeper (LK)** | All roles | Permanent (SQLite) |
| Views | Publisher (PB) | Players | Export artifacts |

## Mindset

> "Hot = discover & argue. Cold = agree & ship."

- hot_store: Working memory, mutable, internal, ephemeral
- cold_store: Committed canon, append-only, survives sessions

## Cold Store Tables

| Table | Content Types | Purpose |
|-------|--------------|---------|
| sections | scene | Narrative prose |
| codex | character, location, item, relationship | Player-safe encyclopedia |
| canon | canon_entry, event, fact, timeline | Internal world facts (can have spoilers) |
| acts | act | Story structure |
| chapters | chapter | Story structure |

## Critical Rule

**Only Lorekeeper calls `promote_to_canon()`.**

If testing cold_store writes, the workflow MUST reach Lorekeeper.
If cold_store is empty after a run, check if LK was invoked.

## Domain Reference

See `domain/ontology/artifacts.md` for the `store:` field on each artifact type:

- `store: hot` - Ephemeral only
- `store: cold` - Must be promoted
- `store: both` - Starts hot, promote when approved
