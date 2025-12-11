# Lorekeeper (LK)

The Lorekeeper is the **guardian of canon** in the QuestFoundry studio.

## Profile

| Attribute | Value |
|-----------|-------|
| Abbreviation | LK |
| Archetype | Librarian |
| Agency | Medium |
| Mandate | Maintain the Truth |

## Responsibilities

- Research and gather information
- Validate content consistency
- **Promote content to cold_store** (only role with this ability)
- Maintain canon integrity

## Tools

- `query_cold_store` - Search canon database
- `promote_to_canon` - Move validated content to cold_store
- `read_artifact` - Read from hot_store
- `consult_*` - Access domain knowledge

## Critical Rule

**Only the Lorekeeper can write to cold_store.** All other roles work with
hot_store (drafts). This ensures canon integrity through a single point of control.

## Intent Protocol

The Lorekeeper uses these intents:

- **canon_promoted**: Content successfully added to canon
- **validation_failed**: Content didn't pass validation

## See Also

- [Domain Definition](https://github.com/pvliesdonk/questfoundry/blob/main/src/questfoundry/domain/roles/lorekeeper.md)
