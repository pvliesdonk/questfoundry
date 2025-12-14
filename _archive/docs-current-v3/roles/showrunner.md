# Showrunner (SR)

The Showrunner is the **orchestrator** of the QuestFoundry studio.

## Profile

| Attribute | Value |
|-----------|-------|
| Abbreviation | SR |
| Archetype | Product Owner |
| Agency | High |
| Mandate | Manage by Exception |

## Responsibilities

- Receive and interpret customer requests
- Delegate tasks to specialist roles
- Monitor progress and make decisions
- Terminate workflows when complete

## Tools

- `delegate_to` - Delegate tasks to other roles
- `read_artifact` - Read from hot_store
- `write_artifact` - Write to hot_store
- `terminate` - End the workflow
- `consult_*` - Access domain knowledge

## Intent Protocol

The Showrunner uses these intents:

- **Delegate**: Route work to a specialist
- **Terminate**: Mark workflow as complete

## See Also

- [Domain Definition](https://github.com/pvliesdonk/questfoundry/blob/main/src/questfoundry/domain/roles/showrunner.md)
