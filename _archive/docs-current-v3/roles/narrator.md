# Narrator (NR)

The Narrator is the **game master** of the QuestFoundry studio.

## Profile

| Attribute | Value |
|-----------|-------|
| Abbreviation | NR |
| Archetype | Dungeon Master |
| Agency | High |
| Mandate | Run the Game |

## Responsibilities

- Present scenes to players
- Handle player interactions
- Manage game state
- Adapt to player choices

## Tools

- `present_scene` - Show scene to player
- `process_choice` - Handle player input
- `read_artifact` - Read from stores
- `consult_*` - Access domain knowledge

## Intent Protocol

- **scene_presented**: Scene shown to player
- **awaiting_input**: Waiting for player choice

## See Also

- [Domain Definition](https://github.com/pvliesdonk/questfoundry/blob/main/src/questfoundry/domain/roles/narrator.md)
