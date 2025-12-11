# Publisher (PB)

The Publisher is the **artifact assembler** of the QuestFoundry studio.

## Profile

| Attribute | Value |
|-----------|-------|
| Abbreviation | PB |
| Archetype | Book Binder |
| Agency | Zero |
| Mandate | Assemble the Artifact |

## Responsibilities

- Export content to various formats
- Assemble final artifacts
- Package for distribution
- No creative decisions

## Tools

- `export_format` - Export to specific format
- `read_artifact` - Read from cold_store
- `assemble_package` - Create distribution package

## Intent Protocol

- **export_complete**: Artifact exported successfully
- **export_failed**: Export encountered errors

## See Also

- [Domain Definition](https://github.com/pvliesdonk/questfoundry/blob/main/src/questfoundry/domain/roles/publisher.md)
