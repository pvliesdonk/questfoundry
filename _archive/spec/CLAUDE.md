# Claude Code Guidelines (spec/)

This file provides Claude Code-specific guidance for work in `spec/` (Layers 0–5 and runtime interface docs). See also `AGENTS.md` and `CONTRIBUTING.md`.

## Golden Rules

1. **spec/ is canonical**: Do not modify implementation code here
2. **Do not hand-bundle**: Don't manually create bundled resources; scripts handle that
3. **Layer boundaries matter**: Keep content in the correct layer
4. **Layer 2 → Layer 3**: Human-readable templates (L2) outrank schemas (L3); fix schemas to match templates

## The Seven Layers

| Layer | Content | Notes |
|-------|---------|-------|
| L0 | Principles, policies | Foundational rules |
| L1 | Roles, their behaviors | Role definitions |
| L2 | Human-readable templates, taxonomies | Player-facing; outranks L3 |
| L3 | Schemas (JSON, YAML, protocol) | Validation; must match L2 |
| L4 | Protocol (intents, envelopes, flows) | Message structure and choreography |
| L5 | Executable definitions, prompts | Runnable specifications |
| L6 | Runtime interface docs | How implementations use bundled resources |

## Before Editing

1. Read the layer's README for context
2. Check nearby docs to understand relationships
3. Read `02-dictionary/` for consistent terminology
4. Validate changes against meta-schemas in `03-schemas/definitions/`

## Hot vs. Cold

- **Hot**: Internal details, implementation spoilers, non-player-facing
- **Cold**: Player-facing, user-visible, safe to leak
- **Rule**: Never leak Hot into Cold

## Making Changes

### Documentation (L0–L2)

```
docs(spec): [description]
```

### Schemas & Protocol (L3–L4)

```
schema(spec): [description]
```

Triggers spec version bump.

### Definitions & Templates (L5)

```
prompt(spec): [description]
```

Triggers spec version bump.

### Breaking Changes

Use `!` and include migration notes:

```
schema(spec)!: [breaking change description]

MIGRATION:
- Old way: ...
- New way: ...
```

## Definition of Done

- ✅ Changes in correct layer
- ✅ Layer 2 → Layer 3 derivation respected
- ✅ Cross-references updated across layers
- ✅ Hot/Cold boundaries maintained
- ✅ JSON/YAML validated against schemas
- ✅ Formatting consistent
- ✅ Commit uses correct `docs/spec`, `schema/spec`, or `prompt/spec` type

## Tips

- Keep terminology consistent with `02-dictionary/`
- Update all cross-references when terms change
- Preserve traceability: if L2 changes, cascade updates to L3 and beyond
- Avoid high-maintenance artifacts; prefer existing structures
- Use automation where possible (bundling, generation, validation)

## Common Tasks

### Adding a New Artifact Type

1. Define in `01-roles/` (L1) — role behavior
2. Create template in `02-dictionary/` (L2) — human-readable example
3. Add schema in `03-schemas/` (L3) — validation rules
4. Reference in protocol flows (L4) if message-based
5. Create example in `05-definitions/` (L5)
6. Update docs in `06-runtime/` (L6) if implementations need guidance

### Updating a Protocol Flow

1. Update flow diagram in `04-protocol/FLOWS/` (L4)
2. Update intent definitions in `04-protocol/INTENTS.md` if needed
3. Add/update envelope schema examples
4. Update implementation docs in `06-runtime/`

### Fixing a Schema-Template Mismatch

1. Check L2 template (source of truth)
2. Update L3 schema to match
3. Validate against examples
4. Update `02-dictionary/` glossary if terminology changed
5. Commit as `schema(spec): fix [artifact] schema to match template`
