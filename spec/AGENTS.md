# Specification Editing Guidelines

This file contains rules for working on the QuestFoundry specification (Layers 0-5).

## Scope

When working in the `spec/` directory, you are editing **documentation, schemas, and prompts** that
define the QuestFoundry specification. This is NOT implementation code.

**Do not modify Python code** in `lib/` when working on the specification. If implementation changes
are needed, do them separately after spec changes are complete.

## Layer Structure

The specification is organized into layers:

- **Layer 0** (`00-north-star/`) — Foundational principles, loops, quality bars
- **Layer 1** (`01-roles/`) — 15 studio roles (charters, briefs)
- **Layer 2** (`02-dictionary/`) — Common language (artifacts, taxonomies, glossary)
- **Layer 3** (`03-schemas/`) — JSON schemas (validation)
- **Layer 4** (`04-protocol/`) — Communication protocol (intents, lifecycles, flows)
- **Layer 5** (`05-behavior/`) — Atomic behavior primitives (expertises, procedures, playbooks, adapters)

## Editing Rules

### 1. Layer Boundaries

Respect the separation of concerns:

- **L0** = principles, policies, quality bars
- **L1** = role definitions (who)
- **L2** = artifact structures (what)
- **L3** = JSON schemas (machine validation)
- **L4** = protocol (how roles communicate)
- **L5** = AI prompts (executable agents)

### 2. L2 is Source of Truth

When Layer 2 (human-readable templates) conflicts with Layer 3 (schemas), **Layer 2 wins**. Schemas
are derived from L2.

### 3. Schema Validation

Any change to a schema in `03-schemas/` must be validated:

- Ensure the schema is valid JSON Schema
- Verify all examples and references are updated
- Update corresponding documentation in Layer 2

### 4. Hot vs Cold

Never leak spoilers or internals from Hot to Cold surfaces. See `00-north-star/SPOILER_HYGIENE.md`
for details.

### 5. Cross-References

When changing a concept, update all layers that reference it:

- Search for the concept name across all layers
- Update references in documentation, schemas, and prompts
- Ensure consistency across the spec

### 6. Prompt Engineering

When editing behavior primitives in `05-behavior/`:

- Follow the loop-focused architecture
- Ensure prompts reference the correct schemas from `03-schemas/`
- Maintain consistency with role charters in `01-roles/`
- Test prompts with actual AI agents when possible

## Conventional Commit Types for Spec Changes

When working in the `spec/` directory, use these **custom conventional commit types** to trigger the appropriate CI/CD workflows:

### `docs(spec): ...`

- **Use for**: Documentation changes in Layers 0-2 (principles, roles, dictionary)
- **Examples**:
  - Updating `00-north-star/WORKING_MODEL.md`
  - Editing role charters in `01-roles/charters/`
  - Modifying artifact documentation in `02-dictionary/`
- **Version impact**: Does NOT trigger a spec version bump
- **Workflow**: Does not trigger `release-spec.yml`

### `schema(spec): ...`

- **Use for**: Any changes to schemas (Layer 3) or protocol (Layer 4)
- **Examples**:
  - Adding/modifying schemas in `03-schemas/*.schema.json`
  - Changing protocol definitions in `04-protocol/`
  - Updating schema validation rules
- **Version impact**: Triggers a spec version bump
- **Workflow**: Triggers `release-spec.yml` workflow

### `prompt(spec): ...`

- **Use for**: Any changes to AI prompts (Layer 5)
- **Examples**:
  - Creating new behavior primitives in `05-behavior/expertises/`, `procedures/`, or `snippets/`
  - Updating playbooks in `05-behavior/playbooks/`
  - Modifying role adapters in `05-behavior/adapters/`
  - Changing loop playbooks
- **Version impact**: Triggers a spec version bump
- **Workflow**: Triggers `release-spec.yml` workflow

### Important Notes

- Use `schema(spec)` or `prompt(spec)` to trigger automated spec releases
- Documentation-only changes use `docs(spec)` and don't create new versions
- These types are specific to the spec/ workspace
- For Python library changes, use standard types: `feat`, `fix`, `refactor`, etc.

## Essential Reading

Before making significant changes, review:

- `00-north-star/WORKING_MODEL.md` — How the studio operates
- `00-north-star/ROLE_INDEX.md` — The 15 internal roles
- `00-north-star/QUALITY_BARS.md` — The 8 quality criteria
- Layer-specific README files for context

## File Organization

- Keep files organized within their appropriate layer
- Use clear, descriptive file names
- Follow existing naming conventions
- Update README files when adding new files

## Quality Standards

All specification files must:

- Use clear, precise language
- Be internally consistent
- Have correct markdown formatting
- Reference other layers accurately
- Be free of typos and grammatical errors

## Working with the Implementation

The Python library in `lib/python/` bundles resources from this spec directory at build time:

- Schemas from `03-schemas/` are bundled into the library package
- The spec compiler (`../../lib/compiler/`) transforms behavior primitives into runtime manifests
- Compiled manifests are bundled into packages at build time
- The bundling script `lib/python/scripts/bundle_resources.py` copies files from spec/

**The spec is the single source of truth**. Never manually edit files in `lib/python/src/questfoundry/resources/` - always edit files in `spec/` and re-run the bundling script.
