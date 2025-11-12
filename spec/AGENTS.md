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
- **Layer 5** (`05-prompts/`) — AI agent prompts (loop playbooks, role prompts)

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

When editing prompts in `05-prompts/`:

- Follow the loop-focused architecture
- Ensure prompts reference the correct schemas from `03-schemas/`
- Maintain consistency with role charters in `01-roles/`
- Test prompts with actual AI agents when possible

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

The Python library in `lib/python/` loads resources from this spec directory:

- Schemas are loaded from `03-schemas/`
- Prompts are loaded from `05-prompts/`

**Never duplicate these resources** into the library code. The spec is the single source of truth.
