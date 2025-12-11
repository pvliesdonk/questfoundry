# Contributing to QuestFoundry Specification

This document provides guidelines for contributing to the QuestFoundry specification (Layers 0-5).

## Table of Contents

- [Overview](#overview)
- [Layer Structure](#layer-structure)
- [Editing Guidelines](#editing-guidelines)
- [Commit Conventions](#commit-conventions)
- [Quality Standards](#quality-standards)
- [Working with Implementation](#working-with-implementation)

## Overview

The `spec/` directory contains **documentation, schemas, and executable definitions** that define the QuestFoundry specification. This is NOT implementation code.

**Important:** Do not modify Python code in `lib/` when working on the specification. If implementation changes are needed, do them separately after spec changes are complete.

## Layer Structure

The specification is organized into 6 layers with clear separation of concerns:

| Layer | Directory | Purpose | Contents |
|-------|-----------|---------|----------|
| **Layer 0** | `00-north-star/` | Foundational principles | Principles, loops, quality bars, operating model |
| **Layer 1** | `01-roles/` | Role definitions | 15 studio roles (charters, briefs, interfaces) |
| **Layer 2** | `02-dictionary/` | Common language | Artifact templates, taxonomies, glossary |
| **Layer 3** | `03-schemas/` | JSON schemas | Machine validation schemas |
| **Layer 4** | `04-protocol/` | Communication protocol | Intents, lifecycles, message flows |
| **Layer 5** | `05-definitions/` | Executable definitions | Role profiles, loop patterns, prompt templates |

### Layer Responsibilities

- **L0** — Principles, policies, quality bars (the "why")
- **L1** — Role definitions (who)
- **L2** — Artifact structures (what)
- **L3** — JSON schemas (machine validation)
- **L4** — Protocol (how roles communicate)
- **L5** — Executable definitions (runtime agents)

## Editing Guidelines

### 1. Respect Layer Boundaries

Each layer has a specific purpose. When making changes:

- Keep foundational principles in Layer 0
- Define roles in Layer 1
- Document artifacts in Layer 2 (source of truth for data structures)
- Derive schemas in Layer 3 from Layer 2
- Specify communication patterns in Layer 4
- Define executable agents in Layer 5

### 2. Layer 2 is Source of Truth for Data Structures

**Critical rule:** When Layer 2 (human-readable templates) conflicts with Layer 3 (schemas), **Layer 2 wins**.

- Schemas in Layer 3 are derived from artifact templates in Layer 2
- Use HTML constraint comments in Layer 2 to guide schema generation
- Update Layer 2 first, then regenerate or update schemas in Layer 3

Example constraint comment in Layer 2:

```html
<!-- Field: status | Type: enum | Required: yes | Taxonomy: Hook Status Lifecycle (taxonomies.md §2) -->
```

### 3. Schema Validation

Any change to a schema in `03-schemas/` must be validated:

- Ensure the schema is valid JSON Schema Draft 2020-12
- Verify all examples and references are updated
- Update corresponding documentation in Layer 2
- Test schema validation with example data

### 4. Hot vs Cold Boundaries

**Never leak spoilers or internals from Hot to Cold surfaces.**

- **Hot**: Discovery space, drafts, spoilers allowed, internal use only
- **Cold**: Player-safe, curated canon, export-ready

See [00-north-star/SPOILER_HYGIENE.md](00-north-star/SPOILER_HYGIENE.md) for detailed rules.

### 5. Maintain Cross-References

When changing a concept, update all layers that reference it:

1. Search for the concept name across all layers
2. Update references in documentation, schemas, and prompts
3. Ensure consistency across the spec
4. Update layer README files if structure changes

### 6. Executable Definitions (Layer 5)

When editing definitions in `05-definitions/`:

- Ensure YAML validates against meta-schemas in `03-schemas/definitions/`
- Reference correct schemas from Layer 3
- Maintain consistency with role charters in Layer 1
- Follow the loop-focused architecture
- Test definitions with the runtime when possible

### 7. Protocol Documentation (Layer 4)

When adding or updating protocol flows:

- Document message sequences completely
- Specify intent types for all communications
- Define lifecycle state transitions clearly
- Provide example messages in `EXAMPLES/`
- Update conformance requirements in `CONFORMANCE.md`

## Commit Conventions

Use **custom conventional commit types** for specification work to trigger appropriate CI/CD workflows:

### `docs(spec): ...`

**Use for:** Documentation changes in Layers 0-2

**Examples:**

- Updating `00-north-star/WORKING_MODEL.md`
- Editing role charters in `01-roles/charters/`
- Modifying artifact documentation in `02-dictionary/`

**Version impact:** Does NOT trigger a spec version bump

### `schema(spec): ...`

**Use for:** Schema or protocol changes (Layers 3-4)

**Examples:**

- Adding/modifying schemas in `03-schemas/*.schema.json`
- Changing protocol definitions in `04-protocol/`
- Updating lifecycle state machines
- Modifying validation rules

**Version impact:** Triggers a spec version bump

### `prompt(spec): ...`

**Use for:** Executable definition changes (Layer 5)

**Examples:**

- Creating new role profiles in `05-definitions/roles/`
- Updating loop patterns in `05-definitions/loops/`
- Modifying prompt templates in `05-definitions/templates/`
- Changing quality gates or transitions

**Version impact:** Triggers a spec version bump

### Breaking Changes

Use `!` suffix for breaking changes:

```
schema(spec)!: change artifact ID format
```

Include `BREAKING CHANGE:` in the commit body with migration details.

### Important Notes

- `docs(spec)` = Documentation only, no version bump
- `schema(spec)` or `prompt(spec)` = Triggers automated spec releases
- For runtime implementation changes, use standard types like `feat(runtime)`, `fix(runtime)`

## Quality Standards

All specification files must meet these standards:

### Content Quality

- Use clear, precise language
- Be internally consistent within and across layers
- Define terms before using them (or reference glossary)
- Provide examples where helpful
- Explain the "why" not just the "what"

### Technical Quality

- Reference correct schema URIs
- Use proper JSON Schema syntax
- Follow YAML conventions for definitions
- Validate all structured data against schemas

### Format Quality

- Use correct Markdown formatting
- Follow existing document structure patterns
- Maintain consistent heading levels
- Use proper code blocks with language tags
- Keep line lengths reasonable

### Cross-Reference Quality

- Reference other layers accurately
- Use relative paths for internal links
- Update normative references when moving content
- Ensure section references are correct (e.g., "taxonomies.md §2")

## Working with Implementation

The Python runtime library in `lib/runtime/` bundles resources from this spec directory at build time:

### Build-Time Bundling

1. **Schemas** from `03-schemas/` are copied into the library package
2. **Definitions** from `05-definitions/` are loaded by the runtime
3. **Templates** from `05-definitions/templates/` are available to agents
4. The bundling script `lib/runtime/scripts/bundle_resources.py` performs the copy

### Runtime Loading

The runtime loads these resources using `importlib.resources`:

- Schemas for validation
- Role profiles for agent creation
- Loop patterns for graph construction
- Prompt templates for LLM interactions

### Critical Rule: Spec is Single Source of Truth

**Never manually edit bundled files** in `lib/runtime/src/questfoundry/resources/`:

1. Always edit files in `spec/`
2. Re-run the bundling script: `cd lib/runtime && uv run hatch run bundle`
3. Test with the runtime
4. Commit spec changes (not bundled files, which are gitignored)

## Essential Reading

Before making significant changes, review:

- [00-north-star/WORKING_MODEL.md](00-north-star/WORKING_MODEL.md) — How the studio operates
- [00-north-star/ROLE_INDEX.md](00-north-star/ROLE_INDEX.md) — The 15 internal roles
- [00-north-star/QUALITY_BARS.md](00-north-star/QUALITY_BARS.md) — The 8 quality criteria
- [02-dictionary/glossary.md](02-dictionary/glossary.md) — System terminology
- [02-dictionary/taxonomies.md](02-dictionary/taxonomies.md) — Classification systems
- Layer-specific README files for context

## File Organization

When adding new files:

- Place files in the appropriate layer directory
- Use clear, descriptive file names following existing conventions
- Update the layer's README file
- Add cross-references from related documents
- Follow existing naming patterns:
  - Markdown: `snake_case.md`
  - Schemas: `artifact_name.schema.json`
  - Definitions: `role_name.yaml` or `loop_name.yaml`

## Validation Checklist

Before submitting specification changes:

- [ ] Changes are in the correct layer
- [ ] Layer 2 templates updated before Layer 3 schemas (if applicable)
- [ ] All cross-references are updated
- [ ] Markdown formatting is correct
- [ ] JSON/YAML is valid and properly formatted
- [ ] Schemas validate against meta-schemas (Layer 5)
- [ ] No Hot content leaked to Cold surfaces
- [ ] Commit follows convention: `docs(spec)`, `schema(spec)`, or `prompt(spec)`
- [ ] Breaking changes are marked with `!` and documented

## Questions?

- Check layer README files for architecture guidance
- Review [00-north-star/](00-north-star/) for foundational principles
- See parent [../CONTRIBUTING.md](../CONTRIBUTING.md) for mono-repo guidelines
- Open an issue for questions or proposals

Thank you for contributing to the QuestFoundry specification!
