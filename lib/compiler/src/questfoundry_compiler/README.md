# QuestFoundry Spec Compiler

## Overview

The QuestFoundry Spec Compiler is responsible for transforming atomic behavior primitives from `../../../../spec/05-behavior/` into runtime-ready artifacts. It validates cross-references, assembles prompts, and generates JSON manifests for execution.

This is the internal implementation of the `questfoundry-compiler` package, located at `lib/compiler/src/questfoundry_compiler/` in the QuestFoundry mono-repo.

## Compiler Pipeline

```
[Atomic Sources] → [Loader] → [Validator] → [Assembler] → [Manifest Builder] → [Output Writer]
     (YAML/MD)      (parse)     (refs)       (compose)       (JSON)            (dist/)
```

## Components

### 1. Loader (`spec_compiler.py`)

Parses YAML frontmatter and markdown content from behavior primitives:

- **Expertises**: Domain-specific knowledge (e.g., `lore_weaver_expertise.md`)
- **Procedures**: Reusable workflow steps (e.g., `canonization_core.md`)
- **Snippets**: Small text blocks (e.g., `spoiler_hygiene_reminder.md`)
- **Playbooks**: Loop definitions (e.g., `lore_deepening.playbook.yaml`)
- **Adapters**: Role configurations (e.g., `lore_weaver.adapter.yaml`)

### 2. Validator (`validators.py`)

Validates all cross-references and schema compliance:

- Expertise references resolve to actual files
- Schema references point to valid L3 schemas
- Role references match L1 role definitions
- No circular dependencies
- No orphaned files (every primitive is referenced)

### 3. Assembler (`assemblers.py`)

Composes prompts by resolving references:

- **Reference Syntax**:
  - `@expertise:lore_weaver_expertise` → Inject full expertise content
  - `@procedure:canonization_core` → Inject full procedure
  - `@procedure:canonization_core#step1` → Inject specific section
  - `@snippet:spoiler_hygiene_reminder` → Inject snippet
  - `@playbook:lore_deepening` → Reference (link, don't inline)
  - `@schema:canon_pack.schema.json` → Reference with validation

### 4. Manifest Builder (`manifest_builder.py`)

Generates JSON runtime manifests for playbooks and adapters:

- Converts YAML to runtime-ready JSON
- Includes all metadata for execution
- Embeds assembled procedure content
- Validates against manifest schemas

### 5. Output Writer

Writes compiled artifacts to `dist/compiled/`:

- **Manifests**: `dist/compiled/manifests/*.manifest.json`
- **Standalone Prompts**: `dist/compiled/standalone_prompts/*_full.md`

## Usage

### Compile All

```bash
qf-compile --spec-dir spec/ --output dist/compiled/
```

### Compile Specific Playbook

```bash
qf-compile --playbook lore_deepening --output dist/compiled/
```

### Validate Only

```bash
qf-compile --validate-only
```

### Watch Mode

```bash
qf-compile --watch
```

## Cross-Reference Syntax

See [../../../../spec/05-behavior/README.md](../../../../spec/05-behavior/README.md) for complete reference syntax specification.

## Output Artifacts

### Playbook Manifest

Example: `dist/compiled/manifests/lore_deepening.manifest.json`

```json
{
  "$schema": "https://questfoundry.liesdonk.nl/manifests/playbook_manifest.schema.json",
  "manifest_version": "2.0.0",
  "playbook_id": "lore_deepening",
  "display_name": "Lore Deepening",
  "compiled_at": "2025-11-13T10:30:00Z",
  "steps": [...],
  "raci": {...},
  "quality_bars": [...]
}
```

### Standalone Prompt

Example: `dist/compiled/standalone_prompts/lore_weaver_full.md`

Complete assembled prompt for a role, composing:

1. Role charter (from adapter)
2. Referenced expertises (full content)
3. Protocol intents (from adapter)
4. Referenced snippets (validation, safety protocols)
5. Loop participation (summary with links)

## Error Handling

The compiler provides detailed error messages for:

- **Invalid References**: References that don't resolve to files
- **Circular Dependencies**: Detected via dependency graph analysis
- **Orphaned Files**: Primitives not referenced by any playbook/adapter
- **Schema Violations**: YAML that doesn't match expected structure
- **Missing Sections**: Section anchors that don't exist (e.g., `#step1`)

## Development

### Running Tests

From the `lib/compiler/` directory:

```bash
pytest tests/
```

### Adding New Primitive Types

1. Update loader to parse new type
2. Add validation rules in `validators.py`
3. Implement assembly logic in `assemblers.py`
4. Update manifest schema if needed
5. Add tests

## Architecture Decisions

### Why Atomic Primitives?

- **Single Source of Truth**: Each piece of logic exists in exactly one place
- **Maintainability**: Update once, propagate everywhere
- **Composability**: Mix and match primitives for new use cases
- **Validation**: Catch broken references at compile time

### Why Compile-Time Assembly?

- **Runtime Performance**: No parsing/assembly overhead
- **Error Detection**: Catch issues before deployment
- **Deterministic Output**: Same inputs always produce same outputs
- **Offline Capability**: Compiled artifacts work without source specs
