# Schema Hydration Process

## Overview

This document describes the schema hydration system for QuestFoundry. The hydration process converts stub schemas (incomplete placeholder schemas) into complete JSON Schema Draft 2020-12 compliant schemas by extracting definitions from artifact documentation.

## Purpose

The schema hydration system:

- Automates schema generation from human-readable artifact documentation
- Ensures consistency between Layer 2 (artifact templates) and Layer 3 (schemas)
- Reduces manual schema maintenance burden
- Validates that schemas match their source documentation

## Process

### 1. Stub Detection

A schema is considered a "stub" if it meets **any** of these criteria:

- Contains `"description": "... (stub ...)"` text
- Has **0 properties** in the `properties` object

Note: The detection uses 0 properties (not ≤1) to avoid false positives with complete schemas that have a single top-level array field.

### 2. Documentation Mapping

For each stub schema `{name}.schema.json`, the hydrator looks for documentation in:

1. `spec/02-dictionary/artifacts/{name}.md` (exact match)
2. `spec/02-dictionary/artifacts/{name}_ENRICHED.md` (enriched version)

If no documentation is found, the schema is skipped with a "Missing Definition" log entry.

### 3. Schema Generation

When documentation is found, the hydrator:

1. Extracts title from the first H1 heading
2. Extracts description from the `> **Use:**` status block
3. Parses field definitions from the "Field Definitions" section
4. Generates JSON Schema properties with appropriate types and constraints
5. Adds standard boilerplate fields (manifest_version, project, created, last_updated)
6. Sets `additionalProperties: false` for strictness

### 4. Type Mapping

| Markdown Source | JSON Schema Result |
|----------------|-------------------|
| "String, required" | `"type": "string"` (Add to `required` list) |
| "Enum: 'A', 'B'" | `"type": "string", "enum": ["A", "B"]` |
| "ISO 8601" | `"format": "date-time"` |
| "SHA-256" | `"pattern": "^[a-f0-9]{64}$"` |
| "YYYY-MM-DD" | `"format": "date", "pattern": "^\d{4}-\d{2}-\d{2}$"` |
| "Array of items" | `"type": "array", "items": { ... }` |
| "Object" | `"type": "object"` |

## Usage

### Running the Hydrator

```bash
python hydrate_schemas.py
```

This will:

- Scan all `*.schema.json` files in `spec/03-schemas/`
- Identify stubs vs complete schemas
- Attempt to hydrate each stub from corresponding markdown documentation
- Report results with detailed logging

### Output

The script outputs:

- **HYDRATE**: `{file}` ← `{markdown}` - Successfully hydrated a stub
- **SKIP**: `{file}` (already complete) - Schema is not a stub
- **SKIP**: `{file}` (no markdown documentation found) - Stub has no documentation

Final summary includes counts for:

- Hydrated schemas
- Skipped (already complete)
- Skipped (no documentation)
- Total processed

## Current Repository State

As of 2025-11-24:

- **60** total schema files
- **28** complete schemas (not stubs)
- **32** stub schemas
- **0** stubs with available documentation
- **22** complete schemas with documentation files

### Complete Schemas (with documentation)

These schemas are complete and have corresponding artifact documentation:

- art_manifest, art_plan, audio_plan, canon_pack, canon_transfer_package
- codex_entry, cuelist, edit_notes, front_matter, gatecheck_report
- hook_card, language_pack, pn_playtest_notes, project_metadata
- register_map, research_memo, shotlist, style_addendum, style_manifest
- tu_brief, view_log, world_genesis_manifest

### Complete Schemas (without documentation)

These are generated/derived schemas:

- cold_art_manifest, cold_book, cold_build_lock, cold_fonts, cold_manifest
- hot_manifest

### Stub Schemas (all without documentation)

These 32 stubs await documentation before they can be hydrated:

- action_items, anchor_map, archive_manifest, art_render, audio_render
- bilingual_glossary, canon_summary, choice, coverage_report, crosslink_map
- determinism_log, friction_report, gateway_map, glossary, harvest_sheet
- hook, merge_approval, message_envelope, phrasing_patterns, playtest_notes
- post_mortem_report, safety_checklist, scene, section, section_brief
- snapshot, topology_notes, translation_pack, tu_checkpoint
- uncertainty_assessment, view_bundle, view_export

## Future Work

To hydrate stub schemas:

1. Create artifact documentation files in `spec/02-dictionary/artifacts/`
2. Follow the structure of existing artifact docs (e.g., `art_manifest.md`)
3. Include "Field Definitions" section with properly formatted field entries
4. Run `python hydrate_schemas.py` to generate schemas

## Schema Structure

All generated schemas follow this structure:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://questfoundry.liesdonk.nl/schemas/{name}.schema.json",
  "title": "Extracted from markdown",
  "description": "Generated from 02-dictionary/artifacts/{name}.md. {description}",
  "type": "object",
  "properties": {
    "manifest_version": { "type": "string", "const": "1.0" },
    "project": { "type": "string", "minLength": 1, "maxLength": 200 },
    "created": { "type": "string", "format": "date-time" },
    "last_updated": { "type": "string", "format": "date-time" },
    "// ... additional fields from documentation"
  },
  "required": ["manifest_version", "project", "created", "last_updated", "..."],
  "additionalProperties": false
}
```

## References

- Problem Statement: Agent Task: QuestFoundry Schema Hydration (Smart Mode)
- Gold Standard: `spec/03-schemas/art_manifest.schema.json`
- Documentation: `spec/02-dictionary/artifacts/`
- JSON Schema: <https://json-schema.org/draft/2020-12/schema>

## Validation

After hydration, schemas should be validated:

1. JSON syntax validation
2. JSON Schema meta-schema compliance (Draft 2020-12)
3. Pre-commit hooks (`pre-commit run --all-files`)
4. Schema-specific validation against example data

## Notes

- The hydrator preserves existing complete schemas (never overwrites)
- All generated schemas use JSON Schema Draft 2020-12
- Standard fields ensure consistency across all schemas
- `additionalProperties: false` enforces strict validation
- Hot/Cold separation must be respected in schema constraints
