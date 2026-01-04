# Schema-First Model Generation

## Overview

JSON Schemas in `schemas/` are the **source of truth** for artifact structure in QuestFoundry. Pydantic models are generated from these schemas, not hand-written.

This approach ensures:
- No drift between validation schema and runtime models
- External tool compatibility (any JSON Schema validator works)
- Human-readable documentation of data formats
- CI enforces consistency automatically

## Directory Structure

```
schemas/
  dream.schema.json      # JSON Schema for DREAM artifact
  brainstorm.schema.json # (future)
  seed.schema.json       # (future)
  ...

src/questfoundry/artifacts/
  generated.py           # AUTO-GENERATED from schemas
  __init__.py            # Re-exports from generated.py

scripts/
  generate_models.py     # Generator script
```

## Workflow

### Modifying Artifact Structure

1. **Edit the JSON Schema** in `schemas/<artifact>.schema.json`
2. **Regenerate models**: `uv run python scripts/generate_models.py`
3. **Commit both files**: schema changes AND generated.py

```bash
# Example: Adding a field to DreamArtifact
vim schemas/dream.schema.json          # Add your field
uv run python scripts/generate_models.py
git add schemas/dream.schema.json src/questfoundry/artifacts/generated.py
git commit -m "feat(artifacts): add new_field to DreamArtifact"
```

### CI Drift Detection

The CI pipeline automatically checks that `generated.py` matches the schemas:

```yaml
- name: Check generated code is up-to-date
  run: |
    uv run python scripts/generate_models.py
    git diff --exit-code src/questfoundry/artifacts/generated.py
```

If you forget to regenerate after schema changes, CI will fail with a diff showing what's out of sync.

## Schema Guidelines

### Required vs Optional Fields

- Mark fields as `required` in schema only if Pydantic model requires them
- Use `scripts/check_schema_model_sync.py` to verify sync

### Constraints

JSON Schema constraints map to Pydantic Field parameters:
- `minLength: 1` -> `Field(min_length=1)`
- `minimum: 5` -> `Field(ge=5)`
- `maxLength: 100` -> `Field(max_length=100)`

### Nested Objects

Nested objects in schemas become nested Pydantic models:

```json
{
  "scope": {
    "type": "object",
    "properties": {
      "target_word_count": {"type": "integer", "minimum": 1000}
    }
  }
}
```

Generates:

```python
class Scope(BaseModel):
    target_word_count: int = Field(ge=1000)

class DreamArtifact(BaseModel):
    scope: Scope | None = None
```

## Why Schema-First?

### Problem: Hand-Maintained Drift

When schemas and models are maintained separately, they drift:
- Developer updates model, forgets schema
- Schema has different constraints than model
- External tools validate against wrong rules

### Solution: Single Source of Truth

With schema-first generation:
- Schema IS the specification
- Models are derived, never edited directly
- CI catches any drift immediately

### Benefits

1. **Correctness**: Models always match schema
2. **Visibility**: Schemas are human-readable documentation
3. **Tooling**: Any JSON Schema tool works (IDE validation, external validators)
4. **Automation**: No manual sync, no review burden

## Related Documentation

- [ADR-006: Schema-First Model Generation](decisions.md#adr-006-schema-first-model-generation)
- [02-artifact-schemas.md](../design/02-artifact-schemas.md) - Artifact format design
