# Validation Architecture

**Version**: 1.0.0
**Last Updated**: 2026-01-03
**Status**: Canonical

---

## Overview

This document describes QuestFoundry's approach to artifact validation, including:

1. **Schema-First Design** — JSON Schema as the source of truth
2. **Code Generation** — Pydantic models generated from schemas
3. **Validate-with-Feedback** — LLM-friendly error reporting for self-correction

---

## Schema-First Design

### Principle

JSON Schema files in `schemas/` are the **canonical definition** of artifact structure. The file format exists independently of program code.

```
schemas/                          # Source of Truth
├── dream.schema.json
├── brainstorm.schema.json
└── ...

src/questfoundry/artifacts/       # Generated from schemas
├── generated.py                  # Auto-generated Pydantic models
└── ...
```

### Rationale

1. **Format Independence** — Artifact files can be validated by any JSON Schema tool, not just Python
2. **Human-Editable** — Schemas are readable documentation of the file format
3. **Single Source of Truth** — No drift between schema and code
4. **Interoperability** — Other tools (editors, validators) can use the same schemas

### Workflow

When artifact structure needs to change:

```bash
# 1. Edit the JSON Schema (source of truth)
vim schemas/dream.schema.json

# 2. Regenerate Pydantic models
uv run python scripts/generate_models.py

# 3. Commit both schema and generated code
git add schemas/ src/questfoundry/artifacts/generated.py
git commit -m "feat(schema): add new field to dream artifact"
```

---

## Code Generation

### Tool: datamodel-code-generator

We use [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator) to generate Pydantic v2 models from JSON Schema.

```bash
# Generate models (run from project root)
uv run python scripts/generate_models.py
```

### Generated Code

The script produces `src/questfoundry/artifacts/generated.py` with:

- Pydantic v2 `BaseModel` classes
- Field constraints from JSON Schema (`minLength`, `minimum`, etc.)
- Type annotations matching schema types
- Docstrings from schema descriptions

### Why Generated Code is Checked In

Generated code is committed to version control because:

1. **Reviewability** — Changes to generated models are visible in PRs
2. **No Build Step** — No generation required at install time
3. **Stability** — Pinned to specific schema version
4. **IDE Support** — Type checking and autocomplete work immediately

---

## Validate-with-Feedback Pattern

### Problem

When LLM output fails validation, simple error messages don't help the model recover:

```json
// BAD: Vague, buried action
{
  "success": false,
  "error_count": 4,
  "errors": [...],
  "hint": "Review the errors above and try again."
}
```

The model often retries with the same mistake because:
- It doesn't know what specific corrections to make
- The action directive is buried at the end
- Field name typos aren't detected as correctable

### Solution: Action-First Structured Feedback

Return feedback designed for LLM self-correction:

```json
// GOOD: Action-first, semantic, correctable
{
  "action_outcome": "rejected",
  "rejection_reason": "validation_failed",
  "recovery_action": "Rename 2 field(s) and add 2 missing field(s), then retry.",
  "field_corrections": {
    "section_title": "rename to 'title'",
    "content": "rename to 'prose'"
  },
  "missing_required": ["anchor", "choices"],
  "error_count": 4,
  "errors": [...]
}
```

### Feedback Structure

| Field | Purpose |
|-------|---------|
| `action_outcome` | What happened: `"saved"` or `"rejected"` |
| `rejection_reason` | Why rejected: `"validation_failed"`, `"permission_denied"`, etc. |
| `recovery_action` | **First thing LLM reads** — clear directive for what to do |
| `field_corrections` | Fuzzy-matched field name fixes (typos, synonyms) |
| `missing_required` | Fields that must be added |
| `error_count` | Number of errors |
| `errors` | Detailed error list (for debugging, not primary guidance) |

### Fuzzy Field Matching

Common LLM mistakes are detected and corrected:

| Pattern | Example | Correction |
|---------|---------|------------|
| Suffix match | `section_title` → `title` | "rename to 'title'" |
| Prefix match | `title_text` → `title` | "rename to 'title'" |
| Synonyms | `content` → `prose` | "rename to 'prose'" |

### Implementation

The validate-with-feedback pattern is implemented in `src/questfoundry/validation/feedback.py`:

```python
from questfoundry.validation.feedback import ValidationFeedback

# Validate and get structured feedback
feedback = ValidationFeedback.from_validation_errors(
    errors=validation_errors,
    artifact_type="dream",
    provided_fields=set(data.keys()),
    required_fields={"genre", "tone", "audience", "themes"},
)

if not feedback.is_valid:
    return feedback.to_dict()  # LLM-friendly response
```

---

## Validation Flow

```
LLM Output
    │
    ▼
┌─────────────────┐
│  Parse YAML/    │
│  Tool Args      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Pydantic       │────→│  Validation     │
│  Validation     │     │  Errors         │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Success        │     │  Build          │
│  Return Result  │     │  Feedback       │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Return to LLM  │
                        │  for Retry      │
                        └─────────────────┘
```

### Retry Behavior

When validation fails:

1. **Don't terminate** — Return feedback to the LLM
2. **Continue loop** — Let the LLM correct and retry
3. **Limit retries** — Maximum 3 validation retries per tool call
4. **Escalate on max retries** — Return error with all feedback

---

## References

- v4 Implementation: [ARCHITECTURE-v3.md Section 9.4](https://github.com/pvliesdonk/questfoundry-v4/blob/main/_deprecated/ARCHITECTURE-v3.md#94-validate-with-feedback-pattern)
- v4 PR #227: [Improved validation feedback structure](https://github.com/pvliesdonk/questfoundry-v4/pull/227)
- JSON Schema: [json-schema.org](https://json-schema.org/)
- datamodel-code-generator: [GitHub](https://github.com/koxudaxi/datamodel-code-generator)

---

## See Also

- [02-artifact-schemas.md](./02-artifact-schemas.md) — Artifact schema definitions
- [13-project-structure.md](./13-project-structure.md) — File organization
