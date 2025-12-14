# Plan: V4-Native Domain Architecture

## Problem Statement

The runtime currently depends on `generated/` which is compiled from v3 domain (MyST markdown). This creates confusion because:

1. `generated/` contains v3 domain knowledge that doesn't match v4
2. `runtime/domain/models.py` was written for v3 and is misaligned with `meta/` schemas
3. Domain knowledge is hardcoded in runtime instead of loaded from domain-v4

## Architecture Principles

1. **`meta/` defines the metamodel** - The structure of agents, artifact-types, playbooks, stores, etc.
2. **`domain-v4/` contains domain instances** - Specific agents, artifacts, playbooks for QuestFoundry
3. **Runtime loads, never hardcodes** - All domain knowledge comes from domain-v4 at runtime
4. **Metamodel is stable** - Python code in runtime matches `meta/` schemas (rarely changes)
5. **Artifact models are dynamic** - Generated from `ArtifactType` definitions at runtime or cached

## Current State

```
meta/schemas/core/           # Metamodel (JSON Schema) - SOURCE OF TRUTH for structure
  agent.schema.json
  artifact-type.schema.json
  playbook.schema.json
  store.schema.json
  ...

domain-v4/                   # Domain instances - SOURCE OF TRUTH for content
  studio.json
  agents/*.json
  artifact-types/*.json
  playbooks/*.json
  stores/*.json

src/questfoundry/
  generated/                 # v3 compiled code - TO BE REMOVED
  runtime/
    domain/
      models.py              # v3 metamodel - NEEDS REWRITE
      loader.py              # Partially works
      artifact_models.py     # Started but incomplete
```

## Target State

```
meta/schemas/core/           # Metamodel (unchanged)

domain-v4/                   # Domain instances (unchanged)

src/questfoundry/
  runtime/
    domain/
      __init__.py            # Public API: load_studio(), get_artifact_model()
      metamodel.py           # NEW: Pydantic models matching meta/ exactly
      loader.py              # Load domain-v4 into metamodel types
      artifact_compiler.py   # NEW: ArtifactType -> Pydantic model
      _cache/                # Optional: cached compiled models (gitignored)
```

## Implementation Phases

### Phase 1: Rewrite Metamodel (match meta/ exactly)

Create `runtime/domain/metamodel.py` with Pydantic models that exactly match `meta/schemas/core/`:

```python
# metamodel.py - Matches meta/schemas/core/*.schema.json

from enum import Enum
from pydantic import BaseModel

# Enums from _definitions.schema.json
class Archetype(str, Enum):
    ORCHESTRATOR = "orchestrator"
    CREATOR = "creator"
    VALIDATOR = "validator"
    RESEARCHER = "researcher"
    CURATOR = "curator"

class StoreSemantics(str, Enum):
    APPEND_ONLY = "append_only"
    MUTABLE = "mutable"
    VERSIONED = "versioned"
    COLD = "cold"

class FieldType(str, Enum):
    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    # ... etc

# Models from individual schema files
class FieldDefinition(BaseModel):
    """Matches _definitions.schema.json#/$defs/field_definition"""
    name: str
    type: FieldType
    description: str | None = None
    required: bool = False
    default: Any = None
    enum: list[str] | None = None
    format: str | None = None
    min: float | None = None
    max: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    # ... all fields from meta/

class ArtifactType(BaseModel):
    """Matches artifact-type.schema.json"""
    id: str
    name: str
    description: str | None = None
    category: str = "document"
    fields: list[FieldDefinition] = []
    json_schema_override: dict | None = None
    lifecycle: Lifecycle | None = None
    validation: ValidationRules | None = None
    default_store: str | None = None
    extends: str | None = None

# ... Agent, Capability, Constraint, Playbook, Store, etc.
```

**Deliverable**: `metamodel.py` with 100% coverage of `meta/schemas/core/`

### Phase 2: Create Artifact Compiler

Create `runtime/domain/artifact_compiler.py` that generates Pydantic models from `ArtifactType`:

```python
# artifact_compiler.py

from pydantic import BaseModel, Field, create_model
from .metamodel import ArtifactType, FieldDefinition, FieldType

def compile_artifact_type(artifact_type: ArtifactType) -> type[BaseModel]:
    """Generate a Pydantic model from an ArtifactType definition."""

    field_definitions = {}
    for field in artifact_type.fields:
        python_type = _field_type_to_python(field)
        field_info = _create_field_info(field)
        field_definitions[field.name] = (python_type, field_info)

    return create_model(
        artifact_type.name.replace(" ", ""),
        **field_definitions
    )

def _field_type_to_python(field: FieldDefinition) -> type:
    """Map FieldType to Python type, handling enums, nested objects, arrays."""
    if field.enum:
        # Create a dynamic Enum
        return Enum(field.name, {v.upper(): v for v in field.enum})

    type_map = {
        FieldType.STRING: str,
        FieldType.TEXT: str,
        FieldType.INTEGER: int,
        FieldType.NUMBER: float,
        FieldType.BOOLEAN: bool,
        # ...
    }
    # Handle array, object recursively
    ...
```

**Deliverable**: `artifact_compiler.py` that handles all `FieldType` variants including nested objects

### Phase 3: Update Loader

Update `runtime/domain/loader.py` to use the new metamodel:

```python
# loader.py

from pathlib import Path
from .metamodel import Studio, Agent, ArtifactType, Playbook, Store
from .artifact_compiler import compile_artifact_type

_studio_cache: dict[Path, Studio] = {}
_artifact_model_cache: dict[str, type[BaseModel]] = {}

def load_studio(studio_path: Path) -> Studio:
    """Load domain-v4/studio.json into metamodel types."""
    if studio_path in _studio_cache:
        return _studio_cache[studio_path]

    # Load and parse
    studio = _load_and_resolve(studio_path)
    _studio_cache[studio_path] = studio
    return studio

def get_artifact_model(artifact_type_id: str) -> type[BaseModel]:
    """Get compiled Pydantic model for an artifact type."""
    if artifact_type_id in _artifact_model_cache:
        return _artifact_model_cache[artifact_type_id]

    studio = load_studio(_default_studio_path())
    artifact_type = studio.artifact_types[artifact_type_id]
    model = compile_artifact_type(artifact_type)
    _artifact_model_cache[artifact_type_id] = model
    return model
```

**Deliverable**: Unified loader API that provides both metamodel objects and compiled artifact models

### Phase 4: Update Runtime Consumers

Update all code that imports from `generated/` to use the new domain loader:

| File | Current Import | New Import |
|------|---------------|------------|
| `cold_store.py` | `from generated.models.artifacts import Choice, Gate` | `get_artifact_model("section")` for nested types |
| `cold_store.py` | `from generated.models.enums import Visibility` | String literals or load from domain |
| `validation.py` | `from generated.models.artifacts import ARTIFACT_REGISTRY` | `studio.artifact_types` |
| `role.py` | `from generated.models.artifacts import Act, Scene...` | `get_artifact_model()` |
| `sr.py` | `from generated.roles import ALL_ROLES` | `studio.agents` |
| `resources.py` | `from generated import roles, loops, models` | Remove or redirect to loader |

**Key changes:**

- Replace `Visibility` enum with string literals (domain defines valid values)
- Replace `PROMOTABLE_ARTIFACTS` with query: stores where `semantics == "cold"`
- Replace `ALL_ROLES`/`ALL_LOOPS` with `studio.agents`/`studio.playbooks`

### Phase 5: Remove v3 Artifacts

1. Delete `src/questfoundry/generated/` directory
2. Delete or archive `src/questfoundry/runtime/domain/models.py` (replaced by `metamodel.py`)
3. Update `pyproject.toml` if needed
4. Update tests

### Phase 6: Documentation

1. Add docstrings linking metamodel classes to `meta/` schema files
2. Update ARCHITECTURE.md with new domain loading flow
3. Add examples in `runtime/domain/__init__.py`

## Migration Strategy

To avoid breaking changes during migration:

1. Create new `metamodel.py` alongside existing `models.py`
2. Update loader to use `metamodel.py`
3. Update consumers one at a time, testing each
4. Once all consumers migrated, delete `models.py` and `generated/`

## Open Questions

1. **Caching strategy**: Compile on every import, or cache to `_cache/` directory?
   - Recommendation: Cache with checksum validation (regenerate if domain-v4 changes)

2. **Nested object types**: How to handle `Choice` and `Gate` embedded in `Section`?
   - Recommendation: Compiler recursively creates nested models from `properties` field

3. **Visibility enum**: Keep as Python Enum or use string validation?
   - Recommendation: String validation against domain-defined values (more flexible)

## Success Criteria

- [ ] `from questfoundry.runtime.domain import load_studio, get_artifact_model` works
- [ ] `studio.agents["showrunner"]` returns properly typed `Agent`
- [ ] `get_artifact_model("section")` returns Pydantic model matching domain-v4/artifact-types/section.json
- [ ] No imports from `questfoundry.generated` anywhere in codebase
- [ ] All tests pass
- [ ] `generated/` directory deleted

## Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: Metamodel | Medium (careful matching to meta/) |
| Phase 2: Compiler | Medium (recursive type handling) |
| Phase 3: Loader | Small (mostly exists) |
| Phase 4: Consumers | Large (many files, careful testing) |
| Phase 5: Cleanup | Small |
| Phase 6: Docs | Small |

---

*This plan addresses the architectural discussion about domain-first design, metamodel stability, and dynamic artifact compilation.*
