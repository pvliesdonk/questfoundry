# Phase 0: Foundation Design

> **Issue**: #144
> **Status**: ✅ Complete
> **Last Updated**: 2024-12-14

## Overview

Foundation layer for the v4 runtime cleanroom rebuild. Implements domain loading, runtime types, configuration, project structure, and initial CLI commands.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Async | Async-first | Harder to retrofit later |
| LLM | LangChain community packages | Ecosystem support |
| Validation | Strict, fail fast | Catch errors early |
| Storage | SQLite for artifacts | Fast queries, single file |
| Config | YAML + env + CLI | Standard, flexible |
| Nested fields | Separate Pydantic objects | Cleaner code, reusable |

---

## Implementation Summary

### Test Coverage

| Component | Tests | File |
|-----------|-------|------|
| Domain Loader | 19 | `tests/runtime/domain/test_loader.py` |
| Schema Compiler | 26 | `tests/runtime/domain/test_compiler.py` |
| Configuration | 17 | `tests/runtime/config/test_config.py` |
| Project Storage | 19 | `tests/runtime/storage/test_project.py` |
| **Total** | **81** | |

### File Structure (Implemented)

```
src/questfoundry/
├── __init__.py
├── cli.py                          # CLI commands: doctor, config, roles, projects
└── runtime/
    ├── __init__.py
    ├── config.py                   # Configuration system with provider states
    ├── domain/
    │   ├── __init__.py             # Exports: load_studio, LoadResult, LoadError
    │   ├── loader.py               # Domain loading and validation
    │   └── compiler.py             # FieldDefinition → JSON Schema
    ├── models/
    │   ├── __init__.py             # Exports all models
    │   ├── enums.py                # Archetype, FieldType, StoreSemantics, etc.
    │   ├── fields.py               # FieldDefinition (recursive)
    │   └── base.py                 # Studio, Agent, Store, Tool, etc.
    └── storage/
        ├── __init__.py             # Exports: Project, ProjectInfo, list_projects
        └── project.py              # SQLite-backed project storage
```

---

## 1. Domain Loader

### Implementation: `runtime/domain/loader.py`

Loads a studio definition from a directory, resolving all file references and building typed Pydantic models.

```python
async def load_studio(domain_path: Path) -> LoadResult:
    """Load and validate a studio definition."""
```

### Loading Flow

1. Load `studio.json`
2. Resolve file path references (agents, stores, tools, etc.)
3. Validate with Pydantic models
4. Check internal consistency (refs point to valid entities)
5. Return `LoadResult` with studio or errors

### Error Handling

```python
@dataclass
class LoadError:
    path: str           # File or reference that caused error
    message: str        # Human-readable description
    severity: Literal["error", "warning"]

@dataclass
class LoadResult:
    studio: Studio | None
    errors: list[LoadError]
    warnings: list[LoadError]

    @property
    def success(self) -> bool:
        return self.studio is not None and len(self.errors) == 0
```

---

## 2. Runtime Types (Pydantic)

### Implementation: `runtime/models/`

#### Enums (`enums.py`)

```python
class FieldType(str, Enum):
    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    URI = "uri"
    ARRAY = "array"
    OBJECT = "object"
    REF = "ref"

class StoreSemantics(str, Enum):
    HOT = "hot"
    COLD = "cold"
    VERSIONED = "versioned"
    EPHEMERAL = "ephemeral"
```

#### Field Definition (`fields.py`)

```python
class FieldDefinition(BaseModel):
    """Recursive field definition for artifact schemas."""
    name: str
    type: FieldType
    description: str | None = None
    required: bool = False
    default: Any = None

    # Object type: nested field definitions
    properties: list[FieldDefinition] | None = None

    # Array type: item structure
    items: FieldDefinition | None = None
    items_type: FieldType | None = None

    # Constraints
    enum: list[str] | None = None
    min: float | None = None
    max: float | None = None
    min_length: int | None = None
    max_length: int | None = None
```

#### Base Models (`base.py`)

- `Studio` - Top-level container with all entities
- `Agent` - Agent definition with capabilities, constraints
- `Store` - Storage definition with semantics
- `Tool` - Tool definition with parameters
- `Playbook` - Workflow definition
- `ArtifactType` - Artifact schema with fields
- `AssetType` - Binary asset definition
- `QualityCriteria` - Quality validation rules

---

## 3. Schema Compiler

### Implementation: `runtime/domain/compiler.py`

Compiles `list[FieldDefinition]` → JSON Schema Draft 2020-12.

```python
def compile_schema(
    fields: list[FieldDefinition],
    title: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Compile field definitions to JSON Schema."""
```

Features:

- All FieldType mappings to JSON Schema types
- Nested object/array support with recursive compilation
- Enum constraints
- Numeric min/max constraints
- String/array length constraints
- Required field tracking
- `$defs` for reusable nested types

---

## 4. Configuration System

### Implementation: `runtime/config.py`

```python
@dataclass
class RuntimeConfig:
    domain_path: Path
    providers: dict[str, ProviderConfig]
    model_classes: ModelClassMapping
    default_provider: str
    log_events: bool
    log_path: Path | None
    langsmith_enabled: bool
    langsmith_project: str

def load_config(
    config_file: Path | None = None,
    env_file: Path | None = None,
) -> RuntimeConfig:
    """Load configuration from all sources."""
```

### Configuration Sources (Precedence)

1. `.env` file (lowest priority)
2. Environment variables
3. `qf.yaml` config file
4. CLI arguments (highest priority)

### Provider States

```python
class ProviderState(str, Enum):
    UNCONFIGURED = "unconfigured"  # No host/key configured
    UNAVAILABLE = "unavailable"    # Configured but connectivity fails
    AVAILABLE = "available"        # Configured and reachable
```

### Model Class Mapping

```python
class ModelClassMapping:
    """Maps abstract model classes to provider-specific models."""
    mappings: dict[str, dict[str, str]]

    def get_model(self, model_class: str, provider: str) -> str | None:
        """Get provider-specific model for a model class."""
```

---

## 5. Project Structure

### Implementation: `runtime/storage/project.py`

```python
class Project:
    """Manages a single story/game project."""

    @classmethod
    def create(cls, path: Path, name: str, ...) -> Project:
        """Create a new project with directory structure."""

    @classmethod
    def open(cls, path: Path) -> Project:
        """Open an existing project."""

    def create_artifact(self, artifact_id, artifact_type, data, ...) -> dict
    def get_artifact(self, artifact_id) -> dict | None
    def update_artifact(self, artifact_id, data, ...) -> dict | None
    def query_artifacts(self, artifact_type, store, ...) -> list[dict]
```

### Directory Layout

```
projects/
└── my_story/
    ├── project.json              # Project metadata
    ├── project.sqlite            # Artifacts, messages, sessions
    ├── assets/                   # Binary files
    ├── logs/                     # Only with --log flag
    └── checkpoints/              # Session checkpoints
```

### SQLite Schema

```sql
CREATE TABLE artifacts (
    _id TEXT PRIMARY KEY,
    _type TEXT NOT NULL,
    _version INTEGER NOT NULL DEFAULT 1,
    _created_at TEXT NOT NULL,
    _updated_at TEXT NOT NULL,
    _created_by TEXT,
    _lifecycle_state TEXT DEFAULT 'draft',
    _store TEXT,
    data JSON NOT NULL
);

CREATE TABLE artifact_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    data JSON NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE NOT NULL,
    message_type TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    payload JSON NOT NULL,
    created_at TEXT NOT NULL,
    processed_at TEXT,
    status TEXT DEFAULT 'pending'
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    entry_agent TEXT NOT NULL,
    status TEXT DEFAULT 'active'
);

CREATE TABLE turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    input TEXT,
    output TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    token_usage JSON
);
```

---

## 6. CLI Commands

### Implementation: `cli.py`

```bash
# System health check
qf doctor
# Domain: QuestFoundry Interactive Fiction Studio v4.0.0
#   12 agents (entry: showrunner, player_narrator)
#   10 tools, 5 stores, 7 playbooks
# Providers:
#   ✓ ollama @ http://... (qwen3:8b)
#   ✓ openai (gpt-4o)
#   ✓ google (gemini-1.5-pro)

# Show resolved configuration
qf config
# YAML dump of merged configuration

# List agents
qf roles
# Table: id | name | archetypes | entry | capabilities

# Project management
qf projects list
qf projects create <name>
qf projects info <name>
```

---

## References

- **Meta schemas**: `meta/schemas/core/`
- **Domain instance**: `domain-v4/`
- **Design philosophy**: `meta/docs/README.md`
- **GitHub issue**: #144
