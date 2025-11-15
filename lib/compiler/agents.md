# QuestFoundry Compiler - Agent Guidelines

## Package Context

The **questfoundry-compiler** is a standalone Python package that compiles atomic behavior primitives from the QuestFoundry specification into runtime-ready artifacts (JSON manifests and assembled prompts).

This package is extracted from the main `questfoundry-py` library to serve multiple use cases:

1. **Build-time dependency** for `questfoundry-py` (static compilation, manifests bundled)
2. **Runtime dependency** for web-based prompt generators (dynamic compilation)
3. **Standalone CLI tool** for spec validation and development

## Development Rules

### Assistant Responsibilities

Follow the core guidelines from the main [QuestFoundry AGENTS.md](../../AGENTS.md):

- Be concise, factual, and systematic
- Express expert opinion on better/worse approaches
- Avoid gratuitous enthusiasm or compliments
- Think step-by-step for complex changes

### Package-Specific Rules

#### 1. Compiler Independence

The compiler must remain **completely independent** from runtime concerns:

- ✅ **DO**: Parse, validate, and assemble spec primitives
- ✅ **DO**: Generate manifests and standalone prompts
- ❌ **DON'T**: Import anything from `questfoundry-py` (circular dependency)
- ❌ **DON'T**: Execute playbooks or instantiate roles
- ❌ **DON'T**: Add runtime-specific logic

#### 2. Minimal Dependencies

Keep dependencies minimal to avoid bloat:

- Core: `pyyaml` only (parsing YAML frontmatter)
- Dev: `pytest`, `mypy`, `ruff`
- **No**: `pydantic`, `jsonschema`, or other heavy dependencies
- Rationale: Web services may compile thousands of times; keep it fast

#### 3. Cross-Reference Validation

All changes to reference syntax or validation must:

1. Update `spec/05-behavior/README.md` reference documentation
2. Add tests for new reference patterns
3. Validate against existing spec files
4. Document in compiler README

#### 4. Output Format Stability

Manifest and prompt output formats are **public APIs**:

- Breaking changes require major version bump
- Add optional fields only (never remove or rename)
- Document schema changes in CHANGELOG
- Coordinate with `questfoundry-py` maintainers

#### 5. Error Messages

Compiler errors must be **actionable**:

```python
# ❌ BAD
raise CompilationError("Invalid reference")

# ✅ GOOD
raise CompilationError(
    f"Invalid reference '@procedure:missing_proc' in playbook "
    f"'{playbook_id}': procedure file 'procedures/missing_proc.md' not found. "
    f"Available procedures: {', '.join(available_procs)}"
)
```

Include:
- What went wrong
- Where (file, line, reference)
- Why (file not found, circular dependency, etc.)
- How to fix (list available options)

#### 6. Testing Requirements

All changes must include tests:

- **Unit tests**: For individual functions/methods
- **Integration tests**: For full compilation workflows
- **Error cases**: Validate error messages and handling
- **Regression tests**: For bug fixes

Run before committing:

```bash
uv run pytest --cov=questfoundry_compiler --cov-report=term-missing
uv run mypy src
uv run ruff check src tests
```

#### 7. Type Annotations

All public APIs must have complete type annotations:

```python
# ✅ GOOD
def compile_playbook(
    self,
    playbook_id: str,
    output_dir: Path
) -> dict[str, Any]:
    ...

# ❌ BAD
def compile_playbook(self, playbook_id, output_dir):
    ...
```

#### 8. Documentation

Update documentation for any public API changes:

- **README.md**: User-facing features
- **Docstrings**: All public classes/functions
- **CHANGELOG.md**: User-visible changes (use commitizen)
- **Type stubs**: If adding complex types

## Code Organization

### Source Structure

```
src/questfoundry_compiler/
├── __init__.py          # Public API exports
├── types.py             # Type definitions, exceptions
├── spec_compiler.py     # Main orchestrator
├── validators.py        # Cross-reference validation
├── assemblers.py        # Content assembly
├── manifest_builder.py  # JSON manifest generation
└── cli.py               # Command-line interface
```

**Public API** (exported from `__init__.py`):

- `SpecCompiler` - Main compiler class
- `BehaviorPrimitive` - Primitive data type
- `CompilationError` - Exception type

**Internal** (not exported):

- Validators, assemblers, manifest builders

### Import Conventions

```python
# External dependencies (top)
import json
from pathlib import Path
from typing import Any

import yaml

# Internal imports (bottom, absolute)
from questfoundry_compiler.types import BehaviorPrimitive, CompilationError
from questfoundry_compiler.validators import ReferenceValidator
```

## Versioning Strategy

Follow semantic versioning strictly:

- **Major** (1.0.0 → 2.0.0): Breaking changes to output format or CLI
- **Minor** (0.1.0 → 0.2.0): New features, backward-compatible
- **Patch** (0.1.0 → 0.1.1): Bug fixes only

Use commitizen for version bumps:

```bash
cz bump
```

## Release Process

1. Ensure all tests pass:
   ```bash
   uv run pytest
   uv run mypy src
   uv run ruff check src tests
   ```

2. Update CHANGELOG:
   ```bash
   cz changelog
   ```

3. Bump version:
   ```bash
   cz bump
   ```

4. Push to GitHub (triggers publish workflow):
   ```bash
   git push --follow-tags
   ```

5. Verify PyPI publish in GitHub Actions

## Common Tasks

### Adding a New Primitive Type

1. Update `spec_compiler.py` loader with new type
2. Add validation rules in `validators.py`
3. Implement assembly logic in `assemblers.py`
4. Update manifest builder if needed
5. Add comprehensive tests
6. Document in README and docstrings

### Changing Reference Syntax

1. Update regex pattern in `spec_compiler.py`
2. Update `validators.py` resolution logic
3. Add tests for new syntax
4. Update `spec/05-behavior/README.md` documentation
5. Consider backward compatibility (major version?)

### Fixing a Bug

1. Add regression test that reproduces bug
2. Fix the bug
3. Verify test passes
4. Add entry to CHANGELOG (via commit message)
5. Patch version bump

## Anti-Patterns to Avoid

### ❌ Runtime Concerns

```python
# DON'T - No role execution
from questfoundry.roles import LoreWeaver
```

### ❌ Heavy Dependencies

```python
# DON'T - Avoid unless absolutely necessary
import pandas
import numpy
```

### ❌ Global State

```python
# DON'T - No module-level state
CACHE = {}  # Bad! Use instance variables

# DO - Instance state only
class SpecCompiler:
    def __init__(self):
        self.primitives: dict[str, BehaviorPrimitive] = {}
```

### ❌ Silent Failures

```python
# DON'T - Always raise or log
try:
    load_primitive(path)
except FileNotFoundError:
    pass  # Silent failure!

# DO - Explicit error handling
try:
    load_primitive(path)
except FileNotFoundError as e:
    raise CompilationError(f"Failed to load {path}: {e}") from e
```

## Questions and Clarifications

For architectural questions or breaking changes:

1. Check existing issues in main QuestFoundry repo
2. Open a new issue with `[compiler]` prefix
3. Tag `@pvliesdonk` for review
4. Wait for consensus before implementing

## Related Documentation

- **Main repo**: [QuestFoundry AGENTS.md](../../AGENTS.md)
- **Specification**: [spec/README.md](../../spec/README.md)
- **Behavior layer**: [spec/05-behavior/README.md](../../spec/05-behavior/README.md)
- **Python library**: [lib/python/README.md](../python/README.md)
