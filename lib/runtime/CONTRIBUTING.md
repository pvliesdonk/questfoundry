# Contributing to QuestFoundry Runtime

This document provides Python development guidelines for the QuestFoundry runtime library (Layer 6).

## Table of Contents

- [Overview](#overview)
- [Development Setup](#development-setup)
- [Python Guidelines](#python-guidelines)
- [Testing](#testing)
- [Resource Loading](#resource-loading)
- [Commit Conventions](#commit-conventions)

## Overview

The `lib/runtime/` directory contains the Python implementation of the QuestFoundry specification (Layer 6). This library provides the runtime execution environment for the QuestFoundry studio.

### Directory Structure

```text
lib/runtime/
├── src/
│   └── questfoundry/     # Main Python package
│       ├── models/       # Data models
│       ├── roles/        # Role implementations
│       ├── loops/        # Loop implementations
│       ├── protocol/     # Protocol handlers
│       ├── providers/    # LLM providers
│       ├── utils/        # Utilities (including resource loading)
│       ├── validation/   # Validation logic
│       └── orchestrator.py  # Main orchestrator
├── tests/                # Pytest test suite
├── scripts/              # Build and utility scripts
├── pyproject.toml        # Dependencies and configuration
├── CONTRIBUTING.md       # This file
└── README.md             # Library documentation
```

### Relationship to Specification

This library **implements** the specification defined in `../../spec/`:

- Bundles schemas from `../../spec/03-schemas/` at build time
- Loads definitions from `../../spec/05-definitions/` at runtime
- Implements roles defined in `../../spec/01-roles/`
- Follows protocols defined in `../../spec/04-protocol/`

**Critical:** This library reads from the spec but **never modifies** it. The spec is the single source of truth. Always edit files in `spec/` and re-run the bundling script.

## Development Setup

### Prerequisites

- Python 3.11, 3.12, or 3.13
- `uv` package manager (install from <https://docs.astral.sh/uv/>)

### One-Time Setup

1. **Install dependencies:**

   ```bash
   cd lib/runtime
   uv sync
   ```

2. **Install pre-commit hooks** (from repo root):

   ```bash
   cd ../..
   pre-commit install
   ```

3. **Bundle resources from spec:**

   ```bash
   cd lib/runtime
   uv run hatch run bundle
   ```

### Development Workflow

Always use `uv` for running code and managing dependencies:

```bash
# Install/sync dependencies
uv sync

# Run linting
uv run ruff check .

# Run type checking
uv run mypy

# Auto-format code
uv run ruff format .

# Run tests
uv run pytest

# Run specific test with output
uv run pytest -s tests/test_example.py

# Bundle resources from spec
uv run hatch run bundle
```

### Definition of Done

Before considering any task complete:

- [ ] All linter checks pass (`uv run ruff check .`)
- [ ] Type checking passes (`uv run mypy`)
- [ ] All tests pass (`uv run pytest`)
- [ ] Code is properly formatted (`uv run ruff format .`)
- [ ] Resources are bundled if spec changed
- [ ] Documentation is updated if API changed

## Python Guidelines

### Version Support

Write code for **Python 3.11-3.13 only**. Use modern Python practices:

- Full type annotations on all functions and methods
- Modern union syntax: `str | None` (not `Optional[str]`)
- Built-in generics: `list[str]`, `dict[str, int]` (not `List`, `Dict`)
- `from __future__ import annotations` when needed

### Code Organization

- All source code in `src/`
- All tests in `tests/`
- Follow Ruff linting standards
- Resolve all type checker errors

### Import Conventions

**Always use absolute imports:**

```python
from questfoundry.models.artifact import HookCard
from questfoundry.utils.resources import load_schema
```

**Never use relative imports:**

```python
# ❌ Don't do this
from .models.artifact import HookCard
from ..utils import load_schema
```

**Import types correctly:**

```python
from collections.abc import Callable, Coroutine, Sequence
from typing_extensions import override  # Use for @override
from pathlib import Path  # Use instead of string paths
```

### Type Annotations

**Modern syntax:**

```python
# ✅ Good
def process(data: str | None) -> list[dict[str, int]]:
    ...

# ❌ Avoid
from typing import Optional, List, Dict
def process(data: Optional[str]) -> List[Dict[str, int]]:
    ...
```

**Always use `@override`:**

```python
from typing_extensions import override

class MyRole(BaseRole):
    @override
    def execute(self, input: str) -> str:
        ...
```

### File Operations

Use `pathlib.Path` for all file operations:

```python
from pathlib import Path

# ✅ Good
content = Path("file.txt").read_text()
data = Path("file.json").read_bytes()

# ❌ Avoid
with open("file.txt", "r") as f:
    content = f.read()
```

### String Formatting

Use `textwrap.dedent` for multi-line strings:

```python
from textwrap import dedent

prompt = dedent("""
    You are a helpful assistant.
    Please respond to: {question}
    """).strip()
```

### Comments and Docstrings

**Comments should explain WHY, not WHAT:**

```python
# ✅ Good: Explains reasoning
# Use exponential backoff to avoid rate limiting
await asyncio.sleep(2 ** attempt)

# ❌ Avoid: States the obvious
# Sleep for 2 to the power of attempt
await asyncio.sleep(2 ** attempt)
```

**Docstrings should be concise:**

```python
def load_role_profile(role_name: str) -> RoleProfile:
    """
    Load a role profile from bundled resources.

    Raises `FileNotFoundError` if the profile doesn't exist.
    """
    ...
```

**Don't write obvious docstrings:**

```python
# ❌ Avoid: Function name and types say it all
def get_user(user_id: int) -> User:
    """Get a user by user ID."""
    ...

# ✅ Better: Add it only if there's something useful to say
def get_user(user_id: int) -> User:
    """
    Fetch user from cache if available, otherwise from database.

    Note: Returns stale data if cache is warm and user was modified
    in the last 5 seconds.
    """
    ...
```

### Error Handling

**Use specific exceptions:**

```python
# ✅ Good
raise ValueError(f"Invalid role name: {role_name}")

# ❌ Avoid
assert False, f"Invalid role name: {role_name}"
```

**Don't suppress type errors unnecessarily:**

```python
# ✅ Good: Fix the type issue
result: str = process_string(input_str)

# ⚠️ Use sparingly: Only when type checker is wrong
result = process_string(input_str)  # pyright: ignore[reportUnknownVariableType]
```

## Testing

### Test Organization

- Place all tests in `tests/` directory
- Mirror source structure: `src/questfoundry/models/` → `tests/test_models.py`
- Use clear, descriptive test names: `test_hook_card_validates_required_fields`

### Test Style

**Keep tests simple:**

```python
def test_load_schema():
    schema = load_schema("hook_card.schema.json")
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "properties" in schema
```

**Don't over-test:**

```python
# ❌ Avoid: Trivial test
def test_constant_value():
    assert SCHEMA_VERSION == "2020-12"

# ❌ Avoid: Testing Pydantic
def test_model_creation():
    model = HookCard(id="HK-20250101-001", status="draft")
    assert model.id == "HK-20250101-001"
```

**Avoid pytest features unless necessary:**

- No fixtures for simple tests
- No parameterization unless testing many similar cases
- Keep it simple: `assert value == expected`

**Don't add assert messages:**

```python
# ✅ Good
assert result == 42

# ❌ Avoid
assert result == 42, "result should be 42"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with output
uv run pytest -s

# Run specific test file
uv run pytest tests/test_models.py

# Run specific test
uv run pytest tests/test_models.py::test_hook_card_validation
```

## Resource Loading

### How Resources are Bundled

Resources are bundled from the spec at build time:

1. **Schemas** from `../../spec/03-schemas/` → `src/questfoundry/resources/schemas/`
2. **Definitions** from `../../spec/05-definitions/` → `src/questfoundry/resources/definitions/`
3. Bundled files are gitignored (not checked in)
4. The spec directory is the single source of truth

### Bundling Script

```bash
cd lib/runtime
uv run hatch run bundle
```

This runs `scripts/bundle_resources.py` which copies files from spec to the package.

### Loading Resources at Runtime

Use `importlib.resources` to load bundled resources:

```python
from questfoundry.utils.resources import load_schema, load_definition

# Load a schema
schema = load_schema("hook_card.schema.json")

# Load a role definition
role = load_definition("roles/lore_weaver.yaml")
```

**Never load from spec directly:**

```python
# ❌ Don't do this
schema_path = Path("../../spec/03-schemas/hook_card.schema.json")
schema = json.loads(schema_path.read_text())

# ✅ Do this instead
from questfoundry.utils.resources import load_schema
schema = load_schema("hook_card.schema.json")
```

### Workflow for Spec Changes

When the spec changes:

1. Edit files in `../../spec/`
2. Run bundling: `uv run hatch run bundle`
3. Test with updated resources
4. Commit spec changes (bundled files are gitignored)

## Commit Conventions

Use **standard conventional commit types** for runtime changes:

### Standard Types

```
feat(runtime): Add support for Anthropic provider
fix(runtime): Resolve resource loading on Windows
refactor(runtime): Simplify validation logic
test(runtime): Add tests for prompt loading
docs(runtime): Update API documentation
chore(runtime): Update dependencies
ci(runtime): Update GitHub Actions workflow
perf(runtime): Optimize schema validation
```

### Type Reference

- `feat(runtime)` — New features (minor version bump: 0.x.0)
- `fix(runtime)` — Bug fixes (patch version bump: 0.0.x)
- `refactor(runtime)` — Code refactoring (patch version bump)
- `test(runtime)` — Test changes (no version bump)
- `docs(runtime)` — Documentation (no version bump)
- `chore(runtime)` — Maintenance (no version bump)
- `ci(runtime)` — CI/CD changes (no version bump)
- `perf(runtime)` — Performance (patch version bump)

### Breaking Changes

Use `!` suffix for breaking changes:

```
feat(runtime)!: Change resource API to use async/await

BREAKING CHANGE: All resource loading functions are now async.
Update calls from `load_schema(name)` to `await load_schema(name)`.
```

This triggers a major version bump (x.0.0).

### Scopes

You can use more specific scopes:

- `feat(roles)` — Role implementation changes
- `fix(protocol)` — Protocol handler fixes
- `refactor(models)` — Model refactoring

### Commit Best Practices

- Keep commits atomic (one logical change)
- Write clear, descriptive subjects
- Use imperative mood: "add" not "added"
- Explain "why" in the body if needed
- Reference issues: `Closes #123`

## Code Review Checklist

Before submitting a PR:

- [ ] All tests pass
- [ ] No linter errors or warnings
- [ ] Type checking passes
- [ ] Code is formatted with Ruff
- [ ] Docstrings added for public APIs
- [ ] Complex logic has explanatory comments
- [ ] Tests added for new functionality
- [ ] Resources bundled if spec changed
- [ ] Commit messages follow conventions
- [ ] Breaking changes documented

## Additional Resources

- Parent contributing guide: [../../CONTRIBUTING.md](../../CONTRIBUTING.md)
- Specification guidelines: [../../spec/CONTRIBUTING.md](../../spec/CONTRIBUTING.md)
- Runtime architecture: [../../spec/06-runtime/ARCHITECTURE.md](../../spec/06-runtime/ARCHITECTURE.md)
- QuestFoundry specification: [../../spec/README.md](../../spec/README.md)

## Questions?

- Check [README.md](README.md) for library documentation
- Review [../../spec/](../../spec/) for specification details
- Open an issue for bugs or feature requests
- Start a discussion for questions

Thank you for contributing to the QuestFoundry runtime!
