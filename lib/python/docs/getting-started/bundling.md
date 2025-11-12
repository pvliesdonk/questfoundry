# Resource Bundling

QuestFoundry bundles schemas and prompts from the specification into the library package.

## Overview

The QuestFoundry mono-repo maintains a single source of truth for schemas and prompts in the `spec/` directory:

- **Schemas**: `spec/03-schemas/` - JSON schema files
- **Prompts**: `spec/05-prompts/` - Role-specific prompt templates

These resources are bundled into the Python library at build time, ensuring the library always has access to the latest schemas and prompts.

## How Bundling Works

### 1. Bundling Script

The `scripts/bundle_resources.py` script copies resources from the spec directory into the library:

```bash
cd lib/python
python scripts/bundle_resources.py
```

This copies:
- All `.schema.json` files from `spec/03-schemas/` to `src/questfoundry/resources/schemas/`
- All role directories from `spec/05-prompts/` to `src/questfoundry/resources/prompts/`

### 2. Resource Loading

The library uses `importlib.resources` to load bundled resources at runtime:

```python
from questfoundry.utils.resources import get_schema, get_prompt

# Load a schema (from bundled resources)
schema = get_schema("hook_card")

# Load a prompt (from bundled resources)
prompt = get_prompt("plotwright")
```

### 3. Package Distribution

When the library is built and distributed via PyPI:

```bash
cd lib/python
uv build
```

The bundled resources are included in the wheel package, so users get all schemas and prompts with the library.

## Development Workflow

### After Spec Changes

If you modify schemas or prompts in the `spec/` directory, you must re-bundle resources:

```bash
cd lib/python
uv run hatch run bundle
```

### Before Building

Always bundle resources before building the library:

```bash
cd lib/python
uv run hatch run bundle
uv build
```

### In CI/CD

The CI workflows automatically bundle resources before testing or publishing:

```yaml
- name: Bundle resources
  run: python scripts/bundle_resources.py
```

## Single Source of Truth

**Important**: The `spec/` directory is the single source of truth for all schemas and prompts.

- Never manually edit files in `src/questfoundry/resources/`
- Always edit files in `spec/` and re-bundle
- The bundled resources are automatically excluded from git (via `.gitignore`)

## Hatch Scripts

Use the following Hatch scripts for common tasks:

```bash
# Bundle resources
uv run hatch run bundle

# Run tests (with bundled resources)
uv run hatch run test

# Lint and format
uv run hatch run lint
uv run hatch run format

# Type check
uv run hatch run typecheck
```

## Next Steps

- [User Guide](../guide/overview.md) - Learn how to use the library
- [Development Guide](../development/contributing.md) - Contribute to QuestFoundry
- [Architecture](../development/architecture.md) - Understand the design
