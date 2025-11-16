# QuestFoundry Spec Compiler

**Transform atomic behavior primitives into runtime-ready artifacts**

[![PyPI version](https://badge.fury.io/py/questfoundry-compiler.svg)](https://badge.fury.io/py/questfoundry-compiler)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The QuestFoundry Spec Compiler (`questfoundry-compiler`) is a standalone tool that compiles atomic behavior primitives from `../../spec/05-behavior/` into runtime-ready artifacts. It validates cross-references, assembles prompts, and generates JSON manifests for execution.

This package is located at `lib/compiler/` in the [QuestFoundry](https://github.com/pvliesdonk/questfoundry) mono-repo and can be used:

- **As a build-time dependency** for `questfoundry-py` (static compilation at package build time)
- **As a runtime dependency** for `qf-generate` CLI tool (dynamic compilation for web agents)
- **As a standalone CLI tool** (`qf-compile`) for spec validation and compilation

## Installation

```bash
pip install questfoundry-compiler
```

## Quick Start

### Compile Behavior Primitives

```bash
# From anywhere (if installed from PyPI)
qf-compile --spec-dir /path/to/spec --output dist/compiled/

# From mono-repo (using local installation)
cd lib/compiler
qf-compile --spec-dir ../../spec --output ../../dist/compiled/

# Validate only (no output)
qf-compile --spec-dir ../../spec --validate-only

# Compile specific playbook
qf-compile --spec-dir ../../spec --playbook lore_deepening --output ../../dist/compiled/
```

### Programmatic Usage

```python
from pathlib import Path
from questfoundry_compiler import SpecCompiler, CompilationError

# Initialize compiler
compiler = SpecCompiler(spec_root=Path("/path/to/spec"))

# Compile all primitives (includes validation)
try:
    output_dir = Path("dist/compiled")
    stats = compiler.compile_all(output_dir)

    print(f"✓ Compiled {stats['primitives_loaded']} primitives")
    print(f"  - Playbook manifests: {stats['playbook_manifests_generated']}")
    print(f"  - Adapter manifests: {stats['adapter_manifests_generated']}")
    print(f"  - Standalone prompts: {stats['standalone_prompts_generated']}")
except CompilationError as e:
    print(f"Compilation failed: {e}")
    exit(1)
```

## What It Compiles

The compiler transforms atomic behavior primitives into runtime artifacts:

### Input: Atomic Primitives

Located in `../../spec/05-behavior/`:

```
spec/05-behavior/
├── expertises/          # Domain knowledge per role
│   └── lore_weaver_expertise.md
├── procedures/          # Reusable workflow steps
│   └── canonization_core.md
├── snippets/           # Small text blocks
│   └── spoiler_hygiene_reminder.md
├── playbooks/          # Loop definitions (YAML)
│   └── lore_deepening.playbook.yaml
└── adapters/           # Role configurations (YAML)
    └── lore_weaver.adapter.yaml
```

### Output: Runtime Artifacts

```
dist/compiled/
├── manifests/                          # JSON manifests for execution
│   ├── lore_deepening.manifest.json
│   └── lore_weaver.manifest.json
└── standalone_prompts/                 # Assembled full prompts
    ├── lore_weaver_full.md
    └── showrunner_full.md
```

## Features

### ✅ Cross-Reference Validation

Validates all `@type:id` references:

- `@expertise:lore_weaver_expertise` → Resolves to file
- `@procedure:canonization_core#step1` → Validates section exists
- `@schema:canon_pack.schema.json` → Checks schema exists
- `@role:lore_weaver` → Verifies role defined in L1

### ✅ Dependency Analysis

- Detects circular dependencies
- Identifies orphaned primitives (not referenced)
- Builds dependency graphs

### ✅ Content Assembly

- Injects referenced content at compile time
- Resolves section anchors (`#section-name`)
- Composes full prompts from primitives

### ✅ Manifest Generation

Generates runtime-ready JSON manifests with:

- Metadata (version, compile timestamp, source files)
- RACI matrices
- Quality bars
- Step definitions with embedded procedure content
- Artifact type references

## CLI Reference

```
qf-compile [OPTIONS]

Options:
  --spec-dir PATH         Path to spec/ directory (default: ./spec)
  --output PATH          Output directory for compiled artifacts (default: ./dist/compiled)
  --playbook ID          Compile specific playbook only
  --adapter ID           Compile specific adapter only
  --validate-only        Validate without generating output
  --watch               Watch for changes and recompile (future)
  --verbose             Show detailed compilation progress
  --help                Show this message and exit
```

## Architecture

### Compilation Pipeline

```
[Atomic Sources] → [Loader] → [Validator] → [Assembler] → [Manifest Builder] → [Output]
     (YAML/MD)      (parse)     (refs)       (compose)       (JSON)            (dist/)
```

### Components

Located in `lib/compiler/src/questfoundry_compiler/`:

- **spec_compiler.py** - Main orchestrator, loads primitives
- **validators.py** - Cross-reference validation, dependency analysis
- **assemblers.py** - Content composition, reference resolution
- **manifest_builder.py** - JSON manifest generation
- **types.py** - Type definitions (`BehaviorPrimitive`, `CompilationError`)
- **cli.py** - Command-line interface (`qf-compile`)

## Use Cases

### 1. Build-Time Static Compilation (questfoundry-py)

The `questfoundry-py` package uses this compiler at build time:

```python
# In build hook or bundle script
from pathlib import Path
from questfoundry_compiler import SpecCompiler, CompilationError

try:
    compiler = SpecCompiler(spec_root=Path("../../spec"))
    stats = compiler.compile_all(output_dir=Path("src/questfoundry/resources/manifests"))
    print(f"✓ Bundled {stats['playbook_manifests_generated']} manifests")
except CompilationError as e:
    print(f"Error: Compilation failed - {e}")
    exit(1)
```

Published package contains pre-compiled manifests, **not** the compiler itself.

### 2. Runtime Dynamic Compilation (qf-generate CLI)

The `qf-generate` tool uses this compiler to dynamically generate prompts:

```python
from pathlib import Path
from questfoundry_compiler import SpecCompiler, CompilationError

# Initialize compiler once at startup
compiler = SpecCompiler(spec_root=Path("../../spec"))
compiler.load_all_primitives()

# Compile specific playbook on demand
try:
    result = compiler.compile_playbook("lore_deepening", output_dir=Path("/tmp/compiled"))
    print(f"✓ Compiled playbook to {result['manifest_path']}")
except CompilationError as e:
    print(f"Error: {e}")
    exit(1)
```

The CLI tool dynamically compiles prompts for web agent simulation.

### 3. Standalone Validation Tool

```bash
# In CI/CD pipeline
qf-compile --spec-dir spec/ --validate-only
```

## Development

### Setup

Install from PyPI:

```bash
pip install questfoundry-compiler
```

Or develop locally:

```bash
cd lib/compiler
uv sync --all-extras
```

### Run Tests

```bash
uv run pytest
uv run pytest --cov=questfoundry_compiler --cov-report=term-missing
```

### Linting & Type Checking

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run mypy src
```

## Version Compatibility

The compiler follows semantic versioning:

- **Major version** changes indicate breaking compilation behavior
- **Minor version** changes add features or fix bugs without breaking existing specs
- **Patch version** changes are bug fixes only

`questfoundry-py` specifies: `questfoundry-compiler>=0.1.0,<1.0.0`

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Related Projects

- **[questfoundry-py](https://pypi.org/project/questfoundry-py/)** - Python runtime library (uses this compiler at build time)
- **[QuestFoundry Spec](../../spec/)** - Complete specification (Layers 0-5)
- **[qf-generate CLI](../../cli/prompt_generator/)** - Prompt generator tool (uses this compiler at runtime)

## Contributing

See the main [QuestFoundry repository](../../) for contribution guidelines.
