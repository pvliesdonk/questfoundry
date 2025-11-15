# QF CLI Implementation Plan

**Version:** 1.0.0
**Date:** 2025-11-15
**Optimized for:** Claude Sonnet 4.5 LLM Implementation
**Based on:** [Gist 811a448413ea751b1da5ad5f50b75a81](https://gist.github.com/pvliesdonk/811a448413ea751b1da5ad5f50b75a81)

---

## Executive Summary

This plan describes the implementation of the **QF CLI** - a unified command-line interface for QuestFoundry. The implementation involves **three distinct components**:

1. **`lib/spec_compiler/`** - Standalone spec compiler library (extracted from lib/python)
2. **Prompt Generator** - Tool to generate standalone prompts for LLM consumption
3. **`qf` CLI** - Unified command-line interface orchestrating all QuestFoundry operations

### Critical Understanding

These are **NOT the same program**. They are:
- **lib/spec_compiler/** - A reusable library with no CLI
- **Prompt Generator** - A feature within the qf CLI
- **qf CLI** - The user-facing command-line tool

---

## Current State Analysis

### What Exists Now

**Compiler Location:** `lib/python/src/questfoundry/compiler/`

**Compiler Components:**
```
lib/python/src/questfoundry/compiler/
├── __init__.py                    # Public API exports
├── spec_compiler.py               # Main orchestrator (574 lines)
├── assemblers.py                  # Reference resolver, prompt assembler
├── manifest_builder.py            # JSON manifest generation
├── validators.py                  # Cross-reference validation
├── types.py                       # BehaviorPrimitive, CompilationError
└── cli.py                         # CLI entry point (qf-compile command)
```

**Current Capabilities:**
1. ✅ Loads behavior primitives (expertises, procedures, snippets, playbooks, adapters)
2. ✅ Validates cross-references (@type:id syntax)
3. ✅ Generates **Manifests** (JSON) - Runtime-ready for PlaybookExecutor
4. ✅ Generates **Standalone Prompts** (Markdown) - Full assembled prompts for LLMs
5. ✅ CLI command: `qf-compile` (defined in lib/python/pyproject.toml:73)

**Current Entry Point:**
```toml
[project.scripts]
qf-compile = "questfoundry.compiler.cli:main"
```

**Dependencies (from lib/python/pyproject.toml):**
- pydantic>=2.0
- jsonschema>=4.0
- pyyaml>=6.0

---

## Target Architecture

### Three-Component Structure

```
questfoundry/
├── lib/
│   ├── spec_compiler/              # NEW: Extracted standalone library
│   │   ├── src/
│   │   │   └── spec_compiler/
│   │   │       ├── __init__.py
│   │   │       ├── compiler.py     # Main SpecCompiler (no CLI)
│   │   │       ├── assemblers.py
│   │   │       ├── manifest_builder.py
│   │   │       ├── validators.py
│   │   │       └── types.py
│   │   ├── tests/
│   │   ├── pyproject.toml
│   │   └── README.md
│   │
│   └── python/                     # UPDATED: Uses lib/spec_compiler
│       ├── src/questfoundry/
│       │   ├── compiler/           # REMOVED or DEPRECATED
│       │   └── ...                 # Rest of questfoundry library
│       └── pyproject.toml          # Add dependency: spec_compiler
│
└── cli/
    └── python/                     # NEW: Unified qf CLI
        ├── src/
        │   └── qf_cli/
        │       ├── __init__.py
        │       ├── main.py         # CLI entry point
        │       ├── commands/
        │       │   ├── __init__.py
        │       │   ├── spec.py     # qf spec subcommands
        │       │   ├── run.py      # qf run subcommands
        │       │   ├── generate.py # qf generate subcommands
        │       │   └── config.py   # qf config subcommands
        │       └── utils/
        │           ├── __init__.py
        │           ├── output.py   # Rich output formatting
        │           └── config.py   # Configuration management
        ├── tests/
        ├── pyproject.toml
        └── README.md
```

---

## Phase 1: Extract Spec Compiler Library

### Objective

Create a **standalone, reusable** spec compiler library at `lib/spec_compiler/` with **no CLI dependencies**.

### Tasks

#### 1.1 Create New Library Structure

**Location:** `lib/spec_compiler/`

**Files to create:**
```
lib/spec_compiler/
├── src/
│   └── spec_compiler/
│       ├── __init__.py
│       ├── compiler.py              # Renamed from spec_compiler.py
│       ├── assemblers.py
│       ├── manifest_builder.py
│       ├── validators.py
│       └── types.py
├── tests/
│   ├── __init__.py
│   ├── test_compiler.py
│   ├── test_assemblers.py
│   ├── test_manifest_builder.py
│   └── test_validators.py
├── pyproject.toml
├── README.md
└── LICENSE
```

#### 1.2 Extract and Clean Compiler Code

**Source:** `lib/python/src/questfoundry/compiler/*.py`
**Destination:** `lib/spec_compiler/src/spec_compiler/`

**Changes Required:**

1. **Remove CLI code** - Extract `cli.py` logic to Phase 2
2. **Update imports** - Change `questfoundry.compiler.*` → `spec_compiler.*`
3. **Keep pure library functions** - Only compilation, validation, assembly logic
4. **Remove CLI-specific output** - No print() statements, use logging instead

**Files to copy and clean:**
- ✅ `types.py` → Copy as-is (BehaviorPrimitive, CompilationError)
- ✅ `spec_compiler.py` → Rename to `compiler.py`, remove CLI references
- ✅ `assemblers.py` → Copy as-is (ReferenceResolver, StandalonePromptAssembler)
- ✅ `manifest_builder.py` → Copy as-is
- ✅ `validators.py` → Copy as-is
- ❌ `cli.py` → Do NOT copy (move logic to qf CLI in Phase 2)

#### 1.3 Create lib/spec_compiler/pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "questfoundry-spec-compiler"
version = "0.1.0"
description = "Standalone spec compiler for QuestFoundry behavior primitives"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [{name = "Peter Liesdonk", email = "peter@liesdonk.nl"}]

dependencies = [
    "pydantic>=2.0",
    "jsonschema>=4.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/spec_compiler"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.ruff]
line-length = 88
target-version = "py311"
```

#### 1.4 Update lib/python to Use New Library

**File:** `lib/python/pyproject.toml`

**Add dependency:**
```toml
dependencies = [
    "questfoundry-spec-compiler",  # NEW
    "pydantic>=2.0",
    "jsonschema>=4.0",
    # ... existing deps
]
```

**Remove old compiler:**
```bash
# Deprecate or remove lib/python/src/questfoundry/compiler/
# Update all internal imports
```

**Update imports in lib/python:**
```python
# OLD
from questfoundry.compiler import SpecCompiler

# NEW
from spec_compiler import SpecCompiler
```

#### 1.5 Testing Strategy

**Port existing tests:**
- Copy tests from `lib/python/tests/compiler/` → `lib/spec_compiler/tests/`
- Update imports
- Ensure 100% test coverage for extracted code

**Run full test suite:**
```bash
cd lib/spec_compiler
uv sync
uv run pytest
```

---

## Phase 2: Create Unified qf CLI

### Objective

Build a **unified CLI** with modern UX using **typer**, **questionary**, and **rich**.

### Library Stack

From the gist specification:
- **typer** - CLI framework (type-safe, modern)
- **questionary** - Interactive prompts
- **rich** - Terminal formatting and progress bars

### CLI Command Structure

```
qf
├── spec
│   ├── compile              # Compile behavior primitives
│   ├── validate             # Validate references only
│   └── watch                # Watch mode (future)
│
├── run
│   ├── init                 # Initialize workspace
│   ├── send                 # Send message to role
│   ├── chat                 # Interactive chat mode
│   └── playbook             # Execute playbook
│
├── generate
│   └── prompt               # Generate standalone prompts
│
└── config
    ├── doctor               # Diagnose configuration
    ├── list-playbooks       # List available playbooks
    ├── list-roles           # List available roles
    └── set-default          # Set default configuration
```

### Tasks

#### 2.1 Create CLI Project Structure

**Location:** `cli/python/`

**Initialize project:**
```bash
cd cli/python
# Create pyproject.toml with dependencies
```

#### 2.2 Create cli/python/pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "questfoundry-cli"
version = "0.1.0"
description = "Unified CLI for QuestFoundry - spec compilation, agentic runtime, and prompt generation"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [{name = "Peter Liesdonk", email = "peter@liesdonk.nl"}]

dependencies = [
    "questfoundry-spec-compiler",    # Spec compiler library
    "questfoundry-py",                # Main QuestFoundry library
    "typer>=0.9.0",                   # CLI framework
    "questionary>=2.0.0",             # Interactive prompts
    "rich>=13.0.0",                   # Terminal formatting
    "click>=8.0.0",                   # typer dependency
]

[project.scripts]
qf = "qf_cli.main:app"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/qf_cli"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

#### 2.3 Implement CLI Entry Point

**File:** `cli/python/src/qf_cli/main.py`

```python
"""QuestFoundry CLI - Unified command-line interface."""

import typer
from rich.console import Console

from qf_cli.commands import config, generate, run, spec

app = typer.Typer(
    name="qf",
    help="QuestFoundry - Collaborative Interactive Fiction Authoring",
    add_completion=True,
)

console = Console()

# Register command groups
app.add_typer(spec.app, name="spec", help="Spec compilation and validation")
app.add_typer(run.app, name="run", help="Agentic runtime interface")
app.add_typer(generate.app, name="generate", help="Prompt generation tools")
app.add_typer(config.app, name="config", help="Configuration utilities")


@app.callback()
def main():
    """QuestFoundry CLI - Layer 7 orchestration."""
    pass


if __name__ == "__main__":
    app()
```

#### 2.4 Implement `qf spec` Commands

**File:** `cli/python/src/qf_cli/commands/spec.py`

```python
"""Spec compilation commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from spec_compiler import SpecCompiler, CompilationError

app = typer.Typer()
console = Console()


@app.command()
def compile(
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
    output: Path = typer.Option(
        Path("dist/compiled"),
        "--output", "-o",
        help="Output directory for compiled artifacts",
    ),
    playbook: Optional[str] = typer.Option(
        None,
        "--playbook",
        help="Compile only specific playbook (by ID)",
    ),
    adapter: Optional[str] = typer.Option(
        None,
        "--adapter",
        help="Compile only specific adapter (by ID)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
):
    """Compile behavior primitives into runtime artifacts."""

    # Validate spec directory
    if not spec_dir.exists():
        console.print(f"[red]Error:[/red] Spec directory not found: {spec_dir}")
        raise typer.Exit(1)

    behavior_dir = spec_dir / "05-behavior"
    if not behavior_dir.exists():
        console.print(f"[red]Error:[/red] Behavior directory not found: {behavior_dir}")
        raise typer.Exit(1)

    # Initialize compiler
    compiler = SpecCompiler(spec_dir)

    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading primitives...", total=100)
            compiler.load_all_primitives()
            progress.update(task, advance=50)

            # Validate
            from spec_compiler.validators import ReferenceValidator
            validator = ReferenceValidator(compiler.primitives, compiler.spec_root)
            errors = validator.validate_all()
            progress.update(task, advance=50)

        if errors:
            actual_errors = [e for e in errors if not e.startswith("Warning:")]
            warnings = [e for e in errors if e.startswith("Warning:")]

            if actual_errors:
                console.print("[red]Validation errors:[/red]")
                for error in actual_errors:
                    console.print(f"  ❌ {error}")
                raise typer.Exit(1)

            if warnings:
                console.print("[yellow]Validation warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  ⚠️  {warning}")

        console.print("✅ [green]Validation passed[/green]")

        # Compile
        if playbook or adapter:
            if playbook:
                result = compiler.compile_playbook(playbook, output)
                console.print(f"✅ Generated: {result['manifest_path']}")

            if adapter:
                result = compiler.compile_adapter(adapter, output)
                console.print(f"✅ Generated: {result['manifest_path']}")
                console.print(f"✅ Generated: {result['prompt_path']}")
        else:
            with Progress() as progress:
                task = progress.add_task("[cyan]Compiling...", total=100)
                stats = compiler.compile_all(output)
                progress.update(task, advance=100)

            # Display stats table
            table = Table(title="📦 Compilation Complete")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green", justify="right")

            table.add_row("Primitives loaded", str(stats['primitives_loaded']))
            table.add_row("Playbook manifests", str(stats['playbook_manifests_generated']))
            table.add_row("Adapter manifests", str(stats['adapter_manifests_generated']))
            table.add_row("Standalone prompts", str(stats['standalone_prompts_generated']))

            console.print(table)
            console.print(f"\n[green]Output directory:[/green] {output}")

    except CompilationError as e:
        console.print(f"[red]Compilation failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
):
    """Validate references without generating output."""

    compiler = SpecCompiler(spec_dir)
    compiler.load_all_primitives()

    from spec_compiler.validators import ReferenceValidator
    validator = ReferenceValidator(compiler.primitives, compiler.spec_root)
    errors = validator.validate_all()

    if errors:
        actual_errors = [e for e in errors if not e.startswith("Warning:")]

        if actual_errors:
            console.print("[red]Validation errors:[/red]")
            for error in actual_errors:
                console.print(f"  ❌ {error}")
            raise typer.Exit(1)

    console.print("✅ [green]Validation passed[/green]")


@app.command()
def watch(
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
    output: Path = typer.Option(
        Path("dist/compiled"),
        "--output", "-o",
        help="Output directory for compiled artifacts",
    ),
):
    """Watch mode: recompile on file changes (not yet implemented)."""
    console.print("[yellow]Watch mode not yet implemented[/yellow]")
    raise typer.Exit(1)
```

#### 2.5 Implement `qf generate` Commands

**File:** `cli/python/src/qf_cli/commands/generate.py`

```python
"""Prompt generation commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from spec_compiler import SpecCompiler

app = typer.Typer()
console = Console()


@app.command()
def prompt(
    adapter: str = typer.Argument(
        ...,
        help="Adapter ID (role name) to generate prompt for",
    ),
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file (default: stdout)",
    ),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format: markdown, json",
    ),
):
    """Generate standalone prompt for a role adapter.

    This assembles a complete, self-contained prompt by:
    1. Loading the adapter configuration
    2. Resolving all @references (expertises, procedures, snippets)
    3. Inlining referenced content into a single markdown file
    4. Formatting for LLM consumption (GPT-4, Claude Sonnet 4.5+)

    Example:
        qf generate prompt lore_weaver
        qf generate prompt scene_smith --output prompts/scene_smith.md
    """

    # Load compiler
    compiler = SpecCompiler(spec_dir)
    compiler.load_all_primitives()

    # Check adapter exists
    adapter_key = f"adapter:{adapter}"
    if adapter_key not in compiler.primitives:
        console.print(f"[red]Error:[/red] Adapter not found: {adapter}")
        console.print("\n[yellow]Available adapters:[/yellow]")
        for key in sorted(compiler.primitives.keys()):
            if key.startswith("adapter:"):
                console.print(f"  - {key.split(':')[1]}")
        raise typer.Exit(1)

    # Generate prompt
    from spec_compiler.assemblers import ReferenceResolver, StandalonePromptAssembler

    resolver = ReferenceResolver(compiler.primitives, compiler.spec_root)
    assembler = StandalonePromptAssembler(compiler.primitives, resolver, compiler.spec_root)

    prompt_text = assembler.assemble_role_prompt(adapter)

    # Output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(prompt_text)
        console.print(f"✅ [green]Generated prompt:[/green] {output}")
    else:
        # Print to stdout with syntax highlighting
        syntax = Syntax(prompt_text, "markdown", theme="monokai", line_numbers=False)
        console.print(syntax)
```

#### 2.6 Implement `qf run` Commands (Stub)

**File:** `cli/python/src/qf_cli/commands/run.py`

```python
"""Agentic runtime interface commands."""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def init():
    """Initialize a QuestFoundry workspace."""
    console.print("[yellow]qf run init: Not yet implemented[/yellow]")


@app.command()
def send():
    """Send a message to a role."""
    console.print("[yellow]qf run send: Not yet implemented[/yellow]")


@app.command()
def chat():
    """Interactive chat mode with a role."""
    console.print("[yellow]qf run chat: Not yet implemented[/yellow]")


@app.command()
def playbook():
    """Execute a playbook."""
    console.print("[yellow]qf run playbook: Not yet implemented[/yellow]")
```

#### 2.7 Implement `qf config` Commands

**File:** `cli/python/src/qf_cli/commands/config.py`

```python
"""Configuration utilities."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from spec_compiler import SpecCompiler

app = typer.Typer()
console = Console()


@app.command()
def doctor():
    """Diagnose QuestFoundry configuration."""

    console.print("[cyan]QuestFoundry Configuration Doctor[/cyan]\n")

    # Check spec directory
    spec_dir = Path("spec")
    if spec_dir.exists():
        console.print(f"✅ Spec directory found: {spec_dir}")
    else:
        console.print(f"❌ Spec directory not found: {spec_dir}")

    # Check behavior primitives
    behavior_dir = spec_dir / "05-behavior"
    if behavior_dir.exists():
        console.print(f"✅ Behavior directory found: {behavior_dir}")

        # Count primitives
        compiler = SpecCompiler(spec_dir)
        compiler.load_all_primitives()

        table = Table(title="Loaded Primitives")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="green", justify="right")

        counts = {}
        for key in compiler.primitives.keys():
            prim_type = key.split(':')[0]
            counts[prim_type] = counts.get(prim_type, 0) + 1

        for prim_type, count in sorted(counts.items()):
            table.add_row(prim_type, str(count))

        console.print(table)
    else:
        console.print(f"❌ Behavior directory not found: {behavior_dir}")


@app.command()
def list_playbooks(
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
):
    """List available playbooks."""

    compiler = SpecCompiler(spec_dir)
    compiler.load_all_primitives()

    playbooks = [
        key.split(':')[1]
        for key in sorted(compiler.primitives.keys())
        if key.startswith("playbook:")
    ]

    if playbooks:
        table = Table(title="Available Playbooks")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")

        for playbook_id in playbooks:
            prim = compiler.primitives[f"playbook:{playbook_id}"]
            name = prim.metadata.get('playbook_name', playbook_id)
            table.add_row(playbook_id, name)

        console.print(table)
    else:
        console.print("[yellow]No playbooks found[/yellow]")


@app.command()
def list_roles(
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
):
    """List available roles (adapters)."""

    compiler = SpecCompiler(spec_dir)
    compiler.load_all_primitives()

    adapters = [
        key.split(':')[1]
        for key in sorted(compiler.primitives.keys())
        if key.startswith("adapter:")
    ]

    if adapters:
        table = Table(title="Available Roles")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Abbreviation", style="yellow")

        for adapter_id in adapters:
            prim = compiler.primitives[f"adapter:{adapter_id}"]
            name = prim.metadata.get('role_name', adapter_id)
            abbr = prim.metadata.get('abbreviation', '—')
            table.add_row(adapter_id, name, abbr)

        console.print(table)
    else:
        console.print("[yellow]No roles found[/yellow]")


@app.command()
def set_default():
    """Set default configuration."""
    console.print("[yellow]qf config set-default: Not yet implemented[/yellow]")
```

---

## Implementation Sequence

### Recommended Order for LLM Implementation

1. **Phase 1.1-1.3** - Extract spec_compiler library structure
2. **Phase 1.4** - Update lib/python dependencies
3. **Phase 1.5** - Port and run tests
4. **Phase 2.1-2.2** - Create CLI project structure
5. **Phase 2.3** - Implement CLI entry point
6. **Phase 2.4** - Implement `qf spec` commands
7. **Phase 2.5** - Implement `qf generate` commands
8. **Phase 2.7** - Implement `qf config` commands
9. **Phase 2.6** - Stub `qf run` commands (future work)

---

## Testing Strategy

### Unit Tests

**lib/spec_compiler/tests/**
- Port existing compiler tests from lib/python
- Add new tests for extracted functionality
- Target: 100% code coverage

**cli/python/tests/**
- Test each command with typer testing utilities
- Mock spec_compiler calls
- Test output formatting with rich
- Test error handling

### Integration Tests

**End-to-end workflows:**
1. Extract library → Update lib/python → Run full test suite
2. Install CLI → Run all commands → Verify output
3. Compile spec → Generate prompts → Validate format

### Manual Testing

```bash
# Phase 1 validation
cd lib/spec_compiler
uv sync
uv run pytest
cd ../python
uv sync
uv run pytest

# Phase 2 validation
cd ../../cli/python
uv sync
qf --help
qf spec compile
qf spec validate
qf generate prompt lore_weaver
qf config doctor
qf config list-playbooks
qf config list-roles
```

---

## Migration Path

### Backward Compatibility

**Deprecation timeline:**
1. ✅ Extract spec_compiler (v0.1.0)
2. ✅ Update lib/python to use new library (v0.7.0)
3. ⚠️ Deprecate `qf-compile` command (v0.8.0)
4. ❌ Remove old compiler from lib/python (v1.0.0)

**Migration guide for users:**
```bash
# OLD
qf-compile --spec-dir spec/ --output dist/compiled/

# NEW
qf spec compile --spec-dir spec/ --output dist/compiled/
```

---

## Dependencies

### lib/spec_compiler/

```toml
dependencies = [
    "pydantic>=2.0",
    "jsonschema>=4.0",
    "pyyaml>=6.0",
]
```

### cli/python/

```toml
dependencies = [
    "questfoundry-spec-compiler",    # Phase 1 output
    "questfoundry-py",                # Existing library
    "typer>=0.9.0",
    "questionary>=2.0.0",
    "rich>=13.0.0",
    "click>=8.0.0",
]
```

---

## Success Criteria

### Phase 1 Complete When:
- ✅ lib/spec_compiler/ is a standalone installable package
- ✅ All tests pass in lib/spec_compiler/
- ✅ lib/python successfully uses the new library
- ✅ All existing functionality works without regression

### Phase 2 Complete When:
- ✅ `qf` command is installed and discoverable
- ✅ `qf spec compile` produces identical output to old `qf-compile`
- ✅ `qf generate prompt` generates valid standalone prompts
- ✅ `qf config` commands provide useful diagnostics
- ✅ All commands have `--help` documentation
- ✅ Rich output formatting works correctly

---

## Open Questions

1. **Versioning Strategy:** How should lib/spec_compiler and cli/python versions be coordinated?
2. **Configuration Storage:** Where should `qf config` store user preferences? (~/.config/questfoundry/?)
3. **Watch Mode:** What file watching library to use? (watchdog?)
4. **qf run:** What is the priority for implementing runtime commands?
5. **Distribution:** Should these be published to PyPI?

---

## Related Documents

- **Gist:** https://gist.github.com/pvliesdonk/811a448413ea751b1da5ad5f50b75a81
- **Spec Readme:** spec/05-behavior/README.md
- **Existing Compiler:** lib/python/src/questfoundry/compiler/

---

## Notes for LLM Implementation

**Claude Sonnet 4.5 Optimization:**

1. **Code Extraction:** The compiler code is well-structured and mostly ready for extraction
2. **Minimal Changes:** Most files can be copied with only import path updates
3. **CLI Framework:** Typer provides type-safe CLI with minimal boilerplate
4. **Testing:** Existing tests can be adapted with minimal changes
5. **Dependencies:** All required libraries are stable and well-documented

**Implementation Tips:**

- Start with Phase 1 (extraction) before Phase 2 (CLI)
- Run tests frequently to catch regressions early
- Use rich for all user-facing output (consistent UX)
- Follow existing code style (ruff, mypy strict)
- Keep CLI commands thin - delegate to libraries

**Potential Pitfalls:**

- Import path changes - use global search/replace carefully
- Circular dependencies - ensure clean separation
- CLI output - avoid print() in library code, use logging
- Configuration - plan for future extensibility

---

**End of Implementation Plan**
