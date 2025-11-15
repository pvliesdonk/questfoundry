# QF CLI Implementation Plan

**Version:** 1.0.0
**Date:** 2025-11-15
**Optimized for:** Claude Sonnet 4.5 LLM Implementation
**Based on:** [Gist 811a448413ea751b1da5ad5f50b75a81](https://gist.github.com/pvliesdonk/811a448413ea751b1da5ad5f50b75a81)

---

## Executive Summary

This plan describes the implementation of **three separate components** for the QuestFoundry ecosystem:

1. **`lib/spec_compiler/`** - Standalone spec compiler library (extracted from lib/python)
2. **`tools/prompt_generator/`** - Standalone prompt generation tool (depends on spec_compiler only)
3. **`cli/python/`** - Unified qf CLI (depends on both spec_compiler and questfoundry-py)

### Critical Understanding

These are **THREE SEPARATE PROGRAMS/REPOS**:
- **lib/spec_compiler/** - A reusable library with no CLI dependencies
- **tools/prompt_generator/** - A standalone tool that uses spec_compiler (NOT part of qf CLI)
- **cli/python/ (qf CLI)** - The unified command-line interface for QuestFoundry runtime

The prompt generator is **independent** from the qf CLI. Any similarity in command names is coincidental.

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
├── tools/
│   └── prompt_generator/           # NEW: Standalone prompt generator
│       ├── src/
│       │   └── qf_prompt_gen/
│       │       ├── __init__.py
│       │       ├── cli.py          # CLI entry point
│       │       ├── generator.py    # Prompt generation logic
│       │       └── formatters.py   # Output formatting
│       ├── tests/
│       ├── pyproject.toml          # Depends ONLY on spec_compiler
│       └── README.md
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
        │       │   └── config.py   # qf config subcommands
        │       └── utils/
        │           ├── __init__.py
        │           ├── output.py   # Rich output formatting
        │           └── config.py   # Configuration management
        ├── tests/
        ├── pyproject.toml          # Depends on spec_compiler + questfoundry-py
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

#### 1.3 Study Spec Layers for Documentation

**CRITICAL**: Before writing any READMEs or documentation, the implementing LLM MUST:

1. **Read and understand spec layers:**
   - `spec/00-north-star/` - Vision, principles, quality bars
   - `spec/01-roles/` - Role definitions and charters
   - `spec/02-dictionary/` - Artifact templates and terminology
   - `spec/03-schemas/` - JSON schema definitions
   - `spec/04-protocol/` - Communication protocols
   - `spec/05-behavior/` - Behavioral primitives (the source material)

2. **Understand the compilation pipeline:**
   - How Layer 5 primitives are loaded
   - How @references are resolved
   - How manifests are generated
   - How standalone prompts are assembled

3. **Document with context:**
   - READMEs should explain HOW the component fits into the QuestFoundry architecture
   - READMEs should reference the appropriate spec layers
   - READMEs should include concrete examples from actual spec files

**Time Investment**: Spend 10-15 minutes reading spec files before writing documentation.

#### 1.4 Create lib/spec_compiler/pyproject.toml

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

#### 1.5 Create Comprehensive README for lib/spec_compiler/

**File:** `lib/spec_compiler/README.md`

**Requirements:**
- Explain what the spec compiler does and WHY it exists
- Describe the Layer 5 behavioral primitive architecture
- Show the compilation pipeline with diagrams
- Include real examples from spec/05-behavior/
- Document all @reference types and resolution rules
- Provide usage examples for both manifest and prompt generation
- Link to relevant spec layers (0-4) for context
- Include API documentation for SpecCompiler, ReferenceResolver, etc.

**Minimum sections:**
1. Overview - What is this and why does it exist?
2. Architecture - How does it fit into QuestFoundry?
3. Behavioral Primitives - What are they? (with examples)
4. Compilation Pipeline - Step-by-step process
5. API Reference - Classes and methods
6. Usage Examples - Common workflows
7. Testing - How to run tests
8. References - Links to spec layers

**Example content to include:**
- Show an actual adapter.yaml from spec/05-behavior/adapters/
- Show how @procedure:canonization_core resolves
- Show a compiled manifest output
- Show a generated standalone prompt

#### 1.6 Update lib/python to Use New Library

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

#### 1.7 Update lib/python README

**File:** `lib/python/README.md`

**Updates Required:**
- Add section on spec_compiler dependency
- Update installation instructions
- Document the change from embedded compiler to external library
- Add migration guide for existing code using questfoundry.compiler
- Update architecture diagram to show spec_compiler as external dependency

#### 1.8 Testing Strategy

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

## Phase 2: Create Standalone Prompt Generator

### Objective

Build a **standalone prompt generation tool** that depends ONLY on `lib/spec_compiler/` (not on questfoundry-py).

### Purpose

Generate standalone, self-contained prompts for LLM consumption by:
1. Loading adapter configurations
2. Resolving all @references (expertises, procedures, snippets)
3. Inlining referenced content into a single markdown file
4. Formatting for LLM consumption (GPT-4, Claude Sonnet 4.5+)

### Command Structure

```
qf-prompt-gen
├── generate              # Generate prompt for a role
├── list                  # List available roles
└── validate              # Validate adapter references
```

## Phase 3: Create Unified qf CLI

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
└── config
    ├── doctor               # Diagnose configuration
    ├── list-playbooks       # List available playbooks
    ├── list-roles           # List available roles
    └── set-default          # Set default configuration
```

**Note:** The qf CLI does NOT include prompt generation - that's handled by the separate `qf-prompt-gen` tool.

### Tasks

#### 2.1 Create Prompt Generator Project Structure

**Location:** `tools/prompt_generator/`

**Initialize project:**
```bash
mkdir -p tools/prompt_generator/src/qf_prompt_gen
mkdir -p tools/prompt_generator/tests
cd tools/prompt_generator
```

#### 2.2 Create tools/prompt_generator/pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "questfoundry-prompt-generator"
version = "0.1.0"
description = "Standalone prompt generator for QuestFoundry role adapters"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [{name = "Peter Liesdonk", email = "peter@liesdonk.nl"}]

dependencies = [
    "questfoundry-spec-compiler",    # ONLY dependency
    "typer>=0.9.0",
    "rich>=13.0.0",
    "click>=8.0.0",
]

[project.scripts]
qf-prompt-gen = "qf_prompt_gen.cli:app"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/qf_prompt_gen"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

#### 2.3 Implement Prompt Generator CLI

**File:** `tools/prompt_generator/src/qf_prompt_gen/cli.py`

```python
"""QuestFoundry Prompt Generator CLI."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from spec_compiler import SpecCompiler, CompilationError
from spec_compiler.assemblers import ReferenceResolver, StandalonePromptAssembler

app = typer.Typer(
    name="qf-prompt-gen",
    help="Generate standalone prompts for QuestFoundry role adapters",
    add_completion=True,
)

console = Console()


@app.command()
def generate(
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
        help="Output format: markdown",
    ),
):
    """Generate standalone prompt for a role adapter.

    This assembles a complete, self-contained prompt by:
    1. Loading the adapter configuration
    2. Resolving all @references (expertises, procedures, snippets)
    3. Inlining referenced content into a single markdown file
    4. Formatting for LLM consumption (GPT-4, Claude Sonnet 4.5+)

    Example:
        qf-prompt-gen generate lore_weaver
        qf-prompt-gen generate scene_smith --output prompts/scene_smith.md
    """

    try:
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
        resolver = ReferenceResolver(compiler.primitives, compiler.spec_root)
        assembler = StandalonePromptAssembler(
            compiler.primitives, resolver, compiler.spec_root
        )

        prompt_text = assembler.assemble_role_prompt(adapter)

        # Output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(prompt_text)
            console.print(f"✅ [green]Generated prompt:[/green] {output}")
            console.print(f"   Characters: {len(prompt_text):,}")
            console.print(f"   Lines: {len(prompt_text.splitlines()):,}")
        else:
            # Print to stdout with syntax highlighting
            syntax = Syntax(prompt_text, "markdown", theme="monokai", line_numbers=False)
            console.print(syntax)

    except CompilationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list(
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
):
    """List available role adapters."""

    try:
        compiler = SpecCompiler(spec_dir)
        compiler.load_all_primitives()

        adapters = [
            key.split(':')[1]
            for key in sorted(compiler.primitives.keys())
            if key.startswith("adapter:")
        ]

        if adapters:
            table = Table(title="Available Role Adapters")
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
            console.print("[yellow]No adapters found[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    spec_dir: Path = typer.Option(
        Path("spec"),
        "--spec-dir",
        help="Root directory of spec/",
    ),
    adapter: Optional[str] = typer.Option(
        None,
        "--adapter",
        help="Validate specific adapter only",
    ),
):
    """Validate adapter references."""

    try:
        compiler = SpecCompiler(spec_dir)
        compiler.load_all_primitives()

        from spec_compiler.validators import ReferenceValidator

        validator = ReferenceValidator(compiler.primitives, compiler.spec_root)
        errors = validator.validate_all()

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

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

#### 2.4 Create Comprehensive README

**File:** `tools/prompt_generator/README.md`

**Requirements:**
- Explain the TWO consumption models for Layer 5 primitives:
  1. Automated multi-agent (lib/python) - Agentic orchestration
  2. Manual human+AI chat (prompt_generator) - Single AI workflows
- Show WHY someone would use this instead of the agentic runtime
- Provide concrete use cases and examples
- Include a complete example showing input (adapter) → output (prompt)
- Explain how @references are resolved and inlined
- Document the relationship to spec layers 0-5

**Minimum sections:**
1. **Overview** - What is this and when to use it?
2. **Architecture** - Two paths for consuming Layer 5
3. **Use Cases** - When to use prompt generator vs agentic runtime
4. **Installation** - Setup instructions
5. **Usage** - All commands with examples
6. **Output Format** - What the generated prompts look like
7. **Examples** - Complete end-to-end example
8. **Dependencies** - Why ONLY spec_compiler (not questfoundry-py)
9. **References** - Links to spec layers

**Example content:**

```markdown
# QuestFoundry Prompt Generator

Standalone tool to generate self-contained LLM prompts from QuestFoundry role adapters.

## Overview

QuestFoundry's Layer 5 behavioral primitives can be consumed in two ways:

1. **Automated Multi-Agent Orchestration** (`lib/python`) - The Showrunner coordinates multiple AI roles working together autonomously
2. **Manual Human+AI Chat** (`prompt_generator`) - Generate standalone prompts for direct human-in-the-loop workflows with a single AI

This tool implements the second approach.

### When to Use This Tool

Use the prompt generator when:
- You want direct control over AI interactions (not autonomous)
- You're working with a single role in a chat interface
- You need a portable, self-contained prompt for Claude/GPT
- You're prototyping or exploring role behaviors
- You don't need multi-agent orchestration

Use `lib/python` (agentic runtime) when:
- You want autonomous multi-agent collaboration
- You need the Showrunner to coordinate work
- You're executing complete playbooks
- You need state management and Hot/Cold storage

## Architecture

```
spec/05-behavior/          # Layer 5: Behavioral primitives (source)
│
├──> spec_compiler         # Compilation engine
│    ├──> Manifests       # → lib/python (agentic runtime)
│    └──> Prompts         # → prompt_generator (manual chat)
│
├──> lib/python            # Automated multi-agent orchestration
│                          # Uses manifests, includes workspace, roles, etc.
│
└──> prompt_generator      # Manual human+AI chat
                           # Uses prompts, NO runtime dependencies
```

## Installation

```bash
cd tools/prompt_generator
uv sync
```

## Usage

### Generate a prompt for a role

```bash
# Output to stdout (with syntax highlighting)
qf-prompt-gen generate lore_weaver

# Save to file
qf-prompt-gen generate scene_smith --output prompts/scene_smith.md

# Specify spec directory
qf-prompt-gen generate gatekeeper --spec-dir /path/to/spec
```

### List available roles

```bash
qf-prompt-gen list
```

Output:
```
Available Role Adapters
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ ID               ┃ Name           ┃ Abbreviation ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ lore_weaver      │ Lore Weaver    │ LW           │
│ scene_smith      │ Scene Smith    │ SS           │
│ gatekeeper       │ Gatekeeper     │ GK           │
└──────────────────┴────────────────┴──────────────┘
```

### Validate adapter references

```bash
qf-prompt-gen validate
```

## Output Format

Generated prompts are comprehensive, self-contained markdown files that include:

1. **Mission** - Role's core purpose from Layer 1 charter
2. **Core Expertise** - Domain knowledge (inlined from @expertise references)
3. **Primary Procedures** - Workflow algorithms (inlined from @procedure references)
4. **Safety & Validation** - Critical reminders (inlined from @snippet references)
5. **Protocol Intents** - What messages the role sends/receives
6. **Loop Participation** - Which playbooks the role participates in
7. **Escalation Rules** - When to ask humans or wake Showrunner

All @references are resolved and inlined, creating a single file ready for LLM consumption.

## Example: Generating a Lore Weaver Prompt

### Input Adapter

`spec/05-behavior/adapters/lore_weaver.adapter.yaml`:
```yaml
adapter_id: lore_weaver
role_name: Lore Weaver
expertise: "@expertise:lore_weaver_expertise"
procedures:
  primary:
    - "@procedure:canonization_core"
    - "@procedure:continuity_check"
safety_protocols:
  - "@snippet:spoiler_hygiene_check"
```

### Command

```bash
qf-prompt-gen generate lore_weaver --output lore_weaver_prompt.md
```

### Output

`lore_weaver_prompt.md` (simplified excerpt):
```markdown
# Lore Weaver — System Prompt

Target: GPT-5, Claude Sonnet 4.5+

## Mission

Resolve the world's deep truth—quietly—then hand clear, spoiler-safe
summaries to neighbors who face the player.

## Core Expertise

[Full content of spec/05-behavior/expertises/lore_weaver_expertise.md
inlined here - 500+ lines covering canon creation, continuity management,
mystery handling, etc.]

## Primary Procedures

### Canonization Core

[Full content of spec/05-behavior/procedures/canonization_core.md
inlined here - algorithm for transforming hooks into canon]

### Continuity Check

[Full content of spec/05-behavior/procedures/continuity_check.md
inlined here - contradiction detection process]

## Safety & Validation

[Full content of spec/05-behavior/snippets/spoiler_hygiene_check.md
inlined here - PN boundary enforcement]
...
```

Result: A ~2000-line standalone prompt ready to paste into Claude/GPT.

## Dependencies

This tool depends ONLY on:
- `questfoundry-spec-compiler` - Compiles Layer 5 primitives
- `typer` - CLI framework
- `rich` - Terminal formatting

**Important:** This tool does NOT depend on `questfoundry-py` because it doesn't
need the agentic runtime, workspace management, or role orchestration. It only
needs to compile and assemble Layer 5 primitives into standalone prompts.

## How It Works

1. **Load primitives** - Reads all files from `spec/05-behavior/`
2. **Find adapter** - Locates the requested role adapter (e.g., `lore_weaver.adapter.yaml`)
3. **Resolve references** - Follows all @expertise, @procedure, @snippet references
4. **Inline content** - Replaces references with actual file contents
5. **Assemble prompt** - Combines everything into a structured markdown file
6. **Format for LLM** - Adds headers, sections, and metadata

## References

- **Spec Overview**: `spec/README.md`
- **Layer 5 Behavioral Primitives**: `spec/05-behavior/README.md`
- **Role Charters** (Layer 1): `spec/01-roles/`
- **Artifact Templates** (Layer 2): `spec/02-dictionary/`
- **JSON Schemas** (Layer 3): `spec/03-schemas/`
- **Spec Compiler Library**: `lib/spec_compiler/README.md`
- **Agentic Runtime** (alternate approach): `lib/python/README.md`

## Contributing

See main repository `CONTRIBUTING.md` for guidelines.

## License

MIT License - See `LICENSE` file in repository root.
```

#### 2.5 Create Examples Directory

**Location:** `tools/prompt_generator/examples/`

Create example outputs:
- `examples/lore_weaver_prompt.md` - Full generated prompt for Lore Weaver
- `examples/gatekeeper_prompt.md` - Full generated prompt for Gatekeeper
- `examples/README.md` - Explains the examples and how to regenerate them

---

## Phase 3 Tasks (Unified qf CLI)

### Tasks

#### 3.1 Create CLI Project Structure

**Location:** `cli/python/`

**Initialize project:**
```bash
cd cli/python
# Create pyproject.toml with dependencies
```

#### 3.2 Create cli/python/pyproject.toml

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

#### 3.3 Implement CLI Entry Point

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

#### 3.4 Implement `qf spec` Commands

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

#### 3.5 Implement `qf run` Commands (Stub)

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

#### 3.6 Implement `qf config` Commands

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

#### 3.7 Create Comprehensive README for qf CLI

**File:** `cli/python/README.md`

**Requirements:**
- Explain that this wraps the agentic runtime (lib/python)
- Distinguish from prompt_generator clearly
- Document all command groups (spec, run, config)
- Provide examples for each command
- Explain relationship to Layer 5 and spec_compiler
- Include architecture diagram showing CLI → lib/python → spec_compiler → spec/

**Minimum sections:**
1. **Overview** - What is the qf CLI?
2. **Architecture** - How it fits into QuestFoundry
3. **Installation** - Setup instructions
4. **Commands** - Complete command reference
   - `qf spec` - Compilation commands
   - `qf run` - Runtime commands (when implemented)
   - `qf config` - Configuration utilities
5. **Examples** - Common workflows
6. **Comparison** - When to use qf CLI vs prompt_generator vs direct library use
7. **References** - Links to lib/python, spec_compiler, spec layers

---

## Implementation Sequence

### Recommended Order for LLM Implementation

**Phase 1: Spec Compiler Extraction**
1. **Phase 1.1-1.2** - Create spec_compiler library structure
2. **Phase 1.3** - **CRITICAL: Study spec layers 0-5** (15 minutes minimum)
3. **Phase 1.4** - Create pyproject.toml
4. **Phase 1.5** - **Write comprehensive README** (using spec layer knowledge)
5. **Phase 1.6** - Update lib/python to use new library
6. **Phase 1.7** - Update lib/python README
7. **Phase 1.8** - Port and run tests

**Phase 2: Prompt Generator**
8. **Phase 2.1** - Create prompt generator project structure
9. **Phase 2.2** - Create pyproject.toml
10. **Phase 2.3** - Implement prompt generator CLI
11. **Phase 2.4** - **Write comprehensive README** (explain two consumption models)
12. **Phase 2.5** - Create example outputs

**Phase 3: QF CLI**
13. **Phase 3.1-3.2** - Create qf CLI project structure
14. **Phase 3.3** - Implement CLI entry point
15. **Phase 3.4** - Implement `qf spec` commands
16. **Phase 3.6** - Implement `qf config` commands
17. **Phase 3.7** - **Write comprehensive README**
18. **Phase 3.5** - Stub `qf run` commands (future work)

**Documentation Quality Gates:**
- Each README must be reviewed against spec layers for accuracy
- Each README must include concrete examples from actual spec files
- Each README must explain architectural context (not just usage)
- Each README must be minimum 200 lines (excluding code examples)

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
# OLD: Compilation
qf-compile --spec-dir spec/ --output dist/compiled/

# NEW: Compilation
qf spec compile --spec-dir spec/ --output dist/compiled/

# NEW: Prompt generation (separate tool)
qf-prompt-gen generate lore_weaver --output prompts/lore_weaver.md
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

### tools/prompt_generator/

```toml
dependencies = [
    "questfoundry-spec-compiler",    # Phase 1 output (ONLY dependency)
    "typer>=0.9.0",
    "rich>=13.0.0",
    "click>=8.0.0",
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
- ✅ **README.md is comprehensive** (minimum 200 lines, references spec layers)
- ✅ **README includes real examples** from spec/05-behavior/
- ✅ **lib/python README updated** to document spec_compiler dependency

### Phase 2 Complete When:
- ✅ `qf-prompt-gen` command is installed and discoverable
- ✅ `qf-prompt-gen generate` produces valid standalone prompts
- ✅ `qf-prompt-gen list` shows all available adapters
- ✅ `qf-prompt-gen validate` checks adapter references
- ✅ Tool depends ONLY on spec_compiler (not questfoundry-py)
- ✅ All commands have `--help` documentation
- ✅ Rich output formatting works correctly
- ✅ **README.md is comprehensive** (minimum 300 lines)
- ✅ **README explains two consumption models** (agentic vs manual)
- ✅ **README includes complete end-to-end example** (input adapter → output prompt)
- ✅ **examples/ directory contains generated prompts** (lore_weaver, gatekeeper)

### Phase 3 Complete When:
- ✅ `qf` command is installed and discoverable
- ✅ `qf spec compile` produces identical output to old `qf-compile`
- ✅ `qf config` commands provide useful diagnostics
- ✅ All commands have `--help` documentation
- ✅ Rich output formatting works correctly
- ✅ **README.md is comprehensive** (minimum 250 lines)
- ✅ **README clearly distinguishes** from prompt_generator
- ✅ **README includes architecture diagram** (CLI → lib → compiler → spec)

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

- **CRITICAL**: Study spec layers 0-5 BEFORE writing documentation (15+ minutes)
- Start with Phase 1 (extraction) before Phase 2 (CLI)
- Run tests frequently to catch regressions early
- Use rich for all user-facing output (consistent UX)
- Follow existing code style (ruff, mypy strict)
- Keep CLI commands thin - delegate to libraries
- **Documentation quality > speed** - take time to write comprehensive READMEs
- Include real examples from actual spec files in documentation
- Explain WHY and HOW, not just WHAT

**Potential Pitfalls:**

- Import path changes - use global search/replace carefully
- Circular dependencies - ensure clean separation
- CLI output - avoid print() in library code, use logging
- Configuration - plan for future extensibility

---

**End of Implementation Plan**
