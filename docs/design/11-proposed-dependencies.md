# Proposed Dependencies

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Proposed — Not mandated

---

## Overview

This document lists proposed dependencies for a v5 implementation. These are **suggestions**, not requirements. Alternative libraries may be substituted.

---

## Core Runtime

### Python

| Dependency | Purpose | Version | Status |
|------------|---------|---------|--------|
| Python | Runtime | >= 3.11 | **Required** |

Python 3.11+ required for:
- Performance improvements
- Better error messages
- `tomllib` built-in

### Package Management

| Tool | Purpose | Status |
|------|---------|--------|
| `uv` | Fast package manager | Recommended |
| `pip` | Standard fallback | Alternative |
| `poetry` | Dependency management | Alternative |

---

## YAML/JSON Processing

| Dependency | Purpose | Status |
|------------|---------|--------|
| `ruamel.yaml` | YAML parsing with comment preservation | Recommended |
| `PyYAML` | Simpler YAML parsing | Alternative |
| `jsonschema` | JSON Schema validation | Recommended |

### Why ruamel.yaml?

Human-edited YAML files should preserve:
- Comments
- Formatting
- Key order

`ruamel.yaml` preserves these; `PyYAML` does not.

```python
# Example
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

with open('artifact.yaml') as f:
    data = yaml.load(f)

# Modify
data['version'] = 2

# Write back (preserves comments)
with open('artifact.yaml', 'w') as f:
    yaml.dump(data, f)
```

---

## LLM Integration

| Dependency | Purpose | Status |
|------------|---------|--------|
| `litellm` | Unified LLM provider interface | Recommended |
| `openai` | OpenAI API client | Alternative (direct) |
| `anthropic` | Anthropic API client | Alternative (direct) |
| `ollama` | Ollama Python client | Alternative (local) |

### Why litellm?

Single interface for multiple providers:

```python
from litellm import completion

# Works with any provider
response = completion(
    model="ollama/qwen3:8b",  # or "gpt-4o" or "claude-3-5-sonnet"
    messages=[{"role": "user", "content": "..."}]
)
```

### Alternative: Direct Clients

If litellm is too heavy, use provider-specific clients:

```python
# Ollama
from ollama import Client

# OpenAI
from openai import OpenAI

# Anthropic
from anthropic import Anthropic
```

---

## Token Estimation

| Dependency | Purpose | Status |
|------------|---------|--------|
| `tiktoken` | OpenAI tokenizer | Recommended |
| Heuristic | 4 chars ≈ 1 token | Fallback |

### Usage

```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback for unknown models
        return len(text) // 4
```

---

## CLI Framework

| Dependency | Purpose | Status |
|------------|---------|--------|
| `click` | CLI framework (mature) | Recommended |
| `typer` | CLI with type hints | Alternative |
| `argparse` | Standard library | Fallback |

### Why Click?

- Battle-tested
- Good documentation
- Composable commands

```python
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--to', help='Run up to stage')
def run(to: str):
    """Run pipeline stages."""
    pass

@cli.command()
@click.argument('stage')
def review(stage: str):
    """Review stage artifact."""
    pass
```

---

## Data Validation

| Dependency | Purpose | Status |
|------------|---------|--------|
| `pydantic` | Data validation with type hints | Recommended |
| `attrs` | Simpler data classes | Alternative |
| `dataclasses` | Standard library | Fallback |

### Why Pydantic?

- JSON Schema generation
- Validation at parse time
- Good error messages

```python
from pydantic import BaseModel

class DreamArtifact(BaseModel):
    type: Literal["dream"]
    version: int
    genre: str
    tone: list[str]
    themes: list[str]

# Validates on construction
dream = DreamArtifact(**yaml_data)
```

---

## Output Formatting

| Dependency | Purpose | Status |
|------------|---------|--------|
| `rich` | Terminal formatting | Recommended |
| Plain print | No dependency | Fallback |

### Rich Features

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Status display
console.print("[green]✓[/green] DREAM completed")

# Tables
table = Table(title="Pipeline Status")
table.add_column("Stage")
table.add_column("Status")
```

---

## Testing

| Dependency | Purpose | Status |
|------------|---------|--------|
| `pytest` | Test framework | Recommended |
| `pytest-asyncio` | Async test support | If using async |
| `pytest-cov` | Coverage reporting | Recommended |

### Test Structure

```
tests/
├── unit/
│   ├── test_prompt_compiler.py
│   ├── test_artifact_validation.py
│   └── test_state_management.py
├── integration/
│   ├── test_stage_transitions.py
│   └── test_context_building.py
└── e2e/
    └── test_full_pipeline.py
```

---

## Optional Dependencies

### Graph Visualization

| Dependency | Purpose | Status |
|------------|---------|--------|
| `networkx` | Graph algorithms | Optional |
| `graphviz` | Graph visualization | Optional |

For topology visualization and reachability analysis.

### Observability

| Dependency | Purpose | Status |
|------------|---------|--------|
| `structlog` | Structured logging | Optional |
| `opentelemetry` | Tracing | Optional |

For production debugging.

---

## Dependency Summary

### Minimal Set

```toml
# pyproject.toml (minimal)
[project]
dependencies = [
    "pyyaml>=6.0",
    "jsonschema>=4.0",
    "click>=8.0",
]
```

### Recommended Set

```toml
# pyproject.toml (recommended)
[project]
dependencies = [
    "ruamel.yaml>=0.18",
    "jsonschema>=4.0",
    "litellm>=1.0",
    "tiktoken>=0.5",
    "click>=8.0",
    "pydantic>=2.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
]
viz = [
    "networkx>=3.0",
    "graphviz>=0.20",
]
```

---

## Version Constraints

| Constraint | Meaning |
|------------|---------|
| `>=3.11` | Minimum version |
| `>=1.0,<2.0` | Range (breaking change protection) |
| `>=1.0` | Any compatible version |

### Recommendation

Use minimum version constraints (`>=`) for most dependencies. Only add upper bounds (`<2.0`) if known breaking changes exist.

---

## Installation

### Development

```bash
# With uv (recommended)
uv pip install -e ".[dev]"

# With pip
pip install -e ".[dev]"
```

### Production

```bash
uv pip install questfoundry
```

---

## See Also

- [12-getting-started.md](./12-getting-started.md) — Setup instructions
- [13-project-structure.md](./13-project-structure.md) — Where code goes
