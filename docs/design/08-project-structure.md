# Project Structure

**Version**: 1.1.0
**Last Updated**: 2026-01-12
**Status**: Canonical

---

## Overview

This document defines the recommended directory structure for a QuestFoundry v5 implementation.

---

## Repository Structure

```
questfoundry/
├── src/
│   └── questfoundry/          # Main package
│       ├── __init__.py
│       ├── cli.py             # CLI entry point
│       ├── config.py          # Configuration handling
│       ├── pipeline/          # Pipeline orchestration
│       ├── stages/            # Stage implementations
│       ├── prompts/           # Prompt compilation
│       ├── artifacts/         # Artifact handling
│       ├── validation/        # Quality validation
│       ├── providers/         # LLM providers
│       └── export/            # Export formats
├── prompts/                   # Prompt templates (not in src/)
│   ├── templates/
│   ├── components/
│   └── schemas/
├── schemas/                   # JSON schemas for artifacts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/                      # Documentation
├── pyproject.toml
├── README.md
└── ARCHITECTURE.md
```

---

## Package Structure

### `src/questfoundry/`

The main Python package.

```
src/questfoundry/
├── __init__.py                # Version, public API
├── cli.py                     # Click command definitions
├── config.py                  # PipelineConfig, ProjectConfig
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py        # PipelineOrchestrator class
│   ├── gates.py               # Human gate handling
│   └── context.py             # Context building for stages
│
├── stages/
│   ├── __init__.py
│   ├── base.py                # Stage abstract base class
│   ├── dream.py
│   ├── brainstorm.py
│   ├── seed.py
│   ├── grow/
│   │   ├── __init__.py
│   │   ├── spine.py
│   │   ├── anchors.py
│   │   ├── fractures.py
│   │   ├── branches.py
│   │   ├── connections.py
│   │   └── briefs.py
│   ├── fill.py
│   └── ship.py
│
├── prompts/
│   ├── __init__.py
│   ├── compiler.py            # PromptCompiler class
│   ├── loader.py              # Template/component loading
│   └── compression.py         # Compression strategies
│
├── artifacts/
│   ├── __init__.py
│   ├── reader.py              # Read YAML artifacts
│   ├── writer.py              # Write YAML artifacts
│   └── validator.py           # JSON Schema validation
│
├── validation/
│   ├── __init__.py
│   ├── topology.py            # Graph reachability, orphans
│   ├── state.py               # State consistency
│   ├── quality_bars.py        # 8 quality bars
│   └── pre_gate.py            # Quick structural checks
│
├── providers/
│   ├── __init__.py
│   ├── base.py                # LLMProvider protocol
│   ├── ollama.py
│   ├── openai.py
│   └── anthropic.py
│
└── export/
    ├── __init__.py
    ├── twee.py                # Twee/Twine export
    ├── html.py                # Standalone HTML
    └── json.py                # Engine-agnostic JSON
```

---

## Prompts Structure

Prompts live **outside** the Python package for easy editing:

```
prompts/
├── templates/                 # Stage-specific templates
│   ├── dream.yaml
│   ├── brainstorm.yaml
│   ├── seed.yaml
│   ├── grow_spine.yaml
│   ├── grow_anchors.yaml
│   ├── grow_fractures.yaml
│   ├── grow_branch.yaml
│   ├── grow_brief.yaml
│   └── fill_scene.yaml
│
├── components/                # Reusable fragments
│   ├── role_setup/
│   │   ├── architect.yaml
│   │   ├── writer.yaml
│   │   └── validator.yaml
│   ├── genre_guidance.yaml
│   ├── arc_patterns.yaml
│   ├── prose_style.yaml
│   └── quality_criteria.yaml
│
└── schemas/                   # Output format specs
    ├── dream_output.yaml
    ├── spine_output.yaml
    └── ...
```

---

## Schemas Structure

JSON Schemas for artifact validation:

```
schemas/
├── dream.schema.json
├── brainstorm.schema.json
├── seed.schema.json
├── grow/
│   ├── spine.schema.json
│   ├── anchors.schema.json
│   ├── fractures.schema.json
│   ├── branch.schema.json
│   ├── connections.schema.json
│   └── brief.schema.json
├── scene.schema.json
└── manifest.schema.json
```

---

## Project Structure (User Projects)

When a user creates a story project:

```
my_story/
├── project.yaml              # Project configuration
├── artifacts/                # Generated artifacts
│   ├── dream.yaml
│   ├── brainstorm.yaml
│   ├── seed.yaml
│   ├── grow/
│   │   ├── spine.yaml
│   │   ├── anchors.yaml
│   │   ├── fractures.yaml
│   │   ├── branches/
│   │   │   ├── main_path.yaml
│   │   │   └── ...
│   │   ├── connections.yaml
│   │   └── briefs/
│   │       ├── opening_001.yaml
│   │       └── ...
│   └── fill/
│       └── scenes/
│           ├── opening_001.yaml
│           └── ...
├── exports/                  # Exported outputs
│   ├── manifest.yaml
│   ├── story.tw
│   ├── story.html
│   └── story.json
├── overrides/                # Human overrides
│   └── validation_overrides.yaml
└── .questfoundry/            # Internal state
    ├── pipeline_state.yaml
    └── validation_cache/
```

---

## Configuration Files

### `project.yaml`

Per-project configuration:

```yaml
# my_story/project.yaml
name: noir_mystery
version: 1

pipeline:
  gates:
    seed: required
    grow.anchors: required
    ship: required
    default: optional

  iteration:
    grow.harvest_rounds: 1
    grow.max_harvest_rounds: 3

providers:
  default: ollama/qwen3:8b
  stages:
    brainstorm: openai/gpt-4o
    fill: anthropic/claude-3-5-sonnet

quality:
  bars:
    accessibility:
      reading_level:
        target: 8
        range: [6, 10]
```

### Global Configuration

User-wide settings:

```yaml
# ~/.config/questfoundry/config.yaml
providers:
  ollama:
    host: http://localhost:11434
  openai:
    api_key: ${OPENAI_API_KEY}
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

defaults:
  provider: ollama/qwen3:8b
  token_budget: 4000
```

---

## Test Structure

```
tests/
├── conftest.py               # Fixtures
├── unit/
│   ├── test_prompt_compiler.py
│   ├── test_artifact_validation.py
│   ├── test_topology_validation.py
│   └── stages/
│       ├── test_dream.py
│       ├── test_seed.py
│       └── ...
├── integration/
│   ├── test_stage_transitions.py
│   ├── test_context_building.py
│   └── test_human_gates.py
├── e2e/
│   ├── test_dream_only.py
│   ├── test_dream_to_seed.py
│   ├── test_full_grow.py
│   └── test_full_pipeline.py
└── fixtures/
    ├── sample_dream.yaml
    ├── sample_seed.yaml
    └── ...
```

---

## Documentation Structure

```
docs/
├── index.md                  # Overview
├── installation.md           # Setup guide
├── quickstart.md             # First story
├── cli-reference.md          # Command reference
├── stages/
│   ├── dream.md
│   ├── brainstorm.md
│   └── ...
├── configuration.md          # Config options
├── providers.md              # LLM provider setup
├── export-formats.md         # Export options
└── development/
    ├── architecture.md
    ├── contributing.md
    └── testing.md
```

---

## Key Principles

### 1. Prompts Outside Package

Prompts in `prompts/` not `src/questfoundry/prompts/`:
- Editable without code changes
- Versionable separately
- Easy to audit

### 2. Schemas Separate

Schemas in `schemas/` not embedded:
- Shareable with other tools
- Clear contract definition
- Easy validation testing

### 3. User Projects Self-Contained

Everything in project directory:
- `my_story/artifacts/`
- `my_story/exports/`
- No hidden state elsewhere

### 4. Configuration Hierarchy

```
defaults (package) → user config (~/.config) → project config → CLI args
```

Each level overrides the previous.

---

## pyproject.toml

```toml
[project]
name = "questfoundry"
version = "0.1.0"
description = "Pipeline-driven interactive fiction generation"
requires-python = ">=3.11"

dependencies = [
    "ruamel.yaml>=0.18",
    "jsonschema>=4.0",
    "click>=8.0",
    "pydantic>=2.0",
    "rich>=13.0",
    "litellm>=1.0",
    "tiktoken>=0.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
]

[project.scripts]
qf = "questfoundry.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/questfoundry"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## See Also

- [00-spec.md](./00-spec.md) — Unified v5 specification
- [06-proposed-dependencies.md](./06-proposed-dependencies.md) — Dependencies
- [07-getting-started.md](./07-getting-started.md) — Implementation order
