# Quick Start

This guide will help you get QuestFoundry up and running.

## Installation

### Prerequisites

- Python 3.11 or later
- [uv](https://github.com/astral-sh/uv) package manager

### Install from Source

```bash
# Clone repository
git clone https://github.com/pvliesdonk/questfoundry.git
cd questfoundry

# Install with uv
uv sync

# Verify installation
uv run qf version
```

## Basic Usage

### Compile Domain to Code

QuestFoundry uses MyST files as the source of truth. Compile them to Python:

```bash
qf compile
```

This generates:

- Pydantic models from `ontology/`
- Role configurations from `roles/`
- LangGraph definitions from `loops/`

### Run a Workflow

Start an interactive story creation session:

```bash
qf ask "Create a mystery story set in a Victorian mansion"
```

The Showrunner will orchestrate the 8 roles to create your story.

### Configure LLM Provider

QuestFoundry supports multiple LLM providers. Configure in your project:

```bash
# Use Ollama (local)
qf config set provider ollama
qf config set model qwen3:8b

# Use OpenAI
qf config set provider openai
qf config set model gpt-4o
```

## Example Project

See `examples/mystery_manor/` for a complete example:

```bash
cd examples/mystery_manor

# Inspect the cold store
sqlite3 project.qfdb "SELECT anchor, title FROM sections;"

# Read a scene
sqlite3 project.qfdb "SELECT content FROM sections WHERE anchor='scene_1';"
```

## Next Steps

- Read the [Architecture](architecture.md) to understand the system design
- Explore the [8 Roles](roles/index.md) and their responsibilities
- Check the [API Reference](api/index.md) for programmatic usage
