# Installation

## Requirements

- Python 3.11 or higher
- pip or uv package manager

## Install from PyPI

```bash
pip install questfoundry-py
```

Or using uv:

```bash
uv add questfoundry-py
```

## Install with Provider Support

QuestFoundry supports multiple LLM providers. Install with provider-specific dependencies:

### OpenAI

```bash
pip install questfoundry-py[openai]
```

### Google Gemini

```bash
pip install questfoundry-py[google]
```

### Ollama

```bash
pip install questfoundry-py[ollama]
```

### All Providers

```bash
pip install questfoundry-py[all-providers]
```

## Development Installation

To contribute to QuestFoundry, clone the repository and install in development mode:

```bash
# Clone the repository
git clone https://github.com/pvliesdonk/questfoundry.git
cd questfoundry/lib/python

# Install with uv (recommended)
uv sync

# Bundle resources
uv run hatch run bundle

# Run tests
uv run pytest
```

## Verify Installation

Verify your installation by importing the library:

```python
import questfoundry

# List available schemas
schemas = questfoundry.list_schemas()
print(f"Found {len(schemas)} schemas")

# List available prompts
prompts = questfoundry.list_prompts()
print(f"Found {len(prompts)} role prompts")
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Learn the basics
- [Resource Bundling](bundling.md) - Understand how resources are bundled
- [User Guide](../guide/overview.md) - Detailed usage documentation
