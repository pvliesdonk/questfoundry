# Quick Start

This guide will help you get started with the QuestFoundry Python library.

## Loading Schemas

QuestFoundry provides JSON schemas for validating artifacts:

```python
from questfoundry.utils.resources import get_schema, list_schemas

# List all available schemas
schemas = list_schemas()
print(f"Available schemas: {schemas}")

# Load a specific schema
hook_card_schema = get_schema("hook_card")
print(hook_card_schema)
```

## Loading Prompts

Access AI agent prompts for different roles:

```python
from questfoundry.utils.resources import get_prompt, list_prompts

# List all available role prompts
prompts = list_prompts()
print(f"Available roles: {prompts}")

# Load a prompt for a specific role
plotwright_prompt = get_prompt("plotwright")
print(plotwright_prompt)
```

## Validating Artifacts

Validate artifacts against their schemas:

```python
from questfoundry.validators.validation import validate_artifact

# Example artifact
artifact = {
    "id": "hook-001",
    "artifact_type": "hook_card",
    "version": "1.0",
    "narrative_hook": {
        "setup": "A mysterious stranger arrives in town.",
        "payoff_promise": "Discover the stranger's true identity."
    }
}

# Validate against schema
try:
    validate_artifact(artifact, "hook_card")
    print("✓ Artifact is valid!")
except Exception as e:
    print(f"✗ Validation error: {e}")
```

## Working with Providers

QuestFoundry supports multiple LLM providers:

```python
from questfoundry.providers.text.bedrock import BedrockProvider

# Initialize provider
provider = BedrockProvider(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1"
)

# Generate text
response = provider.generate(
    prompt="Write a short fantasy story opening.",
    system_prompt="You are a creative storyteller."
)

print(response)
```

## Next Steps

- [Resource Bundling](bundling.md) - Learn how resources are bundled
- [User Guide](../guide/overview.md) - Explore detailed features
- [API Reference](../api/utils.md) - Complete API documentation
