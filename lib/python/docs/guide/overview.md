# User Guide

## Overview

QuestFoundry is designed around a **studio model** for collaborative interactive fiction authoring. The library implements this model with:

- **15 Specialized Roles**: Each role has specific responsibilities and capabilities
- **12 Production Loops**: Structured workflows for narrative development
- **30+ Artifact Types**: Schema-validated data structures for content
- **AI Agent Integration**: Pre-configured prompts for AI-powered workflows

## Core Concepts

### Artifacts

Artifacts are the primary data structures in QuestFoundry. Each artifact:

- Has a specific type (e.g., `hook_card`, `codex_entry`, `art_plan`)
- Follows a JSON schema for validation
- Contains structured narrative or production data
- Can be versioned and tracked

### Roles

QuestFoundry defines 15 specialized roles:

| Role | Responsibility |
|------|---------------|
| **Showrunner** | Orchestrates all production activities |
| **Plotwright** | Designs narrative structure and plot |
| **Scene Smith** | Writes individual scenes and dialogue |
| **Lore Weaver** | Maintains world consistency and canon |
| **Codex Curator** | Creates in-world encyclopedic content |
| **Style Lead** | Defines and enforces narrative voice |
| **Gatekeeper** | Quality assurance and canon validation |
| **Player Narrator** | Generates player-facing narrative |
| **Art Director** | Plans visual assets and art direction |
| **Illustrator** | Creates specific visual assets |
| **Audio Director** | Plans audio design and music |
| **Audio Producer** | Produces audio assets |
| **Book Binder** | Assembles final deliverables |
| **Translator** | Localizes content to other languages |
| **Researcher** | Gathers reference material |

### Loops

Loops are structured workflows that coordinate multiple roles. Examples include:

- **Hook Development Loop**: Create compelling story hooks
- **Scene Production Loop**: Write and refine scenes
- **Canon Integration Loop**: Validate and merge content into canon
- **Asset Production Loop**: Create art and audio assets

### Hot vs Cold

QuestFoundry distinguishes between:

- **Hot Content**: Working drafts, internal notes, spoilers (private)
- **Cold Content**: Player-safe, canon-validated, ready for export (public)

This separation ensures spoiler hygiene and content safety.

## Common Tasks

### Load and Validate a Schema

```python
from questfoundry.utils.resources import get_schema
from questfoundry.validators.validation import validate_artifact

schema = get_schema("hook_card")
artifact = {...}  # Your artifact data
validate_artifact(artifact, "hook_card")
```

### Use a Role Prompt

```python
from questfoundry.utils.resources import get_prompt

plotwright_prompt = get_prompt("plotwright")
# Use with your LLM provider
```

### List Available Resources

```python
from questfoundry.utils.resources import list_schemas, list_prompts

print(f"Schemas: {list_schemas()}")
print(f"Roles: {list_prompts()}")
```
