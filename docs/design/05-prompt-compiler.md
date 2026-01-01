# Prompt Compiler Specification

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

The prompt compiler assembles stage prompts from templates, components, and context. It manages token budgets and applies compression strategies to fit within model limits.

```
Templates + Components + Context + Constraints → Compiled Prompt
```

---

## Core Concept

Instead of prompts embedded in Python code, prompts are compiled from structured files:

```
/prompts
  /templates       # Stage-specific prompt templates
    dream.yaml
    brainstorm.yaml
    seed.yaml
    grow_spine.yaml
    grow_anchors.yaml
    grow_branch.yaml
    ...
  /components      # Reusable prompt fragments
    role_setup.yaml
    genre_guidance.yaml
    quality_criteria.yaml
    output_format.yaml
    ...
  /schemas         # Output structure specifications
    dream_output.yaml
    spine_output.yaml
    ...
```

---

## Template Format

Templates declare which components to include and how to assemble them:

```yaml
# prompts/templates/grow_spine.yaml

name: grow_spine
description: "Generate the story spine (core arc)"
stage: grow.spine

# Token budget for this stage
token_budget: 4000

# Components to include
components:
  # Critical: Always included, never compressed
  - name: role_setup
    priority: critical
    content: |
      You are a story architect designing the core emotional arc.
      You will create a linear spine that every player experiences
      in some form, regardless of their choices.

  # Context: Injected from prior artifacts
  - name: seed_context
    priority: critical
    source: artifacts.seed
    compression: full

  # Standard: Included up to budget, summarized if needed
  - name: genre_guidance
    priority: standard
    ref: components/genre_guidance
    compression: summary_if_needed

  - name: arc_patterns
    priority: standard
    ref: components/arc_patterns
    compression: summary_if_needed

  # Background: First to omit or compress
  - name: spine_examples
    priority: background
    ref: components/spine_examples
    compression: omit_first

# Output specification
output:
  format: yaml
  schema_ref: schemas/spine_output
  include_schema_in_prompt: true

# Temperature and generation settings
generation:
  temperature: 0.7
  max_tokens: 2000
```

---

## Component Format

Components are reusable prompt fragments:

```yaml
# prompts/components/genre_guidance.yaml

name: genre_guidance
description: "Genre-specific writing guidance"

# Content varies based on context
variants:
  - condition:
      context.dream.genre: mystery
    content: |
      For mystery stories, consider:
      - Plant clues early that pay off later
      - Red herrings should be fair (not random)
      - The solution must be deducible from available information
      - Tension builds through revelation, not just action

  - condition:
      context.dream.genre: fantasy
    content: |
      For fantasy stories, consider:
      - Magic systems should have consistent rules
      - Worldbuilding serves story, not vice versa
      - Character growth matters more than spectacle

  # Default fallback
  - condition: default
    content: |
      Focus on emotional truth and character development.
      Genre conventions are tools, not requirements.
```

---

## Priority Levels

| Priority | Behavior When Over Budget |
|----------|--------------------------|
| `critical` | Never compress, error if can't fit |
| `standard` | Summarize if needed |
| `background` | Omit first, then summarize remainder |

### Compression Order

When over budget:

1. Omit `background` components entirely
2. Summarize remaining `background` components
3. Summarize `standard` components
4. Error if still over (critical content doesn't fit)

---

## Compression Strategies

### Full

Include complete content. Default for `critical` priority.

```yaml
- name: seed_context
  compression: full
```

### Summary

LLM-generated overview of content.

```yaml
- name: prior_branches
  compression: summary
  summary_prompt: |
    Summarize these branches in 200 words, focusing on:
    - Key events and revelations
    - State changes
    - How they connect to anchors
```

### Skeleton

Structure only — IDs and relationships, no prose.

```yaml
- name: topology
  compression: skeleton
  skeleton_fields:
    - id
    - connects_to
    - grants
```

### Omit

Exclude entirely if budget requires.

```yaml
- name: examples
  compression: omit_first  # First to omit when over budget
```

---

## Sandwiching

Critical content is split between prompt start and end, leveraging LLM attention patterns:

```
┌─────────────────────────────────────┐
│ CRITICAL: Role and task definition  │  ← High attention
├─────────────────────────────────────┤
│ STANDARD: Context and constraints   │  ← Medium attention
├─────────────────────────────────────┤
│ BACKGROUND: Examples and guides     │  ← Lower attention
├─────────────────────────────────────┤
│ CRITICAL: Output format and rules   │  ← High attention
└─────────────────────────────────────┘
```

Template configuration:

```yaml
components:
  - name: role_setup
    priority: critical
    position: start

  - name: seed_context
    priority: standard
    position: middle

  - name: examples
    priority: background
    position: middle

  - name: output_format
    priority: critical
    position: end

  - name: final_reminders
    priority: critical
    position: end
```

---

## Context Injection

Prior artifacts are injected as context:

```yaml
components:
  - name: seed_context
    source: artifacts.seed
    compression: full
    transform: |
      # Optional: Transform artifact before injection
      Extract protagonist, setting, and central_tension.
      Omit selected_hooks (not needed for spine).
```

### Context Sources

| Source | Description |
|--------|-------------|
| `artifacts.<stage>` | Output from prior stage |
| `context.<key>` | Runtime context values |
| `project.config` | Project configuration |

---

## Token Budget Management

```python
@dataclass
class CompiledPrompt:
    system: str              # System message content
    user: str                # User message content
    token_count: int         # Estimated total tokens
    included_components: list[str]
    compression_applied: list[str]  # What was compressed/omitted
    budget_remaining: int

class PromptCompiler:
    def __init__(
        self,
        prompts_path: Path,
        token_budget: int = 4000,
        tokenizer: str = "cl100k_base"
    ):
        ...

    def compile(
        self,
        template_name: str,
        context: dict[str, Any],
        priority_overrides: dict[str, Priority] | None = None
    ) -> CompiledPrompt:
        """Compile a prompt from template with context."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for budget tracking."""
        ...
```

### Token Estimation

Options (trade-offs):

| Method | Accuracy | Speed | Dependencies |
|--------|----------|-------|--------------|
| `tiktoken` | High | Medium | tiktoken library |
| Character heuristic | Low | Fast | None |
| Model-specific | Highest | Slow | API call |

Recommendation: Use `tiktoken` with fallback to heuristic (4 chars ≈ 1 token).

---

## Context Windowing by Stage

Each stage receives specific context:

| Stage | Context Includes |
|-------|------------------|
| DREAM | User prompt only |
| BRAINSTORM | DREAM output |
| SEED | BRAINSTORM output (full) |
| GROW.Spine | SEED only |
| GROW.Anchors | SEED + Spine |
| GROW.Fractures | SEED + Spine + Anchors |
| GROW.Branches | Compressed spine + full anchors/fractures + prior branches |
| FILL | Brief + relevant branch context |

Configuration in template:

```yaml
# prompts/templates/grow_branch.yaml

context_sources:
  - source: artifacts.seed
    compression: skeleton
    fields: [protagonist.name, setting.key_locations]

  - source: artifacts.grow.spine
    compression: summary
    summary_length: 200

  - source: artifacts.grow.anchors
    compression: full

  - source: artifacts.grow.fractures
    compression: full

  - source: artifacts.grow.branches.*
    compression: summary
    summary_length: 150_per_branch
```

---

## Output Schema Integration

Include output schema in prompt to guide LLM:

```yaml
# prompts/templates/grow_spine.yaml

output:
  format: yaml
  schema_ref: schemas/spine_output
  include_schema_in_prompt: true
  schema_position: end
```

Generated prompt includes:

```
Your output must be valid YAML matching this structure:

```yaml
type: grow_spine
version: 1

arc_shape: <string: rise | fall | rise-fall | fall-rise | ...>

beats:
  - id: <string: unique identifier>
    beat_type: <string: opening | inciting_incident | turning_point | ...>
    description: <string: what happens>
    emotional_state: <string: character's emotional state>
```
```

---

## Caching

Cache compiled prompts for identical inputs:

```python
class PromptCompiler:
    def __init__(self, ..., cache_enabled: bool = True):
        self._cache: dict[str, CompiledPrompt] = {}

    def _cache_key(self, template_name: str, context: dict) -> str:
        """Generate cache key from template + context hash."""
        context_hash = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"{template_name}:{context_hash}"
```

Cache invalidation:
- Template file changes
- Component file changes
- Context changes

---

## Error Handling

### Budget Overflow

If critical content exceeds budget:

```python
class BudgetExceededError(Exception):
    """Critical content doesn't fit in token budget."""
    def __init__(self, required: int, budget: int, components: list[str]):
        self.required = required
        self.budget = budget
        self.components = components
```

Resolution options:
1. Increase budget (if model supports)
2. Mark some critical content as standard
3. Split into multiple calls

### Missing Components

```python
class ComponentNotFoundError(Exception):
    """Referenced component doesn't exist."""
    pass
```

### Invalid Context

```python
class ContextResolutionError(Exception):
    """Context source couldn't be resolved."""
    pass
```

---

## Example: Complete Template

```yaml
# prompts/templates/fill_scene.yaml

name: fill_scene
description: "Generate prose for a scene from its brief"
stage: fill

token_budget: 3000

components:
  # Role and task
  - name: role_setup
    priority: critical
    position: start
    content: |
      You are a prose writer creating a scene for an interactive story.
      You will write evocative, voice-consistent prose that brings
      the scene brief to life while honoring all constraints.

  # The brief being filled
  - name: scene_brief
    priority: critical
    position: start
    source: context.current_brief
    compression: full

  # Voice guidance from seed
  - name: voice_notes
    priority: standard
    source: artifacts.seed.protagonist.voice_notes
    compression: full

  # Style guidance
  - name: style_guide
    priority: standard
    ref: components/prose_style
    compression: summary_if_needed

  # Prior scenes for continuity
  - name: prior_scenes
    priority: background
    source: context.prior_scenes
    compression: skeleton
    skeleton_fields: [passage_id, emotional_beat, key_events]

  # Output format
  - name: output_format
    priority: critical
    position: end
    content: |
      Output valid YAML with:
      - `prose`: The scene text (aim for {brief.prose_guidance.length})
      - `choices`: Array matching brief's choice specifications

      The prose should:
      - Match the {brief.prose_guidance.pov} POV
      - Use {brief.prose_guidance.tense} tense
      - Include sensory details from: {brief.prose_guidance.sensory_focus}

output:
  format: yaml
  schema_ref: schemas/scene_output
  include_schema_in_prompt: true

generation:
  temperature: 0.8
  max_tokens: 1500
```

---

## Directory Structure

```
/prompts
├── templates/
│   ├── dream.yaml
│   ├── brainstorm.yaml
│   ├── seed.yaml
│   ├── grow_spine.yaml
│   ├── grow_anchors.yaml
│   ├── grow_fractures.yaml
│   ├── grow_branch.yaml
│   ├── grow_brief.yaml
│   ├── fill_scene.yaml
│   └── validate_topology.yaml
├── components/
│   ├── role_setup/
│   │   ├── architect.yaml
│   │   ├── writer.yaml
│   │   └── validator.yaml
│   ├── genre_guidance.yaml
│   ├── arc_patterns.yaml
│   ├── prose_style.yaml
│   ├── quality_criteria.yaml
│   └── output_formats/
│       ├── yaml_guidance.yaml
│       └── structured_output.yaml
└── schemas/
    ├── dream_output.yaml
    ├── brainstorm_output.yaml
    ├── seed_output.yaml
    ├── spine_output.yaml
    ├── anchors_output.yaml
    ├── branch_output.yaml
    ├── brief_output.yaml
    └── scene_output.yaml
```

---

## See Also

- [01-pipeline-architecture.md](./01-pipeline-architecture.md) — How prompts are used
- [02-artifact-schemas.md](./02-artifact-schemas.md) — Output schemas
- [10-semantic-conventions.md](./10-semantic-conventions.md) — Naming in prompts
