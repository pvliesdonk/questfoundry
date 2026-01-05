# Interactive Stage Architecture

This document describes the architecture for interactive stage execution in QuestFoundry v5, enabling conversational refinement before structured output.

## Overview

Interactive mode enables multi-turn LLM conversations during stage execution. Instead of a single LLM call producing the final artifact, the LLM engages with the user to discuss and refine the creative vision before calling a finalization tool.

### Key Design Decisions

1. **Single Codepath**: Interactive mode is a flag, not separate implementation
2. **Tool-Gated Finalization**: LLM signals completion by calling a stage-specific tool
3. **Sandwich Pattern**: Critical instructions appear at prompt start AND end
4. **TTY Auto-Detection**: Mode defaults based on terminal interactivity
5. **Validation-Retry Loop**: Failed validation triggers compact feedback for retry

## Architecture Components

### Tool Protocol (`src/questfoundry/tools/base.py`)

```
┌────────────────────┐
│  ToolDefinition    │
│  - name: str       │
│  - description     │
│  - parameters      │  (JSON Schema)
└────────────────────┘

┌────────────────────┐
│  ToolCall          │
│  - id: str         │
│  - name: str       │
│  - arguments: dict │
└────────────────────┘

┌────────────────────┐
│  Tool (Protocol)   │
│  + definition      │
│  + execute()       │
└────────────────────┘
```

### LLMProvider Extension (`src/questfoundry/providers/base.py`)

The provider interface was extended with tool support:

```python
async def complete(
    messages: list[Message],
    tools: list[ToolDefinition] | None = None,
    tool_choice: str | None = None,  # "auto", "required", or tool name
) -> LLMResponse
```

The `LLMResponse` now includes optional `tool_calls`:

```python
@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    tool_calls: list[ToolCall] | None = None
```

### ConversationRunner (`src/questfoundry/conversation/runner.py`)

The `ConversationRunner` manages the multi-turn conversation loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    ConversationRunner                        │
├─────────────────────────────────────────────────────────────┤
│ provider: LLMProvider                                        │
│ tools: list[Tool]                                           │
│ finalization_tool: str          (e.g., "submit_dream")      │
│ max_turns: int                  (default: 10)               │
│ validation_retries: int         (default: 3)                │
├─────────────────────────────────────────────────────────────┤
│ run(initial_messages, user_input_fn, validator)             │
│   → (artifact_data, ConversationState)                      │
└─────────────────────────────────────────────────────────────┘
```

**Conversation Flow:**

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│   System    │ →   │     User      │ →   │     LLM      │
│   Prompt    │     │   Message     │     │   Response   │
└─────────────┘     └───────────────┘     └──────┬───────┘
                                                  │
                    ┌─────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Tool Call Present?   │
        └───────────┬───────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    finalization           research
         │                     │
         ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│   Validate      │  │  Execute Tool   │
│   Artifact      │  │  Add Result to  │
└────────┬────────┘  │  Messages       │
         │           └─────────────────┘
         │                     │
    ┌────┴────┐               │
    │         │               │
  valid    invalid            │
    │         │               │
    ▼         ▼               ▼
 RETURN    RETRY        CONTINUE
           (max 3)      CONVERSATION
```

### Validation Feedback Format (ADR-007)

When validation fails, structured feedback is sent to the LLM for retry:

```json
{
  "result": "validation_failed",

  "issues": {
    "invalid": [
      {
        "field": "audience",
        "provided": "",
        "problem": "empty string not allowed",
        "requirement": "non-empty string, e.g. 'adult', 'young adult'"
      }
    ],
    "missing": [
      {
        "field": "scope.target_word_count",
        "requirement": "integer >= 1000"
      }
    ],
    "unknown": ["passages", "word_count"]
  },

  "issue_count": 4,

  "action": "Call submit_dream() with corrected data. Unknown fields may be typos."
}
```

**Field ordering** is based on prompt engineering principles (primacy/recency effects):
1. `result` first (immediate orientation)
2. `issues` in middle (diagnosis)
3. `action` last (LLM weights final content for instructions)

**Key design choices:**
- `result` uses semantic enum (`accepted`, `validation_failed`, `tool_error`)
- No full schema included (already in tool definition)
- Each field error includes specific `requirement` text
- `unknown` fields help detect wrong field names without fuzzy matching

### Finalization Tools (`src/questfoundry/tools/finalization.py`)

Each stage has a dedicated finalization tool:

| Stage | Tool Name | Purpose |
|-------|-----------|---------|
| DREAM | `submit_dream` | Capture creative vision |
| BRAINSTORM | `submit_brainstorm` | Capture raw material |
| (future) | `submit_seed` | Capture core elements |

The tool schema matches the artifact schema, ensuring structured output.

## Execution Modes

### Interactive Mode (Default with TTY)

1. LLM receives system prompt with discussion instructions
2. Conversation loop: LLM ↔ User (via `user_input_fn`)
3. LLM calls finalization tool when ready
4. Validation with retry on failure
5. Artifact written

### Direct Mode (Non-TTY or `--no-interactive`)

1. LLM receives system prompt with direct generation instructions
2. Same conversation loop as interactive, but `max_turns=1` and no user input
3. Research tools available, finalization tool for completion
4. Validation with retry on failure
5. Artifact written

Both modes use the same `ConversationRunner` with identical 3-phase flow (discuss with research tools → summarize → serialize with finalization tool). The only differences are turn limits and whether user input is collected.

## Prompt Architecture

### Sandwich Pattern

Critical instructions appear at both the start and end of the system prompt:

```yaml
system: |
  You are a creative director for interactive fiction.

  {{ mode_instructions }}        # START: Mode-specific behavior

  The DREAM artifact captures:
  - Genre and subgenre
  - Tone and atmosphere
  ...

  {{ mode_reminder }}            # END: Reinforce key behavior
```

### Mode-Specific Content

The stage provides mode-specific content via template variables:

**Interactive Mode:**
- `mode_instructions`: Engage in conversation, ask clarifying questions
- `mode_reminder`: Only call submit_dream when vision is refined

**Direct Mode:**
- `mode_instructions`: Generate directly, call submit_dream
- `mode_reminder`: (empty)

## Usage

### CLI

```bash
# Auto-detect (interactive if TTY)
qf dream "A noir mystery"

# Force interactive
qf dream -i "A noir mystery"

# Force direct (piped input, scripting)
qf dream -I "A noir mystery"
echo "A noir mystery" | qf dream -I
```

### Programmatic

```python
from questfoundry.pipeline import PipelineOrchestrator

orchestrator = PipelineOrchestrator(project_path)

# Interactive mode (default)
result = await orchestrator.run_stage("dream", {
    "user_prompt": "A noir mystery",
    # "interactive" not set = TTY auto-detect
})

# Force direct mode
result = await orchestrator.run_stage("dream", {
    "user_prompt": "A noir mystery",
    "interactive": False,
})
```

## Extension Points

### Adding Research Tools

Research tools are loaded by the orchestrator and available during discussion:

```python
context = {
    "user_prompt": "...",
    "research_tools": [SearchCorpusTool(), WebSearchTool()],
}
```

Research tools and finalization tools are **not** available simultaneously. The 3-phase pattern uses different tool sets per phase:
- **Discuss phase**: Research tools only
- **Summarize phase**: No tools
- **Serialize phase**: Finalization tool only

### Adding New Stages

1. Create finalization tool in `tools/finalization.py`
2. Register in `FINALIZATION_TOOLS` dict
3. Create stage implementation using `ConversationRunner`
4. Create prompt template with mode variables

## Testing

- **Unit tests**: Mock provider, verify tool call handling
- **Integration tests**: Full stage execution with both modes
- **E2E tests**: Real LLM calls (optional, marked slow)

See `tests/unit/test_conversation_runner.py` and `tests/unit/test_dream_stage.py`.
