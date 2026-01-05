# Interactive Stage Architecture

This document describes the architecture for interactive stage execution in QuestFoundry v5, enabling conversational refinement before structured output.

## Overview

Interactive mode enables multi-turn LLM conversations during stage execution. Instead of a single LLM call producing the final artifact, the LLM engages with the user to discuss and refine the creative vision before producing structured output.

### Key Design Decisions

1. **Unified 3-Phase Pattern**: Both interactive and direct modes use the same code path (Discuss → Summarize → Serialize)
2. **Tool Gating Per Phase**: Different tools available at each phase
3. **Explicit Phase Transitions**: User `/done` or LLM `ready_to_summarize()` tool
4. **Sandwich Pattern**: Critical instructions appear at prompt start AND end
5. **TTY Auto-Detection**: Mode defaults based on terminal interactivity
6. **Validation-Retry Loop**: Failed validation triggers compact feedback for retry
7. **No YAML Fallback**: Tool calling is required; explicit failure if provider skips tool

## The 3-Phase Pattern

Both interactive and direct modes use the same 3-phase pattern, differing only in configuration:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           3-Phase Pattern                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐        ┌───────────┐        ┌───────────┐                 │
│   │ DISCUSS │   →    │ SUMMARIZE │   →    │ SERIALIZE │                 │
│   └────┬────┘        └─────┬─────┘        └─────┬─────┘                 │
│        │                   │                    │                        │
│   Research tools      No tools            Finalization tool              │
│   User input          Generate summary    tool_choice=required           │
│   ready_to_summarize  Auto-proceed        Validation retry               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Phase Details

| Phase | Tools Available | Exit Condition | Purpose |
|-------|-----------------|----------------|---------|
| **Discuss** | Research + `ready_to_summarize` | User `/done`, LLM calls `ready_to_summarize()`, or max turns | Gather information, discuss with user |
| **Summarize** | None | Auto-proceed after summary | Generate compact summary of discussion |
| **Serialize** | Finalization tool only | Valid artifact | Convert summary to structured output |

### Mode Configuration

| Mode | max_discuss_turns | user_input_fn | Behavior |
|------|-------------------|---------------|----------|
| **Direct** | 1 | None | Single discuss turn, then auto-proceed |
| **Interactive** | 10 (default) | Provided | Multi-turn discussion with user |

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

The `ConversationRunner` manages the 3-phase conversation:

```
┌─────────────────────────────────────────────────────────────┐
│                    ConversationRunner                        │
├─────────────────────────────────────────────────────────────┤
│ provider: LLMProvider                                        │
│ research_tools: list[Tool]        (Discuss phase only)      │
│ finalization_tool: Tool           (Serialize phase only)    │
│ max_discuss_turns: int            (1 for direct, 10 for     │
│                                    interactive)             │
│ validation_retries: int           (default: 3)              │
├─────────────────────────────────────────────────────────────┤
│ run(initial_messages, user_input_fn, validator,             │
│     summary_prompt, on_assistant_message)                   │
│   → (artifact_data, ConversationState)                      │
└─────────────────────────────────────────────────────────────┘
```

**3-Phase Conversation Flow:**

```
                    ┌─────────────────────────────────────────────┐
                    │              DISCUSS PHASE                   │
                    │                                              │
                    │   Tools: research_tools + ready_to_summarize │
                    │                                              │
┌─────────────┐     │     ┌───────────────┐     ┌──────────────┐  │
│   System    │ →   │     │     User      │ ↔   │     LLM      │  │
│   Prompt    │     │     │   (optional)  │     │   Response   │  │
└─────────────┘     │     └───────────────┘     └──────┬───────┘  │
                    │                                   │          │
                    │         ┌─────────────────────────┘          │
                    │         │                                    │
                    │         ▼                                    │
                    │  ┌─────────────────────┐                     │
                    │  │ Check Exit Condition │                    │
                    │  └──────────┬──────────┘                     │
                    │             │                                │
                    │   ┌─────────┼─────────┬──────────────┐      │
                    │   │         │         │              │      │
                    │ /done  ready_to_    max       research      │
                    │         summarize  turns       tool         │
                    │   │         │         │              │      │
                    │   ▼         ▼         ▼              ▼      │
                    │  EXIT     EXIT      EXIT        EXECUTE     │
                    │                                 & CONTINUE  │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │             SUMMARIZE PHASE                  │
                    │                                              │
                    │   Tools: None                                │
                    │                                              │
                    │   ┌──────────────────────────────────────┐  │
                    │   │ Generate summary from discussion      │  │
                    │   │ (No user interaction, no tools)       │  │
                    │   └──────────────────────────────────────┘  │
                    │                                              │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │             SERIALIZE PHASE                  │
                    │                                              │
                    │   Tools: finalization_tool only             │
                    │   tool_choice: required                      │
                    │                                              │
                    │   ┌──────────────────────────────────────┐  │
                    │   │ Convert summary to structured output  │  │
                    │   │ using finalization tool               │  │
                    │   └──────────────────────┬───────────────┘  │
                    │                          │                   │
                    │               ┌──────────┴──────────┐       │
                    │               │                     │       │
                    │             valid              invalid      │
                    │               │                     │       │
                    │               ▼                     ▼       │
                    │            RETURN             RETRY         │
                    │                              (max 3)        │
                    └─────────────────────────────────────────────┘
```

### Phase Transition Signals

**User Signal**: `/done` typed as user input transitions from Discuss to Summarize.

**LLM Signal**: `ready_to_summarize()` tool call transitions from Discuss to Summarize.

Both signals are equivalent and can be used interchangeably.

### Validation Feedback Format (ADR-007)

When validation fails in the Serialize phase, structured feedback is sent to the LLM for retry:

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

### Tools (`src/questfoundry/tools/finalization.py`)

**Signal Tool:**

| Tool | Purpose |
|------|---------|
| `ready_to_summarize` | LLM signals discussion is complete |

**Finalization Tools:**

| Stage | Tool Name | Purpose |
|-------|-----------|---------|
| DREAM | `submit_dream` | Capture creative vision |
| BRAINSTORM | `submit_brainstorm` | Capture raw material |
| (future) | `submit_seed` | Capture core elements |

The tool schema matches the artifact schema, ensuring structured output.

## Execution Modes

### Interactive Mode (Default with TTY)

1. LLM receives system prompt with discussion instructions
2. **Discuss Phase**: Conversation loop with research tools available
   - LLM ↔ User (via `user_input_fn`)
   - Research tools can be called
   - Exit on user `/done`, LLM `ready_to_summarize()`, or max turns
3. **Summarize Phase**: LLM generates summary (no tools, no user input)
4. **Serialize Phase**: LLM calls finalization tool
   - `tool_choice="required"` forces tool call
   - Validation with retry on failure
5. Artifact written

### Direct Mode (Non-TTY or `--no-interactive`)

1. LLM receives system prompt with direct generation instructions
2. **Discuss Phase**: Single LLM turn (max_discuss_turns=1)
   - No user input
   - Research tools still available
   - Auto-exit after 1 turn
3. **Summarize Phase**: LLM generates summary (no tools)
4. **Serialize Phase**: LLM calls finalization tool
   - `tool_choice="required"` forces tool call
   - Validation with error on failure
5. Artifact written

**Note**: There is no YAML fallback. If the provider fails to call the finalization tool, a `ConversationError` is raised.

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
- `mode_instructions`: Engage in conversation, ask clarifying questions, mention `ready_to_summarize()` tool
- `mode_reminder`: Discuss the vision with the user first, call `ready_to_summarize()` when ready

**Direct Mode:**
- `mode_instructions`: Generate directly, consider key elements
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

Research tools can be injected into the conversation:

```python
context = {
    "user_prompt": "...",
    "research_tools": [SearchCorpusTool(), WebSearchTool()],
}
```

These are available during the Discuss phase only.

### Adding New Stages

1. Create finalization tool in `tools/finalization.py`
2. Register in `FINALIZATION_TOOLS` dict
3. Create stage implementation using `ConversationRunner`
4. Create prompt template with mode variables

## Testing

- **Unit tests**: Mock provider, verify phase transitions and tool gating
- **Integration tests**: Full stage execution with both modes
- **E2E tests**: Real LLM calls (optional, marked slow)

See `tests/unit/test_conversation_runner.py` and `tests/unit/test_dream_stage.py`.
