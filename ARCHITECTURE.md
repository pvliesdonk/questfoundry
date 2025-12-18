# QuestFoundry V4 Architecture

This document describes the architecture of the QuestFoundry V4 runtime.

## Overview

QuestFoundry is a multi-agent system for collaborative fiction creation. The architecture separates **domain definitions** (what agents exist and what they do) from **runtime execution** (how agents communicate and operate).

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│                    (qf ask, qf projects)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      Runtime Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Agent     │  │  Messaging  │  │    Delegation       │ │
│  │  Execution  │◄─┤    Broker   │◄─┤  Tracker/Executor   │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘ │
│         │                                                   │
│  ┌──────▼──────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Tools     │  │   Storage   │  │     Providers       │ │
│  │  Registry   │  │   (Project) │  │  (Ollama, OpenAI)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      Domain Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Agents    │  │  Playbooks  │  │  Artifact Types     │ │
│  │  (JSON)     │  │   (JSON)    │  │      (JSON)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. Domain as Data, Not Code

Agent definitions, playbooks, and artifact types are JSON files in `domain-v4/`. The runtime loads and interprets them at startup. This enables:

- Easy modification without code changes
- Validation against schemas in `meta/`
- Clear separation of concerns

### 2. Hub-and-Spoke Delegation

The Showrunner is the central orchestrator. Other agents don't communicate directly — they receive delegations from the Showrunner and return results to it:

```
                    ┌─────────────┐
                    │ Showrunner  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Plotwright │    │ Scene Smith │    │ Gatekeeper  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 3. Domain-Aligned Loop Termination

Instead of arbitrary depth limits, loops terminate based on **playbook rework budgets**. Each playbook defines `max_rework_cycles`, and the runtime tracks visits to `is_rework_target` phases. When budget exhausts, escalation occurs — not silent failure.

### 4. Hot/Cold Store Semantics

- **Hot Store (workspace)**: Working drafts, mutable, all agents can write
- **Cold Store (canon)**: Committed content, append-only, Lorekeeper exclusive

Lifecycle states: `draft` → `review` → `approved` → `cold`

## Module Structure

### `runtime/agent/`

Agent execution and the Showrunner loop.

- **`runtime.py`**: `AgentRuntime` — main entry point for agent activation
- **`showrunner_loop.py`**: `ShowrunnerLoop` — orchestration with delegation processing
- **`context.py`**: Context building for agent prompts

### `runtime/delegation/`

Async delegation infrastructure.

- **`tracker.py`**: `PlaybookTracker` — tracks playbook instances and rework budgets
- **`executor.py`**: `AsyncDelegationExecutor` — executes delegations with lifecycle handling
- **`bouncer.py`**: `DelegationBouncer` — pre-flight checks (budgets, concurrent limits)
- **`nudger.py`**: `PlaybookNudger` — generates advisory nudges based on playbook expectations

### `runtime/messaging/`

Inter-agent communication.

- **`message.py`**: `Message` dataclass and factory functions
- **`mailbox.py`**: `AsyncMailbox` — per-agent priority queues
- **`broker.py`**: `AsyncMessageBroker` — central routing and persistence
- **`types.py`**: Enums for message types, priorities, and statuses

Message types:

- `DELEGATION_REQUEST` / `DELEGATION_RESPONSE` — work delegation
- `FEEDBACK` — quality assessment
- `ESCALATION` — budget exhaustion or blocking issues
- `NUDGE` — advisory guidance from runtime
- `DIGEST` — summarized older messages (Secretary pattern)
- `PROGRESS_UPDATE` — status updates (ephemeral, with TTL)

### `runtime/storage/`

Project and artifact persistence.

- **`project.py`**: `Project` — SQLite-backed project storage
- **`store_manager.py`**: `StoreManager` — store access control
- **`lifecycle.py`**: Lifecycle state transitions

### `runtime/tools/`

Tool implementations available to agents.

- **`registry.py`**: `ToolRegistry` — capability-based tool access
- **`base.py`**: `BaseTool` — abstract base for tool implementations
- **`save_artifact.py`**: Artifact CRUD operations

### `runtime/providers/`

LLM provider integrations.

- **`base.py`**: `LLMProvider` — abstract interface
- **`ollama.py`**: `OllamaProvider` — local Ollama integration
- **`openai.py`**: `OpenAIProvider` — OpenAI API integration

### `runtime/checkpoint/`

Session persistence for resume.

- **`manager.py`**: `CheckpointManager` — save/restore session state
- **`types.py`**: `Checkpoint` dataclass

### `runtime/domain/`

Domain loading and validation.

- **`loader.py`**: `load_studio()` — loads domain from JSON files
- **`models.py`**: Pydantic models for domain entities

## Data Flow

### Agent Activation

```
1. User input → CLI
2. CLI → AgentRuntime.activate(agent, input, session)
3. Runtime builds context (system prompt + conversation history)
4. Runtime calls LLM provider
5. LLM response may include tool calls
6. Tool calls executed via ToolRegistry
7. Results returned to agent
8. Agent may request delegation via delegate tool
```

### Delegation Flow

```
1. Agent calls delegate tool
2. Bouncer checks: budget ok? agent available?
3. Delegation request queued in target agent's mailbox
4. ShowrunnerLoop sees pending delegation
5. Executor activates target agent with delegation context
6. Target agent produces result
7. Delegation response sent to requesting agent's mailbox
8. Original agent receives response on next activation
```

### Rework Flow

```
1. Scene Smith produces draft
2. Gatekeeper reviews, finds issues
3. Gatekeeper sends feedback → Scene Smith mailbox
4. Showrunner orchestrates rework phase
5. PlaybookTracker records rework_target phase entry
6. If rework_count > max_rework_cycles:
   - Escalation message sent to Showrunner
   - Playbook marked ESCALATED
```

## Quality Assurance

### Quality Gates

Playbook phases can define `quality_checkpoint`:

```json
{
  "id": "prose_drafting",
  "quality_checkpoint": {
    "validator": "gatekeeper",
    "criteria": ["voice_consistency", "canon_compliance", "structure"]
  }
}
```

The Gatekeeper agent validates against these criteria. Failed checks trigger rework.

### Nudging

The `PlaybookNudger` observes agent behavior against playbook expectations:

- `missing_output` — expected artifact not produced
- `unexpected_state` — phase transition doesn't match playbook
- `quality_gate_reminder` — upcoming quality checkpoint
- `timeout_warning` — budget running low

Nudges are advisory (low priority, short TTL) — agents can ignore them.

## Configuration

### Environment Variables

```bash
OLLAMA_HOST=http://localhost:11434
OPENAI_API_KEY=sk-...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=questfoundry
```

### Domain Configuration

`domain-v4/studio.json` defines the studio configuration:

- Agent definitions (capabilities, tools, knowledge)
- Store definitions (semantics, exclusive writers)
- Playbook definitions (phases, quality checkpoints)
- Artifact type schemas

## Extension Points

### Adding Agents

1. Create `domain-v4/agents/new_agent.json`
2. Define archetypes, capabilities, tools, knowledge
3. Runtime automatically loads on next startup

### Adding Tools

1. Create tool implementation in `runtime/tools/`
2. Decorate with `@register_tool`
3. Define tool schema in `domain-v4/tools/`
4. Grant capability to agents in agent definitions

### Adding Playbooks

1. Create `domain-v4/playbooks/new_playbook.json`
2. Define phases with steps, outputs, quality checkpoints
3. Set `max_rework_cycles` for rework budget
4. Mark `is_rework_target: true` on revision phases

## Testing Strategy

- **Unit tests**: Individual components in isolation
- **Integration tests**: Component interactions
- **E2E tests**: Full workflows with real LLM calls (require long timeouts)

All tests in `tests/runtime/` mirror the module structure.

Run tests with:

```bash
uv run pytest tests/                      # All tests
uv run pytest tests/runtime/ -v           # Runtime tests with verbose output
uv run pytest tests/runtime/messaging/    # Specific module
```
