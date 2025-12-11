# Agent Guidelines (lib/runtime — questfoundry-runtime)

These rules are **mandatory** for work in `lib/runtime/`. Apply with root `AGENTS.md` and
`lib/runtime/CONTRIBUTING.md`.

## Scope & Boundaries

- Implements the spec as a runtime engine (LangGraph-based). Treat `spec/` as authoritative; do not
  modify it here. Re-bundle resources after spec updates.
- Code lives in `src/`, tests in `tests/`; keep structure mirrored.
- Never hand-edit bundled resources in `src/questfoundry/runtime/resources/` (or equivalents); they
  are produced by the bundling script.

## Execution Model (CRITICAL)

Roles communicate via **protocol messages**, not "final answers". This is fundamentally different
from ReAct-style agent loops.

### How Role Execution Works

1. Role receives a message (via graph routing based on `receiver` field)
2. Role does work (reads state, calls tools like `read_state`, `write_state`)
3. Role calls `send_protocol_message(receiver=..., intent=..., content=...)`
4. **Role is DONE for this turn** — return immediately after sending
5. Graph routes message to `receiver`
6. Repeat until `receiver = "__terminate__"`

```
Human Request
    ↓
Showrunner receives human.directive
    ↓
Showrunner calls send_protocol_message(receiver="plotwright", ...)
    ↓ ← Showrunner is DONE for this turn
Graph routes to Plotwright
    ↓
Plotwright does work, calls send_protocol_message(receiver="showrunner", ...)
    ↓ ← Plotwright is DONE for this turn
Graph routes to Showrunner
    ↓
... continues until receiver="__terminate__"
```

### Anti-Patterns (DO NOT IMPLEMENT)

| Anti-Pattern                 | Why It's Wrong                                                      |
| ---------------------------- | ------------------------------------------------------------------- |
| "Final Answer:" termination  | Protocol has no "final answer" — messages ARE the output            |
| Counting all tool iterations | Only count FAILURES; valid tool calls are work, not overhead        |
| Single JSON output           | Roles output messages, not a final JSON blob                        |
| ReAct pattern naming         | Triggers wrong associations; this is "protocol-message execution"   |
| Looping for more output      | After `send_protocol_message`, role is DONE — do NOT continue       |

### The Protocol Message IS the Output

When a role calls `send_protocol_message`, it's saying:

- "Here's my work output"
- "Route this to the receiver"
- "I'm done for now"

Do NOT look for additional output after this. Do NOT loop looking for "Final Answer".

### Parallel Execution

Roles can send multiple messages to different receivers (see `spec/04-protocol/FLOWS/`):

- `receiver="*"` broadcasts to ALL active roles
- Multiple `send_protocol_message` calls can fan out in parallel
- Graph handles parallel routing via LangGraph's built-in capabilities

Example from spec (Hook Harvest flow):

```
SR → LW: hook.accept (HK-20251030-01)
SR → PW: hook.accept (HK-20251030-02)
SR → RS: hook.defer (HK-20251030-05)
SR → Broadcast: hook.reject (HK-20251030-09)
```

All can execute in parallel.

### Iteration Counting

- Count FAILURES only (malformed tool calls, errors, exceptions)
- Valid tool calls ARE work, not overhead
- Reset failure counter on successful tool execution
- Max failures = safety net for broken agents, not a work limit

### Reference

For full protocol details, see `spec/04-protocol/` (ENVELOPE.md, INTENTS.md, LIFECYCLES/, FLOWS/).

### Tool Calling Strategy (bind_tools)

The runtime uses native LLM tool binding (`bind_tools`) for tool calling. This was validated
in December 2025 testing with Ollama Qwen3:8b.

**Tested Models (December 2025):**

| Model | Native `bind_tools` | Notes |
|-------|---------------------|-------|
| Qwen3:8b via Ollama | ✅ Works correctly | Tested: single/multi-tool, streaming, execution loop |
| Llama 3.1 8B | ✅ Works well | Good for native tools |
| Llama 3.2 1B/3B | ⚠️ Use fallback | Too small for reliable tool decisions |
| GPT-4, Claude 3 | ✅ Works | Native provider support |

**Benefits of bind_tools:**

- **Structured IDs**: Tool calls have unique IDs for tracking
- **LangSmith visibility**: Tool calls appear as first-class runs in traces
- **No parsing failures**: No regex needed to extract tool calls
- **Better error handling**: Structured responses vs text parsing
- **Streaming support**: Tool calls visible in stream chunks

**Hybrid Fallback:**

Text-based `Action/Action Input` format is available as fallback for models that don't
support native tools (small models, older providers). The runtime auto-selects based on
model capability.

**Test file:** `tests/test_bind_tools_ollama.py` validates bind_tools behavior.

## Required Practices

- Use `uv` for everything: `uv sync`, `uv run ruff check .`, `uv run mypy`, `uv run pytest`,
  `uv run ruff format .`, `uv run hatch run bundle`.
- Modern Python (3.11–3.13): absolute imports, `from __future__ import annotations`, modern typing
  syntax, and `typing_extensions.override` for overrides. Prefer `pathlib.Path` for I/O.
- Keep comments/docstrings purposeful (explain rationale). Avoid trivial tests/fixtures; focus on
  behavior, edge cases, and integration with bundled manifests.
- Keep edits scoped: no unrelated comment/log churn.

## Commit Conventions

- Conventional commits with scopes like `feat(runtime)`, `fix(runtime)`, `refactor(runtime)`,
  `test(runtime)`, `docs(runtime)`, `chore(runtime)`, `ci(runtime)`, `perf(runtime)`. Use `!` for
  breaking changes.
- Commit atomically; avoid WIP commits.

## Definition of Done (lib/runtime)

- Ruff, mypy, and pytest pass; code formatted with Ruff.
- Resources re-bundled if spec changed; runtime flows validated when relevant.
- Public surface changes documented; breaking changes explicitly noted.
- Tests cover new behavior without trivial or redundant assertions.

## Subagent Usage (for Claude Code / AI Assistants)

When working on this codebase, **use subagents proactively** to parallelize work and maintain
context efficiency.

### When to Use Subagents

| Task Type | Subagent Type | Example |
|-----------|---------------|---------|
| Codebase exploration | `Explore` | "Find all files that handle tool calling" |
| Multi-file implementation | `implementation-executor` | "Implement the async changes across 5 files" |
| Complex planning | `project-planner` | "Plan the migration from sync to async" |
| Documentation lookup | `claude-code-guide` | "How do LangGraph async nodes work?" |

### Parallel Subagent Patterns

**Exploration phase:** Launch multiple Explore agents in parallel:

```
- Explore: "Find tool calling implementation"
- Explore: "Find logging/tracing setup"
- Explore: "Find async patterns used"
- Explore: "Find documentation files"
```

**Implementation phase:** Use implementation-executor for well-defined tasks:

```
- implementation-executor: "Update control_plane.py with async patterns per plan"
- implementation-executor: "Add structured logging to node_factory.py"
```

### Anti-Patterns

- **Don't search manually** when Explore agent can do it faster
- **Don't implement large changes inline** when implementation-executor maintains better context
- **Don't re-read the same docs** when claude-code-guide has them cached

### Task Tool Best Practices

1. **Be specific** in prompts: Include file paths, line numbers, expected behavior
2. **Request structured output**: Ask agents to report findings in tables/lists
3. **Chain appropriately**: Use Explore results to inform implementation-executor prompts
4. **Parallelize independent work**: Launch multiple agents in single message when possible
