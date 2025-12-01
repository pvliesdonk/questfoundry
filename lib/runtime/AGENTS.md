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

### Why Text-Based Tool Calling (Not bind_tools)

The runtime uses text-based `Action/Action Input` format instead of native LLM tool binding
(`bind_tools`). This provides universal compatibility, though the landscape is evolving.

**Current state (December 2025):**

| Model | Native `bind_tools` | Notes |
|-------|---------------------|-------|
| Llama 3.1 8B | ✅ Works well | Recommended for native tools if needed |
| Llama 3.2 1B/3B | ⚠️ Problematic | Over-eager tool calling, JSON bugs, poor "should I call?" decisions |
| Qwen 2.5 / Qwen 3 | ✅ Works (recent Ollama) | Template issues fixed in 2025; official builds work |
| Other providers | Varies | Each uses different special tokens |

**Llama 3.2 (1B/3B) specific issues:**

- Too small for reliable "meta-decisions" (deciding *whether* to call a tool)
- Aggressive tool-first prompt template causes tool calls even for "hello"
- Known open bugs: malformed JSON, split `tool_calls` responses
- Workaround: Don't pass `tools` for generic chat; use separate clients

**Why we still use text-based format:**

- **Universal**: Works across all models without template/parser dependencies
- **Debuggable**: You can see exactly what the LLM outputs (no hidden tokens)
- **Consistent**: Same behavior across Ollama, OpenAI, Anthropic, local models
- **Robust**: No dependency on Ollama version, model version, or template updates

**Note:** Qwen 2.5 native tools now work in recent Ollama (mid-2025+), but we retain
text-based format for consistency and to support the broadest range of models.

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
