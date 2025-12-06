# Claude Code Guidelines (lib/runtime — questfoundry-runtime)

This file provides Claude Code-specific guidance for work in `lib/runtime/`. See also `AGENTS.md` and `CONTRIBUTING.md`.

## Core Model: Protocol Messages, Not ReAct

The runtime is **NOT** a ReAct loop. Roles communicate via **protocol messages**:

1. Role receives a message (routed by `receiver` field)
2. Role does work (calls tools like `read_state`, `write_state`)
3. Role calls `send_protocol_message(receiver=..., intent=..., content=...)`
4. **Role is DONE for this turn** — return immediately
5. Graph routes to `receiver`
6. Repeat until `receiver = "__terminate__"`

### Anti-Patterns (DO NOT DO)

| Pattern | Why It's Wrong |
|---------|----------------|
| "Final Answer:" termination | Protocol has no "final answer" — messages ARE the output |
| Counting all tool iterations | Only count FAILURES; valid calls are work |
| Single JSON output | Roles output messages, not a blob |
| ReAct naming | Triggers wrong mental model |
| Looping after `send_protocol_message` | Role is DONE — don't continue |

### Protocol Details

See `spec/04-protocol/` for:

- ENVELOPE.md — message structure
- INTENTS.md — intent types
- LIFECYCLES/ — role lifecycles
- FLOWS/ — multi-role flows (e.g., Hook Harvest)

### Parallel Execution

Roles can broadcast or fan out to multiple receivers:

- `receiver="*"` → all active roles
- Multiple `send_protocol_message` calls → LangGraph handles parallel routing

## Tool Calling (bind_tools)

The runtime uses native LLM tool binding. This was validated in December 2025:

| Model | Status | Notes |
|-------|--------|-------|
| Qwen3:8b (Ollama) | ✅ | Tested: single/multi-tool, streaming |
| Llama 3.1 8B | ✅ | Good for native tools |
| Llama 3.2 1B/3B | ⚠️ | Too small; use fallback |
| GPT-4, Claude 3 | ✅ | Native support |

**Benefits:**

- Structured IDs for tracking
- LangSmith visibility
- No parsing failures
- Better error handling
- Streaming support

**Fallback:** Text-based `Action/Action Input` available for models without native tool support.

**Test file:** `tests/test_bind_tools_ollama.py`

## Quick Rules

- **Treat spec/ as authoritative**: Changes flow from spec → runtime; re-bundle resources here
- **Structure**: Source in `src/`, tests in `tests/`; keep mirrors
- **No hand-editing**: Bundled resources in `src/questfoundry/runtime/resources/` are auto-generated

## Python Standards

- **Version**: Python 3.11–3.13
- **Imports**: Absolute only
- **Modern syntax**: `list[str]`, `str | None`, `from __future__ import annotations`
- **Type hints**: Full coverage; use `typing_extensions.override`
- **File I/O**: Use `pathlib.Path`
- **Comments**: Explain rationale; avoid trivial comments

## Workflows

```bash
uv sync                      # Install deps
uv run ruff check .          # Lint
uv run mypy                  # Type-check
uv run pytest                # Test
uv run ruff format .         # Format
uv run hatch run bundle      # Re-bundle resources (if spec changed)
```

## Commits

- **Format**: Conventional with scope `runtime` (e.g., `feat(runtime)`, `fix(runtime)`)
- **Size**: Small, atomic; no WIP
- **Breaking changes**: Mark with `!`

## Definition of Done

- ✅ Ruff, mypy, pytest all pass; code formatted with Ruff
- ✅ Resources re-bundled if spec changed
- ✅ Runtime flows validated if relevant
- ✅ Public API changes documented; breaking changes noted
- ✅ Tests cover behavior and edge cases (no trivial assertions)

## Tips

- Keep edits scoped; no unrelated comment/log churn
- Focus on actual failures, not false "iteration limits"
- When debugging, refer to the protocol spec, not ReAct patterns
- Use LangSmith to trace protocol message flows
