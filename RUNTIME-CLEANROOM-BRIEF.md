# Runtime Cleanroom Brief

> **Purpose**: Design a runtime engine for a multi-agent creative studio from scratch.
>
> **Context**: We have a well-defined meta-model (`meta/schemas/`) and domain instances (`domain-v4/`). We need a runtime that executes this specification. A previous implementation exists but has architectural debt — we're starting fresh.
>
> **Rule**: Everything must be reimagined. Old code is archived. Reference conceptual patterns below, but design from first principles based on meta/.

---

## What The CLI Must Do

The CLI is the user's interface to the studio. It should be reimagined alongside the runtime.

### Core Commands (Reconsider)

| Command | Purpose | Notes |
|---------|---------|-------|
| `qf ask` | Talk to studio | Main interaction loop |
| `qf doctor` | System health | Check providers, domain, stores |
| `qf roles` | List agents | Show capabilities |
| `qf projects` | Manage projects | List, create, switch |
| `qf checkpoints` | Manage state | List, resume, delete |

### CLI Patterns That Worked

- **Project-centric**: All operations scoped to a project directory
- **Provider-agnostic**: `--provider ollama/openai/anthropic`
- **Resume support**: `--resume` and `--from-checkpoint N`
- **Rich output**: Tables, panels, progress indicators

### CLI Patterns To Reconsider

- Should `ask` be interactive (REPL) or single-shot?
- How do checkpoints relate to projects?
- Should there be a `qf init` for new projects?
- What about `qf export` for publishing?

---

## What The Runtime Must Do

### 1. Load Domain

- Parse `domain-v4/studio.json` and all referenced files
- Validate against `meta/schemas/core/*.schema.json`
- Make agents, tools, stores, playbooks, artifacts available to the system

### 2. Orchestrate Agents

- Entry agent (Showrunner) receives user input
- Hub-and-spoke model: SR delegates to specialists, they return results
- Agents have capabilities, constraints, knowledge requirements
- Agents use tools to interact with the world

### 3. Execute Tools

- Tools are defined in `domain-v4/tools/*.json`
- Each tool has input/output schemas, constraints
- Agents can only use tools in their `capabilities` list
- Tool execution is logged and traced

### 4. Manage Storage

- Stores defined in `domain-v4/stores/*.json`
- Each store has `semantics` (hot, cold, versioned, ephemeral)
- Artifacts have `_lifecycle_state` (draft, review, approved, cold)
- Lifecycle transitions require validation (not type changes)

### 5. Handle Messaging

- Agents communicate through structured messages
- Flow control prevents overload
- Messages can be prioritized, summarized, expired

### 6. Support Resumption

- Save state at meaningful points
- Resume from any saved state
- Handle context limits gracefully

### 7. Provide Observability

- Trace all LLM calls hierarchically
- Log events in queryable format
- Support debugging and replay

---

## Patterns That Worked Well (Conceptual)

These patterns proved valuable. They should be **reconsidered and potentially reimplemented**, not copy-pasted.

### Checkpointing

**Concept**: After each Showrunner turn completes, save the full conversation state + store snapshots. Allow resuming from any checkpoint.

**Why it worked**: LLM context limits mean long workflows get truncated. Checkpoints let you resume mid-workflow. Also enables debugging (replay from checkpoint N).

**Reconsider**:

- What exactly constitutes a "checkpoint"?
- What state needs saving?
- How does this interact with message-based architecture?

### Structured Logging (JSONL)

**Concept**: Every significant event (delegation, tool call, error) logged as a JSON line with timestamp, event type, and payload.

**Why it worked**: Queryable with `jq`, parseable by tools, enables post-hoc analysis.

**Reconsider**:

- What events matter?
- What's the schema?
- How does this relate to tracing?

### LangSmith Tracing

**Concept**: Hierarchical traces showing: orchestrator run → SR turn → delegation → agent turn → tool calls. Each level shows inputs, outputs, token counts, latency.

**Why it worked**: Essential for debugging multi-agent workflows. See exactly where things went wrong.

**Reconsider**:

- How to structure trace hierarchy?
- What metadata to capture?
- How to make traces actionable?

### Message Broker Patterns

**Concept**: Agents have mailboxes. Messages flow through a broker with these patterns:

- **Bouncer**: Backpressure when mailbox full (reject or queue)
- **Secretary**: Summarize old messages when mailbox overflows
- **TTL**: Messages expire after timeout

**Why it worked**: Prevents context explosion. Agents don't get overwhelmed.

**Reconsider**:

- Is message-based the right model?
- Sync vs async execution?
- How do patterns compose?

### Hub-and-Spoke Delegation

**Concept**: Showrunner is the hub. All other agents are spokes. SR decides who to delegate to. Agents return results + recommendations, SR decides next step.

**Why it worked**: Clear control flow. No agent-to-agent spaghetti. SR maintains narrative coherence.

**Reconsider**:

- Is pure hub-and-spoke too limiting?
- How do "consultation" calls work?
- What about parallel execution?

### Tool Registry

**Concept**: Tools are declarative definitions. At runtime, build actual callable tools from definitions. Agents get filtered view (only their capabilities).

**Why it worked**: Single source of truth. Easy to add tools. Capability enforcement automatic.

**Reconsider**:

- How to compile tool definitions to callables?
- How to handle tool dependencies?
- Sandboxing/permissions?

### Hot/Cold Store Separation

**Concept**: Hot store is working memory (mutable, ephemeral). Cold store is committed canon (append-only, permanent). Promotion moves content from hot to cold.

**Why it worked**: Clear lifecycle. Agents can experiment without polluting canon.

**Reconsider**:

- v4 uses `_lifecycle_state` on artifacts, not separate types
- How does promotion work with lifecycle states?
- What triggers promotion?

---

## Minimum Viable Runtime

### Phase 1: Load & Validate

- [ ] Load studio.json and all referenced definitions
- [ ] Validate against meta/ schemas
- [ ] Expose typed objects (Studio, Agent, Tool, Store, etc.)

### Phase 2: Single Agent Execution

- [ ] Showrunner can receive user input
- [ ] SR can call tools (start with basic ones)
- [ ] SR can return response to user
- [ ] Basic tracing/logging

### Phase 3: Delegation

- [ ] SR can delegate to other agents
- [ ] Delegated agent executes and returns
- [ ] Control returns to SR

### Phase 4: Storage

- [ ] Hot store for working artifacts
- [ ] Cold store for committed artifacts
- [ ] Lifecycle state transitions
- [ ] Promotion protocol

### Phase 5: Checkpointing

- [ ] Save state after SR turns
- [ ] Resume from checkpoint
- [ ] List/manage checkpoints

### Phase 6: Flow Control

- [ ] Message broker infrastructure
- [ ] Bouncer/Secretary/TTL patterns
- [ ] Context management

---

## Non-Goals (For Now)

- Visual asset generation (external service, stub it)
- Audio asset generation (external service, stub it)
- Export/publish functionality (can come later)
- Player-Narrator live mode (can come later)

---

## Reference Material

| Resource | Purpose |
|----------|---------|
| `meta/schemas/core/` | The contract — runtime must implement this |
| `meta/docs/README.md` | Design philosophy and patterns |
| `domain-v4/` | Instance data to load |
| `_archive/runtime-v3/` | Old runtime (reference only, don't import) |
| `_archive/cli-v3.py` | Old CLI (reference only) |

### Debt Gradient in Old Runtime

Not all old code was equally problematic:

| Component | Debt Level | Notes |
|-----------|------------|-------|
| `runtime/domain/` (loader, validation) | **Low** | Already v4-native, loaded from meta/ schemas. Closest to correct. |
| `runtime/orchestrator_v4.py` | Medium | Structure sound, some v3 assumptions |
| `runtime/tools/` | Medium-High | Many v3 imports, but patterns reusable |
| `runtime/stores/` | **High** | Built around v3 Cold* types, needs full redesign |
| `runtime/resources.py` | **High** | Fundamentally v3 concept |

When reimplementing, the **domain loader** can be referenced most directly — it was already doing the right thing. Storage layer needs the most fresh thinking.

---

## How To Use This Brief

1. **Read meta/ first** — understand the schemas and design intent
2. **Design from schemas** — runtime types should mirror meta/ structures
3. **Implement incrementally** — follow the phases above
4. **Reference old code only when stuck** — ask "how did X work?" and I'll point to archive
5. **Question everything** — old patterns may not be right for v4

The goal is a runtime that truly implements meta/, not one that pretends to while hiding v3 underneath.
