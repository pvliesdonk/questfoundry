# Creative Studio Meta-Model

A domain-agnostic meta-model for multi-agent creative AI studios.

## Overview

This meta-model defines the structure for AI studios where multiple LLM-powered agents collaborate on creative work. The model is designed to be:

- **Domain-agnostic**: Works for writing, game design, music production, or any creative domain
- **LLM-friendly**: Simple schemas with minimal required fields, optimized for context limits
- **Agent-driven**: Agents have agency; playbooks provide guidance, not rigid execution paths
- **Extensible**: Studios define their own artifact types, quality criteria, and workflows

## Key Concepts

### Agent-Driven Orchestration

Unlike traditional workflow systems where a runtime enforces state transitions, this model gives agency to an **orchestrator agent** (implementing the `orchestrator` archetype). The orchestrator:

- Receives customer requests
- Consults playbooks for guidance on how to proceed
- Delegates work to other agents
- Makes decisions about workflow based on context

Playbooks are **guidance documents**, not executable state machines. The runtime provides services (storage, validation, messaging) but does not control workflow.

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      STUDIO DEFINITION                       │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Agents    │  │  Playbooks  │  │   Stores    │         │
│  │ (who)       │  │ (guidance)  │  │ (where)     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Artifacts  │  │   Assets    │  │  Quality    │         │
│  │ (what)      │  │ (binaries)  │  │ (standards) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │Constitution │  │  Knowledge  │                          │
│  │ (rules)     │  │ (context)   │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ loaded by
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         RUNTIME                              │
│  (provides services, does NOT orchestrate)                   │
│                                                              │
│  - Agent instantiation    - Storage access                   │
│  - Message passing        - Validation services              │
│  - Knowledge retrieval    - Nudging (discrepancy detection)  │
└─────────────────────────────────────────────────────────────┘
```

### Agent Archetypes

The meta-model defines five archetypes:

| Archetype | Description |
|-----------|-------------|
| `orchestrator` | Coordinates work, interfaces with customer, delegates tasks |
| `creator` | Produces new artifacts and content |
| `validator` | Checks quality and provides feedback |
| `researcher` | Gathers, synthesizes, and retrieves information |
| `curator` | Maintains authoritative canonical knowledge |

Studios define concrete agents that implement one or more archetypes.

### Knowledge Stratification

Knowledge has two independent dimensions: **injection strategy** (how it's accessed) and **scope** (who can access it).

#### Injection Strategy (Layer)

The `layer` field defines the **default** access pattern:

| Layer | Default Injection | Use |
|-------|-------------------|-----|
| `constitution` | Always in system prompt | Inviolable principles |
| `must_know` | Always in system prompt | Critical context |
| `should_know` | Menu in prompt, full via tool | Important guidance |
| `role_specific` | Menu in prompt, full via tool | Specialist knowledge |
| `lookup` | Explicit query only | Reference material |

#### Scope (applicable_to)

The `applicable_to` field in a knowledge entry controls **who** can reference it:

```json
"applicable_to": {
  "agents": ["showrunner"],        // Only these agents
  "archetypes": ["orchestrator"],  // Or agents with this archetype
  "playbooks": ["story_spark"]     // Or during this playbook
}
```

#### Agent Override

An agent's `knowledge_requirements` lists can **override** the default injection:

```json
"knowledge_requirements": {
  "constitution": true,
  "must_know": ["spoiler_hygiene", "my_critical_heuristics"],  // Always inject
  "role_specific": ["reference_manual"],                       // Menu/tool
  "can_lookup": ["corpus_x"]                                   // Query only
}
```

**Key insight**: An entry with `layer: "role_specific"` can be placed in an agent's `must_know[]` list to always inject it for that agent. The agent's list is the injection strategy; the entry's `applicable_to` is the scope guard.

This supports the **menu+consult** pattern: summaries in prompt, details on demand.

#### The Reference Shelf Pattern (Corpus Type)

**Problem**: General domain knowledge (genre guides, craft books, "10 Commandments of Mystery Writing") is useful but doesn't belong in the agent's context menu. Including 500 reference entries alongside "Our Brand Voice" creates noise.

**Solution**: Use `content.type: "corpus"` for RAG-searchable reference material:

```json
{
  "id": "genre_conventions",
  "layer": "lookup",
  "summary": "Genre tropes and conventions. Search with consult_corpus tool.",
  "content": {
    "type": "corpus",
    "corpus_ref": {
      "store_ref": "reference_library",
      "path_pattern": "genres/mystery/**"
    }
  }
}
```

**How it works:**

1. **Menu**: Agent sees summary: "genre_conventions: Genre tropes and conventions"
2. **Trigger**: Agent wonders "What's typical chapter length for noir?"
3. **Consult**: Agent calls `consult_corpus(id="genre_conventions", query="chapter length noir")`
4. **Runtime**: Routes to vector search over the corpus store, returns top chunks

**Key distinction:**

| Type | For | Context Cost |
|------|-----|--------------|
| `inline`/`file_ref` | **Studio Dogma** (internal rules, brand voice) | Full content on demand |
| `corpus` | **Reference Material** (external guides, general craft) | Zero until queried |

This prevents the LLM from confusing a *suggestion* (from a general book) with a *rule* (from your Studio).

### Quality Gates

Quality criteria have two dimensions:

| Dimension | Options |
|-----------|---------|
| **Enforcement** | `runtime` (programmatic) or `llm` (semantic) |
| **Blocking** | `gate` (must pass) or `advisory` (feedback only) |

### Open Floor Principle

Studios operate on an **open floor** - there are no secrets between agents. The Runtime **NEVER denies access**; it only provides guidance and logging.

#### Vocabulary: Guidance, Not Security

The schema deliberately avoids security-loaded terminology to prevent implementers from building denial logic:

| Avoided Term | Used Instead | Rationale |
|--------------|--------------|-----------|
| `access_control` | `workflow_intent` | Focuses on workflow design, not authorization |
| `read`/`write` | `consumption_guidance`/`production_guidance` | Describes intent, not permission |
| `readers`/`writers` | `designated_consumers`/`designated_producers` | Identifies responsibility, not capability |

#### Runtime Behavior on Violations

When an agent acts outside `workflow_intent` (e.g., a non-designated producer writes to a store):

1. **Perform the operation** - never deny
2. **Log a `WorkflowViolation` event** - for audit and improvement
3. **Send a `nudge` message** - inform the agent of the deviation

This approach maintains workflow integrity through observation and feedback, not enforcement.

### Delegation: Responsibility, Not Permission

**Why have delegation at all?** In a "true open floor" (blackboard pattern), artifacts appear and any agent can act on them. This fails in practice:

| Problem | Without Delegation | With Delegation |
|---------|-------------------|-----------------|
| **Bystander Effect** | Everyone assumes someone else will act | Explicit: "Agent B, YOU handle this" |
| **Stampede Effect** | Multiple agents grab same task | Clear assignment prevents duplication |
| **Observability** | Can't track who's responsible | Workflow graph is visible and trackable |
| **Context Explosion** | Every agent watches everything | Bounded scope: "Read X, write Y" |

**Implicit Access Guarantee**: When artifacts are passed in `delegation.context.artifacts`, they are guaranteed available to the delegatee. Since stores never deny access (open floor), no explicit "grant access" field is needed. The act of including an artifact in a delegation IS the guarantee.

### Workflow Intent: Attention, Not Access Control

The `workflow_intent` on stores serves TWO purposes (neither is access control):

1. **Attention Mechanism**: Helps the Runtime decide what to load into each agent's context
   - Without this: every agent needs entire studio state (context window explosion)
   - With this: "Only load SourceCode store for CodeReviewer, ignore MarketingAssets"

2. **Workflow Observability**: Enables nudging when agents deviate from intended patterns
   - Non-designated producer writes? Log it, nudge, but **never deny**

### Artifact Lifecycle Governance

Artifacts can have lifecycle states (e.g., `draft` → `review` → `approved`). Transitions may require validation.

**The Protocol**: Agents cannot directly change lifecycle state. They must:

1. Send `lifecycle_transition_request`: "I believe artifact X should move from A to B"
2. Runtime validates `requires_validation` criteria
3. Runtime responds with `lifecycle_transition_response`: committed, rejected, or deferred

This formal handshake enables governance without breaking the open floor principle.

**Runtime Invariant**: The `_lifecycle_state` field (when present) is **write-protected by the Runtime**. Unlike the open floor principle for data access, this is a hard enforcement:

| Agent Action | Runtime Response |
|--------------|------------------|
| Update artifact content fields | Allowed (open floor) |
| Update `_lifecycle_state` directly | **Rejected** - must use transition protocol |
| Send `lifecycle_transition_request` | Validated, then committed/rejected/deferred |

This is the **one exception** to "never deny" - lifecycle state integrity requires it. The Runtime strips any `_lifecycle_state` mutations from artifact updates and logs the attempt.

## Schema Structure

```text
schemas/
├── core/
│   ├── _definitions.schema.json   # Shared type definitions
│   ├── studio.schema.json         # Top-level container
│   ├── agent.schema.json          # Agent definitions
│   ├── capability.schema.json     # Agent capabilities (permission to use tools)
│   ├── constraint.schema.json     # Agent constraints
│   ├── tool-definition.schema.json # Tool interfaces (what tools accept)
│   ├── artifact-type.schema.json  # Structured data types
│   ├── asset-type.schema.json     # Binary file types
│   ├── store.schema.json          # Storage definitions
│   ├── playbook.schema.json       # Workflow guidance (DAG-based)
│   ├── delegation.schema.json     # Work delegation format
│   └── message.schema.json        # Inter-agent protocol
├── governance/
│   ├── constitution.schema.json   # Inviolable principles
│   └── quality-criteria.schema.json # Validation rules
└── knowledge/
    ├── knowledge-entry.schema.json    # Knowledge items
    └── knowledge-layer.schema.json    # Layer configuration
```

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This overview document |
| [core.md](core.md) | Core primitive details |
| [patterns.md](patterns.md) | General design patterns |
| [tool-patterns.md](tool-patterns.md) | Tools-only orchestrators, `terminates_turn`, `communicate` |
| [knowledge-patterns.md](knowledge-patterns.md) | Budget management, menu+consult, `consult_knowledge` |
| [store-semantics.md](store-semantics.md) | Hot/cold/versioned/ephemeral store tiers, lifecycle states |

## Runtime Implementation Notes

When building a runtime that consumes these schemas:

### 1. Artifact Schema Compiler (The "Transpiler Tax")

The `artifact-type.schema.json` uses an **ordered array** of field definitions (better for LLM prompting), not JSON Schema's unordered `properties` map. This is a deliberate trade-off: engineering complexity for LLM ergonomics.

The runtime must:

- Parse artifact-type definitions with ordered `fields` array
- Compile them into standard JSON Schema Draft 2020-12 for validation
- Keep the ordered array for presenting schemas to LLMs
- Merge any `json_schema_override` for advanced features (oneOf, allOf, patternProperties)

For edge cases where the simplified `field_definition` format is insufficient, artifact types can include a `json_schema_override` object that gets merged into the compiled schema.

### 2. Versioned References

References to artifact types and playbooks support optional version constraints:

```text
story_draft           # Latest version
story_draft@1.0.0     # Exact version
story_draft@^1.0.0    # Compatible with 1.x.x
story_draft@~1.2.0    # Patch-level changes only
core:story_draft@>=2.0.0  # Cross-scope with constraint
```

The Runtime must resolve version constraints and warn when referenced definitions change in breaking ways.

### 3. Playbook Parsing (Never Raw JSON to Agents)

Playbooks are DAGs with `depends_on` relationships. **Never dump raw playbook JSON into agent prompts.** The runtime must:

- Parse the playbook DAG
- Determine current state (which steps are ready, which are blocked)
- Present a **contextual view** to agents: "You are in Phase X, Step Y. Your inputs are ready. Your goal is Z."

### 4. File Paths

All `file_path` references in knowledge entries must be **relative to the studio root**. Never use absolute paths to ensure portability across environments.

### 5. Reserved System Fields

Field names starting with `_` are **reserved for Runtime use**. Studios must not define custom fields with these names:

| Field | Type | Description |
|-------|------|-------------|
| `_id` | string | Unique instance identifier |
| `_type` | artifact_type_ref | Reference to artifact type definition |
| `_version` | semver | Instance version (for versioned stores) |
| `_created_at` | timestamp | When artifact was created |
| `_updated_at` | timestamp | When artifact was last modified |
| `_created_by` | agent_ref | Agent that created this artifact |
| `_lifecycle_state` | id | Current lifecycle state (write-protected) |

The Runtime automatically adds and maintains these fields. The `_lifecycle_state` field can only be modified via the lifecycle transition protocol.

### 6. Inter-Agent Messages

Use `message.schema.json` for all agent communication. Key message types:

| Message Type | Purpose |
|--------------|---------|
| `delegation_request` / `delegation_response` | Work assignment with implicit access guarantee |
| `progress_update` | Status during long-running tasks |
| `clarification_request` / `clarification_response` | Questions between agents |
| `feedback` | Quality check results |
| `nudge` | Runtime discrepancy detection (workflow violations) |
| `lifecycle_transition_request` / `lifecycle_transition_response` | Artifact state changes with validation |
| `escalation` | When agent cannot proceed |
| `completion_signal` | Artifact ready, phase complete, etc. |

### 7. Message Transport: Async Mailbox Pattern

Agents do **not** communicate directly. The Runtime acts as a **message broker** with per-agent mailboxes:

```text
┌─────────────────────────────────────────────────────────────────┐
│                         RUNTIME                                  │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Agent A     │    │ Agent B     │    │ Agent C     │         │
│  │ Mailbox     │    │ Mailbox     │    │ Mailbox     │         │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │         │
│  │ │ msg 1   │ │    │ │ msg 1   │ │    │ │ (empty) │ │         │
│  │ │ msg 2   │ │    │ └─────────┘ │    │ └─────────┘ │         │
│  │ └─────────┘ │    └─────────────┘    └─────────────┘         │
│  └─────────────┘                                                │
│                                                                  │
│  Message Router: to → mailbox, broadcast → all mailboxes        │
└─────────────────────────────────────────────────────────────────┘
```

**How it works:**

1. **Send**: Agent emits message → Runtime validates → routes to recipient mailbox(es)
2. **Receive**: When agent is activated, Runtime injects pending messages into context
3. **Acknowledge**: After processing, messages are marked read (not deleted for audit)

**Key properties:**

| Property | Behavior |
|----------|----------|
| **Asynchronous** | Sender doesn't block waiting for response |
| **Persistent** | Messages survive agent restarts |
| **Ordered** | Messages delivered in send order per sender |
| **Auditable** | Full message history retained |
| **Context-aware** | Runtime filters/prioritizes based on urgency, relevance |

**Broadcast vs. Targeted:**

- `to` field present → targeted delivery to specific agent's mailbox
- `to` field absent → broadcast to all agent mailboxes (rare, typically for announcements)

**Why not direct agent-to-agent?**

| Direct Communication | Async Mailbox |
|---------------------|---------------|
| Agents must be online simultaneously | Async by design |
| No central audit trail | Runtime logs everything |
| Complex routing logic in agents | Runtime handles routing |
| Context explosion (every agent watches all) | Runtime filters by relevance |

This pattern aligns with the **open floor principle**: all messages are visible to the Runtime for observability, but agents only receive messages in their mailbox context.

**Delegation Flow Example:**

```text
1. Orchestrator creates delegation_request message
   ├─ to: "scene_smith"
   ├─ payload.delegation_request: { task, context, expectations }
   └─ Runtime routes → Scene Smith's mailbox

2. Scene Smith activated
   ├─ Runtime injects: pending messages + referenced artifacts
   └─ Agent sees: "You have a delegation from Orchestrator: write scene X"

3. Scene Smith works, sends progress_update messages
   ├─ to: "orchestrator" (or correlation_id routes automatically)
   └─ Runtime routes → Orchestrator's mailbox

4. Scene Smith completes, sends delegation_response
   ├─ status: "success"
   ├─ artifacts_produced: [scene_draft_123]
   └─ Runtime routes → Orchestrator's mailbox
```

### 8. Flow Control: Preventing Mailbox Overflow

Agents have finite context windows. If Agent A sends 50 messages to Agent B, and Agent B only runs once per hour, Agent B's context becomes 90% "Inbox" and 10% "Work." The Runtime must implement **flow control** to prevent this.

**Three strategies:**

| Strategy | Pattern | When Used |
|----------|---------|-----------|
| **Secretary** | Auto-summarize older messages into digests | Inbox exceeds `auto_summarize_threshold` |
| **Bouncer** | Reject new delegations (backpressure) | Agent at `max_active_delegations` capacity |
| **TTL** | Expire ephemeral messages | Message not seen within `ttl_turns` |

**Secretary Pattern (Auto-Summarization):**

```text
Inbox before: [msg1, msg2, msg3, msg4, msg5, msg6]  ← 6 messages
                 ↓ Runtime triggers summarization (threshold=5)
Inbox after:  [digest{msg1-5}, msg6]                 ← 2 messages

The digest contains: summary, original_senders, action_items, urgency
```

**Bouncer Pattern (Backpressure):**

```text
Agent A: "delegate task X to Agent B"
Runtime: Checks Agent B's load
         active_delegations=3, max=3
Runtime → Agent A: "RecipientOverloaded. Agent B cannot accept
                    new delegations. Try a different agent or escalate."
```

**TTL Pattern (Expiration):**

```text
Agent A sends: "Are you there?" (ttl_turns: 2)
Turn 1: Agent B not activated
Turn 2: Agent B not activated
Turn 3: Message expires, deleted ← No value in seeing "Are you there?" hours later
```

**Configuration:**

```json
// studio.defaults.flow_control
{
  "max_inbox_size": 10,
  "auto_summarize_threshold": 5,
  "max_active_delegations_per_agent": 3,
  "default_message_ttl_turns": 24
}

// Per-agent override (e.g., orchestrator handles more load)
{
  "flow_control_override": {
    "max_inbox_size": 20,
    "max_active_delegations": 10
  }
}
```

**Runtime Context Priority (when constructing agent prompt):**

1. **Critical** - Nudges, errors (always shown)
2. **Active Delegations** - Current work (always shown)
3. **New Messages** - Up to `max_inbox_size`, then summarize
4. **Stale Messages** - Archived/hidden

### 9. Implementation Considerations (Hidden Costs)

Building a Runtime that consumes these schemas involves **non-obvious engineering costs**. This section documents them explicitly to inform build-vs-buy decisions.

#### 9.1 Summarizer Risk (LLM-in-the-Loop)

The **Secretary pattern** (auto-summarization) requires an LLM call:

```text
Inbox: [msg1, msg2, msg3, msg4, msg5, msg6] → LLM → "digest of msgs 1-5"
```

**Implications:**

| Concern | Impact |
|---------|--------|
| **Latency** | Agent activation blocked on summarization |
| **Cost** | Every summarization is a billable API call |
| **Fidelity** | Summaries may lose critical nuance |
| **Cascading failure** | If summarizer fails, agent context explodes |

**Mitigation strategies:**

- Use a smaller/cheaper model for summarization
- Pre-compute summaries asynchronously (background job)
- Include `action_items` and `urgency` fields to preserve critical info
- Keep original messages in audit trail (summarizer is lossy for context, not for history)

#### 9.2 Spying Requirement (Output Parsing)

The **lifecycle state protection** requires Runtime to inspect agent outputs:

```text
Agent output: { "content": "...", "_lifecycle_state": "approved" }
Runtime: Strips _lifecycle_state, logs violation, returns sanitized version
```

**Implications:**

- Runtime must parse all artifact mutations
- Can't use pass-through storage (must intercept writes)
- Adds validation overhead to every write operation

**Mitigation:**

- Only enable for artifact types with `lifecycle_states` defined
- Use JSON path-based filtering (cheap) rather than deep inspection

#### 9.3 Transpiler Tax (Schema Compilation)

The `artifact-type.schema.json` uses **ordered field arrays** (LLM-friendly) instead of JSON Schema's unordered `properties` (validator-friendly). This is a deliberate trade-off.

**You must build or integrate:**

1. **Schema compiler**: `field_definition[]` → JSON Schema Draft 2020-12
2. **Bidirectional conversion**: Keep ordered arrays for LLM prompts
3. **Override merging**: Handle `json_schema_override` for advanced features

This is ~500-1000 lines of code and requires maintenance when JSON Schema spec evolves.

#### 9.4 Bootstrapping Validation

Before the first agent runs, the Runtime must validate the studio definition for internal consistency:

| Check | Example Failure |
|-------|-----------------|
| **Reference integrity** | Agent references non-existent store |
| **Entry agent exists** | `entry_agent` points to agent without `is_entry_agent: true` |
| **Circular dependencies** | Playbook DAG has cycles |
| **Exclusive store conflicts** | Two agents claim same exclusive store |
| **Capability without tool** | Agent has capability for undefined tool |
| **Lifecycle state validity** | Transition references undefined state |

**Recommendation:** Build a `validate-studio` CLI command that runs all checks before Runtime startup.

#### 9.5 Tool Definition vs. Capability Gap

**capability.schema.json** grants permission (who can use what).
**tool-definition.schema.json** defines interface (what arguments it accepts).

Both are needed:

```text
Studio defines:  tool-definition { id: "web_search", input_schema: {...} }
Agent has:       capability { type: "tool", tool_ref: "web_search" }
Runtime does:
  1. Check agent has capability → permission granted
  2. Load tool definition → inject into LLM function calling
  3. Validate arguments → input_schema before execution
  4. Execute tool → with timeout, retry policy
```

This separation enables:

- Same tool, different agents (capability grants)
- Tool definitions shared across studios (import/export)
- Runtime validation before expensive tool execution

## Usage Patterns

### consult-schema Pattern

Agent requests schema + documentation before creating artifact:

```text
Agent: "What fields does a [ArtifactType] need?"
Runtime: Returns schema + docs + examples
Agent: Creates artifact following schema
```

### validate-with-feedback Pattern

Strict validation with actionable rejection:

```text
Agent: Creates artifact
Runtime: Validates against schema
If invalid: Returns specific errors with field-level guidance
Agent: Corrects and resubmits
```

### Runtime Nudging

The runtime can detect discrepancies and nudge agents:

```text
Runtime: "According to the playbook, phase X should produce
         artifact Y, but I don't see it. Is this intentional?"
Agent: [Explains or corrects]
```

## Design Philosophy

This meta-model makes deliberate trade-offs:

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| **Ordered field arrays** | Requires transpiler to JSON Schema | Better LLM prompt ergonomics |
| **Agent-driven orchestration** | Less predictable than state machines | Agents handle edge cases humans can't anticipate |
| **Open floor (no denial)** | Can't enforce data isolation | Simplicity; agents are trusted collaborators |
| **Delegation for assignment** | More protocol overhead | Prevents bystander/stampede effects |
| **Lifecycle handshake** | Agents can't directly mutate state | Enables validation gates without denial |
| **Guidance vocabulary** | Requires documentation discipline | Prevents implementers from building denial logic |

### The "Swarm vs. Orchestra" Spectrum

```text
Pure Blackboard          This Model              Rigid Workflow
(chaos)                  (structured autonomy)   (brittle)
     ├────────────────────────┼────────────────────────┤

- No delegation          - Delegation assigns     - State machine
- Everyone watches all     responsibility           enforces steps
- Self-organization      - Workflow intent        - No agent agency
                           guides attention
                         - Agents follow
                           playbook guidance
```

This model sits in the middle: **structured enough** for execution, **flexible enough** for creative work.

## Studio Design Methodology: Nouns First

When designing a new studio, follow the **Nouns First** principle: define *what exists* before defining *who acts* or *how work flows*.

### Recommended Order

```text
1. Constitution    ← Inviolable principles (ethical boundaries, brand voice)
2. Artifact Types  ← What gets created (the "nouns" of your domain)
3. Asset Types     ← Binary files (images, audio, exports)
4. Stores          ← Where artifacts live (with workflow intent)
5. Tools           ← External capabilities (APIs, services)
6. Agents          ← Who does the work (with capabilities + constraints)
7. Playbooks       ← How work flows (DAG-based guidance)
8. Quality Criteria← How work is validated
9. Knowledge       ← What agents need to know
```

### Why This Order?

| Step | Rationale |
|------|-----------|
| **Constitution first** | Sets boundaries before anything else exists |
| **Artifacts before agents** | You can't assign work until you know what work produces |
| **Stores before agents** | Agents reference stores; stores don't reference agents |
| **Tools before agents** | Capabilities grant tool access; tools must exist first |
| **Agents before playbooks** | Playbooks assign steps to agents |
| **Quality last** | Criteria reference artifact types and agents |

### Anti-Pattern: Agent-First Design

Starting with "I need a Writer agent and an Editor agent" leads to:

- Vague artifact definitions (what does "writing" produce?)
- Missing stores (where does work go?)
- Overlapping responsibilities (who owns the outline?)

### Example Walkthrough: Interactive Fiction Studio

```text
1. Constitution
   - "Never generate content harmful to minors"
   - "Maintain consistent narrative voice"

2. Artifact Types
   - story_brief (customer input)
   - world_bible (canonical lore)
   - scene_outline (structure)
   - scene_draft (prose)
   - dialogue_block (character speech)

3. Asset Types
   - character_portrait (image)
   - ambient_audio (sound)
   - export_package (epub/pdf)

4. Stores
   - story_workspace (mutable, all agents)
   - lore_vault (versioned, lorekeeper-exclusive)
   - exports (cold, publisher-exclusive)

5. Tools
   - web_search (research)
   - image_gen (portraits)
   - export_epub (publishing)

6. Agents
   - showrunner (orchestrator)
   - lorekeeper (curator) ← exclusive to lore_vault
   - scene_smith (creator)
   - gatekeeper (validator)
   - publisher (creator) ← exclusive to exports

7. Playbooks
   - new_story_playbook
   - scene_writing_playbook
   - revision_playbook

8. Quality Criteria
   - lore_consistency (llm, gate)
   - prose_quality (llm, advisory)
   - schema_valid (runtime, gate)
```

### The 80/20 Rule

Most studios need:

- 3-5 artifact types (don't over-model)
- 2-3 stores (workspace, canonical, exports)
- 5-8 agents (one orchestrator, several specialists)
- 1-3 playbooks (main workflow + edge cases)

Start minimal. Add complexity only when you hit real problems.

### Checklist Before Runtime

Before starting your Runtime, verify:

- [ ] Every `agent_ref` points to a defined agent
- [ ] Every `store_ref` points to a defined store
- [ ] Every `artifact_type_ref` points to a defined artifact type
- [ ] Every tool capability references a defined tool
- [ ] Exactly one agent has `is_entry_agent: true`
- [ ] `entry_agent` field matches that agent's ID
- [ ] No circular dependencies in playbook DAGs
- [ ] Exclusive stores have exactly one designated agent

## Getting Started

1. Define your studio in `studio.schema.json` format
2. Define agent archetypes and specific agents
3. Create artifact types for your domain
4. Write playbooks for common workflows
5. Configure stores for persistence
6. Add quality criteria for validation

See `docs/examples/` for complete examples.
