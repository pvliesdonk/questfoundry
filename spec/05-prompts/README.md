# Layer 5 — Role Prompts (AI Agent Kits)

Start here: `05-prompts/USAGE_GUIDE.md` for a practical, step‑by‑step guide to upload prompts to
ChatGPT/Claude/Gemini and run a two‑stage production (Stage 1 minimal kit, Stage 2 full kit).

Upload kits/zips: see `05-prompts/upload_kits/README.md` or run `uv run qfspec-build-kits` to
generate ready‑to‑attach folders and archives under `dist/upload_kits/*`.

## Overview

Layer 5 transforms Layer 1 role charters into executable AI agents that communicate via Layer 4
protocol and validate artifacts against Layer 3 schemas.

## Architecture: Loop-Focused Design

Layer 5 uses a **loop-focused architecture** where loops are the primary executable units and roles
participate in loops. This design choice reflects that QuestFoundry workflows are fundamentally
**procedure-driven**: a "Lore Deepening" loop has a specific sequence (open TU → frame questions →
draft canon → pre-gate → handoff to Codex Curator) that remains constant regardless of whether
humans or AI fill the roles.

By making loops the executable units, we achieve:

- **Single source of truth**: Each loop procedure lives in one playbook, not duplicated across N
  role prompts
- **Clear coordination**: The Showrunner loads a playbook and orchestrates; roles respond with their
  domain expertise
- **Role interchangeability**: Swap human/AI implementations without changing the procedure
- **Maintainability**: Update a loop once, not across 7 role prompts that participate in it

The architecture provides three formats to support different use cases:

- **Loop Playbooks** (`loops/`) — 13 executable procedures with message sequences, RACI,
  deliverables
- **Role Adapters** (`role_adapters/`) — 15 thin interface specs (50-100 lines) for multi-role
  orchestration
- **Full Role Prompts** (`[role]/system_prompt.md`) — Comprehensive standalone guides (200-300
  lines) for learning or solo work

## Purpose

- Provide AI agent prompts that implement each of the 15 roles
- Enable human/AI interchangeability for roles
- Support stateful conversation sessions during active loops
- Implement agent-to-human question/answer in interactive mode
- Use Layer 4 protocol for all inter-role communication

## Design Principles

1. **Role Fidelity:** Prompts must faithfully implement Layer 1 charters
2. **Protocol Native:** All communication via Layer 4 envelopes
3. **Schema Compliant:** All artifacts validated against Layer 3 schemas
4. **Context Aware:** Maintain TU, snapshot, hot/cold awareness
5. **Safety First:** Enforce PN boundaries at agent level
6. **Stateful Sessions:** Conversation history during active loops
7. **Human Collaborative:** Agents can ask clarifying questions

## Repository Structure

```
05-prompts/
  README.md                    # This file
  DESIGN.md                    # Detailed design decisions
  ARCHITECTURE.md              # Loop-focused architecture documentation

  loops/                       # PRIMARY: 13 executable loop playbooks
    *.playbook.md              # Complete procedures with message sequences
    examples/                  # Validation message flows
      *_flow.json

  role_adapters/               # NEW: 15 thin interface specs (multi-role orchestration)
    *.adapter.md               # Core expertise, intents, loop participation (50-100 lines)

  _shared/
    context_management.md      # How to track TU, snapshot, hot/cold
    safety_protocol.md         # PN boundary enforcement
    escalation_rules.md        # When to wake Showrunner
    human_interaction.md       # Asking questions to human

  showrunner/
    system_prompt.md           # Index and navigation (110 lines)
    loop_orchestration.md      # How to execute playbooks (100 lines)
    manifest_management.md     # Hot/Cold manifest operations (102 lines)
    initialization.md          # 7-step project setup (223 lines)
    protocol_handlers.md       # Message validation, error handling (230 lines)

  gatekeeper/
    system_prompt.md           # Full role prompt (200-300 lines)

  lore_weaver/
    system_prompt.md           # Full role prompt with loop participation

  scene_smith/
    system_prompt.md           # Full role prompt
  plotwright/
    system_prompt.md           # Full role prompt
  codex_curator/
    system_prompt.md           # Full role prompt
  style_lead/
    system_prompt.md           # Full role prompt
  researcher/
    system_prompt.md           # Full role prompt
  art_director/
    system_prompt.md           # Full role prompt
  illustrator/
    system_prompt.md           # Full role prompt
  audio_director/
    system_prompt.md           # Full role prompt
  audio_producer/
    system_prompt.md           # Full role prompt
  translator/
    system_prompt.md           # Full role prompt
  book_binder/
    system_prompt.md           # Full role prompt
  player_narrator/
    system_prompt.md           # Full role prompt

  tests/
    test_prompts/              # Prompt validation tests
    fixtures/                  # Test scenarios
```

**Dual-Format Strategy:**

- **Full prompts** (`[role]/system_prompt.md`) — Standalone use (ChatGPT sessions, learning)
- **Role adapters** (`role_adapters/[role].adapter.md`) — Multi-role orchestration (efficient
  context)

## Key Concepts

### Loop Playbooks (Primary Executable Units)

Loop playbooks are the **primary way to execute QuestFoundry workflows**. Each playbook contains:

- Complete procedure with message sequences (from Layer 0 + Layer 4)
- RACI matrix defining role responsibilities
- Deliverables and success criteria
- Schema references for validation
- Example message flows for testing

Showrunner loads a playbook and orchestrates roles through the defined steps. Roles respond to
intents using their domain expertise (from adapters or full prompts).

### Stateful Sessions

Agents maintain conversation history during active loops:

```python
class RoleSession:
    role: str
    tu_context: str
    conversation_history: List[Message]
    active_since: datetime

# Session lifetime = loop duration
# Showrunner manages wake (create) / dormancy (archive)
```

### Agent-to-Human Communication

In interactive mode, agents can ask clarifying questions:

```python
# Agent perspective (simple API)
tone = ask_human(
    question="This hook feels ambiguous. Horror or mystery tone?",
    context={"hook_id": "HOOK-001"},
    suggestions=["horror", "mystery", "both"]
)

# Implemented via Layer 4 protocol:
# - Agent sends: intent="human.question"
# - Showrunner routes to CLI
# - Human responds via CLI
# - Response returns as: intent="human.response"
```

### Context Management

Every agent maintains awareness of:

- Current TU
- Active snapshot (if any)
- Hot vs Cold state
- Recent artifacts in conversation
- Role dormancy signals

## Dual Format Strategy

Layer 5 provides **two formats** for different use cases:

### Full Role Prompts (`[role]/system_prompt.md`)

**Use when:**

- Working with single role in isolation
- Learning how a role thinks and operates
- Standalone ChatGPT/Claude session
- Human needs comprehensive guidance

**Contains:**

- Complete mission and operating model
- Domain expertise and decision-making guidance
- Loop participation references
- Examples and edge cases

### Role Adapters (`role_adapters/[role].adapter.md`)

**Use when:**

- Multi-role orchestration via Showrunner
- Loop playbook is primary, role supplements
- Layer 6 library integration
- Efficient context (minimal tokens)

**Contains:**

- Core expertise summary
- Protocol intents handled
- Loop participation RACI
- Handoff protocols

## Prompt Structure

Each role prompt includes:

1. **Role Charter** (from Layer 1)
2. **Mission & Scope**
3. **Protocol Integration**
   - How to receive/send Layer 4 envelopes
   - Intent routing
4. **Schema Validation**
   - Required artifact structure
   - Validation requirements
5. **Context Tracking**
   - TU awareness
   - Hot/Cold boundaries
6. **Safety Guidelines**
   - PN boundary enforcement
   - Spoiler hygiene
7. **Quality Bars** (role-specific)
8. **Human Interaction**
   - When to ask questions
   - How to phrase questions
9. **Dormancy Signals**
   - When to go dormant
   - How to signal Showrunner
10. **Example Conversations**
    - Typical message flows
    - Edge cases

## Implementation Modes

### Guided Mode (Default for MVP)

- Loop-by-loop checkpoints
- Author approves next loop before execution
- Agents run within approved loop
- Human reviews artifacts between loops

### Express Mode (Future)

- Fully autonomous execution
- Setup questions → complete manuscript
- Only stops for gatecheck failures

### Interactive Mode

- Agents ask clarifying questions inline
- Free-form human responses
- Conversational collaboration
- Available as `--interactive` flag

## Provider Support

Agents are provider-agnostic. Layer 6 handles:

- Text generation (OpenAI, Ollama, Google AI Studio, Amazon Bedrock)
- Image generation (A1111, DALL-E, Imagen 4)
- Audio generation (ElevenLabs, etc.)

Per-role provider configuration supported (future).

## Evolution Path

### Phase 1: Core Roles (MVP)

- Showrunner
- Gatekeeper
- Lore Weaver
- Scene Smith

### Phase 2: Content Roles

- Plotwright
- Codex Curator
- Style Lead

### Phase 3: Asset Roles

- Art Director
- Illustrator
- Audio Director
- Audio Producer

### Phase 4: Support Roles

- Researcher
- Translator
- Book Binder

## Testing Strategy

1. **Prompt Validation:** Verify prompts reference correct schemas/protocol
2. **Intent Coverage:** Each role handles required intents
3. **Protocol Compliance:** All messages conform to Layer 4
4. **Schema Compliance:** All artifacts validate against Layer 3
5. **Safety Checks:** PN boundaries enforced
6. **Conversation Tests:** Example flows execute correctly

## Dependencies

- **Layer 0:** Quality bars, loops, PN principles
- **Layer 1:** Role charters and missions
- **Layer 2:** Artifact structures and terminology
- **Layer 3:** JSON schemas for validation
- **Layer 4:** Protocol envelopes and intents

## Integration with Layer 6

Layer 6 (questfoundry-lib) will:

- Bundle these prompts as package resources
- Provide `get_prompt(role_name)` API
- Manage role sessions
- Handle LLM provider routing
- Implement agent-to-human callbacks

## Next Steps

1. ✅ Complete Layer 4 (protocol) → **DONE**
2. 🚧 Design shared prompt patterns (`_shared/`)
3. ⏸️ Implement Showrunner prompt (orchestrator)
4. ⏸️ Implement Gatekeeper prompt (quality enforcement)
5. ⏸️ Implement Lore Weaver prompt (canonization)
6. ⏸️ Implement Scene Smith prompt (prose generation)
7. ⏸️ Create example conversations
8. ⏸️ Test with Layer 6 implementation

## References

- [Layer 1 - Roles](../01-roles/README.md)
- [Layer 4 - Protocol](../04-protocol/README.md)
- [Targeted Loops](../00-north-star/LOOPS/README.md)
- [Quality Bars](../00-north-star/QUALITY_BARS.md)
