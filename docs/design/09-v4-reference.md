# v4 Reference (NON-CANONICAL)

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Reference Only — NOT a template for v5

---

## Warning

> **This document is for historical reference only.**
>
> v4 was a multi-agent orchestration system. v5 is a pipeline-driven system.
> These are fundamentally different architectures.
>
> **Do not** use v4 patterns as templates for v5 implementation.
> **Do not** port v4 code to v5.
> **Do** understand what v4 attempted and why v5 takes a different approach.

---

## What v4 Was

QuestFoundry v4 was a **multi-agent orchestration system** for collaborative fiction creation.

### Core Concept

12 specialized agents communicated through a message broker, with a Showrunner orchestrating work delegation.

```
┌─────────────────────────────────────────────────────┐
│                    Showrunner                        │
│                  (Orchestrator)                      │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Plotwright │ │ Scene Smith │ │ Gatekeeper  │
│  (Creator)  │ │  (Creator)  │ │ (Validator) │
└─────────────┘ └─────────────┘ └─────────────┘
```

### The 12 Agents

| Agent | Archetype | Responsibility |
|-------|-----------|----------------|
| Showrunner | Orchestrator | Hub-and-spoke delegation |
| Lorekeeper | Librarian | Canon management |
| Plotwright | Architect | Story structure |
| Scene Smith | Author | Prose writing |
| Gatekeeper | Validator | Quality enforcement |
| Researcher | Fact Checker | Plausibility |
| Style Lead | Curator | Aesthetic coherence |
| Lore Weaver | Synthesizer | Canon deepening |
| Codex Curator | Documentarian | Player-safe entries |
| Art Director | Planner | Visual planning |
| Audio Director | Planner | Audio planning |
| Book Binder | Publisher | Static export |

---

## Why v4 Was Replaced

### Problem 1: Unpredictable Agent Negotiation

Agents could delegate to each other dynamically. This made:
- Execution flow unpredictable
- Debugging extremely difficult
- Quality dependent on negotiation success

### Problem 2: Emergent Rather Than Enforced Quality

Quality gates existed but agents could argue about them. A Gatekeeper rejection might be negotiated away rather than enforced.

### Problem 3: Complex State Management

Hot/cold stores, lifecycle states, exclusive writers — all requiring runtime enforcement that was difficult to get right.

### Problem 4: Unclear Human Intervention Points

The system ran continuously. Human could interrupt but the natural intervention points weren't clear.

### Problem 5: Context Explosion

Long sessions accumulated conversation history, exceeding context limits and requiring complex summarization.

---

## What v5 Learned from v4

### Agent Archetypes → Stage Responsibilities

The v4 agent archetypes represent real responsibilities that still exist in v5:

| v4 Agent | v5 Equivalent |
|----------|---------------|
| Showrunner | Pipeline Orchestrator |
| Plotwright | GROW stage (structure) |
| Scene Smith | FILL stage (prose) |
| Gatekeeper | Validation checks |
| Lorekeeper | State management |
| Book Binder | SHIP stage |

### Quality Criteria → Quality Bars

v4's governance/quality-criteria informed v5's quality bars:

| v4 Criterion | v5 Bar |
|--------------|--------|
| integrity | Integrity |
| reachability | Reachability |
| style | Style |
| determinism | Determinism |
| presentation | Presentation |
| accessibility | Accessibility |
| nonlinearity | Nonlinearity |
| gateways | Gateways |

### Knowledge Patterns → Prompt Components

v4's knowledge base patterns informed v5's prompt compiler:

| v4 Pattern | v5 Equivalent |
|------------|---------------|
| must_know | Critical priority components |
| should_know | Standard priority components |
| lookup | Context sources |
| constitution | Design principles doc |

---

## Valuable Concepts (Preserved Differently)

These v4 concepts survive in v5 but implemented differently:

### Structure Before Prose

**v4**: Playbooks specified passage_brief → passage workflow.

**v5**: GROW produces briefs, FILL produces prose. Sequential stages enforce the order.

### Lifecycle States

**v4**: draft → review → approved → cold (runtime-managed)

**v5**: Artifacts are either:
- In-progress (current stage)
- Approved (human-gated)
- Shipped (exported)

File-based rather than runtime-managed.

### Exclusive Writers

**v4**: Only Lorekeeper could write to cold store.

**v5**: Each stage owns its artifacts. Human gates control progression.

### Topology Patterns

**v4**: Hubs, loops, gateways as design patterns.

**v5**: Same patterns, defined in ANCHORS rather than discovered by agents.

---

## What NOT to Port

These v4 concepts should **not** be implemented in v5:

### Agent Delegation

v4 agents could delegate work to each other dynamically. v5 uses fixed stage sequence.

### Message Broker

v4 used async mailboxes for inter-agent communication. v5 has no inter-stage communication — artifacts are the only state transfer.

### Playbook Execution

v4 playbooks were runtime-interpreted workflows. v5 has fixed pipeline stages.

### Hot/Cold Store Distinction

v4 had mutable "hot" and immutable "cold" stores. v5 uses file-based artifacts with human gates for immutability.

### Rework Loops

v4 allowed agents to request rework of prior work. v5 requires human-initiated revision (no backflow).

---

## v4 Location (Archived)

v4 materials are archived at:

```
_deprecated/
├── runtime-v3/        # Older runtime (pre-v4)
├── runtime-v4/        # v4 runtime (after v5 migration)
├── tests-v3/          # v3 tests
└── docs-current-v3/   # v3 documentation

domain-v4/             # Domain definitions (still useful for reference)
├── agents/            # Agent definitions
├── governance/        # Quality criteria
├── knowledge/         # Knowledge entries
├── playbooks/         # Workflow definitions
└── stores/            # Store definitions
```

---

## Reading v4 for Understanding

If you need to understand what v4 attempted:

### Start Here

1. `domain-v4/agents/showrunner.json` — Orchestrator definition
2. `domain-v4/governance/constitution.json` — Core principles
3. `domain-v4/knowledge/must_know/story_building_workflow.json` — Workflow patterns

### For Patterns

- `domain-v4/knowledge/must_know/topology_patterns.json` — Hubs, loops, gateways
- `domain-v4/knowledge/must_know/choice_integrity.json` — Choice design
- `domain-v4/governance/quality-criteria/` — Quality standards

### For Artifacts

- `domain-v4/artifact-types/` — What v4 produced
- `meta/schemas/core/artifact-type.schema.json` — Schema format

---

## Summary: v4 vs v5

| Aspect | v4 | v5 |
|--------|----|----|
| Architecture | Multi-agent orchestration | Sequential pipeline |
| Control | Showrunner delegates dynamically | Fixed stage sequence |
| Communication | Message broker | Artifact files |
| State | Runtime hot/cold stores | Git-versioned files |
| Human role | Can interrupt anytime | Gates at stage boundaries |
| Quality | Agents negotiate | Validation enforced |
| Iteration | Rework loops | Human-initiated only |
| Debugging | Trace message flow | Inspect stage artifacts |

---

## See Also

- [00-vision.md](./00-vision.md) — Why v5 exists
- [01-pipeline-architecture.md](./01-pipeline-architecture.md) — What replaced v4
- [08-research-foundation.md](./08-research-foundation.md) — Research that informed v5
