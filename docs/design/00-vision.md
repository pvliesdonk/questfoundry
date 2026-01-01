# QuestFoundry v5: Core Vision

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## What QuestFoundry v5 IS

QuestFoundry v5 is a **pipeline-driven interactive fiction generation system**. It uses large language models as collaborators under constraint, not as autonomous agents.

### The Six-Stage Pipeline

```
DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
```

| Stage | Purpose | Output |
|-------|---------|--------|
| **DREAM** | Establish creative vision | Genre, tone, audience, themes |
| **BRAINSTORM** | Generate raw material | Characters, settings, hooks, what-ifs |
| **SEED** | Crystallize core elements | Protagonist, setting, central tension |
| **GROW** | Build story topology | Complete branching structure with scenes |
| **FILL** | Generate prose | Scene text and player choices |
| **SHIP** | Export to playable format | Twee, JSON, HTML bundles |

Each stage produces **validated YAML artifacts** that feed the next stage. Human review gates between stages ensure quality and creative control.

---

## What QuestFoundry v5 is NOT

### NOT Multi-Agent Orchestration

v5 does NOT use:
- Real-time agent-to-agent communication
- Hub-and-spoke delegation patterns
- Autonomous decision-making between agents
- Persistent agent state across sessions

### NOT Emergent Discovery

v5 does NOT allow:
- Structure emerging during prose generation
- Anchors discovered while writing branches
- Backflow where later stages modify earlier artifacts
- Unbounded iteration seeking "good enough"

---

## Core Philosophy

> **"The LLM as a collaborator under constraint, not an autonomous agent."**

This means:
1. **Humans curate** — Every expansion receives human review
2. **Decisions are legible** — All artifacts are readable YAML/Markdown
3. **Cheap to revisit** — Regenerate from any checkpoint
4. **Structure before prose** — Topology is complete before writing begins

---

## Five Foundational Rules

These rules are **absolute** and define v5's architecture:

### 1. No Persistent Agent State

Each stage execution starts fresh. There is no memory between sessions. Context comes from artifacts, not conversation history.

```yaml
# CORRECT: Context from artifacts
context:
  dream: ./artifacts/dream.yaml
  brainstorm: ./artifacts/brainstorm.yaml

# WRONG: Assuming prior conversation
# "As we discussed earlier..."
```

### 2. One LLM Call Per Stage (or Small Fixed Number)

Each stage makes a predictable number of LLM calls. No unbounded loops.

| Stage | Calls |
|-------|-------|
| DREAM | 1 |
| BRAINSTORM | 1-3 (characters, settings, hooks) |
| SEED | 1 |
| GROW.Spine | 1 |
| GROW.Anchors | 1 |
| GROW.Branches | N (one per branch, sequential) |
| FILL | N (one per scene brief) |
| SHIP | 0 (deterministic compilation) |

### 3. Human Gates Between All Stages

Every stage boundary is a checkpoint where humans:
- Review generated artifacts
- Edit if needed
- Approve to proceed
- Reject to regenerate

Gates can be configured as `required` (must approve) or `optional` (auto-advance with veto window).

### 4. Prompts as Visible Artifacts

All prompts live in readable files, not embedded in code.

```
/prompts
  /templates      # Stage-specific prompts
  /components     # Reusable fragments
  /schemas        # Output format specs
```

**Anti-pattern**: Prompts constructed dynamically in Python code.

### 5. No Backflow

Later stages cannot modify earlier artifacts. If BRANCHES reveals problems with ANCHORS:

1. Human reviews the conflict
2. Human manually edits ANCHORS
3. Pipeline regenerates from ANCHORS forward

**Never**: Automated revision of upstream artifacts.

---

## The GROW Stage: Special Complexity

GROW is the most complex stage. It decomposes into six sequential layers:

```
SPINE → ANCHORS → FRACTURES → BRANCHES → CONNECTIONS → BRIEFS
```

See [03-grow-stage-specification.md](./03-grow-stage-specification.md) for full details.

Key insight: **Anchors are declared, not discovered.** Hubs, gates, and bottlenecks are defined in ANCHORS before any branching occurs. Branches connect to predetermined structure.

---

## Artifact Format

All artifacts are YAML files with:

```yaml
# Standard header
type: seed              # Artifact type
version: 1              # Schema version

# Type-specific content
protagonist:
  name: Maria Chen
  occupation: Private investigator
  flaw: Cannot let go of cold cases
```

**Properties**:
- Human-readable and editable
- Version-controllable (meaningful diffs)
- Schema-validated before writing
- No binary formats
- No opaque identifiers

---

## State Management

All project state lives in versioned files:

```
/project
  /artifacts      # Generated outputs (YAML)
  /prompts        # Instruction templates
  /config         # Pipeline configuration
```

Git is the state manager. No database. No persistent runtime state.

---

## Success Criteria

v5 succeeds when:

1. **A coding agent can implement it** — Reading these docs is sufficient
2. **Short stories complete in hours** — DREAM to SHIP in 2-4 hours with human review
3. **All artifacts are human-editable** — No black boxes
4. **Failures recover gracefully** — Regenerate from any stage
5. **Output is playable** — Compiles to valid Twee/Ink/HTML

---

## Anti-Patterns (Explicit Rejections)

| Anti-Pattern | Why Rejected |
|--------------|--------------|
| Agent negotiation | Unpredictable, hard to debug |
| Incremental structure discovery | Creates inconsistent topology |
| Backflow / upstream revision | Makes pipeline state unpredictable |
| Unbounded iteration | No clear stopping criteria |
| Hidden prompts in code | Impossible to audit or edit |
| Complex state objects | Prefer flat YAML references |
| Autonomous quality decisions | Humans gate all transitions |

---

## Document Map

| Document | Content |
|----------|---------|
| [01-pipeline-architecture.md](./01-pipeline-architecture.md) | Orchestrator, stages, gates |
| [02-artifact-schemas.md](./02-artifact-schemas.md) | YAML formats per stage |
| [03-grow-stage-specification.md](./03-grow-stage-specification.md) | Six-layer GROW decomposition |
| [04-state-mechanics.md](./04-state-mechanics.md) | Codewords, stats, inventory |
| [05-prompt-compiler.md](./05-prompt-compiler.md) | Template system |
| [06-quality-bars.md](./06-quality-bars.md) | Validation criteria |
| [07-design-principles.md](./07-design-principles.md) | Core principles |
| [08-research-foundation.md](./08-research-foundation.md) | Academic grounding |
| [09-v4-reference.md](./09-v4-reference.md) | Historical context (non-canonical) |
| [10-semantic-conventions.md](./10-semantic-conventions.md) | Naming standards |
| [11-proposed-dependencies.md](./11-proposed-dependencies.md) | Tech stack |
| [12-getting-started.md](./12-getting-started.md) | Implementation order |
| [13-project-structure.md](./13-project-structure.md) | Directory layout |

---

## Decision Record

**Why pipeline over agents?**

Multi-agent orchestration (v4) proved difficult because:
- Agent-to-agent negotiation created unpredictable behavior
- Debugging required tracing complex message flows
- Quality emerged rather than being enforced
- Human intervention points were unclear

Pipeline architecture provides:
- Predictable execution flow
- Clear human intervention points
- Auditable artifacts at each stage
- Simple debugging (inspect stage inputs/outputs)

**Why no backflow?**

Backflow (later stages modifying earlier artifacts) creates:
- Circular dependencies
- Unpredictable convergence
- Difficult debugging

Forward-only flow with human-initiated revision provides:
- Clear causality
- Predictable regeneration
- Human control over structural changes
