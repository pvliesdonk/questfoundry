# Research Foundation

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

QuestFoundry v5 is grounded in research on interactive fiction, branching narrative construction, and LLM-driven content generation. This document summarizes key findings that inform the architecture.

**Primary Source**: if-craft-corpus (internal research collection)

---

## Branching Narrative Patterns

### Structural Patterns

Research identifies seven distinct branching structures:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Time Cave** | Pure branching, exponential growth | Short stories only |
| **Gauntlet** | Linear spine with short branches | Tutorial, guided experiences |
| **Branch and Bottleneck** | Diverge then reconverge | Sustainable at any length |
| **Quest/Modular** | Episodic clusters | Anthology structures |
| **Sorting Hat** | Early branching determines track | Replayability focus |
| **Loop and Grow** | Central loop, unlock content each cycle | State-heavy games |
| **Hub and Spoke** | Central hub, branches return | Investigation structures |

### v5 Alignment

v5 primarily uses **Branch and Bottleneck** with **Hub and Spoke** elements:
- ANCHORS define hubs and bottlenecks
- BRANCHES connect between anchors
- All paths reconverge at bottleneck points

---

## LLM-Specific Strategies

### What Works

Research (WHAT-IF 2024, GENEVA 2024) identifies successful approaches:

| Strategy | Description |
|----------|-------------|
| **Meta-prompting** | LLM generates prompts, not just content |
| **Bottom-up iteration** | Generate spine, then one branch at a time |
| **Emotional arc scaffolding** | Specify rise/fall patterns per branch |
| **Three-act anchoring** | Fix inciting incident, crisis, climax |
| **State/goal semantics** | Include explicit character state in each prompt |
| **Validation between phases** | Check topology before prose |

### v5 Implementation

| Strategy | v5 Feature |
|----------|------------|
| Meta-prompting | Prompt compiler assembles prompts from templates |
| Bottom-up iteration | BRANCHES generated sequentially, not in parallel |
| Emotional arc scaffolding | SPINE defines emotional trajectory |
| Three-act anchoring | ANCHORS lock structural points |
| State/goal semantics | Context injection includes prior state |
| Validation between phases | CONNECTIONS validates before FILL |

---

## Known LLM Failure Modes

### Failure: Full Topology in One Shot

**Symptom**: Request complete branching structure in single prompt.

**Result**: Deviation from intent, stagnation of ideas, inconsistent connections.

**Mitigation**: Six-layer GROW decomposition. Each layer is a separate LLM call with validated output.

### Failure: Parallel Branch Generation

**Symptom**: Generate all branches simultaneously.

**Result**: Disconnected parallel narratives, inconsistent world state, contradictory character behavior.

**Mitigation**: Sequential branch generation. Each branch receives prior branches as context.

### Failure: Discovering Anchors During Branching

**Symptom**: Hubs, gates, bottlenecks emerge while writing branches.

**Result**: Inconsistent convergence, orphaned content, structural confusion.

**Mitigation**: ANCHORS declared and locked before FRACTURES or BRANCHES.

### Failure: State Tracking Degradation

**Symptom**: Character state becomes inconsistent over long sessions.

**Result**: Character motivation hallucination, contradictory behavior.

**Mitigation**: Explicit state in context. Character state summarized in each prompt, not assumed from conversation history.

### Failure: Homogeneous Emotional Arcs

**Symptom**: All branches follow "challenge → success → happy ending."

**Result**: Lack of variety, predictable outcomes.

**Mitigation**: SPINE specifies varied arc shapes. Briefs include emotional beat targets.

---

## Decomposition Research

### The WHAT-IF Five-Phase Model (2024)

Research on LLM story generation identified five phases:

1. **Exposition** — Establish world and character
2. **Rising Action** — Introduce conflict and stakes
3. **Climax** — Peak tension and confrontation
4. **Falling Action** — Consequences unfold
5. **Resolution** — New equilibrium

### v5 Mapping

| WHAT-IF Phase | v5 Stage |
|---------------|----------|
| Exposition | DREAM + SEED |
| Rising Action | GROW (Spine beats 1-4) |
| Climax | GROW (Spine beat 5-6) |
| Falling Action | GROW (Spine beat 7) |
| Resolution | GROW (Spine beat 8) |

---

## The Bottom-Up Principle (GENEVA 2024)

> "Bottom-up iteration outperforms top-down generation."

### What This Means

Instead of generating full structure top-down, generate iteratively:

1. Generate spine (linear arc)
2. Generate anchors (structural points)
3. Generate fractures (where branching occurs)
4. Generate branch 1 (with spine + anchors as context)
5. Generate branch 2 (with spine + anchors + branch 1 as context)
6. Continue until complete

### Why It Works

- Each generation has full context of prior decisions
- Consistency maintained through explicit reference
- Errors caught early in validation steps
- Human can intervene between iterations

---

## Quality Standards from IF Industry

### The Nine Quality Bars

Research aggregates quality standards from Choice of Games, ChoiceScript, Emily Short, and BioWare:

1. **Integrity** — Structural completeness
2. **Reachability** — Content accessibility
3. **Comprehension** — Player understanding
4. **Style** — Voice consistency
5. **Safety** — Responsible content handling
6. **Accessibility** — Inclusive access
7. **Canon** — World consistency
8. **Spoiler Hygiene** — Discovery preservation
9. **Research Posture** — Real-world claim support

### v5 Adaptation

v5 uses 8 bars (combining some, deferring others):

| Industry Standard | v5 Bar |
|-------------------|--------|
| Integrity | Integrity |
| Reachability | Reachability |
| Comprehension | (folded into Accessibility) |
| Style | Style |
| Safety | (configuration, not bar) |
| Accessibility | Accessibility |
| Canon | (implicit in state consistency) |
| Spoiler Hygiene | Presentation |
| Research Posture | (deferred to v5.1) |

---

## Mechanics Research

### Stat System Patterns

| Pattern | Pros | Cons |
|---------|------|------|
| **Opposed pairs** | Enforces consistency | Risk of min-maxing |
| **Accumulative** | Enables specialization | May lock out generalists |
| **Hidden variables** | Surprising consequences | Players may feel cheated |

### Skill Check Patterns

| Type | When to Use |
|------|-------------|
| **Threshold** (deterministic) | Competency checks ("you know this or don't") |
| **Probability** (dice) | External chaos ("does the guard look?") |

### v5 Decision

v5.0 uses **threshold-only** for simplicity. Probability checks deferred to v5.1+.

---

## Fail Forward Principle

From Choice of Games and industry practice:

> "Never let a failed check stop the story."

### Implementation

Every check has:
- **Success path**: What happens when passed
- **Failure path**: What happens when failed (continues story with different state)

### Anti-Pattern

```
Check → Fail → "Try again" (loop)
```

### Correct Pattern

```
Check → Fail → Continue with consequence
```

---

## Context Windowing Research

### Token Budget Strategies

From prompt engineering research:

| Strategy | Description |
|----------|-------------|
| **Sandwiching** | Critical content at start and end |
| **Summarization** | LLM-generated overviews of long content |
| **Skeleton extraction** | Structure only, no prose |
| **Priority-based omission** | Low-priority content omitted first |

### Attention Patterns

LLMs pay more attention to:
- Content at prompt start
- Content at prompt end
- Structured content (lists, headers)
- Explicit instructions

Less attention to:
- Middle of long prompts
- Prose-heavy content
- Implicit expectations

### v5 Implementation

Prompt compiler implements:
- Priority levels (critical, standard, background)
- Position control (start, middle, end)
- Compression strategies per component

---

## Research Sources

### Academic

| Source | Contribution |
|--------|--------------|
| WHAT-IF (2024) | Meta-prompting, five-phase decomposition |
| GENEVA (2024) | Bottom-up iteration superiority |
| Emotional Arc Studies (2025) | Arc scaffolding for consistency |

### Industry

| Source | Contribution |
|--------|--------------|
| Choice of Games | Quality standards, fail-forward principle |
| ChoiceScript | Quicktest/randomtest validation patterns |
| Emily Short | Narrative design patterns |
| BioWare | Content QA vs Technical QA distinction |
| Inkle | State management in interactive narrative |

### Standards

| Source | Contribution |
|--------|--------------|
| WCAG 2.1 | Accessibility guidelines |
| IF Competition Standards | Judging criteria, player expectations |

---

## Corpus Location

The full if-craft-corpus is available at:

```
/mnt/code/if-craft-corpus/
```

Key documents:
- `branching_narrative_construction.md` — Detailed structural patterns
- `mechanics_design_patterns.md` — Stat systems and skill checks
- `quality_standards.md` — The nine quality bars
- `llm_strategies.md` — LLM-specific techniques
- `creative_workflow_pipeline.md` — Production pipeline research

---

## See Also

- [00-vision.md](./00-vision.md) — How research informed vision
- [03-grow-stage-specification.md](./03-grow-stage-specification.md) — GROW implements research findings
- [06-quality-bars.md](./06-quality-bars.md) — Quality standards from research
