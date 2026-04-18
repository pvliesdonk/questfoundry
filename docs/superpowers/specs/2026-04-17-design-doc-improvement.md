# Design Doc Improvement — LLM-Debuggable Specification

**Date:** 2026-04-17
**Status:** Approved — ready for implementation planning

---

## Problem

The existing design docs are conceptually correct and clear to a human reader. The failure is specific: when an LLM agent is **debugging** faulty code, it reverts to pattern-matching against a vague mental model formed from narrative prose, rather than reasoning precisely about what is exactly wrong. All failure modes occur: wrong graph structure, missing behavior, and misapplied concepts — and they span multiple stage boundaries.

The root cause is not that the docs are unclear — it is that they are optimized for *understanding*, not for *violation detection*. Rules are stated in prose paragraphs. Invariants are scattered across three files. Wrong behavior is almost never described explicitly. An agent cannot mechanically check "does the current graph satisfy invariant N?" because invariant N does not exist as a numbered, checkable statement.

---

## Goal

Design docs that support **precise reasoning during debugging**, not just understanding during implementation. An LLM agent should be able to:

1. Read a procedure doc and form a correct, precise mental model (not just a vague one)
2. During debugging, iterate through numbered, checkable invariants and determine which one is violated
3. Follow a cross-reference from a broken invariant to the narrative concept it expresses, and from there to the stage that should have produced the correct state

---

## Scope

All design docs are in scope. Two tracks of work with different goals.

### Track 1: Narrative Doc Expansion

**Files:** `how-branching-stories-work.md`, `story-graph-ontology.md`

**Approach:** Interactive collaborative session — author and LLM work through each concept together. The author confirms what is actually underspecified from a narrative standpoint; the LLM flags where the LLM mental model would diverge from the intended meaning. Neither party works alone — the author finds the docs clear and cannot see the LLM blind spots; the LLM cannot judge which expansions are narratively correct.

**What gets added** (identified during the interactive session):

- **Concept boundary sharpening** — where two concepts are defined correctly but their boundary is fuzzy, add an explicit "not to be confused with" statement inline. The Part 8 "danger zones" from the ontology must also appear inline at the concept they govern — an agent reading the `belongs_to` edge definition must see the constraint immediately, not find it in a separate section.

- **Negative examples** — for each concept where LLMs are known to go wrong, add a concrete invalid graph state with an explanation of why it is invalid. Currently the docs describe correct states almost exclusively.

- **Stage attribution completion** — where a concept is created by one stage and consumed by another, both sides must be explicit. If a working annotation is "consumed by GROW," the doc must also say what state is invalid if GROW runs and that annotation still exists (or is missing).

### Track 2: Procedure Doc Rewrites

**Files:** `procedures/dream.md`, `procedures/brainstorm.md`, `procedures/seed.md`, `procedures/grow.md`, `procedures/polish.md`, `procedures/fill.md`, `procedures/dress.md`, `procedures/ship.md`

**Approach:** Full rewrite. Priority order: SEED → GROW → POLISH (bugs in earlier stages cascade into later ones). Remaining stages follow.

---

## Structure of Rewritten Procedure Docs

Each procedure doc follows this template exactly. The nesting reflects that each stage contains phases, and each phase is a mini-stage with its own contracts.

```
# [STAGE] — [one-line purpose]

## Overview
[2–3 sentences: inputs, outputs, what this stage is responsible for
and explicitly NOT responsible for]

## Stage Input Contract
[Numbered list. Each item is a checkable boolean condition that must be
true in the graph before this stage runs. If any item fails, the bug
is in the preceding stage, not this one.]

1. [condition]
2. [condition]

---

## Phase N: [Phase Name]

**Purpose:** [1 sentence]

### Input Contract
[What phase N requires — from the graph or from Phase N-1's output.
Numbered. Checkable.]

### Operations

#### [Operation Name]

**What:** [1–2 sentences: what this does and why it exists — the
minimum narrative context needed to understand the rule]

**Rules:**
R-N.1. [Checkable invariant. Boolean condition, stated precisely.
        "Every X must have exactly one Y" not "X should have a Y".]
R-N.2. [...]

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| [observable wrong state in the graph] | [why this state is wrong] | R-N.1 |

### Output Contract
[What must be true after Phase N completes. These become Phase N+1's
Input Contract. Numbered. Checkable.]

---

## Stage Output Contract
[What must be true after ALL phases complete. These become the next
stage's Stage Input Contract.]

## Implementation Constraints
[CLAUDE.md principles that apply to this stage's implementation.
Stated as rules here, with a pointer to CLAUDE.md for rationale.]

## Cross-References
- [concept] → how-branching-stories-work.md §[section]
- [graph model] → story-graph-ontology.md Part N
- [next stage depends on] → [next-stage].md §Stage Input Contract
- [CLAUDE.md] → CLAUDE.md §[heading]

## Rule Index
[Flat list of every rule in this doc, for fast scanning during debugging.]
R-1.1: [one-line statement]
R-1.2: [one-line statement]
R-3.1: [one-line statement]
...
```

### Key additions vs. current procedure docs

**Input and output contracts** — current docs describe what stages do, not what they guarantee. These contracts let a debugging agent pinpoint which stage produced a broken state: check the stage output contract, then walk backwards through phase output contracts.

**Contract chaining invariant:** Stage N's Stage Input Contract must be identical to Stage N-1's Stage Output Contract. If they diverge, one of the two docs is wrong. During the rewrite, each pair of adjacent stages is checked for contract alignment before either doc is considered complete. A stage's Input Contract is not written independently — it is copied from the preceding stage's Output Contract and then confirmed correct.

**Violations table** — current docs describe correct behavior almost exclusively. Wrong behavior appears in passing. The violations table is the primary debugging tool: given an observed symptom, find the broken rule in one step.

**Rule index** — current docs bury invariants in prose. The rule index gives a flat, scannable list of every checkable condition in the doc, numbered for cross-reference.

**Implementation constraints** — rules from CLAUDE.md that govern this stage's implementation. Prevents agents from needing to cross-read CLAUDE.md during stage implementation or debugging. Specifically: Valid ID Injection Principle, Context Enrichment Principle, Prompt Context Formatting rules, Silent Degradation policy (pipeline must fail loudly — never silently skip a structural constraint).

---

## Cross-Referencing System

Every cross-reference in every doc uses the format:

```
→ [doc-shortname] §[exact heading text]
```

Examples:
- `→ ontology §Part 8: Path Membership ≠ Scene Participation`
- `→ how-branching §Part 2: Commitments and Structure`
- `→ grow.md §Phase 3: Intersection Detection`
- `→ CLAUDE.md §Valid ID Injection Principle`

**Inline placement rule:** cross-references appear at the point of use, not in a "see also" section at the end. If rule R-3.2 depends on a concept defined in the ontology, the cross-reference is on R-3.2's line, not in the Cross-References section at the bottom.

The Cross-References section at the bottom of each doc is a complete index of all outgoing links — useful for navigating, but not a replacement for inline placement.

---

## What Does Not Change

The narrative content of `how-branching-stories-work.md` and `story-graph-ontology.md` is correct. The existing procedure doc content (phases, operations, human gates, worked examples) is carried forward and reorganized into the new structure — nothing is thrown away. The rewrite is a restructuring and augmentation, not a replacement of knowledge.

---

## Success Criteria

A rewrite is complete when:

1. An LLM agent can read a procedure doc, identify every invariant as a numbered rule, and check each one against a graph state without re-reading narrative prose
2. Given a broken graph state, the agent can find the violated rule in the rule index and trace it to the phase that should have prevented it
3. The violations table for each operation contains at least one concrete wrong-state example
4. Every cross-reference resolves to an exact heading in the target doc
5. Every CLAUDE.md implementation principle relevant to a stage appears in that stage's Implementation Constraints section
