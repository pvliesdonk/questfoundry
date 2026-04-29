# DREAM — Establish the creative vision

## Overview

DREAM captures the creative contract that governs every downstream decision: genre, subgenre, tone, themes, audience, scope, content notes, and an optional point-of-view hint. It produces a single Vision node with no edges. DREAM does not create entities, dilemmas, beats, or any graph structure — that begins in BRAINSTORM.

## Stage Input Contract

1. The graph is empty (no nodes, no edges).
2. The human has an initial creative spark (informal idea, may be vague).

---

## Phase 1: Vision Capture

**Purpose:** Explore the creative concept through dialogue, define its boundaries, and serialize it into a validated Vision node.

### Input Contract

1. Graph is empty.
2. Human-provided creative spark is available as conversational input.

### Operations

#### Spark Exploration

**What:** The LLM helps the human articulate the creative vision through open-ended dialogue. Genre, subgenre, tone, themes, audience, and the emotional register of the story are surfaced. The output is a prose understanding — no graph mutation yet.

**Rules:**

R-1.1. The discussion produces a single coherent vision, not a menu of options. When the human is uncertain, the LLM may suggest alternatives — but a decision is required before proceeding to the next operation.

R-1.2. Genre and subgenre are distinct fields. "Mystery" is a genre; "cozy mystery" is a subgenre. Do not conflate them into one field.

R-1.3. Themes are abstract ideas the story explores ("the price of loyalty"), not plot points ("the mentor dies in chapter three"). Plot belongs in BRAINSTORM and later.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Vision has `genre: "cozy mystery"` and no `subgenre` | Genre and subgenre conflated into one field | R-1.2 |
| Vision has `themes: ["the mentor betrays the protagonist"]` | Plot point written as theme — themes must be abstract | R-1.3 |
| Discussion ends without a committed genre ("maybe fantasy, maybe sci-fi") | Exploration never converged on a decision | R-1.1 |

#### Constraint Definition

**What:** The human defines the boundaries of the story — content to include or avoid, structural limits (single POV, linear timeline), and scope (how big this story will be). Constraints enable creativity by defining the sandbox; they are firm commitments, not preferences.

**Rules:**

R-1.4. Scope is a named preset (e.g., `micro`, `short`, `medium`, `long`, or their equivalent implementation values), not a raw beat or passage count. The preset implies approximate sizes for cast, dilemma count, beat count, and passage count.

R-1.5. Content notes define creative direction — what to embrace and what to avoid. They are substantive constraints on BRAINSTORM's output, not after-the-fact filters. "Avoid graphic violence" belongs here; "redact expletives from prose" does not.

R-1.6. Constraints are firm decisions. Reopening them requires looping back to an earlier operation with explicit human approval.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Vision has `scope: 15` (raw number) | Scope must be a named preset, not a count | R-1.4 |
| Content note: "remove any swearing from generated prose" | Content notes shape creative direction; prose scrubbing is a FILL concern | R-1.5 |
| BRAINSTORM surfaces a dilemma that contradicts a declared content note, and the pipeline proceeds | Constraint silently softened instead of surfacing conflict to human | R-1.6 |

#### Vision Synthesis

**What:** The discussed vision is serialized into a Vision node in the graph. This is the only graph write DREAM performs. The node carries all required fields; its `pov_style` is advisory only.

**Rules:**

R-1.7. Exactly one Vision node is created. If synthesis is retried, the previous node is replaced, not duplicated.

R-1.8. All required fields — `genre`, `tone`, `themes`, `audience`, `scope` — are non-empty. `subgenre`, `content_notes`, and `pov_style` are optional but recommended.

R-1.9. `pov_style`, when present, is one of: `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`. It is advisory — FILL makes the final decision.

R-1.10. No graph edges are created. The Vision node has no incoming or outgoing edges; it is retrieved by node-type lookup.

R-1.11. Synthesis captures only what was discussed. The LLM does not invent themes, constraints, or fields that did not surface in the conversation.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Two Vision nodes exist after DREAM | Retry created a second node without removing the first | R-1.7 |
| Vision has empty `themes: []` | Required field left empty — synthesis did not capture a necessary decision | R-1.8 |
| Vision has `pov_style: "omniscient"` | Value outside the permitted set | R-1.9 |
| Synthesis adds a theme ("redemption") that was never discussed | LLM invented content to fill a perceived gap | R-1.11 |
| Downstream stage treats `pov_style` as a hard constraint instead of a starting point | `pov_style` on Vision is advisory, not binding | R-1.9 |

#### Approval

**What:** The human reviews the serialized Vision and either approves it (DREAM completes) or rejects it (loop back to the operation with the misalignment). Approval is an explicit step, not a default.

**Rules:**

R-1.12. DREAM is not complete until the human explicitly approves the Vision node. Silent acceptance (no response) is not approval.

R-1.13. Rejection loops back to the operation that contains the misalignment — spark exploration for vision gaps, constraint definition for boundary issues, synthesis for field errors. It does not silently self-correct.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Pipeline proceeds to BRAINSTORM without recorded human approval | Approval step skipped or implicit | R-1.12 |
| Synthesis retried after rejection without re-exploring the rejected concern | Loop-back operation not identified; fix applied to the wrong layer | R-1.13 |

### Output Contract

1. Exactly one Vision node exists.
2. `genre`, `tone`, `themes`, `audience`, `scope` are all non-empty.
3. `pov_style`, if present, is one of four permitted values.
4. No edges exist in the graph.
5. Human approval is recorded.

---

## Stage Output Contract

1. Exactly one Vision node exists in the graph.
2. The Vision node has non-empty values for: `genre`, `tone`, `themes`, `audience`, `scope`.
3. The Vision node has no incoming or outgoing edges.
4. `pov_style`, if present, is one of `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`; absent or null if deferred to FILL.
5. No other node types exist in the graph.
6. Human approval is recorded.

## Implementation Constraints

- **Prompt Context Formatting:** Any prompt that references Vision fields must format them as human-readable text. Never interpolate raw Python dicts, lists, or enum reprs. → CLAUDE.md §Prompt Context Formatting (CRITICAL)
- **Small Model Prompt Bias:** DREAM's discussion runs on small models during local dev. Write exploration prompts with explicit structure, concrete examples, and clear delimiters. Do not blame the model for vague vision output; fix the prompt first. → CLAUDE.md §Small Model Prompt Bias (CRITICAL)

## Cross-References

- Genre, subgenre, tone, themes, scope narrative meaning → how-branching-stories-work.md §The Vision (DREAM)
- Vision node schema and field semantics → story-graph-ontology.md Part 1: Vision
- Next stage consumes Vision → brainstorm.md §Stage Input Contract

## Rule Index

R-1.1: Discussion produces a single coherent vision, not a menu of options.
R-1.2: Genre and subgenre are distinct fields.
R-1.3: Themes are abstract ideas, not plot points.
R-1.4: Scope is a named preset, not a raw count.
R-1.5: Content notes shape creative direction; they are not after-the-fact filters.
R-1.6: Constraints are firm — reopening them requires explicit approval.
R-1.7: Exactly one Vision node exists; retries replace rather than duplicate.
R-1.8: Required fields (genre, tone, themes, audience, scope) are non-empty.
R-1.9: `pov_style` is one of four permitted values; advisory only.
R-1.10: Vision node has no graph edges.
R-1.11: Synthesis captures only what was discussed; no LLM invention.
R-1.12: DREAM is not complete until human explicitly approves.
R-1.13: Rejection loops back to the operation that contains the misalignment.

---

## Human Gates

| Operation | Gate | Decision |
|-----------|------|----------|
| Spark Exploration | Light — "vision explored enough?" | Continue or probe further |
| Constraint Definition | Required | Approve boundaries |
| Vision Synthesis | Required | Review draft Vision node |
| Approval | Required | Final sign-off |

## Iteration Control

**Forward flow:** Spark Exploration → Constraint Definition → Vision Synthesis → Approval.

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| Constraint Definition | Spark Exploration | Constraints reveal vision gaps |
| Vision Synthesis | Spark Exploration | Synthesis shows underdeveloped areas |
| Vision Synthesis | Constraint Definition | Missing or unclear constraints |
| Approval | Any prior operation | Human requests changes (see R-1.13) |

**Maximum iterations:**

- Spark Exploration: no fixed limit; human decides when "enough."
- Constraint Definition: at most 2 revision passes.
- Vision Synthesis: at most 2 synthesis attempts per approval cycle.
- Approval: single pass — approve or loop back.

## Failure Modes

| Operation | Failure | Detection | Recovery |
|-----------|---------|-----------|----------|
| Spark Exploration | Vision too vague | Human cannot answer basic genre/tone questions | More exploration, try different angles |
| Spark Exploration | Vision too specific | Human already has full plot | Fast-track to Synthesis; note detail for BRAINSTORM |
| Constraint Definition | Constraints too restrictive | No interesting stories fit within bounds | Relax; discuss tradeoffs |
| Constraint Definition | Constraints too loose | No clear boundaries | Push for specificity |
| Vision Synthesis | LLM invents content | Vision contains undiscussed fields (R-1.11) | Strip inventions; re-synthesize |
| Vision Synthesis | LLM loses nuance | Vision oversimplifies the discussion | Human edits directly or re-discusses |

**Escalation.** DREAM has no prior stage. If DREAM cannot complete, the initial spark may be unsuitable for interactive fiction — the human should reconsider the concept.

## Context Management

**Standard (≥128k context):** Include full discussion history for synthesis. No windowing.

**Constrained (~32k context):** Human curates key points from discussion; pass curated summary to Synthesis. Trust the human curation to preserve important decisions.

## Worked Example

### Initial Spark

**Human:** "I want to write a story about an archivist who discovers forbidden knowledge. Dark fantasy, mysterious."

### Spark Exploration

**LLM:** "An archivist with forbidden knowledge — that's rich territory. What draws you to this concept? Is it the dilemma between preservation and destruction? The corruption of knowledge? The isolation of knowing something others don't?"

**Human:** "The corruption angle. And trust — who can you trust when you know something dangerous?"

**LLM:** "Trust is powerful. So we have themes of forbidden knowledge and trust. What's the tone? You said dark fantasy — are we talking grimdark, or more atmospheric and melancholic?"

**Human:** "Atmospheric. Morally ambiguous rather than grimdark. Think of it like a mystery where you're not sure who to believe."

**LLM:** "Atmospheric, morally ambiguous, mystery elements within dark fantasy. What's the scope? Epic quest or intimate character study?"

**Human:** "Intimate. Focused. Maybe just the archivist and a few key relationships."

### Constraint Definition

**LLM:** "For constraints: single-protagonist POV makes sense for intimate focus. Any content boundaries?"

**Human:** "No explicit magic system — keep it mysterious. And no graphic violence."

**LLM:** "Scope? Given the intimate focus, short or medium?"

**Human:** "Short. I want it tight and focused."

### Vision Synthesis

```yaml
vision:
  genre: "dark fantasy"
  subgenre: "mystery"
  tone:
    - "atmospheric"
    - "morally ambiguous"
  themes:
    - "forbidden knowledge"
    - "trust"
    - "corruption"
  audience: "adult readers of literary speculative fiction"
  scope:
    story_size: short              # enum value — unquoted; one of micro/short/medium/long
  content_notes:
    includes:
      - "intimate scope (few key relationships)"
    excludes:
      - "explicit magic system"
      - "graphic violence"
  pov_style: third_person_limited  # enum value — unquoted
```

### Approval

**Human:** "Yes, this captures it."

DREAM complete. Vision node written to graph.
