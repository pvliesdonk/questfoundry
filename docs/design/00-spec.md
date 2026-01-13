# QuestFoundry v5 — Unified Specification

**Version:** 2.0
**Status:** Working draft
**Supersedes:** qf-v5-vision.md, qf-v5-vision-delta001.md, questfoundry-v5-graph-ontology-v1-1.md

---

## Executive Summary

QuestFoundry v5 is a lightweight system for LLM-generated interactive fiction. A human author with a spark of an idea works through a structured pipeline with LLMs and walks away with a playable branching story.

**Core philosophy:**
- LLM as collaborator under constraint, not autonomous agent
- Every creative expansion followed by human curation
- All artifacts are files (version-controllable, diffable, human-editable)
- If your AI coding agent cannot understand the entire system, the system is too complex

---

## Why the Previous Approach Failed

### Failure Mode 1: Agentic Agency

Agents had too much freedom. They could propose hooks, negotiate with each other, and iterate indefinitely. This produced creative output but was impossible to debug, test, or reason about.

### Failure Mode 2: Agent Coordination

Multiple agents operating concurrently with shared mutable state. Race conditions, conflicting proposals, and state synchronization bugs dominated development time.

### Failure Mode 3: Incremental Plot/Prose with Hook Generation

The system built plot and prose incrementally while generating new hooks along the way. Great for emergent creativity, catastrophic for LLM context management. State degraded over long sessions; characters acted inconsistently; plot holes emerged.

### Failure Mode 4: Tooling Complexity

The complexity of the tooling exceeded what an AI coding agent could hold in working memory. Agents fought symptoms without understanding root causes. Every fix introduced new bugs.

### The Core Lesson

**If your AI coding agent cannot understand the entire system, the system is too complex.**

---

## Pipeline Overview

```
DREAM → BRAINSTORM → SEED → GROW → FILL → DRESS → SHIP
```

| Stage | Purpose | Mode |
|-------|---------|------|
| DREAM | Genre, tone, themes, constraints | Human + LLM |
| BRAINSTORM | Entities, tensions, alternatives | LLM-heavy (discuss → summarize → serialize) |
| SEED | Triage: curate entities, promote tensions to threads | Human-heavy |
| GROW | Mutate graph until complete: knots, weaving, passages | Layered, human gates |
| FILL | Generate prose for passages | LLM, sequential |
| DRESS | Illustrations, codex | LLM, optional |
| SHIP | Export to ink/Twee/epub | Deterministic, no LLM |

### Key Constraints

1. **No persistent agent state between sessions.** Each session starts fresh.
2. **All artifacts are files.** JSON, YAML, or Markdown. Version-controllable, diffable, human-editable.
3. **Threads are frozen after SEED.** GROW can mutate beats, create knots, weave arcs—but cannot create new threads.
4. **Human gates between stages.** The human reviews and can edit before proceeding.
5. **Prompts are visible artifacts.** No hidden prompt engineering.

### Design Principle: LLM Prepares, Human Decides

Complex graphs are unwieldy for humans to navigate. The LLM's job is to **identify opportunities and surface decisions**.

**LLM should:**
- Surface all decision points proactively
- Rank options where meaningful
- Propose sensible defaults
- Explain trade-offs briefly

**Human should not need to:**
- Manually scan the graph for opportunities
- Remember all constraints while deciding
- Construct fixes from scratch

This keeps human cognitive load manageable while preserving authorial control.

---

## Graph Ontology

### The Unified Graph

There is **one graph**. The story is the graph. Stages are **graph mutations**.

This is not a pipeline of files where each stage produces a new artifact. Rather:
- All story data lives in a single graph structure
- Each stage reads relevant parts of the graph and writes updates
- The runtime maintains graph consistency
- Storage format is an implementation detail (JSON, YAML, SQLite—doesn't matter)

**Why this matters:**
- No data duplication between "artifacts"
- Clear ownership: each node type has a creating stage
- Validation operates on the whole graph
- Human review shows what changed, not complete file contents

### Stage Operations

| Stage | Creates | Modifies | Reads |
|-------|---------|----------|-------|
| DREAM | Vision metadata | — | — |
| BRAINSTORM | Entity, Tension, Alternative | — | Vision |
| SEED | Thread, Consequence, Beat | Entity (curate), Tension (explore) | Entity, Tension |
| GROW | Arc, Passage, Choice, Codeword; new Beats | Beat (scene_type, knot) | Thread, Beat, Entity |
| FILL | — | Passage (prose), Entity (details) | Passage, Entity, Thread |
| DRESS | Illustration, Codex | — | Passage, Entity |
| SHIP | — (export only) | — | Persistent nodes |

### LLM Output vs Graph Storage

Stages use LLMs to produce structured output. This output is **not** the graph format—it's stage-specific data that the runtime interprets and applies as graph mutations.

```
LLM Output (stage-specific)
    ↓ validate
    ↓ interpret
Runtime applies mutations
    ↓
Graph (unified storage)
```

**Example:** SEED produces entity decisions, thread definitions, and initial beats. The runtime:
1. Validates the output structure
2. Updates existing entity nodes (marking dispositions)
3. Creates new thread, consequence, and beat nodes
4. Creates edges between nodes
5. Validates graph consistency
6. Persists to storage

The LLM doesn't need to know the graph storage format. It produces what makes sense for its task.

### Graph Storage

Storage is a runtime implementation detail. The graph can be stored as:
- Single JSON file (simple, atomic)
- Partitioned files (organized by concern)
- Database (for complex queries)

**What the spec defines:** Node types, edge types, constraints, lifecycle.
**What the runtime decides:** Storage format, serialization, caching.

For human review, the runtime exports readable views (YAML, Markdown) on demand.

### Rollback and Snapshots

Before each stage executes, the runtime snapshots the current graph state. If a stage fails or is rejected at human review, the graph restores to the pre-stage snapshot.

This provides stage-level rollback without complex event sourcing.

---

## Node Types

### Metadata Node

#### Vision

Creative direction established in DREAM. Not a graph node in the traditional sense—more like graph-level configuration.

```yaml
vision:
  genre: string
  tone: string[]
  themes: string[]
  audience: string
  scope:
    length: micro | short | medium | long
    branches: minimal | moderate | extensive
  content_notes: string[]
```

**Lifecycle:** Created in DREAM, read by all subsequent stages for context. Not exported.

---

### Persistent Nodes (survive to export)

#### Passage

Player-facing unit. The prose the player reads.

```yaml
passage:
  id: string
  from_beat: beat_id | null    # traceability (ignored by SHIP)
  summary: string | null       # pre-FILL (ignored by SHIP)
  prose: string | null         # post-FILL (required for SHIP)
```

A passage is complete when `prose` exists.

**Lifecycle:** Created in GROW (1:1 from beats), prose added in FILL. Exported.

**Structural properties (derived):**
- Start passage: no incoming choice edges
- Ending passage: no outgoing choice edges

#### Entity

Character, location, object, or other story element.

```yaml
entity:
  id: string
  type: character | location | object | faction | ...
  base:
    key: value
  overlays:
    - when: codeword[]
      details:
        key: value
```

**Lifecycle:** Created in BRAINSTORM, curated in SEED, details added in FILL. Exported.

**Creation constraint:** Entities can only be created in BRAINSTORM or SEED. Neither GROW nor FILL can create Entity nodes. Minor characters that appear in a single scene are prose detail, not entities.

**Update constraint:** FILL can update existing entities with micro-details discovered during prose generation (e.g., physical appearance, mannerisms). These updates provide consistency for later passages. FILL cannot create new entities.

If GROW or FILL discovers need for a new recurring entity → abort to SEED, add entity, re-run from there.

**State resolution:**
```
active_codewords = all codewords granted on path to current passage
applicable_overlays = overlays where when ⊆ active_codewords
entity_state = base + merge(applicable_overlays)
```

**Overlay conflict resolution:**
1. Most specific wins (overlay with most matching codewords in `when`)
2. If tie, later in list wins
3. Validation flags potential conflicts for human review

Example:
```yaml
overlays:
  - when: [mentor_trusted]
    details: { eyes: "kind" }
  - when: [mentor_ill]
    details: { eyes: "hollow" }
  - when: [mentor_trusted, mentor_ill]    # most specific, wins if both active
    details: { eyes: "kind but tired" }
```

#### Codeword

State marker. Universal mechanism for both plot progression and player choice.

```yaml
codeword:
  id: string
  type: granted  # 'derived' deferred to future version
  tracks: consequence_id | null    # what narrative consequence this represents
  condition: string | null         # for derived: e.g., "gold > 50"
```

**Lifecycle:** Created in GROW (derived from beat grants and choice edges). Exported.

The `tracks` field links a codeword to its originating consequence from SEED. This makes codewords traceable to their narrative purpose. When GROW creates `mentor_protector_committed`, it links back to the `mentor_ally` consequence.

#### Relationship

Connection between two entities with conditional state.

```yaml
relationship:
  id: string
  from_entity: entity_id
  to_entity: entity_id
  base:
    nature: string
    key: value
  overlays:
    - when: codeword[]
      details:
        key: value
```

**Lifecycle:** Created in BRAINSTORM or SEED, updated in FILL. Exported.

#### Illustration

Art asset with caption.

```yaml
illustration:
  id: string
  asset: path
  caption: string
```

**Lifecycle:** Created in DRESS. Exported.

#### Codex

Player-facing encyclopedia entries for entities. Provides in-world information without spoilers.

```yaml
codex_entry:
  id: string
  entity_id: string               # which entity this describes
  visible_when: codeword[]        # player must have these to see entry
  content: string                 # player-safe, no spoilers
```

**Lifecycle:** Created in DRESS. Exported.

---

### Creative Nodes (created in BRAINSTORM, refined in SEED)

#### Tension

The dramatic question that drives meaningful choice. Every interesting story choice implies a road not taken.

```yaml
tension:
  id: string
  question: string              # "Can the mentor be trusted?"
  alternatives:
    - id: string
      description: string       # "Mentor is genuine protector"
      canonical: true           # used for spine arc
    - id: string
      description: string       # "Mentor is manipulating Kay"
      canonical: false          # alternate arc if explored
  involves: entity_id[]
  why_it_matters: string        # thematic stakes
  # Added by SEED:
  explored: alternative_id[]    # which alternatives become threads
```

**Lifecycle:** Created in BRAINSTORM, exploration decisions added in SEED. Not exported.

**Binary constraint:** Exactly two alternatives per tension. This keeps contrasts crisp.

**Canonical flag:** One alternative is marked `canonical: true`. This is the "default" story—used for the spine arc. The non-canonical alternative becomes a branch if promoted to a thread in SEED.

For nuanced situations, use multiple tensions on the same concept:
```yaml
tensions:
  - id: mentor_alignment
    question: "Is the mentor benevolent or self-serving?"
    alternatives:
      - id: mentor_benevolent
      - id: mentor_selfish

  - id: mentor_competence
    question: "Is the mentor capable or flawed?"
    alternatives:
      - id: mentor_capable
      - id: mentor_flawed
```

This yields four combinations (benevolent+capable, benevolent+flawed, etc.) while each tension remains a clear binary contrast.

**Key insight:** "Mentor is a protector" is flat. "Mentor is a protector (not the manipulator they could have been)" has tension—even if we never write the manipulator thread.

#### Alternative

One possible answer to a Tension's question. Extracted as separate nodes in the graph to enable thread/alternative relationships.

```yaml
alternative:
  id: string
  description: string             # "Mentor is genuine protector"
  canonical: bool                 # true = used for spine arc
  tension_id: string              # parent tension
```

**Lifecycle:** Created in BRAINSTORM as part of tension generation. Not exported.

---

### Working Nodes (consumed by GROW, ignored by SHIP)

#### Consequence

Narrative meaning of a thread choice. Bridges the gap between "what this path represents" (alternative) and "how we track it" (codeword).

```yaml
consequence:
  id: string
  thread_id: thread_id              # which thread this belongs to
  description: string               # "Mentor becomes protective ally"
  ripples: string[]                 # story effects this implies
    # - "Shields Kay in confrontation"
    # - "Reveals family connection"
```

**Lifecycle:** Created in SEED when threads are created. Not exported.

GROW creates codewords to track when consequences become active, and creates entity overlays to implement consequence effects.

#### Plot Thread

One explored alternative from a tension. Threads from the same tension are automatically exclusive.

```yaml
thread:
  id: string
  name: string
  tension_id: tension_id
  alternative_id: alternative_id    # which alternative this explores
  shadows: alternative_id[]         # unexplored alternatives (context for FILL)
  tier: major | minor
  description: string
  consequences: consequence_id[]    # narrative meaning of this path
```

**Lifecycle:** Created in SEED. Not exported. (THREAD FREEZE: no new threads after SEED)

**Tier:**
- **Major:** Defines the story. Must interweave with other major threads.
- **Minor:** Supports/enriches. Must touch the story but can be more independent.

**Exclusivity is derived:** All threads sharing a `tension_id` are automatically exclusive. No manual declaration needed.

#### Beat

Narrative unit. Belongs to one or more threads.

```yaml
beat:
  id: string
  summary: string
  scene_type: scene | sequel | micro_beat    # pacing structure (assigned in GROW)
  threads: thread_id[]          # usually one, multiple = knot
  tension_impacts:
    - tension_id: tension_id
      effect: advances | reveals | commits | complicates
      note: string              # "Player sees mentor's private communication"
  requires: beat_id[]           # ordering constraints
  grants: codeword_id[]
  entities: entity_id[]
  relationships: relationship_id[]
  location: entity_id | null              # primary location (assigned in SEED)
  location_alternatives: entity_id[]      # other valid locations (enables knot flexibility)
```

**Lifecycle:** Initial beats created in SEED, mutated and new beats added in GROW. Not exported.

**Location flexibility:** Beats can specify alternative locations where the same dramatic action could occur. If Beat A (at Market) and Beat B (at Docks) both have `location_alternatives` including each other's location, GROW can merge them by choosing a shared setting. This enables knot formation without constraining BRAINSTORM's creative freedom.

**Scene types:**

| Type | Purpose | Prose guidance |
|------|---------|----------------|
| `scene` | Active pursuit: goal → obstacle → outcome | Full dramatic structure, 3+ paragraphs |
| `sequel` | Reactive processing: reaction → dilemma → decision | Breathing room after disaster, 2-3 paragraphs |
| `micro_beat` | Transition, time passage, minor moment | Brief, 1 paragraph |

Scene type is assigned during GROW (Phase 4: Gap Detection) to ensure pacing variety across arcs. GROW may propose additional beats to address pacing gaps (e.g., "three scenes in a row with no sequel").

**Beat types by thread membership:**
- **Single-thread:** Serves one thread's progression
- **Knot:** Serves multiple threads (natural intersection point)

**Tension impact effects:**

| Effect | Meaning |
|--------|---------|
| `advances` | Moves toward resolution without revealing answer |
| `reveals` | Surfaces information bearing on the question |
| `commits` | Point of no return—alternative is now locked in |
| `complicates` | Introduces doubt, new dimension to tension |

#### Arc

Realized weaving of compatible threads.

```yaml
arc:
  id: string
  type: spine | branch
  threads: thread_id[]          # must be from different tensions (compatible)
  sequence: beat_id[]           # ordered result of weaving
  parent: arc_id | null
  diverges_at: beat_id | null
  converges_at: beat_id | null
```

**Lifecycle:** Created in GROW during arc enumeration. Not exported.

---

## Edge Types

> **Naming Convention:** Persistent edges use PascalCase (Choice, Appears) as they appear
> in exports. Working edges use snake_case (belongs_to, has_alternative) as they're internal only.

### Persistent Edges (survive to export)

| Edge | From → To | Properties | Created In | Purpose |
|------|-----------|------------|------------|---------|
| **Choice** | passage → passage | label, requires[], grants[], modifies{} | GROW | Player navigation |
| **Appears** | entity → passage | role | GROW | Entity present in scene |
| **Involves** | relationship → passage | — | GROW | Relationship active in scene |
| **Depicts** | illustration → passage | — | DRESS | Art shown with passage |

**Choice properties:**
```yaml
choice:
  label: string                 # always diegetic ("Wait for nightfall...", never "Continue")
  requires: codeword[]          # gate
  grants: codeword[]            # state change
  modifies:                     # future: numeric state
    state_key: delta
```

### Working Edges (consumed by GROW, not exported)

| Edge | From → To | Created In | Purpose |
|------|-----------|------------|---------|
| **belongs_to** | beat → thread | SEED | Beat serves this thread |
| **involves** | tension → entity | BRAINSTORM | Tension involves these entities |
| **has_alternative** | tension → alternative | BRAINSTORM | Tension's possible answers |
| **explores** | thread → alternative | SEED | Thread explores this alternative |
| **has_consequence** | thread → consequence | SEED | Thread's narrative consequences |
| **requires** | beat → beat | SEED, GROW | Ordering constraint |
| **grants** | beat → codeword | GROW | Beat completion grants codeword |
| **weaves** | arc → thread | GROW | Arc uses this thread |
| **from_beat** | passage → beat | GROW | Traceability |

---

## State Pattern

Entities and relationships share the same conditional state model:

```yaml
base: { ... }
overlays:
  - when: codeword[]
    details: { ... }
```

- `base` = default state
- `overlays` = applied when codewords are active
- Overlays add/refine details

**Conflict resolution order:**
1. Most specific overlay wins (most matching codewords in `when`)
2. If tie, later in list wins
3. If potential conflict detected at validation, human is warned

**Why conflicts can occur:**

Thread exclusivity (from shared tension) prevents some conflicts automatically—you can't have codewords from exclusive threads. But codewords from *different* tensions can coexist, and their overlays might conflict.

Example: `mentor_trusted` (from alignment tension) and `mentor_ill` (from health tension) can both be active. If both set the same key, resolution rules apply.

**Validation:** For each entity, compute all valid codeword combinations. Flag any overlay pairs that (a) can both be active AND (b) set same key to different values.

---

## Stage Specifications

> **Note on Output Schemas:** The YAML schemas shown below represent **LLM output format**,
> not storage format. The runtime interprets these and applies graph mutations accordingly.
> See "LLM Output vs Graph Storage" in the Graph Ontology section.

### Stage 1: DREAM

**Purpose:** Riff on genre, tone, themes, constraints.

**Input:** Human's initial spark ("I want a noir mystery in a space station")

**Output:**
```yaml
dream:
  genre: string
  tone: string
  themes: string[]
  constraints: string[]
  length: micro | short | medium | long
```

**Human Gate:** Approve or refine vision.

---

### Stage 2: BRAINSTORM

**Purpose:** Expansive exploration of story possibilities.

**Process:** Discuss (high temperature) → Summarize (consolidate) → Serialize (low temperature)

**Input:** Approved dream.

**Output:**
```yaml
brainstorm:
  entities:
    - id: string
      type: character | location | object | faction
      concept: string           # one-line essence
      notes: string             # freeform, from discussion

  tensions:
    - id: string
      question: string          # "Can the mentor be trusted?"
      alternatives:
        - id: string
          description: string
          canonical: true       # default story path
        - id: string
          description: string
          canonical: false      # becomes branch if explored
      involves: entity_id[]
      why_it_matters: string
```

BRAINSTORM generates freely without worrying about thread collision. Location flexibility for knot formation is handled in SEED.

**Human Gate:** Review brainstorm output before triage.

---

### Stage 3: SEED

**Purpose:** Triage brainstorm into committed structure. **Thread creation gate.**

**Input:** Approved brainstorm.

**Operations:**
1. Curate entities (in/out)
2. For each tension: decide which alternatives to explore
   - Canonical alternative always becomes a thread (spine path)
   - Non-canonical alternative becomes a thread only if exploring that branch
3. Explored alternatives become threads (with exclusivity inherited from shared tension)
4. Create initial beats per thread

**Output:**
```yaml
seed:
  entities:
    - id: string
      type: character | location | object | faction
      concept: string
      # full entity structure created here

  tensions:
    - tension_id: string
      explored: alternative_id[]      # always includes canonical; may include non-canonical
      implicit: alternative_id[]      # non-explored alternatives (context for FILL)

  threads:
    - id: string
      name: string
      tension_id: tension_id
      alternative_id: alternative_id
      shadows: alternative_id[]
      tier: major | minor
      description: string
      consequences: consequence_id[]

  consequences:
    - id: string
      thread_id: thread_id
      description: string             # "Mentor becomes protective ally"
      ripples: string[]               # story effects this implies

  initial_beats:
    - id: string
      summary: string
      threads: thread_id[]
      tension_impacts:
        - tension_id: tension_id
          effect: advances | reveals | commits | complicates
          note: string
      entities: entity_id[]
      location: entity_id | null            # primary location
      location_alternatives: entity_id[]    # other valid locations (enables knot flexibility)

  convergence_sketch:                 # informal guidance for GROW
    convergence_points: string[]      # "threads should merge by act 2 climax"
    residue_notes: string[]           # "mentor demeanor differs after convergence"
```

**Human Gate:** Approve seed. After this point, no new threads can be created.

**Critical constraint:** THREAD FREEZE. GROW cannot create threads. All branching potential is declared here.

---

### Stage 4: GROW

**Purpose:** Generate the complete story topology through graph mutation.

GROW operates on the graph until completion criteria are met. It can mutate beats, create knots, weave arcs, derive passages and choice edges—but cannot create new threads.

#### Initial State (from SEED)

- Threads with initial beats (single-thread, loose)
- Exclusivity derived from shared tension_id
- Internal ordering (`requires`) declared
- Thread tiers declared (major/minor)
- Core entities created
- Location flexibility annotated on beats (enables knot detection)

#### Knot Operations

Knots are beats serving multiple threads. Three operations:

| Operation | Description | Example |
|-----------|-------------|---------|
| **Mark** | Existing beat serves multiple threads | `investigate` also serves GADGETS thread |
| **Merge** | Combine beats into one (same scene) | `meet_doctor` + `investigate` |
| **Create** | New beat replaces separate beats | `climax` serves MAIN + MENTOR + ROMANCE |

**Shared entities signal knot opportunities.** If Doctor appears in ROMANCE and MAIN threads, consider a knot.

#### Iteration Triggers

- **Gap:** "No ROMANCE beat between act 1 and 3" → add beat
- **Density:** "Three threads climax in one beat" → split or accept
- **Orphan:** "This beat has no natural place" → find knot or cut
- **Conflict:** "Requires can't be satisfied" → reorder or add intermediate

#### Completion Criteria

| Phase | Criterion |
|-------|-----------|
| **Initial** | Each thread's beats form valid DAG (no cycles in `requires`) |
| **Knots** | All threads: ≥2 knots. Major thread pairs: ≥1 shared knot |
| **Weaving** | Total order exists respecting all `requires`. Sequence set per arc |
| **Passages** | 1:1 beat → passage transform. Choice edges derived |
| **Reachability** | All expected nodes reachable from start |

#### Choice Edge Derivation

Compare arc sequences:

| Situation | Result |
|-----------|--------|
| Beat X → Y in all arcs | Single diegetic choice |
| Beat X → Y in arc A, X → Z in arc B | Multiple choices, gated by thread codewords |

Every choice has a diegetic label.

#### GROW Output

- Beats (mutated, with knots)
- Arcs (weaved sequences)
- Passages (1:1 from beats)
- Choice edges (derived from arc sequences)
- Codewords (granted by beats/choices)

**Human Gates:** TBD—iteration control mechanics not yet specified.

---

### Stage 5: FILL

**Purpose:** Generate prose for each passage.

**Input:** Validated topology with passage summaries (from GROW), DREAM vision.

**Mode:** Sequential generation, one passage per LLM call.

#### Phase 0: Voice Determination

Before prose generation, establish the voice document. This synthesizes DREAM's high-level vision with GROW's structural data into concrete stylistic guidance.

**Input:**
- Vision node (genre, tone, themes)
- GROW-created nodes (arcs, beats with scene_type, passages with summaries)

**Voice document schema:**
```yaml
voice:
  # Structural choices
  pov: first | second | third_limited | third_omniscient
  pov_character: entity_id | null    # whose perspective (for limited POVs)
  tense: past | present

  # Stylistic choices
  register: string                   # e.g., "formal", "conversational", "literary", "sparse"
  sentence_rhythm: string            # e.g., "varied", "punchy", "flowing"

  # Guidance
  tone_words: string[]               # adjectives describing the voice
  avoid_words: string[]              # words/phrases to not use
  avoid_patterns: string[]           # patterns to avoid

  # Optional exemplars
  exemplar_passages:
    - text: string
      note: string                   # why this exemplifies the voice
```

**Human Gate:** Approve or modify voice document before proceeding.

#### Phase 1: Sequential Prose Generation

**Traversal order:** Spine arc first, then branch arcs.

Rationale: Spine establishes the canonical voice. Branches are variations that write toward established convergence points.

**One passage per LLM call.** Context per call:
- Voice document
- Beat summary (including `scene_type`)
- Entity states at this point (computed from codewords)
- Relevant shadows for active tensions (derived from thread definitions)
- Sliding window: last N passages of generated prose (recommended: 3-5, implementation-dependent)
- Lookahead (when applicable—see below)

**Lookahead for convergence:**

| Situation | Lookahead content |
|-----------|-------------------|
| Writing convergence passage (spine pass) | Beat summaries of all connecting branches |
| Writing branch passage before convergence | Convergence passage prose (already generated) |
| Writing passage after divergence point | Divergence passage prose (for continuity) |

This ensures smooth transitions at structural junctures.

**Output per passage:**
```yaml
fill_output:
  passage_id: string
  prose: string
  entity_updates:              # updates only, no creation
    - id: entity_id
      field: string
      value: string
  relationship_updates:
    - id: relationship_id
      field: string
      value: string
```

**Human Gate:** After all passages generated, approve to proceed to review.

#### Phase 2: Review

Human and/or LLM review of generated prose.

**Review mechanisms:**
- Human review (reading passages, flagging weak ones)
- LLM review (sliding window, recommended 5-10 passages per window)
- Hybrid (LLM flags candidates, human makes final call)

**Special attention to:**
- Convergence passages (must work for all arrivals)
- Voice consistency across passages
- Scene type adherence (scenes have full structure, sequels have breathing room)

**Output:** List of flagged passages with issue descriptions.

#### Phase 3: Revision

Regenerate flagged passages with issue description in context.

For convergence passage revisions: include all approach passages in context (they now exist).

One passage at a time.

#### Phase 4: Optional Second Cycle

If still unsatisfied after revision, one more review-revise cycle.

**Hard cap:** Maximum 2 review-revise cycles (configurable, human-overridable). After cap, ship as-is. Persistent quality issues indicate upstream problems (DREAM voice, SEED structure) rather than FILL execution.

**Human Gate:** Final approval after last cycle.

#### FILL Output

All passages with `prose` populated. Ready for DRESS or SHIP.

#### Context Budget

FILL context is larger than GROW (prose vs summaries). Estimated ~3,000-5,000 tokens per passage call:
- Voice document: ~200-400 tokens
- Beat summary + scene_type: ~100 tokens
- Entity states: ~300-500 tokens
- Shadows: ~100-200 tokens
- Sliding window (3-5 passages): ~1,000-2,000 tokens
- Lookahead (when applicable): ~200-400 tokens
- Output buffer: ~500 tokens

Fits comfortably in modern context windows. For constrained environments, reduce sliding window size.

#### Deferred: Per-Arc Voice Variations

Future versions may support voice modifiers per arc (e.g., "mentor_manipulator" arc is colder and more paranoid). For v5, single voice document applies to all arcs.

---

### Stage 6: DRESS

**Purpose:** Generate presentation layer.

**Input:** Completed prose.

**Output:**

**Illustrations:**
```
Prose → image prompt extraction → asset generation → illustration node
```

**Codex:**
```yaml
codex_entry:
  entity_id: string
  visible_when: codeword[]      # player must have these
  content: string               # player-safe, no spoilers
```

**Depicts edges:** Link illustrations to passages.

**Human Gate:** Review art direction. Optional—story works without it.

---

### Stage 7: SHIP

**Purpose:** Compile to playable format.

**Input:** Prose + (optional) art manifest.

**Output:** ink, Twee/SugarCube, Markdown, or epub.

**This is deterministic.** No LLM involved.

#### Export Schema

SHIP reads from the graph, ignoring working nodes.

**Nodes required:**

| Node | Required fields |
|------|-----------------|
| Passage | id, prose |
| Entity | id, type, base, overlays |
| Codeword | id, type, condition |
| Relationship | id, from_entity, to_entity, base, overlays |
| Illustration | id, asset, caption |

**Edges required:**

| Edge | Required fields |
|------|-----------------|
| Choice | from, to, label, requires, grants |
| Appears | from (entity), to (passage), role |
| Involves | from (relationship), to (passage) |
| Depicts | from (illustration), to (passage) |

**Derived:**

| Derived | From |
|---------|------|
| Codex entries | Entities + visibility rules |
| Start passage | Passage with no incoming Choice edges |
| Ending passages | Passages with no outgoing Choice edges |

**Ignored by SHIP:**
- Tensions, threads, beats, arcs
- `from_beat`, `summary` on passages
- `requires` edges between beats
- All working edges

---

## Validation

### Post-GROW Reachability Check

```
Start passage (no incoming edges)
    ↓ traverse all choice edges
Reachable passages
    ↓ collect via appears/involves edges
Reachable entities, relationships
    ↓ collect via grants
Reachable codewords
```

### Validation Rules

| Check | Failure means |
|-------|---------------|
| Core entity unreachable | GROW incomplete — fix, don't prune |
| Orphan entity | Prune (unless flagged as core) |
| Dead-end passage (unreachable) | GROW bug or impossible gates |
| Unused codeword | Prune or flag for review |
| Orphan relationship | Prune |

### Pruning (Final GROW Step)

After validation passes:
- Remove unreachable nodes
- Topology is now frozen
- FILL can proceed

---

## Prose Template Conditionals

Passages may contain minor inline conditionals for flavor:

```
Kay approached the guard.
[[if:earned_trust]]"Good to see you," he said warmly.[[endif]]
[[if:!earned_trust]]He barely glanced up.[[endif]]
```

**Scope:** Small diegetic variations only. Major divergence = separate passages.

---

## Anti-Patterns: What NOT to Build

### ❌ Agent Negotiation

Do not build systems where multiple LLM agents propose and negotiate.

### ❌ Thread Creation in GROW

Do not allow GROW to create new threads. All threads are declared in SEED.

### ❌ Unbounded Iteration

Do not allow "keep generating until good." Quality comes from good prompts and human curation, not infinite loops.

### ❌ Backflow

Do not allow later stages to modify nodes they don't own. Each node type has a creating stage (see Stage Operations table). If GROW reveals a problem with SEED's threads, the human must manually revert to pre-GROW snapshot and revise SEED.

### ❌ Hidden Prompts

Do not embed prompts in code. Prompts live in `/prompts` as readable files.

### ❌ Complex State Objects

Do not build elaborate state machines or object graphs. State is flat YAML.

### ❌ Surface Choices Without Tension

Do not generate choices that are purely navigational. Every meaningful choice should connect to a tension, even if the player doesn't see the connection explicitly.

---

## Implementation Risks

### GROW Stage Complexity

GROW is the highest-risk area. The spec defines operations (Mark, Merge, Create) but the triggering heuristics need careful design.

**The Knotting Problem:**

How does the system determine that beats from different threads are compatible enough to merge into a single scene (knot)?

- **Risk:** Pure LLM intuition will hallucinate connections.
- **Requirement:** Rigid compatibility checks before LLM involvement.
- **Signals for knot candidacy:**
  - Shared entity (same character appears in both beats)
  - Shared location
  - Compatible time-deltas
  - No conflicting `requires` constraints

**The Sequencing Clarification:**

Weaving threads into a sequence is *not* an LLM problem. Given a set of beats for an arc, topological sort on `requires` constraints produces the order deterministically.

What the LLM actually decides:
1. **Knot creation:** "These beats should be one scene" (merge/mark)
2. **Gap detection:** "Thread X needs a beat here" (create)
3. **Pruning:** "This beat doesn't fit" (cut)

The LLM does not pick ordering—that's derived from the graph.

### Overlay Conflict Resolution

Overlays from different threads can conflict when both are active.

**Thread exclusivity handles some conflicts:**
Codewords from threads sharing a tension are exclusive—player can never have both. No conflict possible.

**Cross-tension conflicts:**
Codewords from *different* tensions can coexist. If their overlays set the same key, conflict resolution applies:
1. Most specific overlay wins (most matching codewords)
2. If tie, later in list wins

**Validation catches design errors:**
At validation, compute all valid codeword combinations. Flag any overlay pairs that can both be active with conflicting values. Human can:
- Restructure threads to make codewords exclusive
- Add compound overlay for the specific combination
- Accept list-order as tiebreaker

### FILL Stage Context

See Stage 5: FILL for complete context specification (voice document, sliding window, lookahead strategy).

**Knot awareness:** When FILL generates prose for a knot, the brief must include:
- Which threads this beat serves (X and Y)
- The tension impacts for each thread
- Guidance to weave both threads' concerns into one scene

This is derived from beat metadata and passed explicitly in the FILL context.

---

## Context Budget Analysis

### Token Estimates for GROW

Because GROW operates on beats (summaries) rather than passages (prose), the data footprint is small.

**Single beat:**

| Component | Est. Tokens |
|-----------|-------------|
| ID & metadata | ~10 |
| Summary (2-3 sentences) | ~40-60 |
| Tags (threads, tensions, entities) | ~20-30 |
| Logic (requires, grants) | ~10-20 |
| YAML overhead | ~10 |
| **Total per beat** | **~100-130** |

**Full context for arc generation (medium story, 80-100 beats):**

| Component | Content | Est. Tokens |
|-----------|---------|-------------|
| System prompt | GROW rules, knot definition, constraints | ~1,500 |
| Thread definitions | 8-12 threads × ~60 tokens | ~500-700 |
| Tension definitions | 4-6 tensions × ~100 tokens | ~400-600 |
| Entity base definitions | 15-20 entities × ~80 tokens | ~1,200-1,600 |
| Relationship definitions | 10-15 relationships × ~50 tokens | ~500-750 |
| Codeword definitions | All codewords | ~300-500 |
| Arc history | Last 5 beats placed (continuity) | ~600 |
| Candidate beats | Unplaced beats for active threads | ~2,500-8,000 |
| **Total** | | **~8,000-15,000** |

**Worst case:** Early GROW with 60-80 unplaced beats: ~15,000-18,000 tokens.

### Verdict

Even pessimistic estimates use <15% of modern context windows (128k-200k). **No RAG or aggressive chunking needed for GROW.**

The entire relevant subgraph (all unplaced beats for involved threads) can be passed in every API call. This validates the "per arc" generation approach—the model can hold full arc state in working memory.

### FILL Context (Different Concern)

FILL operates on prose, which is larger. A single passage might be 200-400 tokens of prose, plus context.

For FILL, context pruning *is* needed:
- Don't pass all prior passages
- Pass summaries of recent beats + full current brief
- Estimated FILL context: ~3,000-5,000 tokens per passage (manageable)

### Constrained Environments (32k Context)

For local models (Llama 3.1 8B, Qwen, Mistral) with 32k context windows:

**Budget calculation:**

| Component | Tokens |
|-----------|--------|
| System prompt / instructions | ~1,500 |
| Entity/relationship definitions | ~2,350 |
| Thread/tension definitions | ~1,300 |
| Output buffer (safety margin) | ~2,000 |
| **Fixed cost** | **~7,150** |
| **Available for beats** | **~24,850** |

At ~140 tokens per beat: **~177 beats maximum**

This is equivalent to a complete 40,000-word novella or a dense 2-hour branching game. Not a toy.

**Quality degradation risks (8B models):**

- **Needle in haystack:** Model struggles to find connections between beats at opposite ends of context
- **Instruction drift:** As context fills, model may forget constraints (e.g., Thread Freeze rule)

**Recommendation:** Cap at ~120 beats for 8B models to preserve reasoning quality.

**Context windowing for larger stories:**

If story exceeds safe limits:
1. **Summarize frozen arcs:** Once an arc is fully woven, compress to ~500-token summary
2. **Focus on active frontier:** Only pass beats not yet fully connected
3. **Split into acts:** Treat acts as separate project files if necessary

This allows scaling beyond 177 beats while staying within context limits.

---

## Design Decisions

### Iterative Weaving Approach

**Decision:** Arc-at-a-time, not beat-by-beat.

An arc is a complete weave of compatible threads. The process:
1. Generate spine arc (primary thread combination)
2. Validate spine arc
3. For each exclusive thread not in spine, generate divergent arc
4. Divergent arcs share beats with spine until divergence point

Beat-by-beat or sliding window risks drift. Arc-at-a-time maintains coherence.

### Codeword Granularity

**Decision:** Boolean only for v5.0.

Numeric state (`modifies: { trust: +1 }`) is a documented bolt-on for future versions.

For v5.0, relationship tracking uses discrete codewords:
- `trust_none`, `trust_low`, `trust_medium`, `trust_high`

Ugly but explicit. Avoids scope creep into state machine complexity.

---

## File Structure

The project uses a **unified graph** stored as JSON, with snapshots for rollback
and optional human-readable exports.

```
/project
  graph.json              # The unified story graph (single source of truth)
  /snapshots              # Pre-stage snapshots for rollback
    pre-dream.json
    pre-brainstorm.json
    pre-seed.json
    pre-grow.json
    pre-fill.json
    pre-dress.json
  /exports                # Human-readable views (generated on demand)
    graph.yaml            # Full graph in YAML for review
    graph.md              # Markdown summary for reading
    story.ink             # Playable output
    story.html            # Standalone HTML
  /prompts                # Prompt templates (see 01-prompt-compiler.md)
    /components
      /schemas
      /constraints
      /style
      /instructions
    /templates
  /config
    project.yaml          # Project configuration
```

### Graph File Format

The `graph.json` file contains:

```json
{
  "version": "5.0",
  "meta": {
    "last_stage": "seed",
    "last_modified": "2026-01-13T10:30:00Z"
  },
  "nodes": {
    "vision": { "type": "vision", ... },
    "protag_001": { "type": "entity", ... },
    "opening_001": { "type": "passage", ... }
  },
  "edges": [
    { "type": "choice", "from": "opening_001", "to": "fork_001", ... }
  ]
}
```

### Snapshot Strategy

Before each stage runs:
1. Copy current `graph.json` to `snapshots/pre-{stage}.json`
2. Run stage (graph modified in memory)
3. Write updated `graph.json` on success
4. If stage fails, graph.json unchanged (snapshot available for manual recovery)

### Human Review

When users run `qf review`:
1. Generate `exports/graph.yaml` from current graph
2. Generate `exports/graph.md` with readable summary
3. User reviews in their preferred format
4. Changes made via CLI commands, not direct file editing

---

## Summary Tables

### All Node Types

| Node | Persistent | Created in | Required for SHIP |
|------|------------|------------|-------------------|
| Vision | Yes | DREAM | No (context only) |
| Passage | Yes | GROW | Yes (with prose) |
| Entity | Yes | BRAINSTORM/SEED | Yes (FILL can update, not create) |
| Codeword | Yes | GROW | Yes |
| Relationship | Yes | BRAINSTORM/SEED | Yes (FILL can update, not create) |
| Illustration | Yes | DRESS | Yes |
| Codex | Yes | DRESS | Yes |
| Tension | No | BRAINSTORM | No |
| Alternative | No | BRAINSTORM | No |
| Consequence | No | SEED | No |
| Plot Thread | No | SEED | No |
| Beat | No | SEED/GROW | No |
| Arc | No | GROW | No |

### All Edge Types

| Edge | Persistent | Created in | Required for SHIP |
|------|------------|------------|-------------------|
| Choice | Yes | GROW | Yes |
| Appears | Yes | GROW | Yes |
| Involves | Yes | GROW | Yes |
| Depicts | Yes | DRESS | Yes |
| involves (tension) | No | BRAINSTORM | No |
| has_alternative | No | BRAINSTORM | No |
| explores | No | SEED | No |
| has_consequence | No | SEED | No |
| belongs_to | No | SEED/GROW | No |
| requires (beat) | No | SEED/GROW | No |
| grants (beat) | No | GROW | No |
| weaves | No | GROW | No |
| from_beat | No | GROW | No |

### Tension → Thread → Beat Flow

```
Tension (BRAINSTORM)
  "Can the mentor be trusted?"
  ├─ alternative: mentor_protector
  └─ alternative: mentor_manipulator
        ↓ SEED triage
Thread (SEED)
  mentor_protector_thread
    tension_id: mentor_trust
    alternative_id: mentor_protector
    shadows: [mentor_manipulator]    ← unexplored alternative
        ↓ SEED/GROW
Beat (SEED/GROW)
  mentor_reveals_truth
    threads: [mentor_protector_thread]
    tension_impacts:
      - tension_id: mentor_trust
        effect: commits
        note: "Player learns mentor sent warning to family"
```

---

## Open Questions

1. **GROW algorithm specifics:** Compatibility checks, knot detection heuristics, and the exact flow from SEED to completed arcs need detailed specification. (Next priority.)

2. **Human gates within GROW:** How many gates? After each arc? After all arcs? TBD.

3. **Prompt compilation architecture:** The Vision doc's prompt compilation system (components, templates, sandwiching, context budgets) is likely still valid but needs review against the new ontology.

### Resolved

- **Iteration approach:** Arc-at-a-time (see Design Decisions)
- **Codeword granularity:** Boolean only for v5.0 (see Design Decisions)
- **Context budget:** Full subgraph fits in context, no RAG needed (see Context Budget Analysis)
- **Overlay conflicts:** Most specific wins, then list order; validation flags potential conflicts (see State Pattern)
- **Tension cardinality:** Exactly two alternatives (binary); nuance via multiple tensions
- **Entity creation in GROW:** Not allowed; GROW cannot create Entity nodes
- **Derived codewords:** Deferred to future version (forward compatible)

---

## Future Bolt-ons

- **Numeric state:** `modifies` on choices, `derived` codewords
- **Choice visibility:** `visible_when` separate from `requires`
- **Entity absence:** overlay with `present: false`
- **Audit log:** Mutation history (separate from graph)
- **GROW recovery:** Preserve partial GROW work when aborting to SEED (currently discards all)
- **Per-arc voice variations:** Voice document modifiers per arc (e.g., "mentor_manipulator" arc uses colder, more paranoid voice)
