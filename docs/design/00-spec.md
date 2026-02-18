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
| BRAINSTORM | Entities, dilemmas, answers | LLM-heavy (discuss → summarize → serialize) |
| SEED | Triage: curate entities, promote dilemmas to paths | Human-heavy |
| GROW | Mutate graph until complete: intersections, weaving, passages | Layered, human gates |
| FILL | Generate prose for passages | LLM, sequential |
| DRESS | Illustrations, codex | LLM, optional |
| SHIP | Export to ink/Twee/epub | Deterministic, no LLM |

### Key Constraints

1. **No persistent agent state between sessions.** Each session starts fresh.
2. **All artifacts are files.** JSON, YAML, or Markdown. Version-controllable, diffable, human-editable.
3. **Paths are frozen after SEED.** GROW can mutate beats, create intersections, weave arcs—but cannot create new paths.
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
| BRAINSTORM | Entity, Dilemma, Answer | — | Vision |
| SEED | Path, Consequence, Beat | Entity (curate), Dilemma (explore) | Entity, Dilemma |
| GROW | Arc, Passage, Choice, Codeword; new Beats | Beat (scene_type, intersection) | Path, Beat, Entity |
| FILL | Voice | Passage (prose), Entity (details) | Passage, Entity, Path |
| DRESS | ArtDirection, EntityVisual, IllustrationBrief, Illustration, CodexEntry | — | Passage, Entity, Vision, Codeword |
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

**Example:** SEED produces entity decisions, path definitions, and initial beats. The runtime:
1. Validates the output structure
2. Updates existing entity nodes (marking dispositions)
3. Creates new path, consequence, and beat nodes
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
  # Single-beat passage (standard)
  from_beat: beat_id | null    # traceability (ignored by SHIP)
  summary: string | null       # pre-FILL (ignored by SHIP)
  prose: string | null         # post-FILL (required for SHIP)
  # Merged passage fields (only present after Phase 9d collapse)
  from_beats: beat_id[]        # all source beats (replaces from_beat)
  primary_beat: beat_id        # main beat for ID derivation and summary
  merged_from: passage_id[]    # original passages that were collapsed
  transition_points:           # guidance for FILL on transitions
    - index: int               # position in from_beats list
      style: smooth | cut      # transition style
      bridge_entities: entity_id[]
      note: string             # human-readable guidance
```

A passage is complete when `prose` exists.

**Lifecycle:** Created in GROW (1:1 from beats or N:1 after collapse), prose added in FILL. Exported.

**Single vs Merged passages:**
- **Single-beat:** Uses `from_beat` field, created in Phase 8a
- **Merged:** Uses `from_beats`, `primary_beat`, `merged_from`, created in Phase 9d
- Code should check `from_beats` first, fall back to `from_beat`

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

#### ArtDirection

Global visual identity document for the story's presentation layer. Analogous to FILL's voice document but for visual style.

```yaml
art_direction:
  id: string
  style: string                    # e.g., "watercolor", "digital painting", "ink sketch"
  medium: string                   # what it looks like it was made with
  palette: string[]                # dominant colors / mood
  composition_notes: string        # framing preferences
  negative_defaults: string        # global things to avoid in image generation
  aspect_ratio: string             # default dimensions, e.g., "16:9"
```

**Lifecycle:** Created in DRESS (art direction phase). Exported.

#### Illustration

Art asset with diegetic caption. Each illustration is generated from an IllustrationBrief (working node) and linked to a passage via a Depicts edge.

```yaml
illustration:
  id: string
  asset: path                      # e.g., assets/<sha256>.png
  caption: string                  # diegetic — in-world voice, not meta-description
  category: string                 # scene | portrait | vista | item_detail
```

**Lifecycle:** Created in DRESS (image generation phase). Exported.

**Diegetic constraint:** Captions must be written in the story's voice—as if they were part of the world ("The bridge where loyalties shatter"), never meta-descriptive ("An illustration of two characters on a bridge").

#### CodexEntry

Player-facing encyclopedia entries for entities. Provides in-world information without spoilers. Multiple entries per entity enable spoiler-graduated knowledge: players see more as they unlock codewords.

```yaml
codex_entry:
  id: string
  rank: integer                    # display order (1 = base knowledge, higher = deeper)
  visible_when: codeword[]         # all must be present to unlock this tier
  content: string                  # diegetic — in-world voice, player-safe, no spoilers
```

**Lifecycle:** Created in DRESS (codex phase). Exported.

**Entity link:** Via HasEntry edge (codex_entry → entity), not an entity_id field. This follows the ontology's edge-based relationship pattern.

**Cumulative model:** Multiple codex_entry nodes per entity, each with a different rank and visibility gate. SHIP displays all unlocked entries sorted by rank. Each entry is self-contained (readable without other tiers), but the LLM is instructed to minimize redundancy across tiers.

**Example:**
```yaml
# Tier 1: always visible (visible_when: [])
codex_entry:
  id: codex::aldric_basic
  rank: 1
  visible_when: []
  content: "A traveling scholar who offers guidance to those in need."

# Tier 2: after meeting (visible_when: [met_aldric])
codex_entry:
  id: codex::aldric_background
  rank: 2
  visible_when: [met_aldric]
  content: "Claims to be a former court advisor, exiled for speaking truth."

# Tier 3: after discovery (visible_when: [discovered_betrayal])
codex_entry:
  id: codex::aldric_truth
  rank: 3
  visible_when: [discovered_betrayal]
  content: "His exile was self-imposed — he left after orchestrating the king's downfall."
```

---

### Creative Nodes (created in BRAINSTORM, refined in SEED)

#### Dilemma

The dramatic question that drives meaningful choice. Every interesting story choice implies a road not taken. A dilemma presents exactly two answers where both options have costs and benefits.

```yaml
dilemma:
  id: string                      # Format: dilemma::dilemma_name
  question: string                # "Can the mentor be trusted?"
  answers:
    - id: string
      description: string         # "Mentor is genuine protector"
      canonical: true             # used for spine arc
    - id: string
      description: string         # "Mentor is manipulating Kay"
      canonical: false            # alternate arc if explored
  involves: entity_id[]
  why_it_matters: string          # thematic stakes
  # Added by SEED:
  explored: answer_id[]           # which answers LLM intended to explore
  unexplored: answer_id[]         # answers not explored (for FILL shadows)
```

**Lifecycle:** Created in BRAINSTORM, exploration decisions added in SEED. Not exported.

**The `unexplored` field** holds answers that are intentionally NOT explored as paths. These provide narrative context for the FILL stage—the "road not taken" that gives meaning to the chosen path. For example, if we explore "mentor is deceptive", the unexplored "mentor is trustworthy" informs how the deception contrasts with what could have been.

**Derived development states** (computed from path existence, not stored):
- **committed**: Answer has a path in the graph (will become a story path)
- **deferred**: Answer in `explored` but no path (LLM intended to explore but was pruned)
- **latent**: Answer not in `explored` (never intended for exploration, becomes shadow)

The `explored` field records what the LLM *intended* to explore. Actual path existence determines what was *committed*. This separation allows pruning to drop paths without modifying the dilemma's stored intent, keeping the field immutable after SEED.

**Binary constraint:** Exactly two answers per dilemma. This keeps contrasts crisp.

**Canonical flag:** One answer is marked `canonical: true`. This is the "default" story—used for the spine arc. The non-canonical answer becomes a branch if promoted to a path in SEED.

For nuanced situations, use multiple dilemmas on the same concept:
```yaml
dilemmas:
  - id: dilemma::mentor_alignment
    question: "Is the mentor benevolent or self-serving?"
    answers:
      - id: mentor_benevolent
      - id: mentor_selfish

  - id: dilemma::mentor_competence
    question: "Is the mentor capable or flawed?"
    answers:
      - id: mentor_capable
      - id: mentor_flawed
```

This yields four combinations (benevolent+capable, benevolent+flawed, etc.) while each dilemma remains a clear binary contrast.

**Key insight:** "Mentor is a protector" is flat. "Mentor is a protector (not the manipulator they could have been)" has dramatic weight—even if we never write the manipulator path.

#### Answer

One possible answer to a Dilemma's question. Extracted as separate nodes in the graph to enable path/answer relationships.

```yaml
answer:
  id: string
  description: string             # "Mentor is genuine protector"
  canonical: bool                 # true = used for spine arc
  dilemma_id: string              # parent dilemma (dilemma::dilemma_name)
```

**Lifecycle:** Created in BRAINSTORM as part of dilemma generation. Not exported.

---

### Working Nodes (consumed by GROW, ignored by SHIP)

#### Consequence

Narrative meaning of a path choice. Bridges the gap between "what this path represents" (answer) and "how we track it" (codeword).

```yaml
consequence:
  id: string
  path_id: path_id                  # which path this belongs to
  description: string               # "Mentor becomes protective ally"
  ripples: string[]                 # story effects this implies
    # - "Shields Kay in confrontation"
    # - "Reveals family connection"
```

**Lifecycle:** Created in SEED when paths are created. Not exported.

GROW creates codewords to track when consequences become active, and creates entity overlays to implement consequence effects.

#### Path

One explored answer from a dilemma. Paths from the same dilemma are automatically exclusive.

```yaml
path:
  id: string                        # Format: path::dilemma_id__answer_id (hierarchical)
  name: string
  dilemma_id: dilemma_id            # Derivable from path_id
  answer_id: answer_id              # which answer this explores
  shadows: answer_id[]              # unexplored answers (context for FILL)
  tier: major | minor
  description: string
  consequences: consequence_id[]    # narrative meaning of this path
  entity_arcs:                      # per-entity arc descriptors (working, set in GROW Phase 4f)
    - entity_id: entity_id
      arc_line: string              # trajectory in "A → B → C" format (10-200 chars)
      pivot_beat: beat_id           # beat where the arc turns (path-scoped)
      arc_type: string              # computed from entity category (transformation|atmosphere|significance|relationship)
```

**Hierarchical ID format:** Path IDs encode their parent dilemma using `__` separator:
- `path::mentor_trust__benevolent` → dilemma_id is `dilemma::mentor_trust`, answer is `benevolent`
- This prevents LLM confusion between dilemma and path IDs
- The `dilemma_id` field can be derived from the path_id

**Lifecycle:** Created in SEED. Not exported. (PATH FREEZE: no new paths after SEED)

**Tier:**
- **Major:** Defines the story. Must interweave with other major paths.
- **Minor:** Supports/enriches. Must touch the story but can be more independent.

**Exclusivity is derived:** All paths sharing a `dilemma_id` are automatically exclusive. No manual declaration needed.

#### Beat

Narrative unit. Belongs to one or more paths.

```yaml
beat:
  id: string
  summary: string
  scene_type: scene | sequel | micro_beat    # pacing structure (assigned in GROW)
  paths: path_id[]              # usually one, multiple = intersection
  dilemma_impacts:
    - dilemma_id: dilemma_id
      effect: advances | reveals | commits | complicates
      note: string              # "Player sees mentor's private communication"
  sequenced_after: beat_id[]    # topological ordering: this beat must come after these beats (does not imply direct adjacency)
  grants: codeword_id[]
  entities: entity_id[]
  relationships: relationship_id[]
  location: entity_id | null              # primary location (assigned in SEED)
  location_alternatives: entity_id[]      # other valid locations (enables intersection flexibility)
  # Gap beat fields (only present on beats created by GROW Phase 4b/4c)
  is_gap_beat: boolean                    # true for beats inserted to fill narrative/pacing gaps
  transition_style: smooth | cut          # guidance for FILL on how to handle the transition
  bridges_from: beat_id | null            # beat this gap follows (traceability)
  bridges_to: beat_id | null              # beat this gap precedes (traceability)
```

**Lifecycle:** Initial beats created in SEED, mutated and new beats added in GROW. Not exported.

**Location flexibility:** Beats can specify alternative locations where the same dramatic action could occur. If Beat A (at Market) and Beat B (at Docks) both have `location_alternatives` including each other's location, GROW can merge them by choosing a shared setting. This enables intersection formation without constraining BRAINSTORM's creative freedom.

**Scene types:**

| Type | Purpose | Prose guidance |
|------|---------|----------------|
| `scene` | Active pursuit: goal → obstacle → outcome | Full dramatic structure, 3+ paragraphs |
| `sequel` | Reactive processing: reaction → dilemma → decision | Breathing room after disaster, 2-3 paragraphs |
| `micro_beat` | Transition, time passage, minor moment | Brief, 1 paragraph |

Scene type is assigned during GROW (Phase 4: Gap Detection) to ensure pacing variety across arcs. GROW may propose additional beats to address pacing gaps (e.g., "three scenes in a row with no sequel").

**Gap beats:** Beats created by GROW Phase 4b/4c to fill narrative or pacing gaps. Gap beats inherit `entities` and `location` from adjacent beats to maintain context. The `transition_style` field guides FILL:

| Style | When Used | FILL Guidance |
|-------|-----------|---------------|
| `smooth` | Same location, shared entities, scene continuity | Flow naturally, echo imagery |
| `cut` | Location change, scene type change, time jump | Establish new context quickly |

**Beat types by path membership:**
- **Single-path:** Serves one path's progression
- **Intersection:** Serves multiple paths (natural intersection point)

**Dilemma impact effects:**

| Effect | Meaning |
|--------|---------|
| `advances` | Moves toward resolution without revealing answer |
| `reveals` | Surfaces information bearing on the question |
| `commits` | Point of no return—answer is now locked in |
| `complicates` | Introduces doubt, new dimension to dilemma |

#### Arc

Realized weaving of compatible paths.

```yaml
arc:
  id: string
  type: spine | branch
  paths: path_id[]              # must be from different dilemmas (compatible)
  sequence: beat_id[]           # ordered result of weaving
  parent: arc_id | null
  diverges_at: beat_id | null
  converges_at: beat_id | null
```

**Lifecycle:** Created in GROW during arc enumeration. Not exported.

#### EntityVisual

Per-entity visual identity profile. Ensures consistent appearance across all illustrations featuring this entity. Created during the art direction phase of DRESS, linked to its entity via a `describes_visual` edge.

```yaml
entity_visual:
  id: string
  description: string                   # prose description of appearance
  distinguishing_features: string[]     # key visual identifiers
  color_associations: string[]          # colors tied to this entity
  reference_prompt_fragment: string     # injected into every image prompt featuring this entity
```

**Lifecycle:** Created in DRESS (art direction phase). Not exported.

**Purpose:** When illustrating a passage, the image prompt assembler reads EntityVisual nodes for all entities present (via Appears edges) and injects their `reference_prompt_fragment` into the prompt. This ensures a character, location, or object looks the same across all illustrations.

#### IllustrationBrief

Structured image prompt with priority scoring. One brief is generated per passage; only selected briefs are rendered into Illustration nodes.

```yaml
illustration_brief:
  id: string
  priority: integer                     # 1=must-have, 2=important, 3=nice-to-have
  category: string                      # scene | portrait | vista | item_detail | cover
  subject: string                       # what the image depicts
  entities: string[]                    # entity IDs present in scene
  composition: string                   # framing / camera notes
  mood: string                          # emotional tone
  style_overrides: map                  # empty = use global art direction
  negative: string                      # things to avoid in this image
  caption: string                       # proposed diegetic caption
```

**Lifecycle:** Created in DRESS (illustration phase). Not exported.

**Priority scoring (hybrid):** Structural rules provide a base score (spine passages, climax scenes, and endings score higher); LLM judgment adjusts for visual interest and narrative importance. See procedures/dress.md for the full algorithm.

**Linked to passage** via `targets` edge (illustration_brief → passage).

---

## Edge Types

> **Naming Convention:** Persistent edges use PascalCase (Choice, Appears) as they appear
> in exports. Working edges use snake_case (belongs_to, has_answer) as they're internal only.

### Persistent Edges (survive to export)

| Edge | From → To | Properties | Created In | Purpose |
|------|-----------|------------|------------|---------|
| **Choice** | passage → passage | label, requires_codewords[], grants[], modifies{} | GROW | Player navigation |
| **Appears** | entity → passage | role | GROW | Entity present in scene |
| **Involves** | relationship → passage | — | GROW | Relationship active in scene |
| **Depicts** | illustration → passage | — | DRESS | Art shown with passage |
| **HasEntry** | codex_entry → entity | — | DRESS | Codex describes this entity |

**Choice properties:**
```yaml
choice:
  label: string                 # always diegetic ("Wait for nightfall...", never "Continue")
  requires_codewords: codeword[] # gate: player must hold these codewords to traverse this choice
  grants: codeword[]            # state change
  modifies:                     # future: numeric state
    state_key: delta
```

### Working Edges (consumed by GROW, not exported)

| Edge | From → To | Created In | Purpose |
|------|-----------|------------|---------|
| **belongs_to** | beat → path | SEED | Beat serves this path |
| **involves** | dilemma → entity | BRAINSTORM | Dilemma involves these entities |
| **has_answer** | dilemma → answer | BRAINSTORM | Dilemma's possible answers |
| **explores** | path → answer | SEED | Path explores this answer |
| **has_consequence** | path → consequence | SEED | Path's narrative consequences |
| **sequenced_after** | beat → beat | SEED, GROW | Topological ordering: from-beat is sequenced after to-beat (does not imply direct adjacency) |
| **grants** | beat → codeword | GROW | Beat completion grants codeword |
| **weaves** | arc → path | GROW | Arc uses this path |
| **from_beat** | passage → beat | GROW | Traceability |
| **describes_visual** | entity_visual → entity | DRESS | Visual profile for entity |
| **from_brief** | illustration → illustration_brief | DRESS | Traceability to source brief |
| **targets** | illustration_brief → passage | DRESS | Brief targets this passage |

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

Path exclusivity (from shared dilemma) prevents some conflicts automatically—you can't have codewords from exclusive paths. But codewords from *different* dilemmas can coexist, and their overlays might conflict.

Example: `mentor_trusted` (from alignment dilemma) and `mentor_ill` (from health dilemma) can both be active. If both set the same key, resolution rules apply.

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

  dilemmas:
    - id: string                # Format: dilemma::dilemma_name
      question: string          # "Can the mentor be trusted?"
      answers:
        - id: string
          description: string
          canonical: true       # default story path
        - id: string
          description: string
          canonical: false      # becomes branch if explored
      involves: entity_id[]
      why_it_matters: string
```

BRAINSTORM generates freely without worrying about path collision. Location flexibility for intersection formation is handled in SEED.

**Human Gate:** Review brainstorm output before triage.

---

### Stage 3: SEED

**Purpose:** Triage brainstorm into committed structure. **Path creation gate.**

**Input:** Approved brainstorm.

**Operations:**
1. Curate entities (in/out)
2. For each dilemma: decide which answers to explore
   - Canonical answer always becomes a path (spine path)
   - Non-canonical answer becomes a path only if exploring that branch
3. Explored answers become paths (with exclusivity inherited from shared dilemma)
4. Create initial beats per path

**Output:**
```yaml
seed:
  entities:
    - id: string
      type: character | location | object | faction
      concept: string
      # full entity structure created here

  dilemmas:
    - dilemma_id: string              # Format: dilemma::dilemma_name
      explored: answer_id[]           # always includes canonical; may include non-canonical
      unexplored: answer_id[]         # non-explored answers (context for FILL)

  paths:
    - id: string                      # Format: path::dilemma_id__answer_id (hierarchical)
      name: string
      dilemma_id: dilemma_id          # Derivable from path_id
      answer_id: answer_id
      shadows: answer_id[]
      tier: major | minor
      description: string
      consequences: consequence_id[]

  consequences:
    - id: string
      path_id: path_id
      description: string             # "Mentor becomes protective ally"
      ripples: string[]               # story effects this implies

  initial_beats:
    - id: string
      summary: string
      paths: path_id[]
      dilemma_impacts:
        - dilemma_id: dilemma_id
          effect: advances | reveals | commits | complicates
          note: string
      entities: entity_id[]
      location: entity_id | null            # primary location
      location_alternatives: entity_id[]    # other valid locations (enables intersection flexibility)

  convergence_sketch:                 # informal creative guidance for GROW
    convergence_points: string[]      # "paths should merge by act 2 climax"
    residue_notes: string[]           # "mentor demeanor differs after convergence"

  # --- Convergence Analysis (Sections 7-8) ---
  # Machine-actionable structural contracts. Generated post-prune,
  # after all 6 core sections are serialized. See ADR-013.

  dilemma_analyses:
    - dilemma_id: dilemma_id
      convergence_policy: hard | soft | flavor   # topology layer
      ending_salience: high | low | none         # prose layer (endings)
      residue_weight: heavy | light | cosmetic   # prose layer (mid-story)
      payoff_budget: int (2-6)        # minimum exclusive beats before convergence
      reasoning: string               # chain-of-thought justification
      ending_tone: string | null      # required when ending_salience=high

  interaction_constraints:             # sparse — only related dilemma pairs
    - dilemma_a: dilemma_id
      dilemma_b: dilemma_id
      constraint_type: shared_entity | causal_chain | resource_conflict
      description: string
      reasoning: string
```

#### Convergence: Topology Layer vs Prose Layer

Convergence control is split into two orthogonal layers:

- **Topology Layer** (`convergence_policy`): Controls whether and when beat sharing occurs in the graph structure. This is purely structural — it determines the shape of arcs, not the content of prose.
- **Prose Layer** (`ending_salience`, `residue_weight`): Controls how much prose varies based on the player's choice. This is purely narrative — it determines what gets written, not the graph shape.

These layers are independent. Any combination is valid:

| convergence_policy | ending_salience | residue_weight | Meaning |
|---|---|---|---|
| hard | high | heavy | Paths never merge; endings differ; mid-story shows differences |
| soft | high | cosmetic | Paths merge mid-story; endings differ; shared passages ignore choice |
| soft | low | heavy | Paths merge; endings don't depend on it; but shared passages show differences |
| flavor | none | cosmetic | Same structure throughout; choice has no narrative impact |

#### Topology Layer: Convergence Policies

Per-dilemma `convergence_policy` declared by SEED, enforced by GROW. Determines how and whether branch arcs reconverge with the spine.

| Policy | Meaning | GROW Behavior |
|--------|---------|---------------|
| `hard` | Paths never reconverge structurally | `converges_at` is not set. Uses codeword gating at divergence points and encourages separate endings. Shared beats are topologically allowed but gated (see #751). |
| `soft` | Paths reconverge after `payoff_budget` exclusive beats | Backward scan: last exclusive beat marks convergence boundary. If exclusive beats < budget, no convergence. |
| `flavor` | Same structure, different prose via overlays | Immediate convergence at first shared beat. Overlays provide tonal variation. |

**Multi-dilemma arcs:** When an arc spans multiple dilemmas, `hard` dominates; `payoff_budget = max(...)` across all dilemmas.

**`convergence_sketch` vs `convergence_policy`:** These are complementary, not redundant. `convergence_policy` is a machine-actionable structural contract enforced by GROW algorithms. `convergence_sketch` is freeform creative guidance from the LLM to itself, used for narrative hints during prose generation.

#### Prose Layer: Ending Salience

Per-dilemma `ending_salience` controls how much story endings differ based on this dilemma's outcome.

| Value | Meaning | FILL Behavior |
|-------|---------|---------------|
| `high` | Endings MUST differ | Ending family signatures include this dilemma's codewords. `ending_tone` guides prose. |
| `low` | Endings MAY acknowledge | Ending prose may mention choice but must work without it. |
| `none` | Endings MUST NOT reference | Negative obligation injected: "Do NOT reference this choice." |

Only 1-2 dilemmas per story should be `high`. Most should be `low`.

#### Prose Layer: Residue Weight

Per-dilemma `residue_weight` controls how much mid-story prose varies in shared (converged) passages.

| Value | Meaning | FILL Behavior |
|-------|---------|---------------|
| `heavy` | Shared passages MUST show state-specific differences | Variant routing required. Negative obligation: "MUST show differences." |
| `light` | Shared passages MAY acknowledge | Existing behavior. Validation warns if no routing exists. |
| `cosmetic` | Shared passages MUST NOT reference | Filtered from residue candidates. Negative obligation: "Do NOT reference." |

Only 1-2 dilemmas per story should be `heavy`. Most should be `light`.

#### Prose Layer Obligations (Summary)

| Field | Value | Obligation |
|-------|-------|------------|
| `ending_salience` | `high` | Ending prose MUST differ for this choice. |
| `ending_salience` | `low` | Ending prose MAY acknowledge, but must work without it. |
| `ending_salience` | `none` | Ending prose MUST NOT reference this choice. |
| `residue_weight` | `heavy` | Shared passages MUST show state-specific differences. |
| `residue_weight` | `light` | Shared passages MAY acknowledge. |
| `residue_weight` | `cosmetic` | Shared passages MUST NOT reference this choice. |

See the detailed `ending_salience` and `residue_weight` tables above for examples and additional constraints.

#### Unified Variant Routing Primitive

`split_and_reroute()` is the shared mechanism for both ending families and residue passages. Instead of adding extra hub passages, it rewrites incoming choice edges:

1. For each incoming choice to a base passage, clone it per variant
2. Each clone gets a `requires_codewords` gate (variant's codewords) and `is_routing=True`
3. Original incoming choices are deleted (or kept as fallback with `keep_fallback=True`)

**Validation:** `check_routing_coverage()` validates routing choice sets are:
- **Collectively-exhaustive (CE):** Every arc covering the passage has at least one satisfiable route
- **Mutually-exclusive (ME):** At most one route is satisfiable per arc (warn if violated)

**Prose neutrality validation:** `check_prose_neutrality()` validates that shared passages satisfy prose-layer contracts — `heavy`/`high` without routing fails, `light` without routing warns, `cosmetic`/`none` passes.

#### "Residue Must Be Read" Invariant

Every codeword granted must appear in at least one `choice.requires_codewords` gate. Current scope: choice gating and variant routing. Routing choices (`is_routing=True`) satisfy this invariant for their required codewords.

**`converges_at` semantics:** "From this beat onward, all remaining content on this arc is shared with the spine." It is NOT set at intersections (shared beats with later exclusive beats). For `hard` policy, it is never set.

**Human Gate:** Approve seed. After this point, no new paths can be created.

**Critical constraint:** PATH FREEZE. GROW cannot create paths. All branching potential is declared here.

---

### Stage 4: GROW

**Purpose:** Generate the complete story topology through graph mutation.

GROW operates on the graph until completion criteria are met. It can mutate beats, create intersections, weave arcs, derive passages and choice edges—but cannot create new paths.

#### Initial State (from SEED)

- Paths with initial beats (single-path, loose)
- Exclusivity derived from shared dilemma_id
- Internal ordering (`sequenced_after`) declared
- Path tiers declared (major/minor)
- Core entities created
- Location flexibility annotated on beats (enables intersection detection)

#### Intersection Operations

Intersections are beats serving multiple paths. Three operations:

| Operation | Description | Example |
|-----------|-------------|---------|
| **Mark** | Existing beat serves multiple paths | `investigate` also serves GADGETS path |
| **Merge** | Combine beats into one (same scene) | `meet_doctor` + `investigate` |
| **Create** | New beat replaces separate beats | `climax` serves MAIN + MENTOR + ROMANCE |

**Shared entities signal intersection opportunities.** If Doctor appears in ROMANCE and MAIN paths, consider an intersection.

#### Iteration Triggers

- **Gap:** "No ROMANCE beat between act 1 and 3" → add beat
- **Density:** "Three paths climax in one beat" → split or accept
- **Orphan:** "This beat has no natural place" → find intersection or cut
- **Conflict:** "Requires can't be satisfied" → reorder or add intermediate

#### Completion Criteria

| Phase | Criterion |
|-------|-----------|
| **Initial** | Each path's beats form valid DAG (no cycles in `sequenced_after`) |
| **Intersections** | All paths: ≥2 intersections. Major path pairs: ≥1 shared intersection |
| **Weaving** | Total order exists respecting all `sequenced_after`. Sequence set per arc |
| **Passages** | 1:1 beat → passage transform. Choice edges derived |
| **Reachability** | All expected nodes reachable from start |

#### Choice Edge Derivation

Compare arc sequences:

| Situation | Result |
|-----------|--------|
| Beat X → Y in all arcs | Single diegetic choice |
| Beat X → Y in arc A, X → Z in arc B | Multiple choices, gated by path codewords |

Every choice has a diegetic label.

#### GROW Output

- Beats (mutated, with intersections)
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
  # Identity
  story_title: string                # generated title for the story

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
- Relevant shadows for active dilemmas (derived from path definitions)
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

**Purpose:** Generate presentation layer — art direction, illustrations, and codex.

**Input:** Completed prose (all passages), entities, vision, codewords.

**Output:** ArtDirection, EntityVisual[], IllustrationBrief[], Illustration[], Codex[], Depicts edges, HasEntry edges.

> See `docs/design/procedures/dress.md` for the full algorithm specification.

#### Sub-stages

DRESS has three sub-stages with two human gates:

| Sub-stage | Purpose | Creates | Cost |
|-----------|---------|---------|------|
| **Art Direction** | Establish visual identity | ArtDirection, EntityVisual[] | Cheap (LLM text) |
| **Illustration Briefs + Codex** | Generate prompts and encyclopedia | IllustrationBrief[], Codex[] | Cheap (LLM text) |
| **Image Generation** | Render selected briefs | Illustration[] | Expensive (image API) |

**Sub-stage 1: Art Direction** (discuss/summarize/serialize)

Collaborative exploration of visual style, resulting in a global ArtDirection document and per-entity EntityVisual profiles. Follows the standard three-phase pattern (like DREAM).

**Human Gate 1:** Review visual identity and entity visual profiles.

**Sub-stage 2a: Illustration Briefs** (per passage, LLM)

Generates structured image prompts (IllustrationBrief nodes) for all passages with hybrid priority scoring. Reads Appears edges to identify which entities are in each passage and injects EntityVisual prompt fragments for consistency.

**Sub-stage 2b: Codex Generation** (per entity, LLM — parallel with 2a)

Generates cumulative, rank-ordered, diegetic encyclopedia entries for all entities. Spoiler-graduated via codeword-gated tiers.

**Human Gate 2:** Review illustration briefs (sorted by priority), select image generation budget, review codex entries.

**Sub-stage 3: Image Generation** (batch with sample-first, image provider)

Assembles provider-specific prompts from briefs + art direction + entity visuals. Generates one sample image for style confirmation, then batches the remainder. Stores assets as content-addressed files. Creates Illustration nodes + Depicts edges.

**Diegetic constraint:** Both illustration captions and codex entries must be written in the story's voice — as if they are part of the world, not meta-descriptions.

**Image provider:** Uses a provider-independent abstraction layer (not LangChain — no BaseImageModel exists). Single provider per project, starting with OpenAI gpt-image-1.

**Human Gate (overall):** Two gates — after art direction (Gate 1) and after brief generation (Gate 2). Optional — story works without DRESS.

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
| Illustration | id, asset, caption, category |
| CodexEntry | id, rank, visible_when, content |
| ArtDirection | id, style, medium, palette, aspect_ratio |

**Edges required:**

| Edge | Required fields |
|------|-----------------|
| Choice | from, to, label, requires_codewords, grants |
| Appears | from (entity), to (passage), role |
| Involves | from (relationship), to (passage) |
| Depicts | from (illustration), to (passage) |
| HasEntry | from (codex_entry), to (entity) |

**Derived:**

| Derived | From |
|---------|------|
| Codex display | CodexEntry nodes per entity, filtered by player codewords, sorted by rank (cumulative) |
| Start passage | Passage with no incoming Choice edges |
| Ending passages | Passages with no outgoing Choice edges |

**Ignored by SHIP:**
- Dilemmas, paths, beats, arcs
- `from_beat`, `summary` on passages
- `sequenced_after` edges between beats
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

## Post-Convergence Variation

When arcs reconverge at a shared beat, prose must work for all arriving paths.
QuestFoundry uses two mechanisms controlled by the prose layer:

**Variant routing** (`split_and_reroute`): For `residue_weight: heavy` dilemmas,
incoming choices to a shared passage are cloned per variant with codeword gates.
Each variant passage contains state-specific prose. The original passage may be
kept as a fallback for arcs with no applicable variant.

**Residue beats** (GROW Phase 8d): For `residue_weight: light` dilemmas, short
path-specific passages are inserted before the shared convergence point. Each
residue beat carries forward the emotional tone of its arc so the shared passage
can remain neutral.

**Cosmetic dilemmas** (`residue_weight: cosmetic`): No post-convergence variation.
Shared passages must not reference the choice at all. These dilemmas are filtered
out of residue candidate generation.

Entity **overlays** (codeword-gated attribute overrides) handle small state
differences like appearance or mood that carry across convergence boundaries.

---

## Anti-Patterns: What NOT to Build

### ❌ Agent Negotiation

Do not build systems where multiple LLM agents propose and negotiate.

### ❌ Path Creation in GROW

Do not allow GROW to create new paths. All paths are declared in SEED.

### ❌ Unbounded Iteration

Do not allow "keep generating until good." Quality comes from good prompts and human curation, not infinite loops.

### ❌ Backflow

Do not allow later stages to modify nodes they don't own. Each node type has a creating stage (see Stage Operations table). If GROW reveals a problem with SEED's paths, the human must manually revert to pre-GROW snapshot and revise SEED.

### ❌ Hidden Prompts

Do not embed prompts in code. Prompts live in `/prompts` as readable files.

### ❌ Complex State Objects

Do not build elaborate state machines or object graphs. State is flat YAML.

### ❌ Surface Choices Without Dilemma

Do not generate choices that are purely navigational. Every meaningful choice should connect to a dilemma, even if the player doesn't see the connection explicitly.

---

## Implementation Risks

### GROW Stage Complexity

GROW is the highest-risk area. The spec defines operations (Mark, Merge, Create) but the triggering heuristics need careful design.

**The Intersection Problem:**

How does the system determine that beats from different paths are compatible enough to merge into a single scene (intersection)?

- **Risk:** Pure LLM intuition will hallucinate connections.
- **Requirement:** Rigid compatibility checks before LLM involvement.
- **Signals for intersection candidacy:**
  - Shared entity (same character appears in both beats)
  - Shared location
  - Compatible time-deltas
  - No conflicting `sequenced_after` constraints
  - Not a gap beat (`is_gap_beat: false` or absent) — gap beats have path-local `sequenced_after` predecessors and are never eligible as intersection beats

**The Sequencing Clarification:**

Weaving paths into a sequence is *not* an LLM problem. Given a set of beats for an arc, topological sort on `sequenced_after` constraints produces the order deterministically.

**Intersection eligibility constraint:** A beat B may participate in an intersection spanning paths P₁…Pₙ only if every beat in B's `sequenced_after` list is itself reachable from all of P₁…Pₙ. Gap beats (`is_gap_beat: true`) are path-local by construction — their `sequenced_after` predecessors exist on a single path only — and are therefore never eligible as intersection beats.

What the LLM actually decides:
1. **Intersection creation:** "These beats should be one scene" (merge/mark)
2. **Gap detection:** "Path X needs a beat here" (create)
3. **Pruning:** "This beat doesn't fit" (cut)

The LLM does not pick ordering—that's derived from the graph.

### Overlay Conflict Resolution

Overlays from different paths can conflict when both are active.

**Path exclusivity handles some conflicts:**
Codewords from paths sharing a dilemma are exclusive—player can never have both. No conflict possible.

**Cross-dilemma conflicts:**
Codewords from *different* dilemmas can coexist. If their overlays set the same key, conflict resolution applies:
1. Most specific overlay wins (most matching codewords)
2. If tie, later in list wins

**Validation catches design errors:**
At validation, compute all valid codeword combinations. Flag any overlay pairs that can both be active with conflicting values. Human can:
- Restructure paths to make codewords exclusive
- Add compound overlay for the specific combination
- Accept list-order as tiebreaker

### FILL Stage Context

See Stage 5: FILL for complete context specification (voice document, sliding window, lookahead strategy).

**Intersection awareness:** When FILL generates prose for an intersection, the brief must include:
- Which paths this beat serves (X and Y)
- The dilemma impacts for each path
- Guidance to weave both paths' concerns into one scene

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
| Tags (paths, dilemmas, entities) | ~20-30 |
| Logic (sequenced_after, grants) | ~10-20 |
| YAML overhead | ~10 |
| **Total per beat** | **~100-130** |

**Full context for arc generation (medium story, 80-100 beats):**

| Component | Content | Est. Tokens |
|-----------|---------|-------------|
| System prompt | GROW rules, intersection definition, constraints | ~1,500 |
| Path definitions | 8-12 paths × ~60 tokens | ~500-700 |
| Dilemma definitions | 4-6 dilemmas × ~100 tokens | ~400-600 |
| Entity base definitions | 15-20 entities × ~80 tokens | ~1,200-1,600 |
| Relationship definitions | 10-15 relationships × ~50 tokens | ~500-750 |
| Codeword definitions | All codewords | ~300-500 |
| Arc history | Last 5 beats placed (continuity) | ~600 |
| Candidate beats | Unplaced beats for active paths | ~2,500-8,000 |
| **Total** | | **~8,000-15,000** |

**Worst case:** Early GROW with 60-80 unplaced beats: ~15,000-18,000 tokens.

### Verdict

Even pessimistic estimates use <15% of modern context windows (128k-200k). **No RAG or aggressive chunking needed for GROW.**

The entire relevant subgraph (all unplaced beats for involved paths) can be passed in every API call. This validates the "per arc" generation approach—the model can hold full arc state in working memory.

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
| Path/dilemma definitions | ~1,300 |
| Output buffer (safety margin) | ~2,000 |
| **Fixed cost** | **~7,150** |
| **Available for beats** | **~24,850** |

At ~140 tokens per beat: **~177 beats maximum**

This is equivalent to a complete 40,000-word novella or a dense 2-hour branching game. Not a toy.

**Quality degradation risks (8B models):**

- **Needle in haystack:** Model struggles to find connections between beats at opposite ends of context
- **Instruction drift:** As context fills, model may forget constraints (e.g., Path Freeze rule)

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

An arc is a complete weave of compatible paths. The process:
1. Generate spine arc (primary path combination)
2. Validate spine arc
3. For each exclusive path not in spine, generate divergent arc
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

The project uses a **unified graph** stored as a SQLite database, with snapshots
for rollback and optional human-readable exports.

```
/project
  graph.db                # The unified story graph (SQLite, single source of truth)
  /snapshots              # Pre-stage snapshots for rollback
    pre-dream.db
    pre-brainstorm.db
    pre-seed.db
    pre-grow.db
    pre-fill.db
    pre-dress.db
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

The `graph.db` file is a SQLite database with tables for nodes, edges, and
metadata. Each node is stored as an ID + JSON data blob; edges are stored as
typed (edge_type, from_id, to_id) triples. A mutation audit trail records all
graph changes with timestamps.

### Snapshot Strategy

Before each stage runs:
1. Copy current `graph.db` to `snapshots/pre-{stage}.db`
2. Run stage (graph modified via `SqliteGraphStore`)
3. On success, changes are committed to `graph.db`
4. If stage fails, `graph.db` unchanged (snapshot available for manual recovery)

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
| ArtDirection | Yes | DRESS | Yes |
| Illustration | Yes | DRESS | Yes |
| CodexEntry | Yes | DRESS | Yes |
| EntityVisual | No | DRESS | No |
| IllustrationBrief | No | DRESS | No |
| Dilemma | No | BRAINSTORM | No |
| Answer | No | BRAINSTORM | No |
| Consequence | No | SEED | No |
| Path | No | SEED | No |
| Beat | No | SEED/GROW | No |
| Arc | No | GROW | No |

### All Edge Types

| Edge | Persistent | Created in | Required for SHIP |
|------|------------|------------|-------------------|
| Choice | Yes | GROW | Yes |
| Appears | Yes | GROW | Yes |
| Involves | Yes | GROW | Yes |
| Depicts | Yes | DRESS | Yes |
| HasEntry | Yes | DRESS | Yes |
| describes_visual | No | DRESS | No |
| from_brief | No | DRESS | No |
| targets | No | DRESS | No |
| involves (dilemma) | No | BRAINSTORM | No |
| has_answer | No | BRAINSTORM | No |
| explores | No | SEED | No |
| has_consequence | No | SEED | No |
| belongs_to | No | SEED/GROW | No |
| sequenced_after | No | SEED/GROW | No |
| grants (beat) | No | GROW | No |
| weaves | No | GROW | No |
| from_beat | No | GROW | No |

### Dilemma → Path → Beat Flow

```
Dilemma (BRAINSTORM)
  dilemma::mentor_trust
  "Can the mentor be trusted?"
  ├─ answer: mentor_protector
  └─ answer: mentor_manipulator
        ↓ SEED triage
Path (SEED)
  path::mentor_trust__protector          ← hierarchical ID encodes parent
    dilemma_id: dilemma::mentor_trust    ← derivable from path_id
    answer_id: mentor_protector
    shadows: [mentor_manipulator]     ← unexplored answer
        ↓ SEED/GROW
Beat (SEED/GROW)
  mentor_reveals_truth
    paths: [path::mentor_trust__protector]
    dilemma_impacts:
      - dilemma_id: dilemma::mentor_trust
        effect: commits
        note: "Player learns mentor sent warning to family"
```

---

## Open Questions

1. **GROW algorithm specifics:** Compatibility checks, intersection detection heuristics, and the exact flow from SEED to completed arcs need detailed specification. (Next priority.)

2. **Human gates within GROW:** How many gates? After each arc? After all arcs? TBD.

3. **Prompt compilation architecture:** The Vision doc's prompt compilation system (components, templates, sandwiching, context budgets) is likely still valid but needs review against the new ontology.

### Resolved

- **Iteration approach:** Arc-at-a-time (see Design Decisions)
- **Codeword granularity:** Boolean only for v5.0 (see Design Decisions)
- **Context budget:** Full subgraph fits in context, no RAG needed (see Context Budget Analysis)
- **Overlay conflicts:** Most specific wins, then list order; validation flags potential conflicts (see State Pattern)
- **Dilemma cardinality:** Exactly two answers (binary); nuance via multiple dilemmas
- **Entity creation in GROW:** Not allowed; GROW cannot create Entity nodes
- **Derived codewords:** Deferred to future version (forward compatible)

---

## Future Bolt-ons

- **Numeric state:** `modifies` on choices, `derived` codewords
- **Choice visibility:** `visible_when` separate from `requires_codewords`
- **Entity absence:** overlay with `present: false`
- **Audit log:** Mutation history (separate from graph)
- **GROW recovery:** Preserve partial GROW work when aborting to SEED (currently discards all)
- **Per-arc voice variations:** Voice document modifiers per arc (e.g., "mentor_manipulator" arc uses colder, more paranoid voice)
