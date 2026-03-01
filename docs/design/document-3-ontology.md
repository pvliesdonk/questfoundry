# Story Graph Ontology — Data Model for Branching Fiction

> **Status: Authoritative.** This document, together with [Document 1](how-branching-stories-work.md), is the authoritative source of truth for QuestFoundry's graph ontology. Where other design documents contradict this one, this document takes precedence. See [Issue #977](https://github.com/pvliesdonk/questfoundry/issues/977).

## Guiding Principle

The graph serves the story. Every node type, edge type, and property in this ontology exists because a narrative concept from [Document 1](how-branching-stories-work.md) requires it. If a graph concept cannot be traced to a narrative purpose, it does not belong.

This document translates Document 1's storytelling language into a formal data model. It is not a replacement for Document 1 — it is a companion. Document 1 describes what authors are trying to accomplish. This document describes how the graph represents their work.

The direction is always: **narrative concept first, graph representation second.** The current ontology was built partly in the other direction (graph structure first, narrative meaning mapped onto it). Where this document diverges from the current implementation, the narrative intent takes precedence.

---

## Part 1: Primitive Concepts

These are the fundamental building blocks of the graph. Each traces directly to a Document 1 narrative concept.

### Vision

A singleton configuration node. Stores the creative contract established in DREAM: genre, subgenre, tone, themes, audience, scope, style preferences, content guidance, and an optional point-of-view hint.

The vision's fields include:
- **Genre and subgenre** — the primary genre (e.g., "mystery") and a more specific subgenre (e.g., "cozy mystery," "noir detective"). Document 1 discusses subgenre narratively; this document formalizes it as a distinct field.
- **Point-of-view style** — a hint, not a binding constraint. Expressed as one of four narrative perspectives (first person, second person, third person limited, third person omniscient). FILL's voice document makes the final decision — the vision's `pov_style` is advisory context.
- **Content notes** — explicit guidance on what the story should include or exclude (themes to embrace, topics to avoid, content boundaries). This is a substantive creative constraint, not a filter — it shapes what BRAINSTORM generates and what SEED retains.
- **Scope** — expressed as a named preset (e.g., "vignette," "short story," "novella") that implies approximate sizes for the cast, dilemma count, beat count, and passage count. The preset system provides BRAINSTORM and SEED with concrete targets.

The vision has no edges to other nodes. Downstream stages receive it as context — it informs decisions but does not participate in the graph structure. It is working data, not exported to the player.

The voice document (created by FILL) follows the same pattern: a singleton configuration node that governs prose style. It is the operational descendant of the vision — where the vision says "gritty noir," the voice document says "second person, present tense, short sentences, no semicolons."

### Entity

A character, location, object, or faction that populates the story world. Entities are created in BRAINSTORM and persist through to export — they are among the few node types the player's runtime needs.

Each entity carries a **base state**: the facts true regardless of player choices. Name, concept, appearance, personality. FILL adds micro-details to the base state as they are discovered during prose writing (the mentor smokes, the spy has a limp). Once discovered, these details are global — they apply on every arc.

Entities also carry **overlays**: conditional state activated by state flags. "When the mentor is hostile: demeanor is cold, dialogue style is curt." Overlays are the mechanism for representing how choices change the world. The entity remains one node — overlays add or modify properties, they do not create a second entity.

The entity's category (character, location, object, faction) is part of its identity and serves as a namespace: `character::mentor` and `location::mentor` are different nodes.

**Persistent.** Exported by SHIP. The export includes base state, overlays, and whatever FILL and DRESS added. Working metadata (disposition, triage notes) is not exported.

### Dilemma

A binary dramatic question with exactly two compelling answers. The central structural unit of the story's branching.

Each dilemma carries:
- The **question** ("Can the mentor be trusted?")
- Exactly two **answers** (each an answer node linked by `has_answer` edges)
- **Why it matters** — the stakes that make the choice meaningful and the seed of residue
- **Anchored-to edges** — links to the entities central to this dilemma

The `anchored_to` edges are proper graph edges (dilemma → entity), not embedded ID lists. This makes "which dilemmas involve this entity?" a direct graph query. During SEED triage, if an anchored entity is cut, the dilemma must either re-anchor to a surviving entity or be cut itself — a dilemma anchored to nothing is meaningless.

Each dilemma also carries a **role** and associated structural properties, discussed in Part 2.

**Working.** Dilemmas are consumed by the pipeline. By SHIP, they have been absorbed into the story structure — paths, beats, choices. The player never sees "dilemma" as a concept.

### Answer

One possible response to a dilemma. Exactly two per dilemma, linked by `has_answer` edges. Each answer has a description of what this response means narratively.

One answer per dilemma is marked **canonical** — this is the first answer explored when FILL writes prose along its first complete arc. Canonical does not mean primary, default, or more important. It is an authoring convenience: the first path written establishes shared passages, and other paths write toward them. Every answer is narratively equal.

**Working.** Consumed by SEED when creating paths.

### Path

One answer to a dilemma, explored as a complete storyline. Created by SEED when it decides which answers to develop. Each path links to:
- The answer it explores (via `explores` edge)
- The consequences it implies (via `has_consequence` edges)

A path is a container for beats — the sequence of story moments that proves this answer. But the path itself is a working concept: after GROW interleaves beats into a DAG, the path's identity is encoded in beat membership and state flags, not in a separate path node that the player traverses.

An answer that has no path exploring it is a **shadow** — the road not taken. Shadows are derivable (an answer node with no `explores` edge pointing at it) and do not need a dedicated node type. Context builders must find them to provide FILL with the narrative weight of unexplored alternatives.

**Working.** Consumed by GROW.

### Beat

A concrete story moment that advances a dilemma toward resolution. The fundamental unit of the pipeline from SEED onward. Everything downstream — interleaving, intersection, passage creation, prose — operates on beats.

Each beat carries:
- A **summary** of what happens
- **Dilemma impacts** — which dilemma it serves, and how (advances, reveals, commits, or complicates)
- **Path membership** — which path this beat belongs to (via `belongs_to` edge)
- **Entity references** — which entities are present
- **Working annotations** that are consumed during the pipeline and do not persist (see Part 3)

Beats have a lifecycle. SEED creates them as scaffold. GROW adds ordering and intersection relationships. POLISH may add new beats (micro-beats for pacing, residue beats for mood-setting) and groups beats into passages. FILL writes prose for the passages that contain them. Throughout this lifecycle, the beat's identity — what happens — remains stable. The metadata around it evolves.

Beat subtypes (distinguished by a role marker):
- **Regular beat** — a story moment from SEED's scaffold
- **Micro-beat** — a brief transition added by POLISH for pacing
- **Residue beat** — a mood-setter added by POLISH before a shared passage, carrying state-flag-specific prose hints

**Working.** Beats are not exported. They are the authoring abstraction. The player sees passages.

### Consequence

The narrative outcome of a path choice. Created by SEED, linked to a path via `has_consequence` edge. "The mentor becomes your adversary" is a consequence of the distrust path.

Consequences are the bridge between narrative stakes (Document 1's "why it matters") and mechanical state tracking (state flags). Each consequence becomes one or more state flags in GROW.

**Working.** Consumed when state flags are derived.

### State Flag

An internal boolean marker representing a world state that resulted from a committed choice. Derived from consequences during GROW. "The mentor is hostile" is a state flag.

State flags serve two purposes:
- **Routing** — gating choice edges and variant passages so the player sees the right content after soft dilemma convergence
- **Entity overlays** — activating conditional entity state so FILL writes the correct version of an entity

State flags exist for both hard and soft dilemmas. For soft dilemmas, they drive routing after convergence. For hard dilemmas, the graph structure handles routing (paths never rejoin), but state flags still activate entity overlays — the mentor entity is one node, and the overlay needs a flag to know which version to present.

One state flag per soft dilemma suffices for routing: present means path A was taken, absent means path B. Hard dilemmas need flags only for overlay activation.

**Internal.** State flags are implementation machinery. The player does not interact with them in digital formats.

### Codeword

A player-facing state marker used in gamebook (print) formats. The player writes down or marks off a codeword to track their choices, then checks for it at later decision points.

Codewords are a **projection** of state flags — a curated subset surfaced to the player. Not every state flag becomes a codeword. Hard dilemma state flags typically do not need codewords because the gamebook's page structure handles routing (you are physically on a different page). Soft dilemma state flags become codewords because the player must carry state across a convergence point where pages rejoin.

SHIP decides which state flags become player-facing codewords based on dilemma role and convergence structure.

POLISH may also create **cosmetic codewords** — tokens that give the player a feeling of agency ("Write down MOONLIT") without any routing consequence. These are narrative seasoning, not structural.

The total number of codewords is naturally bounded by the number of soft dilemmas — typically well under ten. This keeps the gamebook playable without a spreadsheet.

**Persistent (when present).** Exported by SHIP for gamebook formats. In digital formats, the engine tracks state flags silently and codewords may not exist at all.

### Character Arc Metadata

A per-entity summary of how a character changes across the story, synthesized by POLISH from the beat structure. "The mentor begins as a cryptic authority figure, is gradually revealed as either a protector or a manipulator (depending on path), and ends as either a trusted ally or a defeated adversary."

Character arc metadata is stored as an annotation on entity nodes — it describes the entity's trajectory on each path (start → pivot → end). It is working data for FILL: when the prose writer encounters the mentor in a mid-story scene, they need to know where the mentor has been and where the mentor is going. Without it, the writer sees individual beats in isolation and risks inconsistency.

**Working.** Created by POLISH, consumed by FILL. Not exported.

### Scene Blueprint

A per-passage writing plan created by FILL before prose generation. Each blueprint captures the sensory palette (sight, sound, smell), character gestures, the opening move (dialogue, action, sensory image, or internal thought), a craft constraint, and a one-word emotional arc.

Scene blueprints are working data for FILL's own process — they structure the writing of each passage without affecting the graph. FILL creates them in a planning phase and consumes them during prose generation. They are not passed to other stages.

**Working.** Created and consumed within FILL. Not exported.

---

## Part 2: Dilemma Ordering and Relationships

Not all dilemmas play the same role in a story. Document 1 distinguishes hard dilemmas (the backbone) from soft dilemmas (the subplots), and notes that the ordering of dilemmas has profound structural consequences. This section defines how the graph represents these roles and relationships.

### Dilemma Role

Each dilemma carries a **role** that determines its structural behavior:

**Hard** (backbone) — The central dramatic questions the story is about. They introduce early, commit late (at or near the climax), and carry heavy residue. Paths of a hard dilemma never structurally converge — the worlds are too different. After a hard dilemma commits, the story carries separate beat sequences to separate endings.

**Soft** (subplot) — The secondary questions that enrich the journey. They introduce later, commit earlier, and carry lighter residue. Paths of a soft dilemma reconverge after enough payoff beats — the storylines come back together, though residue persists in prose.

The role is the primary concept. Convergence behavior is **derived**: hard means paths never converge, soft means paths do converge. If a dilemma's paths cannot meaningfully reconverge (the residue is too heavy, the worlds too different), it is hard by definition regardless of narrative intent. Conversely, if paths can reconverge, the dilemma is soft.

This replaces the current `convergence_policy` field. Rather than declaring convergence behavior directly (which invites graph/narrative conflation — "convergence" means different things in graph theory and storytelling), the author declares the dilemma's narrative role, and convergence behavior follows.

### Flavor Choices

Document 1 also describes flavor-level choices that barely diverge — the choice affects tone and details but not which beats the player experiences. These are not full dilemmas in the ontological sense. They are handled by cosmetic state flags and minor prose variation, without the structural machinery of paths, commits, and convergence. POLISH creates them as false branches or minor passage variants.

### Pairwise Relationships

Dilemmas interact with each other. SEED declares these pairwise relationships:

**Wraps** — Dilemma A wraps dilemma B when A introduces before B and B resolves before A. The backbone wraps the subplots: the central question is present from the beginning and resolves at the climax, while secondary questions weave through the middle. Wrapping is a partial order — if A wraps B, A is the outer dilemma.

**Concurrent** — Neither dilemma wraps the other. Both are active at the same time, interleaving but without a nesting relationship. Two hard dilemmas might be concurrent — both introduce early and commit late, their storylines intertwined.

**Serial** — Dilemma A resolves (commits and converges) before dilemma B introduces. The two never interact structurally — they are independent subplots experienced in sequence. Serial soft dilemmas are a major complexity reducer: they never multiply each other's beat count.

**Shared entity** — Two dilemmas are anchored to the same entity. This is not an ordering relationship but a signal for intersection potential: if both dilemmas involve the mentor, their beats may naturally share scenes.

The three ordering relationships (wraps, concurrent, serial) are declared by SEED as **hints** for GROW's interleaving. They express the author's intent for how dilemmas relate in the story's timeline. GROW uses them to guide beat placement; POLISH may adjust within the constraints of the finalized beat DAG. The shared-entity signal requires no explicit declaration — it is derived from the `anchored_to` edges already present in the graph.

### Residue Weight

Orthogonal to the dilemma's role, each dilemma carries a **residue weight** that governs how much prose varies after convergence (for soft dilemmas) or at intersections (for hard dilemmas):

- **Heavy** — genuinely different passages needed. The worlds are too different for one passage to serve both honestly.
- **Light** — a residue beat before a shared passage sets the mood. The shared passage itself can work for both.
- **Cosmetic** — tiny differences handled in prose. Barely affects anything.

Residue weight and dilemma role are independent axes. A soft dilemma can have heavy residue at specific moments (paths reconverge structurally, but some passages need variants). A hard dilemma might have cosmetic residue at an intersection (the dilemma matters enormously for the plot, but at this particular shared scene, the difference is minor).

### Ending Salience

Each dilemma also carries an **ending salience** — how much the story's ending should differ based on this dilemma's resolution:

- **High** — endings must differ meaningfully.
- **Low** — endings may acknowledge the choice but do not structurally differ.
- **None** — endings must not reference this choice.

Hard dilemmas typically have high ending salience. Soft dilemmas vary. Ending salience informs GROW's routing decisions for terminal passages.

---

## Part 3: The Beat DAG — Core Structural Artifact

The beat DAG (directed acyclic graph) is the central artifact of the pipeline. SEED creates the initial beats. GROW weaves them into a coherent structure. POLISH refines and augments them. Everything else — passages, choices, arcs — is derived from this DAG.

### What the DAG Represents

Each node in the DAG is a beat. Each directed edge means "this beat comes before that beat" — a predecessor/successor relationship. The DAG encodes every valid ordering of story moments across all possible playthroughs.

A beat with two successors (one per path of a dilemma) represents a **divergence**: the story splits at the commit. A beat with two predecessors (from different paths) represents a **convergence**: the storylines rejoin. These structural moments are not separate node types — they are visible in the DAG's topology. The commit beat IS the divergence point. The first shared beat after payoff beats IS the convergence point.

### Beat Lifecycle

A beat passes through several stages, accumulating and shedding metadata:

**Created by SEED:**
- Summary, dilemma impacts, path membership (`belongs_to` edge), entity references
- **Working annotations** consumed by GROW:
  - Entity flexibility (substitution edges to alternative entities — "the spy could be the informant")
  - Temporal hints (position relative to other dilemmas — "should come before dilemma B commits")

**Enriched by GROW:**
- Ordering edges (predecessor/successor relationships in the DAG)
- Intersection groupings (co-occurrence with beats from other paths — see Part 4)
- State flag associations (which flags are active when this beat is reached)

**Augmented by POLISH:**
- New beats may be added (micro-beats for pacing, residue beats for mood-setting)
- Ordering edges may be adjusted within linear sections
- Beats are grouped into passages (see Part 5)

**Consumed by FILL:**
- FILL receives passages (which contain beats). It writes prose for the passage, informed by the beats' summaries, entity references, and state context.

Throughout this lifecycle, the beat's core identity — what happens in this story moment — remains stable. The metadata around it evolves as each stage adds its contribution.

**Stage attribution clarification:** When the Node Types table (Part 9) says "Created by: SEED, POLISH," it means both stages create beat nodes — SEED creates regular beats, POLISH creates micro-beats and residue beats. It does NOT mean POLISH mutates every beat SEED created. POLISH may adjust ordering edges on existing beats and adds new beats to the DAG, but it does not rewrite existing beat summaries or dilemma impacts. Pipeline validation tools can use stage attribution on individual beats (`created_by` property) to distinguish SEED-authored beats from POLISH-authored beats.

### Temporal Hints — The SEED→GROW Contract

SEED creates beats for individual paths. Each beat's position relative to its own dilemma is clear from its function: an "advances" beat comes before the commit, a "commits" beat IS the commit, a "consequence" beat comes after. But a beat's position relative to *other* dilemmas is not yet determined — SEED hasn't interleaved the paths.

SEED expresses temporal intent through **hints**: advisory annotations that tell GROW where this beat should fall relative to other dilemmas' commits. These hints interact with the dilemma ordering relationships (Part 2):

- If dilemma A wraps dilemma B, then A's introduction beats should come before B's, and B's resolution should come before A's climax.
- If two dilemmas are serial, their beats do not overlap in time.
- If two dilemmas are concurrent, their beats interleave freely.

The hints are **consumed by GROW** during interleaving. Once GROW produces the DAG with a total order per arc, the temporal positions are structural facts readable from the ordering — the hints have served their purpose and are not carried forward. POLISH may reorder beats within linear sections of the DAG, using its fuller knowledge of the emerging story. It is not bound by SEED's initial hints.

**Temporal hint schema:**

```yaml
temporal_hint:
  relative_to: <dilemma_id>          # The dilemma this hint is relative to
  position: before_commit | after_commit | before_introduce | after_introduce
```

- `before_commit` — this beat should be placed before `relative_to`'s commit beat
- `after_commit` — this beat should be placed after `relative_to`'s commit beat
- `before_introduce` — this beat should be placed before `relative_to`'s first beat
- `after_introduce` — this beat should be placed after `relative_to`'s first beat

Temporal hints are optional. A beat with no hint has no constraint on its placement relative to other dilemmas — GROW places it using structural heuristics and dilemma ordering relationships alone. Hints that conflict with dilemma ordering relationships (e.g., a hint saying "after B's commit" when A wraps B and A's commit comes first) are treated as advisory — GROW resolves the conflict in favor of the ordering relationship.

### The 2^N Law in Graph Terms

Document 1's central structural insight: any beat placed after N committed dilemmas exists in up to 2^N versions. In the DAG, this is visible as the branching factor:

- A beat before any commit has one predecessor path through the DAG — it exists once.
- A beat after one commit has predecessors from two branches — it exists in two versions (or is shared if it belongs to both paths).
- A beat after two commits has predecessors from four branches — up to four versions.

This is not a property of the beat itself but of its **position in the DAG relative to commit beats**. The same beat, moved earlier in the DAG (before a commit), would exist in fewer versions. This is why dilemma ordering matters: hard dilemmas committing late keeps most beats in the shared region, minimizing multiplication.

### Total Order Per Arc

The DAG defines a partial order. Each arc (a specific combination of path choices) defines a **total order** — the exact sequence of beats a player on that arc experiences. This total order is computed from the DAG by selecting, for each dilemma, one path's beats and ordering them according to the DAG's edges.

Arcs are not stored as graph nodes. They are **computed traversals** of the DAG. Any stage that needs an arc's beat sequence computes it on demand from the DAG structure. Diagnostic tools may snapshot pre-computed arc sequences for inspection, but pipeline stages must never read arcs from stored nodes — they traverse the DAG.

If pre-computed arc data is stored for debugging purposes, it uses a `materialized_` prefix to signal that it is derived, read-only, and may be stale.

---

## Part 4: Intersections

Document 1 describes intersections as "where independent storylines share a scene." If the mentor path has "the mentor gives cryptic advice" and the artifact path has "study the artifact's markings," and both could happen in the mentor's study — that is a natural intersection. One scene where both storylines advance simultaneously.

### What an Intersection Is

An intersection is a **co-occurrence declaration**: these beats from different paths happen at the same time, in the same scene. The beats do not merge into one beat. They remain separate beats, each serving their own path and dilemma, each carrying their own dilemma impacts. But they have high cohesion — they share a scene, and when POLISH creates passages, they will be grouped into one passage.

This is a critical distinction from the current implementation, which models intersections by cross-assigning `belongs_to` edges — making a beat "belong to" multiple paths. That conflates two different concepts:

- **Path membership** — "this beat is part of path A's storyline" (the beat advances path A's dilemma)
- **Co-occurrence** — "this beat happens at the same time as a beat from path B"

A beat that co-occurs with another path's beat does not become part of that path. It still advances its own dilemma. The intersection means the two beats share a scene, not that they share a purpose.

### Graph Representation

An intersection is represented as a grouping relationship between beats:

- An **intersection group** links two or more beats from different paths that co-occur in one scene.
- Each beat retains its original `belongs_to` edge to its own path.
- The intersection group carries the resolved scene context: shared location, shared entities, and a rationale for why these beats work as one scene.

The grouping tells POLISH: "when you create passages, these beats should become one passage (or part of one passage)." It tells FILL: "write one scene that advances dilemma A through beat X AND dilemma B through beat Y simultaneously."

### How Intersections Are Found

GROW identifies intersection candidates using the signals SEED provided:

- **Shared entities** — two dilemmas anchored to the same entity (from `anchored_to` edges) naturally produce beats that involve the same character.
- **Entity flexibility** — SEED's substitution annotations ("the spy could be the informant") allow GROW to make two paths share a character they didn't originally share.
- **Location overlap** — beats from different paths that could happen in the same place.
- **Temporal co-occurrence** — beats that fall at roughly the same point in the story's timeline.

### Intersection and Convergence Policy

Intersections must respect dilemma roles. Two beats from the same hard dilemma's paths must never be grouped into an intersection — they are mutually exclusive by definition (the player is on one path or the other, never both). Beats from different dilemmas can always intersect, regardless of those dilemmas' roles. And beats from the same soft dilemma can intersect only if they are in the shared region (before commit or after convergence).

This constraint is structural, not a guideline. Violating it produces a scene that is impossible to reach — the player cannot be on both paths of a hard dilemma simultaneously.

---

## Part 5: The Passage Layer

The player does not see beats. The player sees **passages** — prose units with choices between them. The passage layer is built by POLISH on top of the beat DAG. It is the bridge between the authoring abstraction (beats) and the player experience (prose with choices).

### Passages

A passage is a prose container holding one or more beats. It is what FILL writes and what the player reads.

Passages are created by POLISH through two mechanisms:

- **Grouping by intersection** — beats that co-occur (from intersection groups declared in GROW) become one passage. The passage contains beats from different paths, and FILL writes one scene that advances multiple storylines.
- **Grouping by collapse** — sequential beats from the same path with no choices between them become one passage. Three beats in a row — "search the study," "find the hidden letter," "read the letter" — collapse into one flowing scene. Collapse may produce multiple passages from a chain if the beats have incompatible entities or natural hard breaks.

A passage that contains a single beat is also valid — not everything collapses or intersects.

Each passage carries:
- The **beats** it contains (grouping edges)
- A **summary** derived from its beats
- **Entity references** from its constituent beats
- **Prose** (empty until FILL writes it)

**Persistent.** Passages are exported by SHIP. The export includes prose and structural connections (choices). Working metadata (beat grouping rationale, feasibility audit notes) is not exported.

### Choices

A choice is a directed edge between two passages: "from this passage, the player can go to that passage." Each choice carries:

- A **label** — the text the player sees ("Trust the mentor" / "Confront the mentor")
- **Requires** — state flags that must be active for this choice to be available (gating)
- **Grants** — state flags activated when the player takes this choice

Choices are created by POLISH based on the beat DAG's structure. Where the DAG diverges (a commit beat with successors on different paths), POLISH creates choices with appropriate labels and state flag grants. Where the DAG is linear, passages connect without meaningful choice — or POLISH inserts false branches for the experience of agency.

A choice's `requires` field is empty for most choices — the player can always take them. Gates appear after soft dilemma convergence, where the passage graph rejoins but some choices are only available to players who took a specific path. For hard dilemmas, gating is unnecessary — the passage graph itself is separate, so the player never encounters the "wrong" choice.

### Variant Passages

When heavy residue makes it impossible for one passage to serve all arcs honestly, POLISH creates **variant passages** — separate passages at the same structural position, each gated by different state flags. Same story moment, genuinely different prose.

A variant passage is a full passage in its own right — it contains beats, receives prose from FILL, and connects to the passage graph via choice edges. The gating (via `requires` on incoming choice edges) ensures the player sees the correct variant.

Variants are linked to a **base passage** so the relationship is explicit: "these passages are variants of each other, serving the same structural moment for different state combinations." This is a graph edge (`variant_of`), not a property — it allows traversal in both directions.

### Residue Beats and Residue Passages

When light residue affects how a shared scene should feel, POLISH inserts a **residue beat** before the shared passage. The residue beat is a brief mood-setter — "You enter the vault with confidence" (trust path) vs. "You enter the vault on guard" (distrust path) — that sets the emotional context without requiring the shared passage to vary.

In graph terms, a residue beat becomes a short passage placed before the shared passage, with one variant per path, each gated by the appropriate state flag. The shared passage that follows is ungated — both paths arrive there, and the residue beat has already established the emotional context.

### False Branches

Not every choice needs to be a real dilemma. POLISH creates **false branches** for the experience of agency without structural cost:

- **Diamond** — two choices from passage A lead to passages B and C, which both lead to passage D. The player picks, but the story arrives at the same place.
- **Sidetrack** — one choice goes directly to the next passage, the other takes a one-or-two-beat detour before rejoining. The player who detoured gets extra content but the story continues from the same point.

False branches involve no state flags — they are purely topological patterns in the passage graph.

### The Passage Graph

The complete passage layer — passages connected by choice edges — is a directed graph that the player traverses. It is derived entirely from the beat DAG plus POLISH's decisions about grouping, variants, and false branches.

This passage graph is what SHIP exports. Digital formats traverse it with an engine. Gamebook formats number the passages and print "turn to page X" choices with codeword checks.

---

## Part 6: Entity Overlays and State

Entities exist in a world that changes based on player choices. The mentor who is trusted behaves differently from the mentor who is distrusted. The graph must represent this without creating separate entity nodes for each possible state.

### Base State and Overlays

Every entity has a **base state**: the facts true regardless of player choices. Name, concept, role in the story. FILL enriches the base state with micro-details discovered during prose writing — the mentor smokes, the spy has a limp. These micro-details are global: once discovered, they are true on every arc. A character who smokes on the trust path also smokes on the distrust path.

**Overlays** represent conditional changes to an entity's state, activated by state flags. An overlay says: "when this state flag is active, add or change these properties." The mentor's overlay might say: "when `mentor_hostile` is active: demeanor is cold, dialogue style is curt, avoids eye contact."

The entity remains one node. Overlays modify it conditionally — they do not create a second entity. This is essential: every reference to `character::mentor` throughout the graph points to the same node. The overlay determines which version of the mentor appears in context.

**Implementation note:** Overlays are stored as an embedded list on the entity node, not as separate graph nodes. Each overlay is a dict with `when` (list of state flag IDs) and `details` (key-value property changes). This keeps the entity and all its conditional states as one atomic unit — consistent with the principle that the entity remains one node. A query like "which entities does state flag X affect?" requires scanning entity nodes, but at the scale this pipeline operates (a handful of overlays per story) this is not a performance concern.

### When Overlays Are Needed

Overlays are implied from BRAINSTORM onward — the dilemma's two answers inherently imply two states for the central entity. They become concrete in SEED, where path consequences describe how the entity changes. They are activated in GROW, where state flags are derived from consequences. And they are used by FILL, where the writer needs to know which version of the entity they are portraying.

Both hard and soft dilemmas produce overlays. For soft dilemmas, the overlay is activated by the routing state flag — the same flag that gates post-convergence choices. For hard dilemmas, the graph structure separates the paths, but the entity is still one node referenced from both sides. The hard dilemma's state flag activates the overlay so that FILL (and the player runtime) knows which version of the entity to present.

### Overlay Scope

An overlay activates based on one or more state flags. The simplest case: one state flag, one overlay. "When `mentor_hostile`: these properties change." The absence of the flag implies the other path's state — the base state serves as the default, or a second overlay covers the other path explicitly.

More complex cases arise when multiple dilemmas affect the same entity. If the mentor is central to both the trust dilemma and the artifact dilemma, the mentor might have overlays for each: "when `mentor_hostile`: cold and curt" and "when `artifact_destroyed`: grief-stricken." These compose — a player on the hostile-mentor, destroyed-artifact arc sees both overlays applied.

POLISH audits overlay composition for prose feasibility, following the same logic as passage feasibility: two or three active overlays are manageable, more than that and FILL cannot portray the entity coherently.

### What FILL Adds vs. What Overlays Track

FILL discovers micro-details during prose writing. These update the entity's base state — they are universal facts, not path-dependent. The distinction:

- **Base state** (FILL micro-details): "The mentor smokes." True everywhere. Not gated by state flags.
- **Overlay** (path-dependent): "The mentor is hostile." True only when the distrust path was taken. Gated by state flag.

If FILL discovers something that is path-dependent ("on the trust path, the mentor has a warm smile; on the distrust path, a thin-lipped grimace"), that is an overlay concern, not a base state update. FILL should not modify overlays — that is structural work belonging to earlier stages. FILL's entity updates are limited to universal micro-details.

---

## Part 7: Pipeline Operations on the Graph

Each pipeline stage reads from and writes to the graph. This section summarizes what each stage does — not the how (that belongs in procedure documents) but the what: which node types and edge types are created, modified, or consumed.

### DREAM

| | |
|---|---|
| **Creates** | Vision node (singleton) |
| **Reads** | Nothing |
| **Modifies** | Nothing |
| **Consumes** | Nothing |

DREAM produces the creative contract. One node, no edges.

### BRAINSTORM

| | |
|---|---|
| **Creates** | Entity nodes, dilemma nodes, answer nodes |
| **Edges created** | `has_answer` (dilemma → answer), `anchored_to` (dilemma → entity) |
| **Reads** | Vision node (for genre, tone, scope context) |
| **Modifies** | Nothing |
| **Consumes** | Nothing |

BRAINSTORM populates the world. The cast and the dramatic questions.

### SEED

| | |
|---|---|
| **Creates** | Path nodes, consequence nodes, beat nodes |
| **Edges created** | `explores` (path → answer), `has_consequence` (path → consequence), `belongs_to` (beat → path), entity flexibility edges (beat → alternative entities) |
| **Reads** | All BRAINSTORM output (entities, dilemmas, answers) |
| **Modifies** | Entity nodes (disposition: retained/cut), dilemma nodes (role, residue weight, ending salience) |
| **Declares** | Dilemma pairwise relationships (wraps/serial/concurrent/shared_entity), temporal hints on beats |

SEED is the heaviest mutation stage. It triages, scaffolds, orders, and sketches convergence. Its output is the raw material for GROW: independent paths with complete beat scaffolds, annotated with flexibility and temporal hints.

### GROW

| | |
|---|---|
| **Creates** | Ordering edges (beat → beat), intersection groups, state flags |
| **Edges created** | Predecessor/successor edges in the beat DAG, intersection grouping edges, entity overlay nodes with state flag activation |
| **Reads** | All SEED output (paths, beats, consequences, dilemma relationships, temporal hints) |
| **Modifies** | Beat nodes (enriched with intersection membership) |
| **Consumes** | Entity flexibility annotations (used to find intersections, then discarded), temporal hints (used for interleaving, then discarded) |
| **Validates** | Every computed arc traversal is complete and has no dead ends |

GROW produces the beat DAG — the core structural artifact. It weaves independent paths into one coherent branched structure, identifies intersections, derives state flags from consequences, and creates entity overlays. It does not create passages or choices — that is POLISH's job.

### POLISH

POLISH operates in two phases:

**Phase 1 — Finalize the beat DAG:**

| | |
|---|---|
| **Creates** | Micro-beats (pacing), residue beats (mood-setters) |
| **Edges created** | New ordering edges for inserted beats |
| **Reads** | The beat DAG, intersection groups, state flags, entity overlays, dilemma residue weights |
| **Modifies** | Ordering edges (reordering within linear sections) |

**Phase 2 — Build the passage layer:**

| | |
|---|---|
| **Creates** | Passage nodes, choice edges, variant passages |
| **Edges created** | Beat → passage (grouping), passage → passage (choices with labels/gates/grants), `variant_of` (variant → base passage) |
| **Reads** | Finalized beat DAG, intersection groups, state flags |
| **Decides** | Passage grouping (collapse + intersection), prose feasibility, variant vs shared vs residue beat, false branch placement, character arc metadata |

POLISH transforms the beat DAG into the passage graph. After POLISH, every passage is defined, every choice is wired, and every variant is created. The structure is ready for prose.

### FILL

| | |
|---|---|
| **Creates** | Voice document (singleton) |
| **Reads** | Everything — passages, beats, entities with overlays, state flags, character arc metadata, vision |
| **Modifies** | Passage nodes (writes prose), entity nodes (adds universal micro-details to base state) |
| **Consumes** | Character arc metadata, scene blueprints (working data for writing process) |

FILL is primarily a consumer. It reads the complete graph and writes prose into passages. Its only structural contribution is enriching entity base state with universal micro-details discovered during writing.

### DRESS

| | |
|---|---|
| **Creates** | Art direction node (singleton), entity visual nodes, illustration nodes, codex entry nodes |
| **Reads** | Passages (prose), entities, vision |
| **Modifies** | Nothing structural |

DRESS adds visual identity and reference material. It does not change the story.

### SHIP

| | |
|---|---|
| **Creates** | Export files (Twee, HTML, JSON, gamebook PDF) |
| **Reads** | All persistent nodes: passages (with prose), choice edges, entities (with overlays), state flags, codewords, illustrations, codex entries, art direction |
| **Modifies** | Nothing |
| **Decides** | Which state flags become player-facing codewords (for gamebook format) |
| **Exports** | The player-facing subset of each persistent node's fields |

SHIP reads the graph and produces playable output. It defines the persistent/working boundary — a node is persistent if SHIP exports it. Some persistent nodes have working fields that SHIP does not export.

---

## Part 8: Where the Mapping Breaks

These are places where the intuitive graph interpretation diverges from the narrative meaning. Each is a documented danger zone — a place where an LLM or developer is likely to conflate graph concepts with narrative concepts, producing bugs that are architecturally reasonable but narratively wrong.

### Graph Convergence ≠ Narrative Convergence

In a graph, convergence means "two nodes share a successor." In the story, convergence means "two storylines come back together narratively." A shared successor in the beat DAG might be:

- A genuine narrative convergence (soft dilemma paths rejoining after payoff)
- An intersection (beats co-occurring from different dilemmas — not paths merging)
- A shared beat before any commit (shared because no divergence has happened yet)

Only the first is narrative convergence. An LLM seeing "two edges point to the same node" will default to "these paths converge." The graph structure alone cannot distinguish the three cases — the dilemma role, commit positions, and intersection declarations provide the context needed to interpret what a shared successor means.

### Path Membership ≠ Scene Participation

A beat's `belongs_to` edge means "this beat serves this path's storyline — it advances this path's dilemma." It does NOT mean "this beat only appears on this path" or "this beat is about this path."

Intersection beats participate in scenes with beats from other paths, but they still belong to their original path. The current implementation's cross-assignment of `belongs_to` edges conflated "shares a scene with beats from path B" with "is part of path B's storyline." This produced the hard-convergence violation: beats from mutually exclusive paths appeared to belong to both, creating structurally impossible scenes.

### Beat Ordering ≠ Temporal Position Relative to Commits

The beat DAG says "beat B comes after beat A." That is a prerequisite relationship — a fact about ordering. It does NOT directly encode "beat B is after dilemma X's commit."

Temporal position relative to commits is a higher-level concept computed from the DAG structure: find the commit beat for dilemma X, then determine whether beat B is reachable only through paths that pass through that commit. This computation is well-defined but not trivial, and it is NOT the same as checking a single edge.

### Arcs Are Computed, Not Authored

An arc is a valid traversal of the beat DAG — one combination of path choices producing one complete playthrough. Arcs are the Cartesian product of dilemma paths. They are emergent, not authored.

The danger: treating arcs as primary narrative objects ("this arc needs a scene," "the trust arc should feel warmer"). No one authored an arc. They authored paths and beats. Arcs are what happens when paths combine. Reasoning at the arc level instead of the path level leads to phantom requirements — work that belongs to no path and serves no dilemma.

If arcs are materialized as stored data (for debugging or diagnostics), they must be clearly marked as derived (e.g., `materialized_` prefix) to prevent pipeline stages from treating them as authoritative source data.

### Passages ≠ Beats

A passage is a prose container. A beat is a story moment. They are different abstractions at different levels:

- The author thinks in beats (what happens).
- The player sees passages (what they read).
- One passage can contain multiple beats (from collapse or intersection).
- The same beat can appear in variant passages (same moment, different prose for different states).

Conflating them produces confusion like "edit the passage" when the intent is "change what happens" (a beat concern) vs. "change how it reads" (a passage concern). The beat DAG is the structural truth. The passage graph is the player-facing presentation.

### State Flags ≠ Player Choices

A state flag represents a world state: "the mentor is hostile." It does NOT represent "the player chose to distrust the mentor." The distinction matters because:

- Multiple choices could lead to the same world state (future: distributed commits)
- One choice could trigger multiple state changes
- The prose layer cares about what is true in the world, not which button was pressed

An LLM will naturally write "if the player chose X" when the correct formulation is "if state flag X is active." This conflation is mostly harmless today (one commit per dilemma, one flag per commit) but becomes a real bug if distributed commits or cumulative choices are implemented.

### Codewords ≠ State Flags

State flags are internal implementation machinery — the full set of boolean markers used by GROW, POLISH, entity overlays, and the runtime engine. Codewords are a player-facing subset surfaced in gamebook formats for manual state tracking.

The current codebase uses "codeword" for both concepts. Document 3 defines them as distinct: every codeword is a state flag, but not every state flag is a codeword. Code that manipulates state flags for routing or overlay purposes must not be confused with code that presents codewords to the player.

### Entity Overlays ≠ Entity Variants

An overlay is conditional state on a single entity node: "when hostile, these properties change." The entity remains one node. Two overlays on the same entity compose — a player on both the hostile-mentor and destroyed-artifact arcs sees both overlays applied.

The danger: creating separate entity nodes for each state combination (`mentor_trusted`, `mentor_distrusted`). This breaks every reference to `character::mentor` throughout the graph and produces an entity explosion that scales with the number of dilemmas affecting each entity.

---

## Part 9: Minimal Ontology Summary

### Node Types

| Node | Created by | Persistent | Description |
|---|---|---|---|
| Vision | DREAM | No | Creative contract: genre, tone, themes, audience, scope |
| Voice Document | FILL | No | Prose contract: POV, tense, register, rhythm |
| Entity | BRAINSTORM | Yes (partial) | Character, location, object, or faction. Base state + overlays. |
| Dilemma | BRAINSTORM | No | Binary dramatic question with role, residue weight, ending salience |
| Answer | BRAINSTORM | No | One of two responses to a dilemma. One marked canonical. |
| Path | SEED | No | One answer explored as a storyline |
| Consequence | SEED | No | Narrative outcome of a path choice |
| Beat | SEED, POLISH | No | Story moment. Regular, micro-beat, or residue beat. |
| Intersection Group | GROW | No | Declaration that beats from different paths co-occur |
| State Flag | GROW | Yes | Boolean world-state marker derived from consequence |
| Character Arc Metadata | POLISH | No | Per-entity trajectory summary for FILL context (start → pivot → end per path) |
| Passage | POLISH | Yes (partial) | Prose container holding 1+ beats |
| Scene Blueprint | FILL | No | Per-passage writing plan (sensory palette, opening move) |
| Codeword | SHIP | Yes | Player-facing projection of a state flag (gamebook formats) |
| Art Direction | DRESS | No | Visual identity: style, palette, composition |
| Entity Visual | DRESS | No | Per-entity visual profile for illustration consistency |
| Illustration | DRESS | Yes | Image asset with caption |
| Codex Entry | DRESS | Yes | Diegetic encyclopedia entry |
| Illustration Brief | DRESS | No | Per-passage image generation plan with priority, category, and reference prompts |

"Persistent (partial)" means the node is exported by SHIP, but only a subset of its fields — working metadata is stripped.

### Edge Types

| Edge | From → To | Created by | Description |
|---|---|---|---|
| `has_answer` | Dilemma → Answer | BRAINSTORM | A dilemma's two possible responses |
| `anchored_to` | Dilemma → Entity | BRAINSTORM | Entities central to this dilemma |
| `explores` | Path → Answer | SEED | Which answer this path develops |
| `has_consequence` | Path → Consequence | SEED | Narrative outcomes of this path |
| `belongs_to` | Beat → Path | SEED | Which path this beat serves (single path) |
| `flexibility` | Beat → Entity | SEED | Substitutable entity with role annotation. Working — consumed by GROW. |
| `predecessor` | Beat → Beat | GROW | Ordering in the beat DAG (B comes after A) |
| `intersection` | Beat → Intersection Group | GROW | This beat participates in this co-occurrence group |
| `derived_from` | State Flag → Consequence | GROW | Which consequence this flag represents |
| `grouped_in` | Beat → Passage | POLISH | This beat is part of this passage |
| `choice` | Passage → Passage | POLISH | Player navigation with label, requires, grants |
| `variant_of` | Passage → Passage | POLISH | This passage is a variant of the base passage |
| `wraps` | Dilemma → Dilemma | SEED | A introduces before B, B resolves before A |
| `concurrent` | Dilemma → Dilemma | SEED | Neither wraps the other; active simultaneously |
| `serial` | Dilemma → Dilemma | SEED | A resolves before B introduces; no structural interaction |
| `describes_visual` | Entity Visual → Entity | DRESS | Visual profile for this entity. Working. |
| `targets` | Illustration Brief → Passage | DRESS | Which passage this brief illustrates. Working. |
| `from_brief` | Illustration → Illustration Brief | DRESS | Which brief generated this illustration. Working. |
| `HasEntry` | Codex Entry → Entity | DRESS | This codex entry describes this entity. Persistent. |
| `Depicts` | Illustration → Passage | DRESS | This illustration depicts this passage. Persistent. |

### Dilemma Ordering Relationships

These are edges between dilemma nodes, declared by SEED. They express the author's intent for how dilemmas relate in time.

| Relationship | Meaning |
|---|---|
| Wraps | A introduces before B, B resolves before A |
| Concurrent | Neither wraps the other; active simultaneously |
| Serial | A resolves before B introduces; no structural interaction |

### Dilemma Signals

Distinct from ordering — these are observations about dilemma overlap, not temporal relationships.

| Signal | Meaning |
|---|---|
| Shared Entity | Both dilemmas anchored to same entity; intersection potential (derivable from `anchored_to` edges) |

### State Flag Scoping

| Dilemma Role | State Flag Purpose | Becomes Codeword? |
|---|---|---|
| Hard | Entity overlay activation | Typically no — graph structure handles routing |
| Soft | Routing after convergence + entity overlay activation | Yes — player must track across convergence |
| Cosmetic (POLISH) | Player agency feeling | Optional — no routing consequence |

### The Persistent/Working Boundary

The graph contains two kinds of data:

**Persistent** — needed by the player's runtime. Exported by SHIP. Passages (with prose), choice edges, entities (base state + overlays), state flags, codewords, illustrations, codex entries.

**Working** — consumed during the pipeline. Not exported. Vision, voice document, dilemmas, answers, paths, consequences, beats, intersection groups, scene blueprints, art direction, entity visuals, flexibility annotations, temporal hints, character arc metadata.

Some persistent nodes have working fields that are not exported. SHIP exports the player-facing subset of each persistent node.

Any derived or cached data stored for debugging uses a `materialized_` prefix to signal it is read-only and may be recomputed.

### Future Extensions

Two patterns were identified during design but deferred from the minimal ontology:

**Distributed commits** — A dilemma's commit is spread across multiple smaller choices that accumulate toward resolution, rather than a single dramatic moment. Two implementation paths exist: tree expansion (structurally honest, expensive in content) or threshold state flags (clean, requires a numeric threshold primitive). Neither is needed for the initial implementation. See the research on moral dilemma chains (the "Witcher Principle") for prior art.

**Cosmetic codewords** — Player-facing tokens added by POLISH for the feeling of agency, with no routing consequence. The mechanism is simple (a state flag marked cosmetic, projected as a codeword by SHIP) but the curation of which moments deserve a codeword is a narrative design question that benefits from experience with the pipeline before formalizing.

---

## Appendix: Comparison with Current Ontology

This section documents where the current implementation (`docs/design/00-spec.md`, `src/questfoundry/models/`, `src/questfoundry/graph/`) diverges from the ontology defined in this document. It is a diagnostic — a map of what needs to change, not a criticism of the current code, which was built before Document 1 existed.

### Intersection — Fundamental Redefinition

**Current:** Intersection is modeled by cross-assigning `belongs_to` edges. A beat from path A gets an additional `belongs_to` edge to path B, making it "belong to" both paths. This was the direct cause of the hard-convergence violation fixed on the `fix/hard-convergence-intersection` branch.

**This document:** Intersection is a co-occurrence grouping. Beats retain their single `belongs_to` edge. A separate intersection group declares which beats from different paths share a scene. Path membership and scene participation are distinct concepts.

**Impact:** The `apply_intersection_mark()` function in `grow_algorithms.py` and all code that queries `belongs_to` edges to determine intersection membership needs redesign.

### Codeword → State Flag + Codeword Split

**Current:** The `codeword` node type serves both as internal routing machinery and player-facing state marker. All codewords are treated equally.

**This document:** State flags (internal, full set) and codewords (player-facing, curated subset) are distinct concepts. State flags are created by GROW from consequences. Codewords are projected by SHIP for gamebook formats.

**Impact:** The `Codeword` model, `build_arc_codewords()`, and all routing logic needs to use "state flag" terminology and semantics. SHIP export needs a projection step that selects which state flags become player-facing codewords.

### Convergence Policy → Derived from Dilemma Role

**Current:** `convergence_policy` (hard/soft/flavor) is a directly declared field on `DilemmaAnalysis`. It is the primary concept, with structural behavior derived from it.

**This document:** `dilemma_role` (hard/soft) is the primary concept. Convergence behavior is derived: hard means paths never converge, soft means paths do converge. The `convergence_policy` field is replaced by the role.

**Impact:** `DilemmaAnalysis` in `models/seed.py` and all code that reads `convergence_policy` needs to switch to `dilemma_role`. Flavor choices are handled differently — they are not full dilemmas but minor passage variants created by POLISH.

### `central_entity_ids` → `anchored_to` Edges

**Current:** Dilemmas store central entity references as a list of ID strings (`central_entity_ids` field). Querying "which dilemmas involve this entity?" requires scanning all dilemmas.

**This document:** `anchored_to` edges (dilemma → entity) make this a direct graph query in both directions.

**Impact:** `Dilemma` model in `models/brainstorm.py` and `apply_brainstorm_mutations()` in `mutations.py` need to create edges instead of storing ID lists.

### `is_default_path` → `is_canonical`

**Current:** One answer per dilemma is marked `is_default_path`, suggesting a primary or preferred answer.

**This document:** The field is renamed `is_canonical` and explicitly defined as an authoring convenience (first-written in FILL's writing order), not a narrative preference. Every answer is equally valid.

**Impact:** Rename in `Answer` model and all references. Minor but important for preventing LLM bias toward the "default" path.

### Arc Nodes → Computed Traversals

**Current:** `Arc` is a node type with `arc_id`, `arc_type` (spine/branch), `paths[]`, `sequence[]`. Arcs are created by GROW and stored in the graph.

**This document:** Arcs are computed traversals of the beat DAG, not stored nodes. They are the Cartesian product of path choices. Pipeline stages compute them on demand. Diagnostic snapshots may store them with a `materialized_` prefix.

**Impact:** The `Arc` model in `models/grow.py`, `enumerate_arcs()`, and all code that reads arc nodes needs to be refactored. Arc enumeration becomes a validation utility, not a graph mutation.

### Passage Creation — Moved from GROW to POLISH

**Current:** GROW creates passages (1:1 from beats initially), then passage collapse merges linear chains.

**This document:** GROW produces only the beat DAG. POLISH creates passages by grouping beats (through intersection co-occurrence and collapse), creating choice edges, and adding variants and residue beats.

**Impact:** Phases 7-9 of the current GROW procedure (passage generation, choice creation, routing) move to POLISH. GROW's output boundary changes from "passages and choices" to "the beat DAG with ordering, intersections, and state flags."

### `location_alternatives` → Entity Flexibility Edges

**Current:** Beats have a `location_alternatives` field — a list of alternative location IDs. Only locations can be substituted.

**This document:** Entity flexibility is represented as edges from beats to alternative entities (any category — characters, locations, objects), with a role annotation describing what is being substituted. "The spy could be the informant" is a flexibility edge, not a location swap.

**Impact:** `InitialBeat` model in `models/seed.py` needs flexibility edges instead of `location_alternatives`. GROW's intersection detection needs to read flexibility edges for all entity categories.

### `sequenced_after` → Predecessor/Successor Edges

**Current:** `sequenced_after` edges encode beat ordering as a prerequisite DAG. The name suggests temporal sequence but the semantics are prerequisite relationships.

**This document:** Predecessor/successor edges in the beat DAG. Renamed for clarity — the edge means "this beat comes before that beat" without implying a specific kind of temporal relationship.

**Impact:** Edge type rename. The DAG structure and algorithms are unchanged — only the name and its interpretation.

### Missing: POLISH Stage

**Current:** POLISH does not exist as a pipeline stage. Its responsibilities are split across GROW phases (scene types, gap filling, atmosphere, passage collapse) and not yet implemented (prose feasibility, variant creation, false branching, pacing).

**This document:** POLISH is a full pipeline stage between GROW and FILL, with two phases (finalize beat DAG, build passage layer) and clear responsibilities.

**Impact:** New stage implementation needed. Several current GROW phases (4a-4f, parts of 8-9) migrate to POLISH.

### Missing: Dilemma Ordering

**Current:** No explicit representation of dilemma ordering. Hard/soft role is partially captured in `convergence_policy` but the wraps/serial/concurrent pairwise relationships do not exist. `InteractionConstraint` covers shared_entity, causal_chain, and resource_conflict — related but not the same concepts.

**This document:** Dilemma pairwise relationships (wraps, serial, concurrent, shared_entity) are first-class declarations by SEED. `causal_chain` is subsumed by serial. `resource_conflict` is removed.

**Impact:** New model and edge types for dilemma pairwise relationships. `InteractionConstraint` is redesigned.

### Entity Overlay — Embedded, Not a Separate Node Type

**Early design:** Entity Overlay was specified as a separate node type with `activates` edges (state_flag → entity_overlay), allowing overlays to be independently queried by graph traversal.

**Current design (deliberate):** Overlays are stored as an embedded list on the entity node. Each overlay is `{when: [state_flag_ids], details: {key: value}}`. There is no `entity_overlay` node type and no `activates` edge.

**Rationale:** The spec's own principle is "the entity remains one node." Embedding makes the entity and all its conditional states one atomic read — no join required for the common case (reading an entity in FILL or POLISH context). At the scale this pipeline operates (a few overlays per story), the queryability benefit of separate nodes does not justify the join cost or the node ID management overhead.

**Deferred concern:** If a future stage (e.g., DRESS) needs to reference a specific overlay state by stable ID (e.g., "this illustration depicts the hostile-mentor state"), separate nodes would be preferable. Revisit then.

### Missing: Temporal Hints

**Current:** No mechanism for SEED to express a beat's intended position relative to other dilemmas' commits.

**This document:** Temporal hints are working annotations on beats, consumed by GROW during interleaving. They interact with dilemma ordering relationships to guide beat placement.

**Impact:** New field on `InitialBeat` model. GROW's interleaving algorithm needs to read and respect these hints.

### ADR-017 Routing in GROW vs POLISH

**Current:** ADR-017 (Unified Routing Plan) assigns routing, passage collapse, and false-branch detection to GROW. The `RoutingPlan` architecture computes routing in GROW phases and applies mutations there.

**This document:** All passage-layer work — passage creation, choice edge derivation, variant creation, false branching, and routing — belongs to POLISH. GROW's output boundary is the beat DAG with ordering, intersections, and state flags. GROW does not create passages or choices.

**Impact:** The `RoutingPlan` architecture transfers to POLISH as `PolishPlan`. ADR-017 needs supersession. GROW phases 7–9 (passage creation, choice creation, routing) move to POLISH.

### InitialBeat.paths — List vs Singular belongs_to

**Current:** `InitialBeat.paths` is `list[str]` with `min_length=1`. The mutation creates one `belongs_to` edge per path in the list, allowing a beat to belong to multiple paths simultaneously. The serialize prompt shows multiple paths as valid.

**This document:** Each beat has exactly one `belongs_to` edge to one path (Part 1, Part 3, Part 9). Multi-path membership conflates path membership with scene co-occurrence — the exact pattern that caused the hard-convergence violation. Co-occurrence is modeled by intersection groups (Part 4), not by multiple `belongs_to` edges.

**Impact:** `InitialBeat.paths` becomes singular `path_id: str`. Intersection co-occurrence is signaled through entity flexibility annotations, not multi-path assignment. The mutation, prompt, and validation all need updating.
