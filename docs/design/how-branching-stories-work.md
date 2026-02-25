# How to Build a Branching Interactive Story

> **Status: Authoritative.** This document, together with [Document 3](document-3-ontology.md), is the authoritative source of truth for QuestFoundry's story model. Where other design documents contradict this one, this document takes precedence. See [Issue #977](https://github.com/pvliesdonk/questfoundry/issues/977).

## Common Language

These terms have specific meanings in QuestFoundry. Everyone working with the system — authors, developers, reviewers — must use them consistently.

**Dilemma** — A dramatic question with exactly two compelling answers. "Can the mentor be trusted?" is a dilemma. "Go left or right?" is not — both sides must matter. The power of a dilemma comes from what the player sacrifices by choosing, not what they gain. Each dilemma also carries a statement of *why it matters* — the stakes and consequences that make the choice meaningful.

**Path** — One answer to a dilemma, explored as a storyline. If the dilemma is "Can the mentor be trusted?", one path explores "yes, the mentor is a protector" and the other explores "no, the mentor is a manipulator." Each path is a sequence of beats that proves its answer.

**Shadow** — The answer not explored. Every path has a shadow — the road not taken. Shadows are not dead content; they give weight to the choice. The player chose to trust the mentor, but the *possibility* of betrayal is what makes that trust meaningful. Even unexplored dilemmas (where the player never made an active choice) can cast shadows — the narrative can touch on "what could have been."

**Beat** — A concrete story moment that advances a dilemma toward resolution. Beats are the building blocks between abstract dilemmas and finished prose, and the fundamental unit manipulated throughout the pipeline. BRAINSTORM creates dilemmas. SEED turns them into initial beats. GROW interleaves beats across paths and adds intersections. POLISH reorders, collapses, and splits beats for flow and feasibility. FILL writes prose for each passage. From SEED onward, the work is essentially: creating, ordering, and refining beats.

Each beat describes its relationship to the dilemma it serves — it does one of four things:

- *Advances* the dilemma — builds tension, accumulates evidence
- *Reveals* truth — the protagonist or reader becomes aware of something bearing on the dilemma
- *Commits* the choice — the point of no return, the choice locks in
- *Complicates* the situation — introduces doubt, deepens the dilemma

These describe what a beat does to the dilemma, not its full narrative purpose. When SEED scaffolds a path, it assembles beats into a complete arc — typically introducing the dilemma, developing it through advances and complications, surfacing a reveal, reaching a commit, and playing out the consequences. But this is a common shape, not a formula. A path might need multiple reveals, a try-fail cycle before the commit, emotional reaction beats that process what just happened, or setup beats that establish context without directly touching the dilemma. The scaffold should serve the story, not a template.

**Passage** — What the player actually reads. A passage is the player-facing version of one or more beats, turned into prose. The player sees passages; the author thinks in beats.

**Commit** — The moment a choice becomes irreversible. Before the commit, the story can be the same regardless of what the player will choose. At the commit, the player acts — they draw their sword, sign the contract, kiss the spy. After the commit, the story must reflect the choice. Most commits are a single dramatic moment, but a commit can also be distributed — accumulated across several smaller choices that collectively determine the outcome (see *Future: Distributed Commits* in the SEED section).

**Reveal** — The moment the protagonist or reader becomes aware of a dilemma's truth. A reveal surfaces information; a commit locks in a choice. A story typically reveals before it commits — the player learns the mentor might be lying (reveal), then decides whether to confront them (commit). But they are distinct moments.

**Residue** — The lasting mark a choice leaves on the story. Residue is the narrative memory of choices made — it keeps earlier decisions alive even after their dilemma is fully resolved. A story without residue forgets the player's choices; a story with residue rewards them with callbacks, consequences, and a world that remembers.

- *Heavy residue*: A fundamental change to the world. The mentor is dead. Every future scene must account for their absence.
- *Light residue*: A change in tone or context. The mentor is angry. Future scenes happen, but the mood is different.
- *Cosmetic residue*: A surface-level difference. The cloak is blue, not red. Barely affects anything.

**Arc** — One complete playthrough. A specific combination of choices across all dilemmas. With three dilemmas, each having two paths, there are eight possible arcs — eight different stories a player could experience. Multiple arcs can share the same ending; an arc is the *journey*, not the destination. Every arc should feel like *the* story to the player experiencing it.

**Intersection** — Where independent storylines share a scene. If one path involves stealing a diamond and another involves meeting a spy, and both happen at the same party, that party is an intersection. Intersections make the world feel connected — choices across different dilemmas collide in the same moment.

**Convergence** — Where diverged storylines rejoin. After a dilemma commits and paths split, they may eventually come back together. Whether and when they converge depends on the residue — paths with heavy residue may never converge (the worlds are too different), while paths with light residue can rejoin naturally. Convergence never erases a choice — the residue persists even after paths rejoin.

**Variant Passage** — A separate version of a passage written for a specific combination of active residues. When heavy residue makes it impossible for one passage to serve all arcs honestly, each arc (or group of arcs) gets its own variant. Same story moment, genuinely different prose. Variants are not conditional text within a passage — they are complete, separate passages.

**Residue Beat** — A short passage inserted before a shared passage to establish emotional context. When light residue affects how a shared scene should feel, a residue beat sets the mood without requiring the shared passage itself to vary. "You enter the vault with confidence, the mentor's endorsement giving you courage" (one path) vs. "You enter the vault alone, vigilant" (another path) — then both arrive at the same shared passage.

**Gated** — A passage or choice that only exists for players who made a specific earlier choice. The player experiences gating as consequence: "because I trusted the mentor, this scene happens for me; someone who didn't trust the mentor gets a different scene." Gating is how the story enforces that choices have structural consequences, not just prose differences.

---

## Introduction

A branching interactive story is a living narrative that adapts to the choices a player makes, offering them agency in a world that reacts to their presence.

Writing one is notoriously difficult because every choice multiplies the story. If a player makes just ten binary choices, there are potentially over a thousand different journeys to write. This "combinatorial explosion" is the central challenge of interactive fiction: how do you offer meaningful choices without writing an infinite amount of text?

QuestFoundry solves this by treating the story not as a tree of endless forks, but as a **woven tapestry**. We don't just branch; we merge, weave, and braid storylines back together, ensuring that choices leave lasting marks on the story without fracturing it into unmanageable shards.

The pipeline builds a branching story in stages:

| Stage | Role |
|---|---|
| **DREAM** | Establish the creative vision |
| **BRAINSTORM** | Build the cast and dilemmas |
| **SEED** | Scaffold paths with beats, order dilemmas |
| **GROW** | Weave independent paths into a branching structure |
| **POLISH** | Shape the structure into a prose-ready story *(new — split from GROW)* |
| **FILL** | Write the prose |
| **DRESS** | Create illustrations and reference material |
| **SHIP** | Export to playable formats |

This document explains the creative process at each stage — not the code or the technical specification, but what an author is trying to accomplish and why. For the formal graph ontology that translates these narrative concepts into a data model, see [Document 3 — Story Graph Ontology](document-3-ontology.md).

---

## Part 1: Vision and Raw Material

### The Vision (DREAM)

Every story begins with a creative contract: what kind of experience are we making?

DREAM establishes the boundaries that every later decision must respect. It defines the genre and subgenre (cyberpunk noir, cozy mystery, epic fantasy), the emotional tone (gritty and cynical, whimsical and warm), the themes the story wants to explore ("what is the price of loyalty?"), the intended audience, and preferences for narrative style.

DREAM also sets the scope — how large this story will be. A vignette with two dilemmas and a handful of passages is a fundamentally different undertaking from a full-length adventure with five dilemmas and a hundred passages. Scope constrains everything downstream: how many dilemmas are feasible, how many beats each path can sustain, how much prose needs to be written.

If an idea later in the pipeline contradicts the vision — a slapstick scene in a gritty noir, a light-hearted tone in a story about grief — it gets cut. DREAM is the North Star.

### The Raw Material (BRAINSTORM)

With the vision set, we need a world to tell a story in. BRAINSTORM builds two things: a **cast** and a set of **dilemmas**.

#### The Cast

The cast is the collection of entities that populate the story — characters, locations, objects, and factions. BRAINSTORM is the place to think about who and what belongs in this world, because **the cast is essentially locked after this stage.** Introducing a major character later in the pipeline means they weren't part of any dilemma exploration, they have no beats, and there are no opportunities for their storyline to intersect with others. A new character introduced after BRAINSTORM is limited to a walk-on role — a single paragraph in the final prose at best.

It is easier to build a cohesive cast now and cut what doesn't work than to discover a missing character later and try to weave them in. Think of it as casting a film before shooting begins: everyone who matters needs to be on the call sheet.

#### The Dilemmas

Dilemmas are the engines of a branching story. Each dilemma is a dramatic question with exactly two answers — and both answers must be compelling.

Why exactly two? Because sharp contrast creates meaningful drama. "Do you save the village or pursue your revenge?" forces a sacrifice. "Do you save the village, pursue revenge, or go fishing?" dilutes the tension. It is always more interesting to have two dilemmas than one three-way choice — two binary dilemmas create four arcs, each with a distinct combination of sacrifices, while a three-way choice creates three arcs with weaker contrast between them.

The interesting part of a branching story is not the path taken — it is the **path not taken**. Having an enemy is a story. Having an enemy who *could have been a friend* is a choice that haunts the player. This is why every path carries a shadow — the alternative that makes the chosen path meaningful.

Each dilemma also carries a statement of **why it matters**: the stakes and consequences that make the choice worth agonizing over. "Can the mentor be trusted?" matters because the mentor holds the key to the protagonist's past — trust the wrong person and the truth is lost forever. This "why it matters" is the seed of residue: it describes what lasting mark the choice will leave on the story.

#### Dilemmas and Entities Together

Dilemmas don't exist in isolation — they're anchored to the cast. Each dilemma identifies which entities are central to it. If two dilemmas share a central entity (both involve the Mentor), those dilemmas will naturally interact in the story. If no dilemmas share entities, the story risks feeling like parallel novels that happen to be bound together.

At this stage, dilemmas and entities are sketches — raw material that SEED will triage into a coherent story structure. Not everything survives. But everything that *will* matter must exist here.

---

## Part 2: Commitments and Structure

BRAINSTORM gave us a cast and a set of dilemmas. SEED decides which of those ingredients form a coherent story — and scaffolds the structure to tell it.

### Triage

Not everything from BRAINSTORM survives. SEED selects the characters, locations, and dilemmas that work together as a unified story. A brilliant character who doesn't connect to any dilemma gets cut. A dilemma whose central entities feel disconnected from the rest of the cast gets cut. The goal is a **cohesive ensemble**, not a collection of interesting parts.

For each dilemma that survives, SEED decides which answers to explore as full paths. Often both answers are explored — but sometimes scope or narrative focus means only one side gets a full storyline, with the other remaining as a shadow.

### Scaffolding Paths with Beats

Each explored path needs a skeleton: a sequence of beats that carries the dilemma from introduction to resolution. This scaffold must be **complete** — the arc from beginning to end must be present. A typical scaffold introduces the dilemma, develops it through advances and complications, surfaces one or more reveals, reaches a commit, and plays out the consequences — but the exact shape depends on the story. A path might need a try-fail cycle before the commit, emotional reaction beats after a major reveal, or setup beats that establish a location before the action begins.

Each path's beats should form a coherent story on their own. If you read just one path's beats in order, they should make narrative sense — a beginning, a middle, and an end. GROW will interleave beats from different paths, but it shouldn't need to invent missing structural beats. If the scaffold is incomplete, GROW is forced to fill gaps that are really SEED's responsibility.

A note about pre-commit beats: beats that come before a dilemma's commit are "experienced by every player" — both path A and path B players encounter them. But each pre-commit beat still **belongs to one path**. Path membership describes which dilemma storyline the beat serves, not which players encounter it. A beat on the mentor-trust path that introduces the mentor is encountered by all players (the dilemma hasn't committed yet), but it advances the trust storyline specifically. Its reachability from all starting points comes from the DAG topology — not from belonging to multiple paths.

### Entity Flexibility

While scaffolding beats, SEED annotates them with **flexibility** — hints about what could be changed without breaking the beat's narrative purpose.

A beat that says "the protagonist meets the spy at the docks" might be annotated: the docks could also be the market. The spy could also be the informant. These are not changes — they are invitations. They tell GROW: "if you need to group this beat with another storyline, here's where there's room."

This is how independent paths create the conditions for intersection later. If path A has "meet the spy at the docks" and path B has "find the artifact at the market," and SEED annotated that the spy meeting could happen at the market too — GROW can group them into one scene: meeting the spy at the market, where the artifact also happens to be. The beats remain separate — each still advances its own dilemma — but they co-occur, and POLISH will turn them into a single passage.

### Ordering Dilemmas

Not all dilemmas play the same role in the story. SEED explicitly orders them:

**Hard dilemmas form the backbone.** These are the central dramatic questions — the ones the story is *about*. They introduce early (often in the first act) and commit late (at or near the climax). Because they span the entire story, they carry the most dramatic weight. Their heavy residue means that once they commit, the world is fundamentally different.

**Soft dilemmas are the subplots.** They introduce later and commit earlier — they weave in and out of the middle of the story. Their lighter residue means paths can reconverge after the choice resolves. A romance subplot, a side quest, a secondary loyalty — these enrich the journey without dominating the structure.

This ordering has profound structural consequences. Any beat that comes after a committed dilemma exists in multiple versions — one for each path of that dilemma. Hard dilemmas committing late means most beats come *before* their commit, keeping the story shared and efficient. Soft dilemmas committing early means their brief branching resolves quickly.

SEED also identifies which soft dilemmas can be **serial** — one resolving before another even introduces. Serial soft dilemmas never interact structurally, which is a major complexity reducer. Two soft dilemmas that overlap in time must account for each other; two that are sequential are independent.

#### Future: Distributed Commits

A dilemma's commit is typically a single dramatic moment — one choice that locks in the path. But a commit can also be **distributed** across multiple smaller choices that accumulate toward resolution. Instead of one visible fork, the player makes three or five smaller decisions whose collective weight determines the outcome. The player may not realize which moments mattered until the reckoning — the point where the story reflects their accumulated pattern back to them.

This pattern (known in interactive fiction craft as "moral dilemma chains" or "the Witcher Principle") makes hard choices less obvious and more resistant to save-scumming, but it has significant structural implications. It is not part of the current pipeline and is documented here as a known future direction. See [Document 3](document-3-ontology.md) for a discussion of the implementation options.

### Convergence Sketching

Finally, SEED sketches where diverged paths might rejoin — expressing the author's **intent** for convergence. This is not a binding structure; it is a statement of "I want these storylines to come back together around here."

Convergence intent depends on two independent decisions:

- **Can these paths structurally rejoin?** Hard dilemmas typically never rejoin — the worlds are too different. Soft dilemmas rejoin after enough payoff. Flavor-level choices barely diverge at all.
- **How much should the prose vary when they do?** Heavy residue means genuinely different passages even at convergence. Light residue means a mood-setting beat before a shared passage. Cosmetic residue means tiny differences handled in prose.

These are separate decisions. A soft dilemma can still have heavy residue at specific story moments. A hard dilemma might have low impact on the ending if it resolves before the climax. The convergence sketch captures both dimensions so that GROW knows what to implement and POLISH knows what prose decisions to make.

---

## Part 3: Weaving the Structure

SEED produced independent paths — each a self-contained storyline with its own beats. GROW takes these separate threads and weaves them into a single branched narrative where every valid combination of choices produces a coherent story.

This is the hardest creative act in the pipeline.

### The Problem

Imagine a story with three dilemmas, each with two explored paths. That is six independent storylines, each with their own beats. But the player doesn't experience six separate stories — they experience *one* story in which three dramatic questions unfold simultaneously. GROW must figure out: for every possible combination of choices, what does the player's story look like?

The separate threads need to be **interleaved** — beats from different paths ordered into a coherent narrative sequence. They need to be checked for **intersections** — places where independent storylines can share a scene. And the result must be **validated** — every possible playthrough must be complete and reachable, with no dead ends and no contradictions.

### Interleaving

The most fundamental operation in GROW is deciding the order in which beats from different paths are experienced.

Before any dilemma commits, beats from all paths can be experienced by every player. The early story might look like a linear sequence: introduce the mentor dilemma, introduce the artifact dilemma, develop the mentor's suspicious behavior, introduce the romance subplot, develop the artifact mystery. Different threads, woven into one narrative flow.

The natural ordering follows storytelling logic:

- **Introduction beats** from all paths cluster near the beginning — set up every storyline
- **Development and reveal beats** interleave through the middle — build tension across storylines
- **Commit beats** are distributed for pacing — don't resolve everything at once
- **Consequence beats** cluster toward the end — pay off the choices

This is not a rigid formula. A twist dilemma might introduce late in the story, and that is a valid creative choice. But it has structural consequences.

### The Cost of Branching

Here is the central structural reality of branching fiction:

**Any beat that comes after a committed dilemma exists in multiple versions.** If the mentor dilemma has committed (the player chose to trust or distrust), then every beat that follows exists in two worlds — one where the mentor is trusted, one where they are not. If the artifact dilemma has *also* committed, every beat now exists in four worlds. Three committed dilemmas: eight worlds.

This is where combinatorial explosion actually lives — not in the number of dilemmas, but in the relationship between beats and commits. A beat placed before any commit exists once. A beat placed after all commits exists in every possible world.

**Good storytelling naturally tames this.** Recall from SEED:

- Hard dilemmas (the backbone) introduce early and commit late — near the climax. Because they commit late, most of the story's beats come *before* the hard commit. Those beats are shared across all worlds. And after the hard commit, little story remains — few beats need to be multiplied.

- Soft dilemmas (the subplots) introduce later and commit earlier — in the middle of the story. They branch briefly, then reconverge. Because their residue is light, even the multiplied beats can often share passages.

- Serial soft dilemmas (one resolving before another introduces) never multiply each other at all.

The expensive scenario — many beats multiplied across heavy residue — is minimized not by structural tricks but by the natural shape of a well-told story. The central question hangs over everything but resolves at the climax. The subplots enrich the middle but do not compound.

This is also why introducing a dilemma late is a real creative choice with real costs. A twist dilemma in the final act is dramatically powerful — but every beat after its introduction that also follows an earlier commit now exists in additional worlds. The author must weigh dramatic impact against structural cost.

### Intersections

Independent storylines don't just interleave — they can **collide**. An intersection is where beats from different paths become the same scene.

If the mentor path has "the mentor gives cryptic advice" and the artifact path has "study the artifact's markings," and both could happen in the mentor's study — that is a natural intersection. One scene where the protagonist examines the artifact while the mentor watches, offering cryptic guidance. Two storylines advanced in a single moment.

Intersections are what make a branching story feel like a *world* rather than parallel novels. When different dilemmas collide in the same scene, the player feels that their choices interact — that the story is a living thing responding to them, not a set of isolated tracks.

SEED's entity flexibility annotations are the raw material here. GROW looks for beats that share entities, overlap in location, or could plausibly occur simultaneously — and proposes grouping them into shared scenes. The author approves, rejects, or modifies each proposed intersection.

### Divergence and Convergence

As GROW walks through the interleaved beats, it identifies two critical structural moments:

**Divergence** is where a commit forces the story to branch. Before the mentor dilemma commits, all players experience the same beats. At the commit, the story splits — one version for players who trusted the mentor, another for those who didn't. The divergence point is the last shared beat before the split.

**Convergence** is where branched storylines rejoin. This is governed by the intent SEED expressed:

- For hard dilemmas: paths never structurally rejoin. The worlds are too different. The story carries separate beats all the way to separate endings.
- For soft dilemmas: paths rejoin after enough payoff — a few **gated** beats exclusive to each path that let the player experience the consequences of their choice, then the storylines come back together. The residue persists (the mood is different, the details shift), but the structural path is shared again.
- For flavor-level choices: paths barely diverge. The choice affects tone and details but not which beats the player experiences.

Convergence is never about erasing a choice. Even when paths rejoin, the residue — the narrative memory of what the player chose — must persist. A converged passage might be shared structurally, but the player's experience of it should still reflect their journey.

### Validation

Once the branching structure is complete, GROW validates that it actually works:

- **Every arc is complete.** For every possible combination of choices, is there a coherent sequence of beats from beginning to end?
- **Every beat is reachable.** No orphan beats floating disconnected from the story.
- **Every dilemma resolves.** Each dilemma has a commit beat on every arc that explores it.
- **No contradictions.** No beat requires a condition that is impossible to reach.
- **No dead ends.** Every arc reaches an ending.

If validation fails, the problem is structural — it goes back to GROW or SEED for fixing, not forward to POLISH for patching.

---

## Part 4: Shaping the Story

GROW produced a valid branching structure — one where every combination of choices leads to a complete story. But a valid structure is not yet a story a player wants to read. POLISH takes the frozen structure and shapes it into something ready for prose.

POLISH never changes what happens or on which branches. It changes how the story is presented and ensures that every passage can actually be written well.

*Note: POLISH was historically part of GROW. It is separated here because the concerns are distinct — GROW is about the branching structure, POLISH is about story craft. Mixing them made both harder to reason about.*

### Character Arcs

The branching structure tells us which beats each character appears in, on which paths, in what order. POLISH synthesizes this into explicit arc descriptions: "The mentor begins as a cryptic authority figure, is gradually revealed as either a protector or a manipulator (depending on path), and ends as either a trusted ally or a defeated adversary."

These arc descriptions are metadata for FILL — they ensure that when the prose writer encounters the mentor in scene twelve, they know where the mentor has been and where the mentor is going. Without them, the writer sees individual beats in isolation and risks inconsistency.

### Beat Reordering

Within linear sections of the story (stretches where no branching occurs), beats may be reordered for better narrative flow. GROW placed them in a structurally valid order, but that is not always the most compelling reading order. A scene of quiet reflection might read better before a tense confrontation, or two beats about the same character might work better separated by a beat about someone else.

Reordering is safe here because the branching structure is frozen — no branching logic depends on the order of beats within a linear section.

### Prose Feasibility

This is where POLISH earns its place as a separate stage.

Every passage in the branching structure will eventually need prose. POLISH audits each one: given all the choices that could be active when a player reaches this passage, **can a writer actually produce good prose for it?**

The practical limit is roughly two to three active states. A writer can craft a passage that works when the mentor is trusted *and* the artifact is found — that is two states, manageable. Add a third (the romance resolved happily) and it is still feasible. But five active states with conflicting residues? No single passage can serve that honestly.

POLISH applies a decision tree to each passage:

- **Is this state relevant to this passage?** If the passage is about finding the artifact, the romance subplot's residue probably doesn't matter here. Annotate the passage: "don't address the romance in this scene." Irrelevant states are simply omitted from prose.

- **Two or three compatible states?** The passage can be written as poly-state prose — text that is diegetically honest for all active states simultaneously. "The mentor's expression was unreadable" works whether the mentor is trustworthy or not.

- **Light residue that affects the mood?** Create a **residue beat** — a short passage placed before the shared passage that establishes the emotional context for each path. One path gets "You enter the vault with confidence" and another gets "You enter the vault on guard." The shared passage that follows can be neutral because the residue beat already set the tone.

- **Heavy residue that changes the scene fundamentally?** Create a **variant passage** — a genuinely separate version of the passage for each incompatible state. Same story moment, completely different prose. The mentor is alive in one variant and dead in the other — no amount of careful wording can serve both.

- **Too many conflicting states for any of the above?** The passage needs structural splitting. This is rare if SEED and GROW did their jobs well, but it happens — and it is better caught here than discovered by the writer mid-sentence.

After this audit, every passage in the story is **prose-feasible**: the writer knows exactly what they are working with, how many states to juggle, and where variants or residue beats handle the complexity for them.

### Passage Collapse

Some stretches of the story are linear chains of beats with no choices between them. Three beats in a row — "search the study," "find the hidden letter," "read the letter" — that is three separate passages the player clicks through with no decisions to make.

Passage collapse merges these linear chains into a single passage with transition guidance. Instead of three clicks, the player reads one flowing scene. The transition guidance tells the writer how to connect the beats: "continue the same action smoothly," "shift focus to the arriving character," "mark a brief time skip."

Collapse happens after prose feasibility because collapsed passages need to be feasible too — merging three beats into one passage means the writer juggles whatever states all three beats carry.

### False Branching

Not every choice in a story needs to be a real dilemma. Sometimes the player benefits from a sense of exploration — "Do you check the library or the garden first?" — where both options advance the same plot. The player feels agency; the story structure is unchanged.

False branches give the experience of choice without the structural cost of real branching. They are particularly useful in the early story, where the structure might be linear (all introduction beats, no commits yet) but the player should not feel like they are on rails.

### Pacing

Finally, POLISH checks the rhythm of the story. Too many intense scenes in a row exhausts the reader. Too many quiet reflections stalls the momentum. POLISH can inject micro-beats — brief narrative moments that provide breathing room between major scenes, or small transitions that smooth abrupt jumps.

Like all POLISH operations, pacing adjustments do not change what happens in the story. They change how it feels to experience it.

---

## Part 5: Writing the Narrative

By the end of POLISH, every beat has become a passage — the player-facing unit that FILL will write prose for. Some passages correspond to a single beat; others are collapsed from a chain of beats. Variant passages and residue beats have been created where needed. Every passage is prose-feasible. FILL has one job: write good prose.

FILL does not create, reorder, split, or merge beats. If a passage cannot be written well, the problem is upstream — in POLISH or GROW — not here.

### Voice

Before writing a single passage, FILL establishes a **voice document** — the stylistic identity of the story. This locks concrete decisions that DREAM left open: point of view (first person, second person, third person), tense (past or present), register (formal, conversational, literary), sentence rhythm, and specific patterns to use or avoid.

The voice document is informed by everything that came before — DREAM's creative vision, the full story structure revealed by GROW and POLISH, the arc descriptions, the tone of the dilemmas. A story with a single protagonist navigating intimate moral choices suggests a different voice than an ensemble adventure across multiple locations.

The voice document becomes the North Star for all prose. Fifty or a hundred passages must feel like they belong to the same story. Without a locked voice, prose drifts — early passages sound different from late ones, branches sound different from the main arc.

### Writing Order

FILL writes passages along one complete arc first — one valid playthrough from beginning to end. This establishes the baseline voice and creates the canonical version of every shared passage along that arc.

Other arcs are then written toward the already-established shared passages. When writing a branch that eventually converges with the first arc, the writer knows what prose awaits at the convergence point and can write smoothly toward it. Without this ordering, branches would be written into a void — arriving at convergence points that don't exist yet.

The first arc written is not "the main story." It is a starting point for the authoring process. Every arc should feel like *the* story to the player experiencing it.

### Context

Each passage is written with rich context — not just what this beat is about, but everything the writer needs to make it feel like part of a living story:

- **The voice document** — consistent style throughout
- **The beat summary and scene type** — what happens and at what pace
- **Full entity details** — not just names, but appearance, personality, current state from active residues. A writer cannot portray the mentor well if all they know is "the mentor is here." They need to know the mentor's demeanor is warm (on the trust path) or guarded (on the distrust path), what they look like, how they speak.
- **The preceding passages** — a sliding window of recent prose to maintain continuity and prevent drift
- **The character arcs** — where each entity has been and where they are going
- **The shadows** — the paths not taken, which give weight to the paths that were. The writer knows what *didn't* happen, which informs how they portray what *did*.
- **Lookahead** — when writing toward a convergence point, the prose that awaits there, so the writer can land smoothly

### Review

After prose is generated, it is reviewed for common issues: voice drift (a passage sounds different from its neighbors), continuity breaks (details contradict earlier passages), flat prose (lacks tension or sensory detail), or summary deviation (the prose doesn't match what the beat was supposed to accomplish).

A maximum of two review-and-revision cycles are permitted. If prose still doesn't work after two passes, the problem is not the writing — it is the structure. A broken story cannot be rescued by better prose. The fix belongs upstream.

---

## Part 6: Illustration and Export

### Illustration and Reference (DRESS)

The story has prose. DRESS adds visual identity and reference material.

**Art direction** begins by establishing a visual style consistent with DREAM's vision — a cyberpunk noir calls for different imagery than a cozy mystery. DRESS then creates visual descriptions for the major entities: what the mentor looks like, the atmosphere of the docks at night, the weight and color of the ancient artifact.

For each important passage, DRESS generates a caption and an image generation prompt. These can feed directly into an AI image generator or serve as a brief for a human artist. The goal is that every key moment in the story has a visual anchor — something the player sees that grounds the prose in a concrete image.

**The codex** is a diegetic glossary — an in-world reference for the player. Major characters, locations, factions, and objects get entries written from the perspective of the story's world, not as technical documentation. The codex helps players keep track of a branching story's cast and setting without breaking immersion.

DRESS uses the finished prose as input. It does not change the story — it dresses it.

### Export (SHIP)

SHIP transforms the complete story — prose, art, codex — into playable formats:

- **Twee** — for use in Twine, the most widely used interactive fiction tool
- **HTML** — a standalone playable story in any web browser
- **JSON** — a machine-readable format for custom players or further processing
- **Gamebook PDF** — shuffled, numbered passages with "turn to page X" choices, in the tradition of Choose Your Own Adventure

SHIP is a technical transformation. The creative work is done.

---

## Part 7: The Fundamental Tensions

Building a branching story is a constant negotiation between opposing forces. These tensions do not have solutions — they have trade-offs that the author navigates at every stage.

### Meaningful Choices vs. Writeable Scope

Every dilemma added to the story doubles the number of possible arcs. Every arc needs beats. Every beat needs prose. The author wants rich, consequential choices — but each one multiplies the work.

The resolution is structural: hard dilemmas (the backbone) carry the deepest meaning but commit late, keeping most beats shared. Soft dilemmas (the subplots) enrich the journey but reconverge, bounding their cost. Serial soft dilemmas avoid multiplying each other entirely. The author controls scope not by making choices less meaningful, but by choosing *when* and *how* they branch.

### Efficiency vs. Narrative Honesty

Efficiency wants to reuse — one passage serving every arc, one scene working for every combination of choices. Narrative honesty demands that choices matter — a passage that ignores the player's history feels hollow.

The resolution is layered. Poly-state prose handles two or three compatible states with careful ambiguity. Residue beats set the emotional context before shared passages. Variant passages provide genuinely different prose when residues are incompatible. And irrelevant states are simply omitted — not every beat needs to acknowledge every choice. The key principle: **a shared passage is not a replacement for a branch.** When the story itself differs, the passages must differ.

### Structure vs. Creative Freedom

Structure demands logic, causality, and continuity. Every beat must be reachable, every choice must lead somewhere, every arc must be complete. Creative freedom wants surprise, spontaneity, and moments that don't fit neatly into a framework.

The resolution is staged. BRAINSTORM is wild — generate freely, no constraints. SEED imposes the first discipline — triage and scaffold. GROW enforces structural validity. POLISH shapes the experience. FILL writes within the structure but with full creative voice. Each stage progressively narrows the space while respecting what came before. The wildest ideas are welcome early; by the time prose is written, the structure ensures they actually work.

### The Gap Between Structure and Prose

A branching story has two layers that do not always align. The structure says "this is one passage" — but the narrative might need it to feel completely different depending on how the player arrived. Conversely, the structure says "these are two separate passages" — but the prose might be nearly identical.

This gap is not a flaw to be eliminated. It is the fundamental nature of the medium. POLISH bridges it by auditing every passage for prose feasibility, creating variants and residue beats where structure and narrative diverge. The author's job is to be honest about when structure can be shared and when it cannot — and to never let structural convenience override narrative truth.
