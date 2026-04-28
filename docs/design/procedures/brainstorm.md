# BRAINSTORM — Build the cast and dilemmas

## Overview

BRAINSTORM is the expansive creative stage: it turns an approved Vision into raw story material — the cast of Entity nodes and the set of binary Dilemma nodes that will drive the story's branching. It is deliberately generative (more material than SEED will keep) and does not produce paths, beats, consequences, or any structural machinery — that begins in SEED.

## Stage Input Contract

*Must match DREAM §Stage Output Contract exactly.*

1. Exactly one Vision node exists in the graph.
2. The Vision node has non-empty values for: `genre`, `tone`, `themes`, `audience`, `scope`.
3. The Vision node has no incoming or outgoing edges.
4. `pov_style`, if present, is one of `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`; absent or null if deferred to FILL.
5. No other node types exist in the graph.
6. Human approval of DREAM is recorded.

---

## Phase 1: Discussion

**Purpose:** Generate raw creative material through free-form dialogue seeded by the Vision. Nothing is committed to the graph yet.

### Input Contract

1. Stage Input Contract satisfied.
2. Vision fields are available as conversational context.

### Operations

#### Open Exploration

**What:** The LLM and the human riff on story possibilities — characters, locations, dramatic questions, atmosphere — informed by the Vision. The output is prose discussion notes. High temperature; embrace variety. The expansive mandate is the point: more material here gives SEED more to triage from.

**Rules:**

R-1.1. Discussion aims for abundance. Err toward inclusion — entity and dilemma targets come from the active Size Preset (one of `micro`, `short`, `medium`, `long`), with ranges from 5–10 entities and 2–3 dilemmas (`micro`) up to 20–35 entities and 5–10 dilemmas (`long`); intermediate `short` and `medium` values come from the same preset configuration. SEED will cut; do not pre-triage here.

R-1.2. Discussion writes nothing to the graph. It produces prose notes only. The graph remains empty of BRAINSTORM artifacts through the end of this phase.

R-1.3. Every proposal must be compatible with the Vision. Genre, tone, themes, and content_notes constrain discussion. A proposal that contradicts the Vision must be flagged and rejected — not silently softened.

R-1.4. The LLM is generative; the human guides. The human redirects unproductive paths and encourages promising ones. Filtering for quality is SEED's job, not the human's here.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Discussion ends below the size preset's `entities_min` / `dilemmas_min` | Pre-triaging — self-censored during BRAINSTORM | R-1.1 |
| Entity node exists in the graph after Phase 1 | Graph write happened during discussion | R-1.2 |
| Discussion proposes a slapstick comic-relief character in a gritty noir | Vision's tone not enforced; contradiction accepted | R-1.3 |

### Output Contract

1. Prose discussion notes (in-conversation or curated) sufficient for entity and dilemma extraction.
2. No graph nodes exist yet.

---

## Phase 2: Entity Extraction

**Purpose:** Distill the discussion notes into Entity nodes — the locked cast that every downstream stage references.

### Input Contract

1. Phase 1 Output Contract satisfied.
2. Discussion notes are available.

### Operations

#### Entity Proposal

**What:** The LLM reviews the discussion and proposes an entity list covering every significant character, location, object, and faction. Each entity gets a concept (one-line essence) and notes (freeform context from discussion). The human reviews and edits.

**Rules:**

R-2.1. Every Entity node has non-empty `name`, `category`, and `concept`.

R-2.2. `category` is one of: `character`, `location`, `object`, `faction`. No other values are permitted.

R-2.3. Entity IDs are scoped by category: `character::mentor`, `location::archive`, `object::cipher_device`. `character::mentor` and `location::mentor` are distinct nodes because the category is part of identity.

R-2.4. At least two distinct `location`-category entities are created. Scene variety and intersection formation downstream require multiple locations — a single-location story cannot support them.

R-2.5. The LLM captures what was discussed; it does not invent entities to fill perceived gaps. Missing entities must come from the human, either by surfacing them in discussion first or by adding them explicitly.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Entity has `category: "ally"` | Category outside the permitted set — "ally" is a role, not a category | R-2.2 |
| Entity has empty `concept` | Required field missing — extraction captured only the name | R-2.1 |
| BRAINSTORM produces only one location entity | Insufficient location variety for scene construction downstream | R-2.4 |
| Phase 2 adds an entity that was never mentioned in discussion | LLM invented a character to fill a gap | R-2.5 |
| `archive` exists both as character and location | Category-as-namespace not used; ambiguous ID | R-2.3 |

### Output Contract

1. One or more Entity nodes exist.
2. Each Entity has non-empty `name`, `category`, `concept`.
3. `category` ∈ {`character`, `location`, `object`, `faction`} for every Entity.
4. At least two distinct `location`-category entities exist.
5. Entity IDs are namespaced by category.
6. No Dilemma, Answer, Path, Beat, Consequence, or State Flag nodes exist yet.

---

## Phase 3: Dilemma Formation

**Purpose:** Frame the story's dramatic questions as binary Dilemma nodes, each with two Answer nodes (one canonical) and anchors to the entities they involve. This is the last phase of BRAINSTORM; its Output Contract is the stage's Output Contract.

### Input Contract

1. Phase 2 Output Contract satisfied.
2. Discussion notes remain available (for `why_it_matters` and answer descriptions).

### Operations

#### Dilemma Proposal

**What:** The LLM extracts dramatic questions from the discussion and frames each as a binary dilemma with exactly two answers, a `why_it_matters` statement, and anchors to relevant entities. The human reviews and approves.

**Rules:**

R-3.1. Every Dilemma node has a non-empty `question` and a non-empty `why_it_matters`. The question ends with `?`. The `why_it_matters` is the seed of residue — what lasting mark the choice leaves. → how-branching-stories-work.md §Common Language (Residue).

R-3.2. Every Dilemma has exactly two `has_answer` edges to two distinct Answer nodes. Three-way or four-way dilemmas are forbidden — for nuanced situations, split into multiple binary dilemmas. → how-branching-stories-work.md §The Dilemmas.

R-3.3. Both answers must be genuinely different and both must be compelling. Shades of gray ("protector" vs "well-meaning-but-flawed") and degenerate contrasts ("save now" vs "save later") are violations — they produce weak drama.

R-3.4. Exactly one Answer per Dilemma has `is_canonical: true`. The canonical answer is operationally privileged (FILL writes its arc first) but not narratively superior — see → ontology §Part 1: Answer.

R-3.5. Every Answer has a non-empty `description` stating what this response means narratively.

R-3.6. Every Dilemma has at least one `anchored_to` edge to an Entity. A dilemma anchored to nothing is meaningless — it has no grip on the world. (Entity triage — cutting entities — is SEED's concern, not BRAINSTORM's.)

R-3.7. Dilemma IDs use the `dilemma::` prefix (e.g., `dilemma::mentor_trust`). Answer IDs are unprefixed and scoped within their parent Dilemma (e.g., `mentor_protector`). Answer ID uniqueness is enforced per Dilemma (the pair `<dilemma_id, answer_id>` is globally unique), not globally — two Dilemmas may each have an answer named `benevolent` without collision.

R-3.8. No Path, Beat, Consequence, State Flag, Passage, or Intersection Group nodes are created. Those belong to later stages.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Dilemma has three `has_answer` edges | Three-way dilemma — must be split into binary dilemmas | R-3.2 |
| Both answers: "Save the village now" / "Save the village later" | Degenerate contrast — both do the same thing, just differently timed | R-3.3 |
| Dilemma has `is_canonical: true` on both answers | Exactly one canonical per dilemma | R-3.4 |
| Dilemma has no `is_canonical: true` on either answer | Canonical marking missing | R-3.4 |
| Dilemma has no `anchored_to` edges | Dilemma has no grip on the world — nothing for SEED to scaffold around | R-3.6 |
| Dilemma ID is `mentor_trust` (no prefix) | Missing `dilemma::` namespace | R-3.7 |
| Dilemma has empty `why_it_matters` | Required field missing — no residue seed | R-3.1 |
| Path node exists in the graph after BRAINSTORM | Path creation is SEED's responsibility | R-3.8 |

### Output Contract

1. One or more Dilemma nodes exist.
2. Each Dilemma has non-empty `question` (ending `?`) and `why_it_matters`.
3. Each Dilemma has exactly two `has_answer` edges to distinct Answer nodes.
4. Each Answer has a non-empty `description`.
5. Exactly one Answer per Dilemma has `is_canonical: true`.
6. Each Dilemma has at least one `anchored_to` edge to an Entity.
7. Dilemma IDs use the `dilemma::` prefix.
8. No Path, Beat, Consequence, State Flag, Passage, or Intersection Group nodes exist.

---

## Stage Output Contract

1. One or more Entity nodes exist, each with non-empty `name`, `category`, `concept`; `category` ∈ {`character`, `location`, `object`, `faction`}.
2. At least two distinct `location`-category entities exist.
3. Entity IDs are namespaced by category (e.g., `character::mentor`).
4. One or more Dilemma nodes exist, each with non-empty `question` and `why_it_matters`.
5. Each Dilemma has exactly two `has_answer` edges to two distinct Answer nodes.
6. Each Answer has a non-empty `description`.
7. Exactly one Answer per Dilemma has `is_canonical: true`.
8. Each Dilemma has at least one `anchored_to` edge to an Entity.
9. Dilemma IDs use the `dilemma::` prefix.
10. No Path, Beat, Consequence, State Flag, Passage, or Intersection Group nodes exist.
11. Vision node is unchanged from DREAM's output.

## Implementation Constraints

- **Context Enrichment:** The LLM call that proposes dilemmas must receive the full Vision node (genre, subgenre, tone, themes, audience, scope, content_notes) AND the full Entity list (names, categories, concepts, notes) — not just IDs or a genre string. Bare listings produce generic dilemmas. → CLAUDE.md §Context Enrichment Principle (CRITICAL)
- **Prompt Context Formatting:** Entity and dilemma lists injected into prompts must be formatted as human-readable text (joined strings, bullet points), never as Python list or dict repr. → CLAUDE.md §Prompt Context Formatting (CRITICAL)
- **Valid ID Injection:** Any LLM call that references entity IDs (e.g., for `anchored_to` edges) must receive an explicit `### Valid IDs` section listing every Entity ID created in Phase 2. → CLAUDE.md §Valid ID Injection Principle
- **Small Model Prompt Bias:** BRAINSTORM runs on small models during local dev. Fix the prompt before blaming the model for weak dilemmas. → CLAUDE.md §Small Model Prompt Bias (CRITICAL)
- **Silent Degradation:** Validation of the binary-dilemma invariant (R-3.2), canonical marking (R-3.4), and anchored-to requirement (R-3.6) must produce hard errors, not fallbacks. A dilemma with three answers, two canonical markings, or zero anchors must halt BRAINSTORM — never silently serialize a partial result. → CLAUDE.md §Anti-Patterns to Avoid (Silent degradation of story structure constraints)

## Cross-References

- Cast and Dilemma narrative concepts → how-branching-stories-work.md §The Raw Material (BRAINSTORM)
- Entity node schema → story-graph-ontology.md Part 1: Entity
- Dilemma, Answer node schemas → story-graph-ontology.md Part 1: Dilemma, Answer
- `has_answer`, `anchored_to` edges → story-graph-ontology.md Part 9: Edge Types
- Canonical answer operational privilege → story-graph-ontology.md Part 1: Answer
- Binary dilemma rationale → how-branching-stories-work.md §The Dilemmas
- Previous stage → dream.md §Stage Output Contract
- Next stage → seed.md §Stage Input Contract

## Rule Index

R-1.1: Discussion aims for abundance; do not pre-triage.
R-1.2: Discussion writes nothing to the graph.
R-1.3: Proposals must be compatible with the Vision; contradictions are flagged and rejected.
R-1.4: LLM is generative; human guides but does not over-filter in Phase 1.
R-2.1: Every Entity has non-empty `name`, `category`, `concept`.
R-2.2: `category` ∈ {character, location, object, faction}.
R-2.3: Entity IDs are namespaced by category.
R-2.4: At least two distinct `location` entities exist.
R-2.5: LLM captures what was discussed; no invention.
R-3.1: Every Dilemma has non-empty `question` (ending `?`) and `why_it_matters`.
R-3.2: Every Dilemma has exactly two `has_answer` edges.
R-3.3: Both answers are genuinely different and both compelling.
R-3.4: Exactly one Answer per Dilemma has `is_canonical: true`.
R-3.5: Every Answer has a non-empty `description`.
R-3.6: Every Dilemma has at least one `anchored_to` edge.
R-3.7: Dilemma IDs use the `dilemma::` prefix; Answer IDs are unprefixed and scoped within their Dilemma.
R-3.8: No Path / Beat / Consequence / State Flag / Passage / Intersection Group nodes exist after BRAINSTORM.

---

## Human Gates

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Discussion | Light — "enough raw material?" |
| 2 | Entity Extraction | Required — review entity list |
| 3 | Dilemma Formation | Required — review dilemmas, confirm canonical markings |

## Iteration Control

**Forward flow:** Discussion → Entity Extraction → Dilemma Formation.

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| Entity Extraction | Discussion | Major gaps in entity coverage |
| Dilemma Formation | Discussion | No clear dramatic questions emerged |
| Dilemma Formation | Entity Extraction | Dilemma involves an entity not in the list |
| Any phase | DREAM | Material doesn't fit DREAM vision (genre/tone/constraint conflict) |

**Maximum iterations:**

- Discussion: no fixed limit; human decides when "enough."
- Entity Extraction: at most 2 revision passes.
- Dilemma Formation: at most 2 revision passes.

## Failure Modes

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1. Discussion | Conversation stalls | Human judgment | New prompts, different angle |
| 1. Discussion | Ideas too generic | Human judgment | Push for specificity; add Vision constraints |
| 2. Entity Extraction | Missing important entity | Human review | Add manually, note source in notes field |
| 2. Entity Extraction | Too many entities | Human review | Leave them; SEED will cut |
| 3. Dilemma Formation | Non-binary dilemma proposed | Validation (R-3.2) | Split into multiple binary dilemmas |
| 3. Dilemma Formation | Weak contrast between answers | Human review (R-3.3) | Sharpen answers or cut dilemma |
| 3. Dilemma Formation | No canonical obvious | Human review | Human picks canonical — "narratively equal, operationally privileged" |

**Escalation to DREAM.** Return to DREAM if brainstormed content cannot fit the Vision — genre or tone mismatch emerges, or Vision's content_notes prove too restrictive for any interesting dilemma.

## Context Management

**Standard (≥128k context):** Include full discussion notes in context for Phases 2 and 3. No windowing.

**Constrained (~32k context):** Human curates discussion highlights; pass highlights (not full transcript) to Phase 2 and Phase 3. Trust the curation to preserve material.

## Worked Example

### Input (DREAM output)

```yaml
vision:
  genre: "dark fantasy"
  subgenre: "mystery"
  tone:
    - "atmospheric"
    - "morally ambiguous"
  themes: ["forbidden knowledge", "trust", "corruption"]
  audience: "adult readers of literary speculative fiction"
  scope: short
  content_notes: ["single protagonist POV", "no explicit magic system", "no graphic violence"]
  pov_style: third_person_limited
```

### Phase 1: Discussion (excerpts)

**LLM:** "A dark fantasy mystery with forbidden knowledge — what if the protagonist is an archivist? Someone whose job is to preserve knowledge, now facing knowledge that shouldn't be preserved?"

**Human:** "I like that. What draws them in?"

**LLM:** "A mentor figure — trustworthy on the surface but possibly with their own agenda. The classic 'protecting me or using me?' dilemma."

**Human:** "Yes. And the archive itself should feel alive, dangerous. Layered."

*(Discussion continues, covering characters, locations, artifacts, relationships.)*

### Phase 2: Entity Extraction

```yaml
entities:
  - id: character::kay
    name: "Kay"
    category: character
    concept: "Young archivist drawn into conspiracy"
    notes: "Curious, principled, out of her depth. Family connection to archive."

  - id: character::mentor
    name: "The Mentor"
    category: character
    concept: "Senior archivist with hidden agenda"
    notes: "Ambiguous loyalty. Knows more than they reveal."

  - id: location::archive
    name: "The Archive"
    category: location
    concept: "Ancient repository of forbidden knowledge"
    notes: "Public stacks, restricted collections, forbidden depths."

  - id: location::forbidden_depths
    name: "The Forbidden Depths"
    category: location
    concept: "Lowest level of the archive, where the dangerous knowledge lives"
    notes: "Rumored deaths; few have entered and returned."

  - id: object::cipher_device
    name: "The Cipher Device"
    category: object
    concept: "Artifact that reveals hidden text"
    notes: "Central to investigation and danger. Origin unclear."

  - id: faction::conspiracy
    name: "The Conspiracy"
    category: faction
    concept: "Group with interest in the archive's secrets"
    notes: "Unclear whether they want to protect or exploit."
```

Human reviews, approves.

### Phase 3: Dilemma Formation

```yaml
dilemmas:
  - id: dilemma::mentor_trust
    question: "Can the mentor be trusted?"
    why_it_matters: "Trust determines whether Kay has an ally or is alone against the conspiracy."
    answers:
      - id: mentor_protector
        description: "Mentor is genuinely protecting Kay from forces she doesn't understand"
        is_canonical: true
      - id: mentor_manipulator
        description: "Mentor is using Kay to access forbidden knowledge"
        is_canonical: false
    anchored_to: [character::mentor, character::kay]

  - id: dilemma::archive_nature
    question: "Is the archive's knowledge salvation or corruption?"
    why_it_matters: "Determines whether Kay's quest is heroic or tragic."
    answers:
      - id: archive_salvation
        description: "The forbidden knowledge can save the world if used wisely"
        is_canonical: true
      - id: archive_corruption
        description: "The forbidden knowledge corrupts all who access it"
        is_canonical: false
    anchored_to: [location::archive, object::cipher_device, character::kay]
```

Human reviews, approves. BRAINSTORM complete.
