# BRAINSTORM Procedure

## Summary

**Purpose:** Expansive exploration of story possibilities. Generate raw creative material that SEED will triage into committed structure.

**Input artifacts:**
- `01-dream.yaml` (approved vision)

**Output artifacts:**
- `02-brainstorm.yaml` (entities, dilemmas with answers)

**Mode:** LLM-heavy with human guidance. Discuss → Summarize → Serialize.

---

## Prerequisites

### Required Input Files

| File | Required State |
|------|----------------|
| `01-dream.yaml` | Complete (approved vision) |

### Required Human Decisions from Prior Stages

- DREAM vision approved (genre, tone, themes, constraints)

### Knowledge Context

Inject for LLM:
- Full DREAM vision
- Genre conventions (if relevant corpus documents exist)
- Story length target (from DREAM)

---

## Core Concepts

### Expansive Before Selective

BRAINSTORM is deliberately expansive. The goal is to generate more material than needed—SEED will triage it. Don't self-censor during BRAINSTORM.

**Good BRAINSTORM:** 20 entities, 8 dilemmas, rich notes
**Bad BRAINSTORM:** 5 entities, 2 dilemmas, minimal notes

More raw material gives SEED more options.

### Discuss → Summarize → Serialize

BRAINSTORM follows a three-phase pattern:

1. **Discuss (High Temperature):** Free-form creative exploration. Riff on possibilities. No structure yet.

2. **Summarize (Consolidate):** Extract structured elements from discussion. Identify entities, dilemmas, relationships.

3. **Serialize (Low Temperature):** Convert to YAML. No creativity—just formatting.

This separation keeps creative exploration separate from structural commitment.

### Binary Dilemmas

Every dilemma has exactly two answers. This keeps contrasts crisp and decisions meaningful.

**Good dilemma:**
- Question: "Can the mentor be trusted?"
- Answer A: "Mentor is genuine protector" (canonical)
- Answer B: "Mentor is manipulating Kay" (non-canonical)

**Bad dilemma:**
- Question: "What is the mentor's nature?"
- Answer A: "Protector"
- Answer B: "Manipulator"
- Answer C: "Well-meaning but flawed"
- Answer D: "Secretly the antagonist"

For nuanced situations, use multiple binary dilemmas:
- Dilemma 1: Mentor alignment (benevolent vs selfish)
- Dilemma 2: Mentor competence (capable vs flawed)

This yields four combinations while each dilemma remains binary.

### Creative Freedom (No Anchors)

BRAINSTORM generates freely. Entities, locations, and events emerge naturally from creative exploration—don't constrain them for later stage convenience.

Location flexibility for intersection formation is handled in SEED, not here. BRAINSTORM's job is creative richness, not structural optimization.

---

## Algorithm Phases

### Phase 1: Discussion

**Purpose:** Free-form creative exploration seeded by DREAM vision.

**LLM Involvement:** Discuss (high temperature)

This phase is conversational. LLM and human riff on the story possibilities.

**Discussion prompts:**

*Characters:*
- "Who lives in this world? What makes them interesting?"
- "Who has something to lose? Who has power?"
- "What relationships exist? Who's in conflict with whom?"

*Dramatic questions:*
- "What are the big questions this story asks?"
- "What could go wrong? What's at stake?"
- "What secrets might exist? What truths could be hidden?"

*Setting:*
- "What locations are central to this story?"
- "What's the texture of this world? What makes it feel real?"
- "What events or moments feel essential?"

**LLM role:** Generative and exploratory. Propose ideas, build on human input, suggest connections. High temperature—embrace variety.

**Human role:** Guide, prompt, react. Encourage promising directions, redirect unproductive ones. Don't filter too heavily—save that for SEED.

**Output:** Discussion notes (can be raw transcript or human-curated highlights)

**Human Gate:** Light touch. The gate is "enough raw material generated" not "quality approved."

---

### Phase 2: Entity Extraction

**Purpose:** Distill entities from discussion notes.

**LLM Involvement:** Summarize

LLM reviews discussion and proposes entity list:

```yaml
entities:
  - id: kay
    type: character
    concept: "Young archivist drawn into conspiracy"
    notes: "Curious, principled, out of her depth. Family connection to archive."

  - id: mentor
    type: character
    concept: "Senior archivist with hidden agenda"
    notes: "Ambiguous loyalty. Knows more than they reveal. Could be protector or manipulator."

  - id: archive
    type: location
    concept: "Ancient repository of forbidden knowledge"
    notes: "Layered structure—public areas, restricted sections, forbidden depths."

  - id: cipher_device
    type: object
    concept: "Artifact that reveals hidden text"
    notes: "Central to both investigation and danger. Origin unclear."
```

**Entity types:** character, location, object, faction (extensible)

**For each entity:**
- `id`: Short identifier
- `type`: Category
- `concept`: One-line essence
- `notes`: Freeform context from discussion

### Entity Type Distribution (Guidelines)

For a typical story targeting 15-25 entities, aim for a balanced mix:

| Type | Range | Purpose |
|------|-------|---------|
| Characters | 6-10 | Protagonist, antagonist, allies, suspects, supporting cast |
| Locations | 4-6 | Scene settings; enables intersection formation and scene variety |
| Objects | 4-6 | Puzzles, MacGuffins, clues, symbolic items |
| Factions | 1-3 | Organizations, groups, collectives |

**Location vs Object Classification:**
- **Location**: Has a physical space where scenes can occur; characters navigate to/from it
- **Object**: Carried, single-use, or subordinate to a location

Example: "the_clock_in_the_hallway" implies the hallway is a location; the clock is an object within it. If scenes will happen in the hallway, create it as a location.

**Why locations matter:** SEED requires at least 2 different locations for scene variety. A story with only 1 location limits scene pacing and prevents natural intersection formation in GROW.

**Human Gate:** Yes

Human reviews entity list:
- Any missing? (Add from discussion)
- Any weak/redundant? (Note for SEED to cut)
- Any mischaracterized? (Edit concept/notes)

**Artifacts Modified:**
- Entity list (working draft)

**Completion Criteria:**
- All significant entities from discussion captured
- Human has reviewed list

---

### Phase 3: Dilemma Formation

**Purpose:** Frame dramatic questions as binary dilemmas.

**LLM Involvement:** Summarize

LLM reviews discussion and proposes dilemmas:

```yaml
dilemmas:
  - id: d::mentor_trust
    question: "Can the mentor be trusted?"
    answers:
      - id: mentor_protector
        description: "Mentor is genuinely protecting Kay from forces she doesn't understand"
        canonical: true
      - id: mentor_manipulator
        description: "Mentor is using Kay to access forbidden knowledge"
        canonical: false
    involves: [mentor, kay]
    why_it_matters: "Trust determines whether Kay has an ally or is alone against the conspiracy"

  - id: d::archive_nature
    question: "Is the archive's knowledge salvation or corruption?"
    answers:
      - id: archive_salvation
        description: "The forbidden knowledge can save the world if used wisely"
        canonical: true
      - id: archive_corruption
        description: "The forbidden knowledge corrupts all who access it"
        canonical: false
    involves: [archive, kay, cipher_device]
    why_it_matters: "Determines whether Kay's quest is heroic or tragic"
```

**For each dilemma:**
- `id`: Short identifier with `d::` prefix
- `question`: The dramatic question (ends with ?)
- `answers`: Exactly two (canonical + non-canonical)
- `involves`: Which entities are central to this dilemma
- `why_it_matters`: Thematic stakes

**Canonical flag:** One answer is marked `canonical: true`. This becomes the spine path. The non-canonical answer may become a branch if explored in SEED.

**Human Gate:** Yes

Human reviews dilemmas:
- Are questions genuinely dramatic? (Stakes matter)
- Are answers genuine contrasts? (Not shades of gray)
- Is canonical choice appropriate? (Best default story)
- Are entity involvements correct?

**Artifacts Modified:**
- Dilemma list (working draft)

**Completion Criteria:**
- All major dramatic questions captured as dilemmas
- Each dilemma has exactly two answers
- Human has reviewed and approved dilemmas

---

### Phase 4: Serialization

**Purpose:** Convert approved artifacts to structured YAML.

**LLM Involvement:** None (deterministic formatting)

Take approved entities and dilemmas, format as `02-brainstorm.yaml`:

```yaml
brainstorm:
  entities:
    - id: kay
      type: character
      concept: "Young archivist drawn into conspiracy"
      notes: "Curious, principled, out of her depth..."
    # ... all entities

  dilemmas:
    - id: d::mentor_trust
      question: "Can the mentor be trusted?"
      answers:
        - id: mentor_protector
          description: "Mentor is genuinely protecting Kay..."
          canonical: true
        - id: mentor_manipulator
          description: "Mentor is using Kay..."
          canonical: false
      involves: [mentor, kay]
      why_it_matters: "Trust determines..."
    # ... all dilemmas
```

**Human Gate:** No (deterministic)

**Output:** `02-brainstorm.yaml`

---

## Human Gates Summary

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Discussion | Light: "enough material?" |
| 2 | Entity Extraction | Review entity list |
| 3 | Dilemma Formation | Review dilemmas |
| 4 | Serialization | None (deterministic) |

---

## Iteration Control

### Forward Progress

Normal flow: Phase 1 → 2 → 3 → 4

### Backward Loops

| From Phase | To Phase | Trigger |
|------------|----------|---------|
| 2 (Entities) | 1 (Discussion) | Major gaps in entity coverage |
| 3 (Dilemmas) | 1 (Discussion) | No clear dramatic questions emerged |
| 3 (Dilemmas) | 2 (Entities) | Dilemma involves entity not in list |

### Maximum Iterations

- Phase 1: No fixed limit (but human decides when "enough")
- Phases 2-3: Max 2 revision passes each
- Phase 4: 1 pass (deterministic)

---

## Context Management

### Standard (128k+ context)

Include full discussion notes in context for Phases 2-3. No windowing needed.

### Constrained (32k context)

If discussion is very long:
1. Human curates discussion highlights
2. Pass highlights (not full transcript) to Phases 2-3
3. Trust human curation to preserve important material

---

## Failure Modes and Recovery

### Phase-Specific Failures

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1. Discussion | Conversation stalls | Human judgment | New prompts, different angle |
| 1. Discussion | Ideas too generic | Human judgment | Push for specificity, add constraints |
| 2. Entities | Missing important entity | Human review | Add manually, note source |
| 2. Entities | Too many entities | Human review | Note as cut candidates for SEED |
| 3. Dilemmas | Non-binary dilemma | Validation | Split into multiple binary dilemmas |
| 3. Dilemmas | Weak contrast | Human review | Sharpen answers or cut dilemma |
| 3. Dilemmas | No canonical obvious | Human review | Human decides canonical |

### Escalation to DREAM

Return to DREAM if:
- Brainstormed content doesn't fit DREAM vision
- Genre/tone mismatch emerges
- Constraints prove too restrictive for interesting dilemmas

---

## Worked Example

### DREAM Input

```yaml
dream:
  genre: "dark fantasy mystery"
  tone: "atmospheric, morally ambiguous"
  themes: ["forbidden knowledge", "trust", "sacrifice"]
  constraints: ["single protagonist POV", "no explicit magic system"]
  length: short
```

### Phase 1: Discussion (excerpts)

**LLM:** "A dark fantasy mystery with forbidden knowledge... What if the protagonist is an archivist? Someone whose job is to preserve and protect knowledge, now facing knowledge that shouldn't be preserved?"

**Human:** "I like that. What draws them in?"

**LLM:** "Perhaps a mentor figure—someone who seems trustworthy but might have their own agenda. The classic 'is this person protecting me or using me?' tension."

**Human:** "Yes. And the archive itself should feel alive, dangerous. Layered."

**LLM:** "Different levels of restriction. Public stacks, restricted collections, forbidden depths. The deeper you go, the more dangerous the knowledge..."

*(Discussion continues, covering characters, locations, artifacts, relationships)*

### Phase 2: Entity Extraction

LLM extracts from discussion:
- Kay (protagonist archivist)
- The Mentor (ambiguous guide)
- The Archive (layered repository)
- The Cipher Device (artifact)
- The Forbidden Depths (dangerous location)
- The Conspiracy (faction)
- Kay's Predecessor (missing character)

Human reviews: Approves all, notes "Predecessor might merge with another character in SEED"

### Phase 3: Dilemma Formation

LLM proposes:
- d::mentor_trust: "Can the mentor be trusted?"
- d::archive_nature: "Is the knowledge salvation or corruption?"
- d::predecessor_fate: "What happened to Kay's predecessor?"
- d::conspiracy_goals: "Is the conspiracy protecting or hoarding?"

Human reviews: Approves first three, marks d::conspiracy_goals as "may cut in SEED—overlaps with d::mentor_trust"

### Phase 4: Serialization

All approved artifacts formatted to `02-brainstorm.yaml`

---

## Design Principle: Expansive Generation

BRAINSTORM's job is **creative abundance**, not structural precision. Generate more than needed. Include tangential ideas. Capture nuance in notes.

SEED exists to triage. Don't pre-triage in BRAINSTORM.

**LLM should:**
- Propose many entities, even minor ones
- Surface multiple possible dilemmas
- Include rich notes from discussion
- Err on the side of inclusion

**Human should:**
- Guide creative direction
- Encourage exploration
- Note concerns without blocking
- Trust SEED to filter

---

## Output Checklist

Before BRAINSTORM is complete, verify:

- [ ] Discussion generated sufficient raw material
- [ ] All significant entities captured with concept and notes
- [ ] All dramatic questions framed as binary dilemmas
- [ ] Each dilemma has canonical and non-canonical answers
- [ ] Entity involvement marked on dilemmas
- [ ] why_it_matters populated for each dilemma
- [ ] `02-brainstorm.yaml` written

---

## Summary

BRAINSTORM transforms DREAM vision into raw creative material:

| Input | Output |
|-------|--------|
| Genre, tone, themes | 15-25 entities |
| Constraints | 4-8 dilemmas (binary) |
| Core dilemmas (informal) | Rich notes throughout |

BRAINSTORM generates freely. SEED triages. This separation keeps creative exploration unconstrained while ensuring eventual structure.
