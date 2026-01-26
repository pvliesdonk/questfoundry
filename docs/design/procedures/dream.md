# DREAM Procedure

## Summary

**Purpose:** Establish the creative vision that guides all subsequent stages. Capture genre, tone, themes, constraints, and scope.

**Input artifacts:**
- Human's initial spark (informal idea)

**Output artifacts:**
- `01-dream.yaml` (approved vision)

**Mode:** Human-led with LLM assistance. The human has the vision; the LLM helps articulate, explore, and document it.

---

## Prerequisites

### Required Input Files

None. DREAM is the first stage.

### Required Human Decisions

- Initial story spark or concept (can be vague: "noir mystery in space")

### Knowledge Context

Inject for LLM:
- Genre conventions (from IF-Craft Corpus)
- Interactive fiction craft knowledge (pacing, branching patterns)
- Web search capability for research

---

## Core Concepts

### Human Leads, LLM Assists

Unlike BRAINSTORM (where LLM generates expansively), DREAM is about **capturing human intent**. The LLM's role is:
- Ask clarifying questions
- Suggest possibilities the human might not have considered
- Provide genre knowledge when relevant
- Help articulate vague ideas into concrete terms

The human decides. The LLM facilitates.

### Vision vs Structure

DREAM captures **what kind of story** (vision), not **what happens in the story** (structure). The distinction:

| DREAM (Vision) | BRAINSTORM (Structure) |
|----------------|------------------------|
| "Dark fantasy mystery" | Characters, locations, factions |
| "Themes of trust and sacrifice" | "Can the mentor be trusted?" dilemma |
| "Morally ambiguous tone" | Specific moral dilemmas |
| "Short length" | ~15 entities, ~6 dilemmas |

DREAM stays abstract. BRAINSTORM makes it concrete.

### Constraints as Boundaries

Constraints define what the story **won't** do:
- Content boundaries ("no explicit violence")
- Structural limits ("single protagonist POV")
- Setting limits ("contemporary, no fantasy elements")
- Scope limits ("standalone, no sequel hooks")

Good constraints enable creativity by defining the sandbox.

---

## Algorithm Phases

### Phase 1: Spark Exploration

**Purpose:** Expand the initial spark into a fuller vision.

**LLM Involvement:** Discuss

The human provides an initial idea. LLM explores it through questions:

**Discussion prompts:**

*Genre and tone:*
- "What genre feels right? Any subgenres or mashups?"
- "What's the emotional register? Dark, light, bittersweet?"
- "What existing works have the feel you're going for?"

*Themes:*
- "What ideas do you want this story to explore?"
- "What questions should the reader be asking?"
- "What makes this story meaningful to you?"

*Setting and scope:*
- "What world does this take place in?"
- "How long do you envision this being? A short experience or an epic?"
- "What's the player's role in this world?"

**LLM role:** Curious, supportive. Ask follow-up questions. Offer options when the human seems uncertain. Use corpus knowledge to suggest genre conventions.

**Human role:** Share the vision. React to suggestions. Make decisions.

**Output:** Discussion notes (conversation history)

**Human Gate:** Light touch. Continue until vision feels sufficiently explored.

---

### Phase 2: Constraint Definition

**Purpose:** Define the boundaries of the story.

**LLM Involvement:** Assist

Once the vision is clear, define constraints:

**Discussion prompts:**

*Length and scope:*
- "How long should this be? A quick read (micro) or a substantial experience (long)?"
- "How much branching complexity feels right?"

*Content boundaries:*
- "Any content to avoid? (violence level, themes, etc.)"
- "Any content that must be included?"

*Structural constraints:*
- "Single POV or multiple?"
- "Linear timeline or non-linear?"
- "Any format constraints? (text-only, illustrated, etc.)"

**Length calibration:**

| Length | Word Count | Passages | Playtime |
|--------|------------|----------|----------|
| micro | 2-5k | 10-20 | 5-15 min |
| short | 5-15k | 20-50 | 15-45 min |
| medium | 15-40k | 50-100 | 1-2 hours |
| long | 40k+ | 100+ | 2+ hours |

**LLM role:** Help quantify vague preferences. Suggest constraints the human might not have considered.

**Human role:** Make boundary decisions. These are firm commitments.

**Output:** Constraint list

**Human Gate:** Yes. Constraints must be explicitly approved.

---

### Phase 3: Vision Synthesis

**Purpose:** Consolidate discussion into structured artifact.

**LLM Involvement:** Summarize

LLM synthesizes the discussion into the dream schema:

```yaml
dream:
  genre: string           # e.g., "dark fantasy mystery"
  tone: string            # e.g., "atmospheric, morally ambiguous"
  themes: string[]        # e.g., ["forbidden knowledge", "trust", "sacrifice"]
  constraints: string[]   # e.g., ["single protagonist POV", "no explicit magic system"]
  length: micro | short | medium | long
```

**For each field:**
- `genre`: Primary genre with any subgenres or mashups
- `tone`: Emotional register, atmosphere descriptors
- `themes`: Abstract ideas the story explores (not plot points)
- `constraints`: Firm boundaries (content, structure, scope)
- `length`: Scope commitment

**LLM role:** Extract and organize. Do not invent—only capture what was discussed.

**Human role:** Review for accuracy.

**Output:** Draft `01-dream.yaml`

**Human Gate:** Yes. This is the approval gate.

---

### Phase 4: Approval

**Purpose:** Final sign-off on the vision.

**LLM Involvement:** None

Human reviews the complete artifact:
- Does this capture my vision?
- Are the constraints correct?
- Is the length appropriate?

If approved, DREAM is complete. If not, return to relevant phase.

**Human Gate:** Yes. Must explicitly approve.

**Output:** Approved `01-dream.yaml`

---

## Human Gates Summary

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Spark Exploration | Light: "vision explored enough?" |
| 2 | Constraint Definition | Yes: approve constraints |
| 3 | Vision Synthesis | Yes: review draft |
| 4 | Approval | Yes: final sign-off |

---

## Iteration Control

### Forward Progress

Normal flow: Phase 1 → 2 → 3 → 4

### Backward Loops

| From Phase | To Phase | Trigger |
|------------|----------|---------|
| 2 (Constraints) | 1 (Exploration) | Constraints reveal vision gaps |
| 3 (Synthesis) | 1 (Exploration) | Synthesis shows underdeveloped areas |
| 3 (Synthesis) | 2 (Constraints) | Missing or unclear constraints |
| 4 (Approval) | Any | Human requests changes |

### Maximum Iterations

- Phase 1: No fixed limit (human decides when "enough")
- Phase 2: Max 2 revision passes
- Phase 3: Max 2 synthesis attempts
- Phase 4: 1 pass (approval or loop back)

---

## Context Management

### Standard (128k+ context)

Include full discussion history for synthesis. No windowing needed.

### Constrained (32k context)

If discussion is very long:
1. Human curates key points from discussion
2. Pass curated summary to Phase 3
3. Trust human curation to preserve important decisions

---

## Failure Modes and Recovery

### Phase-Specific Failures

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1. Exploration | Vision too vague | Human can't answer basic genre/tone questions | More exploration, try different angles |
| 1. Exploration | Vision too specific | Human already has full plot in mind | Fast-track to Phase 3, note detail for BRAINSTORM |
| 2. Constraints | Constraints too restrictive | No interesting stories possible within bounds | Relax constraints, discuss tradeoffs |
| 2. Constraints | Constraints too loose | No clear boundaries | Push for specificity |
| 3. Synthesis | LLM invents content | Artifact contains undiscussed elements | Strip inventions, re-synthesize |
| 3. Synthesis | LLM loses nuance | Artifact oversimplifies vision | Human edits directly or re-discusses |

### Escalation

DREAM has no prior stage to escalate to. If DREAM cannot complete:
- The initial spark may be unsuitable for interactive fiction
- Human should reconsider the concept or approach

---

## Worked Example

### Initial Spark

**Human:** "I want to write a story about an archivist who discovers forbidden knowledge. Dark fantasy, mysterious."

### Phase 1: Exploration

**LLM:** "An archivist with forbidden knowledge—that's rich territory. What draws you to this concept? Is it the tension between preservation and destruction? The corruption of knowledge? The isolation of knowing something others don't?"

**Human:** "The corruption angle. And trust—who can you trust when you know something dangerous?"

**LLM:** "Trust is powerful. So we have themes of forbidden knowledge and trust. What's the tone? You said dark fantasy—are we talking grimdark, or more atmospheric and melancholic?"

**Human:** "Atmospheric. Morally ambiguous rather than grimdark. Think of it like a mystery where you're not sure who to believe."

**LLM:** "Atmospheric, morally ambiguous, mystery elements within dark fantasy. What about the scope? Epic quest or intimate character study?"

**Human:** "Intimate. Focused. Maybe just the archivist and a few key relationships."

### Phase 2: Constraints

**LLM:** "For constraints: single protagonist POV makes sense for intimate focus. Any content boundaries?"

**Human:** "No explicit magic system—keep it mysterious. And no graphic violence."

**LLM:** "Length? Given the intimate scope, short or medium?"

**Human:** "Short. I want it tight and focused."

### Phase 3: Synthesis

```yaml
dream:
  genre: "dark fantasy mystery"
  tone: "atmospheric, morally ambiguous"
  themes:
    - "forbidden knowledge"
    - "trust"
    - "corruption"
  constraints:
    - "single protagonist POV"
    - "no explicit magic system"
    - "no graphic violence"
    - "intimate scope (few key relationships)"
  length: short
```

### Phase 4: Approval

**Human:** "Yes, this captures it."

DREAM complete.

---

## Output Checklist

Before DREAM is complete, verify:

- [ ] Genre clearly identified
- [ ] Tone captured (emotional register, atmosphere)
- [ ] At least 2-3 themes defined
- [ ] Constraints explicitly listed
- [ ] Length committed
- [ ] Human has approved final artifact
- [ ] `01-dream.yaml` written

---

## Summary

DREAM transforms a human's initial spark into an approved creative vision:

| Input | Output |
|-------|--------|
| Vague idea | Defined genre and tone |
| Intuitions | Explicit themes |
| Preferences | Firm constraints |
| Sense of scope | Committed length |

DREAM captures **what kind of story**. BRAINSTORM will derive **what happens in it**.
