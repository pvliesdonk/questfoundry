# QuestFoundry v5 — FILL Algorithm Specification

**Status:** Specification Complete
**Parent:** questfoundry-v5-spec.md
**Purpose:** Detailed specification of the FILL stage mechanics

> For the narrative description of the FILL stage, see [Document 1, Part 5](../how-branching-stories-work.md). This document provides the detailed algorithm specification.
>
> **Key changes from Documents 1/3 (2026-02-24):**
> - The Poly-State Prose section below is **superseded** by ADR-015 (residue beats) and Document 1. See the note on that section.
> - FILL receives character arc metadata produced by POLISH. See [Document 1, Part 4](../how-branching-stories-work.md) and [Document 3, Part 1](../document-3-ontology.md).
> - FILL's input comes from POLISH (not directly from GROW). The passage layer, choices, and state flags are created by POLISH.

---

## Overview

FILL transforms passage summaries into prose. It takes a validated story graph from POLISH and produces playable content.

**Input:**
- Validated topology (passages with summaries, choice edges)
- DREAM vision (genre, tone, themes)
- GROW artifacts (arcs, beats with scene_type, entity states)
- Entities and relationships

**Output:**
- All passages with `prose` populated
- Entity updates (micro-details discovered during prose generation)
- Relationship updates

---

## Core Concepts

### Voice Document

The voice document captures the stylistic identity of the story. It's created at the start of FILL, not during DREAM, because:

1. **DREAM is high-level vision.** "Dark fantasy, morally ambiguous" doesn't specify POV or tense.
2. **GROW reveals structure.** Arc shapes, scene types, and beat content inform voice choices.
3. **Voice needs concrete decisions.** First person or third? Past tense or present? These must be locked before prose generation.

The voice document is a **contract** for all FILL calls. Every prose generation receives it, ensuring consistency across 50+ passages.

**What it contains:**

| Field | Purpose |
|-------|---------|
| `pov` | Point of view (first, second, third_limited, third_omniscient) |
| `pov_character` | Whose perspective (for limited POVs) |
| `tense` | Past or present |
| `register` | Formality and style (formal, conversational, literary, sparse) |
| `sentence_rhythm` | Pacing pattern (varied, punchy, flowing) |
| `tone_words` | Adjectives describing the voice (terse, wry, melancholic) |
| `avoid_words` | Words/phrases to never use |
| `avoid_patterns` | Patterns to avoid (adverb-heavy, said-bookisms) |
| `exemplar_passages` | Optional examples of the target voice |

### Sequential Generation

FILL generates passages one at a time, in order. Not parallel.

**Why sequential?**

1. **Voice consistency.** Each passage builds on the voice established by previous passages. The sliding window of recent prose reinforces consistent style.

2. **Continuity.** Details established in passage 5 (a character's gesture, a room's description) should carry forward to passage 6.

3. **Convergence handling.** Branch passages need to know what they're converging toward. Sequential generation ensures convergence passages exist before branches approach them.

**Why spine first?**

The spine arc is the canonical route—all canonical path answers. It establishes:
- The baseline voice
- The canonical version of convergence passages
- The reference point for branch variations

Branches then write toward established spine content, not into a void.

### Sliding Window

Each prose generation call includes recent passages for context. This serves two purposes:

1. **Voice reinforcement.** The LLM sees how recent passages sound, maintaining consistency.
2. **Detail continuity.** Minor details (character descriptions, environmental elements) carry forward naturally.

**What's in the window:**

- Generated prose (not just summaries)
- The most recent N passages in arc order
- Recommended: 3-5 passages (implementation-dependent based on context budget)

**Why prose, not summaries?**

Summaries capture *what happens*. Prose captures *how it sounds*. For voice consistency, the LLM needs to see actual prose.

### Lookahead Strategy

At structural junctures (divergence and convergence points), the LLM needs awareness of what comes before or after to write smooth transitions.

**Convergence (spine pass):**

When writing a convergence passage during spine generation, branches haven't been written yet. But their beat summaries exist. Include:
- Beat summaries of all connecting branches
- Path context (which answers arrive here)

This lets the convergence passage be written with awareness of all arrivals, even without their prose.

**Convergence (branch pass):**

When writing branch passages approaching convergence, the convergence prose exists (written during spine). Include:
- The convergence passage prose as lookahead
- The branch writes *toward* this established target

**Divergence (branch pass):**

When writing the first branch-specific passage after divergence, include:
- The divergence passage prose (for continuity)
- The branch picks up smoothly from the shared content

### Scene Type → Prose Guidance

Each beat has a `scene_type` assigned during GROW. This guides prose structure.

| Scene Type | Prose Guidance |
|------------|----------------|
| `scene` | Full dramatic structure. Goal, obstacle, outcome. Typically 3+ paragraphs. |
| `sequel` | Reactive processing. Reaction, dilemma, decision. Breathing room. 2-3 paragraphs. |
| `micro_beat` | Brief transition. Time passage, minor moment. 1 paragraph. |

**Craft notes (informative, not exhaustive):**

These are guidelines, not rigid rules. A skilled author might write a one-paragraph scene or a four-paragraph sequel. The guidance helps the LLM make reasonable default choices.

**Scene structure (when scene_type = scene):**

A full scene often follows a three-part cadence:

1. **Lead** — Sensory grounding. The character in motion, concrete imagery. Establishes where we are and what's happening.

2. **Middle** — Goal and obstacle. What the character wants, what's in the way. Rising dilemma.

3. **Close** — Decision setup. The scene ends with stakes clarified and choices meaningful. For choice points, this paragraph sets up the options.

**Sequel structure (when scene_type = sequel):**

Sequels provide breathing room after intense scenes:

1. **Reaction** — Emotional response to what just happened.
2. **Dilemma** — Processing options, weighing consequences.
3. **Decision** — Choosing a direction (may lead directly to choice options).

**Micro-beat structure:**

One paragraph. Functional prose that moves the story forward without dwelling. Time transitions, brief observations, minor interactions.

### Entity Updates

During prose generation, the LLM may invent micro-details about entities:
- Physical appearance ("a wiry man with a scar across his left eye")
- Mannerisms ("she had a habit of tapping her ring against the table")
- Voice characteristics ("his voice was softer than expected")

These details should be captured for consistency in later passages.

**Rules:**

1. **Updates only.** FILL cannot create new entities. Only existing entities can be updated.
2. **Additive details.** Updates add information, not contradict existing state.
3. **Automatic capture.** The FILL output schema includes `entity_updates` for this purpose.

If prose reveals that a new recurring entity is needed, FILL should flag and pause for human review. Creating the entity requires returning to SEED.

### Poly-State Prose (Shared Beats)

> **Superseded by ADR-015 and Document 1.** Poly-state prose has been replaced by residue beats (see [ADR-015](../../architecture/decisions.md#adr-015-residue-beats-replace-poly-state-prose)). Shared passages are kept neutral; residue beats set path-specific emotional context before convergence points. The `flag: incompatible_states` escape hatch described below no longer exists. This section is retained for historical context.

Shared beats (path-agnostic) appear in multiple arcs. When writing shared beats, FILL must produce **poly-state prose**: prose that is diegetically accurate for the active state but compatible with all shadow states.

**The challenge:**

A shared beat might be reached from paths where the character has different knowledge or emotional states. The prose must work for all arrivals without contradicting any.

**Context provided to LLM:**

When writing a shared beat, the LLM receives:
- The active state (the arc currently being generated)
- Shadow states (other valid paths that share this beat)
- Instruction: "Write prose that is true for the active state but does not contradict shadow states"

**Success example:**

Beat: "Kay confronts the Mentor about the warning"
- Active state: Kay suspects betrayal
- Shadow state: Kay trusts the Mentor

Poly-state prose: "Kay studied the Mentor's face, searching for the person she thought she knew. 'Why didn't you tell me sooner?'"

This works for both states—suspicious Kay searching for deception, trusting Kay searching for reassurance.

**Failure example:**

Prose: "Kay gripped the knife hidden in her sleeve, knowing the Mentor was a traitor."

This only works for the suspicious state. It contradicts the trusting path.

**The escape hatch:**

If the LLM cannot write poly-state prose due to extreme emotional divergence between paths, it must **flag the beat for splitting**:

```yaml
fill_output:
  passage_id: mentor_confrontation
  prose: null
  flag: incompatible_states
  flag_reason: "Character's emotional state too divergent—suspicious path requires hostile body language incompatible with trusting path"
```

Flagged beats return to GROW for splitting into separate path-specific passages.

**When to flag (guidance for LLM):**
- Internal monologue requires contradictory knowledge
- Body language or actions only make sense for one state
- Dialogue would reveal information only known on one path
- Emotional register is fundamentally different (rage vs warmth)

**When NOT to flag:**
- Ambiguous phrasing can accommodate both states
- Universal emotions apply (uncertainty, curiosity, hope)
- Actions are neutral (observation, movement, waiting)

---

## Algorithm Phases

### Phase 0: Voice Determination

**Purpose:** Establish the voice document that governs all prose generation.

**Input:**
- DREAM vision (genre, tone, themes)
- GROW artifacts (arc structures, beat summaries, scene types)
- Sample beat summaries (to understand content being voiced)

**Operations:**

1. LLM analyzes inputs and proposes voice document:
   - POV and tense based on genre conventions and story needs
   - Register and rhythm based on tone
   - Tone words distilled from DREAM themes
   - Avoid patterns based on genre anti-patterns

2. Human reviews proposal:
   - Approve: proceed with proposed voice
   - Modify: adjust specific fields
   - Override: provide complete voice document manually

3. Optionally, generate 1-2 exemplar passages:
   - Pick representative beat summaries
   - Generate sample prose in proposed voice
   - Include as exemplars if quality is good

**Output:** Approved voice document

**Human Gate:** Required. Voice document must be approved before prose generation begins.

**LLM Involvement:** Proposal generation, exemplar generation (optional)

---

### Phase 1: Sequential Prose Generation

**Purpose:** Generate prose for all passages in order.

**Input:**
- Voice document (from Phase 0)
- Passages with summaries and scene_type
- Arc structure (traversal order)
- Entity states (computed from codewords)
- Path definitions (for shadows)

**Traversal Order:**

1. Spine arc, start to finish
2. For each branch arc (in any consistent order):
   - Start from divergence point
   - Generate branch-specific passages
   - End at convergence or arc end

**Per-Passage Context:**

| Component | Content |
|-----------|---------|
| Voice document | Full document |
| Beat summary | Current passage's summary |
| Scene type | `scene`, `sequel`, or `micro_beat` |
| Entity states | Relevant entities at this point (base + applicable overlays) |
| Shadows | Unexplored answers for active dilemmas |
| Sliding window | Last N passages of generated prose (recommended 3-5) |
| Lookahead | See lookahead strategy in Core Concepts |
| Path context | For intersections: which paths this beat serves |

**Per-Passage Operations:**

1. Assemble context (all components above)
2. LLM generates prose following:
   - Voice document constraints
   - Scene type guidance
   - Beat summary content
3. LLM outputs:
   - `prose`: The generated passage text
   - `entity_updates`: Any micro-details to capture (optional)
   - `relationship_updates`: Any relationship details (optional)
4. Store prose in passage
5. Apply entity/relationship updates
6. Advance to next passage

**Convergence Passage Handling:**

When generating a convergence passage (spine pass):
- Include beat summaries of all connecting branches as lookahead
- Write prose that works for all arrivals
- This passage will be referenced by branches later

**Output:** All passages with prose populated

**Human Gate:** After all passages generated, human approves to proceed to review. May also spot-check during generation for long stories.

**LLM Involvement:** All prose generation

---

### Phase 2: Review

**Purpose:** Identify passages that need revision.

**Input:** All passages with prose

**Review Mechanisms:**

Implementation may use any combination:

1. **Human review:** Read passages, flag weak ones
2. **LLM review:** Pass passages through review prompt, get flags
3. **Hybrid:** LLM flags candidates, human makes final call

**LLM Review Approach:**

If using LLM review, context limits require sliding window:
- Review N passages at a time (recommended 5-10)
- Overlap windows for continuity assessment
- Flag passages with issues

**Review Criteria:**

| Issue | Detection |
|-------|-----------|
| Voice drift | Passage sounds different from surrounding prose |
| Scene type mismatch | Full scene in one paragraph, micro-beat sprawling |
| Summary deviation | Prose doesn't match beat summary content |
| Continuity break | Details contradict earlier passages |
| Convergence awkwardness | Passage doesn't work for one of the arriving branches |
| Flat prose | Lacks dilemma, sensory detail, or emotional engagement |

**Output:** List of flagged passages with issue descriptions

**Human Gate:** Human reviews flagged list, may add/remove items, approves revision targets.

**LLM Involvement:** Optional (for LLM review mechanism)

---

### Phase 3: Revision

**Purpose:** Regenerate flagged passages.

**Input:**
- Flagged passages with issue descriptions
- All prose (including unflagged passages for context)

**Operations:**

For each flagged passage:

1. Assemble context:
   - Voice document
   - Issue description (explicit in context)
   - Extended sliding window (more passages for revision)
   - For convergence passages: all approach passages now exist
2. LLM regenerates prose addressing the flagged issue
3. Store revised prose
4. Apply any entity/relationship updates

**Revision Context Extensions:**

| Situation | Additional Context |
|-----------|-------------------|
| Voice drift | Include more exemplar passages showing correct voice |
| Convergence issues | Include all approach passage prose |
| Continuity break | Include the contradicted passage explicitly |

**Output:** Revised passages

**Human Gate:** After revisions complete, human reviews revised passages.

**LLM Involvement:** All regeneration

---

### Phase 4: Optional Second Cycle

**Purpose:** Final quality pass if needed.

**Operations:**

If human is unsatisfied after Phase 3:
1. Return to Phase 2 (review)
2. Flag remaining issues
3. Revise (Phase 3)

**Hard Cap:** Maximum 2 review-revise cycles.

After cap reached, ship as-is. Persistent quality issues indicate upstream problems:
- Voice document doesn't match the story
- Beat summaries are too vague
- Structure needs rework (return to SEED/GROW)

The cap is configurable and human-overridable, but defaults to 2.

**Human Gate:** Final approval after last cycle.

---

## Human Gates Summary

| After Phase | Human Decision |
|-------------|----------------|
| 0. Voice | Approve/modify voice document |
| 1. Generation | Approve to review (may spot-check during) |
| 2. Review | Approve revision targets |
| 3. Revision | Approve revisions (or trigger Phase 4) |
| 4. Second cycle | Final approval |

---

## Failure Modes and Recovery

FILL failures are generally recoverable within the stage. Unlike GROW, prose problems rarely require returning to earlier stages.

### Phase-Specific Failures

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 0. Voice | Voice doesn't fit story | Human review | Iterate on voice document |
| 0. Voice | Exemplars inconsistent | Human review | Regenerate exemplars or skip |
| 1. Generation | Voice drift mid-story | Review (Phase 2) | Revise with stronger voice reinforcement |
| 1. Generation | Passage too short/long | Review | Revise with explicit length guidance |
| 1. Generation | Summary hallucination | Review | Revise with summary emphasized |
| 1. Generation | Entity detail conflicts | Later generation fails | Update entity, revise both passages |
| 1. Generation | Poly-state prose impossible | LLM flags `incompatible_states` | Return to GROW, split beat into path-specific passages |
| 2. Review | Too many flags | Human overwhelm | Prioritize, accept some imperfection |
| 3. Revision | Revision doesn't fix issue | Human review | Try different approach or accept |

### Structural Failures (Abort to Earlier Stage)

These are rare but indicate upstream problems:

| Condition | Why Abort | Target |
|-----------|-----------|--------|
| Beat summaries too vague for prose | Structure issue | Return to GROW, improve summaries |
| Convergence fundamentally broken | Arcs incompatible | Return to GROW, fix convergence |
| Voice impossible for content | DREAM/structure mismatch | Return to DREAM or SEED |
| New entity needed | Can't create in FILL | Return to SEED, add entity |

### Quality Threshold

FILL has inherent subjectivity. "Good prose" varies by reader. The procedure provides:
- Structure (voice document, scene types)
- Review mechanism
- Iteration cap

It does not guarantee prose quality. Quality ultimately depends on:
- LLM capability
- Prompt design (out of scope for this procedure)
- Human curation effort

---

## Iteration Control

**Principle:** Generate once, review once, revise targeted passages. Cap total cycles.

| Phase | Iteration | Fallback |
|-------|-----------|----------|
| Voice determination | Until human approves | 3 proposals max, then human writes |
| Generation | One pass, strictly sequential | N/A |
| Review | One pass (windowed if LLM) | N/A |
| Revision | One pass per flagged passage | N/A |
| Full cycle | 2 maximum | Ship as-is, note quality concerns |

**Why strict limits?**

Prose quality is unbounded. You can always revise more. Without caps:
- Diminishing returns on iteration
- Human fatigue
- Lost time better spent on structure

The cap forces "good enough" decisions. If prose consistently fails after 2 cycles, the problem is upstream (voice, structure, or model capability).

---

## Context Management

### Standard (128k+ context)

Generous context allows:
- Full voice document (~300-400 tokens)
- Extended sliding window (5+ passages, ~2,000+ tokens)
- Full entity states (~500 tokens)
- Lookahead passages (~400 tokens)
- Output buffer (~500 tokens)

**Total per call:** ~4,000-5,000 tokens

No windowing needed. Include everything.

### Constrained (32k context)

Tight context requires choices:
- Reduce sliding window (2-3 passages)
- Summarize distant entity states
- Shorter voice document (drop exemplars)

**Budget guidance:**

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| Voice document | 300 | 150 (no exemplars) |
| Beat summary + scene_type | 100 | 80 |
| Entity states | 400 | 200 (relevant only) |
| Shadows | 150 | 100 |
| Sliding window | 1,200 (3 passages) | 600 (2 passages) |
| Lookahead | 400 | 200 |
| Output buffer | 500 | 400 |
| **Total** | ~3,050 | ~1,730 |

For 32k context, this leaves ample room for system prompt and generation.

### Implementation Decisions

The procedure specifies *what* to include, not exact sizes. Implementation should:
- Measure actual token usage for the specific model
- Adjust window sizes based on available context
- Prioritize voice document and current beat (never truncate these)
- Truncate sliding window first if needed

---

## Worked Example

### Setup

**Story:** 2 dilemmas, 4 arcs total
- Mentor trust: protector (canonical) vs manipulator
- Artifact nature: saves (canonical) vs corrupts

**Arcs:**
- Spine: mentor_protector + artifact_saves
- Branch A: mentor_protector + artifact_corrupts
- Branch B: mentor_manipulator + artifact_saves
- Branch C: mentor_manipulator + artifact_corrupts

**Passages (simplified):**

| ID | Beat Summary | Scene Type | Arcs |
|----|--------------|------------|------|
| p1 | Kay enters the tower, meets the mentor | scene | all |
| p2 | Mentor explains the artifact's power | scene | all |
| p3_spine | Mentor reveals protective intent, gives artifact | scene | spine, A |
| p3_branch | Mentor's manipulation exposed, takes artifact | scene | B, C |
| p4_spine | Kay uses artifact, it heals the land | scene | spine |
| p4_corrupt | Kay uses artifact, it corrupts her | scene | A, C |
| p4_branch | Kay uses artifact against mentor, saves land | scene | B |
| p5 | Aftermath: Kay reflects on choices | sequel | all (convergence) |

### Phase 0: Voice Determination

**DREAM vision:** Dark fantasy, morally ambiguous, themes of trust and power.

**LLM proposes:**

```yaml
voice:
  pov: third_limited
  pov_character: kay
  tense: past
  register: literary
  sentence_rhythm: varied
  tone_words: [atmospheric, tense, intimate]
  avoid_words: [suddenly, very, really]
  avoid_patterns: [adverb-heavy dialogue tags, excessive internal monologue]
  exemplar_passages: []
```

**Human approves** with modification: adds "melancholic" to tone_words.

### Phase 1: Sequential Prose Generation

**Spine generation (passages in order):**

**p1** (Kay enters tower, meets mentor):
- Context: voice doc, beat summary, scene_type=scene, no sliding window yet
- Generated prose:

> The tower stairs wound upward into darkness, each step groaning beneath Kay's weight. She had expected dust, decay—the usual markers of abandonment. Instead, candlelight flickered from above, warm and steady.
>
> The chamber at the top held more books than walls. And in their midst, a figure turned to face her. The mentor. Older than the stories suggested, but his eyes held the same unsettling clarity.
>
> "You found the way," he said. Not a question. "Sit. We have much to discuss, and the night is not as long as you might hope."

**p2** (Mentor explains artifact):
- Context: voice doc, beat summary, scene_type=scene, sliding window [p1]
- Generated prose:

> The artifact lay on the table between them, no larger than Kay's fist. In the candlelight, it seemed to pulse—though perhaps that was her imagination, her exhaustion, the wine the mentor had poured without asking.
>
> "It responds to intent," the mentor said, his fingers hovering but not touching. "The pure of heart find it heals. The corrupt find it... illuminating." A thin smile. "Most find it somewhere between."
>
> Kay reached for it, then stopped. "And what did you find?"
>
> The mentor's smile didn't waver. "I found that some questions are better lived than answered."

**p5** (Convergence - aftermath):
- Context: voice doc, beat summary, scene_type=sequel, sliding window [p2], lookahead: beat summaries for all arriving branches
- This is a convergence passage. Written during spine pass with awareness of all arrivals.
- Generated prose:

> The land stretched before her, changed. Whether healed or scarred depended on angles, on light, on what Kay allowed herself to remember.
>
> She sat on the tower steps—the same steps she had climbed a lifetime ago, though the candles had long since guttered. The mentor was gone. The artifact was gone, or changed, or perhaps had never been what she thought.
>
> What remained was the weight of choices made. Not regret, exactly. Something quieter. The knowledge that every path closed others, and that she would never know what those other paths held.
>
> She rose. There was still work to do, in whatever world she had made.

**Branch B generation** (mentor_manipulator + artifact_saves):

Starts at divergence. p1 and p2 are shared (already generated).

**p3_branch** (Mentor's manipulation exposed):
- Context: voice doc, beat summary, scene_type=scene, sliding window [p1, p2], divergence passage p2 for continuity
- Generated prose:

> The question hung between them. Kay watched the mentor's face—watched the smile falter, just slightly, at the edges.
>
> "What did I find?" he repeated. And something shifted in his eyes. The warmth Kay had thought she'd seen guttered like a candle in wind. "I found power, girl. The same thing everyone finds, if they're honest."
>
> His hand closed over the artifact before she could react. "And I found that honesty is a luxury. One I can no longer afford."
>
> Kay's hand went to her knife. Too slow. Always too slow.

**p4_branch** (Kay uses artifact against mentor):
- Context: voice doc, beat summary, scene_type=scene, sliding window [p2, p3_branch], lookahead: p5 (convergence prose)
- Branch writes toward the established convergence.
- Generated prose:

> The struggle was brief and brutal. The mentor had age and knowledge; Kay had desperation and the knife she'd carried since childhood. In the end, neither mattered.
>
> The artifact chose.
>
> It burned in her grip—not with heat, but with something colder. Intent. And as the light poured through her, Kay understood what the mentor had never grasped: the artifact didn't respond to purity or corruption. It responded to truth.
>
> The mentor crumpled. The land beyond the tower window shuddered, then stilled. Healed.
>
> Kay dropped the artifact. It lay there, inert, its purpose spent or simply waiting. She didn't want to know which.

### Phase 2: Review

**LLM review flags:**
- p4_corrupt (not shown): "Voice drift—more melodramatic than surrounding passages"
- p3_spine (not shown): "Convergence approach—check works with branch arrivals"

**Human reviews:**
- Accepts p4_corrupt flag
- Dismisses p3_spine concern (convergence is p5, not p3)

### Phase 3: Revision

**p4_corrupt revision:**
- Context: voice doc, issue description ("reduce melodrama, match atmospheric tone"), extended sliding window
- Regenerated with corrected tone

### Phase 4: Not Needed

Human approves after Phase 3. Done.

---

## Design Principle: LLM Generates, Human Curates

FILL is the most generative stage. Unlike GROW (where LLM proposes structural decisions), FILL asks the LLM to create content.

**LLM responsibilities:**
- Generate prose matching voice document
- Follow scene type guidance
- Maintain continuity via sliding window
- Capture entity details

**Human responsibilities:**
- Approve voice document (sets the target)
- Review prose quality (subjective judgment)
- Decide revision targets (prioritize effort)
- Accept "good enough" (avoid infinite polishing)

**The human does not need to:**
- Write prose from scratch
- Manually track entity details
- Remember voice constraints while reviewing

**Quality depends on collaboration:**
- LLM provides volume and consistency
- Human provides taste and curation
- Voice document bridges the gap

---

## Summary

FILL is 5 phases:

| # | Phase | LLM | Human Gate |
|---|-------|-----|------------|
| 0 | Voice determination | Yes (proposal) | Yes |
| 1 | Sequential generation | Yes (all prose) | Yes (approve to review) |
| 2 | Review | Optional (LLM review) | Yes (approve targets) |
| 3 | Revision | Yes (regeneration) | Yes (approve revisions) |
| 4 | Optional second cycle | Yes | Yes (final approval) |

**Phase distribution:**
- LLM-heavy (0, 1, 3): Content generation
- Human-heavy (2, 4): Quality judgment
- All phases have human gates

**Core principle:** LLM generates prose under voice document constraints. Human curates quality through review and revision targeting. Iteration is capped because prose quality is subjective—"good enough" beats "perfect forever."
