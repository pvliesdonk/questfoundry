# Interactive Fiction Craft Guide

A reference corpus for writing compelling interactive fiction with meaningful choices.

---

## Choice Architecture

### The Anatomy of a Meaningful Choice

A meaningful choice in interactive fiction has three essential properties:

1. **Clarity**: The player understands what they're choosing between
2. **Consequence**: The choice materially affects the story or world
3. **Character**: The choice reveals or tests who the protagonist is

Avoid "illusion of choice" where multiple options lead to the same outcome. Players quickly learn to distrust such games.

### Types of Choices

**Branching Choices**: Create divergent story paths. Use sparingly as they multiply content requirements exponentially.

**Stat-Affecting Choices**: Modify character attributes that gate future options. Good for resource management games.

**Relationship Choices**: Affect NPC attitudes and available interactions. Creates social dynamics.

**Knowledge Choices**: Reveal information that changes how players interpret events. Powerful for mysteries.

**Moral Choices**: Test player values without clear "right" answers. Most memorable when consequences are delayed.

### The Hub-and-Spoke Pattern

Structure your story around hubs (safe areas for exploration and preparation) connected by spokes (linear adventure sequences). This balances player agency with authorial control:

- Hubs allow free exploration and relationship building
- Spokes deliver dramatic tension and plot advancement
- Return to hubs feels earned after surviving spokes

---

## Prose Techniques for IF

### Second Person Present Tense

Interactive fiction traditionally uses second person ("you") present tense ("walk") to create immediacy:

> You push open the tavern door. Smoke and laughter wash over you.

This differs from traditional fiction's past tense third person. The present tense reinforces that events are happening NOW and the player is making them happen.

### Brevity and Pacing

Screen reading is tiring. Keep paragraphs short—rarely more than 3-4 sentences. Use white space generously.

Bad:
> The ancient library stretched before you, its towering shelves groaning under the weight of countless leather-bound volumes that generations of scholars had accumulated over the centuries since the monastery's founding during the reign of the third emperor.

Good:
> The library stretches upward, shelf after shelf vanishing into shadow. Dust motes drift through slanted light. Somewhere above, pages rustle—though you see no one.

### Sensory Grounding

Each new location needs sensory anchoring. What does the player:

- See (dominant visual)?
- Hear (ambient sound)?
- Smell/feel (environmental texture)?

This creates presence without lengthy description.

---

## Diegetic Interface Design

### What is Diegetic?

Diegetic elements exist within the story world. Non-diegetic elements exist only for the player (health bars, save icons, chapter titles).

Interactive fiction works best when interface elements feel like part of the world:

**Non-diegetic (avoid)**:
> [LOCKED - Requires: Codeword ASH]

**Diegetic (preferred)**:
> The guard's eyes narrow. "Union members only past this point. Show your badge or move along."

### Gates and Keys

Every locked door needs a key that makes narrative sense:

| Gate Type | Diegetic Key |
|-----------|--------------|
| Physical barrier | Tool, strength, agility |
| Social barrier | Reputation, disguise, bribe |
| Knowledge barrier | Overheard secret, found document |
| Moral barrier | Sacrifice, compromise, commitment |

The player should be able to reason about what might open a gate based on story logic, not game conventions.

---

## World Consistency

### The Codex Pattern

Maintain a codex of established facts:

- Character names, descriptions, relationships
- Location geography and atmosphere
- Timeline of events
- Rules of any magic/technology

Every new scene must be checked against the codex for contradictions.

### Tracking State

Interactive fiction accumulates state:

- What the player knows
- Who the player has met
- What the player has done
- What resources the player has

State should be tracked explicitly and referenced when writing branches. A character shouldn't greet the player as a stranger if they've met before.

### Retroactive Continuity

Sometimes you'll discover a better story requires changing established facts. This is acceptable IF:

- The change improves the experience significantly
- You update ALL references to maintain consistency
- You don't invalidate player choices already made

---

## Genre Conventions

### Fantasy

Fantasy IF often features:

- Clear moral cosmology (good vs evil, order vs chaos)
- Magic with rules and costs
- Quests with defined objectives
- Chosen one narratives

Subvert expectations by making the "dark lord" sympathetic or the "prophecy" a manipulation.

### Mystery

Mystery IF requires careful information management:

- Fair play: all clues must be available before the solution
- Red herrings: some clues should mislead
- Revelation pacing: major discoveries at act breaks
- Multiple solutions: let clever players find shortcuts

### Horror

Horror IF uses:

- Restricted information (what you can't see is scarier)
- Dwindling resources (light, sanity, time)
- Isolation (help is unavailable or untrustworthy)
- Consequence (death or worse should feel possible)

---

## Common Pitfalls

### The Info Dump

Don't front-load world-building. Reveal information as it becomes relevant to player choices.

### The False Dilemma

"Fight or flee" choices often have obvious correct answers. Add genuine tradeoffs:

- Fight: save your friend but reveal your powers
- Flee: escape cleanly but abandon someone

### The Dead End

Never trap players in unwinnable states without warning. Either:

- Make all paths completable (different, not dead)
- Clearly signal point-of-no-return moments
- Allow backtracking to try different approaches

### The Maze

Physical navigation puzzles (go north, go east, go south...) are tedious in text. Use landmarks and purpose instead of compass directions:

- "Head toward the lighthouse" not "Go north"
- "Return to the market square" not "Go back three screens"
