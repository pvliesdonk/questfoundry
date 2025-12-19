# Branching Narrative Craft for Interactive Fiction

Craft guidance for designing meaningful choices—branch architecture, consequence systems, and maintaining narrative coherence across paths.

---

## Choice Architecture

### Types of Choices

**Cosmetic Choices:**

- Change flavor text without affecting story
- Personalization without consequence
- Low cost to implement, low player investment

**Tactical Choices:**

- Affect immediate outcome
- Combat options, puzzle solutions
- Consequences visible within the scene

**Strategic Choices:**

- Affect long-term story direction
- Character builds, faction alignments
- Consequences may be delayed

**Moral Choices:**

- Define character's values
- No "right" answer—tradeoffs and costs
- Player investment highest here

**Identity Choices:**

- Define who the character is
- Backstory, personality, relationships
- Often locked in early

### Meaningful vs Illusory Choice

**Meaningful:**

- Player feels consequences
- Different paths lead to different experiences
- Choice reflects character/player values
- Information to make informed decision

**Illusory:**

- All paths converge immediately
- Cosmetic difference only
- Player detects and resents deception
- Worse than no choice at all

**The Test:** If you removed this choice, would anything change?

### Choice Presentation

**Diegetic (World-Grounded):**

> You could take the mountain pass—faster but exposed. Or follow the river—longer but concealed.

**Mechanical (Game-Speak):**

> [FAST: +2 time, +risk] [SAFE: -2 time, +stealth]

**Best Practice:** Diegetic text with mechanical implications clear through context.

---

## Branch Structures

### Pure Branching (Tree)

Every choice creates permanent divergence. Paths never merge.

**Pros:**

- Maximum consequence
- Truly unique paths
- High replay value

**Cons:**

- Content multiplies exponentially
- Most content seen by few players
- Unsustainable at scale

**Best for:** Short IF, experimental pieces.

### Bottleneck (Funnel)

Paths diverge then reconverge at key story beats.

**Pros:**

- Sustainable at any length
- Major plot points guaranteed
- Variations in journey, convergent destinations

**Cons:**

- Choices can feel less impactful
- Risk of "your choices don't matter" feeling

**Best for:** Most commercial IF.

### Parallel Tracks

Early major choice sets player on one of several distinct tracks.

**Pros:**

- Different story experiences
- Moderate content multiplication
- Major choices feel significant

**Cons:**

- Players miss large portions of content
- Testing burden multiplies
- Must design multiple complete arcs

**Best for:** Medium-to-long IF with strong replay value.

### State-Based

Single narrative path with variations based on accumulated flags/stats.

**Pros:**

- Most efficient use of words
- Personalization without multiplication
- Scales excellently

**Cons:**

- Variations subtle rather than structural
- Less dramatic divergence
- Requires robust state tracking

**Best for:** Long IF, stat-heavy games.

### Hybrid Approaches

Most successful IF combines structures:

- State-based for minor variations
- Bottleneck for sustainable branching
- Occasional pure branching for major moments
- Parallel tracks for act structure

---

## Designing Meaningful Choices

### The Dilemma Principle

Good choices are hard. Both options should have costs and benefits.

**Weak Choice:**

> Save the village or don't bother. (Obvious "right" answer)

**Strong Choice:**

> Save the village but miss the kidnapper's trail, or pursue the kidnapper while the village burns.

### Information and Uncertainty

**Clear Choices:**

- Player knows consequences
- Decisions strategic
- Feels fair

**Blind Choices:**

- Consequences hidden
- Can feel arbitrary
- Sometimes necessary for surprise

**Forecasted Choices:**

- Hints at consequences without certainty
- "This might anger the king..."
- Balance of information and uncertainty

### Delayed Consequences

Not all consequences immediate. Powerful pattern:

1. Make choice in Chapter 2
2. No visible consequence
3. In Chapter 5, consequence emerges
4. Player connects cause and effect

**Requirements:**

- Make the original choice memorable
- Signpost the connection when consequence arrives
- Track state reliably

### Moral Complexity

The best moral choices:

- No clear "good" option
- Costs on all sides
- Reflect real ethical dilemmas
- Let player define their character

**Avoid:**

- Obvious right answers
- Choices where "good" option has no cost
- Punishing players for being good
- Rewarding players for being evil (unless that's your theme)

---

## Managing State

### What to Track

**Essential:**

- Major story decisions
- Character deaths/survivals
- Relationship states
- Resource levels

**Useful:**

- Accumulated personality traits
- Knowledge/discovery flags
- Minor relationship shifts
- Completionist tracking

**Excessive:**

- Every dialogue choice
- Every item examined
- State that never affects anything

### State Representation

**Flags (Boolean):**

```
met_the_queen: true/false
knows_secret: true/false
```

**Counters (Numeric):**

```
trust_with_ally: 0-100
gold: integer
corruption: 0-10
```

**Enums (Category):**

```
faction: rebel/loyalist/neutral
class: warrior/mage/rogue
```

### Checking State

**Conditional Text:**

Different descriptions based on state.

**Conditional Choices:**

Options appear/disappear based on state.

**Conditional Outcomes:**

Same choice, different result based on state.

### State Complexity Management

As state grows, combinations explode. Strategies:

- Test state systematically
- Document all state variables
- Limit variables that interact
- Use state categories (relationship, knowledge, resources)
- Prune unused state

---

## Maintaining Narrative Coherence

### The Continuity Challenge

With branching, maintaining consistent narrative becomes hard. Character who died in one path referenced in another. Events that didn't happen recalled.

### Strategies for Coherence

**Flexible References:**

> "Remember when we faced those bandits?"

Works whether bandits were fought or avoided.

**Conditional Text:**

> "Remember when we faced those bandits?" / "This is quieter than our last journey."

Different text based on path.

**Abstract References:**

> "After everything we've been through..."

Fits any path without specifics.

### Character Consistency Across Paths

Player character should feel consistent:

- Core personality stable
- Choices reflect established character
- Growth believable regardless of path
- Voice consistent even as content varies

### NPC Consistency

NPCs must remain coherent:

- Track relationship state
- Adjust dialogue to history
- Remember player actions
- Maintain personality across encounters

### World Consistency

World state must reflect choices:

- Destroyed buildings stay destroyed
- Dead characters stay dead
- Political changes persist
- Time passes consistently

---

## Convergence Techniques

### The Bottleneck

All paths must pass through this point. Events here affect everyone.

**Techniques:**

- Mandatory story beats
- Plot-critical revelations
- Time jumps that reset minor differences
- Settings where all paths logically converge

### Variable Bottlenecks

Same general event, different specifics based on path:

> Everyone reaches the castle. WHO reaches it and HOW varies.

### Merging Threads

When paths merge, acknowledge differences:

- Reference different routes
- Different resources based on path
- Relationships affected by journey
- Knowledge varies

### The "Felt Difference" Principle

Even when paths converge, player should feel their journey mattered:

- Reference their specific experiences
- State carries forward
- NPCs remember
- Small details differ

---

## Testing Branching Narratives

### Completeness Testing

Every path must be tested:

- All choices lead somewhere
- No dead ends
- No untracked state
- Every combination works

### Continuity Testing

Check for contradictions:

- Dead characters referenced alive
- Unlearned information known
- Wrong relationship states
- Timeline inconsistencies

### Balance Testing

Check path quality:

- All paths satisfying
- No path clearly "worse"
- Consequences proportional to choices
- No path too short or too long

### Systematic Approaches

**State Matrix:**

- List all major state variables
- Test key combinations
- Document expected vs actual outcomes

**Path Maps:**

- Visualize all paths
- Identify convergence points
- Check no orphaned content

---

## Common Mistakes

### The False Choice

Choice that doesn't actually matter. Players recognize and resent.

**Fix:** If choice doesn't matter, don't present it as choice.

### All Roads Lead to Rome

Every choice converges immediately. Removes agency.

**Fix:** Let differences persist. Converge later, not immediately.

### Choice Overload

Too many options. Decision paralysis.

**Fix:** 2-4 choices typical. More only when justified.

### Invisible Consequences

Player doesn't see how choices affected outcome.

**Fix:** Signpost connections. Acknowledge player history.

### Tracking Without Using

Tracking state that never matters. Wasted effort.

**Fix:** Every tracked variable should affect something.

### Inconsistent Consequence Weight

Similar choices with wildly different consequences.

**Fix:** Calibrate consequence to choice significance.

---

## Quick Reference

| Element | Guideline |
|---------|-----------|
| Choice types | Cosmetic < Tactical < Strategic < Moral |
| Meaningful | Consequence visible, reflects values |
| Structure | Bottleneck most sustainable |
| Dilemmas | Both options should have costs |
| State | Track what matters, use what you track |
| Coherence | Acknowledge path differences |
| Convergence | Paths merge, felt differences persist |
| Testing | Every path, every combination |
| Balance | All paths should satisfy |
