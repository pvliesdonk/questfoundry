# Design Principles

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

These principles guide all design decisions in QuestFoundry v5. They are **absolute** — violations are bugs, not trade-offs.

---

## 1. Spoiler Hygiene (Absolute)

**Principle**: Player-facing surfaces never reveal internal implementation.

### What This Means

Players must never see:
- Internal passage IDs (`passage_047`, `chinatown_003`)
- Codeword names (`has_badge_favor`, `knows_jimmy_alive`)
- State variable names (`chinatown_trust >= 30`)
- System messages ("Codeword granted", "Gate unlocked")
- Debug information

### Correct Patterns

```yaml
# WRONG: Exposes implementation
choices:
  - text: "Talk to Chen (requires has_badge_favor)"
    target: chen_scene

# CORRECT: Diegetic feedback
choices:
  - text: "Flash your badge and ask for a favor"
    target: chen_scene
    condition: has_badge_favor
    unavailable_text: "You'd need something official to get his attention"
```

### Gate Feedback

When gates block access, feedback must be in-world:

| Wrong | Correct |
|-------|---------|
| "Requires: investigation >= 30" | "You don't know enough yet to make sense of this" |
| "Missing codeword: has_key" | "The door is locked. You need a key." |
| "Gate: tong_meeting not satisfied" | "The locals don't trust you enough to talk" |

---

## 2. Diegetic Gates (Absolute)

**Principle**: All barriers are expressed through the story world, not game mechanics.

### What This Means

Gates must feel like natural story obstacles, not arbitrary locks.

### Correct Patterns

```yaml
# WRONG: Mechanical gate
gates:
  - id: level_check
    description: "Must have level 5 to proceed"

# CORRECT: Diegetic gate
gates:
  - id: trust_earned
    description: "The community won't speak to outsiders"
    diegetic_barrier: "Social exclusion from tight-knit community"
    how_to_pass: "Build trust through respectful interactions"
```

### Gate Types

| Mechanical (Avoid) | Diegetic (Use) |
|-------------------|----------------|
| "Requires 30 investigation" | "You'd need to know more about the case" |
| "Need 'key' item" | "The door is locked solid" |
| "Stat check failed" | "Your hands shake too much to pick the lock" |

---

## 3. Choice Integrity (Absolute)

**Principle**: Choices must differ in intent, outcome, AND tone — not just flavor.

### What This Means

Every choice must offer meaningfully different:
- **Intent**: What the player is trying to accomplish
- **Outcome**: What happens as a result
- **Tone**: How the character approaches the situation

### Anti-Pattern: Cosmetic Choices

```yaml
# WRONG: Same intent, outcome, and meaning
choices:
  - text: "I'll investigate the warehouse"
    target: warehouse_scene
  - text: "Let's check out the warehouse"
    target: warehouse_scene
  - text: "The warehouse seems like a good lead"
    target: warehouse_scene
```

### Correct Pattern

```yaml
# CORRECT: Different intent and consequence
choices:
  - text: "Go to the warehouse alone"
    target: warehouse_solo
    grants: [went_alone]
    stat_effect: { stat: approach, direction: reckless, amount: 10 }

  - text: "Ask Chen to meet you there"
    target: warehouse_chen
    requires: has_chen_favor
    grants: [went_with_chen]

  - text: "Stake out the warehouse first"
    target: warehouse_stakeout
    stat_effect: { stat: approach, direction: careful, amount: 10 }
```

### The "Two Clear Options" Rule

> "Two clear options beat five near-synonyms."

If you can't make choices meaningfully different, reduce the number of choices.

---

## 4. Structure Before Prose

**Principle**: Complete topology before writing any scene prose.

### What This Means

The order is strict:
1. SPINE (emotional arc)
2. ANCHORS (structure)
3. FRACTURES (divergence points)
4. BRANCHES (paths)
5. CONNECTIONS (validate topology)
6. BRIEFS (scene specs)
7. **Then** FILL (prose)

### Why This Matters

Writing prose before structure leads to:
- Retrofitting structure to match prose (backflow)
- Inconsistent connections
- Orphaned content
- Scope creep

### The Brief Contract

Briefs specify intent; FILL honors the brief:

```yaml
# Brief says WHAT
brief:
  emotional_beat: tension
  location: morgue
  characters: [maria, attendant]
  stakes: "Maria must see the body but dreads confirmation"

# FILL produces HOW
# Prose implements the brief, doesn't change structure
```

---

## 5. Scope Budget Discipline

**Principle**: N passages means at most N unique destinations.

### What This Means

If the scope says "40 passages," you cannot have:
- 50 unique scenes
- Branches that explode combinatorially
- "Just one more path" syndrome

### Scope Enforcement

```yaml
# In dream.yaml
scope:
  estimated_passages: 40
  branching_depth: moderate  # Constrains branch factor

# In CONNECTIONS validation
scope_check:
  declared_passages: 40
  actual_passages: 47
  status: warn
  message: "7 passages over budget. Review for consolidation."
```

### Managing Scope

| Problem | Solution |
|---------|----------|
| Too many branches | Merge paths earlier |
| Exponential growth | Add bottlenecks |
| Scope creep | Use hooks for future content, not current |

---

## 6. Acknowledgment of Convergence

**Principle**: When paths merge, the next scene acknowledges the difference.

### What This Means

If two different paths lead to the same scene, that scene must recognize which path was taken.

### Anti-Pattern: Silent Convergence

```yaml
# WRONG: Ignores how player arrived
# Path A: Maria trusted Lily
# Path B: Maria rejected Lily
# Both lead to: confrontation_scene (identical text)
```

### Correct Patterns

**State-Aware Prose**:
```yaml
# Scene checks prior state
prose_variants:
  - condition: trusted_lily
    prose: "Lily's betrayal stung more because Maria had trusted her."

  - condition: rejected_lily
    prose: "Maria had been right not to trust her. Cold comfort now."
```

**Diegetic Bridge**:
```yaml
# Brief scene acknowledging the journey
passage: convergence_bridge
prose: |
  However she'd gotten here, Maria faced the same choice now.
  [References specific prior path through state-aware text]
```

---

## 7. Traceability

**Principle**: Every change records origin, rationale, and dependencies.

### What This Means

Artifacts include provenance:

```yaml
# In any artifact
_meta:
  created_by: grow.spine
  created_at: 2026-01-01T10:00:00Z
  derived_from:
    - seed.yaml
  rationale: "Opening beat establishes Maria's isolation"
```

### Why This Matters

- Debugging: trace unexpected outputs to inputs
- Regeneration: know what needs updating when sources change
- Review: understand why decisions were made

---

## 8. No Backflow

**Principle**: Later stages cannot modify earlier artifacts.

### What This Means

```
DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
         ─────────────────────────────────►
         NO MODIFICATIONS BACKWARD
```

If FILL reveals a problem in GROW:
1. Human reviews the problem
2. Human edits GROW manually
3. Pipeline regenerates FILL

**Never**: Automated upstream revision.

### Why This Matters

Backflow creates:
- Circular dependencies
- Unpredictable convergence
- Difficult debugging
- Loss of human control

---

## 9. Human Gates Are Sacred

**Principle**: Required gates cannot be bypassed programmatically.

### What This Means

When a gate is `required`:
- Pipeline halts
- Human must explicitly approve
- No timeout auto-advance
- No "skip for testing" in production

### Gate Authority

| Gate | Type | Rationale |
|------|------|-----------|
| SEED | Required | Structural commitment |
| ANCHORS | Required | Structure locked |
| SHIP | Required | Public release |
| Others | Configurable | Author preference |

---

## 10. Fail Forward

**Principle**: Failed checks branch the story, never block it.

### What This Means

Every skill check, stat check, or gate has a failure path that continues the story.

```yaml
# CORRECT: Failure branches
skill_check:
  stat: investigation
  threshold: 30
  success:
    target: find_clue
  failure:
    target: miss_clue
    grants: [missed_important_clue]  # Consequence, not dead end
```

### Anti-Pattern: Blocking Failure

```yaml
# WRONG: Failure loops back
skill_check:
  failure:
    target: try_again  # Frustrating loop
    message: "Try again"
```

---

## Principle Summary

| # | Principle | One-Line |
|---|-----------|----------|
| 1 | Spoiler Hygiene | Players never see internal IDs |
| 2 | Diegetic Gates | Barriers exist in the story world |
| 3 | Choice Integrity | Choices differ in intent, outcome, tone |
| 4 | Structure Before Prose | Topology complete before writing |
| 5 | Scope Budget | Stay within passage limits |
| 6 | Acknowledge Convergence | Merged paths reference their origins |
| 7 | Traceability | Changes record origin and rationale |
| 8 | No Backflow | Later stages don't modify earlier ones |
| 9 | Human Gates Sacred | Required approval cannot be bypassed |
| 10 | Fail Forward | Failed checks continue the story |

---

## See Also

- [00-vision.md](./00-vision.md) — Core philosophy
- [06-quality-bars.md](./06-quality-bars.md) — How principles are validated
- [08-research-foundation.md](./08-research-foundation.md) — Research basis
