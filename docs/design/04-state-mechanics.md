# State and Mechanics Model

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

The state/mechanics model defines how player choices create persistent consequences. v5 expands beyond simple boolean codewords to support richer interactive fiction mechanics.

**Key Principle**: All structural state is defined in ANCHORS. No new state types can be introduced after ANCHORS approval.

---

## State Types

### 1. Codewords (Boolean Flags)

The simplest state type: true or false.

```yaml
codewords:
  - id: found_body
    description: "Player has seen Tommy's body at the morgue"
  - id: has_badge_favor
    description: "Captain Chen gave Maria temporary badge access"
  - id: knows_jimmy_alive
    description: "Player has learned Jimmy is still alive"
```

**Use Cases**:
- Gate conditions ("Can only enter morgue if `has_badge_favor`")
- Narrative flags ("Reference the body discovery if `found_body`")
- Ending requirements ("Requires `found_all_evidence`")

**Operations**:
- `grant(codeword)` — Set to true
- `revoke(codeword)` — Set to false (rare)
- `check(codeword)` — Test if true

### 2. Opposed Stats (Bipolar Spectrums)

Personality or approach traits that slide between two poles.

```yaml
opposed_stats:
  - id: approach
    poles: [careful, reckless]
    initial: 50
    range: [0, 100]
    description: "Player's investigation style"

  - id: disposition
    poles: [stoic, emotional]
    initial: 50
    range: [0, 100]
    description: "Maria's emotional expression"
```

**Interpretation**:
- 0 = fully first pole (careful / stoic)
- 100 = fully second pole (reckless / emotional)
- 50 = balanced / neutral

**Use Cases**:
- Gate conditions ("Only `approach >= 70` can attempt dangerous shortcut")
- Choice availability ("Stoic options only if `disposition < 30`")
- Character consistency ("Prose reflects current position")

**Operations**:
- `shift(stat, direction, amount)` — Move toward a pole
- `set(stat, value)` — Force to specific value (rare)
- `check_range(stat, min, max)` — Test if within range

### 3. Accumulative Stats (Skills/Resources)

Stats that increase (rarely decrease) through play.

```yaml
accumulative_stats:
  - id: investigation
    initial: 0
    max: 100
    description: "Clue-finding and deduction skill"

  - id: street_cred
    initial: 10
    max: 50
    description: "Reputation in the criminal underworld"
```

**Use Cases**:
- Skill checks ("Requires `investigation >= 30`")
- Unlock thresholds ("New dialogue at `street_cred >= 25`")
- Progress tracking

**Operations**:
- `grant_points(stat, amount)` — Increase
- `spend_points(stat, amount)` — Decrease (optional mechanic)
- `check_threshold(stat, minimum)` — Test if meets threshold

### 4. Hidden Variables

State tracked silently, revealed only through consequences.

```yaml
hidden_variables:
  - id: chinatown_trust
    initial: 20
    range: [0, 100]
    description: "Community trust in Maria"
    display: false

  - id: lily_suspicion
    initial: 0
    range: [0, 100]
    description: "How suspicious Lily is of Maria"
    display: false
```

**Use Cases**:
- Relationship tracking (accumulates invisibly)
- Consequences that surprise but feel logical
- NPC behavior modifications

**Key Rule**: Hidden doesn't mean random. Players should feel "I should have seen that coming" when hidden variables trigger consequences.

### 5. Key Items (Narrative Inventory)

Important objects that affect story access or outcomes.

```yaml
key_items:
  - id: fathers_badge
    description: "Maria's father's old police badge"
    narrative_critical: true   # Cannot be lost or sold

  - id: case_file_copy
    description: "Copy of the original case file"
    narrative_critical: false  # Can be lost in certain paths

  - id: tommy_photo
    description: "Old photograph of Tommy and Jimmy together"
    narrative_critical: false
```

**Use Cases**:
- Gate conditions ("Requires `fathers_badge` to enter precinct")
- Dialogue options ("Can show `tommy_photo` to trigger memory")
- Ending requirements

**Operations**:
- `grant_item(item)` — Add to inventory
- `remove_item(item)` — Remove from inventory (if not narrative_critical)
- `has_item(item)` — Check possession

---

## Structural vs Emergent State

### Structural State (ANCHORS only)

State that affects story structure must be defined in ANCHORS:

- New codewords
- New stat types
- New items
- Gate conditions

**After ANCHORS approval, no new state types can be added.**

### Emergent State (BRANCHES can modify)

Branches can modify existing state but not create new types:

```yaml
# In grow/branches/chinatown_path.yaml

# ALLOWED: Grant defined codeword
passages:
  - id: chinatown_003
    grants:
      - spoke_to_mother  # Defined in ANCHORS

# ALLOWED: Modify defined stat
stat_effects:
  - stat: chinatown_trust  # Defined in ANCHORS
    amount: +10

# NOT ALLOWED: Create new state
codewords:
  - id: new_codeword_here  # ERROR: Can't add after ANCHORS
```

### Local Codewords (Branch-Scoped)

Branches can define **scoped codewords** that only matter within that branch subtree:

```yaml
# In grow/branches/chinatown_path.yaml

local_state:
  codewords:
    - id: found_old_photo
      scope: chinatown_path  # Only valid in this branch
```

**Rules for Local Codewords**:
- Cannot be used by gates (gates use structural codewords)
- Cannot affect endings (endings use structural codewords)
- Only affect prose and choice availability within the branch

---

## State Checks

### Threshold Checks (Deterministic)

```yaml
gates:
  - id: convince_guard
    condition:
      type: threshold
      stat: investigation
      minimum: 30
    unlocks: police_records_scene
```

**Behavior**: If `investigation >= 30`, gate opens. Deterministic.

### Codeword Checks (Boolean)

```yaml
gates:
  - id: use_badge
    condition:
      type: codeword
      requires: has_badge_favor
    unlocks: morgue_access
```

**Behavior**: If `has_badge_favor` is true, gate opens.

### Stat Range Checks

```yaml
gates:
  - id: reckless_option
    condition:
      type: stat_range
      stat: approach
      range: [70, 100]  # Must be reckless
    unlocks: dangerous_shortcut
```

**Behavior**: Only available if stat within range.

### Item Checks

```yaml
gates:
  - id: show_badge
    condition:
      type: item
      requires: fathers_badge
    unlocks: precinct_entrance
```

### Compound Checks

```yaml
gates:
  - id: complex_gate
    condition:
      type: compound
      operator: AND
      conditions:
        - type: codeword
          requires: knows_jimmy_alive
        - type: threshold
          stat: investigation
          minimum: 40
```

**Operators**: `AND`, `OR`, `NOT`

---

## Fail Forward

**Core Rule**: Failed checks must never stop the story. They branch, not block.

```yaml
# In grow/briefs/guard_post.yaml

skill_check:
  stat: investigation
  threshold: 30
  success:
    outcome: "Guard lets you in without suspicion"
    target: records_room
  failure:
    outcome: "Guard is suspicious but lets you pass"
    target: records_room
    grants:
      - guard_suspicious  # Consequence, not dead end
```

**Anti-Pattern**: Failed check → game over / loop back

**Correct Pattern**: Failed check → same destination with different state

### Failure Consequences

Failures should create meaningful (but not fatal) consequences:

| Consequence Type | Example |
|-----------------|---------|
| State penalty | `grants: [guard_suspicious]` |
| Stat reduction | `stat_effect: { stat: street_cred, amount: -5 }` |
| Dialogue change | "Success: helpful NPC / Failure: reluctant NPC" |
| Future gate | "Failure now closes option later" |

---

## Display vs Hidden

Configure whether state is shown to players:

```yaml
state:
  opposed_stats:
    - id: approach
      display: true   # Shown in UI as "Careful ◄──●───► Reckless"

  hidden_variables:
    - id: lily_suspicion
      display: false  # Never shown, affects story silently

  accumulative_stats:
    - id: investigation
      display: true   # Shown as "Investigation: 35/100"
```

### Display Formats

For displayed state, define how it appears:

```yaml
display_config:
  opposed_stats:
    format: "spectrum"  # Visual slider
    labels: true        # Show pole names

  accumulative_stats:
    format: "bar"       # Progress bar
    show_max: true      # "35/100" vs just "35"

  codewords:
    format: "hidden"    # Never show codewords (they're implementation detail)
```

---

## State in Prose

FILL stage must reflect state in prose when relevant:

```yaml
# In grow/briefs/confrontation.yaml

state_reflection:
  - condition:
      stat: approach
      range: [70, 100]
    prose_note: "Maria acts impulsively, doesn't wait for backup"

  - condition:
      codeword: trusted_lily
    prose_note: "Maria hesitates, remembering Lily's help"
```

The brief tells FILL what state to consider. Prose adapts accordingly.

---

## Validation Rules

| Rule | Scope | Description |
|------|-------|-------------|
| State defined | ANCHORS | All state types have definitions |
| No new types | BRANCHES | Branches only use/modify existing state |
| Gates satisfiable | CONNECTIONS | Every gate has a reachable path to satisfy |
| No dead ends | CONNECTIONS | Failed checks have forward paths |
| Consistent range | All | Stat modifications stay within defined range |
| Items exist | FILL | Item references point to defined items |

---

## Example: Full State Definition

```yaml
# grow/anchors.yaml (excerpt)

state:
  # Boolean flags
  codewords:
    - id: found_body
      description: "Saw Tommy's body"
    - id: has_badge_favor
      description: "Chen gave badge access"
    - id: knows_jimmy_alive
      description: "Discovered Jimmy is alive"
    - id: trusted_lily
      description: "Accepted Lily's help"
    - id: found_all_evidence
      description: "Collected all key evidence"
    - id: confronted_jimmy
      description: "Had final conversation with Jimmy"

  # Spectrums
  opposed_stats:
    - id: approach
      poles: [careful, reckless]
      initial: 50
      range: [0, 100]
      display: true

  # Skills
  accumulative_stats:
    - id: investigation
      initial: 10
      max: 100
      display: true

  # Hidden tracking
  hidden_variables:
    - id: chinatown_trust
      initial: 20
      range: [0, 100]
      display: false
    - id: chen_loyalty
      initial: 50
      range: [0, 100]
      display: false

  # Important objects
  key_items:
    - id: fathers_badge
      narrative_critical: true
    - id: case_file_copy
      narrative_critical: false
    - id: tommy_photo
      narrative_critical: false
```

---

## Design Questions for Implementation

1. **Complexity Budget**: How much state can small models track reliably?
   - Recommendation: Max 10 codewords, 3 stats, 5 items per story

2. **State Display**: How to show stats in text-only IF?
   - Recommendation: Optional summary command, not inline

3. **Probability Checks**: Support dice rolls?
   - Recommendation: Threshold-only for v5.0; probability in v5.1+

4. **Scoped Codewords**: How to enforce scope?
   - Recommendation: Naming convention (`local_chinatown_*`) + validation

---

## See Also

- [02-artifact-schemas.md](./02-artifact-schemas.md) — State in ANCHORS schema
- [03-grow-stage-specification.md](./03-grow-stage-specification.md) — When state is defined
- [06-quality-bars.md](./06-quality-bars.md) — State consistency validation
