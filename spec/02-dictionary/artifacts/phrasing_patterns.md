# Phrasing Patterns — Diegetic Templates (Layer 2)

> **Use:** Reusable diegetic phrasing templates for common gates, refusals, and state checks.
> Maintained by Style Lead for consistency; used by Scene Smith and Player-Narrator.
>
> **Producer:** Style Lead
> **Consumer:** Scene Smith (drafting), Player-Narrator (performance), Translator (localization)

---

## Normative references

- Principles: `../../00-north-star/PN_PRINCIPLES.md`
- Bars: `../../00-north-star/QUALITY_BARS.md` (§Style, §Presentation)
- Role charters: `../../01-roles/charters/style_lead.md` · `../../01-roles/charters/player_narrator.md`

---

## Purpose

Phrasing patterns provide **diegetic, player-safe templates** for recurring situations:
- Gateway checks ("You need X to proceed")
- Refusals ("That path is blocked")
- State reflections ("You're carrying Y")
- Conditional affordances ("Because you have Z, you can...")

All patterns must be **in-world** (no mechanics, no internal labels).

---

## Structure

### Header

```
Phrasing Patterns — <project/act name>
Voice: <register description>
Author: Style Lead
TU: <tu-id>
Edited: <YYYY-MM-DD>
```

---

## Pattern Categories

### 1) Gateway Checks (Requirements)

**Pattern name:** <descriptive label>
**Condition:** <what the world checks>
**Phrasing:**
- Qualified: "<in-world phrasing when player qualifies>"
- Unqualified: "<in-world phrasing when player doesn't qualify>"
- Hint (optional): "<how player learns about requirement>"

---

### 2) State Reflections (Inventory/Knowledge)

**Pattern name:** <descriptive label>
**State:** <what player has/knows>
**Phrasing:** "<in-world description of state>"

---

### 3) Conditional Affordances

**Pattern name:** <descriptive label>
**Condition:** <state that enables action>
**Phrasing:** "<how the new option appears>"

---

### 4) Refusals (Blocked Actions)

**Pattern name:** <descriptive label>
**Blocker:** <why action fails>
**Phrasing:** "<in-world explanation without mechanics>"

---

## Example

```markdown
Phrasing Patterns — Adventure Bay Mystery
Voice: Warm, encouraging, child-friendly (ages 6-10)
Author: Style Lead
TU: TU-2025-11-24-ST01
Edited: 2025-11-24

---

## Gateway Checks

### Pattern: Maritime Guild Access
**Condition:** Player has guild token visible
**Phrasing:**
- **Qualified:** "The guard spots the guild emblem on your vest and nods you through."
- **Unqualified:** "The guard politely stops you: 'Guild members only beyond this point.'"
- **Hint:** "You notice other pups wearing small bronze pins—guild emblems."

### Pattern: Weather Safety
**Condition:** Weather is calm OR player has climbing gear
**Phrasing:**
- **Qualified (calm):** "The waves have settled. The rocks look climbable now."
- **Qualified (gear):** "With your rope secured, the climb looks manageable."
- **Unqualified:** "The waves crash hard against the rocks. Too dangerous right now."
- **Hint:** "Skye calls down: 'Maybe wait for the tide to change?'"

---

## State Reflections

### Pattern: Carrying Key
**State:** Player obtained lighthouse key
**Phrasing:** "The brass key feels heavy in your paw. The lighthouse symbol etched on its bow catches the light."

### Pattern: Overheard Code
**State:** Player learned maintenance crew code
**Phrasing:** "You remember the three-tap pattern the workers used on the metal door."

---

## Conditional Affordances

### Pattern: Maintenance Access (After Overhearing)
**Condition:** Player overheard crew code earlier
**Phrasing:** "You notice a small metal hatch. The same pattern—three taps—might work here."

### Pattern: Foreman Trust (After Helping)
**Condition:** Player completed foreman's task
**Phrasing:** "The foreman greets you warmly: 'Back for more help? Or maybe you need something?'"

---

## Refusals

### Pattern: Locked Door (No Key)
**Blocker:** Door requires specific key
**Phrasing:** "The door doesn't budge. The lock is solid, and forcing it would probably just hurt your paw."

### Pattern: Time-Gated Event
**Blocker:** Event hasn't occurred yet (time/sequence)
**Phrasing:** "The plaza is quiet. Maybe check back later when the market opens?"

---

## Notes for Scene Smith

- Use these patterns as **starting points**; adapt to context
- Add sensory detail (sound, smell, texture) to ground the refusal
- Avoid negative framing ("you can't"); prefer world-state framing ("the door is locked")

## Notes for Player-Narrator

- Deliver refusals with warmth, not punishment
- Emphasize player agency ("you might try...") over restriction
- Use patterns as **consistency anchors** across performance

## Notes for Translator

- Preserve diegetic tone; avoid literal translation of mechanics
- Adapt idioms to target culture while keeping in-world feel
- Consult register_map for voice consistency

```

---

## Hot vs Cold

**Hot only** — Phrasing patterns are working documents:
- Referenced during section drafting
- Consulted by PN during performance
- Updated as voice evolves
- Not exported to players

---

## Lifecycle

1. **Style Lead** creates initial patterns during early loops
2. **Scene Smith** uses patterns when drafting section prose
3. **PN** uses patterns for consistent performance
4. **Translator** adapts patterns for localization
5. **Style Lead** updates patterns as voice evolves

---

## Validation checklist

- [ ] All patterns use diegetic language (no mechanics/labels)
- [ ] Qualified/unqualified phrasings feel distinct but fair
- [ ] Hints are discoverable without spoiling
- [ ] Voice is consistent across all patterns
- [ ] Refusals are kind, not punishing
- [ ] Patterns adaptable to multiple contexts

---

**Created:** 2025-11-24
**Status:** Initial template
