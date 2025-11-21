---
snippet_id: contrastive_choices
name: Contrastive Choices
description: Ensure verbs & objects differentiate intent; not just synonyms (e.g., different verb, different object, different manner)
applies_to_roles: [style_lead, scene_smith, plotwright, gatekeeper]
quality_bars: [style, nonlinearity, accessibility]
---

# Contrastive Choices

## Core Principle

Choice labels must be meaningfully different, allowing players to make informed decisions without guessing. Near-synonyms are forbidden.

## Forbidden Patterns

### Near-Synonyms

❌ "Go / Proceed"
❌ "Enter / Go in"
❌ "Look around / Investigate"
❌ "Ask / Inquire"

These fail because they don't signal different outcomes or approaches.

### Meta Hedging

❌ "Attempt to X"
❌ "Try to X"
❌ "Maybe X"

These signal uncertainty about mechanics, not meaningful choice.

## Valid Contrast Patterns

### Different Verbs (Different Actions)

✓ "Slip through maintenance / Face the foreman"

- Slip = avoid confrontation
- Face = direct approach

✓ "Knock / Pick the lock"

- Knock = request entry (social)
- Pick = force entry (technical)

### Different Objects

✓ "Take hex-key / Take union token"

- Hex-key = tool (technical path)
- Token = identity (social path)

✓ "Read the manual / Ask the engineer"

- Manual = self-directed learning
- Engineer = social assistance

### Different Manner (Same Action)

✓ "Move quickly / Move carefully"

- Quickly = risky but fast
- Carefully = slow but safe

✓ "Lie convincingly / Tell partial truth"

- Convincingly = full deception (risky)
- Partial truth = safer middle ground

### Different Recipients

✓ "Tell the foreman / Tell the union rep"

- Foreman = management side
- Union rep = worker side

### Different Scope

✓ "Investigate airlock / Investigate entire bay"

- Airlock = focused, quick
- Entire bay = thorough, time-consuming

## Testing Contrast

Ask: "Can player distinguish these without knowledge of outcomes?"

**Good Contrast:**

- "Sneak past guard / Bribe guard"
  → Player knows: sneak = avoid, bribe = interact with money

**Poor Contrast:**

- "Avoid guard / Evade guard"
  → Player confused: what's the difference?

## Context Clarification

When labels alone insufficient, Scene Smith adds 1-2 lines of micro-context:

```markdown
The guard patrols predictably, but the foreman carries petty cash.
- Sneak past guard
- Bribe foreman for distraction
```

Now "sneak" vs "bribe" has context without spoiling outcomes.

## Role-Specific Applications

**Scene Smith:**

- Draft contrastive choice labels
- Add micro-context when needed
- Avoid near-synonyms in prose

**Plotwright:**

- Design choice intents with clear differentiation
- Specify contrast type in section briefs
- Flag ambiguous choice pairs

**Style Lead:**

- Enforce contrastive choice policy
- Provide phrasing alternatives
- Flag near-synonyms in review

**Gatekeeper:**

- Pre-gate check for choice clarity
- Block on near-synonyms
- Suggest contrastive alternatives

## Common Fixes

### Near-Synonym → Different Verb

Before: "Go / Proceed"
After: "Slip through quietly / March confidently"

### Vague → Specific Object

Before: "Take tool / Take item"
After: "Take hex-key / Take union token"

### Generic → Manner Differentiation

Before: "Talk to guard"
After: "Intimidate guard / Charm guard"

### Single Choice → Scoped Options

Before: "Search"
After: "Quick search / Thorough search"

## Validation Checklist

For each choice pair:

- [ ] Verbs different (or manner/object/recipient different)?
- [ ] Player can infer different approaches?
- [ ] No "attempt to" or "try to" hedging?
- [ ] Context provided if labels alone insufficient?
- [ ] No near-synonyms (go/proceed, ask/inquire)?

## Accessibility Connection

Contrastive choices improve accessibility:

- Screen reader users hear labels out of prose context
- Choice distinction must be clear from labels alone
- Synonyms force guessing, reducing agency
- Meaningful contrast enables informed decisions
