# Codex Expansion Loop

> **Goal:** Publish player-safe knowledge from canon without leaking spoilers.

The **Codex Expansion** loop handles the publication of world knowledge for players. It transforms spoiler-heavy canon into accessible codex entries with clear cross-references while strictly maintaining spoiler hygiene.

:::{loop-meta}
id: codex_expansion
name: "Codex Expansion"
trigger: canon_available
entry_point: showrunner
version: 1
:::

## Guidance

This section provides operational context for executing the Codex Expansion workflow.

### When to Trigger

Invoke the Codex Expansion loop when:

- **Post-canonization**: Lore Deepening produces new canon needing player surfaces
- **Term frequency**: Story introduces terms repeatedly that players might not understand
- **Comprehension feedback**: Players or playtesters report confusion
- **Taxonomy hooks**: Hook Harvest identified coverage gaps or red-links

Do NOT invoke when:

- Canon isn't finalized yet (complete Lore Deepening first)
- Content is spoiler-sensitive with no safe summary (defer until reveal)
- Explanation belongs in prose (Scene Smith handles in-story exposition)
- Internal reference only (not everything needs a codex entry)

### Success Criteria

The loop succeeds when:

- [ ] High-frequency terms have matching codex entries
- [ ] No spoilers appear in any player-facing content
- [ ] All "See also" links resolve (no dead ends)
- [ ] Reading level and tone align with style guide
- [ ] Traceability present (TU lineage documented)
- [ ] Crosslink map enables easy navigation

### Common Failure Modes

**Accidental spoilers**

- Symptom: Twist or reveal details appear in codex entry
- Fix: Move detail back to canon notes; rewrite with neutral phrasing
- Prevention: Write player-safe version first, compare against full canon

**Over-technical voice**

- Symptom: Entry reads like internal documentation, not player help
- Fix: Simplify language; add examples; match story tone
- Prevention: Write as if explaining to a curious player

**Link rot**

- Symptom: "See also" links point to non-existent entries
- Fix: Add missing entries as stubs or reduce link fan-out
- Prevention: Verify all links before submission

**Taxonomy creep into canon**

- Symptom: Codex entries start inventing backstory
- Fix: Escalate to Lorekeeper; codex does not invent lore
- Prevention: Codex summarizes canon, never extends it

**Coverage sprawl**

- Symptom: Too many entries planned, none completed
- Fix: Prioritize by player-value; batch releases
- Prevention: Set entry budget per cycle; focus on comprehension bottlenecks

## Loop Participants

The roles that participate in this loop with their operational parameters.

:::{loop-participants}
showrunner:
  timeout: 300
  max_iterations: 5
lorekeeper:
  timeout: 300
  max_iterations: 5
creative_director:
  timeout: 300
  max_iterations: 5
gatekeeper:
  timeout: 300
  max_iterations: 3
:::

## Routing Rules

Decision table for Showrunner: after a role completes work, these rules
describe when to delegate to the next role.

### After Showrunner

:::{routing-rule}
after: showrunner
when: brief_created
delegate_to: lorekeeper
description: SR scopes codex work; LK drafts player-safe entries
:::

:::{routing-rule}
after: showrunner
when: terminate
delegate_to: END
description: Codex expansion complete; end the loop
:::

### After Lorekeeper

:::{routing-rule}
after: lorekeeper
when: entries_drafted
delegate_to: creative_director
description: Entries drafted; CD reviews style and voice
:::

:::{routing-rule}
after: lorekeeper
when: escalation
delegate_to: showrunner
description: LK encounters spoiler issue requiring SR decision
:::

### After Creative Director

:::{routing-rule}
after: creative_director
when: style_approved
delegate_to: gatekeeper
description: Style pass complete; GK validates safety and integrity
:::

:::{routing-rule}
after: creative_director
when: needs_revision
delegate_to: lorekeeper
description: CD requests content changes; LK revises entries
:::

:::{routing-rule}
after: creative_director
when: escalation
delegate_to: showrunner
description: CD encounters voice conflict requiring SR decision
:::

### After Gatekeeper

:::{routing-rule}
after: gatekeeper
when: failed
delegate_to: lorekeeper
description: Validation failed; LK revises entries
:::

:::{routing-rule}
after: gatekeeper
when: passed
delegate_to: showrunner
description: All bars passed; SR approves for merge
:::

## Quality Gates

### Pre-Publication Validation

:::{quality-gate}
before: gatekeeper
role: gatekeeper
bars:

- presentation
- integrity
- style
blocking: true
:::

## Expected Flow

```text
Canon Available
    |
[Showrunner] -> scopes codex work
    |
[Lorekeeper] -> drafts player-safe entries
    |
[Creative Director] -> style pass
    |
[Gatekeeper] -> validates safety and integrity
    | (if passed)
[Showrunner] -> approves for merge
```

## Artifacts Produced

- **Codex Pack**: Collection of player-safe entries
- **Crosslink Map**: Navigation structure for entries
- **Coverage Report**: What's covered and what red-links remain
- **Spoiler Hygiene Note**: Masked details and deferred entries

## Entry Anatomy

Each codex entry should include:

- **Title**: Term or name as players see it
- **Overview**: 2-4 sentences, neutral, spoiler-safe
- **Usage**: How/why the player might encounter this term
- **Context**: High-level setting notes without twist details
- **See also**: 3-5 related entries for navigation
- **Notes**: Accessibility or localization hints
- **Lineage**: TU reference for traceability

### Never Include

- Hidden gate conditions
- Internal IDs or codewords
- Twist explanations
- Implementation details

## Handoffs

After Codex Expansion:

- **Canon Commit**: Approved codex entries for cold_store
- **Publisher**: Codex ready for export pipelines
- **Narrator**: Safe reference material for play sessions
