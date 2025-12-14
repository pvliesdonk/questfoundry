# Spoiler Hygiene

> Keep player surfaces clean. Machinery stays hidden; wonder stays on the page.

This principle defines how QuestFoundry protects players from spoilers while enabling rich internal creative work.

## Core Concept: Hot vs Cold

| Store | Purpose | Spoiler Safety |
|-------|---------|----------------|
| **hot_store** | Working drafts, internal plans, canon notes | May contain spoilers |
| **cold_store** | Committed canon, player-facing content | Must be spoiler-safe |

**Rule:** Surfaces (what players see) come only from cold_store. Hot content never leaks to players.

## Golden Rules

1. **No reveals on surfaces.** If knowing it would change how a first-time reader perceives a choice or scene, it stays off the surface.

2. **No internals.** Never expose: gate logic, unlock conditions, internal IDs, schema fields, debug info.

3. **Diegetic gates only.** Narrator presents access restrictions as in-world conditions (tokens, reputation, knowledge), not mechanics.

4. **Canon is not Codex.** Canon (spoiler-level truth) lives in hot_store. Player-facing summaries go to cold_store.

## Leak Taxonomy

| Leak Type | Bad Example | Good Example |
|-----------|-------------|--------------|
| **Twist telegraph** | "The foreman is secretly Syndicate." | "The foreman eyes your badge; his smile doesn't reach his eyes." |
| **Gate logic** | "Option locked: missing KEY_ASH." | "No union token on your lapel; the foreman waves you back." |
| **Internal mechanics** | "Roll DC 15 to proceed." | "You'd need to prove your credentials somehow." |
| **Canon dump** | "The fire was sabotage by the Syndicate." | "A refinery accident years ago reshaped safety drills." |

## Role Responsibilities

### Gatekeeper

- Block content with **presentation** failures
- Run spoiler checks on all cold_store candidates
- Validate that gate phrasing is diegetic

### Lorekeeper

- Quarantine spoilers in hot_store canon notes
- Provide player-safe summaries for cold_store
- Never leak hot details when verifying consistency

### Narrator

- Enforce gates diegetically (no codewords, no checks by name)
- Present choices without hinting at hidden outcomes
- Reference only cold_store content

### Scene Smith

- Write choices that are distinct and fair without revealing hidden logic
- Use diegetic phrasing for all state-dependent content

### Plotwright

- Define gates in terms of player-meaningful conditions
- Avoid meta-gates ("Missing Key" text)

## Redlines (Never on Surfaces)

- Gate logic or unlock conditions
- Internal identifiers (IDs, schema fields, file paths)
- Hidden plot points or twists
- Debug marks (TODO, FIXME, dev notes)
- Out-of-world directives (click, flag, roll, invoke)
- Determinism details (seeds, models, parameters)

## Phrasing Patterns

**Gate enforcement:**

- "The foreman scans your lapel. No union token—he shakes his head."
- "You recite the salvage clause; the clerk's shoulders drop."
- "You've seen this panel before; two quick twists, and the hatch yields."

**Choice clarity:**

- "Slip through maintenance."
- "Face the foreman directly."

## Gatekeeper Checklist

- [ ] No twist-revealing statements on any player surface
- [ ] No internals (gate logic, IDs, debug info)
- [ ] Gate phrasing is diegetic and natural
- [ ] Codex entries aid comprehension without spoilers
- [ ] All player-facing content is cold_store only

## Remediation

When a spoiler leak is detected:

1. Gatekeeper flags **presentation** failure
2. Content returns to hot_store with remediation note citing this principle
3. Originating role rephrases to diegetic/spoiler-safe version
4. Re-submit for gatecheck

## Summary

Keep wonder on the page and machinery off it. If a player could infer a twist or see the gears, it doesn't belong on the surface. Put it in hot_store, or phrase it in-world.
