---
snippet_id: diegetic_gates
name: Diegetic Gates
description: Enforce access using in-world cues (badge, permit, ritual phrase); never meta speech (option locked, FLAG_X, roll check)
applies_to_roles: [player_narrator, gatekeeper, plotwright, style_lead, lore_weaver]
quality_bars: [gateways, presentation]
---

# Diegetic Gates

## Core Principle

All gates must be enforceable using in-world language. Player-Narrator delivers gates as world obstacles, never mechanical conditions.

## In-World Gate Patterns

### Physical Objects

✓ "The lock requires a hex-key you don't have"
✓ "The scanner flashes red—no clearance badge"
✓ "You need a union token to enter"

❌ "You don't have ITEM_HEX_KEY"
❌ "Option locked: missing CLEARANCE"
❌ "Requires Quest Item: Union Token"

### Knowledge/Skills

✓ "The ritual phrase escapes you"
✓ "The schematic's too complex without training"
✓ "You don't recognize the override sequence"

❌ "Skill check failed"
❌ "Intelligence < 5"
❌ "You haven't learned SKILL_OVERRIDE"

### Social Standing

✓ "The foreman eyes you coldly: 'Union members only'"
✓ "She doesn't trust you yet"
✓ "Your reputation precedes you—access denied"

❌ "Reputation too low"
❌ "Quest 'Foreman's Trust' incomplete"
❌ "Relationship score < 50"

### Environmental/Temporal

✓ "The airlock's on safety lockdown—come back after the shift change"
✓ "The maintenance tunnel's flooded"
✓ "It's too late—the bay's sealed for the night"

❌ "Time gate: wait until 18:00"
❌ "Area locked until EVENT_SHIFT_CHANGE"
❌ "Cooldown: 2 hours remaining"

## Plotwright Design

When designing gates, specify diegetic rationale:

```yaml
gate_id: foreman_office_access
gate_type: social
diegetic_check: "Foreman's approval"
in_world_cue: "Union membership or foreman's explicit invitation"
pn_phrasing: "The foreman blocks the door: 'Union members only'"
acquisition_paths:
  - "Join union (via union rep dialogue)"
  - "Earn foreman's trust (via favor quests)"
signposting:
  - "Union members visible entering office"
  - "Foreman mentions union-only policy in dialogue"
```

## Lore Weaver Support

Provide diegetic rationales (what the world checks), not logic:

```yaml
canon_justification: "Airlocks require EVA certification for safety"
diegetic_mechanism: "Safety system checks badge for EVA cert chip"
pn_enforcement: "The airlock panel blinks: 'EVA certification required'"
NOT: "if player.skills.eva >= 1 then allow"
```

## Style Lead Phrasing

Supply in-world refusals and gate lines:

```yaml
gate_scenario: "Player lacks maintenance access"
meta_version: "You don't have FLAG_MAINTENANCE_ACCESS"
diegetic_version: "The panel stays red—no maintenance clearance"

gate_scenario: "Player hasn't completed prerequisite"
meta_version: "Complete Quest 'Foreman's Trust' first"
diegetic_version: "The foreman doesn't trust you enough yet"
```

## Gatekeeper Validation

Pre-gate checks for diegetic phrasing:

- [ ] No codeword names visible
- [ ] No flag/variable references
- [ ] No skill check mentions
- [ ] No quest prerequisites by meta name
- [ ] In-world cues present (object, knowledge, social, environmental)
- [ ] PN can enforce without revealing mechanics

**Block if:**

- Meta language detected
- No in-world cue provided
- Gate logic exposed
- Enforcement requires mechanic knowledge

## Player-Narrator Performance

PN delivers gates using only in-world language:

```markdown
✓ "The scanner blinks red. No clearance badge, no entry."
✓ "The foreman crosses his arms: 'Union members only.'"
✓ "The airlock panel reads: EVA CERT REQUIRED."
✓ "You don't have the hex-key for this panel."

❌ "Option locked."
❌ "You need FLAG_OMEGA."
❌ "Roll a Persuasion check."
❌ "Quest 'Foreman's Trust' incomplete."
```

## Fairness Requirements

Diegetic gates must be:

1. **Signposted** (player warned 2+ times)
2. **Acquirable** (path to meet condition exists)
3. **Enforceable** (PN can deliver without mechanics)
4. **Fair** (player understands what's needed)

### Signposting Examples

**Gate:** Foreman office requires union membership

**Signpost 1:** Observe union members entering office
**Signpost 2:** Foreman mentions "union-only" policy in dialogue
**Gate Delivery:** "The foreman blocks the door: 'Union members only'"

Player understands WHY gate exists before encountering it.

## Common Violations

### Meta Speech

❌ "You don't have permission to access this area"
✓ "The guard stops you: 'Authorized personnel only'"

### Flag Names

❌ "Missing CODEWORD_OMEGA"
✓ "The terminal prompts for a code phrase you don't know"

### Skill Checks

❌ "Lockpicking failed"
✓ "The lock stays stubborn—you're not getting through this way"

### Quest Prerequisites

❌ "Complete 'Earn Trust' first"
✓ "She doesn't trust you enough to help"

## Translation Considerations

Diegetic gates must remain in-world across languages:

```yaml
source: "The scanner blinks red—no clearance badge"
es: "El escáner parpadea en rojo—sin tarjeta de autorización"
fr: "Le scanner clignote rouge—pas de badge d'accès"

NOT:
es: "Opción bloqueada: falta CLEARANCE_BADGE"
```

Translator receives:

- Diegetic source text
- Cultural context (security systems, badge access)
- Freedom to adapt in-world cue to target culture

## Validation Checklist

For each gate:

- [ ] Diegetic rationale provided (Lore)
- [ ] In-world phrasing specified (Style)
- [ ] Signposted 2+ times (Plotwright)
- [ ] Acquisition path exists (Plotwright)
- [ ] PN can enforce without mechanics (PN validation)
- [ ] No meta language present (Gatekeeper check)
- [ ] Fair to player (Gatekeeper check)
