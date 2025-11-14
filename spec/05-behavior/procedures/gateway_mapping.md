---
procedure_id: gateway_mapping
name: Gateway Mapping
description: Map gateway intents (what the world checks, not mechanics); ensure reachability paths exist; keep conditions diegetic
roles: [plotwright]
references_schemas:
  - gateway_map.schema.json
references_expertises:
  - plotwright_topology
quality_bars: [gateways, reachability]
---

# Gateway Mapping

## Purpose

Design and document gateways (access control points) with diegetic conditions that Player-Narrator can enforce without exposing internals or codewords.

## Gateway Anatomy

### Diegetic Condition

**What the world checks** (not internal game state)

Examples of GOOD diegetic conditions:

- "Has the foreman's token"
- "Knows the maintenance code"
- "Earned union clearance"
- "Carries medical credentials"

Examples of BAD meta conditions:

- "Completed quest X"
- "Has flag Y set"
- "Score > 50"
- "Visited section Z"

### Player-Facing Phrase

**How PN describes the gate when player encounters it**

Examples:

- "The door requires a foreman's token"
- "The terminal asks for your maintenance code"
- "The guard checks your union clearance"
- "Medical Bay requires proper credentials"

### Internal Codeword (Hot Only)

**Technical tracking** (NEVER appears on player surfaces)

Examples:

- `codeword.foreman_token`
- `codeword.maintenance_code`
- `codeword.union_clearance`

### Paths to Acquire

**How player can meet the condition**

Requirements:

- At least one path must exist
- Paths should be signposted
- Multiple paths preferred (player agency)

### Fair Signposting

**Where/how condition is telegraphed to player**

Requirements:

- Player should see gate mentioned before encountering
- Acquisition opportunities should be discoverable
- No "guess the password" scenarios

## Steps

### 1. Identify Gateway Location

- Which section requires access control?
- What's being gated (optional content, critical path, secret)?

### 2. Design Diegetic Condition

- What in-world object, knowledge, or status gates this?
- Can PN phrase this without exposing internals?
- Does this fit world logic (canon-compatible)?

### 3. Map Acquisition Paths

- How can player obtain this condition?
- Are there multiple paths (agency)?
- Are paths reachable from player's current position?

### 4. Plan Signposting

- Where is gate first mentioned?
- Where are acquisition opportunities visible?
- Is timing fair (player has chance to prepare)?

### 5. Validate Fairness

- Can player anticipate this requirement?
- Are there at least 2 signposting moments?
- Does at least one acquisition path exist?
- Is gate enforceable diegetically by PN?

### 6. Document Gateway

Create gateway map entry with all components

## Gateway Map Template

```yaml
gateway_id: engineering_access
location:
  section: "Reach Engineering"
  line: "You approach the sealed engineering hatch"

condition_diegetic: "Maintenance hex-key"
condition_internal: "codeword.maintenance_key"
player_facing_phrase: "The hatch requires a maintenance hex-key"

paths_to_acquire:
  path_1:
    method: "Take from foreman's desk"
    section: "Office"
    requirements: []

  path_2:
    method: "Persuade foreman to lend key"
    section: "Negotiate with Foreman"
    requirements: ["reputation.union_friendly"]

signposting:
  mention_1:
    section: "Cargo Bay"
    line: "The foreman mentions that only maintenance keys open crew passages"

  mention_2:
    section: "Office"
    line: "A six-sided hex-key sits on the foreman's desk"

fairness_notes: "Two signposting moments; two acquisition paths; optional content"
gate_type: "token"
criticality: "optional"  # or "critical" for main path
```

## Gateway Types

### Token Gates

**Requires physical object**

- Keycard, hex-key, badge, data chip
- Easiest to communicate diegetically
- Clear acquisition (find/steal/earn)

### Knowledge Gates

**Requires information**

- Password, code, ritual phrase
- Can be learned through exploration
- Risk: "guess the password" anti-pattern

### Reputation Gates

**Requires earned status**

- Union member, trusted ally, clearance level
- Built through prior actions
- Harder to communicate clearly

### Combination Gates

**Requires multiple conditions**

- "Engineering badge AND maintenance key"
- Use sparingly (complex for player)

## Fairness Criteria

### ✓ Fair Gateway

- Signposted at least twice
- At least one acquisition path exists
- Acquisition path is reachable
- Condition is diegetic and PN-enforceable
- Player can anticipate requirement

### ✗ Unfair Gateway

- No signposting ("surprise gate")
- No acquisition path (impossible)
- Meta condition (breaks immersion)
- Arbitrary timing (no player agency)

## Common Patterns

### Optional Content Gates

- Gate off side content, not critical path
- Rewards exploration
- Multiple acquisition paths preferred

### Critical Path Gates

- Use sparingly
- Require redundant paths (no single point of failure)
- Heavy signposting (3+ mentions)

### Secret Gates

- Hidden condition (password, ritual phrase)
- Multiple discovery paths
- Never block critical content with secrets

## Outputs

- `gateway_map` - Complete map of all gateways with conditions and paths
- `hooks` - For canon justification, codex entries, PN phrasing patterns

## Quality Bars Pressed

- **Gateways:** Conditions enforceable, diegetic, fair
- **Reachability:** Critical content accessible, no impossible gates

## Handoffs

- **To Player-Narrator:** Provide diegetic phrasing patterns for enforcement
- **To Lore Weaver:** Request canon justification for gate conditions
- **To Codex Curator:** Request entries for gate objects/concepts
- **To Gatekeeper:** Submit for Gateways and Reachability validation

## Common Issues

- **Meta Conditions:** "Completed X" ❌ → "Has X's badge" ✓
- **No Signposting:** Player surprised by gate
- **Impossible Gates:** No acquisition path exists
- **Unfair Timing:** Gate blocks player with no warning
- **Guess the Password:** No clues for knowledge gates
