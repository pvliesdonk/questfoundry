---
procedure_id: topology_design
name: Topology Design
description: Craft hubs (fan-out), loops (return-with-difference), and gateways (diegetic state checks); ensure meaningful nonlinearity
roles: [plotwright]
references_schemas:
  - topology_notes.schema.json
  - gateway_map.schema.json
references_expertises:
  - plotwright_topology
quality_bars: [reachability, nonlinearity, gateways]
---

# Topology Design

## Purpose

Design the narrative structure with hubs (fan-out points), loops (return-with-difference mechanics), and gateways (diegetic state checks) to create meaningful nonlinearity that rewards player choice and exploration.

## Core Topology Elements

### Hubs

**Definition:** Fan-out points where player chooses between multiple divergent paths.

**Requirements:**

- Each branch must offer distinct experience (not decorative)
- Divergence must be meaningful (different content, tone, or outcomes)
- Branches can converge later, but each must have unique moments

**Anti-Pattern:** Cosmetic hubs where all branches lead to same content with minor text variations.

### Loops

**Definition:** Return-with-difference mechanics where player revisits a location/situation but experiences change.

**Requirements:**

- Player recognizes the return (familiar setting/context)
- Situation has demonstrably changed (via codewords/state)
- Change is diegetically justified (world responds to prior actions)

**Anti-Pattern:** Decorative loops where return offers no new content or insight.

### Gateways

**Definition:** Diegetic state checks that control access based on what the world knows (not meta game state).

**Requirements:**

- Condition is in-world (has token, knows password, earned reputation)
- Enforceable by Player-Narrator without exposing internals
- Fair and signposted (player can anticipate requirements)
- At least one path to meet condition exists

**Anti-Pattern:** Meta gates ("if completed quest X") or unfair gates (no signposting).

## Steps

### 1. Frame Topology Scope

- Map parts/chapters affected
- Identify structural intent (expand hub, add loop, gate off content)
- Note constraints from canon (timeline, causality)

### 2. Sketch Structural Elements

- Mark hub points (fan-out locations)
- Mark loop returns (revisit opportunities)
- Mark gateway positions (access control points)

### 3. Define Gateway Conditions

For each gateway:

- **Diegetic Condition:** What the world checks (token, reputation, knowledge)
- **Player-Facing Phrase:** How PN describes it ("The foreman's token", "Union clearance")
- **Paths to Acquire:** How player can meet condition
- **Fair Signposting:** Where/how condition is telegraphed

### 4. Validate Reachability

- All critical beats (keystones) reachable via at least one path
- No dead ends that block progress
- Redundant paths around single-point-of-failure bottlenecks

### 5. Validate Nonlinearity

- Hubs offer distinct experiences (not just text variants)
- Loops provide return-with-difference (not empty revisits)
- Gateways create meaningful choice (not artificial delays)

### 6. Document Topology

Create topology notes including:

- Hub diagram with branches
- Loop mechanics (what changes on return)
- Gateway map (conditions and paths)
- Keystone locations for reachability validation

## Outputs

- `topology_notes` - Hubs/loops/gateways overview with rationale
- `gateway_map` - Diegetic gateway checks with fairness notes
- `hooks` - For canon gaps, codex anchors, structural clarifications

## Gateway Mapping Template

```yaml
gateway_id: engineering_access
location: Section "Reach Engineering"
condition_diegetic: "Maintenance hex-key"
condition_internal: codeword.maintenance_key
player_facing_phrase: "The maintenance hex-key unlocks crew passages"
signposting:
  - "Mentioned in Section 'Cargo Bay' (foreman dialogue)"
  - "Visible on foreman's desk in Section 'Office'"
paths_to_acquire:
  - "Take hex-key from foreman's desk (Section 'Office')"
  - "Persuade foreman to lend hex-key (Section 'Negotiate')"
fairness: "Signposted twice; two acquisition paths; optional content"
```

## Common Patterns

### Hub Design

- **Binary Hub:** 2 branches (simple choice)
- **Multi-Hub:** 3+ branches (complex exploration)
- **Weighted Hub:** One "obvious" path + hidden alternatives

### Loop Design

- **Discovery Loop:** Return reveals new information
- **Consequence Loop:** Return shows results of prior actions
- **Escalation Loop:** Return shows situation has worsened/improved

### Gateway Design

- **Token Gate:** Requires physical object
- **Knowledge Gate:** Requires information/password
- **Reputation Gate:** Requires earned status
- **Time Gate:** Requires sequence/timing (use sparingly)

## Anti-Funneling Rule

**Block when:** First-choice options are functionally equivalent (same destination + same opening beats).

**Require:** Divergent destination OR opening beats.

**Example:**

- ❌ "Go / Proceed" → same destination, same opening
- ✓ "Go quickly / Go cautiously" → same destination, different opening beats
- ✓ "Take shuttle / Take cargo hauler" → different destinations

## Quality Bars Pressed

- **Reachability:** Critical beats reachable; no dead ends
- **Nonlinearity:** Hubs/loops intentional, not decorative
- **Gateways:** Conditions enforceable, diegetic, fair

## Handoffs

- **To Scene Smith:** Send section briefs for drafting
- **To Lore Weaver:** Request canon justification for loop mechanics
- **To Codex Curator:** Flag taxonomy/clarity needs for gateway objects
- **To Gatekeeper:** Submit topology for Reachability/Nonlinearity/Gateways pre-gate

## Common Issues

- **Cosmetic Hubs:** Add outcome differences or remove
- **Unfair Gateways:** Add signposting and acquisition paths
- **Dead Ends:** Add exit routes or mark as intentional terminals
- **Topology Sprawl:** Split into smaller TUs and stage changes
