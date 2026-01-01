# GROW Stage Specification

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

GROW is the most complex stage in the v5 pipeline. It transforms the SEED artifact into a complete story topology with branching paths, state management, and scene-level specifications.

GROW decomposes into **six sequential layers**:

```
SPINE → ANCHORS → [HARVEST] → FRACTURES → BRANCHES → CONNECTIONS → BRIEFS
```

Each layer builds on the previous. The optional HARVEST checkpoint allows controlled iteration after ANCHORS.

---

## The Six Layers

### Layer 1: SPINE

**Purpose**: Establish the core emotional arc before any branching.

**Input**: `seed.yaml`

**Output**: `grow/spine.yaml`

**LLM Calls**: 1

**What It Does**:
- Defines the story arc shape (rise, fall, rise-fall, etc.)
- Establishes beat sequence (opening, inciting, crisis, climax, resolution)
- Maps emotional trajectory through the story
- Describes what every player experiences in some form

**Key Rule**: The spine is the narrative through-line, not necessarily the "main path." Even if a player takes divergent branches, the emotional arc follows this spine.

**Anti-Pattern**: Generating branches at this stage. SPINE must be linear.

```yaml
# Example spine beat
beats:
  - id: crisis
    beat_type: crisis
    description: "Evidence points to Maria as the killer"
    emotional_state: desperation
    # NO choices here — this is structural, not interactive
```

### Layer 2: ANCHORS

**Purpose**: Declare structural constraints that will shape branching.

**Input**: `seed.yaml` + `spine.yaml`

**Output**: `grow/anchors.yaml`

**LLM Calls**: 1

**Human Gate**: **Required** — Structure is locked after approval

**What It Defines**:

| Element | Purpose |
|---------|---------|
| **Hubs** | Return points where player choice fans out |
| **Bottlenecks** | Where all paths must reconverge |
| **Gates** | Where progression requires conditions |
| **Endings** | Terminal states (how story can end) |
| **State** | Codewords, stats, items (see [04-state-mechanics.md](./04-state-mechanics.md)) |

**Key Rule**: Anchors constrain branching. They are **declared early**, not **discovered during branch design**.

**Anti-Pattern**: Discovering that you need a new hub while writing branches. If this happens, go back and revise ANCHORS (requires human approval).

```yaml
# ANCHORS defines what BRANCHES must connect to
hubs:
  - id: office_hub
    description: "Maria's office — investigation hub"

bottlenecks:
  - id: arrest_scene
    description: "All paths lead to Maria's arrest"
    after_beat: spine.crisis
    required: true  # Cannot be bypassed

gates:
  - id: morgue_access
    condition:
      type: codeword
      requires: has_badge_favor
```

### HARVEST Checkpoint (Optional)

**Purpose**: Controlled iteration — promote emergent hooks before locking structure.

**When**: After initial ANCHORS, before FRACTURES

**Trigger**: `qf grow --extra-round` or configuration

**Default**: 1 round (effectively skipped after first ANCHORS approval)

**Max**: Configurable, typically 3 rounds

**Process**:

1. Review hooks captured during SPINE/ANCHORS generation
2. Decide which hooks should become structural elements
3. Revise ANCHORS if needed (ANCHORS v2, v3...)
4. Human approves revised ANCHORS
5. Lock structure and proceed to FRACTURES

**Example**:
- During SPINE: LLM mentions "the scar Maria got that night"
- During HARVEST: Human decides this should be a codeword (`has_scar_story`)
- ANCHORS v2: Add codeword and a gate that uses it

**Anti-Pattern**: Unbounded iteration. HARVEST has a maximum round limit.

```yaml
# pipeline.yaml configuration
iteration:
  grow.harvest_rounds: 1       # Default rounds
  grow.max_harvest_rounds: 3   # Hard limit
```

### Layer 3: FRACTURES

**Purpose**: Identify where the spine can meaningfully diverge.

**Input**: `seed.yaml` + `spine.yaml` + `anchors.yaml`

**Output**: `grow/fractures.yaml`

**LLM Calls**: 1

**What It Defines**:
- Points on the spine where meaningful divergence occurs
- What distinguishes each option (not cosmetic differences)
- Stakes at each fracture point
- Which anchors each branch must ultimately reach

**Key Rule**: Fractures are **meaningful divergences**, not cosmetic choices. If options lead to the same outcome with only flavor text differences, it's not a fracture.

```yaml
fractures:
  - id: investigation_approach
    spine_beat: spine.first_turn
    description: "How Maria begins her investigation"
    stakes: |
      This determines who Maria trusts and what she learns first.
      Different paths reveal different pieces of the truth.
    options:
      - id: chinatown_path
        description: "Investigate through family and community"
        must_reach: [chinatown_hub, arrest_bottleneck]
      - id: police_path
        description: "Use old police connections"
        must_reach: [office_hub, arrest_bottleneck]
      - id: underground_path
        description: "Go through criminal contacts"
        must_reach: [arrest_bottleneck]  # Skips some hubs
```

### Layer 4: BRANCHES

**Purpose**: Expand one branch at a time, building on established content.

**Input**: Compressed spine + full anchors/fractures + prior approved branches

**Output**: `grow/branches/<branch_id>.yaml` (one per branch)

**LLM Calls**: N (one per branch, sequential)

**Process**:
1. Select highest-priority unwritten fracture option
2. Generate branch content with full anchor constraints
3. Validate connection to required anchors
4. Human reviews branch
5. Repeat for next fracture option

**Key Rule**: Sequential expansion prevents disconnected parallel narratives. Each branch generation receives:
- Compressed spine summary
- Full anchors and fractures
- All previously approved branches

**Context Windowing**:
```
┌─────────────────────────────────────┐
│ Branch Generation Context           │
├─────────────────────────────────────┤
│ spine.yaml        → Compressed      │
│ anchors.yaml      → Full            │
│ fractures.yaml    → Full            │
│ prior branches    → Full (or compressed if many) │
│ seed.yaml         → Key excerpts    │
└─────────────────────────────────────┘
```

**Anti-Pattern**: Generating all branches in parallel. This causes:
- Inconsistent world state between branches
- Contradictory character behavior
- Broken connections to anchors

**Branch Ordering**:
- Priority 1: Branches from earlier spine beats
- Priority 2: Branches that establish key codewords
- Priority 3: Branches that connect to most anchors

### Layer 5: CONNECTIONS

**Purpose**: Validate complete topology before writing prose.

**Input**: All GROW artifacts

**Output**: `grow/connections.yaml` (validation report + topology map)

**LLM Calls**: 0 (deterministic validation) or 1 (narrative coherence review)

**Validation Checks**:

| Check | Description | Automated |
|-------|-------------|-----------|
| Reachability | All passages reachable from start | Yes |
| Anchor coverage | All branches connect to required anchors | Yes |
| No orphans | No passages without incoming edges | Yes |
| Gate satisfiability | All gate conditions have obtainable paths | Yes |
| Ending reachability | All endings have valid paths | Yes |
| State consistency | Codewords don't create contradictions | Partial |
| Narrative coherence | Story makes sense across branches | LLM |

**Output Format**:
```yaml
type: grow_connections
version: 1

validation:
  status: pass  # pass | fail | warn
  checks:
    - name: reachability
      status: pass
      details: "All 47 passages reachable from opening_001"
    - name: gate_satisfiability
      status: warn
      details: |
        Gate 'tong_meeting_gate' requires chinatown_trust >= 30.
        Only 2 of 5 paths can achieve this threshold.

topology:
  passage_count: 47
  branch_count: 8
  ending_count: 3
  hub_count: 2
  bottleneck_count: 2

  # Adjacency information for graph traversal
  passages:
    - id: opening_001
      outgoing: [phone_call_001, ignore_phone_001]
      incoming: []
    - id: phone_call_001
      outgoing: [morgue_discovery]
      incoming: [opening_001]
```

### Layer 6: BRIEFS

**Purpose**: Scene-level specifications for prose generation.

**Input**: Full topology + relevant branch details

**Output**: `grow/briefs/<passage_id>.yaml` (one per passage)

**LLM Calls**: N (one per passage, can batch similar passages)

**What Each Brief Contains**:
- Passage ID and connections
- Context (what led here, what player knows)
- Location and characters present
- Stakes and emotional beat alignment
- Prose guidance (length, tone, POV, sensory focus)
- Choice text hints

**Key Rule**: Brief = contract for FILL. The prose stage must honor the brief's intent without changing connections or state effects.

**Anti-Pattern**: Including actual prose in briefs. Briefs specify intent; FILL generates prose.

```yaml
# Brief specifies WHAT, not HOW
prose_guidance:
  length: medium
  sensory_focus:
    - smell of cooking
    - sound of rain
  dialogue_weight: heavy
  tone_notes: "Tension masked by small talk"

# NOT this:
prose: "The rain fell heavily as Maria entered..."  # WRONG
```

---

## Context Windowing by Layer

Each layer receives specific context to prevent token overflow:

| Layer | Receives | Compression |
|-------|----------|-------------|
| SPINE | seed.yaml only | Full |
| ANCHORS | seed + spine | Full |
| FRACTURES | seed + spine + anchors | Full |
| BRANCHES | spine (compressed) + anchors/fractures (full) + prior branches | Mixed |
| CONNECTIONS | Full topology | Chunked if needed |
| BRIEFS | Topology skeleton + relevant branch detail | Selective |

**Compression Strategies**:
- **Full**: Complete artifact
- **Summary**: LLM-generated overview
- **Skeleton**: Structure only (IDs, relationships)
- **Selective**: Only portions relevant to current generation

---

## Validation Between Layers

| Transition | Validation |
|------------|------------|
| SEED → SPINE | Seed is complete (protagonist, setting, tension defined) |
| SPINE → ANCHORS | All spine beats have IDs and emotional states |
| ANCHORS → FRACTURES | All state types defined, anchors have unique IDs |
| FRACTURES → BRANCHES | Fractures have clear options with `must_reach` anchors |
| BRANCHES → CONNECTIONS | Each branch connects to its declared anchors |
| CONNECTIONS → BRIEFS | Topology is valid (all passages reachable, no orphans) |

---

## CLI Commands for GROW

```bash
# Run all GROW layers
qf grow

# Run specific layer
qf grow --layer spine
qf grow --layer anchors
qf grow --layer fractures
qf grow --layer branches
qf grow --layer connections
qf grow --layer briefs

# Request additional HARVEST round
qf grow --extra-round

# Skip to specific layer (requires prior layers complete)
qf grow --from fractures

# Validate without regenerating
qf grow --validate-only
```

---

## Error Recovery

### Mid-GROW Failures

If GROW fails during branch generation:

1. Completed branches are preserved
2. Error logged with context
3. Human can:
   - **Retry** the failed branch
   - **Edit** the branch manually
   - **Rollback** to ANCHORS and try different fractures

### Post-CONNECTIONS Failures

If topology validation fails:

1. Specific issues identified
2. Human reviews the validation report
3. Options:
   - **Edit** affected branches manually
   - **Regenerate** specific branches with fixes
   - **Revise ANCHORS** (requires re-running FRACTURES→BRIEFS)

---

## Research Foundation

The GROW decomposition is based on research from the if-craft-corpus:

> **Key insight**: Anchors are structural, declared early. They constrain branching rather than emerging from it.

> **Bottom-up iteration** outperforms top-down generation; each branch references established content before the next is generated.

See [08-research-foundation.md](./08-research-foundation.md) for full research grounding.

---

## Anti-Patterns Summary

| Anti-Pattern | Why It Fails |
|--------------|--------------|
| Branches before anchors | No structure to connect to |
| Parallel branch generation | Inconsistent world state |
| Discovering new anchors in BRANCHES | Breaks locked structure |
| Prose in briefs | Conflates structure and content |
| Topology changes after CONNECTIONS | Invalidates validated graph |
| Unbounded HARVEST | No clear stopping criteria |

---

## See Also

- [02-artifact-schemas.md](./02-artifact-schemas.md) — GROW artifact formats
- [04-state-mechanics.md](./04-state-mechanics.md) — State system defined in ANCHORS
- [08-research-foundation.md](./08-research-foundation.md) — Research basis
