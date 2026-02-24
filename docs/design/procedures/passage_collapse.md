# QuestFoundry v5 — Passage-Level Collapse Design

**Status:** Design Draft — Superseded
**Parent:** grow.md
**Issue:** #634
**Purpose:** Detailed design for passage-level collapse and transition handling

> **Superseded (2026-02-24).** [Document 1, Part 4](../how-branching-stories-work.md) (Passage Collapse section) and [Document 3, Part 5](../document-3-ontology.md) (The Passage Layer) move passage collapse from GROW to the POLISH stage. The concepts described here are largely preserved but the stage assignment changes. Retained for historical reference. The code has not yet been reorganized.

---

## Problem Statement

### Linear Stretches After Passage Creation

The current GROW implementation collapses linear beats in Phase 7b, but this happens
before fork/spoke/gap passages are created (Phases 8-9). This creates new linear
stretches that weren't present at beat level:

```
fork_a → gap_2 → beat_04 → beat_02
```

These linear stretches are flagged by `qf inspect` but were never candidates for
collapse because they didn't exist during Phase 7b.

### Gap Passages Lack Context

Gap beats are created as minimal placeholders without entities or location, causing:

1. Hard transition warnings (`shared_entities: 0`) during FILL
2. No context for FILL to write smooth transitions
3. But gaps ARE the transition — they shouldn't trigger warnings

### 1:1 Beat-Passage Limitation

Each beat becomes exactly one passage. Story beats like "Pim finds the letter"
and "Pim reads the letter" could be one passage, avoiding unnecessary clicks.

---

## Proposed Solution

### 1. Move Collapse to Passage Level

Remove/skip Phase 7b (beat-level collapse). Add new phase after all passage creation:

```
Current phases:
  Phase 7b: collapse_linear_beats (beat-level)
  Phase 8a: Create passages from beats
  Phase 9b/9c: Create fork/spoke passages

New phases:
  Phase 8a: Create passages from beats
  Phase 9b/9c: Create fork/spoke passages
  Phase 9d: collapse_linear_passages ← NEW
  Phase 10: validation
```

### 2. Gap Beats as Transition Bridges

When inserting gap beats (Phase 4b/4c), enrich them with context:

```yaml
gap_beat:
  type: beat
  raw_id: gap_1
  scene_type: micro_beat
  # NEW fields
  entities: [character::pim, character::mentor]  # Union from adjacent beats
  location: location::manor_hall                  # From adjacent, or null if different
  transition_style: smooth | cut                  # LLM decides based on context
  bridges_from: beat::beat_04                     # Source beat
  bridges_to: beat::beat_05                       # Target beat
```

**Transition style heuristics:**
- `smooth`: Same location, shared entities, temporal continuity
- `cut`: Location change, time jump, POV shift, intentional break

### 3. N:1 Beat-to-Passage Mapping

After collapse, passages may derive from multiple beats:

```yaml
# Before collapse
passage::beat_04:
  from_beat: beat::beat_04

passage::gap_1:
  from_beat: beat::gap_1

passage::beat_05:
  from_beat: beat::beat_05

# After collapse
passage::merged_04_05:
  from_beats:
    - beat::beat_04
    - beat::gap_1
    - beat::beat_05
  primary_beat: beat::beat_04    # For ID derivation and main summary
  merged_from:                   # Traceability
    - passage::beat_04
    - passage::gap_1
    - passage::beat_05
  transition_points:             # Guide for FILL
    - index: 1                   # After first beat's content
      style: smooth
      bridge_entities: [character::pim]
      note: "Continue in same scene"
    - index: 2                   # After gap
      style: smooth
      bridge_entities: [character::pim, character::mentor]
      note: "Mentor arrives"
```

### 4. FILL Context for Merged Passages

For merged passages, FILL receives enhanced context:

```yaml
## Passage Context (merged)

**Primary Summary:** Pim discovers the hidden letter behind the portrait.

**Beat Sequence:**
1. [beat::beat_04] Pim searches the study methodically
2. [gap] (smooth transition - continue action)
3. [beat::beat_05] Pim finds the letter and reads it

**Transition Guidance:**
- After "searches the study": Smooth continuation. Keep action flowing.
- After gap: Smooth arrival. Mentor enters naturally.

**Writing Instruction:**
Write as continuous prose with smooth transitions. Do NOT insert
scene breaks or time jumps between beats. The merged passage should
read as one cohesive scene.

**Entities present:** Pim, Mentor (joins mid-scene)
**Location:** Manor Study (unchanged throughout)
```

---

## Ontology Changes

### Current Schema

```yaml
# 1:1 mapping
passage:
  from_beat: beat::some_beat_id
```

### New Schema

```yaml
passage:
  # Single beat (unchanged for simple passages)
  from_beat: beat::some_beat_id

  # OR multiple beats (collapsed passages)
  from_beats:
    - beat::beat_04
    - beat::gap_1
    - beat::beat_05
  primary_beat: beat::beat_04
  merged_from:
    - passage::beat_04
    - passage::gap_1
    - passage::beat_05
  transition_points:
    - index: int         # Position in from_beats list
      style: smooth | cut
      bridge_entities: [entity_id, ...]
      note: str          # Human-readable guidance
```

### Backwards Compatibility

- Single-beat passages keep existing `from_beat` field
- Code checks `from_beats` first, falls back to `from_beat`
- `merged_from` only present on collapsed passages

---

## Algorithm: `collapse_linear_passages`

### Input
- Graph with all passages created (after Phase 9c)
- Collapse threshold (default: 3)

### Algorithm

```python
def collapse_linear_passages(graph: Graph, threshold: int = 3) -> None:
    """Collapse linear passage chains into merged passages.

    A "linear chain" is a sequence of passages where each has exactly
    one outgoing choice leading to the next passage.
    """
    # 1. Find all linear chains
    chains = find_linear_chains(graph, min_length=threshold)

    for chain in chains:
        # 2. Check if chain is collapse-worthy
        if not should_collapse(chain, graph):
            continue

        # 3. Create merged passage
        merged = create_merged_passage(chain, graph)

        # 4. Update graph
        #    - Add merged passage node
        #    - Redirect incoming edges to merged
        #    - Redirect outgoing edges from merged
        #    - Remove original passages (but keep beats)
        update_graph_for_merge(graph, chain, merged)
```

### Chain Detection

```python
def find_linear_chains(graph: Graph, min_length: int) -> list[list[str]]:
    """Find passage chains with single-exit nodes."""
    chains = []
    visited = set()

    for passage_id in graph.get_nodes_by_type("passage"):
        if passage_id in visited:
            continue

        chain = trace_linear_chain(graph, passage_id)
        if len(chain) >= min_length:
            chains.append(chain)
            visited.update(chain)

    return chains
```

### Collapse Criteria

A chain is collapse-worthy if:

1. **Length >= threshold** (default 3)
2. **Scene continuity**: Same or compatible locations
3. **Entity continuity**: Shared entities across chain
4. **No hard cuts**: No `transition_style: cut` in gaps
5. **No structural importance**: Not a hub or key divergence point

```python
def should_collapse(chain: list[str], graph: Graph) -> bool:
    """Determine if a linear chain should be collapsed."""
    if len(chain) < COLLAPSE_THRESHOLD:
        return False

    # Check for hard cuts
    for passage_id in chain:
        passage = graph.get_node(passage_id)
        beat = graph.get_node(passage.get("from_beat"))
        if beat and beat.get("transition_style") == "cut":
            return False

    # Check scene continuity
    locations = [get_passage_location(graph, p) for p in chain]
    if len(set(locations) - {None}) > 1:
        # Multiple distinct locations — don't collapse
        return False

    return True
```

### Merged Passage Creation

```python
def create_merged_passage(chain: list[str], graph: Graph) -> dict:
    """Create a merged passage from a chain."""
    passages = [graph.get_node(p) for p in chain]
    beats = [graph.get_node(p["from_beat"]) for p in passages]

    # Primary beat is first non-gap beat
    primary = next(
        (b for b in beats if not b.get("is_gap")),
        beats[0]
    )

    # Build transition points
    transitions = []
    for i, beat in enumerate(beats[1:], 1):
        if beat.get("is_gap") or beat.get("bridges_from"):
            transitions.append({
                "index": i,
                "style": beat.get("transition_style", "smooth"),
                "bridge_entities": beat.get("entities", []),
                "note": beat.get("summary", ""),
            })

    # Collect all entities
    all_entities = set()
    for beat in beats:
        all_entities.update(beat.get("entities", []))

    return {
        "type": "passage",
        "raw_id": f"merged_{primary['raw_id']}",
        "from_beats": [p["from_beat"] for p in passages],
        "primary_beat": primary["id"],
        "merged_from": chain,
        "transition_points": transitions,
        "entities": list(all_entities),
        "summary": primary.get("summary", ""),
    }
```

---

## Gap Beat Enrichment

### Current Gap Creation (Phase 4b/4c)

```python
def insert_gap_beat(graph, from_beat_id, to_beat_id):
    gap = {
        "type": "beat",
        "raw_id": f"gap_{counter}",
        "scene_type": "micro_beat",
        "summary": "Transition beat",
    }
    graph.create_node(gap_id, gap)
```

### Enhanced Gap Creation

```python
def insert_gap_beat(graph, from_beat_id, to_beat_id):
    from_beat = graph.get_node(from_beat_id)
    to_beat = graph.get_node(to_beat_id)

    # Inherit entities (union of both)
    entities = set(from_beat.get("entities", []))
    entities.update(to_beat.get("entities", []))

    # Determine location
    from_loc = from_beat.get("location")
    to_loc = to_beat.get("location")
    location = from_loc if from_loc == to_loc else None

    # Determine transition style
    style = infer_transition_style(from_beat, to_beat)

    gap = {
        "type": "beat",
        "raw_id": f"gap_{counter}",
        "scene_type": "micro_beat",
        "summary": f"Transition from {from_beat['raw_id']} to {to_beat['raw_id']}",
        "is_gap": True,
        "entities": list(entities),
        "location": location,
        "transition_style": style,
        "bridges_from": from_beat_id,
        "bridges_to": to_beat_id,
    }
    graph.create_node(gap_id, gap)
```

### Transition Style Inference

```python
def infer_transition_style(from_beat: dict, to_beat: dict) -> str:
    """Infer whether transition should be smooth or a cut."""
    # Same location = smooth
    if from_beat.get("location") == to_beat.get("location"):
        # Check for shared entities
        from_entities = set(from_beat.get("entities", []))
        to_entities = set(to_beat.get("entities", []))
        if from_entities & to_entities:
            return "smooth"

    # Scene type changes often warrant cuts
    if from_beat.get("scene_type") != to_beat.get("scene_type"):
        return "cut"

    # Different location = likely cut
    if from_beat.get("location") != to_beat.get("location"):
        return "cut"

    return "smooth"  # Default to smooth
```

---

## FILL Integration

### Detecting Merged Passages

```python
def is_merged_passage(passage: dict) -> bool:
    """Check if a passage is a merge of multiple beats."""
    return bool(passage.get("from_beats"))
```

### Context Formatting for Merged Passages

```python
def format_merged_passage_context(graph: Graph, passage_id: str) -> str:
    """Format rich context for a merged passage."""
    passage = graph.get_node(passage_id)

    if not is_merged_passage(passage):
        return format_passage_context(graph, passage_id)  # Existing function

    lines = ["## Merged Passage Context"]

    # Primary summary
    primary = graph.get_node(passage["primary_beat"])
    lines.append(f"\n**Primary Summary:** {primary.get('summary', '')}")

    # Beat sequence
    lines.append("\n**Beat Sequence:**")
    for i, beat_id in enumerate(passage["from_beats"], 1):
        beat = graph.get_node(beat_id)
        if beat.get("is_gap"):
            style = beat.get("transition_style", "smooth")
            lines.append(f"{i}. [gap] ({style} transition)")
        else:
            lines.append(f"{i}. [{beat_id}] {beat.get('summary', '')}")

    # Transition guidance
    if passage.get("transition_points"):
        lines.append("\n**Transition Guidance:**")
        beats = passage["from_beats"]
        for tp in passage["transition_points"]:
            idx = tp["index"]
            prior_beat = graph.get_node(beats[idx - 1])
            style = tp["style"]
            note = tp.get("note", "")
            lines.append(f"- After \"{prior_beat.get('summary', '')[:50]}...\": {style.title()}. {note}")

    # Writing instruction
    lines.append("\n**Writing Instruction:**")
    lines.append("Write as continuous prose. The beats above should flow as ONE scene.")
    lines.append("Use the transition guidance to smoothly connect each section.")

    return "\n".join(lines)
```

### Suppressing Hard Transition Warnings

Update `format_continuity_warning` to recognize gap passages:

```python
def format_continuity_warning(...) -> str:
    # ... existing code ...

    # Don't warn for gap passages — they ARE the transition
    cur_beat = graph.get_node(cur_passage.get("from_beat", ""))
    if cur_beat and cur_beat.get("is_gap"):
        return ""  # Gap passages don't need transition warnings

    # ... rest of existing logic ...
```

---

## Validation Updates

### `qf inspect` Changes

Update linear stretch detection to understand merged passages:

```python
def find_linear_stretches(graph: Graph) -> list[list[str]]:
    """Find problematic linear stretches, excluding collapsed ones."""

    # Get all passage chains
    chains = trace_passage_chains(graph)

    linear = []
    for chain in chains:
        # Skip if chain is a single merged passage
        if len(chain) == 1:
            passage = graph.get_node(chain[0])
            if is_merged_passage(passage):
                continue

        # Skip if all passages are merged
        if all(is_merged_passage(graph.get_node(p)) for p in chain):
            continue

        # Flag as linear stretch
        if len(chain) >= LINEAR_THRESHOLD:
            linear.append(chain)

    return linear
```

---

## Implementation Phases

### Phase 1: Gap Beat Enrichment
- Update `insert_gap_beat()` in grow.py
- Add entity/location inheritance
- Add `transition_style`, `bridges_from`, `bridges_to` fields
- Unit tests for gap creation

### Phase 2: Suppress Gap Warnings
- Update `format_continuity_warning()` in fill_context.py
- Don't warn when current passage is from a gap beat
- Unit tests

### Phase 3: Passage Schema Extension
- Add `from_beats`, `primary_beat`, `merged_from`, `transition_points` fields
- Update passage validation
- Backwards compatibility for `from_beat`

### Phase 4: Collapse Algorithm
- Implement `find_linear_chains()`
- Implement `should_collapse()`
- Implement `create_merged_passage()`
- Add Phase 9d to GROW

### Phase 5: FILL Integration
- Implement `format_merged_passage_context()`
- Update prose generation to use merged context
- Unit tests

### Phase 6: Validation Updates
- Update `qf inspect` to understand merged passages
- Update linear stretch detection

### Phase 7: Spec Documentation
- Update ontology in 00-spec.md
- Update GROW procedure docs

---

## Configuration

### Collapse Threshold

```yaml
# In project.yaml
grow:
  collapse_threshold: 3    # Minimum chain length to collapse
  collapse_enabled: true   # Enable/disable collapse phase
```

### Transition Style Override

Allow manual override of transition style in beat definitions:

```yaml
beat:
  id: beat::dramatic_reveal
  summary: "The truth is revealed"
  transition_style: cut    # Force hard cut before this beat
```

---

## Risks and Mitigations

### Risk: Over-collapse
Merging too aggressively could create overly long passages.

**Mitigation:**
- Cap merged passage length (e.g., max 5 beats)
- Respect `transition_style: cut` markers
- Allow manual exclusion via beat metadata

### Risk: Lost Granularity
Beat-level analysis relies on 1:1 beat-passage mapping.

**Mitigation:**
- Keep beats intact; only collapse passages
- Maintain `merged_from` traceability
- `primary_beat` preserves main identity

### Risk: FILL Complexity
Merged passages need different context formatting.

**Mitigation:**
- Clear `is_merged_passage()` check
- Dedicated formatting function
- Explicit transition guidance in prompt

---

## Success Criteria

1. [ ] Gap beats have entities/location from adjacent beats
2. [ ] Gap beats have `transition_style` field (smooth/cut)
3. [ ] `collapse_linear_passages` phase implemented
4. [ ] Merged passages have correct schema (`from_beats`, `merged_from`, etc.)
5. [ ] FILL handles merged passages with rich context
6. [ ] `qf inspect` understands merged passages (no false linear warnings)
7. [ ] No "hard transition" warnings for gap passages
8. [ ] Ontology documentation updated
9. [ ] Configuration options (threshold, enable/disable)
