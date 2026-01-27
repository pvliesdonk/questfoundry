# Graph Storage Architecture

**Version**: 1.0.0
**Status**: Proposed

## Overview

This document describes the runtime implementation of the unified story graph.
The design principles and ontology are defined in [00-spec.md](../design/00-spec.md);
this document covers implementation details.

## Design Principles

1. **Docs are truth. Code implements truth.** The ontology lives in design docs.
   Pydantic models implement the ontology but are not the source of truth.

2. **Single graph, multiple views.** One JSON file stores the graph.
   Human-readable exports (YAML, Markdown) are generated on demand.

3. **In-memory during stage.** Graph loads at stage start, modifications happen
   in memory, and the graph persists at stage end.

4. **Stage-level rollback.** Pre-stage snapshots enable reverting failed or
   rejected stages without complex event sourcing.

## Storage Format

### Primary Storage: JSON

The graph is stored as a single JSON file (`graph.json`) in the project root.

```json
{
  "version": "5.0",
  "meta": {
    "project_name": "noir_mystery",
    "last_stage": "seed",
    "last_modified": "2026-01-13T10:30:00Z",
    "stage_history": [
      {"stage": "dream", "completed": "2026-01-13T09:00:00Z"},
      {"stage": "brainstorm", "completed": "2026-01-13T09:30:00Z"},
      {"stage": "seed", "completed": "2026-01-13T10:30:00Z"}
    ]
  },
  "nodes": {
    "vision": {
      "type": "vision",
      "genre": "noir mystery",
      "tone": "gritty, atmospheric",
      "themes": ["trust", "redemption"],
      "constraints": ["no explicit violence"],
      "length": "medium"
    },
    "detective_001": {
      "type": "entity",
      "entity_type": "character",
      "concept": "Weary private eye with a dark past",
      "disposition": "retained",
      "base": {
        "name": "Sam Cross",
        "details": "..."
      },
      "overlays": []
    }
  },
  "edges": [
    {
      "type": "choice",
      "from": "opening_001",
      "to": "investigate_office",
      "label": "Head to the office",
      "requires": [],
      "grants": ["visited_office"]
    },
    {
      "type": "has_answer",
      "from": "mentor_trust",
      "to": "mentor_protector"
    }
  ]
}
```

### Why JSON over YAML for Storage

| Concern | JSON | YAML |
|---------|------|------|
| Parse speed | Fast | Slower |
| Strictness | Strict (catches errors) | Forgiving (hides errors) |
| Round-trip safety | Excellent | Risky (comments, formatting) |
| Tool ecosystem | Universal | Good but quirky |
| Human editing | Not intended | Better for review |

**Decision:** JSON for storage, YAML for human review exports.

## Graph Operations

### Loading

```python
class Graph:
    @classmethod
    def load(cls, project_path: Path) -> "Graph":
        """Load graph from project directory."""
        graph_file = project_path / "graph.json"
        if not graph_file.exists():
            return cls.empty(project_path)
        return cls.load_from_file(graph_file)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "Graph":
        """Load graph from a specific file (e.g., snapshot)."""
        with file_path.open() as f:
            data = json.load(f)
        return cls.from_dict(data)
```

### Saving

```python
class Graph:
    def save(self, file_path: Path) -> None:
        """Persist graph to a file (atomic write)."""
        temp_file = file_path.with_suffix(".tmp")
        with temp_file.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        temp_file.rename(file_path)
```

### Mutation Pattern

Stages don't modify the graph directly. They produce structured output that
the runtime interprets:

```python
async def run_stage(stage: Stage, graph: Graph, project_path: Path) -> Graph:
    # 1. Snapshot before modifications
    snapshot_path = project_path / "snapshots" / f"pre-{stage.name}.json"
    graph.save(snapshot_path)

    # 2. Execute stage (produces structured output)
    stage_output = await stage.execute(graph)

    # 3. Validate output
    validate_stage_output(stage.name, stage_output)

    # 4. Apply mutations to graph
    apply_mutations(graph, stage.name, stage_output)

    # 5. Validate graph consistency
    validate_graph(graph)

    # 6. Persist
    graph.save(project_path / "graph.json")

    return graph
```

## Snapshot Strategy

### Pre-Stage Snapshots

Before each stage runs:

```
project/
  graph.json              # Current state
  snapshots/
    pre-dream.json        # Before DREAM ran
    pre-brainstorm.json   # Before BRAINSTORM ran
    pre-seed.json         # Before SEED ran
```

### Rollback

```python
def rollback_to_stage(project_path: Path, stage_name: str) -> Graph:
    """Restore graph to pre-stage snapshot."""
    snapshot = project_path / "snapshots" / f"pre-{stage_name}.json"
    if not snapshot.exists():
        raise ValueError(f"No snapshot for stage {stage_name}")

    # Load snapshot
    graph = Graph.load_from_file(snapshot)

    # Save as current graph
    graph.save(project_path / "graph.json")

    return graph
```

### Cleanup Policy

Snapshots are kept indefinitely during development. For production:
- Keep last N stages (configurable, default 3)
- Archive older snapshots on successful SHIP

## Human Review Exports

When users run `qf review`, generate readable views:

### YAML Export

Full graph in YAML format for detailed review:

```yaml
# exports/graph.yaml
version: "5.0"
last_stage: seed

nodes:
  vision:
    type: vision
    genre: noir mystery
    # ...

  detective_001:
    type: entity
    entity_type: character
    concept: "Weary private eye with a dark past"
    # ...
```

### Markdown Summary

High-level overview for quick review:

```markdown
# Noir Mystery - Graph Summary

## Stage: SEED (completed 2026-01-13 10:30)

### Entities (5)
- **detective_001** (character): Weary private eye with a dark past [retained]
- **femme_001** (character): Mysterious client with secrets [retained]
- ...

### Threads (2)
- **mentor_protector_thread**: Trust arc exploring mentor as protector
- **trust_betrayal_thread**: Trust arc exploring betrayal

### Beats (8)
- opening_001: First meeting (scene, opening)
- investigation_001: Search the office (scene)
- ...
```

### Export Command

```bash
qf review              # Generate exports, open for review
qf review --format yaml
qf review --format md
qf export graph.yaml   # Just export, don't open
```

## Validation

### Structural Validation

After each stage, validate:

1. **Node references**: All edge endpoints exist
2. **Type consistency**: Edge types match node types
3. **Required fields**: All required fields present
4. **Stage ownership**: Only expected nodes created/modified

### Graph Consistency

Cross-cutting validation:

1. **Reachability**: All passages reachable from start
2. **No orphans**: No disconnected subgraphs (except pending work)
3. **State coherence**: Codeword references valid

### Validation Timing

| Check | When |
|-------|------|
| Node schema | After stage output parsed |
| Edge validity | After mutations applied |
| Reachability | Before human gate |
| Full consistency | Before SHIP |

## Stage Integration

### Stage Contract

Each stage must:
1. Accept graph as read-only context
2. Produce stage-specific output (validated against schema)
3. Not modify graph directly

### Mutation Application

The runtime maps stage output to graph mutations:

```python
def apply_mutations(graph: Graph, stage: str, output: dict) -> None:
    """Apply stage output as graph mutations."""

    if stage == "dream":
        graph.set_node("vision", output["dream"])

    elif stage == "brainstorm":
        for entity in output["entities"]:
            graph.add_node(entity["id"], {"type": "entity", **entity})
        for dilemma in output["tensions"]:
            graph.add_node(dilemma["id"], {"type": "dilemma", **dilemma})
            for alt in dilemma["alternatives"]:
                alt_id = f"{dilemma['id']}_{alt['id']}"
                graph.add_node(alt_id, {"type": "alternative", **alt})
                graph.add_edge("has_answer", dilemma["id"], alt_id)

    elif stage == "seed":
        # Update entity dispositions
        for entity_decision in output["entities"]:
            graph.update_node(entity_decision["id"], {
                "disposition": entity_decision["disposition"]
            })

        # Create paths from explored tensions
        for path in output["paths"]:
            graph.add_node(path["id"], {"type": "path", **path})
            graph.add_edge("explores", path["id"], path["alternative_id"])

        # Create initial beats
        for beat in output["beats"]:
            graph.add_node(beat["id"], {"type": "beat", **beat})
            for thread_id in beat.get("paths", []):
                graph.add_edge("belongs_to", beat["id"], thread_id)
```

## Performance Considerations

### Memory Usage

For medium-size stories (50-100 passages):
- Graph in memory: ~1-5 MB
- JSON on disk: ~100-500 KB
- Load time: <100ms

### Scaling

If graphs grow larger:
1. Lazy loading of node details
2. Index edges by type for faster queries
3. Consider SQLite for very large projects

**Current decision:** Simple in-memory model is sufficient for v5.0 target scope.

## Future Considerations

### Audit Log

Not implemented in v5.0. Could add:
- Append-only log of all mutations
- Enable fine-grained rollback
- Support multi-user collaboration

### Database Storage

For very large projects or multi-user:
- SQLite for local persistence
- PostgreSQL for collaborative editing

### Incremental Persistence

For long-running stages:
- Checkpoint during stage execution
- Resume from checkpoint on failure

## See Also

- [00-spec.md](../design/00-spec.md) - Graph ontology and design
- `src/questfoundry/models/` - Hand-written Pydantic models implementing the ontology
- [langchain-dream-pipeline.md](./langchain-dream-pipeline.md) - Stage execution
