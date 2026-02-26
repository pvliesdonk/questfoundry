# Graph Referential Integrity - Implementation Plan

**Issue**: #147
**Status**: Implemented (PR #149, #150)
**Author**: Claude + Peter

## Overview

Refactor the graph to enforce referential integrity at the data layer, similar to foreign keys in relational databases. This eliminates scattered validation and catches invalid references at the point of mutation.

## Current Problems

1. **Silent creation**: `set_node()` creates nodes implicitly - allows drift/invention
2. **Dangling edges**: `add_edge()` doesn't validate endpoints exist
3. **Scattered validation**: Checks spread across `mutations.py`, `serialize.py`, `orchestrator.py`
4. **Late feedback**: LLM only learns valid IDs after failing validation
5. **No post-mutation checks**: Graph can end up in inconsistent state

## Design Principles

1. **Explicit intent**: Creating vs updating a node are different operations
2. **Fail fast**: Invalid references rejected immediately with actionable feedback
3. **Closed world**: You can only reference nodes that exist
4. **Semantic errors**: Error messages explain WHAT's wrong, WHY, and HOW to fix

---

## Phase 1: Graph API Changes (~150 lines)

### 1.1 New Node Operations

**File**: `src/questfoundry/graph/graph.py`

```python
class Graph:
    def create_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Create a new node. Raises NodeExistsError if already exists."""
        if node_id in self._data["nodes"]:
            raise NodeExistsError(node_id)
        self._data["nodes"][node_id] = data

    def update_node(self, node_id: str, **updates: Any) -> None:
        """Update existing node fields. Raises NodeNotFoundError if doesn't exist."""
        if node_id not in self._data["nodes"]:
            raise NodeNotFoundError(
                node_id,
                available=self._get_nodes_by_prefix(node_id.split("::")[0]),
                context="update_node"
            )
        self._data["nodes"][node_id].update(updates)

    def upsert_node(self, node_id: str, data: dict[str, Any]) -> bool:
        """Create or update node. Returns True if created, False if updated.

        Use sparingly - prefer explicit create/update for clarity.
        """
        created = node_id not in self._data["nodes"]
        self._data["nodes"][node_id] = data
        return created

    def delete_node(self, node_id: str, cascade: bool = False) -> None:
        """Delete node. Raises NodeReferencedError if edges reference it.

        Args:
            node_id: Node to delete
            cascade: If True, also delete referencing edges
        """
        if node_id not in self._data["nodes"]:
            raise NodeNotFoundError(node_id)

        refs = self._find_edges_referencing(node_id)
        if refs and not cascade:
            raise NodeReferencedError(node_id, referenced_by=refs)

        if cascade:
            self._data["edges"] = [
                e for e in self._data["edges"]
                if e.get("from") != node_id and e.get("to") != node_id
            ]

        del self._data["nodes"][node_id]
```

### 1.2 Validated Edge Operations

```python
    def add_edge(
        self,
        edge_type: str,
        from_id: str,
        to_id: str,
        **props: Any
    ) -> None:
        """Add edge with endpoint validation.

        Raises:
            NodeNotFoundError: If from_id or to_id doesn't exist
        """
        if from_id not in self._data["nodes"]:
            raise NodeNotFoundError(
                from_id,
                available=self._get_nodes_by_type(self._infer_type(from_id)),
                context=f"edge '{edge_type}' source"
            )
        if to_id not in self._data["nodes"]:
            raise NodeNotFoundError(
                to_id,
                available=self._get_nodes_by_type(self._infer_type(to_id)),
                context=f"edge '{edge_type}' target"
            )

        edge = {"type": edge_type, "from": from_id, "to": to_id, **props}
        self._data["edges"].append(edge)

    def _infer_type(self, node_id: str) -> str | None:
        """Infer node type from ID prefix (e.g., 'entity::kay' -> 'entity')."""
        if "::" in node_id:
            return node_id.split("::")[0]
        return None
```

### 1.3 Reference Helper

```python
    def ref(self, node_type: str, raw_id: str) -> str:
        """Get validated node reference. Raises if doesn't exist.

        Usage:
            path_ref = graph.ref("path", "trust_path")
            graph.add_edge("belongs_to", beat_ref, path_ref)
        """
        node_id = f"{node_type}::{raw_id}"
        if node_id not in self._data["nodes"]:
            raise NodeNotFoundError(
                node_id,
                available=self._get_nodes_by_type(node_type),
                context=f"reference to {node_type}"
            )
        return node_id
```

---

## Phase 2: Error Types with LLM Feedback (~100 lines)

**File**: `src/questfoundry/graph/errors.py`

```python
from dataclasses import dataclass, field
from difflib import get_close_matches


class GraphIntegrityError(Exception):
    """Base class for graph integrity violations."""

    def to_llm_feedback(self) -> str:
        """Format error as actionable feedback for LLM retry."""
        raise NotImplementedError


@dataclass
class NodeNotFoundError(GraphIntegrityError):
    """Raised when referencing a non-existent node."""

    node_id: str
    available: list[str] = field(default_factory=list)
    context: str = ""

    def __post_init__(self):
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"Node '{self.node_id}' not found"
        if self.context:
            msg += f" ({self.context})"
        return msg

    def _get_suggestions(self) -> list[str]:
        """Find similar IDs that might be typos."""
        # Extract raw_id from prefixed ID (e.g., "entity::kay" -> "kay")
        raw_id = self.node_id.split("::")[-1] if "::" in self.node_id else self.node_id
        raw_available = [a.split("::")[-1] for a in self.available]
        matches = get_close_matches(raw_id, raw_available, n=3, cutoff=0.6)

        # Reconstruct full IDs with prefix
        prefix = self.node_id.split("::")[0] if "::" in self.node_id else ""
        if prefix:
            return [f"{prefix}::{m}" for m in matches]
        return matches

    def to_llm_feedback(self) -> str:
        """Format as actionable LLM feedback."""
        suggestions = self._get_suggestions()

        lines = [
            "## Reference Error: Node Not Found",
            "",
            f"**You referenced**: `{self.node_id}`",
        ]

        if self.context:
            lines.append(f"**Context**: {self.context}")

        lines.extend([
            "",
            "**Problem**: This node does not exist in the graph.",
            ""
        ])

        if suggestions:
            lines.append("**Did you mean one of these?**")
            for s in suggestions:
                lines.append(f"  - `{s}`")
            lines.append("")

        if self.available:
            lines.append("**Valid IDs** (use exactly one of these):")
            # Show up to 20 IDs, sorted for consistency
            for a in sorted(self.available)[:20]:
                lines.append(f"  - `{a}`")
            if len(self.available) > 20:
                lines.append(f"  - ... and {len(self.available) - 20} more")

        return "\n".join(lines)


@dataclass
class NodeExistsError(GraphIntegrityError):
    """Raised when creating a node that already exists."""

    node_id: str

    def __post_init__(self):
        super().__init__(f"Node '{self.node_id}' already exists")

    def to_llm_feedback(self) -> str:
        return f"""## Error: Node Already Exists

**You tried to create**: `{self.node_id}`
**Problem**: A node with this ID already exists.

**Solutions**:
1. Use a different ID if this is a new node
2. Use `update_node()` if you want to modify the existing node
"""


@dataclass
class NodeReferencedError(GraphIntegrityError):
    """Raised when deleting a node that is still referenced."""

    node_id: str
    referenced_by: list[dict] = field(default_factory=list)

    def __post_init__(self):
        super().__init__(f"Node '{self.node_id}' is still referenced by {len(self.referenced_by)} edges")

    def to_llm_feedback(self) -> str:
        lines = [
            "## Error: Cannot Delete Referenced Node",
            "",
            f"**Node**: `{self.node_id}`",
            f"**Problem**: This node is referenced by {len(self.referenced_by)} edge(s).",
            "",
            "**Referenced by**:"
        ]
        for ref in self.referenced_by[:5]:
            lines.append(f"  - {ref.get('type')} from `{ref.get('from')}`")

        lines.extend([
            "",
            "**Solutions**:",
            "1. Delete the referencing edges first",
            "2. Use `cascade=True` to delete node and all references"
        ])

        return "\n".join(lines)
```

---

## Phase 3: Update Mutation Appliers (~200 lines)

**File**: `src/questfoundry/graph/mutations.py`

Replace `set_node()` calls with explicit `create_node()` or `update_node()`:

### 3.1 DREAM Mutations

```python
def apply_dream_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply DREAM stage output to graph."""
    # Vision node - create (first stage, always new)
    graph.create_node("vision", {
        "type": "vision",
        "genre": output.get("genre"),
        "subgenre": output.get("subgenre"),
        # ... other fields
    })
```

### 3.2 BRAINSTORM Mutations

```python
def apply_brainstorm_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply BRAINSTORM stage output to graph."""
    for entity in output.get("entities", []):
        raw_id = entity["entity_id"]
        node_id = f"entity::{raw_id}"

        # Create - BRAINSTORM creates new entities
        graph.create_node(node_id, {
            "type": "entity",
            "raw_id": raw_id,
            "entity_category": entity["entity_category"],
            "concept": entity["concept"],
            "notes": entity.get("notes"),
        })

    for dilemma in output.get("dilemmas", []):
        raw_id = dilemma["dilemma_id"]
        dilemma_node_id = f"dilemma::{raw_id}"

        graph.create_node(dilemma_node_id, {
            "type": "dilemma",
            "raw_id": raw_id,
            "question": dilemma["question"],
            # ...
        })

        for answer in dilemma.get("answers", []):
            answer_id = f"dilemma::{raw_id}::alt::{answer['answer_id']}"
            graph.create_node(answer_id, {
                "type": "answer",
                "raw_id": answer["answer_id"],
                # ...
            })

            # Edge endpoints validated automatically
            graph.add_edge("has_answer", dilemma_node_id, answer_id)
```

### 3.3 SEED Mutations

```python
def apply_seed_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply SEED stage output to graph."""
    # Update existing entities with disposition
    for decision in output.get("entities", []):
        raw_id = decision["entity_id"]
        node_id = f"entity::{raw_id}"

        # Update - entities already exist from BRAINSTORM
        # This will raise NodeNotFoundError if phantom ID
        graph.update_node(node_id, disposition=decision["disposition"])

    # Create new paths
    for path in output.get("paths", []):
        path_id = path["path_id"]

        graph.create_node(path_id, {
            "type": "path",
            "raw_id": path_id.split("::", 1)[-1],
            "dilemma_id": path["dilemma_id"],
            "answer_id": path["answer_id"],
            # ...
        })

        # Edge endpoint validation happens automatically in add_edge()
        answer_ref = f"{path['dilemma_id']}::alt::{path['answer_id']}"
        graph.add_edge("explores", path_id, answer_ref)  # Validates answer exists
```

---

## Phase 4: Valid IDs Context for Serialize (~80 lines)

**File**: `src/questfoundry/graph/context.py`

```python
def format_valid_ids_context(graph: Graph, stage: str) -> str:
    """Format valid IDs as context for LLM serialization.

    Provides the authoritative list of IDs the LLM must use.
    """
    if stage == "seed":
        return _format_seed_valid_ids(graph)
    elif stage == "grow":
        return _format_grow_valid_ids(graph)
    return ""


def _format_seed_valid_ids(graph: Graph) -> str:
    """Format BRAINSTORM IDs for SEED serialization."""
    lines = [
        "## VALID IDS - USE EXACTLY THESE",
        "",
        "You MUST use these exact IDs. Any other ID will be rejected.",
        ""
    ]

    # Group entities by category
    entities = graph.get_nodes_by_type("entity")
    by_category: dict[str, list[str]] = {}
    for node in entities.values():
        cat = node.get("entity_category", "unknown")
        raw_id = node.get("raw_id", "")
        by_category.setdefault(cat, []).append(raw_id)

    lines.append("### Entity IDs (for entity_id, entities, location fields)")
    for category in ["character", "location", "object", "faction"]:
        if category in by_category:
            lines.append("")
            lines.append(f"**{category.title()}s:**")
            for raw_id in sorted(by_category[category]):
                lines.append(f"  - `{raw_id}`")

    # Dilemmas with alternatives
    lines.append("")
    lines.append("### Dilemma IDs with their Answer IDs")
    lines.append("Format: dilemma_id → [answer_ids]")
    lines.append("")

    dilemmas = graph.get_nodes_by_type("dilemma")
    for did, ddata in sorted(dilemmas.items()):
        raw_id = ddata.get("raw_id")
        answers = []
        for edge in graph.get_edges(from_id=did, edge_type="has_answer"):
            answer_node = graph.get_node(edge.get("to", ""))
            if answer_node:
                answer_id = answer_node.get("raw_id")
                default = " (default)" if answer_node.get("is_canonical") else ""
                answers.append(f"`{answer_id}`{default}")

        lines.append(f"- `{raw_id}` → [{', '.join(answers)}]")

    lines.extend([
        "",
        "### Rules",
        "- Every entity above needs a decision (retained/cut)",
        "- Every dilemma above needs a decision (explored/unexplored answers)",
        "- Path answer_id must be from that dilemma's answers list",
    ])

    return "\n".join(lines)
```

### 4.2 Inject into Serialize Phase

**File**: `src/questfoundry/agents/serialize.py`

```python
async def serialize_seed_iteratively(
    model: BaseChatModel,
    brief: str,
    graph: Graph,  # Already have this
    # ...
) -> SeedOutput:
    # NEW: Get valid IDs context
    valid_ids_context = format_valid_ids_context(graph, stage="seed")

    # Prepend to brief so LLM sees it
    brief_with_ids = f"{valid_ids_context}\n\n---\n\n{brief}"

    # ... rest of serialization
```

---

## Phase 5: Cleanup and Integration (~50 lines)

### 5.1 Remove Duplicate Validation

**File**: `src/questfoundry/pipeline/orchestrator.py`

Remove the duplicate Pydantic validation call - it's already done in serialize:

```python
# REMOVE this block (lines ~284-291):
# validation_errors = self._validator.validate(artifact_data, stage_name)
# if validation_errors:
#     log.warning(...)
```

### 5.2 Add Post-Mutation Invariant Check

```python
async def run_stage(self, ...):
    # ... stage execution ...

    try:
        apply_mutations(graph, stage_name, artifact_data)
    except GraphIntegrityError as e:
        # Integrity error during mutation - this is LLM's fault
        # Retry with feedback
        feedback = e.to_llm_feedback()
        return await self._retry_with_feedback(stage, feedback)

    # Post-mutation check - catches code bugs, not LLM errors
    invariant_violations = validate_graph_invariants(graph)
    if invariant_violations:
        graph.rollback_to_snapshot(stage_name)  # Use stage_name as snapshot ID
        raise GraphCorruptionError(invariant_violations)

    graph.save(...)
```

---

## Migration Strategy

### Backward Compatibility

Keep `set_node()` as deprecated alias for `upsert_node()`:

```python
def set_node(self, node_id: str, data: dict[str, Any]) -> None:
    """Deprecated: Use create_node() or update_node() instead."""
    warnings.warn(
        "set_node() is deprecated. Use create_node() or update_node().",
        DeprecationWarning
    )
    self.upsert_node(node_id, data)
```

### Test Updates

All tests using `set_node()` need updating to use explicit operations.

---

## Implementation Order

| PR | Scope | Est. Lines | Dependencies |
|----|-------|------------|--------------|
| PR 1 | Graph API + Error types | ~250 | None |
| PR 2 | Update DREAM/BRAINSTORM mutations | ~100 | PR 1 |
| PR 3 | Update SEED mutations + valid IDs context | ~150 | PR 1, PR 2 |
| PR 4 | Orchestrator cleanup + post-mutation checks | ~80 | PR 1-3 |
| PR 5 | Deprecate set_node(), update tests | ~100 | PR 1-4 |

Total: ~680 lines across 5 PRs

---

## Verification

After implementation:

```bash
# All tests pass
uv run pytest

# Run pipeline - should succeed without validation retries
uv run qf run --to seed projects/test-fk "A mystery story"

# Check no integrity errors
grep -i "integrity\|not found\|not exist" projects/test-fk/logs/debug.jsonl
```

## Open Questions (Resolved)

1. **Cascade behavior**: `delete_node()` defaults to `cascade=False` (safer - requires explicit opt-in for destructive operations). Implemented in PR #149.

2. **Soft references**: Not implemented in initial version. Forward references will be addressed if needed for GROW phase. For now, strict integrity is enforced.

3. **Transaction semantics**: Mutations are NOT all-or-nothing in current implementation. Partial application is possible. The snapshot system can provide manual rollback if needed. Full transaction support deferred to future work if required.

---

## Implementation Notes

### PR #149 (Graph API + Error Types)

Implemented Phase 1 and Phase 2:
- Added `create_node()`, `update_node(**kwargs)`, `upsert_node()`, `delete_node(cascade=)`
- Added `ref()` helper with validation for double-prefixed IDs
- Added `add_edge()` endpoint validation with `EdgeEndpointError`
- Created error types in `src/questfoundry/graph/errors.py`
- Deprecated `add_node()` and `set_node()` with warnings

### PR #150 (Mutation Handlers)

Updated mutation handlers to use new API:
- DREAM: `set_node()` → `upsert_node()` (allows re-running stage)
- BRAINSTORM: `add_node()` → `create_node()` (entities, dilemmas, answers)
- SEED: `add_node()` → `create_node()` (paths, consequences, beats)

### Design Decisions Made During Implementation

1. **Valid IDs context** (Phase 4): Will be injected as a separate section in the prompt template rather than prepending to the user's creative brief. This keeps the brief focused on creativity.

2. **Retry logic**: Semantic validation occurs in the serialize phase before mutations. If `GraphIntegrityError` is raised during mutation, it indicates a bug in validation (not LLM fault) and is logged as error.

3. **Helper methods**: Implemented `_infer_type_from_id()`, `_get_node_ids_by_type()`, and `_find_edges_referencing()` as private methods in `Graph` class.

4. **Method naming**: Kept `add_node()` as deprecated alias pointing to `create_node()`. New code should use explicit `create_node()` or `update_node()`.
