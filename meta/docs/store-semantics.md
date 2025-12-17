# Store Semantics

This document describes the store semantics model for multi-agent systems, defining how data flows through different storage tiers.

## The Problem: Collaborative Data Hygiene

In multi-agent systems, agents need to collaborate on shared artifacts. Without clear semantics:

- Work-in-progress gets mistaken for finished work
- Drafts leak to outputs
- Quality gates get bypassed
- Audit trails become impossible

## The Solution: Semantic Storage Tiers

Stores are classified by **semantics** that define their mutability and purpose:

| Semantics | Mutability | Purpose | Example |
|-----------|------------|---------|---------|
| `hot` | Mutable, any agent | Working memory, drafts, collaboration | workspace |
| `cold` | Append-only, exclusive writer | Committed truth, validated artifacts | canon |
| `versioned` | Immutable snapshots | Export bundles, frozen states | exports |
| `ephemeral` | Session-only | Scratch space, temp files | scratch |

---

## Hot Stores (Workspace)

**Characteristics**:

- **Mutable**: Any authorized agent can read/write
- **Discovery space**: Drafts, experiments, work-in-progress
- **No guarantees**: Content may be inconsistent, incomplete, contradictory
- **Short-lived**: Content should not persist indefinitely

**Use cases**:

- Section drafts before validation
- Hook cards for cross-agent coordination
- Research memos
- Style experiments

**Key rule**: Hot stores are never exported directly. Content must flow through validation to cold stores first.

---

## Cold Stores (Canon)

**Characteristics**:

- **Append-only**: New versions add, don't replace
- **Exclusive writer**: One agent controls writes (e.g., Lore Weaver owns canon)
- **Validated truth**: Only content that passed quality gates
- **Permanent**: Content persists as source of truth

**Use cases**:

- Validated story sections
- Canonical lore and world facts
- Approved character definitions
- Final scene prose

**Key rule**: Promote to cold only after validation passes. Never bypass quality gates.

---

## Versioned Stores (Exports)

**Characteristics**:

- **Immutable**: Once created, never modified
- **Snapshot-based**: Each version captures a point-in-time state
- **Traceable**: Every export cites its source snapshot
- **Distributable**: Ready for external consumption

**Use cases**:

- Published story exports (MD, HTML, EPUB, PDF)
- Release bundles
- Archive snapshots

**Key rule**: Versioned stores are built from cold stores, never from hot.

---

## Ephemeral Stores (Scratch)

**Characteristics**:

- **Session-scoped**: Content disappears when session ends
- **No persistence**: Not checkpointed, not recoverable
- **Fast**: No durability overhead

**Use cases**:

- Intermediate calculation results
- Temporary file staging
- Preview renders

**Key rule**: Never put anything important in ephemeral stores.

---

## Lifecycle States

Within stores, artifacts progress through lifecycle states:

```text
draft → review → approved → cold
```

| State | Meaning |
|-------|---------|
| `draft` | Initial creation, work-in-progress |
| `review` | Awaiting validation |
| `approved` | Passed validation, ready for promotion |
| `cold` | Committed to canon (final) |

**Key insight**: "Cold" is a lifecycle state, not just a store semantics. An artifact transitions to `cold` when committed to canon.

---

## Exclusive Writers

Cold stores have **exclusive writers** to maintain consistency:

| Store | Exclusive Writer | Rationale |
|-------|------------------|-----------|
| canon | `lore_weaver` | Single source of world truth |
| codex | `codex_curator` | Player-safe glossary control |
| exports | `book_binder` | Controlled publishing |

Other agents can *read* these stores but must *request* writes through the owner.

---

## The Promotion Pattern

Content flows from hot to cold through validation:

```text
1. Agent creates artifact in Hot (draft)
2. Agent marks artifact for review
3. Validator checks against quality gates
4. On pass: exclusive writer promotes to Cold
5. On fail: feedback returns to creating agent
```

**Implementation**:

```python
# Request promotion (any agent)
request_lifecycle_transition(
    artifact_id="scene_12",
    from_state="approved",
    to_state="cold",
    justification="Passed all 8 quality bars"
)

# Exclusive writer executes promotion
# Runtime verifies caller has write permission
```

---

## The Snapshot Pattern

When exporting from cold stores:

```text
1. Create snapshot of cold store at current state
2. Tag snapshot with identifier (e.g., "v1.0", "2025-10-28")
3. Build export view from snapshot
4. View cites source snapshot for traceability
```

**Benefits**:

- Exports are reproducible
- Multiple exports can target different snapshots
- Audit trail preserved

---

## Anti-Patterns

### Ship from Hot

**Wrong**: Exporting directly from workspace

```text
workspace/scene_12 → export/chapter_1.md
```

**Why it's wrong**: Skips validation. May export incomplete or invalid content.

**Correct**: Always promote through cold first.

### Bypass Validation

**Wrong**: Promoting directly to cold without gatekeeper

```text
draft → cold (skipping review/approved)
```

**Why it's wrong**: Quality gates exist for a reason.

**Correct**: Every promotion must pass validation.

### Cross-write to Cold

**Wrong**: Scene Smith writing directly to canon

**Why it's wrong**: Violates exclusive writer principle. Creates conflicts.

**Correct**: Request promotion via lifecycle transition tool.

---

## Store Configuration

In domain definitions, stores declare their semantics:

```json
{
  "id": "canon",
  "name": "Canon",
  "description": "Curated world truth",
  "semantics": "cold",
  "exclusive_writer": "lore_weaver",
  "lifecycle_states": ["draft", "review", "approved", "cold"]
}
```

```json
{
  "id": "workspace",
  "name": "Workspace",
  "description": "Collaborative working space",
  "semantics": "hot",
  "lifecycle_states": ["draft", "review", "approved"]
}
```

---

## Summary

| Semantics | Key Property | Access Pattern |
|-----------|--------------|----------------|
| `hot` | Mutable, collaborative | Any agent read/write |
| `cold` | Append-only, validated | Exclusive writer only |
| `versioned` | Immutable snapshots | Read-only after creation |
| `ephemeral` | Session-scoped | Temporary, no persistence |

The semantic tier model ensures data hygiene in collaborative agent systems by enforcing clear boundaries between working space, validated truth, and exports.
