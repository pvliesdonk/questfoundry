# Lifecycle: Artifact States (Hot → Cold)

> **Status:** Normative
> **Version:** 3.0.0

This document defines the **artifact lifecycle**: how artifacts move from creation through stabilization to canon.

---

## Overview

All artifacts in QuestFoundry follow a consistent lifecycle:

```
draft → proposed → in_progress → review → approved → canonized
                                   ↓
                              rejected (terminal)
```

The lifecycle enforces:

1. **Quality gates** — artifacts pass bars before canonization
2. **Clear ownership** — who can transition states
3. **Traceability** — all transitions logged
4. **Safety** — canonical artifacts are immutable

---

## State Machine

:::{lifecycle-states}
id: artifact
states:

- draft: "Initial creation in hot_store"
- proposed: "Ready for work assignment"
- in_progress: "Active work underway"
- review: "Awaiting gatecheck"
- approved: "Passed gatecheck, ready for cold"
- canonized: "In cold_store (terminal)"
- rejected: "Will not pursue (terminal)"
:::

### State Descriptions

| State | Store | Mutability | Description |
|-------|-------|------------|-------------|
| `draft` | hot | mutable | Just created, incomplete |
| `proposed` | hot | mutable | Ready for role assignment |
| `in_progress` | hot | mutable | Role actively working |
| `review` | hot | read-only | Awaiting gatecheck |
| `approved` | hot | read-only | Gatecheck passed |
| `canonized` | cold | immutable | Final, in canon |
| `rejected` | hot | frozen | Will not pursue |

---

## Transitions

:::{lifecycle-transitions}
id: artifact
transitions:

- from: draft
    to: proposed
    trigger: artifact.propose
    sender: any_role

- from: proposed
    to: in_progress
    trigger: artifact.start
    sender: owner_role

- from: in_progress
    to: review
    trigger: artifact.submit
    sender: owner_role

- from: review
    to: approved
    trigger: gate.pass
    sender: gatekeeper

- from: review
    to: in_progress
    trigger: gate.rework
    sender: gatekeeper

- from: approved
    to: canonized
    trigger: artifact.canonize
    sender: showrunner

- from: any_non_terminal
    to: rejected
    trigger: artifact.reject
    sender: showrunner
:::

### Transition Matrix

| From | To | Intent | Sender | Notes |
|------|----|--------|--------|-------|
| `draft` | `proposed` | `artifact.propose` | Any | Ready for assignment |
| `proposed` | `in_progress` | `artifact.start` | Owner | Work begins |
| `in_progress` | `review` | `artifact.submit` | Owner | Request gatecheck |
| `review` | `approved` | `gate.pass` | GK | All bars green |
| `review` | `in_progress` | `gate.rework` | GK | Bars failed |
| `approved` | `canonized` | `artifact.canonize` | SR | Move to cold |
| `*` | `rejected` | `artifact.reject` | SR | Terminal |

---

## Store Semantics

### Hot Store (`hot_store`)

- **Purpose:** Work-in-progress artifacts
- **Mutability:** Read/write
- **Lifecycle:** draft → approved
- **Access:** All roles can read; owner can write

### Cold Store (`cold_store`)

- **Purpose:** Canonical artifacts
- **Mutability:** Append-only
- **Lifecycle:** canonized only
- **Access:** All roles can read; only SR can write (via canonize)

### Promotion Path

```
hot_store (draft → approved)
         │
         ▼ request_gatecheck
    Gatekeeper evaluates
         │
         ▼ gate.pass
    approved in hot_store
         │
         ▼ merge_to_cold (SR only)
cold_store (canonized)
```

---

## Quality Gates

### Pre-Gatecheck (Optional)

Informal review before formal gatecheck:

- Role requests feedback
- GK provides informal assessment
- No state change required

### Formal Gatecheck

Required before canonization:

:::{gatecheck-requirements}
bars:

- integrity: "No contradictions"
- reachability: "All content accessible"
- nonlinearity: "Multiple paths exist"
- gateways: "Valid unlock conditions"
- style: "Voice consistency"
- determinism: "Reproducible outputs"
- presentation: "Format correct"
- accessibility: "All players can access"

decision:
  pass: "All bars green → approved"
  conditional: "Some yellow → approved with notes"
  block: "Any red → back to in_progress"
:::

---

## Role Authorization

### Who Can Transition

| Transition | Authorized Roles |
|------------|------------------|
| draft → proposed | Any role (typically creator) |
| proposed → in_progress | Assigned owner |
| in_progress → review | Owner |
| review → approved | Gatekeeper only |
| review → in_progress | Gatekeeper only |
| approved → canonized | Showrunner only |
| * → rejected | Showrunner only |

### Ownership Assignment

When artifact enters `proposed`:

- SR assigns owner via `proposed_next_step.owner_r`
- Owner has exclusive write access during `in_progress`

---

## Error Conditions

:::{lifecycle-errors}
invalid_transition:
  code: INVALID_STATE_TRANSITION
  example: "Cannot go from draft to canonized directly"

not_authorized:
  code: NOT_AUTHORIZED
  example: "Scene Smith cannot canonize artifacts"

validation_failed:
  code: VALIDATION_FAILED
  example: "Artifact missing required fields"

gatecheck_required:
  code: GATECHECK_REQUIRED
  example: "Cannot canonize without passing gatecheck"
:::

---

## Examples

### Example: Happy Path

```
1. PW creates Brief artifact → draft
2. SR proposes for work → proposed
3. SS assigned, starts work → in_progress
4. SS completes, submits → review
5. GK evaluates, all green → approved
6. SR canonizes → canonized (in cold_store)
```

### Example: Rework Cycle

```
1. SS submits artifact → review
2. GK finds Integrity red → in_progress (rework)
3. SS fixes issues, resubmits → review
4. GK evaluates, all green → approved
5. SR canonizes → canonized
```

### Example: Rejection

```
1. PW creates Branch artifact → draft
2. SR reviews, decides not needed → rejected (terminal)
```

---

## Cross-References

- `domain/protocol/intents.md` — Intent definitions
- `domain/ontology/artifacts.md` — Artifact types
- `runtime/state.py` — StudioState implementation
