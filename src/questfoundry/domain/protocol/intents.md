# Protocol: Intent Catalog

> **Status:** Normative
> **Version:** 3.0.0

This document defines the **message intents** for QuestFoundry v3's SR-centric orchestration model.

---

## Overview

In v3, the Showrunner (SR) is the sole orchestrator. Intents define:

1. **What SR delegates** to roles (task specification)
2. **What roles return** to SR (result status)
3. **What SR decides** based on results (routing logic)

Unlike v2's peer-to-peer protocol, v3 uses **handoff-based delegation** where all communication flows through SR.

---

## Intent Naming Convention

Format: `<domain>.<verb>[.<subverb>]`

| Domain | Purpose |
|--------|---------|
| `task` | SR→Role delegation |
| `result` | Role→SR completion |
| `artifact` | Artifact lifecycle |
| `gate` | Quality validation |
| `query` | Information retrieval |
| `error` | Error conditions |

---

## Delegation Intents (SR → Role)

:::{intent-type}
id: task.delegate
description: "SR delegates work to a specialist role"
sender: showrunner
receiver: any_role
fields:

- task: string  # Task description
- context: object  # Relevant artifacts and state
- constraints: list[string]  # Requirements/boundaries
- expected_output: string  # What SR expects back
:::

:::{intent-type}
id: task.consult
description: "SR requests read-only consultation from a role"
sender: showrunner
receiver: any_role
fields:

- question: string  # What SR needs to know
- artifacts: list[string]  # Relevant artifact IDs
- scope: string  # Domain of consultation
:::

:::{intent-type}
id: task.evaluate
description: "SR requests quality evaluation from Gatekeeper"
sender: showrunner
receiver: gatekeeper
fields:

- artifact_ids: list[string]  # Artifacts to evaluate
- bars: list[string]  # Quality bars to check
- mode: string  # "pre-gate" | "full-gate"
:::

---

## Result Intents (Role → SR)

:::{intent-type}
id: result.completed
description: "Role completed delegated work successfully"
sender: any_role
receiver: showrunner
fields:

- status: string  # "completed"
- artifacts_created: list[string]  # New artifact IDs
- artifacts_modified: list[string]  # Modified artifact IDs
- summary: string  # Brief description of work done
- recommendation: string  # Suggested next action
:::

:::{intent-type}
id: result.blocked
description: "Role cannot proceed without additional input"
sender: any_role
receiver: showrunner
fields:

- status: string  # "blocked"
- blocker: string  # What is blocking
- need: string  # What role needs to proceed
- from_role: string  # Which role can provide it
:::

:::{intent-type}
id: result.needs_review
description: "Role completed work but recommends review"
sender: any_role
receiver: showrunner
fields:

- status: string  # "needs_review"
- artifacts: list[string]  # Artifacts needing review
- concern: string  # What should be reviewed
- recommendation: string  # Suggested reviewer or action
:::

:::{intent-type}
id: result.escalate
description: "Role escalates decision to SR"
sender: any_role
receiver: showrunner
fields:

- status: string  # "escalate"
- decision_needed: string  # What SR must decide
- options: list[string]  # Available choices
- recommendation: string  # Role's recommendation
:::

---

## Artifact Intents

:::{intent-type}
id: artifact.create
description: "Create new artifact in hot_store"
sender: any_role
receiver: showrunner
fields:

- artifact_type: string  # Type from ontology
- artifact_id: string  # Generated ID
- data: object  # Artifact content
- store: string  # "hot" (default) | "cold"
:::

:::{intent-type}
id: artifact.update
description: "Update existing artifact in hot_store"
sender: any_role
receiver: showrunner
fields:

- artifact_id: string  # Existing ID
- updates: object  # Fields to update
- reason: string  # Why updating
:::

:::{intent-type}
id: artifact.promote
description: "Request promotion from hot_store to cold_store"
sender: showrunner
receiver: gatekeeper
fields:

- artifact_ids: list[string]  # Artifacts to promote
- gatecheck_required: bool  # Whether GK approval needed
:::

:::{intent-type}
id: artifact.canonize
description: "Finalize artifact in cold_store (SR only)"
sender: showrunner
receiver: system
fields:

- artifact_ids: list[string]  # Artifacts to canonize
- snapshot_id: string  # Cold snapshot reference
:::

---

## Gate Intents (Gatekeeper)

:::{intent-type}
id: gate.report
description: "Gatekeeper submits evaluation report"
sender: gatekeeper
receiver: showrunner
fields:

- artifact_ids: list[string]  # Evaluated artifacts
- decision: string  # "pass" | "conditional" | "block"
- bars: object  # Per-bar status (green/yellow/red)
- issues: list[object]  # Specific issues found
- remediation: list[object]  # Suggested fixes
:::

:::{intent-type}
id: gate.pass
description: "Gatekeeper approves artifacts for Cold"
sender: gatekeeper
receiver: showrunner
fields:

- artifact_ids: list[string]  # Approved artifacts
- bars_status: object  # All bars green
:::

:::{intent-type}
id: gate.block
description: "Gatekeeper blocks artifacts from Cold"
sender: gatekeeper
receiver: showrunner
fields:

- artifact_ids: list[string]  # Blocked artifacts
- violations: list[object]  # Quality bar violations
- required_fixes: list[object]  # What must be fixed
:::

---

## Query Intents

:::{intent-type}
id: query.artifact
description: "Query artifact from store"
sender: any_role
receiver: system
fields:

- artifact_id: string  # ID to retrieve
- store: string  # "hot" | "cold" | "both"
:::

:::{intent-type}
id: query.lore
description: "Query canon for facts (Lorekeeper)"
sender: lorekeeper
receiver: system
fields:

- question: string  # What to look up
- scope: string  # Domain constraint
- include_hot: bool  # Include draft content
:::

:::{intent-type}
id: query.structure
description: "Query story structure (Plotwright)"
sender: plotwright
receiver: system
fields:

- query_type: string  # "topology" | "branches" | "reachability"
- artifact_ids: list[string]  # Relevant artifacts
:::

---

## Error Intents

:::{intent-type}
id: error.validation
description: "Payload validation failed"
sender: system
receiver: any_role
fields:

- code: string  # "validation_error"
- message: string  # Human-readable error
- schema: string  # Schema that failed
- violations: list[string]  # Specific violations
:::

:::{intent-type}
id: error.authorization
description: "Role not authorized for action"
sender: system
receiver: any_role
fields:

- code: string  # "not_authorized"
- message: string  # Why not authorized
- required_role: string  # Who can do this
:::

:::{intent-type}
id: error.conflict
description: "State conflict prevents action"
sender: system
receiver: any_role
fields:

- code: string  # "conflict"
- message: string  # What conflicted
- current_state: string  # Current artifact state
- attempted_action: string  # What was attempted
:::

:::{intent-type}
id: error.not_found
description: "Requested entity not found"
sender: system
receiver: any_role
fields:

- code: string  # "not_found"
- entity_type: string  # "artifact" | "role" | "loop"
- entity_id: string  # What was not found
:::

---

## Intent Summary

| Intent | Direction | Purpose |
|--------|-----------|---------|
| `task.delegate` | SR → Role | Delegate work |
| `task.consult` | SR → Role | Request consultation |
| `task.evaluate` | SR → GK | Request gatecheck |
| `result.completed` | Role → SR | Work done |
| `result.blocked` | Role → SR | Cannot proceed |
| `result.needs_review` | Role → SR | Needs attention |
| `result.escalate` | Role → SR | Decision needed |
| `artifact.create` | Role → System | Create artifact |
| `artifact.update` | Role → System | Update artifact |
| `artifact.promote` | SR → GK | Request Cold promotion |
| `artifact.canonize` | SR → System | Finalize in Cold |
| `gate.report` | GK → SR | Evaluation results |
| `gate.pass` | GK → SR | Approve for Cold |
| `gate.block` | GK → SR | Block from Cold |
| `query.*` | Role → System | Information retrieval |
| `error.*` | System → Role | Error conditions |

---

## Cross-References

- `domain/roles/*.md` — Role definitions and constraints
- `domain/protocol/delegation.md` — Delegation model details
- `domain/protocol/lifecycles/*.md` — State machines
- `domain/ontology/artifacts.md` — Artifact types
