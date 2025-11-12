# Showrunner — Loop Orchestration

**Module:** Core orchestration patterns for executing loop playbooks

## Overview

Showrunner's primary responsibility is executing loop playbooks with appropriate role coordination.
Loop playbooks (in `../loops/`) define WHAT to do; this module defines HOW Showrunner orchestrates
roles to execute them.

## Loop Orchestration

Read TU brief and loop intent; sequence role work per relevant Flow (see `04-protocol/FLOWS/*`).

**Key Principles:**

- Prefer small, testable steps; checkpoint via `tu.checkpoint` after meaningful progress.
- Request gatecheck when a Hot change targets Cold or a handoff requires a bar decision.

**Cross-Reference:** Before opening first TU, verify `project_metadata.json` exists. If not, run
Project Initialization (see `initialization.md`).

## TU Lifecycle

Showrunner manages the complete Technical Unit (TU) lifecycle through four protocol intents:

### `tu.open` — Initialize Session

- Set up role roster, loop name, and references
- Verify Hot manifest exists and Cold reference is set (see `manifest_management.md`)
- Establish TU brief and objectives

### `tu.update` — Adjust Scope/Roles

- Modify TU scope, add/remove roles mid-session
- Record deltas for traceability
- Update TU brief when objectives shift

### `tu.checkpoint` — Persist Progress

- Capture summary of work completed
- Document risks, blockers, and deferrals
- Create recovery point for long-running sessions
- Trigger after meaningful progress or before role dormancy

### `tu.close` — Archive Session

- Archive conversation logs and session state
- Update Hot manifest with TU outcomes
- Capture final deliverables and handoffs

## Human Interaction Proxy

Showrunner mediates all human interactions on behalf of other roles:

**Request Handling:**

- Accept `human.question` from any role
- Present question to human clearly with context
- Forward `human.response` back with `reply_to` and matching `correlation_id`

**Best Practices:**

- Batch questions when possible to avoid chatty cycles
- Provide sufficient context so human can answer without guessing
- Set reasonable timeout expectations for human responses

**Cross-Reference:** See `_shared/human_interaction.md` for detailed patterns.

## Dormancy & Wake

Showrunner controls role activation states to optimize resource usage:

### Role Wake

- Use `role.wake` to activate optional roles when wake rubric is met
- See `01-roles/interfaces/dormancy_signals.md` for wake criteria per role
- Provide context and expectations when waking roles

### Role Dormancy

- Park roles with `role.dormant` after handoff or inactivity
- Summarize role contributions via `tu.checkpoint` before parking
- Keep dormant roles asleep unless activation criteria met

## Safety & PN Boundaries

**PN Safety Invariant:** Never route Hot discovery state to Player Narrator (PN).

### Enforcement Rules

When `receiver.role = PN`, Showrunner must enforce:

1. **Cold only** — Message context must reference Cold manifest snapshot
2. **Player-safe flag** — `player_safe=true` in envelope
3. **Snapshot present** — Valid `snapshot_id` in context

### Content Hygiene

Apply Presentation and Spoiler Hygiene rules to any player-facing surfaces:

- No meta-commentary or authorial notes
- No unrevealed plot branches
- No structural spoilers (section counts, ending counts)

**Cross-Reference:** See `_shared/safety_protocol.md` for complete PN safety rules.

## Shared Patterns

Loop orchestration integrates with cross-role patterns:

- **Context Management** (`_shared/context_management.md`) - Hot/Cold separation, snapshot
  references
- **Safety Protocol** (`_shared/safety_protocol.md`) - PN boundaries, content hygiene
- **Escalation Rules** (`_shared/escalation_rules.md`) - When to request human intervention
- **Human Interaction** (`_shared/human_interaction.md`) - Question batching, timeout handling
