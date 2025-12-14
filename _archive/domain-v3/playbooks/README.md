# Playbooks

Operational procedures for studio management and recovery.

## Purpose

Playbooks define meta-processes for fixing, recovering, or setting up the studio. Unlike Loops (which produce content artifacts), Playbooks manage operations.

## Distinction from Loops

| Aspect | Loops | Playbooks |
|--------|-------|-----------|
| **Purpose** | Create/Edit Story Content | Fix/Setup the Studio |
| **Output** | Content Artifacts (Scene, Lore, Hook) | Operational Changes |
| **Runtime** | SR uses as heuristic map for delegation | SR reads as "Emergency Manual" |
| **Examples** | `story_spark`, `hook_harvest`, `scene_weave` | `gate_failure`, `role_stuck`, `emergency_retcon` |

## Planned Content

- `gate_failure.md` - Recovery when blocked by Gatekeeper
- `emergency_retcon.md` - Safe rewriting of cold canon
- `role_stuck.md` - Agent reset procedure
- `world_genesis.md` - Project setup workflow

## Format

Each playbook file is prose-only (no compiled directives). The content describes:

1. When to invoke the playbook
2. Step-by-step procedure
3. Decision points and escalation paths
4. Success criteria
