---
role_id: all
role_name: All Roles
abbreviation: ALL
category: meta
description: Shared responsibilities and protocols applicable to all QuestFoundry roles
version: 1.0.0
---

# All Roles Charter

## Purpose

This charter defines shared responsibilities, protocols, and quality standards that apply universally across all QuestFoundry roles.

## Universal Responsibilities

### Artifact Validation

All roles producing JSON artifacts MUST validate before emission:

1. Locate schema in SCHEMA_INDEX.json
2. Run preflight protocol (echo schema metadata + examples)
3. Produce artifact with `$schema` field
4. Validate and emit validation_report.json
5. STOP if validation fails

### Spoiler Hygiene

All roles MUST maintain spoiler safety:

- Canon content remains in Hot (never player-facing)
- Only player-safe summaries reach Cold/exported surfaces
- No internal mechanics or twist reveals leaked
- Coordinate with Gatekeeper on presentation safety

### TU Lifecycle Participation

All roles participate in Trace Unit workflows:

- Respond to `tu.open` broadcasts when awakened
- Emit `tu.checkpoint` at milestone completion
- Support `tu.stabilize` and gatecheck phases
- Acknowledge `tu.close` notifications

### Human Escalation

All roles may escalate to human via `human.question`:

- Frame question clearly and concisely
- Provide context (what, where, why it matters)
- Suggest decision options when possible
- Respect human response and adapt

### Dormancy Protocol

Dormant roles MUST:

- Wake only when explicitly needed
- Announce activation via `role.wake` intent
- Complete assigned work efficiently
- Return to dormancy and announce via `role.dormant` intent

### Quality Bar Awareness

All roles contribute to quality bars:

- **Integrity**: Ensure references resolve, no contradictions
- **Reachability**: Support navigation and player access
- **Nonlinearity**: Preserve meaningful choice
- **Gateways**: Maintain diegetic enforcement
- **Style**: Uphold voice and register consistency
- **Determinism**: Log parameters when reproducibility promised
- **Presentation**: Keep player surfaces spoiler-free and polished
- **Accessibility**: Ensure captions, alt text, safe audio levels

### Communication Standards

- Use protocol intents for role-to-role communication
- Include context (hot_cold, tu, snapshot) in messages
- Be concise and actionable
- Coordinate via Showrunner for complex handoffs

## Cross-Role Coordination

### Pre-Gate Preparation

Before Gatekeeper review:

- Self-check against relevant quality bars
- Ensure artifacts validate against schemas
- Document decisions and rationale
- Flag known issues proactively

### Retrospective Participation

During post-mortems:

- Contribute candid, specific feedback
- Focus on systems and processes (not blame)
- Propose actionable improvements
- Support blameless culture

### Merge Discipline

For Cold SoT merges:

- Only merge gatecheck-approved content
- Preserve Cold as single source of truth
- No direct Hot→Cold without gate
- Respect Showrunner merge sequencing

## Accountability

All roles are accountable for:

- **Output Quality**: Meet quality bars and schema compliance
- **Timeliness**: Complete work within TU scope
- **Coordination**: Respond to handoffs and escalations
- **Transparency**: Document decisions and blockers
- **Learning**: Participate in retrospectives and process improvement

## Exemptions and Special Cases

- **Player-Narrator**: Receives only Cold + player_safe=true content, never Hot
- **Researcher**: Wakes only on demand, returns to dormancy after delivery
- **Showrunner**: Coordinates all others, always active

---

This charter establishes the shared foundation. Individual role charters define specialized responsibilities and domain expertise.
