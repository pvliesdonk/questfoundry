# Shared Pattern — Escalation Rules

Target: GPT-5 (primary)

Purpose

- Provide crisp criteria for escalation vs. autonomous continuation
- Standardize how roles request help or additional activation

References

- 01-roles/interfaces/escalation_rules.md
- 01-roles/interfaces/dormancy_signals.md
- 04-protocol/INTENTS.md

Primary Triggers

- Policy uncertainty or cross-bar conflict → escalate to GK.
- Missing critical input → send `human.question` to SR.
- Specialist lane needed (e.g., Art, Audio, Translation) → ask SR to `role.wake` that role.

Levels (L1–L3)

- L1: Clarification needed (single question, no artifact block). Prefer `human.question` with
  options.
- L2: Artifact risk (bar could slip to yellow/red). Notify SR; suggest waking a specialist.
- L3: Blocker (cannot proceed). Request GK review; include a `tu.checkpoint` summary and proposed
  next steps.

Signals & Rhythm

- SR controls activation via `role.wake` / `role.dormant`.
- After escalation, emit brief updates via `tu.checkpoint` until unblocked.
