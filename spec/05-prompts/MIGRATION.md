# Migration Guide

Scope

- Versioning of prompts and schemas
- Intent renames and deprecations
- Safe rollout sequencing

Versioning

- Use SemVer for protocol/schema changes. Prompt text changes that do not affect wire format are
  patch-level.
- Keep migration notes per release; link to changed intents and examples.

Intent Renames

- tu.start → tu.open
- gate.submit → gate.report.submit
- Ensure all examples and role procedures reflect the new intent names.

Rollout

- Update INTENTS and examples first (Layer 4), then role prompts (Layer 5).
- Add/adjust tests in 05-prompts/tests; run validator locally and in CI.
