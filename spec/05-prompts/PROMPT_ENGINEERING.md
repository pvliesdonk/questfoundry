# Prompt Engineering Guide

Audience: GPT-5 (primary) and authors of role prompts

Goals

- Make prompts predictable, modular, and portable across LLMs
- Keep invariants (safety, protocol) explicit and testable

Standard Structure (per role)

- Goals: what outcomes the role optimizes for
- Invariants: safety, scope, and style guardrails
- Procedures/Handlers: intent-specific steps and constraints
- Examples: envelope-level few-shots (match 04-protocol/envelope.schema.json)

Few-shot Guidance

- Prefer minimal but complete envelopes; include `context`, `safety`, `correlation_id`.
- Show both positive and negative cases when applicable.
- Keep examples normative: treat them as standards, not suggestions.

LLM Adaptation

- Keep role instructions model-agnostic; isolate model-specific toggles (e.g., JSON mode,
  temperature) in the harness.
- Avoid relying on hidden system behavior; encode constraints in the prompt and schema.

Adding Intents or Behaviors

- Update 04-protocol/INTENTS.md first; add examples in 04-protocol/EXAMPLES.
- Extend the role’s procedures and examples; add tests in 05-prompts/tests.
