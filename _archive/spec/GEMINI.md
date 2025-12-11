# Agent Guidelines (Specification: spec/)

These rules are **strict** for all work in `spec/` (Layers 0–5 and runtime interface docs in
`spec/06-runtime`). Follow this file plus `spec/CONTRIBUTING.md` before making changes.

## Scope & Boundaries

- `spec/` is the canonical source of truth. Do not modify implementation code here or bundle runtime
  artifacts manually.
- Respect layers: L0 principles/policies, L1 roles, L2 artifact templates/taxonomies, L3 schemas,
  L4 protocol, L5 executable definitions, L6 interface docs. Keep content in the correct layer.
- Layer 2 (human-readable templates) outranks Layer 3 schemas when conflicts arise; fix schemas to
  match Layer 2, then validate.
- Maintain Hot/Cold separation; never leak spoilers into Cold/player-facing outputs.

## Required Practices

- Read relevant files for context before editing (layer README + nearby docs).
- Keep terminology consistent with `02-dictionary/` (glossary, taxonomies) and update cross-links
  when terms change.
- Validate structured files: JSON Schema draft 2020-12, YAML against meta-schemas in
  `03-schemas/definitions/`, and protocol examples where applicable.
- Preserve traceability: update references across layers (docs, schemas, definitions, templates).
- Avoid new high-maintenance artifacts; prefer existing structures and automation.

## Commit Conventions (spec)

- Use custom types to trigger spec workflows:
  - `docs(spec):` documentation and principles (L0–L2)
  - `schema(spec):` schemas/protocol (L3–L4) — triggers spec version bump
  - `prompt(spec):` executable definitions/templates (L5) — triggers spec version bump
- Mark breaking changes with `!` and include migration notes.

## Definition of Done (spec)

- Changes live in the correct layer and respect Layer 2 → Layer 3 derivation.
- Cross-references and examples are updated; Hot/Cold boundaries respected.
- JSON/YAML validated; formatting consistent; schema/meta-schema checks pass.
- Commit message uses the correct `docs/spec`/`schema/spec`/`prompt/spec` type.
