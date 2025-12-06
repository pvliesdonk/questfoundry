# Layer 5 Rebuild Plan - Structured Knowledge for Agents

This note sketches how to turn Layer 5 (spec/05-definitions) into a machine-readable, regenerable
projection of Layers 0-4, so that agents in lib/runtime can consume the spec as structured input
instead of ad-hoc markdown.

Problem we are solving:

- Agents need precise knowledge of loops, lifecycles, artifacts, and protocol.
- That knowledge already exists in Layers 0-4 (playbooks, charters, taxonomies, schemas, protocol).
- Layer 5 is currently a hand-maintained reimagining of those layers, and consult_* tools mostly
  return long markdown blobs.
- This makes it hard to inject structured knowledge and to derive guardrails directly from the
  spec.

High level direction:

- Treat Layers 0-4 as canonical inputs:
  - L0: 00-north-star (playbooks, loops, quality bars).
  - L1: 01-roles (role charters).
  - L2: 02-dictionary (artifacts, taxonomies, glossary).
  - L3: 03-schemas (JSON Schemas for artifacts and state).
  - L4: 04-protocol (envelope, intents, lifecycles, flows, examples).
- Rebuild Layer 5 (roles/*.yaml, loops/*.yaml, protocol.yaml, quality_gates/*.yaml) as a
  projection of those inputs using a large-context model (e.g. Gemini) that keeps the whole spec
  in context.
- Keep two kinds of structure in Layer 5:
  1) Normative state machines (TU lifecycle, envelope rules, schemas) that can drive strict guards.
  2) Advisory state machines from playbooks/topology (recommended node order, who to wake, etc.)
     that can drive soft guidance.

Strict vs guidance:

- TU lifecycles, envelope schema, and JSON Schemas are law. Violations should be hard errors
  (e.g. invalid state transition, missing required fields, unsafe PN envelope).
- Playbooks are guidance, not law. They describe typical high-signal paths through a loop but
  runs can deviate as long as lifecycle and safety rules are respected. Guards derived from
  playbooks should be warnings or hints, not hard failures.

Integration points (runtime side):

- RuntimeContextAssembler can surface Layer 5 structure in prompts and explicitly tell agents
  which consult_* tools to use before key protocol intents (e.g. consult_playbook + consult_schema
  before tu.open).
- BindToolsExecutor can track consult_* usage and enforce strict guards for lifecycle-critical
  intents (e.g. block tu.open if the loop playbook was never consulted), while using playbook
  structure only for advisory nudges.

This file is intentionally high level. A Gemini-based generator can expand it into a more detailed
spec for roles/loops/protocol once the full spec is loaded into context.
