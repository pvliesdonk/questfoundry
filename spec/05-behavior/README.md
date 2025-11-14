# Layer 5 — Behavior Primitives (v2 Architecture)

## Overview

Layer 5 in v2 uses **atomic, composable behavior primitives** instead of pre-assembled prompts. This architecture eliminates duplication and enables single-source-of-truth maintenance.

**Key Principle:** Define once, reference everywhere.

## Directory Structure

```
05-behavior/
  expertises/          # Domain expertise definitions (one per role/domain)
  procedures/          # Reusable workflow procedures with YAML frontmatter
  snippets/            # Small reusable text blocks
  playbooks/           # Loop playbooks (YAML with references)
  adapters/            # Role adapters (YAML with references)
```

## Atomic Primitive Types

### Expertises

**Purpose:** Define core domain knowledge for each role.

**Location:** `expertises/[role]_[domain].md`

**Structure:** Pure Markdown, no frontmatter (referenced by adapters).

**Examples:**

- `lore_weaver_expertise.md` — Canon creation and continuity management
- `gatekeeper_quality_bars.md` — Quality validation expertise
- `scene_smith_prose_craft.md` — Prose writing techniques

### Procedures

**Purpose:** Reusable workflow algorithms and processes.

**Location:** `procedures/[procedure_name].md`

**Structure:** YAML frontmatter + Markdown content.

**Frontmatter Schema:**

```yaml
---
procedure_id: canonization_core
description: Core algorithm for transforming hooks into canon
version: 2.0.0
references_expertises:
  - lore_weaver_expertise
references_schemas:
  - canon_pack.schema.json
  - hook_card.schema.json
references_roles:
  - lore_weaver
  - researcher
tags:
  - canon
  - validation
---
```

**Examples:**

- `canonization_core.md` — Hook-to-canon transformation
- `continuity_check.md` — Contradiction detection
- `player_safe_summarization.md` — Spoiler-free summary generation

### Snippets

**Purpose:** Small, frequently-reused text blocks (1-5 paragraphs).

**Location:** `snippets/[snippet_name].md`

**Structure:** Pure Markdown, no frontmatter (referenced by procedures/playbooks).

**Examples:**

- `spoiler_hygiene_reminder.md` — PN boundary enforcement
- `validation_protocol.md` — Artifact validation rules
- `human_question_format.md` — How to ask clarifying questions

### Playbooks

**Purpose:** Loop definitions with step sequences, RACI, and deliverables.

**Location:** `playbooks/[loop_name].playbook.yaml`

**Structure:** Pure YAML (references procedures, expertises, snippets).

**Schema:**

```yaml
---
playbook_id: lore_deepening
display_name: Lore Deepening
category: discovery
version: 2.0.0
description: Transform accepted hooks into coherent canon

# Cross-references to behavior primitives
references_procedures:
  - canonization_core
  - continuity_check
references_expertises:
  - lore_weaver_expertise
references_snippets:
  - spoiler_hygiene_reminder

# RACI Matrix
raci:
  responsible: [lore_weaver]
  accountable: [showrunner]
  consulted: [researcher, plotwright]
  informed: [codex_curator]

# Step definitions
steps:
  - step_id: frame_questions
    description: Frame canon questions from hooks
    procedure: "@procedure:canonization_core#step1"
    assigned_roles: [lore_weaver]
    artifacts_input: [hook_card]
    artifacts_output: [canon_pack]
    validation_required: true

# Artifacts reference L3 schemas
artifacts_input: [hook_card]
artifacts_output: [canon_pack]

# Quality bars from L0
quality_bars_pressed: [integrity, gateways, presentation]
---
```

### Adapters

**Purpose:** Role interface specifications for multi-role orchestration.

**Location:** `adapters/[role_name].adapter.yaml`

**Structure:** Pure YAML (references expertises, procedures, snippets).

**Schema:**

```yaml
---
adapter_id: lore_weaver
role_name: Lore Weaver
abbreviation: LW
version: 2.0.0

# Primary expertise reference
expertise: "@expertise:lore_weaver_expertise"

# Mission from L1 charter
mission: "Resolve the world's deep truth—quietly—then hand clear, spoiler-safe summaries to neighbors."

# Protocol intents from L4
protocol_intents:
  receives: [hook.accept, tu.open, canon.validate]
  sends: [canon.create, canon.update, merge.request]

# Loop participation
loops:
  - playbook: lore_deepening
    raci: responsible
  - playbook: hook_harvest
    raci: consulted

# Cross-cutting concerns
safety_protocols:
  - "@snippet:spoiler_hygiene_reminder"

# Handoff protocols
handoffs:
  to_codex_curator: "@procedure:player_safe_summarization"
---
```

## Cross-Reference Syntax

References use `@type:id` or `@type:id#section` format:

### Reference Types

| Syntax | Resolves To | Used In |
|--------|-------------|---------|
| `@expertise:lore_weaver_expertise` | `expertises/lore_weaver_expertise.md` | Adapters, Procedures |
| `@procedure:canonization_core` | `procedures/canonization_core.md` | Playbooks, Adapters |
| `@procedure:canonization_core#step1` | Specific section | Playbooks |
| `@snippet:spoiler_hygiene_reminder` | `snippets/spoiler_hygiene_reminder.md` | Playbooks, Procedures |
| `@playbook:lore_deepening` | `playbooks/lore_deepening.playbook.yaml` | Documentation |
| `@adapter:lore_weaver` | `adapters/lore_weaver.adapter.yaml` | Documentation |
| `@schema:canon_pack.schema.json` | `../03-schemas/canon_pack.schema.json` | Procedures, Playbooks |
| `@role:lore_weaver` | `../01-roles/charters/lore_weaver.md` | Procedures, Playbooks |

### Reference Resolution

The spec compiler (Layer 6 build) validates and resolves all references:

1. **Validation:** All references must resolve to existing files
2. **Assembly:** Referenced content is injected during compilation
3. **Manifests:** Runtime JSON manifests contain assembled content
4. **Standalone Prompts:** Full prompts composed from primitives

## File Naming Conventions

### Expertises

- Pattern: `[role]_[domain].md`
- Examples: `lore_weaver_expertise.md`, `gatekeeper_quality_bars.md`

### Procedures

- Pattern: `[action]_[object].md` or `[process_name].md`
- Examples: `canonization_core.md`, `continuity_check.md`

### Snippets

- Pattern: `[concept]_[type].md`
- Examples: `spoiler_hygiene_reminder.md`, `validation_protocol.md`

### Playbooks

- Pattern: `[loop_name].playbook.yaml`
- Examples: `lore_deepening.playbook.yaml`, `hook_harvest.playbook.yaml`

### Adapters

- Pattern: `[role_name].adapter.yaml`
- Examples: `lore_weaver.adapter.yaml`, `gatekeeper.adapter.yaml`

## Compilation Pipeline

```
[Atomic Sources] → [Validator] → [Assembler] → [Manifest Builder] → [Output Writer]
     (YAML/MD)        (refs)       (compose)       (JSON)           (dist/)
```

**Outputs:**

1. `dist/compiled/manifests/*.manifest.json` — Runtime execution manifests
2. `dist/compiled/standalone_prompts/*.md` — Assembled full prompts

## Migration from v1

v1 architecture (`spec/05-prompts/`) used pre-assembled prompts with significant duplication. v2 extracts atomic primitives and uses a compiler to assemble them.

**See:** `/MIGRATION_V1_TO_V2.md` for complete migration instructions.

## Validation

All cross-references are validated by the spec compiler:

- References must resolve to existing files
- Schema references must point to valid L3 schemas
- Role references must match L1 role definitions
- No circular dependencies allowed
- No orphaned primitives (every file must be referenced)

## Dependencies

- **Layer 0:** Quality bars, loops, PN principles
- **Layer 1:** Role charters and missions
- **Layer 2:** Artifact structures and terminology
- **Layer 3:** JSON schemas for validation
- **Layer 4:** Protocol envelopes and intents

## Next Steps

After authoring behavior primitives:

1. Run spec compiler: `qf-compile --spec-dir spec/ --output dist/compiled/`
2. Validate references: `qf-compile --validate-only`
3. Test with runtime: Use `PlaybookExecutor` with compiled manifests

## Related Documentation

- `/MIGRATION_V1_TO_V2.md` — Complete migration guide
- `../00-north-star/WORKING_MODEL.md` — Studio operating model
- `../04-protocol/README.md` — Protocol layer documentation
