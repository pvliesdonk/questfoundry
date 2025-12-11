# DEPRECATED

**Status:** This directory is deprecated as of 2025-11-21 (Phase 6 of LangGraph Migration)

## Migration Complete

This Layer 5 behavior specification has been replaced by the new "Cartridge" architecture:

**New Location:** `spec/05-definitions/`

## What Changed

- **Old:** `spec/05-behavior/` - Behavior primitives (adapters, expertises, playbooks, procedures, snippets)
- **New:** `spec/05-definitions/` - Executable definitions (roles, loops, quality gates, transitions)

## Architecture Evolution

The behavior layer was reorganized following the "Cartridge Pivot" (ADR-002):

**Layer 5 (NEW):**

- **Roles:** `spec/05-definitions/roles/` - 16 role profiles with full YAML definitions
- **Loops:** `spec/05-definitions/loops/` - 10 loop patterns as state machines
- **Quality Gates:** `spec/05-definitions/quality_gates/` - 8 reusable quality bar validators
- **Transitions:** `spec/05-definitions/transitions/` - 4 lifecycle state machines (hook, tu, gate, view)
- **Templates:** `spec/05-definitions/templates/` - Reusable templates

**Layer 5 (OLD - THIS DIRECTORY):**

- Adapters, expertises, playbooks, procedures, snippets
- These primitives have been consolidated into role profiles and loop patterns

## Migration Mapping

| Old Concept | New Location | Notes |
|-------------|--------------|-------|
| Adapters | Role `behavior.model_config` | LLM configuration in role YAML |
| Expertises | Role `behavior.system_prompt` | Role-specific instructions |
| Playbooks | Loop `nodes` array | Orchestration as state machines |
| Procedures | Loop `edges` conditions | Conditional routing logic |
| Snippets | Role `behavior.template_variables` | Template fragments |

## References

- **Migration Plan:** See `MIGRATION.md` for complete migration details
- **New Architecture:** See `spec/05-definitions/README.md`
- **ADR-002:** Cartridge architecture decision in `MIGRATION.md`

---

**Do not extend this directory.** All new behavior definitions go in `spec/05-definitions/`.
