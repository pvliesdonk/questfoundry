# DEPRECATED

**Status:** This directory is deprecated as of 2025-11-21 (Phase 6 of LangGraph Migration)

## Migration Complete

This imperative Python runtime has been replaced by the new declarative architecture:

**New Location:** `lib/runtime/`

**Architecture:** LangGraph-based runtime that interprets YAML role profiles and loop patterns from `spec/05-definitions/`

## What Changed

- **Old:** Imperative Python code with hardcoded behavior
- **New:** Declarative YAML definitions with spec-driven runtime

## Migration

All functionality has been migrated to:
- Runtime implementation: `lib/runtime/`
- Role definitions: `spec/05-definitions/roles/`
- Loop definitions: `spec/05-definitions/loops/`
- Quality gates: `spec/05-definitions/quality_gates/`
- Lifecycle transitions: `spec/05-definitions/transitions/`

## References

- **Migration Plan:** See `MIGRATION.md` for complete migration details
- **New Architecture:** See `spec/05-definitions/README.md`
- **Runtime Specs:** See `spec/06-runtime/ARCHITECTURE.md`

---

**Do not extend this code.** Use `lib/runtime/` for all new development.
