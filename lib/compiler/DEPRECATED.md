# DEPRECATED

**Status:** This directory is deprecated as of 2025-11-21 (Phase 6 of LangGraph Migration)

## Migration Complete

This spec compiler has been replaced by the new runtime architecture:

**New Location:** `lib/runtime/` - Direct interpretation of YAML definitions

## What Changed

- **Old:** Compiler transformed specs into intermediate representations
- **New:** Runtime directly interprets YAML definitions at execution time

## Migration

The compiler functionality has been replaced by:

- **Schema Registry:** `lib/runtime/src/questfoundry/runtime/core/schema_registry.py`
- **Node Factory:** `lib/runtime/src/questfoundry/runtime/core/node_factory.py`
- **Graph Factory:** `lib/runtime/src/questfoundry/runtime/core/graph_factory.py`

These components load YAML definitions directly from `spec/05-definitions/` and transform them into executable LangGraph StateGraphs at runtime.

## References

- **Migration Plan:** See `MIGRATION.md` for complete migration details
- **New Architecture:** See `spec/06-runtime/ARCHITECTURE.md`

---

**Do not extend this code.** The runtime eliminates the need for a separate compilation step.
