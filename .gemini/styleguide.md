# QuestFoundry Code Review Style Guide

## Project Context

QuestFoundry v5 is a pipeline-driven interactive fiction generation system using LLMs as
collaborators under constraint. It generates branching stories through a six-stage pipeline:
DREAM, BRAINSTORM, SEED, GROW, FILL, SHIP.

## Review Focus Areas

### 1. Design Conformance (CRITICAL for pipeline PRs)

PRs that modify files under `src/questfoundry/pipeline/stages/` or `src/questfoundry/graph/`
MUST include a `## Design Conformance` section in the PR description. If this section is
missing, flag it as a blocking issue.

The design conformance section should reference specific sections from the authoritative
design documents in `docs/design/` and classify each requirement as:

- CONFORMANT: Code implements the requirement (cite file and line)
- PARTIAL: Code exists but diverges from spec
- MISSING: No implementing code found
- DEAD: Code exists but is unreachable (schema exists but nothing populates it)

**DEAD is as bad as MISSING.** A model that the LLM never populates, or a consumer that
never receives data, is not an implementation.

### 2. Removal Discipline

When a PR claims to remove code, verify the removal actually happened:

- The old code must be deleted, not wrapped in a compatibility shim
- Tests must be updated to assert the NEW expected state, not just pass by ignoring removed behavior
- "Tests pass" is NOT sufficient evidence that a removal is complete
- Backward-compatibility layers are only acceptable for external consumers

### 3. Python Standards

- Python 3.11+ with type hints on all public functions
- Google-style docstrings on public functions/methods
- structlog for logging (never print())
- Pydantic models for data boundaries, dataclasses for internal domain
- Imports ordered: stdlib, third-party, local (separated by blank lines)

### 4. Anti-Patterns to Flag

- Agent negotiation between LLM instances
- Backflow (later stages modifying earlier artifacts)
- Hidden prompts in Python code (all prompts must be in `prompts/` directory)
- Unbounded iteration or retry loops
- Backward-compatibility shims during internal refactoring
- Closing removal issues by adding code alongside the thing to be removed
- Silent fallbacks that hide bugs (prefer explicit errors)

### 5. Test Awareness

- Unit tests for individual functions/classes
- Integration tests use real LLM calls (expensive) - flag if added unnecessarily
- Target 70% coverage
- Test fixtures must reflect what the real pipeline produces (not idealized state)

### 6. Prompt Changes

When prompts in `prompts/templates/` or `prompts/components/` are modified:

- Check that the prompt works for small models (4B-8B parameters), not just large ones
- Verify explicit instructions, concrete examples, clear delimiters
- Never suggest "use a larger model" as a fix for prompt quality issues

## Conventional Commits

This project uses conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
Flag commit messages that don't follow this convention.

## PR Size

Target 150-400 lines changed. Flag PRs over 800 lines as too large to review effectively.
