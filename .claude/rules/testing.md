# Testing Guidelines

> **Note**: Runtime is being rebuilt. Tests will be created alongside new implementation.

## E2E Testing Principles

When runtime is implemented, E2E tests will need:

### Long Timeouts

Local models via Ollama are SLOW. Full workflows take 10-20 minutes.

```bash
# Correct
timeout 900 uv run qf ask --project test_proj "create a story"

# WRONG - will timeout mid-workflow
timeout 120 uv run qf ask --project test_proj "create a story"
```

### Checkpoint Support

Design checkpoints to:

- Save state after each SR turn
- Resume from any checkpoint
- Enable incremental testing

### Cold Store Verification

After workflows complete, verify:

- Lifecycle transitions occurred
- Canon contains expected artifacts
- Exclusive writer rules were enforced

## Test Strategy (To Be Designed)

| Level | Purpose | Location |
|-------|---------|----------|
| Unit | Individual components | `tests/unit/` |
| Integration | Component interaction | `tests/integration/` |
| E2E | Full workflows | `tests/e2e/` |

## Before/After Pattern

1. Run tests before changes - capture baseline
2. Make changes
3. Run same tests - compare results
4. If worse, changes caused a regression

## Reference

Old tests archived at `_archive/tests-v3/` for patterns.
