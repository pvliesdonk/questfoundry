# E2E Testing Requirements

## Minimum Timeout: 900 Seconds (15 Minutes)

Local models via Ollama are SLOW. A full workflow with multiple role delegations takes 10-20 minutes.

```bash
# Correct
timeout 900 uv run qf ask --project test_proj "create a story"

# WRONG - will timeout mid-workflow
timeout 120 uv run qf ask --project test_proj "create a story"
```

## Use Checkpoints

Checkpoints save progress. Use them to avoid re-running completed steps:

```bash
# Resume from last checkpoint
qf ask --project test_proj --resume "continue"

# Resume from specific checkpoint
qf ask --project test_proj --from-checkpoint 3 "continue"

# List available checkpoints
sqlite3 test_proj/checkpoints.db "SELECT id, sr_turn FROM checkpoints"
```

## Verify Cold Store Writes

After a full workflow, verify Lorekeeper promoted content:

```bash
# Check sections
sqlite3 test_proj/project.qfdb "SELECT COUNT(*) FROM sections"

# Check codex
sqlite3 test_proj/project.qfdb "SELECT COUNT(*) FROM codex"

# Check canon
sqlite3 test_proj/project.qfdb "SELECT COUNT(*) FROM canon"
```

If counts are 0, the workflow didn't reach Lorekeeper or promote_to_canon wasn't called.

## Before/After Pattern

1. Run tests before changes - capture baseline
2. Make changes
3. Run same tests - compare results
4. If worse, your changes caused a regression
