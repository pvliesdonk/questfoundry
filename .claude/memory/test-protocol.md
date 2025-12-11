# E2E Test Protocol

**Purpose:** Checklist for running end-to-end workflow tests. Follow this exactly.

---

## Pre-Test Checklist

- [ ] **Clean project directory** (if fresh test): `rm -rf project_name`
- [ ] **Baseline tests pass**: `uv run pytest -x`
- [ ] **Document what you're testing**: Write in `current-task.md`
- [ ] **Set appropriate timeout**: 900s minimum for full workflow

---

## Test Commands

### Fresh Workflow Test

```bash
rm -rf project_test && timeout 900 uv run qf ask -vvv --log \
  --project project_test --provider ollama \
  "write a 1-act story with 1 chapter and 1 scene. keep it simple." 2>&1
```

### Resume from Checkpoint

```bash
# Find latest run
sqlite3 project_test/checkpoints.db "SELECT DISTINCT run_id FROM checkpoints"

# Resume
timeout 900 uv run qf ask -vvv --log \
  --project project_test --provider ollama \
  --resume "continue" 2>&1
```

### Resume from Specific Checkpoint

```bash
# List checkpoints
sqlite3 project_test/checkpoints.db "SELECT id, sr_turn, role FROM checkpoints ORDER BY id"

# Resume from checkpoint N
timeout 900 uv run qf ask -vvv --log \
  --project project_test --provider ollama \
  --from-checkpoint N "continue" 2>&1
```

---

## Verification Checklist

### Workflow Completion

- [ ] SR delegated to PW (Plotwright)
- [ ] PW created scene structure in hot_store
- [ ] SS (Scene Smith) added prose content
- [ ] GK (Gatekeeper) validated content
- [ ] **LK (Lorekeeper) promoted to cold_store** ← Most common failure point

### Cold Store Verification

```bash
# Check database exists and has content
sqlite3 project_test/project.qfdb "SELECT COUNT(*) FROM sections"

# List all tables
sqlite3 project_test/project.qfdb ".tables"

# Show section content
sqlite3 project_test/project.qfdb "SELECT id, title FROM sections"
```

### Log Analysis

```bash
# Check which roles ran
grep "role_session_start" project_test/logs/llm.jsonl | jq -r '.role'

# Check for errors
grep -i "error" project_test/logs/llm.jsonl | head -20
```

---

## Expected Timeline (Local Ollama)

| Role | Typical Duration |
|------|------------------|
| SR (planning) | 30-60 seconds |
| PW (structure) | 1-3 minutes |
| SS (prose) | 2-5 minutes |
| GK (validation) | 1-2 minutes |
| LK (canonization) | 30-60 seconds |

**Total:** 5-12 minutes typical, allow 15 minutes buffer

---

## Troubleshooting

### Workflow Stuck

1. Check last checkpoint: `sqlite3 project/checkpoints.db "SELECT * FROM checkpoints ORDER BY id DESC LIMIT 1"`
2. Resume from checkpoint instead of restarting

### Cold Store Empty

1. Verify GK passed (check logs for `gatecheck_report`)
2. Verify LK was delegated to
3. Check `promote_to_canon` was called in logs

### Timeout

1. Increase to 900s or more
2. Use checkpoint resume instead of full restart
3. Consider simpler prompt for faster iteration

---

## Anti-Patterns

- **DON'T** abort after 5 minutes - workflows take longer
- **DON'T** restart from scratch - use checkpoints
- **DON'T** expect cold_store writes before LK runs
- **DON'T** change code while test is running
