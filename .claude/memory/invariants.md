# QuestFoundry v3 Invariants

**Purpose:** Critical facts that must survive context compaction. Read this FIRST when resuming work.

---

## Architecture Invariants

### Source of Truth

- **Domain files** (`src/questfoundry/domain/`) are the source of truth
- **Generated files** (`src/questfoundry/generated/`) are AUTO-GENERATED
- **NEVER edit generated/ directly** - edit domain/ and run `qf compile`

### Storage Model (Three-Tier)

```
hot_store (drafts) → cold_store (canon) → Views (filtered exports)
```

- **hot_store**: Working memory, mutable, dies with session
- **cold_store**: SQLite + files, append-only, survives sessions
- **Views**: Publisher filters cold_store by `visibility` at export time

### Who Writes Where

| Store | Writers | Readers |
|-------|---------|---------|
| hot_store | All roles | All roles |
| cold_store | **ONLY Lorekeeper (LK)** | All roles |
| Views | Publisher (PB) | Players |

**CRITICAL:** Only LK calls `promote_to_canon()`. If testing cold_store writes, you MUST let the workflow reach LK.

### Workflow Chain

```
User Request → SR → PW (structure) → SS (prose) → GK (validate) → LK (canonize) → PB (export)
```

Gatekeeper validates BEFORE Lorekeeper canonizes.

---

## Testing Invariants

### E2E Test Timeouts

- **Minimum timeout:** 900 seconds (15 minutes) for full workflow
- **Ollama local models are SLOW** - expect 2-5 minutes per role delegation
- **Use checkpoints:** `--resume` or `--from-checkpoint N` to skip completed steps

### Test Verification

- **Before changing code:** Run tests, capture baseline
- **After changing code:** Run same tests, compare
- **Cold store test:** Must verify LK ran and wrote to `project.qfdb`

### Checkpoint Recovery

```bash
# List checkpoints for a project
sqlite3 project_name/checkpoints.db "SELECT id, sr_turn, run_id FROM checkpoints"

# Resume from checkpoint
qf ask --project name --from-checkpoint 3 "continue"
```

---

## Development Workflow

### Before Making Changes

1. Read ARCHITECTURE.md sections relevant to your change
2. Run `uv run pytest` to establish baseline
3. Document what's working NOW

### Making Changes

1. Edit source in `domain/` or `compiler/` or `runtime/`
2. If domain change: run `qf compile`
3. Run tests

### After Making Changes

1. Verify no regressions: `uv run pytest`
2. For runtime changes: run e2e test with 15-min timeout
3. Git add and commit atomically

---

## Common Mistakes to Avoid

1. **Editing generated/** - Always edit domain/ and compile
2. **Short timeouts** - E2E tests need 15+ minutes with local models
3. **Forgetting LK writes cold** - Only Lorekeeper promotes to canon
4. **Fixing small bugs mid-task** - Finish current task FIRST, track bugs separately
5. **Not using checkpoints** - Use `--from-checkpoint` to skip completed work
6. **Blaming the model** - See below
7. **Calling regressions "separate bugs"** - See below

---

## Model Capability Bias (DO NOT DO)

> **CRITICAL:** When LLM agents don't follow instructions, the problem is PROMPT ENGINEERING, not model capability.

**Facts:**

- qwen3:8b via Ollama works for QuestFoundry tasks
- GPT-5 and Sonnet 4.5 do NOT perform better on these tasks
- When roles fail to call required tools, it's a **regression** that worked before

**When a role doesn't follow instructions:**

1. Check if ARCHITECTURE.md 9.4 Validate-with-Feedback is applied
2. Strengthen the prompt with explicit enforcement
3. Add validation that returns structured errors
4. DO NOT suggest "use a more capable model"

---

## Key Architecture Patterns

### 9.4 Validate-with-Feedback Pattern

When tool validation fails, return structured feedback and let the LLM retry:

```json
{
  "success": false,
  "error": "Task included 'promote' but promote_to_canon was not called",
  "hint": "Call promote_to_canon(artifact_keys=[...]) before returning"
}
```

This pattern should be applied to ALL stop tools (return_to_sr, terminate) to enforce required behaviors before allowing the role to return.

---

## Regression Accountability (CRITICAL)

> **RULE:** If something was working and now it's broken, and you touched the code, YOU introduced a regression. Do not call it a "separate bug."

**When something breaks after your changes:**

1. **Accept responsibility** - You are the only one who touched the code
2. **Investigate your changes** - Use `git diff` to see what you modified
3. **Trace data flow** - If data was in hot_store and now it's gone, your changes broke something
4. **Do not dismiss** - Never say "that's a separate issue" when you're the cause

**Why this matters:**

- Context compaction loses details of what you changed
- Future sessions inherit your regressions without knowing
- Dismissing as "separate" means the regression never gets fixed

**Investigation steps when data disappears:**

```bash
# What files did you change?
git diff HEAD~5 -- src/questfoundry/runtime/

# Check state tools
grep -n "state\[" src/questfoundry/runtime/tools/*.py

# Check if state is being passed correctly
grep -n "state_dict" src/questfoundry/runtime/roles.py
```
