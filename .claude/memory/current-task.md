# Current Task Tracker

**Purpose:** Track what you're ACTUALLY working on. Update this when switching tasks.

---

## Active Task

**Task:** Fix LK `promote_to_canon` Regression

**Goal:** LK must call `promote_to_canon` when delegated with "promote" task

**Started:** 2025-12-10

---

## Context

**Bug/Feature being addressed:** LK returns "Ready for promotion" without actually calling `promote_to_canon`

**Root Cause:** PROMPT ENGINEERING REGRESSION - NOT a model capability issue

> **CRITICAL NOTE:** Do NOT blame qwen3:8b or claim it's "not capable enough". Testing has shown GPT-5 and Sonnet 4.5 do not perform better on this task. This is a regression - it worked before. The fix is in prompt engineering and enforcement patterns.

**Files likely to change:**

- `src/questfoundry/domain/roles/lorekeeper.md` - prompt improvements
- `src/questfoundry/runtime/tools/role.py` - enforcement via validate-with-feedback

**Tests to verify:**

- E2E test with cold_store verification
- VCR test for LK role

---

## Progress Log

| Time | Action | Result |
|------|--------|--------|
| 16:46 | Started E2E test | Background Bash 32927d |
| 16:47 | SR → PW | Structure created |
| 16:51 | PW → SS | Prose writing |
| 16:54 | SS → GK | Validation |
| 16:57 | GK passed | gatecheck_report created |
| 16:58 | SR → LK | Task: "Verify and promote 5 validated scenes..." |
| 17:00 | LK → SR | **FAILED: Returned without calling promote_to_canon** |
| 17:01 | Timeout | Test hit 900s limit |

---

## Bug Analysis

### What LK Did

1. Listed hot_store keys (found 13 artifacts)
2. Consulted schema for "scene"
3. Read scene_1, scene_2, scene_3, scene_4
4. Wrote plain text analysis (got "must use tools" error)
5. Called `return_to_sr(status="complete")` (got invalid status error)
6. Retried with `return_to_sr(status="completed", message="Ready for promotion")`
7. **Never called `promote_to_canon`**

### What LK Should Have Done

Per ARCHITECTURE.md 9.4 Validate-with-Feedback Pattern and LK role constraints:

1. Verify artifacts
2. Call `promote_to_canon(artifact_keys=[...], section_id="...")`
3. Then call `return_to_sr` with promotion results

### Missing Enforcement

The 9.4 Validate-with-Feedback pattern should be applied to enforce that LK calls the promotion tool before returning. Options:

1. Add `return_to_sr` validation that checks if promote_to_canon was called
2. Strengthen LK prompt with explicit "MUST call promote_to_canon BEFORE return_to_sr"
3. Add a "did_promote" check in the return validation

---

## Blocked By

- [ ] Nothing - ready to implement fix

---

## Side Issues Found (DO NOT FIX NOW)

| Issue | Severity | File |
|-------|----------|------|
| SS validation error at 16:54:03 (recovered) | Low | executor.py |
| LK used "complete" instead of "completed" | Low | LK prompt |
| Only 4 scenes written (asked for 5) | Low | SS prompt |

---

## Fix Strategy

1. **Apply 9.4 Pattern to LK's return_to_sr**
   - When LK returns with "promote" in original task but no promote_to_canon was called
   - Return validation error: "Task included 'promote' but promote_to_canon was not called"
   - Provide hint: "Call promote_to_canon(artifact_keys=[...]) before returning"

2. **Strengthen LK Prompt**
   - Add explicit workflow: "1. Verify, 2. promote_to_canon, 3. return_to_sr"
   - Add constraint: "MUST NOT return_to_sr until promote_to_canon succeeds"

---

## Completion Criteria

- [ ] LK calls promote_to_canon when delegated with promotion task
- [ ] Cold store has content after E2E test
- [ ] Tests still pass
- [ ] No regressions introduced
