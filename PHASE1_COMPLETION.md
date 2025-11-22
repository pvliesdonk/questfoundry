# Phase 1: Critical Runtime Fixes - COMPLETED ✅

**Date:** 2025-11-22
**Scope:** Fix critical bugs preventing basic story_spark loop execution

---

## Summary

All Phase 1 critical fixes have been **successfully implemented and verified**:

1. ✅ Customer directive hallucination fixed
2. ✅ Hot/Cold Sources of Truth architecture implemented
3. ✅ Artifact flow between sequential nodes working
4. ✅ Exit conditions verified correct

The `story_spark` loop now executes successfully end-to-end with proper artifact propagation.

---

## Detailed Fixes

### 1. Customer Directive Hallucination ✅

**Problem:**
Showrunner consistently ignored user's "detective noir" request and hallucinated "foreman's backstory" instead.

**Root Cause:**
Three-way variable name mismatch:

- `showrunner.py:231,279` set: `context["scene_text"]`
- `node_factory.py:252` looked for: `customer_request`
- Showrunner template received: `""` (empty string)

**Fix:**

```python
# lib/runtime/src/questfoundry/runtime/cli/showrunner.py
# Lines 231, 279
context["customer_request"] = customer_message  # Was: context["scene_text"]
```

**Evidence:**
Showrunner now outputs: `"reasoning": "Customer wants to create a small story in detective noir style..."`

---

### 2. Hot/Cold Sources of Truth Architecture ✅

**Problem:**
Artifacts passing between roles was inconsistent and not following spec architecture.

**Decision:**
Implemented full hot_sot/cold_sot structure immediately rather than partial fix per user request: *"passing artifacts between all agents would go through hot sot... So maybe do it immediately"*

**Changes:**

#### A. State Model (`state.py:57-61`)

```python
# Added hot_sot and cold_sot with reducers
hot_sot: Annotated[dict[str, Any], operator.or_]  # Hot: drafts, proposals, hooks, WIP
cold_sot: Annotated[dict[str, Any], operator.or_]  # Cold: stable canon, export-safe
```

#### B. State Initialization (`state_manager.py:91-114`)

```python
# Initialize Hot/Cold Sources of Truth
hot_sot = {
    "tus": [],
    "hooks": [],
    "topology_notes": None,
    "section_briefs": [],
    "draft_sections": [],
    # ... more fields
}

cold_sot = {
    "current_snapshot": None,
    "snapshots": [],
    "canon": {},
    "codex": {},
    "manuscript": {},
}
```

#### C. Artifact Extraction (`node_factory.py:277-368`)

Rewrote `extract_artifacts()` to map LLM JSON outputs to hot_sot keys:

```python
def extract_artifacts(self, role: RoleProfile, llm_output: str, tu_id: str) -> dict[str, Any]:
    """Map LLM output fields to hot_sot keys based on role's outputs spec."""
    output_data = json.loads(llm_output)
    hot_sot_updates = {}

    # Map based on role YAML outputs specification
    for output_spec in role.raw.get("interface", {}).get("outputs", []):
        artifact_type = output_spec.get("artifact_type")
        state_key = output_spec.get("state_key", "")
        # ... mapping logic ...

    return {
        "hot_sot": hot_sot_updates,
        "artifacts": {f"_{role.id}_raw_output": output_data},
    }
```

#### D. Template Context (`node_factory.py:226-264`)

Updated to provide hot_sot variables to templates:

```python
context = {
    "hot_sot": hot_sot,
    "cold_sot": cold_sot,
    # Template-expected variables
    "section_briefs": hot_sot.get("section_briefs", []),
    "style_addendum": hot_sot.get("style_notes"),
    # ...
}
```

**Future Enhancement:**
User noted: *"Eventually hot should be backed by memory/redis and cold should probably be backed by file/sqlite. but that is maybe for later"*

---

### 3. Artifact Flow Between Sequential Nodes ✅

**Problem:**
Scene Smith reported `"No section briefs provided"` despite Plotwright outputting `"briefs_written": [...]` moments earlier.

**Root Cause:**
State reducer wasn't merging hot_sot updates. The node return statement at `node_factory.py:598` incorrectly nested the extracted artifacts:

```python
# WRONG - nests hot_sot inside artifacts
return {"artifacts": extracted_artifacts, "messages": [message]}
# Results in: {"artifacts": {"hot_sot": {...}, "artifacts": {...}}}
```

**Fix:**

```python
# lib/runtime/src/questfoundry/runtime/core/node_factory.py
# Lines 598-602
return {
    **extracted_artifacts,  # Unpacks hot_sot and artifacts to top level
    "messages": [message],
}
# Results in: {"hot_sot": {...}, "artifacts": {...}, "messages": [...]}
```

This allows the `Annotated[dict, operator.or_]` reducer to properly merge hot_sot across nodes.

**Evidence:**
Scene Smith (sequential node) now outputs:

```json
{
  "sections_drafted": ["1", "2", "3"],
  "choices_total": 6,
  "contrastive_check": true,
  "diegetic_gates": true
}
```

---

### 4. Exit Conditions ✅

**Status:** Verified correct
**Location:** `graph_factory.py:280-350`

Graph correctly evaluates both:

- Primary: `state.tu_lifecycle == 'completed'`
- Fallback: `state.iteration_count >= 2` (valid per spec)

No changes needed.

---

## Test Results

### Before Fixes ❌

```
Showrunner: [hallucinated "foreman's backstory"]
Plotwright: "briefs_written": ["section_id_1", "section_id_2"]
Scene Smith: "notes": "No section briefs provided. Cannot proceed."
```

### After Fixes ✅

```
Showrunner: "reasoning": "Customer wants to create a small story in detective noir style..."
Plotwright: "briefs_written": ["story_spark_1", "story_spark_2", "story_spark_3"]
Scene Smith: "sections_drafted": ["1", "2", "3"], "choices_total": 6
```

Full loop execution: **SUCCESS** ✅

---

## New Issue Identified

**Ollama Context Window Truncation:**

```
ai-ollama level=WARN msg="truncating input prompt"
limit=4096 prompt=4199 keep=4 new=4096
```

**Impact:** Prompts exceeding 4096 tokens being truncated
**Status:** Needs investigation and potential mitigation (context pruning, model selection, chunking)
**Action:** User requested to create separate issue

---

## Phase 2 Preview

**Medium Priority:**

- Template variable verification across all roles
- Quality bar propagation testing
- TU lifecycle transition validation
- Message protocol flow verification

**Long-term:**

- Integration test suite for multi-agent loops
- Storage backend plugins (hot: Redis, cold: SQLite)
- Showrunner LLM integration (replace deterministic mapping)
- Protocol Router implementation

---

## References

- Hot/Cold SoT spec: `spec/00-north-star/SOURCES_OF_TRUTH.md`
- Loop definition: `spec/05-definitions/loops/story_spark.yaml`
- Role profiles: `spec/05-definitions/roles/*.yaml`
- State model: `lib/runtime/src/questfoundry/runtime/models/state.py`
