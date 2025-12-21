# Implementation Plan: Issue #228 - Semantic Ambiguity Audit

## Overview

This plan addresses semantic ambiguity in `meta/`, `domain-v4/`, and `src/questfoundry/runtime/` that causes LLM agents to misinterpret tool results, ignore feedback, and behave inconsistently.

## Background

The issue documents three case studies where semantic ambiguity caused agent failures:

1. **PR #220**: `status` conflated task_completion, result_assessment, and action_recommendation
2. **PR #227**: `success` and `hint` in save_artifact were too vague
3. **#242**: Tool descriptions contain role-specific biasing language

## Phase 1: Canonical Vocabulary Specification

**Deliverable**: `meta/docs/semantic-conventions.md` or `domain-v4/knowledge/must_know/semantic_conventions.json`

### Key Semantic Axes to Define

| Axis | Reserved Name | Scope | Values |
|------|--------------|-------|--------|
| Execution outcome | `action_outcome` | Did the operation apply? | `saved`, `rejected`, `deferred`, `queued` |
| Quality assessment | `validation_result` or `overall_assessment` | Is the data acceptable? | `pass`, `fail`, `warning`, `needs_revision` |
| Next-action | `recommendation` (orchestrator) / `recovery_action` (specific) | What to do next | `proceed`, `rework`, `escalate`, `hold` |
| Lifecycle state | `lifecycle_state` | Artifact state | `draft`, `review`, `approved`, `cold` |
| Transition result | `transition_result` | Lifecycle transition outcome | `committed`, `rejected`, `deferred` |
| Transport outcome | `success` (bool) + `error` | Tool invocation status | true/false + error string |

### Content Field Naming Convention

| Role | Preferred Name | Description |
|------|---------------|-------------|
| Player-facing narrative | `prose` | Canon-eligible story text |
| Internal reasoning | `notes` or `internal_notes` | Not player-visible |
| Generic text | `text` | Unspecified role |
| Container object | `content` | Wrapper, not a synonym for prose |

---

## Phase 2: Systematic Audit

### 2.1 Critical Issues (P0 - Immediate)

| Item | Location | Issue | Fix |
|------|----------|-------|-----|
| web_search mismatch | `domain-v4/tools/web_search.json` vs `runtime/tools/web_search.py` | Definition promises `domain_filter`, `recency`; runtime expects `categories` | Align both to same interface |
| consult_knowledge layer | `domain-v4/tools/consult_knowledge.json` | `layer` is open string, should be enum from `layers.json` | Add enum constraint or reference |

### 2.2 High-Impact Renames (P1)

| Item | Location | Current | Proposed |
|------|----------|---------|----------|
| validate_artifact status | `domain-v4/tools/validate_artifact.json` | `green/yellow/red` | `pass/warn/fail` |
| list_artifact_types hint | `domain-v4/tools/list_artifact_types.json` | `hint` | `recommended_action` |
| progress_update status | `meta/schemas/core/message.schema.json` | `status` with `completing` | Remove `completing`, use `in_progress` |
| recency "any" | `domain-v4/tools/web_search.json` | `any` | `all_time` |

### 2.3 Tool Description Neutralization (P1)

Remove role-specific biasing language from tool descriptions:

| Tool | Current Language | Issue |
|------|-----------------|-------|
| `delegate.json` | "The Showrunner uses this extensively" | Steers tool selection |
| `communicate.json` | "This is the ONLY way orchestrators communicate" | Over-emphatic |

**Rule**: Descriptions should explain *what* a tool does, not *who* uses it or *when*.

### 2.4 Artifact Schema Consistency (P2)

Audit content field usage:

| Artifact | Current Field | Should Be |
|----------|--------------|-----------|
| `section.json` | `prose` | `prose` (correct) |
| `codex_entry.json` | `full_entry` | Consider `prose` for consistency |
| Knowledge entries | `content.text` | `content.text` (container pattern, OK) |

Audit lifecycle state naming:

| Artifact | States | Notes |
|----------|--------|-------|
| `section` | draft, review, gatecheck, approved, cold | Standard |
| `codex_entry` | draft, review, validated, published | `published` vs `cold`? |
| `section_brief` | draft, ready, in_use, archived | Different model |

### 2.5 Message Schema Review (P2)

| Field | Location | Issue |
|-------|----------|-------|
| `conditional_pass` | `feedback_payload` | Ambiguous without conditions |
| `nudge` | message type | Passive - should be directive |

---

## Phase 3: Implementation

### Step 1: Create Semantic Conventions Document

Create `meta/docs/semantic-conventions.md` with:

- Canonical vocabulary definitions
- Examples showing correct vs incorrect usage
- Decision tree for choosing field names

### Step 2: Fix Critical P0 Issues

**PR 1: Sync web_search interface**

- Files: `domain-v4/tools/web_search.json`, `src/questfoundry/runtime/tools/web_search.py`
- Align parameters and return values

**PR 2: Constrain consult_knowledge layer**

- Files: `domain-v4/tools/consult_knowledge.json`
- Add enum or reference to valid layers

### Step 3: High-Impact Renames (P1)

**PR 3: validate_artifact semantic clarity**

- Change `green/yellow/red` → `pass/warn/fail`
- Separate execution status from assessment
- Files: `domain-v4/tools/validate_artifact.json`, `src/questfoundry/runtime/tools/validate_artifact.py`

**PR 4: General renames across domain**

- `hint` → `recommended_action`
- `any` → `all_time`
- Remove `completing` from progress_update

**PR 5: Neutralize tool descriptions**

- Remove role-specific language from all tool descriptions
- Add workflow biasing only where structurally needed

### Step 4: Update Knowledge Base (P1)

Update `domain-v4/knowledge/must_know/` entries to:

- Reference canonical field names
- Tell agents which fields to prioritize
- Include examples of correct interpretation

### Step 5: Artifact Schema Alignment (P2)

- Audit all artifact types for content field consistency
- Document decisions for legitimate variations
- Align lifecycle state terminology where possible

### Step 6: Add Enforcement (P3)

Create validation check (`qf validate --semantic`) that:

- Flags use of banned/discouraged field names
- Checks for canonical fields in key schemas
- Reports new terms that may need review

---

## Work Breakdown

| PR | Description | Files | Priority |
|----|-------------|-------|----------|
| 1 | Semantic conventions doc | `meta/docs/semantic-conventions.md` | P0 |
| 2 | Sync web_search interface | `domain-v4/tools/`, `runtime/tools/` | P0 |
| 3 | Constrain consult_knowledge | `domain-v4/tools/consult_knowledge.json` | P0 |
| 4 | validate_artifact clarity | `domain-v4/tools/`, `runtime/tools/` | P1 |
| 5 | Field renames (hint, any, completing) | Various | P1 |
| 6 | Neutralize tool descriptions | `domain-v4/tools/*.json` | P1 |
| 7 | Update must_know knowledge | `domain-v4/knowledge/must_know/` | P1 |
| 8 | Artifact schema audit | `domain-v4/artifact-types/` | P2 |
| 9 | Semantic validation command | `src/questfoundry/cli.py` | P3 |

---

## Success Criteria

1. No tool interface uses `status` to conflate multiple concepts
2. All feedback surfaces use directive language (`recovery_action`, not `hint`)
3. Tool descriptions are role-neutral
4. `consult_knowledge.layer` validates against known layers
5. `web_search` domain and runtime interfaces match
6. Semantic conventions document exists and is referenced in agent knowledge

---

## References

- Issue #228: <https://github.com/pvliesdonk/questfoundry/issues/228>
- PR #220: Semantic clarity for delegation responses
- PR #227: Improve save_artifact validation feedback structure
- PR #242: Tool description biasing investigation
