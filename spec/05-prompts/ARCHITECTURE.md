# Layer 5 Architecture — Loop-Focused Design

**Document Version:** 1.0 (2025-11-06)

## Problem Statement

The original Layer 5 design was **role-focused**: each role had a comprehensive system prompt
containing both domain expertise AND loop procedures. This created several issues:

### Issues with Role-Focused Architecture

1. **Duplication**: Loop procedures duplicated across multiple role prompts
   - Example: Lore Deepening procedure in Lore Weaver, Showrunner, Gatekeeper prompts
2. **Maintenance burden**: Updating a loop required changing 5-7 role prompts
3. **Inconsistency risk**: Different roles might interpret loop procedures differently
4. **Unclear orchestration**: No single source of truth for "how does Lore Deepening work?"
5. **Hard to test**: Loop execution scattered across role implementations

## Solution: Loop-Focused Architecture

The refactored Layer 5 makes **loops the primary executable units** with roles participating:

### Core Principles

1. **Loop playbooks are primary** - Complete procedures with message sequences
2. **Roles participate** - Each role brings expertise, responds to intents
3. **Dual format** - Full prompts for standalone, adapters for orchestration
4. **Single source of truth** - One playbook per loop, referenced by all roles
5. **Clear orchestration** - Showrunner loads playbook → roles execute steps

## Structure

```text
05-prompts/
  loops/                          # PRIMARY: 13 executable loop playbooks
    *.playbook.md
    examples/*.json               # Validation message flows

  role_adapters/                  # NEW: 15 thin interface specs
    *.adapter.md

  [role]/                         # ENHANCED: Full role prompts
    system_prompt.md              # Now includes loop participation

  showrunner/                     # MODULAR: Split into focused modules
    system_prompt.md              # Index
    loop_orchestration.md         # How to execute playbooks
    manifest_management.md        # Hot/Cold operations
    initialization.md             # Project setup
    protocol_handlers.md          # Message validation
```

## Components

### 1. Loop Playbooks (`loops/*.playbook.md`)

**Purpose:** PRIMARY executable units defining complete loop procedures

**Contains:**

- Purpose and activation criteria
- RACI matrix (from Layer 1)
- Complete procedure with message sequences (from Layer 0 + Layer 4)
- Deliverables and success criteria
- Failure modes and remedies
- Schema references (Layer 3)

**Usage:**

- Showrunner orchestration (multi-role sessions)
- Documentation reference
- Validation via example flows

**Example:** `loops/lore_deepening.playbook.md`

```markdown
## Procedure (Message Sequences)

### Step 1: TU Open (Showrunner → Lore Weaver)

[JSON envelope example]

### Step 2: Frame Canon Questions (Lore Weaver)

[Detailed guidance] ...
```

### 2. Role Adapters (`role_adapters/*.adapter.md`)

**Purpose:** Thin interface specs for multi-role orchestration

**Contains:**

- Core expertise (3-5 bullet points)
- Protocol intents handled (receives/sends)
- Loop participation (RACI references)
- Quality bars focus
- Safety boundaries
- Handoff protocols
- Context awareness
- Escalation rules

**Usage:**

- Multi-role orchestration via Showrunner
- Layer 6 library integration
- Efficient context (minimal tokens when playbook is primary)

**Size:** 50-100 lines (vs 200+ for full prompts)

**Example:** `role_adapters/lore_weaver.adapter.md`

```markdown
## Core Expertise

- Canon continuity and referential integrity
- Timeline management (anchors, causality, paradox detection) ...

## Loop Participation

- **Lore Deepening** (R) - Transform accepted hooks into coherent canon
- **Hook Harvest** (C) - Flag canon collisions during triage
```

### 3. Full Role Prompts (`[role]/system_prompt.md`)

**Purpose:** Comprehensive standalone guides for single-role work

**Contains:**

- Complete mission and operating model
- Domain expertise and decision-making guidance
- **NEW: Loop Participation** section with playbook references
- Examples and edge cases
- Full context for independent operation

**Usage:**

- Standalone ChatGPT/Claude sessions
- Learning how a role operates
- Human guidance (comprehensive reference)
- Educational/documentation

**Size:** 200-300 lines (comprehensive)

**Enhancement:** Now includes "Loop Participation" section:

```markdown
## Loop Participation

This role participates in the following loops. For detailed procedures, see loop playbooks in
`../loops/`:

### Primary Loops (Responsible)

- **Lore Deepening** (R) - Transform accepted hooks into coherent canon
  - Playbook: `../loops/lore_deepening.playbook.md`
  - Example: `../loops/examples/lore_deepening_flow.json`
```

### 4. Showrunner Modules

**Purpose:** Modular orchestration system

**Structure:**

- `system_prompt.md` (110 lines) - Index and navigation
- `loop_orchestration.md` (100 lines) - How to execute playbooks
- `manifest_management.md` (102 lines) - Hot/Cold manifest operations
- `initialization.md` (223 lines) - 7-step project setup
- `protocol_handlers.md` (230 lines) - Message validation, error handling

**Benefits:**

- Single responsibility per module
- Easy maintenance (change one module)
- Clear navigation (index → specific module)
- Cross-references between modules

## Workflows

### Scenario 1: Multi-Role Orchestrated Loop (Primary Use Case)

**User Request:** "Run Lore Deepening for character backstory hooks"

**Execution:**

1. **Showrunner** loads `loops/lore_deepening.playbook.md`
2. **Playbook** defines 9-step procedure with message sequences
3. **Showrunner** opens TU per Step 1 → sends `tu.open` to Lore Weaver
4. **Lore Weaver** (using `role_adapters/lore_weaver.adapter.md`) responds with canon drafts
5. **Playbook** Step 8: Pre-gate → Showrunner routes to Gatekeeper
6. **Gatekeeper** (using `role_adapters/gatekeeper.adapter.md`) validates quality bars
7. **Playbook** Step 9: Handoff → Showrunner routes summaries to Codex Curator
8. **Loop completes** - All messages followed playbook procedure

**Resources loaded:**

- 1 loop playbook (lore_deepening.playbook.md)
- 3-4 role adapters (showrunner, lore_weaver, gatekeeper, codex_curator)
- Total: ~500 lines (efficient)

### Scenario 2: Standalone Single-Role Work

**User Request:** "I need help canonizing this backstory" (to ChatGPT)

**Execution:**

1. **User** uploads `lore_weaver/system_prompt.md`
2. **Full prompt** provides comprehensive guidance (no orchestration needed)
3. **Lore Weaver** operates independently with full domain expertise
4. **User** reviews output, iterates conversationally

**Resources loaded:**

- 1 full role prompt (lore_weaver/system_prompt.md)
- Total: ~250 lines (comprehensive, standalone)

### Scenario 3: Learning/Documentation

**User Question:** "How does Lore Deepening work?"

**Answer:**

1. **Read** `loops/lore_deepening.playbook.md` - Complete procedure
2. **Read** `role_adapters/lore_weaver.adapter.md` - Lore Weaver's expertise
3. **Optional** `lore_weaver/system_prompt.md` - Deep dive into decision-making

**Benefits:** Clear, layered documentation

## Benefits

### 1. Single Source of Truth

**Before:** Lore Deepening procedure in 7 role prompts **After:** 1 loop playbook, referenced by all
roles

### 2. Maintainability

**Before:** Update loop → edit 5-7 prompts **After:** Update loop → edit 1 playbook

### 3. Testability

**Before:** Test roles, infer loop behavior **After:** Test loop playbook directly with validation
examples

### 4. Discoverability

**Before:** "How does Lore Deepening work?" → read 5 prompts, synthesize **After:** "How does Lore
Deepening work?" → read 1 playbook

### 5. Role Interchangeability

**Before:** Role prompts tightly coupled to loop logic **After:** Roles = thin adapters, swap
implementations easily

### 6. Clear Orchestration

**Before:** Showrunner references "sequence role work per relevant Flow" **After:** Showrunner loads
playbook, executes steps sequentially

### 7. Efficient Context

**Before:** Load full prompts for all roles (~3000 lines) **After:** Load 1 playbook + adapters
(~1000 lines)

## Migration Path (Completed)

### Phase 1: Create Loop Playbooks ✅

- Extracted procedures from Layer 0 LOOPS/ and Layer 4 FLOWS/
- Created 13 executable playbooks with message sequences
- Added RACI matrices, schemas, success criteria
- Created 13 validation examples

### Phase 2: Create Role Adapters ✅

- Extracted core expertise from full role prompts
- Created 15 thin adapters (50-100 lines each)
- Focused on: intents, loop participation, handoffs
- No procedure duplication

### Phase 3: Enhance Full Role Prompts ✅

- Added "Loop Participation" section to all 15 prompts
- Referenced loop playbooks with relative paths
- Preserved all existing content
- Added dual-format usage guidance

### Phase 4: Modularize Showrunner ✅

- Split 213-line monolith into 5 files total
- system_prompt.md → navigation index (1)
- 4 focused modules: loop_orchestration, manifest_management, initialization, protocol_handlers (4)
- Cross-referenced modules

### Phase 5: Documentation ✅

- Update README.md with architecture explanation
- Update USAGE_GUIDE.md with playbook guidance
- Create ARCHITECTURE.md (this document)

## Layer Alignment

This architecture aligns with QuestFoundry's layer design:

- **Layer 0** (North Star) → Loop purposes, guardrails
- **Layer 1** (Roles) → RACI matrices, role charters
- **Layer 2** (Dictionary) → Terminology, artifact structures
- **Layer 3** (Schemas) → Validation rules
- **Layer 4** (Protocol) → Message sequences, intents
- **Layer 5** (Prompts) → **Executable loop playbooks** + role expertise
- **Layer 6** (Library) → Will load playbooks + adapters programmatically

Loop playbooks bridge Layers 0-4 specification into Layer 5 execution.

## Future Enhancements

### Layer 6 Integration

```python
from questfoundry import Showrunner, loop_playbooks

# Load loop playbook
playbook = loop_playbooks.get("lore_deepening")

# Execute with role adapters
sr = Showrunner(playbook=playbook)
result = sr.execute(
    inputs={"hooks": ["HOOK-001", "HOOK-002"]},
    adapters=["lore_weaver", "gatekeeper", "codex_curator"]
)
```

### Testing Framework

```python
# Validate loop playbook execution
def test_lore_deepening_flow():
    flow = load_example("loops/examples/lore_deepening_flow.json")
    result = execute_playbook("lore_deepening", flow["messages"])
    assert result.deliverables == ["canon_pack_kestrel.json"]
    assert result.success_criteria_met == True
```

### Playbook Versioning

- Tag playbook releases (v1.0.0)
- Support multiple playbook versions
- Backward compatibility guarantees

## References

- **Layer 0 LOOPS:** `/00-north-star/LOOPS/` - Loop specifications
- **Layer 1 RACI:** `/01-roles/raci/by_loop.md` - Role assignments
- **Layer 4 FLOWS:** `/04-protocol/FLOWS/` - Protocol message sequences
- **Loop Playbooks:** `/05-prompts/loops/` - Executable procedures
- **Role Adapters:** `/05-prompts/role_adapters/` - Interface specs
- **Full Prompts:** `/05-prompts/[role]/system_prompt.md` - Comprehensive guides

---

**Document Status:** Normative - This architecture is the canonical design for Layer 5 going
forward.
