# QuestFoundry Migration: v1 → v2 Architectural Refactoring

**Document Version:** 1.0
**Target LLM:** Claude Sonnet 4.5, GPT-5, or equivalent (GPT-5-mini/Haiku for specific phases)
**Migration Type:** Clean break (major version, separate branch, no backward compatibility required)
**Estimated Effort:** 80-120 agent hours across 3 phases

---

## Executive Summary

This document provides complete instructions for an AI agent to migrate QuestFoundry from v1 (pre-assembled prompt architecture) to v2 (atomic, composable behavior primitives). The migration addresses critical maintainability issues where logic like "Lore Weaver Expertise" is duplicated across multiple files, making updates error-prone and inconsistent.

**Migration Goal:** Transform `spec/05-prompts/` from monolithic prompt files into atomic, composable behavior primitives in `spec/05-behavior/`, with a new spec-compiler that assembles runtime artifacts.

**Branch Strategy:** All work occurs on a dedicated migration branch with PRs against that branch. No compatibility with v1 required.

---

## Repository Context

### Current Architecture (v1)

**7-Layer System:**
- **L0-L4:** Specification layers (North Star, Roles, Dictionary, Schemas, Protocol) - **STABLE, DO NOT MODIFY**
- **L5 (Prompts):** `spec/05-prompts/` - Pre-assembled prompt templates - **TARGET FOR MIGRATION**
- **L6 (Python Library):** `lib/python/src/questfoundry/` - Runtime implementation - **REQUIRES REFACTORING**

**Current L5 Structure:**
```
spec/05-prompts/
  loops/*.playbook.md          # 13 loop playbooks (e.g., lore_deepening.playbook.md)
  role_adapters/*.adapter.md   # 15 role adapters (e.g., lore_weaver.adapter.md)
  [role]/system_prompt.md      # 15 full role prompts (e.g., lore_weaver/system_prompt.md)
  [role]/intent_handlers/*.md  # Intent-specific behaviors
  _shared/*.md                 # Shared context/safety protocols
```

**Current L6 Structure:**
```
lib/python/src/questfoundry/
  loops/                       # Hardcoded loop classes (e.g., lore_deepening.py)
  roles/                       # Role implementations
  resources/prompts/           # Bundled prompt files from L5
```

**Problem:** Duplication across playbooks, adapters, standalone prompts, and intent handlers. Changes require N-way updates.

### Target Architecture (v2)

**New L5 Structure (Behavior Primitives):**
```
spec/05-behavior/
  README.md                    # Architecture documentation

  expertises/                  # Atomic domain expertise definitions
    lore_weaver_expertise.md
    gatekeeper_quality_bars.md
    scene_smith_prose.md
    [role]_[domain].md

  procedures/                  # Reusable workflow procedures with YAML frontmatter
    canonization_core.md       # Core canonization algorithm
    continuity_check.md        # Contradiction detection
    player_safe_summarization.md
    gatecheck_validation.md
    [procedure_name].md

  snippets/                    # Small reusable text blocks
    spoiler_hygiene_reminder.md
    validation_protocol.md
    human_question_format.md
    [snippet_name].md

  playbooks/                   # Loop playbooks referencing procedures/expertises
    lore_deepening.playbook.yaml
    hook_harvest.playbook.yaml
    [loop_name].playbook.yaml

  adapters/                    # Role adapters referencing expertises
    lore_weaver.adapter.yaml
    gatekeeper.adapter.yaml
    [role_name].adapter.yaml
```

**New L6 Build (Spec Compiler):**
```
lib/python/src/questfoundry/
  compiler/
    __init__.py
    spec_compiler.py           # Main compiler orchestrator
    validators.py              # Cross-reference validation
    assemblers.py              # Prompt assembly logic
    manifest_builder.py        # JSON manifest generation
```

**New L6a/L7a Runtime:**
```
lib/python/src/questfoundry/
  execution/
    playbook_executor.py       # Generic playbook execution engine
    manifest_loader.py         # Load compiled manifests
```

**New L6b/L7b Outputs:**
```
dist/compiled/
  manifests/
    lore_deepening.manifest.json    # Runtime-ready loop definitions
    lore_weaver.manifest.json        # Runtime-ready role definitions

  standalone_prompts/
    lore_weaver_full.md              # Assembled standalone prompts
    showrunner_full.md
```

---

## Migration Phases

### Phase 1: Deconstruction & Atomization (40-50 hours)

**Objective:** Break existing L5 prompts into atomic components in `spec/05-behavior/`

**Assigned LLM:** Sonnet 4.5 or GPT-5 (requires nuanced understanding)

**Inputs:**
- All files in `spec/05-prompts/`
- Layer 1 role charters (`spec/01-roles/`)
- Layer 3 schemas (`spec/03-schemas/`)
- Layer 4 protocol (`spec/04-protocol/`)

**Outputs:**
- `spec/05-behavior/` directory structure with atomic files
- Migration tracking spreadsheet documenting all extractions

#### Phase 1 Tasks

##### 1.1 Create Directory Structure

```bash
mkdir -p spec/05-behavior/{expertises,procedures,snippets,playbooks,adapters}
```

Create `spec/05-behavior/README.md` documenting:
- Purpose of atomic primitives
- YAML frontmatter schema for procedures/playbooks/adapters
- Cross-reference syntax (e.g., `@expertise:lore_weaver_expertise`, `@procedure:canonization_core`)
- Validation requirements

##### 1.2 Extract Expertises

For each of the 15 roles, extract domain expertise into atomic files:

**Example: `lore_weaver_expertise.md`**

Source locations to extract from:
- `spec/05-prompts/lore_weaver/system_prompt.md` (lines 17-39: "Operating Model" section)
- `spec/05-prompts/role_adapters/lore_weaver.adapter.md` (lines 10-35: "Core Expertise" section)

Extract common content, creating single source of truth:
```markdown
# Lore Weaver Expertise

## Canon Creation
Turn accepted hooks into cohesive canon: backstories, timelines, metaphysics, causal chains, entity/state updates.

## Continuity Management
Keep continuity ledger: who knows what when, what changed, what must remain invariant.

## Player-Safe Summarization
Provide brief, non-spoiling abstracts to Codex Curator for publication; never leak canon to surfaces.

[... full expertise content ...]
```

**Deliverable:** 15+ expertise files (one per role, some roles may have multiple domain areas)

##### 1.3 Extract Procedures

Identify reusable workflow procedures across playbooks and role prompts:

**Example: `canonization_core.md`**

Add YAML frontmatter for metadata and cross-references:
```yaml
---
procedure_id: canonization_core
description: Core algorithm for transforming hooks into canon
references_expertises:
  - lore_weaver_expertise
references_schemas:
  - hook_card.schema.json
  - canon_pack.schema.json
references_roles:
  - lore_weaver
  - researcher
---

# Canonization Core Procedure

## Step 1: Analyze Hook
[... procedure steps ...]
```

Source locations:
- `spec/05-prompts/lore_weaver/system_prompt.md` (lines 24-38: "Canonization algorithm")
- `spec/05-prompts/loops/lore_deepening.playbook.md` (steps 2-8)

**Deliverable:** 30-50 procedure files extracted from playbooks and role prompts

##### 1.4 Extract Snippets

Identify small, frequently-reused text blocks:

**Example: `spoiler_hygiene_reminder.md`**
```markdown
# Spoiler Hygiene Protocol

**CRITICAL:** Canon Packs remain in Hot ALWAYS. NEVER ship canon to player surfaces. Only player-safe summaries go to Codex Curator.
```

Source locations:
- Appears in multiple role prompts and playbooks
- `spec/05-prompts/_shared/safety_protocol.md`

**Deliverable:** 20-30 snippet files

##### 1.5 Convert Playbooks to YAML

Transform each playbook from markdown to YAML with references:

**Example: `lore_deepening.playbook.yaml`**
```yaml
---
playbook_id: lore_deepening
display_name: Lore Deepening
category: discovery
description: Transform accepted hooks into coherent canon

# Cross-references to behavior primitives
references_procedures:
  - canonization_core
  - continuity_check
  - player_safe_summarization

references_expertises:
  - lore_weaver_expertise
  - researcher_fact_checking

references_snippets:
  - spoiler_hygiene_reminder
  - validation_protocol

# RACI Matrix
raci:
  responsible:
    - role: lore_weaver
      steps: [frame_questions, draft_canon, check_contradictions, create_impact_notes, package_canon]
  accountable:
    - role: showrunner
      scope: overall
  consulted:
    - role: researcher
      steps: [draft_canon]
    - role: plotwright
      steps: [create_impact_notes]
  informed:
    - role: codex_curator
      scope: player_safe_summaries

# Steps reference procedures
steps:
  - step_id: frame_questions
    description: Frame canon questions from hooks
    procedure: "@procedure:canonization_core#step1"
    assigned_roles: [lore_weaver]

  - step_id: draft_canon
    description: Draft spoiler-level canon answers
    procedure: "@procedure:canonization_core#step2-3"
    assigned_roles: [lore_weaver]
    consulted_roles: [researcher]

  [... remaining steps ...]

# Artifacts reference L3 schemas
artifacts_input:
  - hook_card
artifacts_output:
  - canon_pack

# Quality bars from L0
quality_bars_pressed:
  - integrity
  - gateways
  - presentation
---
```

**Deliverable:** 13 playbook YAML files

##### 1.6 Convert Adapters to YAML

Transform role adapters to YAML with references:

**Example: `lore_weaver.adapter.yaml`**
```yaml
---
adapter_id: lore_weaver
role_name: Lore Weaver
abbreviation: LW

# Primary expertise reference
expertise: "@expertise:lore_weaver_expertise"

# Mission from L1 charter
mission: "Resolve the world's deep truth—quietly—then hand clear, spoiler-safe summaries to neighbors who face the player."

# Protocol intents from L4
protocol_intents:
  receives:
    - hook.accept
    - tu.open
    - canon.validate
  sends:
    - canon.create
    - canon.update
    - hook.create
    - merge.request

# Loop participation (references playbooks)
loops:
  - playbook: lore_deepening
    raci: responsible
  - playbook: hook_harvest
    raci: consulted
  - playbook: story_spark
    raci: consulted

# Cross-cutting concerns (snippets)
safety_protocols:
  - "@snippet:spoiler_hygiene_reminder"
  - "@snippet:pn_boundary_enforcement"

# Handoff protocols (references other roles)
handoffs:
  to_codex_curator: "@procedure:player_safe_summarization"
  to_plotwright: "@procedure:topology_impact_notes"
  to_scene_smith: "@procedure:prose_callback_notes"
---
```

**Deliverable:** 15 adapter YAML files

##### 1.7 Validation Checkpoint

Create validation scripts:

**Script: `scripts/validate_behavior_refs.py`**

Checks:
1. All `@expertise:`, `@procedure:`, `@snippet:` references resolve to actual files
2. All `references_schemas` point to valid L3 schemas
3. All `references_roles` match L1 role definitions
4. No orphaned files (every atomic file is referenced by at least one playbook/adapter)
5. No circular dependencies

**Deliverable:** Validation script with full pass on all cross-references

##### 1.8 Create Migration Tracking

**Spreadsheet: `spec/05-behavior/MIGRATION_TRACKING.csv`**

Columns:
- Source File (v1)
- Content Extracted
- Target File (v2)
- Cross-References Count
- Status (extracted/validated/reviewed)
- Notes

Track every extraction for human review if needed.

---

### Phase 2: Build Spec Compiler (30-40 hours)

**Objective:** Implement compiler that validates and assembles behavior primitives into runtime artifacts

**Assigned LLM:** Sonnet 4.5 or GPT-5 (complex logic) with Haiku/mini for testing

**Inputs:**
- `spec/05-behavior/` directory structure from Phase 1
- Layer 3 schemas for validation
- Layer 1 roles for RACI validation

**Outputs:**
- Working spec compiler in `lib/python/src/questfoundry/compiler/`
- Compiled manifests in `dist/compiled/manifests/`
- Standalone prompts in `dist/compiled/standalone_prompts/`

#### Phase 2 Tasks

##### 2.1 Design Compiler Architecture

Create `lib/python/src/questfoundry/compiler/README.md` documenting:

**Compiler Pipeline:**
```
[Atomic Sources] → [Validator] → [Assembler] → [Manifest Builder] → [Output Writer]
     (YAML/MD)        (refs)       (compose)       (JSON)           (dist/)
```

**Components:**
1. **Loader:** Parse YAML frontmatter and markdown content
2. **Validator:** Validate cross-references and schema compliance
3. **Assembler:** Compose prompts by resolving references
4. **Manifest Builder:** Generate JSON runtime manifests
5. **Output Writer:** Write to `dist/compiled/`

##### 2.2 Implement Core Compiler

**File: `lib/python/src/questfoundry/compiler/spec_compiler.py`**

```python
from pathlib import Path
from typing import Dict, List, Any
import yaml
import json
from dataclasses import dataclass

@dataclass
class BehaviorPrimitive:
    """Base class for atomic behavior components."""
    id: str
    type: str  # 'expertise', 'procedure', 'snippet', 'playbook', 'adapter'
    content: str
    references: Dict[str, List[str]]
    metadata: Dict[str, Any]

class SpecCompiler:
    """Main spec compiler orchestrator."""

    def __init__(self, spec_root: Path):
        self.spec_root = spec_root
        self.behavior_dir = spec_root / "05-behavior"
        self.primitives: Dict[str, BehaviorPrimitive] = {}

    def load_all_primitives(self) -> None:
        """Load all behavior primitives from disk."""
        # Load expertises, procedures, snippets, playbooks, adapters
        pass

    def validate_references(self) -> List[str]:
        """Validate all cross-references resolve correctly."""
        errors = []
        for prim_id, primitive in self.primitives.items():
            for ref_type, ref_list in primitive.references.items():
                for ref in ref_list:
                    if not self._resolve_reference(ref):
                        errors.append(f"{prim_id}: Invalid reference {ref}")
        return errors

    def _resolve_reference(self, ref: str) -> bool:
        """Check if a reference like @expertise:lore_weaver_expertise resolves."""
        # Parse reference syntax: @type:id or @type:id#section
        pass

    def assemble_playbook(self, playbook_id: str) -> str:
        """Assemble complete playbook markdown by resolving all references."""
        # Load playbook YAML
        # For each reference, inject referenced content
        # Return assembled markdown
        pass

    def build_playbook_manifest(self, playbook_id: str) -> Dict[str, Any]:
        """Build JSON manifest for runtime execution."""
        # Convert YAML playbook to runtime-ready JSON
        # Include all metadata needed by PlaybookExecutor
        pass

    def compile_all(self, output_dir: Path) -> None:
        """Run full compilation pipeline."""
        self.load_all_primitives()

        errors = self.validate_references()
        if errors:
            raise CompilationError(f"Validation failed: {errors}")

        # Build manifests for all playbooks
        # Build manifests for all adapters
        # Assemble standalone prompts
        # Write to output_dir
        pass
```

**Deliverable:** Core compiler implementation

##### 2.3 Implement Reference Resolution

**File: `lib/python/src/questfoundry/compiler/assemblers.py`**

Handle reference syntax:
- `@expertise:lore_weaver_expertise` → Inject full expertise content
- `@procedure:canonization_core` → Inject full procedure
- `@procedure:canonization_core#step1` → Inject specific section
- `@snippet:spoiler_hygiene_reminder` → Inject snippet
- `@playbook:lore_deepening` → Reference (link, don't inline)
- `@schema:canon_pack.schema.json` → Reference with validation

**Deliverable:** Reference resolution engine

##### 2.4 Implement Validators

**File: `lib/python/src/questfoundry/compiler/validators.py`**

```python
class ReferenceValidator:
    """Validate cross-references between primitives."""

    def validate_expertise_refs(self) -> List[str]:
        """Check all expertise references resolve."""
        pass

    def validate_schema_refs(self) -> List[str]:
        """Check all schema references point to valid L3 schemas."""
        pass

    def validate_role_refs(self) -> List[str]:
        """Check all role references match L1 role definitions."""
        pass

    def detect_circular_deps(self) -> List[str]:
        """Detect circular dependencies in references."""
        pass

    def check_orphans(self) -> List[str]:
        """Find primitives not referenced by any playbook/adapter."""
        pass
```

**Deliverable:** Comprehensive validation suite

##### 2.5 Implement Manifest Builder

**File: `lib/python/src/questfoundry/compiler/manifest_builder.py`**

Generate JSON manifests for runtime:

**Example output: `dist/compiled/manifests/lore_deepening.manifest.json`**
```json
{
  "$schema": "https://questfoundry.liesdonk.nl/manifests/playbook_manifest.schema.json",
  "manifest_version": "2.0.0",
  "playbook_id": "lore_deepening",
  "display_name": "Lore Deepening",
  "compiled_at": "2025-11-13T10:30:00Z",
  "source_files": [
    "spec/05-behavior/playbooks/lore_deepening.playbook.yaml",
    "spec/05-behavior/procedures/canonization_core.md",
    "spec/05-behavior/expertises/lore_weaver_expertise.md"
  ],
  "steps": [
    {
      "step_id": "frame_questions",
      "description": "Frame canon questions from hooks",
      "assigned_roles": ["lore_weaver"],
      "procedure_content": "... assembled content ...",
      "artifacts_input": ["hook_card"],
      "artifacts_output": ["canon_pack"]
    }
  ],
  "raci": { ... },
  "quality_bars": ["integrity", "gateways", "presentation"]
}
```

**Deliverable:** Manifest generation for playbooks and adapters

##### 2.6 Implement Standalone Prompt Assembly

**File: `lib/python/src/questfoundry/compiler/assemblers.py`**

Assemble full standalone prompts by composing:
1. Role charter (from adapter)
2. Referenced expertises (full content)
3. Protocol intents (from adapter)
4. Referenced snippets (validation, safety protocols)
5. Loop participation (summary with links)
6. Examples (if present)

**Example output: `dist/compiled/standalone_prompts/lore_weaver_full.md`**

```markdown
# Lore Weaver — System Prompt

Target: GPT-5, Claude Sonnet 4.5+

## Mission
Resolve the world's deep truth—quietly—then hand clear, spoiler-safe summaries to neighbors who face the player.

## References
- 01-roles/charters/lore_weaver.md
- 02-dictionary/artifacts/canon_pack.md
- Compiled from: spec/05-behavior/lore_weaver.adapter.yaml

---

## Core Expertise

[... injected from lore_weaver_expertise.md ...]

---

## Canonization Procedure

[... injected from canonization_core.md ...]

---

## Safety & Validation

[... injected from spoiler_hygiene_reminder.md ...]
[... injected from validation_protocol.md ...]

---

[... remaining sections ...]
```

**Deliverable:** Standalone prompt generation

##### 2.7 Create CLI Interface

**File: `lib/python/src/questfoundry/compiler/cli.py`**

```bash
# Compile all
qf-compile --spec-dir spec/ --output dist/compiled/

# Compile specific playbook
qf-compile --playbook lore_deepening --output dist/compiled/

# Validate only (no output)
qf-compile --validate-only

# Watch mode (recompile on change)
qf-compile --watch
```

**Deliverable:** CLI for compilation

##### 2.8 Write Compiler Tests

**File: `lib/python/tests/compiler/test_spec_compiler.py`**

Test cases:
1. Valid compilation of all playbooks
2. Detection of invalid references
3. Detection of circular dependencies
4. Orphan detection
5. Schema reference validation
6. Manifest JSON schema validation
7. Standalone prompt completeness

**Deliverable:** 95%+ test coverage for compiler

##### 2.9 Create Manifest JSON Schema

**File: `spec/manifests/playbook_manifest.schema.json`**

Define JSON schema for compiled playbook manifests (Layer 3 extension).

**Deliverable:** Manifest schema with validation

---

### Phase 3: Refactor Runtime & Cleanup (20-30 hours)

**Objective:** Replace hardcoded loop classes with generic executor, update library, delete deprecated code

**Assigned LLM:** Sonnet 4.5/GPT-5 for executor, Haiku/mini for cleanup tasks

**Inputs:**
- Compiled manifests from Phase 2
- Existing loop/role implementations in `lib/python/src/questfoundry/`

**Outputs:**
- Generic `PlaybookExecutor` replacing hardcoded loops
- Updated library using compiled manifests
- Deleted `spec/05-prompts/` directory
- Deleted hardcoded loop classes

#### Phase 3 Tasks

##### 3.1 Implement PlaybookExecutor

**File: `lib/python/src/questfoundry/execution/playbook_executor.py`**

```python
from pathlib import Path
from typing import Dict, Any, List
import json

class PlaybookExecutor:
    """Generic executor for compiled playbook manifests."""

    def __init__(self, manifest_path: Path):
        self.manifest = self._load_manifest(manifest_path)
        self.current_step_index = 0
        self.context: Dict[str, Any] = {}

    def _load_manifest(self, path: Path) -> Dict[str, Any]:
        """Load and validate playbook manifest."""
        with open(path) as f:
            manifest = json.load(f)
        # Validate against manifest schema
        return manifest

    def execute_step(self, step_id: str, roles: Dict[str, Any]) -> Any:
        """Execute a single step using assigned roles."""
        step = self._get_step(step_id)

        # Get assigned role instances
        assigned_roles = [roles[r] for r in step['assigned_roles']]

        # Provide step procedure content to role
        procedure_prompt = step['procedure_content']

        # Execute via role
        result = assigned_roles[0].execute(procedure_prompt, self.context)

        # Validate output artifacts if required
        if step.get('validation_required'):
            self._validate_artifacts(result, step['artifacts_output'])

        return result

    def execute_full_loop(self, roles: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entire playbook from start to finish."""
        results = {}
        for step in self.manifest['steps']:
            result = self.execute_step(step['step_id'], roles)
            results[step['step_id']] = result
        return results

    def _validate_artifacts(self, result: Any, expected_types: List[str]) -> None:
        """Validate output artifacts against L3 schemas."""
        # Use existing validation infrastructure
        pass
```

**Key Features:**
- Load any playbook manifest
- Execute steps generically via role interfaces
- Validate artifacts using existing validators
- No hardcoded logic for specific loops

**Deliverable:** Generic playbook executor

##### 3.2 Update Role Implementations

**File: `lib/python/src/questfoundry/roles/base.py`**

Update role interface to accept procedure prompts from manifests:

```python
class Role(ABC):
    """Base class for all QuestFoundry roles."""

    @abstractmethod
    def execute(self, procedure_prompt: str, context: Dict[str, Any]) -> Any:
        """
        Execute a procedure prompt with given context.

        In v2, procedure prompts are assembled from atomic primitives
        and provided by PlaybookExecutor.
        """
        pass
```

**Deliverable:** Updated role base class

##### 3.3 Migrate Loop Registry

**File: `lib/python/src/questfoundry/loops/registry.py`**

Change from class-based registry to manifest-based:

```python
class LoopRegistry:
    """Registry of available loops loaded from compiled manifests."""

    def __init__(self, manifest_dir: Path):
        self.manifest_dir = manifest_dir
        self.loops: Dict[str, Path] = self._discover_loops()

    def _discover_loops(self) -> Dict[str, Path]:
        """Discover all playbook manifests."""
        loops = {}
        for manifest_path in self.manifest_dir.glob("*.manifest.json"):
            with open(manifest_path) as f:
                data = json.load(f)
                loops[data['playbook_id']] = manifest_path
        return loops

    def get_executor(self, loop_id: str) -> PlaybookExecutor:
        """Get executor for a specific loop."""
        manifest_path = self.loops[loop_id]
        return PlaybookExecutor(manifest_path)
```

**Deliverable:** Manifest-based loop registry

##### 3.4 Update Resource Bundling

**File: `scripts/bundle_resources.py`**

Update to bundle compiled artifacts instead of raw prompts:

```python
def bundle_compiled_resources():
    """Bundle compiled manifests and standalone prompts into package."""
    # Copy dist/compiled/manifests/* to lib/python/src/questfoundry/resources/manifests/
    # Copy dist/compiled/standalone_prompts/* to lib/python/src/questfoundry/resources/prompts/
    pass
```

**Deliverable:** Updated bundler

##### 3.5 Delete Deprecated Code

**Directories to delete:**
1. `spec/05-prompts/` — Replaced by `spec/05-behavior/`
2. `lib/python/src/questfoundry/loops/*.py` (except `base.py`, `registry.py`) — Replaced by generic executor
3. `lib/python/src/questfoundry/resources/prompts/` (old bundled prompts)

**Before deletion:**
- Ensure all tests pass with new executor
- Archive deleted files to a git tag: `git tag v1-archive`

**Commands:**
```bash
# Archive v1
git tag -a v1-archive -m "Archive v1 architecture before deletion"
git push origin v1-archive

# Delete deprecated directories
rm -rf spec/05-prompts/
rm -f lib/python/src/questfoundry/loops/{lore_deepening,hook_harvest,story_spark}.py
rm -rf lib/python/src/questfoundry/resources/prompts/
```

**Deliverable:** Clean codebase with only v2 architecture

##### 3.6 Update Tests

**Tasks:**
1. Delete tests for hardcoded loop classes
2. Create tests for `PlaybookExecutor` with various manifests
3. Create tests for manifest loading and validation
4. Update integration tests to use new executor

**Deliverable:** Full test suite passing with v2 architecture

##### 3.7 Update Documentation

**Files to update:**
1. `README.md` — Update architecture description
2. `spec/README.md` — Update layer descriptions (L5 is now "Behavior")
3. `lib/python/README.md` — Update library usage
4. `docs/` — Update MkDocs documentation

**Key changes:**
- Rename "Layer 5: Prompts" to "Layer 5: Behavior"
- Document atomic primitives architecture
- Document spec compiler usage
- Document manifest-based execution

**Deliverable:** Updated documentation

##### 3.8 Update CI/CD

**File: `.github/workflows/lint-test.yml`**

Add compiler validation:
```yaml
- name: Validate spec compilation
  run: |
    qf-compile --validate-only --spec-dir spec/

- name: Compile spec
  run: |
    qf-compile --spec-dir spec/ --output dist/compiled/

- name: Test compiled manifests
  run: |
    pytest tests/execution/test_playbook_executor.py
```

**Deliverable:** CI validates compilation

##### 3.9 Version Bump

Update version to 2.0.0 in:
- `lib/python/pyproject.toml`
- `spec/05-behavior/VERSION`

Create migration guide:
**File: `MIGRATION_V1_TO_V2_USER_GUIDE.md`**

For users of the library, explain:
- Breaking changes
- How to update code using old loop classes
- How to use new `PlaybookExecutor`
- How to compile custom behavior primitives

**Deliverable:** Version 2.0.0 with migration guide

---

## Agent Execution Instructions

### Prerequisites

**Environment Setup:**
```bash
# Navigate to repository
cd /home/user/questfoundry

# Create migration branch
git checkout -b migration/v1-to-v2

# Install dependencies
uv sync

# Verify tests pass in v1
pytest
```

### Phase Execution Order

**Phase 1: Deconstruction (Sonnet 4.5 / GPT-5)**
```bash
# Create tracking document
touch spec/05-behavior/MIGRATION_TRACKING.csv

# Execute tasks 1.1-1.8 sequentially
# Each task should commit intermediate work
git add spec/05-behavior/
git commit -m "feat(phase1): complete deconstruction of [component]"
```

**Phase 2: Compiler (Sonnet 4.5 / GPT-5)**
```bash
# Execute tasks 2.1-2.9 sequentially
# Each major component should be tested and committed
pytest tests/compiler/
git commit -m "feat(phase2): implement spec compiler [component]"
```

**Phase 3: Runtime (Sonnet 4.5 for executor, Haiku for cleanup)**
```bash
# Execute tasks 3.1-3.9 sequentially
# Critical: Don't delete old code until new tests pass
pytest tests/execution/
git commit -m "feat(phase3): implement generic executor"

# Then cleanup
git commit -m "chore(phase3): delete deprecated v1 code"
```

### Validation Gates

**After Phase 1:**
- [ ] All cross-references resolve
- [ ] No orphaned files
- [ ] No circular dependencies
- [ ] All 15 roles have adapter YAML
- [ ] All 13 loops have playbook YAML
- [ ] Migration tracking complete

**After Phase 2:**
- [ ] Compiler runs without errors
- [ ] All manifests validate against schema
- [ ] All standalone prompts generated
- [ ] Compiler tests pass (95%+ coverage)
- [ ] No validation errors

**After Phase 3:**
- [ ] All tests pass with new executor
- [ ] No hardcoded loop classes remain
- [ ] Old prompt directory deleted
- [ ] Documentation updated
- [ ] CI passes
- [ ] Version bumped to 2.0.0

### Commit Strategy

**Commit Message Format:**
```
feat(phase[N]): [component] - [brief description]

- Detailed change 1
- Detailed change 2

Refs: MIGRATION_V1_TO_V2.md Phase [N] Task [N.N]
```

**Branch Strategy:**
```
migration/v1-to-v2              # Main migration branch
  ├─ phase1/[task]              # Optional: sub-branches for complex tasks
  ├─ phase2/[task]
  └─ phase3/[task]
```

**PR Strategy:**
- Create PRs for each phase against `migration/v1-to-v2`
- Final PR from `migration/v1-to-v2` to `main` after all validation gates pass

### Error Recovery

**If Phase 1 fails:**
- Review migration tracking spreadsheet
- Manually inspect duplicated content
- Consult L1/L3/L4 specs for canonical definitions
- Ask human for ambiguous architectural decisions

**If Phase 2 compilation fails:**
- Check reference syntax in YAML files
- Validate YAML frontmatter structure
- Ensure all referenced files exist
- Check for typos in reference IDs

**If Phase 3 tests fail:**
- Compare new executor output with old loop class output
- Ensure manifests contain all required metadata
- Check role interface compatibility
- Verify artifact validation still works

### Human Escalation Points

**Automatic escalation required for:**
1. Ambiguous content in prompts (conflicting logic across files)
2. Architectural decisions not specified in this document
3. Breaking changes to L0-L4 specs (should not happen)
4. Test failures that cannot be resolved by examining manifest structure

**Escalation format:**
```markdown
## Escalation: [Brief Issue]

**Phase:** [N]
**Task:** [N.N]
**Context:** [What were you doing]
**Issue:** [What went wrong]
**Attempted Solutions:** [What you tried]
**Question:** [Specific question for human]
```

---

## Success Criteria

### Technical Criteria
- [ ] All behavior primitives extracted and validated
- [ ] Spec compiler functional and tested
- [ ] Generic executor replaces all hardcoded loops
- [ ] All 819+ tests pass (or replaced with equivalent v2 tests)
- [ ] CI/CD pipeline passes
- [ ] No code duplication in behavior layer

### Documentation Criteria
- [ ] `spec/05-behavior/README.md` complete
- [ ] Compiler usage documented
- [ ] User migration guide complete
- [ ] All layer documentation updated

### Maintainability Criteria
- [ ] Single source of truth for all role expertise
- [ ] No N-way updates required for logic changes
- [ ] Clear cross-reference graph
- [ ] Validation prevents broken references

---

## Appendix A: Cross-Reference Syntax Specification

### Reference Types

**Expertise Reference:**
```yaml
@expertise:lore_weaver_expertise
```
Resolves to: `spec/05-behavior/expertises/lore_weaver_expertise.md`

**Procedure Reference:**
```yaml
@procedure:canonization_core
@procedure:canonization_core#step1  # With section anchor
```
Resolves to: `spec/05-behavior/procedures/canonization_core.md`

**Snippet Reference:**
```yaml
@snippet:spoiler_hygiene_reminder
```
Resolves to: `spec/05-behavior/snippets/spoiler_hygiene_reminder.md`

**Playbook Reference:**
```yaml
@playbook:lore_deepening
```
Resolves to: `spec/05-behavior/playbooks/lore_deepening.playbook.yaml`

**Adapter Reference:**
```yaml
@adapter:lore_weaver
```
Resolves to: `spec/05-behavior/adapters/lore_weaver.adapter.yaml`

**Schema Reference (L3):**
```yaml
@schema:canon_pack.schema.json
```
Resolves to: `spec/03-schemas/canon_pack.schema.json`

**Role Reference (L1):**
```yaml
@role:lore_weaver
```
Resolves to: `spec/01-roles/charters/lore_weaver.md`

### YAML Frontmatter Schema

**For Procedures:**
```yaml
---
procedure_id: canonization_core
description: Core algorithm for transforming hooks into canon
version: 2.0.0
references_expertises:
  - lore_weaver_expertise
references_schemas:
  - canon_pack.schema.json
  - hook_card.schema.json
references_roles:
  - lore_weaver
  - researcher
tags:
  - canon
  - validation
---
```

**For Playbooks:**
```yaml
---
playbook_id: lore_deepening
display_name: Lore Deepening
category: discovery
version: 2.0.0
references_procedures:
  - canonization_core
  - continuity_check
references_expertises:
  - lore_weaver_expertise
references_snippets:
  - spoiler_hygiene_reminder
  - validation_protocol
raci:
  responsible: [lore_weaver]
  accountable: [showrunner]
  consulted: [researcher, plotwright]
  informed: [codex_curator]
steps: [...]
artifacts_input: [hook_card]
artifacts_output: [canon_pack]
quality_bars_pressed: [integrity, gateways, presentation]
---
```

**For Adapters:**
```yaml
---
adapter_id: lore_weaver
role_name: Lore Weaver
abbreviation: LW
version: 2.0.0
expertise: "@expertise:lore_weaver_expertise"
mission: "..."
protocol_intents:
  receives: [hook.accept, tu.open, canon.validate]
  sends: [canon.create, canon.update, merge.request]
loops:
  - playbook: lore_deepening
    raci: responsible
  - playbook: hook_harvest
    raci: consulted
safety_protocols:
  - "@snippet:spoiler_hygiene_reminder"
handoffs:
  to_codex_curator: "@procedure:player_safe_summarization"
---
```

---

## Appendix B: File Naming Conventions

### Expertises
- Pattern: `[role]_[domain].md`
- Examples:
  - `lore_weaver_expertise.md`
  - `gatekeeper_quality_bars.md`
  - `scene_smith_prose_craft.md`
  - `researcher_fact_validation.md`

### Procedures
- Pattern: `[action]_[object].md` or `[process_name].md`
- Examples:
  - `canonization_core.md`
  - `continuity_check.md`
  - `player_safe_summarization.md`
  - `gatecheck_validation.md`
  - `topology_impact_analysis.md`

### Snippets
- Pattern: `[concept]_[type].md`
- Examples:
  - `spoiler_hygiene_reminder.md`
  - `validation_protocol.md`
  - `human_question_format.md`
  - `pn_boundary_enforcement.md`

### Playbooks
- Pattern: `[loop_name].playbook.yaml`
- Examples:
  - `lore_deepening.playbook.yaml`
  - `hook_harvest.playbook.yaml`
  - `story_spark.playbook.yaml`

### Adapters
- Pattern: `[role_name].adapter.yaml`
- Examples:
  - `lore_weaver.adapter.yaml`
  - `gatekeeper.adapter.yaml`
  - `showrunner.adapter.yaml`

---

## Appendix C: Manifest JSON Schema Example

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://questfoundry.liesdonk.nl/manifests/playbook_manifest.schema.json",
  "title": "Playbook Manifest",
  "description": "Runtime manifest for compiled playbooks in QuestFoundry v2",
  "type": "object",
  "required": ["manifest_version", "playbook_id", "display_name", "steps", "compiled_at"],
  "properties": {
    "manifest_version": {
      "type": "string",
      "pattern": "^2\\.\\d+\\.\\d+$",
      "description": "Semantic version of manifest schema"
    },
    "playbook_id": {
      "type": "string",
      "description": "Unique identifier for this playbook"
    },
    "display_name": {
      "type": "string",
      "description": "Human-readable playbook name"
    },
    "compiled_at": {
      "type": "string",
      "format": "date-time",
      "description": "Timestamp of compilation"
    },
    "source_files": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of source files used in compilation"
    },
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["step_id", "description", "assigned_roles", "procedure_content"],
        "properties": {
          "step_id": {"type": "string"},
          "description": {"type": "string"},
          "assigned_roles": {
            "type": "array",
            "items": {"type": "string"}
          },
          "consulted_roles": {
            "type": "array",
            "items": {"type": "string"}
          },
          "procedure_content": {
            "type": "string",
            "description": "Assembled procedure markdown for this step"
          },
          "artifacts_input": {
            "type": "array",
            "items": {"type": "string"}
          },
          "artifacts_output": {
            "type": "array",
            "items": {"type": "string"}
          },
          "validation_required": {"type": "boolean"}
        }
      }
    },
    "raci": {
      "type": "object",
      "properties": {
        "responsible": {"type": "array", "items": {"type": "string"}},
        "accountable": {"type": "array", "items": {"type": "string"}},
        "consulted": {"type": "array", "items": {"type": "string"}},
        "informed": {"type": "array", "items": {"type": "string"}}
      }
    },
    "quality_bars": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["integrity", "reachability", "nonlinearity", "gateways", "style", "determinism", "presentation", "accessibility"]
      }
    }
  }
}
```

---

## End of Migration Instructions

This document contains everything an AI agent needs to execute the v1 → v2 migration. The agent should work sequentially through phases, commit regularly, and escalate to humans only when encountering ambiguities not covered in this specification.

**Estimated Total Time:** 80-120 agent hours across 3 phases
**Final Deliverable:** QuestFoundry v2.0.0 with atomic, composable behavior architecture
