# Changelog — Layer 5 Prompts

All notable changes to Layer 5 prompts will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-11-10

### Added

- **SCHEMA_INDEX.json Updates:**
  - Added `canon_transfer_package` schema entry
    - Path: `03-schemas/canon_transfer_package.schema.json`
    - Roles: Lore Weaver
    - Intents: `canon.transfer.export`, `canon.transfer.import`
  - Added `world_genesis_manifest` schema entry
    - Path: `03-schemas/world_genesis_manifest.schema.json`
    - Roles: Lore Weaver
    - Intents: `canon.genesis.create`
  - Updated version to 0.3.0
  - Updated generated date to 2025-11-10
  - Total schemas: 28 (was 26)

### Context

This release adds schema index support for canon-centric workflows. The two new schemas enable Canon
Transfer (export/import canon between projects for shared universes/sequels) and World Genesis
(proactive worldbuilding before plot design). These schemas are now available to all prompts that
reference SCHEMA_INDEX.json for validation and intent handling.

## [0.2.0] - 2025-11-06

### Added

#### Schema Validation Enforcement (Complete Implementation)

**BREAKING CHANGE:** Schema validation is now **mandatory** for all JSON artifacts. Agents producing
artifacts without validation will be blocked at merge.

**Motivation:** Post-mortem analysis revealed that agents consistently failed to validate artifacts
despite schemas being available. Root cause: Document existence ≠ agent compliance. Solution:
Three-layer enforcement with file ordering and hard gates.

---

#### Phase 1: Validation Contract & Schema Index

**Validation Contract** (`_shared/validation_contract.md`)

- **Purpose:** Non-negotiable validation requirements for all roles
- **Position:** File #1 in all upload kits (LLMs read files sequentially)
- **Content:**
  - Preflight protocol (mandatory before producing artifacts)
  - Hard gate enforcement (validation failures STOP workflow)
  - Schema discovery via SCHEMA_INDEX.json
  - Integrity verification (SHA-256 checksums)
  - validation_report.json format specification
  - Role-specific schema mappings
  - Troubleshooting guide

**Schema Index** (`SCHEMA_INDEX.json`)

- **Purpose:** Canonical registry of all 26 Layer 3 schemas
- **Position:** File #2 in all upload kits
- **Content:** For each schema:
  - `$id` - Canonical URL for `"$schema"` field
  - `path` - Relative path to schema file
  - `draft` - JSON Schema draft version (2020-12)
  - `sha256` - Integrity checksum
  - `roles` - Roles that produce this artifact type
  - `intent` - Protocol intents that use this schema

**Index Generator** (`spec-tools/src/.../generate_schema_index.py`)

- Automated SCHEMA_INDEX.json generation
- Computes SHA-256 hashes for all schemas
- Extracts metadata from schema files
- Maps schemas to roles and protocol intents
- CLI command: `qfspec-generate-schema-index`
- Added to pyproject.toml

**Upload Kit Updates:**

- All 14 manifests updated to place validation files first
- `upload_kits.py` updated to preserve manifest order in zips
- File ordering enforced: validation_contract.md → SCHEMA_INDEX.json → other files
- Orchestration kits: 36 → 38 files
- Full standalone: 23 → 25 files
- Minimal standalone: 10 → 12 files
- All Gemini kits maintained 10-file limit (removed human_interaction.md where needed)

---

#### Phase 2: Role Prompt Validation Sections

**All 15 role system_prompts updated** with "Output Validation (Required)" section:

**Roles with explicit schema mappings (6):**

- **Plotwright** → hook_card
- **Showrunner** → hot_manifest, project_metadata
  - Added orchestration responsibility: verify all roles produce validation_report.json
- **Gatekeeper** → gatecheck_report
  - Added gate integrity note: invalid reports undermine quality enforcement
- **Book Binder** → cold_book, cold_manifest
- **Codex Curator** → codex_entry
- **Illustrator** → cold_art_manifest

**Roles with generic validation (9):**

- Scene Smith, Style Lead, Lore Weaver, Player Narrator
- Art Director, Audio Director, Audio Producer
- Translator, Researcher
- Instructed to check SCHEMA_INDEX.json for their schemas

**Each validation section includes:**

1. Reference to validation_contract.md (file #1)
2. 6-step preflight/validation protocol
3. Schema discovery instructions
4. Hard gate enforcement (STOP on failure)
5. validation_report.json format
6. Role-specific schema listings (where applicable)
7. Validation workflow example

---

#### Phase 3: Loop Playbook Validation Checkpoints

**All 13 loop playbooks updated** with "Validation Requirements (All Artifacts)" section:

**Updated playbooks:**

- archive_snapshot, art_touch_up, audio_pass, binding_run
- codex_expansion, gatecheck, hook_harvest, lore_deepening
- narration_dry_run, post_mortem, story_spark, style_tune_up
- translation_pass

**Each validation section specifies:**

1. Reference to validation_contract.md
2. 7-step validation protocol for artifact-producing steps:
   - Producer looks up schema in SCHEMA_INDEX.json
   - Producer runs preflight protocol
   - Producer validates and emits validation_report.json
   - **If validation fails: STOP loop, escalate to Showrunner**
   - Showrunner verifies both files exist before handoff
   - Showrunner checks `"valid": true` in validation_report.json
   - Only then proceed to next step
3. Hard gate enforcement at loop level
4. Showrunner responsibility to verify validation_report.json at every handoff

**Placement:** Inserted after "## Inputs" and before "## Procedure" for visibility

---

#### Phase 4: Gatekeeper Schema Validation Quality Bar

**New quality bar:** "Schema Validation Quality Bar (All Artifacts)"

**Position:** Added to Gatekeeper prompt after Checklist, before Cold Source of Truth Validation

**Requirements:**

Before issuing ANY `gate.decision` with `pass`, Gatekeeper MUST verify:

1. All JSON artifacts in TU have corresponding validation_report.json
2. All validation_report.json files show `"valid": true`
3. All validation_report.json files have empty `"errors": []`
4. All artifacts have `"$schema"` field pointing to canonical schema $id

**Validation Audit Protocol:**

- 7-step audit for each artifact in TU
- Check artifact has `"$schema"` field
- Verify validation_report.json structure
- BLOCK merge if any validation missing or failed
- Provide specific remediation for each failed artifact

**Hard Stops:**

- Missing validation_report.json → BLOCK
- `"valid": false` in validation_report.json → BLOCK
- Non-empty `"errors"` array → BLOCK
- Missing `"$schema"` field in artifact → BLOCK

**Enforcement:**

- List ALL artifacts with validation issues (not just first)
- Provide clear remediation for each
- Escalate to Showrunner with role assignments

**Integration:** Schema validation is prerequisite for Determinism Bar checks

---

### Changed

**Upload Kit File Counts:**

- orchestration-complete.zip: 36 → 38 files (+validation_contract.md, +SCHEMA_INDEX.json)
- full-standalone.zip: 23 → 25 files (+validation_contract.md, +SCHEMA_INDEX.json)
- minimal-standalone.zip: 10 → 12 files (+validation_contract.md, +SCHEMA_INDEX.json)
- gemini-orchestration-1-foundation.zip: 9 → 10 files (removed human_interaction.md to stay within
  limit)
- gemini-minimal-standalone.zip: 10 → 10 files (removed human_interaction.md)
- gemini-full-standalone-1.zip: 10 → 10 files (removed human_interaction.md + initialization.md)

**SCHEMA_INDEX.json Version:** 0.2.0

---

### Technical Details

**File Ordering Strategy:**

Critical insight: LLMs read uploaded files sequentially. File order determines agent behavior.

**Enforced order in all kits:**

1. validation_contract.md (file #1) ← Rules loaded FIRST
2. SCHEMA_INDEX.json (file #2) ← Discovery mechanism
3. Shared patterns ← Cross-role standards
4. Role/loop content ← Builds on validation foundation

**Implementation:** `upload_kits.py` modified to preserve manifest order when creating zips

**Three-Layer Enforcement:**

1. **Foundation Layer:** validation_contract.md loaded first sets hard gates
2. **Role Layer:** Each role knows to validate their artifacts
3. **Orchestration Layer:** Showrunner enforces validation_report.json at every handoff
4. **Quality Gate Layer:** Gatekeeper audits ALL artifacts before merge

**Result:** 0% → 100% validation compliance (expected)

---

### Migration Notes

**For existing v0.1.0 users:**

1. **Immediate impact:** Agents will see validation requirements as first loaded file
2. **Expected behavior change:**
   - Agents will perform preflight checks before producing artifacts
   - Agents will validate artifacts and emit validation_report.json
   - Invalid artifacts will cause workflow STOP (hard gate)
3. **Compatibility:** v0.1.0 kits remain available for backward compatibility
4. **Migration path:**
   - Test with v0.2.0 kits in non-production workflows
   - Verify agents produce validation_report.json correctly
   - Switch production workflows once validated

**For new users:**

- Start with v0.2.0 kits (includes validation enforcement)
- Follow validation workflow in USAGE_GUIDE.md
- Expect validation_report.json alongside all artifacts

---

### Documentation Updates

**USAGE_GUIDE.md:**

- Added "Schema Validation Workflow" section (150+ lines)
- Explains preflight protocol, validation workflow, file ordering importance
- Provides validation examples and troubleshooting

**Implementation Plan:**

- `docs/post_mortems/2025-11-06_validation_enforcement_implementation_plan.md`
- Complete 5-phase implementation plan (Phases 1-4 complete)
- Critical file ordering rules
- Success metrics and timeline

**Post-Mortem:**

- `docs/post_mortems/2025-11-06_schema_enforcement_and_validation_contracts.md`
- Root cause analysis of validation failures
- Proposed solutions and enforcement mechanisms

---

### Success Criteria

**Achieved:**

- ✅ validation_contract.md created and placed as file #1
- ✅ SCHEMA_INDEX.json generated with 26 schemas and SHA-256 hashes
- ✅ All 15 role prompts updated with validation sections
- ✅ All 13 loop playbooks updated with validation checkpoints
- ✅ Gatekeeper enhanced with schema validation quality bar
- ✅ File ordering enforced in all 14 upload kits
- ✅ All kits rebuild successfully

**Expected outcome:** 100% validation compliance for agents using v0.2.0 kits

---

### Breaking Changes

**BREAKING:** Schema validation is now mandatory. Agents producing artifacts without:

- `"$schema"` field in artifacts
- validation_report.json with `"valid": true`

Will be **blocked at merge** by Gatekeeper.

**Migration required:** Update any custom workflows to include validation steps.

---

## [0.1.0] - 2025-11-06

### Added

#### Loop-Focused Architecture (commits a300224, 428140c)

Complete transformation of Layer 5 from role-focused to loop-focused architecture, making loop
playbooks the primary executable units.

**13 Loop Playbooks** (`loops/*.playbook.md`)

Each playbook contains complete procedures with:

- Purpose and activation criteria
- RACI matrix (from Layer 1)
- Complete procedure with message sequences (Layer 0 + Layer 4)
- Deliverables and success criteria
- Failure modes and remedies
- Schema references to Layer 3
- Validation examples with complete message flows

**Playbooks by category:**

- **Discovery:** Story Spark, Hook Harvest, Lore Deepening
- **Refinement:** Codex Expansion, Style Tune-up
- **Asset Production:** Art Touch-up, Audio Pass, Translation Pass
- **Quality & Export:** Binding Run, Narration Dry-Run, Gatecheck, Post-Mortem, Archive Snapshot

**15 Role Adapters** (`role_adapters/*.adapter.md`)

Thin interface specs (50-100 lines each) optimized for multi-role orchestration:

- Core expertise summary
- Protocol intents handled (receives/sends)
- Loop participation references
- Efficient context usage

**Roles:** Showrunner, Plotwright, Scene Smith, Style Lead, Gatekeeper, Book Binder, Player
Narrator, Lore Weaver, Codex Curator, Researcher, Art Director, Illustrator, Audio Director, Audio
Producer, Translator

**Modular Showrunner** (5 focused modules)

Split 213-line monolith into focused modules with single responsibility:

- `system_prompt.md` (110 lines) - Index and navigation
- `loop_orchestration.md` (100 lines) - Execute playbooks
- `manifest_management.md` (102 lines) - Hot/Cold operations
- `initialization.md` (223 lines) - 7-step project setup
- `protocol_handlers.md` (230 lines) - Message validation

Benefits: Clear navigation, cross-references, maintainability

**Enhanced Role Prompts**

All 15 full role prompts (200-300 lines each) enhanced with:

- "Loop Participation" section
- References to loop playbooks with relative paths
- Preserved all existing domain expertise content
- Use case: Standalone work, learning, ChatGPT sessions

**Documentation**

- `README.md` - Architecture overview, dual-format strategy
- `USAGE_GUIDE.md` - Loop playbook workflows, format selection
- `ARCHITECTURE.md` (NEW) - Comprehensive design rationale (60+ sections)
  - Problem statement, solution principles
  - Component descriptions, workflow scenarios
  - Benefits analysis, migration path
  - Future enhancements

#### Upload Kits and Distribution (commit 426f2b9)

**14 Upload Kits** for ChatGPT, Claude, and Gemini

**Orchestration Mode** ⭐ (NEW - recommended for production)

- `orchestration-complete.zip` (36 files)
  - 4 shared patterns
  - 4 showrunner modules
  - 13 loop playbooks
  - 15 role adapters

**Gemini Orchestration Splits** (10-file limit compliant)

- `gemini-orchestration-1-foundation.zip` (9 files) - Shared + showrunner + SR adapter
- `gemini-orchestration-2-playbooks.zip` (10 files) - Core loop playbooks
- `gemini-orchestration-3-playbooks-extra.zip` (3 files) - Additional loops
- `gemini-orchestration-4-adapters-core.zip` (10 files) - Core role adapters
- `gemini-orchestration-5-adapters-extra.zip` (4 files) - Optional role adapters

**Standalone Mode** (traditional)

- `minimal-standalone.zip` (10 files) - Quick start with core roles
- `optional-standalone.zip` (9 files) - Additional roles (PN, LW, CC, etc.)
- `full-standalone.zip` (23 files) - All role prompts + showrunner modules

**Gemini Standalone Splits** (10-file limit compliant)

- `gemini-minimal-standalone.zip` (10 files)
- `gemini-optional-standalone.zip` (9 files)
- `gemini-full-standalone-1.zip` (10 files) - Shared + showrunner + 1 role
- `gemini-full-standalone-2.zip` (10 files) - 10 role prompts
- `gemini-full-standalone-3.zip` (3 files) - 3 role prompts

**Build Tools**

- `spec-tools/src/questfoundry_spec_tools/upload_kits.py` - Automated kit building
- Preserves original directory structure in zips (prevents filename collisions)
- Command: `uv run qfspec-build-kits`
- Output: `dist/upload_kits/*.zip`

**Documentation**

- `upload_kits/README.md` - Complete kit descriptions, platform guidance, quick starts
- `USAGE_GUIDE.md` - Updated to emphasize orchestration mode as recommended
- Loop playbook workflow examples
- Clear guidance on mode selection

**Manifests** (10 new files in `05-prompts/upload_kits/manifests/`)

- `orchestration-complete.list` (36 files)
- `full-standalone.list` (23 files)
- 8 Gemini split manifests (respecting 10-file limits)

### Features

#### Dual-Format Strategy

**Orchestration Mode** (AI-coordinated) ⭐ **RECOMMENDED**

- Showrunner loads loop playbook + role adapters
- Single-source-of-truth procedures (1 playbook vs 7 role prompts)
- 70% context reduction (adapters: 50-100 lines vs full prompts: 200-300 lines)
- Efficient context usage (~1000 lines vs ~3000 lines)
- Role interchangeability (swap implementations without changing playbooks)
- Clear RACI matrix and coordination

**Standalone Mode** (human-led)

- Full role prompts with complete domain expertise
- Human coordinates workflow via prompts
- Best for learning, exploration, single-role tasks

#### Architecture Benefits

**Single Source of Truth**

- Loop procedures defined once in playbook
- Eliminates duplication across role prompts
- Update loop once, all roles benefit

**Maintainability**

- Clear modular structure
- Single responsibility per component
- Easy to understand and update

**Role Interchangeability**

- Roles implement standard interface from adapters
- Swap implementations without changing playbooks
- Clear separation of concerns

**Clear Orchestration**

- Showrunner loads playbook and coordinates
- Roles respond to protocol intents
- RACI matrix defines responsibilities

**Testability**

- Validation examples for all 13 loops
- Complete message flows included
- Realistic payloads matching Layer 3 schemas

**Efficient Context**

- Adapters: 50-100 lines per role
- Full prompts: 200-300 lines per role
- Load only what's needed for specific loop

#### Platform Support

**ChatGPT**

- Single upload with `orchestration-complete.zip` (36 files)
- Or `minimal-standalone.zip` (10 files) for quick start
- Supports zips and individual files

**Claude**

- Handles multiple attachments well
- Individual files preferred for better grounding
- `orchestration-complete.zip` also supported

**Gemini**

- 10-file limit per zip respected
- 5 orchestration splits or 5 standalone splits
- Upload in sequence: foundation → playbooks → adapters

### Technical Details

#### Protocol Compliance

- Layer 4 envelope v0.2.1 compliant
- All messages follow ENVELOPE.md specification
- Proper context fields (hot_cold, snapshot, TU, loop)
- Traceability via correlation_id and refs

#### Schema Integration

- Cold SoT validation via Layer 3 schemas v0.2.0
- Canonical URLs: `https://questfoundry.liesdonk.nl/schemas/`
- No upload needed (referenced directly)
- Gatekeeper and Book Binder validate against schemas

#### Shared Patterns

4 cross-role patterns included in all kits:

- `context_management.md` - Hot/Cold separation, snapshot references
- `safety_protocol.md` - PN boundaries, content hygiene
- `escalation_rules.md` - When to request human intervention
- `human_interaction.md` - Question batching, timeout handling

### Changed

#### Upload Kit Build Process

**Before:**

- Flattened filenames (e.g., `showrunner.md`)
- Filename collisions (multiple `system_prompt.md` files)
- No orchestration kits
- Manual zip creation

**After:**

- Preserves directory structure (e.g., `05-prompts/showrunner/system_prompt.md`)
- No collisions when extracting zips
- Automated orchestration kit building
- Command: `uv run qfspec-build-kits`

#### Documentation Focus

**Before:**

- Emphasized standalone mode
- Role-focused workflows
- Limited guidance on multi-role coordination

**After:**

- Emphasizes orchestration mode as recommended for production
- Loop-focused workflows with clear procedures
- Comprehensive guidance on mode selection
- Platform-specific instructions (ChatGPT, Claude, Gemini)

### Files Modified/Created

**Total: 51 files changed across 3 commits**

**Commit a300224 (Loop playbooks):**

- 13 loop playbooks created
- 13 validation example flows created
- 26 files added

**Commit 428140c (Dual formats):**

- 15 role adapters created
- 15 role prompts enhanced (Loop Participation sections)
- 5 showrunner modules (4 new, 1 rewritten)
- 3 documentation files (README.md, USAGE_GUIDE.md, ARCHITECTURE.md)
- 38 files modified/created

**Commit 426f2b9 (Upload kits):**

- `upload_kits.py` rewritten (preserve directory structure)
- `upload_kits/README.md` complete rewrite
- `USAGE_GUIDE.md` updated (emphasize orchestration)
- 10 new manifest files created
- 13 files modified/created

### Version Alignment

**Prompts Version:** v0.1.0 **Compatible with:**

- schemas-v0.2.0 (Layer 3)
- protocol-v0.2.1 (Layer 4)

**Dependencies:**

- Schemas: All validation via canonical URLs
- Protocol: Envelope v0.2.1 for all messages
- Layer 1: RACI matrices for all loops
- Layer 0: Workflows and sequences

---

## Migration Notes

### From Pre-Loop Architecture

**If you were using role prompts before commit a300224:**

1. **Keep using standalone mode** - Full role prompts still available and fully supported
2. **Try orchestration mode** - 70% context savings, clearer coordination
3. **Update Showrunner** - Now modular (5 files instead of 1)
4. **Use loop playbooks** - Single-source-of-truth procedures

**Breaking changes:** None - all previous workflows still supported via standalone mode

### Upload Kit Changes

**If you were using old upload kits:**

1. **Zips now preserve paths** - Extract with `unzip` to see directory structure
2. **New orchestration kits available** - Recommended for production
3. **Build locally** - `uv run qfspec-build-kits` for fresh kits
4. **Gemini splits provided** - 10-file limit respected

**Breaking changes:** Zip internal structure changed (but content identical)

---

## Release Notes

This is the **first versioned release** of Layer 5 prompts, representing the completion of the
loop-focused architecture transformation across three major commits:

1. **a300224** - Added 13 loop playbooks as primary executable units
2. **428140c** - Added dual-format strategy (full prompts + role adapters)
3. **426f2b9** - Aligned upload kits with new architecture

### Key Achievements

✅ **Loop-focused architecture** - Single-source-of-truth procedures ✅ **Dual-format strategy** -
Full prompts for standalone, adapters for orchestration ✅ **70% context reduction** - Efficient
multi-role coordination ✅ **14 upload kits** - Optimized for ChatGPT, Claude, Gemini ✅ **Complete
documentation** - Usage guides, architecture docs, examples ✅ **Protocol compliance** - Layer 4
envelope v0.2.1 ✅ **Schema validation** - Layer 3 schemas v0.2.0

### Recommended Usage

**For production workflows:** Use **orchestration mode** with `orchestration-complete.zip`

- 70% context savings
- Clear coordination via loop playbooks
- Role interchangeability

**For learning/exploration:** Use **standalone mode** with `minimal-standalone.zip`

- Complete role prompts
- Human-led workflows
- Easier to understand

### Quick Start

**ChatGPT/Claude:**

```bash
# Download orchestration kit
wget https://github.com/pvliesdonk/questfoundry-spec/releases/download/prompts-v0.1.0/orchestration-complete.zip

# Upload to platform and prompt:
"Load the Story Spark playbook from loops/ and execute it for a 3-scene mystery."
```

**Gemini:**

```bash
# Download all 5 orchestration splits
wget https://github.com/pvliesdonk/questfoundry-spec/releases/download/prompts-v0.1.0/gemini-orchestration-{1..5}-*.zip

# Upload in sequence: foundation → playbooks → adapters
# Then prompt: "Load Story Spark playbook and execute it."
```

See `05-prompts/USAGE_GUIDE.md` for complete instructions.

---

## Future Versions

Future releases will follow semantic versioning:

- **PATCH** (0.1.X) - Bug fixes, documentation updates
- **MINOR** (0.X.0) - New loops, new roles, enhanced features (backward compatible)
- **MAJOR** (X.0.0) - Breaking changes to architecture or protocol alignment

Track development in the [main repository](https://github.com/pvliesdonk/questfoundry-spec).
