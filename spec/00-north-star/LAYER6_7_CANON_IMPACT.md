# Layer 6/7 Implementation Impact — Canon Workflows

**Audience:** Library authors (Layer 6) and UI/CLI developers (Layer 7).

**Purpose:** Document the implementation requirements for **Canon Transfer** (shared universes) and
**World Genesis** (canon-first worldbuilding) workflows introduced in this spec revision.

**Date:** 2025-11-10

---

## Executive Summary

Three new canon-centric **workflow patterns** were added (documented in WORKING_MODEL.md §12):

1. **World Genesis** — Proactive worldbuilding before plot design (runs Lore Deepening proactively)
2. **Canon Transfer (Export)** — Package canon for sequel/shared universe reuse
3. **Canon Transfer (Import)** — Seed new project with imported canon

These workflow patterns introduce:

- 2 new artifact types (with schemas)
- 3 new protocol intents
- New Hot/Cold merge patterns (invariant vs. mutable canon)
- New validation requirements (conflict detection, timeline anchors, entity registry)

---

## What Changed Across Layers 0-5

### Layer 0 (North Star)

**Modified:**

- `WORKING_MODEL.md` — Added §12 "Workflow Patterns (canon-centric alternatives)"

**Added:**

- `PLAYBOOKS/playbook_world_genesis.md` — One-page playbook for World Genesis pattern
- `PLAYBOOKS/playbook_canon_transfer_export.md` — One-page playbook for export pattern
- `PLAYBOOKS/playbook_canon_transfer_import.md` — One-page playbook for import pattern

**Note:** Canon workflows are **patterns** (different orderings/uses of the 13 standard loops), not
distinct loops themselves. They are documented in WORKING_MODEL.md §12 and guided by playbooks.

### Layer 2 (Dictionary)

**Added:**

- `artifacts/canon_transfer_package.md` — Human-readable template for Canon Transfer Package
- `artifacts/world_genesis_manifest.md` — Human-readable template for World Genesis Manifest

### Layer 3 (Schemas)

**Added:**

- `canon_transfer_package.schema.json` — JSON Schema for Canon Transfer Package
- `world_genesis_manifest.schema.json` — JSON Schema for World Genesis Manifest

### Layer 4 (Protocol)

**Modified:**

- `INTENTS.md` — Added `canon.*` domain with 3 new intents:
  - `canon.transfer.export` (R: LW, A: SR, schema: `canon_transfer_package.schema.json`)
  - `canon.transfer.import` (R: LW, A: SR, schema: `canon_transfer_package.schema.json`)
  - `canon.genesis.create` (R: LW, A: SR, schema: `world_genesis_manifest.schema.json`)

### Layer 5 (Prompts)

**Added:**

- 3 playbooks (see Layer 0 above)

---

## Layer 6 (Libraries) — Implementation Requirements

### 6.1) New Intent Handlers

Implement handlers for the following intents (see `04-protocol/INTENTS.md`):

#### `canon.transfer.export`

**Purpose:** Export canon from completed project as Canon Transfer Package.

**Inputs:**

- Current Cold snapshot (canon packs, codex entries, timeline, entity registry)
- Tagging instructions (invariant/mutable/local per canon element)
- Project metadata (title, slug, snapshot_id)

**Processing:**

1. Load all canon packs from Cold
2. Tag each canon element as invariant (immutable), mutable (extensible), or local (exclude)
3. Extract timeline anchors (T0/T1/T2)
4. Build entity registry (characters, places, factions, items)
5. Select player-safe codex entries for universe-wide transfer
6. Assemble Canon Transfer Package (validate against `canon_transfer_package.schema.json`)
7. Run Gatekeeper validation (integrity, presentation, consistency, schema compliance)

**Output:**

- `canon_transfer_package_<project-slug>.json`
- Gatekeeper report
- TU record (`TU-<date>-LW-CanonTransferExport`)

**Error conditions:**

- Schema validation failure → return validation errors
- Entity reference broken → return missing entity list
- Codex spoilers detected → return affected entry IDs
- Timeline conflicts → return conflicting anchors

#### `canon.transfer.import`

**Purpose:** Import Canon Transfer Package into new project.

**Inputs:**

- Canon Transfer Package (`canon_transfer_package_<source-project>.json`)
- New project seed ideas (user's initial concept)
- Project metadata (title, slug)

**Processing:**

1. Load and validate transfer package against schema
2. **Conflict detection:** Compare imported invariant canon against project seed ideas
   - If conflict detected: escalate to Showrunner with resolution options (reject/revise/downgrade)
3. Merge invariant canon to Cold (mark `immutable: true`, `source: <source-project>`)
4. Seed mutable canon to Hot (mark `immutable: false`, `source: <source-project>`)
5. Import timeline anchors (T0/T1/T2 from source; T3+ for new project)
6. Import entity registry (mark entities with `source: <source-project>`)
7. Import codex baseline (mark `inherited: true`, `source: <source-project>`)
8. Generate constraint documentation for creative roles (invariants, mutables, timeline offsets,
   entity baseline)
9. Run Gatekeeper validation (integrity, conflict resolution, presentation, schema compliance)

**Output:**

- Cold canon packs (invariant canon, `cold-merged`)
- Hot canon packs (mutable canon, `hot-accepted`)
- Codex entries (inherited baseline, `cold-merged`)
- Entity registry (baseline for new project)
- Timeline foundation (T0/T1/T2 from source; T3+ new)
- Constraint documentation (for Plotwright/Scene Smith)
- Gatekeeper report
- TU record (`TU-<date>-LW-CanonTransferImport`)

**Error conditions:**

- Schema validation failure → return validation errors
- Unresolved conflict with project seed → return conflict details + resolution options
- Entity reference broken → return missing entity list
- Timeline confusion → return timeline offset errors

#### `canon.genesis.create`

**Purpose:** Execute World Genesis workflow pattern (proactive worldbuilding using Lore Deepening
before plot).

**Inputs:**

- Project concept (genre, setting, scope)
- Worldbuilding scope (user-defined themes: geography, magic, history, factions, etc.)
- Worldbuilding budget (minimal 1-2h / standard 4-8h / epic 20+h)
- Style preferences (narrative voice, register, motifs)
- Research constraints (if Researcher active: real-world inspirations)

**Processing:**

1. Frame the world (define scope per theme)
2. **Lore Weaver (proactive mode):** Run Lore Deepening proactively (not reactively from hooks) to
   create canon packs for each theme:
   - Geography (regions, landmarks, travel times, invariants)
   - Magic/Metaphysics (rules, limits, sources, invariants)
   - History (timeline depth, major events, invariants)
   - Factions (powers, structures, rivalries, invariants)
   - Culture/Technology (optional: customs, tech baseline)
3. **Researcher (optional):** Corroborate real-world inspirations; attach citations or mark
   `uncorroborated:<risk>`
4. **Codex Curator:** Build codex baseline (player-safe entries, taxonomy, cross-links)
5. **Style Lead:** Define style anchors (voice, dialogue patterns, motifs)
6. **Gatekeeper validation:** Preview validation (Integrity, Style, Presentation)
7. **Lore Weaver stabilization:** Iterate until all themes covered, no contradictions, Gatekeeper
   approves
8. Generate deliverables (canon packs, codex baseline, style addendum, timeline foundation,
   constraint manifest, entity registry)
9. Assemble World Genesis Manifest (validate against `world_genesis_manifest.schema.json`)

**Output:**

- Canon packs (one per theme, `cold-merged` after gatecheck)
- Codex baseline (player-safe entries)
- Style addendum (world voice/register)
- Timeline foundation (T0/T1/T2 anchors)
- Constraint manifest (invariants, affordances, can/cannot for Plotwright)
- Entity registry (characters, places, factions, items)
- World Genesis Manifest (`world_genesis_manifest_<slug>.json`)
- Gatekeeper report
- TU record (`TU-<date>-LW-WorldGenesis`)

**Error conditions:**

- Internal contradictions in genesis canon → return contradiction details
- Codex spoilers → return affected entry IDs
- Style drift → return inconsistency details
- Over-building (unused canon) → warn when canon exceeds likely Plotwright needs

### 6.2) Schema Validators

Implement validators for:

1. **`canon_transfer_package.schema.json`**
   - Validate all required fields (metadata, invariant_canon, mutable_canon, codex_baseline,
     timeline_anchors, entity_registry)
   - Enforce string length constraints (10-600 chars for canon, 3-80 for IDs)
   - Validate enum values (entity types, mutability flags)
   - Validate array min/max items

2. **`world_genesis_manifest.schema.json`**
   - Validate all required fields (project_metadata, worldbuilding_scope, canon_packs_created,
     timeline_foundation, entity_registry, codex_baseline, style_anchors, constraint_manifest,
     iteration_summary)
   - Enforce iteration tracking structure (iteration_num, steps_completed, issues, result)
   - Validate stabilization status

### 6.3) Hot/Cold Merge Logic

Extend merge path logic to handle:

1. **Invariant canon merge to Cold:**
   - Mark canon packs with `immutable: true`, `source: <source-project>`
   - Status: `cold-merged` (cannot be changed)
   - Block any future edits to invariant canon

2. **Mutable canon seed to Hot:**
   - Mark canon packs with `immutable: false`, `source: <source-project>`
   - Status: `hot-accepted` (can be extended via Lore Deepening)
   - Allow extension but not contradiction

3. **World Genesis canon to Cold:**
   - After Gatekeeper approval, merge all genesis canon packs to Cold
   - Mark with `source: world-genesis`, `immutable: true` (unless explicitly marked mutable)

### 6.4) Conflict Detection Engine

Implement conflict detection for Canon Transfer (Import):

**Algorithm:**

1. Parse invariant canon rules from transfer package
2. Parse project seed ideas (user input)
3. For each invariant rule:
   - Check if project seed contradicts invariant
   - Example: Invariant says "Wormhole 3 collapsed"; seed wants "repair Wormhole 3"
4. If conflict detected:
   - Generate conflict report (invariant ID, conflict description, affected project seed element)
   - Suggest resolution options:
     - **Option A:** Reject import (wrong canon baseline)
     - **Option B:** Revise project seed (honor invariant)
     - **Option C:** Downgrade invariant to mutable (coordinate with source project owner)
5. Escalate to Showrunner for resolution

**Output:** Conflict report with resolution options

### 6.5) Entity Registry Management

Implement entity registry CRUD operations:

**Data model:**

```json
{
  "characters": [
    {
      "name": "string (3-80 chars)",
      "role": "string (10-160 chars)",
      "status": "alive|dead|unknown",
      "description": "string (20-400 chars)",
      "source": "string (project slug or 'world-genesis')"
    }
  ],
  "places": [
    {
      "name": "string (3-80 chars)",
      "type": "hub|gate|landmark|other",
      "geography": "string (20-400 chars)",
      "source": "string (project slug or 'world-genesis')"
    }
  ],
  "factions": [
    {
      "name": "string (3-80 chars)",
      "structure": "string (20-300 chars)",
      "influence": "string (20-300 chars)",
      "source": "string (project slug or 'world-genesis')"
    }
  ],
  "items": [
    {
      "name": "string (3-80 chars)",
      "function": "string (20-300 chars)",
      "availability": "string (10-160 chars)",
      "source": "string (project slug or 'world-genesis')"
    }
  ]
}
```

**Operations:**

- **Create:** Add entities during World Genesis or Canon Transfer (Import)
- **Read:** Query entity registry by name, type, source
- **Update:** Extend entity details (only if mutable; block if from invariant canon)
- **Validate:** Ensure all entity references in canon resolve to registry

### 6.6) Timeline Anchor Management

Implement timeline anchor operations:

**Data model:**

```json
{
  "timeline_anchors": [
    {
      "id": "T0|T1|T2|T3|...",
      "description": "string (20-300 chars)",
      "era": "past|recent|now|future",
      "source": "string (project slug or 'world-genesis')"
    }
  ]
}
```

**Operations:**

- **Import from source project:** T0/T1/T2 become baseline
- **Extend for new project:** T3/T4/... for new events
- **Document offsets:** Calculate and display timeline offsets between projects
- **Validate chronology:** Ensure anchors are ordered correctly

### 6.7) Constraint Manifest Generator

Implement constraint documentation generator for Plotwright/Scene Smith:

**Inputs:**

- Invariant canon (list of immutable rules)
- Mutable canon (list of extensible rules)
- Timeline foundation (anchor offsets)
- Entity registry baseline

**Output:**

```markdown
## Invariants (CANNOT CHANGE)

- [List of immutable rules with consequences]

## Mutables (CAN EXTEND)

- [List of extensible rules with extension guidance and constraints]

## Timeline (extend from T3+)

- [Timeline offsets and current anchor]

## Entities (baseline)

- [List of inherited entities with status]

## You CAN:

- [Bulleted list of allowed operations]

## You CANNOT:

- [Bulleted list of forbidden operations with rationale]
```

---

## Layer 7 (UI/CLI) — Implementation Requirements

### 7.1) New Commands/Menu Items

#### Command: `genesis` or `world-genesis`

**Purpose:** Start World Genesis workflow pattern (canon-first worldbuilding using Lore Deepening
proactively).

**User flow:**

1. User invokes command: `qf genesis` or `qf world-genesis`
2. CLI prompts for:
   - Project concept (genre, setting, scope)
   - Worldbuilding themes (checkboxes: geography, magic, history, factions, metaphysics, culture)
   - Worldbuilding budget (dropdown: minimal 1-2h / standard 4-8h / epic 20+h)
   - Style preferences (narrative voice, register, motifs)
   - Optional: Research constraints (real-world inspirations)
3. CLI opens TU: `TU-<date>-LW-WorldGenesis`
4. CLI invokes `canon.genesis.create` intent
5. Display progress:
   - "Creating geography canon..." (with spinner)
   - "Creating magic canon..." (with spinner)
   - "Building codex baseline..." (with spinner)
   - "Running Gatekeeper validation..." (with spinner)
6. On completion:
   - Display summary: "World Genesis complete. Created 5 canon packs, 24 codex entries, 1 style
     addendum."
   - Display deliverables:
     - `world_genesis_manifest_<slug>.json`
     - List of canon packs created
     - Constraint manifest preview
   - Prompt: "Ready to start Story Spark? (y/n)"

**Error handling:**

- Contradiction detected → Display details; prompt to revise or continue
- Codex spoilers → Display affected entries; prompt to revise
- Over-building warning → "Canon exceeds likely Plotwright needs. Continue? (y/n)"

#### Command: `export-canon` or `canon-export`

**Purpose:** Export canon from completed project for sequel/shared universe.

**User flow:**

1. User invokes command: `qf export-canon` or `qf canon-export`
2. CLI validates: Project must be complete (final Binding Run exists)
3. CLI prompts for tagging:
   - Display all canon packs from Cold
   - For each canon element, prompt: "Invariant (immutable) / Mutable (extensible) / Local
     (exclude)?"
   - Provide quick-tag options: "Tag all as invariant", "Tag all as mutable", "Manual tagging"
4. CLI extracts timeline, entity registry, codex baseline
5. CLI invokes `canon.transfer.export` intent
6. Display progress:
   - "Tagging canon elements..." (with progress bar)
   - "Extracting timeline anchors..." (with spinner)
   - "Building entity registry..." (with spinner)
   - "Running Gatekeeper validation..." (with spinner)
7. On completion:
   - Display summary: "Canon export complete. Package contains 15 invariant rules, 8 mutable rules,
     3 timeline anchors, 12 entities."
   - Display deliverable: `canon_transfer_package_<project-slug>.json`
   - Display file size and location
   - Prompt: "Copy package path to clipboard? (y/n)"

**Error handling:**

- Entity reference broken → Display missing entities; prompt to add or exclude
- Codex spoilers → Display affected entries; prompt to revise or exclude
- Timeline conflicts → Display conflicts; prompt to reorder or exclude

#### Command: `import-canon` or `canon-import`

**Purpose:** Import canon from prior project to seed new story.

**User flow:**

1. User invokes command: `qf import-canon` or `qf canon-import`
2. CLI prompts for:
   - Canon Transfer Package path (file picker or manual input)
   - New project seed ideas (multi-line text input)
3. CLI validates package against schema
4. CLI invokes `canon.transfer.import` intent (with conflict detection)
5. If conflicts detected:
   - Display conflict report:
     - "Conflict: Invariant says 'Wormhole 3 collapsed'; your seed wants 'repair Wormhole 3'"
     - "Resolution options:"
       - "A) Reject import (use different canon baseline)"
       - "B) Revise seed (honor invariant; story cannot repair Wormhole 3)"
       - "C) Downgrade invariant to mutable (allow repair; coordinate with source project owner)"
   - Prompt for resolution choice
6. Display progress:
   - "Merging invariant canon to Cold..." (with progress bar)
   - "Seeding mutable canon to Hot..." (with progress bar)
   - "Importing timeline anchors..." (with spinner)
   - "Importing entity registry..." (with spinner)
   - "Generating constraint documentation..." (with spinner)
   - "Running Gatekeeper validation..." (with spinner)
7. On completion:
   - Display summary: "Canon import complete. 15 invariants → Cold, 8 mutables → Hot, timeline
     extends from Y+2."
   - Display deliverables:
     - List of Cold canon packs (invariant)
     - List of Hot canon packs (mutable)
     - Constraint documentation preview
   - Prompt: "View constraint documentation? (y/n)"
   - Prompt: "Ready to start Story Spark? (y/n)"

**Error handling:**

- Schema validation failure → Display validation errors; prompt to fix package or cancel
- Unresolved conflict → Block import; display conflict details; prompt for resolution
- Timeline confusion → Display timeline offset errors; prompt to clarify

### 7.2) UI Enhancements

#### Project Initialization Flow

**Enhancement:** Add canon workflow options to project setup wizard.

**Mockup:**

```
QuestFoundry — New Project Setup

How do you want to start?

○ Story-Driven (Standard)
  Start with Story Spark; plot drives canon discovery.

○ Canon-First (World Genesis)
  Build detailed worldbuilding before plot design.
  Ideal for: epic fantasy, deep sci-fi, rich settings.

○ Shared Universe (Canon Transfer)
  Import canon from a prior project (sequel/franchise).

[Continue]
```

#### Canon Tagging Interface

**Purpose:** Visual interface for tagging canon elements during export.

**Mockup:**

```
Canon Export — Tagging

Tag each canon element as Invariant / Mutable / Local:

Canon Pack: CP-Geography-001 (Geography)
Canon: "Wormhole 3 collapsed in Y-18; irreparable."
Tag: ○ Invariant (immutable)  ○ Mutable (extensible)  ○ Local (exclude)

Canon Pack: CP-Factions-004 (Factions)
Canon: "Toll Syndicate has 5 known council members."
Tag: ○ Invariant (immutable)  ● Mutable (extensible)  ○ Local (exclude)

[Quick Actions]
- Tag all as Invariant
- Tag all as Mutable
- Auto-suggest (AI)

[Continue] [Cancel]
```

#### Conflict Resolution Dialog

**Purpose:** Visual interface for resolving import conflicts.

**Mockup:**

```
Canon Import — Conflict Detected

Conflict #1 of 2:

Invariant Canon (from "The Lighthouse Keeper"):
"Wormhole 3 collapsed in Y-18; irreparable with current technology."

Your Project Seed:
"Harbor master investigates wormhole repair efforts."

Resolution Options:
○ Reject import (use different canon baseline)
○ Revise project seed (honor invariant; change "repair efforts" to "failed repair attempts")
○ Downgrade invariant to mutable (allow repair; requires source project owner approval)

[Resolve] [Cancel Import]
```

#### Constraint Documentation Viewer

**Purpose:** Display imported canon constraints for creative roles.

**Mockup:**

```
Canon Constraints — The Lighthouse Keeper (imported)

✗ INVARIANTS (Cannot Change)
  - Wormhole 3 collapsed in Y-18 (irreparable)
  - Toll Syndicate has 5-tier hierarchy (structure fixed)

✓ MUTABLES (Can Extend)
  - Toll Syndicate member roster (can add members)
  - Dock 7 layout (can expand)

⏱ TIMELINE
  - Source project ends at Y+0
  - New project starts at Y+2 (2 years later)

👥 ENTITIES (Baseline)
  - Kestrel Var (alive, reformed)
  - Ena Roe (alive, promoted)

✓ You CAN:
  - Add new Toll members
  - Expand Dock 7 geography
  - Introduce new characters
  - Explore failed repair attempts

✗ You CANNOT:
  - Repair Wormhole 3 (violates invariant)
  - Change Toll hierarchy (violates invariant)
  - Contradict Kestrel/Ena fates (violates entity registry)

[Export as PDF] [Close]
```

### 7.3) Status Indicators

Add status indicators for canon workflows:

**Project status display:**

```
Project: The Harbor Master
Status: Canon imported from "The Lighthouse Keeper"
  - 15 invariants (Cold)
  - 8 mutables (Hot)
  - Timeline: Y+2 (2 years after source)
Ready for: Story Spark
```

**Canon pack metadata display:**

```
Canon Pack: CP-Geography-001
Source: The Lighthouse Keeper
Mutability: Invariant (cannot change)
Status: cold-merged
```

### 7.4) Help/Documentation

Add help topics for new commands:

**Help text for `qf genesis`:**

```
qf genesis — Start World Genesis workflow pattern

Build detailed worldbuilding before plot design. Ideal for epic fantasy,
deep sci-fi, and rich settings where extensive canon (geography, magic,
history, factions) should be established proactively using Lore Deepening.

Usage:
  qf genesis

Options:
  --budget <minimal|standard|epic>  Set worldbuilding budget
  --themes <list>                   Specify themes (geography,magic,...)

Output:
  - Canon packs (one per theme)
  - Codex baseline (player-safe entries)
  - Style anchors (voice/register)
  - Timeline foundation (T0/T1/T2)
  - Constraint manifest (for Plotwright)
  - world_genesis_manifest_<slug>.json

Next step: Story Spark (plot within canon constraints)

See also: qf help export-canon, qf help import-canon
```

---

## Integration Points

### 7.5) Workflow Transitions

Ensure seamless transitions between canon workflows and standard flows:

**World Genesis → Story Spark:**

- Auto-populate Story Spark with constraint manifest
- Display canon constraints in sidebar during topology design
- Validate Story Spark output against invariants

**Canon Import → Story Spark:**

- Auto-populate Story Spark with imported constraint docs
- Display inherited entities in sidebar for reference
- Block Story Spark changes that violate imported invariants

**Lore Deepening → Canon Export:**

- Track which canon packs are export candidates
- Suggest tagging based on hook sources (story-specific → local; universe-wide → invariant/mutable)

### 7.6) Gatekeeper Integration

Extend Gatekeeper validation UI to handle:

**New validation criteria for canon workflows:**

- **Conflict resolution** (Canon Import only): No unresolved contradictions with project seed
- **Timeline chronology** (all canon workflows): Anchors are ordered correctly
- **Entity reference integrity** (all canon workflows): All references resolve to entity registry

**Validation report enhancements:**

```
Gatekeeper Report — Canon Transfer (Import)

✅ Integrity: All entity references resolve
✅ Conflict Resolution: No unresolved contradictions
✅ Presentation: Codex baseline is player-safe
✅ Schema Compliance: Package validates against schema

Status: APPROVED

[View Details] [Close]
```

---

## Testing Requirements

### Layer 6 Testing

**Unit tests:**

- Schema validators (valid/invalid packages)
- Conflict detection algorithm (conflict/no-conflict cases)
- Entity registry CRUD operations
- Timeline anchor validation
- Constraint manifest generator

**Integration tests:**

- Full World Genesis flow (minimal/standard/epic budgets)
- Full Canon Transfer (Export) flow (invariant/mutable/local tagging)
- Full Canon Transfer (Import) flow (with/without conflicts)
- Hot/Cold merge logic (invariant→Cold, mutable→Hot)
- Gatekeeper validation for canon workflows

### Layer 7 Testing

**UI/UX tests:**

- Project initialization wizard (all 3 workflow options)
- Canon tagging interface (keyboard/mouse navigation)
- Conflict resolution dialog (all 3 resolution options)
- Constraint documentation viewer (display/export)
- Command-line flows (all 3 commands with various inputs)

**Edge cases:**

- Empty timeline (no T0/T1/T2 in source project)
- Massive entity registry (1000+ entities)
- Conflicting invariants (multiple conflicts in single import)
- Downgrade invariant to mutable (coordination with source project)

---

## Migration & Backward Compatibility

**For existing projects:**

- Projects without canon workflows: No impact; standard Story Spark flow unchanged
- Projects with existing canon packs: Can retroactively export via `canon-export` command (all canon
  tagged as mutable by default; user refines)

**For existing libraries:**

- Layer 6: New intents are additive; existing intents unchanged
- Hot/Cold merge logic: Extended, not replaced; existing merge paths still valid

**For existing UI:**

- Layer 7: New commands are additive; existing commands unchanged
- Project initialization wizard: Enhanced with new options; default remains Story-Driven

---

## Performance Considerations

**World Genesis:**

- Epic budget (20+ hours, 100+ codex entries) may require progress persistence (resume if
  interrupted)
- Canon pack creation should be parallelizable where possible
- Codex baseline generation should be incremental (build taxonomy as entries are added)

**Canon Transfer (Import):**

- Large transfer packages (10+ MB) should stream validation (don't load entire JSON into memory)
- Conflict detection should be batched (check all invariants before escalating to user)
- Entity registry import should deduplicate entities (avoid redundant entries)

**Canon Transfer (Export):**

- Large Cold snapshots (1000+ canon packs) should paginate tagging UI (don't display all at once)
- Entity registry extraction should index by reference (fast lookup)

---

## Open Questions for Layer 6/7 Authors

1. **Tagging UI:** Should canon element tagging support batch operations (e.g., "Tag all geography
   as invariant, all character arcs as local")? Or is per-element tagging sufficient?

2. **Conflict resolution:** Should Option C (downgrade invariant to mutable) trigger automated
   coordination with source project owner (e.g., email notification, shared workspace)? Or is manual
   coordination expected?

3. **Entity registry deduplication:** If new project introduces an entity with the same name as an
   imported entity (e.g., both projects have "Guard Captain"), how should the UI handle this?
   Namespace by source project? Prompt for rename?

4. **Timeline anchor visualization:** Would a timeline graph (T0 → T1 → T2 → T3 → ...) be helpful in
   the UI? Or is text-based display sufficient?

5. **World Genesis progress persistence:** If epic budget (20+ hours) is interrupted, should the
   loop support resume-from-checkpoint? Or is full re-run expected?

6. **Export format:** Should Canon Transfer Packages support additional formats (YAML, TOML) beyond
   JSON? Or is JSON sufficient?

7. **Constraint manifest:** Should constraint documentation be version-controlled alongside canon
   packs? Or is it derived on-demand from current Cold state?

8. **Role-based views:** Should constraint documentation have role-specific views (e.g., Plotwright
   sees affordances/constraints; Scene Smith sees entity details; Codex Curator sees taxonomy)?

---

## References

**Normative:**

- `00-north-star/WORKING_MODEL.md` (§12 Workflow Patterns)
- `03-schemas/canon_transfer_package.schema.json`
- `03-schemas/world_genesis_manifest.schema.json`
- `04-protocol/INTENTS.md`
- `02-dictionary/loop_names.md` (defines the 13 standard loops)

**Informative:**

- `00-north-star/PLAYBOOKS/playbook_world_genesis.md` (guides World Genesis pattern)
- `00-north-star/PLAYBOOKS/playbook_canon_transfer_export.md` (guides export pattern)
- `00-north-star/PLAYBOOKS/playbook_canon_transfer_import.md` (guides import pattern)
- `02-dictionary/artifacts/canon_transfer_package.md`
- `02-dictionary/artifacts/world_genesis_manifest.md`

---

**TL;DR for Layer 6:** Implement 3 new intent handlers (`canon.transfer.export`,
`canon.transfer.import`, `canon.genesis.create`), 2 schema validators, conflict detection engine,
entity registry/timeline management, and constraint manifest generator.

**TL;DR for Layer 7:** Add 3 new commands (`genesis`, `export-canon`, `import-canon`), enhance
project initialization wizard, implement canon tagging UI, conflict resolution dialog, and
constraint documentation viewer.
