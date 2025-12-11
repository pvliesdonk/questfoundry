# Layer 3 — JSON Schemas (Generated from Layer 2)

> **Purpose:** Machine-readable schemas derived from enriched Layer 2 artifact templates.

---

## Status

✅ **COMPLETE** — All schemas generated and validated

- **Current version:** `schemas-v0.2.0` (2025-11-05)
- **Total schemas:** 28 JSON Schema files (Draft 2020-12)
  - 22 artifact schemas (from Layer 2 templates)
  - 6 system schemas (5 Cold SoT manifests + 1 Hot manifest)
- **Source:** Layer 2 artifact templates in `02-dictionary/artifacts/*.md`
- **Note:** Protocol envelope schema (`envelope.schema.json`) lives in `04-protocol/`, not here
- **Validation:** All schemas pass JSON Schema Draft 2020-12 meta-validation
- **Published:** Canonical URLs at `https://questfoundry.liesdonk.nl/schemas/`

See [CHANGELOG.md](./CHANGELOG.md) for version history.

---

## Schema Generation Process

### 1. Source Material

Each enriched template contains **HTML constraint comments** with field metadata:

```html
<!-- Field: Status | Type: enum | Required: yes | Taxonomy: Hook Status Lifecycle (taxonomies.md §2) -->
<!-- Allowed values: proposed | accepted | in-progress | resolved | canonized | deferred | rejected -->
```

### 2. Extraction Pattern

For each field, extract:

- **Field name** — Property name in schema
- **Type** — JSON Schema type (string, enum, array, object, etc.)
- **Required** — Boolean (yes/no)
- **Format** — Pattern, date format, or reference format
- **Enum values** — For controlled vocabularies
- **Validation rules** — Length, pattern, cross-field constraints
- **Description** — Human-readable explanation from prose

### 3. Output Format

JSON Schema Draft 2020-12 with:

- `$schema`, `$id`, `title`, `description`
- `type: "object"`
- `properties: {}` — All fields
- `required: []` — Required field names
- `definitions: {}` — Reusable types (role names, dates, IDs)

---

## Validation Feedback Process

**Architectural Risk:** The L2→L3 translation boundary is a deliberate design choice that
prioritizes human readability (Layer 2 templates) over machine validation (Layer 3 schemas). This
creates a governance challenge: **how do schema validation failures get surfaced and corrected in
the human-readable Layer 2 source?**

### When Validation Fails

Validation failures can occur at two stages:

1. **Schema generation** — L2 template constraints are ambiguous or incomplete, preventing schema
   generation
2. **Artifact validation** — An artifact instance fails validation against its generated schema

### Feedback Loop

**Schema generation failures:**

1. Schema generator (human or tool) identifies inconsistency in L2 template
2. Issue is logged with specific file, field, and constraint reference
3. L2 template is corrected with clarified constraints
4. Schema is regenerated and validated against JSON Schema meta-schema
5. Example artifacts are tested against new schema

**Artifact validation failures:**

1. Artifact fails validation (e.g., role produces invalid `hook_card.json`)
2. Validation report references schema constraint that was violated
3. Two possible root causes:
   - **Artifact is wrong** — Role prompt or implementation needs correction
   - **Schema is wrong** — L2 template constraint was too strict or unclear
4. If schema is wrong, correction flows back to L2 template
5. Schema is regenerated and all existing artifacts are re-validated

### Governance Checkpoint

The **Gatekeeper** role enforces this feedback loop:

- Before merging artifact changes to Cold, GK validates against L3 schemas
- If validation fails, GK blocks merge and requests either:
  - Artifact correction (if artifact violated correct schema), or
  - L2 template review (if schema constraint is unclear or incorrect)

This ensures the L2 "meaning" layer and L3 "validation" layer remain aligned.

### Tools and Automation

The `spec-tools` validation CLI automates this process:

- `qfspec-validate-schemas` — Validate all schemas against JSON Schema meta-schema
- `qfspec-validate-artifact` — Validate artifact instance against its schema
- CI integration runs both checks on every commit

---

## Schema Template

See `SCHEMA_TEMPLATE.json` for the standard pattern.

See `EXTRACTION_GUIDE.md` for step-by-step extraction instructions.

See `hook_card.schema.json` for a complete reference example.

---

## File Naming Convention

**Pattern:** `{artifact_name}.schema.json`

**Examples:**

- `hook_card.schema.json`
- `tu_brief.schema.json`
- `gatecheck_report.schema.json`
- `canon_pack.schema.json`
- etc.

---

## Schema Index (28 Total)

### Artifact Schemas (22)

**Core Workflow:**

- ✅ `hook_card.schema.json` — Hook tracking and routing
- ✅ `tu_brief.schema.json` — Trace Unit (work unit) tracking

**Creation & Content:**

- ✅ `canon_pack.schema.json` — Canonical story facts
- ✅ `canon_transfer_package.schema.json` — Canon import/export for cross-project sharing
- ✅ `codex_entry.schema.json` — Player-facing encyclopedia
- ✅ `style_addendum.schema.json` — Style and voice guidelines
- ✅ `edit_notes.schema.json` — Editorial feedback

**Research & Planning:**

- ✅ `research_memo.schema.json` — Factual research documentation
- ✅ `shotlist.schema.json` — Individual illustration specs
- ✅ `cuelist.schema.json` — Audio cue specifications
- ✅ `art_plan.schema.json` — Illustration planning
- ✅ `audio_plan.schema.json` — Audio production planning

**Localization:**

- ✅ `language_pack.schema.json` — Localization translations
- ✅ `register_map.schema.json` — Language register specs

**Quality & Export:**

- ✅ `gatecheck_report.schema.json` — Quality validation reports
- ✅ `view_log.schema.json` — Export metadata
- ✅ `front_matter.schema.json` — Book front matter
- ✅ `pn_playtest_notes.schema.json` — Player-Narrator feedback

**Project Metadata:**

- ✅ `project_metadata.schema.json` — Project configuration
- ✅ `world_genesis_manifest.schema.json` — World-building foundation (Layer 0 kickoff)
- ✅ `art_manifest.schema.json` — Complete art asset inventory
- ✅ `style_manifest.schema.json` — Typography and style settings

### System Schemas (6)

**Cold Source of Truth:**

- ✅ `cold_manifest.schema.json` — Top-level file index with SHA-256 hashes
- ✅ `cold_book.schema.json` — Story structure and bibliographic metadata
- ✅ `cold_art_manifest.schema.json` — Asset mappings with provenance
- ✅ `cold_fonts.schema.json` — Font file mappings
- ✅ `cold_build_lock.schema.json` — Tool version pinning

**Hot Discovery Space:**

- ✅ `hot_manifest.schema.json` — Master index for Hot discovery space

> **Note:** The protocol envelope schema (`envelope.schema.json`) lives in `04-protocol/`, not in
> this directory.

---

## Validation

Each schema should:

- ✅ Pass JSON Schema meta-validation
- ✅ Reference taxonomies from Layer 2
- ✅ Include all required fields from enriched template
- ✅ Include all enum values from constraint comments
- ✅ Include format patterns (dates, IDs)
- ✅ Include descriptions from template prose

---

## Usage

Schemas will be used for:

- **Validation tooling** — CLI validators for artifacts
- **API design** — REST/GraphQL endpoint definitions
- **UI generation** — Form builders from schemas
- **Documentation** — Auto-generated field reference

---

## Cross-References

- **Source templates:** `../02-dictionary/artifacts/*.md` (enriched with HTML constraint comments)
- **Taxonomies:** `../02-dictionary/taxonomies.md` (enumerations)
- **Field registry:** `../02-dictionary/field_registry.md` (field catalog)
- **Validation rules:** Embedded in Layer 2 artifact template comments
- **Changelog:** `./CHANGELOG.md` (version history)

---

**Created:** 2025-10-30 **Method:** Automated extraction from enriched Layer 2 templates
