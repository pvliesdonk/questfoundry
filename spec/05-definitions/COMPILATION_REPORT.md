# Layer 5 Compilation Report

**Generated:** 2025-11-25
**Compiler Version:** QuestFoundry Compiler v1.0.0
**Branch:** `claude/review-cartridge-model-011QBMoWpFRyuaYvSp8nPk5Y`

---

## Summary

This report documents the compilation of Layer 5 Runtime Definitions from authoritative source layers (0, 1, 4). The compilation transpiled role charters and loop specifications into executable YAML definitions for LangGraph runtime execution.

| Category | Count | Status |
|----------|-------|--------|
| Role Profiles | 15 | Generated |
| Loop Patterns | 12 | Generated |
| Schema Validation | Partial | See findings |

---

## 1. Roles Generated

### 1.1 Role Summary

| ID | Name | Abbr | Dormancy | Role Type | Charter |
|----|------|------|----------|-----------|---------|
| showrunner | Showrunner | SR | always_on | reasoning_agent | spec/01-roles/charters/showrunner.md |
| gatekeeper | Gatekeeper | GK | always_on | reasoning_agent | spec/01-roles/charters/gatekeeper.md |
| plotwright | Plotwright | PW | default_on | reasoning_agent | spec/01-roles/charters/plotwright.md |
| scene_smith | Scene Smith | SS | default_on | reasoning_agent | spec/01-roles/charters/scene_smith.md |
| lore_weaver | Lore Weaver | LW | default_on | reasoning_agent | spec/01-roles/charters/lore_weaver.md |
| codex_curator | Codex Curator | CC | default_on | reasoning_agent | spec/01-roles/charters/codex_curator.md |
| style_lead | Style Lead | ST | default_on | reasoning_agent | spec/01-roles/charters/style_lead.md |
| researcher | Researcher | RS | optional | reasoning_agent | spec/01-roles/charters/researcher.md |
| art_director | Art Director | AD | optional | reasoning_agent | spec/01-roles/charters/art_director.md |
| illustrator | Illustrator | IL | optional | executor_agent | spec/01-roles/charters/illustrator.md |
| audio_director | Audio Director | AuD | optional | reasoning_agent | spec/01-roles/charters/audio_director.md |
| audio_producer | Audio Producer | AuP | optional | executor_agent | spec/01-roles/charters/audio_producer.md |
| translator | Translator | TR | optional | reasoning_agent | spec/01-roles/charters/translator.md |
| book_binder | Book Binder | BB | optional | executor_agent | spec/01-roles/charters/book_binder.md |
| player_narrator | Player-Narrator | PN | optional | reasoning_agent | spec/01-roles/charters/player_narrator.md |

### 1.2 Dormancy Distribution

- **Always-On (2):** Showrunner, Gatekeeper
- **Default-On (5):** Plotwright, Scene Smith, Lore Weaver, Codex Curator, Style Lead
- **Optional (8):** Researcher, Art Director, Illustrator, Audio Director, Audio Producer, Translator, Book Binder, Player-Narrator

---

## 2. Tools Injected Per Role

### 2.1 Tool Assignment Matrix

| Role | Internal Tools | External Tools | Rationale |
|------|---------------|----------------|-----------|
| **Showrunner** | read_hot_sot, write_hot_sot, read_cold_sot, write_cold_sot, create_snapshot, send_protocol_message, update_tu, wake_role, trigger_gatecheck | - | Full orchestration authority; manages all SoT operations, TU lifecycle, role activation, and gatechecks |
| **Gatekeeper** | read_hot_sot, write_hot_sot, read_cold_sot, validate_artifact, send_protocol_message, evaluate_quality_bar | - | Quality enforcement; needs artifact validation and bar evaluation for gatechecks |
| **Plotwright** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Topology design; reads existing structure, writes topology updates |
| **Scene Smith** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Prose production; reads context and style, writes section drafts |
| **Lore Weaver** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Canon management; reads hooks/existing canon, writes canon packs |
| **Codex Curator** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Player-safe entries; reads canon, writes codex entries |
| **Style Lead** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Voice consistency; reads drafts/style docs, writes addenda |
| **Researcher** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | web_search | Fact verification; needs web search for claim corroboration |
| **Art Director** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Visual planning; writes art plans (no image generation) |
| **Illustrator** | read_hot_sot, write_hot_sot, send_protocol_message | generate_image | Image rendering; needs image generation tool for renders |
| **Audio Director** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Audio planning; writes audio plans (no audio production) |
| **Audio Producer** | read_hot_sot, write_hot_sot, send_protocol_message | generate_audio | Audio rendering; needs audio synthesis tool for assets |
| **Translator** | read_hot_sot, write_hot_sot, read_cold_sot, send_protocol_message | - | Localization; reads source surfaces, writes language packs |
| **Book Binder** | read_hot_sot, write_hot_sot, read_cold_sot, write_exports, validate_artifact, send_protocol_message | - | Export assembly; needs export write and artifact validation |
| **Player-Narrator** | read_cold_sot, send_protocol_message | - | In-world performance; Cold-only access, no Hot visibility |

### 2.2 Tool Categories

**Internal Tools (Studio Operations):**
- `read_hot_sot` / `write_hot_sot` - Hot Source of Truth access
- `read_cold_sot` / `write_cold_sot` - Cold Source of Truth access
- `create_snapshot` - Cold snapshot creation
- `send_protocol_message` - Layer 4 protocol messaging
- `update_tu` - Trace Unit lifecycle management
- `wake_role` - Dormant role activation
- `trigger_gatecheck` - Quality gate invocation
- `validate_artifact` - Schema validation
- `evaluate_quality_bar` - Bar status evaluation
- `write_exports` - Export view generation

**External Tools (Service Integrations):**
- `web_search` - Web search API (SerpAPI/similar)
- `generate_image` - Image generation API (Stable Diffusion)
- `generate_audio` - Audio synthesis API (ElevenLabs/similar)

---

## 3. Missing Capabilities

The following charter requirements have no matching tool in the current registry:

| Role | Charter Requirement | Expected Tool | Current Status |
|------|---------------------|---------------|----------------|
| **Book Binder** | Document format conversion (EPUB/PDF/HTML) | `pandoc` | Defined in registry but not assigned |
| **Codex Curator** | Lore index search for cross-references | `lore_index` | Defined in registry but not assigned |
| **Lore Weaver** | Canon consistency checking with index | `lore_index` | Defined in registry but not assigned |
| **Audio Producer** | Audio synthesis with TTS | `audio_synthesis` | Defined in registry but needs renaming (`generate_audio` used instead) |
| **Researcher** | Citation management | - | Not in registry; could benefit from citation database tool |
| **Translator** | Translation memory lookup | - | Not in registry; would improve consistency |

### 3.1 Recommendations

1. **Assign `pandoc` to Book Binder** - Required for EPUB/PDF export generation
2. **Assign `lore_index` to Codex Curator and Lore Weaver** - Enables semantic canon/codex search
3. **Standardize audio tool naming** - Align `generate_audio` with registry's `audio_synthesis`
4. **Consider adding citation management tool** - Would strengthen Researcher's memo quality
5. **Consider adding translation memory tool** - Would improve Translator's consistency

---

## 4. Loops Generated

### 4.1 Loop Summary

| ID | Name | Category | Phase | Primary Roles |
|----|------|----------|-------|---------------|
| story_spark | Story Spark | Discovery | Ideation | PW, SS |
| hook_harvest | Hook Harvest | Discovery | Triage | SR |
| lore_deepening | Lore Deepening | Discovery | Canonization | LW |
| codex_expansion | Codex Expansion | Refinement | Codex | CC |
| style_tune_up | Style Tune-up | Refinement | Voice | ST |
| art_touch_up | Art Touch-up | Assets | Visual | AD, IL |
| audio_pass | Audio Pass | Assets | Audio | AuD, AuP |
| translation_pass | Translation Pass | Localization | Translation | TR |
| binding_run | Binding Run | Export | Assembly | BB |
| narration_dry_run | Narration Dry-Run | Export | Testing | PN |
| post_mortem | Post-Mortem | Reflection | Retrospective | SR |
| full_production_run | Full Production Run | Full Cycle | Production | SR (all) |

### 4.2 Loop Categories

- **Discovery (3):** story_spark, hook_harvest, lore_deepening
- **Refinement (2):** codex_expansion, style_tune_up
- **Assets (2):** art_touch_up, audio_pass
- **Localization (1):** translation_pass
- **Export (2):** binding_run, narration_dry_run
- **Reflection (1):** post_mortem
- **Full Cycle (1):** full_production_run

---

## 5. Schema Validation Findings

### 5.1 Role Profiles

**Status:** Conformant

The generated role profiles conform to `role_profile.schema.json` with all required fields:
- `id` - Present and valid (snake_case pattern)
- `identity` - Contains name, abbreviation, charter_ref, dormancy_policy
- `interface` - Contains inputs, outputs
- `behavior` - Contains prompt template
- `protocol` - Contains intents, lifecycles
- `constraints` - Contains safety (can_see_hot, can_see_spoilers, pn_safe)

### 5.2 Loop Patterns

**Status:** Non-conformant - Structural Mismatch

The generated loop patterns use a **documentation-style format** that differs from the `loop_pattern.schema.json` executable format.

**Schema Requires:**
```yaml
id: string
metadata:
  name: string (required)
  type: enum (required)
topology:
  entry_node: string (required)
  nodes: array (required)
  edges: array (required)
protocol_flow: object (required)
gates: object (required)
traceability:
  tu_lifecycle:
    required: boolean (required)
  produces_cold: boolean (required)
```

**Generated Format:**
```yaml
id: string
identity:
  name: string
  description: string
  category: string
  phase: string
purpose: string
outcome: string
triggers: array
context: object
topology:
  nodes: array
  edges: array
raci: object
handoffs: array
success_criteria: array
failure_modes: array
quality_bars: array
merge_path: object
metadata: object
```

### 5.3 Required Actions

1. **Restructure loop patterns** to match executable schema format:
   - Move `identity.name` to `metadata.name`
   - Add `metadata.type` from category mapping
   - Add `topology.entry_node` from first node
   - Add `protocol_flow` section (can be minimal)
   - Add `gates` section with quality_bars
   - Add `traceability` section with tu_lifecycle and produces_cold

2. **Alternative:** Update schema to reflect documentation format if that's the intended Layer 5 representation

---

## 6. File Manifest

### 6.1 Role Profiles (`spec/05-definitions/roles/`)

```
showrunner.yaml
gatekeeper.yaml
plotwright.yaml
scene_smith.yaml
lore_weaver.yaml
codex_curator.yaml
style_lead.yaml
researcher.yaml
art_director.yaml
illustrator.yaml
audio_director.yaml
audio_producer.yaml
translator.yaml
book_binder.yaml
player_narrator.yaml
```

### 6.2 Loop Patterns (`spec/05-definitions/loops/`)

```
story_spark.yaml
hook_harvest.yaml
lore_deepening.yaml
codex_expansion.yaml
style_tune_up.yaml
art_touch_up.yaml
audio_pass.yaml
translation_pass.yaml
binding_run.yaml
narration_dry_run.yaml
post_mortem.yaml
full_production_run.yaml
```

---

## 7. Input Sources Used

| Layer | Source | Purpose |
|-------|--------|---------|
| 0 | spec/00-north-star/ROLE_INDEX.md | Role metadata, dormancy policies |
| 0 | spec/00-north-star/LOOPS/*.md | Loop specifications and workflows |
| 1 | spec/01-roles/charters/*.md | Role charters (15 files) |
| 4 | spec/04-protocol/INTENTS.md | Protocol message intents |
| 6 | spec/06-runtime/interfaces/tool_registry.yaml | Tool definitions |

---

## 8. Next Steps

1. **Schema alignment:** Decide whether to update loop patterns to match executable schema or update schema to match documentation format
2. **Tool assignment review:** Assign missing tools (pandoc, lore_index) to appropriate roles
3. **Runtime integration:** Wire Layer 5 definitions into LangGraph node factory
4. **Validation tooling:** Create automated schema validation in CI/CD pipeline

---

*Report generated by QuestFoundry Compiler Agent*
