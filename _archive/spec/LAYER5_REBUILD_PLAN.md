# Layer 5 Rebuild Plan - Structured Knowledge for Agents

This document details how to turn Layer 5 (spec/05-definitions) into a machine-readable, regenerable
projection of Layers 0-4, so that agents in lib/runtime can consume the spec as structured input
instead of ad-hoc markdown.

---

## Problem Statement

- Agents need precise knowledge of loops, lifecycles, artifacts, and protocol.
- That knowledge already exists in Layers 0-4 (playbooks, charters, taxonomies, schemas, protocol).
- Layer 5 is currently a hand-maintained reimagining of those layers.
- consult_* tools mostly return long markdown blobs instead of structured data.
- This makes it hard to inject structured knowledge and to derive guardrails directly from the spec.

**Goal:** Generate L5 YAML definitions from L0-L4 sources such that:

1. L5 files are **projections**, not hand-authored duplicates
2. Runtime can consume structured data instead of parsing markdown
3. Changes to L0-L4 can be reflected in L5 via regeneration
4. Normative rules (hard enforcement) are distinguished from advisory guidance (soft hints)

---

## Complete Source Layer Inventory

### L0: North Star (`00-north-star/`) — 15 Core Documents + 25 Loop/Playbook Files

#### Core Policy Documents (15 files)

| File | Content Summary | L5 Target |
|------|-----------------|-----------|
| `README.md` | Navigation, quick start, 6-step reading path | — (meta) |
| `NORTH_STAR.md` | 8 non-negotiable principles, 6 success criteria, 14 roles, 10 loops | `principles.yaml` |
| `WORKING_MODEL.md` | 15 roles, 10 loops, 5-stage merge path, RACI matrix (15×4), 3 workflow patterns | `workflow_patterns.yaml` |
| `SOURCES_OF_TRUTH.md` | Hot/Cold definitions, state objects, permissions table, stabilization path | `hot_cold_policy.yaml` |
| `ROLE_INDEX.md` | 15 role definitions, RACI (10×4), 7 naming aliases, dormancy policy | (feeds role profiles) |
| `PN_PRINCIPLES.md` | 6 golden rules, 4 diegetic gate techniques, 4 tone guardrails, 4 failure modes | `pn_constraints.yaml` |
| `QUALITY_BARS.md` | 8 bars with checks/failures/fixes, exception framework, 4 anti-patterns | `quality_gates/*.yaml` |
| `TRACEABILITY.md` | TU (10 fields), 6 TU requirement categories, linking rules, 4 failure modes | `tu_schema_extension.yaml` |
| `TRACELOG.md` | Append-only Cold snapshot ledger format | — (runtime artifact) |
| `EVERGREEN_MANUSCRIPT.md` | View definition, 12 contents, 7 export options, 11-item checklist | `view_policy.yaml` |
| `SPOILER_HYGIENE.md` | 5 golden rules, 6-row leak taxonomy, 8 redlines, 7-item review checklist | `spoiler_constraints.yaml` |
| `ACCESSIBILITY_AND_CONTENT_NOTES.md` | 5 principles, 6 surfaces, 7 global requirements, per-surface anatomy | `accessibility_policy.yaml` |
| `INCIDENT_RESPONSE.md` | 4 incident categories, 7-step protocol, 5 playbooks (A–E), escalation matrix | `incident_playbooks.yaml` |
| `COLD_SOT_FORMAT.md` | Directory structure, 3 required files, Binder contract, validation rules | `cold_sot_schema.yaml` |
| `HOOKS.md` | 13 hook types, 7 status lifecycle, 9 hook fields, 8 lifecycle stages, 4 examples | `hook_lifecycle.yaml` |

#### Loop Guides (`LOOPS/`) — 12 files, 9-section structure each

| Loop | Triggers | Key Deliverables | Quality Bars |
|------|----------|------------------|--------------|
| `story_spark.md` | New chapter/restructure | Topology notes, section briefs, hook list | Integrity, Reachability, Nonlinearity |
| `hook_harvest.md` | Accumulated proposals | Triaged hook list, owner assignments | — |
| `lore_deepening.md` | Accepted hooks need canon | Canon packs, player-safe summaries | Integrity, Gateways |
| `codex_expansion.md` | Taxonomy gaps | Codex entries, cross-refs | Presentation, Accessibility |
| `style_tune_up.md` | Voice drift detected | Style addenda, PN patterns | Style, Presentation |
| `art_touch_up.md` | Visual needs identified | Art plans, shotlists | Presentation, Accessibility |
| `audio_pass.md` | Audio needs identified | Audio plans, cuelists | Presentation, Accessibility |
| `translation_pass.md` | Locale needs | Language packs, register maps | Style, Presentation |
| `binding_run.md` | Export requested | View logs, front matter | All 8 bars |
| `narration_dry_run.md` | View ready for playtest | PN playtest notes | Presentation, Style |
| `post_mortem.md` | Session complete | Post-mortem report, improvement hooks | — |
| `full_production_run.md` | Multi-slice coordination | Coordinated deliverables | All 8 bars |

#### Playbooks (`PLAYBOOKS/`) — 15 files, compact checklist format

Each playbook condenses its loop guide into: Use when, One-minute scope, Inputs, Do the thing,
Deliverables, Hand-off map, Fast triage rubric, RACI (micro), Anti-patterns, Cheat line.

**Additional playbooks beyond loop mirrors:**

- `playbook_world_genesis.md` — Canon-first workflow
- `playbook_canon_transfer_export.md` — Export canon for reuse
- `playbook_canon_transfer_import.md` — Import canon from prior project

---

### L1: Roles (`01-roles/`) — 52 Files Total

#### Role Charters (`charters/`) — 16 files (15 roles + 1 "all" meta)

**Fixed 12-Section Skeleton per Charter:**

```
§1  Canon & Mission — name, aliases, one-sentence mission, normative refs (6 L0 links)
§2  Scope & Shape — in-scope bullets, out-of-scope bullets, MAY/SHOULD/MUST decisions
§3  Inputs & Outputs — reads (Hot/Cold), produces (named artifacts)
§4  Participation in Loops — RACI per loop, definition of done
§5  Hook Policy — types, size, tags, lineage
§6  Player-Surface Obligations — spoiler hygiene, accessibility, PN boundaries
§7  Dormancy & Wake — default state, wake signals, plan-only merges
§8  Cross-Domain & Escalation — coordination patterns, pair guide refs
§9  Anti-patterns — 3-5 named patterns with descriptions
§10 Mini-Checklist — 6-item pre-handoff checklist
§11 Tiny Examples — 1-3 before→after pairs
§12 Metadata — lineage (TU ID), related links
```

**Role Categories:**

- **Always-On (2):** Showrunner, Gatekeeper
- **Default-On (5):** Plotwright, Scene Smith, Style Lead, Lore Weaver, Codex Curator
- **Optional/Dormant (6):** Researcher, Art Director, Illustrator, Audio Director, Audio Producer, Translator
- **Downstream (2):** Book Binder, Player-Narrator

#### Role Briefs (`briefs/`) — 15 files

**Fixed 10-Section Structure per Brief:**

```
§0  Normative references — 6 L0 links + charter ref
§1  Operating principles — 5 heuristics (small steps, player-first, hooks not detours, etc.)
§2  Inputs & outputs (quick view) — reads, produces, lineage note
§3  Small-step policy — scope, TU, timebox, DoD
§4  Heuristics — 6 practical working heuristics
§5  Safety rails — spoilers, internals, accessibility, PN phrasing
§6  Communication rules — notify neighbors, pair guides, dormancy, escalation
§7  When to pause & escalate — 4 situations
§8  Tiny examples — 2-4 line examples
§9  Done checklist — 6 items
§10 Metadata — role, lineage, charter ref, loop guide ref
```

#### Interface Documents (`interfaces/`) — 3 files

| File | Content | L5 Target |
|------|---------|-----------|
| `pair_guides.md` | 11 pair handshakes with 5-minute protocols, anti-patterns, escalation | `pair_handshakes.yaml` |
| `dormancy_signals.md` | Wake rubric (4 factors), hard triggers, per-role signals, TU scaffolds | `dormancy_policy.yaml` |
| `escalation_rules.md` | 5 principles, 11 lane owners, 3 severity levels, ADR skeleton | `escalation_policy.yaml` |

#### Checklists (`checklists/`) — 2 files

| File | Content | L5 Target |
|------|---------|-----------|
| `quality_bars_by_role.md` | Per-role bar ownership, self-checks, typical fails, escalation | (feeds role profiles) |
| `role_readiness.md` | Per-role pre-flight: inputs, neighbors, ready-if, red flags | (feeds role profiles) |

#### RACI (`raci/`) — 1 file

| File | Content | L5 Target |
|------|---------|-----------|
| `by_loop.md` | 11 micro-loop RACI matrices, bar focus, quick matrix appendix | (feeds loop patterns) |

---

### L2: Dictionary (`02-dictionary/`) — 47 Files Total

#### Core Reference Files (6 files)

| File | Content | L5 Target |
|------|---------|-----------|
| `glossary.md` | 80+ term definitions (topology, Hot/Cold, roles, quality, localization) | `glossary.yaml` |
| `taxonomies.md` | 10 controlled vocabularies (100+ enum values) | `taxonomies.yaml` |
| `field_registry.md` | 237 fields across 17 artifacts in 10 categories | `field_definitions.yaml` |
| `loop_names.md` | 13 loops: display name ↔ file name ↔ abbreviation | `loop_names.yaml` |
| `role_abbreviations.md` | 15 roles: full name ↔ abbreviation ↔ deferral tag | `role_abbreviations.yaml` |
| `conventions/choice_integrity.md` | Convergence patterns (reflection, bridge, state-aware) | `choice_integrity.yaml` |

#### Taxonomies Detail (10 Controlled Vocabularies)

1. **Hook Types** (13): narrative, scene, factual, taxonomy, structure, canon, research, style/pn, translation, art, audio, binder/nav, accessibility
2. **Hook Status** (7): proposed → accepted → in-progress → resolved → canonized; deferred, rejected
3. **TU/Loop Types** (13): story-spark, hook-harvest, lore-deepening, codex-expansion, style-tuneup, art-touchup, audio-pass, translation-pass, binding-run, narration-dry-run, gatecheck, post-mortem, archive-snapshot
4. **Gate Types** (6+): token, reputation, knowledge, physical, temporal, composite
5. **Quality Bars** (8): Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism, Presentation, Accessibility
6. **Artifact Status** (6): draft, review, blocked, approved, merged, published
7. **Deferral Types** (4): deferred:art, deferred:audio, deferred:translation, deferred:research
8. **Research Posture** (6): corroborated, plausible, disputed, uncorroborated:low/medium/high
9. **Role Dormancy** (4 categories): always-on, default-on, optional, downstream
10. **Loop Classifications**: Discovery, Refinement, Asset, Export, Validation

#### Artifact Templates (`artifacts/`) — 41 files

**Core Workflow (2):**

- `hook_card.md` — 28 fields, 7 status lifecycle
- `tu_brief.md` — 27 fields, 6 status lifecycle

**Creation & Content (4):**

- `canon_pack.md` — 34 fields (most complex), Hot answers + Cold summaries
- `codex_entry.md` — 29 fields, player-safe terminology
- `style_addendum.md` — 15+ fields, register/patterns
- `research_memo.md` — 12+ fields, question → answer → posture

**Quality & Export (3):**

- `gatecheck_report.md` — 20+ fields, 8 bar evaluations
- `view_log.md` — 15+ fields, export metadata
- `front_matter.md` — 9 fields, player-facing intro

**Planning (5):**

- `edit_notes.md` — 10+ fields, line-level fixes
- `shotlist.md` — 10+ fields, art slot index
- `cuelist.md` — 10+ fields, audio cue index
- `art_plan.md` — 36 fields (most complex), illustration spec
- `audio_plan.md` — 25+ fields, audio cue spec

**Localization (2):**

- `language_pack.md` — 25+ fields, glossary + register + coverage
- `register_map.md` — 30 fields, voice equivalence across locales

**Additional (25+ more):**

- `section_brief.md`, `section_draft.md`, `section.md`
- `anchor_map.md`, `gateway_map.md`
- `pre_gate_note.md`, `safety_checklist.md`, `spoiler_hygiene_note.md`
- `harvest_sheet.md`, `meeting_minutes.md`, `phrasing_patterns.md`
- `codex_pack.md`, `pn_playtest_notes.md`, `determinism_log.md`
- `world_genesis_manifest.md`, `post_mortem_report.md`
- `project_metadata.md`, `art_manifest.md`, `style_manifest.md`

---

### L3: Schemas (`03-schemas/`) — 57+ JSON Schema Files

**Artifact Schemas:** One per artifact type (hook_card, tu_brief, canon_pack, etc.)

**Meta-Schemas (L5 Definition Schemas):**

- `definitions/role_profile.schema.json`
- `definitions/loop_pattern.schema.json`
- `definitions/quality_gate.schema.json`

**Cold SoT Schemas:**

- `cold_manifest.schema.json`
- `cold_book.schema.json`
- `cold_art_manifest.schema.json`

---

### L4: Protocol (`04-protocol/`) — 33 Files Total

#### Core Documents (5 files)

| File | Content | L5 Target |
|------|---------|-----------|
| `README.md` | Overview, deliverables index | — (meta) |
| `ENVELOPE.md` | Transport wrapper spec, 14 fields, PN safety invariant | `envelope_schema.yaml` |
| `envelope.schema.json` | JSON Schema for envelope validation | (normative schema) |
| `INTENTS.md` | 33 intent definitions, authorization rules | `intents.yaml` |
| `CONFORMANCE.md` | 4 conformance levels, test matrix | `conformance_tests.yaml` |

#### Intents Catalog (33 Intents)

**General (2):** `ack`, `error`
**Human (2):** `human.question`, `human.response`
**Role (2):** `role.wake`, `role.dormant`
**TU (10):** `tu.open`, `tu.start`, `tu.defer`, `tu.reject`, `tu.submit_gate`, `tu.rework`, `tu.reactivate`, `tu.merge`, `tu.close`, `tu.update`, `tu.checkpoint`
**Hook (2):** `hook.create`, `hook.update_status`
**Gate (4):** `gate.report.submit`, `gate.decision`, `gate.defer`, `gate.feedback`
**Merge (3):** `merge.request`, `merge.approve`, `merge.reject`
**View (7):** `view.export.request`, `view.export.result`, `view.bind`, `view.bind_failed`, `view.bound`, `view.feedback`, `view.publish`
**PN (1):** `pn.playtest.submit`
**Canon (3):** `canon.transfer.export`, `canon.transfer.import`, `canon.genesis.create`

#### Lifecycles (`LIFECYCLES/`) — 4 files

| Lifecycle | States | Transitions | Guards |
|-----------|--------|-------------|--------|
| `hooks.md` | 7 states | 11 transitions | Owner assignment, canonization requires merge |
| `tu.md` | 6 states | 10 transitions | Snapshot binding, merge requires pass |
| `gate.md` | 4 states | 5 transitions | All 8 bars evaluated, decision rules |
| `view.md` | 5 states | 6 transitions | PN safety invariant, snapshot consistency |

#### Flows (`FLOWS/`) — 6 documented + 7 pending

**Documented:**

- `hook_harvest.md` — Create and triage hooks
- `lore_deepening.md` — Develop canon
- `codex_expansion.md` — Create codex entries
- `gatecheck.md` — Quality bar validation
- `binding_run.md` — Cold export to PN
- `narration_dry_run.md` — PN playtest feedback

**Pending:** story_spark, style_tune_up, art_touch_up, audio_pass, translation_pass, full_production_run, post_mortem

#### Examples (`EXAMPLES/`) — 15 JSON envelope examples

---

## L5 Target Structure (Complete)

### Role Profiles (`05-definitions/roles/*.yaml`)

Each profile compiles from L1 charter + brief + L0 policy documents:

| Source | L5 Field | Runtime Usage |
|--------|----------|---------------|
| Charter §1 name | `identity.name` | Display name |
| Charter §1 aliases | `identity.aliases` | Alternative names |
| Charter §1 mission | `prompt_content.core_mandate` | IDENTITY layer |
| Charter §1 normative refs | `normative_references[]` | Policy links |
| Charter §2 in-scope | `prompt_content.operating_principles[]` | IDENTITY layer |
| Charter §2 out-of-scope | `prompt_content.out_of_scope[]` | What NOT to do |
| Charter §2 MAY/SHOULD/MUST | `protocol.authority.*` | Permission levels |
| Charter §3 reads | `interface.inputs[]` | STATE MANAGEMENT |
| Charter §3 produces | `interface.outputs[]` | Artifact extraction |
| Charter §4 RACI | `raci.loops.*` | Loop membership |
| Charter §4 DoD | `success_criteria.role_done[]` | Self-validation |
| Charter §5 hook types | `protocol.lifecycles.hook.can_create` | Hook permissions |
| Charter §5 hook tags | `protocol.lifecycles.hook.tags[]` | Hook categorization |
| Charter §6 spoiler hygiene | `constraints.safety.spoiler_hygiene_level` | PN safety |
| Charter §6 accessibility | `constraints.safety.accessibility_obligations` | A11y requirements |
| Charter §7 dormancy | `identity.dormancy_policy` | Activation logic |
| Charter §7 wake signals | `identity.wake_conditions[]` | When to activate |
| Charter §8 escalation | `protocol.escalation.*` | Routing rules |
| Charter §9 anti-patterns | `prompt_content.anti_patterns[]` | IDENTITY warnings |
| Charter §10 checklist | `prompt_content.checklist[]` | Self-validation |
| Charter §11 examples | `prompt_content.heuristics[].examples` | IDENTITY examples |
| Brief §1 principles | `prompt_content.mindset[]` | Operating heuristics |
| Brief §4 heuristics | `prompt_content.heuristics[]` | Practical guidance |
| Brief §5 safety rails | `constraints.safety.quick_checks[]` | Validation rules |
| Brief §7 escalation triggers | `protocol.escalation.triggers[]` | When to pause |
| `quality_bars_by_role.md` | `quality_bars_owned[]` | Bar responsibility |
| `role_readiness.md` | `readiness.inputs[]`, `.neighbors[]`, `.red_flags[]` | Pre-flight |
| `pair_guides.md` | `protocol.pair_handshakes[]` | Coordination |
| `dormancy_signals.md` | `identity.dormancy.*` | Wake/sleep rules |
| L4 INTENTS.md | `protocol.intents.can_send[]`, `.can_receive[]` | Message auth |
| L4 lifecycles | `protocol.lifecycles.*` | State machines |

**Critical Fields (Runtime Dependencies):**

```yaml
interface:
  inputs:
    - artifact_type: hook_card       # L3 schema name
      state_key: hot_sot.hooks       # StudioState path
      required: false
      validation_mode: strict
  outputs:
    - artifact_type: section_brief
      state_key: hot_sot.section_briefs
      validation_required: true
      merge_strategy: append

protocol:
  intents:
    can_send: [hook.create, artifact.submit, feedback.request, ack, error]
    can_receive: [tu.assign, feedback.provide, artifact.request, ack, error]
  lifecycles:
    hook:
      can_create: true
      can_transition: []
    tu:
      can_open: false
      can_close: false
  envelope_defaults:
    safety:
      player_safe: false
      spoilers: allowed
    context:
      hot_cold: hot
```

### Loop Patterns (`05-definitions/loops/*.yaml`)

Each pattern compiles from L0 loop guide + playbook + L1 RACI:

| Source | L5 Field | Runtime Usage |
|--------|----------|---------------|
| Guide §1 triggers | `triggers[]` | Loop preconditions |
| Guide §2 inputs | `context.required_artifacts[]` | Input validation |
| Guide §3 roles (R) | `roles.responsible[]` | Primary owners |
| Guide §3 roles (A) | `roles.accountable` | Decision maker |
| Guide §3 roles (C) | `roles.consulted[]` | Must consult |
| Guide §3 roles (I) | `roles.informed[]` | Notify only |
| Guide §3 role descriptions | `topology.nodes[]` | Graph nodes |
| Guide §4 procedure | `topology.edges[]` | Graph transitions |
| Guide §5 deliverables | `deliverables[]` | Exit artifacts |
| Guide §6 merge path | `handoffs[]` | Downstream loops |
| Guide §7 success criteria | `success_criteria.custom_checks[]` | Exit validation |
| Guide §8 failure modes | `failure_modes[]` | Recovery patterns |
| Guide quality bar refs | `gates.quality_bars[]` | Gate enforcement |
| Playbook cheat line | `metadata.summary` | One-line description |
| Playbook anti-patterns | `anti_patterns[]` | What not to do |
| RACI by_loop.md | `raci.*` | Authoritative RACI |
| L4 flow | `protocol_flow.*` | Message sequences |

**Critical Fields (Runtime Dependencies):**

```yaml
topology:
  entry_node: topology_draft
  nodes:
    - role_id: plotwright
      node_id: topology_draft
      description: "Map parts/chapters and designate hubs..."
      inputs: [cold_snapshot, prior_topology]
      outputs: [topology_notes, section_briefs]
  edges:
    - source: topology_draft
      target: section_briefs
      type: conditional
      condition:
        evaluator: state_key_match
        expression: "state.topology_complete == true"

gates:
  quality_bars: [Integrity, Reachability, Nonlinearity, Gateways, Style, Presentation]
  pre_gate_required: true
  blocking_bars: [Integrity, Reachability]
```

### Quality Gates (`05-definitions/quality_gates/*.yaml`)

Each gate compiles from L0 QUALITY_BARS.md:

| Source | L5 Field | Runtime Usage |
|--------|----------|---------------|
| Bar name | `bar_name` | Display |
| What it means | `definition.what_it_means` | Agent understanding |
| Quick checks | `validation_checks[]` | Gatekeeper evaluation |
| Common failures | `failure_patterns[]` | Pattern recognition |
| Fixes | `remediation_guidance[]` | Recovery steps |
| Owned by roles | `ownership.primary_owners[]` | Responsibility |
| Checked in loops | `ownership.checked_in_loops[]` | When applied |
| Blocking policy | `blocking_policy` | Merge control |
| Exception rules | `exception_framework` | Waiver conditions |

### NEW: Protocol Definitions (`05-definitions/protocol/*.yaml`)

**Intents (`intents/*.yaml`):**

```yaml
# intents/tu.open.yaml
id: tu.open
domain: tu
verb: open
sender_roles: [SR, owner_a]
receiver_roles: [broadcast]
payload_type: tu_brief
transitions:
  from: hot-proposed
  to: stabilizing
authorization:
  requires_tu: true
  requires_snapshot: false
envelope_constraints:
  context.hot_cold: hot
```

**Lifecycles (`lifecycles/*.yaml`):**

```yaml
# lifecycles/hook.yaml
id: hook_lifecycle
states:
  - id: proposed
    terminal: false
  - id: accepted
    terminal: false
  - id: in-progress
    terminal: false
  - id: resolved
    terminal: false
  - id: canonized
    terminal: true
  - id: deferred
    terminal: false
  - id: rejected
    terminal: true
transitions:
  - from: proposed
    to: accepted
    intent: hook.update_status
    sender: SR
    guards: [owner_assigned]
  # ... all 11 transitions
```

**Envelope Defaults (`envelope_defaults.yaml`):**

```yaml
required_fields:
  - protocol.name
  - protocol.version
  - id
  - time
  - sender.role
  - receiver.role
  - intent
  - context.hot_cold
  - context.loop
  - safety.player_safe
  - safety.spoilers
  - payload.type
  - payload.data

pn_safety_invariant:
  when: receiver.role == "PN"
  enforce:
    - context.hot_cold: cold
    - context.snapshot: required
    - safety.player_safe: true
    - safety.spoilers: forbidden
```

### NEW: Taxonomy Definitions (`05-definitions/taxonomies/*.yaml`)

```yaml
# taxonomies/hook_types.yaml
id: hook_types
values:
  - id: narrative
    description: Story entities, locations, stakes, topology
  - id: scene
    description: Concrete scene details (traits, tells, props)
  - id: factual
    description: Real-world claims requiring verification
  # ... all 13 types

# taxonomies/quality_bars.yaml
id: quality_bars
values:
  - id: integrity
    display: Integrity
    description: Anchors resolve, IDs unique, no orphans
  # ... all 8 bars
```

### NEW: Artifact Definitions (`05-definitions/artifacts/*.yaml`)

```yaml
# artifacts/hook_card.yaml
id: hook_card
schema_ref: spec/03-schemas/hook_card.schema.json
state_key: hot_sot.hooks
field_count: 28
lifecycle: hook_lifecycle
status_field: header.status
required_fields:
  - header.id
  - header.status
  - classification.type_primary
  - player_safe_summary
  - proposed_next_step.loop
hot_cold: hot
player_safe: false
```

### NEW: Policy Definitions (`05-definitions/policies/*.yaml`)

```yaml
# policies/spoiler_hygiene.yaml
id: spoiler_hygiene
source: spec/00-north-star/SPOILER_HYGIENE.md
golden_rules:
  - id: no_reveals
    description: Never reveal hidden beats, codewords, gate conditions
  - id: no_internals
    description: Never print codeword names, gate logic, RNG, seeds
  # ... all 5 rules

leak_taxonomy:
  - type: twist_telegraph
    bad_example: "The foreman is secretly Syndicate."
    good_example: "The foreman eyes your badge; his smile doesn't reach his eyes."
  # ... all 6 leak types

redlines:
  - Codeword names or gate logic
  - Determinism or production details
  # ... all 8 categories

role_responsibilities:
  showrunner: Enforce this doc as non-negotiable bar
  gatekeeper: Block merges with Presentation failures
  # ... all 8 roles
```

---

## Complete Artifact Type → State Key Mapping

The generator must resolve human-readable names to schema types and state keys:

| Human Name (in Charters) | artifact_type | state_key | L3 Schema |
|--------------------------|---------------|-----------|-----------|
| Hook Card / Hook List / Hooks | hook_card | hot_sot.hooks | hook_card.schema.json |
| TU Brief / Trace Unit | tu_brief | hot_sot.tus | tu_brief.schema.json |
| Canon Pack / Canon | canon_pack | hot_sot.canon | canon_pack.schema.json |
| Codex Entry / Codex entries | codex_entry | hot_sot.codex | codex_entry.schema.json |
| Codex Pack | codex_pack | cold_sot.codex | codex_pack.schema.json |
| Style Addendum / Style guardrails | style_addendum | hot_sot.style | style_addendum.schema.json |
| Research Memo | research_memo | hot_sot.research | research_memo.schema.json |
| Edit Notes | edit_notes | hot_sot.edit_notes | edit_notes.schema.json |
| Topology Notes | topology_notes | hot_sot.topology | topology_notes.schema.json |
| Section Brief / Section briefs | section_brief | hot_sot.section_briefs | section_brief.schema.json |
| Section Draft | section_draft | hot_sot.drafts | section_draft.schema.json |
| Section | section | cold_sot.sections | section.schema.json |
| Gateway Map | gateway_map | hot_sot.gateway_map | gateway_map.schema.json |
| Anchor Map | anchor_map | hot_sot.anchor_map | anchor_map.schema.json |
| Art Plan | art_plan | hot_sot.art_plans | art_plan.schema.json |
| Audio Plan | audio_plan | hot_sot.audio_plans | audio_plan.schema.json |
| Shotlist | shotlist | hot_sot.shotlists | shotlist.schema.json |
| Cuelist | cuelist | hot_sot.cuelists | cuelist.schema.json |
| Language Pack | language_pack | hot_sot.language_packs | language_pack.schema.json |
| Register Map | register_map | hot_sot.register_maps | register_map.schema.json |
| Gatecheck Report | gatecheck_report | hot_sot.gatechecks | gatecheck_report.schema.json |
| View Log | view_log | cold_sot.view_logs | view_log.schema.json |
| Front Matter | front_matter | cold_sot.front_matter | front_matter.schema.json |
| PN Playtest Notes | pn_playtest_notes | hot_sot.playtest_notes | pn_playtest_notes.schema.json |
| Cold Snapshot / Current snapshot | cold_snapshot | cold_sot.snapshot | cold_manifest.schema.json |
| Post-Mortem Report | post_mortem_report | hot_sot.post_mortems | post_mortem_report.schema.json |
| Determinism Log | determinism_log | hot_sot.determinism_logs | determinism_log.schema.json |

---

## Normative vs Advisory Rules

### Normative (Hard Enforcement — Runtime MUST Block)

**From L4 Protocol:**

1. **TU Lifecycle Transitions**
   - States: hot-proposed → stabilizing → gatecheck → cold-merged
   - Invalid transitions raise `StateError`
   - Merge requires `context.snapshot` present

2. **Hook Lifecycle Transitions**
   - States: proposed → accepted → in-progress → resolved → canonized
   - Terminal states: canonized, rejected
   - Canonization requires Cold merge

3. **Gate Lifecycle Transitions**
   - States: pre-gate → gatecheck → decision
   - Decision requires all 8 bars evaluated
   - Pass requires all bars green

4. **View Lifecycle Transitions**
   - States: snapshot-selected → export-binding → pn-dry-run → feedback-collected → view-published
   - PN handoff requires PN safety invariant

5. **Envelope Schema**
   - Required fields: protocol, id, time, sender, receiver, intent, context, safety, payload
   - PN safety invariant when receiver.role = "PN"

6. **Intent Authorization**
   - Sender must be in `intent.sender_roles`
   - Receiver must be in `intent.receiver_roles`
   - Payload type must match `intent.payload_type`

**From L3 Schemas:**

7. **Artifact Schema Validation**
   - All artifacts validate against their L3 schema
   - Required fields per artifact type
   - Enum values from L2 taxonomies

**From L5 Role Profiles:**

8. **Hot/Cold Permissions**
   - `constraints.hot_cold_permissions.cold.write: false` blocks cold writes
   - PN has `constraints.safety.pn_safe: true` requirement

9. **Intent Permissions**
   - `protocol.intents.can_send` limits which intents a role can emit
   - `protocol.intents.can_receive` limits which intents a role can receive

### Advisory (Soft Guidance — Warn but Allow)

1. **Playbook Node Order** — Recommended sequence, skipping allowed
2. **Consult Requirements** — "Should consult playbook before tu.open"
3. **Role Wake Suggestions** — Dormancy wake signals are hints
4. **Success Criteria Checks** — Failing warns but doesn't block handoff
5. **Pair Guide Protocols** — 5-minute protocols are suggestions
6. **Checklist Items** — Pre-handoff checklists are self-validation

---

## Generator Architecture

### Phase 1: Parsers

```python
# Section-aware markdown parsers

class RoleCharterParser:
    """Parse 12-section charter structure."""
    def parse(self, path: Path) -> CharterData:
        sections = self._split_sections(path)
        return CharterData(
            canon_mission=self._parse_section_1(sections[1]),
            scope_shape=self._parse_section_2(sections[2]),
            inputs_outputs=self._parse_section_3(sections[3]),  # CRITICAL
            participation=self._parse_section_4(sections[4]),
            hook_policy=self._parse_section_5(sections[5]),
            player_surface=self._parse_section_6(sections[6]),
            dormancy=self._parse_section_7(sections[7]),
            escalation=self._parse_section_8(sections[8]),
            anti_patterns=self._parse_section_9(sections[9]),
            checklist=self._parse_section_10(sections[10]),
            examples=self._parse_section_11(sections[11]),
            metadata=self._parse_section_12(sections[12]),
        )

class LoopGuideParser:
    """Parse 9-section loop guide structure."""
    def parse(self, path: Path) -> LoopGuideData:
        ...

class PlaybookParser:
    """Parse compact playbook structure."""
    def parse(self, path: Path) -> PlaybookData:
        ...

class QualityBarsParser:
    """Parse QUALITY_BARS.md into 8 bar definitions."""
    def parse(self, path: Path) -> list[QualityBarData]:
        ...

class IntentsParser:
    """Parse INTENTS.md into 33 intent definitions."""
    def parse(self, path: Path) -> list[IntentData]:
        ...

class LifecycleParser:
    """Parse LIFECYCLES/*.md into state machines."""
    def parse(self, path: Path) -> LifecycleData:
        ...

class TaxonomiesParser:
    """Parse taxonomies.md into controlled vocabularies."""
    def parse(self, path: Path) -> dict[str, list[TaxonomyValue]]:
        ...

class ArtifactTemplateParser:
    """Parse artifacts/*.md into field definitions."""
    def parse(self, path: Path) -> ArtifactTemplateData:
        ...
```

### Phase 2: Cross-Reference Resolution

```python
class ReferenceResolver:
    """Resolve human names to schema types and state keys."""

    def __init__(self):
        self.artifact_mapping = self._load_artifact_mapping()
        self.role_abbreviations = self._load_role_abbreviations()
        self.loop_names = self._load_loop_names()
        self.taxonomies = self._load_taxonomies()

    def resolve_artifact(self, human_name: str) -> ArtifactRef:
        """'Hook List' → ArtifactRef(type='hook_card', state_key='hot_sot.hooks')"""
        ...

    def resolve_role(self, name_or_abbrev: str) -> str:
        """'Plotwright' or 'PW' → 'plotwright'"""
        ...

    def resolve_loop(self, display_name: str) -> str:
        """'Story Spark' → 'story_spark'"""
        ...

    def resolve_intent(self, intent_str: str) -> IntentRef:
        """'hook.create' → IntentRef with sender_roles, receiver_roles, etc."""
        ...
```

### Phase 3: L5 Generation

```python
class L5Generator:
    """Generate L5 YAML from parsed L0-L4 sources."""

    def generate_role_profile(self, role_id: str) -> dict:
        charter = self.charter_parser.parse(f"charters/{role_id}.md")
        brief = self.brief_parser.parse(f"briefs/{role_id}.md")
        quality_bars = self.quality_bars_by_role[role_id]
        readiness = self.role_readiness[role_id]
        pair_guides = self.pair_guides_for_role[role_id]
        intents = self.intents_for_role[role_id]
        lifecycles = self.lifecycles_for_role[role_id]

        return {
            "id": role_id,
            "identity": self._build_identity(charter, brief),
            "prompt_content": self._build_prompt_content(charter, brief),
            "interface": self._build_interface(charter),  # CRITICAL
            "protocol": self._build_protocol(charter, intents, lifecycles),
            "constraints": self._build_constraints(charter, brief),
            "raci": self._build_raci(charter),
            "readiness": self._build_readiness(readiness),
            "metadata": self._build_metadata(charter),
        }

    def generate_loop_pattern(self, loop_id: str) -> dict:
        guide = self.loop_parser.parse(f"LOOPS/{loop_id}.md")
        playbook = self.playbook_parser.parse(f"PLAYBOOKS/playbook_{loop_id}.md")
        raci = self.raci_by_loop[loop_id]
        flow = self.flows.get(loop_id)

        return {
            "id": loop_id,
            "metadata": self._build_loop_metadata(guide, playbook),
            "topology": self._build_topology(guide),
            "roles": self._build_roles(guide, raci),
            "context": self._build_context(guide),
            "gates": self._build_gates(guide),
            "handoffs": self._build_handoffs(guide),
            "success_criteria": self._build_success_criteria(guide),
            "protocol_flow": self._build_protocol_flow(flow) if flow else None,
        }

    def generate_quality_gate(self, bar_data: QualityBarData) -> dict:
        ...

    def generate_intent(self, intent_data: IntentData) -> dict:
        ...

    def generate_lifecycle(self, lifecycle_data: LifecycleData) -> dict:
        ...

    def generate_taxonomy(self, taxonomy_id: str, values: list) -> dict:
        ...

    def generate_artifact_definition(self, artifact_data: ArtifactTemplateData) -> dict:
        ...

    def generate_policy(self, policy_id: str, source_path: Path) -> dict:
        ...
```

### Phase 4: Validation

```bash
# Validate all generated L5 files against meta-schemas

for file in spec/05-definitions/roles/*.yaml; do
    jsonschema --instance "$file" spec/03-schemas/definitions/role_profile.schema.json
done

for file in spec/05-definitions/loops/*.yaml; do
    jsonschema --instance "$file" spec/03-schemas/definitions/loop_pattern.schema.json
done

for file in spec/05-definitions/quality_gates/*.yaml; do
    jsonschema --instance "$file" spec/03-schemas/definitions/quality_gate.schema.json
done

# Additional validation for new definition types
for file in spec/05-definitions/protocol/intents/*.yaml; do
    jsonschema --instance "$file" spec/03-schemas/definitions/intent.schema.json
done

for file in spec/05-definitions/protocol/lifecycles/*.yaml; do
    jsonschema --instance "$file" spec/03-schemas/definitions/lifecycle.schema.json
done
```

---

## Implementation Phases

### Phase 1: Infrastructure & Parsers

- [ ] Create `spec/tools/generate_l5.py` with CLI
- [ ] Implement `RoleCharterParser` (12 sections)
- [ ] Implement `RoleBriefParser` (10 sections)
- [ ] Implement `LoopGuideParser` (9 sections)
- [ ] Implement `PlaybookParser` (compact format)
- [ ] Implement `QualityBarsParser`
- [ ] Implement `IntentsParser`
- [ ] Implement `LifecycleParser` (4 lifecycles)
- [ ] Implement `TaxonomiesParser` (10 vocabularies)
- [ ] Implement `ArtifactTemplateParser`
- [ ] Create `ReferenceResolver` with mapping tables

### Phase 2: Role Profile Generation

- [ ] Parse all 15 role charters
- [ ] Parse all 15 role briefs
- [ ] Extract `interface.inputs/outputs` from Charter §3
- [ ] Extract `prompt_content.*` from Charter §2, §9, §11 + Brief §1, §4
- [ ] Extract `raci.loops` from Charter §4
- [ ] Extract `protocol.intents` from L4 INTENTS.md
- [ ] Extract `protocol.lifecycles` from L4 LIFECYCLES/
- [ ] Extract `constraints.*` from Charter §6, §7 + Brief §5
- [ ] Extract `readiness.*` from `role_readiness.md`
- [ ] Generate and validate 15 role profiles

### Phase 3: Loop Pattern Generation

- [ ] Parse all 12 loop guides
- [ ] Parse all 15 playbooks
- [ ] Extract `topology.nodes` from Guide §3
- [ ] Extract `topology.edges` from Guide §4
- [ ] Extract `success_criteria` from Guide §7
- [ ] Extract `failure_modes` from Guide §8
- [ ] Extract `raci.*` from `raci/by_loop.md`
- [ ] Extract `protocol_flow` from L4 FLOWS/
- [ ] Generate and validate 12 loop patterns

### Phase 4: Quality Gate Generation

- [ ] Parse QUALITY_BARS.md
- [ ] Extract 8 quality gate definitions
- [ ] Generate and validate 8 quality gates

### Phase 5: Protocol Definitions Generation

- [ ] Parse INTENTS.md → generate 33 intent definitions
- [ ] Parse LIFECYCLES/*.md → generate 4 lifecycle definitions
- [ ] Parse ENVELOPE.md → generate envelope defaults
- [ ] Parse CONFORMANCE.md → generate conformance tests

### Phase 6: Taxonomy & Artifact Definitions

- [ ] Parse taxonomies.md → generate 10 taxonomy definitions
- [ ] Parse artifacts/*.md → generate 30+ artifact definitions
- [ ] Validate all enums match taxonomies

### Phase 7: Policy Definitions Generation

- [ ] Parse SPOILER_HYGIENE.md → `spoiler_hygiene.yaml`
- [ ] Parse PN_PRINCIPLES.md → `pn_constraints.yaml`
- [ ] Parse ACCESSIBILITY_AND_CONTENT_NOTES.md → `accessibility_policy.yaml`
- [ ] Parse SOURCES_OF_TRUTH.md → `hot_cold_policy.yaml`
- [ ] Parse INCIDENT_RESPONSE.md → `incident_playbooks.yaml`

### Phase 8: Runtime Integration

- [ ] Update `SchemaToolGenerator._discover_artifact_mappings()` to use L5
- [ ] Update `RuntimeContextAssembler` to use all L5 definition types
- [ ] Add lifecycle enforcement using L5 lifecycle definitions
- [ ] Add intent authorization using L5 intent definitions
- [ ] Update `consult_*` tools to return structured L5 data
- [ ] Add taxonomy validation against L5 taxonomy definitions

---

## Success Criteria

1. **15 role profiles** with:
   - Non-empty `interface.inputs[]` and `interface.outputs[]`
   - Correct `state_key` mappings
   - Full `prompt_content` with operating_principles, anti_patterns, heuristics
   - Complete `protocol.intents` and `protocol.lifecycles`
   - All fields from Charter §1-§12 and Brief §0-§10 captured

2. **12 loop patterns** with:
   - Complete `topology.nodes[]` with role assignments
   - `topology.edges[]` reflecting procedure steps
   - `success_criteria.custom_checks[]` from success criteria
   - `failure_modes[]` from Guide §8
   - `protocol_flow` where L4 flow exists

3. **8 quality gates** with:
   - Full bar definitions, checks, failures, fixes
   - Ownership and loop assignments
   - Exception framework

4. **33 intent definitions** with:
   - Sender/receiver authorization
   - Payload type requirements
   - Lifecycle transition mappings

5. **4 lifecycle definitions** with:
   - All states and transitions
   - Guards and authorization rules

6. **10 taxonomy definitions** with:
   - All enum values from taxonomies.md
   - Descriptions for each value

7. **30+ artifact definitions** with:
   - Schema refs, state keys, field counts
   - Lifecycle associations
   - Hot/Cold and player_safe flags

8. **Policy definitions** for:
   - Spoiler hygiene rules and leak taxonomy
   - PN constraints and golden rules
   - Accessibility requirements
   - Hot/Cold boundaries

9. **Runtime tests pass** with generated L5 files

10. **No warnings** about missing artifact mappings or tool mappings

---

## Key Lessons from Failed Attempt

The previous attempt failed because:

1. **Captured only metadata, not content** — Added YAML frontmatter but didn't extract the rich
   structured content (operating_principles, interface.inputs/outputs, heuristics with examples)

2. **Ignored section structure** — The markdown files have consistent numbered sections that map
   directly to L5 fields. The generator must parse these sections, not just add frontmatter.

3. **Missed interface.inputs/outputs** — This is the MOST CRITICAL field for runtime. It tells
   `RuntimeContextAssembler` what tools to present and `SchemaToolGenerator` what write tools to
   generate. Without it, agents don't know what to read/write.

4. **Didn't resolve artifact names to state keys** — Charter says "Hook List" but L5 needs
   `artifact_type: hook_card, state_key: hot_sot.hooks`. This mapping is essential.

5. **Missed vast amounts of L0-L4 content** — The spec contains 15 core L0 documents, 52 L1 files,
   47 L2 files, 57+ L3 schemas, and 33 L4 files. The previous plan only addressed a fraction.

The correct approach requires:

- Section-aware markdown parsing for all document types
- Semantic extraction of bullets, tables, examples, and structured content
- Cross-reference resolution (artifact names → schema types → state keys)
- Complete coverage of all L0-L4 content, not just charters and loop guides
- Validation against L3 meta-schemas
- New L5 definition types for intents, lifecycles, taxonomies, artifacts, and policies

---

## References

**Templates:**

- Role charter: `spec/01-roles/_templates/ROLE_CHARTER.template.md`
- Agent brief: `spec/01-roles/_templates/AGENT_BRIEF.template.md`

**Meta-Schemas:**

- Role profile: `spec/03-schemas/definitions/role_profile.schema.json`
- Loop pattern: `spec/03-schemas/definitions/loop_pattern.schema.json`
- Quality gate: `spec/03-schemas/definitions/quality_gate.schema.json`

**Runtime Consumers:**

- Context assembly: `lib/runtime/src/questfoundry/runtime/core/runtime_context_assembler.py`
- Tool generation: `lib/runtime/src/questfoundry/runtime/core/schema_tool_generator.py`
- Node factory: `lib/runtime/src/questfoundry/runtime/core/node_factory.py`

**Source Inventories:**

- L0: 15 core docs + 12 loop guides + 15 playbooks = 42 files
- L1: 16 charters + 15 briefs + 3 interfaces + 2 checklists + 1 RACI = 37 files
- L2: 6 core refs + 41 artifact templates = 47 files
- L3: 57+ schema files
- L4: 5 core docs + 4 lifecycles + 6 flows + 15 examples = 30 files

**Total source files to parse:** ~210+
