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

## Source Layer Structure

### L0: North Star (`00-north-star/`)

- **Loop Guides** (`LOOPS/*.md`) - 12 files, 9-section structure
- **Playbooks** (`PLAYBOOKS/*.md`) - 15 files, compact checklist format
- **Quality Bars** (`QUALITY_BARS.md`) - 8 bars with checks and ownership
- **Principles** (PN_PRINCIPLES, SPOILER_HYGIENE, SOURCES_OF_TRUTH, TRACEABILITY, etc.)

### L1: Roles (`01-roles/`)

- **Role Charters** (`charters/*.md`) - 15 files, fixed 12-section skeleton
- **Role Briefs** (`briefs/*.md`) - compact runtime summaries
- **Template** (`_templates/ROLE_CHARTER.template.md`) - canonical structure

### L2: Dictionary (`02-dictionary/`)

- Artifact templates, taxonomies, glossary
- Human-readable examples that outrank L3 schemas

### L3: Schemas (`03-schemas/`)

- JSON Schemas for artifacts (57 files)
- Meta-schemas for L5 definitions (`role_profile.schema.json`, `loop_pattern.schema.json`)

### L4: Protocol (`04-protocol/`)

- Envelope structure, intents, lifecycles, flows
- Message choreography between roles

---

## Target Layer Structure (L5)

### Role Profiles (`05-definitions/roles/*.yaml`)

Each role profile compiles from its L1 charter. The mapping:

| Charter Section | L5 Field | Runtime Usage |
|-----------------|----------|---------------|
| §1 Canon & Mission | `identity.*`, `prompt_content.core_mandate` | IDENTITY layer |
| §1 Canonical name | `identity.name` | Display name |
| §1 Aliases | `identity.aliases` | Alternative names |
| §2 In scope (SHOULD) | `prompt_content.operating_principles[]` | IDENTITY layer guidance |
| §2 Out of scope | `prompt_content.task_guidance` (implicit) | What NOT to do |
| §2 MAY/SHOULD/MUST | `protocol.intents`, `constraints.*` | Permissions |
| §3 Reads (Hot/Cold) | `interface.inputs[]` | STATE MANAGEMENT layer |
| §3 Produces | `interface.outputs[]` | Artifact extraction |
| §4 Participation + RACI | `raci.loops.*` | Loop membership |
| §5 Hook types | `protocol.lifecycles.hook.can_create` | Hook permissions |
| §6 Player-Surface | `constraints.safety.*`, `protocol.envelope_defaults` | PN safety |
| §7 Dormancy policy | `identity.dormancy_policy` | Activation logic |
| §7 Wake signals | `identity.wake_conditions[]` | When to activate |
| §9 Anti-patterns | `prompt_content.anti_patterns[]` | IDENTITY layer warnings |
| §10 Mini-Checklist | `prompt_content.checklist[]` | Self-validation |
| §11 Tiny Examples | `prompt_content.heuristics[]` with `.examples` | IDENTITY layer examples |

**Critical Fields for Runtime:**

```yaml
interface:
  inputs:
    - artifact_type: hook_card       # Schema name
      state_key: hot_sot.hooks       # Where to read
      required: false                # Mandatory?
      validation_mode: strict        # Schema validation
  outputs:
    - artifact_type: section_brief
      state_key: hot_sot.section_briefs
      validation_required: true
      merge_strategy: append         # How to combine
```

These drive:

- `RuntimeContextAssembler._build_state_management_layer()` - tells agent what tools to use
- `NodeFactory.extract_artifacts()` - knows where to find outputs
- `SchemaToolGenerator._discover_artifact_mappings()` - generates typed write tools

### Loop Patterns (`05-definitions/loops/*.yaml`)

Each loop pattern compiles from its L0 loop guide + playbook. The mapping:

| Guide Section | L5 Field | Runtime Usage |
|---------------|----------|---------------|
| §1 Triggers | `context.required_artifacts[]` | Loop preconditions |
| §3 Roles & Responsibilities | `topology.nodes[]` | Graph nodes |
| §3 Role designations (R/A/C/I) | `roles.required[]`, `.optional[]` | Participant validation |
| §4 Procedure steps | `topology.edges[]` | Graph transitions |
| §5 Deliverables (Hot) | `success_criteria.artifacts_produced[]` | Exit validation |
| §6 Merge Path | `handoffs[]` | Downstream loops |
| §7 Success Criteria | `success_criteria.custom_checks[]` | Exit validation |
| §8 RACI table | (stored in role_profile.raci, not duplicated) | Cross-reference |
| Quality bar refs | `gates.quality_bars[]` | Gate enforcement |

**Critical Fields for Runtime:**

```yaml
topology:
  entry_node: topology_draft
  nodes:
    - role_id: plotwright
      node_id: topology_draft
      description: "Map parts/chapters and designate hubs..."
  edges:
    - source: topology_draft
      target: section_briefs
      type: conditional
      condition:
        evaluator: state_key_match
        expression: "state.topology_complete == true"
```

These drive:

- `SchemaRegistry.load_loop()` - builds LoopPattern object
- `RuntimeContextAssembler` - MISSION layer with node context
- LangGraph graph construction

### Quality Gates (`05-definitions/quality_gates/*.yaml`)

Each quality gate compiles from `QUALITY_BARS.md`. The mapping:

| Quality Bar Field | L5 Field | Runtime Usage |
|-------------------|----------|---------------|
| Bar name | `bar_name` | Display |
| What it means | `bar_definition.what_it_means` | Agent understanding |
| Automation profile | `automation` | How to evaluate |
| Blocking policy | `blocking_policy` | Merge control |
| Checks | `validation_checks[]` | Gatekeeper evaluation |
| Owned by | `ownership.primary_owner` | Responsibility |
| Checked in | `ownership.checked_in_loops[]` | When applied |

---

## Section Parsing Patterns

### Role Charter: Fixed 12-Section Skeleton

All 15 role charters follow this exact structure:

```
## 1) Canon & Mission
   - Canonical name, aliases, one-sentence mission
   - Normative references (L0 links)

## 2) Scope & Shape
   - In scope (SHOULD focus on): bullets
   - Out of scope (SHOULD NOT own): bullets
   - Decisions & authority: MAY/SHOULD/MUST structure

## 3) Inputs & Outputs (human level)
   - Reads (inputs): Hot/Cold distinction
   - Produces (outputs): Named artifacts

## 4) Participation in Loops
   - Primary loops with RACI: **R:** role · **A:** Showrunner · **C:** ... · **I:** ...
   - Definition of done: bullets

## 5) Hook Policy (small ideas, big futures)
   - May propose hooks: types
   - Size, Tags, Lineage

## 6) Player-Surface Obligations
   - Spoiler Hygiene, PN boundaries, Accessibility

## 7) Dormancy & Wake Conditions
   - Default ON/dormant
   - Wake signals

## 8) Cross-Domain & Escalation
   - Coordination patterns

## 9) Anti-patterns (don't do this)
   - Named patterns with descriptions

## 10) Mini-Checklist (run every time)
   - Checkbox items

## 11) Tiny Examples
   - Before → After pairs

## 12) Metadata
   - Lineage, Related links
```

### Loop Guide: Consistent 9-Section Pattern

All 12 loop guides follow this structure:

```
## 1) Triggers (Showrunner)
   - Activation conditions
   - TU opening steps

## 2) Inputs
   - Cold snapshot, prior notes, hooks, QA findings

## 3) Roles & Responsibilities
   - Per-role with (R/A/C/I) designation
   - Specific action verbs

## 4) Procedure
   - Numbered steps (6-10 typically)
   - Mixed role participation

## 5) Deliverables (Hot)
   - Bulleted artifact list

## 6) Merge Path / Prioritization / Handoffs
   - Downstream loop routing

## 7) Success Criteria
   - Observable outcomes + quality bar refs

## 8) Failure Modes & Remedies / RACI
   - Problem/solution pairs OR RACI table

## 9) [Optional additions]
   - RACI if not in §8
   - Anti-patterns
   - Notes on dormant roles
```

### Playbook: Compact Checklist Format

Playbooks condense loop guides into action-ready checklists:

```
## Use when: [preamble + outcome]

## One-minute scope (Showrunner)
   - Checkboxes

## Inputs you need on screen
   - Bulleted list

## Do the thing (compact procedure)
   - Per-role blocks with numbered steps

## Deliverables (Hot)
   - Bulleted list

## Hand-off map / Definition of done
   - Directional arrows or DoD checklist

## Fast triage rubric / Success criteria
   - 3-5 decision criteria

## RACI (micro)
   - Compact table

## Anti-patterns to catch
   - Bullets

## Cheat line
   - Single summary sentence
```

---

## Normative vs Advisory Rules

### Normative (Hard Enforcement)

These are LAW. Violations block execution:

1. **TU Lifecycle Transitions** (from L4 protocol)
   - Valid states: hot-proposed → stabilizing → gatecheck → cold-merged
   - Invalid transitions raise `StateError`

2. **Envelope Schema** (from L4)
   - Required fields: sender, receiver, intent, tu_id
   - Safety fields: player_safe, spoilers

3. **Artifact Schemas** (from L3)
   - JSON Schema validation on write
   - Required fields per artifact type

4. **Hot/Cold Permissions** (from L5 role profiles)
   - `constraints.hot_cold_permissions.cold.write: false` blocks cold writes
   - PN roles have `pn_safe: true` requirement

5. **Protocol Intent Permissions** (from L5)
   - `protocol.can_send` limits which intents a role can emit
   - Sending unauthorized intent is blocked

### Advisory (Soft Guidance)

These are HINTS. Deviations log warnings but don't block:

1. **Playbook Node Order**
   - Recommended sequence from procedure steps
   - Skipping steps is allowed if lifecycle rules pass

2. **Consult Requirements**
   - "Should consult playbook before tu.open"
   - Warn if not consulted, but allow proceed

3. **Role Wake Suggestions**
   - Dormancy wake signals are hints
   - Showrunner can override

4. **Success Criteria Checks**
   - `success_criteria.custom_checks[]` are advisory
   - Failing a check warns but doesn't block handoff

---

## Generator Architecture

### Phase 1: Parser Development

Build parsers for each source format:

```
RoleCharterParser:
  - Input: 01-roles/charters/{role}.md
  - Output: Structured dict with all 12 sections
  - Key extractions:
    - Section 2 bullets → operating_principles
    - Section 3 Reads/Produces → interface.inputs/outputs
    - Section 4 RACI patterns → raci.loops
    - Section 9 anti-patterns → anti_patterns
    - Section 11 examples → heuristics with examples

LoopGuideParser:
  - Input: 00-north-star/LOOPS/{loop}.md
  - Output: Structured dict with all 9 sections
  - Key extractions:
    - Section 3 role designations → topology.nodes
    - Section 4 procedure → topology.edges
    - Section 7 success criteria → success_criteria.custom_checks

PlaybookParser:
  - Input: 00-north-star/PLAYBOOKS/playbook_{loop}.md
  - Output: Compact procedure + RACI
  - Used to augment loop patterns with execution hints

QualityBarsParser:
  - Input: 00-north-star/QUALITY_BARS.md
  - Output: 8 quality gate definitions
  - Key extractions:
    - Bar definitions, checks, ownership, blocking policy
```

### Phase 2: Cross-Reference Resolution

After parsing, resolve references:

1. **Artifact Types → Schemas**
   - "Hook List" in charter → `hook_card.schema.json`
   - "Section Briefs" → `section_brief.schema.json`

2. **State Keys**
   - "Hot: prior topology notes" → `hot_sot.topology_notes`
   - "Cold: current snapshot" → `cold_sot.snapshot`

3. **RACI Loops**
   - "Story Spark — **R:** Plotwright" → `raci.loops.story_spark.responsible: true`

4. **Quality Bars**
   - "Reachability/Nonlinearity/Gateways" → `quality_bars_owned: [...]`

### Phase 3: L5 Generation

Generate YAML with full structure:

```python
def generate_role_profile(role_id: str, parsed_charter: dict) -> dict:
    return {
        "id": role_id,
        "identity": extract_identity(parsed_charter),
        "prompt_content": extract_prompt_content(parsed_charter),
        "interface": extract_interface(parsed_charter),  # CRITICAL
        "behavior": infer_behavior(parsed_charter),
        "protocol": extract_protocol(parsed_charter),
        "constraints": extract_constraints(parsed_charter),
        "raci": extract_raci(parsed_charter),
        "metadata": generate_metadata(),
    }

def extract_interface(parsed: dict) -> dict:
    """Extract interface.inputs/outputs from Section 3."""
    inputs = []
    outputs = []

    for read in parsed["section_3"]["reads"]:
        inputs.append({
            "artifact_type": resolve_artifact_type(read["artifact"]),
            "state_key": resolve_state_key(read["location"]),  # hot_sot.* or cold_sot.*
            "required": read.get("required", False),
            "validation_mode": "strict",
        })

    for produce in parsed["section_3"]["produces"]:
        outputs.append({
            "artifact_type": resolve_artifact_type(produce["artifact"]),
            "state_key": resolve_state_key(produce["location"]),
            "validation_required": True,
            "merge_strategy": infer_merge_strategy(produce),
        })

    return {"inputs": inputs, "outputs": outputs, "side_effects": [...]}
```

### Phase 4: Validation

Validate generated YAML against L3 meta-schemas:

```bash
# Validate all role profiles
for file in spec/05-definitions/roles/*.yaml; do
    jsonschema --instance "$file" spec/03-schemas/definitions/role_profile.schema.json
done

# Validate all loop patterns
for file in spec/05-definitions/loops/*.yaml; do
    jsonschema --instance "$file" spec/03-schemas/definitions/loop_pattern.schema.json
done
```

---

## Runtime Integration Points

### RuntimeContextAssembler

Consumes L5 to build 6-layer prompts:

1. **IDENTITY** ← `role.prompt_content` + `role.identity`
2. **PROTOCOL** ← `role.protocol` + L4 definitions
3. **STATE** ← current `StudioState` snapshot
4. **STATE MANAGEMENT** ← `role.interface.inputs/outputs` (teaches agent which tools)
5. **MISSION** ← `loop.topology.nodes[current].description` + `role.task_guidance`
6. **INTERFACE** ← available tools from capabilities

### NodeFactory

Uses L5 for node creation:

- `identity.dormancy_policy` → `should_execute_role()`
- `behavior.model_config` → `select_llm()`
- `interface.outputs` → `extract_artifacts()`

### SchemaToolGenerator

Uses L5 for tool generation:

- `interface.outputs[].artifact_type` → generates `write_{artifact_type}` tools
- `interface.outputs[].state_key` → maps to `hot_sot.*` keys

### BindToolsExecutor

Can enforce consult requirements:

```python
# Before tu.open, verify playbook was consulted
if intent == "tu.open":
    if not state.get("_consulted_playbook"):
        raise ConsultRequiredError("Must consult_playbook before tu.open")
```

---

## Artifact Type → State Key Mapping

The generator must resolve human-readable artifact names to state keys:

| Human Name (Charter) | artifact_type | state_key |
|----------------------|---------------|-----------|
| Hook List / Hooks | hook_card | hot_sot.hooks |
| Canon Pack / Canon summaries | canon_pack | hot_sot.canon |
| Style guardrails / Style addendum | style_addendum | hot_sot.style |
| Topology notes | topology_notes | hot_sot.topology |
| Section briefs | section_brief | hot_sot.section_briefs |
| Section drafts | section_draft | hot_sot.drafts |
| Gateway map | gateway_map | hot_sot.gateway_map |
| Codex entries | codex_entry | hot_sot.codex |
| Art plan | art_plan | hot_sot.art_plan |
| Audio plan | audio_plan | hot_sot.audio_plan |
| Current snapshot | cold_snapshot | cold_sot.snapshot |
| Manuscript sections | section | cold_sot.sections |
| Published codex | codex_pack | cold_sot.codex |

---

## Implementation Phases

### Phase 1: Infrastructure

- [ ] Create `spec/tools/generate_l5.py` with argument parsing
- [ ] Implement section-aware markdown parser for charters
- [ ] Implement section-aware markdown parser for loop guides
- [ ] Create artifact type → state key mapping table

### Phase 2: Role Profile Generation

- [ ] Parse all 15 role charters
- [ ] Extract `interface.inputs/outputs` from Section 3
- [ ] Extract `prompt_content.*` from Sections 2, 9, 11
- [ ] Extract `raci.loops` from Section 4
- [ ] Generate and validate role profiles

### Phase 3: Loop Pattern Generation

- [ ] Parse all 12 loop guides
- [ ] Extract `topology.nodes` from Section 3
- [ ] Extract `topology.edges` from Section 4
- [ ] Extract `success_criteria` from Section 7
- [ ] Generate and validate loop patterns

### Phase 4: Quality Gate Generation

- [ ] Parse QUALITY_BARS.md
- [ ] Extract 8 quality gate definitions
- [ ] Generate and validate quality gates

### Phase 5: Runtime Integration

- [ ] Update `SchemaToolGenerator._discover_artifact_mappings()` to use L5
- [ ] Update `RuntimeContextAssembler` to surface structured L5 data
- [ ] Add consult requirement enforcement in `BindToolsExecutor`
- [ ] Update `consult_*` tools to return structured frontmatter

---

## Success Criteria

1. **All 15 role profiles generate** from charters with:
   - Non-empty `interface.inputs` and `interface.outputs`
   - Correct `state_key` mappings (hot_sot.*/ cold_sot.*)
   - Full `prompt_content` with operating_principles, anti_patterns, heuristics

2. **All 12 loop patterns generate** from guides with:
   - Complete `topology.nodes` with role assignments
   - `topology.edges` reflecting procedure steps
   - `success_criteria.custom_checks` from success criteria

3. **All 8 quality gates generate** from QUALITY_BARS.md

4. **Runtime tests pass** with generated L5 files

5. **No warnings** about missing artifact mappings or tool mappings

---

## Key Lessons from Failed Attempt

The previous attempt failed because:

1. **Captured only metadata, not content** - Added YAML frontmatter but didn't extract the rich
   structured content (operating_principles, interface.inputs/outputs, heuristics with examples)

2. **Ignored section structure** - The markdown files have consistent numbered sections that map
   directly to L5 fields. The generator must parse these sections, not just add frontmatter.

3. **Missed interface.inputs/outputs** - This is the MOST CRITICAL field for runtime. It tells
   `RuntimeContextAssembler` what tools to present and `SchemaToolGenerator` what write tools to
   generate. Without it, agents don't know what to read/write.

4. **Didn't resolve artifact names to state keys** - Charter says "Hook List" but L5 needs
   `artifact_type: hook_card, state_key: hot_sot.hooks`. This mapping is essential.

The correct approach requires:

- Section-aware markdown parsing
- Semantic extraction of bullets, tables, and examples
- Cross-reference resolution (artifact names → schema types → state keys)
- Validation against L3 meta-schemas

---

## References

- Role charter template: `spec/01-roles/_templates/ROLE_CHARTER.template.md`
- L5 role schema: `spec/03-schemas/definitions/role_profile.schema.json`
- L5 loop schema: `spec/03-schemas/definitions/loop_pattern.schema.json`
- Runtime context assembly: `lib/runtime/src/questfoundry/runtime/core/runtime_context_assembler.py`
- Schema tool generation: `lib/runtime/src/questfoundry/runtime/core/schema_tool_generator.py`
