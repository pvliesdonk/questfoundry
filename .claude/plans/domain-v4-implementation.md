# Phase 4: QuestFoundry Mapping to New Meta-Model

**Tracking Issue**: [#130](https://github.com/pvliesdonk/questfoundry/issues/130)

## Status

- [x] **Wave 1**: Core Writing Studio (commit `3c3d19f1`)
- [ ] **Wave 2**: Research & Export (#131)
- [ ] **Wave 3**: Canon & Style (#132)
- [ ] **Wave 4**: Visual Assets (#133)
- [ ] **Wave 5**: Audio Assets (#134)
- [ ] **Wave 6**: Localization & Play (#135)

---

## Summary

Map QuestFoundry **v2 spec** (`_archive/spec/`) to the new domain-agnostic meta-model in `meta/`.
This replaces v3 domain entirely (cleanroom approach).

---

## Source Material: v2 Spec Structure

```
_archive/spec/
├── 00-north-star/     # Principles, policies, loop definitions
├── 01-roles/          # 15 role charters & briefs
├── 02-dictionary/     # Glossary, taxonomies, 40 artifact templates
├── 03-schemas/        # JSON Schema definitions
├── 04-protocol/       # Message flows (implied)
├── 05-definitions/    # Executable loop/role/gate definitions
└── 06-runtime/        # Implementation interfaces
```

---

## Mapping Analysis

### Core Primitives

| Meta-Model | v2 Source | Notes |
|------------|-----------|-------|
| **Studio** | (create new) | Container for all QF definitions |
| **Agent** | `01-roles/` | 15 roles → agents with archetypes |
| **Artifact Type** | `02-dictionary/artifacts/` | 40 templates → ordered field arrays |
| **Store** | `00-north-star/SOURCES_OF_TRUTH.md` | Hot/Cold/Snapshot/View semantics |
| **Playbook** | `00-north-star/LOOPS/` | 12 loops → DAG playbooks |
| **Tool** | `03-schemas/capabilities.schema.json` | Role capabilities |
| **Quality Criteria** | `00-north-star/QUALITY_BARS.md` | 8 mandatory bars |
| **Constitution** | `00-north-star/*.md` | North Star, Spoiler Hygiene, PN Principles |
| **Knowledge** | `02-dictionary/glossary.md` + prose | Stratify by layer |

### Agent Archetypes Mapping (15 Roles)

| v2 Role | Archetype(s) | Status | Notes |
|---------|--------------|--------|-------|
| **Showrunner** | `orchestrator` | Always-on | Entry agent for authoring |
| **Gatekeeper** | `validator` | Always-on | Quality bar enforcement |
| **Plotwright** | `creator` | Default-on | Topology design |
| **Scene Smith** | `creator` | Default-on | Prose writing |
| **Lore Weaver** | `curator` | Default-on | Exclusive to cold_store |
| **Codex Curator** | `curator` | Default-on | Player-safe references |
| **Style Lead** | `validator` | Default-on | Voice/register consistency |
| **Researcher** | `researcher` | Optional | Factual verification |
| **Art Director** | `creator` | Optional | Illustration planning |
| **Illustrator** | `creator` | Optional | Asset production |
| **Audio Director** | `creator` | Optional | Audio planning |
| **Audio Producer** | `creator` | Optional | Audio production |
| **Translator** | `creator` | Optional | Localization |
| **Book Binder** | `creator` | Downstream | Export assembly |
| **Player-Narrator** | `orchestrator` | Downstream | Runtime/play mode |

### Store Semantics Mapping

| v2 Concept | Meta-Model Store | Semantics | workflow_intent |
|------------|------------------|-----------|-----------------|
| Hot SoT | `workspace` | `mutable` | production: all |
| Cold SoT | `canon` | `cold` | production: exclusive (Lore Weaver) |
| Snapshot | `canon` version | `versioned` | Tagged immutable state |
| View | `exports` | `mutable` | Derived from snapshot |
| (new) | `audit` | `append_only` | Hook history, TU log |

---

## Phased Implementation

### Wave 1: Core Writing Studio ✅

**Agents** (4):

- Showrunner (orchestrator, entry_agent)
- Gatekeeper (validator)
- Plotwright (creator - structure/topology)
- Scene Smith (creator - prose/voice)

**Playbooks** (3):

- **Story Spark** - Structure loop (Plotwright → Section Briefs)
- **Scene Weave** - Prose loop (Scene Smith → Sections)
- **Hook Harvest** - Triage loop (Showrunner → Accepted Hooks)

**Key Insight**: Decouple logic from art

- Fix game-breaking logic in Story Spark without rewriting dialogue
- Polish voice in Scene Weave without risking quest logic

**Artifacts**: Section Brief, Section, Hook Card, Gatecheck Report

### Wave 2: Research & Export

**Agents** (2): Researcher, Book Binder
**Playbooks** (1): Binding Run

### Wave 3: Canon & Style

**Agents** (3): Style Lead, Lore Weaver, Codex Curator
**Playbooks** (1): Lore Deepening

### Wave 4: Visual Assets

**Agents** (2): Art Director, Illustrator
**Playbooks** (1): Art Pass

### Wave 5: Audio Assets

**Agents** (2): Audio Director, Audio Producer
**Playbooks** (1): Audio Pass

### Wave 6: Localization & Play

**Agents** (2): Translator, Player-Narrator
**Playbooks** (2): Translation Pass, Playtest

---

## Wave 1 Implementation Steps (All Complete)

1. ✅ Constitution - 10 inviolable principles
2. ✅ Artifact Types - Section Brief, Section, Hook Card, Gatecheck Report
3. ✅ Stores - workspace, canon, audit
4. ✅ Tools - consult_schema, validate_artifact, search_workspace, delegate
5. ✅ Agents - Showrunner, Gatekeeper, Plotwright, Scene Smith
6. ✅ Playbooks - Story Spark, Scene Weave, Hook Harvest
7. ✅ Quality Criteria - 8 bars with LLM rubrics
8. ✅ Knowledge - must_know (6) + role_specific (4)

---

## Directory Structure (Wave 1)

```text
domain-v4/
├── studio.json
├── agents/           # 4 files
├── artifact-types/   # 4 files
├── stores/           # 3 files
├── tools/            # 4 files
├── playbooks/        # 3 files
├── governance/
│   ├── constitution.json
│   └── quality-criteria/  # 8 files
└── knowledge/
    ├── layers.json
    ├── must_know/    # 6 files
    └── role_specific/  # 4 files
```

---

## Implementation Methodology

**Nouns First** from `meta/docs/README.md`:

1. Constitution (principles)
2. Artifact Types (what gets created)
3. Stores (where things live)
4. Tools (external capabilities)
5. Agents (who does work)
6. Playbooks (how work flows)
7. Quality Criteria (validation)
8. Knowledge (what agents know)
