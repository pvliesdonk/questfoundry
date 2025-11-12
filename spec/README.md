# QuestFoundry Specification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JSON Schema](https://img.shields.io/badge/JSON%20Schema-2020--12-blue)](https://json-schema.org/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**A layered, multi-agent specification for creating interactive nonlinear gamebooks**

QuestFoundry separates _what we do_ (roles, loops, quality bars) from _how machines speak_ (schemas,
protocol) and _how tools run_ (prompts, libraries, UI). Both humans and AI agents can play the
roles—as long as they communicate via structured, validated data.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture Layers](#architecture-layers)
- [Core Concepts](#core-concepts)
- [The Studio Workflow](#the-studio-workflow)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Overview

QuestFoundry is a **complete specification** for a collaborative interactive fiction authoring
studio. It defines:

- **15 roles** (Showrunner, Gatekeeper, Plotwright, Scene Smith, Lore Weaver, etc.)
- **22 artifact types** (Hook Cards, Trace Units, Canon Packs, Codex Entries, etc.)
- **28 JSON schemas** (22 artifact + 6 system) for machine-readable validation
- **Communication protocol** with state machines and message envelopes
- **AI agent prompts** implementing roles for Claude/ChatGPT/Gemini
- **8 quality bars** ensuring integrity, reachability, style consistency, accessibility

The specification is deliberately **layered** for clarity, testability, and traceability.

---

## Quick Start

### For Readers (Understanding the Spec)

1. **Start here**: Read [`00-north-star/README.md`](00-north-star/README.md) (5 min overview)
2. **Learn the model**: [`00-north-star/WORKING_MODEL.md`](00-north-star/WORKING_MODEL.md)
3. **Understand roles**: Browse [`01-roles/charters/`](01-roles/charters/)
4. **Explore artifacts**: [`02-dictionary/artifacts/`](02-dictionary/artifacts/)

### For Implementers (Building Tools)

1. **Protocol first**: [`04-protocol/ENVELOPE.md`](04-protocol/ENVELOPE.md)
2. **Schema reference**: [`03-schemas/`](03-schemas/)
3. **Implementation**: See `../lib/python/` for the Layer 6 Python library

### For AI Agents (Running Roles)

1. **Browse prompts**: [`05-prompts/`](05-prompts/)
2. **Usage guide**: [`05-prompts/USAGE_GUIDE.md`](05-prompts/USAGE_GUIDE.md)
3. **Example conversations**: Each role directory contains `/examples/`

---

## Architecture Layers

QuestFoundry is organized into **7 layers**, each with clear separation of concerns:

| Layer | Name                | Focus                                      | Status     | Entry Point                                          |
| ----- | ------------------- | ------------------------------------------ | ---------- | ---------------------------------------------------- |
| **0** | **North Star**      | Vision, principles, operating model        | ✅ 100%    | [`00-north-star/README.md`](00-north-star/README.md) |
| **1** | **Roles**           | Role charters, responsibilities, workflows | ✅ 100%    | [`01-roles/README.md`](01-roles/README.md)           |
| **2** | **Common Language** | Data dictionary, artifact templates        | ✅ 100%    | [`02-dictionary/README.md`](02-dictionary/README.md) |
| **3** | **Schemas**         | JSON Schema specifications (Draft 2020-12) | ✅ 100%    | [`03-schemas/README.md`](03-schemas/README.md)       |
| **4** | **Protocol**        | Message envelopes, intents, state machines | ✅ 100%    | [`04-protocol/README.md`](04-protocol/README.md)     |
| **5** | **Prompts**         | AI agent system prompts                    | ✅ 100%    | [`05-prompts/README.md`](05-prompts/README.md)       |
| **6** | **Libraries**       | Python SDK, validators, clients            | 🚧 Active  | `../lib/python/`                                     |
| **7** | **CLI**             | Command-line interface tools               | 📋 Planned | `../cli/python/`                                     |

**Legend:** ✅ Complete | 🚧 In Progress | 📋 Planned

### Why Layers?

- **Clarity** — Understand the studio without reading code
- **Replaceability** — Swap AI models/tools without changing canon or roles
- **Traceability** — Every change has a Trace Unit (TU) and Cold snapshot
- **Safety** — Player-Narrator (PN) sees only player-safe surfaces

---

## Core Concepts

### 🎭 The 15 Roles

QuestFoundry defines **15 roles** that can be played by humans or AI agents:

**Always On:**

- **Showrunner (SR)** — Orchestrates work, wakes roles, sequences loops
- **Gatekeeper (GK)** — Enforces quality bars, validates merges

**Default On:**

- **Plotwright (PW)** — Designs topology (hubs, loops, gateways)
- **Scene Smith (SS)** — Writes prose to topology & style
- **Style Lead (ST)** — Maintains voice, register, motifs
- **Lore Weaver (LW)** — Converts hooks into spoiler-level canon
- **Codex Curator (CC)** — Creates player-safe encyclopedia entries

**Optional/Dormant:**

- **Researcher (RS)** — Fact verification & corroboration
- **Art Director (AD)** / **Illustrator (IL)** — Visual planning/creation
- **Audio Director (AuD)** / **Audio Producer (AuP)** — Sound planning/creation
- **Translator (TR)** — Localization

**Downstream:**

- **Book Binder (BB)** — Assembles export views from Cold snapshots
- **Player-Narrator (PN)** — Performs the book in-world, enforces diegetic gates

See [`00-north-star/ROLE_INDEX.md`](00-north-star/ROLE_INDEX.md) for the complete directory.

### 📦 The 22 Artifact Types

All work in QuestFoundry produces **structured artifacts** with JSON schemas:

**Core Workflow:**

- `hook_card` — Small, traceable follow-ups to discovered needs
- `tu_brief` — Trace Unit work order tracking changes

**Content Creation:**

- `canon_pack` — Spoiler-level canon compilation
- `codex_entry` — Player-safe encyclopedia entries
- `style_addendum` — Voice/register/motif guidance
- `edit_notes` — Copyediting instructions

**Planning:**

- `research_memo` — Fact-checking & corroboration
- `shotlist` / `cuelist` — Visual/audio asset planning
- `art_plan` / `audio_plan` — Asset design briefs

**Quality:**

- `gatecheck_report` — Quality bar validation results
- `view_log` — Export manifest
- `front_matter` — Book metadata
- `pn_playtest_notes` — Player-Narrator testing feedback

**Localization:**

- `language_pack` — Translation structure
- `register_map` — Terminology mapping across languages

**Project:**

- `project_metadata` — Project-wide settings
- `art_manifest` / `style_manifest` — Asset catalogs

Browse templates: [`02-dictionary/artifacts/`](02-dictionary/artifacts/)

### 🔄 Hot vs. Cold (Sources of Truth)

QuestFoundry uses **two Sources of Truth**:

- **Hot** — Discovery space: drafts, hooks, spoilers, internal reasoning
- **Cold** — Curated canon & player-safe surfaces approved by Gatekeeper
- **Snapshots** — Immutable Cold exports tagged by date (`Cold @ YYYY-MM-DD`)
- **Views** — Specific exports of Cold snapshots (EPUB, web, etc.)

Changes move through **Trace Units (TUs)** with states:

```text
hot-proposed → stabilizing → gatecheck → cold-merged
```

See [`00-north-star/SOURCES_OF_TRUTH.md`](00-north-star/SOURCES_OF_TRUTH.md) for details.

### 🎯 The 8 Quality Bars

Before anything merges to Cold, **Gatekeeper** validates against 8 criteria:

1. **Integrity** — No dead references, valid IDs
2. **Reachability** — Keystones reachable from start
3. **Nonlinearity** — Hubs/loops/gateways meaningful
4. **Gateways** — Coherent diegetic checks
5. **Style** — Voice/register/motifs consistent
6. **Determinism** — Promised for assets when needed
7. **Presentation** — No spoilers on player surfaces
8. **Accessibility** — Navigation clear, alt text present, sensory considerations

See [`00-north-star/QUALITY_BARS.md`](00-north-star/QUALITY_BARS.md) for full criteria.

---

## The Studio Workflow

### 🎬 Micro-Loops (Targeted Work Cycles)

QuestFoundry organizes work into **12 focused loops** (Layer 5 includes 13 playbooks—the 12 loops
below plus a standalone Gatecheck playbook):

**Discovery:**

- **Story Spark** — Initial brainstorming
- **Hook Harvest** — Capture follow-up ideas
- **Lore Deepening** — Expand canon from hooks
- **Codex Expansion** — Create player-safe entries

**Refinement:**

- **Style Tune-up** — Voice/register consistency pass

**Assets:**

- **Art Touch-up** — Visual planning/creation
- **Audio Pass** — Sound planning/creation

**Localization:**

- **Translation Pass** — Target-language slice

**Export:**

- **Binding Run** — Export view on Cold
- **Narration Dry-Run** — PN playtesting

**Full Cycle:**

- **Full Production Run** — Orchestrates all loops

**Reflection:**

- **Post-Mortem** — Retrospective and lessons learned

Detailed guides: [`00-north-star/LOOPS/`](00-north-star/LOOPS/)

### 📋 Example Workflow

1. **Showrunner** scopes a Story Spark loop
2. **Plotwright** + **Scene Smith** create draft topology & prose in **Hot**
3. **Lore Weaver** documents hooks for canon expansion
4. **Gatekeeper** reviews against Quality Bars
5. Approved changes merge to **Cold**
6. **Book Binder** exports a Cold snapshot as EPUB
7. **Player-Narrator** playtests and files feedback

---

## Repository Structure

This directory contains **Layers 0-5** of the QuestFoundry specification:

```text
spec/
├── 00-north-star/          # Layer 0: Vision, principles, loops
│   ├── README.md           # Navigator
│   ├── WORKING_MODEL.md    # Operating model (Hot/Cold, roles, loops)
│   ├── QUALITY_BARS.md     # Gatekeeper validation criteria
│   ├── LOOPS/              # 12 detailed loop guides
│   └── ROLE_INDEX.md       # Directory of all roles
├── 01-roles/               # Layer 1: Role charters & briefs
│   ├── charters/           # 15 role charter documents
│   ├── briefs/             # Agent briefs for each role
│   └── interfaces/         # Role interaction patterns
├── 02-dictionary/          # Layer 2: Common language (human-level)
│   ├── artifacts/          # 22 artifact templates (markdown)
│   └── glossary.md         # Terminology reference
├── 03-schemas/             # Layer 3: JSON Schema specifications
│   ├── *.schema.json       # 28 JSON schemas (Draft 2020-12)
│   └── README.md           # Schema generation guide
├── 04-protocol/            # Layer 4: Communication protocol
│   ├── ENVELOPE.md         # Message format spec
│   ├── INTENTS.md          # Intent catalog
│   ├── LIFECYCLES/         # State machines (hooks, TUs)
│   ├── FLOWS/              # Message sequence diagrams
│   └── EXAMPLES/           # 20+ example messages
├── 05-prompts/             # Layer 5: AI agent prompts
│   ├── {role_name}/        # Per-role system prompts
│   │   ├── system_prompt.md
│   │   ├── intent_handlers/
│   │   └── examples/
│   └── _shared/            # Shared patterns (context, safety, escalation)
├── agents.md               # Specification editing guidelines
└── README.md               # This file
```

**Implementation** (Layers 6-7) is located in the parent mono-repo:

- **Layer 6**: `../lib/python/` — Python library implementation
- **Layer 7**: `../cli/python/` — CLI tools (planned)

---

## Key Documents

**Getting Started:**

- [`00-north-star/README.md`](00-north-star/README.md) — Navigator for Layer 0
- [`00-north-star/WORKING_MODEL.md`](00-north-star/WORKING_MODEL.md) — Studio operating model
- [`00-north-star/PN_PRINCIPLES.md`](00-north-star/PN_PRINCIPLES.md) — Player-Narrator boundaries

**Policy & Governance:**

- [`00-north-star/QUALITY_BARS.md`](00-north-star/QUALITY_BARS.md) — Gatekeeper validation criteria
- [`00-north-star/TRACEABILITY.md`](00-north-star/TRACEABILITY.md) — Trace Unit system
- [`00-north-star/SPOILER_HYGIENE.md`](00-north-star/SPOILER_HYGIENE.md) — Player-surface safety

**Technical Specs:**

- [`04-protocol/ENVELOPE.md`](04-protocol/ENVELOPE.md) — Message format specification
- [`04-protocol/INTENTS.md`](04-protocol/INTENTS.md) — Complete intent catalog
- [`04-protocol/LIFECYCLES/`](04-protocol/LIFECYCLES/) — State machines for hooks & TUs
- [`03-schemas/README.md`](03-schemas/README.md) — Schema generation methodology

**Implementation:**

- [`05-prompts/USAGE_GUIDE.md`](05-prompts/USAGE_GUIDE.md) — AI agent prompt usage
- `../lib/python/README.md` — Python library documentation

---

## License

This project is licensed under the **MIT License**. See [`../LICENSE`](../LICENSE) for details.

---

## Project Status

| Component            | Status           | Notes                                                                  |
| -------------------- | ---------------- | ---------------------------------------------------------------------- |
| Layer 0 (North Star) | ✅ 100% complete | 12 loops with full guides, 8 quality bars                              |
| Layer 1 (Roles)      | ✅ 100% complete | All 15 charters, 15 briefs, interfaces complete                        |
| Layer 2 (Dictionary) | ✅ 100% complete | All 22 artifacts enriched, glossary, taxonomies, cross-refs complete   |
| Layer 3 (Schemas)    | ✅ 100% complete | 28 schemas (22 artifacts + 6 system schemas) validated                 |
| Layer 4 (Protocol)   | ✅ 100% complete | protocol-v1.0.0: 4 lifecycles, 6 flows, intents, conformance, examples |
| Layer 5 (Prompts)    | ✅ 100% complete | Loop-focused architecture: 13 playbooks, 15 adapters, 15 full prompts  |
| Layer 6 (Libraries)  | 🚧 In Progress   | Python library at `../lib/python/`                                     |
| Layer 7 (CLI)        | 📋 Planned       | CLI tools at `../cli/python/`                                          |

---

**Built with ❤️ for interactive storytellers, game designers, and AI collaborators.**
