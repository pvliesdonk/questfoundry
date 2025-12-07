# Layer 2 — Common Language (Data Dictionary)

> **Status:** ✅ **ACTIVE — Phase 3 enrichment complete (2025-10-30)**
>
> Layer 2 defines the common language used across all roles and loops. Phase 3 (completed
> 2025-10-30) enriched all 20 artifact templates with inline field constraints, validation rules,
> and common error examples to support Layer 3 schema generation.
>
> This layer contains taxonomies, field registry, glossary, and artifact templates that form the
> data dictionary for QuestFoundry studio operations.

---

## What Layer 2 Is

**Purpose:** Define the common language — terminology, data structures, and artifact types — used
across all roles and loops.

**Scope:**

- Human-readable glossary of all system terms
- Taxonomies and classification systems (hook types, gate types, loop types, etc.)
- Artifact structure definitions (what fields/sections each work product has)
- Non-technical data dictionary (technical schemas come in Layer 3)

**Not here:**

- Policy/principles (Layer 0)
- Role responsibilities (Layer 1)
- JSON schemas (Layer 3)
- Protocol/wire formats (Layer 4)

---

## Current Contents (Parked)

### Glossary

- `glossary.md` — System terminology (moved from `00-north-star/TERMINOLOGY.md`)

### Artifacts

- `artifacts/*.md` — 17 work artifact templates (moved from `01-roles/templates/`)
  - Defines structure of hooks, TUs, canon packs, codex entries, style addenda, etc.
  - Currently human-readable templates; will inform Layer 3 schemas

---

## Layer 2 Structure (Planned)

```text
02-dictionary/
├── README.md                    # This file
├── glossary.md                  # All system terms (✅ parked)
├── taxonomies.md                # Classification systems (🚧 to be drafted)
│   ├── Hook lifecycle & types
│   ├── TU types
│   ├── Gate types & conditions
│   ├── Quality Bar categories
│   └── Loop classifications
│
└── artifacts/                   # Work product structures (✅ parked)
    ├── README.md               # Artifact index
    ├── hook_card.md
    ├── tu_brief.md
    ├── canon_pack.md
    ├── codex_entry.md
    ├── style_addendum.md
    ├── research_memo.md
    ├── shotlist.md
    ├── cuelist.md
    ├── art_plan.md
    ├── audio_plan.md
    ├── gatecheck_report.md
    ├── view_log.md
    ├── language_pack.md
    ├── register_map.md
    ├── edit_notes.md
    ├── front_matter.md
    └── pn_playtest_notes.md
```

---

## Normative References (Layer 0)

Layer 2 definitions must align with:

- `../00-north-star/QUALITY_BARS.md` — Quality standards
- `../00-north-star/SOURCES_OF_TRUTH.md` — Hot/Cold policy
- `../00-north-star/SPOILER_HYGIENE.md` — Player-safety rules
- `../00-north-star/PN_PRINCIPLES.md` — Presentation boundaries
- `../00-north-star/TRACEABILITY.md` — Trace policies

---

## Migration Notes

**Moved from Layer 0:**

- `00-north-star/TERMINOLOGY.md` → `02-dictionary/glossary.md`
  - Reason: Pure data dictionary, not policy

**Moved from Layer 1:**

- `01-roles/templates/*.md` → `02-dictionary/artifacts/`
  - Reason: Define WHAT artifacts ARE (data structure), not WHO creates them (role responsibility)
  - Layer 1 still defines which roles produce which artifacts (in charters/briefs)

**Cross-reference updates:**

- Layer 0 and Layer 1 files referencing these locations have been updated
- See
  [ADR-20251029-01-layer-boundary-clarification.md](../DECISIONS/ADR-20251029-01-layer-boundary-clarification.md)
  for rationale

---

## Contributing to Layer 2

Until Layer 2 is properly drafted:

1. Treat parked files as **read-only references**
2. Do not add new artifacts or terms here yet
3. If you need to reference a structure, point to the parked file but note it's pending review
4. Layer 2 proper draft should happen after Layer 0/1 stabilize

Once Layer 2 drafting begins:

1. Review all parked content for consistency
2. Add missing taxonomies
3. Ensure all terms are player-safe or clearly marked as Hot-only
4. Create artifact index with usage guidance
5. Align with Layer 3 schema planning

---

## Status

- ✅ Structure created
- ✅ Parked documents migrated
- ✅ Taxonomies complete (`taxonomies.md` with 8 sections)
- ✅ Field registry complete (`field_registry.md`)
- ✅ All 20 artifact templates enriched with constraints (Phase 3, 2025-10-29/30)
- ✅ Cross-references updated (loop_names.md, role_abbreviations.md)
- ✅ Conventions documented (choice_integrity.md)
- ✅ Cross-layer consistency review complete (2025-11-06)
