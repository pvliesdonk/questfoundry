# Story Architecture Overhaul - Implementation Plan

**Issue:** #284
**Tracking:** #285
**Branch:** `feature/284-story-architecture`

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PR Scope | All 3 Epics in one PR | Single comprehensive change |
| Terminology | Adopt Twine/Twee terms | IF ecosystem compatibility |
| Topology Versioning | Single mutable artifact | Simpler model |

## Terminology Migration

| Current | New (Twine) | Notes |
|---------|-------------|-------|
| `section` | `passage` | Core narrative unit |
| `section_brief` | `passage_brief` | Planning document |
| `anchor` field | `pid` | Passage identifier |
| `target_anchor` | `target` | Link destination |
| (new) | `story` | Root manifest with IFID |
| (new) | `ifid` | UUID v4 story identity |

## Epic 1: The Structural Foundation

Create `story` and `topology` artifacts; refactor `story_spark`.

### New Artifacts

**story.json** - Root manifest with:
- `ifid` (UUID v4 story identity)
- `name`, `start`, `format`, `logline`, `theme`, `genre`, `tone`
- `current_pass`, `active_phase`
- `tags` (Twine-compatible)

**topology.json** - Graph structure with:
- `ifid` (references parent story)
- `passages[]` - nodes with `pid`, `name`, `tags`, `is_start`, `is_ending`, `topology_role`
- `links[]` - edges with `from`, `to`, `gate`

## Epic 2: The Narrative Alignment

Anchor prose generation to topology.

- Add `topology_passage_ref` to passage and passage_brief
- Create `passage_from_topology` relationship
- Update `scene_weave` to read and validate against topology

## Epic 3: Knowledge & Lifecycle Refinement

Improve hook targeting and pass tracking.

- Enhance `hook_card` with polymorphic `target_ref`
- Enhance `canon_pack` with `origin_hooks`
- Implement pass tracking on `story` artifact

## Files Impacted

- **New files:** 5 (story, topology, 3 relationships)
- **Renamed files:** 3 (section→passage, section_brief→passage_brief, relationship)
- **Modified files:** ~42 (agents, playbooks, knowledge, stores)

## Meta Schema Compliance

No meta schema changes required. All new artifacts conform to existing schemas.
