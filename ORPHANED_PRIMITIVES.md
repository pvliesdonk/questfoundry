# Orphaned Primitives Report

**Generated:** 2025-11-15
**Validation Status:** ✅ PASS (53 warnings for unreferenced primitives)

## Overview

This document tracks behavior primitives (expertises, procedures, snippets) that exist in the spec but are not currently referenced by any playbook or adapter. These are **not errors** but rather opportunities for:

1. **Integration**: Reference them from appropriate playbooks/adapters
2. **Consolidation**: Merge with similar primitives if redundant
3. **Removal**: Delete if truly obsolete or superseded
4. **Documentation**: Understand gaps in current workflow coverage

## Summary by Type

- **Expertises**: 6 orphaned
- **Procedures**: 25 orphaned
- **Snippets**: 22 orphaned
- **Total**: 53 orphaned primitives

---

## Expertises (6)

These domain expertise definitions are not currently referenced by any playbook or adapter:

### 1. `translator_terminology`

**File:** `spec/05-behavior/expertises/translator_terminology.md`
**Purpose:** Bilingual glossary management, translation term consistency
**Why Orphaned:** Created during spec validation fix; needs integration into translation_pass playbook
**Recommendation:** Reference from `translator.adapter.yaml`

### 2. `gatekeeper_presentation`

**File:** `spec/05-behavior/expertises/gatekeeper_presentation.md`
**Purpose:** Enforce Presentation Quality Bar (#7) - spoiler safety, player-facing polish
**Why Orphaned:** Created during spec validation fix; Gatekeeper may use generic quality bar expertise
**Recommendation:** Reference from `gatekeeper.adapter.yaml` or merge into existing gatekeeper expertise

### 3. `codex_curator_terminology`

**File:** `spec/05-behavior/expertises/codex_curator_terminology.md`
**Purpose:** Glossary governance, cross-entry consistency, spoiler prevention
**Why Orphaned:** Created during spec validation fix; needs integration into codex_expansion playbook
**Recommendation:** Reference from `codex_curator.adapter.yaml`

### 4. `audio_producer_safety`

**File:** `spec/05-behavior/expertises/audio_producer_safety.md`
**Purpose:** Audio safety standards (volume, frequency, accessibility)
**Why Orphaned:** Created during spec validation fix; needs integration into audio_pass playbook
**Recommendation:** Reference from `audio_producer.adapter.yaml`

### 5. `lore_weaver_summarization`

**File:** `spec/05-behavior/expertises/lore_weaver_summarization.md`
**Purpose:** Canon compression, player-safe summarization for codex
**Why Orphaned:** Created during spec validation fix; may be covered by main lore_weaver_expertise
**Recommendation:** Reference from `lore_weaver.adapter.yaml` or merge into `lore_weaver_expertise`

### 6. `researcher_fact_checking`

**File:** `spec/05-behavior/expertises/researcher_fact_checking.md`
**Purpose:** Fact verification, uncertainty assessment, research memo production
**Why Orphaned:** Created during spec validation fix; Researcher role may be dormant in current workflows
**Recommendation:** Reference from `researcher.adapter.yaml`

---

## Procedures (25)

These workflow procedures are not currently referenced by any playbook or adapter:

### Audio Procedures (7)

1. **`audio_text_equivalents_captions`**
   Purpose: Create accessible text equivalents and captions for audio
   Recommendation: Reference from `audio_pass.playbook.yaml`

2. **`audio_reproducibility_planning`**
   Purpose: Plan deterministic audio rendering parameters
   Recommendation: Reference from `audio_pass.playbook.yaml` or merge into `audio_plan_authoring`

3. **`audio_caption_text_alignment`**
   Purpose: Align captions with audio timing and content
   Recommendation: Reference from `audio_pass.playbook.yaml`

4. **`audio_dynamic_range_safety`**
   Purpose: Ensure safe volume levels and dynamic range
   Note: Created during v1 extraction - should likely be referenced
   Recommendation: Reference from `audio_pass.playbook.yaml` or `audio_producer.adapter.yaml`

5. **`audio_determinism_logging`**
   Purpose: Log generation parameters for reproducible audio
   Note: Created during v1 extraction - should likely be referenced
   Recommendation: Reference from `audio_pass.playbook.yaml` or `audio_producer.adapter.yaml`

6. **`audio_mix_ready_delivery`**
   Purpose: Deliver mix-ready audio with proper formatting
   Recommendation: Reference from `audio_pass.playbook.yaml`

7. **`leitmotif_use_policy`**
   Purpose: Policy for recurring musical motif usage
   Note: May overlap with `leitmotif_management` (which IS referenced)
   Recommendation: Merge into `leitmotif_management` or differentiate purpose

### Art Procedures (3)

8. **`art_determinism_planning`**
   Purpose: Plan deterministic image generation parameters
   Recommendation: Reference from `art_touch_up.playbook.yaml`

9. **`art_caption_alt_guidance`**
   Purpose: Guidance for creating captions and alt text
   Note: May overlap with `alt_text_creation` (created during v1 extraction)
   Recommendation: Merge with `alt_text_creation` or differentiate purpose

10. **`visual_language_motif`**
    Purpose: Maintain visual motif consistency
    Note: May overlap with `visual_language_maintenance` (created during v1 extraction)
    Recommendation: Merge with `visual_language_maintenance`

### Binder/Export Procedures (3)

11. **`binder_integrity_enforcement`**
    Purpose: Validate anchors, links, cross-references
    Note: May overlap with `integrity_enforcement` (created during v1 extraction)
    Recommendation: Determine which to keep; likely merge or delete this one

12. **`binder_presentation_enforcement`**
    Purpose: Ensure exported content meets presentation quality bar
    Recommendation: Reference from `binding_run.playbook.yaml`

13. **`view_log_maintenance`**
    Purpose: Maintain view export logs for traceability
    Recommendation: Reference from `binding_run.playbook.yaml`

### Content Creation Procedures (5)

14. **`lore_translation`**
    Purpose: Translate canon to player-safe language
    Recommendation: Reference from `lore_deepening.playbook.yaml` or `codex_curator.adapter.yaml`

15. **`player_safe_encyclopedia`**
    Purpose: Create player-safe codex entries from canon
    Recommendation: Reference from `codex_expansion.playbook.yaml`

16. **`prose_drafting`**
    Purpose: Draft scene prose content
    Recommendation: Reference from `story_spark.playbook.yaml` or `scene_smith.adapter.yaml`

17. **`curator_gap_identification`**
    Purpose: Identify missing codex entries or glossary gaps
    Note: Created during v1 extraction
    Recommendation: Reference from `codex_expansion.playbook.yaml`

18. **`sensory_anchoring`**
    Purpose: Add sensory details to ground scenes
    Recommendation: Reference from `story_spark.playbook.yaml`

### Choice & Topology Procedures (2)

19. **`micro_context_management`**
    Purpose: Manage contextual details within scene micro-segments
    Recommendation: Reference from `story_spark.playbook.yaml`

20. **`contrastive_choice_design`**
    Purpose: Design meaningfully different choice options
    Note: May overlap with `contrastive_choice_polishing` (created during v1 extraction)
    Recommendation: Differentiate design (creation) vs polishing (refinement) or merge

### Translation Procedure (1)

21. **`register_map_idiom_strategy`**
    Purpose: Strategy for translating idioms while maintaining register
    Note: May overlap with `register_map_maintenance` (created during v1 extraction)
    Recommendation: Merge into `register_map_maintenance`

### Style Procedure (1)

22. **`voice_register_coherence`**
    Purpose: Ensure consistent voice and register
    Note: May overlap with `voice_coherence` (created during v1 extraction)
    Recommendation: Merge into `voice_coherence`

### Gatekeeper Procedures (2)

23. **`smallest_viable_fixes`**
    Purpose: Minimal fixes to pass gatecheck without over-correction
    Recommendation: Reference from gatecheck-related playbooks or `gatekeeper.adapter.yaml`

24. **`quality_bar_enforcement`**
    Purpose: Enforce all 8 quality bars during gatecheck
    Recommendation: Reference from `gatekeeper.adapter.yaml`

### Orchestration Procedure (1)

25. **`loop_orchestration`**
    Purpose: Coordinate multi-role loop execution
    Recommendation: Reference from `showrunner.adapter.yaml`

---

## Snippets (22)

These small reusable text blocks are not currently referenced:

### Safety & Hygiene (5)

1. **`spoiler_hygiene`** - Spoiler prevention reminder
2. **`pn_boundaries`** - Player-Narrator safety boundaries
3. **`pn_safety_invariant`** - PN receives only Cold + player_safe=true
4. **`no_internals`** - Never expose internal mechanics to players
5. **`cold_only_rule`** - Binder uses only Cold SoT

**Recommendation:** These are fundamental safety protocols. Reference from relevant adapters (lore_weaver, codex_curator, player_narrator, book_binder) or playbooks that involve player-facing content.

### Accessibility (4)

6. **`alt_text_quality`** - Alt text standards
7. **`text_equivalents_captions`** - Text equivalent requirements
8. **`safety_critical_audio`** - Audio safety warnings
9. **`accessibility`** - General accessibility standards

**Recommendation:** Reference from audio_pass, art_touch_up, binding_run playbooks and relevant adapters (audio_producer, illustrator, book_binder).

### Determinism & Technical (3)

10. **`determinism`** - Determinism logging requirements
11. **`technique_off_surfaces`** - Keep technical details in Hot
12. **`cold_manifest_validation`** - Cold manifest integrity checks

**Recommendation:** Reference from audio_pass, art_touch_up playbooks and relevant adapters (audio_producer, illustrator, book_binder).

### Style & Voice (4)

13. **`register_alignment`** - Register consistency across translations
14. **`presentation_normalization`** - Presentation quality standards
15. **`contrastive_choices`** - Choice differentiation principles
16. **`diegetic_gates`** - Diegetic gateway enforcement

**Recommendation:** Reference from translation_pass, style_tune_up, narration_dry_run playbooks and relevant adapters (translator, style_lead, player_narrator).

### Workflow & Process (6)

17. **`terminology`** - Terminology management standards
18. **`continuity_check_quick`** - Quick continuity validation
19. **`research_posture`** - Research uncertainty posture taxonomy
20. **`dormancy_policy`** - Dormant role wake/sleep discipline
21. **`handoff_checklist`** - Role handoff requirements
22. **`localization_support`** - Localization coordination
23. **`human_question_template`** - Format for escalating to human

**Recommendation:** Reference from appropriate adapters (codex_curator, researcher, showrunner, translator) and playbooks involving those roles.

---

## Action Items for Future Work

### High Priority (Likely Missing References)

1. **Audio safety procedures** (`audio_dynamic_range_safety`, `audio_determinism_logging`) - Should be in audio_pass playbook
2. **New expertises** (all 6) - Should be referenced from corresponding role adapters
3. **Safety snippets** (`spoiler_hygiene`, `pn_safety_invariant`, `no_internals`) - Core safety, should be widely referenced
4. **Accessibility snippets** - Should be in audio/art/binding playbooks

### Medium Priority (Possible Duplicates to Merge)

1. `leitmotif_use_policy` vs `leitmotif_management`
2. `art_caption_alt_guidance` vs `alt_text_creation`
3. `visual_language_motif` vs `visual_language_maintenance`
4. `binder_integrity_enforcement` vs `integrity_enforcement`
5. `contrastive_choice_design` vs `contrastive_choice_polishing`
6. `voice_register_coherence` vs `voice_coherence`
7. `register_map_idiom_strategy` vs `register_map_maintenance`

### Low Priority (Evaluate Need)

1. Specialized procedures that may be context-specific
2. Snippets that might be better integrated into procedures
3. Procedures that may have been superseded by newer patterns

---

## Investigation Workflow

For each orphaned primitive, determine:

1. **Is it needed?** Does it fill a gap in current workflows?
2. **Is it duplicate?** Does it overlap with another primitive?
3. **Where should it go?** Which playbook(s) or adapter(s) should reference it?
4. **Should it merge?** Can it be consolidated with similar primitives?

### Example Investigation: `audio_dynamic_range_safety`

- ✅ **Needed?** Yes - audio safety is critical for accessibility
- ✅ **Duplicate?** No - unique purpose (safety vs general audio work)
- 📍 **Where?** Should be in `audio_pass.playbook.yaml` and `audio_producer.adapter.yaml`
- ❌ **Merge?** No - distinct from audio_rendering or audio_plan_authoring

---

## Notes

- Orphaned primitives are **not errors** per spec design philosophy
- Some may be intentionally standalone (future use, optional enhancements)
- v1→v2 migration may have created duplicates that need reconciliation
- Safety and accessibility snippets should likely be widely referenced
