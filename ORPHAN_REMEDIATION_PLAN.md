# Orphaned Primitives Remediation Plan

**Status:** ✅ Validation passes (53 orphan warnings)
**Goal:** Integrate, merge, or remove orphaned primitives systematically
**Approach:** Combine Gemini's execution framework with detailed primitive analysis

---

## Executive Summary

We have **53 orphaned primitives** that validation reports as unreferenced:
- **6 Expertises** - Created during v1 extraction, need adapter integration
- **25 Procedures** - Mix of orphans and likely duplicates
- **22 Snippets** - Reusable blocks awaiting reference

This plan provides a **systematic remediation strategy** using:
1. **Gemini's 3-phase approach** (validate → remediate → confirm)
2. **Claude's detailed categorization** (integration vs merge vs delete decisions)

---

## Phase 0: Current State Assessment

### Validation Status
```bash
cd lib/compiler && uv run qf-compile --spec-dir ../../spec --validate-only
```
**Result:** ✅ PASS with 53 orphan warnings (not errors)

### Key Finding
These orphans fall into **three categories**:

| Category | Count | Action Required |
|----------|-------|----------------|
| **Integration Needed** | ~20 | Reference from playbooks/adapters |
| **Likely Duplicates** | ~8 | Merge into canonical version |
| **Evaluate/Delete** | ~25 | Determine if truly obsolete |

---

## Phase 1: Systematic Remediation

### Strategy: Triage by Type

We'll process orphans in order of **confidence and impact**:

1. **High Confidence: New Expertises** (6 files) - Clearly should be referenced
2. **Medium Confidence: Duplicates** (8 files) - Merge into canonical versions
3. **Low Confidence: Procedures** (17 files) - Requires case-by-case analysis
4. **Lowest Priority: Snippets** (22 files) - Evaluate last

---

## Remediation Queue

### Priority 1: Reference New Expertises (6 fixes)

These were created during v1 extraction and clearly belong in adapters:

#### 1.1 `translator_terminology` → `translator.adapter.yaml`
**File:** `spec/05-behavior/expertises/translator_terminology.md`
**Action:** Add to `translator.adapter.yaml`:
```yaml
references_expertises:
  - translator_terminology
```
**Rationale:** Bilingual glossary management is core to translator role

#### 1.2 `codex_curator_terminology` → `codex_curator.adapter.yaml`
**File:** `spec/05-behavior/expertises/codex_curator_terminology.md`
**Action:** Add to `codex_curator.adapter.yaml`:
```yaml
references_expertises:
  - codex_curator_terminology
```
**Rationale:** Glossary governance is core to curator role

#### 1.3 `audio_producer_safety` → `audio_producer.adapter.yaml`
**File:** `spec/05-behavior/expertises/audio_producer_safety.md`
**Action:** Add to `audio_producer.adapter.yaml`:
```yaml
references_expertises:
  - audio_producer_safety
```
**Rationale:** Audio safety standards are critical for producer role

#### 1.4 `researcher_fact_checking` → `researcher.adapter.yaml`
**File:** `spec/05-behavior/expertises/researcher_fact_checking.md`
**Action:** Add to `researcher.adapter.yaml`:
```yaml
references_expertises:
  - researcher_fact_checking
```
**Rationale:** Fact verification is core to researcher role

#### 1.5 `gatekeeper_presentation` → `gatekeeper.adapter.yaml`
**File:** `spec/05-behavior/expertises/gatekeeper_presentation.md`
**Decision Required:** Reference OR merge into existing `gatekeeper_quality_bars`
**Recommendation:** **Merge** into `gatekeeper_quality_bars` (consolidate quality bar expertise)

#### 1.6 `lore_weaver_summarization` → `lore_weaver.adapter.yaml`
**File:** `spec/05-behavior/expertises/lore_weaver_summarization.md`
**Decision Required:** Reference OR merge into existing `lore_weaver_expertise`
**Recommendation:** **Merge** into `lore_weaver_expertise` (consolidate lore expertise)

---

### Priority 2: Merge Duplicate Procedures (8 merges)

These have clear canonical versions that should subsume the orphans:

#### 2.1 Merge `leitmotif_use_policy` → `leitmotif_management`
**Orphan:** `spec/05-behavior/procedures/leitmotif_use_policy.md`
**Canonical:** `spec/05-behavior/procedures/leitmotif_management.md` (IS referenced)
**Action:**
1. Review both files
2. Merge unique content from `leitmotif_use_policy` into `leitmotif_management`
3. Delete `leitmotif_use_policy.md`

#### 2.2 Merge `binder_integrity_enforcement` → `integrity_enforcement`
**Orphan:** `spec/05-behavior/procedures/binder_integrity_enforcement.md`
**Canonical:** `spec/05-behavior/procedures/integrity_enforcement.md` (IS referenced)
**Action:**
1. Review both files (both do anchor/link validation)
2. Merge unique content from `binder_integrity_enforcement` into `integrity_enforcement`
3. Delete `binder_integrity_enforcement.md`

#### 2.3 Merge `contrastive_choice_design` → `contrastive_choice_polishing`
**Orphan:** `spec/05-behavior/procedures/contrastive_choice_design.md`
**Canonical:** `spec/05-behavior/procedures/contrastive_choice_polishing.md` (created during v1 extraction)
**Decision:** Differentiate (design = creation, polishing = refinement) OR merge?
**Recommendation:** **Differentiate** - keep both, but reference `contrastive_choice_design` from `story_spark.playbook.yaml`

#### 2.4 Merge `voice_register_coherence` → `voice_coherence`
**Orphan:** `spec/05-behavior/procedures/voice_register_coherence.md`
**Canonical:** `spec/05-behavior/procedures/voice_coherence.md` (created during v1 extraction)
**Action:**
1. Merge into `voice_coherence`
2. Delete `voice_register_coherence.md`

#### 2.5 Merge `art_caption_alt_guidance` → `alt_text_creation`
**Orphan:** `spec/05-behavior/procedures/art_caption_alt_guidance.md`
**Canonical:** `spec/05-behavior/procedures/alt_text_creation.md` (created during v1 extraction)
**Action:**
1. Merge guidance into `alt_text_creation`
2. Delete `art_caption_alt_guidance.md`

#### 2.6 Merge `visual_language_motif` → `visual_language_maintenance`
**Orphan:** `spec/05-behavior/procedures/visual_language_motif.md`
**Canonical:** `spec/05-behavior/procedures/visual_language_maintenance.md` (created during v1 extraction)
**Action:**
1. Merge motif-specific content into `visual_language_maintenance`
2. Delete `visual_language_motif.md`

#### 2.7 Merge `register_map_idiom_strategy` → `register_map_maintenance`
**Orphan:** `spec/05-behavior/procedures/register_map_idiom_strategy.md`
**Canonical:** `spec/05-behavior/procedures/register_map_maintenance.md` (created during v1 extraction)
**Action:**
1. Merge idiom strategy into `register_map_maintenance`
2. Delete `register_map_idiom_strategy.md`

#### 2.8 Review `audio_determinism_logging` vs `determinism_logging`
**Orphan:** `spec/05-behavior/procedures/audio_determinism_logging.md`
**Canonical:** `spec/05-behavior/procedures/determinism_logging.md` (created during v1 extraction, covers both audio & art)
**Action:**
1. Verify `determinism_logging` covers audio-specific needs
2. If yes, delete `audio_determinism_logging.md`
3. If no, differentiate and reference from `audio_pass.playbook.yaml`

---

### Priority 3: Reference High-Value Procedures (9 references)

These should be referenced from playbooks/adapters:

#### 3.1 Audio Procedures → `audio_pass.playbook.yaml`

Add to `audio_pass.playbook.yaml` references:
```yaml
references_procedures:
  - audio_text_equivalents_captions     # Step: Create captions
  - audio_caption_text_alignment        # Step: Align captions
  - audio_dynamic_range_safety          # Step: Safety check
  - audio_mix_ready_delivery            # Step: Final delivery
```

#### 3.2 Audio Procedures → `audio_producer.adapter.yaml`

Add to `audio_producer.adapter.yaml`:
```yaml
references_procedures:
  - audio_reproducibility_planning      # For deterministic rendering
```

#### 3.3 Art Procedure → `art_touch_up.playbook.yaml`

Add to `art_touch_up.playbook.yaml`:
```yaml
references_procedures:
  - art_determinism_planning
```

#### 3.4 Binder Procedure → `binding_run.playbook.yaml`

Add to `binding_run.playbook.yaml`:
```yaml
references_procedures:
  - binder_presentation_enforcement     # Pre-export quality check
  - view_log_maintenance                # Logging
```

#### 3.5 Content Procedures

- `lore_translation` → Reference from `lore_weaver.adapter.yaml` or `codex_curator.adapter.yaml`
- `player_safe_encyclopedia` → Reference from `codex_expansion.playbook.yaml`
- `curator_gap_identification` → Reference from `codex_expansion.playbook.yaml`
- `prose_drafting` → Reference from `story_spark.playbook.yaml`
- `sensory_anchoring` → Reference from `story_spark.playbook.yaml`

---

### Priority 4: Evaluate/Delete Low-Priority Procedures (8 evaluations)

These require case-by-case review:

#### 4.1 `smallest_viable_fixes`
**Decision:** Reference from `gatekeeper.adapter.yaml` OR delete if covered by existing procedures
**Action:** Review gatekeeper procedures, add if useful

#### 4.2 `quality_bar_enforcement`
**Decision:** Reference from `gatekeeper.adapter.yaml` OR merge into existing quality procedures
**Action:** Review overlap with other gatekeeper procedures

#### 4.3 `loop_orchestration`
**Decision:** Reference from `showrunner.adapter.yaml` OR delete if covered by existing orchestration
**Action:** Check if showrunner already has orchestration procedures

#### 4.4 `micro_context_management`
**Decision:** Reference from `story_spark.playbook.yaml` OR delete
**Action:** Evaluate if this granularity is needed

#### 4.5-4.8 Other specialized procedures
Evaluate each for unique value vs redundancy with existing procedures.

---

### Priority 5: Snippet Triage (22 snippets)

**Strategy:** Batch reference from relevant playbooks/adapters

#### 5.1 Safety Snippets (High Priority)
Reference from multiple locations:
```yaml
# In lore_weaver.adapter.yaml, codex_curator.adapter.yaml, player_narrator.adapter.yaml:
references_snippets:
  - spoiler_hygiene
  - pn_boundaries
  - pn_safety_invariant
  - no_internals
```

#### 5.2 Accessibility Snippets
Reference from audio/art playbooks:
```yaml
# In audio_pass.playbook.yaml, art_touch_up.playbook.yaml:
references_snippets:
  - alt_text_quality
  - text_equivalents_captions
  - safety_critical_audio
  - accessibility
```

#### 5.3 Technical Snippets
Reference from relevant contexts:
```yaml
# In audio_pass.playbook.yaml, art_touch_up.playbook.yaml:
references_snippets:
  - determinism
  - technique_off_surfaces

# In binding_run.playbook.yaml:
references_snippets:
  - cold_manifest_validation
  - cold_only_rule
```

#### 5.4 Process Snippets
Reference from appropriate adapters:
```yaml
# Various adapters:
references_snippets:
  - terminology                    # codex_curator, translator
  - continuity_check_quick         # lore_weaver
  - research_posture               # researcher
  - dormancy_policy                # all dormant roles
  - handoff_checklist              # showrunner
  - human_question_template        # all roles
```

---

## Phase 2: Implementation Workflow

### Step-by-Step Process

For each remediation item:

1. **Create branch:** `git checkout -b fix/orphan-[category]`
2. **Make changes:**
   - For **references**: Edit adapter/playbook YAML
   - For **merges**: Combine files, delete orphan
   - For **deletes**: Remove file
3. **Validate:** `qf-compile --validate-only`
4. **Commit:** Clear commit message explaining decision
5. **Test:** Ensure no new errors introduced
6. **Merge:** Back to main work branch

### Batch Strategy

Group related changes:
- **Batch 1:** All 6 expertise references (single commit)
- **Batch 2:** All 8 merges (individual commits per merge)
- **Batch 3:** Audio procedure references (single commit)
- **Batch 4:** Art/binder procedure references (single commit)
- **Batch 5:** Safety snippet references (single commit)
- **Batch 6:** Remaining snippet references (single commit)

---

## Phase 3: Validation & Verification

### Final Checks

1. **Run validator:**
   ```bash
   qf-compile --validate-only
   ```
   **Target:** Zero orphan warnings (or documented justification for remaining)

2. **Review MIGRATION_TRACKING.csv:**
   Document all merges and deletions

3. **Spot-check references:**
   Ensure added references make semantic sense in context

4. **Run full compilation:**
   ```bash
   qf-compile --spec-dir spec/ --output dist/compiled/
   ```
   **Target:** Successful compilation with no errors

---

## Decision Matrix

For each orphaned primitive, apply this decision tree:

```
Is there a similar/duplicate primitive that IS referenced?
├─ YES → Merge orphan into canonical, delete orphan
└─ NO
   └─ Does this primitive provide unique value?
      ├─ YES → Reference from appropriate playbook/adapter
      └─ NO
         └─ Is this a v1 fragment with no v2 equivalent?
            ├─ YES → Delete
            └─ UNSURE → Escalate for human review
```

---

## Success Criteria

- [ ] All 6 new expertises referenced from adapters (or merged)
- [ ] All 8 duplicate procedures merged and orphans deleted
- [ ] All high-value procedures (9) referenced from playbooks
- [ ] Safety snippets (5) widely referenced
- [ ] Accessibility snippets (4) referenced from relevant playbooks
- [ ] Technical/process snippets (13) appropriately referenced or deleted
- [ ] Validator reports ≤10 orphan warnings (down from 53)
- [ ] All changes documented in MIGRATION_TRACKING.csv

---

## Risk Mitigation

### Risks

1. **Over-deletion:** Removing content that should be preserved
2. **Over-merging:** Losing semantic distinctions between similar procedures
3. **Breaking references:** Deleting files that have hidden/indirect references

### Mitigations

1. **Git safety:** All work on feature branches, easy rollback
2. **Validation gates:** Run validator after every change batch
3. **Review merges:** Carefully review before deleting "duplicate" content
4. **Document decisions:** Clear commit messages explaining rationale

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Execute Priority 1** (expertise references) - highest confidence
3. **Execute Priority 2** (merges) - requires file review
4. **Validate** after each priority tier
5. **Iterate** based on results
6. **Document** all decisions in MIGRATION_TRACKING.csv

---

## Appendix: Quick Reference

### Files to Merge (8)
1. `leitmotif_use_policy` → `leitmotif_management`
2. `binder_integrity_enforcement` → `integrity_enforcement`
3. `voice_register_coherence` → `voice_coherence`
4. `art_caption_alt_guidance` → `alt_text_creation`
5. `visual_language_motif` → `visual_language_maintenance`
6. `register_map_idiom_strategy` → `register_map_maintenance`
7. `audio_determinism_logging` → `determinism_logging` (verify first)
8. `gatekeeper_presentation` → `gatekeeper_quality_bars` (expertise)

### Files to Reference (15+ procedures, 22 snippets)
See Priority 3-5 sections above for specific playbook/adapter targets.

### Files to Evaluate (8 procedures)
See Priority 4 section - requires case-by-case review.
