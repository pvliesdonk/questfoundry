# Spec-Compliance Audit Report

**Date:** 2026-04-18 (audit begins)
**Spec:** `docs/superpowers/specs/2026-04-18-spec-compliance-audit-design.md`
**Plan:** `docs/superpowers/plans/2026-04-19-spec-compliance-audit.md`

This report records every gap found between the authoritative design
specs and current code + tests. Per CLAUDE.md §Design Doc Authority,
specs supersede code and tests; findings below document where code/tests
diverge from the spec and need to be brought into compliance.

Each section corresponds to one milestone. Cluster issues filed in GitHub
reference the section anchors in this document.

---

## Status

| Milestone | Status | Clusters | Epic | Filed |
|---|---|---|---|---|
| M-DREAM-spec | issues-filed | 3 | #1268 | 3 |
| M-BRAINSTORM-spec | issues-filed | 8 | #1272 | 8 |
| M-SEED-spec | issues-filed | 14 | #1281 | 14 |
| M-GROW-spec | issues-filed | 13 | #1296 | 13 |
| M-POLISH-spec | issues-filed | 8 | #1310 | 8 |
| M-FILL-spec | issues-filed | 5 | #1319 | 5 |
| M-DRESS-spec | issues-filed | 5 | #1325 | 5 |
| M-SHIP-spec | issues-filed | 7 | #1331 | 7 |
| M-logging-compliance | issues-filed | 3 | #1339 | 3 |
| M-silent-degradation | issues-filed | 3 | #1343 | 2 |
| M-contract-chaining | issues-filed | 2 | #1346 | 2 |

---

## Runtime-verification deferred list

Rules that can only be verified against a live LLM run (prompt quality,
diegetic voice, etc.). These accumulate here and are NOT filed as
compliance issues — they are a follow-on track.

(none yet)

---

## M-DREAM-spec

### Summary
- Rules checked: 13
- Compliant: 4 | Drift: 2 | Missing: 2 | Uncheckable: 5

### Cluster: POV Style Enum Values Don't Match Spec

**Rules covered:** R-1.9

**Current state:** The Pydantic model defines `pov_style` with values `"first"`, `"second"`, `"third_limited"`, `"third_omniscient"`, stripped of the `_person` suffix.

**Gap:** The authoritative spec (dream.md, line 75) requires the values to be `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient` — with the `_person` suffix included. The implementation omits this suffix, causing a mismatch between the spec and code.

**Recommended fix:** Update the Pydantic model's `pov_style` Literal to include the `_person` suffix on all four values to match the authoritative spec exactly.

**Code refs:** `src/questfoundry/models/dream.py:63`, `src/questfoundry/prompts/templates/dream.yaml:25-27`

**Test refs:** `tests/unit/test_dream_stage.py` (no tests validate pov_style values against spec)

### Cluster: Scope Preset Names Diverge from Spec Examples

**Rules covered:** R-1.4

**Current state:** The Scope model defines `story_size` as one of `["vignette", "short", "standard", "long"]`.

**Gap:** The authoritative spec (dream.md, line 51) states scope presets should be "e.g., `micro`, `short`, `medium`, `long`, or their equivalent implementation values." The spec examples include `micro` and `medium`, but the code uses `vignette` and `standard` instead. While the spec does say "or their equivalent implementation values," using entirely different names (`vignette` vs. `micro`, `standard` vs. `medium`) creates ambiguity about whether these are truly equivalent or a silent divergence.

**Recommended fix:** Align terminology to match the spec exactly. Rename the enum values to `micro`, `short`, `medium`, `long`. Update `src/questfoundry/models/dream.py:34`, `src/questfoundry/prompts/templates/dream.yaml:16`, and any downstream consumers that match against the old names (BRAINSTORM/SEED preset lookups, CLI help text, tests).

**Code refs:** `src/questfoundry/models/dream.py:34`, `src/questfoundry/prompts/templates/dream.yaml:16`

**Test refs:** `tests/unit/test_dream_stage.py` (tests use `standard` directly), `tests/integration/test_dream_pipeline.py:59` (tests validate artifact structure but not scope values against spec)

### Cluster: Vision approval not explicitly recorded; rejection loop missing

**Rules covered:** R-1.12, R-1.13

**Current state:** The DREAM stage runs discuss → summarize → serialize and writes the Vision via `apply_dream_mutations()`. When `qf dream` is run with `--no-interactive`, the flag implies explicit user pre-approval of all gates, so the stage completing without an interactive prompt is legitimate. In interactive mode a prompt is shown. However, (a) the approval mode is not recorded on the Vision node, and (b) no rejection loop exists in the interactive mode — rejection currently re-runs the whole stage rather than re-entering the operation with the misalignment.

**Gap:** R-1.12 is satisfied by the `--no-interactive` pre-approval pattern plus interactive-mode prompting; the gap is traceability — the Vision node looks identical regardless of approval mode. R-1.13 is not satisfied: rejection does not loop back to a specific operation (spark exploration / constraint definition / synthesis); it only restarts the stage.

**Recommended fix:** (1) Stamp the Vision node at gate-pass time with `approved_at: <timestamp>` and `approval_mode: "interactive" | "no_interactive"`. (2) On interactive rejection, prompt the user to identify which operation to re-enter (spark exploration / constraint definition / synthesis) and re-run from that operation rather than restarting the stage. (3) Downstream stages (BRAINSTORM) can assert the Vision has the approval stamp before proceeding.

**Code refs:** `src/questfoundry/pipeline/stages/dream.py:170-191`, `src/questfoundry/graph/mutations.py:561-591`, `src/questfoundry/cli.py:784-869`

**Test refs:** `tests/unit/test_dream_stage.py`, `tests/integration/test_dream_pipeline.py`

### Uncheckable rules

- R-1.1: Discussion produces a single coherent vision — requires live LLM dialogue to verify convergence (is one vision picked, or does the LLM offer a menu?)
- R-1.3: Themes are abstract ideas, not plot points — requires semantic theme validation beyond schema constraints
- R-1.5: Content notes shape creative direction — BRAINSTORM enforcement is external to DREAM; DREAM stores notes correctly but validation of downstream respect happens in BRAINSTORM stage
- R-1.6: Constraints are firm — enforcement depends on BRAINSTORM respecting content notes; DREAM has no mechanism to validate this
- R-1.11: Synthesis captures only what was discussed — requires monitoring LLM behavior against discussion context

---

## M-BRAINSTORM-spec

### Summary
- Rules checked: 19
- Compliant: 8 | Drift: 7 | Missing: 3 | Uncheckable: 1

### Cluster: Entity name field and minimum requirements

**Rules covered:** R-2.1

**Current state:** Entity model allows `name` to be `None` (optional field with `default=None`). The spec requires entities to have "non-empty `name`, `category`, and `concept`" as part of the Stage Output Contract.

**Gap:** The `name` field is semantically optional in the data model, but the spec designates it as required. While the prompt guidance notes "Names will be generated later" if missing, the spec's "non-empty name" language is stricter than the implementation enforces.

**Recommended fix:** Add `min_length=1` to the `name` field in Entity model to enforce non-null names at serialization time, aligning the data contract with the spec's structural requirement.

**Code refs:** `src/questfoundry/models/brainstorm.py:44-50` (Entity.name field)

**Test refs:** `tests/unit/test_brainstorm_stage.py:421-452` (test_brainstorm_output_model_validates)

### Cluster: Location entity minimum count not enforced

**Rules covered:** R-2.4

**Current state:** The spec requires "at least two distinct `location`-category entities exist." No validation code enforces this constraint. The prompt mentions "aim for `{size_locations}` distinct settings" but does not validate the output minimum.

**Gap:** `validate_brainstorm_mutations()` performs internal consistency checks (duplicates, ID references, canonical answer count) but does NOT check that the entity list contains at least 2 locations. A brainstorm run could produce 0 location entities without error.

**Recommended fix:** Add a check in `validate_brainstorm_mutations()` to count location-category entities and append a `BrainstormValidationError` if the count is less than 2. Return that error to the serialize-to-artifact repair loop so the LLM can correct it.

**Code refs:** `src/questfoundry/graph/mutations.py:594-713` (validate_brainstorm_mutations function, missing location count check)

**Test refs:** `tests/unit/test_brainstorm_stage.py` (no test for location minimum)

### Cluster: Question punctuation validation missing

**Rules covered:** R-3.1

**Current state:** The Dilemma model validates that `question` has `min_length=1`, but does NOT validate that it ends with a question mark `?`. The spec explicitly states the question "ends with `?`".

**Gap:** A dilemma like `"Will the hero succeed"` (no `?`) would serialize without error, violating the spec's narrative requirement that the question format ends with punctuation.

**Recommended fix:** Add a field validator to `Dilemma.question` that checks `question.rstrip().endswith("?")` and raises a clear error with example format if missing.

**Code refs:** `src/questfoundry/models/brainstorm.py:94-162` (Dilemma class, missing question punctuation validator)

**Test refs:** `tests/unit/test_brainstorm_stage.py` (no test for question ending with `?`)

### Cluster: Dilemma-to-entity anchoring not validated as structural requirement

**Rules covered:** R-3.6

**Current state:** `validate_brainstorm_mutations()` does NOT check that every dilemma has at least one `anchored_to` reference to an entity. When `apply_brainstorm_mutations()` encounters an entity ID in `central_entity_ids` that cannot be resolved in the graph, it logs a WARNING and skips creating the `anchored_to` edge. A dilemma can end up with zero anchored_to edges without error.

**Gap:** The spec (R-3.6 and Stage Output Contract) requires "Each Dilemma has at least one `anchored_to` edge to an Entity." Silently creating dilemmas without any anchors violates this hard structural requirement. Per CLAUDE.md §Anti-Patterns, "Silent degradation of story structure constraints" is forbidden.

**Recommended fix:** Add validation in `validate_brainstorm_mutations()` to check that each dilemma's `central_entity_ids` list is non-empty. After `apply_brainstorm_mutations()` creates dilemma nodes, add a post-mutation check that every dilemma node has at least one outgoing `anchored_to` edge; raise `MutationError` if not. This ensures the invariant is enforced, not degraded.

**Code refs:** `src/questfoundry/graph/mutations.py:666-677` (validation of central_entity_ids references), `src/questfoundry/graph/mutations.py:899-910` (silent skip of unresolvable entities)

**Test refs:** `tests/unit/test_brainstorm_stage.py` (no test for anchored_to requirement)

### Cluster: Dilemma ID format validation incomplete

**Rules covered:** R-3.7

**Current state:** The Dilemma model validates that `dilemma_id` has `min_length=1` and rejects IDs ending with `_or_` (common LLM error). However, it does NOT validate the required `dilemma::` prefix. The prefix is applied by `apply_brainstorm_mutations()` via `_prefix_id()`, but validation accepts unprefixed IDs.

**Gap:** A dilemma with `dilemma_id: "mentor_trust"` (no prefix) would pass Pydantic validation. The spec and Stage Output Contract require "Dilemma IDs use the `dilemma::` prefix." This allows incorrect IDs to pass through phases before failing during serialization or graph operations.

**Recommended fix:** Add a field validator to `Dilemma.dilemma_id` that either (a) accepts unprefixed IDs and auto-prefixes them in a `@model_validator(mode="before")` hook (similar to the existing `migrate_alternatives_field`), or (b) requires the `dilemma::` prefix explicitly and rejects unprefixed IDs with a helpful error.

**Code refs:** `src/questfoundry/models/brainstorm.py:109-130` (Dilemma.dilemma_id field, missing prefix validation)

**Test refs:** `tests/unit/test_brainstorm_stage.py:490-527` (test_dilemma_rejects_trailing_or_in_id tests _or_ but not prefix)

### Cluster: No validation that output contains only brainstorm-allowed node types

**Rules covered:** R-3.8

**Current state:** Neither `validate_brainstorm_mutations()` nor the BrainstormOutput schema checks that the artifact contains ONLY entities, dilemmas, and answers. If an LLM accidentally included Path, Beat, Consequence, or other node types in the output, they would be silently ignored or cause schema validation errors rather than being caught as spec violations.

**Gap:** The spec explicitly forbids "Path, Beat, Consequence, State Flag, Passage, or Intersection Group nodes" after BRAINSTORM. While most of these are unlikely from the LLM, a malformed output could include unexpected fields. There is no defensive check that the output schema matches the expected types.

**Recommended fix:** Add a comment or docstring to `BrainstormOutput` documenting that ONLY `entities` and `dilemmas` are permitted. Add an optional test that instantiates BrainstormOutput with an unexpected field (e.g., `paths=[]`) and verifies it is rejected or ignored as expected.

**Code refs:** `src/questfoundry/models/brainstorm.py:170-192` (BrainstormOutput class)

**Test refs:** `tests/unit/test_brainstorm_stage.py` (no test for forbidden node types)

### Cluster: No validation that discussion abundance target is met

**Rules covered:** R-1.1

**Current state:** The spec targets 15–25 entities and 4–8 dilemmas for a short story. The `execute()` method logs entity and dilemma counts at completion but performs no validation that the minimums are met. The prompt includes guidance `{size_entities}` and `{size_dilemmas}` placeholders but does not enforce them in code.

**Gap:** A brainstorm run with 2 entities and 1 dilemma would complete without error or warning (beyond the logged counts). The spec's guidance on abundance is advisory to the LLM, not enforced by the stage. Users have no automated signal that the material is insufficient for downstream stages.

**Recommended fix:** Add optional validation in `serialize_to_artifact()` or the serialize loop that warns (or errors, depending on scope choice) if entity count is below 10 or dilemma count is below 3. Alternatively, document in the stage docstring that abundance validation is the human's responsibility at the Phase 2 and Phase 3 review gates.

**Code refs:** `src/questfoundry/pipeline/stages/brainstorm.py:328-332` (logs counts but no validation)

**Test refs:** `tests/unit/test_brainstorm_stage.py:421-452` (test with 1 entity, 1 dilemma; no test for minimum counts)

### Cluster: Prompt does not enforce Vision compatibility at serialization time

**Rules covered:** R-1.3

**Current state:** The discuss phase prompt includes the vision context and guidance to reject contradictions ("A proposal that contradicts the Vision must be flagged and rejected"). However, the serialize prompt does not re-enforce vision compatibility, and there is no validation step that checks the final entities and dilemmas against the vision for contradictions.

**Gap:** An LLM could generate a "slapstick comic-relief character in a gritty noir" during discussion, which the human might overlook, and it would serialize into BRAINSTORM output without catching the tone mismatch. The spec requires contradictions to be "flagged and rejected," but no stage code enforces this.

**Recommended fix:** Add vision compatibility context to `get_brainstorm_serialize_prompt()` so the LLM performs a final sanity check that all entities and dilemmas fit the genre, tone, and themes. Alternatively, document that vision validation is a human responsibility at the Phase 2 and Phase 3 gates, and add a prompt reminder: "Does every entity and dilemma fit the vision? If not, flag it for revision."

**Code refs:** `src/questfoundry/pipeline/stages/brainstorm.py:307-310` (serialize prompt call without vision re-injection), `prompts/templates/serialize_brainstorm.yaml` (serialize prompt)

**Test refs:** `tests/unit/test_brainstorm_stage.py` (no test for vision-contradiction detection)

### Uncheckable rules

- R-3.3: Both answers are genuinely different and both compelling — requires LLM-level narrative judgment of drama and contrast quality, not programmatically verifiable. Human review at Phase 3 gate determines this.

---

## M-SEED-spec

### Summary
- Rules checked: 66
- Compliant: 45 | Drift: 13 | Missing: 7 | Uncheckable: 1

### Cluster: Missing Y-shape shared beat enforcement

**Rules covered:** R-3.6, R-3.10

**Current state:** The Y-shape structure is described in code comments and YAML migration logic (`_migrate_paths_to_path_id`, `also_belongs_to` field), but SEED prompts, context builders, and beat-generation serialization do not explicitly enforce or verify that dilemmas produce pre-commit beats with `also_belongs_to` set.

**Gap:** R-3.10 requires "every Dilemma with two explored Answers has ≥1 pre-commit beat," but there is no mechanism validating this in `validate_seed_mutations()`. The function checks beat count per path and commit/post-commit beat properties, but does not assert that each dilemma has at least one beat with `also_belongs_to` set to both explored paths of that dilemma.

**Recommended fix:** Add a validation rule in `validate_seed_mutations()` that iterates each dilemma with two explored answers and verifies at least one beat with `also_belongs_to` is assigned to both of its explored paths. If missing, report a COMPLETENESS error with guidance to regenerate beats.

**Code refs:** `src/questfoundry/graph/mutations.py:1068-1500`, `src/questfoundry/models/seed.py:235-353`

**Test refs:** `tests/unit/test_seed_stage.py` (lacks explicit Y-shape validation test)

### Cluster: Cross-dilemma `belongs_to` prohibition not enforced

**Rules covered:** R-3.9

**Current state:** The YAML model (`InitialBeat._migrate_paths_to_path_id`) rejects `paths` with more than 2 entries, preventing >2 `belongs_to` edges. However, there is no validation that both `path_id` and `also_belongs_to` belong to the same dilemma.

**Gap:** R-3.9 forbids "beats must not have `belongs_to` edges referencing Paths from different Dilemmas." A beat could theoretically be created with `path_id: path::mentor_trust__protector` and `also_belongs_to: path::artifact_nature__salvation`. No semantic validator catches this violation.

**Recommended fix:** Add a post-model validator in `InitialBeat` (after `_also_belongs_to_differs_from_path_id`) or in `validate_seed_mutations()` that extracts the dilemma_id from both path IDs and asserts they match. If they differ, raise ValueError with a clear message citing R-3.9.

**Code refs:** `src/questfoundry/models/seed.py:322-327`, `src/questfoundry/graph/mutations.py:1328-1340`

**Test refs:** `tests/unit/test_seed_models.py` (no cross-dilemma Y-shape test)

### Cluster: Setup and epilogue beat semantics not validated

**Rules covered:** R-3.14, R-3.15

**Current state:** `InitialBeat` model does not enforce that setup/epilogue beats are structural (zero `belongs_to`, zero `dilemma_impacts`). The model allows any beat to declare itself as setup/epilogue via comment or metadata, but no field flags them and no validator enforces the zero-`belongs_to` constraint.

**Gap:** R-3.14 requires setup/epilogue beats to be "structural beats with zero `belongs_to` edges and zero `dilemma_impacts`." Currently, a beat with `path_id: path_x` could still be labeled as setup in prose/comments without validation catching the violation.

**Recommended fix:** Add an optional `beat_type` field to `InitialBeat` (enum: `"narrative" | "setup" | "epilogue"`; default "narrative"). Add post-model validator asserting that setup/epilogue beats have null `path_id`, null `also_belongs_to`, and empty `dilemma_impacts`.

**Code refs:** `src/questfoundry/models/seed.py:235-353`

**Test refs:** `tests/unit/test_seed_models.py`, `tests/unit/test_seed_stage.py`

### Cluster: `explored` field immutability not enforced or tested

**Rules covered:** R-2.3, R-5.2

**Current state:** Phase 5 calls `prune_to_arc_limit()` from `src/questfoundry/graph/seed_pruning.py`. The pruning logic drops Path nodes from the artifact but must preserve the `explored` field on DilemmaDecision objects unchanged. No assertion or test verifies this invariant: `pruned_artifact.dilemmas[i].explored` should remain identical to the pre-pruning value for each dilemma. Existing Phase 5 tests check final arc count but do not verify `explored` immutability before/after pruning.

**Gap:** If the pruning code ever modifies `explored` (e.g., by dropping demoted answers), this invariant is silently broken. Both the runtime check and the test coverage are missing.

**Recommended fix:** In `seed.py:execute()` after `prune_to_arc_limit()` returns, add an assertion loop comparing original and pruned artifact dilemma `explored` fields. Log an ERROR and raise `SeedStageError` on mismatch. Add a test case that creates an artifact with 5 dilemmas (3 of which are demoted by pruning), serializes it, and verifies that the pruned artifact's `dilemmas[i].explored` matches the original for all dilemmas.

**Code refs:** `src/questfoundry/pipeline/stages/seed.py:467-476`, `src/questfoundry/graph/seed_pruning.py`

**Test refs:** `tests/unit/test_seed_stage.py:838-879` (tests final arc count, not explored immutability)

### Cluster: Convergence analysis LLM failure handling missing

**Rules covered:** R-7.5

**Current state:** Phase 4 (Convergence Analysis) calls `serialize_convergence_analysis()`. If the LLM call fails or returns invalid role/weight values, the result is returned as-is. There is no explicit WARNING log if defaults are applied.

**Gap:** R-7.5 requires "On LLM failure, defaults are applied... but the failure is logged at WARNING level with the Dilemma IDs affected. Silent default application is forbidden." The code does not handle LLM failure in the convergence analysis phase or log fallback application.

**Recommended fix:** Modify `serialize_convergence_analysis()` and/or its call site to catch and handle LLM failures (malformed response, validation failure). If defaults are applied for any dilemma, log at WARNING with affected dilemma IDs and the reason for fallback.

**Code refs:** `src/questfoundry/pipeline/stages/seed.py:451-461`, `src/questfoundry/agents/serialize.py`

**Test refs:** `tests/unit/test_seed_stage.py` (no convergence analysis failure test)

### Cluster: Dilemma ordering LLM failure silent application

**Rules covered:** R-8.5

**Current state:** Phase 8 (Dilemma Ordering Relationships) calls `serialize_dilemma_relationships()`. If the LLM fails, the code silently returns empty relationships (or a partial list) without logging a WARNING.

**Gap:** R-8.5 states "If the LLM call fails, no relationships are declared — the graph is left with zero ordering edges. Failure logged at WARNING." The code does not log at WARNING on LLM failure during relationship serialization.

**Recommended fix:** Modify `serialize_dilemma_relationships()` to wrap the LLM call in a try/except block. On failure, log at WARNING with the error details and affected dilemma count. Return an empty list rather than propagating the exception.

**Code refs:** `src/questfoundry/pipeline/stages/seed.py:502-514`, `src/questfoundry/agents/serialize.py`

**Test refs:** None (no test for dilemma relationships LLM failure)

### Cluster: Dilemma role "flavor" deprecation incomplete

**Rules covered:** R-7.1

**Current state:** `DilemmaAnalysis` model has a `_migrate_legacy_fields` validator that converts `dilemma_role='flavor'` to `'soft'` with a DeprecationWarning. However, the role field's docstring and enum values do not explicitly forbid `'flavor'`.

**Gap:** R-7.1 requires `dilemma_role ∈ {hard, soft}` exactly. The migration silently upgrades `'flavor'` to `'soft'`, hiding the real intent. Future code reading stored artifacts may not know that a soft dilemma with `residue_weight='cosmetic'` was originally `'flavor'`.

**Recommended fix:** Update the `_migrate_legacy_fields` logic to log at WARNING when `'flavor'` is encountered, including the dilemma_id and note that `'soft' + 'cosmetic'` is the canonical form. Ensure tests verify the migration and logging.

**Code refs:** `src/questfoundry/models/seed.py:376-403`

**Test refs:** `tests/unit/test_seed_models.py` (no flavor migration test found)

### Cluster: Concurrent dilemma ordering normalization not enforced post-serialization

**Rules covered:** R-8.3

**Current state:** `DilemmaRelationship._validate_and_normalize_pair()` normalizes `concurrent` pairs to canonical order (dilemma_a < dilemma_b alphabetically). The test suite does not verify this normalization or reject non-normalized pairs created directly.

**Gap:** If code bypasses the Pydantic validator (e.g., direct dict insertion into graph), non-normalized concurrent pairs could exist. The code provides no post-serialization check that all concurrent edges in the artifact are normalized.

**Recommended fix:** Add a validator in Phase 8 (after `serialize_dilemma_relationships()` returns) that iterates all concurrent relationships and asserts lex ordering. If violation found, log at ERROR and raise exception before returning.

**Code refs:** `src/questfoundry/models/seed.py:490-503`, `src/questfoundry/pipeline/stages/seed.py:502-514`

**Test refs:** `tests/unit/test_seed_models.py` (DilemmaRelationship normalization test needed)

### Cluster: Shared entity derivation vs. declaration guard absent

**Rules covered:** R-8.4

**Current state:** The `DilemmaRelationship` model does not have a `shared_entity` field. Relationships are created only as `wraps`, `concurrent`, or `serial`. The code structure implies `shared_entity` should be derived from graph `anchored_to` edges in downstream stages.

**Gap:** R-8.4 states "`shared_entity` is NOT a declared relationship — it is derived from `anchored_to` edges. Do not create `shared_entity` edges." No validation prevents a human or LLM from incorrectly declaring a `shared_entity` edge type in SEED output if such a field were added to the model later.

**Recommended fix:** Add a note/docstring to the stage specifying that `shared_entity` relationships are computed downstream and should never be hand-declared. If a model field or edge type for `shared_entity` is ever added, include a validator rejecting it with a clear error message.

**Code refs:** `src/questfoundry/models/seed.py:453-509`

**Test refs:** None (design constraint, not runtime check)

### Cluster: Arc count threshold guardrail weakened by optional Phase 7/8

**Rules covered:** R-5.1

**Current state:** Phase 5 prunes to arc limit (`max_arcs=size_profile.max_arcs`). After pruning, the artifact is checked for minimum arcs. However, phases 7 and 8 run after this check and could theoretically further reduce arc count if they drop dilemmas.

**Gap:** R-5.1 requires "Arc count ≤ 16" and implies the pruning phase is the enforcement point. If Phase 7 (Convergence Analysis) or Phase 8 (Dilemma Ordering) later decide to drop a dilemma (due to LLM classification), the final artifact could violate the guardrail. The code does not re-validate arc count after these phases.

**Recommended fix:** Add a post-Phase-8 arc count check (before returning from `execute()`) that logs at WARNING if final arc count has drifted below minimum, and ERROR if above max. Document that phases 7/8 must not reduce arc count.

**Code refs:** `src/questfoundry/pipeline/stages/seed.py:487-514`

**Test refs:** `tests/unit/test_seed_stage.py:838-879`

### Cluster: Beat entities field validation missing for narrative beats

**Rules covered:** R-3.13

**Current state:** `InitialBeat` model requires `summary` and `entities` fields with `min_length=1` for `summary`. However, validation does NOT check that `entities` is non-empty for narrative beats (it may be empty for structural beats, but narrative beats must reference entities).

**Gap:** R-3.13 states "Every beat has non-empty `summary` and `entities`." The model allows `entities: []` by default. For narrative beats (not setup/epilogue), this violates the rule. The model lacks a conditional validator enforcing `entities` ≥ 1 for non-structural beats.

**Recommended fix:** Extend `InitialBeat` with a post-model validator that checks: if `beat_type` (see the Setup/epilogue cluster) is `"narrative"`, then `len(entities) >= 1`. If not set, assume narrative and require entities.

**Code refs:** `src/questfoundry/models/seed.py:235-353`

**Test refs:** `tests/unit/test_seed_stage.py:958-1029` (tests shared beat structure but not entity list completeness)

### Cluster: `path_importance` field not defined in spec

**Rules covered:** R-3.1 (tangential) — spec-vs-code mismatch

**Current state:** The `Path` model includes a `path_importance: PathTier = Literal["major", "minor"]` field. This field is persisted and indexed for LLM context but does not appear in the authoritative SEED procedure spec.

**Gap:** Per CLAUDE.md §Design Doc Authority, specs supersede code — fields that exist in code but not in spec are code drift. The spec does not define `path_importance` or how it affects branching, pruning, or prose generation. This is either (a) a field to be removed from code, or (b) a field the spec should formally add (which is a separate spec-update track — not part of this audit's fix).

**Recommended fix:** Raise the `path_importance` field with maintainers. If kept: update the spec to define it as a Hint (similar to `temporal_hint`) and describe its downstream effect. If dropped: remove from the `Path` model and any downstream consumers. Default stance per CLAUDE.md: the code matches the spec — if kept, spec must be updated first.

**Code refs:** `src/questfoundry/models/seed.py:177-179`

**Test refs:** `tests/unit/test_seed_stage.py:499-512`

### Cluster: Consequence ripples validation absent

**Rules covered:** R-3.4

**Current state:** The `Consequence` model has a `narrative_effects: list[str]` field described as "Story effects this consequence implies (cascading impacts)." There is no model validation that this list is non-empty.

**Gap:** R-3.4 requires "Every Consequence has a non-empty `description` and at least one ripple." The model allows `narrative_effects: []` by default. No validator enforces `len(narrative_effects) >= 1` for any Consequence.

**Recommended fix:** Add a post-model validator in `Consequence` that asserts `len(narrative_effects) >= 1`. If empty, raise ValueError with guidance to describe at least one story effect.

**Code refs:** `src/questfoundry/models/seed.py:122-143`

**Test refs:** `tests/unit/test_seed_models.py` (no Consequence ripple validation test)

### Cluster: Path Freeze approval not explicitly recorded

**Rules covered:** R-6.4

**Current state:** When `qf seed` is run with `--no-interactive`, the flag implies explicit user pre-approval of all gates including Path Freeze. Interactive mode uses an interactive gate. The gate mechanism itself is compliant with R-6.4's "approval is required." What is missing is the "explicitly recorded" half — the graph state after Phase 6 does not carry a marker indicating the approval mode or timestamp.

**Gap:** R-6.4 requires approval to be "explicitly recorded." The gate pass is enforced but not recorded. GROW cannot check whether Path Freeze was user-approved (interactive) vs auto-approved (`--no-interactive`), and has no recorded timestamp to verify recency.

**Recommended fix:** Stamp a graph-level marker at Path Freeze gate-pass time (e.g., a `path_freeze_approved_at: <timestamp>` and `path_freeze_approval_mode: "interactive" | "no_interactive"` pair — either on a dedicated PathFreeze node or as metadata on an existing stage-boundary artifact). GROW's Stage Input Contract validation should check the marker exists before proceeding.

**Code refs:** `src/questfoundry/pipeline/stages/seed.py` (no approval stamping)

**Test refs:** None (approval state not currently representable)

### Uncheckable rules

- R-3.5: Consequences describe world state vs. player actions — requires semantic understanding of prose. Cannot be programmatically checked at the model level. Enforced via SEED prompt engineering; any runtime check would require an LLM-based critique call. Deferred to runtime-verification follow-on track.

---

## M-GROW-spec

### Summary
- Rules checked: 51
- Compliant: 33 | Drift: 11 | Missing: 5 | Uncheckable: 2

### Cluster: All-intersections-rejected not raised to ERROR (silent degradation)

**Rules covered:** R-2.3, R-2.8

**Current state:** Phase 3 (`_phase_3_intersections`) attempts to validate and apply intersections but logs rejections at WARNING/DEBUG and returns `status="failed"` only when all proposals are rejected after structural retry exhaustion. No explicit ERROR log occurs when the all-rejected outcome is reached.

**Gap:** Per CLAUDE.md §Anti-Patterns (Silent degradation), an all-intersections-rejected outcome is a pipeline failure, not a warning. R-2.8 requires rejection logs at INFO, but the aggregate failure mode (zero groups formed when signals suggested some should) must be ERROR and halt.

**Recommended fix:** When all intersection proposals are rejected, log at ERROR with the rejection reasons aggregated and halt the pipeline. Per-candidate rejection logging stays at INFO (R-2.8). Differentiate "some accepted, some rejected" (normal, INFO) from "all rejected" (ERROR, halt).

**Code refs:** `src/questfoundry/pipeline/stages/grow/llm_phases.py:299-312`, `src/questfoundry/pipeline/stages/grow/llm_phases.py:329-336`

**Test refs:** `tests/unit/test_grow_gates.py`, `tests/unit/test_grow_validators.py`

### Cluster: Temporal hint acyclicity invariant violation not logged at ERROR

**Rules covered:** R-3.2, R-3.3, R-3.6, R-3.7

**Current state:** Phase 3 `_phase_resolve_temporal_hints` applies mandatory drops, requests LLM resolution for swap pairs, and calls `verify_hints_acyclic()` as a postcondition. The function raises `TemporalHintResolutionInvariantError` if surviving hints cycle, caught by the stage executor as `status="failed"`.

**Gap:** The error is propagated as a phase failure but not logged at ERROR level before the exception is raised. R-3.7 requires a hard invariant violation to halt with an error and be visible in logs — currently the log entry (if any) is at the phase-result level, not as an explicit ERROR for the invariant.

**Recommended fix:** Log at ERROR level before raising `TemporalHintResolutionInvariantError` with full context (surviving beat IDs, cycle path). Ensure the exception propagates to the stage executor without being swallowed.

**Code refs:** `src/questfoundry/pipeline/stages/grow/llm_phases.py:553-558`, `src/questfoundry/graph/errors.py`

**Test refs:** `tests/unit/test_grow_deterministic.py` (temporal hint tests)

### Cluster: State flag derivation edge validation missing

**Rules covered:** R-6.1, R-6.4

**Current state:** Phase 8b `phase_state_flags` creates one state flag per consequence and adds a `derived_from` edge to that consequence. The edge type is hardcoded without validation that the consequence actually exists.

**Gap:** R-6.1 mandates every State Flag has a `derived_from` edge to exactly one Consequence, no ad-hoc creation. If a consequence is missing or the edge creation silently no-ops, no validation catches this.

**Recommended fix:** Add a post-creation validation check in `phase_state_flags` that iterates every created state flag and asserts a `derived_from` edge exists pointing to an existing consequence node. Log at ERROR if any flag has no target.

**Code refs:** `src/questfoundry/pipeline/stages/grow/deterministic.py:594-603`

**Test refs:** `tests/unit/test_grow_models.py`, `tests/unit/test_grow_validators.py`

### Cluster: Transition beat zero-overlap seam detection missing

**Rules covered:** R-5.1, R-5.2, R-5.3

**Current state:** Phase 5 `_phase_transition_gaps` is scheduled in the registry but the implementation does not show explicit zero-overlap seam detection before inserting transition beats. The code calls `_validate_and_insert_gaps()` without the deterministic seam-detection algorithm required by R-5.2.

**Gap:** R-5.2 specifies transition beats are inserted ONLY at cross-dilemma seams with zero entity/location overlap. Without this check, transition beats may be inserted at seams with partial overlap, violating the spec.

**Recommended fix:** Before the LLM phase for transition drafting, implement a deterministic seam-detection algorithm that checks every cross-dilemma `predecessor` edge for entity/location overlap. Only flag seams with zero overlap as candidates for LLM transition drafting.

**Code refs:** `src/questfoundry/pipeline/stages/grow/llm_phases.py` (`_phase_transition_gaps`, offset ~1250-1350)

**Test refs:** `tests/unit/test_grow_stage.py` (transition beat tests if present)

### Cluster: Soft dilemma without structural convergence — no halt on classification error

**Rules covered:** R-7.1, R-7.3, R-7.4

**Current state:** Phase 7 `phase_convergence` computes convergence beats for soft dilemmas using `find_dag_convergence_beat()` and persists `converges_at` and `convergence_payoff`. If no convergence is found, the code continues without halting. Hard dilemmas are left with null fields.

**Gap:** R-7.4 states: "If a soft Dilemma has no structural convergence beat, this is a classification error — the Dilemma should be hard. Halt with error identifying the Dilemma." Current code logs at WARNING but does not halt. A mis-classified dilemma proceeds through the pipeline.

**Recommended fix:** After computing convergence for each soft dilemma, check if `converges_at` is None. If so, log at ERROR and halt with a classification-error message identifying the dilemma by ID.

**Code refs:** `src/questfoundry/pipeline/stages/grow/deterministic.py:480-507`

**Test refs:** `tests/unit/test_grow_deterministic.py` (convergence tests)

### Cluster: Materialized arc data prefix convention unenforced

**Rules covered:** R-8.2

**Current state:** Arcs are computed on-the-fly by `enumerate_arcs()` and returned as Arc dataclass instances. No arc nodes are stored in the graph. Comments correctly state arcs are computed, not stored.

**Gap:** R-8.2 states "If materialized for debugging, they must use the `materialized_` prefix." No code uses or enforces this prefix. If debugging code ever materializes arcs, the convention must be applied, but there is no enforcement mechanism.

**Recommended fix:** Add a note in code comments clarifying the `materialized_` prefix requirement for any future arc materialization. If arc nodes are ever created, enforce the prefix at creation time with a validation check that rejects non-prefixed arc nodes.

**Code refs:** `src/questfoundry/graph/grow_algorithms.py:603-748` (enumerate_arcs docstring)

**Test refs:** `tests/unit/test_grow_algorithms.py` (arc enumeration tests)

### Cluster: Dead passage/choice counting code from pre-split era

**Rules covered:** Stage Output Contract item 16 (tangential — dead code hygiene)

**Current state:** Code at `stage.py:359-362` counts passage and choice nodes in the graph at GROW completion. This is a leftover from before POLISH was spun out as a separate stage; those node types should never exist at the end of GROW now.

**Gap:** The counting code is dead (always zero in a correctly-ordered pipeline) but left in place. Per the CLAUDE.md anti-pattern about Dead Code / Backward-Compatibility shims, leftover references to removed scope should be deleted directly rather than lingering as "just in case" guards.

**Recommended fix:** Remove the passage/choice (and variant-passage, residue-beat, character-arc-metadata) counting from `stage.py`. Replace with an end-of-stage assertion that these node counts are zero; if non-zero, halt with a stage-integrity ERROR identifying which node type leaked (indicates a pipeline-ordering bug).

**Code refs:** `src/questfoundry/pipeline/stages/grow/stage.py:359-362, 376`

**Test refs:** `tests/unit/test_grow_stage.py` (output contract tests)

### Cluster: Logging-level misuse at validation failure sites

**Rules covered:** CLAUDE.md §Logging (cross-cutting — see M-logging-compliance milestone for full treatment)

**Current state:** Phase validation (`phase_validation`) logs fail results at ERROR (correct). Phase 3 logs intersection rejections at WARNING. Temporal hint resolution logs at INFO on success but WARNING on LLM failure.

**Gap:** Per CLAUDE.md §Logging litmus test, WARNING is for degraded states that do not stop execution; ERROR is for failures that stop execution. Some sites use WARNING where ERROR is correct (e.g., phase returns `status="failed"`). Some use WARNING for individual rejections that are retried (which is actually compliant per R-2.8 requiring INFO, so these may be over-leveled).

**Recommended fix:** Audit all phase failures and ensure they log at ERROR when the pipeline halts. Use INFO for per-candidate decisions that are normal (rejection + retry with another candidate). Use WARNING only for degraded states where the pipeline continues but attention is warranted.

**Code refs:** `src/questfoundry/pipeline/stages/grow/llm_phases.py:234-240, 299-312`

**Test refs:** `tests/unit/test_grow_stage.py` (logging checks)

### Cluster: Intra-path predecessor Y-fork postcondition unchecked

**Rules covered:** R-1.3, R-1.4, R-1.5, R-1.6

**Current state:** Phase 1a `phase_intra_path_predecessors` builds per-path beat chains, filtering to chainable beats (exclusive or intra-dilemma shared) and linking them via `predecessor` edges. Beats are sorted with intra-dilemma shared beats first, then exclusive beats.

**Gap:** R-1.4 specifies "The last shared pre-commit beat has one `predecessor` successor per explored path of its Dilemma." No postcondition verifies this: if a shared beat is missing a successor, the phase does not detect it. This breaks the Y-shape the pipeline requires.

**Recommended fix:** After building intra-path predecessor chains, add a validation check that the last intra-dilemma shared beat of each dilemma has exactly one successor per path of that dilemma. Log at ERROR if any shared pre-commit beat has fewer successors than paths.

**Code refs:** `src/questfoundry/pipeline/stages/grow/deterministic.py:198-248`

**Test refs:** `tests/unit/test_grow_deterministic.py` (intra-path predecessor tests)

### Cluster: No-Conditional-Prerequisites invariant not checked

**Rules covered:** R-2.7

**Current state:** Phase 2 (intersections) runs before Phase 1b (interleave). After intersections are applied, Phase 1b creates cross-path `predecessor` edges. The phase does NOT explicitly check R-2.7 (`paths(B) ⊇ paths(A_post_intersection)`) before or after applying intersections.

**Gap:** If R-2.7 is violated, arc enumeration will drop the edge silently during traversal, producing inconsistent orderings and `passage_dag_cycles` failures in POLISH. The code does not validate this invariant before accepting intersections.

**Recommended fix:** Implement a deterministic check in `check_intersection_compatibility()` (or as a pre-interleave step) verifying: for each intersection group and each beat A in the group, every beat B that is a direct successor of A has `paths(B) ⊇ paths(A_post_intersection)`. Log at DEBUG for all intersections checked; log at INFO (per R-2.8) and reject any intersection that fails this check.

**Code refs:** `src/questfoundry/graph/grow_algorithms.py:check_intersection_compatibility` (location not visible in audit — check around line 1300+)

**Test refs:** `tests/unit/test_grow_validators.py` (intersection compatibility tests)

### Cluster: State flag names not validated for world-state phrasing

**Rules covered:** R-6.2

**Current state:** Phase 8b creates state flag names using `f"{cons_raw}_committed"`. If a consequence is named for a player action, the resulting flag will be action-phrased.

**Gap:** R-6.2 requires "State flag names express world state, not player actions." A consequence named `player_chose_distrust` produces flag `player_chose_distrust_committed` — action-phrased. Spec examples use flags like `mentor_hostile_adversary` (world state).

**Recommended fix:** Add a validation check in `phase_state_flags` that verifies each created flag name does not contain action-verb patterns (e.g., "chose", "decided", "picked", "selected"). Log at WARNING for apparently action-phrased flags. Document the expected world-state naming convention for consequences in SEED's stage contract (this may drive a SEED-side fix too).

**Code refs:** `src/questfoundry/pipeline/stages/grow/deterministic.py:590-603`

**Test refs:** `tests/unit/test_grow_models.py` (state flag naming tests)

### Cluster: Entity overlay composition not validated

**Rules covered:** R-6.7, R-6.8

**Current state:** Phase 8c `_phase_8c_overlays` creates overlays on entity nodes, each with a `when` list of state flag IDs and a `details` dict. Multiple overlays can apply to the same entity when multiple flags are active.

**Gap:** R-6.7 states "Overlays may be composed — if multiple state flags affect the same entity, multiple overlays apply on arcs where their flags are all active." The code creates overlays but does not validate composition: no check for conflicting `details` keys across overlays, no check that hard-dilemma entities also receive overlays (R-6.8), no rejection of overlays with empty `when`.

**Recommended fix:** After creating overlays, validate: (a) no two overlays for the same entity have conflicting `details` keys (or define a merge strategy); (b) hard-dilemma entities have overlays created despite paths not converging (R-6.8); (c) overlays with empty `when` lists are rejected — every overlay must have activation conditions. Log WARNING for composition conflicts, ERROR for invalid overlay configs.

**Code refs:** `src/questfoundry/pipeline/stages/grow/llm_phases.py:_phase_8c_overlays` (offset 1320+)

**Test refs:** `tests/unit/test_grow_validators.py` (overlay composition tests)

### Cluster: Intersection candidate signals unverified as deterministic

**Rules covered:** R-2.1

**Current state:** Intersection candidate generation (`build_intersection_candidates`) uses signals "derivable from the graph": anchored_to overlap, flexibility-based entity substitutability, temporal proximity. The implementation is not fully visible in the audit.

**Gap:** R-2.1 requires these signals to be DETERMINISTIC — no LLM calls during candidate generation (LLM is only for clustering). If the function makes any LLM calls during generation, R-2.1 is violated.

**Recommended fix:** Verify `build_intersection_candidates()` uses only deterministic, graph-derivable signals. Document the algorithm for each signal type. Add unit tests that verify candidate generation is deterministic (same graph always produces same candidates).

**Code refs:** `src/questfoundry/graph/grow_algorithms.py:build_intersection_candidates` (location around line 1300+)

**Test refs:** `tests/unit/test_grow_algorithms.py` (intersection candidate tests)

### Uncheckable rules

- R-2.2: Clustering LLM receives full beat context (not bare IDs). Requires inspecting the actual prompt rendering at runtime; code review of the prompt template is partial evidence but not sufficient to verify end-to-end without a live LLM call.
- R-5.4: Transition-beat drafting LLM receives full context for both bridging beats. Same reasoning as R-2.2 — verifiable only at runtime with a live LLM.

---

## M-POLISH-spec

### Summary
- Rules checked: 72
- Compliant: 59 | Drift: 8 | Missing: 3 | Uncheckable: 2

*(Subagent originally included 9 additional "clusters" that it concluded were compliant — these are rolled back into the Compliant count in this summary.)*

### Cluster: POLISH consumes intersection groups (spec violation — GROW-internal only)

**Rules covered:** R-4a.4

**Current state:** Code explicitly reads intersection groups in Phase 4a and creates passages directly from them (`grouping_type="intersection"`, `deterministic.py:213-234`). Code comment says "beats from intersection groups", confirming the design.

**Gap:** Spec R-4a.4 states "POLISH does NOT consume Intersection Group nodes as a constraint on grouping." Current implementation creates passages directly from intersection group `beat_ids`, which violates this. The spec intends intersection groups to be GROW-internal only, with POLISH making fresh grouping assessment from the finalized DAG alone.

**Recommended fix:** Remove the intersection-group iteration in `compute_beat_grouping()`. Fold all beats into collapse or singleton groupings based only on DAG topology and path membership. Intersection groups should not influence passage grouping during POLISH.

**Code refs:** `src/questfoundry/pipeline/stages/polish/deterministic.py:213-234`

**Test refs:** `tests/unit/test_polish*.py` (existing tests may validate intersection grouping as expected — need updating to verify grouping is DAG-driven instead)

### Cluster: Zero-choice condition does not halt with ERROR

**Rules covered:** R-4c.2

**Current state:** Phase 4c computes choice edges from DAG divergences but contains no explicit check for zero-choice condition. If the DAG produces zero choices, this silently passes to Phase 6 without error.

**Gap:** Per spec R-4c.2, zero choice edges indicate a SEED/GROW bug and MUST halt with ERROR identifying the broken upstream stage. The current code logs choices count but does not validate that it is non-zero. This violates the Silent Degradation policy.

**Recommended fix:** Add validation after `plan.choice_specs = compute_choice_edges(...)` that checks `if not plan.choice_specs` and returns a failed PhaseResult with ERROR-level log and message identifying SEED/GROW as the upstream source of the failure.

**Code refs:** `src/questfoundry/pipeline/stages/polish/deterministic.py:146`

**Test refs:** `tests/unit/test_polish*.py` (no test for zero-choice detection)

### Cluster: Residue beat passage-layer mapping decision not recorded in plan

**Rules covered:** R-5.7, R-5.8

**Current state:** Phase 5b (residue content generation) does not record which passage-layer mapping choice (residue-passage-with-variants vs. parallel-passages) was selected. `ResidueSpec` has no field to store this choice. Phase 6 applies residue beats assuming a fixed mapping strategy without reference to a spec-recorded decision.

**Gap:** Spec R-5.8 requires "The chosen mapping is recorded in the plan so Phase 6 can apply it atomically." Current code cannot distinguish between the two mapping options per residue spec, violating atomic plan application independence and the separation between planning (Phase 5) and application (Phase 6).

**Recommended fix:** Add a `mapping_strategy` field to `ResidueSpec` (enum: `"residue_passage_with_variants"` | `"parallel_passages"`). Phase 5b's LLM call should include context on both options and request a choice. Phase 6's `_create_residue_beat_and_passage()` should consult this field and apply the corresponding strategy.

**Code refs:** `src/questfoundry/models/polish.py:181-194`, `src/questfoundry/pipeline/stages/polish/deterministic.py:1176-1241`

**Test refs:** `tests/unit/test_polish_apply.py` (Phase 6 application tests should verify mapping choice is honored)

### Cluster: Character arc metadata stored as separate nodes instead of entity annotations

**Rules covered:** R-3.3

**Current state:** Phase 3 creates `character_arc_metadata` nodes and links them via `has_arc_metadata` edges to entities (`llm_phases.py:286-299`). These are separate graph nodes of type `character_arc_metadata`.

**Gap:** Spec R-3.3 mandates "Arc metadata is stored as an annotation on Entity nodes, not as separate graph nodes." Current implementation violates this by creating separate nodes and linking via edges, which contradicts the ontology's annotation pattern.

**Recommended fix:** Instead of creating separate `character_arc_metadata` nodes, augment the entity node's data dict directly with a `character_arc` field containing `start`, `pivots`, `end_per_path`. Do not create separate nodes or `has_arc_metadata` edges. Update existing tests that expect the edge creation.

**Code refs:** `src/questfoundry/pipeline/stages/polish/llm_phases.py:286-299`

**Test refs:** `tests/unit/test_polish_phases.py:398-436` (`test_has_arc_metadata_edge_created` expects edge creation — must change to verify annotation)

### Cluster: False-branch beat role name does not match spec

**Rules covered:** R-5.10

**Current state:** Sidetrack beats created in Phase 6 `_apply_sidetrack()` set `role: "sidetrack_beat"` (`deterministic.py:1349`). No `belongs_to` edges are added (compliant). Diamond alternative passages do not create beats; they are pure passage alternatives.

**Gap:** Spec R-5.10 requires false-branch beats (both diamonds and sidetracks conceptually) to have `role: "false_branch_beat"`. Current code uses `"sidetrack_beat"` — nomenclature drift.

**Recommended fix:** Rename role from `"sidetrack_beat"` to `"false_branch_beat"`. If diamond alternatives ever create beats, they should also receive this role. Update all code checking `role == "sidetrack_beat"` to `role == "false_branch_beat"`, plus related tests.

**Code refs:** `src/questfoundry/pipeline/stages/polish/deterministic.py:1349`

**Test refs:** `tests/unit/test_polish_apply.py`

### Cluster: Post-convergence soft-dilemma gating incomplete

**Rules covered:** R-4c.3, R-4c.4

**Current state:** Phase 4c computes `requires` for intersection-passage choices by calling `compute_active_flags_at_beat()` on the target beat (`deterministic.py:790-813`). For non-intersection choices, `requires` is set to empty list (`deterministic.py:832`).

**Gap:** Spec R-4c.3 requires ALL post-convergence soft-dilemma choices to have `requires` set to the appropriate state flag. Current code gates only intersection-passage choices, which is a subset. Non-intersection post-convergence soft-dilemma choices may not receive `requires` gating.

**Recommended fix:** Expand the `requires` computation to detect post-convergence soft-dilemma choices (not just intersection-based) and set `requires` accordingly. Track which dilemmas are soft vs hard (from dilemma nodes). Hard-dilemma choices keep empty `requires` (R-4c.4).

**Code refs:** `src/questfoundry/pipeline/stages/polish/deterministic.py:790-814`

**Test refs:** `tests/unit/test_polish_deterministic.py` (choice-computation tests should verify `requires` gating for all post-convergence soft choices)

### Cluster: Choice label distinctness within source passage not validated

**Rules covered:** R-5.2

**Current state:** Phase 5a receives choice contexts and the LLM generates labels. The code deduplicates on `(from_passage, to_passage)` key but does not verify that multiple choices from the same passage have distinct labels.

**Gap:** Spec R-5.2 requires "Labels are distinct within a source passage — two choices from the same passage have clearly different labels." If two choices from the same passage receive the same label, it is silently applied.

**Recommended fix:** After collecting labels in Phase 5a, group by `from_passage` and check that labels within each group are unique (case-insensitive). If duplicates found, either trigger an LLM re-call for that passage's labels or log a warning and flag for human review.

**Code refs:** `src/questfoundry/pipeline/stages/polish/llm_phases.py:341-372`

**Test refs:** `tests/unit/test_polish_llm_phases.py` (should test label-distinctness validation)

### Cluster: Structural beats missing `created_by: POLISH` attribution

**Rules covered:** R-2.5

**Current state:** Phase 2 creates micro-beat nodes with `role: "micro_beat"`, zero `dilemma_impacts`, zero `belongs_to` (correct for R-2.1). The `created_by: POLISH` attribution (R-2.5) is NOT set on the node data dict. Same likely holds for residue beats and false-branch beats created in Phase 5/6.

**Gap:** Spec R-2.5 requires `created_by: POLISH` attribution for stage-attribution tracking (consumed by `created_by` property for pipeline validation tools).

**Recommended fix:** Add `"created_by": "POLISH"` to the beat-node data dict when creating micro-beats in Phase 2, residue beats in Phase 5/6, and false-branch beats in Phase 6. Update affected tests to assert the attribution field is set.

**Code refs:** `src/questfoundry/pipeline/stages/polish/llm_phases.py:1004-1015`, `deterministic.py:1189-1200, 1342-1354`

**Test refs:** `tests/unit/test_polish*.py` (should verify `created_by` field on structural beats)

### Uncheckable rules

- R-1.1: Reordering only within linear (single-predecessor, single-successor) sections. Enforced by code filtering logic; full verification requires LLM-mocked integration test to confirm reorderings never cross boundaries in practice.
- R-5.4: Label-generation LLM call receives full context. Requires inspection of the rendered prompt at runtime; code review of templates is partial evidence but not sufficient.

---

## M-FILL-spec

### Summary
- Rules checked: 17
- Compliant: 11 | Drift: 4 | Missing: 1 | Uncheckable: 1

*(Subagent originally produced 14 clusters; several were self-described as compliant or spec-gap observations. Refined here to 5 actionable findings; compliant clusters rolled back into Compliant count.)*

### Cluster: Voice Document approval not explicitly recorded on the node

**Rules covered:** R-1.7

**Current state:** `_phase_0_voice()` creates the Voice Document singleton, then proceeds through an `AutoApprovePhaseGate()` when the CLI is invoked with `--no-interactive` (which implies user pre-approval of all gates). Interactive mode uses an interactive gate. The approval itself — whether interactive or auto — is not recorded on the Voice node.

**Gap:** R-1.7 requires "Human approval of the Voice Document is required before Phase 2." `--no-interactive` is a legitimate form of pre-approval (the user has explicitly opted in at invocation time), so the gate mechanism itself is compliant. The gap is that the approval is not explicitly recorded on the artifact — a Voice node after Phase 0 looks identical whether it was interactively approved or auto-approved, and downstream readers can't tell.

**Recommended fix:** Stamp the Voice node at gate-pass time with `approved_at: <timestamp>` and `approval_mode: "interactive" | "no_interactive"`. Phase 2 entry can assert the stamp exists before starting prose generation. This makes the approval traceable without changing the approval flow.

**Code refs:** `src/questfoundry/pipeline/stages/fill.py:817-886` (Phase 0), `src/questfoundry/pipeline/stages/fill.py:299-340` (execute flow with gate)

**Test refs:** `tests/unit/test_fill_stage.py` (no test for approval stamping)

### Cluster: Entity-update silent skip on missing entity (R-2.14 + silent degradation + wrong log level)

**Rules covered:** R-2.14, plus Silent Degradation (CLAUDE.md §Anti-Patterns), plus §Logging

**Current state:** When `_resolve_entity_id()` returns None (entity not found in graph), the code logs at WARNING and silently skips the update. No escalation surface reaches the user; the narrative detail is lost silently.

**Gap:** R-2.14 requires FILL to halt and escalate to SEED when a new entity is needed — silent skip is the opposite. The log level (WARNING) is also wrong per CLAUDE.md §Logging: a lost detail suggests either a phantom entity ID from the LLM or a graph inconsistency, both of which are failures, not "degraded state the system handled correctly." This is a compound finding: silent degradation + wrong log level + missing escalation.

**Recommended fix:** Replace the skip-and-warn with an explicit escalation. Options: (a) raise `FillStageError` with a guidance message pointing to SEED, or (b) collect all missing-entity events into an escalation report surfaced to the CLI at stage end. Log at ERROR in either case. Do not silently skip.

**Code refs:** `src/questfoundry/pipeline/stages/fill.py:179-210, 1293-1306, 1729-1739`, `src/questfoundry/models/fill.py:65-75`

**Test refs:** `tests/unit/test_fill_stage.py` (no test for missing-entity handling)

### Cluster: Convergence-passage lookahead asymmetric

**Rules covered:** R-2.6

**Current state:** `format_lookahead_context()` provides lookahead for branch passages approaching convergence (branch sees convergence prose), but does NOT provide the inverse: when writing a convergence passage during the spine pass, the context does not include beat summaries of all connecting branches.

**Gap:** R-2.6 requires "At convergence points, the canonical-arc pass receives beat summaries of all arriving branches as lookahead so the convergence prose works for all arrivals." The asymmetry means canonical convergence prose is written without awareness of incoming branch context — later branches may arrive to prose that doesn't accommodate them.

**Recommended fix:** In `format_lookahead_context()`, add a convergence-detection path: if the current passage is a convergence point in the canonical arc, identify all branch passages that converge here, extract their beat summaries, and include them in the context.

**Code refs:** `src/questfoundry/graph/fill_context.py:877-950`, `src/questfoundry/pipeline/stages/fill.py:1196`

**Test refs:** `tests/unit/test_fill_context.py` (no test for convergence-direction lookahead)

### Cluster: Review-cycle cap escalation not visibly surfaced

**Rules covered:** R-5.2

**Current state:** After the final revision cycle (`final_cycle=True`), passages with unresolved flags are logged at WARNING as `upstream_escalation`. The stage completes with `status=completed` regardless. No ERROR, no stage failure, no structured escalation artifact for the user.

**Gap:** R-5.2 requires "Persistent quality issues escalate upstream — do not ship silently-low-quality prose." Logging at WARNING buries the escalation. Ship-as-is combined with warning-only logging is silent degradation.

**Recommended fix:** After the final revision cycle, if any passages retain unresolved flags: (a) log at ERROR (not WARNING), and (b) surface the escalation in the stage result (e.g., a structured `escalations` field on the FillResult) so the CLI can display it at stage end. Consider failing the stage with a clear message if the unresolved count exceeds a threshold.

**Code refs:** `src/questfoundry/pipeline/stages/fill.py:280-296, 1825-1835, 1753-1765`

**Test refs:** `tests/unit/test_fill_stage.py` (no test for escalation surfacing)

### Cluster: Code has phases and mechanisms not defined in spec

**Rules covered:** Spec-vs-code mismatch (multiple non-spec additions)

**Current state:** Code implements several phases and mechanisms not described in `docs/design/procedures/fill.md`:
- **Phase 1c: Mechanical Quality Gate** — deterministic prose-quality checks (near-duplicate detection via `rapidfuzz`, opening-trigram collision, vocabulary diversity, sentence-length variance). Runs between Phase 1 (generation) and Phase 2 (review).
- **Phase 4: Arc-Level Validation** — `_phase_4_arc_validation()` calls `run_arc_validation()` with checks for intensity progression, dramatic-question closure, narrative-function variety, dilemma-prose coverage. Spec's Phase 4 is only "Optional Second Cycle."
- **Two-step prose generation** (`_fill_prose_call` + `_fill_extract_call`) — default behavior; spec describes structured-output single-call.
- **Lexical-diversity tracking** (`compute_lexical_diversity` over a rolling window) — not mentioned in spec.

**Gap:** Per CLAUDE.md §Design Doc Authority, the spec supersedes code. Code-has-extras-not-in-spec is drift: either the spec must be updated to describe these mechanisms (preferred given their apparent value), or the mechanisms must be removed. Currently they exist in a grey zone where a reader of the spec has no way to know about them.

**Recommended fix:** The right first step is a spec update (per CLAUDE.md: update spec first, then code). Document Phase 1c and Phase 4 in the spec, explain two-step generation as an allowed implementation choice, and note lexical-diversity tracking as implementation guidance. After the spec update, this cluster's findings resolve to compliant.

**Code refs:** `src/questfoundry/pipeline/stages/fill.py:1349-1507` (Phase 1c), `src/questfoundry/pipeline/stages/fill.py:1864-1909` (Phase 4), `src/questfoundry/pipeline/stages/fill.py:630-716, 1215-1278` (two-step), `src/questfoundry/pipeline/stages/fill.py:1136-1247` (lexical diversity)

**Test refs:** `tests/unit/test_fill_validation.py` (Phase 4 checks tested)

### Uncheckable rules

- R-2.3: Every prose call receives the Voice Document. Code passes `voice_document` into context; full verification requires inspecting the rendered prompt at runtime to confirm the LLM actually reads it.

---

## M-DRESS-spec

### Summary
- Rules checked: 34
- Compliant: 24 | Drift: 6 | Missing: 1 | Uncheckable: 3

*(Subagent originally reported slightly different counts and non-matching rule numbers — reclassified to spec's R-1.x through R-5.x numbering. Two overlapping approval-recording clusters merged.)*

### Cluster: EntityVisual per-Entity coverage not validated

**Rules covered:** R-1.3, R-1.4

**Current state:** Phase 1's LLM generates EntityVisual nodes for entities mentioned in the output, but code does not validate that ALL entities with an `appears` edge receive visual profiles. If the LLM omits an entity, no error is raised.

**Gap:** R-1.3 requires "Every Entity with ≥1 `appears` edge has an EntityVisual node." R-1.4 requires non-empty `reference_prompt_fragment`. Neither is validated post-serialization — a DRESS run could complete with some appearing entities lacking a visual profile, leading to inconsistent illustrations downstream.

**Recommended fix:** After Phase 1 serialization, iterate all entities with `appears` edges and confirm each has an EntityVisual with non-empty `reference_prompt_fragment`. On mismatch, log ERROR and either halt or inject a minimal fallback profile with a warning. Alternatively, include all appearing Entity IDs in the serialize prompt's Valid IDs section so the LLM cannot omit them.

**Code refs:** `src/questfoundry/pipeline/stages/dress.py:500-507` (`apply_dress_art_direction`)

**Test refs:** `tests/unit/test_dress_stage.py:214` (`test_phase0_creates_nodes` does not validate all-entities coverage)

### Cluster: Passages pre-filtered out of brief generation

**Rules covered:** R-2.1

**Current state:** Phase 2 pre-filters passages by priority BEFORE brief generation. Passages with insufficient structural score or low LLM adjustment are skipped entirely — no brief is created.

**Gap:** R-2.1 requires "Every Passage has exactly one IllustrationBrief node with a `targets` edge," including low-priority ones (which the spec labels `priority: skip` — still a brief, just not rendered). Pre-filtering means the user at Gate 2 cannot see or reprioritize low-scoring passages because no brief exists to review.

**Recommended fix:** Generate briefs for every passage that has prose. Apply priority filtering as a brief-level attribute (`priority: skip`) rather than by suppressing brief creation. Phase 4 (Gate 2) then displays all briefs sorted by priority; the user can select, edit, or reprioritize.

**Code refs:** `src/questfoundry/pipeline/stages/dress.py:668-706` (priority filtering happens before LLM calls)

**Test refs:** `tests/unit/test_dress_stage.py` (no test for all-passages-get-briefs)

### Cluster: Gate 2 (image budget) not interactive; approval mode not recorded

**Rules covered:** R-4.1, R-4.4

**Current state:** Phase 4 auto-approves all briefs via `AutoApprovePhaseGate` and does not present a budget selection dialog. There is no conditional interactive prompt based on CLI mode. The approval — whether interactive or auto via `--no-interactive` — is not recorded on any artifact.

**Gap:** R-4.1 requires "Human explicitly sets the rendering budget." R-4.4 requires "Gate 2 approval recorded." `--no-interactive` is a legitimate pre-approval pattern, so auto-approving is compliant when that flag is set — but interactive mode should prompt, and either way the approval mode + timestamp should be recorded for traceability.

**Recommended fix:** (1) In interactive mode, present briefs sorted by priority and prompt for budget selection (number of briefs to render or priority cutoff). (2) In `--no-interactive` mode, apply a default (e.g., "all priority 1 + priority 2") and record the default choice. (3) Stamp approval metadata (`approved_at`, `approval_mode: "interactive" | "no_interactive"`, `budget: {…}`) on an artifact visible to Phase 5 and SHIP.

**Code refs:** `src/questfoundry/pipeline/stages/dress.py:932-978` (`_phase_3_review`), `src/questfoundry/pipeline/stages/dress.py:151` (`AutoApprovePhaseGate` hardcoded), `src/questfoundry/cli.py`

**Test refs:** `tests/unit/test_dress_stage.py:379` (`test_selects_all_briefs` mocks gate, does not test interactive mode)

### Cluster: Codex spoiler-ordering validation missing; no retry loop

**Rules covered:** R-3.6

**Current state:** Phase 3 generates codex entries in batches. `validate_dress_codex_entries()` checks that rank=1 exists and that state flag references are valid, but does not enforce spoiler ordering (lower-rank entries must not leak content gated behind higher-rank entries). No retry loop exists for failed batches.

**Gap:** R-3.6 requires "A lower-ranked (earlier-visible) entry must not prematurely disclose content whose reveal is gated behind a higher-ranked tier." LLM validation for spoilers is mentioned in the spec with max 2 retries per entity. Currently: (a) no spoiler check, (b) no retry logic. An entry that leaks rank-2 content at rank 1 would pass silently.

**Recommended fix:** (1) Add spoiler-leak detection: for each entity, cross-check lower-rank content against higher-rank reveals (either deterministic keyword/entity check or a secondary LLM "does rank N leak content from rank N+1" call). (2) Wrap batch processing in a retry loop (max 2 per entity per the spec). (3) On retry exhaustion, fall back to a minimal rank-1-only codex with a WARNING, rather than silently skipping.

**Code refs:** `src/questfoundry/pipeline/stages/dress.py:859-926` (`_phase_2_codex`, no retry loop), `src/questfoundry/graph/dress_mutations.py:236-284` (`validate_dress_codex_entries`, no spoiler checks)

**Test refs:** `tests/unit/test_dress_stage.py` (no spoiler-validation tests, no batch-failure recovery tests)

### Cluster: Codex rank 1 `visible_when: []` invariant not validated

**Rules covered:** R-3.2

**Current state:** The CodexEntry model allows `visible_when` to default to empty list. Code assumes rank 1 is always visible, but no validator enforces that (a) at least one rank-1 entry exists per entity, and (b) that rank-1 entry has `visible_when: []`.

**Gap:** R-3.2 requires "Rank 1 (rank=1) has `visible_when: []` — always visible from the start." Current validation checks that rank-1 exists but not that it is unconditional. A malformed rank-1 entry with a gate would pass.

**Recommended fix:** In `validate_dress_codex_entries()`, for each entity's entries: assert that at least one entry has `rank=1`, and that entry's `visible_when == []`. Reject entries that violate either condition.

**Code refs:** `src/questfoundry/graph/dress_mutations.py:256-262`

**Test refs:** `tests/unit/test_dress_mutations.py` (no test for `visible_when=[]` on rank 1)

### Uncheckable rules

- R-2.4: Caption diegetic-voice enforcement. Prompt-enforced; verifying caption prose is in-world voice requires LLM-based critique, not programmatic.
- R-3.4: Codex entries diegetic. Same reasoning as R-2.4.
- R-3.5: Codex entries self-contained. Requires narrative judgment of whether each entry is readable without prior tiers; not programmatically verifiable.

---

## M-SHIP-spec

### Summary
- Rules checked: 26
- Compliant: 15 | Drift: 7 | Missing: 4 | Uncheckable: 0

*(Subagent originally produced 10 clusters; one was a compliant-with-deprecation-note observation and three were test-coverage-only gaps on otherwise-compliant behaviour. The three test-coverage items are bundled below; the deprecation note is dropped as non-actionable.)*

### Cluster: Phase 4 Export Validation not implemented

**Rules covered:** R-4.1, R-4.2, R-4.3, R-4.4

**Current state:** SHIP lacks an explicit Phase 4 implementation for per-format validation. The stage runs exporters but does not parse Twee link reachability, headless-render HTML, schema-validate JSON, or verify PDF page count. No post-export validation gate.

**Gap:** Spec requires Phase 4 to run per-format validation and halt with ERROR on failure (R-4.2). Currently, no validation code exists; broken exports (e.g., Twee referencing a non-existent passage, PDF page-count mismatch) would be delivered silently.

**Recommended fix:** Implement per-exporter validation functions: Twee parser verifying reachability, JSON schema validator, HTML headless-render smoke check, PDF page-count checker. Add a post-export validation phase in `ShipStage.execute()` that raises `ShipStageError` on any failure. Validation logs at ERROR, halting bundle delivery.

**Code refs:** `src/questfoundry/pipeline/stages/ship.py:43-164`

**Test refs:** `tests/unit/test_ship_stage.py` (no validation tests)

### Cluster: Deterministic metadata headers missing on exports

**Rules covered:** R-3.6

**Current state:** Exporters produce output files without metadata headers. Twee/HTML/JSON/PDF are generated without pipeline version, graph snapshot hash, format version, or generation timestamp.

**Gap:** R-3.6 mandates every export include a deterministic header with those four fields. Required for reproducibility audits and provenance tracking.

**Recommended fix:** Add a metadata generation function computing: pipeline version from package metadata; graph snapshot hash (deterministic content hash); format version per exporter; ISO timestamp. Embed this per format — Twee as a metadata passage, HTML as meta tags, JSON as top-level fields, PDF as a metadata section.

**Code refs:** `src/questfoundry/export/twee_exporter.py`, `src/questfoundry/export/json_exporter.py`, `src/questfoundry/export/html_exporter.py`, `src/questfoundry/export/pdf_exporter.py`

**Test refs:** `tests/unit/test_*_exporter.py` (no metadata header tests across all four)

### Cluster: Codeword threshold warning missing

**Rules covered:** R-1.7

**Current state:** Codeword projection correctly filters to soft-dilemma flags (R-1.1 compliant). However, no threshold check triggers WARNING when projected codeword count exceeds the playability threshold of ~10.

**Gap:** R-1.7 requires logging WARNING when codeword count > ~10 (a human-facing signal that the gamebook may become unwieldy). Currently the count is emitted at INFO without a threshold-based warning.

**Recommended fix:** After `_project_state_flags_to_codewords()` returns, count the projected codewords. If count > 10, emit a WARNING log with the count and a note recommending reduction or manual review.

**Code refs:** `src/questfoundry/export/context.py:130-175`

**Test refs:** `tests/unit/test_export_context.py:287-343` (tests projection but not threshold)

### Cluster: HTML voice-document styling not integrated

**Rules covered:** R-3.3

**Current state:** HTML exporter renders passages with fixed CSS. No voice-document data is read or applied to styling.

**Gap:** R-3.3 requires HTML to use "voice-document-informed CSS/typography when available" — the voice document's register, sentence rhythm, and related fields should affect visual typography. Currently absent entirely.

**Recommended fix:** Extend `ExportContext` to include voice-document fields. In the HTML exporter, apply CSS classes or inline styles based on those fields (e.g., `class="register-sparse"` applies tighter spacing). Fall back to default CSS when the voice document is absent.

**Code refs:** `src/questfoundry/export/html_exporter.py`

**Test refs:** `tests/unit/test_html_exporter.py` (no voice-document-aware styling tests)

### Cluster: PDF pagination map not exported

**Rules covered:** R-3.5

**Current state:** PDF exporter uses seeded-random shuffling for passage numbering (R-3.5 reproducibility compliant). But the numbering map — passage-ID → PDF page — is not exported.

**Gap:** Spec states "a page-number map is included in the JSON metadata for debugging." Currently the map is computed internally and discarded; users cannot trace which passage became which page after a bug report.

**Recommended fix:** Compute the passage numbering in a shared utility (not only in the PDF exporter). Include the `passage_id → page_number` map as a field in the JSON export (or alongside the PDF as a sidecar `.map.json` file) when PDF format is targeted. Document the schema in the spec's Phase 3 notes.

**Code refs:** `src/questfoundry/export/pdf_exporter.py:305-344`

**Test refs:** `tests/unit/test_pdf_exporter.py` (no test for map output)

### Cluster: Partial-DRESS presence not warned

**Rules covered:** R-3.8

**Current state:** DRESS-absent case is handled gracefully (empty lists, `None` for art direction). However, if DRESS produced a partial art direction (e.g., style present but palette missing), the exporter silently propagates the partial data without warning.

**Gap:** R-3.8 requires graceful degradation — this case is silent rather than graceful. Partial art direction yields partial illustrations without signaling the gap to the user.

**Recommended fix:** In `_extract_art_direction()`, check for expected fields (style, palette, composition_notes). If any are missing on an otherwise-present art direction, log WARNING identifying the missing fields. Continue execution; this is degradation, not failure.

**Code refs:** `src/questfoundry/export/context.py:299-306`

**Test refs:** `tests/unit/test_export_context.py:241-247` (tests absence but not partial presence)

### Cluster: Cross-cutting test coverage gaps (determinism, format consistency, label verbatim)

**Rules covered:** R-2.4, R-3.1, R-3.2 — all implementation is compliant; tests absent

**Current state:** Three rules are correctly implemented in code but lack regression tests:
- **R-2.4 determinism** — SHIP does not mutate the graph, but no test verifies that running SHIP twice on the same graph produces byte-identical output.
- **R-3.1 format divergence** — All four exporters read from a single `ExportContext`, but no cross-format consistency test asserts that passage/choice/codeword/codex counts match across formats.
- **R-3.2 Twee label verbatim** — `TweeExporter._render_choice()` uses `choice.label` unchanged, but no test asserts an exact-match verbatim preservation (e.g., round-tripping a label with special characters).

**Gap:** Per the audit-design principle that the spec supersedes code AND tests, implementation-compliant-but-untested is still drift — regression tests are required to keep the implementation compliant as the code evolves.

**Recommended fix:** Add three targeted tests:
1. `tests/unit/test_ship_stage.py` — run SHIP twice on the same graph, assert `hash(output_1) == hash(output_2)` per format.
2. `tests/integration/test_ship_format_consistency.py` — export the same story to all four formats; assert passage/choice/codeword/codex counts are identical.
3. `tests/unit/test_twee_exporter.py` — create a choice with `label = "Trust the mentor—say nothing"` (em-dash, quotes); assert the Twee output contains that exact string.

**Code refs:** `src/questfoundry/pipeline/stages/ship.py`, `src/questfoundry/export/twee_exporter.py:192-213`, all exporters.

**Test refs:** (new tests to add — currently absent)

---

## M-logging-compliance

### Summary

Cross-cutting audit of all 486 log call sites in `src/questfoundry/` against CLAUDE.md §Logging. Pattern-based audit (individual classification of 486 sites was infeasible); findings drawn from targeted grep + sample inspection of high-density modules.

- Log call sites total: 486
- Sites matching a misuse pattern (estimated): ~50–80
- Clusters: 4

### Cluster: LLM-proposal rejection/skip events logged at WARNING (should be INFO)

**Pattern:** `log.warning("phase3_insufficient_valid_beats", …)`, `log.warning("phase3_incompatible_intersection", …)`, `log.warning("phase3_stage1_compatibility_failed", …)`, `log.warning("serialize_paths_retry_failed", …)`, `log.warning("missing_section_brief", …)`, etc.

**CLAUDE.md litmus test:** *"If the system detected a problem AND handled it correctly (rejected bad input, used a fallback, skipped an invalid proposal), that's `INFO` or `DEBUG` — not `WARNING`."*

**Current state:** High concentration of WARNING logs in `grow/llm_phases.py` (23 calls), `agents/serialize.py` (17), `polish/llm_phases.py` (11), `grow/llm_helper.py` (12) — many are per-proposal rejections where the pipeline continues with other proposals. These are "handled correctly," not "something unexpected happened that warrants attention."

**Gap:** These events are normal pipeline operation (candidate rejected → try next), not degraded states. WARNING obscures the real warnings (e.g., actual contract violations or recovery fallbacks).

**Recommended fix:** Audit the rejection/skip warnings in the four high-density modules listed above. Reclassify:
- Per-candidate rejection during retry loop → INFO (or DEBUG for high-frequency)
- Actual degraded state that may produce lower-quality output → WARNING (keep)
- Failure mode requiring user attention that the pipeline cannot auto-recover from → ERROR

Expect ~30–50 sites to move from WARNING to INFO/DEBUG.

**Code refs:** `src/questfoundry/pipeline/stages/grow/llm_phases.py:234, 245, 291, 479, 498, 507, 523, 639, 888, 972`, `src/questfoundry/agents/serialize.py:2082, 2487`, `src/questfoundry/pipeline/stages/polish/llm_phases.py`, `src/questfoundry/pipeline/stages/grow/llm_helper.py`

**Test refs:** None (log-level tests typically absent; not prioritized as regression concern)

### Cluster: Fallback-with-default at WARNING — some should be INFO, others ERROR

**Pattern:** `log.warning("invalid_aspect_ratio", raw=raw, fallback="16:9")`, `log.warning("unknown_aspect_ratio", aspect_ratio=aspect_ratio, fallback="1:1")`, `log.warning("convergence_dilemma_role_missing", dilemma_id=...)`, `log.warning("asset_missing", path=..., passage=...)`, `log.warning("entity_missing_type", entity_id=...)`, `log.warning("missing_scene_type", beat_id=...)`, `log.warning("validation_skipped", reason=...)`.

**Current state:** Mixed bag of fallback logs. Some represent "invalid input from external source, defaulted cleanly" (benign → INFO). Others represent "graph integrity issue detected" (real WARNING or ERROR).

**Gap:** Per CLAUDE.md litmus:
- `invalid_aspect_ratio ... fallback="16:9"`: input mistyped, default applied. System handled correctly → INFO.
- `asset_missing path=... passage=...`: file referenced but not on disk. This is a real integrity problem. Should be ERROR.
- `convergence_dilemma_role_missing dilemma_id=...`: dilemma missing its role field. This violates the SEED contract per R-7.1; should be ERROR.
- `entity_missing_type` / `missing_scene_type`: similar — upstream contract violation, should be ERROR.
- `validation_skipped`: depends on why. If intentional user opt-out → INFO; if a skip-because-we-failed → ERROR.

**Recommended fix:** Reclassify each listed site per the litmus test. Rough estimate: 3–5 sites should move from WARNING to ERROR (integrity issues); a few should move from WARNING to INFO (clean defaults for malformed external input).

**Code refs:** `src/questfoundry/pipeline/stages/dress.py:116`, `src/questfoundry/pipeline/stages/grow/deterministic.py:483`, `src/questfoundry/providers/image_placeholder.py:118`, `src/questfoundry/export/assets.py:56, 101`, `src/questfoundry/inspection.py:523`, `src/questfoundry/graph/fill_context.py:1493`, `src/questfoundry/graph/grow_algorithms.py:1251, 2572`

**Test refs:** None

### Cluster: Silent `except ... pass` handlers without compensating log

**Pattern:** Narrow `except` blocks with bare `pass` or `pass` preceded by a comment.

**Current state:** 5+ sites across:
- `src/questfoundry/agents/summarize.py:153-154` — "If parsing fails, stick with the raw content" — benign-ish.
- `src/questfoundry/providers/factory.py:372-373` — `except ValueError: pass` — context unclear without inspection.
- `src/questfoundry/observability/langchain_callbacks.py:46-47` — `except ValueError: pass` in callback parsing — likely benign.
- `src/questfoundry/artifacts/validator.py:489-490, 496-497` — skipping invalid class types — benign.

**Gap:** Per CLAUDE.md Silent Degradation policy, silent failure of structural operations is forbidden. Most of these sites are defensive narrow exceptions, but a log at DEBUG level would make them visible without being noisy. Currently there's no record if any of these fire unexpectedly.

**Recommended fix:** Add DEBUG log to each `except...pass` site identifying the swallowed exception type and a brief reason. Audit whether each site really represents a benign case or whether the handler is masking a real bug (per-site judgment required).

**Code refs:** see Current state list.

**Test refs:** None

### Cluster: Entity-update silent-skip logging (cross-reference)

**Pattern:** `log.warning("entity_update_skipped", entity_id=...)` on missing-entity lookup.

**Current state:** In `fill.py:1302-1306, 1734-1739` and `graph/mutations.py` (entity lookup in BRAINSTORM phase — see M-BRAINSTORM-spec §Dilemma-to-entity anchoring cluster).

**Gap:** Already filed as compliance issues under FILL (#1321) and BRAINSTORM (#1273) as full silent-degradation violations, not just logging issues. Noted here so the logging cross-cutting audit does not treat the sites as independent findings.

**Recommended fix:** Resolved by fixing #1321 and #1273. No separate action required in M-logging-compliance.

**Code refs:** Cross-reference #1321 (FILL), #1273 (BRAINSTORM).

**Test refs:** See the linked issues.

---

## M-silent-degradation

### Summary

Cross-cutting audit against CLAUDE.md §Anti-Patterns ("Silent degradation of story structure constraints"). Per the policy: *"if the pipeline cannot satisfy a structural requirement (cross-dilemma ordering, intersection formation, DAG consistency), it MUST fail loudly. Silently skipping constraints and producing a weaker story is not acceptable."*

The audit combines (a) direct grep of known violation signatures and (b) aggregation of silent-degradation findings already surfaced by the 8 per-stage audits. Per-stage findings are rolled up by reference — the cross-cutting milestone adds clusters that cut across stages or that were missed by the stage audits.

- Direct new findings: 2 clusters
- Cross-referenced from stage audits: 1 roll-up cluster
- Clusters total: 3

### Cluster: Topological-sort cycle silently falls back to sorted order (DAG-integrity violation)

**Rules covered:** CLAUDE.md §Anti-Patterns ("Silent degradation of story structure constraints"), Story Graph Ontology Part 8 (beat DAG invariants), indirect: POLISH R-4c.1 (upstream DAG must be acyclic).

**Pattern:** Topological-sort helpers catch cycle conditions and return a sorted-by-ID fallback instead of raising. The rest of the pipeline treats the degraded output as valid order.

**Current state:** Three sites silently produce wrong beat order on cycle:
- `src/questfoundry/graph/algorithms.py:366-369` — `_topological_sort_subset`. Docstring explicitly says *"Falls back to sorted order if cycles are detected."* No log, no raise. Wraps the Kahn's-algorithm loop and appends any remaining (cyclic) beats in alphabetical order.
- `src/questfoundry/graph/grow_algorithms.py:677-678` — `except ValueError: pass  # Fallback: no reference if global beat set has cycles` inside `enumerate_arcs`. Leaves `reference_positions = {}`; subsequent per-arc sort proceeds without reference.
- `src/questfoundry/graph/grow_algorithms.py:699-700` — inside the same `enumerate_arcs` per-arc loop: `except ValueError: sequence = sorted(beat_set)  # Fallback for cycles`. No log.
- `src/questfoundry/graph/grow_algorithms.py:2607-2615` — `get_path_beat_sequence` logs WARNING `interleave_path_cycle_fallback` and returns `sorted(beats)`. Comment says "should not happen."

**Gap:** A cycle in the beat DAG is a structural invariant violation, not a recoverable condition. The "should not happen" comments confirm this. Silently producing sorted output means downstream POLISH operates on a degraded sequence (wrong narrative order); the cycle is never surfaced. Per CLAUDE.md, this must fail loudly.

**Recommended fix:** Replace each silent/warning fallback with `raise PipelineInvariantError` (or an equivalent loud exception type). Let it propagate to the phase runner so the phase status becomes `failed` and the stage halts. If callers currently rely on the fallback for defensive purposes, add an explicit invariant check earlier in the phase so the failure site is clear.

**Code refs:** `src/questfoundry/graph/algorithms.py:332-371`, `src/questfoundry/graph/grow_algorithms.py:670-710`, `src/questfoundry/graph/grow_algorithms.py:2596-2616`.

**Test refs:** `tests/unit/test_grow_algorithms.py` — add cycle-input tests that assert the exception is raised.

### Cluster: FILL LLM-failure silent continuation (voice / blueprint / extract)

**Rules covered:** CLAUDE.md §Anti-Patterns (Silent degradation), FILL procedure general "escalate, don't swallow" posture (related to R-2.14 / R-5.2 already filed under #1321 and FILL quality escalation cluster).

**Pattern:** Broad `except Exception:` around LLM sub-calls in FILL, logs WARNING, continues with an empty / skipped result. The stage completes successfully with a degraded artifact.

**Current state:** Three sites:
- `src/questfoundry/pipeline/stages/fill.py:835-842` — voice research: `except Exception: log.warning("voice_research_failed", exc_info=True)`. `research_notes` stays `None`, prose generation proceeds without voice guidance; no indication in the final artifact.
- `src/questfoundry/pipeline/stages/fill.py:1091-1097` — blueprint validation: `except Exception: log.warning("blueprint_validation_failed", passage_id=full_pid); continue`. A passage proceeds without its expand blueprint; downstream generation falls back to defaults silently.
- `src/questfoundry/pipeline/stages/fill.py:1260-1276` — entity extraction: `except (ValidationError, ValueError, RuntimeError): log.warning("entity_extract_failed", ...)`. `entity_updates` stays empty; narrative entity state diverges from the generated prose with no surfaced signal.

**Gap:** Each case is LLM failure absorbed as a shrug. The degraded output ships. Per the Silent Degradation policy, LLM failures in structural paths must surface loudly (ERROR, halt, or an explicit degradation report). Logging at WARNING and continuing is exactly the pattern CLAUDE.md names as a violation.

**Recommended fix:** For each site, pick one of:
1. **Raise a `FillStageError`** with context (which sub-call failed, which passage). Pipeline halts. Appropriate when the missing output corrupts the artifact.
2. **Accumulate failures into a FILL-stage escalation report** surfaced at stage completion (e.g., `GrowPhaseResult.detail` analog, or an explicit summary in the run output). Log at ERROR. User sees "FILL completed with N degraded passages: …" and can decide.

Voice-research: degraded prose quality but not structurally broken → option 2. Blueprint + entity-extract: structural → option 1, or option 2 if repeatable per-passage and bounded.

**Code refs:** `src/questfoundry/pipeline/stages/fill.py:835-842`, `src/questfoundry/pipeline/stages/fill.py:1091-1097`, `src/questfoundry/pipeline/stages/fill.py:1260-1276`.

**Test refs:** `tests/unit/test_fill_stage.py` — add tests that inject LLM-call failures and assert the stage either raises or surfaces the degradation report.

### Cluster: Roll-up of per-stage silent-degradation findings (cross-reference, no new issue)

**Pattern:** Silent-degradation findings that were already surfaced and filed as compliance issues under their owning stage. Listed here so the cross-cutting milestone shows the full picture without duplicating the per-stage issues.

**Cross-referenced per-stage issues:**

| Stage | Issue | Finding |
|---|---|---|
| BRAINSTORM | #1273 | Dilemma `anchored_to` edge silently dropped when entity_id cannot be resolved |
| SEED | (cluster: convergence LLM failure silent default, R-7.5) — filed under M-SEED-spec | LLM failure in convergence analysis applies defaults without WARNING |
| SEED | (cluster: dilemma ordering LLM failure silent application) | `serialize_dilemma_relationships()` returns empty on LLM failure with no log |
| SEED | (cluster: `explored` invariant silent break) | If pruning ever mutates `explored`, no runtime check catches it |
| GROW | (cluster: all-intersections-rejected not raised to ERROR) | Zero intersection groups formed when signals suggested some should is logged at WARNING; must be ERROR + halt |
| GROW | (cluster: R-2.7 intersection edge silently dropped) | Intersection edge across path-shared commit beats silently ignored during traversal |
| POLISH | (cluster: zero-choice silent pass to Phase 6) | Choice count of zero passes through; spec requires ERROR halt |
| POLISH | (cluster: duplicate choice labels silently applied) | Two choices from same source with same label silently accepted |
| FILL | #1321 | Entity-update silent skip on missing entity (R-2.14 + wrong log level + missing escalation) |
| FILL | (cluster: quality-escalation silent) | Persistent quality issues shipped with WARNING instead of escalating upstream per R-5.2 |
| DRESS | (cluster: spoiler-leak silent pass) | No spoiler-leak detection; rank-1 can silently disclose rank-2 content |
| SHIP | (cluster: per-format validation missing) | Broken exports (dangling Twee passage refs, PDF page mismatch) shipped silently |
| SHIP | (cluster: partial DRESS propagation) | Partial art direction (e.g., style without palette) silently propagated by exporters |
| logging | #1340 | Rejection/skip events logged at WARNING hide real warnings — tangentially masks silent-degradation visibility |

**Recommended fix:** Fixed by resolving the linked issues. No new issue filed under M-silent-degradation for this cluster.

**Code refs:** See each linked issue / stage-milestone cluster.

**Test refs:** See each linked issue.

---

## M-contract-chaining

### Summary

Cross-cutting audit of Stage Input Contract / Stage Output Contract enforcement at the 7 adjacent-stage seams. Per the authoritative specs, each stage's procedure doc defines an explicit contract for what must be true at entry and what will be true at exit, and the "verbatim match" claim requires Stage N Input ≡ Stage N-1 Output.

**Per-seam summary:**

| Seam | Entry-contract check | Exit-contract check | Classification |
|---|---|---|---|
| DREAM → BRAINSTORM | Vision node existence only (`BrainstormStage._get_vision_context`) | None | drift — minimal entry, no exit |
| BRAINSTORM → SEED | Loads brainstorm context from graph; no explicit validator | None | missing — implicit seam |
| SEED → GROW | `last_stage` sentinel check only (`grow/stage.py:287-291`) | None | missing — sentinel-only seam |
| GROW → POLISH | **Explicit `validate_grow_output(graph)`** (`polish/stage.py:239`) | Exit: **`validate_polish_output(graph)`** (`polish/deterministic.py:1395-1397`) | **compliant** — both sides enforce |
| POLISH → FILL | `last_stage` sentinel check only (`fill.py:391-396`) | None | missing — sentinel-only seam |
| FILL → DRESS | `last_stage` sentinel check only (`dress.py:283-286`) | None | missing — sentinel-only seam |
| DRESS → SHIP | `last_stage`-like: checks "passages have prose" only (`ship.py:76-89`); does not validate DRESS output contract | None | drift — partial entry, wrong upstream |

- Seams total: 7
- Compliant: 1 (GROW → POLISH)
- Drift: 2 (DREAM → BRAINSTORM; DRESS → SHIP)
- Missing: 4 (BRAINSTORM → SEED; SEED → GROW; POLISH → FILL; FILL → DRESS)
- Clusters: 2

### Cluster: Stage entry skips explicit upstream-contract validation (6 of 7 seams)

**Rules covered:** Cross-stage contract invariant — Stage N Input Contract ≡ Stage N-1 Output Contract, per every `docs/design/procedures/*.md` "Stage Input Contract" section.

**Pattern:** Each downstream stage relies on a `last_stage` sentinel (or a narrow existence check) as a proxy for upstream completion. Only POLISH runs an explicit validator that mirrors the upstream (GROW) Stage Output Contract.

**Current state — seams 1–6 (excluding GROW→POLISH):**
- `src/questfoundry/pipeline/stages/brainstorm.py:152-160` — DREAM→BRAINSTORM: checks `graph.get_node("vision") is not None`. Does not validate the vision has the fields the DREAM Stage Output Contract requires (genre, tone, themes, audience, scope, etc.).
- `src/questfoundry/pipeline/stages/seed.py:291` — BRAINSTORM→SEED: loads brainstorm context from graph via `_get_brainstorm_context`. No structural validation; relies on implicit "the LLM will cope."
- `src/questfoundry/pipeline/stages/grow/stage.py:287-291` — SEED→GROW: `last_stage in ("seed", "grow", …)` sentinel; no validator.
- `src/questfoundry/pipeline/stages/fill.py:391-396` — POLISH→FILL: `last_stage in ("polish", …)` sentinel; no validator.
- `src/questfoundry/pipeline/stages/dress.py:283-286` — FILL→DRESS: `last_stage in ("fill", …)` sentinel; no validator.
- `src/questfoundry/pipeline/stages/ship.py:76-89` — DRESS→SHIP: checks "passages have prose" (which is really a FILL contract, not a DRESS contract); no DRESS-output validator.

**Gap:** Each seam is a point where the "verbatim match" claim in the procedure docs can drift silently. A malformed or partial upstream artifact passes through as long as the sentinel says the stage ran. When downstream later fails, the error message points at downstream code rather than the upstream contract break.

The GROW→POLISH seam demonstrates the correct pattern: `validate_grow_output(graph)` runs at POLISH entry, and a contract break produces a clear error that identifies GROW as the broken upstream.

**Recommended fix:** For each of the 6 seams, create a `validate_<upstream>_output(graph) -> list[str]` helper alongside the upstream stage (mirroring `polish_validation.validate_grow_output`). Call it at the downstream stage's entry, after rewind and before any phase dispatch. On errors, raise with a message that names the upstream stage explicitly (so the failure lands on the right step).

Recommended new helpers:
- `validate_dream_output(graph)` — vision node field coverage per DREAM Stage Output Contract
- `validate_brainstorm_output(graph)` — entities + dilemmas + answers counts, required fields, anchored_to coverage
- `validate_seed_output(graph)` — paths, commit/pre-commit beats, Y-shape skeleton invariants
- `validate_fill_output(graph)` — every passage has non-empty prose
- `validate_dress_output(graph)` — codex entries + entity appears edges + art direction (see also cluster 2 below)

Stage-owning milestones (not this epic) will own implementing these — this epic only tracks the entry-validation wiring.

**Code refs:** see Current state list.

**Test refs:** `tests/integration/test_stage_chaining.py` (new) — fixtures that produce partial upstream artifacts and assert the downstream stage halts with a clear error pointing at the upstream.

### Cluster: Stage exit contract not enforced (6 of 7 stages produce output without validation)

**Rules covered:** Every procedure doc's "Stage Output Contract" section; Design Doc Authority (specs supersede code, so if the spec says an output has N fields, the code must produce all N).

**Pattern:** The upstream side of a seam — the stage's exit — produces an artifact and calls `graph.set_last_stage(...)` without validating that the artifact actually satisfies its documented Stage Output Contract. Only POLISH runs `validate_polish_output(graph)` at exit (`polish/deterministic.py:1395-1397`).

**Current state:** DREAM, BRAINSTORM, SEED, GROW, FILL, DRESS all lack an explicit exit-validation call. Errors that slip through downstream become hard to localize because the upstream was allowed to "succeed" with a malformed artifact.

Examples of issues this hides (cross-referenced from per-stage milestones):
- BRAINSTORM #1273 — dilemma written without an `anchored_to` edge; no exit-validator would have caught this.
- SEED `explored` silent break (M-SEED-spec cluster) — no runtime check.
- GROW all-intersections-rejected / R-2.7 silent drop (M-GROW-spec clusters).
- FILL degraded passage (M-FILL-spec #1321 + M-silent-degradation #1345).

**Gap:** Exit validation is the mirror of entry validation. Without it, the seam is only single-sided enforcement at best. If the downstream stage never checks (see cluster 1), malformed artifacts pass through both sides of the seam unnoticed.

**Recommended fix:** For each of DREAM, BRAINSTORM, SEED, GROW, FILL, DRESS, add a stage-exit call to the stage's `validate_<stage>_output(graph)` (the same function created in cluster 1). The call runs after the last phase / last mutation and before `graph.set_last_stage(...)`. On errors, raise the stage-local error type, so the stage that produced the bad artifact owns the failure.

POLISH is already compliant (keep as reference pattern).

**Code refs:** `src/questfoundry/pipeline/stages/dream.py`, `brainstorm.py`, `seed.py`, `grow/stage.py`, `fill.py`, `dress.py` — each needs an exit hook alongside where `set_last_stage(...)` is called.

**Test refs:** `tests/integration/test_stage_chaining.py` (same file as cluster 1) — fixtures that mutate a stage's output to violate the contract and assert the stage raises at exit, not silently saves.

---

<!-- Milestone sections appended below as each audit stage completes. -->
