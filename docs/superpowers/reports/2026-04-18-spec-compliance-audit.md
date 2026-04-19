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
| M-SEED-spec | drafted | 14 | — | — |
| M-GROW-spec | pending | — | — | — |
| M-POLISH-spec | pending | — | — | — |
| M-FILL-spec | pending | — | — | — |
| M-DRESS-spec | pending | — | — | — |
| M-SHIP-spec | pending | — | — | — |
| M-logging-compliance | pending | — | — | — |
| M-silent-degradation | pending | — | — | — |
| M-contract-chaining | pending | — | — | — |

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

### Cluster: Missing Human Approval Gate for Vision Node

**Rules covered:** R-1.12, R-1.13

**Current state:** The DREAM stage executes (discuss → summarize → serialize) and writes the Vision node to the graph via `apply_dream_mutations()`, but there is no explicit human approval step before the stage marks itself complete. The CLI command `qf dream` runs the stage and saves the artifact without waiting for user confirmation. There is no rejection loop mechanism for the user to request changes and loop back.

**Gap:** The authoritative spec (dream.md, R-1.12 and R-1.13) requires DREAM to be "not complete until the human explicitly approves the Vision node" and rejection must "loop back to the operation that contains the misalignment." The current implementation skips both. The Vision node is upserted to the graph immediately after serialization with no approval gate. If the user is unsatisfied, there is no in-pipeline mechanism to request changes and re-run a specific operation (spark exploration, constraint definition, or synthesis).

**Recommended fix:** Introduce an approval gate after the serialize phase: display the serialized Vision to the user and require an explicit approve/reject response before marking DREAM complete. On rejection, support user indication of which operation failed (spark exploration, constraint definition, or synthesis) and re-enter that operation rather than re-serializing. This may require changes to the orchestrator's phase-progression model and the CLI's interactive flow.

**Code refs:** `src/questfoundry/pipeline/stages/dream.py:170-191` (no approval check), `src/questfoundry/graph/mutations.py:561-591` (upsert without approval), `src/questfoundry/cli.py:784-869` (no approval prompt)

**Test refs:** `tests/unit/test_dream_stage.py` (no tests for approval gate), `tests/integration/test_dream_pipeline.py` (no approval simulation)

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

### Cluster: Path Freeze approval gate not recorded

**Rules covered:** R-6.4

**Current state:** The spec requires "Path Freeze: Human approval is required and explicitly recorded." The code does not show a mechanism for recording or checking this approval. No gate or flag is set after Phase 6. Nothing prevents Phase 6 output from being consumed by GROW if the human did not explicitly approve Path Freeze.

**Gap:** R-6.4 is a code-side requirement (the approval must be explicitly recorded somewhere checkable by downstream stages). There is no recorded approval state in the artifact or graph.

**Recommended fix:** Add an explicit approval flag/timestamp to the SEED artifact or a graph-level marker (e.g., a `path_freeze_approved_at` field). Downstream stages (GROW) should check for this flag before proceeding. The CLI layer should prompt for approval at stage completion and set the flag only on explicit confirmation.

**Code refs:** `src/questfoundry/pipeline/stages/seed.py` (no approval gate visible)

**Test refs:** None (approval state not currently representable)

### Uncheckable rules

- R-3.5: Consequences describe world state vs. player actions — requires semantic understanding of prose. Cannot be programmatically checked at the model level. Enforced via SEED prompt engineering; any runtime check would require an LLM-based critique call. Deferred to runtime-verification follow-on track.

---

<!-- Milestone sections appended below as each audit stage completes. -->
