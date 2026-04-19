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
| M-SEED-spec | pending | — | — | — |
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

<!-- Milestone sections appended below as each audit stage completes. -->
