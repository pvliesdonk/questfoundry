"""LLM-powered phase implementations for the GROW stage.

Contains _LLMPhaseMixin with all phases that require LLM calls:
phases 3, 4a-4f, 8c.

GrowStage inherits this mixin so ``execute()`` can delegate to
``self._phase_3_intersections()``, etc.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import (
    ENTITY_CATEGORIES,
    strip_scope_prefix,
)
from questfoundry.graph.context_compact import (
    ContextItem,
    compact_items,
    truncate_summary,
)
from questfoundry.graph.graph import Graph
from questfoundry.models.grow import GrowPhaseResult
from questfoundry.pipeline.stages.grow._helpers import (
    GrowStageError,
    _format_structural_feedback,
    log,
)
from questfoundry.pipeline.stages.grow.registry import grow_phase

# Maximum characters to show from a beat summary when building LLM context for
# temporal hint resolution prompts.
_TEMPORAL_HINT_SUMMARY_TRUNCATION = 100

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.mutations import GrowValidationError


class _LLMPhaseMixin:
    """Mixin providing LLM-powered GROW phases (3, 4a-4f, 8c).

    Expects the host class to provide (via ``_LLMHelperMixin`` or directly):

    - ``_grow_llm_call()``
    - ``_validate_and_insert_gaps()``
    - ``_compact_config()``
    - ``_max_concurrency``
    - ``_on_connectivity_error``
    - ``_lang_instruction``
    - ``PROLOGUE_ID``
    """

    @grow_phase(name="intersections", depends_on=["validate_dag"], priority=1)
    async def _phase_3_intersections(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 3: Intersection detection (structural, multi-dilemma).

        Pre-clusters beats from different dilemmas into candidate groups
        using algorithmic signal detection (shared locations/entities),
        then asks the LLM to evaluate which groups form natural scenes.
        Intersections are purely structural (multi-dilemma scene merging);
        prose differentiation for same-dilemma convergences is handled
        by Phase 8d (residue beats).

        **Runs before interleave_beats** so that the beat DAG is clean (no
        predecessor edges yet) when intersection compatibility is checked.
        This eliminates the conditional-prerequisite rejection problem that
        occurred when interleave ran first (#1124).

        Preconditions:
        - Beat DAG validated (Phase 1 passed).
        - Beats have belongs_to edges with single-dilemma mapping.
        - Beat nodes have locations, entities for candidate clustering.
        - No predecessor edges exist yet (interleave has not run).

        Postconditions:
        - Accepted intersections marked on beat nodes via apply_intersection_mark.
        - Resolved locations stored on intersected beats.
        - Incompatible proposals rejected (requires conflicts, same dilemma).

        Invariants:
        - Pre-clustering is deterministic (shared locations/entities).
        - LLM only evaluates pre-clustered candidate groups.
        - Structural retry: up to 2 attempts with targeted error feedback.
        - All intersections applied in batch to avoid cascade effects.
        """
        from questfoundry.graph.grow_algorithms import (
            apply_intersection_mark,
            build_intersection_candidates,
            check_intersection_compatibility,
            format_intersection_candidates,
            resolve_intersection_location,
        )
        from questfoundry.models.grow import Phase3Output

        # Build candidate pool
        candidates = build_intersection_candidates(graph)
        if not candidates:
            return GrowPhaseResult(
                phase="intersections",
                status="completed",
                detail="No intersection candidates found (no beats share signals across dilemmas)",
            )

        # Build beat-to-dilemma mapping and valid beat ID set
        beat_nodes = graph.get_nodes_by_type("beat")
        path_nodes = graph.get_nodes_by_type("path")

        from collections import defaultdict

        from questfoundry.graph.context import normalize_scoped_id

        beat_dilemmas: dict[str, set[str]] = defaultdict(set)
        belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
        for edge in belongs_to_edges:
            beat_id = edge["from"]
            path_id = edge["to"]
            path_data = path_nodes.get(path_id)
            if not path_data:
                continue
            dilemma_id = path_data.get("dilemma_id")
            if dilemma_id:
                beat_dilemmas[beat_id].add(normalize_scoped_id(dilemma_id, "dilemma"))

        # Collect valid beat IDs (beats that map to exactly 1 dilemma)
        valid_beat_ids: set[str] = set()
        for candidate in candidates:
            for bid in candidate.beat_ids:
                dilemma_ids = beat_dilemmas.get(bid, set())
                if len(dilemma_ids) == 1:
                    valid_beat_ids.add(bid)

        if not valid_beat_ids:
            return GrowPhaseResult(
                phase="intersections",
                status="completed",
                detail=(
                    "No intersection candidates found "
                    "(all candidate beats span multiple dilemmas or lack dilemma mapping)"
                ),
            )

        # Format candidates as pre-clustered groups for the LLM
        candidate_groups_text = format_intersection_candidates(
            candidates, beat_nodes, beat_dilemmas, graph=graph
        )

        context: dict[str, str] = {
            "candidate_groups": candidate_groups_text,
            "valid_beat_ids": ", ".join(sorted(valid_beat_ids)),
            "structural_feedback": "",
        }

        # Call LLM with structural retry loop
        from questfoundry.graph.grow_validators import validate_phase3_output

        validator = partial(validate_phase3_output, valid_beat_ids=valid_beat_ids)

        # 2 attempts: initial call + 1 structural retry
        max_structural_retries = 2
        total_llm_calls = 0
        total_tokens = 0
        # Initialised before the retry loop so pyright sees them as always bound;
        # each iteration overwrites them inside the loop body.
        accepted: list[tuple[list[str], str | None, list[str], str]] = []
        applied_count = 0
        skipped_count = 0

        for structural_attempt in range(max_structural_retries):
            try:
                result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                    model=model,
                    template_name="grow_phase3_intersections",
                    context=context,
                    output_schema=Phase3Output,
                    semantic_validator=validator,
                )
            except GrowStageError as e:
                return GrowPhaseResult(
                    phase="intersections",
                    status="failed",
                    detail=str(e),
                    llm_calls=total_llm_calls,
                    tokens_used=total_tokens,
                )

            total_llm_calls += llm_calls
            total_tokens += tokens

            # Validate and apply intersections (reset per retry attempt)
            applied_count = 0
            skipped_count = 0
            pre_intersection_graph = Graph.from_dict(graph.to_dict())
            accepted = []
            structural_errors: list[GrowValidationError] = []

            for proposal in result.intersections:
                # Filter to valid beat IDs
                valid_ids = [bid for bid in proposal.beat_ids if bid in valid_beat_ids]
                if len(valid_ids) < 2:
                    log.info(
                        "phase3_insufficient_valid_beats",
                        proposed=proposal.beat_ids,
                        valid=valid_ids,
                    )
                    skipped_count += 1
                    continue

                # Run compatibility check
                errors = check_intersection_compatibility(pre_intersection_graph, valid_ids)
                if errors:
                    log.info(
                        "phase3_incompatible_intersection",
                        beat_ids=valid_ids,
                        errors=[e.issue for e in errors],
                    )
                    structural_errors.extend(errors)
                    skipped_count += 1
                    continue

                # Resolve location (prefer LLM proposal, fallback to algorithm)
                # Guard against qwen3:4b returning the literal string "null".
                raw_loc = proposal.resolved_location
                llm_location: str | None = (
                    None if raw_loc in (None, "null", "NULL", "") else raw_loc
                )
                location: str | None
                if llm_location:
                    location = llm_location
                else:
                    location = resolve_intersection_location(pre_intersection_graph, valid_ids)
                    log.debug(
                        "phase3_location_resolved",
                        beat_ids=valid_ids,
                        resolved=location,
                    )

                accepted.append((valid_ids, location, proposal.shared_entities, proposal.rationale))
                log.debug(
                    "phase3_intersection_accepted",
                    beat_ids=valid_ids,
                    location=location,
                )

            # If some were accepted, break out of retry loop
            if accepted:
                break

            # All rejected -- retry with targeted feedback if structural errors
            # exist and attempts remain. Skip retry when rejections are
            # non-structural (e.g. invalid beat IDs) since feedback would be empty.
            if (
                structural_errors
                and len(result.intersections) > 0
                and structural_attempt < max_structural_retries - 1
            ):
                context["structural_feedback"] = _format_structural_feedback(structural_errors)
                log.info(
                    "phase3_structural_retry",
                    attempt=structural_attempt + 1,
                    errors=len(structural_errors),
                )
                continue

            # Final attempt exhausted or non-structural failures -- fail
            if len(result.intersections) > 0:
                return GrowPhaseResult(
                    phase="intersections",
                    status="failed",
                    detail=(
                        f"All {len(result.intersections)} proposed intersections rejected "
                        f"after {structural_attempt + 1} attempt(s). "
                        f"Story structure lacks cross-dilemma scene overlap. "
                        f"Common causes: insufficient shared locations, isolated storylines, "
                        f"or characters confined to a single dilemma."
                    ),
                    llm_calls=total_llm_calls,
                    tokens_used=total_tokens,
                )

        # R-2.3 / R-2.8: if the graph had strong candidate signals but the LLM
        # returned zero accepted groups, that is a Silent Degradation violation.
        # Two failure modes both require ERROR + halt:
        #   (a) LLM proposed intersections but ALL were rejected (caught above and
        #       returned status="failed", which stage.py escalates to GrowMutationError).
        #   (b) LLM proposed ZERO intersections despite candidates existing.
        # Case (b) is caught here.
        if not accepted and len(candidates) > 0:
            from questfoundry.graph.grow_validation import (
                GrowContractError,  # local to avoid circular import
            )

            log.error(
                "all_intersections_rejected",
                candidate_count=len(candidates),
                proposed_count=len(result.intersections),  # pyright: ignore[reportPossiblyUnboundVariable]  # result bound: loop runs ≥1 and early-returns on error
            )
            raise GrowContractError(
                f"R-2.3 / R-2.8: all intersection candidates rejected — "
                f"{len(candidates)} candidate(s) generated from graph signals, "
                f"{len(result.intersections)} proposed by LLM, 0 accepted. "  # pyright: ignore[reportPossiblyUnboundVariable]
                f"Pipeline failure — halting."
            )

        # Apply accepted intersections in a batch to avoid cascade effects.
        for beat_ids, location, shared_entities, rationale in accepted:
            apply_intersection_mark(
                graph, beat_ids, location, shared_entities=shared_entities, rationale=rationale
            )
            applied_count += 1
            log.debug(
                "phase3_intersection_applied",
                beat_ids=beat_ids,
                location=location,
            )

        # Defensive check: if proposals were made but none accepted after
        # exhausting the loop (shouldn't happen given the checks above,
        # but guards against future logic changes).
        if len(result.intersections) > 0 and not accepted:  # pyright: ignore[reportPossiblyUnboundVariable]  # result bound: loop runs ≥1 and early-returns on error
            return GrowPhaseResult(
                phase="intersections",
                status="failed",
                detail=(f"All {len(result.intersections)} proposed intersections were rejected."),  # pyright: ignore[reportPossiblyUnboundVariable]
                llm_calls=total_llm_calls,
                tokens_used=total_tokens,
            )

        return GrowPhaseResult(
            phase="intersections",
            status="completed",
            detail=(
                f"Proposed {len(result.intersections)} intersections: "  # pyright: ignore[reportPossiblyUnboundVariable]
                f"{applied_count} applied, {skipped_count} skipped"
            ),
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    @grow_phase(name="resolve_temporal_hints", depends_on=["intra_path_predecessors"], priority=2)
    async def _phase_resolve_temporal_hints(
        self, graph: Graph, model: BaseChatModel
    ) -> GrowPhaseResult:
        """Detect and resolve temporal hint ordering conflicts before interleave (#1123, #1140).

        Uses ``build_hint_conflict_graph`` to perform a complete conflict analysis:

        1. Builds a base DAG from all non-hint edges.
        2. Tests each hint alone — mandatory solo drops are applied immediately
           without an LLM call.
        3. Tests surviving hints pairwise for mutual exclusion — swap pairs are
           presented to the LLM for narrative resolution.
        4. Verifies all survivors are acyclic (postcondition) before returning.
           Raises ``TemporalHintResolutionInvariantError`` if the check fails.

        Preconditions:
        - Beat DAG validated (Phase 1 passed).
        - Intersections applied (Phase 3 complete).
        - Beats carry ``temporal_hint`` values from SEED serialization.

        Postconditions:
        - No hint cycles remain: ``interleave_beats`` can apply all surviving
          hints without silently dropping any as cycle-creating.
        - Dropped hints are nulled out on beat nodes.

        Invariants:
        - No-op if no conflicts detected (no LLM call).
        - LLM call only made when swap pairs exist.
        - Does not create any graph edges — only strips hints from nodes.
        """
        from questfoundry.graph.errors import TemporalHintResolutionInvariantError
        from questfoundry.graph.grow_algorithms import (
            build_hint_conflict_graph,
            strip_temporal_hints_by_id,
            verify_hints_acyclic,
        )
        from questfoundry.models.grow import TemporalResolutionOutput

        result = build_hint_conflict_graph(graph)

        if not result.conflicts:
            log.info("resolve_temporal_hints_no_conflicts")
            return GrowPhaseResult(
                phase="resolve_temporal_hints",
                status="completed",
                detail="No temporal hint conflicts detected",
            )

        log.info(
            "resolve_temporal_hints_conflicts_found",
            mandatory_drops=len(result.mandatory_drops),
            swap_pairs=len(result.swap_pairs),
            mandatory_beat_ids=sorted(result.mandatory_drops),
        )

        # Apply mandatory drops immediately — no LLM needed
        beats_to_drop: set[str] = set(result.mandatory_drops)

        total_llm_calls = 0
        total_tokens = 0

        # If there are swap pairs, ask the LLM to choose
        if result.swap_pairs:
            # Build context for each swap pair
            beat_nodes = graph.get_nodes_by_type("beat")
            # Pre-build O(1) lookup for swap conflict by beat pair.
            # Non-mandatory conflicts always have beat_b set (swap pairs).
            swap_conflict_by_pair: dict[frozenset[str], str] = {
                frozenset({c.beat_a, c.beat_b}): c.default_drop
                for c in result.conflicts
                if not c.mandatory and c.beat_b is not None
            }
            swap_pairs_lines: list[str] = []
            for idx, (beat_a_id, beat_b_id) in enumerate(result.swap_pairs, 1):
                group_id = f"P{idx}"
                data_a = beat_nodes.get(beat_a_id, {})
                data_b = beat_nodes.get(beat_b_id, {})
                hint_a = data_a.get("temporal_hint") or {}
                hint_b = data_b.get("temporal_hint") or {}
                pos_a = hint_a.get("position", "unknown")
                rel_a = hint_a.get("relative_to", "unknown")
                pos_b = hint_b.get("position", "unknown")
                rel_b = hint_b.get("relative_to", "unknown")
                strength_a = (
                    "STRONG: after/before_commit"
                    if "commit" in pos_a
                    else "WEAK: after/before_introduce"
                )
                strength_b = (
                    "STRONG: after/before_commit"
                    if "commit" in pos_b
                    else "WEAK: after/before_introduce"
                )
                summary_a = (data_a.get("summary") or "(no summary)")[
                    :_TEMPORAL_HINT_SUMMARY_TRUNCATION
                ]
                summary_b = (data_b.get("summary") or "(no summary)")[
                    :_TEMPORAL_HINT_SUMMARY_TRUNCATION
                ]

                # O(1) lookup for default_drop
                default_drop = swap_conflict_by_pair.get(
                    frozenset({beat_a_id, beat_b_id}), beat_a_id
                )

                swap_pairs_lines.append(
                    f"### Swap Pair {group_id} — DROP EXACTLY ONE:\n"
                    f"  Option A: `{beat_a_id}` | hint: `{pos_a} {rel_a}` [{strength_a}]\n"
                    f"            Summary: {summary_a}\n"
                    f"  Option B: `{beat_b_id}` | hint: `{pos_b} {rel_b}` [{strength_b}]\n"
                    f"            Summary: {summary_b}\n"
                    f"  Mechanical default: drop Option {'A' if default_drop == beat_a_id else 'B'} "
                    f"(based on hint strength + beat role heuristic)"
                )

            swap_pairs_context = "\n\n".join(swap_pairs_lines)
            context: dict[str, str] = {"swap_pairs_context": swap_pairs_context}

            try:
                llm_result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                    model=model,
                    template_name="grow_phase_temporal_resolution",
                    context=context,
                    output_schema=TemporalResolutionOutput,
                    semantic_validator=None,
                )
                total_llm_calls += llm_calls
                total_tokens += tokens
            except GrowStageError as e:
                log.warning(
                    "resolve_temporal_hints_llm_failed_using_defaults",
                    error=str(e),
                    swap_pairs=len(result.swap_pairs),
                )
                # Fall back to mechanical defaults
                for beat_a_id, beat_b_id in result.swap_pairs:
                    default_drop = swap_conflict_by_pair.get(
                        frozenset({beat_a_id, beat_b_id}), beat_a_id
                    )
                    beats_to_drop.add(default_drop)
            else:
                # Validate and apply LLM resolutions
                valid_swap_beats: dict[str, tuple[str, str]] = {
                    f"P{idx}": (a, b) for idx, (a, b) in enumerate(result.swap_pairs, 1)
                }
                for resolution in llm_result.resolutions:
                    pair = valid_swap_beats.get(resolution.group_id)
                    if pair is None:
                        log.info(
                            "resolve_temporal_hints_invalid_group_id",
                            group_id=resolution.group_id,
                        )
                        continue
                    beat_a_id, beat_b_id = pair
                    if resolution.drop_beat_id in (beat_a_id, beat_b_id):
                        beats_to_drop.add(resolution.drop_beat_id)
                    else:
                        log.info(
                            "resolve_temporal_hints_invalid_drop_beat",
                            group_id=resolution.group_id,
                            drop_beat_id=resolution.drop_beat_id,
                            valid_options=f"`{beat_a_id}` or `{beat_b_id}`",
                        )
                        # Fall back to mechanical default
                        beats_to_drop.add(
                            swap_conflict_by_pair.get(frozenset({beat_a_id, beat_b_id}), beat_a_id)
                        )

                # Fill in any missing resolutions with mechanical defaults
                resolved_groups = {r.group_id for r in llm_result.resolutions}
                for idx, (beat_a_id, beat_b_id) in enumerate(result.swap_pairs, 1):
                    group_id = f"P{idx}"
                    if group_id not in resolved_groups:
                        log.info(
                            "resolve_temporal_hints_missing_resolution",
                            group_id=group_id,
                            using_default=True,
                        )
                        beats_to_drop.add(
                            swap_conflict_by_pair.get(frozenset({beat_a_id, beat_b_id}), beat_a_id)
                        )

        # Strip the resolved hints
        stripped = strip_temporal_hints_by_id(graph, beats_to_drop)

        log.info(
            "temporal_hint_conflict_resolved",
            mandatory_drops=len(result.mandatory_drops),
            swap_pairs=len(result.swap_pairs),
            total_hints_dropped=stripped,
            dropped_beats=sorted(beats_to_drop),
        )

        # Postcondition: verify no surviving hints still cycle.
        # NOTE: strip_temporal_hints_by_id has already run at this point.
        # _iter_temporal_hint_edges skips beats with temporal_hint=None, so
        # surviving_beat_ids acts as a cross-check rather than a filter here.
        all_beat_ids_with_hints: set[str] = set()
        for bid, data in graph.get_nodes_by_type("beat").items():
            if data.get("temporal_hint") is not None:
                all_beat_ids_with_hints.add(bid)

        surviving = all_beat_ids_with_hints - beats_to_drop
        still_cyclic = verify_hints_acyclic(graph, surviving)
        if still_cyclic:
            raise TemporalHintResolutionInvariantError(
                still_cyclic=still_cyclic,
                dropped=beats_to_drop,
            )

        return GrowPhaseResult(
            phase="resolve_temporal_hints",
            status="completed",
            detail=(
                f"Resolved {len(result.conflicts)} temporal hint conflict(s): "
                f"dropped {stripped} hint(s) from {sorted(beats_to_drop)}"
            ),
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    @grow_phase(name="scene_types", depends_on=["interleave_beats"], priority=4)
    async def _phase_4a_scene_types(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4a: Tag beats with scene type classification.

        Asks the LLM to classify each beat as scene (active conflict/action),
        sequel (reaction/reflection), or micro_beat (brief transition).

        Preconditions:
        - DAG validation complete.
        - Beat nodes exist with summaries.

        Postconditions:
        - Each beat annotated with scene_type (scene/sequel/micro_beat).
        - Each beat annotated with narrative_function and exit_mood.
        - Invalid beat IDs from LLM output silently skipped.

        Invariants:
        - All beats classified in a single LLM call.
        - Beat summaries truncated to 80 chars in context.
        - Uses compact_items for context budget management.
        """
        from questfoundry.models.grow import Phase4aOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="scene_types",
                status="completed",
                detail="No beats to classify",
            )

        # Build compact beat summaries (drop paths/impacts arrays, truncate)
        beat_items: list[ContextItem] = []
        for bid in sorted(beat_nodes.keys()):
            data = beat_nodes[bid]
            summary = truncate_summary(data.get("summary", ""), 80)
            n_impacts = len(data.get("dilemma_impacts", []))
            line = f'- {bid}: "{summary}" [impacts={n_impacts}]'
            beat_items.append(ContextItem(id=bid, text=line))

        context = {
            "beat_summaries": compact_items(beat_items, self._compact_config()),  # type: ignore[attr-defined]
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
            "beat_count": str(len(beat_nodes)),
        }

        from questfoundry.graph.grow_validators import validate_phase4a_output

        validator = partial(validate_phase4a_output, valid_beat_ids=set(beat_nodes.keys()))
        try:
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase4a_scene_types",
                context=context,
                output_schema=Phase4aOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="scene_types",
                status="failed",
                detail=str(e),
            )

        # Validate and apply tags
        applied = 0
        for tag in result.tags:
            if tag.beat_id not in beat_nodes:
                log.info("phase4a_invalid_beat_id", beat_id=tag.beat_id)
                continue
            graph.update_node(
                tag.beat_id,
                scene_type=tag.scene_type,
                narrative_function=tag.narrative_function,
                exit_mood=tag.exit_mood,
            )
            applied += 1

        return GrowPhaseResult(
            phase="scene_types",
            status="completed",
            detail=f"Tagged {applied}/{len(beat_nodes)} beats with scene types",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    # NOTE: GROW Phase 4b (narrative_gaps) was MOVED to POLISH Phase 1a
    # per the spec migration in PR #1366. The implementation lives in
    # ``polish/llm_phases.py``. The shared gap-insertion helper lives in
    # ``graph/gap_insertion.py``. See issue #1368 for the migration epic.

    # NOTE: GROW Phase 4c (pacing_gaps) was MOVED to POLISH Phase 2 (extended)
    # per the spec migration in PR #1366 / issue #1368 PR C. POLISH's
    # ``_phase_2_pacing`` already detects 3+ consecutive same-scene_type
    # runs and inserts correction beats; per spec R-2.7 those correction
    # beats now carry ``is_gap_beat: True`` to distinguish their origin
    # from regular micro-beats.

    # NOTE: GROW Phase 4d (atmospheric) was MOVED to POLISH Phase 5e
    # per the spec migration in PR #1366 / issue #1368 PR B. The
    # implementation lives in ``polish/llm_phases.py`` as
    # ``_phase_5e_atmospheric``.

    # NOTE: GROW Phase 4e (path_arcs) was MOVED to POLISH Phase 5f
    # per the spec migration in PR #1366 / issue #1368 PR B. The
    # implementation lives in ``polish/llm_phases.py`` as
    # ``_phase_5f_path_thematic``.

    # NOTE: GROW Phase 4f (entity_arcs) was MOVED to POLISH Phase 3 (extended)
    # per the spec migration in PR #1366 / issue #1368 PR C. POLISH's
    # ``_phase_3_character_arcs`` now also produces ``arcs_per_path`` per
    # entity (per spec R-3.6), in the same LLM call that synthesizes
    # start/pivots/end_per_path.

    @grow_phase(name="transition_gaps", depends_on=["interleave_beats"], priority=8)
    async def _phase_transition_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4g: Insert transition beats at hard cross-dilemma seams.

        Detects predecessor edges that cross dilemma boundaries with no shared
        entities or location (hard transitions), then asks the LLM to draft
        1-2 sentence bridge beats for each seam.

        Preconditions:
        - Beats have summaries, entities, and locations.
        - Predecessor edges from interleave_beats are in place.

        Postconditions:
        - Hard-transition seams receive a new ``transition_beat`` beat node.
        - The original predecessor edge is removed and replaced with two edges:
          predecessor(transition, earlier) and predecessor(later, transition).

        Invariants:
        - Only cross-dilemma hard transitions (no shared entity/location) processed.
        - Bridge beats have role="transition_beat", scene_type="micro_beat".
        - No dilemma_impacts on bridge beats (atmospheric only).
        """
        from questfoundry.graph.context import strip_scope_prefix
        from questfoundry.graph.grow_algorithms import detect_cross_dilemma_hard_transitions
        from questfoundry.models.grow import TransitionGapsOutput

        transitions = detect_cross_dilemma_hard_transitions(graph)
        if not transitions:
            return GrowPhaseResult(
                phase="transition_gaps",
                status="completed",
                detail="No hard transitions detected",
            )

        # Build context lines for each transition
        beat_nodes = graph.get_nodes_by_type("beat")
        transition_lines: list[str] = []
        transition_map: dict[str, tuple[str, str]] = {}

        for earlier, later in transitions:
            tid = f"{earlier}|{later}"
            earlier_data = beat_nodes.get(earlier, {})
            later_data = beat_nodes.get(later, {})

            earlier_summary = truncate_summary(earlier_data.get("summary", ""), 80)
            later_summary = truncate_summary(later_data.get("summary", ""), 80)
            earlier_entities = earlier_data.get("entities") or []
            later_entities = later_data.get("entities") or []
            earlier_location = earlier_data.get("location", "")
            later_location = later_data.get("location", "")

            earlier_ents_str = (
                ", ".join(str(e) for e in earlier_entities) if earlier_entities else "none"
            )
            later_ents_str = ", ".join(str(e) for e in later_entities) if later_entities else "none"

            transition_lines.append(
                f"transition_id: {tid}\n"
                f"  FROM: {earlier} — {earlier_summary}\n"
                f"    entities: {earlier_ents_str}\n"
                f"    location: {earlier_location or 'unknown'}\n"
                f"  TO:   {later} — {later_summary}\n"
                f"    entities: {later_ents_str}\n"
                f"    location: {later_location or 'unknown'}"
            )
            transition_map[tid] = (earlier, later)

        # Get genre/tone from vision node
        vision_node = graph.get_node("vision")
        genre = ""
        tone = ""
        if vision_node is not None:
            genre = vision_node.get("genre", "")
            tone_val = vision_node.get("tone")
            if isinstance(tone_val, list):
                tone = ", ".join(str(t) for t in tone_val)
            elif isinstance(tone_val, str):
                tone = tone_val

        context = {
            "transition_count": str(len(transitions)),
            "transitions_context": "\n\n".join(transition_lines),
            "genre": genre or "(not specified)",
            "tone": tone or "(not specified)",
        }

        try:
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase4g_transition_gaps",
                context=context,
                output_schema=TransitionGapsOutput,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="transition_gaps",
                status="failed",
                detail=str(e),
            )

        # Insert bridge beats
        inserted = 0
        for bridge in result.bridges:
            pair = transition_map.get(bridge.transition_id.strip())
            if pair is None:
                log.info(
                    "phase4g_unknown_transition_id",
                    transition_id=bridge.transition_id,
                )
                continue

            earlier, later = pair
            earlier_raw = strip_scope_prefix(earlier)
            later_raw = strip_scope_prefix(later)
            beat_id = f"beat::transition_{earlier_raw}_{later_raw}"

            # Skip if already exists (e.g. from a retry)
            if graph.has_node(beat_id):
                log.info("phase4g_transition_beat_exists", beat_id=beat_id)
                continue

            graph.create_node(
                beat_id,
                {
                    "type": "beat",
                    "raw_id": f"transition_{earlier_raw}_{later_raw}",
                    "summary": bridge.summary,
                    "role": "transition_beat",
                    "scene_type": "micro_beat",
                    "entities": bridge.entities,
                    "location": bridge.location or "",
                    "dilemma_impacts": [],
                },
            )

            # Transition beats have zero belongs_to — they are DAG
            # infrastructure, not part of any dilemma's Y-shape.  Arc
            # traversals reach them by walking the predecessor chain.
            # See Story Graph Ontology Part 3 "Total Order Per Arc"
            # and Part 8 "Zero-belongs_to beats".

            # Replace the old predecessor edge with two new ones
            graph.remove_edge("predecessor", later, earlier)
            graph.add_edge("predecessor", beat_id, earlier)
            graph.add_edge("predecessor", later, beat_id)
            inserted += 1

        if inserted == 0 and transitions:
            log.warning(
                "phase4g_no_bridges_matched",
                transitions_detected=len(transitions),
                bridges_returned=len(result.bridges),
                hint="LLM may have returned transition_ids that don't match the expected format",
            )

        return GrowPhaseResult(
            phase="transition_gaps",
            status="completed",
            detail=f"Inserted {inserted} transition bridge beats from {len(transitions)} hard transitions",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    # -------------------------------------------------------------------------
    # Late LLM phases (state_flags → validation)
    # -------------------------------------------------------------------------

    @grow_phase(name="overlays", depends_on=["state_flags"], priority=16)
    async def _phase_8c_overlays(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 8c: Create cosmetic entity overlays conditioned on state flags.

        For each consequence/state flag pair, proposes entity-level presentation
        changes that activate when those state flags are granted. Overlays are
        cosmetic (how entities appear/behave).

        Preconditions:
        - State flags created (state_flags phase complete).
        - Entity nodes exist with concept, entity_type.
        - Consequence nodes linked to paths and dilemmas.

        Postconditions:
        - Entity nodes gain overlays list with {when: [state_flag_ids], details: {...}}.
        - Overlays modify entity presentation when state flags are granted.
        - Invalid entity/state flag IDs from LLM output silently skipped.

        Invariants:
        - Entity IDs resolved through all category prefixes for robustness.
        - Enriched context traces state flag -> consequence -> path -> dilemma.
        - Overlays appended to existing overlays list (not replaced).
        """
        from questfoundry.models.grow import Phase8cOutput

        state_flag_nodes = graph.get_nodes_by_type("state_flag")
        entity_nodes = graph.get_nodes_by_type("entity")

        if not state_flag_nodes or not entity_nodes:
            return GrowPhaseResult(
                phase="overlays",
                status="completed",
                detail="No state flags or entities to process",
            )

        # Build enriched consequence context per state flag:
        # state flag → consequence → path → dilemma (with central entities + effects)
        consequence_nodes = graph.get_nodes_by_type("consequence")
        valid_state_flag_ids: list[str] = []
        # Maps state_flag_id → dilemma_id for mutual-exclusion validation.
        # Two flags sharing the same dilemma can never both be active.
        flag_to_dilemma: dict[str, str] = {}

        # Collect per-flag data then group by dilemma so the context makes
        # mutual exclusivity explicit to the model.
        # Structure: dilemma_id → list of (sf_id, path_name, cons_desc, effects)
        # Flags with no dilemma are placed in a sentinel group "".
        from collections import defaultdict

        # (sf_id, path_name, cons_desc, effects) — inlined to avoid local alias in type expression
        dilemma_groups: dict[str, list[tuple[str, str, str, list[str]]]] = defaultdict(list)
        dilemma_questions: dict[str, str] = {}
        dilemma_central_entities: dict[str, list[str]] = {}

        for sf_id, sf_data in sorted(state_flag_nodes.items()):
            valid_state_flag_ids.append(sf_id)
            derived_from_id = sf_data.get("derived_from", "")
            cons_data = consequence_nodes.get(derived_from_id, {})
            cons_desc = cons_data.get("description", "unknown consequence")
            narrative_effects: list[str] = cons_data.get("ripples", [])

            # Trace: consequence → path → dilemma for rich context
            path_id = cons_data.get("path_id", "")
            path_node = graph.get_node(path_id) if path_id else None
            dilemma_id = ""
            path_name = path_id
            if path_node:
                path_name = path_node.get("name", path_id)
                dilemma_id = path_node.get("dilemma_id", "")
                if dilemma_id:
                    flag_to_dilemma[sf_id] = dilemma_id
                    if dilemma_id not in dilemma_questions:
                        dilemma_node = graph.get_node(dilemma_id)
                        if dilemma_node:
                            dilemma_questions[dilemma_id] = dilemma_node.get("question", "")
                            anchored = graph.get_edges(from_id=dilemma_id, edge_type="anchored_to")
                            dilemma_central_entities[dilemma_id] = (
                                [strip_scope_prefix(e["to"]) for e in anchored] if anchored else []
                            )

            dilemma_groups[dilemma_id].append((sf_id, path_name, cons_desc, narrative_effects))

        # Emit grouped context: one block per dilemma, flags listed inside
        consequence_lines: list[str] = []
        for dilemma_id, entries in sorted(dilemma_groups.items()):
            if dilemma_id:
                question = dilemma_questions.get(dilemma_id, dilemma_id)
                central = dilemma_central_entities.get(dilemma_id, [])
                central_str = f" — central entities: {', '.join(central)}" if central else ""
                consequence_lines.append(
                    f'DILEMMA GROUP "{question}"{central_str}'
                    f" (flags are mutually exclusive — use only ONE per overlay):"
                )
            else:
                consequence_lines.append("UNGROUPED FLAGS (no dilemma):")

            for sf_id, path_name, cons_desc, effects in entries:
                consequence_lines.append(f'  - {sf_id}  →  Path "{path_name}"')
                consequence_lines.append(f"    Consequence: {cons_desc}")
                if effects:
                    for effect in effects:
                        consequence_lines.append(f"    Effect: {effect}")

            consequence_lines.append("")  # blank line between groups

        consequence_context = "\n".join(consequence_lines).rstrip()

        # Build entity context: entity details for overlay candidates
        entity_lines: list[str] = []
        valid_entity_ids: list[str] = []

        for ent_id, ent_data in sorted(entity_nodes.items()):
            valid_entity_ids.append(ent_id)
            category = ent_data.get("entity_category", ent_data.get("entity_type", "unknown"))
            concept = ent_data.get("concept", "")
            entity_lines.append(f"- {ent_id} ({category}): {concept}")

        entity_context = "\n".join(entity_lines)

        context = {
            "consequence_context": consequence_context,
            "entity_context": entity_context,
            "valid_entity_ids": ", ".join(valid_entity_ids),
            "valid_state_flag_ids": ", ".join(valid_state_flag_ids),
        }

        from questfoundry.graph.grow_validators import validate_phase8c_output

        validator = partial(
            validate_phase8c_output,
            valid_entity_ids=set(valid_entity_ids),
            valid_state_flag_ids=set(valid_state_flag_ids),
            flag_to_dilemma=flag_to_dilemma,
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model,
                "grow_phase8c_overlays",
                context,
                Phase8cOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(phase="overlays", status="failed", detail=str(e))

        # Validate and apply overlays
        valid_entity_set = set(valid_entity_ids)
        valid_state_flag_set = set(valid_state_flag_ids)
        overlay_count = 0

        for overlay in result.overlays:
            # Validate entity_id exists - try resolving through category prefixes
            prefixed_eid = overlay.entity_id
            if "::" not in overlay.entity_id:
                # Try all category prefixes to find a match
                raw_id = overlay.entity_id
                found = False
                for category in ENTITY_CATEGORIES:
                    candidate = f"{category}::{raw_id}"
                    if candidate in valid_entity_set:
                        prefixed_eid = candidate
                        found = True
                        break
                # Try legacy entity:: prefix for backwards compatibility
                if not found:
                    legacy = f"entity::{raw_id}"
                    if legacy in valid_entity_set:
                        prefixed_eid = legacy
                        found = True
                if not found:
                    log.info(
                        "phase8c_invalid_entity",
                        entity_id=overlay.entity_id,
                        tried_categories=list(ENTITY_CATEGORIES),
                    )
                    continue
            elif prefixed_eid not in valid_entity_set:
                # Already prefixed but not found
                log.info(
                    "phase8c_invalid_entity",
                    entity_id=overlay.entity_id,
                    prefixed=prefixed_eid,
                )
                continue

            # Validate all state flag IDs in 'when' exist
            invalid_flags = [cw for cw in overlay.when if cw not in valid_state_flag_set]
            if invalid_flags:
                log.info(
                    "phase8c_invalid_state_flags",
                    entity_id=overlay.entity_id,
                    invalid=invalid_flags,
                )
                continue

            # Store overlay on entity node
            entity_data = graph.get_node(prefixed_eid)
            if entity_data is None:
                log.error("phase8c_entity_disappeared", entity_id=prefixed_eid)
                continue

            existing_overlays: list[dict[str, Any]] = entity_data.get("overlays", [])
            existing_overlays.append(
                {
                    "when": overlay.when,
                    "details": overlay.details_as_dict(),
                }
            )
            graph.update_node(prefixed_eid, overlays=existing_overlays)
            overlay_count += 1

        return GrowPhaseResult(
            phase="overlays",
            status="completed",
            detail=f"Created {overlay_count} overlays",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )
