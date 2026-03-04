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
    build_narrative_frame,
    compact_items,
    enrich_beat_line,
    truncate_summary,
)
from questfoundry.graph.graph import Graph
from questfoundry.models.grow import GrowPhaseResult
from questfoundry.pipeline.batching import batch_llm_calls
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


def _build_path_dilemma_context(
    graph: Graph,
    path_nodes: dict[str, Any],
) -> tuple[str, str]:
    """Build path-to-dilemma mapping text and valid dilemma ID text for LLM context.

    Args:
        graph: The graph store to query for dilemma nodes.
        path_nodes: Dict of path_id → path data.

    Returns:
        A tuple of (path_dilemma_map_text, valid_dilemma_ids_text) ready for
        injection into a prompt context dict.
    """
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    valid_dilemma_ids = sorted(dilemma_nodes.keys())
    path_dilemma_lines = []
    for pid in sorted(path_nodes.keys()):
        pdata = path_nodes[pid]
        dilemma_id = pdata.get("dilemma_id", "")
        if dilemma_id and not dilemma_id.startswith("dilemma::"):
            dilemma_id = f"dilemma::{dilemma_id}"
        question = ""
        if dilemma_id and dilemma_id in dilemma_nodes:
            question = dilemma_nodes[dilemma_id].get("question", "")
        suffix = f' ("{question}")' if question else ""
        path_dilemma_lines.append(f"  {pid} → {dilemma_id or '(none)'}{suffix}")
    path_dilemma_map_text = "\n".join(path_dilemma_lines) or "(no paths with dilemmas)"
    valid_dilemma_ids_text = ", ".join(valid_dilemma_ids) or "(none)"
    return path_dilemma_map_text, valid_dilemma_ids_text


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

            # Validate and apply intersections
            applied_count = 0
            skipped_count = 0
            pre_intersection_graph = Graph.from_dict(graph.to_dict())
            accepted: list[tuple[list[str], str | None, list[str], str]] = []
            structural_errors: list[GrowValidationError] = []

            for proposal in result.intersections:
                # Filter to valid beat IDs
                valid_ids = [bid for bid in proposal.beat_ids if bid in valid_beat_ids]
                if len(valid_ids) < 2:
                    log.warning(
                        "phase3_insufficient_valid_beats",
                        proposed=proposal.beat_ids,
                        valid=valid_ids,
                    )
                    skipped_count += 1
                    continue

                # Run compatibility check
                errors = check_intersection_compatibility(pre_intersection_graph, valid_ids)
                if errors:
                    log.warning(
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
                log.warning(
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
        if len(result.intersections) > 0 and not accepted:
            return GrowPhaseResult(
                phase="intersections",
                status="failed",
                detail=(f"All {len(result.intersections)} proposed intersections were rejected."),
                llm_calls=total_llm_calls,
                tokens_used=total_tokens,
            )

        return GrowPhaseResult(
            phase="intersections",
            status="completed",
            detail=(
                f"Proposed {len(result.intersections)} intersections: "
                f"{applied_count} applied, {skipped_count} skipped"
            ),
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    @grow_phase(name="resolve_temporal_hints", depends_on=["intersections"], priority=2)
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
                    for c in result.conflicts:
                        if not c.mandatory and {c.beat_a, c.beat_b} == {beat_a_id, beat_b_id}:
                            beats_to_drop.add(c.default_drop)
                            break
            else:
                # Validate and apply LLM resolutions
                valid_swap_beats: dict[str, tuple[str, str]] = {
                    f"P{idx}": (a, b) for idx, (a, b) in enumerate(result.swap_pairs, 1)
                }
                for resolution in llm_result.resolutions:
                    pair = valid_swap_beats.get(resolution.group_id)
                    if pair is None:
                        log.warning(
                            "resolve_temporal_hints_invalid_group_id",
                            group_id=resolution.group_id,
                        )
                        continue
                    beat_a_id, beat_b_id = pair
                    if resolution.drop_beat_id in (beat_a_id, beat_b_id):
                        beats_to_drop.add(resolution.drop_beat_id)
                    else:
                        log.warning(
                            "resolve_temporal_hints_invalid_drop_beat",
                            group_id=resolution.group_id,
                            drop_beat_id=resolution.drop_beat_id,
                            valid_options=f"`{beat_a_id}` or `{beat_b_id}`",
                        )
                        # Fall back to mechanical default
                        for c in result.conflicts:
                            if not c.mandatory and {c.beat_a, c.beat_b} == {beat_a_id, beat_b_id}:
                                beats_to_drop.add(c.default_drop)
                                break

                # Fill in any missing resolutions with mechanical defaults
                resolved_groups = {r.group_id for r in llm_result.resolutions}
                for idx, (beat_a_id, beat_b_id) in enumerate(result.swap_pairs, 1):
                    group_id = f"P{idx}"
                    if group_id not in resolved_groups:
                        log.warning(
                            "resolve_temporal_hints_missing_resolution",
                            group_id=group_id,
                            using_default=True,
                        )
                        for c in result.conflicts:
                            if not c.mandatory and {c.beat_a, c.beat_b} == {beat_a_id, beat_b_id}:
                                beats_to_drop.add(c.default_drop)
                                break

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
                log.warning("phase4a_invalid_beat_id", beat_id=tag.beat_id)
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

    @grow_phase(name="narrative_gaps", depends_on=["scene_types"], priority=4)
    async def _phase_4b_narrative_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4b: Detect narrative gaps in path beat sequences.

        For each path, traces the beat sequence and asks the LLM
        to identify missing beats (e.g., a path jumps from setup
        to climax without a development beat).

        Preconditions:
        - Scene types assigned (Phase 4a complete).
        - Paths have 2+ beats with requires-based ordering.

        Postconditions:
        - Gap beats inserted into the graph with requires edges.
        - New beats have belongs_to edges linking them to their path.
        - Beat summaries provided by LLM for each gap.

        Invariants:
        - Only paths with 2+ beats assessed.
        - Inserted beats placed between valid before/after beat pairs.
        - Invalid proposals (bad IDs, wrong ordering) silently rejected.
        """
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence
        from questfoundry.models.grow import Phase4bOutput

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="completed",
                detail="No paths to check for gaps",
            )

        # Build path sequences with truncated summaries
        path_sequences: list[str] = []
        valid_beat_ids: set[str] = set()
        for pid in sorted(path_nodes.keys()):
            sequence = get_path_beat_sequence(graph, pid)
            if len(sequence) < 2:
                continue
            beat_list: list[str] = []
            for bid in sequence:
                node = graph.get_node(bid)
                summary = truncate_summary(node.get("summary", ""), 80) if node else ""
                scene_type = node.get("scene_type", "untagged") if node else "untagged"
                beat_list.append(f"    {bid} [{scene_type}]: {summary}")
                valid_beat_ids.add(bid)
            raw_pid = path_nodes[pid].get("raw_id", pid)
            path_sequences.append(f"  Path: {raw_pid} ({pid})\n" + "\n".join(beat_list))

        if not path_sequences:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="completed",
                detail="No paths with 2+ beats to check",
            )

        path_dilemma_map_text, valid_dilemma_ids_text = _build_path_dilemma_context(
            graph, path_nodes
        )

        context = {
            "path_sequences": "\n\n".join(path_sequences),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(valid_beat_ids)),
            "path_dilemma_map": path_dilemma_map_text,
            "valid_dilemma_ids": valid_dilemma_ids_text,
        }

        from questfoundry.graph.grow_validators import validate_phase4_output

        validator = partial(
            validate_phase4_output,
            valid_path_ids=set(path_nodes.keys()),
            valid_beat_ids=valid_beat_ids,
            graph=graph,
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase4b_narrative_gaps",
                context=context,
                output_schema=Phase4bOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="failed",
                detail=str(e),
            )

        # Validate and insert gap beats
        report = self._validate_and_insert_gaps(  # type: ignore[attr-defined]
            graph, result.gaps, path_nodes, valid_beat_ids, "phase4b"
        )

        return GrowPhaseResult(
            phase="narrative_gaps",
            status="completed",
            detail=f"Inserted {report.inserted} gap beats from {len(result.gaps)} proposals",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    @grow_phase(name="pacing_gaps", depends_on=["narrative_gaps"], priority=5)
    async def _phase_4c_pacing_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4c: Detect and fix pacing issues (3+ same scene_type in a row).

        Runs deterministic pacing detection first, then asks the LLM
        to propose correction beats for any violations found.

        Preconditions:
        - Narrative gaps resolved (Phase 4b complete).
        - Beats have scene_type tags from Phase 4a.

        Postconditions:
        - Pacing violations (3+ consecutive same scene_type) corrected.
        - Correction beats inserted to break monotonous sequences.

        Invariants:
        - Skipped if no scene_type tags found (Phase 4a did not run).
        - Only affected paths included in LLM context.
        - Deterministic pacing detection precedes LLM correction.
        """
        from questfoundry.graph.grow_algorithms import (
            detect_pacing_issues,
            get_path_beat_sequence,
        )
        from questfoundry.models.grow import Phase4bOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="completed",
                detail="No beats to check for pacing",
            )

        # Check if scene types have been assigned
        has_scene_types = any(b.get("scene_type") for b in beat_nodes.values())
        if not has_scene_types:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="skipped",
                detail="No scene_type tags found (Phase 4a may not have run)",
            )

        issues = detect_pacing_issues(graph)
        if not issues:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="completed",
                detail="No pacing issues detected",
            )

        # Build path sequences for affected paths with truncated summaries
        path_nodes = graph.get_nodes_by_type("path")
        affected_pids = {issue.path_id for issue in issues}
        path_sequences: list[str] = []
        valid_beat_ids: set[str] = set()
        for pid in sorted(affected_pids):
            sequence = get_path_beat_sequence(graph, pid)
            if len(sequence) < 2:
                continue
            beat_list: list[str] = []
            for idx, bid in enumerate(sequence, 1):
                node = graph.get_node(bid)
                summary = truncate_summary(node.get("summary", ""), 80) if node else ""
                scene_type = node.get("scene_type", "untagged") if node else "untagged"
                beat_list.append(f"    #{idx} {bid} [{scene_type}]: {summary}")
                valid_beat_ids.add(bid)
            raw_pid = pid.removeprefix("path::")
            path_sequences.append(f"  Path: {raw_pid} ({pid})\n" + "\n".join(beat_list))

        # Build issue descriptions with truncated summaries
        issue_descriptions: list[str] = []
        for issue in issues:
            issue_beats: list[str] = []
            for bid in issue.beat_ids:
                node = graph.get_node(bid)
                summary = truncate_summary(node.get("summary", ""), 80) if node else ""
                issue_beats.append(f"    {bid}: {summary}")
            raw_pid = issue.path_id.removeprefix("path::")
            issue_descriptions.append(
                f"  Path {raw_pid}: {len(issue.beat_ids)} consecutive "
                f"'{issue.scene_type}' beats:\n" + "\n".join(issue_beats)
            )

        path_dilemma_map_text_4c, valid_dilemma_ids_text_4c = _build_path_dilemma_context(
            graph, path_nodes
        )

        context = {
            "path_sequences": "\n\n".join(path_sequences),
            "pacing_issues": "\n\n".join(issue_descriptions),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(valid_beat_ids)),
            "issue_count": str(len(issues)),
            "path_dilemma_map": path_dilemma_map_text_4c,
            "valid_dilemma_ids": valid_dilemma_ids_text_4c,
        }

        from questfoundry.graph.grow_validators import validate_phase4_output

        validator = partial(
            validate_phase4_output,
            valid_path_ids=set(path_nodes.keys()),
            valid_beat_ids=valid_beat_ids,
            graph=graph,
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase4c_pacing_gaps",
                context=context,
                output_schema=Phase4bOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="failed",
                detail=str(e),
            )

        # Insert correction beats
        report = self._validate_and_insert_gaps(  # type: ignore[attr-defined]
            graph, result.gaps, path_nodes, valid_beat_ids, "phase4c"
        )
        if report.total_invalid > 0:
            log.warning(
                "phase4c_invalid_gap_proposals",
                invalid=report.total_invalid,
                invalid_before=report.invalid_before_beat,
                invalid_after=report.invalid_after_beat,
                invalid_path=report.invalid_path_id,
                invalid_order=report.invalid_beat_order,
                not_in_sequence=report.beat_not_in_sequence,
            )

        return GrowPhaseResult(
            phase="pacing_gaps",
            status="completed",
            detail=(
                f"Found {len(issues)} pacing issues, inserted {report.inserted} correction beats"
            ),
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    @grow_phase(name="atmospheric", depends_on=["pacing_gaps"], priority=6)
    async def _phase_4d_atmospheric(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4d: Atmospheric detail for beats.

        Generates sensory environment details for all beats.

        Preconditions:
        - Pacing gaps resolved (Phase 4c complete).
        - Beat nodes have summaries.

        Postconditions:
        - All beats annotated with atmospheric_detail (sensory environment).

        Invariants:
        - Single LLM call for all beats.
        - Narrative frame built from dilemma questions and path themes.
        """
        from questfoundry.models.grow import Phase4dOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="atmospheric",
                status="completed",
                detail="No beats to annotate",
            )

        # Build enriched beat summaries with entity names
        beat_items: list[ContextItem] = []
        for bid in sorted(beat_nodes.keys()):
            data = beat_nodes[bid]
            line = enrich_beat_line(graph, bid, data, include_entities=True)
            beat_items.append(ContextItem(id=bid, text=line))

        # Build narrative frame from dilemmas and paths
        dilemma_ids = sorted(graph.get_nodes_by_type("dilemma").keys())
        path_ids = sorted(graph.get_nodes_by_type("path").keys())
        narrative_frame = build_narrative_frame(graph, dilemma_ids=dilemma_ids, path_ids=path_ids)

        context = {
            "narrative_frame": narrative_frame,
            "beat_summaries": compact_items(beat_items, self._compact_config()),  # type: ignore[attr-defined]
            "beat_count": str(len(beat_nodes)),
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
        }

        try:
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase4d_atmospheric",
                context=context,
                output_schema=Phase4dOutput,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="atmospheric",
                status="failed",
                detail=str(e),
            )

        # Apply atmospheric details
        applied_details = 0
        for detail in result.details:
            if detail.beat_id not in beat_nodes:
                log.warning("phase4d_invalid_beat_id", beat_id=detail.beat_id)
                continue
            graph.update_node(
                detail.beat_id,
                atmospheric_detail=detail.atmospheric_detail,
            )
            applied_details += 1

        return GrowPhaseResult(
            phase="atmospheric",
            status="completed",
            detail=f"Applied atmospheric details to {applied_details}/{len(beat_nodes)} beats",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    @grow_phase(name="path_arcs", depends_on=["atmospheric"], priority=7)
    async def _phase_4e_path_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4e: Per-path thematic mini-arcs.

        Generates a thematic through-line and mood descriptor for each path.

        Preconditions:
        - Atmospheric details assigned (Phase 4d complete).
        - Each path has beats with scene_type and narrative_function.
        - Entity nodes have concept and entity_type fields.

        Postconditions:
        - Each path annotated with path_theme and path_mood.
        - Themes and moods derived from beat sequence and entity context.

        Invariants:
        - One LLM call per path via batch_llm_calls.
        - Dilemma question and stakes included in per-path context.
        - Entity arcs from path node included for thematic coherence.
        """
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence
        from questfoundry.models.grow import PathMiniArc

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            return GrowPhaseResult(
                phase="path_arcs",
                status="completed",
                detail="No paths to annotate",
            )

        applied = 0

        # Pre-compute context for each path (graph reads are not async)
        path_items: list[tuple[str, dict[str, str]]] = []
        for pid in sorted(path_nodes.keys()):
            pdata = path_nodes[pid]
            dilemma_id = pdata.get("dilemma_id", "")
            dilemma_node = graph.get_node(dilemma_id) if dilemma_id else None
            dilemma_question = dilemma_node.get("question", "") if dilemma_node else ""
            dilemma_stakes = dilemma_node.get("why_it_matters", "") if dilemma_node else ""
            path_description = pdata.get("description", "")

            try:
                beat_ids = get_path_beat_sequence(graph, pid)
            except ValueError:
                log.warning("phase4e_cycle_in_path", path_id=pid)
                continue

            if not beat_ids:
                log.warning("phase4e_no_beats_for_path", path_id=pid)
                continue

            # Collect entity IDs from all beats in this path
            beat_entity_ids: set[str] = set()
            beat_lines: list[str] = []
            for i, bid in enumerate(beat_ids, 1):
                bdata = graph.get_node(bid)
                if not bdata:
                    continue
                summary = bdata.get("summary", "")
                narrative_fn = bdata.get("narrative_function", "")
                scene_type = bdata.get("scene_type", "")
                beat_lines.append(
                    f"{i}. {bid}: {summary} [function={narrative_fn}, scene_type={scene_type}]"
                )
                for eid in bdata.get("entities", []):
                    beat_entity_ids.add(eid)

            # Format entity context (name + concept for all entities)
            entity_lines: list[str] = []
            for eid in sorted(beat_entity_ids):
                enode = graph.get_node(eid)
                if enode:
                    name = enode.get("name") or enode.get("raw_id", eid)
                    concept = enode.get("concept", "")
                    entity_lines.append(
                        f"- {name}: {concept}" if concept else f"- {name}: (no concept yet)"
                    )
                else:
                    entity_lines.append(f"- {eid}: (not in graph)")

            # Format entity arcs from path node (subset: entity_id + arc_line only;
            # pivot_beat and arc_type are not needed for thematic context)
            arc_lines: list[str] = []
            for arc in pdata.get("entity_arcs", []):
                arc_entity = arc.get("entity_id", "")
                arc_line = arc.get("arc_line", "")
                if arc_entity and arc_line:
                    ename = arc_entity
                    enode = graph.get_node(arc_entity)
                    if enode:
                        ename = enode.get("name") or enode.get("raw_id", arc_entity)
                    arc_lines.append(f"- {ename}: {arc_line}")

            context = {
                "path_id": pid,
                "dilemma_question": dilemma_question or "(none)",
                "dilemma_stakes": dilemma_stakes or "(none)",
                "path_description": path_description or "(none)",
                "entity_context": "\n".join(entity_lines) if entity_lines else "(none)",
                "entity_arcs": "\n".join(arc_lines) if arc_lines else "(none)",
                "beat_sequence": "\n".join(beat_lines) if beat_lines else "(none)",
            }
            path_items.append((pid, context))

        async def _arc_for_path(
            item: tuple[str, dict[str, str]],
        ) -> tuple[tuple[str, PathMiniArc], int, int]:
            pid, ctx = item
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase4e_path_arcs",
                context=ctx,
                output_schema=PathMiniArc,
            )
            return (pid, result), llm_calls, tokens

        results, total_llm_calls, total_tokens, errors = await batch_llm_calls(
            path_items,
            _arc_for_path,
            self._max_concurrency,  # type: ignore[attr-defined]
            on_connectivity_error=self._on_connectivity_error,  # type: ignore[attr-defined]
        )

        for item in results:
            if item is None:
                continue
            pid, result = item
            graph.update_node(
                pid,
                path_theme=result.path_theme,
                path_mood=result.path_mood,
            )
            applied += 1

        if errors:
            for idx, e in errors:
                pid = path_items[idx][0]
                log.warning("phase4e_llm_failed", path_id=pid, error=str(e))

        return GrowPhaseResult(
            phase="path_arcs",
            status="completed",
            detail=f"Applied path arcs to {applied}/{len(path_nodes)} paths",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    @grow_phase(name="entity_arcs", depends_on=["interleave_beats"], priority=8)
    async def _phase_4f_entity_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4f: Per-entity arc trajectories on each path.

        Runs post-interleave so beat topology (including cross-path predecessor
        edges) is final. For each path, selects eligible entities
        (deterministic), then asks the LLM to generate arc_line and pivot_beat
        per entity.

        Preconditions:
        - Interleave complete, predecessor edges exist for path sequencing.
        - Entity nodes have entity_type and concept fields.
        - Path nodes have beat sequences via belongs_to + predecessor.

        Postconditions:
        - Each path annotated with entity_arcs list.
        - Each entity arc has entity_id, arc_type, arc_line, pivot_beat.
        - arc_type derived deterministically from entity category.

        Invariants:
        - One LLM call per path via batch_llm_calls.
        - Only eligible entities (2+ beat appearances) included.
        - Pivot beat on shared beat triggers a warning but is allowed.
        """
        from functools import partial as partial_fn

        from questfoundry.graph.grow_algorithms import (
            ARC_TYPE_BY_ENTITY_TYPE,
            get_path_beat_sequence,
            select_entities_for_arc,
        )
        from questfoundry.graph.grow_validators import validate_phase4f_output
        from questfoundry.models.grow import Phase4fOutput

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            return GrowPhaseResult(
                phase="entity_arcs",
                status="completed",
                detail="No paths to annotate",
            )

        applied = 0

        # Pre-compute context for each path
        path_items: list[tuple[str, dict[str, str], set[str], set[str]]] = []
        for pid in sorted(path_nodes.keys()):
            pdata = path_nodes[pid]
            dilemma_id = pdata.get("dilemma_id", "")
            dilemma_node = graph.get_node(dilemma_id) if dilemma_id else None
            dilemma_question = dilemma_node.get("question", "") if dilemma_node else ""

            try:
                beat_ids = get_path_beat_sequence(graph, pid)
            except ValueError:
                log.warning("phase4f_cycle_in_path", path_id=pid)
                continue

            if not beat_ids:
                log.warning("phase4f_no_beats_for_path", path_id=pid)
                continue

            eligible = select_entities_for_arc(graph, pid, beat_ids)
            if not eligible:
                log.debug("phase4f_no_eligible_entities", path_id=pid)
                continue

            # Build beat sequence lines (same format as 4e)
            beat_lines: list[str] = []
            for i, bid in enumerate(beat_ids, 1):
                bdata = graph.get_node(bid)
                if not bdata:
                    continue
                summary = bdata.get("summary", "")
                entities = bdata.get("entities", [])
                entities_str = ", ".join(entities) if entities else "none"
                beat_lines.append(f"{i}. {bid}: {summary} [entities: {entities_str}]")

            # Build entity list with concept for context
            entity_lines: list[str] = []
            for eid in eligible:
                edata = graph.get_node(eid)
                concept = edata.get("concept", "") if edata else ""
                etype = edata.get("entity_type", "character") if edata else "character"
                entity_lines.append(f"- {eid} ({etype}): {concept}")

            # Valid IDs section
            valid_ids = (
                f"### Entity IDs (generate arc for EACH)\n"
                f"{chr(10).join(f'- `{eid}`' for eid in eligible)}\n\n"
                f"### Path ID\n"
                f"This path: `{pid}`\n\n"
                f"### Beat IDs on This Path (use ONLY these for pivot_beat)\n"
                f"{chr(10).join(f'- `{bid}`' for bid in beat_ids)}\n"
                f"Total beats: {len(beat_ids)}"
            )

            context = {
                "path_id": pid,
                "dilemma_question": dilemma_question or "(no dilemma question)",
                "beat_sequence": "\n".join(beat_lines) if beat_lines else "(no beats)",
                "entity_list": "\n".join(entity_lines),
                "entity_count": str(len(eligible)),
                "valid_ids_section": valid_ids,
            }
            valid_entity_set = set(eligible)
            valid_beat_set = set(beat_ids)
            path_items.append((pid, context, valid_entity_set, valid_beat_set))

        async def _arcs_for_path(
            item: tuple[str, dict[str, str], set[str], set[str]],
        ) -> tuple[tuple[str, Phase4fOutput], int, int]:
            pid, ctx, valid_eids, valid_bids = item
            validator = partial_fn(
                validate_phase4f_output,
                valid_entity_ids=valid_eids,
                valid_beat_ids=valid_bids,
            )
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase4f_entity_arcs",
                context=ctx,
                output_schema=Phase4fOutput,
                semantic_validator=validator,
            )
            return (pid, result), llm_calls, tokens

        results, total_llm_calls, total_tokens, errors = await batch_llm_calls(
            path_items,
            _arcs_for_path,
            self._max_concurrency,  # type: ignore[attr-defined]
            on_connectivity_error=self._on_connectivity_error,  # type: ignore[attr-defined]
        )

        for item in results:
            if item is None:
                continue
            pid, result = item

            # Build entity_arcs with computed arc_type
            entity_arcs: list[dict[str, str]] = []
            for arc in result.arcs:
                edata = graph.get_node(arc.entity_id)
                if edata is None:
                    log.error("phase4f_missing_entity", entity_id=arc.entity_id, path_id=pid)
                    continue
                etype = edata.get("entity_type", "character")
                arc_type = ARC_TYPE_BY_ENTITY_TYPE.get(etype, "transformation")

                # Warn if pivot is on a shared beat (belongs to multiple paths)
                pivot_paths = graph.get_edges(from_id=arc.pivot_beat, edge_type="belongs_to")
                if len(pivot_paths) > 1:
                    log.warning(
                        "shared_pivot_beat",
                        entity_id=arc.entity_id,
                        pivot_beat=arc.pivot_beat,
                        path_id=pid,
                    )

                entity_arcs.append(
                    {
                        "entity_id": arc.entity_id,
                        "arc_type": arc_type,
                        "arc_line": arc.arc_line,
                        "pivot_beat": arc.pivot_beat,
                    }
                )

            graph.update_node(pid, entity_arcs=entity_arcs)
            applied += 1

        if errors:
            for idx, e in errors:
                pid = path_items[idx][0]
                log.warning("phase4f_llm_failed", path_id=pid, error=str(e))

        return GrowPhaseResult(
            phase="entity_arcs",
            status="completed",
            detail=f"Applied entity arcs to {applied}/{len(path_nodes)} paths",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
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

        FlagEntry = tuple[str, str, str, list[str]]  # (sf_id, path_name, cons_desc, effects)
        dilemma_groups: dict[str, list[FlagEntry]] = defaultdict(list)
        dilemma_questions: dict[str, str] = {}
        dilemma_central_entities: dict[str, list[str]] = {}

        for sf_id, sf_data in sorted(state_flag_nodes.items()):
            valid_state_flag_ids.append(sf_id)
            derived_from_id = sf_data.get("derived_from", "")
            cons_data = consequence_nodes.get(derived_from_id, {})
            cons_desc = cons_data.get("description", "unknown consequence")
            narrative_effects: list[str] = cons_data.get("narrative_effects", [])

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
                    log.warning(
                        "phase8c_invalid_entity",
                        entity_id=overlay.entity_id,
                        tried_categories=list(ENTITY_CATEGORIES),
                    )
                    continue
            elif prefixed_eid not in valid_entity_set:
                # Already prefixed but not found
                log.warning(
                    "phase8c_invalid_entity",
                    entity_id=overlay.entity_id,
                    prefixed=prefixed_eid,
                )
                continue

            # Validate all state flag IDs in 'when' exist
            invalid_flags = [cw for cw in overlay.when if cw not in valid_state_flag_set]
            if invalid_flags:
                log.warning(
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
