"""LLM-powered phase implementations for the GROW stage.

Contains _LLMPhaseMixin with all phases that require LLM calls:
phases 2, 3, 4a-4f (early phases).

Later phases (8c, 9, 9b, 9c) will be added in a subsequent PR.

GrowStage inherits this mixin so ``execute()`` can delegate to
``self._phase_2_path_agnostic()``, etc.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.mutations import GrowValidationError


class _LLMPhaseMixin:
    """Mixin providing LLM-powered GROW phases (2, 3, 4a-4f).

    Expects the host class to provide (via ``_LLMHelperMixin`` or directly):

    - ``_grow_llm_call()``
    - ``_validate_and_insert_gaps()``
    - ``_compact_config()``
    - ``_max_concurrency``
    - ``_on_connectivity_error``
    """

    async def _phase_2_path_agnostic(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 2: Path-agnostic assessment.

        Identifies beats whose prose is compatible across multiple paths
        of the same dilemma. Path-agnostic beats don't need separate
        renderings per path -- they read the same regardless of path.

        This is about prose compatibility, not logical compatibility.
        A beat is path-agnostic if its narrative content doesn't reference
        path-specific choices or consequences.
        """
        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output

        # Collect dilemmas with multiple paths
        dilemma_nodes = graph.get_nodes_by_type("dilemma")
        path_nodes = graph.get_nodes_by_type("path")
        beat_nodes = graph.get_nodes_by_type("beat")

        if not dilemma_nodes or not path_nodes or not beat_nodes:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="completed",
                detail="No dilemmas/paths/beats to assess",
            )

        # Build dilemma -> paths mapping from path node dilemma_id properties
        from questfoundry.graph.grow_algorithms import build_dilemma_paths

        dilemma_paths = build_dilemma_paths(graph)

        # Only assess dilemmas with multiple paths
        multi_path_dilemmas = {did: paths for did, paths in dilemma_paths.items() if len(paths) > 1}

        if not multi_path_dilemmas:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="completed",
                detail="No multi-path dilemmas to assess",
            )

        # Build beat -> paths mapping via belongs_to edges
        beat_path_map: dict[str, list[str]] = {}
        belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
        for edge in belongs_to_edges:
            beat_id = edge["from"]
            path_id = edge["to"]
            beat_path_map.setdefault(beat_id, []).append(path_id)

        # Find beats that belong to multiple paths of the same dilemma
        # These are candidates for path-agnostic assessment
        candidate_beats: dict[str, list[str]] = {}  # beat_id -> list of dilemma_ids
        for beat_id, beat_paths in beat_path_map.items():
            if beat_id not in beat_nodes:
                continue
            for dilemma_id, dilemma_path_list in multi_path_dilemmas.items():
                # Count how many of this dilemma's paths the beat belongs to
                shared = [p for p in beat_paths if p in dilemma_path_list]
                if len(shared) > 1:
                    candidate_beats.setdefault(beat_id, []).append(dilemma_id)

        if not candidate_beats:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="completed",
                detail="No candidate beats for path-agnostic assessment",
            )

        # Build context for LLM
        beat_summaries: list[str] = []
        valid_beat_ids: list[str] = []
        valid_dilemma_ids: list[str] = []

        for beat_id, dilemma_id_list in sorted(candidate_beats.items()):
            beat_data = beat_nodes[beat_id]
            summary = beat_data.get("summary", "No summary")
            dilemmas_str = ", ".join(
                dilemma_nodes[did].get("raw_id", did) for did in dilemma_id_list
            )
            beat_summaries.append(
                f"- beat_id: {beat_id}\n  summary: {summary}\n  dilemmas: [{dilemmas_str}]"
            )
            valid_beat_ids.append(beat_id)
            for did in dilemma_id_list:
                raw_did = dilemma_nodes[did].get("raw_id", did)
                if raw_did not in valid_dilemma_ids:
                    valid_dilemma_ids.append(raw_did)

        context = {
            "beat_summaries": "\n".join(beat_summaries),
            "valid_beat_ids": ", ".join(valid_beat_ids),
            "valid_dilemma_ids": ", ".join(valid_dilemma_ids),
        }

        # Call LLM with semantic validation
        from questfoundry.graph.grow_validators import validate_phase2_output

        validator = partial(
            validate_phase2_output,
            valid_beat_ids=set(valid_beat_ids),
            valid_dilemma_ids=set(valid_dilemma_ids),
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="grow_phase2_agnostic",
                context=context,
                output_schema=Phase2Output,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="failed",
                detail=str(e),
            )

        # Semantic validation: check all IDs exist
        valid_assessments: list[PathAgnosticAssessment] = []
        for assessment in result.assessments:
            if assessment.beat_id not in beat_nodes:
                log.warning(
                    "phase2_invalid_beat_id",
                    beat_id=assessment.beat_id,
                )
                continue
            # Filter agnostic_for to valid dilemma raw_ids
            invalid_dilemmas = [d for d in assessment.agnostic_for if d not in valid_dilemma_ids]
            if invalid_dilemmas:
                log.warning(
                    "phase2_invalid_dilemma_ids",
                    beat_id=assessment.beat_id,
                    invalid_ids=invalid_dilemmas,
                )
            valid_dilemmas = [d for d in assessment.agnostic_for if d in valid_dilemma_ids]
            if valid_dilemmas:
                valid_assessments.append(
                    PathAgnosticAssessment(
                        beat_id=assessment.beat_id,
                        agnostic_for=valid_dilemmas,
                    )
                )

        # Apply results to graph
        agnostic_count = 0
        for assessment in valid_assessments:
            graph.update_node(
                assessment.beat_id,
                path_agnostic_for=assessment.agnostic_for,
            )
            agnostic_count += 1

        return GrowPhaseResult(
            phase="path_agnostic",
            status="completed",
            detail=f"Assessed {len(candidate_beats)} beats, {agnostic_count} marked agnostic",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_3_intersections(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 3: Intersection detection.

        Pre-clusters beats from different dilemmas into candidate groups
        using algorithmic signal detection (shared locations/entities),
        then asks the LLM to evaluate which groups form natural scenes.

        Includes a structural retry: if all proposed intersections are
        rejected by compatibility checks, the LLM is re-invoked with
        targeted error feedback.

        An intersection is valid when:
        - Beats are from different dilemmas
        - No requires conflicts between the beats
        - Location is resolvable (shared location exists)
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
            "candidate_count": str(len(candidates)),
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
            accepted: list[tuple[list[str], str | None]] = []
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
                location: str | None
                if proposal.resolved_location:
                    location = proposal.resolved_location
                else:
                    location = resolve_intersection_location(pre_intersection_graph, valid_ids)
                    log.debug(
                        "phase3_location_resolved",
                        beat_ids=valid_ids,
                        resolved=location,
                    )

                accepted.append((valid_ids, location))
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
        for beat_ids, location in accepted:
            apply_intersection_mark(graph, beat_ids, location)
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

    async def _phase_4a_scene_types(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4a: Tag beats with scene type classification.

        Asks the LLM to classify each beat as scene (active conflict/action),
        sequel (reaction/reflection), or micro_beat (brief transition).
        Updates beat nodes with scene_type field.
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

    async def _phase_4b_narrative_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4b: Detect narrative gaps in path beat sequences.

        For each path, traces the beat sequence and asks the LLM
        to identify missing beats (e.g., a path jumps from setup
        to climax without a development beat). Inserts proposed gap
        beats into the graph.
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

        context = {
            "path_sequences": "\n\n".join(path_sequences),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(valid_beat_ids)),
        }

        from questfoundry.graph.grow_validators import validate_phase4_output

        validator = partial(
            validate_phase4_output,
            valid_path_ids=set(path_nodes.keys()),
            valid_beat_ids=valid_beat_ids,
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

    async def _phase_4c_pacing_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4c: Detect and fix pacing issues (3+ same scene_type in a row).

        Runs deterministic pacing detection first. If issues are found,
        asks the LLM to propose correction beats. Only proceeds if
        Phase 4a has tagged beats with scene types.
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

        context = {
            "path_sequences": "\n\n".join(path_sequences),
            "pacing_issues": "\n\n".join(issue_descriptions),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(valid_beat_ids)),
            "issue_count": str(len(issues)),
        }

        from questfoundry.graph.grow_validators import validate_phase4_output

        validator = partial(
            validate_phase4_output,
            valid_path_ids=set(path_nodes.keys()),
            valid_beat_ids=valid_beat_ids,
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

    async def _phase_4d_atmospheric(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4d: Atmospheric detail and entry states for beats.

        Generates sensory environment details for all beats and per-path
        entry moods for shared (path-agnostic) beats. Single batch LLM call.
        """
        from questfoundry.models.grow import Phase4dOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="atmospheric",
                status="completed",
                detail="No beats to annotate",
            )

        path_nodes = graph.get_nodes_by_type("path")

        # Build enriched beat summaries with entity names
        beat_items: list[ContextItem] = []
        shared_beats: list[str] = []
        for bid in sorted(beat_nodes.keys()):
            data = beat_nodes[bid]
            line = enrich_beat_line(graph, bid, data, include_entities=True)
            beat_items.append(ContextItem(id=bid, text=line))
            if data.get("path_agnostic_for"):
                shared_beats.append(bid)

        # Build narrative frame from dilemmas and paths
        dilemma_ids = sorted(graph.get_nodes_by_type("dilemma").keys())
        path_ids = sorted(path_nodes.keys())
        narrative_frame = build_narrative_frame(graph, dilemma_ids=dilemma_ids, path_ids=path_ids)

        # Build path info with theme/mood for entry state context
        path_info_lines: list[str] = []
        for pid in sorted(path_nodes.keys()):
            pdata = path_nodes[pid]
            dilemma = pdata.get("dilemma_id", "")
            theme = pdata.get("path_theme", "")
            mood = pdata.get("path_mood", "")
            line = f"- {pid} [dilemma={dilemma}]"
            if theme:
                line += f"\n    theme: {theme}"
            if mood:
                line += f"\n    mood: {mood}"
            path_info_lines.append(line)

        context = {
            "narrative_frame": narrative_frame,
            "beat_summaries": compact_items(beat_items, self._compact_config()),  # type: ignore[attr-defined]
            "beat_count": str(len(beat_nodes)),
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "shared_beats": ", ".join(shared_beats) if shared_beats else "(none)",
            "path_info": "\n".join(path_info_lines) if path_info_lines else "(no paths)",
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

        # Apply entry states (shared beats only)
        applied_entries = 0
        valid_path_set = set(path_nodes.keys())
        for entry_beat in result.entry_states:
            if entry_beat.beat_id not in beat_nodes:
                log.warning("phase4d_invalid_entry_beat_id", beat_id=entry_beat.beat_id)
                continue
            if entry_beat.beat_id not in shared_beats:
                log.warning(
                    "phase4d_entry_for_non_shared_beat",
                    beat_id=entry_beat.beat_id,
                )
                continue
            valid_moods = [
                {"path_id": em.path_id, "mood": em.mood}
                for em in entry_beat.moods
                if em.path_id in valid_path_set
            ]
            if valid_moods:
                graph.update_node(entry_beat.beat_id, entry_states=valid_moods)
                applied_entries += 1

        return GrowPhaseResult(
            phase="atmospheric",
            status="completed",
            detail=(
                f"Applied atmospheric details to {applied_details}/{len(beat_nodes)} beats, "
                f"entry states to {applied_entries}/{len(shared_beats)} shared beats"
            ),
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_4e_path_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4e: Per-path thematic mini-arcs.

        Generates a thematic through-line and mood descriptor for each path.
        One LLM call per path.
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

    async def _phase_4f_entity_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4f: Per-entity arc trajectories on each path.

        Runs post-intersection so beat topology is final. For each path,
        selects eligible entities (deterministic), then asks the LLM to
        generate arc_line and pivot_beat per entity. The arc_type is
        computed from entity category, not LLM output.

        Results are stored as ``entity_arcs`` on path nodes.
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
                beat_lines.append(f"{i}. {bid}: {summary} [entities={entities}]")

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

                # Warn if pivot is on a shared beat
                pivot_data = graph.get_node(arc.pivot_beat)
                if pivot_data and pivot_data.get("path_agnostic_for"):
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
