"""LLM-powered phase implementations for the POLISH stage.

Contains _PolishLLMPhaseMixin with Phases 1-3 (beat reordering,
pacing/micro-beat injection, character arc synthesis) and Phase 5
(LLM enrichment of the deterministic plan).

PolishStage inherits this mixin so ``execute()`` can delegate to
``self._phase_1_beat_reordering()``, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.polish_context import (
    format_ambiguous_feasibility_context,
    format_choice_label_context,
    format_entity_arc_context,
    format_false_branch_context,
    format_linear_section_context,
    format_pacing_context,
    format_residue_content_context,
    format_transition_guidance_context,
    format_variant_summary_context,
)
from questfoundry.models.pipeline import PhaseResult
from questfoundry.models.polish import (
    AmbiguousFeasibilityCase,
    FalseBranchSpec,
    Phase1Output,
    Phase2Output,
    Phase3Output,
    Phase5aOutput,
    Phase5bOutput,
    Phase5cOutput,
    Phase5dOutput,
    Phase5eOutput,
    Phase5fOutput,
    ResidueSpec,
    VariantSpec,
)
from questfoundry.pipeline.stages.polish._helpers import _PRE_PLAN_WARNINGS_NODE, log
from questfoundry.pipeline.stages.polish.deterministic import _load_plan_data
from questfoundry.pipeline.stages.polish.registry import polish_phase

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph


class _PolishLLMPhaseMixin:
    """Mixin providing LLM-powered POLISH phases (1, 2, 3, 5).

    Expects the host class to provide (via ``_PolishLLMHelperMixin``):

    - ``_polish_llm_call()``
    """

    @polish_phase(name="beat_reordering", depends_on=[], priority=0)
    async def _phase_1_beat_reordering(self, graph: Graph, model: BaseChatModel) -> PhaseResult:
        """Phase 1: Beat Reordering.

        Within linear sections of the beat DAG (3+ beats, single
        predecessor and single successor per beat), proposes reorderings
        for better narrative flow.

        Postconditions:
        - Predecessor edges updated within reordered sections.
        - Sections with invalid proposals keep original order.
        """
        beat_nodes = graph.get_nodes_by_type("beat")
        predecessor_edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, predecessor_edges)

        if not sections:
            log.info("phase1_no_sections", detail="No linear sections with 3+ beats found")
            return PhaseResult(
                phase="beat_reordering",
                status="skipped",
                detail="No linear sections with 3+ beats found",
            )

        total_llm_calls = 0
        total_tokens = 0
        reordered_count = 0
        warnings: list[str] = []

        for section in sections:
            section_id = section["section_id"]
            beat_ids = section["beat_ids"]
            before_beat = section.get("before_beat")
            after_beat = section.get("after_beat")

            context = format_linear_section_context(
                graph, section_id, beat_ids, before_beat, after_beat
            )

            result, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase1_reorder",
                context=context,
                output_schema=Phase1Output,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Find the matching section in the output
            matched = None
            for rs in result.reordered_sections:
                if rs.section_id == section_id:
                    matched = rs
                    break

            if matched is None:
                # LLM didn't propose a reordering for this section — keep original
                continue

            # Validate: must contain exactly the same beats
            if set(matched.beat_ids) != set(beat_ids):
                msg = (
                    f"Section {section_id}: reordering rejected — "
                    f"beat set mismatch (expected {len(beat_ids)}, got {len(matched.beat_ids)})"
                )
                warnings.append(msg)
                log.info(
                    "phase1_set_mismatch",
                    section=section_id,
                    expected=len(beat_ids),
                    got=len(matched.beat_ids),
                )
                continue

            # Validate: commit beats must not precede their dilemma's advance/reveal beats
            if not _validate_reorder_constraints(graph, beat_ids, matched.beat_ids):
                msg = (
                    f"Section {section_id}: reordering rejected — "
                    f"hard constraint violation (commit before advance/reveal)"
                )
                warnings.append(msg)
                log.info("phase1_constraint_violation", section=section_id)
                continue

            # Apply: update predecessor edges within the section
            _apply_reorder(graph, beat_ids, matched.beat_ids, before_beat, after_beat)
            reordered_count += 1
            log.debug(
                "phase1_section_reordered",
                section=section_id,
                rationale=matched.rationale[:80],
            )

        # Persist pre-plan warnings to graph so phase_plan_computation (a free
        # function without access to self) can drain them into PolishPlan.warnings.
        if warnings:
            existing = graph.get_node(_PRE_PLAN_WARNINGS_NODE)
            prior: list[str] = existing.get("warnings", []) if existing else []
            _upsert_pre_plan_warnings(graph, prior + warnings)

        detail = f"Reordered {reordered_count}/{len(sections)} sections"
        if warnings:
            detail += f"; {len(warnings)} warning(s)"

        return PhaseResult(
            phase="beat_reordering",
            status="completed",
            detail=detail,
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    @polish_phase(name="narrative_gaps", depends_on=["beat_reordering"], priority=1)
    async def _phase_1a_narrative_gaps(self, graph: Graph, model: BaseChatModel) -> PhaseResult:
        """Phase 1a: Narrative Gap Insertion.

        For each path with 2+ beats, the LLM identifies missing
        intermediate beats (e.g., a path goes setup → climax with no
        development beat) and proposes new beats to insert at specified
        positions. Insertion validates IDs, ordering, and cycle safety.

        Per the structural-vs-narrative migration (PR #1366), this is
        narrative-prep work — POLISH territory, not GROW. See
        ``docs/design/procedures/polish.md`` §Phase 1a for the spec.

        Postconditions:
        - Gap beats inserted into the graph with predecessor edges,
          ``belongs_to`` to their path, ``is_gap_beat=True``,
          ``role: gap_beat``, ``created_by: "POLISH"``.
        - Per-path cap: maximum 2 gap beats per path.
        """
        from questfoundry.graph.gap_insertion import validate_and_insert_gaps
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence
        from questfoundry.models.grow import Phase4bOutput

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            log.info("phase1a_no_paths", detail="No paths to check for gaps")
            return PhaseResult(
                phase="narrative_gaps",
                status="skipped",
                detail="No paths to check for gaps",
            )

        # Build path sequences with truncated summaries (matches GROW Phase 4b's
        # original context shape so the prompt can be migrated verbatim).
        path_sequences: list[str] = []
        valid_beat_ids: set[str] = set()
        for pid in sorted(path_nodes.keys()):
            sequence = get_path_beat_sequence(graph, pid)
            if len(sequence) < 2:
                continue
            beat_list: list[str] = []
            for bid in sequence:
                node = graph.get_node(bid)
                summary = (node.get("summary", "") or "")[:80] if node else ""
                scene_type = node.get("scene_type", "untagged") if node else "untagged"
                beat_list.append(f"    {bid} [{scene_type}]: {summary}")
                valid_beat_ids.add(bid)
            raw_pid = path_nodes[pid].get("raw_id", pid)
            path_sequences.append(f"  Path: {raw_pid} ({pid})\n" + "\n".join(beat_list))

        if not path_sequences:
            log.info("phase1a_no_multibeat_paths", detail="No paths with 2+ beats")
            return PhaseResult(
                phase="narrative_gaps",
                status="skipped",
                detail="No paths with 2+ beats to check",
            )

        # Path → dilemma map for the prompt's Valid IDs section.
        # Matches the helper used by GROW Phase 4b/4c (kept in grow llm_phases
        # while pacing_gaps still lives there; will move with PR C).
        from questfoundry.pipeline.stages.grow.llm_phases import (
            _build_path_dilemma_context,
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

        # The LLM call uses POLISH's helper. POLISH's _polish_llm_call has no
        # semantic_validator parameter, so invalid IDs in the response are
        # caught by validate_and_insert_gaps at insertion time instead of
        # forcing a retry. If retry-on-invalid-IDs becomes important, add
        # semantic_validator support to _polish_llm_call (mirroring GROW's).
        result, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
            model=model,
            template_name="polish_phase1a_narrative_gaps",
            context=context,
            output_schema=Phase4bOutput,
        )

        # Validate proposals and insert gap beats.
        report = validate_and_insert_gaps(
            graph,
            result.gaps,
            valid_path_ids=set(path_nodes.keys()),
            valid_beat_ids=valid_beat_ids,
            phase_name="phase1a",
        )

        log.info(
            "phase1a_complete",
            inserted=report.inserted,
            invalid=report.total_invalid,
            proposals=len(result.gaps),
        )
        return PhaseResult(
            phase="narrative_gaps",
            status="completed",
            detail=f"Inserted {report.inserted} gap beats from {len(result.gaps)} proposals",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    @polish_phase(name="pacing", depends_on=["narrative_gaps"], priority=1)
    async def _phase_2_pacing(self, graph: Graph, model: BaseChatModel) -> PhaseResult:
        """Phase 2: Pacing & Micro-beat Injection.

        Detects pacing issues (3+ scene/sequel in a row, no sequel after
        commit) and proposes micro-beat insertions to smooth the rhythm.

        Postconditions:
        - Micro-beat nodes created with role="micro_beat".
        - Predecessor edges updated to include micro-beats in DAG.
        """
        beat_nodes = graph.get_nodes_by_type("beat")
        predecessor_edges = graph.get_edges(edge_type="predecessor")

        pacing_flags = _detect_pacing_flags(beat_nodes, predecessor_edges, graph)

        if not pacing_flags:
            log.info("phase2_no_flags", detail="No pacing issues detected")
            return PhaseResult(
                phase="pacing",
                status="skipped",
                detail="No pacing issues detected",
            )

        context = format_pacing_context(graph, pacing_flags)

        result, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
            model=model,
            template_name="polish_phase2_pacing",
            context=context,
            output_schema=Phase2Output,
        )

        # Apply micro-beats
        inserted = 0
        for mb in result.micro_beats:
            if mb.after_beat_id not in beat_nodes:
                log.info(
                    "phase2_invalid_after_beat",
                    after_beat_id=mb.after_beat_id,
                )
                continue

            micro_beat_id = f"beat::micro_{inserted}_{mb.after_beat_id.split('::')[-1]}"
            _insert_micro_beat(graph, micro_beat_id, mb.after_beat_id, mb.summary, mb.entity_ids)
            inserted += 1

        return PhaseResult(
            phase="pacing",
            status="completed",
            detail=f"Inserted {inserted} micro-beat(s) from {len(pacing_flags)} pacing flag(s)",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    @polish_phase(name="character_arcs", depends_on=["pacing"], priority=2)
    async def _phase_3_character_arcs(self, graph: Graph, model: BaseChatModel) -> PhaseResult:
        """Phase 3: Character Arc Synthesis.

        For entities appearing in 2+ beats, synthesizes arc descriptions
        (start, pivots, end per path) for FILL's prose consistency.

        Postconditions:
        - CharacterArcMetadata nodes created for arc-worthy entities.
        """
        beat_nodes = graph.get_nodes_by_type("beat")
        entity_nodes = graph.get_nodes_by_type("entity")

        # Find entities appearing in 2+ beats
        entity_beats = _collect_entity_appearances(beat_nodes, graph)
        arc_worthy = {
            eid: beats
            for eid, beats in entity_beats.items()
            if len(beats) >= 2 and eid in entity_nodes
        }

        if not arc_worthy:
            log.info("phase3_no_entities", detail="No entities with 2+ beat appearances")
            return PhaseResult(
                phase="character_arcs",
                status="skipped",
                detail="No entities with 2+ beat appearances",
            )

        total_llm_calls = 0
        total_tokens = 0
        arcs_created = 0

        # Process entities in batches via a single LLM call per entity
        for entity_id, beat_appearances in sorted(arc_worthy.items()):
            context = format_entity_arc_context(graph, entity_id, beat_appearances)

            result, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase3_arcs",
                context=context,
                output_schema=Phase3Output,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Store arc metadata as an annotation on the Entity node (R-3.3).
            # The old pattern (separate character_arc_metadata nodes +
            # has_arc_metadata edges) violated R-3.3 and ontology Part 1.
            for arc in result.character_arcs:
                if arc.entity_id != entity_id:
                    log.info(
                        "phase3_entity_mismatch",
                        expected=entity_id,
                        got=arc.entity_id,
                    )
                    continue

                # pivots is a list of ArcPivot models with (path_id, beat_id)
                # attributes; normalize to path_id → beat_id so the R-3.3
                # validator's shape check passes.
                pivots_by_path = {p.path_id: p.beat_id for p in arc.pivots}

                graph.update_node(
                    entity_id,
                    character_arc={
                        "start": arc.start,
                        "pivots": pivots_by_path,
                        "end_per_path": dict(arc.end_per_path),
                    },
                )
                arcs_created += 1

        return PhaseResult(
            phase="character_arcs",
            status="completed",
            detail=f"Created {arcs_created} character arc(s) from {len(arc_worthy)} entities",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    @polish_phase(name="atmospheric_annotation", depends_on=["character_arcs"], priority=3)
    async def _phase_5e_atmospheric(self, graph: Graph, model: BaseChatModel) -> PhaseResult:
        """Phase 5e: Atmospheric Annotation.

        For every beat in the frozen DAG, generate an
        ``atmospheric_detail`` string describing the sensory environment
        (sight, sound, smell, texture). Single LLM call covers all beats.

        Per spec ``polish.md`` §Phase 5e (R-5e.1 through R-5e.3) and the
        structural-vs-narrative migration in PR #1366: sensory grounding
        is prose-prep, POLISH territory. Migrated from GROW Phase 4d
        per issue #1368 (PR B).

        R-5e.3: runs after Beat DAG Freeze, so transition beats inserted
        by GROW Phase 4c receive ``atmospheric_detail`` like any other
        beat (auto-fixes the audit Q3 gap).
        """
        from questfoundry.graph.context_compact import (
            ContextItem,
            build_narrative_frame,
            compact_items,
            enrich_beat_line,
        )
        from questfoundry.models.grow import Phase4dOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            log.info("phase5e_no_beats", detail="No beats to annotate")
            return PhaseResult(
                phase="atmospheric_annotation",
                status="skipped",
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

        # POLISH's _polish_llm_call doesn't accept a compact-config the way
        # GROW does; pass the items directly with a generous default budget
        # since the prompt is small and the LLM can handle many beats.
        context = {
            "narrative_frame": narrative_frame,
            "beat_summaries": compact_items(beat_items, None),
            "beat_count": str(len(beat_nodes)),
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
        }

        result, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
            model=model,
            template_name="polish_phase5e_atmospheric",
            context=context,
            output_schema=Phase4dOutput,
        )

        # Apply atmospheric details (R-5e.1: partial coverage emits WARNING)
        applied = 0
        for detail in result.details:
            if detail.beat_id not in beat_nodes:
                log.info("phase5e_invalid_beat_id", beat_id=detail.beat_id)
                continue
            graph.update_node(
                detail.beat_id,
                atmospheric_detail=detail.atmospheric_detail,
            )
            applied += 1

        if applied < len(beat_nodes):
            log.warning(
                "phase5e_partial_coverage",
                applied=applied,
                total=len(beat_nodes),
                missing=len(beat_nodes) - applied,
            )

        return PhaseResult(
            phase="atmospheric_annotation",
            status="completed",
            detail=f"Applied atmospheric details to {applied}/{len(beat_nodes)} beats",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    @polish_phase(
        name="path_thematic_annotation",
        depends_on=["atmospheric_annotation"],
        priority=3,
    )
    async def _phase_5f_path_thematic(self, graph: Graph, model: BaseChatModel) -> PhaseResult:
        """Phase 5f: Path Thematic Annotation.

        For each multi-beat path, generate ``path_theme`` (controlling
        idea) and ``path_mood`` (tonal palette). One LLM call per path.

        Per spec ``polish.md`` §Phase 5f (R-5f.1 through R-5f.3) and the
        structural-vs-narrative migration in PR #1366: per-path narrative
        identity is prose-prep, POLISH territory. Migrated from GROW
        Phase 4e per issue #1368 (PR B).
        """
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence
        from questfoundry.models.grow import PathMiniArc
        from questfoundry.pipeline.batching import batch_llm_calls

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            log.info("phase5f_no_paths", detail="No paths to annotate")
            return PhaseResult(
                phase="path_thematic_annotation",
                status="skipped",
                detail="No paths to annotate",
            )

        applied = 0

        # Pre-compute context for each path (graph reads are sync)
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
                log.info("phase5f_cycle_in_path", path_id=pid)
                continue

            # R-5f.1: skip paths with <2 beats (no narrative arc to summarize)
            if len(beat_ids) < 2:
                log.info("phase5f_skip_short_path", path_id=pid, beats=len(beat_ids))
                continue

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
            result, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase5f_path_thematic",
                context=ctx,
                output_schema=PathMiniArc,
            )
            return (pid, result), llm_calls, tokens

        # PolishStage doesn't carry _max_concurrency / _on_connectivity_error
        # the way GrowStage does (those were added for GROW's batch phases).
        # Default to concurrency=2 and no retry hook; if PolishStage grows
        # those attrs later, getattr picks them up automatically.
        results, total_llm_calls, total_tokens, errors = await batch_llm_calls(
            path_items,
            _arc_for_path,
            getattr(self, "_max_concurrency", 2),
            on_connectivity_error=getattr(self, "_on_connectivity_error", None),
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

        # R-5f.3: per-path failures log at WARNING; field stays unpopulated
        if errors:
            for idx, e in errors:
                pid = path_items[idx][0]
                log.warning("phase5f_llm_failed", path_id=pid, error=str(e))

        return PhaseResult(
            phase="path_thematic_annotation",
            status="completed",
            detail=f"Annotated {applied}/{len(path_items)} multi-beat paths",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    @polish_phase(name="llm_enrichment", depends_on=["plan_computation"], priority=4)
    async def _phase_5_llm_enrichment(self, graph: Graph, model: BaseChatModel) -> PhaseResult:
        """Phase 5: LLM Enrichment of the deterministic plan.

        Enriches the plan from Phase 4 with creative content:
        5a: Choice labels (diegetic, distinct, concise)
        5b: Residue beat content (mood-setting prose hints)
        5c: False branch decisions (skip/diamond/sidetrack)
        5d: Variant passage summaries

        Postconditions:
        - Plan node updated with enriched specs.
        """
        plan_data = _load_plan_data(graph)
        if plan_data is None:
            return PhaseResult(
                phase="llm_enrichment",
                status="failed",
                detail="No polish plan found — Phase 4 must run first",
            )

        passage_specs = plan_data.get("passage_specs", [])
        choice_specs = plan_data.get("choice_specs", [])
        residue_specs = plan_data.get("residue_specs", [])
        false_branch_candidates = plan_data.get("false_branch_candidates", [])
        variant_specs = plan_data.get("variant_specs", [])

        total_llm_calls = 0
        total_tokens = 0
        enrichment_parts: list[str] = []

        # 5a: Choice labels
        if choice_specs:
            context = format_choice_label_context(graph, choice_specs, passage_specs)
            result, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase5a_choice_labels",
                context=context,
                output_schema=Phase5aOutput,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Apply labels to choice specs — detect duplicate (from, to) pairs
            label_lookup: dict[tuple[str, str], str] = {}
            for item in result.choice_labels:
                key = (item.from_passage, item.to_passage)
                if key in label_lookup:
                    log.info(
                        "phase5a_duplicate_choice_label",
                        from_passage=item.from_passage,
                        to_passage=item.to_passage,
                        kept=label_lookup[key],
                        discarded=item.label,
                    )
                else:
                    label_lookup[key] = item.label
            for spec in choice_specs:
                key = (spec["from_passage"], spec["to_passage"])
                if key in label_lookup:
                    spec["label"] = label_lookup[key]

            # R-5.2: labels are distinct within a source passage.  Case-insensitive
            # uniqueness — detect collisions, log a WARNING so humans can review.
            for collision in _detect_duplicate_labels_in_passage(choice_specs):
                log.warning(
                    "phase5a_duplicate_labels_in_passage",
                    from_passage=collision["from_passage"],
                    duplicate_label=collision["label"],
                    conflicting_targets=collision["targets"],
                    hint="Human review recommended; R-5.2 requires distinct labels within a passage.",
                )

            enrichment_parts.append(f"{len(result.choice_labels)} choice labels")
            log.debug("phase5a_complete", labels=len(result.choice_labels))

        # 5b: Residue beat content
        if residue_specs:
            context = format_residue_content_context(graph, residue_specs, passage_specs)
            result_b, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase5b_residue",
                context=context,
                output_schema=Phase5bOutput,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Apply content hints and mapping_strategy to residue specs
            item_lookup = {item.residue_id: item for item in result_b.residue_content}
            for spec in residue_specs:
                rid = spec.get("residue_id", "")
                if rid in item_lookup:
                    item = item_lookup[rid]
                    spec["content_hint"] = item.content_hint
                    spec["mapping_strategy"] = item.mapping_strategy

            enrichment_parts.append(f"{len(result_b.residue_content)} residue hints")
            log.debug("phase5b_complete", hints=len(result_b.residue_content))

        # 5c: False branch decisions
        false_branch_specs: list[dict[str, Any]] = []
        if false_branch_candidates:
            context = format_false_branch_context(graph, false_branch_candidates, passage_specs)
            result_c, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase5c_false_branches",
                context=context,
                output_schema=Phase5cOutput,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            for decision in result_c.decisions:
                idx = decision.candidate_index
                if idx < 0 or idx >= len(false_branch_candidates):
                    log.info("phase5c_invalid_index", index=idx)
                    continue

                candidate = false_branch_candidates[idx]
                fb_spec = FalseBranchSpec(
                    candidate_passage_ids=candidate.get("passage_ids", []),
                    branch_type=decision.decision,
                    details=decision.details,
                    diamond_summary_a=decision.diamond_summary_a,
                    diamond_summary_b=decision.diamond_summary_b,
                    sidetrack_summary=decision.sidetrack_summary,
                    sidetrack_entities=decision.sidetrack_entities,
                    choice_label_enter=decision.choice_label_enter,
                    choice_label_return=decision.choice_label_return,
                )
                false_branch_specs.append(fb_spec.model_dump())

            enrichment_parts.append(f"{len(false_branch_specs)} false branch decisions")
            log.debug("phase5c_complete", decisions=len(false_branch_specs))

        # 5d: Variant passage summaries
        if variant_specs:
            context = format_variant_summary_context(graph, variant_specs, passage_specs)
            result_d, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase5d_variants",
                context=context,
                output_schema=Phase5dOutput,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Apply summaries to variant specs
            summary_lookup = {item.variant_id: item.summary for item in result_d.variant_summaries}
            for spec in variant_specs:
                vid = spec.get("variant_id", "")
                if vid in summary_lookup:
                    spec["summary"] = summary_lookup[vid]

            enrichment_parts.append(f"{len(result_d.variant_summaries)} variant summaries")
            log.debug("phase5d_complete", summaries=len(result_d.variant_summaries))

        # 5e: Resolve ambiguous feasibility cases
        ambiguous_specs_raw = plan_data.get("ambiguous_specs", [])
        if ambiguous_specs_raw:
            ambiguous_cases = [AmbiguousFeasibilityCase(**a) for a in ambiguous_specs_raw]
            context = format_ambiguous_feasibility_context(graph, ambiguous_cases, passage_specs)
            result_e, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase5e_feasibility",
                context=context,
                output_schema=Phase5eOutput,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Build lookup for fast access
            case_lookup: dict[str, AmbiguousFeasibilityCase] = {
                c.passage_id: c for c in ambiguous_cases
            }

            resolved_count = 0
            for decision in result_e.feasibility_decisions:
                passage_id = decision.passage_id
                flag_index = decision.flag_index
                case = case_lookup.get(passage_id)
                if case is None:
                    log.info("phase5e_unknown_passage", passage_id=passage_id)
                    continue
                if flag_index < 0 or flag_index >= len(case.flags):
                    log.info(
                        "phase5e_invalid_flag_index",
                        passage_id=passage_id,
                        flag_index=flag_index,
                        flags_count=len(case.flags),
                    )
                    continue

                flag = case.flags[flag_index]
                d = decision.decision

                if d == "variant":
                    variant_counter = len(variant_specs)
                    variant_specs.append(
                        VariantSpec(
                            base_passage_id=passage_id,
                            variant_id=f"passage::variant_{variant_counter}",
                            requires=[flag],
                            summary="",
                        ).model_dump()
                    )
                elif d == "residue":
                    path_id = flag.split(":")[-1] if ":" in flag else ""
                    passage_raw = passage_id.split("::")[-1]
                    residue_specs.append(
                        ResidueSpec(
                            target_passage_id=passage_id,
                            residue_id=f"residue::{passage_raw}_{flag.replace(':', '_')}",
                            flag=flag,
                            path_id=path_id,
                        ).model_dump()
                    )
                elif d == "irrelevant":
                    # Append to feasibility_annotations
                    ann_key = passage_id
                    existing = plan_data.get("feasibility_annotations", {})
                    existing.setdefault(ann_key, []).append(flag)
                else:
                    log.info("phase5e_unknown_decision", decision=d, passage_id=passage_id)
                    continue

                resolved_count += 1

            enrichment_parts.append(f"{resolved_count} ambiguous cases resolved")
            log.debug("phase5e_complete", resolved=resolved_count)
        else:
            enrichment_parts.append("0 ambiguous cases resolved")
            log.info("phase5e_skipped", status="skipped", detail="No ambiguous feasibility cases")

        # 5f: Transition guidance for collapsed passages
        collapsed_specs = [
            p
            for p in passage_specs
            if p.get("grouping_type") == "collapse" and len(p.get("beat_ids", [])) >= 2
        ]
        if collapsed_specs:
            context = format_transition_guidance_context(graph, passage_specs)
            result_f, llm_calls, tokens = await self._polish_llm_call(  # type: ignore[attr-defined]
                model=model,
                template_name="polish_phase5f_transitions",
                context=context,
                output_schema=Phase5fOutput,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            passage_lookup: dict[str, dict[str, Any]] = {p["passage_id"]: p for p in passage_specs}
            guides_applied = 0
            for item in result_f.transition_guidance:
                spec = passage_lookup.get(item.passage_id)
                if spec is None:
                    log.info(
                        "phase5f_unknown_passage",
                        passage_id=item.passage_id,
                    )
                    continue
                expected = len(spec.get("beat_ids", [])) - 1
                if len(item.transitions) != expected:
                    log.info(
                        "phase5f_transition_count_mismatch",
                        passage_id=item.passage_id,
                        expected=expected,
                        got=len(item.transitions),
                    )
                    continue
                spec["transition_guidance"] = item.transitions
                guides_applied += 1

            enrichment_parts.append(f"{guides_applied} transition guides generated")
            log.debug("phase5f_complete", guides=guides_applied)
        else:
            enrichment_parts.append("0 transition guides generated")
            log.info(
                "phase5f_skipped",
                status="skipped",
                detail="No collapsed passages with 2+ beats",
            )

        # Store enriched plan back to graph
        _update_plan_data(
            graph,
            choice_specs=choice_specs,
            residue_specs=residue_specs,
            false_branch_specs=false_branch_specs,
            variant_specs=variant_specs,
            ambiguous_specs=[],  # Resolved — clear from plan
            passage_specs=passage_specs,
            feasibility_annotations=plan_data.get("feasibility_annotations", {}),
        )

        detail = "; ".join(enrichment_parts) if enrichment_parts else "No enrichment needed"

        return PhaseResult(
            phase="llm_enrichment",
            status="completed",
            detail=detail,
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )


# ---------------------------------------------------------------------------
# Phase 5 helpers
# ---------------------------------------------------------------------------


def _detect_duplicate_labels_in_passage(
    choice_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """R-5.2: return collisions where two+ choices from the same passage share a label.

    Pure function — no side effects.  Used by Phase 5a after the LLM
    assigns labels to detect case-insensitive within-passage collisions.
    The caller logs each collision; the LLM re-call decision is left to
    the operator (empty return means no collisions).

    Each collision is a dict with:
      - ``from_passage``: the source passage.
      - ``label``: the case-folded label string that collided.
      - ``targets``: sorted list of target passages sharing the label.
    """
    from collections import defaultdict

    labels_by_passage: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for spec in choice_specs:
        label = spec.get("label") or ""
        if not label:
            continue
        labels_by_passage[spec["from_passage"]][label.lower()].append(spec["to_passage"])

    collisions: list[dict[str, Any]] = []
    for from_passage in sorted(labels_by_passage):
        label_map = labels_by_passage[from_passage]
        for lower_label in sorted(label_map):
            targets = label_map[lower_label]
            if len(targets) > 1:
                collisions.append(
                    {
                        "from_passage": from_passage,
                        "label": lower_label,
                        "targets": sorted(targets),
                    }
                )
    return collisions


def _update_plan_data(
    graph: Graph,
    *,
    choice_specs: list[dict[str, Any]],
    residue_specs: list[dict[str, Any]],
    false_branch_specs: list[dict[str, Any]],
    variant_specs: list[dict[str, Any]],
    ambiguous_specs: list[dict[str, Any]] | None = None,
    passage_specs: list[dict[str, Any]] | None = None,
    feasibility_annotations: dict[str, list[str]] | None = None,
) -> None:
    """Update the plan node with enriched data from Phase 5."""
    updates: dict[str, Any] = {
        "choice_specs": choice_specs,
        "residue_specs": residue_specs,
        "false_branch_specs": false_branch_specs,
        "variant_specs": variant_specs,
    }
    if ambiguous_specs is not None:
        updates["ambiguous_specs"] = ambiguous_specs
    if passage_specs is not None:
        updates["passage_specs"] = passage_specs
    if feasibility_annotations is not None:
        updates["feasibility_annotations"] = feasibility_annotations
    graph.update_node("polish_plan::current", **updates)


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


def _upsert_pre_plan_warnings(graph: Graph, warnings: list[str]) -> None:
    """Persist Phase 1 rejection warnings to a temporary graph node.

    The node is created or updated so that ``phase_plan_computation``
    (a free function) can drain these warnings into ``PolishPlan.warnings``
    without needing a reference to the ``PolishStage`` instance.
    """
    if not warnings:
        return
    existing = graph.get_node(_PRE_PLAN_WARNINGS_NODE)
    if existing is None:
        graph.create_node(
            _PRE_PLAN_WARNINGS_NODE,
            {
                "type": "polish_meta",
                "raw_id": "pre_plan_warnings",
                "warnings": warnings,
            },
        )
    else:
        graph.update_node(_PRE_PLAN_WARNINGS_NODE, warnings=warnings)


def _find_linear_sections(
    beat_nodes: dict[str, dict[str, Any]],
    predecessor_edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Find maximal linear sections in the beat DAG.

    A linear section is a chain of beats where each has exactly one
    predecessor and one successor (within the section). Only returns
    sections with 3+ beats.

    Returns:
        List of dicts with keys: section_id, beat_ids, before_beat, after_beat.
    """
    # Build adjacency
    children: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    parents: dict[str, list[str]] = {bid: [] for bid in beat_nodes}

    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            parents[from_id].append(to_id)
            children[to_id].append(from_id)

    # Find linear chains: start from beats with != 1 parent or with a parent
    # that has != 1 child (i.e., boundary beats)
    visited: set[str] = set()
    sections: list[dict[str, Any]] = []
    section_counter = 0

    for bid in sorted(beat_nodes.keys()):
        if bid in visited:
            continue

        # Check if this is a potential chain start:
        # - Has exactly 1 child
        # - Has != 1 parent OR parent has != 1 child (i.e., is a boundary)
        p = parents[bid]
        c = children[bid]

        is_chain_start = len(c) == 1 and (len(p) != 1 or (p and len(children[p[0]]) != 1))
        # Also start from root beats (no parents)
        if not is_chain_start and len(p) != 0:
            continue

        # Walk the chain
        chain = [bid]
        current = bid
        while True:
            c = children[current]
            if len(c) != 1:
                break
            next_beat = c[0]
            if len(parents[next_beat]) != 1:
                break
            chain.append(next_beat)
            current = next_beat

        for b in chain:
            visited.add(b)

        if len(chain) < 3:
            continue

        before_beat = parents[chain[0]][0] if parents[chain[0]] else None
        after_beat = children[chain[-1]][0] if children[chain[-1]] else None

        sections.append(
            {
                "section_id": f"section_{section_counter}",
                "beat_ids": chain,
                "before_beat": before_beat,
                "after_beat": after_beat,
            }
        )
        section_counter += 1

    return sections


def _validate_reorder_constraints(
    graph: Graph,
    original_order: list[str],  # noqa: ARG001
    proposed_order: list[str],
) -> bool:
    """Validate that a reordering preserves hard constraints.

    Commit beats must come after their dilemma's advance/reveal beats
    within the same section.

    Returns:
        True if the reordering is valid.
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    # Map dilemma_id → list of (beat_id, effect) in proposed order
    dilemma_beats: dict[str, list[tuple[str, str]]] = {}
    for bid in proposed_order:
        data = beat_nodes.get(bid, {})
        for impact in data.get("dilemma_impacts", []):
            dilemma_id = impact.get("dilemma_id", "")
            effect = impact.get("effect", "")
            if dilemma_id and effect:
                dilemma_beats.setdefault(dilemma_id, []).append((bid, effect))

    # For each dilemma, commits must not precede advances/reveals
    for _dilemma_id, beats in dilemma_beats.items():
        commit_indices = []
        advance_reveal_indices = []
        for i, (_bid, effect) in enumerate(beats):
            if effect == "commits":
                commit_indices.append(i)
            elif effect in ("advances", "reveals"):
                advance_reveal_indices.append(i)

        if commit_indices and advance_reveal_indices:
            earliest_commit = min(commit_indices)
            latest_advance_reveal = max(advance_reveal_indices)
            if earliest_commit < latest_advance_reveal:
                return False

    return True


def _apply_reorder(
    graph: Graph,
    original_order: list[str],
    new_order: list[str],
    before_beat: str | None,
    after_beat: str | None,
) -> None:
    """Apply a reordering by updating predecessor edges within the section.

    Removes old predecessor edges between section beats and creates
    new ones reflecting the proposed order.
    """
    # Remove old internal predecessor edges
    for i in range(1, len(original_order)):
        graph.remove_edge("predecessor", original_order[i], original_order[i - 1])

    # Remove edges connecting section to before/after
    if before_beat:
        graph.remove_edge("predecessor", original_order[0], before_beat)
    if after_beat:
        graph.remove_edge("predecessor", after_beat, original_order[-1])

    # Add new internal edges
    for i in range(1, len(new_order)):
        graph.add_edge("predecessor", new_order[i], new_order[i - 1])

    # Reconnect to before/after
    if before_beat:
        graph.add_edge("predecessor", new_order[0], before_beat)
    if after_beat:
        graph.add_edge("predecessor", after_beat, new_order[-1])


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------


def _detect_pacing_flags(
    beat_nodes: dict[str, dict[str, Any]],
    predecessor_edges: list[dict[str, str]],
    graph: Graph,
) -> list[dict[str, Any]]:
    """Detect pacing issues in the beat DAG.

    Flags:
    - 3+ scene beats in a row without a sequel/reflection
    - 3+ sequel beats in a row without an action/scene
    - No sequel after a commits beat

    Returns:
        List of dicts with keys: issue_type, beat_ids, path_id.
    """
    # Build adjacency for path-local walks
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    _dp_accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            _dp_accum.setdefault(edge["from"], set()).add(edge["to"])
    beat_to_paths: dict[str, frozenset[str]] = {bid: frozenset(ps) for bid, ps in _dp_accum.items()}

    def _primary_path(bid: str) -> str:
        """First-by-sort-order path membership. Stable for reporting."""
        return next(iter(sorted(beat_to_paths.get(bid, frozenset()))), "")

    children: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    parents: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            parents[from_id].append(to_id)
            children[to_id].append(from_id)

    flags: list[dict[str, Any]] = []

    # Walk linear chains and detect patterns
    visited: set[str] = set()
    for bid in sorted(beat_nodes.keys()):
        if bid in visited or parents[bid]:
            continue

        # Walk from roots
        chain = _walk_linear_chain(bid, children, parents)
        for b in chain:
            visited.add(b)

        # Check for consecutive scene/sequel runs
        _check_consecutive_runs(chain, beat_nodes, _primary_path, flags)

        # Check for missing sequel after commits
        _check_post_commit_sequel(chain, beat_nodes, _primary_path, flags)

    return flags


def _walk_linear_chain(
    start: str,
    children: dict[str, list[str]],
    parents: dict[str, list[str]],
) -> list[str]:
    """Walk a linear chain from start, following single-child links."""
    chain = [start]
    current = start
    while True:
        c = children[current]
        if len(c) != 1:
            break
        next_beat = c[0]
        if len(parents[next_beat]) != 1:
            break
        chain.append(next_beat)
        current = next_beat
    return chain


def _check_consecutive_runs(
    chain: list[str],
    beat_nodes: dict[str, dict[str, Any]],
    get_primary_path: Callable[[str], str],
    flags: list[dict[str, Any]],
) -> None:
    """Check for 3+ consecutive same-type beats in a chain."""
    if len(chain) < 3:
        return

    run_type: str | None = None
    run_beats: list[str] = []

    for bid in chain:
        scene_type = beat_nodes.get(bid, {}).get("scene_type", "unknown")

        if scene_type == run_type:
            run_beats.append(bid)
        else:
            if len(run_beats) >= 3 and run_type in ("scene", "sequel"):
                flags.append(
                    {
                        "issue_type": f"consecutive_{run_type}",
                        "beat_ids": list(run_beats),
                        "path_id": get_primary_path(run_beats[0]),
                    }
                )
            run_type = scene_type
            run_beats = [bid]

    # Check final run
    if len(run_beats) >= 3 and run_type in ("scene", "sequel"):
        flags.append(
            {
                "issue_type": f"consecutive_{run_type}",
                "beat_ids": list(run_beats),
                "path_id": get_primary_path(run_beats[0]),
            }
        )


def _check_post_commit_sequel(
    chain: list[str],
    beat_nodes: dict[str, dict[str, Any]],
    get_primary_path: Callable[[str], str],
    flags: list[dict[str, Any]],
) -> None:
    """Check for missing sequel beats after commit beats."""
    for i, bid in enumerate(chain):
        data = beat_nodes.get(bid, {})
        impacts = data.get("dilemma_impacts", [])

        is_commit = any(imp.get("effect") == "commits" for imp in impacts)
        if not is_commit:
            continue

        # Check if next beat is a sequel (or if commit is at end of chain)
        if i + 1 >= len(chain):
            # Commit is the last beat in the chain — no sequel follows
            flags.append(
                {
                    "issue_type": "no_sequel_after_commit",
                    "beat_ids": chain[max(0, i - 1) : i + 1],
                    "path_id": get_primary_path(bid),
                }
            )
        else:
            next_data = beat_nodes.get(chain[i + 1], {})
            next_type = next_data.get("scene_type", "unknown")
            if next_type != "sequel":
                flags.append(
                    {
                        "issue_type": "no_sequel_after_commit",
                        "beat_ids": chain[max(0, i - 1) : i + 3],
                        "path_id": get_primary_path(bid),
                    }
                )


def _insert_micro_beat(
    graph: Graph,
    micro_beat_id: str,
    after_beat_id: str,
    summary: str,
    entity_ids: list[str],
) -> None:
    """Insert a micro-beat node after the specified beat in the DAG.

    Updates predecessor edges to splice the micro-beat into the chain.
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    # Determine path from the after_beat
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    path_id = None
    for edge in belongs_to_edges:
        if edge["from"] == after_beat_id:
            path_id = edge["to"]
            break

    # Create the micro-beat node
    graph.create_node(
        micro_beat_id,
        {
            "type": "beat",
            "raw_id": micro_beat_id.split("::")[-1],
            "summary": summary,
            "role": "micro_beat",
            "scene_type": "micro_beat",
            "dilemma_impacts": [],
            "entities": entity_ids,
            "created_by": "POLISH",
        },
    )

    # Add belongs_to edge
    if path_id:
        graph.add_edge("belongs_to", micro_beat_id, path_id)

    # Find children of after_beat that are on the same path (linear successors)
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    successors = []
    for edge in predecessor_edges:
        if edge["to"] == after_beat_id and edge["from"] in beat_nodes:
            successors.append(edge["from"])

    # Splice: micro-beat becomes predecessor of after_beat's successors
    # and after_beat becomes predecessor of micro-beat
    graph.add_edge("predecessor", micro_beat_id, after_beat_id)

    # For linear chains, reconnect the first successor
    if len(successors) == 1:
        graph.remove_edge("predecessor", successors[0], after_beat_id)
        graph.add_edge("predecessor", successors[0], micro_beat_id)


# ---------------------------------------------------------------------------
# Phase 3 helpers
# ---------------------------------------------------------------------------


def _collect_entity_appearances(
    beat_nodes: dict[str, dict[str, Any]],
    graph: Graph,
) -> dict[str, list[str]]:
    """Collect which beats each entity appears in, in topological order.

    Returns:
        Dict mapping entity_id → list of beat_ids (topologically sorted).
    """
    # Build topological order
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    topo_order = _topological_sort(beat_nodes, predecessor_edges)

    entity_beats: dict[str, list[str]] = {}
    for bid in topo_order:
        data = beat_nodes.get(bid, {})
        entities = data.get("entities", [])
        for eid in entities:
            entity_beats.setdefault(eid, []).append(bid)

    return entity_beats


def _topological_sort(
    beat_nodes: dict[str, dict[str, Any]],
    predecessor_edges: list[dict[str, str]],
) -> list[str]:
    """Kahn's algorithm for topological sort of beat DAG.

    Returns beat IDs in topological order (parents before children).
    Uses a sorted list as a priority queue for deterministic ordering.
    """
    from collections import deque

    in_degree: dict[str, int] = dict.fromkeys(beat_nodes, 0)
    adj: dict[str, list[str]] = {bid: [] for bid in beat_nodes}

    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            in_degree[from_id] += 1
            adj[to_id].append(from_id)

    queue = deque(sorted(bid for bid, deg in in_degree.items() if deg == 0))
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in sorted(adj[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result
