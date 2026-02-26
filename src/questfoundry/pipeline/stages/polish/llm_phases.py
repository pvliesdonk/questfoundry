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
    format_choice_label_context,
    format_entity_arc_context,
    format_false_branch_context,
    format_linear_section_context,
    format_pacing_context,
    format_residue_content_context,
    format_variant_summary_context,
)
from questfoundry.models.pipeline import PhaseResult
from questfoundry.models.polish import (
    FalseBranchSpec,
    Phase1Output,
    Phase2Output,
    Phase3Output,
    Phase5aOutput,
    Phase5bOutput,
    Phase5cOutput,
    Phase5dOutput,
)
from questfoundry.pipeline.stages.polish._helpers import log
from questfoundry.pipeline.stages.polish.registry import polish_phase

if TYPE_CHECKING:
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
                warnings.append(
                    f"Section {section_id}: reordering rejected — "
                    f"beat set mismatch (expected {len(beat_ids)}, got {len(matched.beat_ids)})"
                )
                log.warning(
                    "phase1_set_mismatch",
                    section=section_id,
                    expected=len(beat_ids),
                    got=len(matched.beat_ids),
                )
                continue

            # Validate: commit beats must not precede their dilemma's advance/reveal beats
            if not _validate_reorder_constraints(graph, beat_ids, matched.beat_ids):
                warnings.append(
                    f"Section {section_id}: reordering rejected — "
                    f"hard constraint violation (commit before advance/reveal)"
                )
                log.warning("phase1_constraint_violation", section=section_id)
                continue

            # Apply: update predecessor edges within the section
            _apply_reorder(graph, beat_ids, matched.beat_ids, before_beat, after_beat)
            reordered_count += 1
            log.debug(
                "phase1_section_reordered",
                section=section_id,
                rationale=matched.rationale[:80],
            )

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

    @polish_phase(name="pacing", depends_on=["beat_reordering"], priority=1)
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
                log.warning(
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

            # Store arc metadata
            for arc in result.character_arcs:
                if arc.entity_id != entity_id:
                    log.warning(
                        "phase3_entity_mismatch",
                        expected=entity_id,
                        got=arc.entity_id,
                    )
                    continue

                arc_node_id = f"character_arc_metadata::{entity_id.split('::')[-1]}"
                graph.create_node(
                    arc_node_id,
                    {
                        "type": "character_arc_metadata",
                        "raw_id": entity_id.split("::")[-1],
                        "entity_id": entity_id,
                        "start": arc.start,
                        "pivots": [p.model_dump() for p in arc.pivots],
                        "end_per_path": arc.end_per_path,
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

            # Apply labels to choice specs
            label_lookup = {
                (item.from_passage, item.to_passage): item.label for item in result.choice_labels
            }
            for spec in choice_specs:
                key = (spec["from_passage"], spec["to_passage"])
                if key in label_lookup:
                    spec["label"] = label_lookup[key]

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

            # Apply content hints to residue specs
            hint_lookup = {item.residue_id: item.content_hint for item in result_b.residue_content}
            for spec in residue_specs:
                rid = spec.get("residue_id", "")
                if rid in hint_lookup:
                    spec["content_hint"] = hint_lookup[rid]

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
                    log.warning("phase5c_invalid_index", index=idx)
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

        # Store enriched plan back to graph
        _update_plan_data(
            graph,
            choice_specs=choice_specs,
            residue_specs=residue_specs,
            false_branch_specs=false_branch_specs,
            variant_specs=variant_specs,
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


def _load_plan_data(graph: Graph) -> dict[str, Any] | None:
    """Load the plan node data from the graph."""
    plan_nodes = graph.get_nodes_by_type("polish_plan")
    if not plan_nodes:
        return None
    return plan_nodes.get("polish_plan::current")


def _update_plan_data(
    graph: Graph,
    *,
    choice_specs: list[dict[str, Any]],
    residue_specs: list[dict[str, Any]],
    false_branch_specs: list[dict[str, Any]],
    variant_specs: list[dict[str, Any]],
) -> None:
    """Update the plan node with enriched data from Phase 5."""
    graph.update_node(
        "polish_plan::current",
        choice_specs=choice_specs,
        residue_specs=residue_specs,
        false_branch_specs=false_branch_specs,
        variant_specs=variant_specs,
    )


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


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
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            beat_to_path[edge["from"]] = edge["to"]

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
        _check_consecutive_runs(chain, beat_nodes, beat_to_path, flags)

        # Check for missing sequel after commits
        _check_post_commit_sequel(chain, beat_nodes, beat_to_path, flags)

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
    beat_to_path: dict[str, str],
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
                        "path_id": beat_to_path.get(run_beats[0], ""),
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
                "path_id": beat_to_path.get(run_beats[0], ""),
            }
        )


def _check_post_commit_sequel(
    chain: list[str],
    beat_nodes: dict[str, dict[str, Any]],
    beat_to_path: dict[str, str],
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
                    "path_id": beat_to_path.get(bid, ""),
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
                        "path_id": beat_to_path.get(bid, ""),
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
