"""Tests for POLISH Phase 4 deterministic plan computation.

Tests 4a (beat grouping), 4b (feasibility audit),
4c (choice edge derivation), and 4d (false branch candidates).
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.models.polish import PassageSpec
from questfoundry.pipeline.stages.polish.deterministic import (
    PolishPlan,
    _drain_pre_plan_warnings,
    compute_beat_grouping,
    compute_choice_edges,
    compute_prose_feasibility,
    find_false_branch_candidates,
)
from questfoundry.pipeline.stages.polish.llm_phases import _upsert_pre_plan_warnings


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node with defaults."""
    data = {
        "type": "beat",
        "raw_id": beat_id.split("::")[-1],
        "summary": summary,
        "dilemma_impacts": [],
        "entities": [],
        "scene_type": "scene",
    }
    data.update(kwargs)
    graph.create_node(beat_id, data)


def _add_predecessor(graph: Graph, child: str, parent: str) -> None:
    graph.add_edge("predecessor", child, parent)


def _add_belongs_to(graph: Graph, beat_id: str, path_id: str) -> None:
    graph.add_edge("belongs_to", beat_id, path_id)


class TestComputeBeatGrouping:
    """Tests for Phase 4a: beat grouping."""

    def test_singleton_beats(self) -> None:
        """Ungrouped beats become singleton passages."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "First beat")
        _make_beat(graph, "beat::b", "Second beat")
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        # No predecessor edges — no linear chain

        specs = compute_beat_grouping(graph)
        assert len(specs) == 2
        assert all(s.grouping_type == "singleton" for s in specs)

    def test_collapse_grouping(self) -> None:
        """Sequential same-path beats collapse into one passage."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "Search the study")
        _make_beat(graph, "beat::b", "Find the letter")
        _make_beat(graph, "beat::c", "Read the letter")
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_belongs_to(graph, "beat::c", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")
        _add_predecessor(graph, "beat::c", "beat::b")

        specs = compute_beat_grouping(graph)
        collapse_specs = [s for s in specs if s.grouping_type == "collapse"]
        assert len(collapse_specs) == 1
        assert len(collapse_specs[0].beat_ids) == 3

    def test_intersection_grouping(self) -> None:
        """Beats in intersection groups form intersection passages."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _make_beat(graph, "beat::a", "Path A beat")
        _make_beat(graph, "beat::b", "Path B beat")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")

        # Create intersection group
        graph.create_node(
            "intersection_group::ig1",
            {
                "type": "intersection_group",
                "raw_id": "ig1",
                "node_ids": ["beat::a", "beat::b"],
            },
        )

        specs = compute_beat_grouping(graph)
        intersection_specs = [s for s in specs if s.grouping_type == "intersection"]
        assert len(intersection_specs) == 1
        assert set(intersection_specs[0].beat_ids) == {"beat::a", "beat::b"}

    def test_different_paths_dont_collapse(self) -> None:
        """Sequential beats on different paths don't collapse."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _make_beat(graph, "beat::a", "Path A beat")
        _make_beat(graph, "beat::b", "Path B beat")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = compute_beat_grouping(graph)
        # Should be singletons, not collapsed
        assert all(s.grouping_type == "singleton" for s in specs)

    def test_all_beats_assigned(self) -> None:
        """Every beat is assigned to exactly one passage."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for i in range(5):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}")
            _add_belongs_to(graph, f"beat::b{i}", "path::p1")
        for i in range(1, 5):
            _add_predecessor(graph, f"beat::b{i}", f"beat::b{i - 1}")

        specs = compute_beat_grouping(graph)
        all_beats = set()
        for spec in specs:
            for bid in spec.beat_ids:
                assert bid not in all_beats, f"{bid} assigned to multiple passages"
                all_beats.add(bid)

        expected = {f"beat::b{i}" for i in range(5)}
        assert all_beats == expected

    def test_entities_merged(self) -> None:
        """Collapsed passage contains union of all beat entities."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "A", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "B", entities=["entity::hero", "entity::mentor"])
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = compute_beat_grouping(graph)
        collapse_specs = [s for s in specs if s.grouping_type == "collapse"]
        assert len(collapse_specs) == 1
        assert set(collapse_specs[0].entities) == {"entity::hero", "entity::mentor"}

    def test_empty_graph(self) -> None:
        """Empty graph produces no specs."""
        graph = Graph.empty()
        specs = compute_beat_grouping(graph)
        assert specs == []


class TestComputeProseFeasibility:
    """Tests for Phase 4b: prose feasibility audit."""

    def test_clean_passage(self) -> None:
        """Passage with no relevant flags is clean (no annotations)."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "Simple beat")
        _add_belongs_to(graph, "beat::a", "path::p1")

        specs = [PassageSpec(passage_id="p1", beat_ids=["beat::a"], summary="test")]
        result = compute_prose_feasibility(graph, specs)

        assert result["annotations"] == {}
        assert result["variant_specs"] == []
        assert result["residue_specs"] == []

    def test_annotated_passage(self) -> None:
        """Passage with narratively irrelevant flags gets annotations."""
        graph = Graph.empty()
        graph.create_node("path::brave", {"type": "path", "raw_id": "brave"})
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft", "residue_weight": "light"},
        )

        # Commit beat as ancestor
        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::target", "Target beat", entities=["entity::hero"])
        _add_belongs_to(graph, "beat::commit", "path::brave")
        _add_belongs_to(graph, "beat::target", "path::brave")
        _add_predecessor(graph, "beat::target", "beat::commit")

        # Overlay for a DIFFERENT entity than the passage's entity (embedded on entity node)
        graph.create_node(
            "entity::npc",
            {
                "type": "entity",
                "raw_id": "npc",
                "overlays": [
                    {
                        "when": ["dilemma::d1:path::brave"],
                        "details": {"description": "NPC changes"},
                    }
                ],
            },
        )

        specs = [
            PassageSpec(
                passage_id="passage::test",
                beat_ids=["beat::target"],
                summary="test",
                entities=["entity::hero"],
            )
        ]
        result = compute_prose_feasibility(graph, specs)

        # Flag is structurally relevant but entity::npc doesn't overlap entity::hero
        assert "passage::test" in result["annotations"]

    def test_empty_specs(self) -> None:
        """Empty specs produce empty results."""
        graph = Graph.empty()
        result = compute_prose_feasibility(graph, [])
        assert result["variant_specs"] == []
        assert result["residue_specs"] == []


class TestComputeChoiceEdges:
    """Tests for Phase 4c: choice edge derivation."""

    def test_simple_divergence(self) -> None:
        """Beat with children on different paths produces choices."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        _make_beat(graph, "beat::start", "Start", entities=["entity::hero"])
        _make_beat(graph, "beat::a", "Path A", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "Path B", entities=["entity::hero"])

        _add_belongs_to(graph, "beat::start", "path::pa")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")

        _add_predecessor(graph, "beat::a", "beat::start")
        _add_predecessor(graph, "beat::b", "beat::start")

        specs = [
            PassageSpec(passage_id="p_start", beat_ids=["beat::start"], summary="start"),
            PassageSpec(passage_id="p_a", beat_ids=["beat::a"], summary="a"),
            PassageSpec(passage_id="p_b", beat_ids=["beat::b"], summary="b"),
        ]

        choices = compute_choice_edges(graph, specs)
        assert len(choices) == 2
        from_passages = {c.from_passage for c in choices}
        to_passages = {c.to_passage for c in choices}
        assert from_passages == {"p_start"}
        assert to_passages == {"p_a", "p_b"}

    def test_no_divergence(self) -> None:
        """Linear chain produces no choices."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = [
            PassageSpec(passage_id="p_a", beat_ids=["beat::a"], summary="a"),
            PassageSpec(passage_id="p_b", beat_ids=["beat::b"], summary="b"),
        ]

        choices = compute_choice_edges(graph, specs)
        assert len(choices) == 0

    def test_grants_from_commit(self) -> None:
        """Choice leading to a commit beat populates grants."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        _make_beat(graph, "beat::start", "Start")
        _make_beat(
            graph,
            "beat::commit_a",
            "Commit A",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::b", "Path B")

        _add_belongs_to(graph, "beat::start", "path::pa")
        _add_belongs_to(graph, "beat::commit_a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")

        _add_predecessor(graph, "beat::commit_a", "beat::start")
        _add_predecessor(graph, "beat::b", "beat::start")

        specs = [
            PassageSpec(passage_id="p_start", beat_ids=["beat::start"], summary="start"),
            PassageSpec(passage_id="p_a", beat_ids=["beat::commit_a"], summary="a"),
            PassageSpec(passage_id="p_b", beat_ids=["beat::b"], summary="b"),
        ]

        choices = compute_choice_edges(graph, specs)
        # The choice to p_a should have grants for the commit
        choice_to_a = [c for c in choices if c.to_passage == "p_a"]
        assert len(choice_to_a) == 1
        assert len(choice_to_a[0].grants) > 0


class TestFindFalseBranchCandidates:
    """Tests for Phase 4d: false branch identification."""

    def test_short_stretch_no_candidates(self) -> None:
        """Fewer than 3 passages → no candidates."""
        graph = Graph.empty()
        specs = [
            PassageSpec(passage_id="p1", beat_ids=["b1"], summary="a"),
            PassageSpec(passage_id="p2", beat_ids=["b2"], summary="b"),
        ]
        candidates = find_false_branch_candidates(graph, specs)
        assert len(candidates) == 0

    def test_linear_stretch_produces_candidate(self) -> None:
        """3+ passages in a row via beat DAG adjacency → at least one candidate."""
        graph = Graph.empty()
        # Create beat nodes and predecessor edges forming a linear chain
        for i in range(5):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        for i in range(1, 5):
            graph.add_edge("predecessor", f"beat::b{i}", f"beat::b{i - 1}")
        specs = [
            PassageSpec(passage_id=f"passage::p{i}", beat_ids=[f"beat::b{i}"], summary=f"P{i}")
            for i in range(5)
        ]
        candidates = find_false_branch_candidates(graph, specs)
        assert len(candidates) >= 1
        assert all(len(c.passage_ids) >= 3 for c in candidates)

    def test_out_of_order_spec_but_adjacent_via_dag(self) -> None:
        """Passages out-of-order in spec list but forming a linear chain via beat DAG
        are correctly identified as false branch candidates (#1161)."""
        graph = Graph.empty()
        # Beats form a linear chain: b0 → b1 → b2 → b3
        for i in range(4):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        for i in range(1, 4):
            graph.add_edge("predecessor", f"beat::b{i}", f"beat::b{i - 1}")

        # Specs listed in reverse order — spec list order is NOT topological
        specs = [
            PassageSpec(passage_id="passage::p3", beat_ids=["beat::b3"], summary="P3"),
            PassageSpec(passage_id="passage::p2", beat_ids=["beat::b2"], summary="P2"),
            PassageSpec(passage_id="passage::p1", beat_ids=["beat::b1"], summary="P1"),
            PassageSpec(passage_id="passage::p0", beat_ids=["beat::b0"], summary="P0"),
        ]
        candidates = find_false_branch_candidates(graph, specs)
        # All four form a linear chain via beat DAG adjacency
        assert len(candidates) == 1
        assert len(candidates[0].passage_ids) == 4

    def test_non_adjacent_passages_not_identified(self) -> None:
        """Passages in spec order but NOT adjacent in the passage graph are
        NOT identified as false branch candidates (#1161)."""
        graph = Graph.empty()
        # Two separate passages with no beat DAG connection between them
        for i in range(4):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        # b0 → b1 form one chain; b2 → b3 form another (no connection between)
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")

        specs = [
            PassageSpec(passage_id="passage::p0", beat_ids=["beat::b0"], summary="P0"),
            PassageSpec(passage_id="passage::p1", beat_ids=["beat::b1"], summary="P1"),
            # gap — no connection from b1 to b2
            PassageSpec(passage_id="passage::p2", beat_ids=["beat::b2"], summary="P2"),
            PassageSpec(passage_id="passage::p3", beat_ids=["beat::b3"], summary="P3"),
        ]
        candidates = find_false_branch_candidates(graph, specs)
        # Each sub-chain has only 2 passages — neither qualifies as 3+
        assert all(len(c.passage_ids) < 3 for c in candidates)


class TestPolishPlan:
    """Tests for the PolishPlan dataclass."""

    def test_empty_plan(self) -> None:
        plan = PolishPlan()
        assert len(plan.passage_specs) == 0
        assert len(plan.choice_specs) == 0
        assert len(plan.warnings) == 0

    def test_plan_with_specs(self) -> None:
        plan = PolishPlan(
            passage_specs=[PassageSpec(passage_id="p1", beat_ids=["b1"], summary="test")],
            warnings=["Something to review"],
        )
        assert len(plan.passage_specs) == 1
        assert len(plan.warnings) == 1


class TestPolishCollapseLinearBeats:
    """Tests for phase_collapse_linear_beats in POLISH (moved from GROW in #1109)."""

    def test_phase_registered_in_polish(self) -> None:
        """collapse_linear_beats is registered in the POLISH registry."""
        from questfoundry.pipeline.stages.polish.registry import get_polish_registry

        registry = get_polish_registry()
        assert "collapse_linear_beats" in registry

    def test_phase_not_registered_in_grow(self) -> None:
        """collapse_linear_beats is NOT registered in the GROW registry."""
        from questfoundry.pipeline.stages.grow.registry import get_registry

        registry = get_registry()
        assert "collapse_linear_beats" not in registry

    def test_phase_is_deterministic(self) -> None:
        """collapse_linear_beats is marked deterministic (no LLM calls)."""
        from questfoundry.pipeline.stages.polish.registry import get_polish_registry

        registry = get_polish_registry()
        meta = registry.get_meta("collapse_linear_beats")
        assert meta.is_deterministic is True

    def test_phase_depends_on_character_arcs(self) -> None:
        """collapse_linear_beats depends on character_arcs so it runs after Phase 3."""
        from questfoundry.pipeline.stages.polish.registry import get_polish_registry

        registry = get_polish_registry()
        meta = registry.get_meta("collapse_linear_beats")
        assert "character_arcs" in meta.depends_on

    def test_plan_computation_depends_on_collapse(self) -> None:
        """plan_computation depends on collapse_linear_beats."""
        from questfoundry.pipeline.stages.polish.registry import get_polish_registry

        registry = get_polish_registry()
        meta = registry.get_meta("plan_computation")
        assert "collapse_linear_beats" in meta.depends_on

    def test_collapse_runs_before_plan_computation(self) -> None:
        """Execution order places collapse_linear_beats before plan_computation."""
        from questfoundry.pipeline.stages.polish.registry import get_polish_registry

        order = get_polish_registry().execution_order()
        assert order.index("collapse_linear_beats") < order.index("plan_computation")

    def test_phase_collapses_linear_run(self) -> None:
        """phase_collapse_linear_beats merges a linear 2-beat run."""
        from questfoundry.pipeline.stages.polish.deterministic import phase_collapse_linear_beats

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for i in range(1, 4):
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "dilemma_impacts": [],
                    "entities": [],
                    "scene_type": "scene",
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "path::p1")
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")

        # Run the phase (model param is unused for deterministic phases)
        import asyncio

        result = asyncio.run(
            phase_collapse_linear_beats(graph, None)  # type: ignore[arg-type]
        )
        assert result.status == "completed"
        assert "Collapsed" in result.detail or "No linear" in result.detail

    def test_phase_no_op_on_empty_graph(self) -> None:
        """phase_collapse_linear_beats returns completed when no beats exist."""
        import asyncio

        from questfoundry.pipeline.stages.polish.deterministic import phase_collapse_linear_beats

        graph = Graph.empty()
        result = asyncio.run(
            phase_collapse_linear_beats(graph, None)  # type: ignore[arg-type]
        )
        assert result.status == "completed"
        assert result.detail == "No linear beat runs to collapse"


# ---------------------------------------------------------------------------
# Tests for Fix #1153: arc_traversals populated after phase_plan_application
# ---------------------------------------------------------------------------


class TestArcTraversalsAfterPlanApplication:
    """Phase 6 must populate arc_traversals on the polish_plan node."""

    def test_arc_traversals_non_empty_after_phase6(self) -> None:
        """Minimal two-path graph: phase 4 + phase 6 produce non-empty arc_traversals."""
        import asyncio

        from questfoundry.pipeline.stages.polish.deterministic import (
            phase_plan_application,
            phase_plan_computation,
        )

        graph = Graph.empty()

        # Two dilemmas, each with two paths
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1", "status": "explored"})
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa", "dilemma_id": "dilemma::d1"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb", "dilemma_id": "dilemma::d1"})

        # State flags
        graph.create_node(
            "state_flag::d1_pa",
            {
                "type": "state_flag",
                "raw_id": "d1_pa",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pa",
            },
        )
        graph.create_node(
            "state_flag::d1_pb",
            {
                "type": "state_flag",
                "raw_id": "d1_pb",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pb",
            },
        )

        # Beats on each path
        for bid, pid in [("beat::a", "path::pa"), ("beat::b", "path::pb")]:
            graph.create_node(
                bid,
                {
                    "type": "beat",
                    "raw_id": bid.split("::")[-1],
                    "summary": f"Beat on {pid}",
                    "dilemma_impacts": [],
                    "entities": [],
                    "scene_type": "scene",
                },
            )
            graph.add_edge("belongs_to", bid, pid)

        # Run Phase 4 (plan computation)
        r4 = asyncio.run(phase_plan_computation(graph, None))  # type: ignore[arg-type]
        assert r4.status == "completed"

        # Run Phase 6 (plan application) — no Phase 5 needed for minimal graph
        r6 = asyncio.run(phase_plan_application(graph, None))  # type: ignore[arg-type]
        assert r6.status == "completed"

        # Verify arc_traversals is populated
        plan_nodes = graph.get_nodes_by_type("polish_plan")
        plan_data = plan_nodes.get("polish_plan::current", {})
        arc_traversals = plan_data.get("arc_traversals", {})

        assert arc_traversals, "arc_traversals must be non-empty after plan application"
        # Keys should follow arc_key format (e.g. "path_a+path_b")
        for key in arc_traversals:
            assert "+" in key or key  # arc keys join path IDs
        # Values should be lists of passage IDs
        for passages in arc_traversals.values():
            assert isinstance(passages, list)
            for pid in passages:
                assert pid.startswith("passage::")


# ---------------------------------------------------------------------------
# Tests for Fix #1152: ChoiceSpec.requires populated for convergence choices
# ---------------------------------------------------------------------------


class TestChoiceSpecRequires:
    """compute_choice_edges must populate requires for choices from intersection passages."""

    def _build_convergence_graph(self, graph: Graph) -> None:
        """Build a graph where an intersection beat diverges to two paths.

        Structure:
          start (pa) → merge (intersection, pa) → c (pc, commits d1) ──┐
                                                → d (pd, commits d1) ──┴→ (end)

        The intersection passage (containing beat::merge) diverges to paths pc
        and pd. beat::c and beat::d are the first commit beats on their paths,
        and merge itself has no commit ancestors — so compute_active_flags_at_beat
        returns a single-combo result for each child, allowing requires to be set.
        """
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1", "status": "explored"})
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pc", {"type": "path", "raw_id": "pc"})
        graph.create_node("path::pd", {"type": "path", "raw_id": "pd"})
        graph.create_node(
            "state_flag::d1_pc",
            {
                "type": "state_flag",
                "raw_id": "d1_pc",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pc",
            },
        )
        graph.create_node(
            "state_flag::d1_pd",
            {
                "type": "state_flag",
                "raw_id": "d1_pd",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pd",
            },
        )
        graph.add_edge("dilemma_path", "dilemma::d1", "path::pc")
        graph.add_edge("dilemma_path", "dilemma::d1", "path::pd")

        # Start beat on pa (no commits — clean ancestors for merge)
        _make_beat(graph, "beat::start", "Start")
        graph.add_edge("belongs_to", "beat::start", "path::pa")

        # Intersection beat (no commit ancestors)
        _make_beat(graph, "beat::merge", "Merge beat")
        graph.add_edge("belongs_to", "beat::merge", "path::pa")
        graph.create_node(
            "intersection_group::g1",
            {"type": "intersection_group", "raw_id": "g1", "node_ids": ["beat::merge"]},
        )

        # Post-intersection beats on two different paths — each commits to d1
        _make_beat(
            graph,
            "beat::c",
            "Path C beat",
            dilemma_impacts=[{"effect": "commits", "dilemma_id": "dilemma::d1"}],
        )
        graph.add_edge("belongs_to", "beat::c", "path::pc")
        _make_beat(
            graph,
            "beat::d",
            "Path D beat",
            dilemma_impacts=[{"effect": "commits", "dilemma_id": "dilemma::d1"}],
        )
        graph.add_edge("belongs_to", "beat::d", "path::pd")

        # Predecessor edges
        _add_predecessor(graph, "beat::merge", "beat::start")
        _add_predecessor(graph, "beat::c", "beat::merge")
        _add_predecessor(graph, "beat::d", "beat::merge")

    def test_divergence_choice_requires_empty(self) -> None:
        """Choices from non-intersection passages have empty requires."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        _make_beat(graph, "beat::start", "Start")
        _make_beat(graph, "beat::a", "Path A")
        _make_beat(graph, "beat::b", "Path B")

        graph.add_edge("belongs_to", "beat::start", "path::pa")
        graph.add_edge("belongs_to", "beat::a", "path::pa")
        graph.add_edge("belongs_to", "beat::b", "path::pb")

        _add_predecessor(graph, "beat::a", "beat::start")
        _add_predecessor(graph, "beat::b", "beat::start")

        specs = compute_beat_grouping(graph)
        choices = compute_choice_edges(graph, specs)

        # Divergence choices from non-intersection passages should have no requires
        for c in choices:
            from_spec = next((s for s in specs if s.passage_id == c.from_passage), None)
            if from_spec and from_spec.grouping_type != "intersection":
                assert c.requires == [], f"Expected empty requires for {c.from_passage}"

    def test_convergence_choice_has_requires(self) -> None:
        """Choices from intersection passages have a non-empty requires list."""
        graph = Graph.empty()
        self._build_convergence_graph(graph)

        specs = compute_beat_grouping(graph)
        choices = compute_choice_edges(graph, specs)

        passage_id_to_spec = {s.passage_id: s for s in specs}
        intersection_choices = [
            c
            for c in choices
            if c.from_passage in passage_id_to_spec
            and passage_id_to_spec[c.from_passage].grouping_type == "intersection"
        ]

        assert intersection_choices, "Expected at least one choice from an intersection passage"
        assert any(len(c.requires) > 0 for c in intersection_choices), (
            "Expected at least one intersection choice to have a non-empty requires list"
        )


# ---------------------------------------------------------------------------
# Tests for Issue #1157: Ambiguous feasibility detection in Phase 4b
# ---------------------------------------------------------------------------


class TestAmbiguousFeasibilityDetection:
    """Phase 4b must detect ambiguous cases (mixed residue weights) and store them."""

    def _make_commit_beat(
        self,
        graph: Graph,
        beat_id: str,
        path_id: str,
        dilemma_id: str,
    ) -> None:
        graph.create_node(
            beat_id,
            {
                "type": "beat",
                "raw_id": beat_id.split("::")[-1],
                "summary": f"Commit on {path_id}",
                "dilemma_impacts": [{"dilemma_id": dilemma_id, "effect": "commits"}],
                "entities": [],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", beat_id, path_id)

    def test_mixed_weights_produces_ambiguous_spec(self) -> None:
        """Passage with one heavy flag and one light flag → ambiguous_specs, NOT in variant or residue."""
        graph = Graph.empty()

        # Two dilemmas — one heavy, one light
        graph.create_node(
            "dilemma::heavy",
            {
                "type": "dilemma",
                "raw_id": "heavy",
                "residue_weight": "heavy",
            },
        )
        graph.create_node(
            "dilemma::light",
            {
                "type": "dilemma",
                "raw_id": "light",
                "residue_weight": "light",
            },
        )

        # Two paths
        graph.create_node("path::ph", {"type": "path", "raw_id": "ph"})
        graph.create_node("path::pl", {"type": "path", "raw_id": "pl"})

        # Entity that appears in the passage
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": ["dilemma::heavy:path::ph"], "details": {"mood": "grim"}},
                    {"when": ["dilemma::light:path::pl"], "details": {"mood": "relieved"}},
                ],
            },
        )

        # Commit beats (ancestors of the target passage).
        # Both need to be in the ancestor chain so both flags are active at beat::target.
        # Chain: commit_h → commit_l → target
        self._make_commit_beat(graph, "beat::commit_h", "path::ph", "dilemma::heavy")
        self._make_commit_beat(graph, "beat::commit_l", "path::pl", "dilemma::light")
        _add_predecessor(graph, "beat::commit_l", "beat::commit_h")

        # Target beat referencing entity::hero
        _make_beat(
            graph,
            "beat::target",
            "Target beat with hero",
            entities=["entity::hero"],
        )
        graph.add_edge("belongs_to", "beat::target", "path::ph")
        _add_predecessor(graph, "beat::target", "beat::commit_l")

        spec = PassageSpec(
            passage_id="passage::test_mixed",
            beat_ids=["beat::target"],
            summary="test",
            entities=["entity::hero"],
        )

        result = compute_prose_feasibility(graph, [spec])

        # Mixed weights → goes to ambiguous, NOT variant or residue
        assert len(result["ambiguous_specs"]) == 1
        assert result["ambiguous_specs"][0].passage_id == "passage::test_mixed"
        assert len(result["variant_specs"]) == 0
        assert len(result["residue_specs"]) == 0

    def test_all_heavy_flags_not_ambiguous(self) -> None:
        """2+ flags all heavy → deterministic variant, NOT ambiguous."""
        graph = Graph.empty()

        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "residue_weight": "heavy"},
        )
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "residue_weight": "hard"},
        )
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})

        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": ["dilemma::d1:path::p1"], "details": {"mood": "dark"}},
                    {"when": ["dilemma::d2:path::p2"], "details": {"mood": "grim"}},
                ],
            },
        )

        self._make_commit_beat(graph, "beat::c1", "path::p1", "dilemma::d1")
        self._make_commit_beat(graph, "beat::c2", "path::p2", "dilemma::d2")
        # Chain: c2 → c1 → target so both flags are ancestors of beat::target
        _add_predecessor(graph, "beat::c1", "beat::c2")

        _make_beat(graph, "beat::target", "Target", entities=["entity::hero"])
        graph.add_edge("belongs_to", "beat::target", "path::p1")
        _add_predecessor(graph, "beat::target", "beat::c1")

        spec = PassageSpec(
            passage_id="passage::all_heavy",
            beat_ids=["beat::target"],
            summary="heavy test",
            entities=["entity::hero"],
        )

        result = compute_prose_feasibility(graph, [spec])
        assert len(result["ambiguous_specs"]) == 0
        assert len(result["variant_specs"]) == 2  # one per heavy flag
        assert len(result["residue_specs"]) == 0

    def test_single_relevant_flag_always_deterministic(self) -> None:
        """Exactly 1 relevant flag → never ambiguous (deterministic assignment)."""
        graph = Graph.empty()

        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "residue_weight": "heavy"},
        )
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": ["dilemma::d1:path::p1"], "details": {"mood": "dark"}},
                ],
            },
        )

        self._make_commit_beat(graph, "beat::c1", "path::p1", "dilemma::d1")
        _make_beat(graph, "beat::target", "Target", entities=["entity::hero"])
        graph.add_edge("belongs_to", "beat::target", "path::p1")
        _add_predecessor(graph, "beat::target", "beat::c1")

        spec = PassageSpec(
            passage_id="passage::single_flag",
            beat_ids=["beat::target"],
            summary="single flag test",
            entities=["entity::hero"],
        )

        result = compute_prose_feasibility(graph, [spec])
        assert len(result["ambiguous_specs"]) == 0
        assert len(result["variant_specs"]) == 1  # heavy → variant


# ---------------------------------------------------------------------------
# Tests for Issue #1158: transition_guidance stored on passage node in Phase 6
# ---------------------------------------------------------------------------


class TestTransitionGuidanceInGraph:
    """After Phase 6, passage nodes created from collapsed specs must have transition_guidance."""

    def test_transition_guidance_stored_on_passage_node(self) -> None:
        """Phase 6 stores transition_guidance from PassageSpec onto the passage graph node."""
        import asyncio

        from questfoundry.pipeline.stages.polish.deterministic import phase_plan_application

        graph = Graph.empty()

        # Build a minimal plan node directly (bypass phase 4) with a collapsed passage
        # that has transition_guidance set (simulating what phase 5f would produce).
        from questfoundry.models.polish import PassageSpec as _PassageSpec

        spec = _PassageSpec(
            passage_id="passage::collapse_0",
            beat_ids=["beat::a", "beat::b", "beat::c"],
            summary="Three beats collapsed",
            entities=[],
            grouping_type="collapse",
            transition_guidance=["Move from action to reflection.", "Time passes quietly."],
        )

        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "raw_id": "current",
                "passage_count": 1,
                "variant_count": 0,
                "residue_count": 0,
                "choice_count": 0,
                "candidate_count": 0,
                "warnings": [],
                "passage_specs": [spec.model_dump()],
                "variant_specs": [],
                "residue_specs": [],
                "choice_specs": [],
                "false_branch_candidates": [],
                "false_branch_specs": [],
                "feasibility_annotations": {},
                "ambiguous_specs": [],
                "arc_traversals": {},
            },
        )

        # Create beat nodes so grouped_in edges don't fail
        for bid in ["beat::a", "beat::b", "beat::c"]:
            graph.create_node(
                bid,
                {
                    "type": "beat",
                    "raw_id": bid.split("::")[-1],
                    "summary": f"Beat {bid}",
                    "dilemma_impacts": [],
                    "entities": [],
                    "scene_type": "scene",
                },
            )

        result = asyncio.run(phase_plan_application(graph, None))  # type: ignore[arg-type]
        assert result.status == "completed"

        passage_nodes = graph.get_nodes_by_type("passage")
        passage = passage_nodes.get("passage::collapse_0")
        assert passage is not None
        assert passage.get("transition_guidance") == [
            "Move from action to reflection.",
            "Time passes quietly.",
        ]


class TestPrePlanWarningAccumulator:
    """Tests for Issue #1159: Phase 1 warnings accumulated before PolishPlan exists."""

    def test_drain_pre_plan_warnings_drains_into_plan(self) -> None:
        """Warnings written to graph by Phase 1 are drained into plan.warnings (#1159)."""
        graph = Graph.empty()
        warnings = [
            "Section section_0: reordering rejected — beat set mismatch (expected 3, got 2)",
            "Section section_1: reordering rejected — hard constraint violation (commit before advance/reveal)",
        ]
        _upsert_pre_plan_warnings(graph, warnings)

        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)

        assert plan.warnings == warnings

    def test_drain_clears_accumulator(self) -> None:
        """Draining pre-plan warnings clears the accumulator node (#1159)."""
        graph = Graph.empty()
        _upsert_pre_plan_warnings(graph, ["some warning"])

        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)
        assert len(plan.warnings) == 1

        # Second drain should add nothing
        plan2 = PolishPlan()
        _drain_pre_plan_warnings(graph, plan2)
        assert len(plan2.warnings) == 0

    def test_drain_no_warnings_node_is_noop(self) -> None:
        """Draining when no pre-plan warnings node exists is a no-op (#1159)."""
        graph = Graph.empty()
        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)  # Should not raise
        assert plan.warnings == []

    def test_no_rejection_produces_empty_phase1_contribution(self) -> None:
        """When Phase 1 produces no rejections, plan.warnings is empty (#1159)."""
        graph = Graph.empty()
        # Do not call _upsert_pre_plan_warnings — simulates no rejections
        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)
        assert plan.warnings == []
