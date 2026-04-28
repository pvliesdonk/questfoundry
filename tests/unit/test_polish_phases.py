"""Tests for POLISH LLM phase helpers (deterministic logic).

Tests the pure functions that support Phases 1-3, not the LLM calls
themselves. LLM integration is tested separately.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from questfoundry.graph.graph import Graph
from questfoundry.models.polish import ArcPivot, CharacterArcMetadata, Phase3Output
from questfoundry.pipeline.stages.polish.llm_phases import (
    _check_consecutive_runs,
    _check_post_commit_sequel,
    _collect_entity_appearances,
    _detect_pacing_flags,
    _find_linear_sections,
    _PolishLLMPhaseMixin,
    _topological_sort,
    _validate_reorder_constraints,
)


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node."""
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
    """Helper to create a predecessor edge."""
    graph.add_edge("predecessor", child, parent)


class TestFindLinearSections:
    """Tests for _find_linear_sections."""

    def test_simple_chain(self) -> None:
        """Linear chain of 4 beats → one section."""
        graph = Graph.empty()
        for i in range(4):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}")
        for i in range(1, 4):
            _add_predecessor(graph, f"beat::b{i}", f"beat::b{i - 1}")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        assert len(sections) == 1
        assert len(sections[0]["beat_ids"]) == 4

    def test_short_chain_excluded(self) -> None:
        """Chain of 2 beats is too short → no sections."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")
        _add_predecessor(graph, "beat::b", "beat::a")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        assert len(sections) == 0

    def test_branching_splits_sections(self) -> None:
        """Branch point splits the DAG into separate sections.

        Structure: a → b → c (one parent)
                   a → d → e → f (branch)
        """
        graph = Graph.empty()
        for name in ["a", "b", "c", "d", "e", "f"]:
            _make_beat(graph, f"beat::{name}", f"Beat {name}")

        _add_predecessor(graph, "beat::b", "beat::a")
        _add_predecessor(graph, "beat::c", "beat::b")
        _add_predecessor(graph, "beat::d", "beat::a")
        _add_predecessor(graph, "beat::e", "beat::d")
        _add_predecessor(graph, "beat::f", "beat::e")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        # The branch from a splits — d→e→f forms a linear section of 3
        # a→b→c doesn't form a section because a has 2 children (b, d)
        section_lengths = sorted(len(s["beat_ids"]) for s in sections)
        assert 3 in section_lengths

    def test_before_after_context(self) -> None:
        """Section tracks before/after beats for context."""
        graph = Graph.empty()
        for i in range(5):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}")
        for i in range(1, 5):
            _add_predecessor(graph, f"beat::b{i}", f"beat::b{i - 1}")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        assert len(sections) == 1
        # First beat has no predecessor, last beat has no successor
        assert sections[0]["before_beat"] is None
        assert sections[0]["after_beat"] is None

    def test_no_beats(self) -> None:
        """Empty graph → no sections."""
        sections = _find_linear_sections({}, [])
        assert sections == []


class TestValidateReorderConstraints:
    """Tests for _validate_reorder_constraints."""

    def test_valid_reorder(self) -> None:
        """Reorder that preserves constraints passes."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Setup")
        _make_beat(
            graph,
            "beat::b",
            "Advance",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "advances"}],
        )
        _make_beat(
            graph,
            "beat::c",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "commits"}],
        )

        # Original: a, b, c. Proposed: a, b, c (same — valid)
        assert _validate_reorder_constraints(
            graph, ["beat::a", "beat::b", "beat::c"], ["beat::a", "beat::b", "beat::c"]
        )

    def test_commit_before_advance_rejected(self) -> None:
        """Reorder putting commit before advance fails."""
        graph = Graph.empty()
        _make_beat(
            graph,
            "beat::advance",
            "Advance",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "advances"}],
        )
        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::other", "Other")

        # Proposed: commit before advance — should fail
        assert not _validate_reorder_constraints(
            graph,
            ["beat::advance", "beat::commit", "beat::other"],
            ["beat::commit", "beat::advance", "beat::other"],
        )

    def test_no_dilemma_impacts_always_valid(self) -> None:
        """Beats without dilemma impacts can be freely reordered."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")
        _make_beat(graph, "beat::c", "C")

        assert _validate_reorder_constraints(
            graph, ["beat::a", "beat::b", "beat::c"], ["beat::c", "beat::a", "beat::b"]
        )


class TestDetectPacingFlags:
    """Tests for _detect_pacing_flags."""

    def test_three_consecutive_scenes(self) -> None:
        """Three scene beats in a row triggers a flag."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for i in range(3):
            _make_beat(graph, f"beat::s{i}", f"Scene {i}", scene_type="scene")
            graph.add_edge("belongs_to", f"beat::s{i}", "path::p1")
        _add_predecessor(graph, "beat::s1", "beat::s0")
        _add_predecessor(graph, "beat::s2", "beat::s1")

        edges = graph.get_edges(edge_type="predecessor")
        beat_nodes = graph.get_nodes_by_type("beat")

        flags = _detect_pacing_flags(beat_nodes, edges, graph)
        assert len(flags) >= 1
        assert any(f["issue_type"] == "consecutive_scene" for f in flags)

    def test_mixed_types_no_flag(self) -> None:
        """Alternating scene/sequel doesn't trigger flags."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        types = ["scene", "sequel", "scene"]
        for i, st in enumerate(types):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}", scene_type=st)
            graph.add_edge("belongs_to", f"beat::b{i}", "path::p1")
        _add_predecessor(graph, "beat::b1", "beat::b0")
        _add_predecessor(graph, "beat::b2", "beat::b1")

        edges = graph.get_edges(edge_type="predecessor")
        beat_nodes = graph.get_nodes_by_type("beat")

        flags = _detect_pacing_flags(beat_nodes, edges, graph)
        # No consecutive runs of 3+
        consecutive_flags = [f for f in flags if "consecutive" in f["issue_type"]]
        assert len(consecutive_flags) == 0

    def test_no_sequel_after_commit(self) -> None:
        """Commit beat followed by non-sequel triggers a flag."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            scene_type="scene",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::action", "Action", scene_type="scene")
        graph.add_edge("belongs_to", "beat::commit", "path::p1")
        graph.add_edge("belongs_to", "beat::action", "path::p1")
        _add_predecessor(graph, "beat::action", "beat::commit")

        edges = graph.get_edges(edge_type="predecessor")
        beat_nodes = graph.get_nodes_by_type("beat")

        flags = _detect_pacing_flags(beat_nodes, edges, graph)
        assert any(f["issue_type"] == "no_sequel_after_commit" for f in flags)


class TestConsecutiveRuns:
    """Tests for _check_consecutive_runs helper."""

    def test_detects_scene_run(self) -> None:
        beat_nodes = {
            "a": {"scene_type": "scene"},
            "b": {"scene_type": "scene"},
            "c": {"scene_type": "scene"},
        }
        flags: list = []
        _check_consecutive_runs(["a", "b", "c"], beat_nodes, lambda _bid: "", flags)
        assert len(flags) == 1
        assert flags[0]["issue_type"] == "consecutive_scene"

    def test_detects_sequel_run(self) -> None:
        beat_nodes = {
            "a": {"scene_type": "sequel"},
            "b": {"scene_type": "sequel"},
            "c": {"scene_type": "sequel"},
        }
        flags: list = []
        _check_consecutive_runs(["a", "b", "c"], beat_nodes, lambda _bid: "", flags)
        assert len(flags) == 1
        assert flags[0]["issue_type"] == "consecutive_sequel"

    def test_short_chain_ignored(self) -> None:
        flags: list = []
        _check_consecutive_runs(["a", "b"], {"a": {}, "b": {}}, lambda _bid: "", flags)
        assert len(flags) == 0


class TestPostCommitSequel:
    """Tests for _check_post_commit_sequel helper."""

    def test_commit_followed_by_sequel_ok(self) -> None:
        beat_nodes = {
            "a": {
                "dilemma_impacts": [{"effect": "commits"}],
                "scene_type": "scene",
            },
            "b": {"dilemma_impacts": [], "scene_type": "sequel"},
        }
        flags: list = []
        _check_post_commit_sequel(["a", "b"], beat_nodes, lambda _bid: "", flags)
        assert len(flags) == 0

    def test_commit_followed_by_scene_flagged(self) -> None:
        beat_nodes = {
            "a": {
                "dilemma_impacts": [{"effect": "commits"}],
                "scene_type": "scene",
            },
            "b": {"dilemma_impacts": [], "scene_type": "scene"},
        }
        flags: list = []
        _check_post_commit_sequel(["a", "b"], beat_nodes, lambda _bid: "", flags)
        assert len(flags) == 1

    def test_commit_at_end_of_chain_flagged(self) -> None:
        """Commit as the last beat in a chain triggers no-sequel flag."""
        beat_nodes = {
            "a": {"dilemma_impacts": [], "scene_type": "scene"},
            "b": {
                "dilemma_impacts": [{"effect": "commits"}],
                "scene_type": "scene",
            },
        }
        flags: list = []
        _check_post_commit_sequel(["a", "b"], beat_nodes, lambda _bid: "", flags)
        assert len(flags) == 1
        assert flags[0]["issue_type"] == "no_sequel_after_commit"


class TestTopologicalSort:
    """Tests for _topological_sort."""

    def test_linear_chain(self) -> None:
        beat_nodes = {"a": {}, "b": {}, "c": {}}
        edges = [
            {"from": "b", "to": "a", "type": "predecessor"},
            {"from": "c", "to": "b", "type": "predecessor"},
        ]
        result = _topological_sort(beat_nodes, edges)
        assert result == ["a", "b", "c"]

    def test_single_node(self) -> None:
        result = _topological_sort({"x": {}}, [])
        assert result == ["x"]

    def test_diamond(self) -> None:
        """Diamond DAG: a → b, a → c, b → d, c → d."""
        beat_nodes = {"a": {}, "b": {}, "c": {}, "d": {}}
        edges = [
            {"from": "b", "to": "a", "type": "predecessor"},
            {"from": "c", "to": "a", "type": "predecessor"},
            {"from": "d", "to": "b", "type": "predecessor"},
            {"from": "d", "to": "c", "type": "predecessor"},
        ]
        result = _topological_sort(beat_nodes, edges)
        assert result[0] == "a"
        assert result[-1] == "d"
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")


class TestCollectEntityAppearances:
    """Tests for _collect_entity_appearances."""

    def test_entities_collected_in_order(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "First", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "Second", entities=["entity::hero", "entity::npc"])
        _make_beat(graph, "beat::c", "Third", entities=["entity::npc"])
        _add_predecessor(graph, "beat::b", "beat::a")
        _add_predecessor(graph, "beat::c", "beat::b")

        beat_nodes = graph.get_nodes_by_type("beat")
        result = _collect_entity_appearances(beat_nodes, graph)

        assert result["entity::hero"] == ["beat::a", "beat::b"]
        assert result["entity::npc"] == ["beat::b", "beat::c"]

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        result = _collect_entity_appearances({}, graph)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests for Phase 3 has_arc_metadata edge (Fix #1154)
# ---------------------------------------------------------------------------


class _FakePolishLLMHost(_PolishLLMPhaseMixin):
    """Minimal host that satisfies _PolishLLMHelperMixin by providing _polish_llm_call."""

    def __init__(self, llm_result: object) -> None:
        self._llm_result = llm_result

    async def _polish_llm_call(
        self,
        model: object,  # noqa: ARG002
        template_name: str,  # noqa: ARG002
        context: str,  # noqa: ARG002
        output_schema: object,  # noqa: ARG002
    ) -> tuple[object, int, int]:
        return (self._llm_result, 1, 100)


class TestPhase3CharacterArcField:
    """Phase 3 must annotate Entity nodes with character_arc field per R-3.3."""

    def test_character_arc_field_on_entity(self) -> None:
        """R-3.3: Phase 3 annotates Entity node's data dict with 'character_arc';
        no separate 'character_arc_metadata' node and no 'has_arc_metadata' edge
        are created."""
        graph = Graph.empty()

        # Create entity node
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "name": "Hero"},
        )

        # Create two beats so entity qualifies as arc-worthy
        _make_beat(graph, "beat::a", "First beat", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "Second beat", entities=["entity::hero"])
        _add_predecessor(graph, "beat::b", "beat::a")

        # Create a path (needed to make end_per_path work)
        graph.create_node(
            "path::brave",
            {"type": "path", "raw_id": "brave"},
        )

        # Phase 3 output
        arc_output = Phase3Output(
            character_arcs=[
                CharacterArcMetadata(
                    entity_id="entity::hero",
                    start="Nervous newcomer",
                    pivots=[
                        ArcPivot(
                            path_id="path::brave",
                            beat_id="beat::b",
                            description="Gains confidence",
                        )
                    ],
                    end_per_path={"path::brave": "Confident hero"},
                )
            ]
        )

        host = _FakePolishLLMHost(arc_output)
        result = asyncio.run(host._phase_3_character_arcs(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"

        # Verify character_arc field exists on entity node
        entity_data = graph.get_node("entity::hero")
        assert entity_data is not None
        assert "character_arc" in entity_data
        arc = entity_data["character_arc"]
        assert "start" in arc
        assert arc["start"] == "Nervous newcomer"
        assert "pivots" in arc
        assert "path::brave" in arc["pivots"]
        assert arc["pivots"]["path::brave"] == "beat::b"
        assert "end_per_path" in arc
        assert arc["end_per_path"]["path::brave"] == "Confident hero"

        # And the forbidden shapes are absent:
        assert graph.get_nodes_by_type("character_arc_metadata") == {}
        assert graph.get_edges(edge_type="has_arc_metadata") == []


# ---------------------------------------------------------------------------
# Tests for Issue #1157: Phase 5e — ambiguous feasibility resolution
# ---------------------------------------------------------------------------


class TestPhase5eAmbiguousFeasibility:
    """Phase 5e must resolve ambiguous cases via LLM and apply decisions to plan."""

    def _build_plan_with_ambiguous(self, graph: Graph) -> None:
        """Build a minimal polish_plan node with one ambiguous feasibility case."""
        from questfoundry.models.polish import AmbiguousFeasibilityCase, PassageSpec

        passage = PassageSpec(
            passage_id="passage::ambig_0",
            beat_ids=["beat::a"],
            summary="Ambiguous passage",
            entities=["entity::hero"],
            grouping_type="singleton",
        )
        ambiguous = AmbiguousFeasibilityCase(
            passage_id="passage::ambig_0",
            passage_summary="Ambiguous passage",
            entities=["entity::hero"],
            flags=["dilemma::heavy:path::ph"],
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
                "passage_specs": [passage.model_dump()],
                "variant_specs": [],
                "residue_specs": [],
                "choice_specs": [],
                "false_branch_candidates": [],
                "false_branch_specs": [],
                "feasibility_annotations": {},
                "ambiguous_specs": [ambiguous.model_dump()],
                "arc_traversals": {},
            },
        )

    def test_variant_decision_adds_variant_spec(self) -> None:
        """LLM decision 'variant' → new VariantSpec appended to plan."""
        from questfoundry.models.polish import FeasibilityDecisionItem, Phase5eOutput

        graph = Graph.empty()
        self._build_plan_with_ambiguous(graph)

        llm_output = Phase5eOutput(
            feasibility_decisions=[
                FeasibilityDecisionItem(
                    passage_id="passage::ambig_0",
                    flag_index=0,
                    decision="variant",
                )
            ]
        )

        host = _FakePolishLLMHost(llm_output)
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"

        # Plan node should have a variant_spec for the resolved case
        plan_nodes = graph.get_nodes_by_type("polish_plan")
        plan_data = plan_nodes.get("polish_plan::current", {})
        variant_specs = plan_data.get("variant_specs", [])
        assert len(variant_specs) == 1
        assert variant_specs[0]["base_passage_id"] == "passage::ambig_0"
        assert variant_specs[0]["requires"] == ["dilemma::heavy:path::ph"]

    def test_residue_decision_adds_residue_spec(self) -> None:
        """LLM decision 'residue' → new ResidueSpec appended to plan."""
        from questfoundry.models.polish import FeasibilityDecisionItem, Phase5eOutput

        graph = Graph.empty()
        self._build_plan_with_ambiguous(graph)

        llm_output = Phase5eOutput(
            feasibility_decisions=[
                FeasibilityDecisionItem(
                    passage_id="passage::ambig_0",
                    flag_index=0,
                    decision="residue",
                )
            ]
        )

        host = _FakePolishLLMHost(llm_output)
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"

        plan_nodes = graph.get_nodes_by_type("polish_plan")
        plan_data = plan_nodes.get("polish_plan::current", {})
        residue_specs = plan_data.get("residue_specs", [])
        assert len(residue_specs) == 1
        assert residue_specs[0]["target_passage_id"] == "passage::ambig_0"
        assert residue_specs[0]["flag"] == "dilemma::heavy:path::ph"

    def test_irrelevant_decision_updates_feasibility_annotations(self) -> None:
        """LLM decision 'irrelevant' → flag recorded in feasibility_annotations, not variant/residue."""
        from questfoundry.models.polish import FeasibilityDecisionItem, Phase5eOutput

        graph = Graph.empty()
        self._build_plan_with_ambiguous(graph)

        llm_output = Phase5eOutput(
            feasibility_decisions=[
                FeasibilityDecisionItem(
                    passage_id="passage::ambig_0",
                    flag_index=0,
                    decision="irrelevant",
                )
            ]
        )

        host = _FakePolishLLMHost(llm_output)
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"

        plan_nodes = graph.get_nodes_by_type("polish_plan")
        plan_data = plan_nodes.get("polish_plan::current", {})

        # Flag recorded in feasibility_annotations for the passage
        annotations = plan_data.get("feasibility_annotations", {})
        assert "passage::ambig_0" in annotations
        assert "dilemma::heavy:path::ph" in annotations["passage::ambig_0"]

        # Nothing added to variant or residue specs
        assert len(plan_data.get("variant_specs", [])) == 0

    def test_skipped_when_no_ambiguous_cases(self) -> None:
        """Phase 5e is skipped when ambiguous_specs is empty."""
        from questfoundry.models.polish import Phase5eOutput

        graph = Graph.empty()
        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "raw_id": "current",
                "passage_count": 0,
                "variant_count": 0,
                "residue_count": 0,
                "choice_count": 0,
                "candidate_count": 0,
                "warnings": [],
                "passage_specs": [],
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

        host = _FakePolishLLMHost(Phase5eOutput())
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"
        assert "0 ambiguous cases resolved" in result.detail


class _RecordingPolishLLMHost(_PolishLLMPhaseMixin):
    """Like _FakePolishLLMHost, but records (template_name, context, schema) per call.

    Used by Phase 5a tests to assert which specs were forwarded to the LLM.
    """

    def __init__(self, results_by_template: dict[str, object]) -> None:
        self._results = results_by_template
        self.calls: list[tuple[str, object, object]] = []

    async def _polish_llm_call(
        self,
        model: object,  # noqa: ARG002
        template_name: str,
        context: object,
        output_schema: object,
    ) -> tuple[object, int, int]:
        self.calls.append((template_name, context, output_schema))
        result = self._results.get(template_name)
        if result is None:
            raise AssertionError(f"Unexpected template call: {template_name}")
        return (result, 1, 100)


class TestPhase5aContinueFiltering:
    """Phase 5a must NOT forward Continue (R-4c.7) choices to the labeling LLM.

    Continue edges already carry the literal label `"Continue"`, set by
    `compute_choice_edges`. Sending them to the LLM wastes tokens and risks
    relabeling them to diegetic phrases.
    """

    def _build_plan_with_mixed_choices(self, graph: Graph) -> None:
        from questfoundry.models.polish import ChoiceSpec, PassageSpec

        passages = [
            PassageSpec(
                passage_id=f"passage::p{i}",
                beat_ids=[f"beat::b{i}"],
                summary=f"summary {i}",
                entities=[],
                grouping_type="singleton",
            )
            for i in range(3)
        ]

        # Two choices: one fork (no label yet — to be filled by 5a) and one
        # Continue (already labeled by R-4c.7).
        choices = [
            ChoiceSpec(
                from_passage="passage::p0",
                to_passage="passage::p1",
                grants=[],
                requires=[],
                label="",
            ).model_dump(),
            ChoiceSpec(
                from_passage="passage::p1",
                to_passage="passage::p2",
                grants=[],
                requires=[],
                label="Continue",
            ).model_dump(),
        ]

        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "raw_id": "current",
                "passage_count": 3,
                "variant_count": 0,
                "residue_count": 0,
                "choice_count": 2,
                "candidate_count": 0,
                "warnings": [],
                "passage_specs": [p.model_dump() for p in passages],
                "variant_specs": [],
                "residue_specs": [],
                "choice_specs": choices,
                "false_branch_candidates": [],
                "false_branch_specs": [],
                "feasibility_annotations": {},
                "ambiguous_specs": [],
                "arc_traversals": {},
            },
        )

    def test_continue_specs_not_forwarded_to_llm(self) -> None:
        from questfoundry.models.polish import (
            ChoiceLabelItem,
            Phase5aOutput,
        )

        graph = Graph.empty()
        self._build_plan_with_mixed_choices(graph)

        labeled = Phase5aOutput(
            choice_labels=[
                ChoiceLabelItem(
                    from_passage="passage::p0",
                    to_passage="passage::p1",
                    label="Trust the mentor",
                )
            ]
        )

        host = _RecordingPolishLLMHost({"polish_phase5a_choice_labels": labeled})
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]
        assert result.status == "completed"

        # Locate the Phase 5a call. The context dict carries "choice_count".
        phase5a_calls = [c for c in host.calls if c[0] == "polish_phase5a_choice_labels"]
        assert len(phase5a_calls) == 1
        _, context, _ = phase5a_calls[0]
        assert isinstance(context, dict)
        assert context["choice_count"] == "1"  # Continue spec excluded

        # The fork choice should now carry the LLM-supplied label.
        plan_data = graph.get_nodes_by_type("polish_plan")["polish_plan::current"]
        labels = {
            (c["from_passage"], c["to_passage"]): c["label"] for c in plan_data["choice_specs"]
        }
        assert labels[("passage::p0", "passage::p1")] == "Trust the mentor"
        # Continue label preserved exactly.
        assert labels[("passage::p1", "passage::p2")] == "Continue"

    def test_only_continue_choices_skips_phase5a_llm_call(self) -> None:
        """When every choice is a Continue, Phase 5a must not invoke the LLM."""
        from questfoundry.models.polish import ChoiceSpec, PassageSpec

        graph = Graph.empty()
        passages = [
            PassageSpec(
                passage_id=f"passage::p{i}",
                beat_ids=[f"beat::b{i}"],
                summary=f"summary {i}",
                entities=[],
                grouping_type="singleton",
            )
            for i in range(2)
        ]
        choices = [
            ChoiceSpec(
                from_passage="passage::p0",
                to_passage="passage::p1",
                grants=[],
                requires=[],
                label="Continue",
            ).model_dump(),
        ]
        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "raw_id": "current",
                "passage_count": 2,
                "variant_count": 0,
                "residue_count": 0,
                "choice_count": 1,
                "candidate_count": 0,
                "warnings": [],
                "passage_specs": [p.model_dump() for p in passages],
                "variant_specs": [],
                "residue_specs": [],
                "choice_specs": choices,
                "false_branch_candidates": [],
                "false_branch_specs": [],
                "feasibility_annotations": {},
                "ambiguous_specs": [],
                "arc_traversals": {},
            },
        )

        host = _RecordingPolishLLMHost({})  # no Phase 5a result registered
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]
        assert result.status == "completed"

        phase5a_calls = [c for c in host.calls if c[0] == "polish_phase5a_choice_labels"]
        assert phase5a_calls == []


# ---------------------------------------------------------------------------
# Tests for Issue #1158: Phase 5f — transition guidance for collapsed passages
# ---------------------------------------------------------------------------


class TestPhase5fTransitionGuidance:
    """Phase 5f must generate and apply transition guidance for collapsed passages."""

    def _build_plan_with_collapsed(self, graph: Graph) -> None:
        """Build a minimal polish_plan node with one collapsed passage (3 beats)."""
        from questfoundry.models.polish import PassageSpec

        passage = PassageSpec(
            passage_id="passage::collapse_0",
            beat_ids=["beat::a", "beat::b", "beat::c"],
            summary="Three beats in one scene",
            entities=["entity::hero"],
            grouping_type="collapse",
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
                "passage_specs": [passage.model_dump()],
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

    def test_transition_guidance_applied_to_passage_spec(self) -> None:
        """LLM returns 2 transitions for 3-beat collapsed passage → spec updated."""
        from questfoundry.models.polish import Phase5fOutput, TransitionGuidanceItem

        graph = Graph.empty()
        self._build_plan_with_collapsed(graph)

        llm_output = Phase5fOutput(
            transition_guidance=[
                TransitionGuidanceItem(
                    passage_id="passage::collapse_0",
                    transitions=[
                        "As the dust settles, a new threat emerges.",
                        "Time passes; the hero steels their resolve.",
                    ],
                )
            ]
        )

        host = _FakePolishLLMHost(llm_output)
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"
        assert "transition guides generated" in result.detail

        # Passage spec in plan should have transition_guidance
        plan_nodes = graph.get_nodes_by_type("polish_plan")
        plan_data = plan_nodes.get("polish_plan::current", {})
        passage_specs = plan_data.get("passage_specs", [])
        assert len(passage_specs) == 1
        assert passage_specs[0]["transition_guidance"] == [
            "As the dust settles, a new threat emerges.",
            "Time passes; the hero steels their resolve.",
        ]

    def test_wrong_transition_count_skipped(self) -> None:
        """If transitions count != beat_count - 1, the spec is skipped with a warning."""
        from questfoundry.models.polish import Phase5fOutput, TransitionGuidanceItem

        graph = Graph.empty()
        self._build_plan_with_collapsed(graph)

        # Wrong count: 3 beats → need 2 transitions, but LLM returns 3
        llm_output = Phase5fOutput(
            transition_guidance=[
                TransitionGuidanceItem(
                    passage_id="passage::collapse_0",
                    transitions=["T1.", "T2.", "T3."],  # too many
                )
            ]
        )

        host = _FakePolishLLMHost(llm_output)
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"
        # Guide not applied — count says 0
        assert "0 transition guides generated" in result.detail

    def test_skipped_when_no_collapsed_passages(self) -> None:
        """Phase 5f is skipped when no collapsed passages with 2+ beats exist."""
        from questfoundry.models.polish import PassageSpec, Phase5fOutput

        graph = Graph.empty()

        singleton = PassageSpec(
            passage_id="passage::single_0",
            beat_ids=["beat::x"],
            summary="singleton",
            entities=[],
            grouping_type="singleton",
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
                "passage_specs": [singleton.model_dump()],
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

        host = _FakePolishLLMHost(Phase5fOutput())
        result = asyncio.run(host._phase_5_llm_enrichment(graph, MagicMock()))  # type: ignore[arg-type]

        assert result.status == "completed"
        assert "0 transition guides generated" in result.detail
