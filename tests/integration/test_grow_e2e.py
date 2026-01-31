"""End-to-end integration tests for the GROW stage.

Runs all 15 GROW phases on the E2E fixture graph with a mocked LLM,
verifying that phases execute correctly and produce valid output
structure. Also tests context formatting stays within token budgets.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.graph.graph import Graph
from tests.fixtures.grow_fixtures import make_e2e_fixture_graph


def _make_e2e_mock_model(graph: Graph) -> MagicMock:
    """Create a mock model that returns valid structured output for all LLM phases.

    Builds realistic Phase 2 assessments from the fixture graph structure.
    Other phases return empty/minimal results for simplicity.
    """
    from questfoundry.models.grow import (
        PathAgnosticAssessment,
        Phase2Output,
        Phase3Output,
        Phase4aOutput,
        Phase4bOutput,
        Phase8cOutput,
        Phase9Output,
        SceneTypeTag,
    )

    # Build Phase 2: identify shared beats (path-agnostic)
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    path_nodes = graph.get_nodes_by_type("path")
    beat_nodes = graph.get_nodes_by_type("beat")

    dilemma_paths: dict[str, list[str]] = {}
    for edge in graph.get_edges(edge_type="explores"):
        path_id = edge["from"]
        dilemma_id = edge["to"]
        if path_id in path_nodes and dilemma_id in dilemma_nodes:
            dilemma_paths.setdefault(dilemma_id, []).append(path_id)

    beat_path_map: dict[str, list[str]] = {}
    for edge in graph.get_edges(edge_type="belongs_to"):
        beat_path_map.setdefault(edge["from"], []).append(edge["to"])

    assessments: list[PathAgnosticAssessment] = []
    for beat_id, bp_list in beat_path_map.items():
        if beat_id not in beat_nodes:
            continue
        agnostic_dilemmas: list[str] = []
        for dilemma_id, d_paths in dilemma_paths.items():
            shared = [p for p in bp_list if p in d_paths]
            if len(shared) > 1:
                raw_did = dilemma_nodes[dilemma_id].get("raw_id", dilemma_id)
                agnostic_dilemmas.append(raw_did)
        if agnostic_dilemmas:
            assessments.append(
                PathAgnosticAssessment(beat_id=beat_id, agnostic_for=agnostic_dilemmas)
            )

    phase2_output = Phase2Output(assessments=assessments)
    phase3_output = Phase3Output(intersections=[])

    # Phase 4a: scene type tags for all beats
    scene_types = ["scene", "sequel", "micro_beat"]
    tags = [
        SceneTypeTag(
            narrative_function="introduce",
            exit_mood="quiet dread",
            beat_id=bid,
            scene_type=scene_types[i % 3],
        )
        for i, bid in enumerate(sorted(beat_nodes.keys()))
    ]
    phase4a_output = Phase4aOutput(tags=tags)
    phase4b_output = Phase4bOutput(gaps=[])
    phase8c_output = Phase8cOutput(overlays=[])
    phase9_output = Phase9Output(labels=[])

    output_by_schema: dict[type, object] = {
        Phase2Output: phase2_output,
        Phase3Output: phase3_output,
        Phase4aOutput: phase4a_output,
        Phase4bOutput: phase4b_output,
        Phase8cOutput: phase8c_output,
        Phase9Output: phase9_output,
    }

    def _with_structured_output(schema: type, **_kwargs: object) -> AsyncMock:
        output = output_by_schema.get(schema, phase2_output)
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=output)
        return mock_structured

    mock_model = MagicMock()
    mock_model.with_structured_output = MagicMock(side_effect=_with_structured_output)
    return mock_model


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the full GROW pipeline once and share the result across tests.

    Uses tmp_path_factory for module-scoped temp directory.
    Returns a dict with result_dict, llm_calls, and tokens.
    """
    import asyncio

    from questfoundry.pipeline.stages.grow import GrowStage

    tmp_path = tmp_path_factory.mktemp("grow_e2e")
    graph = make_e2e_fixture_graph()
    graph.save(tmp_path / "graph.json")

    stage = GrowStage(project_path=tmp_path)
    mock_model = _make_e2e_mock_model(graph)

    result_dict, llm_calls, tokens = asyncio.run(stage.execute(model=mock_model, user_prompt=""))

    return {"result_dict": result_dict, "llm_calls": llm_calls, "tokens": tokens}


class TestGrowFullPipeline:
    """E2E tests running all 15 GROW phases on the fixture graph."""

    def test_all_phases_complete(self, pipeline_result: dict[str, Any]) -> None:
        """Verify all 15 phases complete with no failures."""
        phases = pipeline_result["result_dict"]["phases_completed"]
        assert len(phases) == 15
        for phase in phases:
            assert phase["status"] in ("completed", "skipped"), (
                f"Phase {phase['phase']} has unexpected status: {phase['status']}"
            )

    def test_arc_enumeration(self, pipeline_result: dict[str, Any]) -> None:
        """Verify 4 arcs are enumerated from 2 dilemmas x 2 paths."""
        result_dict = pipeline_result["result_dict"]
        assert result_dict["arc_count"] == 4
        assert result_dict["spine_arc_id"] is not None

    def test_passages_created(self, pipeline_result: dict[str, Any]) -> None:
        """Verify passages are created for all beats (10 beats = 10 passages)."""
        assert pipeline_result["result_dict"]["passage_count"] == 10

    def test_codewords_derived(self, pipeline_result: dict[str, Any]) -> None:
        """Verify codewords are created from consequences (4 consequences)."""
        assert pipeline_result["result_dict"]["codeword_count"] == 4

    def test_choices_created(self, pipeline_result: dict[str, Any]) -> None:
        """Verify choices are created at divergence points."""
        assert pipeline_result["result_dict"]["choice_count"] > 0

    def test_validation_phase_passes(self, pipeline_result: dict[str, Any]) -> None:
        """Verify Phase 10 validation passes on the fixture graph."""
        phases = pipeline_result["result_dict"]["phases_completed"]
        validation_phase = next((p for p in phases if p["phase"] == "validation"), None)
        assert validation_phase is not None
        assert validation_phase["status"] == "completed"

    def test_result_structure(self, pipeline_result: dict[str, Any]) -> None:
        """Verify GrowResult contains expected keys and counts."""
        result_dict = pipeline_result["result_dict"]
        expected_keys = {
            "phases_completed",
            "arc_count",
            "spine_arc_id",
            "passage_count",
            "codeword_count",
            "choice_count",
        }
        assert expected_keys.issubset(set(result_dict.keys()))

        # LLM calls should be tracked
        assert pipeline_result["llm_calls"] >= 0
        assert pipeline_result["tokens"] >= 0


class TestGrowContextFormatting:
    """Tests for context formatting functions."""

    def test_valid_ids_contains_all_types(self) -> None:
        """Verify format_grow_valid_ids returns all ID types."""
        from questfoundry.graph.grow_context import format_grow_valid_ids

        graph = make_e2e_fixture_graph()
        ids = format_grow_valid_ids(graph)

        assert "valid_beat_ids" in ids
        assert "valid_path_ids" in ids
        assert "valid_dilemma_ids" in ids
        assert "valid_entity_ids" in ids
        assert "valid_passage_ids" in ids
        assert "valid_choice_ids" in ids

        # Before GROW runs, only SEED-stage IDs should be populated
        assert ids["valid_beat_ids"] != ""
        assert ids["valid_path_ids"] != ""
        assert ids["valid_dilemma_ids"] != ""
        assert ids["valid_entity_ids"] != ""
        # Passages and choices are created during GROW
        assert ids["valid_passage_ids"] == ""
        assert ids["valid_choice_ids"] == ""

    def test_valid_ids_context_for_grow_stage(self) -> None:
        """Verify format_valid_ids_context handles stage='grow'."""
        from questfoundry.graph.context import format_valid_ids_context

        graph = make_e2e_fixture_graph()
        context = format_valid_ids_context(graph, stage="grow")

        assert "VALID IDs FOR GROW PHASES" in context
        assert "Beat IDs" in context
        assert "Path IDs" in context
        assert "Dilemma IDs" in context

    def test_empty_graph_context(self) -> None:
        """Verify context formatting handles empty graph gracefully."""
        from questfoundry.graph.grow_context import format_grow_valid_ids

        graph = Graph.empty()

        ids = format_grow_valid_ids(graph)
        assert all(v == "" for v in ids.values())
