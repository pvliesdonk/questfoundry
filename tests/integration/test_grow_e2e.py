"""End-to-end integration tests for the GROW stage.

Runs all 15 GROW phases on the E2E fixture graph with a mocked LLM,
verifying that phases execute correctly and produce valid output
structure. Also tests context formatting stays within token budgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.models.grow import (
    AtmosphericDetail,
    PathMiniArc,
    Phase3Output,
    Phase4aOutput,
    Phase4bOutput,
    Phase4dOutput,
    Phase4fOutput,
    Phase8cOutput,
    SceneTypeTag,
)
from tests.fixtures.grow_fixtures import make_e2e_fixture_graph

if TYPE_CHECKING:
    from collections.abc import Iterator


def _make_e2e_mock_model(graph: Graph) -> MagicMock:
    """Create a mock model that returns valid structured output for all LLM phases.

    Returns empty/minimal results for each phase for simplicity.
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    phase3_output = Phase3Output(intersections=[])

    # Phase 4a: scene type tags for all beats
    scene_types: list[Literal["scene", "sequel", "micro_beat"]] = [
        "scene",
        "sequel",
        "micro_beat",
    ]
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

    # Phase 4d: atmospheric details for all beats
    phase4d_output = Phase4dOutput(
        details=[
            AtmosphericDetail(
                beat_id=bid,
                atmospheric_detail="Dim light filters through dusty windows",
            )
            for bid in sorted(beat_nodes.keys())
        ],
    )
    phase4f_output = Phase4fOutput(arcs=[])

    # Phase 4e: generic path arc (called per-path with PathMiniArc schema)
    phase4e_output = PathMiniArc(
        path_id="placeholder",
        path_theme="A journey through uncertainty and choice",
        path_mood="quiet tension",
    )

    phase8c_output = Phase8cOutput(overlays=[])

    # Map schema title -> output (schema is now a dict with "title" field)
    output_by_title: dict[str, object] = {
        "Phase3Output": phase3_output,
        "Phase4aOutput": phase4a_output,
        "Phase4bOutput": phase4b_output,
        "Phase4dOutput": phase4d_output,
        "Phase4fOutput": phase4f_output,
        "PathMiniArc": phase4e_output,
        "Phase8cOutput": phase8c_output,
    }

    def _with_structured_output(schema: dict[str, Any], **_kwargs: object) -> AsyncMock:
        title = schema.get("title", "") if isinstance(schema, dict) else ""
        output = output_by_title.get(title, phase3_output)
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=output)
        return mock_structured

    mock_model = MagicMock()
    mock_model.with_structured_output = MagicMock(side_effect=_with_structured_output)
    return mock_model


@pytest.fixture(scope="module")
def pipeline_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Any]]:
    """Run the full GROW pipeline once and share the result across tests.

    Uses tmp_path_factory for module-scoped temp directory.
    Returns a dict with result_dict, llm_calls, and tokens.
    """
    import asyncio

    from questfoundry.graph import grow_validation as grow_validation
    from questfoundry.graph.grow_validation import ValidationCheck, ValidationReport
    from questfoundry.pipeline.stages.grow import GrowStage

    tmp_path = tmp_path_factory.mktemp("grow_e2e")
    graph = make_e2e_fixture_graph()
    graph.save(tmp_path / "graph.db")

    # Module-scoped fixture cannot depend on function-scoped monkeypatch.
    mp = pytest.MonkeyPatch()

    def _mock_run_all_checks(_graph: Graph) -> ValidationReport:
        return ValidationReport(checks=[ValidationCheck(name="mock_validation", severity="pass")])

    # Mock outputs do not guarantee semantic validity; bypass validation so
    # this fixture exercises pipeline wiring rather than validation outcomes.
    mp.setattr(grow_validation, "run_all_checks", _mock_run_all_checks)

    stage = GrowStage(project_path=tmp_path)
    mock_model = _make_e2e_mock_model(graph)

    result_dict, llm_calls, tokens = asyncio.run(stage.execute(model=mock_model, user_prompt=""))

    yield {
        "result_dict": result_dict,
        "llm_calls": llm_calls,
        "tokens": tokens,
        "project_path": tmp_path,
    }

    mp.undo()


class TestGrowFullPipeline:
    """E2E tests running all 15 GROW phases on the fixture graph."""

    def test_all_phases_complete(self, pipeline_result: dict[str, Any]) -> None:
        """Verify the stage completes and returns the GrowResult summary."""
        result_dict = pipeline_result["result_dict"]
        expected_keys = {
            "arc_count",
            "passage_count",
            "choice_count",
            "state_flag_count",
            "overlay_count",
            "spine_arc_id",
            "phases_completed",
        }
        assert set(result_dict.keys()) == expected_keys

    def test_arc_enumeration(self, pipeline_result: dict[str, Any]) -> None:
        """Verify 4 arcs are enumerated from 2 dilemmas x 2 paths."""
        result_dict = pipeline_result["result_dict"]
        assert result_dict["arc_count"] == 4
        assert result_dict["spine_arc_id"] is not None

    def test_passages_created(self, pipeline_result: dict[str, Any]) -> None:
        """Verify passages are created (S3: apply_routing creates variants only when
        choice edges exist; base passage count after consolidation is the minimum)."""
        assert pipeline_result["result_dict"]["passage_count"] >= 7

    def test_state_flags_derived(self, pipeline_result: dict[str, Any]) -> None:
        """Verify state flags are created from consequences (4 consequences)."""
        assert pipeline_result["result_dict"]["state_flag_count"] == 4

    def test_choices_created(self, pipeline_result: dict[str, Any]) -> None:
        """Verify choices are created at divergence points."""
        assert pipeline_result["result_dict"]["choice_count"] > 0

    def test_validation_phase_passes(self, pipeline_result: dict[str, Any]) -> None:
        """Verify the resulting graph is structurally valid."""
        saved_graph = Graph.load(pipeline_result["project_path"])
        assert saved_graph.validate_invariants() == []

    def test_result_structure(self, pipeline_result: dict[str, Any]) -> None:
        """Verify the GrowResult summary contains expected count fields."""
        result_dict = pipeline_result["result_dict"]
        expected_keys = {
            "arc_count",
            "passage_count",
            "state_flag_count",
            "choice_count",
            "overlay_count",
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
