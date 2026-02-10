"""Tests for GROW semantic validators (grow_validators.py)."""

from __future__ import annotations

from questfoundry.graph.grow_validators import (
    count_entries,
    format_semantic_errors,
    validate_phase2_output,
    validate_phase3_output,
    validate_phase4a_output,
    validate_phase4f_output,
    validate_phase8c_output,
    validate_phase9_output,
    validate_phase9c_output,
)
from questfoundry.graph.mutations import GrowValidationError
from questfoundry.models.grow import (
    ChoiceLabel,
    EntityArcDescriptor,
    HubProposal,
    IntersectionProposal,
    OverlayProposal,
    PathAgnosticAssessment,
    Phase2Output,
    Phase3Output,
    Phase4aOutput,
    Phase4fOutput,
    Phase8cOutput,
    Phase9cOutput,
    Phase9Output,
    SceneTypeTag,
    SpokeProposal,
)


class TestValidatePhase2Output:
    def test_valid_output_no_errors(self) -> None:
        result = Phase2Output(
            assessments=[
                PathAgnosticAssessment(beat_id="beat::b1", agnostic_for=["t1"]),
                PathAgnosticAssessment(beat_id="beat::b2", agnostic_for=["t1", "t2"]),
            ]
        )
        errors = validate_phase2_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2"},
            valid_dilemma_ids={"t1", "t2"},
        )
        assert errors == []

    def test_invalid_beat_id(self) -> None:
        result = Phase2Output(
            assessments=[
                PathAgnosticAssessment(beat_id="beat::phantom", agnostic_for=[]),
            ]
        )
        errors = validate_phase2_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2"},
            valid_dilemma_ids=set(),
        )
        assert len(errors) == 1
        assert errors[0].field_path == "assessments.0.beat_id"
        assert "phantom" in errors[0].issue
        assert errors[0].provided == "beat::phantom"

    def test_invalid_dilemma_id(self) -> None:
        result = Phase2Output(
            assessments=[
                PathAgnosticAssessment(beat_id="beat::b1", agnostic_for=["t_bad"]),
            ]
        )
        errors = validate_phase2_output(
            result,
            valid_beat_ids={"beat::b1"},
            valid_dilemma_ids={"t1", "t2"},
        )
        assert len(errors) == 1
        assert errors[0].field_path == "assessments.0.agnostic_for"
        assert "t_bad" in errors[0].issue

    def test_multiple_errors(self) -> None:
        result = Phase2Output(
            assessments=[
                PathAgnosticAssessment(beat_id="beat::bad", agnostic_for=["t_bad1", "t_bad2"]),
            ]
        )
        errors = validate_phase2_output(
            result,
            valid_beat_ids={"beat::b1"},
            valid_dilemma_ids={"t1"},
        )
        # 1 bad beat_id + 2 bad dilemma_ids = 3 errors
        assert len(errors) == 3

    def test_empty_assessments(self) -> None:
        result = Phase2Output(assessments=[])
        errors = validate_phase2_output(
            result,
            valid_beat_ids={"beat::b1"},
            valid_dilemma_ids={"t1"},
        )
        assert errors == []


class TestValidatePhase3Output:
    def test_valid_output_no_errors(self) -> None:
        result = Phase3Output(
            intersections=[
                IntersectionProposal(beat_ids=["beat::b1", "beat::b2"], rationale="test"),
            ]
        )
        errors = validate_phase3_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2", "beat::b3"},
        )
        assert errors == []

    def test_rejects_oversized_intersection(self) -> None:
        result = Phase3Output(
            intersections=[
                IntersectionProposal(
                    beat_ids=["beat::b1", "beat::b2", "beat::b3", "beat::b4"],
                    rationale="too big",
                )
            ]
        )
        errors = validate_phase3_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2", "beat::b3", "beat::b4"},
            max_intersection_size=3,
        )
        assert len(errors) == 1
        assert "maximum allowed is 3" in errors[0].issue

    def test_invalid_beat_id(self) -> None:
        result = Phase3Output(
            intersections=[
                IntersectionProposal(beat_ids=["beat::b1", "beat::phantom"], rationale="test"),
            ]
        )
        errors = validate_phase3_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2"},
        )
        assert len(errors) == 1
        assert "phantom" in errors[0].issue

    def test_beat_reused_across_intersections(self) -> None:
        result = Phase3Output(
            intersections=[
                IntersectionProposal(beat_ids=["beat::b1", "beat::b2"], rationale="intersection1"),
                IntersectionProposal(beat_ids=["beat::b2", "beat::b3"], rationale="intersection2"),
            ]
        )
        errors = validate_phase3_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2", "beat::b3"},
        )
        # beat::b2 reused → 1 error
        assert len(errors) == 1
        assert "reused" in errors[0].issue
        assert errors[0].provided == "beat::b2"

    def test_both_invalid_and_reused(self) -> None:
        result = Phase3Output(
            intersections=[
                IntersectionProposal(beat_ids=["beat::b1", "beat::bad"], rationale="intersection1"),
                IntersectionProposal(beat_ids=["beat::b1", "beat::b2"], rationale="intersection2"),
            ]
        )
        errors = validate_phase3_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2"},
        )
        # beat::bad invalid + beat::b1 reused = 2 errors
        assert len(errors) == 2


class TestValidatePhase4aOutput:
    def test_valid_output_no_errors(self) -> None:
        result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b1",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b2",
                    scene_type="sequel",
                ),
            ]
        )
        errors = validate_phase4a_output(
            result,
            valid_beat_ids={"beat::b1", "beat::b2"},
        )
        assert errors == []

    def test_invalid_beat_id(self) -> None:
        result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::phantom",
                    scene_type="scene",
                ),
            ]
        )
        errors = validate_phase4a_output(
            result,
            valid_beat_ids={"beat::b1"},
        )
        assert len(errors) == 1
        assert errors[0].field_path == "tags.0.beat_id"
        assert "phantom" in errors[0].issue


class TestValidatePhase8cOutput:
    def test_valid_output_no_errors(self) -> None:
        result = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::e1",
                    when=["cw::c1"],
                    details=[{"key": "state", "value": "active"}],
                ),
            ]
        )
        errors = validate_phase8c_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_codeword_ids={"cw::c1"},
        )
        assert errors == []

    def test_invalid_entity_id(self) -> None:
        result = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::bad",
                    when=["cw::c1"],
                    details=[{"key": "state", "value": "active"}],
                ),
            ]
        )
        errors = validate_phase8c_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_codeword_ids={"cw::c1"},
        )
        assert len(errors) == 1
        assert "entity::bad" in errors[0].issue

    def test_invalid_codeword_id(self) -> None:
        result = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::e1",
                    when=["cw::bad"],
                    details=[{"key": "state", "value": "active"}],
                ),
            ]
        )
        errors = validate_phase8c_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_codeword_ids={"cw::c1"},
        )
        assert len(errors) == 1
        assert "cw::bad" in errors[0].issue

    def test_multiple_invalid_codewords(self) -> None:
        result = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::e1",
                    when=["cw::bad1", "cw::bad2"],
                    details=[{"key": "state", "value": "active"}],
                ),
            ]
        )
        errors = validate_phase8c_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_codeword_ids={"cw::c1"},
        )
        assert len(errors) == 2


class TestValidatePhase9Output:
    def test_valid_output_no_errors(self) -> None:
        result = Phase9Output(
            labels=[
                ChoiceLabel(from_passage="passage::a", to_passage="passage::b", label="go left"),
            ]
        )
        errors = validate_phase9_output(
            result,
            valid_passage_ids={"passage::a", "passage::b"},
        )
        assert errors == []

    def test_invalid_from_passage(self) -> None:
        result = Phase9Output(
            labels=[
                ChoiceLabel(from_passage="passage::bad", to_passage="passage::b", label="go"),
            ]
        )
        errors = validate_phase9_output(
            result,
            valid_passage_ids={"passage::a", "passage::b"},
        )
        assert len(errors) == 1
        assert errors[0].field_path == "labels.0.from_passage"
        assert "passage::bad" in errors[0].issue

    def test_invalid_to_passage(self) -> None:
        result = Phase9Output(
            labels=[
                ChoiceLabel(from_passage="passage::a", to_passage="passage::bad", label="go"),
            ]
        )
        errors = validate_phase9_output(
            result,
            valid_passage_ids={"passage::a", "passage::b"},
        )
        assert len(errors) == 1
        assert errors[0].field_path == "labels.0.to_passage"

    def test_both_invalid(self) -> None:
        result = Phase9Output(
            labels=[
                ChoiceLabel(from_passage="passage::bad1", to_passage="passage::bad2", label="go"),
            ]
        )
        errors = validate_phase9_output(
            result,
            valid_passage_ids={"passage::a", "passage::b"},
        )
        assert len(errors) == 2

    def test_available_ids_in_error(self) -> None:
        result = Phase9Output(
            labels=[
                ChoiceLabel(from_passage="passage::bad", to_passage="passage::a", label="go"),
            ]
        )
        errors = validate_phase9_output(
            result,
            valid_passage_ids={"passage::a", "passage::b", "passage::c"},
        )
        assert len(errors) == 1
        assert "passage::a" in errors[0].available
        assert "passage::b" in errors[0].available
        assert "passage::c" in errors[0].available


class TestValidatePhase4fOutput:
    def test_valid_output_no_errors(self) -> None:
        result = Phase4fOutput(
            arcs=[
                EntityArcDescriptor(
                    entity_id="entity::e1",
                    arc_line="trusted ally → doubts surface → revealed as spy",
                    pivot_beat="beat::b2",
                ),
            ]
        )
        errors = validate_phase4f_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_beat_ids={"beat::b1", "beat::b2", "beat::b3"},
        )
        assert errors == []

    def test_invalid_entity_id(self) -> None:
        result = Phase4fOutput(
            arcs=[
                EntityArcDescriptor(
                    entity_id="entity::phantom",
                    arc_line="trusted ally → doubts surface → revealed as spy",
                    pivot_beat="beat::b2",
                ),
            ]
        )
        errors = validate_phase4f_output(
            result,
            valid_entity_ids={"entity::e1", "entity::e2"},
            valid_beat_ids={"beat::b1", "beat::b2"},
        )
        assert len(errors) == 1
        assert errors[0].field_path == "arcs.0.entity_id"
        assert "phantom" in errors[0].issue
        assert errors[0].provided == "entity::phantom"

    def test_invalid_pivot_beat(self) -> None:
        result = Phase4fOutput(
            arcs=[
                EntityArcDescriptor(
                    entity_id="entity::e1",
                    arc_line="safe harbor → tension creeps in → site of confrontation",
                    pivot_beat="beat::not_on_path",
                ),
            ]
        )
        errors = validate_phase4f_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_beat_ids={"beat::b1", "beat::b2"},
        )
        assert len(errors) == 1
        assert errors[0].field_path == "arcs.0.pivot_beat"
        assert "not_on_path" in errors[0].issue

    def test_both_invalid(self) -> None:
        result = Phase4fOutput(
            arcs=[
                EntityArcDescriptor(
                    entity_id="entity::bad",
                    arc_line="mundane letter → imbued with dread → proof of betrayal",
                    pivot_beat="beat::bad",
                ),
            ]
        )
        errors = validate_phase4f_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_beat_ids={"beat::b1"},
        )
        assert len(errors) == 2

    def test_multiple_arcs_mixed_validity(self) -> None:
        result = Phase4fOutput(
            arcs=[
                EntityArcDescriptor(
                    entity_id="entity::e1",
                    arc_line="trusted ally → doubts surface → revealed as spy",
                    pivot_beat="beat::b2",
                ),
                EntityArcDescriptor(
                    entity_id="entity::phantom",
                    arc_line="safe harbor → tension creeps in → site of confrontation",
                    pivot_beat="beat::b2",
                ),
            ]
        )
        errors = validate_phase4f_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_beat_ids={"beat::b1", "beat::b2"},
        )
        assert len(errors) == 1
        assert errors[0].provided == "entity::phantom"

    def test_available_ids_in_error(self) -> None:
        result = Phase4fOutput(
            arcs=[
                EntityArcDescriptor(
                    entity_id="entity::bad",
                    arc_line="trusted ally → doubts surface → revealed as spy",
                    pivot_beat="beat::b1",
                ),
            ]
        )
        errors = validate_phase4f_output(
            result,
            valid_entity_ids={"entity::e1", "entity::e2"},
            valid_beat_ids={"beat::b1"},
        )
        assert len(errors) == 1
        assert "entity::e1" in errors[0].available
        assert "entity::e2" in errors[0].available

    def test_empty_arcs(self) -> None:
        result = Phase4fOutput(arcs=[])
        errors = validate_phase4f_output(
            result,
            valid_entity_ids={"entity::e1"},
            valid_beat_ids={"beat::b1"},
        )
        assert errors == []


class TestFormatSemanticErrors:
    def test_basic_formatting(self) -> None:
        errors = [
            GrowValidationError(
                field_path="assessments.0.beat_id",
                issue="Beat ID not found: beat::bad",
                provided="beat::bad",
                available=["beat::b1", "beat::b2"],
            ),
        ]
        text = format_semantic_errors(errors)
        assert "Semantic validation errors" in text
        assert "assessments.0.beat_id" in text
        assert "beat::bad" in text
        assert "beat::b1" in text
        assert "Please fix" in text

    def test_truncates_available_to_five(self) -> None:
        errors = [
            GrowValidationError(
                field_path="tags.0.beat_id",
                issue="Beat ID not found: x",
                provided="x",
                available=["a", "b", "c", "d", "e", "f", "g"],
            ),
        ]
        text = format_semantic_errors(errors)
        assert "and 2 more" in text

    def test_no_available_ids(self) -> None:
        errors = [
            GrowValidationError(
                field_path="intersections.0.beat_ids",
                issue="Beat reused across intersections: beat::b1",
                provided="beat::b1",
            ),
        ]
        text = format_semantic_errors(errors)
        assert "Valid options" not in text
        assert "reused" in text


class TestCountEntries:
    def test_counts_assessments(self) -> None:
        result = Phase2Output(
            assessments=[
                PathAgnosticAssessment(beat_id="b1", agnostic_for=[]),
                PathAgnosticAssessment(beat_id="b2", agnostic_for=[]),
            ]
        )
        assert count_entries(result) == 2

    def test_counts_intersections(self) -> None:
        result = Phase3Output(
            intersections=[IntersectionProposal(beat_ids=["b1", "b2"], rationale="test")]
        )
        assert count_entries(result) == 1

    def test_counts_tags(self) -> None:
        result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="b1",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="b2",
                    scene_type="sequel",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="b3",
                    scene_type="micro_beat",
                ),
            ]
        )
        assert count_entries(result) == 3

    def test_counts_overlays(self) -> None:
        result = Phase8cOutput(overlays=[])
        assert count_entries(result) == 0

    def test_counts_labels(self) -> None:
        result = Phase9Output(labels=[ChoiceLabel(from_passage="a", to_passage="b", label="go")])
        assert count_entries(result) == 1

    def test_counts_arcs(self) -> None:
        result = Phase4fOutput(
            arcs=[
                EntityArcDescriptor(
                    entity_id="entity::e1",
                    arc_line="trusted ally → doubts → revealed as spy",
                    pivot_beat="b2",
                ),
                EntityArcDescriptor(
                    entity_id="entity::e2",
                    arc_line="safe harbor → tension → confrontation",
                    pivot_beat="b3",
                ),
            ]
        )
        assert count_entries(result) == 2

    def test_fallback_for_unknown_object(self) -> None:
        assert count_entries(object()) == 1


# ---------------------------------------------------------------------------
# Phase 9c: Hub-spoke spoke grants validation
# ---------------------------------------------------------------------------


class TestValidatePhase9cGrants:
    """Tests for spoke grant ID validation in Phase 9c."""

    def _make_phase9c_output(self, grants: list[str] | None = None) -> Phase9cOutput:
        spoke_with_grants = SpokeProposal(
            summary="Examine the ancient mural.",
            grants=grants or [],
        )
        spoke_plain = SpokeProposal(summary="Listen to the crowd.")
        hub = HubProposal(
            passage_id="passage::market",
            spokes=[spoke_with_grants, spoke_plain],
            forward_label="Continue onward",
        )
        return Phase9cOutput(hubs=[hub])

    def test_valid_spoke_grants_pass(self) -> None:
        result = self._make_phase9c_output(grants=["codeword::cw_mural"])
        errors = validate_phase9c_output(
            result,
            valid_passage_ids={"passage::market"},
            valid_codeword_ids={"codeword::cw_mural"},
        )
        assert not errors

    def test_invalid_spoke_grants_rejected(self) -> None:
        result = self._make_phase9c_output(grants=["codeword::nonexistent"])
        errors = validate_phase9c_output(
            result,
            valid_passage_ids={"passage::market"},
            valid_codeword_ids={"codeword::cw_mural"},
        )
        assert len(errors) == 1
        assert "nonexistent" in errors[0].issue

    def test_no_codeword_validation_when_none(self) -> None:
        """When valid_codeword_ids is None, grants are not validated."""
        result = self._make_phase9c_output(grants=["codeword::anything"])
        errors = validate_phase9c_output(
            result,
            valid_passage_ids={"passage::market"},
            valid_codeword_ids=None,
        )
        assert not errors

    def test_unscoped_grant_id_normalized(self) -> None:
        """Grant IDs without 'codeword::' prefix are normalized."""
        result = self._make_phase9c_output(grants=["cw_mural"])
        errors = validate_phase9c_output(
            result,
            valid_passage_ids={"passage::market"},
            valid_codeword_ids={"codeword::cw_mural"},
        )
        assert not errors
