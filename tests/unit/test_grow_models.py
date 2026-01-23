"""Tests for GROW stage models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from questfoundry.models.grow import (
    Arc,
    Choice,
    ChoiceLabel,
    Codeword,
    EntityOverlay,
    GapProposal,
    GrowPhaseResult,
    GrowResult,
    KnotProposal,
    OverlayProposal,
    Passage,
    SceneTypeTag,
    ThreadAgnosticAssessment,
)


class TestArc:
    def test_valid_spine_arc(self) -> None:
        arc = Arc(
            arc_id="mentor_trust+artifact_quest",
            arc_type="spine",
            threads=["mentor_trust_canonical", "artifact_quest_canonical"],
            sequence=["beat_1", "beat_2", "beat_3"],
        )
        assert arc.arc_id == "mentor_trust+artifact_quest"
        assert arc.arc_type == "spine"
        assert arc.diverges_from is None
        assert arc.converges_to is None

    def test_valid_branch_arc(self) -> None:
        arc = Arc(
            arc_id="mentor_trust+artifact_quest",
            arc_type="branch",
            threads=["mentor_trust_alt", "artifact_quest_canonical"],
            sequence=["beat_1", "beat_4"],
            diverges_from="arc::spine",
            diverges_at="beat_2",
        )
        assert arc.arc_type == "branch"
        assert arc.diverges_from == "arc::spine"
        assert arc.diverges_at == "beat_2"

    def test_empty_arc_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="arc_id"):
            Arc(arc_id="", arc_type="spine", threads=["t1"])

    def test_empty_threads_rejected(self) -> None:
        with pytest.raises(ValidationError, match="threads"):
            Arc(arc_id="test", arc_type="spine", threads=[])

    def test_invalid_arc_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="arc_type"):
            Arc(arc_id="test", arc_type="invalid", threads=["t1"])  # type: ignore[arg-type]

    def test_empty_sequence_allowed(self) -> None:
        arc = Arc(arc_id="test", arc_type="spine", threads=["t1"])
        assert arc.sequence == []

    def test_convergence_fields(self) -> None:
        arc = Arc(
            arc_id="test",
            arc_type="branch",
            threads=["t1"],
            converges_to="arc::spine",
            converges_at="beat_5",
        )
        assert arc.converges_to == "arc::spine"
        assert arc.converges_at == "beat_5"


class TestPassage:
    def test_valid_passage(self) -> None:
        passage = Passage(
            passage_id="passage::beat_1",
            from_beat="beat_1",
            summary="The hero enters the cave.",
            entities=["entity::hero", "entity::cave"],
        )
        assert passage.passage_id == "passage::beat_1"
        assert passage.from_beat == "beat_1"
        assert len(passage.entities) == 2

    def test_empty_entities_allowed(self) -> None:
        passage = Passage(passage_id="p1", from_beat="b1", summary="A scene.")
        assert passage.entities == []

    def test_empty_passage_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="passage_id"):
            Passage(passage_id="", from_beat="b1", summary="text")

    def test_empty_from_beat_rejected(self) -> None:
        with pytest.raises(ValidationError, match="from_beat"):
            Passage(passage_id="p1", from_beat="", summary="text")

    def test_empty_summary_rejected(self) -> None:
        with pytest.raises(ValidationError, match="summary"):
            Passage(passage_id="p1", from_beat="b1", summary="")


class TestCodeword:
    def test_valid_codeword(self) -> None:
        cw = Codeword(
            codeword_id="codeword::mentor_trust_committed",
            tracks="consequence::mentor_trust",
        )
        assert cw.codeword_id == "codeword::mentor_trust_committed"
        assert cw.codeword_type == "granted"

    def test_default_type_is_granted(self) -> None:
        cw = Codeword(codeword_id="cw1", tracks="c1")
        assert cw.codeword_type == "granted"

    def test_empty_codeword_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="codeword_id"):
            Codeword(codeword_id="", tracks="c1")

    def test_empty_tracks_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tracks"):
            Codeword(codeword_id="cw1", tracks="")

    def test_invalid_codeword_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="codeword_type"):
            Codeword(codeword_id="cw1", tracks="c1", codeword_type="revoked")  # type: ignore[arg-type]


class TestChoice:
    def test_valid_choice(self) -> None:
        choice = Choice(
            from_passage="p1",
            to_passage="p2",
            label="Go left",
            requires=["cw1"],
            grants=["cw2"],
        )
        assert choice.from_passage == "p1"
        assert choice.requires == ["cw1"]

    def test_empty_requires_grants_allowed(self) -> None:
        choice = Choice(from_passage="p1", to_passage="p2", label="Continue")
        assert choice.requires == []
        assert choice.grants == []

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="label"):
            Choice(from_passage="p1", to_passage="p2", label="")


class TestEntityOverlay:
    def test_valid_overlay(self) -> None:
        overlay = EntityOverlay(
            entity_id="entity::hero",
            when=["cw_trust_committed"],
            details={"mood": "grateful", "dialogue_style": "warm"},
        )
        assert overlay.entity_id == "entity::hero"
        assert overlay.when == ["cw_trust_committed"]
        assert overlay.details["mood"] == "grateful"

    def test_empty_when_rejected(self) -> None:
        with pytest.raises(ValidationError, match="when"):
            EntityOverlay(entity_id="e1", when=[], details={})

    def test_empty_details_allowed(self) -> None:
        overlay = EntityOverlay(entity_id="e1", when=["cw1"])
        assert overlay.details == {}


class TestThreadAgnosticAssessment:
    def test_valid_assessment(self) -> None:
        ta = ThreadAgnosticAssessment(
            beat_id="beat_1",
            agnostic_for=["tension_mentor_trust"],
        )
        assert ta.beat_id == "beat_1"
        assert ta.agnostic_for == ["tension_mentor_trust"]

    def test_empty_agnostic_for_allowed(self) -> None:
        ta = ThreadAgnosticAssessment(beat_id="b1")
        assert ta.agnostic_for == []

    def test_empty_beat_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="beat_id"):
            ThreadAgnosticAssessment(beat_id="")


class TestKnotProposal:
    def test_valid_knot(self) -> None:
        knot = KnotProposal(
            beat_ids=["beat_1", "beat_2", "beat_3"],
            resolved_location="beat_4",
            rationale="Shared location signal",
        )
        assert len(knot.beat_ids) == 3
        assert knot.resolved_location == "beat_4"
        assert knot.rationale == "Shared location signal"

    def test_single_beat_rejected(self) -> None:
        with pytest.raises(ValidationError, match="beat_ids"):
            KnotProposal(beat_ids=["beat_1"])

    def test_empty_beat_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="beat_ids"):
            KnotProposal(beat_ids=[])

    def test_no_resolved_location_allowed(self) -> None:
        knot = KnotProposal(beat_ids=["b1", "b2"], rationale="Entity overlap")
        assert knot.resolved_location is None


class TestSceneTypeTag:
    @pytest.mark.parametrize(
        "scene_type",
        [
            pytest.param("scene", id="scene"),
            pytest.param("sequel", id="sequel"),
            pytest.param("micro_beat", id="micro_beat"),
        ],
    )
    def test_valid_scene_types(self, scene_type: str) -> None:
        tag = SceneTypeTag(beat_id="b1", scene_type=scene_type)  # type: ignore[arg-type]
        assert tag.scene_type == scene_type

    def test_invalid_scene_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="scene_type"):
            SceneTypeTag(beat_id="b1", scene_type="epic")  # type: ignore[arg-type]


class TestGapProposal:
    def test_valid_gap(self) -> None:
        gap = GapProposal(
            thread_id="thread_main",
            after_beat="beat_2",
            before_beat="beat_3",
            summary="Hero reflects on the battle.",
            scene_type="sequel",
        )
        assert gap.thread_id == "thread_main"
        assert gap.scene_type == "sequel"

    def test_default_scene_type_is_sequel(self) -> None:
        gap = GapProposal(thread_id="t1", summary="A transition.")
        assert gap.scene_type == "sequel"

    def test_no_before_after_allowed(self) -> None:
        gap = GapProposal(thread_id="t1", summary="Opening scene.")
        assert gap.after_beat is None
        assert gap.before_beat is None

    def test_empty_summary_rejected(self) -> None:
        with pytest.raises(ValidationError, match="summary"):
            GapProposal(thread_id="t1", summary="")


class TestOverlayProposal:
    def test_valid_proposal(self) -> None:
        op = OverlayProposal(
            entity_id="e1",
            when=["cw1", "cw2"],
            details={"state": "active"},
        )
        assert op.entity_id == "e1"
        assert len(op.when) == 2

    def test_empty_when_rejected(self) -> None:
        with pytest.raises(ValidationError, match="when"):
            OverlayProposal(entity_id="e1", when=[])


class TestChoiceLabel:
    def test_valid_label(self) -> None:
        cl = ChoiceLabel(from_passage="p1", to_passage="p2", label="Fight")
        assert cl.label == "Fight"

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="label"):
            ChoiceLabel(from_passage="p1", to_passage="p2", label="")


class TestGrowPhaseResult:
    @pytest.mark.parametrize(
        "status",
        [
            pytest.param("completed", id="completed"),
            pytest.param("skipped", id="skipped"),
            pytest.param("failed", id="failed"),
        ],
    )
    def test_valid_statuses(self, status: str) -> None:
        r = GrowPhaseResult(phase="phase_1", status=status)  # type: ignore[arg-type]
        assert r.status == status

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            GrowPhaseResult(phase="p1", status="running")  # type: ignore[arg-type]

    def test_empty_phase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="phase"):
            GrowPhaseResult(phase="", status="completed")

    def test_default_detail_is_empty(self) -> None:
        r = GrowPhaseResult(phase="p1", status="completed")
        assert r.detail == ""


class TestGrowResult:
    def test_default_values(self) -> None:
        result = GrowResult()
        assert result.arc_count == 0
        assert result.passage_count == 0
        assert result.codeword_count == 0
        assert result.phases_completed == []
        assert result.spine_arc_id is None

    def test_with_phases(self) -> None:
        result = GrowResult(
            arc_count=4,
            passage_count=8,
            codeword_count=2,
            phases_completed=[
                GrowPhaseResult(phase="validate", status="completed"),
                GrowPhaseResult(phase="arcs", status="completed"),
            ],
            spine_arc_id="arc::thread_a+thread_b",
        )
        assert result.arc_count == 4
        assert len(result.phases_completed) == 2
        assert result.spine_arc_id == "arc::thread_a+thread_b"

    def test_model_dump_roundtrip(self) -> None:
        result = GrowResult(
            arc_count=2,
            phases_completed=[
                GrowPhaseResult(phase="validate", status="completed", detail="ok"),
            ],
            spine_arc_id="arc::main",
        )
        data = result.model_dump()
        restored = GrowResult.model_validate(data)
        assert restored == result

    def test_model_validate_from_dict(self) -> None:
        data = {
            "arc_count": 3,
            "passage_count": 6,
            "codeword_count": 1,
            "phases_completed": [
                {"phase": "validate", "status": "completed", "detail": ""},
            ],
            "spine_arc_id": "arc::spine",
        }
        result = GrowResult.model_validate(data)
        assert result.arc_count == 3
        assert result.phases_completed[0].phase == "validate"
