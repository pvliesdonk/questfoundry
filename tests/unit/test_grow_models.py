"""Tests for GROW stage models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from questfoundry.models.grow import (
    Arc,
    AtmosphericDetail,
    Choice,
    ChoiceLabel,
    Codeword,
    EntityOverlay,
    EntryMood,
    EntryStateBeat,
    GapProposal,
    GrowPhaseResult,
    GrowResult,
    IntersectionProposal,
    OverlayProposal,
    Passage,
    PathAgnosticAssessment,
    PathMiniArc,
    Phase4dOutput,
    Phase4eOutput,
    SceneTypeTag,
)


class TestArc:
    def test_valid_spine_arc(self) -> None:
        arc = Arc(
            arc_id="mentor_trust+artifact_quest",
            arc_type="spine",
            paths=["mentor_trust_canonical", "artifact_quest_canonical"],
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
            paths=["mentor_trust_alt", "artifact_quest_canonical"],
            sequence=["beat_1", "beat_4"],
            diverges_from="arc::spine",
            diverges_at="beat_2",
        )
        assert arc.arc_type == "branch"
        assert arc.diverges_from == "arc::spine"
        assert arc.diverges_at == "beat_2"

    def test_empty_arc_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="arc_id"):
            Arc(arc_id="", arc_type="spine", paths=["t1"])

    def test_empty_paths_rejected(self) -> None:
        with pytest.raises(ValidationError, match="paths"):
            Arc(arc_id="test", arc_type="spine", paths=[])

    def test_invalid_arc_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="arc_type"):
            Arc(arc_id="test", arc_type="invalid", paths=["t1"])  # type: ignore[arg-type]

    def test_empty_sequence_allowed(self) -> None:
        arc = Arc(arc_id="test", arc_type="spine", paths=["t1"])
        assert arc.sequence == []

    def test_convergence_fields(self) -> None:
        arc = Arc(
            arc_id="test",
            arc_type="branch",
            paths=["t1"],
            converges_to="arc::spine",
            converges_at="beat_5",
        )
        assert arc.converges_to == "arc::spine"
        assert arc.converges_at == "beat_5"

    def test_paths_migration(self) -> None:
        """Verify backward compat: 'paths' field migrates to 'paths'."""
        arc = Arc(
            arc_id="test",
            arc_type="spine",
            paths=["path_a", "path_b"],  # type: ignore[call-arg]
        )
        assert arc.paths == ["path_a", "path_b"]
        # Backward compat property still works
        assert arc.paths == ["path_a", "path_b"]


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


class TestPathAgnosticAssessment:
    def test_valid_assessment(self) -> None:
        ta = PathAgnosticAssessment(
            beat_id="beat_1",
            agnostic_for=["dilemma_mentor_trust"],
        )
        assert ta.beat_id == "beat_1"
        assert ta.agnostic_for == ["dilemma_mentor_trust"]

    def test_empty_agnostic_for_allowed(self) -> None:
        ta = PathAgnosticAssessment(beat_id="b1")
        assert ta.agnostic_for == []

    def test_empty_beat_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="beat_id"):
            PathAgnosticAssessment(beat_id="")


class TestIntersectionProposal:
    def test_valid_intersection(self) -> None:
        intersection = IntersectionProposal(
            beat_ids=["beat_1", "beat_2", "beat_3"],
            resolved_location="beat_4",
            rationale="Shared location signal",
        )
        assert len(intersection.beat_ids) == 3
        assert intersection.resolved_location == "beat_4"
        assert intersection.rationale == "Shared location signal"

    def test_single_beat_rejected(self) -> None:
        with pytest.raises(ValidationError, match="beat_ids"):
            IntersectionProposal(beat_ids=["beat_1"], rationale="test")

    def test_empty_beat_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="beat_ids"):
            IntersectionProposal(beat_ids=[], rationale="test")

    def test_no_resolved_location_allowed(self) -> None:
        intersection = IntersectionProposal(beat_ids=["b1", "b2"], rationale="Entity overlap")
        assert intersection.resolved_location is None


class TestSceneTypeTag:
    def _make_tag(self, **overrides: str) -> SceneTypeTag:
        defaults = {
            "beat_id": "b1",
            "scene_type": "scene",
            "narrative_function": "introduce",
            "exit_mood": "quiet dread",
        }
        defaults.update(overrides)
        return SceneTypeTag(**defaults)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "scene_type",
        [
            pytest.param("scene", id="scene"),
            pytest.param("sequel", id="sequel"),
            pytest.param("micro_beat", id="micro_beat"),
        ],
    )
    def test_valid_scene_types(self, scene_type: str) -> None:
        tag = self._make_tag(scene_type=scene_type)
        assert tag.scene_type == scene_type

    def test_invalid_scene_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="scene_type"):
            self._make_tag(scene_type="epic")

    @pytest.mark.parametrize(
        "func",
        [
            pytest.param("introduce", id="introduce"),
            pytest.param("develop", id="develop"),
            pytest.param("complicate", id="complicate"),
            pytest.param("confront", id="confront"),
            pytest.param("resolve", id="resolve"),
        ],
    )
    def test_valid_narrative_functions(self, func: str) -> None:
        tag = self._make_tag(narrative_function=func)
        assert tag.narrative_function == func

    def test_invalid_narrative_function_rejected(self) -> None:
        with pytest.raises(ValidationError, match="narrative_function"):
            self._make_tag(narrative_function="climax")

    def test_exit_mood_valid(self) -> None:
        tag = self._make_tag(exit_mood="shaken resolve")
        assert tag.exit_mood == "shaken resolve"

    def test_exit_mood_too_short_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exit_mood"):
            self._make_tag(exit_mood="x")

    def test_exit_mood_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exit_mood"):
            self._make_tag(exit_mood="a" * 41)


class TestGapProposal:
    def test_valid_gap(self) -> None:
        gap = GapProposal(
            path_id="path_main",
            after_beat="beat_2",
            before_beat="beat_3",
            summary="Hero reflects on the battle.",
            scene_type="sequel",
        )
        assert gap.path_id == "path_main"
        assert gap.scene_type == "sequel"

    def test_default_scene_type_is_sequel(self) -> None:
        gap = GapProposal(path_id="t1", summary="A transition.")
        assert gap.scene_type == "sequel"

    def test_no_before_after_allowed(self) -> None:
        gap = GapProposal(path_id="t1", summary="Opening scene.")
        assert gap.after_beat is None
        assert gap.before_beat is None

    def test_empty_summary_rejected(self) -> None:
        with pytest.raises(ValidationError, match="summary"):
            GapProposal(path_id="t1", summary="")


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
            spine_arc_id="arc::path_a+path_b",
        )
        assert result.arc_count == 4
        assert len(result.phases_completed) == 2
        assert result.spine_arc_id == "arc::path_a+path_b"

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


class TestAtmosphericDetail:
    def test_valid_detail(self) -> None:
        ad = AtmosphericDetail(
            beat_id="b1",
            atmospheric_detail="Cold stone walls slick with condensation",
        )
        assert ad.beat_id == "b1"
        assert "stone walls" in ad.atmospheric_detail

    def test_empty_beat_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="beat_id"):
            AtmosphericDetail(beat_id="", atmospheric_detail="Some detail here enough")

    def test_too_short_detail_rejected(self) -> None:
        with pytest.raises(ValidationError, match="atmospheric_detail"):
            AtmosphericDetail(beat_id="b1", atmospheric_detail="short")

    def test_too_long_detail_rejected(self) -> None:
        with pytest.raises(ValidationError, match="atmospheric_detail"):
            AtmosphericDetail(beat_id="b1", atmospheric_detail="x" * 201)


class TestEntryMood:
    def test_valid_mood(self) -> None:
        em = EntryMood(path_id="path_trust", mood="wary hope")
        assert em.path_id == "path_trust"
        assert em.mood == "wary hope"

    def test_empty_path_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path_id"):
            EntryMood(path_id="", mood="wary hope")

    def test_too_short_mood_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mood"):
            EntryMood(path_id="p1", mood="x")

    def test_too_long_mood_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mood"):
            EntryMood(path_id="p1", mood="x" * 51)


class TestEntryStateBeat:
    def test_valid_entry_state(self) -> None:
        esb = EntryStateBeat(
            beat_id="b1",
            moods=[EntryMood(path_id="p1", mood="quiet dread")],
        )
        assert esb.beat_id == "b1"
        assert len(esb.moods) == 1

    def test_empty_moods_rejected(self) -> None:
        with pytest.raises(ValidationError, match="moods"):
            EntryStateBeat(beat_id="b1", moods=[])

    def test_multiple_moods(self) -> None:
        esb = EntryStateBeat(
            beat_id="shared_beat",
            moods=[
                EntryMood(path_id="p1", mood="wary hope"),
                EntryMood(path_id="p2", mood="bitter resolve"),
            ],
        )
        assert len(esb.moods) == 2

    def test_duplicate_path_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"path_id.*unique"):
            EntryStateBeat(
                beat_id="b1",
                moods=[
                    EntryMood(path_id="p1", mood="quiet dread"),
                    EntryMood(path_id="p1", mood="bitter resolve"),
                ],
            )


class TestPhase4dOutput:
    def test_default_empty(self) -> None:
        out = Phase4dOutput()
        assert out.details == []
        assert out.entry_states == []

    def test_with_details_and_entry_states(self) -> None:
        out = Phase4dOutput(
            details=[
                AtmosphericDetail(
                    beat_id="b1",
                    atmospheric_detail="Dusty library with creaking shelves",
                ),
            ],
            entry_states=[
                EntryStateBeat(
                    beat_id="b2",
                    moods=[EntryMood(path_id="p1", mood="quiet dread")],
                ),
            ],
        )
        assert len(out.details) == 1
        assert len(out.entry_states) == 1

    def test_duplicate_detail_beat_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"beat_id.*details.*unique"):
            Phase4dOutput(
                details=[
                    AtmosphericDetail(
                        beat_id="b1", atmospheric_detail="Dusty library with creaking shelves"
                    ),
                    AtmosphericDetail(
                        beat_id="b1", atmospheric_detail="Rain-slicked cobblestones at dusk"
                    ),
                ],
            )

    def test_duplicate_entry_state_beat_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"beat_id.*entry_states.*unique"):
            Phase4dOutput(
                entry_states=[
                    EntryStateBeat(beat_id="b1", moods=[EntryMood(path_id="p1", mood="dread")]),
                    EntryStateBeat(beat_id="b1", moods=[EntryMood(path_id="p2", mood="hope")]),
                ],
            )


class TestPathMiniArc:
    def test_valid_mini_arc(self) -> None:
        arc = PathMiniArc(
            path_id="path_trust",
            path_theme="The cost of vulnerability in a world that punishes openness",
            path_mood="melancholy determination",
        )
        assert arc.path_id == "path_trust"
        assert "vulnerability" in arc.path_theme

    def test_empty_path_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path_id"):
            PathMiniArc(path_id="", path_theme="A long enough theme", path_mood="quiet dread")

    def test_too_short_theme_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path_theme"):
            PathMiniArc(path_id="p1", path_theme="short", path_mood="quiet dread")

    def test_too_short_mood_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path_mood"):
            PathMiniArc(path_id="p1", path_theme="A long enough theme", path_mood="x")


class TestPhase4eOutput:
    def test_default_empty(self) -> None:
        out = Phase4eOutput()
        assert out.arcs == []

    def test_with_arcs(self) -> None:
        out = Phase4eOutput(
            arcs=[
                PathMiniArc(
                    path_id="p1",
                    path_theme="Trust earned through shared vulnerability",
                    path_mood="fragile warmth",
                ),
                PathMiniArc(
                    path_id="p2",
                    path_theme="Self-preservation at the cost of connection",
                    path_mood="cold clarity",
                ),
            ],
        )
        assert len(out.arcs) == 2

    def test_duplicate_path_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"path_id.*unique"):
            Phase4eOutput(
                arcs=[
                    PathMiniArc(
                        path_id="p1",
                        path_theme="Trust earned through vulnerability",
                        path_mood="warmth",
                    ),
                    PathMiniArc(
                        path_id="p1", path_theme="Different theme entirely here", path_mood="cold"
                    ),
                ],
            )
