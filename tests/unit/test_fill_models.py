"""Tests for FILL stage models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from questfoundry.models.fill import (
    EntityUpdate,
    FillPassageOutput,
    FillPhase0Output,
    FillPhase1Output,
    FillPhase2Output,
    FillPhaseResult,
    FillResult,
    ReviewFlag,
    VoiceDocument,
)
from questfoundry.models.pipeline import PhaseResult

# ---------------------------------------------------------------------------
# VoiceDocument
# ---------------------------------------------------------------------------


class TestVoiceDocument:
    def test_minimal_creation(self) -> None:
        doc = VoiceDocument(
            pov="third_limited",
            tense="past",
            voice_register="literary",
            sentence_rhythm="varied",
            tone_words=["atmospheric"],
        )
        assert doc.pov == "third_limited"
        assert doc.tense == "past"
        assert doc.voice_register == "literary"
        assert doc.sentence_rhythm == "varied"
        assert doc.tone_words == ["atmospheric"]
        assert doc.pov_character == ""
        assert doc.avoid_words == []
        assert doc.avoid_patterns == []
        assert doc.exemplar_passages == []

    def test_full_creation(self) -> None:
        doc = VoiceDocument(
            pov="first",
            pov_character="kay",
            tense="present",
            voice_register="conversational",
            sentence_rhythm="punchy",
            tone_words=["terse", "wry"],
            avoid_words=["suddenly", "very"],
            avoid_patterns=["adverb-heavy dialogue tags"],
            exemplar_passages=["The rain fell. I didn't care."],
        )
        assert doc.pov_character == "kay"
        assert len(doc.avoid_words) == 2
        assert len(doc.exemplar_passages) == 1

    def test_all_pov_values(self) -> None:
        for pov in ("first", "second", "third_limited", "third_omniscient"):
            doc = VoiceDocument(
                pov=pov,
                tense="past",
                voice_register="literary",
                sentence_rhythm="varied",
                tone_words=["dark"],
            )
            assert doc.pov == pov

    def test_invalid_pov_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VoiceDocument(
                pov="fourth_wall",  # type: ignore[arg-type]
                tense="past",
                voice_register="literary",
                sentence_rhythm="varied",
                tone_words=["dark"],
            )

    def test_invalid_tense_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VoiceDocument(
                pov="first",
                tense="future",  # type: ignore[arg-type]
                voice_register="literary",
                sentence_rhythm="varied",
                tone_words=["dark"],
            )

    def test_empty_tone_words_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VoiceDocument(
                pov="first",
                tense="past",
                voice_register="literary",
                sentence_rhythm="varied",
                tone_words=[],
            )

    def test_all_register_values(self) -> None:
        for reg in ("formal", "conversational", "literary", "sparse"):
            doc = VoiceDocument(
                pov="first",
                tense="past",
                voice_register=reg,
                sentence_rhythm="varied",
                tone_words=["dark"],
            )
            assert doc.voice_register == reg

    def test_all_rhythm_values(self) -> None:
        for rhythm in ("varied", "punchy", "flowing"):
            doc = VoiceDocument(
                pov="first",
                tense="past",
                voice_register="literary",
                sentence_rhythm=rhythm,
                tone_words=["dark"],
            )
            assert doc.sentence_rhythm == rhythm


# ---------------------------------------------------------------------------
# EntityUpdate
# ---------------------------------------------------------------------------


class TestEntityUpdate:
    def test_creation(self) -> None:
        update = EntityUpdate(
            entity_id="kay",
            field="appearance",
            value="a wiry woman with a scar across her left eye",
        )
        assert update.entity_id == "kay"
        assert update.field == "appearance"
        assert "scar" in update.value

    def test_empty_entity_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityUpdate(entity_id="", field="appearance", value="tall")

    def test_empty_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityUpdate(entity_id="kay", field="", value="tall")

    def test_empty_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityUpdate(entity_id="kay", field="appearance", value="")


# ---------------------------------------------------------------------------
# FillPassageOutput
# ---------------------------------------------------------------------------


class TestFillPassageOutput:
    def test_minimal_creation(self) -> None:
        output = FillPassageOutput(passage_id="p1")
        assert output.passage_id == "p1"
        assert output.prose == ""
        assert output.flag == "ok"
        assert output.flag_reason == ""
        assert output.entity_updates == []

    def test_with_prose(self) -> None:
        output = FillPassageOutput(
            passage_id="p1",
            prose="The tower stairs wound upward into darkness.",
        )
        assert "tower" in output.prose

    def test_incompatible_states_flag(self) -> None:
        output = FillPassageOutput(
            passage_id="mentor_confrontation",
            flag="incompatible_states",
            flag_reason="Character emotional state too divergent between paths",
        )
        assert output.flag == "incompatible_states"
        assert output.prose == ""
        assert "divergent" in output.flag_reason

    def test_with_entity_updates(self) -> None:
        output = FillPassageOutput(
            passage_id="p1",
            prose="Kay studied the mentor's weathered face.",
            entity_updates=[
                EntityUpdate(
                    entity_id="mentor",
                    field="appearance",
                    value="weathered face",
                )
            ],
        )
        assert len(output.entity_updates) == 1
        assert output.entity_updates[0].entity_id == "mentor"

    def test_empty_passage_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FillPassageOutput(passage_id="")


# ---------------------------------------------------------------------------
# ReviewFlag
# ---------------------------------------------------------------------------


class TestReviewFlag:
    def test_creation(self) -> None:
        flag = ReviewFlag(
            passage_id="p4",
            issue="Voice drift — more melodramatic than surrounding passages",
            issue_type="voice_drift",
        )
        assert flag.passage_id == "p4"
        assert "melodramatic" in flag.issue
        assert flag.issue_type == "voice_drift"

    def test_all_issue_types(self) -> None:
        types = [
            "voice_drift",
            "scene_type_mismatch",
            "summary_deviation",
            "continuity_break",
            "convergence_awkwardness",
            "flat_prose",
        ]
        for issue_type in types:
            flag = ReviewFlag(
                passage_id="p1",
                issue="test issue",
                issue_type=issue_type,
            )
            assert flag.issue_type == issue_type

    def test_invalid_issue_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewFlag(
                passage_id="p1",
                issue="test",
                issue_type="bad_type",  # type: ignore[arg-type]
            )

    def test_empty_issue_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewFlag(passage_id="p1", issue="", issue_type="voice_drift")


# ---------------------------------------------------------------------------
# Phase output wrappers
# ---------------------------------------------------------------------------


class TestFillPhase0Output:
    def test_wraps_voice_document(self) -> None:
        voice = VoiceDocument(
            pov="third_limited",
            tense="past",
            voice_register="literary",
            sentence_rhythm="varied",
            tone_words=["atmospheric"],
        )
        output = FillPhase0Output(voice=voice)
        assert output.voice.pov == "third_limited"


class TestFillPhase1Output:
    def test_wraps_passage_output(self) -> None:
        passage = FillPassageOutput(
            passage_id="p1",
            prose="The rain fell.",
        )
        output = FillPhase1Output(passage=passage)
        assert output.passage.passage_id == "p1"


class TestFillPhase2Output:
    def test_empty_flags(self) -> None:
        output = FillPhase2Output()
        assert output.flags == []

    def test_with_flags(self) -> None:
        output = FillPhase2Output(
            flags=[
                ReviewFlag(
                    passage_id="p4",
                    issue="voice drift",
                    issue_type="voice_drift",
                ),
            ]
        )
        assert len(output.flags) == 1


# ---------------------------------------------------------------------------
# FillPhaseResult and FillResult
# ---------------------------------------------------------------------------


class TestFillPhaseResult:
    def test_inherits_phase_result(self) -> None:
        assert issubclass(FillPhaseResult, PhaseResult)

    def test_is_phase_result_instance(self) -> None:
        result = FillPhaseResult(phase="voice", status="completed")
        assert isinstance(result, PhaseResult)

    def test_all_inherited_fields(self) -> None:
        result = FillPhaseResult(
            phase="generate",
            status="completed",
            detail="45 passages filled",
            llm_calls=45,
            tokens_used=90000,
        )
        assert result.llm_calls == 45
        assert result.tokens_used == 90000


class TestFillResult:
    def test_defaults(self) -> None:
        result = FillResult()
        assert result.passages_filled == 0
        assert result.passages_flagged == 0
        assert result.entity_updates_applied == 0
        assert result.review_cycles == 0
        assert result.phases_completed == []

    def test_full_creation(self) -> None:
        result = FillResult(
            passages_filled=45,
            passages_flagged=3,
            entity_updates_applied=12,
            review_cycles=1,
            phases_completed=[
                FillPhaseResult(phase="voice", status="completed"),
                FillPhaseResult(phase="generate", status="completed", llm_calls=45),
                FillPhaseResult(phase="review", status="completed", llm_calls=5),
                FillPhaseResult(phase="revision", status="completed", llm_calls=3),
            ],
        )
        assert result.passages_filled == 45
        assert len(result.phases_completed) == 4
