"""Tests for POLISH stage Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from questfoundry.models.polish import (
    ArcPivot,
    CharacterArcMetadata,
    MicroBeatProposal,
    Phase1Output,
    Phase2Output,
    Phase3Output,
    PolishPhaseResult,
    ReorderedSection,
)


class TestReorderedSection:
    """Tests for Phase 1 output model."""

    def test_valid_section(self) -> None:
        section = ReorderedSection(
            section_id="section_0",
            beat_ids=["beat::a", "beat::b", "beat::c"],
            rationale="Better scene-sequel rhythm",
        )
        assert section.section_id == "section_0"
        assert len(section.beat_ids) == 3

    def test_empty_section_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            ReorderedSection(section_id="", beat_ids=["beat::a"], rationale="test")

    def test_empty_beat_ids_fails(self) -> None:
        with pytest.raises(ValidationError):
            ReorderedSection(section_id="s1", beat_ids=[], rationale="test")

    def test_empty_rationale_fails(self) -> None:
        with pytest.raises(ValidationError):
            ReorderedSection(section_id="s1", beat_ids=["beat::a"], rationale="")


class TestPhase1Output:
    """Tests for Phase 1 output container."""

    def test_empty_reorderings(self) -> None:
        output = Phase1Output(reordered_sections=[])
        assert output.reordered_sections == []

    def test_with_sections(self) -> None:
        output = Phase1Output(
            reordered_sections=[
                ReorderedSection(
                    section_id="s0",
                    beat_ids=["beat::a", "beat::b"],
                    rationale="test",
                )
            ]
        )
        assert len(output.reordered_sections) == 1

    def test_defaults_to_empty(self) -> None:
        output = Phase1Output()
        assert output.reordered_sections == []


class TestMicroBeatProposal:
    """Tests for Phase 2 output model."""

    def test_valid_micro_beat(self) -> None:
        mb = MicroBeatProposal(
            after_beat_id="beat::conflict",
            summary="A moment of quiet settles over the room",
            entity_ids=["entity::mentor"],
        )
        assert mb.after_beat_id == "beat::conflict"
        assert len(mb.entity_ids) == 1

    def test_empty_summary_fails(self) -> None:
        with pytest.raises(ValidationError):
            MicroBeatProposal(after_beat_id="beat::a", summary="")

    def test_no_entities_ok(self) -> None:
        mb = MicroBeatProposal(
            after_beat_id="beat::a",
            summary="Wind whistles through the corridor",
        )
        assert mb.entity_ids == []


class TestPhase2Output:
    """Tests for Phase 2 output container."""

    def test_empty_micro_beats(self) -> None:
        output = Phase2Output(micro_beats=[])
        assert output.micro_beats == []

    def test_defaults_to_empty(self) -> None:
        output = Phase2Output()
        assert output.micro_beats == []


class TestCharacterArcMetadata:
    """Tests for Phase 3 output model."""

    def test_full_arc(self) -> None:
        arc = CharacterArcMetadata(
            entity_id="entity::mentor",
            start="The mentor appears as a calm authority figure",
            pivots=[
                ArcPivot(
                    path_id="path::trust",
                    beat_id="beat::reveal",
                    description="The mentor reveals hidden knowledge",
                ),
            ],
            end_per_path={
                "path::trust": "The mentor becomes a true ally",
                "path::doubt": "The mentor is estranged",
            },
        )
        assert arc.entity_id == "entity::mentor"
        assert len(arc.pivots) == 1
        assert len(arc.end_per_path) == 2

    def test_empty_entity_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            CharacterArcMetadata(
                entity_id="",
                start="test",
            )

    def test_no_pivots_ok(self) -> None:
        arc = CharacterArcMetadata(
            entity_id="entity::npc",
            start="A background character",
        )
        assert arc.pivots == []
        assert arc.end_per_path == {}


class TestPhase3Output:
    """Tests for Phase 3 output container."""

    def test_defaults_to_empty(self) -> None:
        output = Phase3Output()
        assert output.character_arcs == []


class TestPolishPhaseResult:
    """Tests for the phase result container."""

    def test_default_values(self) -> None:
        result = PolishPhaseResult(phase="test")
        assert result.status == "completed"
        assert result.llm_calls == 0
        assert result.tokens_used == 0

    def test_custom_values(self) -> None:
        result = PolishPhaseResult(
            phase="beat_reordering",
            status="completed",
            detail="Reordered 3/5 sections",
            llm_calls=5,
            tokens_used=12000,
        )
        assert result.llm_calls == 5
