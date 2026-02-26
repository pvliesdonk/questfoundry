"""Tests for POLISH Phase 5 LLM output models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from questfoundry.models.polish import (
    ChoiceLabelItem,
    FalseBranchDecisionItem,
    FalseBranchSpec,
    Phase5aOutput,
    Phase5bOutput,
    Phase5cOutput,
    Phase5dOutput,
    ResidueContentItem,
    ResidueSpec,
    VariantSummaryItem,
)


class TestChoiceLabelItem:
    def test_valid(self) -> None:
        item = ChoiceLabelItem(from_passage="p1", to_passage="p2", label="Trust the mentor")
        assert item.label == "Trust the mentor"

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChoiceLabelItem(from_passage="p1", to_passage="p2", label="")


class TestPhase5aOutput:
    def test_empty_labels(self) -> None:
        out = Phase5aOutput()
        assert out.choice_labels == []

    def test_with_labels(self) -> None:
        out = Phase5aOutput(
            choice_labels=[
                ChoiceLabelItem(from_passage="p1", to_passage="p2", label="Go left"),
                ChoiceLabelItem(from_passage="p1", to_passage="p3", label="Go right"),
            ]
        )
        assert len(out.choice_labels) == 2


class TestResidueContentItem:
    def test_valid(self) -> None:
        item = ResidueContentItem(residue_id="r1", content_hint="You feel a chill")
        assert item.content_hint == "You feel a chill"

    def test_empty_hint_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResidueContentItem(residue_id="r1", content_hint="")


class TestPhase5bOutput:
    def test_empty(self) -> None:
        out = Phase5bOutput()
        assert out.residue_content == []


class TestFalseBranchDecisionItem:
    def test_skip(self) -> None:
        item = FalseBranchDecisionItem(candidate_index=0, decision="skip")
        assert item.decision == "skip"
        assert item.diamond_summary_a == ""

    def test_diamond(self) -> None:
        item = FalseBranchDecisionItem(
            candidate_index=1,
            decision="diamond",
            diamond_summary_a="Research path",
            diamond_summary_b="Accident path",
        )
        assert item.diamond_summary_a == "Research path"

    def test_sidetrack(self) -> None:
        item = FalseBranchDecisionItem(
            candidate_index=2,
            decision="sidetrack",
            sidetrack_summary="Meet a stranger",
            sidetrack_entities=["entity::stranger"],
            choice_label_enter="Approach",
            choice_label_return="Move on",
        )
        assert item.sidetrack_entities == ["entity::stranger"]

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FalseBranchDecisionItem(candidate_index=-1, decision="skip")


class TestPhase5cOutput:
    def test_empty(self) -> None:
        out = Phase5cOutput()
        assert out.decisions == []


class TestVariantSummaryItem:
    def test_valid(self) -> None:
        item = VariantSummaryItem(variant_id="v1", summary="Confident entry")
        assert item.summary == "Confident entry"


class TestPhase5dOutput:
    def test_empty(self) -> None:
        out = Phase5dOutput()
        assert out.variant_summaries == []


class TestResidueSpecContentHint:
    """Test the content_hint field added to ResidueSpec."""

    def test_default_empty(self) -> None:
        spec = ResidueSpec(target_passage_id="p1", residue_id="r1", flag="flag1")
        assert spec.content_hint == ""

    def test_with_content_hint(self) -> None:
        spec = ResidueSpec(
            target_passage_id="p1",
            residue_id="r1",
            flag="flag1",
            content_hint="You enter confidently",
        )
        assert spec.content_hint == "You enter confidently"


class TestFalseBranchSpecExtended:
    """Test the extended fields on FalseBranchSpec."""

    def test_default_fields(self) -> None:
        spec = FalseBranchSpec(
            candidate_passage_ids=["p1", "p2", "p3"],
            branch_type="skip",
        )
        assert spec.diamond_summary_a == ""
        assert spec.sidetrack_entities == []

    def test_diamond_fields(self) -> None:
        spec = FalseBranchSpec(
            candidate_passage_ids=["p1", "p2", "p3"],
            branch_type="diamond",
            diamond_summary_a="Option A",
            diamond_summary_b="Option B",
        )
        assert spec.diamond_summary_a == "Option A"

    def test_sidetrack_fields(self) -> None:
        spec = FalseBranchSpec(
            candidate_passage_ids=["p1", "p2", "p3"],
            branch_type="sidetrack",
            sidetrack_summary="Detour",
            sidetrack_entities=["entity::npc"],
            choice_label_enter="Take detour",
            choice_label_return="Continue",
        )
        assert spec.choice_label_enter == "Take detour"
