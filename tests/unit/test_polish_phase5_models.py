"""Tests for POLISH Phase 5 LLM output models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from questfoundry.models.polish import (
    ChoiceLabelItem,
    FalseBranchDecisionItem,
    FalseBranchSpec,
    PassageSpec,
    Phase5aOutput,
    Phase5bOutput,
    Phase5cOutput,
    Phase5dOutput,
    ResidueContentItem,
    ResidueSpec,
    VariantSummaryItem,
)


class TestPassageSpecSummaryRequired:
    """Stage Output Contract item 2 + #1527: every passage MUST have a non-empty summary.

    Phase 4a constructs PassageSpec from beat summaries (which are guaranteed
    non-empty by SEED's InitialBeat.summary `min_length=1` and POLISH's
    GapProposal.summary `min_length=1`). FILL has no prose context without it.
    """

    def test_empty_summary_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"summary"):
            PassageSpec(passage_id="passage::test", beat_ids=["beat::a"], summary="")

    def test_missing_summary_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"summary"):
            PassageSpec(passage_id="passage::test", beat_ids=["beat::a"])  # type: ignore[call-arg]

    def test_non_empty_summary_accepted(self) -> None:
        spec = PassageSpec(
            passage_id="passage::test", beat_ids=["beat::a"], summary="Hero enters the cave."
        )
        assert spec.summary == "Hero enters the cave."


class TestChoiceLabelItem:
    def test_valid(self) -> None:
        item = ChoiceLabelItem(from_passage="p1", to_passage="p2", label="Trust the mentor")
        assert item.label == "Trust the mentor"

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChoiceLabelItem(from_passage="p1", to_passage="p2", label="")


class TestPhase5aOutput:
    def test_empty_labels_rejected(self) -> None:
        """Phase 5a empty choice_labels = LLM failure (#1527 retry-bypass).

        POLISH has no semantic_validator hook (#1498); Pydantic is the only
        in-retry enforcement point. Empty list silently leaves all
        ChoiceSpec.label at their default `""`; Phase 7 R-7.7 catches it
        at exit AFTER Phase 6 commits — too late for repair.
        """
        with pytest.raises(ValidationError, match=r"at least 1 item"):
            Phase5aOutput(choice_labels=[])

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

    def test_sidetrack_requires_summary(self) -> None:
        """R-5.9: sidetrack decision requires non-empty sidetrack_summary (#1527).

        Closes a retry-bypass: empty summary previously passed Pydantic
        and Phase 6 created sidetrack beats with no prose context.
        """
        with pytest.raises(ValidationError, match=r"R-5\.9"):
            FalseBranchDecisionItem(
                candidate_index=0,
                decision="sidetrack",
                sidetrack_summary="",  # empty — must be rejected
            )

    @pytest.mark.parametrize(
        ("summary_a", "summary_b"),
        [
            pytest.param("Path A", "", id="missing_b"),
            pytest.param("", "Path B", id="missing_a"),
            pytest.param("", "", id="missing_both"),
        ],
    )
    def test_diamond_requires_both_summaries(self, summary_a: str, summary_b: str) -> None:
        """R-5.10: diamond decision requires both diamond_summary_a AND _b (#1527).

        Each parametrize case runs independently — if `missing_b` were to
        regress, `missing_a` and `missing_both` still get their own
        assertion (vs nested `pytest.raises` blocks where the first
        unexpectedly-not-raising case short-circuits the rest).
        """
        with pytest.raises(ValidationError, match=r"R-5\.10"):
            FalseBranchDecisionItem(
                candidate_index=0,
                decision="diamond",
                diamond_summary_a=summary_a,
                diamond_summary_b=summary_b,
            )


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
