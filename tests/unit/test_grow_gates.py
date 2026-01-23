"""Tests for GROW phase gate infrastructure."""

from __future__ import annotations

import pytest

from questfoundry.graph.mutations import (
    GrowErrorCategory,
    GrowMutationError,
    GrowValidationError,
    has_mutation_handler,
)
from questfoundry.models.grow import GrowPhaseResult
from questfoundry.pipeline.gates import AutoApprovePhaseGate, PhaseGateHook


class TestPhaseGateHookProtocol:
    """Verify PhaseGateHook Protocol conformance."""

    def test_auto_approve_phase_gate_satisfies_protocol(self) -> None:
        gate: PhaseGateHook = AutoApprovePhaseGate()
        assert hasattr(gate, "on_phase_complete")

    @pytest.mark.asyncio
    async def test_custom_gate_satisfies_protocol(self) -> None:
        class RejectAllPhaseGate:
            async def on_phase_complete(
                self, _stage: str, _phase: str, _result: GrowPhaseResult
            ) -> str:
                return "reject"

        gate: PhaseGateHook = RejectAllPhaseGate()  # type: ignore[assignment]
        result = GrowPhaseResult(phase="test", status="completed")
        decision = await gate.on_phase_complete("grow", "test", result)
        assert decision == "reject"


class TestAutoApprovePhaseGate:
    @pytest.mark.asyncio
    async def test_always_returns_approve(self) -> None:
        gate = AutoApprovePhaseGate()
        result = GrowPhaseResult(phase="validate_dag", status="completed")
        decision = await gate.on_phase_complete("grow", "validate_dag", result)
        assert decision == "approve"

    @pytest.mark.asyncio
    async def test_approves_failed_phase(self) -> None:
        gate = AutoApprovePhaseGate()
        result = GrowPhaseResult(phase="arcs", status="failed", detail="error")
        decision = await gate.on_phase_complete("grow", "arcs", result)
        assert decision == "approve"

    @pytest.mark.asyncio
    async def test_approves_skipped_phase(self) -> None:
        gate = AutoApprovePhaseGate()
        result = GrowPhaseResult(phase="knots", status="skipped")
        decision = await gate.on_phase_complete("grow", "knots", result)
        assert decision == "approve"


class TestHasMutationHandlerGrow:
    def test_grow_is_registered(self) -> None:
        assert has_mutation_handler("grow") is True

    def test_other_stages_still_registered(self) -> None:
        assert has_mutation_handler("dream") is True
        assert has_mutation_handler("brainstorm") is True
        assert has_mutation_handler("seed") is True

    def test_unknown_stage_not_registered(self) -> None:
        assert has_mutation_handler("unknown") is False


class TestGrowErrorCategory:
    def test_structural_category(self) -> None:
        assert GrowErrorCategory.STRUCTURAL.name == "STRUCTURAL"

    def test_combinatorial_category(self) -> None:
        assert GrowErrorCategory.COMBINATORIAL.name == "COMBINATORIAL"

    def test_reference_category(self) -> None:
        assert GrowErrorCategory.REFERENCE.name == "REFERENCE"

    def test_fatal_category(self) -> None:
        assert GrowErrorCategory.FATAL.name == "FATAL"

    def test_categories_are_distinct(self) -> None:
        categories = [
            GrowErrorCategory.STRUCTURAL,
            GrowErrorCategory.COMBINATORIAL,
            GrowErrorCategory.REFERENCE,
            GrowErrorCategory.FATAL,
        ]
        assert len(set(categories)) == 4


class TestGrowValidationError:
    def test_minimal_creation(self) -> None:
        error = GrowValidationError(field_path="arc.threads", issue="empty list")
        assert error.field_path == "arc.threads"
        assert error.issue == "empty list"
        assert error.available == []
        assert error.provided == ""
        assert error.category is None

    def test_full_creation(self) -> None:
        error = GrowValidationError(
            field_path="arc.threads.0",
            issue="thread not found in graph",
            available=["thread_a", "thread_b"],
            provided="thread_x",
            category=GrowErrorCategory.REFERENCE,
        )
        assert error.provided == "thread_x"
        assert error.category == GrowErrorCategory.REFERENCE
        assert "thread_a" in error.available


class TestGrowMutationError:
    def test_creation_with_single_error(self) -> None:
        errors = [
            GrowValidationError(
                field_path="dag",
                issue="cycle detected between beat_1 and beat_3",
                category=GrowErrorCategory.STRUCTURAL,
            )
        ]
        exc = GrowMutationError(errors)
        assert len(exc.errors) == 1
        assert "cycle detected" in str(exc)

    def test_creation_with_multiple_errors(self) -> None:
        errors = [
            GrowValidationError(
                field_path="arc.threads.0",
                issue="not found",
                available=["t1", "t2"],
                provided="t_invalid",
                category=GrowErrorCategory.REFERENCE,
            ),
            GrowValidationError(
                field_path="arc_count",
                issue="exceeds limit of 32",
                category=GrowErrorCategory.COMBINATORIAL,
            ),
        ]
        exc = GrowMutationError(errors)
        assert len(exc.errors) == 2

    def test_to_feedback_includes_categories(self) -> None:
        errors = [
            GrowValidationError(
                field_path="dag",
                issue="cycle detected",
                category=GrowErrorCategory.STRUCTURAL,
            )
        ]
        exc = GrowMutationError(errors)
        feedback = exc.to_feedback()
        assert "[STRUCTURAL]" in feedback
        assert "cycle detected" in feedback

    def test_to_feedback_no_category(self) -> None:
        errors = [GrowValidationError(field_path="test", issue="problem")]
        exc = GrowMutationError(errors)
        feedback = exc.to_feedback()
        assert "problem" in feedback
        assert "[" not in feedback.split("problem")[0].split("test")[-1]

    def test_to_feedback_with_suggestions(self) -> None:
        errors = [
            GrowValidationError(
                field_path="arc.threads.0",
                issue="not found",
                available=["mentor_trust_canonical"],
                provided="mentor_trust_canon",
                category=GrowErrorCategory.REFERENCE,
            )
        ]
        exc = GrowMutationError(errors)
        feedback = exc.to_feedback()
        assert "mentor_trust_canonical" in feedback

    def test_is_mutation_error(self) -> None:
        exc = GrowMutationError([])
        assert isinstance(exc, ValueError)

    def test_truncates_many_errors(self) -> None:
        errors = [
            GrowValidationError(field_path=f"field_{i}", issue=f"error {i}") for i in range(20)
        ]
        exc = GrowMutationError(errors)
        feedback = exc.to_feedback()
        assert "... and 12 more errors" in feedback
