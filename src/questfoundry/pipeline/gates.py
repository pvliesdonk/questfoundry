"""Gate hooks for pipeline stage transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from questfoundry.models.pipeline import PhaseResult
    from questfoundry.pipeline.orchestrator import StageResult


class GateHook(Protocol):
    """Protocol for gate hooks that approve/reject stage transitions."""

    async def on_stage_complete(
        self,
        stage: str,
        result: StageResult,
    ) -> Literal["approve", "reject"]:
        """Called when a stage completes.

        Args:
            stage: Name of the completed stage.
            result: Result of the stage execution.

        Returns:
            "approve" to continue or "reject" to halt pipeline.
        """
        ...


class AutoApproveGate:
    """Gate that automatically approves all stage transitions.

    This is the default gate for Slice 1 where human review
    is not yet implemented.
    """

    async def on_stage_complete(
        self,
        _stage: str,
        _result: StageResult,
    ) -> Literal["approve", "reject"]:
        """Automatically approve all completed stages.

        Args:
            _stage: Name of the completed stage (unused).
            _result: Result of the stage execution (unused).

        Returns:
            Always returns "approve".
        """
        return "approve"


class RequireSuccessGate:
    """Gate that only approves successful stage completions.

    Rejects any stage that completed with errors.
    """

    async def on_stage_complete(
        self,
        _stage: str,
        result: StageResult,
    ) -> Literal["approve", "reject"]:
        """Approve only if stage completed without errors.

        Args:
            _stage: Name of the completed stage (unused).
            result: Result of the stage execution.

        Returns:
            "approve" if no errors, "reject" otherwise.
        """
        if result.errors:
            return "reject"
        return "approve"


class PhaseGateHook(Protocol):
    """Protocol for intra-stage phase gates used by multi-phase stages.

    Unlike GateHook which gates between pipeline stages, PhaseGateHook
    gates between phases within a single stage (GROW, FILL). This allows
    stopping execution mid-stage and rolling back to a previous phase snapshot.
    """

    async def on_phase_complete(
        self,
        stage: str,
        phase: str,
        result: PhaseResult,
    ) -> Literal["approve", "reject"]:
        """Called when a phase within a stage completes.

        Args:
            stage: Name of the stage (e.g., "grow").
            phase: Name of the completed phase (e.g., "validate_dag").
            result: Result of the phase execution.

        Returns:
            "approve" to continue or "reject" to halt and rollback.
        """
        ...


class AutoApprovePhaseGate:
    """Phase gate that automatically approves all phase transitions.

    Default gate for multi-phase stages where human review is not yet implemented.
    """

    async def on_phase_complete(
        self,
        _stage: str,
        _phase: str,
        _result: PhaseResult,
    ) -> Literal["approve", "reject"]:
        """Automatically approve all completed phases.

        Args:
            _stage: Name of the stage (unused).
            _phase: Name of the phase (unused).
            _result: Phase result (unused).

        Returns:
            Always returns "approve".
        """
        return "approve"
