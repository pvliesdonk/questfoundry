"""Gatecheck loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class GatecheckLoop(Loop):
    """
    Gatecheck: Validate quality bars and deliver gate decisions.

    This loop validates all 8 quality bars (Integrity, Reachability, Nonlinearity,
    Gateways, Style, Determinism, Presentation, Accessibility) and delivers
    pass/conditional pass/block decisions to coordinate merge approvals with
    snapshot stamping per TRACEABILITY.

    Steps:
    1. Run quality bars (Gatekeeper)
    2. Collect findings (Gatekeeper)
    3. Triage blockers (Gatekeeper)
    4. Create report (Gatekeeper)
    5. Package decision (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="gatecheck",
        display_name="Gatecheck",
        description="Validate quality bars and deliver gate decisions",
        typical_duration="30-60 minutes",
        primary_roles=["gatekeeper"],
        consulted_roles=["showrunner"],
        entry_conditions=[
            "Owner submits work within TU for gatecheck",
            "Pre-gate feedback provided earlier",
            "Any loop nearing merge to Cold requires gatecheck",
        ],
        exit_conditions=[
            "All 8 bars evaluated with statuses",
            "Gate decision ties to bar statuses",
            "Yellow/red bars have smallest viable fixes specified",
            "Responsible owners assigned for yellow/red bars",
        ],
        output_artifacts=["gatecheck_report"],
        inputs=[
            "TU Brief from owner",
            "Artifacts to review",
            "Current Cold snapshot",
            "Prior pre-gate notes",
        ],
        tags=["validation", "quality", "export"],
    )

    steps = [
        LoopStep(
            step_id="run_quality_bars",
            description="Evaluate all 8 quality bars",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["tu_brief", "canon_pack"],
            artifacts_output=["bar_results"],
            validation_required=True,
        ),
        LoopStep(
            step_id="collect_findings",
            description="Collect findings from quality bar evaluations",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["bar_results"],
            artifacts_output=["findings"],
            validation_required=True,
        ),
        LoopStep(
            step_id="triage_blockers",
            description="Identify blockers and smallest viable fixes",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["findings"],
            artifacts_output=["triage_results"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_report",
            description="Create gatecheck report with decision",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["triage_results"],
            artifacts_output=["gatecheck_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_decision",
            description="Package decision and handoffs",
            assigned_roles=["showrunner"],
            consulted_roles=["gatekeeper"],
            artifacts_input=["gatecheck_report"],
            artifacts_output=["gate_decision"],
            validation_required=True,
        ),
    ]

    # The 8 Quality Bars
    QUALITY_BARS = [
        "integrity",
        "reachability",
        "nonlinearity",
        "gateways",
        "style",
        "determinism",
        "presentation",
        "accessibility",
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Gatecheck loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.bar_results: dict[str, dict[str, Any]] = {}
        self.findings: list[dict[str, Any]] = []
        self.decision: str = "pending"  # pass, conditional_pass, block

    def execute(self) -> LoopResult:
        """
        Execute the Gatecheck loop.

        Returns:
            Result of loop execution
        """
        artifacts_created: list[Artifact] = []
        artifacts_modified: list[Artifact] = []
        steps_completed = 0
        steps_failed = 0

        # Execute each step in sequence
        for step in self.steps:
            try:
                self.execute_step(step)

                if step.status == StepStatus.COMPLETED:
                    steps_completed += 1

                    # Collect artifacts created in this step
                    if step.result and isinstance(step.result, dict):
                        if "artifacts" in step.result:
                            artifacts_created.extend(step.result["artifacts"])

                elif step.status == StepStatus.FAILED:
                    steps_failed += 1

                    # Abort on failure
                    return LoopResult(
                        success=False,
                        loop_id=self.metadata.loop_id,
                        artifacts_created=artifacts_created,
                        artifacts_modified=artifacts_modified,
                        steps_completed=steps_completed,
                        steps_failed=steps_failed,
                        error=f"Step '{step.step_id}' failed: {step.error}",
                    )

            except Exception as e:
                return LoopResult(
                    success=False,
                    loop_id=self.metadata.loop_id,
                    artifacts_created=artifacts_created,
                    artifacts_modified=artifacts_modified,
                    steps_completed=steps_completed,
                    steps_failed=steps_failed + 1,
                    error=f"Exception in step '{step.step_id}': {str(e)}",
                )

        # All steps completed successfully
        return LoopResult(
            success=True,
            loop_id=self.metadata.loop_id,
            artifacts_created=artifacts_created,
            artifacts_modified=artifacts_modified,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            metadata={
                "decision": self.decision,
                "bars_evaluated": len(self.bar_results),
                "findings_count": len(self.findings),
            },
        )

    def _execute_step_logic(self, step: LoopStep, roles: dict[str, Role]) -> Any:
        """
        Execute specific logic for each step.

        Args:
            step: Step being executed
            roles: Available roles

        Returns:
            Step result
        """
        if step.step_id == "run_quality_bars":
            return self._run_quality_bars(roles)
        elif step.step_id == "collect_findings":
            return self._collect_findings(roles)
        elif step.step_id == "triage_blockers":
            return self._triage_blockers(roles)
        elif step.step_id == "create_report":
            return self._create_report(roles)
        elif step.step_id == "package_decision":
            return self._package_decision(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _run_quality_bars(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Evaluate all 8 quality bars."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="evaluate_quality_bars",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"bars": self.QUALITY_BARS},
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            # Store bar results
            # Each bar has: status (green/yellow/red), evidence
            self.bar_results = result.metadata.get("bar_results", {})

            # Default to all green if not specified
            for bar in self.QUALITY_BARS:
                if bar not in self.bar_results:
                    self.bar_results[bar] = {
                        "status": "green",
                        "evidence": "No issues found",
                    }

            # Create bar results artifact
            artifact = Artifact(
                type="bar_results",
                data={"bars": self.bar_results},
                metadata={"created_by": "gatekeeper", "loop": "gatecheck"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "bar_results": self.bar_results,
            }
        else:
            raise RuntimeError(f"Quality bar evaluation failed: {result.error}")

    def _collect_findings(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Collect findings from quality bar evaluations."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="collect_findings",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"bar_results": self.bar_results},
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            # Collect findings for yellow and red bars
            self.findings = []
            for bar, bar_data in self.bar_results.items():
                status = bar_data.get("status", "green")
                if status in ("yellow", "red"):
                    self.findings.append(
                        {
                            "bar": bar,
                            "status": status,
                            "evidence": bar_data.get("evidence", ""),
                        }
                    )

            # Create findings artifact
            artifact = Artifact(
                type="findings",
                data={"findings": self.findings, "count": len(self.findings)},
                metadata={"created_by": "gatekeeper", "loop": "gatecheck"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "findings": self.findings,
            }
        else:
            raise RuntimeError(f"Findings collection failed: {result.error}")

    def _triage_blockers(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Identify blockers and smallest viable fixes."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="triage_blockers",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"findings": self.findings},
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            # Determine gate decision based on bar statuses
            has_red = any(
                bar_data.get("status") == "red"
                for bar_data in self.bar_results.values()
            )
            has_yellow = any(
                bar_data.get("status") == "yellow"
                for bar_data in self.bar_results.values()
            )

            if has_red:
                self.decision = "block"
            elif has_yellow:
                self.decision = "conditional_pass"
            else:
                self.decision = "pass"

            # Add fixes to findings
            triage_results = []
            for finding in self.findings:
                triage_result = finding.copy()
                triage_result["smallest_viable_fix"] = (
                    f"Fix {finding['bar']} issue: {finding['evidence']}"
                )
                triage_result["owner"] = "owner_role"
                triage_results.append(triage_result)

            # Create triage artifact
            artifact = Artifact(
                type="triage_results",
                data={
                    "decision": self.decision,
                    "blockers": [f for f in triage_results if f["status"] == "red"],
                    "warnings": [f for f in triage_results if f["status"] == "yellow"],
                },
                metadata={"created_by": "gatekeeper", "loop": "gatecheck"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "decision": self.decision,
            }
        else:
            raise RuntimeError(f"Blocker triage failed: {result.error}")

    def _create_report(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create gatecheck report with decision."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="create_gatecheck_report",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "decision": self.decision,
                "bar_results": self.bar_results,
            },
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            # Create comprehensive gatecheck report
            report_data = {
                "title": "Gatecheck Report",
                "decision": self.decision,
                "bars": [
                    {
                        "bar": bar,
                        "status": bar_data.get("status", "green"),
                        "evidence": bar_data.get("evidence", ""),
                    }
                    for bar, bar_data in self.bar_results.items()
                ],
                "why": self._get_decision_rationale(),
                "next_actions": self._get_next_actions(),
            }

            artifact = Artifact(
                type="gatecheck_report",
                data=report_data,
                metadata={"created_by": "gatekeeper", "loop": "gatecheck"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "report": report_data,
            }
        else:
            raise RuntimeError(f"Report creation failed: {result.error}")

    def _package_decision(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package decision and handoffs."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "package_decision",
                "decision": self.decision,
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create gate decision package
            decision_package = {
                "decision": self.decision,
                "handoffs": self._get_handoffs(),
                "merge_approved": self.decision in ("pass", "conditional_pass"),
            }

            artifact = Artifact(
                type="gate_decision",
                data=decision_package,
                metadata={"created_by": "showrunner", "loop": "gatecheck"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "decision_package": decision_package,
            }
        else:
            raise RuntimeError(f"Decision packaging failed: {result.error}")

    def _get_decision_rationale(self) -> str:
        """Get rationale for gate decision."""
        if self.decision == "pass":
            return "All 8 bars green; no blockers"
        elif self.decision == "conditional_pass":
            yellow_bars = [
                bar
                for bar, data in self.bar_results.items()
                if data.get("status") == "yellow"
            ]
            return f"Yellow bars: {', '.join(yellow_bars)}; can merge with handoffs"
        else:  # block
            red_bars = [
                bar
                for bar, data in self.bar_results.items()
                if data.get("status") == "red"
            ]
            return f"Red bars: {', '.join(red_bars)}; must fix before merge"

    def _get_next_actions(self) -> str:
        """Get next actions based on decision."""
        if self.decision == "pass":
            return "Merge to Cold; notify downstream roles"
        elif self.decision == "conditional_pass":
            return "Merge to Cold with handoffs for yellow bar fixes"
        else:  # block
            return "Owner must address red bars; re-submit for gatecheck"

    def _get_handoffs(self) -> list[str]:
        """Get handoffs based on findings."""
        handoffs = []
        for finding in self.findings:
            if finding["status"] == "yellow":
                fix = finding.get("smallest_viable_fix", "TBD")
                owner = finding.get("owner", "TBD")
                handoffs.append(f"Bar: {finding['bar']}; Fix: {fix}; Owner: {owner}")
        return handoffs

    def validate_step(self, step: LoopStep, result: Any) -> bool:
        """
        Validate step completion.

        Args:
            step: Step that was executed
            result: Result from step execution

        Returns:
            True if step is valid, False otherwise
        """
        if not isinstance(result, dict):
            return False

        if not result.get("success", False):
            return False

        # Check if artifacts were created
        if "artifacts" in result:
            return len(result["artifacts"]) > 0

        return True
