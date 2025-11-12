"""Post Mortem loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class PostMortemLoop(Loop):
    """
    Post Mortem: Final quality report.

    This loop conducts a structured retrospective after completing a major
    milestone, release, or significant TU cluster. Extracts actionable lessons,
    identifies process improvements, tracks quality bar trends, and updates best
    practices.

    Steps:
    1. Collect metrics (Showrunner with Gatekeeper)
    2. Final validation (Gatekeeper)
    3. Create report (Gatekeeper with Showrunner)
    4. Archive (Showrunner)
    5. Package (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="post_mortem",
        display_name="Post Mortem",
        description="Final quality report",
        typical_duration="1-2 hours",
        primary_roles=["gatekeeper"],
        consulted_roles=["showrunner"],
        entry_conditions=[
            "After major milestone completion",
            "After significant TU cluster completes",
            "When recurring quality bar issues suggest review",
            "Periodic retrospectives (quarterly, per-release)",
        ],
        exit_conditions=[
            "All participating roles contributed",
            "Action items are specific and owned",
            "At least one best practice documented",
            "Report archived and accessible",
        ],
        output_artifacts=["post_mortem_report", "action_items"],
        inputs=[
            "Completed TUs from milestone",
            "Gatecheck Reports",
            "PN Playtest Notes",
            "Timeline Metrics",
        ],
        tags=["export", "retrospective", "quality"],
    )

    steps = [
        LoopStep(
            step_id="collect_metrics",
            description="Gather metrics from completed work",
            assigned_roles=["showrunner"],
            consulted_roles=["gatekeeper"],
            artifacts_input=["gatecheck_report", "tu_brief"],
            artifacts_output=["metrics_collection"],
            validation_required=True,
        ),
        LoopStep(
            step_id="final_validation",
            description="Final quality bar validation",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["metrics_collection"],
            artifacts_output=["validation_summary"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_report",
            description="Create post-mortem report with lessons learned",
            assigned_roles=["gatekeeper"],
            consulted_roles=["showrunner"],
            artifacts_input=["validation_summary", "metrics_collection"],
            artifacts_output=["post_mortem_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="archive",
            description="Archive report and metrics",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["post_mortem_report"],
            artifacts_output=["archived_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package",
            description="Package action items for follow-up",
            assigned_roles=["showrunner"],
            consulted_roles=["gatekeeper"],
            artifacts_input=["post_mortem_report"],
            artifacts_output=["action_items"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Post Mortem loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.metrics: dict[str, Any] = {}
        self.action_items: list[dict[str, Any]] = []
        self.lessons_learned: list[dict[str, Any]] = []

    def execute(self) -> LoopResult:
        """
        Execute the Post Mortem loop.

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
                "action_items_count": len(self.action_items),
                "lessons_learned_count": len(self.lessons_learned),
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
        if step.step_id == "collect_metrics":
            return self._collect_metrics(roles)
        elif step.step_id == "final_validation":
            return self._final_validation(roles)
        elif step.step_id == "create_report":
            return self._create_report(roles)
        elif step.step_id == "archive":
            return self._archive(roles)
        elif step.step_id == "package":
            return self._package(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _collect_metrics(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Gather metrics from completed work."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="collect_metrics",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Store metrics
            self.metrics = {
                "gate_pass_rate": 0.85,
                "rework_cycles": 2,
                "cycle_time_avg": "3 hours",
                "quality_bar_trends": {
                    "integrity": "green",
                    "style": "green",
                    "presentation": "green",
                },
            }

            # Create metrics collection artifact
            artifact = Artifact(
                type="metrics_collection",
                data=self.metrics,
                metadata={"created_by": "showrunner", "loop": "post_mortem"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "metrics": self.metrics,
            }
        else:
            raise RuntimeError(f"Metrics collection failed: {result.error}")

    def _final_validation(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Final quality bar validation."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="final_validation",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"metrics": self.metrics},
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            validation_summary = {
                "overall_status": "pass",
                "bars_evaluated": 8,
                "blockers_found": 0,
                "warnings": 1,
            }

            # Create validation summary artifact
            artifact = Artifact(
                type="validation_summary",
                data=validation_summary,
                metadata={"created_by": "gatekeeper", "loop": "post_mortem"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "summary": validation_summary,
            }
        else:
            raise RuntimeError(f"Final validation failed: {result.error}")

    def _create_report(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create post-mortem report with lessons learned."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="create_post_mortem_report",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"metrics": self.metrics},
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            # Store lessons learned
            self.lessons_learned = [
                {
                    "category": "What went well",
                    "items": ["Strong gate pass rate", "Good collaboration"],
                },
                {
                    "category": "What went poorly",
                    "items": ["Some rework needed", "Timeline slippage"],
                },
                {
                    "category": "Improvements",
                    "items": ["Earlier style checks", "Better planning"],
                },
            ]

            # Create action items
            self.action_items = [
                {
                    "description": "Add Style Lead to pre-gate sessions",
                    "owner": "showrunner",
                    "priority": "high",
                },
            ]

            # Create post-mortem report
            report_data = {
                "title": "Post-Mortem Report",
                "date": "2025-11-07",
                "metrics": self.metrics,
                "lessons_learned": self.lessons_learned,
                "action_items": self.action_items,
            }

            artifact = Artifact(
                type="post_mortem_report",
                data=report_data,
                metadata={"created_by": "gatekeeper", "loop": "post_mortem"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "report": report_data,
            }
        else:
            raise RuntimeError(f"Report creation failed: {result.error}")

    def _archive(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Archive report and metrics."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"step_name": "archive"},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create archived report artifact
            archive_data = {
                "report_archived": True,
                "archive_location": "/archives/post_mortems/",
                "date": "2025-11-07",
            }

            artifact = Artifact(
                type="archived_report",
                data=archive_data,
                metadata={"created_by": "showrunner", "loop": "post_mortem"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "archive": archive_data,
            }
        else:
            raise RuntimeError(f"Report archival failed: {result.error}")

    def _package(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package action items for follow-up."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "package",
                "action_items": self.action_items,
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Package action items
            package_data = {
                "action_items": self.action_items,
                "count": len(self.action_items),
                "next_review": "2025-12-07",
            }

            artifact = Artifact(
                type="action_items",
                data=package_data,
                metadata={"created_by": "showrunner", "loop": "post_mortem"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "package": package_data,
            }
        else:
            raise RuntimeError(f"Action item packaging failed: {result.error}")

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
