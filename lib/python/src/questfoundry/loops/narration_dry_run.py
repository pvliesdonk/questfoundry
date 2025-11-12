"""Narration Dry Run loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class NarrationDryRunLoop(Loop):
    """
    Narration Dry Run: Test player experience.

    This loop has the Player-Narrator play through the current Cold snapshot
    exactly as a player would experience it—in-world, spoiler-safe—to surface
    UX issues in comprehension, choice clarity, pacing, and diegetic gate
    enforcement.

    Steps:
    1. Select path (Showrunner with PN)
    2. Simulate playthrough (Player Narrator)
    3. Identify issues (Player Narrator)
    4. Report (Player Narrator)
    5. Package (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="narration_dry_run",
        display_name="Narration Dry Run",
        description="Test player experience",
        typical_duration="2-4 hours",
        primary_roles=["player_narrator"],
        consulted_roles=["showrunner", "gatekeeper"],
        entry_conditions=[
            "After a Binding Run exports a view on Cold",
            "Before user playtests, recordings, or live demos",
            "When PN phrasing patterns were recently updated",
        ],
        exit_conditions=[
            "Gates enforced diegetically without confusion",
            "Choices read as distinct and fair",
            "Navigation is smooth across formats",
            "Actionable follow-ups documented",
        ],
        output_artifacts=["playtest_notes", "issue_summary"],
        inputs=[
            "Exported bundle from Binding Run",
            "PN Principles",
            "Style guardrails",
            "Language Pack (if testing localization)",
        ],
        tags=["export", "testing", "player-facing"],
    )

    steps = [
        LoopStep(
            step_id="select_path",
            description="Choose representative paths to test",
            assigned_roles=["showrunner"],
            consulted_roles=["player_narrator"],
            artifacts_input=["export_bundle"],
            artifacts_output=["path_plan"],
            validation_required=True,
        ),
        LoopStep(
            step_id="simulate_playthrough",
            description="Narrate in-voice and perform the book",
            assigned_roles=["player_narrator"],
            consulted_roles=[],
            artifacts_input=["path_plan", "export_bundle"],
            artifacts_output=["playthrough_log"],
            validation_required=True,
        ),
        LoopStep(
            step_id="identify_issues",
            description="Log UX issues with actionable notes",
            assigned_roles=["player_narrator"],
            consulted_roles=["gatekeeper"],
            artifacts_input=["playthrough_log"],
            artifacts_output=["playtest_notes"],
            validation_required=True,
        ),
        LoopStep(
            step_id="report",
            description="Create summary of findings by issue type",
            assigned_roles=["player_narrator"],
            consulted_roles=["showrunner"],
            artifacts_input=["playtest_notes"],
            artifacts_output=["issue_summary"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package",
            description="Package findings and route to follow-up loops",
            assigned_roles=["showrunner"],
            consulted_roles=["player_narrator"],
            artifacts_input=["issue_summary"],
            artifacts_output=["followup_package"],
            validation_required=True,
        ),
    ]

    # Issue types from playbook
    ISSUE_TYPES = [
        "choice-ambiguity",
        "gate-friction",
        "recap-needed",
        "codex-invite",
        "leak-risk",
        "nav-bug",
        "tone-wobble",
        "accessibility",
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Narration Dry Run loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.paths_selected: list[str] = []
        self.issues_found: list[dict[str, Any]] = []
        self.issue_counts: dict[str, int] = {t: 0 for t in self.ISSUE_TYPES}

    def execute(self) -> LoopResult:
        """
        Execute the Narration Dry Run loop.

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
                "paths_tested": len(self.paths_selected),
                "issues_found": len(self.issues_found),
                "issue_counts": self.issue_counts,
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
        if step.step_id == "select_path":
            return self._select_path(roles)
        elif step.step_id == "simulate_playthrough":
            return self._simulate_playthrough(roles)
        elif step.step_id == "identify_issues":
            return self._identify_issues(roles)
        elif step.step_id == "report":
            return self._report(roles)
        elif step.step_id == "package":
            return self._package(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _select_path(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Choose representative paths to test."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"step_name": "select_path"},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Store selected paths
            self.paths_selected = result.metadata.get(
                "paths", ["hub-route", "loop-return", "gated-branch"]
            )

            # Create path plan artifact
            artifact = Artifact(
                type="path_plan",
                data={"paths": self.paths_selected, "count": len(self.paths_selected)},
                metadata={"created_by": "showrunner", "loop": "narration_dry_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "paths": self.paths_selected,
            }
        else:
            raise RuntimeError(f"Path selection failed: {result.error}")

    def _simulate_playthrough(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Narrate in-voice and perform the book."""
        player_narrator = roles["player_narrator"]

        context = RoleContext(
            task="perform_narration",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"paths": self.paths_selected},
        )

        result = player_narrator.execute_task(context)

        if result.success:
            playthrough_data = result.metadata.get("playthrough", {})

            # Create playthrough log artifact
            artifact = Artifact(
                type="playthrough_log",
                data=playthrough_data,
                metadata={"created_by": "player_narrator", "loop": "narration_dry_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "playthrough": playthrough_data,
            }
        else:
            raise RuntimeError(f"Playthrough simulation failed: {result.error}")

    def _identify_issues(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Log UX issues with actionable notes."""
        player_narrator = roles["player_narrator"]

        context = RoleContext(
            task="identify_issues",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = player_narrator.execute_task(context)

        if result.success:
            # Store issues
            self.issues_found = result.metadata.get("issues", [])

            # Count issues by type
            for issue in self.issues_found:
                issue_type = issue.get("type", "unknown")
                if issue_type in self.issue_counts:
                    self.issue_counts[issue_type] += 1

            # Create playtest notes artifact
            artifact = Artifact(
                type="playtest_notes",
                data={"issues": self.issues_found, "count": len(self.issues_found)},
                metadata={"created_by": "player_narrator", "loop": "narration_dry_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "issues": self.issues_found,
            }
        else:
            raise RuntimeError(f"Issue identification failed: {result.error}")

    def _report(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create summary of findings by issue type."""
        player_narrator = roles["player_narrator"]

        context = RoleContext(
            task="create_report",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"issues": self.issues_found},
        )

        result = player_narrator.execute_task(context)

        if result.success:
            # Create issue summary
            summary_data = {
                "total_issues": len(self.issues_found),
                "by_type": self.issue_counts,
                "severity": result.metadata.get("severity", "low"),
                "recommended_followups": self._get_recommended_followups(),
            }

            # Create issue summary artifact
            artifact = Artifact(
                type="issue_summary",
                data=summary_data,
                metadata={"created_by": "player_narrator", "loop": "narration_dry_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "summary": summary_data,
            }
        else:
            raise RuntimeError(f"Report creation failed: {result.error}")

    def _package(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package findings and route to follow-up loops."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"step_name": "package"},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create followup package
            package_data = {
                "issues": self.issues_found,
                "followup_loops": self._get_recommended_followups(),
                "priority": "medium",
            }

            artifact = Artifact(
                type="followup_package",
                data=package_data,
                metadata={"created_by": "showrunner", "loop": "narration_dry_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "package": package_data,
            }
        else:
            raise RuntimeError(f"Packaging failed: {result.error}")

    def _get_recommended_followups(self) -> list[str]:
        """Get recommended follow-up loops based on issues found."""
        followups = []

        if self.issue_counts.get("tone-wobble", 0) > 0:
            followups.append("Style Tune-up")
        if self.issue_counts.get("codex-invite", 0) > 0:
            followups.append("Codex Expansion")
        if self.issue_counts.get("nav-bug", 0) > 0:
            followups.append("Binding Run")

        return followups if followups else ["None"]

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
