"""Story Spark loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class StorySparkLoop(Loop):
    """
    Story Spark: Transform initial quest concept into first draft.

    This loop introduces or reshapes narrative structure, creating topology,
    section briefs, and scene drafts. It's the foundational loop for quest
    creation.

    Steps:
    1. Generate quest hooks (Plotwright)
    2. Create narrative topology (Plotwright)
    3. Create TU brief (Plotwright)
    4. Create section briefs (Plotwright)
    5. Draft scenes (Scene Smith)
    6. Pre-gate check (Gatekeeper)
    7. Refinement iteration (if needed)
    """

    metadata = LoopMetadata(
        loop_id="story_spark",
        display_name="Story Spark",
        description="Introduce or reshape narrative structure",
        typical_duration="2-4 hours",
        primary_roles=["plotwright", "scene_smith"],
        consulted_roles=["style_lead", "lore_weaver", "codex_curator", "gatekeeper"],
        entry_conditions=[
            "New chapter/act/subplot needed",
            "Restructure request",
            "Reachability/nonlinearity fixes",
        ],
        exit_conditions=[
            "Topology stabilized",
            "Section briefs complete",
            "Scene drafts ready",
            "Gatekeeper preview passed",
        ],
        output_artifacts=["tu_brief", "hook_card", "canon_pack"],
        inputs=["Cold snapshot", "Prior topology notes", "Open hooks", "QA findings"],
        tags=["structure", "content", "foundation"],
    )

    steps = [
        LoopStep(
            step_id="generate_hooks",
            description="Generate quest hooks",
            assigned_roles=["plotwright"],
            consulted_roles=["lore_weaver"],
            artifacts_input=["project_metadata"],
            artifacts_output=["hook_card"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_topology",
            description="Design narrative topology (hubs, loops, gateways)",
            assigned_roles=["plotwright"],
            consulted_roles=["lore_weaver", "style_lead"],
            artifacts_input=["hook_card"],
            artifacts_output=["tu_brief"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_tu_brief",
            description="Create TU brief with narrative structure",
            assigned_roles=["plotwright"],
            consulted_roles=["gatekeeper"],
            artifacts_input=["hook_card"],
            artifacts_output=["tu_brief"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_section_briefs",
            description="Create section briefs for Scene Smith",
            assigned_roles=["plotwright"],
            consulted_roles=["style_lead", "codex_curator"],
            artifacts_input=["tu_brief"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="draft_scenes",
            description="Draft scenes from section briefs",
            assigned_roles=["scene_smith"],
            consulted_roles=["style_lead"],
            artifacts_input=["tu_brief", "canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="pre_gate_check",
            description="Gatekeeper pre-gate quality check",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["tu_brief", "canon_pack"],
            artifacts_output=["gatecheck_report"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Story Spark loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.hooks_generated: list[dict[str, Any]] = []
        self.iteration_count = 0
        self.max_iterations = context.config.get("max_iterations", 3)

    def execute(self) -> LoopResult:
        """
        Execute the Story Spark loop.

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

                    # Decide how to handle failure
                    if step.step_id == "pre_gate_check":
                        # Check if we should iterate
                        if self.iteration_count < self.max_iterations:
                            self.iteration_count += 1
                            # Go back to refinement
                            return self._handle_refinement_iteration(
                                step, artifacts_created, artifacts_modified
                            )
                        else:
                            # Max iterations reached
                            return LoopResult(
                                success=False,
                                loop_id=self.metadata.loop_id,
                                artifacts_created=artifacts_created,
                                artifacts_modified=artifacts_modified,
                                steps_completed=steps_completed,
                                steps_failed=steps_failed,
                                error=(
                                    f"Max iterations ({self.max_iterations}) "
                                    "reached, quality not achieved"
                                ),
                            )
                    else:
                        # Other step failed - abort
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
            metadata={"iterations": self.iteration_count},
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
        if step.step_id == "generate_hooks":
            return self._generate_hooks(roles)
        elif step.step_id == "create_topology":
            return self._create_topology(roles)
        elif step.step_id == "create_tu_brief":
            return self._create_tu_brief(roles)
        elif step.step_id == "create_section_briefs":
            return self._create_section_briefs(roles)
        elif step.step_id == "draft_scenes":
            return self._draft_scenes(roles)
        elif step.step_id == "pre_gate_check":
            return self._pre_gate_check(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _generate_hooks(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Generate quest hooks."""
        plotwright = roles["plotwright"]

        context = RoleContext(
            task="generate_hooks",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = plotwright.execute_task(context)

        if result.success:
            # Store hooks for later use
            self.hooks_generated = result.metadata.get("hooks", [])

            # Create hook card artifacts
            artifacts = []
            for hook in self.hooks_generated:
                artifact = Artifact(
                    type="hook_card",
                    data=hook,
                    metadata={"created_by": "plotwright", "loop": "story_spark"},
                )
                artifacts.append(artifact)
                self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": artifacts,
                "hooks": self.hooks_generated,
            }
        else:
            raise RuntimeError(f"Hook generation failed: {result.error}")

    def _create_topology(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create narrative topology."""
        plotwright = roles["plotwright"]

        context = RoleContext(
            task="create_topology",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = plotwright.execute_task(context)

        if result.success:
            # Create topology artifact
            artifact = Artifact(
                type="tu_brief",
                data={"topology": result.output},
                metadata={"created_by": "plotwright", "loop": "story_spark"},
            )
            self.context.artifacts.append(artifact)

            return {"success": True, "artifacts": [artifact]}
        else:
            raise RuntimeError(f"Topology creation failed: {result.error}")

    def _create_tu_brief(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create TU brief."""
        plotwright = roles["plotwright"]

        context = RoleContext(
            task="create_tu_brief",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = plotwright.execute_task(context)

        if result.success:
            # Create or update TU brief artifact
            artifact = Artifact(
                type="tu_brief",
                data={"content": result.output},
                metadata={"created_by": "plotwright", "loop": "story_spark"},
            )
            self.context.artifacts.append(artifact)

            return {"success": True, "artifacts": [artifact]}
        else:
            raise RuntimeError(f"TU brief creation failed: {result.error}")

    def _create_section_briefs(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create section briefs."""
        plotwright = roles["plotwright"]

        # Get section count from config or default
        section_count = self.context.config.get("section_count", 3)

        context = RoleContext(
            task="create_section_briefs",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"section_count": section_count},
        )

        result = plotwright.execute_task(context)

        if result.success:
            # Create section brief artifact
            artifact = Artifact(
                type="canon_pack",
                data={"section_briefs": result.output},
                metadata={"created_by": "plotwright", "loop": "story_spark"},
            )
            self.context.artifacts.append(artifact)

            return {"success": True, "artifacts": [artifact]}
        else:
            raise RuntimeError(f"Section brief creation failed: {result.error}")

    def _draft_scenes(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Draft scenes from section briefs."""
        scene_smith = roles["scene_smith"]

        context = RoleContext(
            task="draft_scene",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = scene_smith.execute_task(context)

        if result.success:
            # Create scene artifact
            artifact = Artifact(
                type="canon_pack",
                data={"scenes": result.output},
                metadata={"created_by": "scene_smith", "loop": "story_spark"},
            )
            self.context.artifacts.append(artifact)

            return {"success": True, "artifacts": [artifact]}
        else:
            raise RuntimeError(f"Scene drafting failed: {result.error}")

    def _pre_gate_check(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Run pre-gate quality check."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="pre_gate",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            status = result.metadata.get("status", "unknown")

            # Create gatecheck report
            artifact = Artifact(
                type="gatecheck_report",
                data={
                    "status": status,
                    "blockers": result.metadata.get("blockers", []),
                    "quick_wins": result.metadata.get("quick_wins", []),
                    "review_needed": result.metadata.get("review_needed", []),
                },
                metadata={"created_by": "gatekeeper", "loop": "story_spark"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "status": status,
                "passed": status == "pass",
            }
        else:
            raise RuntimeError(f"Pre-gate check failed: {result.error}")

    def _handle_refinement_iteration(
        self,
        failed_step: LoopStep,
        artifacts_created: list[Artifact],
        artifacts_modified: list[Artifact],
    ) -> LoopResult:
        """
        Handle refinement iteration when pre-gate fails.

        For now, just return partial success with iteration metadata.
        In a full implementation, this would loop back to earlier steps.

        Args:
            failed_step: The step that failed (pre_gate_check)
            artifacts_created: Artifacts created so far
            artifacts_modified: Artifacts modified so far

        Returns:
            Loop result indicating iteration needed
        """
        completed_steps = [s for s in self.steps if s.status == StepStatus.COMPLETED]
        return LoopResult(
            success=False,
            loop_id=self.metadata.loop_id,
            artifacts_created=artifacts_created,
            artifacts_modified=artifacts_modified,
            steps_completed=len(completed_steps),
            steps_failed=1,
            error="Pre-gate check failed, iteration needed",
            metadata={
                "iterations": self.iteration_count,
                "refinement_needed": True,
                "gate_status": (
                    failed_step.result.get("status")
                    if failed_step.result
                    else "unknown"
                ),
            },
        )

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

        # For pre_gate_check, validate based on status
        if step.step_id == "pre_gate_check":
            return bool(result.get("passed", False))

        # For other steps, check if artifacts were created
        if "artifacts" in result:
            return len(result["artifacts"]) > 0

        return True
