"""Scene Forge loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class SceneForgeLoop(Loop):
    """
    Scene Forge: Create scene content from hooks and canon.

    This loop takes hooks and canonical lore and transforms them into
    concrete scene drafts. Scene Smith is the primary role, with Showrunner
    coordinating and Style Lead potentially consulted for voice consistency.

    Steps:
    1. Select scenes to draft (Scene Smith with Showrunner)
    2. Gather canon and topology context (Scene Smith)
    3. Draft scene prose (Scene Smith)
    4. Style pass (Scene Smith with Style Lead)
    5. Package scene drafts (Scene Smith)
    """

    metadata = LoopMetadata(
        loop_id="scene_forge",
        display_name="Scene Forge",
        description="Create scene content from hooks and canon",
        typical_duration="2-4 hours",
        primary_roles=["scene_smith"],
        consulted_roles=["showrunner", "style_lead", "lore_weaver"],
        entry_conditions=[
            "After Hook Harvest accepts scene hooks",
            "After Lore Deepening provides canon context",
            "When section briefs need prose",
        ],
        exit_conditions=[
            "Scene drafts complete",
            "Style consistent with guardrails",
            "Canon references accurate",
            "Ready for review/gatecheck",
        ],
        output_artifacts=["canon_pack"],
        inputs=["Hook cards", "Canon packs", "Section briefs", "TU briefs"],
        tags=["content", "prose", "drafting"],
    )

    steps = [
        LoopStep(
            step_id="select_scenes",
            description="Choose which scenes to draft this pass",
            assigned_roles=["scene_smith"],
            consulted_roles=["showrunner"],
            artifacts_input=["hook_card", "tu_brief"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="gather_context",
            description="Collect canon, topology, and style context",
            assigned_roles=["scene_smith"],
            consulted_roles=["lore_weaver"],
            artifacts_input=["canon_pack", "tu_brief", "hook_card"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="draft_scenes",
            description="Write scene prose with choices and descriptions",
            assigned_roles=["scene_smith"],
            consulted_roles=["style_lead"],
            artifacts_input=["canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="style_pass",
            description="Ensure voice, tone, and consistency",
            assigned_roles=["scene_smith"],
            consulted_roles=["style_lead"],
            artifacts_input=["canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_scenes",
            description="Package scene drafts for review",
            assigned_roles=["scene_smith"],
            consulted_roles=[],
            artifacts_input=["canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Scene Forge loop.

        Args:
            context: Loop execution context
        """
        logger.debug("Initializing SceneForgeLoop")
        super().__init__(context)
        self.selected_scenes: list[dict[str, Any]] = []
        self.scene_drafts: list[dict[str, Any]] = []
        self.scene_count = context.config.get("scene_count", 3)
        logger.trace("Scene count configured: %d", self.scene_count)

    def execute(self) -> LoopResult:
        """
        Execute the Scene Forge loop.

        Returns:
            Result of loop execution
        """
        logger.info("Starting Scene Forge loop execution")
        logger.debug("Total steps to execute: %d", len(self.steps))

        artifacts_created: list[Artifact] = []
        artifacts_modified: list[Artifact] = []
        steps_completed = 0
        steps_failed = 0

        # Execute each step in sequence
        for step in self.steps:
            try:
                logger.debug("Processing step: %s", step.step_id)
                self.execute_step(step)

                if step.status == StepStatus.COMPLETED:
                    steps_completed += 1
                    logger.info("Step completed: %s", step.step_id)

                    # Collect artifacts created in this step
                    if step.result and isinstance(step.result, dict):
                        if "artifacts" in step.result:
                            artifacts_created.extend(step.result["artifacts"])
                            logger.trace(
                                "Collected %d artifacts from step",
                                len(step.result["artifacts"]),
                            )

                elif step.status == StepStatus.FAILED:
                    steps_failed += 1
                    logger.error("Step failed: %s - %s", step.step_id, step.error)

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
                logger.error("Exception in step %s: %s", step.step_id, e, exc_info=True)
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
        logger.info(
            (
                "Scene Forge loop completed successfully: %d/%d steps, "
                "%d artifacts created"
            ),
            steps_completed,
            len(self.steps),
            len(artifacts_created),
        )
        return LoopResult(
            success=True,
            loop_id=self.metadata.loop_id,
            artifacts_created=artifacts_created,
            artifacts_modified=artifacts_modified,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            metadata={
                "scenes_drafted": len(self.scene_drafts),
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
        if step.step_id == "select_scenes":
            return self._select_scenes(roles)
        elif step.step_id == "gather_context":
            return self._gather_context(roles)
        elif step.step_id == "draft_scenes":
            return self._draft_scenes(roles)
        elif step.step_id == "style_pass":
            return self._style_pass(roles)
        elif step.step_id == "package_scenes":
            return self._package_scenes(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _select_scenes(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Choose which scenes to draft this pass."""
        scene_smith = roles["scene_smith"]

        context = RoleContext(
            task="draft_scene",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"scene_count": self.scene_count},
        )

        result = scene_smith.execute_task(context)

        if result.success:
            # Create list of scenes to draft
            # In real implementation, would analyze hooks and briefs
            self.selected_scenes = [
                {"scene_id": f"scene_{i}", "title": f"Scene {i}"}
                for i in range(1, self.scene_count + 1)
            ]

            artifact = Artifact(
                type="canon_pack",
                data={
                    "status": "scenes_selected",
                    "scenes": self.selected_scenes,
                },
                metadata={"created_by": "scene_smith", "loop": "scene_forge"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "scenes": self.selected_scenes,
            }
        else:
            raise RuntimeError(f"Scene selection failed: {result.error}")

    def _gather_context(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Collect canon, topology, and style context."""
        scene_smith = roles["scene_smith"]

        # Gather relevant canon packs
        canon_artifacts = [a for a in self.context.artifacts if a.type == "canon_pack"]

        context = RoleContext(
            task="draft_scene",
            artifacts=canon_artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "task_description": "Gather context for scene drafting"
            },
        )

        result = scene_smith.execute_task(context)

        if result.success:
            artifact = Artifact(
                type="canon_pack",
                data={
                    "status": "context_gathered",
                    "canon_summary": "Canon context compiled",
                },
                metadata={"created_by": "scene_smith", "loop": "scene_forge"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
            }
        else:
            raise RuntimeError(f"Context gathering failed: {result.error}")

    def _draft_scenes(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Write scene prose with choices and descriptions."""
        scene_smith = roles["scene_smith"]

        # Draft each selected scene
        for scene in self.selected_scenes:
            context = RoleContext(
                task="draft_scene",
                artifacts=self.context.artifacts,
                project_metadata=self.context.project_metadata,
                additional_context={
                    "scene_id": scene.get("scene_id", ""),
                    "scene_title": scene.get("title", ""),
                },
            )

            result = scene_smith.execute_task(context)

            if result.success:
                draft = {
                    "scene_id": scene.get("scene_id", ""),
                    "title": scene.get("title", ""),
                    "content": result.output,
                }
                self.scene_drafts.append(draft)

        # Create drafts artifact
        artifact = Artifact(
            type="canon_pack",
            data={
                "status": "scenes_drafted",
                "drafts": self.scene_drafts,
            },
            metadata={"created_by": "scene_smith", "loop": "scene_forge"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "drafts": self.scene_drafts,
        }

    def _style_pass(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Ensure voice, tone, and consistency."""
        scene_smith = roles["scene_smith"]

        # Review each draft for style
        context = RoleContext(
            task="draft_scene",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "task_description": "Style pass on scene drafts",
                "drafts": self.scene_drafts,
            },
        )

        result = scene_smith.execute_task(context)

        if result.success:
            artifact = Artifact(
                type="canon_pack",
                data={
                    "status": "style_passed",
                    "drafts": self.scene_drafts,
                },
                metadata={"created_by": "scene_smith", "loop": "scene_forge"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
            }
        else:
            raise RuntimeError(f"Style pass failed: {result.error}")

    def _package_scenes(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package scene drafts for review."""
        # Create final scene package
        scene_package = {
            "date": "2025-11-07",
            "tu_id": "scene-forge",
            "scenes": self.scene_drafts,
            "scene_count": len(self.scene_drafts),
            "ready_for_review": True,
        }

        artifact = Artifact(
            type="canon_pack",
            data=scene_package,
            metadata={"created_by": "scene_smith", "loop": "scene_forge"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "scene_package": scene_package,
        }

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
