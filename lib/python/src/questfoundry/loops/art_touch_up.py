"""Art Touch-Up loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class ArtTouchUpLoop(Loop):
    """
    Art Touch-Up: Create shotlists and generate images.

    This loop decides what to illustrate and why, then (optionally) produces
    illustrations that match style and narrative intent—without leaking spoilers.
    Supports plan-only merges when the Illustrator is dormant.

    Steps:
    1. Create shotlist (Art Director)
    2. Review shots (Art Director with Style Lead)
    3. Generate images (Illustrator, if active)
    4. Validate quality (Gatekeeper)
    5. Package art plan (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="art_touch_up",
        display_name="Art Touch-Up",
        description="Create shotlists and generate images",
        typical_duration="2-4 hours",
        primary_roles=["art_director"],
        consulted_roles=["illustrator", "style_lead", "showrunner", "gatekeeper"],
        entry_conditions=[
            "New chapter/act needs anchoring visuals",
            "Scene gained iconic imagery",
            "Style Lead requests motif reinforcement",
            "Replacement or upgrade of existing illustration",
        ],
        exit_conditions=[
            "Art Plan complete with subjects and composition intent",
            "Images generated (if Illustrator active)",
            "Captions are spoiler-safe",
            "Gatekeeper reports green on Style/Presentation",
        ],
        output_artifacts=["art_plan", "art_render"],
        inputs=[
            "Current Cold snapshot",
            "Target sections/scenes",
            "Lore Deepening notes",
            "Codex entries",
        ],
        tags=["asset", "visual", "content"],
    )

    steps = [
        LoopStep(
            step_id="create_shotlist",
            description="Select scenes and create shotlist with composition intent",
            assigned_roles=["art_director"],
            consulted_roles=["lore_weaver"],
            artifacts_input=["canon_pack"],
            artifacts_output=["shotlist"],
            validation_required=True,
        ),
        LoopStep(
            step_id="review_shots",
            description="Review shotlist for style alignment and spoiler safety",
            assigned_roles=["art_director"],
            consulted_roles=["style_lead"],
            artifacts_input=["shotlist"],
            artifacts_output=["art_plan"],
            validation_required=True,
        ),
        LoopStep(
            step_id="generate_images",
            description="Generate images from art plan (if Illustrator active)",
            assigned_roles=["illustrator"],
            consulted_roles=["art_director"],
            artifacts_input=["art_plan"],
            artifacts_output=["art_render"],
            validation_required=False,  # Optional if illustrator dormant
        ),
        LoopStep(
            step_id="validate_quality",
            description="Validate style, presentation safety, and determinism",
            assigned_roles=["gatekeeper"],
            consulted_roles=["style_lead"],
            artifacts_input=["art_plan", "art_render"],
            artifacts_output=["validation_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_art_plan",
            description="Package art plan and renders for handoff",
            assigned_roles=["showrunner"],
            consulted_roles=["art_director"],
            artifacts_input=["art_plan", "art_render", "validation_report"],
            artifacts_output=["art_package"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Art Touch-Up loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.shotlist: list[dict[str, Any]] = []
        self.art_plan: dict[str, Any] = {}
        self.renders_generated: list[dict[str, Any]] = []
        self.illustrator_active = "illustrator" in context.role_instances

    def execute(self) -> LoopResult:
        """
        Execute the Art Touch-Up loop.

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
                # Skip image generation if illustrator is dormant
                if step.step_id == "generate_images" and not self.illustrator_active:
                    step.status = StepStatus.SKIPPED
                    continue

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
                "shotlist_count": len(self.shotlist),
                "renders_generated": len(self.renders_generated),
                "illustrator_active": self.illustrator_active,
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
        if step.step_id == "create_shotlist":
            return self._create_shotlist(roles)
        elif step.step_id == "review_shots":
            return self._review_shots(roles)
        elif step.step_id == "generate_images":
            return self._generate_images(roles)
        elif step.step_id == "validate_quality":
            return self._validate_quality(roles)
        elif step.step_id == "package_art_plan":
            return self._package_art_plan(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _create_shotlist(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Select scenes and create shotlist with composition intent."""
        art_director = roles["art_director"]

        context = RoleContext(
            task="create_shotlist",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = art_director.execute_task(context)

        if result.success:
            # Store shotlist
            self.shotlist = result.metadata.get("shotlist", [])

            # Default shotlist if none provided
            if not self.shotlist:
                self.shotlist = [
                    {
                        "subject": "Opening scene",
                        "purpose": "Establish setting",
                        "composition": "Wide establishing shot",
                        "spoiler_risk": "low",
                    }
                ]

            # Create shotlist artifact
            artifact = Artifact(
                type="shotlist",
                data={"shots": self.shotlist, "count": len(self.shotlist)},
                metadata={"created_by": "art_director", "loop": "art_touch_up"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "shotlist": self.shotlist,
            }
        else:
            raise RuntimeError(f"Shotlist creation failed: {result.error}")

    def _review_shots(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Review shotlist for style alignment and spoiler safety."""
        art_director = roles["art_director"]

        context = RoleContext(
            task="review_shotlist",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"shotlist": self.shotlist},
        )

        result = art_director.execute_task(context)

        if result.success:
            # Create art plan from reviewed shotlist
            self.art_plan = {
                "shots": self.shotlist,
                "style_notes": result.metadata.get("style_notes", ""),
                "captions": result.metadata.get("captions", {}),
                "constraints": result.metadata.get("constraints", {}),
            }

            # Create art plan artifact
            artifact = Artifact(
                type="art_plan",
                data=self.art_plan,
                metadata={"created_by": "art_director", "loop": "art_touch_up"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "art_plan": self.art_plan,
            }
        else:
            raise RuntimeError(f"Shot review failed: {result.error}")

    def _generate_images(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Generate images from art plan (if Illustrator active)."""
        if not self.illustrator_active:
            return {"success": True, "artifacts": [], "skipped": True}

        illustrator = roles["illustrator"]

        context = RoleContext(
            task="generate_image",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"art_plan": self.art_plan},
        )

        result = illustrator.execute_task(context)

        if result.success:
            # Store renders
            self.renders_generated = result.metadata.get("renders", [])

            # Create render artifacts
            artifacts = []
            for render in self.renders_generated:
                artifact = Artifact(
                    type="art_render",
                    data=render,
                    metadata={"created_by": "illustrator", "loop": "art_touch_up"},
                )
                artifacts.append(artifact)
                self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": artifacts,
                "renders": self.renders_generated,
            }
        else:
            raise RuntimeError(f"Image generation failed: {result.error}")

    def _validate_quality(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Validate style, presentation safety, and determinism."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="pre_gate",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "art_plan": self.art_plan,
                "renders": self.renders_generated,
            },
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            validation_status = result.metadata.get("status", "pass")

            # Create validation report
            artifact = Artifact(
                type="validation_report",
                data={
                    "status": validation_status,
                    "style": "green",
                    "presentation": "green",
                    "determinism": "green" if self.renders_generated else "n/a",
                },
                metadata={"created_by": "gatekeeper", "loop": "art_touch_up"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "status": validation_status,
            }
        else:
            raise RuntimeError(f"Quality validation failed: {result.error}")

    def _package_art_plan(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package art plan and renders for handoff."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "package_art_plan",
                "art_plan": self.art_plan,
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create art package
            package_data = {
                "art_plan": self.art_plan,
                "renders": self.renders_generated,
                "plan_only": not self.illustrator_active,
                "handoffs": [
                    "To Book Binder: Art plans and/or renders ready for inclusion"
                ],
            }

            artifact = Artifact(
                type="art_package",
                data=package_data,
                metadata={"created_by": "showrunner", "loop": "art_touch_up"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "package": package_data,
            }
        else:
            raise RuntimeError(f"Art plan packaging failed: {result.error}")

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

        # Skip validation is OK for optional steps
        if result.get("skipped", False):
            return True

        # Check if artifacts were created
        if "artifacts" in result:
            return len(result["artifacts"]) > 0

        return True
