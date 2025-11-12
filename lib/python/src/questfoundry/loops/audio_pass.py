"""Audio Pass loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class AudioPassLoop(Loop):
    """
    Audio Pass: Create cuelists and generate audio.

    This loop decides what the audience should hear and why—ambience, foley,
    stingers, or voice—then (optionally) produces audio that fits style and
    narrative intent without leaking spoilers. Supports plan-only merges when
    the Audio Producer is dormant.

    Steps:
    1. Create cuelist (Audio Director)
    2. Review cues (Audio Director with Style Lead)
    3. Generate audio (Audio Producer, if active)
    4. Validate quality (Gatekeeper)
    5. Package audio plan (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="audio_pass",
        display_name="Audio Pass",
        description="Create cuelists and generate audio",
        typical_duration="2-4 hours",
        primary_roles=["audio_director"],
        consulted_roles=["audio_producer", "style_lead", "showrunner", "gatekeeper"],
        entry_conditions=[
            "Chapter/scene needs mood scaffolding or sound cues",
            "Style Lead requests motif reinforcement via sound",
            "Replacement/upgrade of existing sounds or VO",
            "Export targets include audio plan/assets",
        ],
        exit_conditions=[
            "Audio Plan complete with cues and placement",
            "Audio assets generated (if Audio Producer active)",
            "Descriptions are spoiler-safe",
            "Gatekeeper reports green on Style/Presentation",
        ],
        output_artifacts=["audio_plan", "audio_asset"],
        inputs=[
            "Cold snapshot",
            "Target scenes/sections",
            "PN Principles",
            "Accessibility requirements",
        ],
        tags=["asset", "audio", "content"],
    )

    steps = [
        LoopStep(
            step_id="create_cuelist",
            description="Select cues and create cuelist with placement",
            assigned_roles=["audio_director"],
            consulted_roles=["style_lead"],
            artifacts_input=["canon_pack"],
            artifacts_output=["cuelist"],
            validation_required=True,
        ),
        LoopStep(
            step_id="review_cues",
            description="Review cuelist for style alignment and spoiler safety",
            assigned_roles=["audio_director"],
            consulted_roles=["style_lead"],
            artifacts_input=["cuelist"],
            artifacts_output=["audio_plan"],
            validation_required=True,
        ),
        LoopStep(
            step_id="generate_audio",
            description="Generate audio assets from plan (if Audio Producer active)",
            assigned_roles=["audio_producer"],
            consulted_roles=["audio_director"],
            artifacts_input=["audio_plan"],
            artifacts_output=["audio_asset"],
            validation_required=False,  # Optional if producer dormant
        ),
        LoopStep(
            step_id="validate_quality",
            description="Validate style, presentation safety, and reproducibility",
            assigned_roles=["gatekeeper"],
            consulted_roles=["style_lead"],
            artifacts_input=["audio_plan", "audio_asset"],
            artifacts_output=["validation_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_audio_plan",
            description="Package audio plan and assets for handoff",
            assigned_roles=["showrunner"],
            consulted_roles=["audio_director"],
            artifacts_input=["audio_plan", "audio_asset", "validation_report"],
            artifacts_output=["audio_package"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Audio Pass loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.cuelist: list[dict[str, Any]] = []
        self.audio_plan: dict[str, Any] = {}
        self.assets_generated: list[dict[str, Any]] = []
        self.audio_producer_active = "audio_producer" in context.role_instances

    def execute(self) -> LoopResult:
        """
        Execute the Audio Pass loop.

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
                # Skip audio generation if producer is dormant
                if step.step_id == "generate_audio" and not self.audio_producer_active:
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
                "cuelist_count": len(self.cuelist),
                "assets_generated": len(self.assets_generated),
                "audio_producer_active": self.audio_producer_active,
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
        if step.step_id == "create_cuelist":
            return self._create_cuelist(roles)
        elif step.step_id == "review_cues":
            return self._review_cues(roles)
        elif step.step_id == "generate_audio":
            return self._generate_audio(roles)
        elif step.step_id == "validate_quality":
            return self._validate_quality(roles)
        elif step.step_id == "package_audio_plan":
            return self._package_audio_plan(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _create_cuelist(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Select cues and create cuelist with placement."""
        audio_director = roles["audio_director"]

        context = RoleContext(
            task="create_cuelist",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = audio_director.execute_task(context)

        if result.success:
            # Store cuelist
            self.cuelist = result.metadata.get("cuelist", [])

            # Default cuelist if none provided
            if not self.cuelist:
                self.cuelist = [
                    {
                        "cue_id": "AMB-001",
                        "type": "ambience",
                        "scene_anchor": "opening",
                        "purpose": "Establish mood",
                        "spoiler_risk": "low",
                    }
                ]

            # Create cuelist artifact
            artifact = Artifact(
                type="cuelist",
                data={"cues": self.cuelist, "count": len(self.cuelist)},
                metadata={"created_by": "audio_director", "loop": "audio_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "cuelist": self.cuelist,
            }
        else:
            raise RuntimeError(f"Cuelist creation failed: {result.error}")

    def _review_cues(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Review cuelist for style alignment and spoiler safety."""
        audio_director = roles["audio_director"]

        context = RoleContext(
            task="review_cuelist",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"cuelist": self.cuelist},
        )

        result = audio_director.execute_task(context)

        if result.success:
            # Create audio plan from reviewed cuelist
            self.audio_plan = {
                "cues": self.cuelist,
                "style_notes": result.metadata.get("style_notes", ""),
                "captions": result.metadata.get("captions", {}),
                "safety_notes": result.metadata.get("safety_notes", {}),
            }

            # Create audio plan artifact
            artifact = Artifact(
                type="audio_plan",
                data=self.audio_plan,
                metadata={"created_by": "audio_director", "loop": "audio_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "audio_plan": self.audio_plan,
            }
        else:
            raise RuntimeError(f"Cue review failed: {result.error}")

    def _generate_audio(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Generate audio assets from plan (if Audio Producer active)."""
        if not self.audio_producer_active:
            return {"success": True, "artifacts": [], "skipped": True}

        audio_producer = roles["audio_producer"]

        context = RoleContext(
            task="generate_audio",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"audio_plan": self.audio_plan},
        )

        result = audio_producer.execute_task(context)

        if result.success:
            # Store assets
            self.assets_generated = result.metadata.get("assets", [])

            # Create asset artifacts
            artifacts = []
            for asset in self.assets_generated:
                artifact = Artifact(
                    type="audio_asset",
                    data=asset,
                    metadata={"created_by": "audio_producer", "loop": "audio_pass"},
                )
                artifacts.append(artifact)
                self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": artifacts,
                "assets": self.assets_generated,
            }
        else:
            raise RuntimeError(f"Audio generation failed: {result.error}")

    def _validate_quality(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Validate style, presentation safety, and reproducibility."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="pre_gate",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "audio_plan": self.audio_plan,
                "assets": self.assets_generated,
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
                    "reproducibility": "green" if self.assets_generated else "n/a",
                },
                metadata={"created_by": "gatekeeper", "loop": "audio_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "status": validation_status,
            }
        else:
            raise RuntimeError(f"Quality validation failed: {result.error}")

    def _package_audio_plan(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package audio plan and assets for handoff."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "package_audio_plan",
                "audio_plan": self.audio_plan,
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create audio package
            package_data = {
                "audio_plan": self.audio_plan,
                "assets": self.assets_generated,
                "plan_only": not self.audio_producer_active,
                "handoffs": [
                    "To Book Binder: Audio plans and/or assets ready for inclusion"
                ],
            }

            artifact = Artifact(
                type="audio_package",
                data=package_data,
                metadata={"created_by": "showrunner", "loop": "audio_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "package": package_data,
            }
        else:
            raise RuntimeError(f"Audio plan packaging failed: {result.error}")

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
