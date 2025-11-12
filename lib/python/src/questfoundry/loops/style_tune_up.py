"""Style Tune-Up loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class StyleTuneUpLoop(Loop):
    """
    Style Tune-Up: Refine scene text for style consistency.

    This loop detects and corrects style drift across prose, captions, and
    PN surfaces without re-architecting the story. Tighten voice, register,
    motifs, and visual guardrails so the book reads like one mind made it.

    Steps:
    1. Drift diagnosis (Style Lead)
    2. Create style addendum (Style Lead)
    3. Generate edit notes (Style Lead)
    4. Apply revisions (Scene Smith)
    5. Pre-gate validation (Gatekeeper)
    6. Package results (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="style_tune_up",
        display_name="Style Tune-Up",
        description="Detect and correct style drift across content",
        typical_duration="2-4 hours",
        primary_roles=["style_lead"],
        consulted_roles=["showrunner", "scene_smith", "gatekeeper"],
        entry_conditions=[
            "Readers/PN report tonal wobble",
            "New chapter introduces variant tone",
            "Art/Audio plans landed and nudge aesthetics",
        ],
        exit_conditions=[
            "Style drift diagnosed and documented",
            "Edit notes provided to owners",
            "Gatekeeper reports Style/Presentation green",
        ],
        output_artifacts=["style_addendum", "edit_notes"],
        inputs=["Cold style guide", "Recent prose drafts", "PN lines"],
        tags=["refinement", "style", "quality"],
    )

    steps = [
        LoopStep(
            step_id="diagnose_drift",
            description="Sample sections and tag style issues",
            assigned_roles=["style_lead"],
            consulted_roles=[],
            artifacts_input=["manuscript_section"],
            artifacts_output=["drift_diagnosis"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_addendum",
            description="Write style addendum with rules and exemplars",
            assigned_roles=["style_lead"],
            consulted_roles=[],
            artifacts_input=["drift_diagnosis"],
            artifacts_output=["style_addendum"],
            validation_required=True,
        ),
        LoopStep(
            step_id="generate_edit_notes",
            description="Create targeted fix suggestions for owners",
            assigned_roles=["style_lead"],
            consulted_roles=[],
            artifacts_input=["drift_diagnosis", "style_addendum"],
            artifacts_output=["edit_notes"],
            validation_required=True,
        ),
        LoopStep(
            step_id="apply_revisions",
            description="Scene Smith applies minimal changes to hit style",
            # Showrunner coordinates, scene_smith consulted
            assigned_roles=["showrunner"],
            consulted_roles=["scene_smith", "style_lead"],
            artifacts_input=["edit_notes"],
            artifacts_output=["manuscript_section"],
            validation_required=True,
        ),
        LoopStep(
            step_id="pre_gate_validation",
            description="Check Style and Presentation Safety",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["style_addendum", "manuscript_section"],
            artifacts_output=["gate_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_results",
            description="Bundle addendum and edit notes into TU",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["style_addendum", "edit_notes", "gate_report"],
            artifacts_output=["tu_brief"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Style Tune-Up loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.drift_issues: list[dict[str, Any]] = []
        self.addendum_created = False
        self.edit_notes_count = 0

    def execute(self) -> LoopResult:
        """
        Execute the Style Tune-Up loop.

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
                "drift_issues_found": len(self.drift_issues),
                "addendum_created": self.addendum_created,
                "edit_notes_count": self.edit_notes_count,
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
        if step.step_id == "diagnose_drift":
            return self._diagnose_drift(roles)
        elif step.step_id == "create_addendum":
            return self._create_addendum(roles)
        elif step.step_id == "generate_edit_notes":
            return self._generate_edit_notes(roles)
        elif step.step_id == "apply_revisions":
            return self._apply_revisions(roles)
        elif step.step_id == "pre_gate_validation":
            return self._pre_gate_validation(roles)
        elif step.step_id == "package_results":
            return self._package_results(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _diagnose_drift(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Sample sections and tag style issues."""
        style_lead = roles["style_lead"]

        # Find manuscript sections to analyze
        manuscript_sections = [
            a for a in self.context.artifacts if a.type == "manuscript_section"
        ]

        context = RoleContext(
            task="diagnose_drift",
            artifacts=manuscript_sections,
            project_metadata=self.context.project_metadata,
            additional_context={
                "sections": [
                    f"{a.artifact_id}: {a.data.get('title', 'Untitled')}"
                    if isinstance(a.data, dict)
                    else a.artifact_id
                    for a in manuscript_sections[:10]  # Sample first 10
                ],
            },
        )

        result = style_lead.execute_task(context)

        if result.success:
            # Extract drift issues from result
            issues = result.metadata.get("issues", [])
            self.drift_issues = issues

            # Create drift diagnosis artifact
            artifact = Artifact(
                type="drift_diagnosis",
                data={
                    "issues": issues,
                    "issue_count": len(issues),
                    "sections_analyzed": len(manuscript_sections[:10]),
                },
                metadata={"created_by": "style_lead", "loop": "style_tune_up"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "issues_found": len(issues),
            }
        else:
            raise RuntimeError(f"Drift diagnosis failed: {result.error}")

    def _create_addendum(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Write style addendum with rules and exemplars."""
        style_lead = roles["style_lead"]

        context = RoleContext(
            task="create_style_addendum",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"issues": self.drift_issues},
        )

        result = style_lead.execute_task(context)

        if result.success:
            self.addendum_created = True
            addendum_data = result.metadata.get("addendum", {})

            # Create style addendum artifact
            artifact = Artifact(
                type="style_addendum",
                data=addendum_data,
                metadata={"created_by": "style_lead", "loop": "style_tune_up"},
            )
            self.context.artifacts.append(artifact)

            return {"success": True, "artifacts": [artifact], "addendum": addendum_data}
        else:
            raise RuntimeError(f"Style addendum creation failed: {result.error}")

    def _generate_edit_notes(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create targeted fix suggestions for owners."""
        style_lead = roles["style_lead"]

        context = RoleContext(
            task="generate_edit_notes",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"issues": self.drift_issues},
        )

        result = style_lead.execute_task(context)

        if result.success:
            notes_data = result.metadata.get("notes", {})
            # Count all notes across owners
            total_notes = sum(
                len(owner_notes)
                for owner_notes in notes_data.values()
                if isinstance(owner_notes, list)
            )
            self.edit_notes_count = total_notes

            # Create edit notes artifact
            artifact = Artifact(
                type="edit_notes",
                data=notes_data,
                metadata={"created_by": "style_lead", "loop": "style_tune_up"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "notes_count": total_notes,
            }
        else:
            raise RuntimeError(f"Edit notes generation failed: {result.error}")

    def _apply_revisions(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Scene Smith applies minimal changes to hit style."""
        showrunner = roles["showrunner"]

        # Showrunner coordinates revision application
        # Scene Smith is consulted but may not be active
        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "apply_revisions",
                "from_role": "style_lead",
                "to_role": "scene_smith",
                "edit_notes_count": self.edit_notes_count,
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # In a real implementation, scene_smith would modify sections
            # For now, just track that revisions were coordinated
            return {"success": True, "artifacts": [], "applied": self.edit_notes_count}
        else:
            raise RuntimeError(f"Revision coordination failed: {result.error}")

    def _pre_gate_validation(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Check Style and Presentation Safety."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="pre_gate",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "bars": ["style", "presentation"],
                "mode": "pre-gate",
            },
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            gate_status = result.metadata.get("all_green", True)

            # Create gate report artifact
            artifact = Artifact(
                type="gate_report",
                data={
                    "status": "green" if gate_status else "yellow",
                    "bars_checked": ["style", "presentation"],
                    "mode": "pre-gate",
                },
                metadata={"created_by": "gatekeeper", "loop": "style_tune_up"},
            )
            self.context.artifacts.append(artifact)

            return {"success": True, "artifacts": [artifact], "all_green": gate_status}
        else:
            raise RuntimeError(f"Pre-gate validation failed: {result.error}")

    def _package_results(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Bundle addendum and edit notes into TU."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="review_progress",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "tu_id": "style-tune-up",
                "steps_completed": [
                    "Diagnose",
                    "Addendum",
                    "Edit Notes",
                    "Revisions",
                    "Pre-gate",
                ],
                "steps_remaining": [],
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create TU brief artifact
            tu_brief = {
                "tu_id": "style-tune-up",
                "loop": "style_tune_up",
                "deliverables": [
                    "Style Addendum with rules and exemplars",
                    f"Edit Notes ({self.edit_notes_count} notes)",
                    "Pre-gate validation complete",
                ],
                "drift_issues_resolved": len(self.drift_issues),
            }

            artifact = Artifact(
                type="tu_brief",
                data=tu_brief,
                metadata={"created_by": "showrunner", "loop": "style_tune_up"},
            )
            self.context.artifacts.append(artifact)

            return {"success": True, "artifacts": [artifact], "tu_brief": tu_brief}
        else:
            raise RuntimeError(f"Results packaging failed: {result.error}")

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
        # (except for apply_revisions which may not create artifacts)
        if step.step_id != "apply_revisions" and "artifacts" in result:
            return len(result["artifacts"]) > 0

        return True
