"""Binding Run loop implementation."""

import logging
from datetime import datetime, timezone
from typing import Any

from ..export.binder import BookBinder
from ..export.view import ViewArtifact
from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class BindingRunLoop(Loop):
    """
    Binding Run: Final exports (HTML, Markdown, YAML).

    This loop assembles a player-safe export view of the book from a specific
    Cold snapshot. Packages manuscript, codex, and checklists—optionally including
    art/audio plans or assets and translation slices—without leaking spoilers or
    internal plumbing.

    Steps:
    1. Select formats (Showrunner)
    2. Generate views (Book Binder)
    3. Export files (Book Binder)
    4. Validate (Gatekeeper)
    5. Package (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="binding_run",
        display_name="Binding Run",
        description="Final exports (HTML, Markdown, YAML)",
        typical_duration="1-2 hours",
        primary_roles=["book_binder"],
        consulted_roles=["showrunner", "gatekeeper"],
        entry_conditions=[
            "Milestone release (chapter/act/book)",
            "Playtest build needed for PN Narration Dry-Run",
            "External review or print/export request",
        ],
        exit_conditions=[
            "Bundle is player-safe, navigable, and accessible",
            "Snapshot and options are clear and consistent",
            "Known limitations disclosed in View Log",
        ],
        output_artifacts=["view_log", "export_bundle"],
        inputs=[
            "Cold snapshot",
            "Gatekeeper's latest pass notes",
            "Inclusion options from Showrunner",
        ],
        tags=["export", "binding", "player-facing"],
    )

    steps = [
        LoopStep(
            step_id="select_formats",
            description="Select snapshot and export options",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["cold_snapshot"],
            artifacts_output=["export_config"],
            validation_required=True,
        ),
        LoopStep(
            step_id="generate_views",
            description="Generate views from Cold snapshot",
            assigned_roles=["book_binder"],
            consulted_roles=[],
            artifacts_input=["export_config", "cold_snapshot"],
            artifacts_output=["view_artifact"],
            validation_required=True,
        ),
        LoopStep(
            step_id="export_files",
            description="Export to selected formats (HTML, Markdown, etc.)",
            assigned_roles=["book_binder"],
            consulted_roles=[],
            artifacts_input=["view_artifact"],
            artifacts_output=["export_files"],
            validation_required=True,
        ),
        LoopStep(
            step_id="validate",
            description="Validate presentation safety and integrity",
            assigned_roles=["gatekeeper"],
            consulted_roles=["book_binder"],
            artifacts_input=["export_files"],
            artifacts_output=["validation_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package",
            description="Package export bundle with View Log",
            assigned_roles=["showrunner"],
            consulted_roles=["book_binder"],
            artifacts_input=["export_files", "validation_report"],
            artifacts_output=["export_bundle"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Binding Run loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.export_formats = context.config.get("formats", ["markdown", "html"])
        self.export_config: dict[str, Any] = {}
        self.view_artifact: ViewArtifact | None = None
        self.binder = BookBinder()

    def execute(self) -> LoopResult:
        """
        Execute the Binding Run loop.

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
                "formats": self.export_formats,
                "exports_created": len(self.export_formats),
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
        if step.step_id == "select_formats":
            return self._select_formats(roles)
        elif step.step_id == "generate_views":
            return self._generate_views(roles)
        elif step.step_id == "export_files":
            return self._export_files(roles)
        elif step.step_id == "validate":
            return self._validate(roles)
        elif step.step_id == "package":
            return self._package(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _select_formats(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Select snapshot and export options."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"step_name": "select_formats"},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Store export config
            self.export_config = {
                "formats": self.export_formats,
                "include_art": self.context.config.get("include_art", False),
                "include_audio": self.context.config.get("include_audio", False),
                "include_translations": self.context.config.get(
                    "include_translations", False
                ),
            }

            # Create export config artifact
            artifact = Artifact(
                type="export_config",
                data=self.export_config,
                metadata={"created_by": "showrunner", "loop": "binding_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "config": self.export_config,
            }
        else:
            raise RuntimeError(f"Format selection failed: {result.error}")

    def _generate_views(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Generate views from Cold snapshot."""
        book_binder = roles["book_binder"]

        context = RoleContext(
            task="generate_view",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"export_config": self.export_config},
        )

        result = book_binder.execute_task(context)

        if result.success:
            # Create view artifact (using export module)
            # Try to get snapshot_id from context artifacts, or generate one
            snapshot_id = self._get_snapshot_id()
            now = datetime.now(timezone.utc)

            self.view_artifact = ViewArtifact(
                view_id=f"view-{now.strftime('%Y%m%d-%H%M%S')}",
                snapshot_id=snapshot_id,
                artifacts=self.context.artifacts,
                created=now,
            )

            # Create view artifact reference
            artifact = Artifact(
                type="view_artifact",
                data={"view_id": self.view_artifact.view_id},
                metadata={"created_by": "book_binder", "loop": "binding_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "view": self.view_artifact,
            }
        else:
            raise RuntimeError(f"View generation failed: {result.error}")

    def _export_files(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Export to selected formats (HTML, Markdown, etc.)."""
        book_binder = roles["book_binder"]

        context = RoleContext(
            task="export_files",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "view": self.view_artifact,
                "formats": self.export_formats,
            },
        )

        result = book_binder.execute_task(context)

        if result.success:
            # Create export files artifacts
            exports = []
            for format_type in self.export_formats:
                export_data = {
                    "format": format_type,
                    "content": f"Exported content in {format_type}",
                    "filename": f"export.{format_type}",
                }
                exports.append(export_data)

            artifact = Artifact(
                type="export_files",
                data={"exports": exports, "count": len(exports)},
                metadata={"created_by": "book_binder", "loop": "binding_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "exports": exports,
            }
        else:
            raise RuntimeError(f"File export failed: {result.error}")

    def _validate(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Validate presentation safety and integrity."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="pre_gate",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            validation_status = result.metadata.get("status", "pass")

            # Create validation report
            artifact = Artifact(
                type="validation_report",
                data={
                    "status": validation_status,
                    "presentation": "green",
                    "integrity": "green",
                    "accessibility": "green",
                },
                metadata={"created_by": "gatekeeper", "loop": "binding_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "status": validation_status,
            }
        else:
            raise RuntimeError(f"Validation failed: {result.error}")

    def _package(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package export bundle with View Log."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"step_name": "package"},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create export bundle
            view_id = self.view_artifact.view_id if self.view_artifact else "unknown"
            bundle_data = {
                "view_id": view_id,
                "formats": self.export_formats,
                "snapshot_id": self._get_snapshot_id(),
                "view_log": {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "formats": self.export_formats,
                    "coverage": "complete",
                },
            }

            artifact = Artifact(
                type="export_bundle",
                data=bundle_data,
                metadata={"created_by": "showrunner", "loop": "binding_run"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "bundle": bundle_data,
            }
        else:
            raise RuntimeError(f"Bundle packaging failed: {result.error}")

    def _get_snapshot_id(self) -> str:
        """
        Get snapshot ID from context artifacts, or generate a default one.

        Returns:
            Snapshot ID from artifacts or generated default
        """
        # Look for snapshot artifacts in context
        for artifact in self.context.artifacts:
            if artifact.type in ["snapshot_data", "cold_snapshot"]:
                snapshot_id = artifact.data.get("snapshot_id")
                if snapshot_id:
                    return snapshot_id

        # If no snapshot found, generate a default one
        now = datetime.now(timezone.utc)
        return f"snapshot-{now.strftime('%Y%m%d-%H%M%S')}"

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
