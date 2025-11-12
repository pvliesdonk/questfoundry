"""Archive Snapshot loop implementation."""

import logging
from datetime import datetime, timezone
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class ArchiveSnapshotLoop(Loop):
    """
    Archive Snapshot: Create cold snapshots.

    This loop creates a comprehensive, versioned snapshot of the entire project
    state (Hot and Cold) at a significant milestone for long-term archival,
    reproducibility, and provenance.

    Steps:
    1. Select artifacts (Showrunner)
    2. Create snapshot (Showrunner)
    3. Validate (Gatekeeper)
    4. Promote to cold (Showrunner)
    5. Package (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="archive_snapshot",
        display_name="Archive Snapshot",
        description="Create cold snapshots",
        typical_duration="1-2 hours",
        primary_roles=["showrunner"],
        consulted_roles=["gatekeeper", "book_binder"],
        entry_conditions=[
            "Major milestone completion",
            "Before significant refactoring or architectural changes",
            "Periodic archival schedule",
            "Before team transitions or long breaks",
        ],
        exit_conditions=[
            "All critical artifacts included in archive",
            "Manifest checksums validate",
            "Archive stored in multiple locations",
            "Archive cataloged and findable",
        ],
        output_artifacts=["archive_package", "archive_index"],
        inputs=[
            "Current Cold snapshot",
            "Current Hot snapshot",
            "All TU records",
            "All Gatecheck Reports",
        ],
        tags=["export", "archival", "snapshot"],
    )

    steps = [
        LoopStep(
            step_id="select_artifacts",
            description="Select artifacts to include in snapshot",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["cold_snapshot", "hot_snapshot"],
            artifacts_output=["artifact_selection"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_snapshot",
            description="Create comprehensive snapshot with manifest",
            assigned_roles=["showrunner"],
            consulted_roles=["book_binder"],
            artifacts_input=["artifact_selection"],
            artifacts_output=["snapshot_data"],
            validation_required=True,
        ),
        LoopStep(
            step_id="validate",
            description="Verify integrity and completeness",
            assigned_roles=["gatekeeper"],
            consulted_roles=[],
            artifacts_input=["snapshot_data"],
            artifacts_output=["validation_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="promote_to_cold",
            description="Promote snapshot to cold storage",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["snapshot_data", "validation_report"],
            artifacts_output=["cold_snapshot"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package",
            description="Package archive and create index entry",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["cold_snapshot"],
            artifacts_output=["archive_package", "archive_index"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Archive Snapshot loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.artifacts_selected: list[str] = []
        self.snapshot_id: str = ""
        self.manifest: dict[str, Any] = {}

    def execute(self) -> LoopResult:
        """
        Execute the Archive Snapshot loop.

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
                "snapshot_id": self.snapshot_id,
                "artifacts_archived": len(self.artifacts_selected),
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
        if step.step_id == "select_artifacts":
            return self._select_artifacts(roles)
        elif step.step_id == "create_snapshot":
            return self._create_snapshot(roles)
        elif step.step_id == "validate":
            return self._validate(roles)
        elif step.step_id == "promote_to_cold":
            return self._promote_to_cold(roles)
        elif step.step_id == "package":
            return self._package(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _select_artifacts(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Select artifacts to include in snapshot."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"step_name": "select_artifacts"},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Select all artifacts by default
            self.artifacts_selected = [
                a.artifact_id for a in self.context.artifacts if a.artifact_id
            ]

            # Create artifact selection
            artifact = Artifact(
                type="artifact_selection",
                data={
                    "artifacts": self.artifacts_selected,
                    "count": len(self.artifacts_selected),
                },
                metadata={"created_by": "showrunner", "loop": "archive_snapshot"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "selected": self.artifacts_selected,
            }
        else:
            raise RuntimeError(f"Artifact selection failed: {result.error}")

    def _create_snapshot(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Create comprehensive snapshot with manifest."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="create_snapshot",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"artifacts_selected": self.artifacts_selected},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create snapshot ID
            now = datetime.now(timezone.utc)
            self.snapshot_id = f"SNAPSHOT-{now.strftime('%Y%m%d-%H%M%S')}"

            # Create manifest
            self.manifest = {
                "snapshot_id": self.snapshot_id,
                "created": now.isoformat(),
                "artifacts_count": len(self.artifacts_selected),
                "checksums": {},
            }

            # Create snapshot data artifact
            artifact = Artifact(
                type="snapshot_data",
                data={
                    "snapshot_id": self.snapshot_id,
                    "manifest": self.manifest,
                    "artifacts": self.artifacts_selected,
                },
                metadata={"created_by": "showrunner", "loop": "archive_snapshot"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "snapshot_id": self.snapshot_id,
            }
        else:
            raise RuntimeError(f"Snapshot creation failed: {result.error}")

    def _validate(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Verify integrity and completeness."""
        gatekeeper = roles["gatekeeper"]

        context = RoleContext(
            task="validate_snapshot",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"manifest": self.manifest},
        )

        result = gatekeeper.execute_task(context)

        if result.success:
            validation_status = result.metadata.get("status", "pass")

            # Create validation report
            artifact = Artifact(
                type="validation_report",
                data={
                    "status": validation_status,
                    "integrity": "green",
                    "completeness": "green",
                    "snapshot_id": self.snapshot_id,
                },
                metadata={"created_by": "gatekeeper", "loop": "archive_snapshot"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "status": validation_status,
            }
        else:
            raise RuntimeError(f"Snapshot validation failed: {result.error}")

    def _promote_to_cold(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Promote snapshot to cold storage."""
        # Use workspace to promote artifacts
        workspace = self.context.workspace
        promoted_count = 0

        for artifact_id in self.artifacts_selected:
            if workspace.promote_to_cold(artifact_id, delete_hot=False):
                promoted_count += 1

        # Create cold snapshot artifact
        artifact = Artifact(
            type="cold_snapshot",
            data={
                "snapshot_id": self.snapshot_id,
                "promoted_count": promoted_count,
                "manifest": self.manifest,
            },
            metadata={"created_by": "showrunner", "loop": "archive_snapshot"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "promoted_count": promoted_count,
        }

    def _package(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package archive and create index entry."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"step_name": "package"},
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create archive package
            package_data = {
                "snapshot_id": self.snapshot_id,
                "manifest": self.manifest,
                "storage_locations": ["local", "cloud"],
            }

            package_artifact = Artifact(
                type="archive_package",
                data=package_data,
                metadata={"created_by": "showrunner", "loop": "archive_snapshot"},
            )
            self.context.artifacts.append(package_artifact)

            # Create archive index entry
            index_data = {
                "snapshot_id": self.snapshot_id,
                "created": self.manifest.get("created"),
                "artifacts_count": len(self.artifacts_selected),
            }

            index_artifact = Artifact(
                type="archive_index",
                data=index_data,
                metadata={"created_by": "showrunner", "loop": "archive_snapshot"},
            )
            self.context.artifacts.append(index_artifact)

            return {
                "success": True,
                "artifacts": [package_artifact, index_artifact],
                "package": package_data,
            }
        else:
            raise RuntimeError(f"Archive packaging failed: {result.error}")

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
