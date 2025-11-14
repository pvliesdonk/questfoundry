"""Generic executor for compiled playbook manifests."""

import logging
from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext, RoleResult
from ..state.workspace import WorkspaceManager
from .manifest_loader import ManifestLoader

logger = logging.getLogger(__name__)


class PlaybookExecutor:
    """Generic executor for compiled playbook manifests.

    This replaces hardcoded loop classes with a generic execution engine
    that interprets compiled manifests from the spec compiler.
    """

    def __init__(
        self,
        manifest_path: Path | None = None,
        playbook_id: str | None = None,
        manifest_dir: Path | None = None,
    ):
        """Initialize playbook executor.

        Args:
            manifest_path: Direct path to manifest file (takes precedence)
            playbook_id: ID of playbook to load from manifest_dir
            manifest_dir: Directory containing manifests (default: dist/compiled/manifests)

        Raises:
            ValueError: If neither manifest_path nor playbook_id provided
        """
        if manifest_path:
            self.manifest_path = Path(manifest_path)
            with open(self.manifest_path) as f:
                import json

                self.manifest = json.load(f)
        elif playbook_id and manifest_dir:
            loader = ManifestLoader(Path(manifest_dir))
            self.manifest = loader.load_manifest(playbook_id)
            self.manifest_path = Path(manifest_dir) / f"{playbook_id}.manifest.json"
        elif playbook_id:
            # Default manifest directory
            default_dir = Path.cwd() / "dist" / "compiled" / "manifests"
            loader = ManifestLoader(default_dir)
            self.manifest = loader.load_manifest(playbook_id)
            self.manifest_path = default_dir / f"{playbook_id}.manifest.json"
        else:
            msg = "Either manifest_path or playbook_id must be provided"
            raise ValueError(msg)

        self.playbook_id = self.manifest["playbook_id"]
        self.display_name = self.manifest["display_name"]
        self.steps = self.manifest["steps"]
        self.current_step_index = 0
        self.context: dict[str, Any] = {}
        self.step_results: dict[str, Any] = {}

    def execute_step(
        self,
        step_id: str,
        roles: dict[str, Role],
        artifacts: list[Artifact] | None = None,
        workspace: WorkspaceManager | None = None,
        project_metadata: dict[str, Any] | None = None,
    ) -> RoleResult:
        """Execute a single step using assigned roles.

        Args:
            step_id: ID of the step to execute
            roles: Dictionary of role instances keyed by role name
            artifacts: Available artifacts for this step
            workspace: Workspace manager for file operations
            project_metadata: Project-level metadata

        Returns:
            RoleResult from executing the step

        Raises:
            ValueError: If step not found or required role not available
        """
        step = self._get_step(step_id)
        if not step:
            msg = f"Step '{step_id}' not found in playbook '{self.playbook_id}'"
            raise ValueError(msg)

        # Get assigned role instances
        assigned_role_names = step["assigned_roles"]
        if not assigned_role_names:
            msg = f"Step '{step_id}' has no assigned roles"
            raise ValueError(msg)

        # Use first assigned role (RACI: Responsible)
        primary_role_name = assigned_role_names[0]
        if primary_role_name not in roles:
            msg = f"Required role '{primary_role_name}' not available for step '{step_id}'"
            raise ValueError(msg)

        primary_role = roles[primary_role_name]

        # Build role context
        procedure_content = step["procedure_content"]
        role_context = RoleContext(
            task=step["description"],
            artifacts=artifacts or [],
            project_metadata=project_metadata or {},
            workspace_path=workspace.workspace_path if workspace else None,
            additional_context={
                "step_id": step_id,
                "procedure": procedure_content,
                "playbook_id": self.playbook_id,
                "consulted_roles": step.get("consulted_roles", []),
                "step_results": self.step_results,
            },
        )

        logger.info(
            "Executing step '%s' in playbook '%s' with role '%s'",
            step_id,
            self.playbook_id,
            primary_role_name,
        )

        # Execute via role
        result = primary_role.execute(role_context)

        # Store result for subsequent steps
        self.step_results[step_id] = result

        # Validate output artifacts if required
        if step.get("validation_required", True):
            self._validate_artifacts(result, step.get("artifacts_output", []))

        return result

    def execute_full_loop(
        self,
        roles: dict[str, Role],
        artifacts: list[Artifact] | None = None,
        workspace: WorkspaceManager | None = None,
        project_metadata: dict[str, Any] | None = None,
    ) -> dict[str, RoleResult]:
        """Execute entire playbook from start to finish.

        Args:
            roles: Dictionary of role instances
            artifacts: Initial artifacts
            workspace: Workspace manager
            project_metadata: Project metadata

        Returns:
            Dictionary mapping step IDs to their results
        """
        results: dict[str, RoleResult] = {}

        for step in self.steps:
            step_id = step["step_id"]

            try:
                result = self.execute_step(
                    step_id,
                    roles,
                    artifacts,
                    workspace,
                    project_metadata,
                )
                results[step_id] = result

                # Update artifacts with outputs from this step
                if result.artifacts:
                    if artifacts is None:
                        artifacts = []
                    artifacts.extend(result.artifacts)

            except Exception as e:
                logger.exception(
                    "Failed to execute step '%s' in playbook '%s'",
                    step_id,
                    self.playbook_id,
                )
                # Store failed result
                results[step_id] = RoleResult(
                    success=False,
                    output="",
                    error=str(e),
                )
                # Stop execution on failure
                break

        return results

    def _get_step(self, step_id: str) -> dict[str, Any] | None:
        """Get step by ID.

        Args:
            step_id: Step identifier

        Returns:
            Step dictionary or None if not found
        """
        for step in self.steps:
            if step["step_id"] == step_id:
                return step
        return None

    def _validate_artifacts(
        self, result: RoleResult, expected_types: list[str]
    ) -> None:
        """Validate output artifacts against expected types.

        Args:
            result: RoleResult containing artifacts
            expected_types: List of expected artifact types

        Note:
            Currently logs warnings for missing artifacts but doesn't fail.
            Full validation against L3 schemas should use existing validators.
        """
        if not expected_types:
            return

        produced_types = {artifact.type for artifact in result.artifacts}

        for expected_type in expected_types:
            if expected_type not in produced_types:
                logger.warning(
                    "Expected artifact type '%s' not produced by step",
                    expected_type,
                )

    def get_raci(self) -> dict[str, list[str]]:
        """Get RACI matrix for this playbook.

        Returns:
            RACI dictionary with responsible, accountable, consulted, informed
        """
        return self.manifest.get("raci", {})

    def get_quality_bars(self) -> list[str]:
        """Get quality bars pressed by this playbook.

        Returns:
            List of quality bar names
        """
        return self.manifest.get("quality_bars", [])

    def get_source_files(self) -> list[str]:
        """Get source files used to compile this manifest.

        Returns:
            List of source file paths
        """
        return self.manifest.get("source_files", [])
