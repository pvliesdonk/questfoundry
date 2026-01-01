"""Artifact writing to YAML files."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime
from typing import Any

from pydantic import BaseModel
from ruamel.yaml import YAML


class ArtifactWriteError(Exception):
    """Raised when an artifact file can't be written."""

    def __init__(self, stage_name: str, path: Path, reason: str) -> None:
        self.stage_name = stage_name
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to write {stage_name} artifact at {path}: {reason}")


class ArtifactWriter:
    """Write artifacts to the project artifacts directory."""

    def __init__(self, project_path: Path) -> None:
        """Initialize writer with project path.

        Args:
            project_path: Path to the project root directory.
        """
        self.project_path = project_path
        self.artifacts_path = project_path / "artifacts"
        self._yaml = YAML()
        self._yaml.default_flow_style = False
        self._yaml.preserve_quotes = True
        self._yaml.indent(mapping=2, sequence=4, offset=2)

    def _ensure_artifacts_dir(self) -> None:
        """Ensure the artifacts directory exists."""
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

    def _get_artifact_path(self, stage_name: str) -> Path:
        """Get the path to an artifact file.

        Args:
            stage_name: Name of the stage (e.g., "dream", "seed").

        Returns:
            Path to the artifact YAML file.
        """
        return self.artifacts_path / f"{stage_name}.yaml"

    def write(self, artifact: BaseModel | dict[str, Any], stage_name: str) -> Path:
        """Write an artifact to disk.

        Args:
            artifact: Pydantic model or dictionary to write.
            stage_name: Name of the stage.

        Returns:
            Path to the written artifact file.

        Raises:
            ArtifactWriteError: If the artifact can't be written.
        """
        path = self._get_artifact_path(stage_name)

        try:
            self._ensure_artifacts_dir()

            # Convert Pydantic model to dict if needed
            if isinstance(artifact, BaseModel):
                data = artifact.model_dump(exclude_none=True)
            else:
                data = artifact

            with path.open("w", encoding="utf-8") as f:
                self._yaml.dump(data, f)

            return path
        except Exception as e:
            if isinstance(e, ArtifactWriteError):
                raise
            raise ArtifactWriteError(stage_name, path, str(e)) from e

    def write_raw(self, data: dict[str, Any], stage_name: str) -> Path:
        """Write raw dictionary data to disk.

        This is an alias for write() that makes intent clearer.

        Args:
            data: Dictionary to write.
            stage_name: Name of the stage.

        Returns:
            Path to the written artifact file.
        """
        return self.write(data, stage_name)
