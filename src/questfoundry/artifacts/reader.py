"""Artifact reading from YAML files."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TypeVar

from pydantic import BaseModel
from ruamel.yaml import YAML

T = TypeVar("T", bound=BaseModel)


class ArtifactNotFoundError(Exception):
    """Raised when an artifact file doesn't exist."""

    def __init__(self, stage_name: str, path: Path) -> None:
        self.stage_name = stage_name
        self.path = path
        super().__init__(f"Artifact not found: {stage_name} at {path}")


class ArtifactParseError(Exception):
    """Raised when an artifact file can't be parsed."""

    def __init__(self, stage_name: str, path: Path, reason: str) -> None:
        self.stage_name = stage_name
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse {stage_name} artifact at {path}: {reason}")


class ArtifactReader:
    """Read artifacts from the project artifacts directory."""

    def __init__(self, project_path: Path) -> None:
        """Initialize reader with project path.

        Args:
            project_path: Path to the project root directory.
        """
        self.project_path = project_path
        self.artifacts_path = project_path / "artifacts"
        self._yaml = YAML()
        self._yaml.preserve_quotes = True

    def _get_artifact_path(self, stage_name: str) -> Path:
        """Get the path to an artifact file.

        Args:
            stage_name: Name of the stage (e.g., "dream", "seed").

        Returns:
            Path to the artifact YAML file.
        """
        return self.artifacts_path / f"{stage_name}.yaml"

    def exists(self, stage_name: str) -> bool:
        """Check if an artifact exists.

        Args:
            stage_name: Name of the stage.

        Returns:
            True if the artifact file exists.
        """
        return self._get_artifact_path(stage_name).exists()

    def read(self, stage_name: str) -> dict[str, object]:
        """Read an artifact as a raw dictionary.

        Args:
            stage_name: Name of the stage.

        Returns:
            Dictionary containing the artifact data.

        Raises:
            ArtifactNotFoundError: If the artifact doesn't exist.
            ArtifactParseError: If the artifact can't be parsed.
        """
        path = self._get_artifact_path(stage_name)

        if not path.exists():
            raise ArtifactNotFoundError(stage_name, path)

        try:
            with path.open("r", encoding="utf-8") as f:
                data = self._yaml.load(f)
                if data is None:
                    raise ArtifactParseError(stage_name, path, "Empty file")
                return dict(data)
        except Exception as e:
            if isinstance(e, (ArtifactNotFoundError, ArtifactParseError)):
                raise
            raise ArtifactParseError(stage_name, path, str(e)) from e

    def read_validated(self, stage_name: str, model: type[T]) -> T:
        """Read and validate an artifact against a Pydantic model.

        Args:
            stage_name: Name of the stage.
            model: Pydantic model class to validate against.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ArtifactNotFoundError: If the artifact doesn't exist.
            ArtifactParseError: If the artifact can't be parsed.
            pydantic.ValidationError: If validation fails.
        """
        data = self.read(stage_name)
        return model.model_validate(data)
