"""Project configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import Any

from ruamel.yaml import YAML


@dataclass
class ProviderConfig:
    """Configuration for LLM provider."""

    name: str = "ollama"
    model: str = "qwen3:8b"


@dataclass
class GateConfig:
    """Configuration for stage gates."""

    stage: str
    required: bool = False


@dataclass
class ProjectConfig:
    """Configuration for a QuestFoundry project."""

    name: str
    version: int = 1
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    stages: list[str] = field(
        default_factory=lambda: ["dream", "brainstorm", "seed", "grow", "fill", "ship"]
    )
    gates: list[GateConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary containing config fields.

        Returns:
            ProjectConfig instance.
        """
        # Parse provider
        provider_data = data.get("providers", {})
        default_provider = provider_data.get("default", "ollama/qwen3:8b")
        if "/" in default_provider:
            provider_name, model = default_provider.split("/", 1)
        else:
            provider_name, model = default_provider, "qwen3:8b"

        provider = ProviderConfig(name=provider_name, model=model)

        # Parse stages
        pipeline_data = data.get("pipeline", {})
        stages = pipeline_data.get(
            "stages", ["dream", "brainstorm", "seed", "grow", "fill", "ship"]
        )

        # Parse gates
        gates_data = pipeline_data.get("gates", {})
        gates = [
            GateConfig(stage=stage, required=(value == "required"))
            for stage, value in gates_data.items()
        ]

        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", 1),
            provider=provider,
            stages=stages,
            gates=gates,
        )


class ProjectConfigError(Exception):
    """Raised when project configuration cannot be loaded."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load project config at {path}: {reason}")


def load_project_config(project_path: Path) -> ProjectConfig:
    """Load project configuration from project.yaml.

    Args:
        project_path: Path to the project root directory.

    Returns:
        ProjectConfig instance.

    Raises:
        ProjectConfigError: If config cannot be loaded.
    """
    config_path = project_path / "project.yaml"

    if not config_path.exists():
        raise ProjectConfigError(config_path, "File not found")

    yaml = YAML()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.load(f)

        if data is None:
            raise ProjectConfigError(config_path, "Empty file")

        return ProjectConfig.from_dict(dict(data))
    except Exception as e:
        if isinstance(e, ProjectConfigError):
            raise
        raise ProjectConfigError(config_path, str(e)) from e


def create_default_config(name: str) -> ProjectConfig:
    """Create a default project configuration.

    Args:
        name: Project name.

    Returns:
        ProjectConfig with default values.
    """
    return ProjectConfig(name=name)
