"""Project configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import Any

from ruamel.yaml import YAML

# Default configuration values
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_STAGES = ["dream", "brainstorm", "seed", "grow", "fill", "ship"]


@dataclass
class ProviderConfig:
    """Configuration for LLM provider."""

    name: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


@dataclass
class GateConfig:
    """Configuration for stage gates."""

    stage: str
    required: bool = False


@dataclass
class ResearchToolsConfig:
    """Configuration for research tools.

    Controls which research tools are available to pipeline stages.
    Tools can be enabled/disabled independently.

    Note:
        Setting a tool to True means "enabled if available". The tool will
        only be loaded if its dependencies are installed. If the dependency
        is missing, a warning is logged but execution continues without
        the tool. This allows graceful degradation when optional packages
        are not installed.

    Attributes:
        corpus: Enable IF Craft Corpus tools if ifcraftcorpus is installed.
        web_search: Enable web search tool if pvl-webtools is installed and SEARXNG_URL set.
        web_fetch: Enable web fetch tool if pvl-webtools is installed.
    """

    corpus: bool = True
    web_search: bool = True
    web_fetch: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchToolsConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary with corpus, web_search, web_fetch bool fields.

        Returns:
            ResearchToolsConfig instance.
        """
        return cls(
            corpus=data.get("corpus", True),
            web_search=data.get("web_search", True),
            web_fetch=data.get("web_fetch", True),
        )


@dataclass
class ProjectConfig:
    """Configuration for a QuestFoundry project."""

    name: str
    version: int = 1
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    stages: list[str] = field(default_factory=lambda: list(DEFAULT_STAGES))
    gates: list[GateConfig] = field(default_factory=list)
    research_tools: ResearchToolsConfig = field(default_factory=ResearchToolsConfig)

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
        default_provider = provider_data.get("default", f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}")
        if "/" in default_provider:
            provider_name, model = default_provider.split("/", 1)
        else:
            provider_name, model = default_provider, DEFAULT_MODEL

        provider = ProviderConfig(name=provider_name, model=model)

        # Parse stages
        pipeline_data = data.get("pipeline", {})
        stages = pipeline_data.get("stages", list(DEFAULT_STAGES))

        # Parse gates
        gates_data = pipeline_data.get("gates", {})
        gates = [
            GateConfig(stage=stage, required=(value == "required"))
            for stage, value in gates_data.items()
        ]

        # Parse research tools config
        research_tools_data = data.get("research_tools", {})
        research_tools = ResearchToolsConfig.from_dict(research_tools_data)

        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", 1),
            provider=provider,
            stages=stages,
            gates=gates,
            research_tools=research_tools,
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
