"""Project configuration loading."""

from __future__ import annotations

import os
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
    """Configuration for LLM provider (legacy single-provider format)."""

    name: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


@dataclass
class ProvidersConfig:
    """Configuration for LLM providers with phase-specific overrides.

    Supports hybrid model configurations where different phases use different
    LLM providers. Each phase (discuss, summarize, serialize) can optionally
    override the default provider.

    Resolution order for each phase (within this class):
    1. Environment variable (e.g., QF_PROVIDER_SERIALIZE)
    2. Project config (e.g., providers.serialize)
    3. Default provider (from providers.default)

    Note: CLI flags and QF_PROVIDER are handled by the orchestrator for
    backward compatibility, not by this class.

    Attributes:
        default: Default provider string (e.g., "ollama/qwen3:8b"). Required.
        discuss: Optional provider override for the discuss phase.
        summarize: Optional provider override for the summarize phase.
        serialize: Optional provider override for the serialize phase.
    """

    default: str
    discuss: str | None = None
    summarize: str | None = None
    serialize: str | None = None

    def get_discuss_provider(self) -> str:
        """Get the effective provider for the discuss phase.

        Checks environment variable QF_PROVIDER_DISCUSS, then config,
        then falls back to default.
        """
        return os.getenv("QF_PROVIDER_DISCUSS") or self.discuss or self.default

    def get_summarize_provider(self) -> str:
        """Get the effective provider for the summarize phase.

        Checks environment variable QF_PROVIDER_SUMMARIZE, then config,
        then falls back to default.
        """
        return os.getenv("QF_PROVIDER_SUMMARIZE") or self.summarize or self.default

    def get_serialize_provider(self) -> str:
        """Get the effective provider for the serialize phase.

        Checks environment variable QF_PROVIDER_SERIALIZE, then config,
        then falls back to default.
        """
        return os.getenv("QF_PROVIDER_SERIALIZE") or self.serialize or self.default

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvidersConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary with provider configuration. Can have:
                - default: Required default provider string
                - discuss: Optional discuss phase provider
                - summarize: Optional summarize phase provider
                - serialize: Optional serialize phase provider

        Returns:
            ProvidersConfig instance.
        """
        default = data.get("default", f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}")
        return cls(
            default=default,
            discuss=data.get("discuss"),
            summarize=data.get("summarize"),
            serialize=data.get("serialize"),
        )


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
    providers: ProvidersConfig = field(
        default_factory=lambda: ProvidersConfig(default=f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}")
    )
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
        # Parse providers (new format with hybrid support)
        provider_data = data.get("providers", {})
        providers_config = ProvidersConfig.from_dict(provider_data)

        # Also populate legacy provider field for backward compatibility
        default_provider = providers_config.default
        if "/" in default_provider:
            provider_name, model = default_provider.split("/", 1)
        else:
            # Use provider-specific default model, not hardcoded DEFAULT_MODEL
            from questfoundry.providers.factory import get_default_model

            provider_name = default_provider
            model = get_default_model(provider_name) or DEFAULT_MODEL
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
            providers=providers_config,
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


def create_default_config(
    name: str,
    provider: str | None = None,
) -> ProjectConfig:
    """Create a default project configuration.

    Args:
        name: Project name.
        provider: Optional default provider string (e.g., "ollama/qwen3:8b").
            If not provided, uses the system default.

    Returns:
        ProjectConfig with default values.
    """
    # Determine provider string
    provider_string = provider or f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}"

    # Parse into legacy ProviderConfig
    if "/" in provider_string:
        provider_name, model = provider_string.split("/", 1)
    else:
        # Use provider-specific default model, not hardcoded DEFAULT_MODEL
        from questfoundry.providers.factory import get_default_model

        provider_name = provider_string
        model = get_default_model(provider_name) or DEFAULT_MODEL

    return ProjectConfig(
        name=name,
        provider=ProviderConfig(name=provider_name, model=model),
        providers=ProvidersConfig(default=provider_string),
    )
