"""Type stubs for base role classes"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from ..providers.audio import AudioProvider
from ..providers.base import ImageProvider, TextProvider
from .human_callback import HumanCallback
from .session import RoleSession

MAX_ARTIFACT_VALUE_LENGTH: int
MAX_ARTIFACTS_IN_CONTEXT: int
MAX_FORMATTED_CONTEXT_SIZE: int

@dataclass
class RoleContext:
    """Context provided to a role for task execution."""

    task: str
    artifacts: list[Artifact] = ...
    project_metadata: dict[str, Any] = ...
    workspace_path: Path | None = None
    additional_context: dict[str, Any] = ...

@dataclass
class RoleResult:
    """Result of a role execution."""

    success: bool
    output: str
    artifacts: list[Artifact] = ...
    metadata: dict[str, Any] = ...
    error: str | None = None

class Role(ABC):
    """Base class for all QuestFoundry roles."""

    @property
    def name(self) -> str: ...
    @property
    def abbreviation(self) -> str: ...
    @property
    def role_name(self) -> str: ...

    text_provider: TextProvider
    image_provider: ImageProvider | None
    audio_provider: AudioProvider | None
    session: RoleSession | None
    human_callback: HumanCallback | None
    has_image_provider: bool
    has_audio_provider: bool

    def __init__(
        self,
        text_provider: TextProvider | None = None,
        image_provider: ImageProvider | None = None,
        audio_provider: AudioProvider | None = None,
        session: RoleSession | None = None,
        human_callback: HumanCallback | None = None,
        provider: TextProvider | None = None,
        spec_path: Path | None = None,
        config: dict[str, Any] | None = None,
        role_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
    @abstractmethod
    def execute(self, context: RoleContext) -> RoleResult: ...
    def format_context(self, context: RoleContext) -> str: ...
    def parse_output(self, output: str) -> RoleResult: ...
    def load_system_prompt(self) -> str: ...
    def ask_human(
        self, question: str, context: dict[str, Any] | None = None
    ) -> str: ...
    def build_system_prompt(self, *args: Any, **kwargs: Any) -> str: ...
    def format_artifacts(self, artifacts: list[Artifact]) -> str: ...
    def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str: ...
    def _parse_json_from_response(self, response: str) -> dict[str, Any]: ...
    def execute_task(self, *args: Any, **kwargs: Any) -> RoleResult: ...
