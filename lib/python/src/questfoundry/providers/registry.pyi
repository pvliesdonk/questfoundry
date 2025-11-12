"""Type stubs for provider registry"""

from typing import Any

from .audio import AudioProvider
from .base import ImageProvider, TextProvider
from .config import ProviderConfig

class ProviderRegistry:
    """Registry for managing provider instances."""

    config: ProviderConfig

    def __init__(self, config: ProviderConfig) -> None: ...
    def register_text_provider(
        self, name: str, provider_class: type[TextProvider]
    ) -> None: ...
    def register_image_provider(
        self, name: str, provider_class: type[ImageProvider]
    ) -> None: ...
    def register_audio_provider(
        self, name: str, provider_class: type[AudioProvider]
    ) -> None: ...
    def get_text_provider(
        self, name: str | None = None, config: dict[str, Any] | None = None
    ) -> TextProvider: ...
    def get_image_provider(
        self, name: str | None = None, config: dict[str, Any] | None = None
    ) -> ImageProvider: ...
    def get_audio_provider(
        self, name: str | None = None, config: dict[str, Any] | None = None
    ) -> AudioProvider: ...
    def list_providers(self) -> dict[str, list[str]]: ...
