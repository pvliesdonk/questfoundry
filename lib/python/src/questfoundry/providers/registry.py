"""Provider registry for discovering and instantiating providers"""

import logging

from .audio import AudioProvider
from .base import ImageProvider, TextProvider
from .config import ProviderConfig

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Registry for managing text and image providers.

    The registry handles provider instantiation and configuration,
    allowing easy access to providers by name.

    Example:
        >>> config = ProviderConfig()
        >>> registry = ProviderRegistry(config)
        >>> text_provider = registry.get_text_provider("openai")
        >>> image_provider = registry.get_image_provider("dalle")
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize registry with configuration.

        Args:
            config: Provider configuration
        """
        logger.debug("Initializing ProviderRegistry")
        logger.trace("Registry config path: %s", config.config_path)
        self.config = config
        self._text_providers: dict[str, type[TextProvider]] = {}
        self._image_providers: dict[str, type[ImageProvider]] = {}
        self._audio_providers: dict[str, type[AudioProvider]] = {}
        self._text_instances: dict[str, TextProvider] = {}
        self._image_instances: dict[str, ImageProvider] = {}
        self._audio_instances: dict[str, AudioProvider] = {}

        # Register built-in providers
        logger.debug("Registering built-in providers")
        self._register_builtin_providers()
        logger.info(
            "ProviderRegistry initialized with %d text, %d image, %d audio providers",
            len(self._text_providers),
            len(self._image_providers),
            len(self._audio_providers),
        )

    def register_text_provider(
        self, name: str, provider_class: type[TextProvider]
    ) -> None:
        """
        Register a text provider class.

        Args:
            name: Provider name (e.g., 'openai', 'ollama')
            provider_class: Provider class to register
        """
        logger.debug("Registering text provider: %s", name)
        self._text_providers[name] = provider_class
        logger.trace("Text provider %s registered (%s)", name, provider_class.__name__)

    def register_image_provider(
        self, name: str, provider_class: type[ImageProvider]
    ) -> None:
        """
        Register an image provider class.

        Args:
            name: Provider name (e.g., 'dalle', 'a1111')
            provider_class: Provider class to register
        """
        logger.debug("Registering image provider: %s", name)
        self._image_providers[name] = provider_class
        logger.trace("Image provider %s registered (%s)", name, provider_class.__name__)

    def register_audio_provider(
        self, name: str, provider_class: type[AudioProvider]
    ) -> None:
        """
        Register an audio provider class.

        Args:
            name: Provider name (e.g., 'elevenlabs', 'google-tts')
            provider_class: Provider class to register
        """
        logger.debug("Registering audio provider: %s", name)
        self._audio_providers[name] = provider_class
        logger.trace("Audio provider %s registered (%s)", name, provider_class.__name__)

    def get_text_provider(self, name: str | None = None) -> TextProvider:
        """
        Get or create a text provider instance.

        Args:
            name: Provider name. If None, uses default from config.

        Returns:
            Text provider instance

        Raises:
            ValueError: If provider not found or not registered
        """
        logger.debug("Getting text provider: %s", name)

        if name is None:
            logger.trace("No provider name specified, using default")
            name = self.config.get_default_provider("text")
            if name is None:
                logger.error("No default text provider configured")
                raise ValueError("No default text provider configured")
            logger.trace("Default text provider: %s", name)

        # Return cached instance if available
        if name in self._text_instances:
            logger.trace("Returning cached text provider instance: %s", name)
            return self._text_instances[name]

        # Get provider class
        if name not in self._text_providers:
            logger.error(
                "Text provider %s not registered. Available: %s",
                name,
                list(self._text_providers.keys()),
            )
            raise ValueError(f"Text provider '{name}' not registered")

        provider_class = self._text_providers[name]
        logger.debug("Creating new text provider instance: %s", name)

        # Get configuration
        try:
            provider_config = self.config.get_provider_config("text", name)
            logger.trace("Loaded configuration for text provider %s", name)
        except KeyError:
            logger.debug(
                "No configuration found for text provider %s, using empty config", name
            )
            provider_config = {}

        # Create and cache instance
        logger.trace("Instantiating text provider class: %s", provider_class.__name__)
        instance = provider_class(provider_config)
        logger.trace("Validating text provider configuration: %s", name)
        instance.validate_config()
        self._text_instances[name] = instance
        logger.info("Text provider %s initialized and cached successfully", name)

        return instance

    def get_image_provider(self, name: str | None = None) -> ImageProvider:
        """
        Get or create an image provider instance.

        Args:
            name: Provider name. If None, uses default from config.

        Returns:
            Image provider instance

        Raises:
            ValueError: If provider not found or not registered
        """
        logger.debug("Getting image provider: %s", name)

        if name is None:
            logger.trace("No provider name specified, using default")
            name = self.config.get_default_provider("image")
            if name is None:
                logger.error("No default image provider configured")
                raise ValueError("No default image provider configured")

        # Return cached instance if available
        if name in self._image_instances:
            logger.trace("Returning cached image provider instance: %s", name)
            return self._image_instances[name]

        # Get provider class
        if name not in self._image_providers:
            logger.error("Image provider %s not registered", name)
            raise ValueError(f"Image provider '{name}' not registered")

        provider_class = self._image_providers[name]
        logger.debug("Creating new image provider instance: %s", name)

        # Get configuration
        try:
            provider_config = self.config.get_provider_config("image", name)
            logger.trace("Loaded configuration for image provider %s", name)
        except KeyError:
            logger.debug("No configuration found for image provider %s", name)
            provider_config = {}

        # Create and cache instance
        instance = provider_class(provider_config)
        instance.validate_config()
        self._image_instances[name] = instance
        logger.info("Image provider %s initialized and cached successfully", name)

        return instance

    def get_audio_provider(self, name: str | None = None) -> AudioProvider:
        """
        Get or create an audio provider instance.

        Args:
            name: Provider name. If None, uses default from config.

        Returns:
            Audio provider instance

        Raises:
            ValueError: If provider not found or not registered
        """
        logger.debug("Getting audio provider: %s", name)

        if name is None:
            logger.trace("No provider name specified, using default")
            name = self.config.get_default_provider("audio")
            if name is None:
                logger.error("No default audio provider configured")
                raise ValueError("No default audio provider configured")

        # Return cached instance if available
        if name in self._audio_instances:
            logger.trace("Returning cached audio provider instance: %s", name)
            return self._audio_instances[name]

        # Get provider class
        if name not in self._audio_providers:
            logger.error("Audio provider %s not registered", name)
            raise ValueError(f"Audio provider '{name}' not registered")

        provider_class = self._audio_providers[name]
        logger.debug("Creating new audio provider instance: %s", name)

        # Get configuration
        try:
            provider_config = self.config.get_provider_config("audio", name)
            logger.trace("Loaded configuration for audio provider %s", name)
        except KeyError:
            logger.debug("No configuration found for audio provider %s", name)
            provider_config = {}

        # Create and cache instance
        instance = provider_class(provider_config)
        instance.validate_config()
        self._audio_instances[name] = instance
        logger.info("Audio provider %s initialized and cached successfully", name)

        return instance

    def list_text_providers(self) -> list[str]:
        """
        List registered text providers.

        Returns:
            List of text provider names
        """
        return list(self._text_providers.keys())

    def list_image_providers(self) -> list[str]:
        """
        List registered image providers.

        Returns:
            List of image provider names
        """
        return list(self._image_providers.keys())

    def list_audio_providers(self) -> list[str]:
        """
        List registered audio providers.

        Returns:
            List of audio provider names
        """
        return list(self._audio_providers.keys())

    def close_all(self) -> None:
        """Close all provider instances and release resources."""
        logger.debug("Closing all provider instances")
        text_count = 0
        for text_provider in self._text_instances.values():
            logger.trace("Closing text provider instance")
            try:
                text_provider.close()
                text_count += 1
            except Exception as e:
                logger.warning("Error closing text provider: %s", str(e), exc_info=True)

        image_count = 0
        for image_provider in self._image_instances.values():
            logger.trace("Closing image provider instance")
            try:
                image_provider.close()
                image_count += 1
            except Exception as e:
                logger.warning(
                    "Error closing image provider: %s", str(e), exc_info=True
                )

        audio_count = 0
        for audio_provider in self._audio_instances.values():
            logger.trace("Closing audio provider instance")
            try:
                audio_provider.close()
                audio_count += 1
            except Exception as e:
                logger.warning(
                    "Error closing audio provider: %s", str(e), exc_info=True
                )

        self._text_instances.clear()
        self._image_instances.clear()
        self._audio_instances.clear()
        logger.info(
            "Closed %d text, %d image, %d audio provider instances",
            text_count,
            image_count,
            audio_count,
        )

    def _register_builtin_providers(self) -> None:
        """Register built-in providers."""
        # Import here to avoid circular dependencies and to make
        # provider dependencies optional
        logger.trace("Attempting to register built-in providers")

        try:
            from .text.openai import OpenAIProvider

            self.register_text_provider("openai", OpenAIProvider)
            logger.debug("OpenAI text provider registered successfully")
        except ImportError:
            logger.debug("OpenAI provider not available (optional dependency)")

        try:
            from .text.ollama import OllamaProvider

            self.register_text_provider("ollama", OllamaProvider)
            logger.debug("Ollama text provider registered successfully")
        except ImportError:
            logger.debug("Ollama provider not available (optional dependency)")

        # Register image providers
        try:
            from .image.dalle import DalleProvider

            self.register_image_provider("dalle", DalleProvider)
            logger.debug("DALL-E image provider registered successfully")
        except ImportError:
            logger.debug("DALL-E provider not available (optional dependency)")

        try:
            from .image.a1111 import Automatic1111Provider

            self.register_image_provider("a1111", Automatic1111Provider)
            logger.debug("Automatic1111 image provider registered successfully")
        except ImportError:
            logger.debug("Automatic1111 provider not available (optional dependency)")

        # Register audio providers
        try:
            from .audio.elevenlabs import ElevenLabsProvider

            self.register_audio_provider("elevenlabs", ElevenLabsProvider)
            logger.debug("ElevenLabs audio provider registered successfully")
        except ImportError:
            logger.debug("ElevenLabs provider not available (optional dependency)")

        try:
            from .audio.mock import MockAudioProvider

            self.register_audio_provider("mock", MockAudioProvider)
            logger.debug("Mock audio provider registered successfully")
        except ImportError:
            logger.debug("Mock audio provider not available (optional dependency)")
