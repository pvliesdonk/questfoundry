"""Base class for audio generation providers."""

from abc import abstractmethod
from typing import Any

from ..base import Provider


class AudioProvider(Provider):
    """
    Base class for audio generation providers.

    Audio providers generate speech/audio from text using text-to-speech (TTS)
    or other audio generation models.
    """

    @abstractmethod
    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate audio from text.

        Args:
            text: Text to convert to speech
            voice: Optional voice ID/name to use
            **kwargs: Provider-specific parameters (e.g., stability, speed)

        Returns:
            Audio data as bytes (typically WAV or MP3 format)

        Raises:
            ValueError: If text is empty or voice is invalid
            RuntimeError: If audio generation fails
        """
        pass

    @abstractmethod
    def list_voices(self) -> list[dict[str, Any]]:
        """
        List available voices for this provider.

        Returns:
            List of voice dictionaries with at least:
            - voice_id: Unique identifier
            - name: Human-readable name
            - description: Optional description
            - language: Language code (e.g., "en-US")
            - gender: Optional gender (e.g., "male", "female")

        Example:
            >>> provider.list_voices()
            [
                {
                    "voice_id": "21m00Tcm4TlvDq8ikWAM",
                    "name": "Rachel",
                    "description": "Calm and professional",
                    "language": "en-US",
                    "gender": "female"
                }
            ]
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate that the provider is properly configured.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported audio formats.

        Returns:
            List of format strings (e.g., ["mp3", "wav", "pcm"])
        """
        # Default to common formats, subclasses can override
        return ["mp3", "wav"]

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}()"
