"""ElevenLabs text-to-speech provider."""

import os
from typing import Any

from . import AudioProvider


class ElevenLabsProvider(AudioProvider):
    """
    ElevenLabs text-to-speech provider.

    Provides high-quality neural voice synthesis using ElevenLabs API.
    Supports multiple voices, languages, and voice settings.

    Configuration:
        api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
        model: Model to use (default: "eleven_monolingual_v1")
        voice_id: Default voice ID (optional)
        stability: Voice stability 0.0-1.0 (default: 0.5)
        similarity_boost: Similarity boost 0.0-1.0 (default: 0.75)
    """

    API_BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, config: dict[str, Any]):
        """
        Initialize ElevenLabs provider.

        Args:
            config: Configuration with api_key and optional settings

        Raises:
            ValueError: If api_key is missing
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.get("api_key") or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key required. "
                "Set 'api_key' in config or ELEVENLABS_API_KEY env var"
            )

        # Voice settings
        self.model = config.get("model", "eleven_monolingual_v1")
        self.default_voice_id = config.get("voice_id")
        self.stability = config.get("stability", 0.5)
        self.similarity_boost = config.get("similarity_boost", 0.75)

    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate audio using ElevenLabs TTS.

        Args:
            text: Text to convert to speech
            voice: Voice ID to use (uses default if not specified)
            **kwargs: Additional parameters:
                - stability: Voice stability (0.0-1.0)
                - similarity_boost: Similarity boost (0.0-1.0)
                - model: Model to use

        Returns:
            MP3 audio data as bytes

        Raises:
            ValueError: If text is empty or voice is invalid
            RuntimeError: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Determine voice to use
        voice_id = voice or self.default_voice_id
        if not voice_id:
            raise ValueError(
                "Voice ID required. "
                "Specify 'voice' parameter or set 'voice_id' in config"
            )

        # Build request parameters
        stability = kwargs.get("stability", self.stability)
        similarity_boost = kwargs.get("similarity_boost", self.similarity_boost)
        model = kwargs.get("model", self.model)

        # Prepare request
        url = f"{self.API_BASE_URL}/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }
        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        }

        # Make API request
        try:
            import requests  # type: ignore[import-untyped]

            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.content

        except ImportError:
            raise RuntimeError(
                "requests library required for ElevenLabs provider. "
                "Install with: pip install requests"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ElevenLabs API call failed: {e}")

    def list_voices(self) -> list[dict[str, Any]]:
        """
        List available ElevenLabs voices.

        Returns:
            List of voice dictionaries

        Raises:
            RuntimeError: If API call fails
        """
        url = f"{self.API_BASE_URL}/voices"
        headers = {"xi-api-key": self.api_key}

        try:
            import requests

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Transform to standard format
            voices = []
            for voice_data in data.get("voices", []):
                voices.append(
                    {
                        "voice_id": voice_data.get("voice_id"),
                        "name": voice_data.get("name"),
                        "description": voice_data.get("description", ""),
                        "language": voice_data.get("labels", {}).get("language", "en"),
                        "gender": voice_data.get("labels", {}).get("gender", "neutral"),
                        "category": voice_data.get("category", "general"),
                    }
                )

            return voices

        except ImportError:
            raise RuntimeError(
                "requests library required for ElevenLabs provider. "
                "Install with: pip install requests"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ElevenLabs API call failed: {e}")

    def validate_config(self) -> None:
        """
        Validate configuration by testing API key.

        Raises:
            ValueError: If API key is invalid
        """
        try:
            # Test API key by listing voices
            self.list_voices()
        except RuntimeError as e:
            raise ValueError(f"Invalid configuration: {e}") from e

    def get_supported_formats(self) -> list[str]:
        """
        Get supported audio formats.

        Returns:
            List with MP3 format
        """
        return ["mp3"]

    def __repr__(self) -> str:
        """String representation."""
        has_key = bool(self.api_key)
        return f"ElevenLabsProvider(has_api_key={has_key}, model={self.model})"
