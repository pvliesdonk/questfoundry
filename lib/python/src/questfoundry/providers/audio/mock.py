"""Mock audio provider for testing."""

import wave
from io import BytesIO
from typing import Any

from . import AudioProvider


class MockAudioProvider(AudioProvider):
    """
    Mock audio provider for testing and development.

    Returns silent WAV audio files for any text input. Useful for:
    - Unit testing without real API calls
    - Development without audio provider accounts
    - CI/CD pipelines without credentials
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize mock audio provider.

        Args:
            config: Optional configuration (unused for mock)
        """
        super().__init__(config or {})
        self.generate_count = 0  # Track number of generations for testing

    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate silent WAV audio.

        Args:
            text: Text to "convert" (length affects duration)
            voice: Voice to use (recorded but not used)
            **kwargs: Additional parameters (recorded but not used)

        Returns:
            Silent WAV audio as bytes

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        self.generate_count += 1

        # Calculate duration based on text length (roughly 150 words per minute)
        words = len(text.split())
        duration_seconds = max(1.0, words / 2.5)  # Minimum 1 second

        # Generate silent WAV file
        return self._create_silent_wav(duration_seconds)

    def list_voices(self) -> list[dict[str, Any]]:
        """
        List mock voices.

        Returns:
            List of mock voice configurations
        """
        return [
            {
                "voice_id": "mock-voice-001",
                "name": "Mock Voice 1",
                "description": "Test voice for male narration",
                "language": "en-US",
                "gender": "male",
            },
            {
                "voice_id": "mock-voice-002",
                "name": "Mock Voice 2",
                "description": "Test voice for female narration",
                "language": "en-US",
                "gender": "female",
            },
            {
                "voice_id": "mock-voice-003",
                "name": "Mock Voice 3",
                "description": "Test voice for neutral narration",
                "language": "en-US",
                "gender": "neutral",
            },
        ]

    def validate_config(self) -> None:
        """
        Validate configuration (always valid for mock).

        Raises:
            ValueError: Never raises for mock provider
        """
        pass

    def get_supported_formats(self) -> list[str]:
        """
        Get supported formats.

        Returns:
            List with WAV format
        """
        return ["wav"]

    def _create_silent_wav(self, duration: float) -> bytes:
        """
        Create a silent WAV file of specified duration.

        Args:
            duration: Duration in seconds

        Returns:
            WAV file as bytes
        """
        # WAV parameters
        sample_rate = 44100  # CD quality
        num_channels = 1  # Mono
        sample_width = 2  # 16-bit
        num_frames = int(sample_rate * duration)

        # Create WAV in memory
        wav_buffer = BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)

            # Write silent frames (zeros)
            silent_data = b"\x00\x00" * num_frames
            wav_file.writeframes(silent_data)

        return wav_buffer.getvalue()

    def reset_count(self) -> None:
        """Reset generation counter (useful for testing)."""
        self.generate_count = 0

    def __repr__(self) -> str:
        """String representation."""
        return f"MockAudioProvider(generate_count={self.generate_count})"
