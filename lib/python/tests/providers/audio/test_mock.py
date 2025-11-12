"""Tests for mock audio provider."""

import wave
from io import BytesIO

import pytest

from questfoundry.providers.audio.mock import MockAudioProvider


@pytest.fixture
def provider():
    """Create a mock audio provider."""
    return MockAudioProvider()


def test_provider_initialization():
    """Test provider initializes correctly."""
    provider = MockAudioProvider()
    assert provider.generate_count == 0
    provider.validate_config()  # Should not raise


def test_provider_initialization_with_config():
    """Test provider accepts config dict."""
    config = {"key": "value"}
    provider = MockAudioProvider(config)
    assert provider.config == config


def test_generate_audio_simple(provider):
    """Test basic audio generation."""
    text = "Hello, world!"
    audio_data = provider.generate_audio(text)

    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
    assert provider.generate_count == 1


def test_generate_audio_empty_text(provider):
    """Test that empty text raises error."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        provider.generate_audio("")

    with pytest.raises(ValueError, match="Text cannot be empty"):
        provider.generate_audio("   ")


def test_generate_audio_with_voice(provider):
    """Test audio generation with voice parameter."""
    text = "Hello, world!"
    audio_data = provider.generate_audio(text, voice="mock-voice-001")

    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0


def test_generate_audio_multiple_calls(provider):
    """Test multiple generations increment counter."""
    provider.generate_audio("First call")
    provider.generate_audio("Second call")
    provider.generate_audio("Third call")

    assert provider.generate_count == 3


def test_generate_audio_is_valid_wav(provider):
    """Test that generated audio is valid WAV format."""
    text = "Test audio generation"
    audio_data = provider.generate_audio(text)

    # Try to parse as WAV
    wav_buffer = BytesIO(audio_data)
    with wave.open(wav_buffer, "rb") as wav_file:
        # Check WAV parameters
        assert wav_file.getnchannels() == 1  # Mono
        assert wav_file.getsampwidth() == 2  # 16-bit
        assert wav_file.getframerate() == 44100  # 44.1kHz
        assert wav_file.getnframes() > 0  # Has frames


def test_generate_audio_duration_scales_with_text(provider):
    """Test that longer text produces longer audio."""
    short_text = "Hi"
    long_text = " ".join(["Hello"] * 100)

    short_audio = provider.generate_audio(short_text)
    long_audio = provider.generate_audio(long_text)

    # Parse WAV files to check duration
    short_wav = wave.open(BytesIO(short_audio), "rb")
    long_wav = wave.open(BytesIO(long_audio), "rb")

    short_frames = short_wav.getnframes()
    long_frames = long_wav.getnframes()

    short_wav.close()
    long_wav.close()

    # Longer text should have more frames
    assert long_frames > short_frames


def test_list_voices(provider):
    """Test listing available voices."""
    voices = provider.list_voices()

    assert isinstance(voices, list)
    assert len(voices) > 0

    # Check voice structure
    for voice in voices:
        assert "voice_id" in voice
        assert "name" in voice
        assert "description" in voice
        assert "language" in voice
        assert "gender" in voice


def test_list_voices_returns_expected_voices(provider):
    """Test that specific mock voices are included."""
    voices = provider.list_voices()
    voice_ids = [v["voice_id"] for v in voices]

    assert "mock-voice-001" in voice_ids
    assert "mock-voice-002" in voice_ids
    assert "mock-voice-003" in voice_ids


def test_validate_config(provider):
    """Test configuration validation."""
    provider.validate_config()  # Should not raise


def test_get_supported_formats(provider):
    """Test supported format list."""
    formats = provider.get_supported_formats()

    assert isinstance(formats, list)
    assert "wav" in formats


def test_reset_count(provider):
    """Test resetting generation counter."""
    provider.generate_audio("Test 1")
    provider.generate_audio("Test 2")

    assert provider.generate_count == 2

    provider.reset_count()

    assert provider.generate_count == 0


def test_repr(provider):
    """Test string representation."""
    repr_str = repr(provider)

    assert "MockAudioProvider" in repr_str
    assert "generate_count=0" in repr_str

    provider.generate_audio("Test")
    repr_str = repr(provider)

    assert "generate_count=1" in repr_str


def test_minimum_duration(provider):
    """Test that even short text gets minimum duration."""
    text = "a"  # Single character
    audio_data = provider.generate_audio(text)

    wav = wave.open(BytesIO(audio_data), "rb")
    duration = wav.getnframes() / wav.getframerate()
    wav.close()

    # Should have at least 1 second
    assert duration >= 1.0
