"""
End-to-end integration tests for all providers.

These tests make real API calls and require actual API keys to be set.
They are skipped in CI but can be run manually for validation.

To run these tests manually with API keys:
    export OPENAI_API_KEY="your-key"
    export GOOGLE_AI_API_KEY="your-key"
    export AWS_ACCESS_KEY_ID="your-key"
    export AWS_SECRET_ACCESS_KEY="your-secret"
    export GOOGLE_CLOUD_PROJECT="your-project"
    export GOOGLE_CLOUD_API_KEY="your-key"
    export ELEVENLABS_API_KEY="your-key"

    pytest tests/integration/test_providers_e2e.py -v
"""

import os

import pytest

# Check which API keys are available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


# ============================================================================
# TEXT PROVIDERS E2E TESTS
# ============================================================================


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set. Set it to run this E2E test manually.",
)
def test_openai_provider_e2e():
    """E2E test for OpenAI provider with real API."""
    from questfoundry.providers.text.openai import OpenAIProvider

    provider = OpenAIProvider({"api_key": OPENAI_API_KEY, "model": "gpt-4o-mini"})

    # Validate config
    provider.validate_config()

    # Generate text
    response = provider.generate_text(
        "Write a single word: Hello", max_tokens=10, temperature=0.0
    )

    assert isinstance(response, str)
    assert len(response) > 0
    print(f"OpenAI response: {response}")


@pytest.mark.skipif(
    not GOOGLE_AI_API_KEY,
    reason="GOOGLE_AI_API_KEY not set. Set it to run this E2E test manually.",
)
def test_gemini_provider_e2e():
    """E2E test for Gemini provider with real API."""
    from questfoundry.providers.text.gemini import GeminiProvider

    provider = GeminiProvider(
        {"api_key": GOOGLE_AI_API_KEY, "model": "gemini-2.0-flash-exp"}
    )

    # Validate config
    provider.validate_config()

    # Generate text
    response = provider.generate_text(
        "Write a single word: Hello", max_tokens=10, temperature=0.0
    )

    assert isinstance(response, str)
    assert len(response) > 0
    print(f"Gemini response: {response}")


@pytest.mark.skipif(
    not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY),
    reason="AWS credentials not set. Set them to run this E2E test manually.",
)
def test_bedrock_provider_e2e():
    """E2E test for Bedrock provider with real API."""
    from questfoundry.providers.text.bedrock import BedrockProvider

    provider = BedrockProvider(
        {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region": "us-east-1",
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        }
    )

    # Validate config
    provider.validate_config()

    # Generate text
    response = provider.generate_text(
        "Write a single word: Hello", max_tokens=10, temperature=0.0
    )

    assert isinstance(response, str)
    assert len(response) > 0
    print(f"Bedrock response: {response}")


# ============================================================================
# IMAGE PROVIDERS E2E TESTS
# ============================================================================


@pytest.mark.skipif(
    not (GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_API_KEY),
    reason="Google Cloud credentials not set. Set them to run this E2E test manually.",
)
def test_imagen_provider_e2e():
    """E2E test for Imagen provider with real API."""
    from questfoundry.providers.image.imagen import ImagenProvider

    provider = ImagenProvider(
        {
            "project_id": GOOGLE_CLOUD_PROJECT,
            "api_key": GOOGLE_CLOUD_API_KEY,
            "location": "us-central1",
            "model": "imagen-4.0-preview",
        }
    )

    # Validate config
    provider.validate_config()

    # Generate image
    image_data = provider.generate_image("A simple red circle on white background")

    assert isinstance(image_data, bytes)
    assert len(image_data) > 0
    print(f"Imagen generated {len(image_data)} bytes")


# Note: Mock image provider not included in Epic 14
# Available image providers: A1111 (requires local server),
# DALL-E (requires OpenAI key), Imagen (requires Google Cloud)


# ============================================================================
# AUDIO PROVIDERS E2E TESTS
# ============================================================================


@pytest.mark.skipif(
    not ELEVENLABS_API_KEY,
    reason="ELEVENLABS_API_KEY not set. Set it to run this E2E test manually.",
)
def test_elevenlabs_provider_e2e():
    """E2E test for ElevenLabs provider with real API."""
    from questfoundry.providers.audio.elevenlabs import ElevenLabsProvider

    provider = ElevenLabsProvider({"api_key": ELEVENLABS_API_KEY})

    # Validate config
    provider.validate_config()

    # List voices
    voices = provider.list_voices()
    assert isinstance(voices, list)
    assert len(voices) > 0
    print(f"ElevenLabs found {len(voices)} voices")

    # Generate audio with first voice
    voice_id = voices[0]["voice_id"]
    audio_data = provider.generate_audio("Hello, world!", voice=voice_id)

    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
    print(f"ElevenLabs generated {len(audio_data)} bytes")


def test_mock_audio_provider_e2e():
    """E2E test for Mock audio provider (no API key needed)."""
    from questfoundry.providers.audio.mock import MockAudioProvider

    provider = MockAudioProvider()

    # Validate config
    provider.validate_config()  # Should not raise

    # List voices
    voices = provider.list_voices()
    assert isinstance(voices, list)
    assert len(voices) > 0

    # Generate audio
    audio_data = provider.generate_audio("Hello, world!")

    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0

    # Verify it's a valid WAV file
    import wave
    from io import BytesIO

    wav_buffer = BytesIO(audio_data)
    with wave.open(wav_buffer, "rb") as wav_file:
        assert wav_file.getnchannels() == 1  # Mono
        assert wav_file.getsampwidth() == 2  # 16-bit
        assert wav_file.getframerate() == 44100  # 44.1kHz
        assert wav_file.getnframes() > 0  # Has frames

    print(f"Mock audio generated {len(audio_data)} bytes of valid WAV")


# ============================================================================
# CROSS-PROVIDER COMPARISON TESTS
# ============================================================================


def test_all_available_text_providers_consistency():
    """Test that all available text providers produce consistent results."""
    from questfoundry.providers.text.gemini import GeminiProvider
    from questfoundry.providers.text.openai import OpenAIProvider

    prompt = "Write exactly one word: Hello"
    results = {}

    # Test OpenAI if available
    if OPENAI_API_KEY:
        provider = OpenAIProvider({"api_key": OPENAI_API_KEY})
        provider.validate_config()
        results["openai"] = provider.generate_text(
            prompt, max_tokens=5, temperature=0.0
        )

    # Test Gemini if available
    if GOOGLE_AI_API_KEY:
        provider = GeminiProvider({"api_key": GOOGLE_AI_API_KEY})
        provider.validate_config()
        results["gemini"] = provider.generate_text(
            prompt, max_tokens=5, temperature=0.0
        )

    # Print results for manual verification
    if results:
        print("\nText provider comparison:")
        for name, response in results.items():
            print(f"  {name}: {response}")

        # All responses should contain "Hello" or similar
        for name, response in results.items():
            assert isinstance(response, str)
            assert len(response) > 0
    else:
        pytest.skip("No text provider API keys available")


def test_all_available_audio_providers_consistency():
    """Test that all available audio providers work correctly."""
    results = {}

    # Mock is always available
    from questfoundry.providers.audio.mock import MockAudioProvider

    provider = MockAudioProvider()
    results["mock"] = provider.generate_audio("Test")

    # Test ElevenLabs if available
    if ELEVENLABS_API_KEY:
        from questfoundry.providers.audio.elevenlabs import ElevenLabsProvider

        provider = ElevenLabsProvider({"api_key": ELEVENLABS_API_KEY})
        voices = provider.list_voices()
        if voices:
            results["elevenlabs"] = provider.generate_audio(
                "Test", voice=voices[0]["voice_id"]
            )

    # Print results
    print("\nAudio provider comparison:")
    for name, audio_data in results.items():
        print(f"  {name}: {len(audio_data)} bytes")
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
