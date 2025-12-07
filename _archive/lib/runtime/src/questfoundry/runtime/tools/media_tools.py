"""
Media Tools - Image and audio generation tools.

These tools wrap the underlying generation backends:
- generate_image: Wraps StableDiffusion/DALL-E/Gemini providers
- generate_audio: Audio synthesis (TTS, music, SFX)

Provider selection is automatic based on available API keys:
- Image: OPENAI_API_KEY (DALL-E), GOOGLE_API_KEY (Gemini), A1111_URL (SD)
- Audio: ELEVENLABS_API_KEY (TTS), SUNO_API_KEY (music)
"""

import logging
import os
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class GenerateImage(BaseTool):
    """
    Generate an image from a text prompt.

    The Illustrator uses this to create visual assets for the story.
    Delegates to StableDiffusion tool which handles multi-provider support
    (DALL-E, Gemini, A1111/Stable Diffusion).

    Provider is auto-selected based on available API keys:
    - OPENAI_API_KEY -> DALL-E
    - GOOGLE_API_KEY -> Gemini Imagen
    - A1111_URL -> Local Stable Diffusion
    - IMAGE_PROVIDER env var can override
    """

    name: str = "generate_image"
    description: str = (
        "Generate an image from a text description. "
        "Input: prompt (detailed description of the image to generate), "
        "style (optional: art style like 'digital art', 'watercolor', 'photorealistic'), "
        "size (optional: 'small', 'medium', 'large')"
    )

    def _run(
        self,
        prompt: str,
        style: str = "digital art",
        size: str = "medium",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate an image from prompt using available provider.

        Args:
            prompt: Text description of the image to generate
            style: Art style hint (appended to prompt)
            size: Size hint ('small', 'medium', 'large')
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with status, provider, and url/path to generated image
        """
        # Delegate to StableDiffusion which handles provider selection
        from questfoundry.runtime.tools.creative_tools import StableDiffusion

        # Enhance prompt with style
        full_prompt = f"{prompt}, {style}" if style else prompt

        # Map size to dimensions for providers that support it
        size_map = {
            "small": {"width": 512, "height": 512},
            "medium": {"width": 768, "height": 768},
            "large": {"width": 1024, "height": 1024},
        }
        dimensions = size_map.get(size, size_map["medium"])

        sd_tool = StableDiffusion()
        result = sd_tool._run(prompt=full_prompt, **dimensions, **kwargs)

        # Standardize response format
        if result.get("status") == "mock":
            logger.warning(
                "No image provider configured. Set OPENAI_API_KEY, "
                "GOOGLE_API_KEY, or A1111_URL to enable image generation."
            )

        return result


class GenerateAudio(BaseTool):
    """
    Generate audio from parameters (TTS, music, sound effects).

    The Audio Producer uses this to create audio assets.
    Supports multiple audio generation backends via API keys:
    - TTS: ELEVENLABS_API_KEY, AZURE_SPEECH_KEY, GOOGLE_APPLICATION_CREDENTIALS
    - Music/SFX: SUNO_API_KEY, ELEVENLABS_API_KEY
    """

    name: str = "generate_audio"
    description: str = (
        "Generate audio content. "
        "Input: audio_type ('tts' for text-to-speech, 'music' for background music, "
        "'sfx' for sound effects), "
        "content (text for TTS, description for music/sfx), "
        "voice (optional: voice ID for TTS), "
        "duration (optional: duration in seconds for music/sfx)"
    )

    def _run(
        self,
        audio_type: str,
        content: str,
        voice: str | None = None,
        duration: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate audio content using available provider.

        Args:
            audio_type: Type of audio ('tts', 'music', 'sfx')
            content: Text for TTS or description for music/sfx
            voice: Voice ID for TTS providers
            duration: Duration in seconds for music/sfx
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with status, provider, and path/url to generated audio
        """
        valid_types = ["tts", "music", "sfx"]
        if audio_type not in valid_types:
            return {
                "status": "error",
                "error": f"Invalid audio_type: {audio_type}. Valid types: {valid_types}",
            }

        # Check for available providers based on type
        if audio_type == "tts":
            return self._generate_tts(content, voice, **kwargs)
        else:  # music or sfx
            return self._generate_audio_content(audio_type, content, duration, **kwargs)

    def _generate_tts(
        self, text: str, voice: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate text-to-speech audio."""
        # Check providers in priority order
        if os.getenv("ELEVENLABS_API_KEY"):
            return self._run_elevenlabs_tts(text, voice, **kwargs)
        elif os.getenv("AZURE_SPEECH_KEY"):
            return self._run_azure_tts(text, voice, **kwargs)
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return self._run_google_tts(text, voice, **kwargs)

        # No provider available
        logger.warning(
            "No TTS provider configured. Set ELEVENLABS_API_KEY, "
            "AZURE_SPEECH_KEY, or GOOGLE_APPLICATION_CREDENTIALS to enable TTS."
        )
        return {
            "status": "mock",
            "provider": "none",
            "audio_type": "tts",
            "message": "No TTS provider configured; returning mock",
            "content_preview": text[:100] if len(text) > 100 else text,
        }

    def _generate_audio_content(
        self, audio_type: str, description: str, duration: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate music or sound effects."""
        # Check providers in priority order
        if os.getenv("SUNO_API_KEY"):
            return self._run_suno(audio_type, description, duration, **kwargs)
        elif os.getenv("ELEVENLABS_API_KEY") and audio_type == "sfx":
            return self._run_elevenlabs_sfx(description, duration, **kwargs)

        # No provider available
        logger.warning(
            f"No {audio_type} provider configured. Set SUNO_API_KEY for music/sfx generation."
        )
        return {
            "status": "mock",
            "provider": "none",
            "audio_type": audio_type,
            "message": f"No {audio_type} provider configured; returning mock",
            "description_preview": description[:100] if len(description) > 100 else description,
            "duration_requested": duration,
        }

    def _run_elevenlabs_tts(
        self, text: str, voice: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate TTS using ElevenLabs API."""
        try:
            import httpx
            import tempfile
            from pathlib import Path

            api_key = os.getenv("ELEVENLABS_API_KEY")
            voice_id = voice or "21m00Tcm4TlvDq8ikWAM"  # Default voice (Rachel)

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "Accept": "audio/mpeg",
                        "xi-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": kwargs.get("model_id", "eleven_turbo_v2"),
                        "voice_settings": {
                            "stability": kwargs.get("stability", 0.75),
                            "similarity_boost": kwargs.get("similarity_boost", 0.75),
                        },
                    },
                )
            resp.raise_for_status()

            # Save audio to temp file
            fd, path = tempfile.mkstemp(suffix=".mp3", prefix="qf_tts_")
            with os.fdopen(fd, "wb") as f:
                f.write(resp.content)

            return {
                "status": "success",
                "provider": "elevenlabs",
                "audio_type": "tts",
                "path": str(Path(path)),
                "voice_id": voice_id,
            }
        except Exception as e:
            logger.warning(f"ElevenLabs TTS failed: {e}")
            return {
                "status": "error",
                "provider": "elevenlabs",
                "audio_type": "tts",
                "error": str(e),
            }

    def _run_azure_tts(
        self, text: str, voice: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate TTS using Azure Cognitive Services."""
        # Placeholder for Azure TTS implementation
        logger.info(f"Azure TTS not yet implemented, text: {text[:50]}...")
        return {
            "status": "not_implemented",
            "provider": "azure",
            "audio_type": "tts",
            "message": "Azure TTS integration pending",
        }

    def _run_google_tts(
        self, text: str, voice: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate TTS using Google Cloud Text-to-Speech."""
        # Placeholder for Google TTS implementation
        logger.info(f"Google TTS not yet implemented, text: {text[:50]}...")
        return {
            "status": "not_implemented",
            "provider": "google",
            "audio_type": "tts",
            "message": "Google TTS integration pending",
        }

    def _run_suno(
        self, audio_type: str, description: str, duration: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate audio using Suno AI."""
        # Placeholder for Suno implementation
        logger.info(f"Suno audio not yet implemented, desc: {description[:50]}...")
        return {
            "status": "not_implemented",
            "provider": "suno",
            "audio_type": audio_type,
            "message": "Suno AI integration pending",
        }

    def _run_elevenlabs_sfx(
        self, description: str, duration: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate sound effects using ElevenLabs."""
        try:
            import httpx
            import tempfile
            from pathlib import Path

            api_key = os.getenv("ELEVENLABS_API_KEY")

            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    "https://api.elevenlabs.io/v1/sound-generation",
                    headers={
                        "Accept": "audio/mpeg",
                        "xi-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": description,
                        "duration_seconds": duration,
                        "prompt_influence": kwargs.get("prompt_influence", 0.3),
                    },
                )
            resp.raise_for_status()

            # Save audio to temp file
            fd, path = tempfile.mkstemp(suffix=".mp3", prefix="qf_sfx_")
            with os.fdopen(fd, "wb") as f:
                f.write(resp.content)

            return {
                "status": "success",
                "provider": "elevenlabs",
                "audio_type": "sfx",
                "path": str(Path(path)),
                "description": description,
            }
        except Exception as e:
            logger.warning(f"ElevenLabs SFX failed: {e}")
            return {
                "status": "error",
                "provider": "elevenlabs",
                "audio_type": "sfx",
                "error": str(e),
            }
