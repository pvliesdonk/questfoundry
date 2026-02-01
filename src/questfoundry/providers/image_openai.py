"""OpenAI image generation provider.

Supports gpt-image-1 and dall-e-3 via the OpenAI Images API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, NoReturn

from questfoundry.observability.logging import get_logger
from questfoundry.providers.image import (
    ImageContentPolicyError,
    ImageProviderConnectionError,
    ImageProviderError,
    ImageResult,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

log = get_logger(__name__)

# Aspect ratio → OpenAI size mapping
_ASPECT_RATIO_TO_SIZE: dict[str, str] = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}

# Output format → MIME type mapping
_FORMAT_TO_CONTENT_TYPE: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class OpenAIImageProvider:
    """Image generation via OpenAI's Images API.

    Args:
        model: Model name (e.g., ``gpt-image-1``, ``dall-e-3``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        output_format: Image format (``png``, ``jpeg``, ``webp``).
    """

    def __init__(
        self,
        model: str = "gpt-image-1",
        api_key: str | None = None,
        output_format: str = "png",
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._output_format = output_format

        if not self._api_key:
            raise ImageProviderError(
                "openai",
                "API key required. Set OPENAI_API_KEY environment variable.",
            )

        if output_format not in _FORMAT_TO_CONTENT_TYPE:
            supported = ", ".join(sorted(_FORMAT_TO_CONTENT_TYPE))
            msg = f"Unsupported output_format '{output_format}'. Supported: {supported}"
            raise ImageProviderError("openai", msg)

        self._content_type = _FORMAT_TO_CONTENT_TYPE[output_format]

        # Create client once, reuse across calls
        self._client: AsyncOpenAI = self._create_client()

    def _create_client(self) -> AsyncOpenAI:
        """Create the AsyncOpenAI client (deferred import to keep openai optional)."""
        try:
            from openai import AsyncOpenAI as _AsyncOpenAI
        except ImportError as e:
            raise ImageProviderError(
                "openai", "openai package not installed. Run: uv add openai"
            ) from e
        return _AsyncOpenAI(api_key=self._api_key)

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        aspect_ratio: str = "1:1",
        quality: str = "standard",
    ) -> ImageResult:
        """Generate an image via OpenAI Images API.

        OpenAI has no separate negative_prompt parameter, so negative
        content is appended to the prompt as an avoidance clause.

        Args:
            prompt: Positive text prompt.
            negative_prompt: Appended as "Avoid: ..." to the prompt.
            aspect_ratio: Maps to OpenAI size parameter. Must be a supported
                ratio (1:1, 16:9, 9:16, 3:2, 2:3).
            quality: Passed directly to API (``standard``, ``hd``).

        Returns:
            ImageResult with generated image.

        Raises:
            ImageProviderError: On API errors or unsupported aspect_ratio.
            ImageContentPolicyError: On content policy rejection.
            ImageProviderConnectionError: On network errors.
        """
        # Build effective prompt
        effective_prompt = prompt
        if negative_prompt:
            effective_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        # Map aspect ratio to size — reject unknown ratios
        size = _ASPECT_RATIO_TO_SIZE.get(aspect_ratio)
        if size is None:
            supported = ", ".join(sorted(_ASPECT_RATIO_TO_SIZE))
            msg = f"Unsupported aspect_ratio '{aspect_ratio}'. Supported: {supported}"
            raise ImageProviderError("openai", msg)

        log.debug(
            "image_generate_start",
            model=self._model,
            size=size,
            quality=quality,
            prompt_length=len(effective_prompt),
        )

        try:
            # Build kwargs — use Any to avoid Literal type mismatches
            # from dynamic values (size comes from dict lookup, quality from caller)
            api_kwargs: dict[str, Any] = {
                "model": self._model,
                "prompt": effective_prompt,
                "n": 1,
                "size": size,
                "quality": quality,
                "response_format": "b64_json",
                "output_format": self._output_format,
            }
            response = await self._client.images.generate(**api_kwargs)
        except ImageProviderError:
            raise
        except Exception as e:
            self._handle_error(e)

        # Extract image data from response
        if not response.data:
            raise ImageProviderError("openai", "Empty response from image API")

        image_item = response.data[0]
        b64_data = image_item.b64_json
        if not b64_data:
            raise ImageProviderError("openai", "No image data in response")

        # Collect metadata
        metadata: dict[str, Any] = {
            "model": self._model,
            "size": size,
            "quality": quality,
        }
        revised_prompt = getattr(image_item, "revised_prompt", None)
        if revised_prompt:
            metadata["revised_prompt"] = revised_prompt

        log.info(
            "image_generate_complete",
            model=self._model,
            size=size,
        )

        return ImageResult.from_base64(
            b64_data,
            content_type=self._content_type,
            **metadata,
        )

    def _handle_error(self, error: Exception) -> NoReturn:
        """Convert OpenAI exceptions to ImageProvider exceptions.

        Uses OpenAI typed exceptions when available, falls back to
        string matching for untyped errors.
        """
        try:
            from openai import APIConnectionError, APIStatusError
        except ImportError:
            # If openai isn't available, fall back to string matching
            raise ImageProviderError("openai", f"Image generation failed: {error}") from error

        if isinstance(error, APIConnectionError):
            raise ImageProviderConnectionError("openai", f"Connection error: {error}") from error

        if isinstance(error, APIStatusError):
            status = error.status_code
            if status == 400 and "content_policy" in str(error).lower():
                raise ImageContentPolicyError(
                    "openai", f"Content policy rejection: {error}"
                ) from error
            raise ImageProviderError("openai", f"API error (HTTP {status}): {error}") from error

        raise ImageProviderError("openai", f"Image generation failed: {error}") from error
