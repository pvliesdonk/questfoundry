"""OpenAI image generation provider.

Supports gpt-image-1 and dall-e-3 via the OpenAI Images API.
"""

from __future__ import annotations

import os
from typing import Any

from questfoundry.observability.logging import get_logger
from questfoundry.providers.image import (
    ImageContentPolicyError,
    ImageProviderConnectionError,
    ImageProviderError,
    ImageResult,
)

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
            aspect_ratio: Maps to OpenAI size parameter.
            quality: Passed directly to API (``standard``, ``hd``).

        Returns:
            ImageResult with generated image.

        Raises:
            ImageProviderError: On API errors.
            ImageContentPolicyError: On content policy rejection.
            ImageProviderConnectionError: On network errors.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImageProviderError(
                "openai", "openai package not installed. Run: uv add openai"
            ) from e

        # Build effective prompt
        effective_prompt = prompt
        if negative_prompt:
            effective_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        # Map aspect ratio to size
        size = _ASPECT_RATIO_TO_SIZE.get(aspect_ratio, "1024x1024")

        log.debug(
            "image_generate_start",
            model=self._model,
            size=size,
            quality=quality,
            prompt_length=len(effective_prompt),
        )

        try:
            client = AsyncOpenAI(api_key=self._api_key)

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
            response = await client.images.generate(**api_kwargs)
        except Exception as e:
            return self._handle_error(e)

        # Extract image data from response
        if not response.data:
            raise ImageProviderError("openai", "Empty response from image API")

        image_item = response.data[0]
        b64_data = image_item.b64_json
        if not b64_data:
            raise ImageProviderError("openai", "No image data in response")

        content_type = _FORMAT_TO_CONTENT_TYPE.get(self._output_format, "image/png")

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
            content_type=content_type,
            **metadata,
        )

    def _handle_error(self, error: Exception) -> ImageResult:
        """Convert OpenAI exceptions to ImageProvider exceptions.

        This method always raises — the return type is for type checker
        compatibility only.
        """
        error_str = str(error).lower()

        if "content_policy" in error_str or "safety" in error_str:
            raise ImageContentPolicyError("openai", f"Content policy rejection: {error}") from error

        if "connection" in error_str or "timeout" in error_str:
            raise ImageProviderConnectionError("openai", f"Connection error: {error}") from error

        raise ImageProviderError("openai", f"Image generation failed: {error}") from error


def create_image_provider(
    provider_spec: str,
    **kwargs: Any,
) -> OpenAIImageProvider:
    """Factory: create an image provider from a spec string.

    Args:
        provider_spec: Format ``provider/model`` (e.g., ``openai/gpt-image-1``).
        **kwargs: Additional provider options.

    Returns:
        Configured image provider.

    Raises:
        ImageProviderError: If provider is unknown.
    """
    if "/" in provider_spec:
        provider, model = provider_spec.split("/", 1)
    else:
        provider = provider_spec
        model = "gpt-image-1"

    provider_lower = provider.lower()

    if provider_lower == "openai":
        return OpenAIImageProvider(model=model, **kwargs)

    raise ImageProviderError(provider_lower, f"Unknown image provider: {provider_lower}")
