"""OpenAI image generation provider.

Supports gpt-image-1 (and legacy dall-e-3) via the OpenAI Images API.

gpt-image-1 and dall-e-3 differ in supported parameters:

* **gpt-image-1**: ``output_format`` (no ``response_format``),
  sizes ``1024x1024 / 1536x1024 / 1024x1536 / auto``,
  quality ``low / medium / high``.
* **dall-e-3** (deprecated May 2026): ``response_format`` (``b64_json``),
  sizes ``1024x1024 / 1792x1024 / 1024x1792``,
  quality ``standard / hd``.
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

    from questfoundry.providers.image_brief import ImageBrief

log = get_logger(__name__)

# -- Model-specific size mappings -------------------------------------------

_GPT_IMAGE_SIZES: dict[str, str] = {
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "9:16": "1024x1536",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}

_DALLE3_SIZES: dict[str, str] = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    "3:2": "1792x1024",
    "2:3": "1024x1792",
}

# Kept for backward-compat with tests that import the name directly.
_ASPECT_RATIO_TO_SIZE = _GPT_IMAGE_SIZES

# Output format → MIME type mapping
_FORMAT_TO_CONTENT_TYPE: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def _is_gpt_image_model(model: str) -> bool:
    """Return True for gpt-image-* models (not dall-e)."""
    return model.startswith("gpt-image")


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
        self._is_gpt_image = _is_gpt_image_model(model)
        self._sizes = _GPT_IMAGE_SIZES if self._is_gpt_image else _DALLE3_SIZES

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

    # -- PromptDistiller implementation ------------------------------------

    async def distill_prompt(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Format a structured brief as natural-language prose for DALL-E.

        DALL-E 3 and gpt-image-1 handle prose paragraphs well, so this
        keeps the full detail rather than condensing to tags.
        """
        parts: list[str] = []

        if brief.entity_fragments:
            parts.append("; ".join(brief.entity_fragments))
        parts.append(brief.subject)
        if brief.composition:
            parts.append(f"Composition: {brief.composition}")
        if brief.mood:
            parts.append(f"Mood: {brief.mood}")
        if brief.art_style:
            parts.append(f"Style: {brief.art_style}")
        if brief.art_medium:
            parts.append(f"Medium: {brief.art_medium}")
        if brief.style_overrides:
            parts.append(brief.style_overrides)
        if brief.palette:
            parts.append(f"Palette: {', '.join(brief.palette)}")

        prompt = "\n".join(p for p in parts if p)

        negative = brief.negative
        if brief.negative_defaults:
            negative = (
                f"{negative}, {brief.negative_defaults}" if negative else brief.negative_defaults
            )
        return prompt, negative or None

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        aspect_ratio: str = "1:1",
        quality: str = "standard",  # mapped to "high" for gpt-image models
    ) -> ImageResult:
        """Generate an image via OpenAI Images API.

        OpenAI has no separate negative_prompt parameter, so negative
        content is appended to the prompt as an avoidance clause.

        Args:
            prompt: Positive text prompt.
            negative_prompt: Appended as "Avoid: ..." to the prompt.
            aspect_ratio: Maps to OpenAI size parameter. Must be a supported
                ratio (1:1, 16:9, 9:16, 3:2, 2:3).
            quality: Passed directly to API. gpt-image-1 accepts
                ``low``/``medium``/``high``; dall-e-3 accepts ``standard``/``hd``.

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

        # Map quality values between model families
        if self._is_gpt_image:
            quality = {"standard": "high", "hd": "high"}.get(quality, quality)

        # Map aspect ratio to size — reject unknown ratios
        size = self._sizes.get(aspect_ratio)
        if size is None:
            supported = ", ".join(sorted(self._sizes))
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
            api_kwargs: dict[str, Any] = {
                "model": self._model,
                "prompt": effective_prompt,
                "n": 1,
                "size": size,
                "quality": quality,
            }
            if self._is_gpt_image:
                # gpt-image-1: output_format instead of response_format
                api_kwargs["output_format"] = self._output_format
            else:
                # dall-e-3: response_format required for base64
                api_kwargs["response_format"] = "b64_json"
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
            "quality": "high",
            "api_quality": quality,
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
