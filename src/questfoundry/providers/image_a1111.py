"""Automatic1111 (Stable Diffusion WebUI) image provider.

Generates images via the A1111 REST API (``/sdapi/v1/txt2img``).
Requires ``A1111_HOST`` environment variable pointing to a running
WebUI instance (e.g., ``http://athena:7860``).

The provider spec string selects the SD checkpoint::

    a1111                   # Use whatever checkpoint is loaded
    a1111/dreamshaper_8     # Override to dreamshaper_8
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from questfoundry.providers.image import (
    ImageProviderConnectionError,
    ImageProviderError,
    ImageResult,
)

# SD-friendly pixel dimensions (multiples of 8).
_ASPECT_RATIO_TO_SIZE: dict[str, tuple[int, int]] = {
    "1:1": (512, 512),
    "16:9": (768, 432),
    "9:16": (432, 768),
    "3:2": (768, 512),
    "2:3": (512, 768),
}

_DEFAULT_TIMEOUT = 120.0  # Image generation can be slow on consumer GPUs


class A1111ImageProvider:
    """Image provider using Automatic1111 Stable Diffusion WebUI.

    Args:
        model: Optional SD checkpoint name (e.g., ``dreamshaper_8``).
            When set, the request includes ``override_settings.sd_model_checkpoint``.
        host: WebUI base URL. Falls back to ``A1111_HOST`` env var.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ) -> None:
        self._host = host or os.environ.get("A1111_HOST")
        if not self._host:
            raise ImageProviderError(
                "a1111",
                "A1111_HOST environment variable is required. "
                "Set it to the WebUI URL (e.g., http://localhost:7860).",
            )
        # Strip trailing slash for consistent URL construction
        self._host = self._host.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        aspect_ratio: str = "1:1",
        quality: str = "standard",  # noqa: ARG002
    ) -> ImageResult:
        """Generate an image via A1111 txt2img API.

        Args:
            prompt: Positive text prompt.
            negative_prompt: Negative prompt (natively supported by SD).
            aspect_ratio: Desired aspect ratio (e.g., "16:9").
            quality: Ignored — SD quality is controlled by steps/cfg.

        Returns:
            ImageResult with PNG data and provider metadata.

        Raises:
            ImageProviderConnectionError: If A1111 is unreachable.
            ImageProviderError: On API errors or unexpected responses.
        """
        width, height = _ASPECT_RATIO_TO_SIZE.get(aspect_ratio, (512, 512))

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": width,
            "height": height,
            "steps": 25,
            "cfg_scale": 7.0,
            "sampler_name": "Euler a",
        }

        if self._model:
            payload["override_settings"] = {"sd_model_checkpoint": self._model}

        url = f"{self._host}/sdapi/v1/txt2img"

        try:
            response = await self._client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise ImageProviderConnectionError(
                "a1111", f"Cannot connect to A1111 at {self._host}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise ImageProviderConnectionError(
                "a1111", f"Request to A1111 timed out ({_DEFAULT_TIMEOUT}s): {e}"
            ) from e

        if response.status_code != 200:
            body_preview = response.text[:200]
            raise ImageProviderError(
                "a1111",
                f"A1111 returned HTTP {response.status_code}: {body_preview}",
            )

        data = response.json()
        images = data.get("images")
        if not images:
            raise ImageProviderError("a1111", "A1111 response missing 'images' field")

        # Extract seed from response info if available
        seed = None
        info_str = data.get("info")
        if info_str:
            try:
                info = json.loads(info_str) if isinstance(info_str, str) else info_str
                seed = info.get("seed")
            except (json.JSONDecodeError, TypeError):
                pass

        metadata: dict[str, Any] = {
            "quality": "low",
            "model": self._model,
            "size": f"{width}x{height}",
            "steps": 25,
        }
        if seed is not None:
            metadata["seed"] = seed

        return ImageResult.from_base64(
            images[0],
            content_type="image/png",
            **metadata,
        )
