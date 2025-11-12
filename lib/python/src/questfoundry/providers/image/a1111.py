"""Automatic1111 Stable Diffusion image generation provider"""

import base64
import logging
from typing import Any

import httpx

from ..base import ImageProvider

logger = logging.getLogger(__name__)


class Automatic1111Provider(ImageProvider):
    """
    Automatic1111 Stable Diffusion WebUI provider.

    This provider interfaces with the Automatic1111 Stable Diffusion WebUI
    API for local image generation.

    Configuration:
        base_url: A1111 API URL (default: http://localhost:7860)
        model: Default model/checkpoint to use (optional)
        timeout: Request timeout in seconds (default: 300)
        steps: Number of sampling steps (default: 20)
        cfg_scale: Classifier-free guidance scale (default: 7.0)
        sampler: Sampling method (default: Euler a)

    Example config:
        providers:
          image:
            a1111:
              base_url: http://localhost:7860
              model: sd_xl_base_1.0.safetensors
              steps: 30
              cfg_scale: 8.0
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Automatic1111 provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        self.base_url = config.get("base_url", "http://localhost:7860")
        self.default_model = config.get("model")
        self.timeout = config.get("timeout", 300)
        self.steps = config.get("steps", 20)
        self.cfg_scale = config.get("cfg_scale", 7.0)
        self.sampler = config.get("sampler", "Euler a")

        # Create HTTP client (will be set in validate_config)
        self._client: httpx.Client | None = None

    def validate_config(self) -> None:
        """
        Validate A1111 configuration.

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If cannot connect to A1111 server
        """
        try:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
            )

            # Test connection by getting available models
            response = self._client.get("/sdapi/v1/sd-models")
            response.raise_for_status()

        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Automatic1111 server at {self.base_url}. "
                f"Make sure the WebUI is running with --api flag. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to validate A1111 configuration: {e}") from e

    def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate an image using Automatic1111 Stable Diffusion.

        Args:
            prompt: The text prompt describing the image
            model: Model/checkpoint to use (uses default if not specified)
            width: Image width in pixels (default: 512)
            height: Image height in pixels (default: 512)
            **kwargs: Additional A1111-specific parameters:
                     - negative_prompt: Negative prompt text
                     - steps: Number of sampling steps
                     - cfg_scale: CFG scale
                     - sampler_name: Sampler to use
                     - seed: Random seed (-1 for random)
                     - batch_size: Number of images to generate

        Returns:
            Image data as bytes (PNG format)

        Raises:
            ValueError: If client not initialized
            RuntimeError: If generation fails
        """
        if self._client is None:
            raise ValueError(
                "A1111 client not initialized. Call validate_config first."
            )

        # If model specified, switch to it
        if model or self.default_model:
            self._set_model(model or self.default_model)

        # Build request parameters
        params = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "width": width or 512,
            "height": height or 512,
            "steps": kwargs.get("steps", self.steps),
            "cfg_scale": kwargs.get("cfg_scale", self.cfg_scale),
            "sampler_name": kwargs.get("sampler_name", self.sampler),
            "seed": kwargs.get("seed", -1),
            "batch_size": kwargs.get("batch_size", 1),
            "n_iter": 1,
        }

        try:
            response = self._client.post("/sdapi/v1/txt2img", json=params)
            response.raise_for_status()

            result = response.json()

            # Extract first image
            if not result.get("images"):
                raise RuntimeError("No images in A1111 response")

            # Images are returned as base64 strings
            image_b64 = result["images"][0]
            return base64.b64decode(image_b64)

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"A1111 API returned error status {e.response.status_code}: "
                f"{e.response.text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"A1111 image generation failed: {e}") from e

    def _set_model(self, model: str | None) -> None:
        """
        Set the active model/checkpoint.

        Args:
            model: Model name or path
        """
        if not model or self._client is None:
            return

        try:
            response = self._client.post(
                "/sdapi/v1/options",
                json={"sd_model_checkpoint": model},
            )
            response.raise_for_status()
        except Exception as e:
            # Don't fail if model switch fails, just log
            # The generation will use whatever model is currently loaded
            logger.warning(
                "Failed to switch to model '%s': %s. Using currently loaded model.",
                model,
                e,
            )

    def close(self) -> None:
        """Close the A1111 client."""
        if self._client is not None:
            self._client.close()
            self._client = None
