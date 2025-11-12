"""OpenAI DALL-E image generation provider"""

from typing import Any

from ..base import ImageProvider


class DalleProvider(ImageProvider):
    """
    OpenAI DALL-E image generation provider.

    Configuration:
        api_key: OpenAI API key (required)
        model: Default model to use (default: dall-e-3)
        organization: Optional organization ID
        quality: Image quality - 'standard' or 'hd' (default: standard)
        style: Image style - 'vivid' or 'natural' (default: vivid)

    Example config:
        providers:
          image:
            dalle:
              api_key: ${OPENAI_API_KEY}
              model: dall-e-3
              quality: standard
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize DALL-E provider.

        Args:
            config: Provider configuration

        Raises:
            ImportError: If openai package not installed
        """
        super().__init__(config)

        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "DALL-E provider requires 'openai' package. "
                "Install with: pip install questfoundry-py[openai]"
            ) from e

        self.default_model = config.get("model", "dall-e-3")
        self.api_key = config.get("api_key")
        self.organization = config.get("organization")
        self.quality = config.get("quality", "standard")
        self.style = config.get("style", "vivid")

        # Create client (will be set in validate_config)
        self._client: Any = None

    def validate_config(self) -> None:
        """
        Validate DALL-E configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.api_key:
            raise ValueError("DALL-E provider requires 'api_key' in configuration")

        # Import here to avoid dependency at module level
        from openai import OpenAI

        # Create client
        self._client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
        )

    def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate an image using DALL-E.

        Args:
            prompt: The text prompt describing the image
            model: Model to use (uses default if not specified)
            width: Image width (DALL-E 3: must be 1024, 1792, or combined)
            height: Image height (DALL-E 3: must be 1024, 1792, or combined)
            **kwargs: Additional DALL-E-specific parameters
                     (quality, style, n, response_format)

        Returns:
            Image data as bytes (PNG format)

        Raises:
            ValueError: If client not initialized or parameters invalid
            RuntimeError: If API call fails

        Note:
            DALL-E 3 supports these sizes:
            - 1024x1024 (square)
            - 1792x1024 (landscape)
            - 1024x1792 (portrait)
        """
        if self._client is None:
            raise ValueError(
                "DALL-E client not initialized. Call validate_config first."
            )

        model = model or self.default_model

        # DALL-E 3 only supports specific sizes
        valid_sizes = {"1024x1024", "1792x1024", "1024x1792"}

        # Determine size from width/height
        if width and height:
            size = f"{width}x{height}"
            if size not in valid_sizes:
                raise ValueError(
                    f"Invalid size {size}. DALL-E 3 supports: "
                    f"{', '.join(sorted(valid_sizes))}"
                )
        elif width or height:
            # If only one dimension specified, use square size
            size = "1024x1024"
        else:
            # Default to square
            size = "1024x1024"

        # Build API parameters
        api_params: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": kwargs.get("quality", self.quality),
            "style": kwargs.get("style", self.style),
            "n": kwargs.get("n", 1),
            "response_format": kwargs.get("response_format", "b64_json"),
        }

        try:
            response = self._client.images.generate(**api_params)

            # Extract image data
            if api_params["response_format"] == "b64_json":
                import base64

                image_b64 = response.data[0].b64_json
                if image_b64 is None:
                    raise RuntimeError("No image data in response")
                return base64.b64decode(image_b64)
            else:
                # URL format - fetch the image
                import httpx

                image_url = response.data[0].url
                if image_url is None:
                    raise RuntimeError("No image URL in response")
                img_response = httpx.get(image_url)
                img_response.raise_for_status()
                return bytes(img_response.content)

        except Exception as e:
            raise RuntimeError(f"DALL-E image generation failed: {e}") from e

    def close(self) -> None:
        """Close the DALL-E client."""
        if self._client is not None:
            self._client.close()
            self._client = None
