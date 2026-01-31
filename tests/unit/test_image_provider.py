"""Tests for ImageProvider protocol, ImageResult, and factory."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

from questfoundry.providers.image import (
    ImageContentPolicyError,
    ImageProvider,
    ImageProviderConnectionError,
    ImageProviderError,
    ImageResult,
)

# ---------------------------------------------------------------------------
# ImageResult
# ---------------------------------------------------------------------------


class TestImageResult:
    def test_basic_creation(self) -> None:
        result = ImageResult(image_data=b"png_bytes", content_type="image/png")
        assert result.image_data == b"png_bytes"
        assert result.content_type == "image/png"
        assert result.provider_metadata == {}

    def test_size_bytes(self) -> None:
        result = ImageResult(image_data=b"12345")
        assert result.size_bytes == 5

    def test_from_base64(self) -> None:
        original = b"test image data"
        b64 = base64.b64encode(original).decode()
        result = ImageResult.from_base64(b64, content_type="image/webp", model="test")
        assert result.image_data == original
        assert result.content_type == "image/webp"
        assert result.provider_metadata["model"] == "test"

    def test_frozen(self) -> None:
        result = ImageResult(image_data=b"data")
        with pytest.raises(AttributeError):
            result.image_data = b"other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestImageProviderProtocol:
    def test_openai_provider_conforms(self) -> None:
        """OpenAIImageProvider satisfies the ImageProvider protocol."""
        from questfoundry.providers.image_openai import OpenAIImageProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIImageProvider(model="gpt-image-1")

        assert isinstance(provider, ImageProvider)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_base_error(self) -> None:
        err = ImageProviderError("openai", "something failed")
        assert err.provider == "openai"
        assert "[openai]" in str(err)

    def test_content_policy_inherits(self) -> None:
        err = ImageContentPolicyError("openai", "rejected")
        assert isinstance(err, ImageProviderError)

    def test_connection_error_inherits(self) -> None:
        err = ImageProviderConnectionError("openai", "timeout")
        assert isinstance(err, ImageProviderError)


# ---------------------------------------------------------------------------
# OpenAIImageProvider
# ---------------------------------------------------------------------------


class TestOpenAIImageProvider:
    def test_missing_api_key_raises(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ImageProviderError, match="API key"),
        ):
            OpenAIImageProvider(api_key=None)

    def test_explicit_api_key(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        provider = OpenAIImageProvider(api_key="sk-test")
        assert provider._api_key == "sk-test"

    def test_env_api_key(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            provider = OpenAIImageProvider()

        assert provider._api_key == "sk-env"

    @pytest.mark.asyncio()
    async def test_generate_success(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        # Create mock response matching OpenAI API structure
        mock_image_item = type(
            "ImageItem",
            (),
            {
                "b64_json": base64.b64encode(b"fake_png_data").decode(),
                "revised_prompt": "A beautiful scene",
            },
        )()
        mock_response = type(
            "ImagesResponse",
            (),
            {
                "data": [mock_image_item],
            },
        )()

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        provider = OpenAIImageProvider(api_key="sk-test", model="gpt-image-1")

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await provider.generate(
                "A watercolor landscape",
                negative_prompt="photorealistic",
                aspect_ratio="16:9",
                quality="hd",
            )

        assert result.image_data == b"fake_png_data"
        assert result.content_type == "image/png"
        assert result.provider_metadata["model"] == "gpt-image-1"
        assert result.provider_metadata["revised_prompt"] == "A beautiful scene"
        assert result.provider_metadata["size"] == "1792x1024"

        # Verify API was called with correct params
        call_kwargs = mock_client.images.generate.call_args
        assert "Avoid: photorealistic" in call_kwargs.kwargs["prompt"]
        assert call_kwargs.kwargs["size"] == "1792x1024"

    @pytest.mark.asyncio()
    async def test_generate_content_policy_error(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(
            side_effect=Exception("content_policy_violation: unsafe content")
        )

        provider = OpenAIImageProvider(api_key="sk-test")

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ImageContentPolicyError),
        ):
            await provider.generate("test prompt")

    @pytest.mark.asyncio()
    async def test_generate_connection_error(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(side_effect=Exception("Connection timeout"))

        provider = OpenAIImageProvider(api_key="sk-test")

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ImageProviderConnectionError),
        ):
            await provider.generate("test prompt")

    @pytest.mark.asyncio()
    async def test_generate_empty_response(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        mock_response = type("ImagesResponse", (), {"data": []})()
        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        provider = OpenAIImageProvider(api_key="sk-test")

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ImageProviderError, match="Empty response"),
        ):
            await provider.generate("test prompt")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateImageProvider:
    def test_openai_with_model(self) -> None:
        from questfoundry.providers.image_openai import create_image_provider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            provider = create_image_provider("openai/gpt-image-1")

        assert isinstance(provider, ImageProvider)
        assert provider._model == "gpt-image-1"

    def test_openai_default_model(self) -> None:
        from questfoundry.providers.image_openai import create_image_provider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            provider = create_image_provider("openai")

        assert provider._model == "gpt-image-1"

    def test_unknown_provider_raises(self) -> None:
        from questfoundry.providers.image_openai import create_image_provider

        with pytest.raises(ImageProviderError, match="Unknown image provider"):
            create_image_provider("midjourney/v6")

    def test_aspect_ratio_mapping(self) -> None:
        from questfoundry.providers.image_openai import _ASPECT_RATIO_TO_SIZE

        assert _ASPECT_RATIO_TO_SIZE["1:1"] == "1024x1024"
        assert _ASPECT_RATIO_TO_SIZE["16:9"] == "1792x1024"
        assert _ASPECT_RATIO_TO_SIZE["9:16"] == "1024x1792"
