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

    def test_invalid_output_format_raises(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        with pytest.raises(ImageProviderError, match="Unsupported output_format"):
            OpenAIImageProvider(api_key="sk-test", output_format="bmp")

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

        provider = OpenAIImageProvider(api_key="sk-test", model="gpt-image-1")
        provider._client = AsyncMock()
        provider._client.images.generate = AsyncMock(return_value=mock_response)

        result = await provider.generate(
            "A watercolor landscape",
            negative_prompt="photorealistic",
            aspect_ratio="16:9",
            quality="high",
        )

        assert result.image_data == b"fake_png_data"
        assert result.content_type == "image/png"
        assert result.provider_metadata["model"] == "gpt-image-1"
        assert result.provider_metadata["revised_prompt"] == "A beautiful scene"
        assert result.provider_metadata["size"] == "1536x1024"

        # Verify API was called with correct params
        call_kwargs = provider._client.images.generate.call_args
        assert "Avoid: photorealistic" in call_kwargs.kwargs["prompt"]
        assert call_kwargs.kwargs["size"] == "1536x1024"
        # gpt-image-1 uses output_format, not response_format
        assert "response_format" not in call_kwargs.kwargs
        assert call_kwargs.kwargs["output_format"] == "png"

    @pytest.mark.asyncio()
    async def test_generate_content_policy_error(self) -> None:
        from openai import APIStatusError

        from questfoundry.providers.image_openai import OpenAIImageProvider

        # Create a real APIStatusError for content policy
        mock_request = type(
            "Request",
            (),
            {
                "url": "https://api.openai.com/v1/images/generations",
                "method": "POST",
                "headers": {},
            },
        )()
        mock_response = type(
            "Response", (), {"status_code": 400, "headers": {}, "request": mock_request}
        )()
        error = APIStatusError(
            message="content_policy_violation: unsafe content",
            response=mock_response,  # type: ignore[arg-type]
            body=None,
        )

        provider = OpenAIImageProvider(api_key="sk-test")
        provider._client = AsyncMock()
        provider._client.images.generate = AsyncMock(side_effect=error)

        with pytest.raises(ImageContentPolicyError):
            await provider.generate("test prompt")

    @pytest.mark.asyncio()
    async def test_generate_connection_error(self) -> None:
        from openai import APIConnectionError

        from questfoundry.providers.image_openai import OpenAIImageProvider

        error = APIConnectionError(request=AsyncMock())

        provider = OpenAIImageProvider(api_key="sk-test")
        provider._client = AsyncMock()
        provider._client.images.generate = AsyncMock(side_effect=error)

        with pytest.raises(ImageProviderConnectionError):
            await provider.generate("test prompt")

    @pytest.mark.asyncio()
    async def test_generate_empty_response(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        mock_response = type("ImagesResponse", (), {"data": []})()

        provider = OpenAIImageProvider(api_key="sk-test")
        provider._client = AsyncMock()
        provider._client.images.generate = AsyncMock(return_value=mock_response)

        with pytest.raises(ImageProviderError, match="Empty response"):
            await provider.generate("test prompt")

    @pytest.mark.asyncio()
    async def test_invalid_aspect_ratio_raises(self) -> None:
        from questfoundry.providers.image_openai import OpenAIImageProvider

        provider = OpenAIImageProvider(api_key="sk-test")

        with pytest.raises(ImageProviderError, match="Unsupported aspect_ratio"):
            await provider.generate("test prompt", aspect_ratio="4:3")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateImageProvider:
    def test_openai_with_model(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            provider = create_image_provider("openai/gpt-image-1")

        assert isinstance(provider, ImageProvider)
        assert provider._model == "gpt-image-1"

    def test_openai_default_model(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            provider = create_image_provider("openai")

        assert provider._model == "gpt-image-1"

    def test_unknown_provider_raises(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        with pytest.raises(ImageProviderError, match="Unknown image provider"):
            create_image_provider("midjourney/v6")

    def test_aspect_ratio_mapping_gpt_image(self) -> None:
        from questfoundry.providers.image_openai import _GPT_IMAGE_SIZES

        assert _GPT_IMAGE_SIZES["1:1"] == "1024x1024"
        assert _GPT_IMAGE_SIZES["16:9"] == "1536x1024"
        assert _GPT_IMAGE_SIZES["9:16"] == "1024x1536"

    def test_aspect_ratio_mapping_dalle3(self) -> None:
        from questfoundry.providers.image_openai import _DALLE3_SIZES

        assert _DALLE3_SIZES["1:1"] == "1024x1024"
        assert _DALLE3_SIZES["16:9"] == "1792x1024"
        assert _DALLE3_SIZES["9:16"] == "1024x1792"


# ---------------------------------------------------------------------------
# OpenAI PromptDistiller
# ---------------------------------------------------------------------------


class TestOpenAIDistillPrompt:
    @pytest.mark.asyncio()
    async def test_formats_as_prose(self) -> None:
        from questfoundry.providers.image import PromptDistiller
        from questfoundry.providers.image_brief import ImageBrief
        from questfoundry.providers.image_openai import (
            OpenAIImageProvider as _OpenAI,
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            provider = _OpenAI()

        assert isinstance(provider, PromptDistiller)

        brief = ImageBrief(
            subject="Battle in the courtyard",
            composition="Wide shot, low angle",
            mood="epic",
            art_style="oil painting",
            art_medium="canvas",
            entity_fragments=["tall warrior, scarred face"],
            palette=["crimson", "gold"],
            negative="gore",
            negative_defaults="photorealism",
        )

        positive, negative = await provider.distill_prompt(brief)

        assert "Battle in the courtyard" in positive
        assert "Composition: Wide shot, low angle" in positive
        assert "Style: oil painting" in positive
        assert "tall warrior, scarred face" in positive
        assert "Palette: crimson, gold" in positive
        assert negative is not None
        assert "gore" in negative
        assert "photorealism" in negative

    @pytest.mark.asyncio()
    async def test_includes_style_overrides(self) -> None:
        from questfoundry.providers.image_brief import ImageBrief
        from questfoundry.providers.image_openai import (
            OpenAIImageProvider as _OpenAI,
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            provider = _OpenAI()

        brief = ImageBrief(
            subject="Storm",
            composition="",
            mood="",
            style_overrides="darker and grittier",
        )

        positive, _ = await provider.distill_prompt(brief)
        assert "darker and grittier" in positive
