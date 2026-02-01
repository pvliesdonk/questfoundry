"""Tests for A1111 (Automatic1111) image provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from questfoundry.providers.image import ImageProviderConnectionError, ImageProviderError
from questfoundry.providers.image_a1111 import _ASPECT_RATIO_TO_SIZE, A1111ImageProvider


def _fake_response(
    image_b64: str = "",
    seed: int = 42,
    status_code: int = 200,
) -> httpx.Response:
    """Build a fake httpx.Response mimicking A1111 txt2img output."""
    if not image_b64:
        image_b64 = base64.b64encode(b"fake_png_data").decode()
    body = {
        "images": [image_b64],
        "info": json.dumps({"seed": seed}),
    }
    return httpx.Response(status_code, json=body)


class TestA1111Provider:
    @pytest.mark.asyncio()
    async def test_generate_returns_image(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_response = _fake_response()

        with patch.object(
            provider._client, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await provider.generate("a cat sitting on a roof")

        assert result.image_data == b"fake_png_data"
        assert result.content_type == "image/png"
        assert result.provider_metadata["quality"] == "low"

    @pytest.mark.asyncio()
    async def test_checkpoint_override(self) -> None:
        provider = A1111ImageProvider(model="dreamshaper_8", host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test prompt")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["override_settings"]["sd_model_checkpoint"] == "dreamshaper_8"

    @pytest.mark.asyncio()
    async def test_no_checkpoint_when_unset(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test prompt")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "override_settings" not in payload

    @pytest.mark.asyncio()
    async def test_negative_prompt_passed(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("a castle", negative_prompt="blurry, low quality")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["negative_prompt"] == "blurry, low quality"

    @pytest.mark.asyncio()
    async def test_aspect_ratio_mapping(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        for ratio, (expected_w, expected_h) in _ASPECT_RATIO_TO_SIZE.items():
            mock_post = AsyncMock(return_value=_fake_response())
            with patch.object(provider._client, "post", mock_post):
                await provider.generate("test", aspect_ratio=ratio)

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["width"] == expected_w, f"Wrong width for {ratio}"
            assert payload["height"] == expected_h, f"Wrong height for {ratio}"

    @pytest.mark.asyncio()
    async def test_unknown_ratio_defaults_512(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test", aspect_ratio="4:3")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["width"] == 512
        assert payload["height"] == 512

    @pytest.mark.asyncio()
    async def test_connection_error(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("refused"),
            ),
            pytest.raises(ImageProviderConnectionError, match="Cannot connect"),
        ):
            await provider.generate("test")

    @pytest.mark.asyncio()
    async def test_timeout_error(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ReadTimeout("timeout"),
            ),
            pytest.raises(ImageProviderConnectionError, match="timed out"),
        ):
            await provider.generate("test")

    @pytest.mark.asyncio()
    async def test_http_error(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        error_response = httpx.Response(500, text="Internal Server Error")

        with (
            patch.object(
                provider._client, "post", new_callable=AsyncMock, return_value=error_response
            ),
            pytest.raises(ImageProviderError, match="HTTP 500"),
        ):
            await provider.generate("test")

    @pytest.mark.asyncio()
    async def test_missing_images_field(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        bad_response = httpx.Response(200, json={"parameters": {}})

        with (
            patch.object(
                provider._client, "post", new_callable=AsyncMock, return_value=bad_response
            ),
            pytest.raises(ImageProviderError, match="missing 'images'"),
        ):
            await provider.generate("test")

    def test_missing_host_raises(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ImageProviderError, match="A1111_HOST"),
        ):
            A1111ImageProvider()

    @pytest.mark.asyncio()
    async def test_quality_metadata(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with patch.object(
            provider._client, "post", new_callable=AsyncMock, return_value=_fake_response()
        ):
            result = await provider.generate("test")

        assert result.provider_metadata["quality"] == "low"

    @pytest.mark.asyncio()
    async def test_seed_in_metadata(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            return_value=_fake_response(seed=12345),
        ):
            result = await provider.generate("test")

        assert result.provider_metadata["seed"] == 12345


class TestA1111FactoryRouting:
    def test_factory_creates_a1111_provider(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        provider = create_image_provider("a1111/dreamshaper_8", host="http://localhost:7860")
        assert isinstance(provider, A1111ImageProvider)
        assert provider._model == "dreamshaper_8"

    def test_factory_a1111_without_model(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        provider = create_image_provider("a1111", host="http://localhost:7860")
        assert isinstance(provider, A1111ImageProvider)
        assert provider._model is None
