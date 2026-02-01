"""Tests for A1111 (Automatic1111) image provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from questfoundry.providers.image import ImageProviderConnectionError, ImageProviderError
from questfoundry.providers.image_a1111 import _ASPECT_RATIO_TO_SIZE, A1111ImageProvider
from questfoundry.providers.image_brief import ImageBrief


def _fake_response(
    image_b64: str = "",
    seed: int = 42,
    status_code: int = 200,
    sd_model_name: str = "Dreamshaper",
) -> httpx.Response:
    """Build a fake httpx.Response mimicking A1111 txt2img output."""
    if not image_b64:
        image_b64 = base64.b64encode(b"fake_png_data").decode()
    body = {
        "images": [image_b64],
        "info": json.dumps({"seed": seed, "sd_model_name": sd_model_name}),
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

        payload = mock_post.call_args.kwargs["json"]
        assert payload["override_settings"]["sd_model_checkpoint"] == "dreamshaper_8"

    @pytest.mark.asyncio()
    async def test_no_checkpoint_when_unset(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test prompt")

        payload = mock_post.call_args.kwargs["json"]
        assert "override_settings" not in payload

    @pytest.mark.asyncio()
    async def test_negative_prompt_passed(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("a castle", negative_prompt="blurry, low quality")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["negative_prompt"] == "blurry, low quality"

    @pytest.mark.asyncio()
    async def test_aspect_ratio_mapping(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        for ratio, (expected_w, expected_h) in _ASPECT_RATIO_TO_SIZE.items():
            mock_post = AsyncMock(return_value=_fake_response())
            with patch.object(provider._client, "post", mock_post):
                await provider.generate("test", aspect_ratio=ratio)

            payload = mock_post.call_args.kwargs["json"]
            assert payload["width"] == expected_w, f"Wrong width for {ratio}"
            assert payload["height"] == expected_h, f"Wrong height for {ratio}"

    @pytest.mark.asyncio()
    async def test_unknown_ratio_defaults_512(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test", aspect_ratio="4:3")

        payload = mock_post.call_args.kwargs["json"]
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
            pytest.raises(ImageProviderConnectionError, match="timed out after"),
        ):
            await provider.generate("test")

    @pytest.mark.asyncio()
    async def test_pool_timeout_error(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.PoolTimeout("pool exhausted"),
            ),
            pytest.raises(ImageProviderConnectionError, match="timed out after"),
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

    @pytest.mark.asyncio()
    async def test_empty_images_array(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        empty_response = httpx.Response(200, json={"images": []})

        with (
            patch.object(
                provider._client, "post", new_callable=AsyncMock, return_value=empty_response
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

    @pytest.mark.asyncio()
    async def test_active_model_captured_when_no_override(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            return_value=_fake_response(sd_model_name="Dreamshaper"),
        ):
            result = await provider.generate("test")

        assert result.provider_metadata["model"] == "Dreamshaper"

    @pytest.mark.asyncio()
    async def test_explicit_model_not_overridden_by_response(self) -> None:
        provider = A1111ImageProvider(model="my_checkpoint", host="http://localhost:7860")

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            return_value=_fake_response(sd_model_name="Dreamshaper"),
        ):
            result = await provider.generate("test")

        assert result.provider_metadata["model"] == "my_checkpoint"

    @pytest.mark.asyncio()
    async def test_aclose(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with patch.object(provider._client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.aclose()

        mock_close.assert_called_once()


class TestA1111DistillPrompt:
    """Tests for PromptDistiller implementation."""

    def _make_brief(self, **overrides: object) -> ImageBrief:
        defaults = {
            "subject": "Battle scene in courtyard",
            "composition": "Wide shot, low angle, golden hour lighting",
            "mood": "epic and foreboding",
            "art_style": "watercolor",
            "art_medium": "traditional paper",
            "entity_fragments": ["tall warrior, scarred face"],
            "palette": ["crimson", "gold"],
            "negative": "modern elements",
            "negative_defaults": "photorealism",
        }
        defaults.update(overrides)
        return ImageBrief(**defaults)

    def test_conforms_to_prompt_distiller(self) -> None:
        from questfoundry.providers.image import PromptDistiller

        provider = A1111ImageProvider(host="http://localhost:7860")
        assert isinstance(provider, PromptDistiller)

    @pytest.mark.asyncio()
    async def test_rule_based_tag_order(self) -> None:
        """Style/medium should come before subject in SD prompts."""
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief()
        positive, negative = await provider.distill_prompt(brief)

        # Style anchors should appear before subject
        style_pos = positive.find("watercolor")
        subject_pos = positive.find("Battle scene")
        assert style_pos < subject_pos, "Style should precede subject for CLIP priority"

        assert negative is not None
        assert "modern elements" in negative
        assert "photorealism" in negative

    @pytest.mark.asyncio()
    async def test_rule_based_truncation(self) -> None:
        """Prompts exceeding ~60 words should be truncated."""
        long_composition = " ".join(f"word{i}" for i in range(80))
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief(composition=long_composition)
        positive, _ = await provider.distill_prompt(brief)

        word_count = len(positive.split())
        assert word_count <= 60

    @pytest.mark.asyncio()
    async def test_rule_based_no_art_direction(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief(art_style=None, art_medium=None, palette=[])
        positive, _ = await provider.distill_prompt(brief)

        assert "Battle scene" in positive
        assert "watercolor" not in positive

    @pytest.mark.asyncio()
    async def test_rule_based_includes_style_overrides(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief(style_overrides="darker palette")
        positive, _ = await provider.distill_prompt(brief)
        assert "darker palette" in positive

    @pytest.mark.asyncio()
    async def test_llm_distillation(self) -> None:
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "watercolor, warrior battle, courtyard, epic, golden hour"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        positive, negative = await provider.distill_prompt(brief)

        assert positive == "watercolor, warrior battle, courtyard, epic, golden hour"
        mock_llm.ainvoke.assert_called_once()

        # Negative still passed through directly
        assert negative is not None
        assert "modern elements" in negative

    @pytest.mark.asyncio()
    async def test_no_llm_falls_back_to_rule_based(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        assert provider._llm is None
        brief = self._make_brief()
        positive, _ = await provider.distill_prompt(brief)
        # Should still produce valid output via rule-based path
        assert "watercolor" in positive
        assert "Battle scene" in positive

    def test_factory_passes_llm(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        mock_llm = AsyncMock()
        provider = create_image_provider(
            "a1111/dreamshaper_8", host="http://localhost:7860", llm=mock_llm
        )
        assert isinstance(provider, A1111ImageProvider)
        assert provider._llm is mock_llm


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
