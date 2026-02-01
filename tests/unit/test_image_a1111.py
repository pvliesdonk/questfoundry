"""Tests for A1111 (Automatic1111) image provider."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from questfoundry.providers.image import ImageProviderConnectionError, ImageProviderError
from questfoundry.providers.image_a1111 import (
    _SD15_PRESET,
    _SDXL_PRESET,
    A1111ImageProvider,
    _resolve_preset,
)
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
        assert result.provider_metadata["quality"] == "medium"

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
    async def test_aspect_ratio_mapping_sd15(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        for ratio, (expected_w, expected_h) in _SD15_PRESET.sizes.items():
            mock_post = AsyncMock(return_value=_fake_response())
            with patch.object(provider._client, "post", mock_post):
                await provider.generate("test", aspect_ratio=ratio)

            payload = mock_post.call_args.kwargs["json"]
            assert payload["width"] == expected_w, f"Wrong width for {ratio}"
            assert payload["height"] == expected_h, f"Wrong height for {ratio}"

    @pytest.mark.asyncio()
    async def test_aspect_ratio_mapping_sdxl(self) -> None:
        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860")

        for ratio, (expected_w, expected_h) in _SDXL_PRESET.sizes.items():
            mock_post = AsyncMock(return_value=_fake_response())
            with patch.object(provider._client, "post", mock_post):
                await provider.generate("test", aspect_ratio=ratio)

            payload = mock_post.call_args.kwargs["json"]
            assert payload["width"] == expected_w, f"Wrong width for {ratio}"
            assert payload["height"] == expected_h, f"Wrong height for {ratio}"

    @pytest.mark.asyncio()
    async def test_unknown_ratio_falls_back_to_1x1(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test", aspect_ratio="4:3")

        payload = mock_post.call_args.kwargs["json"]
        # Falls back to preset 1:1 size (768x768 for SD 1.5)
        assert payload["width"] == 768
        assert payload["height"] == 768

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
    async def test_quality_metadata_sd15(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")

        with patch.object(
            provider._client, "post", new_callable=AsyncMock, return_value=_fake_response()
        ):
            result = await provider.generate("test")

        assert result.provider_metadata["quality"] == "medium"

    @pytest.mark.asyncio()
    async def test_quality_metadata_sdxl(self) -> None:
        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860")

        with patch.object(
            provider._client, "post", new_callable=AsyncMock, return_value=_fake_response()
        ):
            result = await provider.generate("test")

        assert result.provider_metadata["quality"] == "high"

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


class TestA1111Presets:
    """Tests for model-aware generation presets."""

    def test_sd15_preset_default(self) -> None:
        assert _resolve_preset(None) is _SD15_PRESET

    def test_sd15_preset_dreamshaper(self) -> None:
        assert _resolve_preset("dreamshaper_8") is _SD15_PRESET

    def test_sdxl_preset_sdxl(self) -> None:
        assert _resolve_preset("sdxl_base") is _SDXL_PRESET

    def test_sdxl_preset_xl_suffix(self) -> None:
        assert _resolve_preset("realvisxl_v40") is _SDXL_PRESET

    def test_sdxl_preset_case_insensitive(self) -> None:
        assert _resolve_preset("SDXL_turbo") is _SDXL_PRESET

    @pytest.mark.asyncio()
    async def test_sdxl_sampler_in_payload(self) -> None:
        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["sampler_name"] == "DPM++ 2M"
        assert payload["scheduler"] == "karras"
        assert payload["steps"] == 35
        assert payload["cfg_scale"] == 7.5

    @pytest.mark.asyncio()
    async def test_sd15_sampler_in_payload(self) -> None:
        provider = A1111ImageProvider(host="http://localhost:7860")
        mock_post = AsyncMock(return_value=_fake_response())

        with patch.object(provider._client, "post", mock_post):
            await provider.generate("test")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["sampler_name"] == "DPM++ 2M"
        assert payload["scheduler"] == "karras"
        assert payload["steps"] == 30
        assert payload["cfg_scale"] == 7.0


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
    async def test_no_llm_raises(self) -> None:
        """distill_prompt raises when no LLM is provided."""
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief()
        with pytest.raises(ImageProviderError, match="requires an LLM"):
            await provider.distill_prompt(brief)

    @pytest.mark.asyncio()
    async def test_llm_system_prompt_tag_limit(self) -> None:
        """SD 1.5 distiller should enforce 40-tag limit."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "warrior battle, courtyard, epic, watercolor"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        await provider.distill_prompt(brief)

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "40 tags" in system_msg
        assert "HARD LIMIT" in system_msg
        assert "subject" in system_msg.lower()

    @pytest.mark.asyncio()
    async def test_llm_sdxl_break_instruction(self) -> None:
        """SDXL LLM distiller should mention BREAK and 75-tag limit."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "scene tags BREAK style tags"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        await provider.distill_prompt(brief)

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "BREAK" in system_msg
        assert "75 tags" in system_msg

    @pytest.mark.asyncio()
    async def test_llm_entity_cap(self) -> None:
        """LLM distiller should also cap entities."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "test prompt"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief(entity_fragments=["frag1", "frag2", "frag3", "frag4"])
        await provider.distill_prompt(brief)

        call_args = mock_llm.ainvoke.call_args[0][0]
        human_msg = call_args[1].content
        # SD 1.5: max 2 entities
        assert "frag3" not in human_msg
        assert "frag4" not in human_msg

    @pytest.mark.asyncio()
    async def test_llm_distillation_two_line_output(self) -> None:
        """LLM returns positive on line 1, negative on line 2."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = (
            "warrior battle, courtyard, epic, watercolor, golden hour\n"
            "photorealism, modern elements, text"
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        positive, negative = await provider.distill_prompt(brief)

        assert positive == "warrior battle, courtyard, epic, watercolor, golden hour"
        assert negative == "photorealism, modern elements, text"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio()
    async def test_llm_distillation_single_line_output(self) -> None:
        """LLM returns only positive — negative is None."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "warrior battle, courtyard, epic, watercolor"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        positive, negative = await provider.distill_prompt(brief)

        assert positive == "warrior battle, courtyard, epic, watercolor"
        assert negative is None

    @pytest.mark.asyncio()
    async def test_llm_strips_labels(self) -> None:
        """LLM output with 'Positive:' / 'Negative:' labels gets stripped."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = (
            "Positive: warrior, courtyard, watercolor\nNegative: photorealism, text"
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        positive, negative = await provider.distill_prompt(brief)

        assert positive == "warrior, courtyard, watercolor"
        assert negative == "photorealism, text"

    @pytest.mark.asyncio()
    async def test_llm_receives_negative_in_brief(self) -> None:
        """Negative prompt text is included in the brief sent to the LLM."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "tags\nbad things"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief(negative="blurry", negative_defaults="text, watermark")
        await provider.distill_prompt(brief)

        human_msg = mock_llm.ainvoke.call_args[0][0][1].content
        assert "blurry" in human_msg
        assert "text, watermark" in human_msg

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
