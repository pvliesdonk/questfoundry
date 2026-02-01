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
    _condense_to_tags,
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


class TestCondenseToTags:
    """Tests for prose-to-tag condensation helper."""

    def test_strips_articles_prepositions(self) -> None:
        text = "a tall warrior with a scarred face in the courtyard"
        result = _condense_to_tags(text)
        assert "a " not in f" {result} ".replace(",", " ")
        assert "tall warrior" in result
        assert "scarred face" in result
        assert "courtyard" in result

    def test_splits_on_semicolons(self) -> None:
        text = "clean vector shapes; halftone textures; prismatic lens flares"
        result = _condense_to_tags(text)
        assert "clean vector shapes" in result
        assert "halftone textures" in result
        assert "prismatic lens flares" in result

    def test_preserves_commas_as_tags(self) -> None:
        text = "chrome highlights, neon bloom, light film grain"
        result = _condense_to_tags(text)
        assert "chrome highlights" in result
        assert "neon bloom" in result
        assert "light film grain" in result

    def test_empty_input(self) -> None:
        assert _condense_to_tags("") == ""
        assert _condense_to_tags("   ") == ""

    def test_prose_heavy_example(self) -> None:
        text = (
            "clean vector shapes with confident black linework, "
            "chrome highlights, neon bloom, light film grain; "
            "occasional halftone textures, speedlines, "
            "and prismatic lens flares for mythic objects"
        )
        result = _condense_to_tags(text)
        # Filler words stripped
        assert "with" not in result.split(", ")
        assert "occasional" not in result.lower().split(", ")
        # Content preserved
        assert "clean vector shapes" in result
        assert "chrome highlights" in result


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
    async def test_sd15_subject_first(self) -> None:
        """Subject should come before style in SD 1.5 prompts."""
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief()
        positive, negative = await provider.distill_prompt(brief)

        subject_pos = positive.find("Battle scene")
        style_pos = positive.find("watercolor")
        assert subject_pos < style_pos, "Subject should precede style"

        assert negative is not None
        assert "modern elements" in negative
        assert "photorealism" in negative

    @pytest.mark.asyncio()
    async def test_sd15_entity_cap(self) -> None:
        """SD 1.5 should include at most 2 entity fragments."""
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief(
            entity_fragments=[
                "tall warrior, scarred face",
                "old mage, white beard",
                "young thief, hooded cloak",
            ]
        )
        positive, _ = await provider.distill_prompt(brief)

        # Third entity should not appear
        assert "hooded cloak" not in positive
        # First two should be present (condensed)
        assert "tall warrior" in positive
        assert "old mage" in positive

    @pytest.mark.asyncio()
    async def test_sd15_truncation(self) -> None:
        """SD 1.5 prompts should not exceed 60 words."""
        long_composition = " ".join(f"word{i}" for i in range(80))
        provider = A1111ImageProvider(host="http://localhost:7860")
        brief = self._make_brief(composition=long_composition)
        positive, _ = await provider.distill_prompt(brief)

        word_count = len(positive.split())
        assert word_count <= 60

    @pytest.mark.asyncio()
    async def test_sdxl_break_separated(self) -> None:
        """SDXL prompts should use BREAK to separate scene from style."""
        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860")
        brief = self._make_brief()
        positive, _ = await provider.distill_prompt(brief)

        assert " BREAK " in positive
        scene, style = positive.split(" BREAK ")
        # Subject in scene chunk
        assert "Battle scene" in scene
        # Style in style chunk
        assert "watercolor" in style

    @pytest.mark.asyncio()
    async def test_sdxl_entity_cap(self) -> None:
        """SDXL should include at most 3 entity fragments."""
        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860")
        brief = self._make_brief(
            entity_fragments=[
                "tall warrior, scarred face",
                "old mage, white beard",
                "young thief, hooded cloak",
                "dark knight, obsidian armor",
            ]
        )
        positive, _ = await provider.distill_prompt(brief)

        # Fourth entity should not appear
        assert "obsidian armor" not in positive
        # First three should be present
        assert "tall warrior" in positive
        assert "old mage" in positive
        assert "hooded cloak" in positive

    @pytest.mark.asyncio()
    async def test_sdxl_quality_boosters(self) -> None:
        """SDXL style chunk should include quality boosters."""
        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860")
        brief = self._make_brief()
        positive, _ = await provider.distill_prompt(brief)

        _, style = positive.split(" BREAK ")
        assert "masterpiece" in style
        assert "best quality" in style

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
    async def test_llm_system_prompt_subject_first(self) -> None:
        """LLM distiller should instruct subject-first ordering."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "warrior battle, courtyard, epic, watercolor"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        await provider.distill_prompt(brief)

        # Check the system message content
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "subject first" in system_msg.lower()
        assert "style/medium goes last" in system_msg.lower()

    @pytest.mark.asyncio()
    async def test_llm_sdxl_break_instruction(self) -> None:
        """SDXL LLM distiller should mention BREAK."""
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "scene BREAK style"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(model="sdxl_base", host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        await provider.distill_prompt(brief)

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "BREAK" in system_msg
        assert "110" in system_msg

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
    async def test_llm_distillation(self) -> None:
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "warrior battle, courtyard, epic, watercolor, golden hour"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        provider = A1111ImageProvider(host="http://localhost:7860", llm=mock_llm)
        brief = self._make_brief()
        positive, negative = await provider.distill_prompt(brief)

        assert positive == "warrior battle, courtyard, epic, watercolor, golden hour"
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
