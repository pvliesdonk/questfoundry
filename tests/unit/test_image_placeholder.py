"""Tests for PlaceholderImageProvider."""

from __future__ import annotations

import pytest

from questfoundry.providers.image import ImageProvider
from questfoundry.providers.image_placeholder import PlaceholderImageProvider, _make_png


class TestMakePng:
    """Test the pure-Python PNG generator."""

    def test_produces_valid_png_signature(self) -> None:
        data = _make_png(2, 2, 128, 128, 128)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_different_sizes_produce_different_lengths(self) -> None:
        small = _make_png(1, 1, 0, 0, 0)
        large = _make_png(100, 100, 0, 0, 0)
        assert len(large) > len(small)

    def test_different_colors_produce_different_data(self) -> None:
        red = _make_png(4, 4, 255, 0, 0)
        blue = _make_png(4, 4, 0, 0, 255)
        assert red != blue


class TestPlaceholderImageProvider:
    """Test the placeholder provider."""

    def test_conforms_to_protocol(self) -> None:
        provider = PlaceholderImageProvider()
        assert isinstance(provider, ImageProvider)

    @pytest.mark.asyncio()
    async def test_generate_returns_image_result(self) -> None:
        provider = PlaceholderImageProvider()
        result = await provider.generate("test prompt")

        assert result.image_data[:8] == b"\x89PNG\r\n\x1a\n"
        assert result.content_type == "image/png"
        assert result.size_bytes > 0

    @pytest.mark.asyncio()
    async def test_quality_metadata(self) -> None:
        provider = PlaceholderImageProvider()
        result = await provider.generate("test prompt")

        assert result.provider_metadata["quality"] == "placeholder"

    @pytest.mark.asyncio()
    async def test_size_metadata_default_aspect(self) -> None:
        provider = PlaceholderImageProvider()
        result = await provider.generate("test", aspect_ratio="1:1")

        assert result.provider_metadata["size"] == "256x256"

    @pytest.mark.asyncio()
    async def test_16_9_aspect_ratio(self) -> None:
        provider = PlaceholderImageProvider()
        result = await provider.generate("test", aspect_ratio="16:9")

        assert result.provider_metadata["size"] == "640x360"

    @pytest.mark.asyncio()
    async def test_unknown_aspect_ratio_falls_back(self) -> None:
        """Unknown aspect ratios fall back to 1:1 instead of erroring."""
        provider = PlaceholderImageProvider()
        result = await provider.generate("test", aspect_ratio="4:3")

        assert result.provider_metadata["size"] == "256x256"

    @pytest.mark.asyncio()
    async def test_deterministic_color(self) -> None:
        """Same prompt always produces the same color."""
        provider = PlaceholderImageProvider()
        r1 = await provider.generate("hello world")
        r2 = await provider.generate("hello world")

        assert r1.provider_metadata["color"] == r2.provider_metadata["color"]

    @pytest.mark.asyncio()
    async def test_different_prompts_may_differ(self) -> None:
        """Different prompts can produce different colors."""
        provider = PlaceholderImageProvider()
        r1 = await provider.generate("alpha")
        r2 = await provider.generate("beta")

        # Not guaranteed different, but with 6 palette colors
        # these specific prompts happen to differ
        assert r1.provider_metadata["color"] != r2.provider_metadata["color"]

    @pytest.mark.asyncio()
    async def test_prompt_preview_in_metadata(self) -> None:
        provider = PlaceholderImageProvider()
        result = await provider.generate("A long prompt about a watercolor landscape")

        assert "watercolor" in result.provider_metadata["prompt_preview"]


class TestFactoryPlaceholder:
    """Test factory routing for placeholder provider."""

    def test_factory_creates_placeholder(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        provider = create_image_provider("placeholder")
        assert isinstance(provider, PlaceholderImageProvider)

    def test_factory_ignores_model_for_placeholder(self) -> None:
        from questfoundry.providers.image_factory import create_image_provider

        # placeholder/anything should still work
        provider = create_image_provider("placeholder/ignored")
        assert isinstance(provider, PlaceholderImageProvider)
