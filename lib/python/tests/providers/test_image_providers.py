"""Tests for image providers"""

import base64
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from questfoundry.providers import ImageProvider


class MockImageProvider(ImageProvider):
    """Mock image provider for testing"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.generate_called = False

    def validate_config(self) -> None:
        """Validate configuration"""
        if "required_key" in self.config and not self.config["required_key"]:
            raise ValueError("required_key is empty")

    def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate mock image"""
        self.generate_called = True
        # Return a minimal PNG header
        return b"\x89PNG\r\n\x1a\n"


def test_mock_image_provider():
    """Test mock image provider"""
    provider = MockImageProvider({"required_key": "test"})
    provider.validate_config()

    image_data = provider.generate_image("test prompt", width=512, height=512)
    assert provider.generate_called
    assert isinstance(image_data, bytes)
    assert image_data.startswith(b"\x89PNG")


@pytest.mark.skipif(True, reason="Skip DALL-E tests - requires openai package mocking")
def test_dalle_provider_requires_openai_package():
    """Test DALL-E provider import error without openai package"""
    pass


def test_dalle_provider_validates_api_key():
    """Test DALL-E provider requires API key"""
    from questfoundry.providers.image.dalle import DalleProvider

    try:
        provider = DalleProvider({})  # No API key
    except ImportError:
        pytest.skip("openai package not available")
        return

    with pytest.raises(ValueError, match="api_key"):
        provider.validate_config()


def test_dalle_provider_basic_config():
    """Test DALL-E provider basic configuration"""
    from questfoundry.providers.image.dalle import DalleProvider

    try:
        # Just test that provider can be created with config
        provider = DalleProvider(
            {"api_key": "test-key", "model": "dall-e-3", "quality": "hd"}
        )
    except ImportError:
        pytest.skip("openai package not available")
        return

    assert provider.api_key == "test-key"
    assert provider.default_model == "dall-e-3"
    assert provider.quality == "hd"


def test_a1111_provider_validates_connection():
    """Test A1111 provider connection validation"""
    with patch("questfoundry.providers.image.a1111.httpx.Client") as mock_client_cls:
        from questfoundry.providers.image.a1111 import Automatic1111Provider

        # Mock successful connection
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = Automatic1111Provider({"base_url": "http://localhost:7860"})
        provider.validate_config()  # Should not raise

        mock_client.get.assert_called_with("/sdapi/v1/sd-models")


def test_a1111_provider_generate_image():
    """Test A1111 provider image generation"""
    with patch("questfoundry.providers.image.a1111.httpx.Client") as mock_client_cls:
        from questfoundry.providers.image.a1111 import Automatic1111Provider

        # Mock HTTP client
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock validation response
        mock_models_response = Mock()
        mock_models_response.status_code = 200
        mock_client.get.return_value = mock_models_response

        # Mock generation response
        fake_image_b64 = base64.b64encode(b"fake_sd_image").decode()
        mock_gen_response = Mock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {"images": [fake_image_b64]}
        mock_client.post.return_value = mock_gen_response

        # Create provider
        provider = Automatic1111Provider({})
        provider.validate_config()

        # Generate image
        image_data = provider.generate_image(
            "a landscape",
            width=512,
            height=512,
            negative_prompt="blurry",
            steps=30,
        )

        # Verify
        assert image_data == b"fake_sd_image"
        assert mock_client.post.call_count >= 1

        # Check that txt2img was called
        calls = [
            call
            for call in mock_client.post.call_args_list
            if "/sdapi/v1/txt2img" in str(call)
        ]
        assert len(calls) == 1

        call_kwargs = calls[0][1]
        assert call_kwargs["json"]["prompt"] == "a landscape"
        assert call_kwargs["json"]["width"] == 512
        assert call_kwargs["json"]["height"] == 512
        assert call_kwargs["json"]["negative_prompt"] == "blurry"
        assert call_kwargs["json"]["steps"] == 30


def test_image_provider_context_manager():
    """Test image provider as context manager"""
    provider = MockImageProvider({"required_key": "test"})
    provider.validate_config()

    with provider as p:
        assert p is provider
        p.generate_image("test")

    # close() should have been called
    # For mock provider, this is a no-op, but interface should work


def test_dalle_provider_default_parameters():
    """Test DALL-E provider default parameters"""
    from questfoundry.providers.image.dalle import DalleProvider

    try:
        # Test with defaults
        provider = DalleProvider({"api_key": "test"})
    except ImportError:
        pytest.skip("openai package not available")
        return

    assert provider.default_model == "dall-e-3"
    assert provider.quality == "standard"
    assert provider.style == "vivid"

    # Test with custom defaults
    provider2 = DalleProvider({"api_key": "test", "quality": "hd", "style": "natural"})
    assert provider2.quality == "hd"
    assert provider2.style == "natural"
