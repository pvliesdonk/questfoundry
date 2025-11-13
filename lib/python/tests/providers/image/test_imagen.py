"""Tests for Google Imagen 4 image provider."""

import base64
import os
from unittest.mock import MagicMock, patch

import pytest

from questfoundry.providers.image.imagen import ImagenProvider


@pytest.fixture
def mock_imagen_config():
    """Create mock Imagen config."""
    return {
        "project_id": "test-project-123",
        "api_key": "test-google-cloud-api-key-xyz",
        "location": "us-central1",
        "model": "imagen-4.0-preview",
    }


@pytest.fixture
def mock_google_modules():
    """Create mocked Google Cloud modules that persist for the test."""
    mock_aiplatform = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.aiplatform = mock_aiplatform

    mock_vision_models = MagicMock()
    mock_image_gen_model = MagicMock()
    mock_vision_models.ImageGenerationModel = mock_image_gen_model

    mock_vertexai = MagicMock()
    mock_preview = MagicMock()
    mock_preview.vision_models = mock_vision_models
    mock_vertexai.preview = mock_preview

    return {
        "google": MagicMock(),
        "google.cloud": mock_cloud,
        "google.cloud.aiplatform": mock_aiplatform,
        "vertexai": mock_vertexai,
        "vertexai.preview": mock_preview,
        "vertexai.preview.vision_models": mock_vision_models,
    }


@pytest.fixture
def provider(mock_imagen_config, mock_google_modules):
    """Create an Imagen provider with mocked config."""
    with patch.dict("sys.modules", mock_google_modules):
        provider_instance = ImagenProvider(mock_imagen_config)
        # Keep the patch active during the test
        yield provider_instance


def test_provider_initialization(mock_imagen_config):
    """Test provider initializes correctly."""
    # Mock the Google Cloud modules for initialization
    mock_aiplatform = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.aiplatform = mock_aiplatform

    mock_vision_models = MagicMock()
    mock_image_gen_model = MagicMock()
    mock_vision_models.ImageGenerationModel = mock_image_gen_model

    mock_vertexai = MagicMock()
    mock_preview = MagicMock()
    mock_preview.vision_models = mock_vision_models
    mock_vertexai.preview = mock_preview

    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.cloud": mock_cloud,
            "google.cloud.aiplatform": mock_aiplatform,
            "vertexai": mock_vertexai,
            "vertexai.preview": mock_preview,
            "vertexai.preview.vision_models": mock_vision_models,
        },
    ):
        provider = ImagenProvider(mock_imagen_config)

        assert provider.project_id == "test-project-123"
        assert provider.api_key == "test-google-cloud-api-key-xyz"
        assert provider.location == "us-central1"
        assert provider.model == "imagen-4.0-preview"
        assert provider.aspect_ratio == "1:1"
        assert provider.safety_filter_level == "BLOCK_SOME"

        # Verify Vertex AI was initialized
        mock_aiplatform.init.assert_called_once_with(
            project="test-project-123", location="us-central1"
        )


def test_provider_initialization_with_env_vars(monkeypatch):
    """Test provider uses credentials from environment."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project-456")
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "env-api-key-abc")

    # Mock the Google Cloud modules for initialization
    mock_aiplatform = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.aiplatform = mock_aiplatform

    mock_vision_models = MagicMock()
    mock_vision_models.ImageGenerationModel = MagicMock()

    mock_vertexai = MagicMock()
    mock_preview = MagicMock()
    mock_preview.vision_models = mock_vision_models
    mock_vertexai.preview = mock_preview

    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.cloud": mock_cloud,
            "google.cloud.aiplatform": mock_aiplatform,
            "vertexai": mock_vertexai,
            "vertexai.preview": mock_preview,
            "vertexai.preview.vision_models": mock_vision_models,
        },
    ):
        provider = ImagenProvider({})

        assert provider.project_id == "env-project-456"
        assert provider.api_key == "env-api-key-abc"


def test_provider_initialization_missing_project_id():
    """Test provider raises error when project ID is missing."""
    with pytest.raises(ValueError, match="Google Cloud project ID required"):
        ImagenProvider({"api_key": "test-key"})


def test_provider_initialization_missing_api_key():
    """Test provider raises error when API key is missing."""
    with pytest.raises(ValueError, match="Google Cloud API key required"):
        ImagenProvider({"project_id": "test-project"})


def test_provider_initialization_with_custom_settings():
    """Test provider initializes with custom settings."""
    config = {
        "project_id": "custom-project",
        "api_key": "custom-key",
        "location": "europe-west1",
        "model": "imagen-3.0",
        "aspect_ratio": "16:9",
        "safety_filter_level": "BLOCK_MOST",
    }

    # Mock the Google Cloud modules for initialization
    mock_aiplatform = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.aiplatform = mock_aiplatform

    mock_vision_models = MagicMock()
    mock_vision_models.ImageGenerationModel = MagicMock()

    mock_vertexai = MagicMock()
    mock_preview = MagicMock()
    mock_preview.vision_models = mock_vision_models
    mock_vertexai.preview = mock_preview

    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.cloud": mock_cloud,
            "google.cloud.aiplatform": mock_aiplatform,
            "vertexai": mock_vertexai,
            "vertexai.preview": mock_preview,
            "vertexai.preview.vision_models": mock_vision_models,
        },
    ):
        provider = ImagenProvider(config)

        assert provider.location == "europe-west1"
        assert provider.model == "imagen-3.0"
        assert provider.aspect_ratio == "16:9"
        assert provider.safety_filter_level == "BLOCK_MOST"


def test_generate_image_success_with_bytes(provider):
    """Test successful image generation with _image_bytes attribute."""
    # Mock model and response
    mock_model = MagicMock()
    mock_image = MagicMock()
    mock_image._image_bytes = b"fake-image-data"

    mock_response = MagicMock()
    mock_response.images = [mock_image]
    mock_model.generate_images.return_value = mock_response

    # Mock the from_pretrained method on the cached ImageGenerationModel class
    provider._ImageGenerationModel.from_pretrained = MagicMock(return_value=mock_model)

    result = provider.generate_image("A beautiful sunset")

    assert result == b"fake-image-data"
    provider._ImageGenerationModel.from_pretrained.assert_called_once_with(
        "imagen-4.0-preview"
    )
    mock_model.generate_images.assert_called_once()


def test_generate_image_with_base64_string(provider):
    """Test image generation when response is base64 string."""
    # Mock model and response with base64 string
    mock_model = MagicMock()
    image_bytes = b"fake-image-data"
    base64_image = base64.b64encode(image_bytes).decode()

    mock_response = MagicMock()
    mock_response.images = [base64_image]
    mock_model.generate_images.return_value = mock_response

    provider._ImageGenerationModel.from_pretrained = MagicMock(return_value=mock_model)

    result = provider.generate_image("Test prompt")

    assert result == image_bytes


def test_generate_image_with_dimensions(provider):
    """Test image generation with width and height."""
    mock_model = MagicMock()
    mock_image = MagicMock()
    mock_image._image_bytes = b"image-data"

    mock_response = MagicMock()
    mock_response.images = [mock_image]
    mock_model.generate_images.return_value = mock_response

    provider._ImageGenerationModel.from_pretrained = MagicMock(return_value=mock_model)

    # Test 1:1 aspect ratio
    provider.generate_image("Test", width=512, height=512)
    call_kwargs = mock_model.generate_images.call_args[1]
    assert call_kwargs["aspect_ratio"] == "1:1"

    # Test 16:9 aspect ratio (wide)
    provider.generate_image("Test", width=1920, height=1080)
    call_kwargs = mock_model.generate_images.call_args[1]
    assert call_kwargs["aspect_ratio"] == "16:9"

    # Test 9:16 aspect ratio (tall)
    provider.generate_image("Test", width=1080, height=1920)
    call_kwargs = mock_model.generate_images.call_args[1]
    assert call_kwargs["aspect_ratio"] == "9:16"


def test_generate_image_no_images_returned(provider):
    """Test error handling when no images are returned."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.images = []
    mock_model.generate_images.return_value = mock_response

    provider._ImageGenerationModel.from_pretrained = MagicMock(return_value=mock_model)

    with pytest.raises(RuntimeError, match="No images returned"):
        provider.generate_image("Test")


def test_generate_image_model_load_failure(provider):
    """Test error handling when model fails to load."""
    provider._ImageGenerationModel.from_pretrained = MagicMock(
        side_effect=Exception("Model not found")
    )

    with pytest.raises(RuntimeError, match="Failed to load Imagen model"):
        provider.generate_image("Test")


def test_generate_image_api_error(provider):
    """Test error handling for API failures."""
    mock_model = MagicMock()
    mock_model.generate_images.side_effect = Exception("API Error")

    provider._ImageGenerationModel.from_pretrained = MagicMock(return_value=mock_model)

    with pytest.raises(RuntimeError, match="Imagen API call failed"):
        provider.generate_image("Test")


def test_generate_image_missing_library():
    """Test error when google-cloud-aiplatform library is not installed."""
    # Mock ImportError during initialization
    with patch.dict("sys.modules", {"google.cloud": None}):
        with pytest.raises(
            RuntimeError, match="google-cloud-aiplatform library required"
        ):
            ImagenProvider({"project_id": "test", "api_key": "test"})


def test_validate_config_success(provider):
    """Test successful config validation."""
    # With our new implementation, validate_config is essentially a no-op
    # since initialization already happened in __init__
    provider.validate_config()  # Should not raise


def test_validate_config_invalid_credentials():
    """Test config validation with invalid credentials during initialization."""
    # Mock initialization failure
    mock_aiplatform = MagicMock()
    mock_aiplatform.init.side_effect = Exception("Invalid credentials")
    mock_cloud = MagicMock()
    mock_cloud.aiplatform = mock_aiplatform

    mock_vision_models = MagicMock()
    mock_vision_models.ImageGenerationModel = MagicMock()

    mock_vertexai = MagicMock()
    mock_preview = MagicMock()
    mock_preview.vision_models = mock_vision_models
    mock_vertexai.preview = mock_preview

    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.cloud": mock_cloud,
            "google.cloud.aiplatform": mock_aiplatform,
            "vertexai": mock_vertexai,
            "vertexai.preview": mock_preview,
            "vertexai.preview.vision_models": mock_vision_models,
        },
    ):
        # The error now happens during __init__, not validate_config
        with pytest.raises(Exception, match="Invalid credentials"):
            ImagenProvider({"project_id": "test", "api_key": "test"})


def test_validate_config_missing_library():
    """Test validation error when library is not installed."""
    # The error now happens during __init__, not validate_config
    with patch.dict("sys.modules", {"google.cloud": None}):
        with pytest.raises(
            RuntimeError, match="google-cloud-aiplatform library required"
        ):
            ImagenProvider({"project_id": "test", "api_key": "test"})


def test_repr(provider):
    """Test string representation."""
    repr_str = repr(provider)

    assert "ImagenProvider" in repr_str
    assert "model=imagen-4.0-preview" in repr_str
    assert "project=test-project-123" in repr_str
    assert "location=us-central1" in repr_str
    assert "has_credentials=True" in repr_str


def test_repr_without_credentials():
    """Test string representation when credentials are missing."""
    # Mock the Google Cloud modules for initialization
    mock_aiplatform = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.aiplatform = mock_aiplatform

    mock_vision_models = MagicMock()
    mock_vision_models.ImageGenerationModel = MagicMock()

    mock_vertexai = MagicMock()
    mock_preview = MagicMock()
    mock_preview.vision_models = mock_vision_models
    mock_vertexai.preview = mock_preview

    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.cloud": mock_cloud,
                "google.cloud.aiplatform": mock_aiplatform,
                "vertexai": mock_vertexai,
                "vertexai.preview": mock_preview,
                "vertexai.preview.vision_models": mock_vision_models,
            },
        ):
            try:
                provider = ImagenProvider({"project_id": "", "api_key": ""})
            except ValueError:
                # If it raises on init, create with fake credentials and clear them
                provider = ImagenProvider({"project_id": "temp", "api_key": "temp"})
                provider.project_id = None
                provider.api_key = None

    repr_str = repr(provider)
    assert "has_credentials=False" in repr_str
