"""Tests for Amazon Bedrock text provider."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from questfoundry.providers.text.bedrock import BedrockProvider


@pytest.fixture
def mock_bedrock_config():
    """Create mock Bedrock config."""
    return {
        "aws_access_key_id": "test-access-key-id",
        "aws_secret_access_key": "test-secret-access-key",
        "aws_region": "us-east-1",
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "temperature": 0.7,
    }


@pytest.fixture
def mock_boto3_module():
    """Create mocked boto3 module."""
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    return mock_boto3


@pytest.fixture
def provider(mock_bedrock_config, mock_boto3_module):
    """Create a Bedrock provider with mocked config."""
    with patch.dict("sys.modules", {"boto3": mock_boto3_module}):
        provider_instance = BedrockProvider(mock_bedrock_config)
        yield provider_instance


def test_provider_initialization(mock_bedrock_config):
    """Test provider initializes correctly."""
    # Mock boto3 for initialization
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        provider = BedrockProvider(mock_bedrock_config)

        assert provider.aws_access_key_id == "test-access-key-id"
        assert provider.aws_secret_access_key == "test-secret-access-key"
        assert provider.aws_region == "us-east-1"
        assert provider.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 4096
        assert provider.top_p == 0.9

        # Verify boto3 client was created
        mock_boto3.client.assert_called_once_with(
            "bedrock-runtime",
            aws_access_key_id="test-access-key-id",
            aws_secret_access_key="test-secret-access-key",
            region_name="us-east-1",
        )


def test_provider_initialization_with_env_vars(monkeypatch):
    """Test provider uses AWS credentials from environment."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret-key")

    # Mock boto3 for initialization
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        provider = BedrockProvider({})

        assert provider.aws_access_key_id == "env-access-key"
        assert provider.aws_secret_access_key == "env-secret-key"


def test_provider_initialization_missing_credentials():
    """Test provider raises error when credentials are missing."""
    with pytest.raises(ValueError, match="AWS credentials required"):
        BedrockProvider({})


def test_provider_initialization_with_custom_settings():
    """Test provider initializes with custom settings."""
    config = {
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret",
        "aws_region": "us-west-2",
        "model": "anthropic.claude-3-opus-20240229-v1:0",
        "temperature": 0.5,
        "max_tokens": 2048,
        "top_p": 0.95,
    }

    # Mock boto3 for initialization
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        provider = BedrockProvider(config)

        assert provider.aws_region == "us-west-2"
        assert provider.model == "anthropic.claude-3-opus-20240229-v1:0"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 2048
        assert provider.top_p == 0.95


def test_generate_text_success_claude(provider):
    """Test successful text generation with Claude model."""
    # Mock response
    mock_response = {
        "body": MagicMock(
            read=lambda: json.dumps(
                {"content": [{"text": "This is the generated response."}]}
            ).encode()
        )
    }
    provider._client.invoke_model.return_value = mock_response

    result = provider.generate_text("Test prompt")

    assert result == "This is the generated response."
    provider._client.invoke_model.assert_called_once()


def test_generate_text_with_max_tokens(provider):
    """Test text generation with max_tokens override."""
    mock_response = {
        "body": MagicMock(
            read=lambda: json.dumps({"content": [{"text": "Response"}]}).encode()
        )
    }
    provider._client.invoke_model.return_value = mock_response

    provider.generate_text("Test", max_tokens=1024)

    # Check that request body includes max_tokens
    call_args = provider._client.invoke_model.call_args
    body = json.loads(call_args[1]["body"])
    assert body["max_tokens"] == 1024


def test_generate_text_with_temperature_override(provider):
    """Test text generation with temperature override."""
    mock_response = {
        "body": MagicMock(
            read=lambda: json.dumps({"content": [{"text": "Response"}]}).encode()
        )
    }
    provider._client.invoke_model.return_value = mock_response

    provider.generate_text("Test", temperature=0.9)

    # Check that request body uses overridden temperature
    call_args = provider._client.invoke_model.call_args
    body = json.loads(call_args[1]["body"])
    assert body["temperature"] == 0.9


def test_generate_text_generic_model():
    """Test text generation with non-Claude model."""
    config = {
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret",
        "model": "meta.llama2-13b-v1",
    }

    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_response = {
        "body": MagicMock(
            read=lambda: json.dumps({"completion": "Generic model response"}).encode()
        )
    }
    mock_client.invoke_model.return_value = mock_response

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        provider = BedrockProvider(config)
        result = provider.generate_text("Test prompt")

    assert result == "Generic model response"


def test_generate_text_unexpected_format_claude(provider):
    """Test error handling for unexpected Claude response format."""
    mock_response = {
        "body": MagicMock(read=lambda: json.dumps({"content": []}).encode())
    }
    provider._client.invoke_model.return_value = mock_response

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        provider.generate_text("Test")


def test_generate_text_unexpected_format_generic():
    """Test error handling for unexpected generic model response format."""
    config = {
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret",
        "model": "meta.llama2-13b-v1",
    }

    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_response = {
        "body": MagicMock(read=lambda: json.dumps({"unknown": "format"}).encode())
    }
    mock_client.invoke_model.return_value = mock_response

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        provider = BedrockProvider(config)
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            provider.generate_text("Test")


def test_generate_text_api_error(provider):
    """Test error handling for API failures."""
    provider._client.invoke_model.side_effect = Exception("API Error")

    with pytest.raises(RuntimeError, match="Bedrock API call failed: API Error"):
        provider.generate_text("Test")


def test_generate_text_missing_library(provider):
    """Test that missing library error is caught at initialization, not at call time."""
    # Since the library is now imported and cached during __init__,
    # this test verifies that once a provider is created, it can be used
    # even if sys.modules is later modified. The library check happens at init.
    mock_response = {
        "body": MagicMock(
            read=lambda: json.dumps({"content": [{"text": "Response"}]}).encode()
        )
    }
    provider._client.invoke_model.return_value = mock_response

    # This should work fine because _client was cached during init
    with patch.dict("sys.modules", {"boto3": None}):
        result = provider.generate_text("Test")
        assert result == "Response"


def test_generate_text_streaming_not_implemented(provider):
    """Test that streaming raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Streaming not yet implemented"):
        provider.generate_text_streaming("Test")


def test_validate_config_success(provider):
    """Test successful config validation."""
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_foundation_models.return_value = {"modelSummaries": []}

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        provider.validate_config()  # Should not raise

    mock_boto3.client.assert_called_once_with(
        "bedrock",
        aws_access_key_id="test-access-key-id",
        aws_secret_access_key="test-secret-access-key",
        region_name="us-east-1",
    )


def test_validate_config_invalid_credentials(provider):
    """Test config validation with invalid credentials."""
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.list_foundation_models.side_effect = Exception("Invalid credentials")

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        with pytest.raises(ValueError, match="Invalid Bedrock configuration"):
            provider.validate_config()


def test_validate_config_missing_library(provider):
    """Test validation error when library is not installed."""
    with patch.dict("sys.modules", {"boto3": None}):
        with pytest.raises(ValueError, match="boto3 library required"):
            provider.validate_config()


def test_repr(provider):
    """Test string representation."""
    repr_str = repr(provider)

    assert "BedrockProvider" in repr_str
    assert "model=anthropic.claude-3-5-sonnet-20241022-v2:0" in repr_str
    assert "region=us-east-1" in repr_str
    assert "has_credentials=True" in repr_str


def test_repr_without_credentials():
    """Test string representation when credentials are missing."""
    # Mock boto3 for initialization
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    # Override environment to ensure no credentials
    with patch.dict(os.environ, {}, clear=True):
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            try:
                provider = BedrockProvider(
                    {"aws_access_key_id": "", "aws_secret_access_key": ""}
                )
            except ValueError:
                # If it raises on init, create with fake credentials and clear them
                provider = BedrockProvider(
                    {"aws_access_key_id": "temp", "aws_secret_access_key": "temp"}
                )
                provider.aws_access_key_id = None
                provider.aws_secret_access_key = None

    repr_str = repr(provider)
    assert "has_credentials=False" in repr_str
