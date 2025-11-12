"""Amazon Bedrock text generation provider."""

import json
import os
from typing import Any

from ..base import TextProvider


class BedrockProvider(TextProvider):
    """
    Amazon Bedrock text generation provider.

    Provides access to foundation models via AWS Bedrock, including
    Claude, Llama, Mistral, and other models.

    Configuration:
        aws_access_key_id: AWS access key (or set AWS_ACCESS_KEY_ID env var)
        aws_secret_access_key: AWS secret key (or set AWS_SECRET_ACCESS_KEY env var)
        aws_region: AWS region (default: "us-east-1")
        model: Model ID (default: "anthropic.claude-3-5-sonnet-20241022-v2:0")
        temperature: Temperature 0.0-1.0 (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4096)
        top_p: Top-p sampling (default: 0.9)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Bedrock provider.

        Args:
            config: Configuration with AWS credentials and optional settings

        Raises:
            ValueError: If AWS credentials are missing
            RuntimeError: If boto3 library not installed
        """
        super().__init__(config)

        # Get AWS credentials from config or environment
        self.aws_access_key_id = config.get("aws_access_key_id") or os.getenv(
            "AWS_ACCESS_KEY_ID"
        )
        self.aws_secret_access_key = config.get("aws_secret_access_key") or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.aws_region = config.get("aws_region", "us-east-1")

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError(
                "AWS credentials required. "
                "Set 'aws_access_key_id' and 'aws_secret_access_key' in config "
                "or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars"
            )

        # Model settings
        self.model = config.get("model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.top_p = config.get("top_p", 0.9)

        # Initialize boto3 client once during initialization
        try:
            import boto3  # type: ignore

            self._client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region,
            )
        except ImportError:
            raise RuntimeError(
                "boto3 library required for Bedrock provider. "
                "Install with: pip install boto3"
            )

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Amazon Bedrock.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Temperature (overrides config)

        Returns:
            Generated text

        Raises:
            RuntimeError: If API call fails
        """
        # Use provided model or default
        model_id = model if model is not None else self.model

        # Build request based on model type
        if "anthropic.claude" in model_id:
            # Claude models use Messages API format
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "temperature": (
                    temperature if temperature is not None else self.temperature
                ),
                "top_p": self.top_p,
            }
        else:
            # Generic format for other models
            request_body = {
                "prompt": prompt,
                "max_tokens_to_sample": (
                    max_tokens if max_tokens is not None else self.max_tokens
                ),
                "temperature": (
                    temperature if temperature is not None else self.temperature
                ),
                "top_p": self.top_p,
            }

        # Invoke model
        try:
            response = self._client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
            )

            # Parse response
            response_body = json.loads(response["body"].read())

            # Extract text based on model type
            if "anthropic.claude" in model_id:
                # Claude response format
                if "content" in response_body and len(response_body["content"]) > 0:
                    return response_body["content"][0]["text"]
                else:
                    raise RuntimeError("Unexpected response format from Bedrock API")
            else:
                # Generic response format
                if "completion" in response_body:
                    return response_body["completion"]
                else:
                    raise RuntimeError("Unexpected response format from Bedrock API")

        except RuntimeError:
            # Re-raise our own RuntimeErrors
            raise
        except Exception as e:
            # Wrap boto3/botocore exceptions (ClientError, ParamValidationError, etc.)
            raise RuntimeError(f"Bedrock API call failed: {e}") from e

    def generate_text_streaming(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate text with streaming (not implemented yet).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature

        Raises:
            NotImplementedError: Streaming not yet implemented
        """
        raise NotImplementedError("Streaming not yet implemented for Bedrock provider")

    def validate_config(self) -> None:
        """
        Validate configuration by testing AWS credentials.

        Raises:
            ValueError: If configuration is invalid
        """
        # Client already initialized in __init__, test by making a simple API call
        try:
            import boto3

            # Create a Bedrock control plane client (different from runtime)
            bedrock_client = boto3.client(
                "bedrock",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region,
            )
            # Try to list models to validate credentials
            bedrock_client.list_foundation_models(byOutputModality="TEXT")
        except ImportError:
            raise ValueError(
                "boto3 library required for Bedrock provider. "
                "Install with: pip install boto3"
            )
        except Exception as e:
            raise ValueError(f"Invalid Bedrock configuration: {e}") from e

    def __repr__(self) -> str:
        """String representation."""
        has_credentials = bool(self.aws_access_key_id and self.aws_secret_access_key)
        return (
            f"BedrockProvider(model={self.model}, region={self.aws_region}, "
            f"has_credentials={has_credentials})"
        )
