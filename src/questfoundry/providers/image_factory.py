"""Image provider factory.

Creates image provider instances from spec strings (e.g., ``openai/gpt-image-1``).
Provider implementations are lazily imported to avoid pulling in optional
dependencies until actually needed.
"""

from __future__ import annotations

from typing import Any

from questfoundry.providers.image import ImageProvider, ImageProviderError


def create_image_provider(
    provider_spec: str,
    **kwargs: Any,
) -> ImageProvider:
    """Create an image provider from a spec string.

    Args:
        provider_spec: Format ``provider/model`` (e.g., ``openai/gpt-image-1``).
            If no model is specified, a provider-specific default is used.
        **kwargs: Additional provider options forwarded to the constructor.

    Returns:
        Configured image provider.

    Raises:
        ImageProviderError: If provider is unknown.
    """
    if "/" in provider_spec:
        provider, model = provider_spec.split("/", 1)
    else:
        provider = provider_spec
        model = None

    provider_lower = provider.lower()

    if provider_lower == "openai":
        from questfoundry.providers.image_openai import OpenAIImageProvider

        return OpenAIImageProvider(model=model or "gpt-image-1", **kwargs)

    raise ImageProviderError(provider_lower, f"Unknown image provider: {provider_lower}")
