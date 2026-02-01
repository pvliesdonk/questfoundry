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
    *,
    llm: Any | None = None,
    **kwargs: Any,
) -> ImageProvider:
    """Create an image provider from a spec string.

    Args:
        provider_spec: Format ``provider/model`` (e.g., ``openai/gpt-image-1``).
            If no model is specified, a provider-specific default is used.
        llm: Optional LLM model for providers that do prompt distillation
            (currently A1111 only).
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

    if provider_lower == "placeholder":
        from questfoundry.providers.image_placeholder import PlaceholderImageProvider

        return PlaceholderImageProvider(**kwargs)

    if provider_lower == "openai":
        from questfoundry.providers.image_openai import OpenAIImageProvider

        if model:
            return OpenAIImageProvider(model=model, **kwargs)
        return OpenAIImageProvider(**kwargs)

    if provider_lower == "a1111":
        from questfoundry.providers.image_a1111 import A1111ImageProvider

        return A1111ImageProvider(model=model, llm=llm, **kwargs)

    raise ImageProviderError(provider_lower, f"Unknown image provider: {provider_lower}")
