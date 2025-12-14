"""Google AI Studio (Gemini) LLM provider using LangChain.

Provides factory functions for creating Gemini-backed LLMs that support
tool calling via the bind_tools pattern.

Usage
-----
::

    from questfoundry.runtime.providers.google import create_google_llm

    # Basic usage (requires GOOGLE_API_KEY env var)
    llm = create_google_llm()

    # Specify model
    llm = create_google_llm(model="gemini-2.5-pro-preview-05-06")

    # With thinking mode
    llm = create_google_llm(model="gemini-2.5-flash", thinking_budget=1024)

Setup
-----
Set your API key::

    export GOOGLE_API_KEY="your-api-key-here"

Or use a .env file::

    GOOGLE_API_KEY=your-api-key-here
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Default model - Gemini 2.5 Pro for best quality
DEFAULT_MODEL = "gemini-2.5-pro-preview-05-06"


def create_google_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    api_key: str | None = None,
    thinking_budget: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a Google AI Studio (Gemini) LLM instance.

    Parameters
    ----------
    model : str
        Gemini model name. Options include:
        - "gemini-2.5-pro-preview-05-06" (best quality, default)
        - "gemini-2.5-flash-preview-05-20" (faster, cheaper)
        - "gemini-1.5-pro" (stable, well-tested)
        - "gemini-1.5-flash" (fast, good for simple tasks)
    temperature : float
        Sampling temperature (0.0 = deterministic, 1.0+ = creative).
    api_key : str | None
        Google API key. If None, reads from GOOGLE_API_KEY env var.
    thinking_budget : int | None
        Thinking budget in tokens for reasoning models:
        - None: Use model default
        - 0: Disable thinking
        - -1: Dynamic (model decides)
        - >0: Max thinking tokens
    **kwargs
        Additional arguments passed to ChatGoogleGenerativeAI.

    Returns
    -------
    BaseChatModel
        LangChain chat model with tool support.

    Raises
    ------
    ImportError
        If langchain-google-genai is not installed.
    ValueError
        If no API key is provided or found in environment.

    Examples
    --------
    Basic usage::

        llm = create_google_llm()
        response = await llm.ainvoke("Hello!")

    With tool binding::

        llm = create_google_llm(model="gemini-2.5-flash")
        llm_with_tools = llm.bind_tools([my_tool])
        response = await llm_with_tools.ainvoke(messages)

    With thinking mode (for reasoning tasks)::

        llm = create_google_llm(
            model="gemini-2.5-flash",
            thinking_budget=1024
        )
        response = await llm.ainvoke("Solve this step by step...")

    Notes
    -----
    Tool-calling support:
    - gemini-2.5-pro: Excellent tool support, best quality
    - gemini-2.5-flash: Good tool support, faster
    - gemini-1.5-pro/flash: Good tool support, stable
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(
            "langchain-google-genai is required for Google AI Studio support. "
            "Install with: pip install langchain-google-genai"
        ) from e

    # Get API key from parameter or environment
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Google API key required. Set GOOGLE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    logger.info(f"Creating Google AI Studio LLM: model={model}")

    # Build kwargs
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "google_api_key": resolved_api_key,
        **kwargs,
    }

    # Add thinking budget if specified
    if thinking_budget is not None:
        llm_kwargs["thinking_budget"] = thinking_budget
        logger.info(f"Thinking budget: {thinking_budget}")

    return ChatGoogleGenerativeAI(**llm_kwargs)


def check_google_available(api_key: str | None = None) -> bool:
    """Check if Google AI Studio is available.

    Parameters
    ----------
    api_key : str | None
        Google API key to check. If None, reads from environment.

    Returns
    -------
    bool
        True if API key is set and valid format, False otherwise.

    Notes
    -----
    This only checks if an API key is configured, not if it's valid.
    Use list_google_models() for a full connectivity check.
    """
    resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        return False

    # Basic format check (Google API keys typically start with "AI")
    return len(resolved_key) > 10


def list_google_models() -> list[str]:
    """List available Gemini models.

    Returns
    -------
    list[str]
        List of recommended model names.

    Notes
    -----
    Unlike Ollama, Google doesn't have a dynamic model list API.
    This returns the recommended models for QuestFoundry.
    """
    return [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-05-20",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]


def get_google_model_info(model: str) -> dict[str, Any]:
    """Get information about a Gemini model.

    Parameters
    ----------
    model : str
        Model name.

    Returns
    -------
    dict[str, Any]
        Model information including context window, tier, etc.
    """
    model_info: dict[str, dict[str, Any]] = {
        "gemini-2.5-pro-preview-05-06": {
            "name": "Gemini 2.5 Pro",
            "context_window": 1000000,
            "tier": "pro",
            "thinking": True,
            "multimodal": True,
        },
        "gemini-2.5-flash-preview-05-20": {
            "name": "Gemini 2.5 Flash",
            "context_window": 1000000,
            "tier": "flash",
            "thinking": True,
            "multimodal": True,
        },
        "gemini-1.5-pro": {
            "name": "Gemini 1.5 Pro",
            "context_window": 2000000,
            "tier": "pro",
            "thinking": False,
            "multimodal": True,
        },
        "gemini-1.5-flash": {
            "name": "Gemini 1.5 Flash",
            "context_window": 1000000,
            "tier": "flash",
            "thinking": False,
            "multimodal": True,
        },
    }

    return model_info.get(
        model,
        {
            "name": model,
            "context_window": 128000,
            "tier": "unknown",
            "thinking": False,
            "multimodal": False,
        },
    )
