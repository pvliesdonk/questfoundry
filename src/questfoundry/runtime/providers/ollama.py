"""Ollama LLM provider using LangChain.

Provides a simple factory for creating Ollama-backed LLMs that support
tool calling via the bind_tools pattern.

Usage
-----
    from questfoundry.runtime.providers import create_ollama_llm

    llm = create_ollama_llm(model="qwen3:8b")
    # or with custom base URL
    llm = create_ollama_llm(model="llama3.1:8b", base_url="http://localhost:11434")
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Default model for QuestFoundry
# Qwen 3 8B has good tool-calling support and runs well on consumer hardware
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_BASE_URL = "http://localhost:11434"


def create_ollama_llm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an Ollama LLM instance.

    Parameters
    ----------
    model : str
        Ollama model name (e.g., "qwen3:8b", "llama3.1:8b", "mistral:7b").
    base_url : str
        Ollama server URL.
    temperature : float
        Sampling temperature (0.0 = deterministic, 1.0 = creative).
    **kwargs
        Additional arguments passed to ChatOllama.

    Returns
    -------
    BaseChatModel
        LangChain chat model with tool support.

    Raises
    ------
    ImportError
        If langchain-ollama is not installed.
    ConnectionError
        If Ollama server is not reachable.

    Examples
    --------
    Basic usage::

        llm = create_ollama_llm()
        response = await llm.ainvoke("Hello!")

    With tool binding::

        llm = create_ollama_llm(model="qwen3:8b")
        llm_with_tools = llm.bind_tools([my_tool])
        response = await llm_with_tools.ainvoke(messages)

    Notes
    -----
    Tool-calling support varies by model:
    - qwen3:8b - excellent tool support
    - llama3.1:8b - good tool support
    - mistral:7b - moderate tool support
    - smaller models may struggle with complex tool schemas
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama is required for Ollama support. "
            "Install with: pip install langchain-ollama"
        ) from e

    logger.info(f"Creating Ollama LLM: model={model}, base_url={base_url}")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )


def check_ollama_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check if Ollama server is available.

    Parameters
    ----------
    base_url : str
        Ollama server URL to check.

    Returns
    -------
    bool
        True if server is reachable, False otherwise.
    """
    import urllib.error
    import urllib.request

    try:
        # Ollama's health endpoint
        url = f"{base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, TimeoutError):
        return False


def list_ollama_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """List available models on the Ollama server.

    Parameters
    ----------
    base_url : str
        Ollama server URL.

    Returns
    -------
    list[str]
        List of available model names.
    """
    import json
    import urllib.error
    import urllib.request

    try:
        url = f"{base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to list Ollama models: {e}")
        return []
