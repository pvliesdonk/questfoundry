"""Integration test configuration and fixtures.

Provides fixtures for running integration tests against real LLM providers.
Tests are automatically skipped if the required provider is not configured.
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003 - Used at runtime in integration_project fixture
from typing import TYPE_CHECKING

import pytest
from dotenv import load_dotenv

# Load .env file at import time so provider availability checks work
load_dotenv()

if TYPE_CHECKING:
    from collections.abc import Generator

    from langchain_core.language_models import BaseChatModel


def _ollama_available() -> bool:
    """Check if Ollama is configured and reachable."""
    host = os.getenv("OLLAMA_HOST")
    if not host:
        return False

    try:
        import httpx

        response = httpx.get(f"{host}/api/tags", timeout=5.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, OSError):
        return False


def _openai_available() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


def _anthropic_available() -> bool:
    """Check if Anthropic API key is configured."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


# Skip markers
requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="OLLAMA_HOST not set or Ollama not reachable",
)

requires_openai = pytest.mark.skipif(
    not _openai_available(),
    reason="OPENAI_API_KEY not set",
)

requires_anthropic = pytest.mark.skipif(
    not _anthropic_available(),
    reason="ANTHROPIC_API_KEY not set",
)

requires_any_provider = pytest.mark.skipif(
    not (_ollama_available() or _openai_available() or _anthropic_available()),
    reason="No LLM provider configured (need OLLAMA_HOST, OPENAI_API_KEY, or ANTHROPIC_API_KEY)",
)


@pytest.fixture
def ollama_model() -> Generator[BaseChatModel, None, None]:
    """Create an Ollama chat model for integration tests.

    Skipped if OLLAMA_HOST is not configured.
    Uses qwen3:8b as the default model for consistency.
    """
    if not _ollama_available():
        pytest.skip("OLLAMA_HOST not set or Ollama not reachable")

    from questfoundry.providers.factory import create_chat_model

    model = create_chat_model("ollama", "qwen3:8b")
    yield model


@pytest.fixture
def openai_model() -> Generator[BaseChatModel, None, None]:
    """Create an OpenAI chat model for integration tests.

    Skipped if OPENAI_API_KEY is not configured.
    Uses gpt-4o-mini for cost efficiency.
    """
    if not _openai_available():
        pytest.skip("OPENAI_API_KEY not set")

    from questfoundry.providers.factory import create_chat_model

    model = create_chat_model("openai", "gpt-4o-mini")
    yield model


@pytest.fixture
def anthropic_model() -> Generator[BaseChatModel, None, None]:
    """Create an Anthropic chat model for integration tests.

    Skipped if ANTHROPIC_API_KEY is not configured.
    Uses claude-3-haiku for cost efficiency.
    """
    if not _anthropic_available():
        pytest.skip("ANTHROPIC_API_KEY not set")

    from questfoundry.providers.factory import create_chat_model

    model = create_chat_model("anthropic", "claude-3-haiku-20240307")
    yield model


@pytest.fixture(params=["ollama", "openai"])
def any_model(request: pytest.FixtureRequest) -> Generator[BaseChatModel, None, None]:
    """Parametrized fixture that provides both Ollama and OpenAI models.

    Tests using this fixture run twice - once per provider (if available).
    Skips providers that are not configured.
    """
    provider = request.param

    if provider == "ollama":
        if not _ollama_available():
            pytest.skip("OLLAMA_HOST not set or Ollama not reachable")
        from questfoundry.providers.factory import create_chat_model

        yield create_chat_model("ollama", "qwen3:8b")
    elif provider == "openai":
        if not _openai_available():
            pytest.skip("OPENAI_API_KEY not set")
        from questfoundry.providers.factory import create_chat_model

        yield create_chat_model("openai", "gpt-4o-mini")


@pytest.fixture
def integration_project(tmp_path: Path) -> Path:
    """Create a temporary project directory for integration tests.

    Creates the standard project structure:
    - project.yaml
    - artifacts/
    - logs/
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create artifacts directory
    (project_dir / "artifacts").mkdir()

    # Create logs directory
    (project_dir / "logs").mkdir()

    # Create minimal project.yaml
    project_yaml = project_dir / "project.yaml"
    project_yaml.write_text("""# Test project configuration
name: integration_test
provider:
  name: ollama
  model: qwen3:8b
stages:
  - dream
""")

    return project_dir


@pytest.fixture
def simple_story_prompt() -> str:
    """Provide a simple, consistent story prompt for integration tests.

    Using a specific prompt ensures reproducible tests while being
    simple enough for quick LLM responses.
    """
    return "A cozy mystery story set in a small coastal town bookshop."
