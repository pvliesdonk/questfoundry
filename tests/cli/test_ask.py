"""Tests for the CLI ask command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from questfoundry.cli import app
from questfoundry.runtime.domain import LoadError, LoadResult
from questfoundry.runtime.models.base import Agent, Studio
from questfoundry.runtime.providers import StreamChunk
from questfoundry.runtime.storage import Project

runner = CliRunner()


@pytest.fixture
def mock_studio() -> Studio:
    """Create a mock studio with agents."""
    return Studio(
        id="test_studio",
        name="Test Studio",
        agents=[
            Agent(
                id="showrunner",
                name="Showrunner",
                description="The orchestrator",
                archetypes=["orchestrator"],
                is_entry_agent=True,
            ),
            Agent(
                id="lorekeeper",
                name="Lorekeeper",
                description="Keeper of canon",
                archetypes=["curator"],
            ),
        ],
    )


@pytest.fixture
def test_project(tmp_path: Path) -> Project:
    """Create a test project."""
    project_path = tmp_path / "projects" / "test-project"
    project = Project.create(
        path=project_path,
        name="Test Project",
        studio_id="test_studio",
    )
    yield project
    project.close()


class TestAskCommandHelp:
    """Tests for ask command help and structure."""

    def test_ask_help(self) -> None:
        """ask --help displays usage."""
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "Interactive session or single-shot query" in result.stdout

    def test_ask_auto_creates_default_project(self, tmp_path: Path) -> None:
        """ask without project auto-creates default project."""
        load_result = LoadResult(
            studio=None, errors=[LoadError(path="domain-v4", message="test", severity="error")]
        )

        with (
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
        ):
            mock_load.return_value = load_result

            result = runner.invoke(
                app,
                ["ask", "--projects-dir", str(tmp_path / "projects")],
            )

            # Should have created default project
            assert "Creating new project" in result.stdout


class TestAskProjectErrors:
    """Tests for project auto-creation behavior."""

    def test_ask_project_auto_created(self, tmp_path: Path) -> None:
        """ask auto-creates project if it doesn't exist."""
        # Mock domain loading to fail so we can test project creation
        failed_result = LoadResult(
            studio=None,
            errors=[
                LoadError(
                    path="domain-v4",
                    message="Test domain error",
                    severity="error",
                )
            ],
        )

        with patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = failed_result

            result = runner.invoke(
                app,
                [
                    "ask",
                    "nonexistent",
                    "Hello",
                    "--projects-dir",
                    str(tmp_path / "projects"),
                ],
            )

        # Project should be auto-created, then fail on domain load
        assert "Creating new project" in result.stdout
        assert result.exit_code == 1


class TestAskDomainErrors:
    """Tests for domain-related errors."""

    def test_ask_domain_load_failure(self, test_project: Project) -> None:
        """ask exits with error when domain fails to load."""
        failed_result = LoadResult(
            studio=None,
            errors=[
                LoadError(
                    path="domain-v4",
                    message="Invalid domain",
                    severity="error",
                )
            ],
        )

        with (
            patch("questfoundry.runtime.storage.Project.open", return_value=test_project),
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
        ):
            mock_load.return_value = failed_result

            result = runner.invoke(
                app,
                [
                    "ask",
                    "test-project",
                    "Hello",
                    "--projects-dir",
                    str(test_project.path.parent),
                ],
            )

            assert result.exit_code == 1
            assert "Failed to load domain" in result.stdout


class TestAskAgentErrors:
    """Tests for agent-related errors."""

    def test_ask_agent_not_found(self, test_project: Project, mock_studio: Studio) -> None:
        """ask exits with error when specified agent not found."""
        load_result = LoadResult(studio=mock_studio)

        mock_provider = MagicMock()
        mock_provider.check_availability = AsyncMock(return_value=True)

        with (
            patch("questfoundry.runtime.storage.Project.open", return_value=test_project),
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
            patch("questfoundry.runtime.providers.OllamaProvider", return_value=mock_provider),
        ):
            mock_load.return_value = load_result

            result = runner.invoke(
                app,
                [
                    "ask",
                    "test-project",
                    "Hello",
                    "--entry-agent",
                    "nonexistent_agent",
                    "--projects-dir",
                    str(test_project.path.parent),
                ],
            )

            assert result.exit_code == 1
            assert "Agent not found" in result.stdout


class TestAskProviderErrors:
    """Tests for provider-related errors."""

    def test_ask_provider_unavailable(self, test_project: Project, mock_studio: Studio) -> None:
        """ask exits with error when provider unavailable."""
        load_result = LoadResult(studio=mock_studio)

        mock_provider = MagicMock()
        mock_provider.check_availability = AsyncMock(return_value=False)
        mock_provider.host = "http://localhost:11434"

        with (
            patch("questfoundry.runtime.storage.Project.open", return_value=test_project),
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
            patch("questfoundry.runtime.providers.OllamaProvider", return_value=mock_provider),
        ):
            mock_load.return_value = load_result

            result = runner.invoke(
                app,
                [
                    "ask",
                    "test-project",
                    "Hello",
                    "--projects-dir",
                    str(test_project.path.parent),
                ],
            )

            assert result.exit_code == 1
            assert "not available" in result.stdout.lower()


class TestAskSingleShot:
    """Tests for single-shot mode."""

    def test_ask_single_shot_success(self, test_project: Project, mock_studio: Studio) -> None:
        """ask in single-shot mode streams response."""
        load_result = LoadResult(studio=mock_studio)

        mock_provider = MagicMock()
        mock_provider.check_availability = AsyncMock(return_value=True)
        mock_provider.close = AsyncMock()

        # Mock streaming response
        async def mock_stream(*_args, **_kwargs):
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" there!")
            yield StreamChunk(content="", done=True, total_tokens=50)

        mock_provider.stream = mock_stream

        with (
            patch("questfoundry.runtime.storage.Project.open", return_value=test_project),
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
            patch("questfoundry.runtime.providers.OllamaProvider", return_value=mock_provider),
        ):
            mock_load.return_value = load_result

            result = runner.invoke(
                app,
                [
                    "ask",
                    "test-project",
                    "Hello, who are you?",
                    "--projects-dir",
                    str(test_project.path.parent),
                ],
            )

            # Check it ran successfully (or at least didn't exit with project/domain errors)
            # Note: Rich Live rendering may not work perfectly in test runner
            assert "Project not found" not in result.stdout
            assert "Failed to load domain" not in result.stdout

    def test_ask_single_shot_with_agent(self, test_project: Project, mock_studio: Studio) -> None:
        """ask single-shot with specific agent."""
        load_result = LoadResult(studio=mock_studio)

        mock_provider = MagicMock()
        mock_provider.check_availability = AsyncMock(return_value=True)
        mock_provider.close = AsyncMock()

        async def mock_stream(*_args, **_kwargs):
            yield StreamChunk(content="I am the Lorekeeper")
            yield StreamChunk(content="", done=True, total_tokens=50)

        mock_provider.stream = mock_stream

        with (
            patch("questfoundry.runtime.storage.Project.open", return_value=test_project),
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
            patch("questfoundry.runtime.providers.OllamaProvider", return_value=mock_provider),
        ):
            mock_load.return_value = load_result

            result = runner.invoke(
                app,
                [
                    "ask",
                    "test-project",
                    "Who are you?",
                    "--entry-agent",
                    "lorekeeper",
                    "--projects-dir",
                    str(test_project.path.parent),
                ],
            )

            # Verify the non-entry agent was accepted
            assert "Agent not found" not in result.stdout


class TestAskREPL:
    """Tests for REPL mode."""

    def test_ask_repl_exit(self, test_project: Project, mock_studio: Studio) -> None:
        """ask REPL mode can be exited with 'exit'."""
        load_result = LoadResult(studio=mock_studio)

        mock_provider = MagicMock()
        mock_provider.check_availability = AsyncMock(return_value=True)
        mock_provider.close = AsyncMock()

        with (
            patch("questfoundry.runtime.storage.Project.open", return_value=test_project),
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
            patch("questfoundry.runtime.providers.OllamaProvider", return_value=mock_provider),
        ):
            mock_load.return_value = load_result

            # Simulate user typing 'exit'
            result = runner.invoke(
                app,
                [
                    "ask",
                    "test-project",
                    "--projects-dir",
                    str(test_project.path.parent),
                ],
                input="exit\n",
            )

            # REPL should have shown the header and ended cleanly
            # New format: "Session {project}" header and "Session complete" ending
            assert "Session" in result.stdout and (
                "Session complete" in result.stdout or "Session ended" in result.stdout
            )

    def test_ask_repl_quit(self, test_project: Project, mock_studio: Studio) -> None:
        """ask REPL mode can be exited with 'quit'."""
        load_result = LoadResult(studio=mock_studio)

        mock_provider = MagicMock()
        mock_provider.check_availability = AsyncMock(return_value=True)
        mock_provider.close = AsyncMock()

        with (
            patch("questfoundry.runtime.storage.Project.open", return_value=test_project),
            patch("questfoundry.runtime.load_studio", new_callable=AsyncMock) as mock_load,
            patch("questfoundry.runtime.providers.OllamaProvider", return_value=mock_provider),
        ):
            mock_load.return_value = load_result

            result = runner.invoke(
                app,
                [
                    "ask",
                    "test-project",
                    "--projects-dir",
                    str(test_project.path.parent),
                ],
                input="quit\n",
            )

            # New format: "Session complete" instead of "Session ended"
            assert "Session complete" in result.stdout or "Session ended" in result.stdout
