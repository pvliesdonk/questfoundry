"""Test CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from questfoundry import __version__
from questfoundry.cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_version_command() -> None:
    """Test qf version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"v{__version__}" in result.stdout


def test_no_args_shows_help() -> None:
    """Test that no arguments shows help."""
    result = runner.invoke(app, [])
    # no_args_is_help=True returns exit code 2 (not 0 like --help)
    assert result.exit_code == 2
    assert "QuestFoundry" in result.stdout


# --- Init Command Tests ---


def test_init_creates_project(tmp_path: Path) -> None:
    """Test qf init creates project structure."""
    result = runner.invoke(app, ["init", "my_story", "--path", str(tmp_path)])

    assert result.exit_code == 0
    assert "Created project" in result.stdout
    assert "my_story" in result.stdout

    # Check structure created
    project_path = tmp_path / "my_story"
    assert project_path.exists()
    assert (project_path / "project.yaml").exists()
    assert (project_path / "artifacts").is_dir()


def test_init_project_yaml_content(tmp_path: Path) -> None:
    """Test qf init creates valid project.yaml."""
    import yaml

    runner.invoke(app, ["init", "test_project", "--path", str(tmp_path)])

    config_file = tmp_path / "test_project" / "project.yaml"
    with config_file.open() as f:
        config = yaml.safe_load(f)

    assert config["name"] == "test_project"
    assert config["version"] == 1
    assert "stages" in config["pipeline"]
    assert "dream" in config["pipeline"]["stages"]
    assert "default" in config["providers"]


def test_init_existing_directory_fails(tmp_path: Path) -> None:
    """Test qf init fails if directory exists."""
    # Create directory first
    (tmp_path / "existing").mkdir()

    result = runner.invoke(app, ["init", "existing", "--path", str(tmp_path)])

    assert result.exit_code == 1
    assert "already exists" in result.stdout


# --- Status Command Tests ---


def test_status_no_project_fails() -> None:
    """Test qf status fails without project.yaml."""
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "No project.yaml found" in result.stdout


def test_status_shows_stages(tmp_path: Path) -> None:
    """Test qf status displays pipeline stages."""
    # Create minimal project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])

    result = runner.invoke(app, ["status", "--project", str(tmp_path / "test")])

    assert result.exit_code == 0
    assert "Pipeline Status" in result.stdout
    assert "dream" in result.stdout
    assert "pending" in result.stdout


def test_status_shows_completed_stage(tmp_path: Path) -> None:
    """Test qf status shows completed stages."""
    import yaml

    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Create fake artifact
    artifact = {"type": "dream", "version": 1, "genre": "fantasy"}
    with (project_path / "artifacts" / "dream.yaml").open("w") as f:
        yaml.safe_dump(artifact, f)

    result = runner.invoke(app, ["status", "--project", str(project_path)])

    assert result.exit_code == 0
    assert "completed" in result.stdout


# --- Dream Command Tests ---


def test_dream_no_project_fails() -> None:
    """Test qf dream fails without project.yaml."""
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["dream", "A fantasy story"])

        assert result.exit_code == 1
        assert "No project.yaml found" in result.stdout


def test_dream_with_mock_provider(tmp_path: Path) -> None:
    """Test qf dream runs stage with mocked provider."""
    from questfoundry.pipeline import StageResult

    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Mock the orchestrator
    mock_result = StageResult(
        stage="dream",
        status="completed",
        artifact_path=project_path / "artifacts" / "dream.yaml",
        llm_calls=1,
        tokens_used=500,
        duration_seconds=1.5,
    )

    # Create mock artifact
    import yaml

    artifact = {
        "type": "dream",
        "version": 1,
        "genre": "fantasy",
        "subgenre": "epic",
        "tone": ["adventurous", "dramatic"],
        "themes": ["heroism", "destiny"],
    }
    with (project_path / "artifacts" / "dream.yaml").open("w") as f:
        yaml.safe_dump(artifact, f)

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["dream", "An epic fantasy quest", "--project", str(project_path)],
        )

    assert result.exit_code == 0
    assert "DREAM stage completed" in result.stdout
    assert "fantasy" in result.stdout
    assert "Tokens: 500" in result.stdout


def test_dream_failed_stage(tmp_path: Path) -> None:
    """Test qf dream handles failed stage."""
    from questfoundry.pipeline import StageResult

    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Mock failed result
    mock_result = StageResult(
        stage="dream",
        status="failed",
        errors=["LLM connection failed", "Timeout after 30s"],
    )

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["dream", "A story", "--project", str(project_path)],
        )

    assert result.exit_code == 1
    assert "DREAM stage failed" in result.stdout
    assert "LLM connection failed" in result.stdout


def test_dream_prompts_for_input(tmp_path: Path) -> None:
    """Test qf dream prompts for input when not provided."""
    from questfoundry.pipeline import StageResult

    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    mock_result = StageResult(
        stage="dream",
        status="completed",
        artifact_path=project_path / "artifacts" / "dream.yaml",
        llm_calls=1,
        tokens_used=100,
        duration_seconds=0.5,
    )

    # Create minimal artifact
    import yaml

    with (project_path / "artifacts" / "dream.yaml").open("w") as f:
        yaml.safe_dump({"type": "dream", "genre": "test"}, f)

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_get.return_value = mock_orchestrator

        # Simulate user input
        result = runner.invoke(
            app,
            ["dream", "--project", str(project_path)],
            input="A mystery story\n",
        )

    assert result.exit_code == 0
    # Verify the stage was called with the user's input
    mock_orchestrator.run_stage.assert_called_once()
    call_args = mock_orchestrator.run_stage.call_args
    assert call_args[0][0] == "dream"
    assert call_args[0][1]["user_prompt"] == "A mystery story"


# --- Verbosity Flag Tests ---


def test_verbose_flag_exists() -> None:
    """Test -v flag is recognized."""
    result = runner.invoke(app, ["-v", "version"])
    assert result.exit_code == 0


def test_verbose_flag_countable() -> None:
    """Test -vv and -vvv are recognized."""
    result = runner.invoke(app, ["-vv", "version"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["-vvv", "version"])
    assert result.exit_code == 0


# --- Doctor Command Tests ---


def test_doctor_help() -> None:
    """Test qf doctor --help shows command help."""
    result = runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    assert "configuration" in result.stdout.lower()
    assert "connectivity" in result.stdout.lower()


def test_doctor_shows_configuration() -> None:
    """Test qf doctor shows configuration section."""
    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}, clear=False),
        patch("questfoundry.cli._check_providers", return_value=True),
    ):
        result = runner.invoke(app, ["doctor"])

    assert "Configuration" in result.stdout


def test_doctor_checks_ollama_host() -> None:
    """Test qf doctor checks OLLAMA_HOST."""
    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}, clear=False),
        patch("questfoundry.cli._check_providers", return_value=True),
    ):
        result = runner.invoke(app, ["doctor"])

    assert "OLLAMA_HOST" in result.stdout
    assert "http://test:11434" in result.stdout


def test_doctor_masks_api_keys() -> None:
    """Test qf doctor masks API keys in output."""
    with (
        patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-1234567890abcdef"},
            clear=False,
        ),
        patch("questfoundry.cli._check_providers", return_value=True),
    ):
        result = runner.invoke(app, ["doctor"])

    # Should mask the key (first 7 chars + ... + last 3 chars)
    assert "sk-1234" in result.stdout
    assert "...def" in result.stdout
    # Should NOT show full key
    assert "sk-1234567890abcdef" not in result.stdout


def test_doctor_shows_unconfigured() -> None:
    """Test qf doctor shows unconfigured providers."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("questfoundry.cli._check_providers", return_value=False),
    ):
        result = runner.invoke(app, ["doctor"])

    assert "not configured" in result.stdout


def test_doctor_checks_project_when_present(tmp_path: Path) -> None:
    """Test qf doctor checks project.yaml when present."""
    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("questfoundry.cli._check_providers", return_value=True),
    ):
        result = runner.invoke(app, ["doctor", "--project", str(project_path)])

    assert "Project" in result.stdout
    assert "project.yaml" in result.stdout


def test_doctor_exit_code_on_failure() -> None:
    """Test qf doctor exits with code 1 on failure."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("questfoundry.cli._check_providers", return_value=False),
    ):
        result = runner.invoke(app, ["doctor"])

    # No providers configured should result in failure
    assert result.exit_code == 1
