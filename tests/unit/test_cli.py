"""Test CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from questfoundry import __version__
from questfoundry.cli import (
    DEFAULT_GROW_PROMPT,
    DEFAULT_INTERACTIVE_BRAINSTORM_PROMPT,
    DEFAULT_INTERACTIVE_DREAM_PROMPT,
    DEFAULT_INTERACTIVE_SEED_PROMPT,
    DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT,
    DEFAULT_NONINTERACTIVE_SEED_PROMPT,
    STAGE_ORDER,
    STAGE_PROMPTS,
    _preview_grow_artifact,
    _resolve_project_path,
    app,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

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
    assert "pipeline" not in config  # Stages are no longer written to project.yaml
    assert "default" in config["providers"]


def test_init_with_provider(tmp_path: Path) -> None:
    """Test qf init --provider sets provider in project.yaml."""
    import yaml

    runner.invoke(
        app,
        ["init", "test_project", "--path", str(tmp_path), "--provider", "openai/gpt-4o"],
    )

    config_file = tmp_path / "test_project" / "project.yaml"
    with config_file.open() as f:
        config = yaml.safe_load(f)

    assert config["providers"]["default"] == "openai/gpt-4o"


def test_init_existing_directory_fails(tmp_path: Path) -> None:
    """Test qf init fails if directory exists."""
    # Create directory first
    (tmp_path / "existing").mkdir()

    result = runner.invoke(app, ["init", "existing", "--path", str(tmp_path)])

    assert result.exit_code == 1
    # Check for both words (may be split across lines due to terminal wrapping)
    assert "already" in result.stdout
    assert "exists" in result.stdout


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


def test_dream_no_prompt_noninteractive_fails_fast(tmp_path: Path) -> None:
    """Test qf dream fails fast when no prompt in non-interactive mode."""
    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Run without prompt in non-interactive mode (no TTY in test runner)
    result = runner.invoke(
        app,
        ["dream", "--project", str(project_path)],
    )

    # Should fail fast with helpful error, not hang waiting for input
    assert result.exit_code == 1
    assert "Prompt required in non-interactive mode" in result.stdout
    assert "--interactive" in result.stdout or "-i" in result.stdout


# --- Verbosity and Log Flag Tests ---


def test_verbose_flag_exists() -> None:
    """Test -v flag is recognized."""
    result = runner.invoke(app, ["-v", "version"])
    assert result.exit_code == 0


def test_log_flag_exists() -> None:
    """Test --log flag is recognized."""
    result = runner.invoke(app, ["--log", "version"])
    assert result.exit_code == 0


def test_verbose_and_log_flags_together() -> None:
    """Test -v and --log can be used together."""
    result = runner.invoke(app, ["-v", "--log", "version"])
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
        patch("questfoundry.cli._check_providers", return_value=(True, {})),
    ):
        result = runner.invoke(app, ["doctor"])

    assert "Configuration" in result.stdout


def test_doctor_checks_ollama_host() -> None:
    """Test qf doctor checks OLLAMA_HOST."""
    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}, clear=False),
        patch("questfoundry.cli._check_providers", return_value=(True, {})),
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
        patch("questfoundry.cli._check_providers", return_value=(True, {})),
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
        patch("questfoundry.cli._check_providers", return_value=(False, {})),
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
        patch("questfoundry.cli._check_providers", return_value=(True, {})),
    ):
        result = runner.invoke(app, ["doctor", "--project", str(project_path)])

    assert "Project" in result.stdout
    assert "project.yaml" in result.stdout


def test_doctor_exit_code_on_failure() -> None:
    """Test qf doctor exits with code 1 on failure."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("questfoundry.cli._check_providers", return_value=(False, {})),
    ):
        result = runner.invoke(app, ["doctor"])

    # No providers configured should result in failure
    assert result.exit_code == 1


# --- _resolve_project_path Tests ---


def test_resolve_project_path_none_returns_cwd() -> None:
    """Test _resolve_project_path returns current dir when project is None."""
    from pathlib import Path

    result = _resolve_project_path(None)
    assert result == Path()


def test_resolve_project_path_existing_path_returns_as_is(tmp_path: Path) -> None:
    """Test _resolve_project_path returns existing path unchanged."""

    # Create a project structure
    project = tmp_path / "my_project"
    project.mkdir()
    (project / "project.yaml").touch()

    result = _resolve_project_path(project)
    assert result == project


def test_resolve_project_path_simple_name_looks_in_projects_dir(tmp_path: Path) -> None:
    """Test _resolve_project_path checks projects dir for simple names."""
    from pathlib import Path

    import questfoundry.cli as cli_module

    # Create projects directory and a project in it
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project = projects_dir / "my_story"
    project.mkdir()
    (project / "project.yaml").touch()

    # Patch _projects_dir to point to our test directory
    original_projects_dir = cli_module._projects_dir
    try:
        cli_module._projects_dir = projects_dir

        # Simple name should resolve to projects_dir/name
        result = _resolve_project_path(Path("my_story"))
        assert result == project
    finally:
        cli_module._projects_dir = original_projects_dir


def test_resolve_project_path_simple_name_nonexistent_returns_as_is(
    tmp_path: Path,
) -> None:
    """Test _resolve_project_path returns simple name as-is if not in projects dir."""
    from pathlib import Path

    import questfoundry.cli as cli_module

    # Create empty projects directory
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    original_projects_dir = cli_module._projects_dir
    try:
        cli_module._projects_dir = projects_dir

        # Non-existent simple name returns as-is
        result = _resolve_project_path(Path("nonexistent"))
        assert result == Path("nonexistent")
    finally:
        cli_module._projects_dir = original_projects_dir


def test_resolve_project_path_with_separator_not_checked_in_projects_dir(
    tmp_path: Path,
) -> None:
    """Test _resolve_project_path skips projects dir check for paths with separators."""
    from pathlib import Path

    import questfoundry.cli as cli_module

    # Create projects directory with a matching structure
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    nested = projects_dir / "foo" / "bar"
    nested.mkdir(parents=True)

    original_projects_dir = cli_module._projects_dir
    try:
        cli_module._projects_dir = projects_dir

        # Path with separator is returned as-is (not looked up in projects dir)
        result = _resolve_project_path(Path("foo/bar"))
        assert result == Path("foo/bar")
    finally:
        cli_module._projects_dir = original_projects_dir


# --- Default Prompt Constant Tests ---


def test_default_interactive_dream_prompt_constant() -> None:
    """Test DEFAULT_INTERACTIVE_DREAM_PROMPT is defined and non-empty."""
    assert DEFAULT_INTERACTIVE_DREAM_PROMPT
    assert "interactive fiction" in DEFAULT_INTERACTIVE_DREAM_PROMPT.lower()
    assert "creative vision" in DEFAULT_INTERACTIVE_DREAM_PROMPT.lower()


def test_default_interactive_brainstorm_prompt_constant() -> None:
    """Test DEFAULT_INTERACTIVE_BRAINSTORM_PROMPT is defined and non-empty."""
    assert DEFAULT_INTERACTIVE_BRAINSTORM_PROMPT
    assert "brainstorm" in DEFAULT_INTERACTIVE_BRAINSTORM_PROMPT.lower()


def test_default_interactive_seed_prompt_constant() -> None:
    """Test DEFAULT_INTERACTIVE_SEED_PROMPT is defined and non-empty."""
    assert DEFAULT_INTERACTIVE_SEED_PROMPT
    # SEED stage triages brainstorm into structure
    assert "triage" in DEFAULT_INTERACTIVE_SEED_PROMPT.lower()


def test_default_noninteractive_brainstorm_prompt_constant() -> None:
    """Test DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT is defined and non-empty."""
    assert DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT
    # Should reference entities/dilemmas from DREAM stage
    assert (
        "entities" in DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT.lower()
        or "dilemmas" in DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT.lower()
    )


def test_default_noninteractive_seed_prompt_constant() -> None:
    """Test DEFAULT_NONINTERACTIVE_SEED_PROMPT is defined and non-empty."""
    assert DEFAULT_NONINTERACTIVE_SEED_PROMPT
    # Should reference triaging brainstorm into structure
    assert (
        "triage" in DEFAULT_NONINTERACTIVE_SEED_PROMPT.lower()
        or "structure" in DEFAULT_NONINTERACTIVE_SEED_PROMPT.lower()
    )


# --- Non-Interactive Mode with Default Prompts ---


def test_brainstorm_no_prompt_noninteractive_uses_default(tmp_path: Path) -> None:
    """Test qf brainstorm uses default prompt in non-interactive mode."""
    from questfoundry.pipeline import StageResult

    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Mock successful result
    mock_result = StageResult(
        stage="brainstorm",
        status="completed",
        artifact_path=project_path / "graph.json",
        llm_calls=2,
        tokens_used=300,
    )

    # Mock _get_orchestrator to capture the context
    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["brainstorm", "--project", str(project_path)],
        )

        # Should succeed (not exit 1 with "Prompt required" error)
        assert result.exit_code == 0
        # run_stage should have been called with default prompt in context
        mock_orchestrator.run_stage.assert_called_once()
        call_args = mock_orchestrator.run_stage.call_args
        context = call_args[0][1]  # Second positional arg is context
        assert context["user_prompt"] == DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT


def test_seed_no_prompt_noninteractive_uses_default(tmp_path: Path) -> None:
    """Test qf seed uses default prompt in non-interactive mode."""
    from questfoundry.pipeline import StageResult

    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Mock successful result
    mock_result = StageResult(
        stage="seed",
        status="completed",
        artifact_path=project_path / "graph.json",
        llm_calls=2,
        tokens_used=400,
    )

    # Mock _get_orchestrator to capture the context
    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["seed", "--project", str(project_path)],
        )

        # Should succeed (not exit 1 with "Prompt required" error)
        assert result.exit_code == 0
        # run_stage should have been called with default prompt in context
        mock_orchestrator.run_stage.assert_called_once()
        call_args = mock_orchestrator.run_stage.call_args
        context = call_args[0][1]  # Second positional arg is context
        assert context["user_prompt"] == DEFAULT_NONINTERACTIVE_SEED_PROMPT


# --- STAGE_ORDER and STAGE_PROMPTS Constants Tests ---


def test_stage_order_constant() -> None:
    """Test STAGE_ORDER contains expected stages in order."""
    assert STAGE_ORDER == ["dream", "brainstorm", "seed", "grow", "fill", "dress", "ship"]


def test_stage_prompts_has_all_stages() -> None:
    """Test STAGE_PROMPTS has entry for each stage in STAGE_ORDER."""
    for stage in STAGE_ORDER:
        assert stage in STAGE_PROMPTS


def test_stage_prompts_dream_requires_prompt() -> None:
    """Test DREAM stage has no default non-interactive prompt (requires explicit)."""
    interactive_prompt, noninteractive_prompt = STAGE_PROMPTS["dream"]
    assert interactive_prompt  # Has interactive prompt
    assert noninteractive_prompt is None  # No non-interactive default


def test_stage_prompts_brainstorm_has_defaults() -> None:
    """Test BRAINSTORM stage has both interactive and non-interactive prompts."""
    interactive_prompt, noninteractive_prompt = STAGE_PROMPTS["brainstorm"]
    assert interactive_prompt
    assert noninteractive_prompt


def test_stage_prompts_seed_has_defaults() -> None:
    """Test SEED stage has both interactive and non-interactive prompts."""
    interactive_prompt, noninteractive_prompt = STAGE_PROMPTS["seed"]
    assert interactive_prompt
    assert noninteractive_prompt


# --- Run Command Tests ---


def test_run_command_no_project_fails() -> None:
    """Test qf run fails without project.yaml."""
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["run", "--to", "dream", "--prompt", "test"])

        assert result.exit_code == 1
        assert "No project.yaml found" in result.stdout


def test_run_command_unknown_to_stage_fails(tmp_path: Path) -> None:
    """Test qf run fails with unknown --to stage."""
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    result = runner.invoke(
        app,
        ["run", "--to", "invalid", "--project", str(project_path)],
    )

    assert result.exit_code == 1
    assert "Unknown stage" in result.stdout
    assert "invalid" in result.stdout


def test_run_command_unknown_from_stage_fails(tmp_path: Path) -> None:
    """Test qf run fails with unknown --from stage."""
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    result = runner.invoke(
        app,
        ["run", "--to", "seed", "--from", "invalid", "--project", str(project_path)],
    )

    assert result.exit_code == 1
    assert "Unknown stage" in result.stdout
    assert "invalid" in result.stdout


def test_run_command_from_after_to_fails(tmp_path: Path) -> None:
    """Test qf run fails when --from comes after --to."""
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    result = runner.invoke(
        app,
        [
            "run",
            "--to",
            "dream",
            "--from",
            "seed",
            "--project",
            str(project_path),
            "--prompt",
            "test",
        ],
    )

    assert result.exit_code == 1
    assert "--from stage must come before --to stage" in result.stdout


def test_run_command_dream_requires_prompt(tmp_path: Path) -> None:
    """Test qf run requires prompt when DREAM stage will run."""
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_status = MagicMock()
        mock_status.stages = {}  # No stages completed
        mock_orchestrator.get_status.return_value = mock_status
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["run", "--to", "seed", "--project", str(project_path)],
        )

    assert result.exit_code == 1
    assert "DREAM stage requires a prompt" in result.stdout


def test_run_command_runs_stages(tmp_path: Path) -> None:
    """Test qf run executes specified stages."""
    from questfoundry.pipeline import StageResult

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Mock successful results for each stage
    def make_result(stage: str) -> StageResult:
        return StageResult(
            stage=stage,
            status="completed",
            artifact_path=project_path / "graph.json",
            llm_calls=2,
            tokens_used=300,
        )

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_status = MagicMock()
        mock_status.stages = {}  # No stages completed
        mock_orchestrator.get_status.return_value = mock_status
        mock_orchestrator.run_stage = AsyncMock(side_effect=lambda s, _: make_result(s))
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            [
                "run",
                "--to",
                "brainstorm",
                "--project",
                str(project_path),
                "--prompt",
                "A mystery story",
            ],
        )

    assert result.exit_code == 0
    assert "Pipeline run complete" in result.stdout
    # Should have run both dream and brainstorm
    assert mock_orchestrator.run_stage.call_count == 2


def test_run_command_skips_completed_stages(tmp_path: Path) -> None:
    """Test qf run skips already completed stages."""
    from questfoundry.pipeline import StageResult

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    mock_result = StageResult(
        stage="brainstorm",
        status="completed",
        artifact_path=project_path / "graph.json",
        llm_calls=2,
        tokens_used=300,
    )

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()

        # Mock status: dream is completed
        mock_dream_info = MagicMock()
        mock_dream_info.status = "completed"
        mock_status = MagicMock()
        mock_status.stages = {"dream": mock_dream_info}
        mock_orchestrator.get_status.return_value = mock_status

        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["run", "--to", "brainstorm", "--project", str(project_path)],
        )

    assert result.exit_code == 0
    # Should only run brainstorm (dream is skipped)
    assert mock_orchestrator.run_stage.call_count == 1
    call_args = mock_orchestrator.run_stage.call_args
    assert call_args[0][0] == "brainstorm"


def test_run_command_force_reruns_completed(tmp_path: Path) -> None:
    """Test qf run --force re-runs already completed stages."""
    from questfoundry.pipeline import StageResult

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    def make_result(stage: str) -> StageResult:
        return StageResult(
            stage=stage,
            status="completed",
            artifact_path=project_path / "graph.json",
            llm_calls=2,
            tokens_used=300,
        )

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()

        # Mock status: dream is completed
        mock_dream_info = MagicMock()
        mock_dream_info.status = "completed"
        mock_status = MagicMock()
        mock_status.stages = {"dream": mock_dream_info}
        mock_orchestrator.get_status.return_value = mock_status

        mock_orchestrator.run_stage = AsyncMock(side_effect=lambda s, _: make_result(s))
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            [
                "run",
                "--to",
                "brainstorm",
                "--project",
                str(project_path),
                "--prompt",
                "A story",
                "--force",
            ],
        )

    assert result.exit_code == 0
    # Should run both stages (force overrides completed status)
    assert mock_orchestrator.run_stage.call_count == 2


def test_run_command_all_completed_no_force(tmp_path: Path) -> None:
    """Test qf run shows message when all stages already completed."""
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()

        # Mock status: all stages completed
        mock_dream_info = MagicMock()
        mock_dream_info.status = "completed"
        mock_brainstorm_info = MagicMock()
        mock_brainstorm_info.status = "completed"
        mock_status = MagicMock()
        mock_status.stages = {"dream": mock_dream_info, "brainstorm": mock_brainstorm_info}
        mock_orchestrator.get_status.return_value = mock_status
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["run", "--to", "brainstorm", "--project", str(project_path)],
        )

    assert result.exit_code == 0
    assert "All stages already completed" in result.stdout
    assert "--force" in result.stdout


# --- Phase-Specific Provider Flag Tests ---


def test_dream_phase_provider_flags_passed_to_orchestrator(tmp_path: Path) -> None:
    """Test dream command passes phase-specific provider flags to orchestrator."""
    from questfoundry.pipeline import StageResult

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

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

    artifact = {"type": "dream", "version": 1, "genre": "fantasy"}
    with (project_path / "artifacts" / "dream.yaml").open("w") as f:
        yaml.safe_dump(artifact, f)

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            [
                "dream",
                "A story",
                "--project",
                str(project_path),
                "--provider-discuss",
                "ollama/qwen3:4b-instruct-32k",
                "--provider-summarize",
                "openai/gpt-5-mini",
                "--provider-serialize",
                "openai/o1-mini",
            ],
        )

    assert result.exit_code == 0
    # Verify orchestrator was called with phase-specific overrides
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args[1]
    assert call_kwargs["provider_discuss_override"] == "ollama/qwen3:4b-instruct-32k"
    assert call_kwargs["provider_summarize_override"] == "openai/gpt-5-mini"
    assert call_kwargs["provider_serialize_override"] == "openai/o1-mini"


def test_brainstorm_phase_provider_flags_passed_to_orchestrator(tmp_path: Path) -> None:
    """Test brainstorm command passes phase-specific provider flags to orchestrator."""
    from questfoundry.pipeline import StageResult

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    mock_result = StageResult(
        stage="brainstorm",
        status="completed",
        artifact_path=project_path / "artifacts" / "brainstorm.yaml",
        llm_calls=1,
        tokens_used=500,
        duration_seconds=1.5,
    )

    # Create mock artifact
    import yaml

    artifact = {"type": "brainstorm", "version": 1, "entities": [], "dilemmas": []}
    with (project_path / "artifacts" / "brainstorm.yaml").open("w") as f:
        yaml.safe_dump(artifact, f)

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            [
                "brainstorm",
                "--project",
                str(project_path),
                "--provider-serialize",
                "openai/o1-mini",
            ],
        )

    assert result.exit_code == 0
    call_kwargs = mock_get.call_args[1]
    assert call_kwargs["provider_serialize_override"] == "openai/o1-mini"


def test_seed_phase_provider_flags_passed_to_orchestrator(tmp_path: Path) -> None:
    """Test seed command passes phase-specific provider flags to orchestrator."""
    from questfoundry.pipeline import StageResult

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    mock_result = StageResult(
        stage="seed",
        status="completed",
        artifact_path=project_path / "artifacts" / "seed.yaml",
        llm_calls=1,
        tokens_used=500,
        duration_seconds=1.5,
    )

    # Create mock artifact
    import yaml

    artifact = {"type": "seed", "version": 1, "entities": [], "paths": [], "initial_beats": []}
    with (project_path / "artifacts" / "seed.yaml").open("w") as f:
        yaml.safe_dump(artifact, f)

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            [
                "seed",
                "--project",
                str(project_path),
                "--provider-discuss",
                "anthropic/claude-3",
            ],
        )

    assert result.exit_code == 0
    call_kwargs = mock_get.call_args[1]
    assert call_kwargs["provider_discuss_override"] == "anthropic/claude-3"


def test_run_phase_provider_flags_passed_to_orchestrator(tmp_path: Path) -> None:
    """Test run command passes phase-specific provider flags to each stage."""
    from questfoundry.pipeline import StageResult

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Create mock artifacts directory
    import yaml

    # Create dream artifact
    artifact = {"type": "dream", "version": 1, "genre": "fantasy"}
    with (project_path / "artifacts" / "dream.yaml").open("w") as f:
        yaml.safe_dump(artifact, f)

    def make_result(stage: str) -> StageResult:
        artifact_file = project_path / "artifacts" / f"{stage}.yaml"
        artifact_file.write_text(f"type: {stage}\nversion: 1\n")
        return StageResult(
            stage=stage,
            status="completed",
            artifact_path=artifact_file,
            llm_calls=1,
            tokens_used=100,
            duration_seconds=0.5,
        )

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_status = MagicMock()
        mock_status.stages = {}  # No completed stages
        mock_orchestrator.get_status.return_value = mock_status
        mock_orchestrator.run_stage = AsyncMock(side_effect=lambda s, _: make_result(s))
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            [
                "run",
                "--to",
                "dream",
                "--project",
                str(project_path),
                "--prompt",
                "A story",
                "--provider-serialize",
                "openai/o1-mini",
            ],
        )

    assert result.exit_code == 0
    # Check that phase-specific flags were passed through
    call_kwargs = mock_get.call_args[1]
    assert call_kwargs["provider_serialize_override"] == "openai/o1-mini"


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_dream_help_shows_phase_provider_flags() -> None:
    """Test dream --help shows all phase-specific provider flags."""
    result = runner.invoke(app, ["dream", "--help"])
    output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "--provider-discuss" in output
    assert "--provider-summarize" in output
    assert "--provider-serialize" in output


def test_run_help_shows_phase_provider_flags() -> None:
    """Test run --help shows all phase-specific provider flags."""
    result = runner.invoke(app, ["run", "--help"])
    output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "--provider-discuss" in output
    # Rich may truncate long option names with ellipsis depending on column width
    assert "--provider-summar" in output
    assert "--provider-serial" in output


# --- Run --init Tests ---


def test_run_init_creates_project(tmp_path: Path) -> None:
    """Test qf run --init creates project when it doesn't exist."""
    project_path = tmp_path / "new-project"

    result = runner.invoke(
        app,
        [
            "run",
            "--to",
            "dream",
            "--prompt",
            "test",
            "--project",
            str(project_path),
            "--init",
            "--no-interactive",
        ],
    )

    # The project should have been created (run may still fail on LLM, but init succeeds)
    assert (project_path / "project.yaml").exists()
    assert (project_path / "artifacts").is_dir()
    assert "Created project" in result.stdout


def test_run_init_existing_project_continues(tmp_path: Path) -> None:
    """Test qf run --init with existing project proceeds normally."""
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Should not error about project already existing
    result = runner.invoke(
        app,
        [
            "run",
            "--to",
            "dream",
            "--prompt",
            "test",
            "--project",
            str(project_path),
            "--init",
            "--no-interactive",
        ],
    )

    # Should proceed to stage execution (may fail on LLM, but not on init)
    assert "already exists" not in result.stdout


def test_run_no_init_no_project_fails_non_interactive() -> None:
    """Test qf run without --init fails when project doesn't exist."""
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["run", "--to", "dream", "--prompt", "test"])

        assert result.exit_code == 1
        assert "No project.yaml found" in result.stdout


def test_run_init_interactive_prompt_decline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test qf run without --init prompts in interactive mode and user declines."""
    project_path = tmp_path / "new-project"

    # Mock TTY detection to return True
    monkeypatch.setattr("questfoundry.cli._is_interactive_tty", lambda: True)

    result = runner.invoke(
        app,
        [
            "run",
            "--to",
            "dream",
            "--prompt",
            "test",
            "--project",
            str(project_path),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert not (project_path / "project.yaml").exists()


def test_run_init_interactive_prompt_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test qf run without --init prompts in interactive mode and user accepts."""
    project_path = tmp_path / "new-project"

    # Mock TTY detection to return True
    monkeypatch.setattr("questfoundry.cli._is_interactive_tty", lambda: True)

    result = runner.invoke(
        app,
        [
            "run",
            "--to",
            "dream",
            "--prompt",
            "test",
            "--project",
            str(project_path),
        ],
        input="y\n",
    )

    assert (project_path / "project.yaml").exists()
    assert "Created project" in result.stdout


# --- GROW Command Tests ---


def test_grow_command_exists() -> None:
    """Test qf grow command is registered and shows help."""
    result = runner.invoke(app, ["grow", "--help"])
    output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "branching structure" in output.lower()
    assert "--project" in output
    assert "--provider" in output


def test_fill_command_exists() -> None:
    """Test qf fill command is registered and shows help."""
    result = runner.invoke(app, ["fill", "--help"])
    output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "prose" in output.lower()
    assert "--project" in output
    assert "--provider" in output


def test_fill_no_project_fails() -> None:
    """Test qf fill fails without project.yaml."""
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["fill"])

        assert result.exit_code == 1
        assert "No project.yaml found" in result.stdout


def test_grow_no_project_fails() -> None:
    """Test qf grow fails without project.yaml."""
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["grow"])

        assert result.exit_code == 1
        assert "No project.yaml found" in result.stdout


def test_grow_with_mock_provider(tmp_path: Path) -> None:
    """Test qf grow runs stage with mocked provider."""
    from questfoundry.pipeline import StageResult

    # Create project
    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    # Mock the orchestrator
    mock_result = StageResult(
        stage="grow",
        status="completed",
        artifact_path=project_path / "artifacts" / "grow.yaml",
        llm_calls=5,
        tokens_used=2000,
        duration_seconds=10.0,
    )

    # Create mock artifact
    import yaml

    artifact = {
        "arc_count": 3,
        "passage_count": 12,
        "choice_count": 8,
        "codeword_count": 4,
        "overlay_count": 2,
        "spine_arc_id": "arc_main",
        "phases_completed": [
            {"phase": 1, "status": "completed"},
            {"phase": 2, "status": "completed"},
        ],
    }
    with (project_path / "artifacts" / "grow.yaml").open("w") as f:
        yaml.safe_dump(artifact, f)

    with patch("questfoundry.cli._get_orchestrator") as mock_get:
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_stage = AsyncMock(return_value=mock_result)
        mock_orchestrator.close = AsyncMock()
        mock_orchestrator.config.provider.name = "test"
        mock_get.return_value = mock_orchestrator

        result = runner.invoke(
            app,
            ["grow", "--project", str(project_path)],
        )

    assert result.exit_code == 0
    assert "GROW stage completed" in result.stdout
    assert "Tokens: 2,000" in result.stdout


def test_grow_has_no_interactive_flag() -> None:
    """Test qf grow does not offer --interactive flag."""
    result = runner.invoke(app, ["grow", "--help"])
    output = _strip_ansi(result.stdout)

    assert "--interactive" not in output


def test_grow_preview_function_basic() -> None:
    """Test _preview_grow_artifact displays arc/passage/choice counts."""
    from io import StringIO

    from rich.console import Console

    import questfoundry.cli as cli_module

    artifact = {
        "arc_count": 3,
        "passage_count": 12,
        "choice_count": 8,
        "codeword_count": 4,
        "overlay_count": 0,
        "spine_arc_id": "arc_main",
        "phases_completed": [
            {"phase": 1, "status": "completed"},
            {"phase": 2, "status": "completed"},
            {"phase": 3, "status": "completed"},
        ],
    }

    # Capture console output
    output = StringIO()
    test_console = Console(file=output, highlight=False)
    original_console = cli_module.console
    try:
        cli_module.console = test_console
        _preview_grow_artifact(artifact)
    finally:
        cli_module.console = original_console

    text = output.getvalue()
    assert "3" in text  # arc_count
    assert "12" in text  # passage_count
    assert "8" in text  # choice_count
    assert "4" in text  # codeword_count
    assert "arc_main" in text  # spine_arc_id
    assert "3 completed" in text  # phases


def test_grow_preview_function_with_failures() -> None:
    """Test _preview_grow_artifact shows failed phases."""
    from io import StringIO

    from rich.console import Console

    import questfoundry.cli as cli_module

    artifact = {
        "arc_count": 1,
        "passage_count": 5,
        "choice_count": 3,
        "codeword_count": 2,
        "overlay_count": 0,
        "spine_arc_id": "arc_main",
        "phases_completed": [
            {"phase": 1, "status": "completed"},
            {"phase": 2, "status": "failed", "detail": "Validation error"},
        ],
    }

    output = StringIO()
    test_console = Console(file=output, highlight=False)
    original_console = cli_module.console
    try:
        cli_module.console = test_console
        _preview_grow_artifact(artifact)
    finally:
        cli_module.console = original_console

    text = output.getvalue()
    assert "Failed phases: 1" in text


def test_grow_preview_function_with_overlays() -> None:
    """Test _preview_grow_artifact shows overlay count when non-zero."""
    from io import StringIO

    from rich.console import Console

    import questfoundry.cli as cli_module

    artifact = {
        "arc_count": 2,
        "passage_count": 8,
        "choice_count": 5,
        "codeword_count": 3,
        "overlay_count": 4,
        "spine_arc_id": "arc_spine",
        "phases_completed": [],
    }

    output = StringIO()
    test_console = Console(file=output, highlight=False)
    original_console = cli_module.console
    try:
        cli_module.console = test_console
        _preview_grow_artifact(artifact)
    finally:
        cli_module.console = original_console

    text = output.getvalue()
    assert "Overlays" in text
    assert "4" in text


# --- Generate Images Command Tests ---


def test_generate_images_no_project_fails() -> None:
    """generate-images fails when no project exists."""
    result = runner.invoke(app, ["generate-images", "--project", "/nonexistent/path"])
    assert result.exit_code != 0


def test_generate_images_no_provider_fails(tmp_path: Path) -> None:
    """generate-images fails when no image provider is specified."""
    from ruamel.yaml import YAML

    project = tmp_path / "test-proj"
    project.mkdir()
    yaml_writer = YAML()
    with (project / "project.yaml").open("w") as f:
        yaml_writer.dump(
            {
                "name": "test-proj",
                "version": "0.1.0",
                "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
            },
            f,
        )

    result = runner.invoke(app, ["generate-images", "--project", str(project)])
    assert result.exit_code != 0
    assert "No image provider" in result.stdout


def test_generate_images_no_selection_fails(tmp_path: Path) -> None:
    """generate-images fails when dress has not been run yet."""
    from ruamel.yaml import YAML

    from questfoundry.graph.graph import Graph

    project = tmp_path / "test-proj"
    project.mkdir()
    yaml_writer = YAML()
    with (project / "project.yaml").open("w") as f:
        yaml_writer.dump(
            {
                "name": "test-proj",
                "version": "0.1.0",
                "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
            },
            f,
        )

    g = Graph()
    g.set_last_stage("dress")
    g.save(project / "graph.json")

    result = runner.invoke(
        app, ["generate-images", "--project", str(project), "--image-provider", "placeholder"]
    )
    assert result.exit_code != 0
    assert "No brief selection" in result.stdout


def test_generate_images_success(tmp_path: Path) -> None:
    """generate-images succeeds with valid graph and provider."""
    from ruamel.yaml import YAML

    from questfoundry.graph.graph import Graph

    project = tmp_path / "test-proj"
    project.mkdir()
    yaml_writer = YAML()
    with (project / "project.yaml").open("w") as f:
        yaml_writer.dump(
            {
                "name": "test-proj",
                "version": "0.1.0",
                "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
            },
            f,
        )

    g = Graph()
    g.set_last_stage("dress")
    g.upsert_node(
        "dress_meta::selection",
        {"type": "dress_meta", "selected_briefs": [], "total_briefs": 0},
    )
    g.save(project / "graph.json")

    result = runner.invoke(
        app, ["generate-images", "--project", str(project), "--image-provider", "placeholder"]
    )
    assert result.exit_code == 0
    assert "complete" in result.stdout.lower()


def test_stage_prompts_grow_has_defaults() -> None:
    """Test GROW stage has both interactive and non-interactive prompts (same value)."""
    interactive_prompt, noninteractive_prompt = STAGE_PROMPTS["grow"]
    assert interactive_prompt == DEFAULT_GROW_PROMPT
    assert noninteractive_prompt == DEFAULT_GROW_PROMPT


def test_grow_prompt_constant() -> None:
    """Test DEFAULT_GROW_PROMPT is defined and describes branching."""
    assert DEFAULT_GROW_PROMPT
    assert "branching" in DEFAULT_GROW_PROMPT.lower()
    assert "arcs" in DEFAULT_GROW_PROMPT.lower()
