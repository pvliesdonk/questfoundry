"""Test CLI commands."""

from typer.testing import CliRunner

from questfoundry import __version__
from questfoundry.cli import app

runner = CliRunner()


def test_version_command() -> None:
    """Test qf version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"v{__version__}" in result.stdout


def test_status_command() -> None:
    """Test qf status command (stub)."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Not implemented" in result.stdout


def test_no_args_shows_help() -> None:
    """Test that no arguments shows help."""
    result = runner.invoke(app, [])
    # Typer returns exit code 0 for --help, but 2 for no_args_is_help
    # The important thing is that help text is shown
    assert "QuestFoundry" in result.stdout
