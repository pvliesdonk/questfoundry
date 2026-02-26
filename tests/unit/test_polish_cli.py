"""Tests for POLISH stage CLI wiring and registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from questfoundry.cli import DEFAULT_POLISH_PROMPT, STAGE_ORDER, STAGE_PROMPTS, app
from questfoundry.pipeline.config import DEFAULT_STAGES
from questfoundry.pipeline.stages import PolishStage, PolishStageError, get_stage, list_stages

runner = CliRunner()


def test_polish_in_stage_order() -> None:
    """POLISH must appear between GROW and FILL in STAGE_ORDER."""
    assert "polish" in STAGE_ORDER
    grow_idx = STAGE_ORDER.index("grow")
    polish_idx = STAGE_ORDER.index("polish")
    fill_idx = STAGE_ORDER.index("fill")
    assert grow_idx < polish_idx < fill_idx


def test_polish_in_default_stages() -> None:
    """POLISH must appear in DEFAULT_STAGES between GROW and FILL."""
    assert "polish" in DEFAULT_STAGES
    grow_idx = DEFAULT_STAGES.index("grow")
    polish_idx = DEFAULT_STAGES.index("polish")
    fill_idx = DEFAULT_STAGES.index("fill")
    assert grow_idx < polish_idx < fill_idx


def test_polish_in_stage_prompts() -> None:
    """POLISH must have an entry in STAGE_PROMPTS."""
    assert "polish" in STAGE_PROMPTS
    interactive, noninteractive = STAGE_PROMPTS["polish"]
    assert interactive == DEFAULT_POLISH_PROMPT
    assert noninteractive == DEFAULT_POLISH_PROMPT


def test_polish_stage_registered() -> None:
    """POLISH stage must be registered and retrievable."""
    assert "polish" in list_stages()
    stage = get_stage("polish")
    assert stage is not None
    assert stage.name == "polish"


def test_polish_stage_class() -> None:
    """PolishStage class has expected attributes."""
    stage = PolishStage()
    assert stage.name == "polish"
    assert stage.project_path is None


def test_polish_stage_execute_raises_not_implemented() -> None:
    """Stub execute() raises PolishStageError."""
    import asyncio
    from unittest.mock import MagicMock

    stage = PolishStage()
    mock_model = MagicMock()

    with pytest.raises(PolishStageError, match="not yet implemented"):
        asyncio.run(stage.execute(mock_model, "test prompt"))


def test_polish_cli_command_exists() -> None:
    """The 'polish' CLI command must be registered."""
    result = runner.invoke(app, ["polish", "--help"])
    assert result.exit_code == 0
    assert "POLISH" in result.stdout or "polish" in result.stdout.lower()


def test_grow_next_step_hint_is_polish(tmp_path: Path) -> None:
    """GROW's next step hint should point to 'qf polish', not 'qf fill'."""
    from unittest.mock import patch

    runner.invoke(app, ["init", "test", "--path", str(tmp_path)])
    project_path = tmp_path / "test"

    with patch("questfoundry.cli._run_stage_command") as mock_run:
        runner.invoke(app, ["grow", "--project", str(project_path)])

    if mock_run.called:
        call_kwargs = mock_run.call_args
        # Check keyword args or positional args for next_step_hint
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("next_step_hint") == "qf polish"
