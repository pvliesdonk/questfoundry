"""Tests for pipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.pipeline import (
    AutoApproveGate,
    PipelineOrchestrator,
    ProjectConfig,
    RequireSuccessGate,
    StageNotFoundError,
    StageResult,
    create_default_config,
    load_project_config,
)
from questfoundry.pipeline.config import ProjectConfigError
from questfoundry.pipeline.stages import get_stage, list_stages, register_stage

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.default_model = "test-model"
    provider.complete = AsyncMock()
    provider.close = AsyncMock()
    return provider


# --- ProjectConfig Tests ---


def test_create_default_config() -> None:
    """create_default_config returns config with defaults."""
    config = create_default_config("test_project")

    assert config.name == "test_project"
    assert config.version == 1
    assert config.provider.name == "ollama"
    assert config.provider.model == "qwen3:8b"
    assert "dream" in config.stages


def test_config_from_dict_minimal() -> None:
    """ProjectConfig.from_dict handles minimal config."""
    data = {"name": "minimal"}
    config = ProjectConfig.from_dict(data)

    assert config.name == "minimal"
    assert config.provider.name == "ollama"


def test_config_from_dict_full() -> None:
    """ProjectConfig.from_dict parses full config."""
    data = {
        "name": "full",
        "version": 2,
        "pipeline": {
            "stages": ["dream", "seed"],
            "gates": {
                "dream": "optional",
                "seed": "required",
            },
        },
        "providers": {
            "default": "openai/gpt-4o",
        },
    }
    config = ProjectConfig.from_dict(data)

    assert config.name == "full"
    assert config.version == 2
    assert config.provider.name == "openai"
    assert config.provider.model == "gpt-4o"
    assert config.stages == ["dream", "seed"]
    assert len(config.gates) == 2
    assert any(g.stage == "seed" and g.required for g in config.gates)


def test_load_project_config(tmp_path: Path) -> None:
    """load_project_config loads from project.yaml."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: loaded
version: 1
providers:
  default: ollama/llama3
"""
    )

    config = load_project_config(tmp_path)

    assert config.name == "loaded"
    assert config.provider.model == "llama3"


def test_load_project_config_not_found(tmp_path: Path) -> None:
    """load_project_config raises error for missing file."""
    with pytest.raises(ProjectConfigError) as exc_info:
        load_project_config(tmp_path)

    assert "File not found" in str(exc_info.value)


# --- Gate Tests ---


@pytest.mark.asyncio
async def test_auto_approve_gate() -> None:
    """AutoApproveGate always approves."""
    gate = AutoApproveGate()
    result = StageResult(stage="test", status="completed")

    decision = await gate.on_stage_complete("test", result)

    assert decision == "approve"


@pytest.mark.asyncio
async def test_require_success_gate_approves_success() -> None:
    """RequireSuccessGate approves stages without errors."""
    gate = RequireSuccessGate()
    result = StageResult(stage="test", status="completed", errors=[])

    decision = await gate.on_stage_complete("test", result)

    assert decision == "approve"


@pytest.mark.asyncio
async def test_require_success_gate_rejects_errors() -> None:
    """RequireSuccessGate rejects stages with errors."""
    gate = RequireSuccessGate()
    result = StageResult(stage="test", status="completed", errors=["Something went wrong"])

    decision = await gate.on_stage_complete("test", result)

    assert decision == "reject"


# --- Stage Registry Tests ---


def test_register_and_get_stage() -> None:
    """Stages can be registered and retrieved."""
    # Create a mock stage
    mock_stage = MagicMock()
    mock_stage.name = "test_stage"

    register_stage(mock_stage)

    assert get_stage("test_stage") is mock_stage


def test_get_stage_not_found() -> None:
    """get_stage returns None for unknown stages."""
    assert get_stage("nonexistent_stage") is None


def test_list_stages() -> None:
    """list_stages returns registered stage names."""
    # Register a stage for the test
    mock_stage = MagicMock()
    mock_stage.name = "listed_stage"
    register_stage(mock_stage)

    stages = list_stages()

    assert "listed_stage" in stages


# --- PipelineOrchestrator Tests ---


def test_orchestrator_initialization(tmp_path: Path) -> None:
    """Orchestrator initializes without project.yaml."""
    orchestrator = PipelineOrchestrator(tmp_path)

    assert orchestrator.config.name == "unnamed"


def test_orchestrator_with_config(tmp_path: Path) -> None:
    """Orchestrator loads project configuration."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: test_project
providers:
  default: ollama/qwen3:8b
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)

    assert orchestrator.config.name == "test_project"


@pytest.mark.asyncio
async def test_orchestrator_run_stage_not_found(tmp_path: Path) -> None:
    """Orchestrator raises error for unknown stage."""
    orchestrator = PipelineOrchestrator(tmp_path)

    with pytest.raises(StageNotFoundError):
        await orchestrator.run_stage("nonexistent")


@pytest.mark.asyncio
async def test_orchestrator_run_stage_with_mock(tmp_path: Path) -> None:
    """Orchestrator executes stage with mocked components."""
    # Register a mock stage
    mock_stage = MagicMock()
    mock_stage.name = "mock"
    mock_stage.execute = AsyncMock(
        return_value=(
            {"type": "mock", "version": 1},
            1,
            100,
        )
    )
    register_stage(mock_stage)

    orchestrator = PipelineOrchestrator(tmp_path)
    # Inject mock chat model directly
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "mock"

    # Run stage with user_prompt
    result = await orchestrator.run_stage("mock", {"user_prompt": "test prompt"})

    assert result.stage == "mock"
    assert result.llm_calls == 1
    assert result.tokens_used == 100
    assert result.artifact_path is not None
    assert result.artifact_path.exists()


@pytest.mark.asyncio
async def test_orchestrator_run_stage_with_errors(tmp_path: Path) -> None:
    """Orchestrator handles stage execution errors."""
    # Register a stage that raises an error
    mock_stage = MagicMock()
    mock_stage.name = "failing"
    mock_stage.execute = AsyncMock(side_effect=RuntimeError("Stage failed"))
    register_stage(mock_stage)

    orchestrator = PipelineOrchestrator(tmp_path)
    # Inject mock chat model directly
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "mock"

    result = await orchestrator.run_stage("failing", {"user_prompt": "test prompt"})

    assert result.stage == "failing"
    assert result.status == "failed"
    assert "Stage failed" in result.errors[0]


def test_orchestrator_get_status(tmp_path: Path) -> None:
    """Orchestrator reports pipeline status."""
    # Create project.yaml
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: status_test
pipeline:
  stages:
    - dream
    - seed
"""
    )

    # Create one artifact
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "dream.yaml").write_text("type: dream\nversion: 1")

    orchestrator = PipelineOrchestrator(tmp_path)
    status = orchestrator.get_status()

    assert status.project_name == "status_test"
    assert status.stages["dream"].status == "completed"
    assert status.stages["dream"].artifact_path is not None
    assert status.stages["seed"].status == "pending"
    assert status.stages["seed"].artifact_path is None


@pytest.mark.asyncio
async def test_orchestrator_gate_rejection(tmp_path: Path) -> None:
    """Orchestrator respects gate rejection."""
    # Register mock stage
    mock_stage = MagicMock()
    mock_stage.name = "gated"
    mock_stage.execute = AsyncMock(
        return_value=(
            {"type": "gated", "version": 1},
            1,
            50,
        )
    )
    register_stage(mock_stage)

    # Create rejecting gate
    class RejectGate:
        async def on_stage_complete(self, _stage: str, _result: StageResult) -> str:
            return "reject"

    orchestrator = PipelineOrchestrator(tmp_path, gate=RejectGate())
    # Inject mock chat model directly
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "mock"

    result = await orchestrator.run_stage("gated", {"user_prompt": "test prompt"})

    assert result.status == "pending_review"


@pytest.mark.asyncio
async def test_orchestrator_close_sync(tmp_path: Path) -> None:
    """Orchestrator closes chat model with sync close method."""
    orchestrator = PipelineOrchestrator(tmp_path)
    # Inject mock chat model directly
    mock_model = MagicMock()
    mock_model.close = MagicMock(return_value=None)  # Sync close
    orchestrator._chat_model = mock_model

    # Close orchestrator
    await orchestrator.close()

    assert orchestrator._chat_model is None
    mock_model.close.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_close_async(tmp_path: Path) -> None:
    """Orchestrator closes chat model with async close method."""
    orchestrator = PipelineOrchestrator(tmp_path)
    # Inject mock chat model with async close
    mock_model = MagicMock()
    mock_model.close = AsyncMock(return_value=None)  # Async close
    orchestrator._chat_model = mock_model

    # Close orchestrator
    await orchestrator.close()

    assert orchestrator._chat_model is None
    mock_model.close.assert_awaited_once()
