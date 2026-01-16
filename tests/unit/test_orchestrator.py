"""Tests for pipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.graph import Graph
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


def test_orchestrator_model_info_before_run(tmp_path: Path) -> None:
    """Orchestrator model_info is None before first stage run."""
    orchestrator = PipelineOrchestrator(tmp_path)

    assert orchestrator.model_info is None


def test_orchestrator_model_info_after_model_creation(tmp_path: Path) -> None:
    """Orchestrator model_info is populated after model creation."""
    orchestrator = PipelineOrchestrator(tmp_path)
    # Inject mock chat model and model info directly
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "openai"
    orchestrator._model_name = "gpt-4o"

    # Manually populate model_info as _get_chat_model would
    from questfoundry.providers.model_info import get_model_info

    orchestrator._model_info = get_model_info("openai", "gpt-4o")

    assert orchestrator.model_info is not None
    assert orchestrator.model_info.context_window == 128_000
    assert orchestrator.model_info.supports_vision is True


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


# --- Post-Mutation Validation Integration Tests ---


@pytest.fixture
def project_with_graph(tmp_path: Path) -> Path:
    """Create project directory with initialized graph."""
    # Create project config
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: validation_test
providers:
  default: ollama/qwen3:8b
"""
    )

    # Create artifacts directory
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    # Initialize graph with some entities (as BRAINSTORM would)
    graph = Graph()
    graph.create_node(
        "entity::detective",
        {"type": "entity", "raw_id": "detective", "entity_type": "character"},
    )
    graph.save(tmp_path / "graph.json")

    return tmp_path


@pytest.mark.asyncio
async def test_orchestrator_calls_validate_invariants(project_with_graph: Path) -> None:
    """Orchestrator calls validate_invariants() after apply_mutations()."""
    # Register a stage with mutation handler
    mock_stage = MagicMock()
    mock_stage.name = "validation_stage"
    mock_stage.execute = AsyncMock(
        return_value=(
            {"type": "validation_stage", "version": 1},
            1,
            100,
        )
    )
    register_stage(mock_stage)

    orchestrator = PipelineOrchestrator(project_with_graph)
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "mock"

    # Patch has_mutation_handler to return True for our stage
    # and validate_invariants to track if it was called
    with (
        patch("questfoundry.pipeline.orchestrator.has_mutation_handler", return_value=True),
        patch("questfoundry.pipeline.orchestrator.apply_mutations") as mock_apply,
        patch.object(Graph, "validate_invariants", return_value=[]) as mock_validate,
    ):
        result = await orchestrator.run_stage("validation_stage", {"user_prompt": "test"})

    assert result.status == "completed"
    mock_apply.assert_called_once()
    mock_validate.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_rollback_on_graph_corruption(
    project_with_graph: Path,
) -> None:
    """Orchestrator rolls back on violations and returns failed result."""
    mock_stage = MagicMock()
    mock_stage.name = "corrupt_stage"
    mock_stage.execute = AsyncMock(
        return_value=(
            {"type": "corrupt_stage", "version": 1},
            1,
            100,
        )
    )
    register_stage(mock_stage)

    orchestrator = PipelineOrchestrator(project_with_graph)
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "mock"

    fake_violations = ["Dangling edge: entity::missing -> entity::nonexistent"]

    with (
        patch("questfoundry.pipeline.orchestrator.has_mutation_handler", return_value=True),
        patch("questfoundry.pipeline.orchestrator.apply_mutations"),
        patch("questfoundry.pipeline.orchestrator.save_snapshot"),
        patch("questfoundry.pipeline.orchestrator.rollback_to_snapshot") as mock_rollback,
        patch.object(Graph, "validate_invariants", return_value=fake_violations),
    ):
        result = await orchestrator.run_stage("corrupt_stage", {"user_prompt": "test"})

    # Verify rollback was called
    mock_rollback.assert_called_once_with(project_with_graph, "corrupt_stage")

    # Verify result indicates failure with corruption error
    assert result.status == "failed"
    assert any("Graph corruption" in err for err in result.errors)


@pytest.mark.asyncio
async def test_orchestrator_graph_restored_after_rollback(
    project_with_graph: Path,
) -> None:
    """Graph state is restored after rollback on corruption."""
    mock_stage = MagicMock()
    mock_stage.name = "rollback_test"
    mock_stage.execute = AsyncMock(
        return_value=(
            {"type": "rollback_test", "version": 1},
            1,
            100,
        )
    )
    register_stage(mock_stage)

    orchestrator = PipelineOrchestrator(project_with_graph)
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "mock"

    # Get initial graph state
    original_graph = Graph.load(project_with_graph)
    original_nodes = set(original_graph._data["nodes"].keys())

    fake_violations = ["Test violation"]

    # Track if rollback_to_snapshot was called
    rollback_called = False

    def mock_rollback(_project_path: Path, _stage: str) -> None:
        nonlocal rollback_called
        rollback_called = True
        # Simulate rollback by doing nothing (graph already at original state)

    with (
        patch("questfoundry.pipeline.orchestrator.has_mutation_handler", return_value=True),
        patch("questfoundry.pipeline.orchestrator.apply_mutations"),
        patch("questfoundry.pipeline.orchestrator.save_snapshot"),
        patch(
            "questfoundry.pipeline.orchestrator.rollback_to_snapshot",
            side_effect=mock_rollback,
        ),
        patch.object(Graph, "validate_invariants", return_value=fake_violations),
    ):
        # Orchestrator catches GraphCorruptionError and returns failed result
        result = await orchestrator.run_stage("rollback_test", {"user_prompt": "test"})

    assert rollback_called
    assert result.status == "failed"

    # Verify graph state unchanged (rollback would restore it)
    restored_graph = Graph.load(project_with_graph)
    restored_nodes = set(restored_graph._data["nodes"].keys())
    assert original_nodes == restored_nodes


@pytest.mark.asyncio
async def test_orchestrator_no_validation_without_mutation_handler(
    project_with_graph: Path,
) -> None:
    """Orchestrator skips validation for stages without mutation handlers."""
    mock_stage = MagicMock()
    mock_stage.name = "no_mutation_stage"
    mock_stage.execute = AsyncMock(
        return_value=(
            {"type": "no_mutation_stage", "version": 1},
            1,
            100,
        )
    )
    register_stage(mock_stage)

    orchestrator = PipelineOrchestrator(project_with_graph)
    mock_model = MagicMock()
    orchestrator._chat_model = mock_model
    orchestrator._provider_name = "mock"

    with (
        patch("questfoundry.pipeline.orchestrator.has_mutation_handler", return_value=False),
        patch("questfoundry.pipeline.orchestrator.apply_mutations") as mock_apply,
        patch.object(Graph, "validate_invariants") as mock_validate,
    ):
        result = await orchestrator.run_stage("no_mutation_stage", {"user_prompt": "test"})

    assert result.status == "completed"
    # Neither apply_mutations nor validate_invariants should be called
    mock_apply.assert_not_called()
    mock_validate.assert_not_called()


# --- Phase-Specific Provider Override Tests ---


def test_orchestrator_stores_phase_overrides(tmp_path: Path) -> None:
    """Orchestrator stores phase-specific CLI overrides."""
    # Create project config
    config_file = tmp_path / "project.yaml"
    config_file.write_text("name: override_test\n")

    orchestrator = PipelineOrchestrator(
        tmp_path,
        provider_override="ollama/default",
        provider_discuss_override="ollama/discuss",
        provider_summarize_override="openai/gpt-4o",
        provider_serialize_override="openai/o1-mini",
    )

    assert orchestrator._provider_override == "ollama/default"
    assert orchestrator._provider_discuss_override == "ollama/discuss"
    assert orchestrator._provider_summarize_override == "openai/gpt-4o"
    assert orchestrator._provider_serialize_override == "openai/o1-mini"


def test_orchestrator_discuss_override_precedence(tmp_path: Path) -> None:
    """Phase-specific discuss CLI override takes precedence over general override."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text("name: precedence_test\n")

    orchestrator = PipelineOrchestrator(
        tmp_path,
        provider_override="ollama/general",
        provider_discuss_override="openai/specific",
    )

    # Phase-specific should win over general
    resolved = orchestrator._get_resolved_discuss_provider()
    assert resolved == "openai/specific"


def test_orchestrator_discuss_fallback_to_general(tmp_path: Path) -> None:
    """Discuss falls back to general override when phase-specific not set."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text("name: fallback_test\n")

    orchestrator = PipelineOrchestrator(
        tmp_path,
        provider_override="ollama/general",
        # No provider_discuss_override
    )

    resolved = orchestrator._get_resolved_discuss_provider()
    assert resolved == "ollama/general"


def test_orchestrator_phase_provider_precedence(tmp_path: Path) -> None:
    """Phase-specific CLI overrides take precedence for summarize/serialize."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: phase_test
providers:
  default: ollama/config-default
  summarize: ollama/config-summarize
  serialize: ollama/config-serialize
"""
    )

    orchestrator = PipelineOrchestrator(
        tmp_path,
        provider_override="ollama/cli-general",
        provider_summarize_override="openai/cli-summarize",
        provider_serialize_override="openai/cli-serialize",
    )

    # Phase-specific CLI should win
    assert orchestrator._get_resolved_phase_provider("summarize") == "openai/cli-summarize"
    assert orchestrator._get_resolved_phase_provider("serialize") == "openai/cli-serialize"


def test_orchestrator_phase_provider_env_precedence(tmp_path: Path) -> None:
    """Phase-specific env vars take precedence over config but not CLI."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: env_test
providers:
  default: ollama/config-default
  summarize: ollama/config-summarize
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)

    # Test env var precedence (no CLI override)
    with patch.dict("os.environ", {"QF_PROVIDER_SUMMARIZE": "anthropic/env-summarize"}):
        resolved = orchestrator._get_resolved_phase_provider("summarize")
        assert resolved == "anthropic/env-summarize"


def test_orchestrator_phase_provider_full_precedence_chain(tmp_path: Path) -> None:
    """Full 6-level precedence chain for phase providers."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: chain_test
providers:
  default: ollama/level6
  serialize: ollama/level5
"""
    )

    # Level 1: Phase CLI wins
    orchestrator = PipelineOrchestrator(
        tmp_path,
        provider_override="ollama/level2",
        provider_serialize_override="ollama/level1",
    )
    with patch.dict(
        "os.environ",
        {"QF_PROVIDER_SERIALIZE": "ollama/level3", "QF_PROVIDER": "ollama/level4"},
    ):
        assert orchestrator._get_resolved_phase_provider("serialize") == "ollama/level1"

    # Level 2: General CLI wins (no phase CLI)
    orchestrator = PipelineOrchestrator(
        tmp_path,
        provider_override="ollama/level2",
    )
    with patch.dict(
        "os.environ",
        {"QF_PROVIDER_SERIALIZE": "ollama/level3", "QF_PROVIDER": "ollama/level4"},
    ):
        assert orchestrator._get_resolved_phase_provider("serialize") == "ollama/level2"

    # Level 3: Phase env wins (no CLI)
    orchestrator = PipelineOrchestrator(tmp_path)
    with patch.dict(
        "os.environ",
        {"QF_PROVIDER_SERIALIZE": "ollama/level3", "QF_PROVIDER": "ollama/level4"},
    ):
        assert orchestrator._get_resolved_phase_provider("serialize") == "ollama/level3"

    # Level 4: General env wins (no phase env)
    orchestrator = PipelineOrchestrator(tmp_path)
    with patch.dict("os.environ", {"QF_PROVIDER": "ollama/level4"}, clear=False):
        # Need to ensure QF_PROVIDER_SERIALIZE is not set
        import os

        orig = os.environ.pop("QF_PROVIDER_SERIALIZE", None)
        try:
            assert orchestrator._get_resolved_phase_provider("serialize") == "ollama/level4"
        finally:
            if orig:
                os.environ["QF_PROVIDER_SERIALIZE"] = orig

    # Level 5 & 6: Config (tested in test_config.py)
