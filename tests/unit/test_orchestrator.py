"""Tests for pipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.graph import Graph, apply_mutations
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


def _inject_mock_models(orchestrator: PipelineOrchestrator) -> MagicMock:
    """Inject mock models for all roles (creative, balanced, structured).

    This helper sets up mocks for the chat model and role-specific models
    that would otherwise require provider configuration (e.g., OLLAMA_HOST).

    Returns:
        The mock model instance used for all roles.
    """
    mock_model = MagicMock()
    # Creative role model
    orchestrator._creative_model = mock_model
    orchestrator._provider_name = "mock"
    orchestrator._model_name = "mock-model"
    # Balanced role model
    orchestrator._balanced_model = mock_model
    orchestrator._balanced_provider_name = "mock"
    orchestrator._balanced_model_name = "mock-model"
    # Structured role model
    orchestrator._structured_model = mock_model
    orchestrator._structured_provider_name = "mock"
    orchestrator._structured_model_name = "mock-model"
    return mock_model


# --- ProjectConfig Tests ---


def test_create_default_config() -> None:
    """create_default_config returns config with defaults."""
    config = create_default_config("test_project")

    assert config.name == "test_project"
    assert config.version == 1
    assert config.provider.name == "ollama"
    assert config.provider.model == "qwen3:4b-instruct-32k"
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
            "default": "openai/gpt-5-mini",
        },
    }
    config = ProjectConfig.from_dict(data)

    assert config.name == "full"
    assert config.version == 2
    assert config.provider.name == "openai"
    assert config.provider.model == "gpt-5-mini"
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
  default: ollama/qwen3:4b-instruct-32k
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
    orchestrator._creative_model = mock_model
    orchestrator._provider_name = "openai"
    orchestrator._model_name = "gpt-5-mini"

    # Manually populate model_info as _get_chat_model would
    from questfoundry.providers.model_info import get_model_info

    orchestrator._model_info = get_model_info("openai", "gpt-5-mini")

    assert orchestrator.model_info is not None
    assert orchestrator.model_info.context_window == 400_000
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
    _inject_mock_models(orchestrator)

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
    _inject_mock_models(orchestrator)

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
    _inject_mock_models(orchestrator)

    result = await orchestrator.run_stage("gated", {"user_prompt": "test prompt"})

    assert result.status == "pending_review"


@pytest.mark.asyncio
async def test_orchestrator_close_sync(tmp_path: Path) -> None:
    """Orchestrator closes chat model with sync close method."""
    orchestrator = PipelineOrchestrator(tmp_path)
    # Inject mock chat model directly
    mock_model = MagicMock()
    mock_model.close = MagicMock(return_value=None)  # Sync close
    orchestrator._creative_model = mock_model

    # Close orchestrator
    await orchestrator.close()

    assert orchestrator._creative_model is None
    mock_model.close.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_close_async(tmp_path: Path) -> None:
    """Orchestrator closes chat model with async close method."""
    orchestrator = PipelineOrchestrator(tmp_path)
    # Inject mock chat model with async close
    mock_model = MagicMock()
    mock_model.close = AsyncMock(return_value=None)  # Async close
    orchestrator._creative_model = mock_model

    # Close orchestrator
    await orchestrator.close()

    assert orchestrator._creative_model is None
    mock_model.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_close_unloads_different_ollama_structured_model(
    tmp_path: Path,
) -> None:
    """Close unloads structured model when different from creative (Ollama)."""
    from unittest.mock import AsyncMock, patch

    # Create project config with different Ollama models
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: test
providers:
  creative: ollama/mistral:7b
  structured: ollama/phi3:mini
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)

    # Inject mock structured model
    mock_structured = MagicMock()
    mock_structured.base_url = "http://localhost:11434"
    mock_structured.model = "phi3:mini"
    orchestrator._structured_model = mock_structured

    with patch(
        "questfoundry.pipeline.orchestrator.unload_ollama_model",
        new_callable=AsyncMock,
    ) as mock_unload:
        await orchestrator.close()

        # Should have called unload with the structured model
        mock_unload.assert_awaited_once_with(mock_structured)


@pytest.mark.asyncio
async def test_orchestrator_close_skips_unload_when_same_model(tmp_path: Path) -> None:
    """Close skips unload when structured and creative use same model."""
    from unittest.mock import AsyncMock, patch

    # Create project config with same model for all phases
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: test
providers:
  default: ollama/mistral:7b
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)
    orchestrator._structured_model = MagicMock()

    with patch(
        "questfoundry.pipeline.orchestrator.unload_ollama_model",
        new_callable=AsyncMock,
    ) as mock_unload:
        await orchestrator.close()

        # Should NOT have called unload (same model)
        mock_unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_close_skips_unload_for_non_ollama(tmp_path: Path) -> None:
    """Close skips unload when structured provider is not Ollama."""
    from unittest.mock import AsyncMock, patch

    # Create project config with non-Ollama structured provider
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: test
providers:
  creative: ollama/mistral:7b
  structured: openai/gpt-4o
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)
    orchestrator._structured_model = MagicMock()

    with patch(
        "questfoundry.pipeline.orchestrator.unload_ollama_model",
        new_callable=AsyncMock,
    ) as mock_unload:
        await orchestrator.close()

        # Should NOT have called unload (not Ollama)
        mock_unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_close_clears_all_model_references(tmp_path: Path) -> None:
    """Close sets _balanced_model and _structured_model to None."""
    orchestrator = PipelineOrchestrator(tmp_path)

    # Set all model references
    orchestrator._creative_model = MagicMock()
    orchestrator._balanced_model = MagicMock()
    orchestrator._structured_model = MagicMock()

    await orchestrator.close()

    # All should be cleared
    assert orchestrator._creative_model is None
    assert orchestrator._balanced_model is None
    assert orchestrator._structured_model is None


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
  default: ollama/qwen3:4b-instruct-32k
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
    _inject_mock_models(orchestrator)

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
    _inject_mock_models(orchestrator)

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
    _inject_mock_models(orchestrator)

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
    _inject_mock_models(orchestrator)

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
        provider_summarize_override="openai/gpt-5-mini",
        provider_serialize_override="openai/o1-mini",
    )

    assert orchestrator._provider_override == "ollama/default"
    # Legacy params are mapped to role-based attributes
    assert orchestrator._provider_creative_override == "ollama/discuss"
    assert orchestrator._provider_balanced_override == "openai/gpt-5-mini"
    assert orchestrator._provider_structured_override == "openai/o1-mini"


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


def test_orchestrator_legacy_env_var_alias(tmp_path: Path) -> None:
    """Legacy env vars (QF_PROVIDER_DISCUSS) work as aliases."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text("name: legacy_env_test\n")

    orchestrator = PipelineOrchestrator(tmp_path)

    with patch.dict("os.environ", {"QF_PROVIDER_DISCUSS": "openai/legacy-discuss"}):
        resolved = orchestrator._get_resolved_role_provider("creative")
        assert resolved == "openai/legacy-discuss"


def test_orchestrator_user_config_fallback(tmp_path: Path) -> None:
    """User config provides fallback when project config has no role-specific setting."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text("name: user_config_test\n")

    orchestrator = PipelineOrchestrator(tmp_path)

    # Inject user config with creative role set
    from questfoundry.pipeline.config import ProvidersConfig

    orchestrator._user_config = ProvidersConfig(
        default="openai/user-default",
        creative="openai/user-creative",
    )

    resolved = orchestrator._get_resolved_role_provider("creative")
    assert resolved == "openai/user-creative"


def test_orchestrator_project_config_beats_user_config(tmp_path: Path) -> None:
    """Role-specific project config takes precedence over user config."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: precedence_test
providers:
  default: ollama/proj-default
  creative: ollama/proj-creative
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)

    from questfoundry.pipeline.config import ProvidersConfig

    orchestrator._user_config = ProvidersConfig(
        default="openai/user-default",
        creative="openai/user-creative",
    )

    resolved = orchestrator._get_resolved_role_provider("creative")
    assert resolved == "ollama/proj-creative"


def test_orchestrator_user_role_beats_project_default(tmp_path: Path) -> None:
    """User role-specific config takes precedence over project default (no project role)."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: precedence_test
providers:
  default: ollama/proj-default
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)

    from questfoundry.pipeline.config import ProvidersConfig

    orchestrator._user_config = ProvidersConfig(
        default="openai/user-default",
        creative="openai/user-creative",
    )

    # User role-specific (level 6) should beat project default (level 7)
    resolved = orchestrator._get_resolved_role_provider("creative")
    assert resolved == "openai/user-creative"


# --- Tests for image provider precedence ---


def test_orchestrator_image_provider_cli_override(tmp_path: Path) -> None:
    """CLI --image-provider flag takes highest precedence."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: img_test
providers:
  default: ollama/qwen3:4b-instruct-32k
  image: openai/gpt-image-1
"""
    )

    orchestrator = PipelineOrchestrator(
        tmp_path,
        image_provider_override="placeholder",
    )

    with patch.dict("os.environ", {"QF_IMAGE_PROVIDER": "a1111"}):
        assert orchestrator._get_resolved_image_provider() == "placeholder"


def test_orchestrator_image_provider_env_precedence(tmp_path: Path) -> None:
    """QF_IMAGE_PROVIDER env var takes precedence over config."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: img_test
providers:
  default: ollama/qwen3:4b-instruct-32k
  image: openai/gpt-image-1
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)

    with patch.dict("os.environ", {"QF_IMAGE_PROVIDER": "placeholder"}):
        assert orchestrator._get_resolved_image_provider() == "placeholder"


def test_orchestrator_image_provider_config_fallback(tmp_path: Path) -> None:
    """Falls back to providers.image from project config."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text(
        """
name: img_test
providers:
  default: ollama/qwen3:4b-instruct-32k
  image: openai/gpt-image-1
"""
    )

    orchestrator = PipelineOrchestrator(tmp_path)

    with patch.dict("os.environ", {}, clear=True):
        assert orchestrator._get_resolved_image_provider() == "openai/gpt-image-1"


def test_orchestrator_image_provider_none_by_default(tmp_path: Path) -> None:
    """Returns None when no image provider configured at any level."""
    config_file = tmp_path / "project.yaml"
    config_file.write_text("name: img_test\n")

    orchestrator = PipelineOrchestrator(tmp_path)

    with patch.dict("os.environ", {}, clear=True):
        assert orchestrator._get_resolved_image_provider() is None


# ---------------------------------------------------------------------------
# Re-run snapshot restoration for orchestrator-managed stages
# ---------------------------------------------------------------------------

# Minimal DREAM artifact for apply_dream_mutations (uses upsert_node)
_DREAM_ARTIFACT = {
    "genre": "fantasy",
    "tone": ["dark"],
    "themes": ["betrayal"],
    "audience": "adult",
}

# Minimal BRAINSTORM artifact with one entity and one dilemma
_BRAINSTORM_ARTIFACT = {
    "entities": [
        {
            "entity_id": "test_hero",
            "entity_category": "character",
            "concept": "A brave hero",
        },
    ],
    "dilemmas": [
        {
            "dilemma_id": "fight_or_flee",
            "question": "Fight or flee?",
            "central_entity_ids": ["test_hero"],
            "answers": [
                {"answer_id": "fight", "answer_text": "Fight", "is_default_path": True},
                {"answer_id": "flee", "answer_text": "Flee", "is_default_path": False},
            ],
        },
    ],
}


def _simulate_mutation_block(
    project_path: Path,
    stage_name: str,
    artifact_data: dict[str, Any],
) -> Graph:
    """Simulate the orchestrator's mutation block with re-run detection.

    Calls the production _load_graph_for_mutation helper directly,
    then applies mutations and saves — matching the orchestrator's run_stage() flow.
    """
    from questfoundry.pipeline.orchestrator import _load_graph_for_mutation

    graph = _load_graph_for_mutation(project_path, stage_name)
    apply_mutations(graph, stage_name, artifact_data)
    graph.set_last_stage(stage_name)
    graph.save(project_path / "graph.json")
    return graph


class TestMutationRerunDetection:
    """Tests for re-run snapshot restoration in the orchestrator mutation block."""

    def test_first_run_saves_snapshot(self, tmp_path: Path) -> None:
        """First brainstorm run saves a pre-stage snapshot."""
        # Set up: graph with last_stage=dream (brainstorm prerequisite)
        graph = Graph.empty()
        graph.upsert_node("vision", {"type": "vision", "genre": "fantasy"})
        graph.set_last_stage("dream")
        graph.save(tmp_path / "graph.json")

        _simulate_mutation_block(tmp_path, "brainstorm", _BRAINSTORM_ARTIFACT)

        snapshot_path = tmp_path / "snapshots" / "pre-brainstorm.json"
        assert snapshot_path.exists()

        # Snapshot should contain dream-only state (no brainstorm nodes)
        snapshot_graph = Graph.load_from_file(snapshot_path)
        assert snapshot_graph.get_node("character::test_hero") is None
        assert snapshot_graph.get_last_stage() == "dream"

    def test_rerun_restores_snapshot(self, tmp_path: Path) -> None:
        """Re-running brainstorm restores the clean pre-stage snapshot."""
        # First run
        graph = Graph.empty()
        graph.upsert_node("vision", {"type": "vision", "genre": "fantasy"})
        graph.set_last_stage("dream")
        graph.save(tmp_path / "graph.json")

        _simulate_mutation_block(tmp_path, "brainstorm", _BRAINSTORM_ARTIFACT)

        # Verify first run succeeded
        result = Graph.load(tmp_path)
        assert result.get_node("character::test_hero") is not None
        assert result.get_last_stage() == "brainstorm"

        # Re-run with a DIFFERENT entity to prove we're not just appending
        rerun_artifact = {
            "entities": [
                {
                    "entity_id": "rerun_hero",
                    "entity_category": "character",
                    "concept": "A different hero",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "stay_or_go",
                    "question": "Stay or go?",
                    "central_entity_ids": ["rerun_hero"],
                    "answers": [
                        {"answer_id": "stay", "answer_text": "Stay", "is_default_path": True},
                        {"answer_id": "go", "answer_text": "Go", "is_default_path": False},
                    ],
                },
            ],
        }
        _simulate_mutation_block(tmp_path, "brainstorm", rerun_artifact)

        # New entities should exist, old should not
        result = Graph.load(tmp_path)
        assert result.get_node("character::rerun_hero") is not None
        assert result.get_node("character::test_hero") is None

    def test_rerun_after_downstream_stage(self, tmp_path: Path) -> None:
        """Brainstorm re-run after seed has completed restores pre-brainstorm snapshot."""
        # Set up: run dream, then brainstorm
        graph = Graph.empty()
        graph.upsert_node("vision", {"type": "vision", "genre": "fantasy"})
        graph.set_last_stage("dream")
        graph.save(tmp_path / "graph.json")

        _simulate_mutation_block(tmp_path, "brainstorm", _BRAINSTORM_ARTIFACT)

        # Simulate seed having run (just set last_stage, no real seed mutations)
        result = Graph.load(tmp_path)
        result.set_last_stage("seed")
        result.save(tmp_path / "graph.json")

        # Now re-run brainstorm — should not crash
        _simulate_mutation_block(tmp_path, "brainstorm", _BRAINSTORM_ARTIFACT)

        result = Graph.load(tmp_path)
        assert result.get_node("character::test_hero") is not None
        assert result.get_last_stage() == "brainstorm"

    def test_downstream_stage_after_rerun_is_first_run(self, tmp_path: Path) -> None:
        """Seed run after brainstorm re-run treats seed as first run (saves new snapshot)."""
        # Set up: dream → brainstorm → seed
        graph = Graph.empty()
        graph.upsert_node("vision", {"type": "vision", "genre": "fantasy"})
        graph.set_last_stage("dream")
        graph.save(tmp_path / "graph.json")

        _simulate_mutation_block(tmp_path, "brainstorm", _BRAINSTORM_ARTIFACT)

        # After brainstorm, last_stage == "brainstorm" which is seed's prerequisite.
        # So running dream mutations (via apply_mutations) for seed should be a first run.
        # We can't easily run real seed mutations without valid brainstorm data,
        # but we can verify the prerequisite logic: last_stage should == "brainstorm"
        result = Graph.load(tmp_path)
        assert result.get_last_stage() == "brainstorm"

        # Seed's prerequisite is "brainstorm", and last_stage == "brainstorm"
        # So this should save a new pre-seed snapshot (first run).
        from questfoundry.pipeline.orchestrator import _MUTATION_STAGE_PREREQUISITES

        prerequisite = _MUTATION_STAGE_PREREQUISITES["seed"]
        assert result.get_last_stage() == prerequisite  # First run condition

    def test_rerun_no_snapshot_raises_error(self, tmp_path: Path) -> None:
        """Re-run without snapshot raises ValueError with guidance."""
        # Set up: graph with last_stage=brainstorm but no snapshot file
        graph = Graph.empty()
        graph.upsert_node("vision", {"type": "vision", "genre": "fantasy"})
        graph.set_last_stage("brainstorm")
        graph.save(tmp_path / "graph.json")

        # No pre-brainstorm.json exists — simulate snapshot deletion
        snapshot_path = tmp_path / "snapshots" / "pre-brainstorm.json"
        assert not snapshot_path.exists()

        # Re-run without snapshot should raise, not silently corrupt
        with pytest.raises(ValueError, match="requires the pre-stage snapshot"):
            _simulate_mutation_block(tmp_path, "brainstorm", _BRAINSTORM_ARTIFACT)

    def test_dream_first_run_none_prerequisite(self, tmp_path: Path) -> None:
        """Dream first run works with None prerequisite (empty graph)."""
        # Empty graph — last_stage is None, dream prerequisite is None
        graph = Graph.empty()
        graph.save(tmp_path / "graph.json")

        _simulate_mutation_block(tmp_path, "dream", _DREAM_ARTIFACT)

        result = Graph.load(tmp_path)
        assert result.get_node("vision") is not None
        assert result.get_last_stage() == "dream"

        # Snapshot saved
        snapshot_path = tmp_path / "snapshots" / "pre-dream.json"
        assert snapshot_path.exists()
