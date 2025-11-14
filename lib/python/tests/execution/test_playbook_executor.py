"""Tests for PlaybookExecutor."""

import json
from unittest.mock import Mock

import pytest

from questfoundry.execution.playbook_executor import PlaybookExecutor
from questfoundry.models.artifact import Artifact
from questfoundry.roles.base import Role, RoleContext, RoleResult


@pytest.fixture
def sample_manifest():
    """Sample playbook manifest for testing."""
    return {
        "manifest_version": "2.0.0",
        "playbook_id": "test_loop",
        "display_name": "Test Loop",
        "compiled_at": "2025-01-01T00:00:00Z",
        "source_files": ["spec/05-behavior/playbooks/test_loop.playbook.yaml"],
        "steps": [
            {
                "step_id": "step1",
                "description": "First step",
                "assigned_roles": ["test_role"],
                "consulted_roles": [],
                "procedure_content": "Do step 1",
                "artifacts_input": [],
                "artifacts_output": ["output1"],
                "validation_required": True,
            },
            {
                "step_id": "step2",
                "description": "Second step",
                "assigned_roles": ["test_role"],
                "consulted_roles": ["consultant"],
                "procedure_content": "Do step 2",
                "artifacts_input": ["output1"],
                "artifacts_output": ["output2"],
                "validation_required": False,
            },
        ],
        "raci": {
            "responsible": ["test_role"],
            "accountable": ["showrunner"],
            "consulted": ["consultant"],
            "informed": [],
        },
        "quality_bars": ["integrity", "presentation"],
    }


@pytest.fixture
def manifest_file(tmp_path, sample_manifest):
    """Create a temporary manifest file."""
    manifest_path = tmp_path / "test_loop.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(sample_manifest, f)
    return manifest_path


@pytest.fixture
def mock_role():
    """Mock role for testing."""
    role = Mock(spec=Role)
    role.execute.return_value = RoleResult(
        success=True,
        output="Test output",
        artifacts=[
            Artifact(
                type="output1",
                artifact_id="test-1",
                content="Test content",
            )
        ],
    )
    return role


class TestPlaybookExecutor:
    """Tests for PlaybookExecutor."""

    def test_init_with_manifest_path(self, manifest_file):
        """Test initialization with direct manifest path."""
        executor = PlaybookExecutor(manifest_path=manifest_file)

        assert executor.playbook_id == "test_loop"
        assert executor.display_name == "Test Loop"
        assert len(executor.steps) == 2
        assert executor.current_step_index == 0

    def test_init_with_playbook_id(self, tmp_path, manifest_file):
        """Test initialization with playbook ID and manifest dir."""
        executor = PlaybookExecutor(
            playbook_id="test_loop",
            manifest_dir=tmp_path,
        )

        assert executor.playbook_id == "test_loop"
        assert len(executor.steps) == 2

    def test_init_missing_params(self):
        """Test initialization fails without required params."""
        with pytest.raises(ValueError, match="Either manifest_path or playbook_id"):
            PlaybookExecutor()

    def test_execute_step(self, manifest_file, mock_role):
        """Test executing a single step."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role}

        result = executor.execute_step("step1", roles)

        assert result.success
        assert result.output == "Test output"
        assert len(result.artifacts) == 1
        assert result.artifacts[0].type == "output1"

        # Check that role.execute was called with correct context
        mock_role.execute.assert_called_once()
        call_args = mock_role.execute.call_args[0][0]
        assert isinstance(call_args, RoleContext)
        assert call_args.task == "First step"
        assert call_args.additional_context["step_id"] == "step1"
        assert call_args.additional_context["procedure"] == "Do step 1"

    def test_execute_step_not_found(self, manifest_file, mock_role):
        """Test executing non-existent step raises error."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role}

        with pytest.raises(ValueError, match="Step 'invalid' not found"):
            executor.execute_step("invalid", roles)

    def test_execute_step_missing_role(self, manifest_file):
        """Test executing step without required role raises error."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {}  # Empty roles dict

        with pytest.raises(ValueError, match="Required role 'test_role' not available"):
            executor.execute_step("step1", roles)

    def test_execute_full_loop(self, manifest_file, mock_role):
        """Test executing full playbook."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role, "consultant": Mock(spec=Role)}

        results, aggregated_artifacts = executor.execute_full_loop(roles)

        assert len(results) == 2
        assert "step1" in results
        assert "step2" in results
        assert results["step1"].success
        assert results["step2"].success
        assert len(aggregated_artifacts) == 2
        assert executor.latest_artifacts == aggregated_artifacts

        # Check that role was called for each step
        assert mock_role.execute.call_count == 2

    def test_execute_full_loop_mutates_passed_artifact_list(
        self,
        manifest_file,
        mock_role,
    ):
        """Artifacts list is mutated in place when provided."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role, "consultant": Mock(spec=Role)}
        initial_artifacts: list[Artifact] = []

        _, aggregated_artifacts = executor.execute_full_loop(
            roles,
            artifacts=initial_artifacts,
        )

        assert aggregated_artifacts is initial_artifacts
        assert len(initial_artifacts) == 2

    def test_execute_full_loop_stops_on_failure(self, manifest_file, mock_role):
        """Test that execution stops on step failure."""
        # Make first step fail
        mock_role.execute.side_effect = Exception("Step failed")

        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role}

        results, aggregated_artifacts = executor.execute_full_loop(roles)

        # Only first step should be attempted
        assert len(results) == 1
        assert "step1" in results
        assert not results["step1"].success
        assert "Step failed" in results["step1"].error
        assert aggregated_artifacts == []

    def test_get_raci(self, manifest_file):
        """Test getting RACI matrix."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        raci = executor.get_raci()

        assert raci["responsible"] == ["test_role"]
        assert raci["accountable"] == ["showrunner"]
        assert raci["consulted"] == ["consultant"]

    def test_get_quality_bars(self, manifest_file):
        """Test getting quality bars."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        quality_bars = executor.get_quality_bars()

        assert quality_bars == ["integrity", "presentation"]

    def test_get_source_files(self, manifest_file):
        """Test getting source files."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        source_files = executor.get_source_files()

        assert len(source_files) == 1
        assert "test_loop.playbook.yaml" in source_files[0]

    def test_step_results_context(self, manifest_file, mock_role):
        """Test that step results are passed to subsequent steps."""
        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role, "consultant": Mock(spec=Role)}

        executor.execute_step("step1", roles)
        executor.execute_step("step2", roles)

        # Check second call has step results from first
        second_call_args = mock_role.execute.call_args_list[1][0][0]
        assert "step1" in second_call_args.additional_context["step_results"]

    def test_step_results_store_failures(self, manifest_file, mock_role):
        """Ensure failed role executions are tracked in step_results."""
        mock_role.execute.return_value = RoleResult(
            success=False,
            output="",
            error="Validation failed",
            artifacts=[],
        )

        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role}

        result = executor.execute_step("step1", roles)

        assert not result.success
        assert executor.step_results["step1"].success is False

    def test_execute_step_creates_fallback_artifact(self, manifest_file, mock_role):
        """Generate synthetic artifact when role does not supply one."""
        mock_role.execute.return_value = RoleResult(
            success=True,
            output="Generated text",
            artifacts=[],
        )

        executor = PlaybookExecutor(manifest_path=manifest_file)
        roles = {"test_role": mock_role}

        result = executor.execute_step("step1", roles)

        assert result.artifacts
        assert result.artifacts[0].metadata["source_role"] == "test_role"
