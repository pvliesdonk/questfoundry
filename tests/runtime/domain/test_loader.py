"""
TDD tests for domain loader.

These tests define the expected behavior of the domain loader.
Implementation should make these tests pass.
"""

import json
from pathlib import Path


class TestLoadResult:
    """Tests for LoadResult dataclass."""

    def test_load_result_has_required_attributes(self):
        """LoadResult must have studio, errors, and warnings."""
        from questfoundry.runtime.domain.loader import LoadResult

        result = LoadResult(studio=None, errors=[], warnings=[])

        assert hasattr(result, "studio")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

    def test_load_result_success_when_no_errors(self):
        """LoadResult.success is True when no errors and studio present."""
        from questfoundry.runtime.domain.loader import LoadResult
        from questfoundry.runtime.models.base import Studio

        # Create minimal valid studio
        studio = Studio(id="test", name="Test Studio")
        result = LoadResult(studio=studio, errors=[], warnings=[])

        assert result.success is True

    def test_load_result_failure_when_errors(self):
        """LoadResult.success is False when errors present."""
        from questfoundry.runtime.domain.loader import LoadError, LoadResult

        error = LoadError(path="test.json", message="File not found", severity="error")
        result = LoadResult(studio=None, errors=[error], warnings=[])

        assert result.success is False

    def test_load_result_failure_when_no_studio(self):
        """LoadResult.success is False when studio is None."""
        from questfoundry.runtime.domain.loader import LoadResult

        result = LoadResult(studio=None, errors=[], warnings=[])

        assert result.success is False


class TestLoadError:
    """Tests for LoadError dataclass."""

    def test_load_error_has_required_attributes(self):
        """LoadError must have path, message, and severity."""
        from questfoundry.runtime.domain.loader import LoadError

        error = LoadError(path="test.json", message="Something wrong", severity="error")

        assert error.path == "test.json"
        assert error.message == "Something wrong"
        assert error.severity == "error"

    def test_load_error_severity_types(self):
        """LoadError severity must be 'error' or 'warning'."""
        from questfoundry.runtime.domain.loader import LoadError

        error = LoadError(path="test.json", message="Bad", severity="error")
        warning = LoadError(path="test.json", message="Meh", severity="warning")

        assert error.severity == "error"
        assert warning.severity == "warning"


class TestLoadStudio:
    """Tests for load_studio function."""

    async def test_load_studio_returns_load_result(self, domain_v4_path: Path):
        """load_studio must return a LoadResult."""
        from questfoundry.runtime.domain.loader import LoadResult, load_studio

        result = await load_studio(domain_v4_path)

        assert isinstance(result, LoadResult)

    async def test_load_studio_loads_valid_domain(self, domain_v4_path: Path):
        """load_studio successfully loads domain-v4/."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success, f"Expected success, got errors: {result.errors}"
        assert result.studio is not None
        assert result.studio.id == "questfoundry"
        assert result.studio.name == "QuestFoundry Interactive Fiction Studio"

    async def test_load_studio_resolves_agent_refs(self, domain_v4_path: Path):
        """load_studio resolves agent file references to actual agents."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success
        # Count actual agent files in domain to avoid hardcoding
        agent_files = list((domain_v4_path / "agents").glob("*.json"))
        assert len(result.studio.agents) == len(agent_files)
        # Check first agent is properly loaded
        showrunner = next((a for a in result.studio.agents if a.id == "showrunner"), None)
        assert showrunner is not None
        assert showrunner.name == "Showrunner"
        assert showrunner.is_entry_agent is True

    async def test_load_studio_resolves_store_refs(self, domain_v4_path: Path):
        """load_studio resolves store file references."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success
        # Count stores referenced in studio.json (not all files in directory)
        studio_json = json.loads((domain_v4_path / "studio.json").read_text())
        store_refs = studio_json.get("stores", [])
        assert len(result.studio.stores) == len(store_refs)
        # Check workspace store
        workspace = next((s for s in result.studio.stores if s.id == "workspace"), None)
        assert workspace is not None

    async def test_load_studio_resolves_tool_refs(self, domain_v4_path: Path):
        """load_studio resolves tool file references."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success
        # Count actual tool files in domain to avoid hardcoding
        tool_files = list((domain_v4_path / "tools").glob("*.json"))
        assert len(result.studio.tools) == len(tool_files)

    async def test_load_studio_resolves_playbook_refs(self, domain_v4_path: Path):
        """load_studio resolves playbook file references."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success
        # Count actual playbook files in domain to avoid hardcoding
        playbook_files = list((domain_v4_path / "playbooks").glob("*.json"))
        assert len(result.studio.playbooks) == len(playbook_files)

    async def test_load_studio_resolves_artifact_type_refs(self, domain_v4_path: Path):
        """load_studio resolves artifact type file references."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success
        # Count actual artifact type files in domain to avoid hardcoding
        artifact_type_files = list((domain_v4_path / "artifact-types").glob("*.json"))
        assert len(result.studio.artifact_types) == len(artifact_type_files)

    async def test_load_studio_missing_directory_returns_error(self, tmp_path: Path):
        """load_studio returns error for non-existent directory."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(tmp_path / "nonexistent")

        assert not result.success
        assert len(result.errors) > 0
        assert any(
            "not found" in e.message.lower() or "not exist" in e.message.lower()
            for e in result.errors
        )

    async def test_load_studio_missing_studio_json_returns_error(self, tmp_path: Path):
        """load_studio returns error when studio.json is missing."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(tmp_path)

        assert not result.success
        assert any("studio.json" in e.message for e in result.errors)

    async def test_load_studio_missing_referenced_file_returns_error(self, tmp_path: Path):
        """load_studio returns error for missing referenced files."""
        from questfoundry.runtime.domain.loader import load_studio

        # Create studio.json with bad reference
        studio_data = {
            "id": "test",
            "name": "Test Studio",
            "agents": ["agents/missing.json"],
        }
        (tmp_path / "studio.json").write_text(json.dumps(studio_data))

        result = await load_studio(tmp_path)

        assert not result.success
        assert any("missing.json" in e.message for e in result.errors)

    async def test_load_studio_collects_multiple_errors(self, tmp_path: Path):
        """load_studio collects all errors, doesn't stop at first."""
        from questfoundry.runtime.domain.loader import load_studio

        # Create studio.json with multiple bad references
        studio_data = {
            "id": "test",
            "name": "Test Studio",
            "agents": ["agents/missing1.json", "agents/missing2.json"],
            "stores": ["stores/missing.json"],
        }
        (tmp_path / "studio.json").write_text(json.dumps(studio_data))

        result = await load_studio(tmp_path)

        assert not result.success
        # Should have errors for all missing files
        assert len(result.errors) >= 3


class TestEntryAgents:
    """Tests for entry agent handling."""

    async def test_studio_has_entry_agents(self, domain_v4_path: Path):
        """Studio should have entry_agents mapping."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success
        assert result.studio.entry_agents is not None
        assert "authoring" in result.studio.entry_agents
        assert result.studio.entry_agents["authoring"] == "showrunner"

    async def test_entry_agents_are_marked(self, domain_v4_path: Path):
        """Agents referenced as entry should have is_entry_agent=True."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)

        assert result.success
        showrunner = next((a for a in result.studio.agents if a.id == "showrunner"), None)
        player_narrator = next((a for a in result.studio.agents if a.id == "player_narrator"), None)

        assert showrunner is not None
        assert showrunner.is_entry_agent is True
        assert player_narrator is not None
        assert player_narrator.is_entry_agent is True
