"""Tests for Cold Store role tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from questfoundry.runtime.cold_store import ColdStore
from questfoundry.runtime.state import Artifact
from questfoundry.runtime.tools.role import PromoteToCanon, ReadColdSot


class TestReadColdSot:
    """Tests for ReadColdSot tool."""

    @pytest.fixture
    def cold_store(self, tmp_path: Path) -> ColdStore:
        """Create a cold store with some content."""
        cold = ColdStore.create(tmp_path / "test.qfproj")
        cold.add_section("chapter_1", "First chapter content", {"title": "Chapter 1"})
        cold.add_section("chapter_2", "Second chapter content", {"title": "Chapter 2"})
        cold.create_snapshot("Initial chapters")
        yield cold
        cold.close()

    @pytest.fixture
    def tool(self, cold_store: ColdStore) -> ReadColdSot:
        """Create ReadColdSot tool with cold store."""
        t = ReadColdSot()
        t.cold_store = cold_store
        return t

    def test_read_section(self, tool: ReadColdSot) -> None:
        """Read a specific section."""
        result = json.loads(tool._run("chapter_1"))
        assert result["success"]
        assert result["section_id"] == "chapter_1"
        assert result["content"] == "First chapter content"
        assert result["metadata"]["title"] == "Chapter 1"

    def test_read_section_not_found(self, tool: ReadColdSot) -> None:
        """Read non-existent section returns error."""
        result = json.loads(tool._run("nonexistent"))
        assert not result["success"]
        assert "not found" in result["error"]
        assert "available_sections" in result

    def test_list_command(self, tool: ReadColdSot) -> None:
        """List all sections and snapshots."""
        result = json.loads(tool._run("list"))
        assert result["success"]
        assert result["section_count"] == 2
        assert "chapter_1" in result["sections"]
        assert "chapter_2" in result["sections"]
        assert result["snapshot_count"] == 1

    def test_latest_snapshot_command(self, tool: ReadColdSot) -> None:
        """Get latest snapshot."""
        result = json.loads(tool._run("latest_snapshot"))
        assert result["success"]
        assert "snapshot_id" in result
        assert result["section_count"] == 2
        assert "chapter_1" in result["section_ids"]
        assert "chapter_2" in result["section_ids"]

    def test_read_snapshot_by_id(self, tool: ReadColdSot, cold_store: ColdStore) -> None:
        """Read snapshot by ID."""
        latest = cold_store.get_latest_snapshot()
        assert latest is not None
        result = json.loads(tool._run(latest.id))
        assert result["success"]
        assert result["snapshot_id"] == latest.id

    def test_no_cold_store(self) -> None:
        """Error when cold store not available."""
        tool = ReadColdSot()
        tool.cold_store = None
        result = json.loads(tool._run("anything"))
        assert not result["success"]
        assert "not available" in result["error"]

    def test_empty_key(self, tool: ReadColdSot) -> None:
        """Error when key is empty."""
        result = json.loads(tool._run(""))
        assert not result["success"]
        assert "required" in result["error"]


class TestPromoteToCanon:
    """Tests for PromoteToCanon tool."""

    @pytest.fixture
    def cold_store(self, tmp_path: Path) -> ColdStore:
        """Create empty cold store."""
        cold = ColdStore.create(tmp_path / "test.qfproj")
        yield cold
        cold.close()

    @pytest.fixture
    def state_with_artifacts(self) -> dict:
        """Create state with hot store artifacts."""
        artifact = Artifact(
            id="scene_001",
            type="scene",
            status="draft",
            created_by="scene_smith",
            data={
                "title": "Opening Scene",
                "content": "The story begins in a dark forest...",
                "tags": ["intro", "forest"],
            },
        )
        return {
            "hot_store": {
                "scene_001": artifact,
                "raw_data": {"key": "value", "prose": "Some prose content"},
            },
            "cold_store": {},
        }

    @pytest.fixture
    def gatekeeper_tool(
        self, state_with_artifacts: dict, cold_store: ColdStore
    ) -> PromoteToCanon:
        """Create PromoteToCanon tool for Gatekeeper."""
        tool = PromoteToCanon()
        tool.state = state_with_artifacts
        tool.cold_store = cold_store
        tool.role_id = "gatekeeper"
        return tool

    def test_promote_artifact(
        self, gatekeeper_tool: PromoteToCanon, cold_store: ColdStore
    ) -> None:
        """Promote an artifact to cold store."""
        result = json.loads(gatekeeper_tool._run(["scene_001"]))
        assert result["success"]
        assert "scene_001" in result["promoted"]
        assert result["snapshot_id"] is not None

        # Verify in cold store
        section = cold_store.get_section("scene_001")
        assert section is not None
        assert "dark forest" in section.content
        assert section.metadata["source_artifact_id"] == "scene_001"

    def test_promote_multiple(
        self, gatekeeper_tool: PromoteToCanon, cold_store: ColdStore
    ) -> None:
        """Promote multiple artifacts."""
        result = json.loads(gatekeeper_tool._run(["scene_001", "raw_data"]))
        assert result["success"]
        assert len(result["promoted"]) == 2

        # Both should be in cold store
        assert cold_store.get_section("scene_001") is not None
        assert cold_store.get_section("raw_data") is not None

    def test_promote_without_snapshot(
        self, gatekeeper_tool: PromoteToCanon, cold_store: ColdStore
    ) -> None:
        """Promote without creating snapshot."""
        result = json.loads(
            gatekeeper_tool._run(["scene_001"], create_snapshot=False)
        )
        assert result["success"]
        assert result["snapshot_id"] is None

        # Section should exist but no snapshot
        assert cold_store.get_section("scene_001") is not None
        assert cold_store.get_latest_snapshot() is None

    def test_promote_with_description(
        self, gatekeeper_tool: PromoteToCanon, cold_store: ColdStore
    ) -> None:
        """Promote with custom snapshot description."""
        result = json.loads(
            gatekeeper_tool._run(
                ["scene_001"],
                snapshot_description="Opening scene approved",
            )
        )
        assert result["success"]

        snapshot = cold_store.get_latest_snapshot()
        assert snapshot is not None
        assert snapshot.description == "Opening scene approved"

    def test_promote_nonexistent(self, gatekeeper_tool: PromoteToCanon) -> None:
        """Promote non-existent artifact reports error."""
        result = json.loads(gatekeeper_tool._run(["nonexistent"]))
        assert not result["success"]
        assert result["errors"] is not None
        assert any("not found" in e for e in result["errors"])

    def test_non_gatekeeper_blocked(
        self, state_with_artifacts: dict, cold_store: ColdStore
    ) -> None:
        """Non-Gatekeeper roles cannot promote."""
        tool = PromoteToCanon()
        tool.state = state_with_artifacts
        tool.cold_store = cold_store
        tool.role_id = "plotwright"  # Not gatekeeper!

        result = json.loads(tool._run(["scene_001"]))
        assert not result["success"]
        assert "cannot promote" in result["error"]
        assert "plotwright" in result["error"]

    def test_no_cold_store(self, state_with_artifacts: dict) -> None:
        """Error when cold store not available."""
        tool = PromoteToCanon()
        tool.state = state_with_artifacts
        tool.cold_store = None
        tool.role_id = "gatekeeper"

        result = json.loads(tool._run(["scene_001"]))
        assert not result["success"]
        assert "not available" in result["error"]

    def test_empty_artifact_ids(self, gatekeeper_tool: PromoteToCanon) -> None:
        """Error when artifact_ids is empty."""
        result = json.loads(gatekeeper_tool._run([]))
        assert not result["success"]
        assert "required" in result["error"]


class TestRoleToolIntegration:
    """Integration tests for role tools with cold store."""

    @pytest.fixture
    def cold_store(self, tmp_path: Path) -> ColdStore:
        """Create cold store."""
        cold = ColdStore.create(tmp_path / "test.qfproj")
        yield cold
        cold.close()

    def test_full_workflow(self, cold_store: ColdStore) -> None:
        """Test full hot -> cold workflow."""
        # 1. Create artifact in hot store (simulating role work)
        artifact = Artifact(
            id="hook_001",
            type="hook_card",
            status="draft",
            created_by="showrunner",
            data={
                "title": "The Mystery Letter",
                "content": "A mysterious letter arrives...",
            },
        )
        state = {"hot_store": {"hook_001": artifact}}

        # 2. Gatekeeper promotes to cold
        promote_tool = PromoteToCanon()
        promote_tool.state = state
        promote_tool.cold_store = cold_store
        promote_tool.role_id = "gatekeeper"

        result = json.loads(promote_tool._run(["hook_001"]))
        assert result["success"]
        snapshot_id = result["snapshot_id"]

        # 3. Any role can read from cold
        read_tool = ReadColdSot()
        read_tool.cold_store = cold_store

        # Read section
        result = json.loads(read_tool._run("hook_001"))
        assert result["success"]
        assert "mysterious letter" in result["content"]

        # Read snapshot
        result = json.loads(read_tool._run(snapshot_id))
        assert result["success"]
        assert "hook_001" in result["section_ids"]
