"""Tests for PlaybookLoader."""

import pytest

from questfoundry.runtime.delegation import PlaybookLoader
from questfoundry.runtime.models.base import Playbook


class TestPlaybookLoader:
    """Tests for PlaybookLoader class."""

    @pytest.fixture
    def sample_playbooks_dict(self) -> dict:
        """Sample playbook definitions as dicts."""
        return {
            "story_spark": {
                "id": "story_spark",
                "name": "Story Spark",
                "max_rework_cycles": 3,
                "entry_phase": "topology_design",
                "phases": {
                    "topology_design": {
                        "name": "Topology Design",
                        "is_rework_target": False,
                    },
                    "brief_creation": {
                        "name": "Brief Creation",
                        "is_rework_target": True,
                    },
                    "preview_gate": {
                        "name": "Preview Gate",
                        "is_rework_target": False,
                    },
                },
            },
            "scene_weave": {
                "id": "scene_weave",
                "name": "Scene Weave",
                "max_rework_cycles": 2,
                "entry_phase": "drafting",
                "phases": {
                    "drafting": {
                        "name": "Drafting",
                        "is_rework_target": True,
                    },
                    "polish": {
                        "name": "Polish",
                        "is_rework_target": True,
                    },
                },
            },
        }

    @pytest.fixture
    def sample_playbook_models(self) -> list[Playbook]:
        """Sample Playbook model objects."""
        return [
            Playbook(
                id="story_spark",
                name="Story Spark",
                purpose="Design story structure",
                max_rework_cycles=3,
                entry_phase="topology_design",
                phases={
                    "topology_design": {
                        "name": "Topology Design",
                        "is_rework_target": False,
                    },
                    "brief_creation": {
                        "name": "Brief Creation",
                        "is_rework_target": True,
                    },
                },
            ),
            Playbook(
                id="scene_weave",
                name="Scene Weave",
                purpose="Draft prose",
                max_rework_cycles=2,
                entry_phase="drafting",
                phases={
                    "drafting": {
                        "name": "Drafting",
                        "is_rework_target": True,
                    },
                },
            ),
        ]

    def test_init_from_dict(self, sample_playbooks_dict: dict):
        """Test initialization from playbook dict."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.list_playbooks() == ["story_spark", "scene_weave"]

    def test_from_playbooks_factory(self, sample_playbook_models: list[Playbook]):
        """Test factory method from Playbook models."""
        loader = PlaybookLoader.from_playbooks(sample_playbook_models)

        assert set(loader.list_playbooks()) == {"story_spark", "scene_weave"}

    def test_get_max_rework_cycles(self, sample_playbooks_dict: dict):
        """Test getting max rework cycles."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.get_max_rework_cycles("story_spark") == 3
        assert loader.get_max_rework_cycles("scene_weave") == 2

    def test_get_max_rework_cycles_default(self, sample_playbooks_dict: dict):
        """Test default max rework cycles for missing playbook."""
        loader = PlaybookLoader(sample_playbooks_dict)

        # Missing playbook returns default
        assert loader.get_max_rework_cycles("nonexistent") == 3

    def test_is_rework_target_true(self, sample_playbooks_dict: dict):
        """Test identifying rework target phases."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.is_rework_target("story_spark", "brief_creation") is True
        assert loader.is_rework_target("scene_weave", "drafting") is True
        assert loader.is_rework_target("scene_weave", "polish") is True

    def test_is_rework_target_false(self, sample_playbooks_dict: dict):
        """Test non-rework target phases."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.is_rework_target("story_spark", "topology_design") is False
        assert loader.is_rework_target("story_spark", "preview_gate") is False

    def test_is_rework_target_missing_playbook(self, sample_playbooks_dict: dict):
        """Test rework target check for missing playbook."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.is_rework_target("nonexistent", "some_phase") is False

    def test_is_rework_target_missing_phase(self, sample_playbooks_dict: dict):
        """Test rework target check for missing phase."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.is_rework_target("story_spark", "nonexistent_phase") is False

    def test_get_phase(self, sample_playbooks_dict: dict):
        """Test getting phase definition."""
        loader = PlaybookLoader(sample_playbooks_dict)

        phase = loader.get_phase("story_spark", "brief_creation")
        assert phase is not None
        assert phase["name"] == "Brief Creation"
        assert phase["is_rework_target"] is True

    def test_get_phase_missing(self, sample_playbooks_dict: dict):
        """Test getting missing phase."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.get_phase("story_spark", "nonexistent") is None
        assert loader.get_phase("nonexistent", "brief_creation") is None

    def test_get_entry_phase(self, sample_playbooks_dict: dict):
        """Test getting entry phase."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.get_entry_phase("story_spark") == "topology_design"
        assert loader.get_entry_phase("scene_weave") == "drafting"

    def test_get_entry_phase_missing(self, sample_playbooks_dict: dict):
        """Test getting entry phase for missing playbook."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.get_entry_phase("nonexistent") is None

    def test_get_rework_target_phases(self, sample_playbooks_dict: dict):
        """Test getting all rework target phases."""
        loader = PlaybookLoader(sample_playbooks_dict)

        targets = loader.get_rework_target_phases("story_spark")
        assert targets == ["brief_creation"]

        targets = loader.get_rework_target_phases("scene_weave")
        assert set(targets) == {"drafting", "polish"}

    def test_get_rework_target_phases_none(self):
        """Test playbook with no rework targets."""
        loader = PlaybookLoader(
            {
                "no_rework": {
                    "id": "no_rework",
                    "name": "No Rework",
                    "phases": {
                        "phase1": {"name": "Phase 1"},
                        "phase2": {"name": "Phase 2"},
                    },
                }
            }
        )

        assert loader.get_rework_target_phases("no_rework") == []

    def test_get_rework_target_phases_missing(self, sample_playbooks_dict: dict):
        """Test rework targets for missing playbook."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.get_rework_target_phases("nonexistent") == []

    def test_get_playbook(self, sample_playbooks_dict: dict):
        """Test getting full playbook definition."""
        loader = PlaybookLoader(sample_playbooks_dict)

        playbook = loader.get_playbook("story_spark")
        assert playbook is not None
        assert playbook["id"] == "story_spark"
        assert playbook["max_rework_cycles"] == 3

    def test_get_playbook_missing(self, sample_playbooks_dict: dict):
        """Test getting missing playbook."""
        loader = PlaybookLoader(sample_playbooks_dict)

        assert loader.get_playbook("nonexistent") is None

    def test_get_playbook_dict(self, sample_playbooks_dict: dict):
        """Test getting raw playbook dictionary."""
        loader = PlaybookLoader(sample_playbooks_dict)

        playbook_dict = loader.get_playbook_dict()
        assert playbook_dict is sample_playbooks_dict

    def test_from_playbooks_preserves_phases(self, sample_playbook_models: list[Playbook]):
        """Test that from_playbooks preserves phase structure."""
        loader = PlaybookLoader.from_playbooks(sample_playbook_models)

        # Verify phases are properly transferred
        phase = loader.get_phase("story_spark", "brief_creation")
        assert phase is not None
        assert phase["is_rework_target"] is True

    def test_playbook_dict_compatible_with_nudger(self, sample_playbooks_dict: dict):
        """Test that playbook dict can be used with PlaybookNudger."""
        loader = PlaybookLoader(sample_playbooks_dict)

        # PlaybookNudger expects dict[str, dict[str, Any]]
        playbook_dict = loader.get_playbook_dict()
        assert isinstance(playbook_dict, dict)
        for pb_id, pb_def in playbook_dict.items():
            assert isinstance(pb_id, str)
            assert isinstance(pb_def, dict)
            assert "phases" in pb_def
