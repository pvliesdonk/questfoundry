"""Tests for artifact lifecycle management."""

import pytest

from questfoundry.runtime.storage.lifecycle import (
    ArtifactLifecycle,
    LifecycleManager,
    LifecycleState,
    LifecycleTransition,
)


class TestLifecycleState:
    """Tests for LifecycleState dataclass."""

    def test_from_dict_minimal(self):
        """Create state with minimal data."""
        state = LifecycleState.from_dict({"id": "draft"})
        assert state.id == "draft"
        assert state.name == "draft"  # Defaults to ID
        assert state.description is None
        assert state.terminal is False

    def test_from_dict_full(self):
        """Create state with full data."""
        state = LifecycleState.from_dict(
            {
                "id": "cold",
                "name": "Cold",
                "description": "Committed to canon",
                "terminal": True,
            }
        )
        assert state.id == "cold"
        assert state.name == "Cold"
        assert state.description == "Committed to canon"
        assert state.terminal is True


class TestLifecycleTransition:
    """Tests for LifecycleTransition dataclass."""

    def test_from_dict_minimal(self):
        """Create transition with minimal data."""
        trans = LifecycleTransition.from_dict(
            {
                "from": "draft",
                "to": "review",
            }
        )
        assert trans.from_state == "draft"
        assert trans.to_state == "review"
        assert trans.allowed_agents == []
        assert trans.requires_validation == []

    def test_from_dict_full(self):
        """Create transition with full data."""
        trans = LifecycleTransition.from_dict(
            {
                "from": "approved",
                "to": "cold",
                "allowed_agents": ["gatekeeper"],
                "requires_validation": ["integrity", "style"],
            }
        )
        assert trans.from_state == "approved"
        assert trans.to_state == "cold"
        assert trans.allowed_agents == ["gatekeeper"]
        assert trans.requires_validation == ["integrity", "style"]

    def test_is_agent_allowed_empty_list(self):
        """Empty allowed_agents means any agent allowed."""
        trans = LifecycleTransition(from_state="draft", to_state="review")
        assert trans.is_agent_allowed("any_agent")
        assert trans.is_agent_allowed(None)

    def test_is_agent_allowed_specific(self):
        """Specific allowed_agents filters by agent."""
        trans = LifecycleTransition(
            from_state="approved",
            to_state="cold",
            allowed_agents=["gatekeeper", "lorekeeper"],
        )
        assert trans.is_agent_allowed("gatekeeper")
        assert trans.is_agent_allowed("lorekeeper")
        assert not trans.is_agent_allowed("scene_smith")


class TestArtifactLifecycle:
    """Tests for ArtifactLifecycle."""

    @pytest.fixture
    def section_lifecycle_data(self):
        """Lifecycle data from section.json."""
        return {
            "states": [
                {"id": "draft", "name": "Draft", "description": "Section is being written"},
                {"id": "review", "name": "Review", "description": "Under review"},
                {"id": "gatecheck", "name": "Gatecheck", "description": "Awaiting validation"},
                {"id": "approved", "name": "Approved", "description": "Ready for cold"},
                {
                    "id": "cold",
                    "name": "Cold",
                    "description": "Committed to canon",
                    "terminal": True,
                },
            ],
            "initial_state": "draft",
            "transitions": [
                {"from": "draft", "to": "review"},
                {"from": "review", "to": "draft"},
                {"from": "review", "to": "gatecheck"},
                {"from": "gatecheck", "to": "draft"},
                {"from": "gatecheck", "to": "approved"},
                {
                    "from": "approved",
                    "to": "cold",
                    "requires_validation": ["integrity", "style"],
                },
            ],
        }

    def test_from_dict_none(self):
        """None data returns None."""
        lifecycle = ArtifactLifecycle.from_dict("test", None)
        assert lifecycle is None

    def test_from_dict_full(self, section_lifecycle_data):
        """Create lifecycle from full data."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        assert lifecycle.artifact_type_id == "section"
        assert lifecycle.initial_state == "draft"
        assert len(lifecycle.states) == 5
        assert len(lifecycle.transitions) == 6

    def test_get_state(self, section_lifecycle_data):
        """Get state by ID."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        draft = lifecycle.get_state("draft")
        assert draft is not None
        assert draft.name == "Draft"

        cold = lifecycle.get_state("cold")
        assert cold.terminal is True

        assert lifecycle.get_state("nonexistent") is None

    def test_is_terminal_state(self, section_lifecycle_data):
        """Check terminal state."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        assert not lifecycle.is_terminal_state("draft")
        assert not lifecycle.is_terminal_state("approved")
        assert lifecycle.is_terminal_state("cold")

    def test_get_valid_transitions(self, section_lifecycle_data):
        """Get valid transitions from a state."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        # From draft - can only go to review
        draft_trans = lifecycle.get_valid_transitions("draft")
        assert len(draft_trans) == 1
        assert draft_trans[0].to_state == "review"

        # From review - can go to draft or gatecheck
        review_trans = lifecycle.get_valid_transitions("review")
        assert len(review_trans) == 2
        targets = {t.to_state for t in review_trans}
        assert targets == {"draft", "gatecheck"}

        # From cold - no valid transitions (terminal)
        cold_trans = lifecycle.get_valid_transitions("cold")
        assert len(cold_trans) == 0

    def test_get_valid_transitions_with_agent(self):
        """Filter transitions by agent."""
        lifecycle = ArtifactLifecycle(
            artifact_type_id="test",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "approved": LifecycleState(id="approved", name="Approved"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="approved"),
                LifecycleTransition(
                    from_state="approved",
                    to_state="cold",
                    allowed_agents=["gatekeeper"],
                ),
            ],
            initial_state="draft",
        )

        # Gatekeeper can make all transitions
        gk_trans = lifecycle.get_valid_transitions("approved", agent_id="gatekeeper")
        assert len(gk_trans) == 1

        # Scene smith cannot transition approved -> cold
        ss_trans = lifecycle.get_valid_transitions("approved", agent_id="scene_smith")
        assert len(ss_trans) == 0

    def test_get_transition(self, section_lifecycle_data):
        """Get specific transition."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        trans = lifecycle.get_transition("approved", "cold")
        assert trans is not None
        assert trans.requires_validation == ["integrity", "style"]

        assert lifecycle.get_transition("draft", "cold") is None

    def test_validate_transition_success(self, section_lifecycle_data):
        """Valid transition passes."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        allowed, reason = lifecycle.validate_transition("draft", "review")
        assert allowed is True
        assert reason is None

    def test_validate_transition_from_terminal(self, section_lifecycle_data):
        """Cannot transition from terminal state."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        allowed, reason = lifecycle.validate_transition("cold", "draft")
        assert allowed is False
        assert "terminal" in reason

    def test_validate_transition_not_defined(self, section_lifecycle_data):
        """Reject undefined transition."""
        lifecycle = ArtifactLifecycle.from_dict("section", section_lifecycle_data)

        allowed, reason = lifecycle.validate_transition("draft", "cold")
        assert allowed is False
        assert "No transition defined" in reason

    def test_validate_transition_agent_not_allowed(self):
        """Reject if agent not in allowed_agents."""
        lifecycle = ArtifactLifecycle(
            artifact_type_id="test",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(
                    from_state="draft",
                    to_state="cold",
                    allowed_agents=["gatekeeper"],
                ),
            ],
            initial_state="draft",
        )

        allowed, reason = lifecycle.validate_transition("draft", "cold", agent_id="scene_smith")
        assert allowed is False
        assert "not allowed" in reason
        assert "gatekeeper" in reason


class TestLifecycleManager:
    """Tests for LifecycleManager."""

    @pytest.fixture
    def manager_with_lifecycles(self):
        """Create manager with test lifecycles."""
        manager = LifecycleManager()

        section_lifecycle = ArtifactLifecycle(
            artifact_type_id="section",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "review": LifecycleState(id="review", name="Review"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="review"),
                LifecycleTransition(
                    from_state="review",
                    to_state="cold",
                    allowed_agents=["gatekeeper"],
                    requires_validation=["integrity"],
                ),
            ],
            initial_state="draft",
        )
        manager.register_lifecycle(section_lifecycle)

        return manager

    def test_empty_manager(self):
        """Empty manager has no lifecycles."""
        manager = LifecycleManager()
        assert manager.get_lifecycle("section") is None
        assert not manager.has_lifecycle("section")

    def test_register_and_get(self, manager_with_lifecycles):
        """Register and retrieve lifecycle."""
        lifecycle = manager_with_lifecycles.get_lifecycle("section")
        assert lifecycle is not None
        assert lifecycle.artifact_type_id == "section"

    def test_has_lifecycle(self, manager_with_lifecycles):
        """Check lifecycle existence."""
        assert manager_with_lifecycles.has_lifecycle("section")
        assert not manager_with_lifecycles.has_lifecycle("unknown")

    def test_get_initial_state(self, manager_with_lifecycles):
        """Get initial state for artifact type."""
        assert manager_with_lifecycles.get_initial_state("section") == "draft"
        # Unknown type defaults to draft
        assert manager_with_lifecycles.get_initial_state("unknown") == "draft"

    def test_validate_transition(self, manager_with_lifecycles):
        """Validate transition via manager."""
        # Valid transition
        allowed, reason = manager_with_lifecycles.validate_transition("section", "draft", "review")
        assert allowed is True

        # Invalid - agent not allowed
        allowed, reason = manager_with_lifecycles.validate_transition(
            "section", "review", "cold", agent_id="scene_smith"
        )
        assert allowed is False

        # Unknown type - allow any transition
        allowed, reason = manager_with_lifecycles.validate_transition("unknown", "any", "state")
        assert allowed is True

    def test_get_required_validations(self, manager_with_lifecycles):
        """Get required validations for transition."""
        validations = manager_with_lifecycles.get_required_validations("section", "review", "cold")
        assert validations == ["integrity"]

        # No validations for other transitions
        validations = manager_with_lifecycles.get_required_validations("section", "draft", "review")
        assert validations == []

        # Unknown type returns empty
        validations = manager_with_lifecycles.get_required_validations("unknown", "any", "state")
        assert validations == []

    def test_from_artifact_types_dict(self):
        """Create manager from dict artifact types."""
        artifact_types = [
            {
                "id": "section",
                "lifecycle": {
                    "states": [
                        {"id": "draft", "name": "Draft"},
                        {"id": "cold", "name": "Cold", "terminal": True},
                    ],
                    "initial_state": "draft",
                    "transitions": [{"from": "draft", "to": "cold"}],
                },
            },
            {
                "id": "note",
                # No lifecycle
            },
        ]

        manager = LifecycleManager.from_artifact_types(artifact_types)

        assert manager.has_lifecycle("section")
        assert not manager.has_lifecycle("note")

    def test_from_artifact_types_pydantic_models(self):
        """Create manager from pydantic ArtifactType models with Lifecycle."""
        from questfoundry.runtime.models.base import (
            ArtifactType as ModelArtifactType,
        )
        from questfoundry.runtime.models.base import (
            Lifecycle as ModelLifecycle,
        )
        from questfoundry.runtime.models.base import (
            LifecycleState as ModelLifecycleState,
        )
        from questfoundry.runtime.models.base import (
            LifecycleTransition as ModelLifecycleTransition,
        )

        artifact_type = ModelArtifactType(
            id="section",
            name="Section",
            lifecycle=ModelLifecycle(
                states=[
                    ModelLifecycleState(id="draft", name="Draft"),
                    ModelLifecycleState(id="cold", name="Cold", terminal=True),
                ],
                initial_state="draft",
                transitions=[
                    ModelLifecycleTransition(
                        from_state="draft",
                        to_state="cold",
                        allowed_agents=["gatekeeper"],
                    )
                ],
            ),
        )

        manager = LifecycleManager.from_artifact_types([artifact_type])

        assert manager.has_lifecycle("section")
        lifecycle = manager.get_lifecycle("section")
        assert lifecycle is not None

        # The stored transition should be the storage-layer dataclass,
        # with is_agent_allowed available.
        trans = lifecycle.get_transition("draft", "cold")
        assert trans is not None
        assert trans.is_agent_allowed("gatekeeper")
