"""Tests for the runtime state model."""

from questfoundry.runtime import (
    Artifact,
    Intent,
    StudioState,
    create_initial_state,
)


def test_create_initial_state() -> None:
    """Test creating initial state."""
    state = create_initial_state("story_spark")

    assert state["loop_id"] == "story_spark"
    assert state["iteration"] == 0
    assert state["hot_store"] == {}
    assert state["cold_store"] == {}
    assert len(state["messages"]) == 0


def test_create_initial_state_with_input() -> None:
    """Test creating initial state with user input."""
    state = create_initial_state("story_spark", "Create a mystery story")

    assert len(state["messages"]) == 1
    assert "mystery" in state["messages"][0].content.lower()


def test_artifact_model() -> None:
    """Test artifact creation."""
    artifact = Artifact(
        id="hook-001",
        type="hook_card",
        data={"title": "Test Hook"},
    )

    assert artifact.id == "hook-001"
    assert artifact.status == "draft"
    assert artifact.data["title"] == "Test Hook"


def test_intent_model() -> None:
    """Test intent creation."""
    intent = Intent(
        type="handoff",
        source_role="showrunner",
        status="completed",
    )

    assert intent.type == "handoff"
    assert intent.source_role == "showrunner"


def test_studio_state_typing() -> None:
    """Test that StudioState is a proper TypedDict."""
    # This should type-check correctly
    state: StudioState = {
        "hot_store": {},
        "cold_store": {},
        "messages": [],
        "current_role": "showrunner",
        "pending_intents": [],
        "loop_id": "test",
        "iteration": 0,
        "metadata": {},
    }

    assert state["current_role"] == "showrunner"
