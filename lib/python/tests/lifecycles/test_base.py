"""Tests for base lifecycle"""

import pytest

from questfoundry.lifecycles import Lifecycle, StateTransitionError


class SimpleLifecycle(Lifecycle):
    """Simple lifecycle for testing"""

    STATES = {"draft", "review", "published"}
    INITIAL_STATE = "draft"
    TRANSITIONS = {
        ("draft", "review"): None,
        ("review", "published"): None,
        ("review", "draft"): None,  # Can go back
    }


def test_lifecycle_initialization():
    """Test lifecycle initializes with initial state"""
    lifecycle = SimpleLifecycle()
    assert lifecycle.current_state == "draft"
    assert lifecycle.history == []


def test_lifecycle_with_custom_initial_state():
    """Test lifecycle can start in custom state"""
    lifecycle = SimpleLifecycle(current_state="review")
    assert lifecycle.current_state == "review"


def test_lifecycle_invalid_initial_state():
    """Test lifecycle rejects invalid initial state"""
    with pytest.raises(ValueError, match="Invalid state"):
        SimpleLifecycle(current_state="invalid")


def test_can_transition_to_valid():
    """Test can_transition_to returns True for valid transitions"""
    lifecycle = SimpleLifecycle()
    assert lifecycle.can_transition_to("review") is True


def test_can_transition_to_invalid():
    """Test can_transition_to returns False for invalid transitions"""
    lifecycle = SimpleLifecycle()
    assert lifecycle.can_transition_to("published") is False


def test_can_transition_to_same_state():
    """Test can always transition to same state"""
    lifecycle = SimpleLifecycle()
    assert lifecycle.can_transition_to("draft") is True


def test_can_transition_to_nonexistent_state():
    """Test can_transition_to returns False for nonexistent states"""
    lifecycle = SimpleLifecycle()
    assert lifecycle.can_transition_to("nonexistent") is False


def test_transition_to_valid():
    """Test valid transition updates state"""
    lifecycle = SimpleLifecycle()
    lifecycle.transition_to("review")
    assert lifecycle.current_state == "review"


def test_transition_to_invalid():
    """Test invalid transition raises error"""
    lifecycle = SimpleLifecycle()
    with pytest.raises(StateTransitionError, match="Cannot transition"):
        lifecycle.transition_to("published")


def test_transition_records_history():
    """Test transitions are recorded in history"""
    lifecycle = SimpleLifecycle()
    lifecycle.transition_to("review", reason="Ready for review")

    assert len(lifecycle.history) == 1
    assert lifecycle.history[0]["from_state"] == "draft"
    assert lifecycle.history[0]["to_state"] == "review"
    assert lifecycle.history[0]["reason"] == "Ready for review"
    assert "timestamp" in lifecycle.history[0]


def test_multiple_transitions():
    """Test multiple transitions work correctly"""
    lifecycle = SimpleLifecycle()
    lifecycle.transition_to("review")
    lifecycle.transition_to("published")

    assert lifecycle.current_state == "published"
    assert len(lifecycle.history) == 2


def test_get_valid_transitions():
    """Test getting valid next states"""
    lifecycle = SimpleLifecycle()

    # From draft, can go to review
    valid = lifecycle.get_valid_transitions()
    assert valid == ["review"]

    # From review, can go to draft or published
    lifecycle.transition_to("review")
    valid = lifecycle.get_valid_transitions()
    assert set(valid) == {"draft", "published"}


def test_get_history():
    """Test getting history copy"""
    lifecycle = SimpleLifecycle()
    lifecycle.transition_to("review")

    history = lifecycle.get_history()
    assert len(history) == 1

    # Modifying copy shouldn't affect original
    history.append({"test": "data"})
    assert len(lifecycle.history) == 1


def test_lifecycle_with_validation():
    """Test lifecycle with validation function"""

    def validate_review(data):
        return data.get("approved", False)

    class ValidatedLifecycle(Lifecycle):
        STATES = {"draft", "published"}
        INITIAL_STATE = "draft"
        TRANSITIONS = {
            ("draft", "published"): validate_review,
        }

    lifecycle = ValidatedLifecycle()

    # Should fail without approval
    assert lifecycle.can_transition_to("published", {}) is False
    with pytest.raises(StateTransitionError):
        lifecycle.transition_to("published", {})

    # Should succeed with approval
    assert lifecycle.can_transition_to("published", {"approved": True}) is True
    lifecycle.transition_to("published", {"approved": True})
    assert lifecycle.current_state == "published"
