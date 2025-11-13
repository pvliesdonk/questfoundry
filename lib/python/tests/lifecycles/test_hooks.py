"""Tests for Hook lifecycle"""

import pytest

from questfoundry.lifecycles import HookLifecycle, StateTransitionError


def test_hook_lifecycle_initial_state():
    """Test hook starts in proposed state"""
    lifecycle = HookLifecycle()
    assert lifecycle.current_state == "proposed"


def test_hook_full_lifecycle():
    """Test complete hook lifecycle: proposed → canonized"""
    lifecycle = HookLifecycle()

    # proposed → accepted
    lifecycle.accept(reason="Good idea")
    assert lifecycle.current_state == "accepted"

    # accepted → in-progress
    lifecycle.start_work(reason="Beginning work")
    assert lifecycle.current_state == "in-progress"

    # in-progress → resolved
    lifecycle.resolve(reason="Work complete")
    assert lifecycle.current_state == "resolved"

    # resolved → canonized
    lifecycle.canonize(reason="Merged to cold")
    assert lifecycle.current_state == "canonized"

    # Check history
    assert len(lifecycle.history) == 4


def test_hook_rejection_path():
    """Test hook rejection from proposed"""
    lifecycle = HookLifecycle()

    lifecycle.reject(reason="Not aligned with project")
    assert lifecycle.current_state == "rejected"


def test_hook_deferral_paths():
    """Test hook can be deferred from multiple states"""
    # From proposed
    lifecycle1 = HookLifecycle()
    lifecycle1.defer(reason="Low priority")
    assert lifecycle1.current_state == "deferred"

    # From accepted
    lifecycle2 = HookLifecycle()
    lifecycle2.accept()
    lifecycle2.defer(reason="Resource constraints")
    assert lifecycle2.current_state == "deferred"

    # From in-progress
    lifecycle3 = HookLifecycle()
    lifecycle3.accept()
    lifecycle3.start_work()
    lifecycle3.defer(reason="Blocking issue")
    assert lifecycle3.current_state == "deferred"


def test_hook_reactivation():
    """Test reactivating deferred hook"""
    lifecycle = HookLifecycle()

    lifecycle.defer()
    lifecycle.reactivate(reason="Priority increased")

    assert lifecycle.current_state == "proposed"


def test_hook_invalid_transitions():
    """Test invalid transitions raise errors"""
    lifecycle = HookLifecycle()

    # Can't go directly from proposed to resolved
    with pytest.raises(StateTransitionError):
        lifecycle.resolve()

    # Can't reject from accepted state
    lifecycle.accept()
    with pytest.raises(StateTransitionError):
        lifecycle.reject()

    # Can't start work from proposed
    lifecycle = HookLifecycle()
    with pytest.raises(StateTransitionError):
        lifecycle.start_work()


def test_hook_cannot_canonize_unresolved():
    """Test can't canonize hook that isn't resolved"""
    lifecycle = HookLifecycle()

    lifecycle.accept()
    lifecycle.start_work()

    with pytest.raises(StateTransitionError):
        lifecycle.canonize()


def test_hook_valid_transitions_from_proposed():
    """Test getting valid transitions from proposed state"""
    lifecycle = HookLifecycle()

    valid = lifecycle.get_valid_transitions()
    assert set(valid) == {"accepted", "deferred", "rejected"}


def test_hook_valid_transitions_from_accepted():
    """Test getting valid transitions from accepted state"""
    lifecycle = HookLifecycle()
    lifecycle.accept()

    valid = lifecycle.get_valid_transitions()
    assert set(valid) == {"deferred", "in-progress"}


def test_hook_valid_transitions_from_in_progress():
    """Test getting valid transitions from in-progress state"""
    lifecycle = HookLifecycle()
    lifecycle.accept()
    lifecycle.start_work()

    valid = lifecycle.get_valid_transitions()
    assert set(valid) == {"deferred", "resolved"}


def test_hook_valid_transitions_from_resolved():
    """Test getting valid transitions from resolved state"""
    lifecycle = HookLifecycle()
    lifecycle.accept()
    lifecycle.start_work()
    lifecycle.resolve()

    valid = lifecycle.get_valid_transitions()
    assert valid == ["canonized"]


def test_hook_valid_transitions_from_deferred():
    """Test getting valid transitions from deferred state"""
    lifecycle = HookLifecycle()
    lifecycle.defer()

    valid = lifecycle.get_valid_transitions()
    assert valid == ["proposed"]


def test_hook_states_are_correct():
    """Test all expected states are defined"""
    expected_states = {
        "proposed",
        "accepted",
        "in-progress",
        "resolved",
        "canonized",
        "deferred",
        "rejected",
    }
    assert HookLifecycle.STATES == expected_states


def test_hook_with_existing_state():
    """Test creating hook lifecycle with existing state"""
    lifecycle = HookLifecycle(current_state="in-progress")
    assert lifecycle.current_state == "in-progress"

    # Should be able to continue from that state
    lifecycle.resolve()
    assert lifecycle.current_state == "resolved"
