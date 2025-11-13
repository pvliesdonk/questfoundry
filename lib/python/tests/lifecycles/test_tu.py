"""Tests for TU lifecycle"""

import pytest

from questfoundry.lifecycles import StateTransitionError, TULifecycle


def test_tu_lifecycle_initial_state():
    """Test TU starts in hot-proposed state"""
    lifecycle = TULifecycle()
    assert lifecycle.current_state == "hot-proposed"


def test_tu_full_successful_lifecycle():
    """Test complete TU lifecycle: hot-proposed → cold-merged"""
    lifecycle = TULifecycle()

    # hot-proposed → stabilizing
    lifecycle.begin_stabilization(reason="Starting refinement")
    assert lifecycle.current_state == "stabilizing"

    # stabilizing → gatecheck
    lifecycle.request_gatecheck(reason="Ready for validation")
    assert lifecycle.current_state == "gatecheck"

    # gatecheck → cold-merged (with passing data)
    lifecycle.merge_to_cold(data={"gatecheck_passed": True}, reason="All bars green")
    assert lifecycle.current_state == "cold-merged"

    # Check history
    assert len(lifecycle.history) == 3


def test_tu_gatecheck_failure_path():
    """Test TU returns to stabilization after failed gatecheck"""
    lifecycle = TULifecycle()

    lifecycle.begin_stabilization()
    lifecycle.request_gatecheck()

    # Gatecheck fails
    lifecycle.return_to_stabilization(reason="Quality bar: Integrity failed")
    assert lifecycle.current_state == "stabilizing"

    # Can try again
    lifecycle.request_gatecheck(reason="Issues fixed")
    assert lifecycle.current_state == "gatecheck"


def test_tu_cannot_merge_without_passing_gatecheck():
    """Test TU cannot merge to cold without passing gatecheck"""
    lifecycle = TULifecycle()

    lifecycle.begin_stabilization()
    lifecycle.request_gatecheck()

    # Try to merge without gatecheck_passed flag
    with pytest.raises(StateTransitionError):
        lifecycle.merge_to_cold(data={})

    # Try to merge with gatecheck_passed=False
    with pytest.raises(StateTransitionError):
        lifecycle.merge_to_cold(data={"gatecheck_passed": False})


def test_tu_can_return_to_proposed():
    """Test TU can return to proposed from stabilizing"""
    lifecycle = TULifecycle()

    lifecycle.begin_stabilization()
    lifecycle.return_to_proposed(reason="Needs major rework")

    assert lifecycle.current_state == "hot-proposed"


def test_tu_invalid_transitions():
    """Test invalid transitions raise errors"""
    lifecycle = TULifecycle()

    # Can't go directly from hot-proposed to gatecheck
    with pytest.raises(StateTransitionError):
        lifecycle.request_gatecheck()

    # Can't go directly from hot-proposed to cold-merged
    with pytest.raises(StateTransitionError):
        lifecycle.merge_to_cold(data={"gatecheck_passed": True})

    # Can't return to stabilization from cold-merged state
    lifecycle = TULifecycle()
    lifecycle.begin_stabilization()
    lifecycle.request_gatecheck()
    lifecycle.merge_to_cold(data={"gatecheck_passed": True})

    with pytest.raises(StateTransitionError):
        lifecycle.return_to_stabilization()


def test_tu_valid_transitions_from_hot_proposed():
    """Test getting valid transitions from hot-proposed state"""
    lifecycle = TULifecycle()

    valid = lifecycle.get_valid_transitions()
    assert valid == ["stabilizing"]


def test_tu_valid_transitions_from_stabilizing():
    """Test getting valid transitions from stabilizing state"""
    lifecycle = TULifecycle()
    lifecycle.begin_stabilization()

    valid = lifecycle.get_valid_transitions()
    assert set(valid) == {"gatecheck", "hot-proposed"}


def test_tu_valid_transitions_from_gatecheck():
    """Test getting valid transitions from gatecheck state"""
    lifecycle = TULifecycle()
    lifecycle.begin_stabilization()
    lifecycle.request_gatecheck()

    valid = lifecycle.get_valid_transitions()
    # cold-merged requires validation, so both should be listed
    assert set(valid) == {"cold-merged", "stabilizing"}


def test_tu_states_are_correct():
    """Test all expected states are defined"""
    expected_states = {
        "hot-proposed",
        "stabilizing",
        "gatecheck",
        "cold-merged",
    }
    assert TULifecycle.STATES == expected_states


def test_tu_with_existing_state():
    """Test creating TU lifecycle with existing state"""
    lifecycle = TULifecycle(current_state="stabilizing")
    assert lifecycle.current_state == "stabilizing"

    # Should be able to continue from that state
    lifecycle.request_gatecheck()
    assert lifecycle.current_state == "gatecheck"


def test_tu_rework_cycle():
    """Test TU can go through multiple rework cycles"""
    lifecycle = TULifecycle()

    # First attempt
    lifecycle.begin_stabilization()  # 1
    lifecycle.request_gatecheck()  # 2
    lifecycle.return_to_stabilization(reason="Failed integrity")  # 3

    # Second attempt
    lifecycle.request_gatecheck(reason="Fixed issues")  # 4
    lifecycle.return_to_stabilization(reason="Failed style")  # 5

    # Third attempt - success
    lifecycle.request_gatecheck(reason="All issues resolved")  # 6
    lifecycle.merge_to_cold(data={"gatecheck_passed": True})  # 7

    assert lifecycle.current_state == "cold-merged"
    assert len(lifecycle.history) == 7
