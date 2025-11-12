"""Tests for agent-to-human communication callback functions."""

from questfoundry.roles.human_callback import batch_mode_callback


def test_batch_mode_callback_with_suggestions():
    """Test batch mode callback returns first suggestion."""
    question = "What color?"
    context = {"suggestions": ["red", "blue", "green"]}

    answer = batch_mode_callback(question, context)

    assert answer == "red"


def test_batch_mode_callback_without_suggestions():
    """Test batch mode callback returns empty string when no suggestions."""
    question = "What color?"
    context = {}

    answer = batch_mode_callback(question, context)

    assert answer == ""


def test_batch_mode_callback_with_empty_suggestions():
    """Test batch mode callback with empty suggestions list."""
    question = "What color?"
    context = {"suggestions": []}

    answer = batch_mode_callback(question, context)

    assert answer == ""


def test_batch_mode_callback_ignores_question():
    """Test batch mode callback ignores the question text."""
    question1 = "First question?"
    question2 = "Second question?"
    context = {"suggestions": ["answer"]}

    answer1 = batch_mode_callback(question1, context)
    answer2 = batch_mode_callback(question2, context)

    assert answer1 == answer2 == "answer"


def test_batch_mode_callback_context_structure():
    """Test batch mode callback works with various context structures."""
    # With suggestions
    assert batch_mode_callback("Q?", {"suggestions": ["a", "b"]}) == "a"

    # Without suggestions key
    assert batch_mode_callback("Q?", {}) == ""

    # With None suggestions
    assert batch_mode_callback("Q?", {"suggestions": None}) == ""

    # With other context data (should be ignored)
    assert (
        batch_mode_callback(
            "Q?", {"suggestions": ["yes"], "role": "test", "context": {}}
        )
        == "yes"
    )
