"""Tests for analyze_story_graph tool."""

from questfoundry.runtime.tools.analyze_story_graph import _get_choice_target


class TestGetChoiceTarget:
    """Tests for _get_choice_target helper function."""

    def test_target_anchor_field(self) -> None:
        """Should extract target from target_anchor field (schema-correct)."""
        choice = {"intent": "Go north", "target_anchor": "anchor002"}
        assert _get_choice_target(choice) == "anchor002"

    def test_target_field(self) -> None:
        """Should extract target from target field (common LLM mistake)."""
        choice = {"intent": "Go south", "target": "anchor003"}
        assert _get_choice_target(choice) == "anchor003"

    def test_target_anchor_preferred_over_target(self) -> None:
        """Should prefer target_anchor when both fields present."""
        choice = {
            "intent": "Go east",
            "target_anchor": "anchor004",
            "target": "anchor005",
        }
        assert _get_choice_target(choice) == "anchor004"

    def test_neither_field(self) -> None:
        """Should return None when neither field present."""
        choice = {"intent": "Stay here"}
        assert _get_choice_target(choice) is None

    def test_empty_target_anchor_falls_back_to_target(self) -> None:
        """Should fall back to target when target_anchor is empty string."""
        choice = {"intent": "Go west", "target_anchor": "", "target": "anchor006"}
        assert _get_choice_target(choice) == "anchor006"

    def test_none_target_anchor_falls_back_to_target(self) -> None:
        """Should fall back to target when target_anchor is None."""
        choice = {"intent": "Go down", "target_anchor": None, "target": "anchor007"}
        assert _get_choice_target(choice) == "anchor007"
