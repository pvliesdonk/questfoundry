"""Tests for story size profiles and DREAM Scope integration."""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.models.dream import Scope
from questfoundry.pipeline.size import (
    PRESETS,
    VALID_PRESETS,
    get_size_profile,
    resolve_size_from_graph,
    size_template_vars,
)


class TestSizeProfile:
    def test_all_presets_exist(self) -> None:
        assert {"vignette", "short", "standard", "long"} == VALID_PRESETS
        for name in VALID_PRESETS:
            profile = get_size_profile(name)
            assert profile.preset == name

    def test_standard_matches_current_hardcoded_values(self) -> None:
        """Standard preset must exactly match current hardcoded defaults."""
        s = get_size_profile("standard")
        # seed.py:431 — max_arcs=16
        assert s.max_arcs == 16
        # seed_pruning.py:57 — log2(16) = 4 fully explored
        assert s.fully_explored == 4
        # summarize_brainstorm.yaml — Characters 5-10
        assert s.characters_min == 5
        assert s.characters_max == 10
        # summarize_brainstorm.yaml — Locations 3-6
        assert s.locations_min == 3
        assert s.locations_max == 6
        # summarize_brainstorm.yaml — Objects 3-5
        assert s.objects_min == 3
        assert s.objects_max == 5
        # summarize_brainstorm.yaml — Dilemmas 4-8
        assert s.dilemmas_min == 4
        assert s.dilemmas_max == 8
        # discuss_brainstorm.yaml — 15-25 entities
        assert s.entities_min == 15
        assert s.entities_max == 25
        # discuss_seed.yaml — 2-4 beats per path
        assert s.beats_per_path_min == 2
        assert s.beats_per_path_max == 4
        # fill_phase0_voice.yaml — 3-5 tone words
        assert s.tone_words_min == 3
        assert s.tone_words_max == 5

    def test_invalid_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown story_size preset"):
            get_size_profile("huge")

    def test_presets_are_frozen(self) -> None:
        profile = get_size_profile("standard")
        with pytest.raises(AttributeError):
            profile.max_arcs = 32  # type: ignore[misc]

    def test_range_str_formatting(self) -> None:
        profile = get_size_profile("standard")
        assert profile.range_str("characters") == "5-10"
        assert profile.range_str("dilemmas") == "4-8"
        assert profile.range_str("locations") == "3-6"
        assert profile.range_str("tone_words") == "3-5"

    def test_range_str_invalid_prefix_raises(self) -> None:
        profile = get_size_profile("standard")
        with pytest.raises(AttributeError):
            profile.range_str("nonexistent")

    def test_preset_ordering_max_arcs(self) -> None:
        """Presets scale monotonically by max_arcs."""
        order = ["vignette", "short", "standard", "long"]
        arcs = [get_size_profile(name).max_arcs for name in order]
        assert arcs == sorted(arcs)
        assert len(set(arcs)) == 4  # All distinct

    def test_preset_ordering_est_passages(self) -> None:
        """Estimated passages scale monotonically."""
        order = ["vignette", "short", "standard", "long"]
        passages = [get_size_profile(name).est_passages_max for name in order]
        assert passages == sorted(passages)

    def test_preset_ordering_fully_explored(self) -> None:
        order = ["vignette", "short", "standard", "long"]
        explored = [get_size_profile(name).fully_explored for name in order]
        assert explored == sorted(explored)

    def test_all_presets_have_consistent_ranges(self) -> None:
        """Min <= max for all range pairs in all presets."""
        range_prefixes = [
            "characters",
            "locations",
            "objects",
            "dilemmas",
            "entities",
            "beats_per_path",
            "convergence_points",
            "est_passages",
            "est_words",
            "tone_words",
        ]
        for name, profile in PRESETS.items():
            for prefix in range_prefixes:
                lo = getattr(profile, f"{prefix}_min")
                hi = getattr(profile, f"{prefix}_max")
                assert lo <= hi, f"{name}.{prefix}: {lo} > {hi}"


class TestResolveSizeFromGraph:
    def test_with_story_size_in_scope(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "vision",
            {
                "type": "vision",
                "scope": {"story_size": "short", "estimated_passages": 20},
            },
        )
        profile = resolve_size_from_graph(graph)
        assert profile.preset == "short"
        assert profile.max_arcs == 8

    def test_missing_vision_node_defaults_to_standard(self) -> None:
        graph = Graph.empty()
        profile = resolve_size_from_graph(graph)
        assert profile.preset == "standard"

    def test_missing_scope_defaults_to_standard(self) -> None:
        graph = Graph.empty()
        graph.create_node("vision", {"type": "vision", "genre": "fantasy"})
        profile = resolve_size_from_graph(graph)
        assert profile.preset == "standard"

    def test_missing_story_size_field_defaults_to_standard(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "vision",
            {
                "type": "vision",
                "scope": {"estimated_passages": 20},
            },
        )
        profile = resolve_size_from_graph(graph)
        assert profile.preset == "standard"

    def test_invalid_story_size_defaults_to_standard(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "vision",
            {
                "type": "vision",
                "scope": {"story_size": "enormous"},
            },
        )
        profile = resolve_size_from_graph(graph)
        assert profile.preset == "standard"

    def test_vignette_from_graph(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "vision",
            {
                "type": "vision",
                "scope": {"story_size": "vignette"},
            },
        )
        profile = resolve_size_from_graph(graph)
        assert profile.preset == "vignette"
        assert profile.max_arcs == 2

    def test_long_from_graph(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "vision",
            {
                "type": "vision",
                "scope": {"story_size": "long"},
            },
        )
        profile = resolve_size_from_graph(graph)
        assert profile.preset == "long"
        assert profile.max_arcs == 32


class TestScopeStorySize:
    """Tests for story_size field on the DREAM Scope model."""

    def test_story_size_is_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Scope()  # type: ignore[call-arg]

    def test_explicit_story_size(self) -> None:
        scope = Scope(story_size="vignette", estimated_passages=10, target_word_count=3000)
        assert scope.story_size == "vignette"

    def test_story_size_in_model_dump(self) -> None:
        scope = Scope(story_size="short", estimated_passages=20, target_word_count=10000)
        data = scope.model_dump()
        assert data["story_size"] == "short"

    def test_story_size_from_dict(self) -> None:
        scope = Scope.model_validate(
            {
                "story_size": "long",
                "estimated_passages": 80,
                "target_word_count": 40000,
            }
        )
        assert scope.story_size == "long"

    def test_story_size_absent_raises_error(self) -> None:
        """story_size is required — omitting it raises a validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Scope.model_validate({"estimated_passages": 30, "target_word_count": 15000})


class TestSizeTemplateVars:
    def test_standard_template_vars(self) -> None:
        profile = get_size_profile("standard")
        vars_ = size_template_vars(profile)
        assert vars_["size_characters"] == "5-10"
        assert vars_["size_dilemmas"] == "4-8"
        assert vars_["size_locations"] == "3-6"
        assert vars_["size_beats_per_path"] == "2-4"
        assert vars_["size_preset"] == "standard"

    def test_vignette_template_vars(self) -> None:
        profile = get_size_profile("vignette")
        vars_ = size_template_vars(profile)
        assert vars_["size_characters"] == "2-4"
        assert vars_["size_dilemmas"] == "2-3"
        assert vars_["size_locations"] == "1-2"

    def test_default_uses_standard(self) -> None:
        vars_ = size_template_vars(None)
        assert vars_["size_preset"] == "standard"
        assert vars_["size_characters"] == "5-10"

    def test_all_expected_keys_present(self) -> None:
        vars_ = size_template_vars()
        expected = {
            "size_characters",
            "size_locations",
            "size_objects",
            "size_dilemmas",
            "size_entities",
            "size_beats_per_path",
            "size_convergence_points",
            "size_est_passages",
            "size_est_words",
            "size_tone_words",
            "size_preset",
        }
        assert set(vars_.keys()) == expected
