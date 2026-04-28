"""Tests for pipeline.summary — per-stage CLI summary builders."""

from __future__ import annotations

from typing import Any

from questfoundry.pipeline.summary import build_stage_summary


class TestUnknownAndEmpty:
    def test_unknown_stage_returns_empty(self) -> None:
        assert build_stage_summary("nope", {"foo": "bar"}) == []

    def test_non_dict_artifact_returns_empty(self) -> None:
        assert build_stage_summary("dream", "not a dict") == []  # type: ignore[arg-type]

    def test_empty_dict_returns_empty(self) -> None:
        assert build_stage_summary("dream", {}) == []


class TestDream:
    def test_full_dream(self) -> None:
        data: dict[str, Any] = {
            "genre": "dark fantasy",
            "subgenre": "gothic horror",
            "audience": "adult",
            "tone": ["bleak", "claustrophobic"],
            "themes": ["guilt", "redemption", "sacrifice"],
            "scope": {"story_size": "short"},
        }
        lines = build_stage_summary("dream", data)
        assert lines == [
            "Genre: dark fantasy (gothic horror)",
            "Audience: adult",
            "Tone: bleak, claustrophobic",
            "Themes: guilt, redemption, sacrifice",
            "Scope: short",
        ]

    def test_dream_no_subgenre(self) -> None:
        data = {"genre": "noir", "audience": "adult", "themes": ["loss"], "tone": ["cold"]}
        lines = build_stage_summary("dream", data)
        assert lines[0] == "Genre: noir"

    def test_dream_long_themes_truncated(self) -> None:
        data = {
            "genre": "g",
            "themes": ["a", "b", "c", "d", "e", "f"],
            "tone": ["t"],
        }
        lines = build_stage_summary("dream", data)
        # _format_str_list default limit=4
        assert "Themes: a, b, c, d, ..." in lines


class TestBrainstorm:
    def test_entities_with_categories(self) -> None:
        data = {
            "entities": [
                {"entity_category": "character"},
                {"entity_category": "character"},
                {"entity_category": "location"},
            ],
            "dilemmas": [{"id": "d1"}, {"id": "d2"}],
        }
        lines = build_stage_summary("brainstorm", data)
        assert lines[0].startswith("Entities: 3 (")
        assert "2 character" in lines[0]
        assert "1 location" in lines[0]
        assert lines[-1] == "Dilemmas: 2"

    def test_entities_without_categories(self) -> None:
        data = {"entities": [{"foo": "bar"}]}
        lines = build_stage_summary("brainstorm", data)
        assert lines == ["Entities: 1"]


class TestSeed:
    def test_seed_counts(self) -> None:
        data = {
            "entities": [{}, {}, {}],
            "dilemmas": [{}, {}],
            "paths": [{}, {}, {}, {}],
            "consequences": [{}],
            "initial_beats": [{}, {}, {}, {}, {}],
        }
        lines = build_stage_summary("seed", data)
        assert lines == [
            "Entities: 3",
            "Dilemmas: 2",
            "Paths: 4",
            "Consequences: 1",
            "Initial beats: 5",
        ]

    def test_seed_empty_lists_skipped(self) -> None:
        data = {"entities": [], "dilemmas": [{}]}
        lines = build_stage_summary("seed", data)
        assert lines == ["Dilemmas: 1"]


class TestGrow:
    def test_grow_int_counts(self) -> None:
        data = {"arc_count": 3, "state_flag_count": 5, "overlay_count": 2}
        lines = build_stage_summary("grow", data)
        assert lines == ["Arcs: 3", "State flags: 5", "Overlays: 2"]

    def test_grow_zero_counts_skipped(self) -> None:
        data = {"arc_count": 0, "state_flag_count": 4}
        lines = build_stage_summary("grow", data)
        assert lines == ["State flags: 4"]


class TestFill:
    def test_fill_counts(self) -> None:
        data = {
            "passages_filled": 12,
            "passages_flagged": 1,
            "entity_updates_applied": 4,
            "review_cycles": 2,
        }
        lines = build_stage_summary("fill", data)
        assert lines == [
            "Passages filled: 12",
            "Passages flagged: 1",
            "Entity updates: 4",
            "Review cycles: 2",
        ]


class TestDress:
    def test_dress_full(self) -> None:
        data = {
            "art_direction_created": True,
            "entity_visuals_created": 5,
            "codex_entries_created": 8,
            "briefs_created": 3,
            "illustrations_generated": 3,
            "illustrations_failed": 0,
        }
        lines = build_stage_summary("dress", data)
        assert lines[0] == "Art direction: created"
        assert "Entity visuals: 5" in lines
        assert "Codex entries: 8" in lines
        assert "Illustrations failed: 0" not in lines  # zero skipped

    def test_dress_no_art_direction(self) -> None:
        data = {"art_direction_created": False, "codex_entries_created": 3}
        lines = build_stage_summary("dress", data)
        assert lines == ["Codex entries: 3"]


class TestPolish:
    def test_polish_counts(self) -> None:
        data = {
            "passage_count": 24,
            "choice_count": 18,
            "variant_count": 4,
            "residue_count": 2,
            "sidetrack_count": 1,
            "false_branch_count": 3,
        }
        lines = build_stage_summary("polish", data)
        assert lines == [
            "Passages: 24",
            "Choices: 18",
            "Variants: 4",
            "Residue beats: 2",
            "Sidetrack beats: 1",
            "False branches: 3",
        ]

    def test_polish_zero_skipped(self) -> None:
        data = {"passage_count": 24, "choice_count": 0}
        lines = build_stage_summary("polish", data)
        assert lines == ["Passages: 24"]
