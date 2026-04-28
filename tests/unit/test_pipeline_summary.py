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
        # Mirrors the actual graph-derived shape from DressStage._extract_artifact:
        # dict-of-dicts keyed by node id, plus an art_direction dict.
        data = {
            "art_direction": {"art_direction_id": "main", "style": "noir"},
            "entity_visuals": {"ev1": {}, "ev2": {}, "ev3": {}, "ev4": {}, "ev5": {}},
            "briefs": {"b1": {}, "b2": {}, "b3": {}},
            "codex_entries": {f"c{i}": {} for i in range(8)},
            "illustrations": {"i1": {}, "i2": {}, "i3": {}},
        }
        lines = build_stage_summary("dress", data)
        assert lines[0] == "Art direction: created"
        assert "Entity visuals: 5" in lines
        assert "Codex entries: 8" in lines
        assert "Illustration briefs: 3" in lines
        assert "Illustrations: 3" in lines

    def test_dress_no_art_direction(self) -> None:
        # Graph extraction returns {} when the art_direction node is absent.
        data = {"art_direction": {}, "codex_entries": {"c1": {}, "c2": {}, "c3": {}}}
        lines = build_stage_summary("dress", data)
        assert lines == ["Codex entries: 3"]

    def test_dress_empty_collections_skipped(self) -> None:
        data = {
            "art_direction": {},
            "entity_visuals": {},
            "briefs": {"b1": {}},
            "codex_entries": {},
            "illustrations": {},
        }
        lines = build_stage_summary("dress", data)
        assert lines == ["Illustration briefs: 1"]


class TestPolish:
    # POLISH writes its outputs to the graph (passages, choices, state flags),
    # not into the artifact dict. The artifact is just `{"phases_completed": [...]}`,
    # so the summary surfaces phase tally only — for headline counts use `qf status`.
    def test_polish_phases_completed(self) -> None:
        data = {"phases_completed": [{"phase": "5a"}, {"phase": "5b"}, {"phase": "5c"}]}
        lines = build_stage_summary("polish", data)
        assert lines == ["Phases completed: 3"]

    def test_polish_empty_phases(self) -> None:
        data: dict[str, Any] = {"phases_completed": []}
        lines = build_stage_summary("polish", data)
        assert lines == []

    def test_polish_missing_phases_key(self) -> None:
        data: dict[str, Any] = {"some_other_field": 42}
        lines = build_stage_summary("polish", data)
        assert lines == []
