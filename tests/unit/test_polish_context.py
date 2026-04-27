"""Tests for POLISH context builders."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_context import (
    format_entity_arc_context,
    format_linear_section_context,
    format_pacing_context,
)


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node with optional extras."""
    data = {
        "type": "beat",
        "raw_id": beat_id.split("::")[-1],
        "summary": summary,
        "dilemma_impacts": [],
        "entities": [],
        "scene_type": "scene",
    }
    data.update(kwargs)
    graph.create_node(beat_id, data)


class TestFormatLinearSectionContext:
    """Tests for Phase 1 context builder."""

    def test_basic_section(self) -> None:
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "First action")
        _make_beat(graph, "beat::b", "Second action")
        _make_beat(graph, "beat::c", "Third action")

        ctx = format_linear_section_context(
            graph, "section_0", ["beat::a", "beat::b", "beat::c"], None, None
        )

        assert ctx["section_id"] == "section_0"
        # Beat IDs in beat_details lines are backtick-wrapped per @prompt-engineer Rule 4.
        assert "`beat::a`" in ctx["beat_details"]
        assert "`beat::b`" in ctx["beat_details"]
        assert "`beat::c`" in ctx["beat_details"]
        assert ctx["beat_count"] == "3"
        assert ctx["valid_beat_ids"] == "`beat::a`, `beat::b`, `beat::c`"

    def test_with_context_beats(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::before", "Before section", scene_type="sequel")
        _make_beat(graph, "beat::a", "Section beat")
        _make_beat(graph, "beat::after", "After section")

        ctx = format_linear_section_context(graph, "s0", ["beat::a"], "beat::before", "beat::after")

        assert "preceding" in ctx["before_context"]
        assert "beat::before" in ctx["before_context"]
        assert "following" in ctx["after_context"]
        assert "beat::after" in ctx["after_context"]

    def test_no_context_beats(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Only beat")

        ctx = format_linear_section_context(graph, "s0", ["beat::a"], None, None)

        assert "start/end" in ctx["before_context"]
        assert "start/end" in ctx["after_context"]

    def test_empty_section_falls_back_to_none(self) -> None:
        """Empty `beat_ids` MUST render `(none)` for both `beat_details` and
        `valid_beat_ids` per the consistent empty-fallback pattern across
        polish_context render functions."""
        graph = Graph.empty()
        ctx = format_linear_section_context(graph, "section_0", [], None, None)
        assert ctx["beat_details"] == "(none)"
        assert ctx["valid_beat_ids"] == "(none)"
        assert ctx["beat_count"] == "0"

    def test_section_beat_with_entities_renders_backticks(self) -> None:
        """A section beat whose `entities` field is populated renders the
        entity list with backticks per @prompt-engineer Rule 4 — never as a
        Python repr-style bracket list. Covers the linear-section
        equivalent of `test_pacing_beat_with_entities_renders_backticks`."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Hero acts", entities=["entity::hero"])

        ctx = format_linear_section_context(graph, "s0", ["beat::a"], None, None)

        assert "entities: `entity::hero`" in ctx["beat_details"]
        # No bracket-format leak per @prompt-engineer Rule 4.
        assert "entities=[" not in ctx["beat_details"]

    def test_dilemma_impacts_shown(self) -> None:
        graph = Graph.empty()
        _make_beat(
            graph,
            "beat::commit",
            "Commit beat",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )

        ctx = format_linear_section_context(graph, "s0", ["beat::commit"], None, None)

        assert "commits" in ctx["beat_details"]
        # Dilemma IDs are backtick-wrapped within the impacts: clause.
        assert "`dilemma::d1`" in ctx["beat_details"]
        # No bracket-format leaks per @prompt-engineer Rule 4.
        assert "impacts=[" not in ctx["beat_details"]


class TestFormatPacingContext:
    """Tests for Phase 2 context builder."""

    def test_with_pacing_flags(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Action 1", scene_type="scene")
        _make_beat(graph, "beat::b", "Action 2", scene_type="scene")
        _make_beat(graph, "beat::c", "Action 3", scene_type="scene")
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero", "name": "Hero"})

        flags = [
            {
                "issue_type": "consecutive_scene",
                "beat_ids": ["beat::a", "beat::b", "beat::c"],
                "path_id": "path::p1",
            }
        ]

        ctx = format_pacing_context(graph, flags)

        assert "consecutive_scene" in ctx["pacing_issues"]
        # Beat IDs and path IDs backtick-wrapped per @prompt-engineer Rule 4.
        assert "`beat::a`" in ctx["pacing_issues"]
        assert "Path: `path::p1`" in ctx["pacing_issues"]
        # valid_entity_ids backtick-wrapped, with `(none)` fallback.
        assert "`entity::hero`" in ctx["valid_entity_ids"]
        assert ctx["entity_count"] == "1"

    def test_no_flags(self) -> None:
        graph = Graph.empty()
        ctx = format_pacing_context(graph, [])
        assert "No pacing issues" in ctx["pacing_issues"]
        # No entities → valid_entity_ids falls back to `(none)`.
        assert ctx["valid_entity_ids"] == "(none)"

    def test_pacing_beat_with_entities_renders_backticks(self) -> None:
        """A pacing flag whose beats reference entities renders the entity
        list with backticks per @prompt-engineer Rule 4 — never as a Python
        repr-style bracket list."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Hero acts", entities=["entity::hero"])
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero", "name": "Hero"})

        flags = [
            {"issue_type": "consecutive_scene", "beat_ids": ["beat::a"], "path_id": "path::p1"}
        ]
        ctx = format_pacing_context(graph, flags)

        assert "entities: `entity::hero`" in ctx["pacing_issues"]
        # No bracket-format leaking through (@prompt-engineer Rule 4).
        assert "entities=[" not in ctx["pacing_issues"]


class TestFormatEntityArcContext:
    """Tests for Phase 3 context builder."""

    def test_basic_entity_context(self) -> None:
        graph = Graph.empty()
        graph.create_node("path::brave", {"type": "path", "raw_id": "brave"})
        graph.create_node(
            "entity::mentor",
            {
                "type": "entity",
                "raw_id": "mentor",
                "name": "The Mentor",
                "description": "A wise guide",
            },
        )

        _make_beat(
            graph, "beat::intro", "Mentor introduces themselves", entities=["entity::mentor"]
        )
        _make_beat(graph, "beat::reveal", "Mentor reveals a secret", entities=["entity::mentor"])

        graph.add_edge("belongs_to", "beat::intro", "path::brave")
        graph.add_edge("belongs_to", "beat::reveal", "path::brave")
        graph.add_edge("predecessor", "beat::reveal", "beat::intro")

        ctx = format_entity_arc_context(graph, "entity::mentor", ["beat::intro", "beat::reveal"])

        assert ctx["entity_id"] == "entity::mentor"
        assert ctx["entity_name"] == "The Mentor"
        assert "wise guide" in ctx["entity_description"]
        # IDs in beat_appearances lines are backtick-wrapped per @prompt-engineer Rule 4
        # rule 1 — matches the valid_*_ids lists so a model doesn't need to
        # mentally strip backticks when matching IDs across surfaces.
        assert "`beat::intro`" in ctx["beat_appearances"]
        assert "`beat::reveal`" in ctx["beat_appearances"]
        assert "(path: `path::brave`)" in ctx["beat_appearances"]
        # Same backtick convention for the standalone ID lists.
        assert "`path::brave`" in ctx["path_ids"]
        # `valid_path_ids` is entity-scoped (same as `path_ids`) per #1410 —
        # the Phase 3 prompt forbids arcs on paths the entity isn't in.
        assert ctx["valid_path_ids"] == ctx["path_ids"]
        assert "`beat::intro`" in ctx["valid_beat_ids"]
        assert "`beat::reveal`" in ctx["valid_beat_ids"]

    def test_entity_with_overlays(self) -> None:
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node(
            "entity::npc",
            {
                "type": "entity",
                "raw_id": "npc",
                "name": "NPC",
                "overlays": [
                    {
                        "when": ["dilemma::d1:path::p1"],
                        "details": {"demeanor": "hostile"},
                    }
                ],
            },
        )

        _make_beat(graph, "beat::b1", "Meet NPC", entities=["entity::npc"])
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        ctx = format_entity_arc_context(graph, "entity::npc", ["beat::b1"])

        assert "hostile" in ctx["overlay_data"]
        # Flag IDs are backtick-wrapped per @prompt-engineer Rule 4 — matches the
        # DRESS overlay renderer (closes #1406).
        assert "`dilemma::d1:path::p1`" in ctx["overlay_data"]

    def test_entity_overlay_list_values_render_human_readable(self) -> None:
        """List-valued details render as comma-joined strings (e.g. `umm, well`)
        — never as Python repr (`['umm', 'well']`) per @prompt-engineer Rule 4.
        Pinned because this is exactly the bracket-format the rule forbids."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node(
            "entity::npc",
            {
                "type": "entity",
                "raw_id": "npc",
                "name": "NPC",
                "overlays": [
                    {
                        "when": ["state_flag::met_npc"],
                        "details": {"speech_tics": ["umm", "well"]},
                    }
                ],
            },
        )
        _make_beat(graph, "beat::b1", "Meet NPC", entities=["entity::npc"])
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        ctx = format_entity_arc_context(graph, "entity::npc", ["beat::b1"])
        assert "speech_tics: umm, well" in ctx["overlay_data"]
        # Belt-and-braces: explicitly assert the bracket-format is GONE.
        assert "['umm', 'well']" not in ctx["overlay_data"]

    def test_id_lists_fall_back_to_none_when_empty(self) -> None:
        """Empty source sets MUST render as `(none)` rather than an empty
        string so the prompt never receives a bare empty injection. Matches
        the existing `anchored_dilemmas` fallback pattern; pinned because
        empty-input behaviour is otherwise easy to regress silently."""
        graph = Graph.empty()
        graph.create_node(
            "entity::loner",
            {"type": "entity", "raw_id": "loner", "name": "Loner", "description": ""},
        )
        # No beats, no paths, no anchored_to edges.

        ctx = format_entity_arc_context(graph, "entity::loner", [])
        assert ctx["path_ids"] == "(none)"
        assert ctx["valid_path_ids"] == "(none)"
        assert ctx["valid_beat_ids"] == "(none)"
        assert ctx["anchored_dilemmas"] == "(none)"
        # `beat_appearances` uses the same fallback (with the indent the
        # rendered lines normally carry) so the prompt never receives an
        # empty injection.
        assert ctx["beat_appearances"] == "  (none)"

    def test_valid_path_ids_excludes_paths_entity_does_not_appear_on(self) -> None:
        """`valid_path_ids` MUST be scoped to paths where the entity actually
        appears (closes #1410). Showing the broader story-wide list confused
        models into inventing arcs on paths the entity is never in."""
        graph = Graph.empty()
        # Two paths exist in the story.
        graph.create_node("path::story_a", {"type": "path", "raw_id": "story_a"})
        graph.create_node("path::story_b", {"type": "path", "raw_id": "story_b"})
        graph.create_node(
            "entity::loner",
            {"type": "entity", "raw_id": "loner", "name": "Loner", "description": "x"},
        )
        # Entity appears only on path_a (via beat::b1).
        _make_beat(graph, "beat::b1", "Loner appears", entities=["entity::loner"])
        graph.add_edge("belongs_to", "beat::b1", "path::story_a")

        ctx = format_entity_arc_context(graph, "entity::loner", ["beat::b1"])
        # Both path_ids and valid_path_ids show only the entity's path,
        # NOT path::story_b (which exists in the graph but doesn't include
        # the entity).
        assert ctx["path_ids"] == "`path::story_a`"
        assert ctx["valid_path_ids"] == "`path::story_a`"
        assert "story_b" not in ctx["valid_path_ids"]

    def test_anchored_dilemmas_backtick_wrapped(self) -> None:
        """Dilemmas the entity is `anchored_to` are backtick-wrapped per
        @prompt-engineer Rule 4 — same convention as overlay flag IDs and the
        valid_*_ids lists."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node(
            "entity::mentor",
            {"type": "entity", "raw_id": "mentor", "name": "Mentor", "description": "guide"},
        )
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.add_edge("anchored_to", "entity::mentor", "dilemma::trust")
        _make_beat(graph, "beat::b1", "Meet mentor", entities=["entity::mentor"])
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        ctx = format_entity_arc_context(graph, "entity::mentor", ["beat::b1"])
        assert ctx["anchored_dilemmas"] == "`dilemma::trust`"

    def test_entity_overlay_details_sorted_for_determinism(self) -> None:
        """Detail keys MUST be iterated in sorted order so the rendered string
        is byte-identical across runs regardless of dict insertion order."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node(
            "entity::npc",
            {
                "type": "entity",
                "raw_id": "npc",
                "name": "NPC",
                "overlays": [
                    {
                        "when": ["state_flag::met_npc"],
                        # Inserted in non-alphabetical order intentionally.
                        "details": {"voice": "soft", "demeanor": "warm"},
                    }
                ],
            },
        )
        _make_beat(graph, "beat::b1", "Meet NPC", entities=["entity::npc"])
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        ctx = format_entity_arc_context(graph, "entity::npc", ["beat::b1"])
        # `demeanor` sorts before `voice` alphabetically.
        d_idx = ctx["overlay_data"].index("demeanor: warm")
        v_idx = ctx["overlay_data"].index("voice: soft")
        assert d_idx < v_idx

    def test_entity_not_found(self) -> None:
        """Missing entity returns empty fields gracefully."""
        graph = Graph.empty()
        _make_beat(graph, "beat::b1", "Some beat")

        ctx = format_entity_arc_context(graph, "entity::missing", ["beat::b1"])

        assert ctx["entity_id"] == "entity::missing"
        assert ctx["entity_name"] == "entity::missing"


def test_format_entity_context_lists_all_paths_for_pre_commit_beat() -> None:
    """Pre-commit beats with dual belongs_to show both paths in appearance lines."""
    graph = Graph.empty()
    graph.create_node("entity::mentor", {"type": "entity", "name": "Mentor", "description": "x"})
    graph.create_node("path::trust__a", {"type": "path", "raw_id": "trust__a"})
    graph.create_node("path::trust__b", {"type": "path", "raw_id": "trust__b"})
    graph.create_node(
        "beat::shared",
        {
            "type": "beat",
            "summary": "Shared setup.",
            "scene_type": "scene",
            "raw_id": "shared",
            "dilemma_impacts": [],
            "entities": ["entity::mentor"],
        },
    )
    graph.add_edge("belongs_to", "beat::shared", "path::trust__a")
    graph.add_edge("belongs_to", "beat::shared", "path::trust__b")
    graph.add_edge("appears", "entity::mentor", "beat::shared")

    ctx = format_entity_arc_context(graph, "entity::mentor", ["beat::shared"])
    beat_appearances = ctx["beat_appearances"]
    assert "path::trust__a" in beat_appearances
    assert "path::trust__b" in beat_appearances
