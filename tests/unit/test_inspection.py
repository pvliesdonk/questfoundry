"""Tests for project inspection module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.graph import Graph

if TYPE_CHECKING:
    from pathlib import Path
from questfoundry.inspection import (
    InspectionReport,
    _branching_stats,
    _coverage_stats,
    _graph_summary,
    _prose_stats,
    inspect_project,
)


def _make_graph_with_passages(prose_texts: list[str | None]) -> Graph:
    """Build a graph with passages having given prose values."""
    graph = Graph.empty()
    for i, prose in enumerate(prose_texts):
        data: dict = {
            "type": "passage",
            "raw_id": f"p{i}",
            "from_beat": f"beat::p{i}",
            "summary": f"passage {i}",
        }
        if prose is not None:
            data["prose"] = prose
        else:
            data["prose"] = None
            data["flag"] = "incompatible_states"
            data["flag_reason"] = "test reason"
        graph.create_node(f"passage::p{i}", data)
    return graph


def _make_full_graph() -> Graph:
    """Build a graph with passages, choices, entities, codex, and briefs."""
    graph = _make_graph_with_passages(
        ["The hero stood tall in the morning light.", "She ran through the dark forest quickly."]
    )

    # Add choices
    graph.create_node(
        "choice::p0__p1",
        {
            "type": "choice",
            "from_passage": "passage::p0",
            "to_passage": "passage::p1",
            "label": "Enter the forest",
            "requires": [],
            "grants": [],
        },
    )
    graph.create_node(
        "choice::p0__p1_continue",
        {
            "type": "choice",
            "from_passage": "passage::p0",
            "to_passage": "passage::p1",
            "label": "continue",
            "requires": [],
            "grants": [],
        },
    )
    graph.add_edge("choice_from", "choice::p0__p1", "passage::p0")
    graph.add_edge("choice_to", "choice::p0__p1", "passage::p1")
    graph.add_edge("choice_from", "choice::p0__p1_continue", "passage::p0")
    graph.add_edge("choice_to", "choice::p0__p1_continue", "passage::p1")

    # Add entities
    graph.create_node(
        "entity::hero",
        {"type": "entity", "entity_type": "character", "concept": "The Hero"},
    )
    graph.create_node(
        "entity::forest",
        {"type": "entity", "entity_type": "location", "concept": "Dark Forest"},
    )

    # Add codex entries with HasEntry edges
    graph.create_node(
        "codex_entry::hero_1",
        {"type": "codex_entry", "title": "Hero", "rank": 1, "content": "A brave soul."},
    )
    graph.add_edge("HasEntry", "codex_entry::hero_1", "entity::hero")

    # Add illustration briefs
    graph.create_node(
        "illustration_brief::ib1",
        {"type": "illustration_brief", "subject": "hero", "priority": 1},
    )

    return graph


class TestGraphSummary:
    def test_counts_nodes_and_edges(self, tmp_path: Path) -> None:
        graph = _make_full_graph()
        summary = _graph_summary(graph, tmp_path)

        assert summary.total_nodes > 0
        assert summary.total_edges > 0
        assert "passage" in summary.node_counts
        assert "choice" in summary.node_counts

    def test_node_counts_sorted_by_count(self, tmp_path: Path) -> None:
        graph = _make_full_graph()
        summary = _graph_summary(graph, tmp_path)

        counts = list(summary.node_counts.values())
        assert counts == sorted(counts, reverse=True)

    def test_empty_graph(self, tmp_path: Path) -> None:
        graph = Graph.empty()
        summary = _graph_summary(graph, tmp_path)

        assert summary.total_nodes == 0
        assert summary.total_edges == 0


class TestProseStats:
    def test_word_counts(self) -> None:
        graph = _make_graph_with_passages(["Hello world", "One two three four five"])
        stats = _prose_stats(graph)

        assert stats is not None
        assert stats.total_passages == 2
        assert stats.passages_with_prose == 2
        assert stats.total_words == 7
        assert stats.min_words == 2
        assert stats.max_words == 5
        assert stats.avg_words == 3.5

    def test_flagged_passages_detected(self) -> None:
        graph = _make_graph_with_passages(["Some prose here.", None])
        stats = _prose_stats(graph)

        assert stats is not None
        assert stats.passages_with_prose == 1
        assert len(stats.flagged_passages) == 1
        assert stats.flagged_passages[0]["id"] == "passage::p1"
        assert stats.flagged_passages[0]["flag"] == "incompatible_states"

    def test_no_passages_returns_none(self) -> None:
        graph = Graph.empty()
        assert _prose_stats(graph) is None

    def test_lexical_diversity(self) -> None:
        graph = _make_graph_with_passages(["the the the the", "new new new new"])
        stats = _prose_stats(graph)

        assert stats is not None
        assert stats.lexical_diversity is not None
        assert 0.0 < stats.lexical_diversity < 1.0

    def test_single_passage_no_diversity(self) -> None:
        graph = _make_graph_with_passages(["Just one passage."])
        stats = _prose_stats(graph)

        assert stats is not None
        assert stats.lexical_diversity is None


class TestBranchingStats:
    def test_structural_classification(self) -> None:
        """Choices classified by graph structure: meaningful (2+ outgoing), contextual, continue."""
        graph = _make_full_graph()
        stats = _branching_stats(graph)

        assert stats is not None
        assert stats.total_choices == 2
        # Both choices originate from passage::p0 which has 2 outgoing → both meaningful
        assert stats.meaningful_choices == 2
        assert stats.contextual_choices == 0
        assert stats.continue_choices == 0

    def test_three_way_classification(self) -> None:
        """Passages with 1 outgoing choice classified as contextual or continue by label."""
        graph = Graph.empty()
        # Create 3 passages
        for i in range(3):
            graph.create_node(
                f"passage::p{i}",
                {
                    "type": "passage",
                    "raw_id": f"p{i}",
                    "from_beat": f"beat::p{i}",
                    "summary": f"p{i}",
                },
            )

        # p0 → p1: single outgoing, contextual label
        graph.create_node(
            "choice::p0__p1",
            {
                "type": "choice",
                "from_passage": "passage::p0",
                "to_passage": "passage::p1",
                "label": "Search the room",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p0__p1", "passage::p0")
        graph.add_edge("choice_to", "choice::p0__p1", "passage::p1")

        # p1 → p2: single outgoing, continue label
        graph.create_node(
            "choice::p1__p2",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "continue",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p1__p2", "passage::p1")
        graph.add_edge("choice_to", "choice::p1__p2", "passage::p2")

        stats = _branching_stats(graph)

        assert stats is not None
        assert stats.total_choices == 2
        assert stats.meaningful_choices == 0  # no passage has 2+ outgoing
        assert stats.contextual_choices == 1  # p0→p1 has non-continue label
        assert stats.continue_choices == 1  # p1→p2 has "continue" label

    def test_no_passages_returns_none(self) -> None:
        graph = Graph.empty()
        assert _branching_stats(graph) is None

    def test_start_and_ending_passages(self) -> None:
        graph = _make_full_graph()
        stats = _branching_stats(graph)

        assert stats is not None
        # p0 has no incoming → start, p1 has no outgoing → ending
        assert stats.start_passages >= 1
        assert stats.ending_passages >= 1

    def test_start_passages_ignore_return_links(self) -> None:
        """Return links (spoke→hub) should not count as incoming for start detection."""
        graph = Graph.empty()
        graph.create_node(
            "passage::p0",
            {"type": "passage", "raw_id": "p0", "from_beat": "beat::p0", "summary": "p0"},
        )
        graph.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::p1", "summary": "p1"},
        )
        graph.create_node(
            "passage::spoke_0",
            {
                "type": "passage",
                "raw_id": "spoke_0",
                "from_beat": "beat::spoke_0",
                "summary": "spoke_0",
            },
        )

        # Ensure p0 has 2 outgoing (hub), then add a return link spoke→p0.
        graph.create_node(
            "choice::p0__p1",
            {
                "type": "choice",
                "from_passage": "passage::p0",
                "to_passage": "passage::p1",
                "label": "continue",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p0__p1", "passage::p0")
        graph.add_edge("choice_to", "choice::p0__p1", "passage::p1")
        graph.create_node(
            "choice::p0__spoke_0",
            {
                "type": "choice",
                "from_passage": "passage::p0",
                "to_passage": "passage::spoke_0",
                "label": "Look around",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p0__spoke_0", "passage::p0")
        graph.add_edge("choice_to", "choice::p0__spoke_0", "passage::spoke_0")
        graph.create_node(
            "choice::spoke_0_return",
            {
                "type": "choice",
                "from_passage": "passage::spoke_0",
                "to_passage": "passage::p0",
                "label": "Return",
                "is_return": True,
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::spoke_0_return", "passage::spoke_0")
        graph.add_edge("choice_to", "choice::spoke_0_return", "passage::p0")

        stats = _branching_stats(graph)
        assert stats is not None
        assert stats.start_passages == 1


class TestCoverageStats:
    def test_entity_counts(self, tmp_path: Path) -> None:
        graph = _make_full_graph()
        stats = _coverage_stats(graph, tmp_path)

        assert stats.entity_count == 2
        assert stats.entity_types["character"] == 1
        assert stats.entity_types["location"] == 1

    def test_codex_entry_count(self, tmp_path: Path) -> None:
        graph = _make_full_graph()
        stats = _coverage_stats(graph, tmp_path)

        assert stats.codex_entries == 1
        assert stats.entities_with_codex == 1

    def test_entities_with_codex_counts_unique_entities(self, tmp_path: Path) -> None:
        """Multiple codex entries per entity should count the entity once."""
        graph = _make_full_graph()
        # Add a second codex entry for the same entity (hero)
        graph.create_node(
            "codex_entry::hero_2",
            {"type": "codex_entry", "title": "Hero Backstory", "rank": 2, "content": "More lore."},
        )
        graph.add_edge("HasEntry", "codex_entry::hero_2", "entity::hero")

        stats = _coverage_stats(graph, tmp_path)
        assert stats.codex_entries == 2  # Two entries total
        assert stats.entities_with_codex == 1  # But only one unique entity

    def test_illustration_brief_count(self, tmp_path: Path) -> None:
        graph = _make_full_graph()
        stats = _coverage_stats(graph, tmp_path)

        assert stats.illustration_briefs == 1
        assert stats.illustration_nodes == 0

    def test_asset_file_counting(self, tmp_path: Path) -> None:
        graph = _make_full_graph()
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "img1.png").write_bytes(b"fake")
        (assets / "img2.png").write_bytes(b"fake")

        stats = _coverage_stats(graph, tmp_path)
        assert stats.asset_files == 2

    def test_no_assets_dir(self, tmp_path: Path) -> None:
        graph = Graph.empty()
        stats = _coverage_stats(graph, tmp_path)
        assert stats.asset_files == 0


class TestInspectProject:
    def test_integration(self, tmp_path: Path) -> None:
        """Full integration: write graph, run inspect."""
        import json

        graph = _make_full_graph()
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(graph.to_dict()))

        # Write a minimal project.yaml
        (tmp_path / "project.yaml").write_text("name: test-project\nversion: 1\n")

        report = inspect_project(tmp_path)

        assert isinstance(report, InspectionReport)
        assert report.summary.project_name == "test-project"
        assert report.summary.total_nodes > 0
        assert report.prose is not None
        assert report.branching is not None
        assert report.coverage.entity_count == 2

    def test_json_serializable(self, tmp_path: Path) -> None:
        """Report can be serialized to JSON."""
        import dataclasses
        import json

        graph = _make_full_graph()
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(graph.to_dict()))
        (tmp_path / "project.yaml").write_text("name: test\nversion: 1\n")

        report = inspect_project(tmp_path)
        # Should not raise
        result = json.dumps(dataclasses.asdict(report))
        assert isinstance(result, str)
