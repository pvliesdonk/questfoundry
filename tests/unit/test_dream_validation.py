"""Tests for DREAM Stage Output Contract validator."""

from __future__ import annotations

import pytest

from questfoundry.graph.dream_validation import validate_dream_output
from questfoundry.graph.graph import Graph


def _build_compliant_vision() -> dict[str, object]:
    """A vision payload that satisfies every rule in DREAM's output contract."""
    return {
        "type": "vision",
        "genre": "dark fantasy",
        "subgenre": "mystery",
        "tone": ["atmospheric", "morally ambiguous"],
        "themes": ["forbidden knowledge", "trust"],
        "audience": "adult",
        "scope": {"story_size": "short"},
        "content_notes": {"includes": [], "excludes": ["graphic violence"]},
        "pov_style": "third_person_limited",
        "human_approved": True,
    }


@pytest.fixture
def compliant_graph() -> Graph:
    graph = Graph()
    graph.create_node("vision", _build_compliant_vision())
    return graph


def test_valid_graph_passes(compliant_graph: Graph) -> None:
    assert validate_dream_output(compliant_graph) == []


def test_R_1_7_no_vision_node() -> None:
    graph = Graph()
    errors = validate_dream_output(graph)
    assert errors, "expected error for missing vision node"
    assert any("vision" in e.lower() for e in errors)


def test_R_1_7_two_vision_nodes() -> None:
    graph = Graph()
    graph.create_node("vision", _build_compliant_vision())
    graph.create_node("vision::extra", {**_build_compliant_vision(), "raw_id": "extra"})
    errors = validate_dream_output(graph)
    assert any("exactly one" in e.lower() or "vision node" in e.lower() for e in errors)


@pytest.mark.parametrize("missing_field", ["genre", "tone", "themes", "audience", "scope"])
def test_R_1_8_required_field_missing(missing_field: str) -> None:
    payload = _build_compliant_vision()
    del payload[missing_field]
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any(missing_field in e for e in errors), (
        f"expected an error mentioning '{missing_field}', got {errors}"
    )


@pytest.mark.parametrize(
    "empty_field,empty_value",
    [
        ("genre", ""),
        ("tone", []),
        ("themes", []),
        ("audience", ""),
    ],
)
def test_R_1_8_required_field_empty(empty_field: str, empty_value: object) -> None:
    payload = _build_compliant_vision()
    payload[empty_field] = empty_value
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any(empty_field in e for e in errors)


def test_R_1_9_invalid_pov_style() -> None:
    payload = _build_compliant_vision()
    payload["pov_style"] = "omniscient"  # not in allowed set
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any("pov_style" in e for e in errors)


def test_R_1_9_pov_style_absent_is_ok(compliant_graph: Graph) -> None:
    # pov_style is optional per R-1.9.
    node = compliant_graph.get_node("vision")
    assert node is not None
    data = dict(node)
    data.pop("pov_style", None)
    graph = Graph()
    graph.create_node("vision", data)
    assert validate_dream_output(graph) == []


def test_R_1_10_vision_has_no_edges(compliant_graph: Graph) -> None:
    # Adding a dummy node and an edge from vision to it violates R-1.10.
    compliant_graph.create_node("entity::kay", {"type": "entity", "name": "Kay"})
    compliant_graph.add_edge("anchored_to", "vision", "entity::kay")
    errors = validate_dream_output(compliant_graph)
    assert any("edge" in e.lower() for e in errors)


def test_output5_no_other_node_types_exist(compliant_graph: Graph) -> None:
    compliant_graph.create_node("entity::kay", {"type": "entity", "name": "Kay"})
    errors = validate_dream_output(compliant_graph)
    assert any("entity" in e.lower() or "other node" in e.lower() for e in errors)


def test_output6_human_approval_recorded() -> None:
    payload = _build_compliant_vision()
    payload["human_approved"] = False
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any("approv" in e.lower() for e in errors)


def test_output6_human_approval_missing() -> None:
    payload = _build_compliant_vision()
    del payload["human_approved"]
    graph = Graph()
    graph.create_node("vision", payload)
    errors = validate_dream_output(graph)
    assert any("approv" in e.lower() for e in errors)
