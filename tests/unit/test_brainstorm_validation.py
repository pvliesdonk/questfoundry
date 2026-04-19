"""Tests for BRAINSTORM Stage Output Contract validator."""

from __future__ import annotations

import pytest

from questfoundry.graph.brainstorm_validation import validate_brainstorm_output
from questfoundry.graph.graph import Graph


def _seed_vision(graph: Graph) -> None:
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "subgenre": "mystery",
            "tone": ["atmospheric"],
            "themes": ["forbidden knowledge"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "pov_style": "third_person_limited",
            "human_approved": True,
        },
    )


def _seed_entity(graph: Graph, entity_id: str, category: str, name: str = "X") -> None:
    graph.create_node(
        entity_id,
        {
            "type": "entity",
            "raw_id": entity_id.split("::", 1)[-1],
            "name": name,
            "category": category,
            "concept": "one-line essence",
        },
    )


def _seed_dilemma(
    graph: Graph,
    dilemma_id: str,
    anchored_to: list[str],
    answers: list[tuple[str, bool]],
) -> None:
    graph.create_node(
        dilemma_id,
        {
            "type": "dilemma",
            "raw_id": dilemma_id.split("::", 1)[-1],
            "question": "What matters?",
            "why_it_matters": "Because.",
        },
    )
    for target in anchored_to:
        graph.add_edge("anchored_to", dilemma_id, target)
    for raw, is_canonical in answers:
        ans_id = f"{dilemma_id}::alt::{raw}"
        graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": raw,
                "description": f"desc-{raw}",
                "is_canonical": is_canonical,
            },
        )
        graph.add_edge("has_answer", dilemma_id, ans_id)


@pytest.fixture
def compliant_graph() -> Graph:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character", "Kay")
    _seed_entity(graph, "character::mentor", "character", "Mentor")
    _seed_entity(graph, "location::archive", "location", "Archive")
    _seed_entity(graph, "location::depths", "location", "Forbidden Depths")
    _seed_entity(graph, "object::cipher", "object", "Cipher")
    _seed_dilemma(
        graph,
        "dilemma::mentor_trust",
        anchored_to=["character::mentor", "character::kay"],
        answers=[("protector", True), ("manipulator", False)],
    )
    return graph


def test_valid_graph_passes(compliant_graph: Graph) -> None:
    assert validate_brainstorm_output(compliant_graph) == []


@pytest.mark.parametrize("missing_field", ["name", "category", "concept"])
def test_R_2_1_entity_missing_field(compliant_graph: Graph, missing_field: str) -> None:
    compliant_graph.update_node("character::kay", **{missing_field: None})
    errors = validate_brainstorm_output(compliant_graph)
    assert any("character::kay" in e and missing_field in e for e in errors)


def test_R_2_2_invalid_entity_category(compliant_graph: Graph) -> None:
    compliant_graph.update_node("character::kay", category="ally")
    errors = validate_brainstorm_output(compliant_graph)
    assert any("category" in e and "ally" in e for e in errors)


def test_R_2_4_insufficient_location_entities() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character")
    _seed_entity(graph, "location::archive", "location")
    _seed_dilemma(
        graph,
        "dilemma::x",
        anchored_to=["character::kay"],
        answers=[("a", True), ("b", False)],
    )
    errors = validate_brainstorm_output(graph)
    assert any("location" in e.lower() and "2" in e for e in errors)


def test_R_2_3_entity_id_missing_category_prefix() -> None:
    graph = Graph()
    _seed_vision(graph)
    graph.create_node(
        "kay",
        {"type": "entity", "raw_id": "kay", "name": "Kay", "category": "character", "concept": "x"},
    )
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    _seed_dilemma(
        graph,
        "dilemma::x",
        anchored_to=["location::a"],
        answers=[("a", True), ("b", False)],
    )
    errors = validate_brainstorm_output(graph)
    assert any("kay" in e and ("prefix" in e or "namespace" in e) for e in errors)


def test_R_3_1_dilemma_question_missing_qmark(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", question="not a question")
    errors = validate_brainstorm_output(compliant_graph)
    assert any("question" in e and "?" in e for e in errors)


def test_R_3_1_dilemma_missing_why_it_matters(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", why_it_matters=None)
    errors = validate_brainstorm_output(compliant_graph)
    assert any("why_it_matters" in e for e in errors)


def test_R_3_2_dilemma_not_binary(compliant_graph: Graph) -> None:
    compliant_graph.create_node(
        "dilemma::mentor_trust::alt::third",
        {"type": "answer", "raw_id": "third", "description": "d", "is_canonical": False},
    )
    compliant_graph.add_edge(
        "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::third"
    )
    errors = validate_brainstorm_output(compliant_graph)
    assert any("has_answer" in e or "answer" in e.lower() for e in errors)


def test_R_3_5_answer_description_empty(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust::alt::protector", description="")
    errors = validate_brainstorm_output(compliant_graph)
    assert any("description" in e for e in errors)


def test_R_3_4_no_canonical_answer(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust::alt::protector", is_canonical=False)
    errors = validate_brainstorm_output(compliant_graph)
    assert any("canonical" in e for e in errors)


def test_R_3_4_two_canonical_answers(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust::alt::manipulator", is_canonical=True)
    errors = validate_brainstorm_output(compliant_graph)
    assert any("canonical" in e for e in errors)


def test_R_3_6_dilemma_missing_anchored_to() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character")
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    graph.create_node(
        "dilemma::orphan",
        {
            "type": "dilemma",
            "raw_id": "orphan",
            "question": "Why?",
            "why_it_matters": "stakes",
        },
    )
    for raw, is_canonical in [("yes", True), ("no", False)]:
        ans_id = f"dilemma::orphan::alt::{raw}"
        graph.create_node(
            ans_id,
            {"type": "answer", "raw_id": raw, "description": "d", "is_canonical": is_canonical},
        )
        graph.add_edge("has_answer", "dilemma::orphan", ans_id)
    errors = validate_brainstorm_output(graph)
    assert any("dilemma::orphan" in e and "anchored_to" in e for e in errors)


def test_R_3_7_dilemma_id_missing_prefix() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "character::kay", "character")
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    graph.create_node(
        "mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Q?",
            "why_it_matters": "stakes",
        },
    )
    graph.add_edge("anchored_to", "mentor_trust", "character::kay")
    for raw, is_canonical in [("yes", True), ("no", False)]:
        ans_id = f"mentor_trust::alt::{raw}"
        graph.create_node(
            ans_id,
            {"type": "answer", "raw_id": raw, "description": "d", "is_canonical": is_canonical},
        )
        graph.add_edge("has_answer", "mentor_trust", ans_id)
    errors = validate_brainstorm_output(graph)
    assert any("prefix" in e.lower() or "dilemma::" in e for e in errors)


@pytest.mark.parametrize(
    "forbidden",
    ["path", "beat", "consequence", "state_flag", "passage", "intersection_group"],
)
def test_R_3_8_forbidden_node_type_present(compliant_graph: Graph, forbidden: str) -> None:
    compliant_graph.create_node(f"{forbidden}::x", {"type": forbidden, "raw_id": "x"})
    errors = validate_brainstorm_output(compliant_graph)
    assert any(forbidden in e for e in errors)


def test_output11_vision_still_exists(compliant_graph: Graph) -> None:
    compliant_graph.delete_node("vision")
    errors = validate_brainstorm_output(compliant_graph)
    assert any("vision" in e.lower() for e in errors)


def test_R_1_1_no_entities() -> None:
    graph = Graph()
    _seed_vision(graph)
    errors = validate_brainstorm_output(graph)
    assert any("entit" in e.lower() for e in errors)


def test_R_1_1_no_dilemmas() -> None:
    graph = Graph()
    _seed_vision(graph)
    _seed_entity(graph, "location::a", "location")
    _seed_entity(graph, "location::b", "location")
    errors = validate_brainstorm_output(graph)
    assert any("dilemma" in e.lower() for e in errors)


def test_output11_vision_corrupted_by_brainstorm(compliant_graph: Graph) -> None:
    """If BRAINSTORM somehow wipes a required vision field, validate_brainstorm_output must fail."""
    compliant_graph.update_node("vision", genre=None)
    errors = validate_brainstorm_output(compliant_graph)
    assert any("genre" in e.lower() for e in errors)
