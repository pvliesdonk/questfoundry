"""BRAINSTORM Stage Output Contract validator.

Validates the graph satisfies every rule in
docs/design/procedures/brainstorm.md §Stage Output Contract.

Called at BRAINSTORM exit (from apply_brainstorm_mutations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class BrainstormContractError(ValueError):
    """Raised when BRAINSTORM's Stage Output Contract is violated."""


_ALLOWED_ENTITY_CATEGORIES = frozenset({"character", "location", "object", "faction"})

_FORBIDDEN_NODE_TYPES = frozenset(
    {"path", "beat", "consequence", "state_flag", "passage", "intersection_group"}
)


def validate_brainstorm_output(graph: Graph) -> list[str]:
    """Verify the graph satisfies BRAINSTORM's Stage Output Contract.

    Returns:
        List of human-readable error strings. Empty means compliant.
        Pure read-only — never mutates the graph.
    """
    errors: list[str] = []

    # Output-11: vision node must still be present (BRAINSTORM must not remove it)
    vision_nodes = graph.get_nodes_by_type("vision")
    if not vision_nodes:
        errors.append("Output-11: vision node is missing (BRAINSTORM must not remove it)")

    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    # R-1.1: minimum floors — at least one entity and one dilemma
    if not entity_nodes:
        errors.append("R-1.1: BRAINSTORM must produce at least one entity")
    if not dilemma_nodes:
        errors.append("R-1.1: BRAINSTORM must produce at least one dilemma")

    # R-2.1 / R-2.2 / R-2.3 / R-2.4: entity field checks
    location_count = 0
    for entity_id, entity in entity_nodes.items():
        category = entity.get("category")
        if not entity.get("name"):
            errors.append(f"R-2.1: entity {entity_id!r} has empty/missing name")
        if not category:
            errors.append(f"R-2.1: entity {entity_id!r} has empty/missing category")
        if not entity.get("concept"):
            errors.append(f"R-2.1: entity {entity_id!r} has empty/missing concept")

        if category and category not in _ALLOWED_ENTITY_CATEGORIES:
            errors.append(
                f"R-2.2: entity {entity_id!r} has invalid category {category!r}; "
                f"must be one of {sorted(_ALLOWED_ENTITY_CATEGORIES)}"
            )

        if "::" not in entity_id:
            errors.append(
                f"R-2.3: entity id {entity_id!r} missing category namespace prefix "
                "(expected e.g. 'character::...', 'location::...')"
            )
        elif category:
            prefix = entity_id.split("::", 1)[0]
            if prefix != category:
                errors.append(
                    f"R-2.3: entity id {entity_id!r} prefix {prefix!r} "
                    f"does not match category {category!r}"
                )

        if category == "location":
            location_count += 1

    if location_count < 2:
        errors.append(
            f"R-2.4: BRAINSTORM must produce at least 2 location entities, found {location_count}"
        )

    # Gather edges once for dilemma checks.
    has_answer_edges = graph.get_edges(edge_type="has_answer")
    anchored_to_edges = graph.get_edges(edge_type="anchored_to")

    answers_per_dilemma: dict[str, list[str]] = {}
    for edge in has_answer_edges:
        answers_per_dilemma.setdefault(edge["from"], []).append(edge["to"])

    anchors_per_dilemma: dict[str, list[str]] = {}
    for edge in anchored_to_edges:
        anchors_per_dilemma.setdefault(edge["from"], []).append(edge["to"])

    # R-3.1 / R-3.2 / R-3.4 / R-3.5 / R-3.6 / R-3.7: dilemma + answer checks
    for dilemma_id, dilemma in dilemma_nodes.items():
        # R-3.7: dilemma id must start with 'dilemma::'
        if not dilemma_id.startswith("dilemma::"):
            errors.append(f"R-3.7: dilemma id {dilemma_id!r} missing 'dilemma::' prefix")

        # R-3.1: question must be present and end with '?'
        question = dilemma.get("question")
        if not question:
            errors.append(f"R-3.1: dilemma {dilemma_id!r} has empty/missing question")
        elif not question.rstrip().endswith("?"):
            errors.append(
                f"R-3.1: dilemma {dilemma_id!r} question must end with '?' (got {question!r})"
            )
        # R-3.1: why_it_matters must be present
        if not dilemma.get("why_it_matters"):
            errors.append(f"R-3.1: dilemma {dilemma_id!r} has empty/missing why_it_matters")

        # R-3.6: must have at least one anchored_to edge to an entity
        anchors = anchors_per_dilemma.get(dilemma_id, [])
        if not anchors:
            errors.append(f"R-3.6: dilemma {dilemma_id!r} has no anchored_to edge to an entity")

        # R-3.2: must have exactly 2 distinct answers
        answers = answers_per_dilemma.get(dilemma_id, [])
        distinct_answers = set(answers)
        if len(distinct_answers) != 2:
            errors.append(
                f"R-3.2: dilemma {dilemma_id!r} must have exactly 2 distinct "
                f"has_answer edges, got {len(distinct_answers)}"
            )

        # R-3.4 / R-3.5: answer field checks
        canonical_count = 0
        for ans_id in distinct_answers:
            ans = graph.get_node(ans_id) or {}
            if ans.get("is_canonical") is True:
                canonical_count += 1
            if not ans.get("description"):
                errors.append(f"R-3.5: answer {ans_id!r} has empty/missing description")
        if len(distinct_answers) >= 2 and canonical_count != 1:
            errors.append(
                f"R-3.4: dilemma {dilemma_id!r} must have exactly one canonical answer, "
                f"found {canonical_count}"
            )

    # R-3.8: forbidden node types must not be present
    for node_type in _FORBIDDEN_NODE_TYPES:
        forbidden = graph.get_nodes_by_type(node_type)
        if forbidden:
            errors.append(
                f"R-3.8: BRAINSTORM must not create {node_type!r} nodes; "
                f"found {len(forbidden)}: {sorted(forbidden.keys())[:3]}"
            )

    return errors
