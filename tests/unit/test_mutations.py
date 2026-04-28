"""Tests for stage mutation appliers."""

from __future__ import annotations

from typing import Any

import pytest

from questfoundry.graph import Graph, MutationError, SeedMutationError
from questfoundry.graph.mutations import (
    BrainstormMutationError,
    BrainstormValidationError,
    SeedErrorCategory,
    SeedValidationError,
    _backfill_explored_from_paths,
    _format_available_with_suggestions,
    _normalize_id,
    _prefix_id,
    _sort_by_similarity,
    apply_brainstorm_mutations,
    apply_dream_mutations,
    apply_mutations,
    apply_seed_mutations,
    categorize_error,
    categorize_errors,
    has_mutation_handler,
    validate_brainstorm_mutations,
    validate_seed_mutations,
)


def _blocking_errors(errors: list[SeedValidationError]) -> list[SeedValidationError]:
    """Filter out WARNING-category errors, returning only blocking errors."""
    return [e for e in errors if e.category != SeedErrorCategory.WARNING]


def _create_compliant_vision(graph: Graph) -> None:
    """Create a vision node that satisfies DREAM contract."""
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["forbidden knowledge"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )


def _create_compliant_brainstorm_graph(
    *,
    extra_entities: list[tuple[str, dict]] | None = None,
    extra_dilemmas: list[tuple[str, str, list[tuple[str, str, bool]], str]] | None = None,
) -> Graph:
    """Create a graph that fully satisfies the BRAINSTORM output contract.

    This is required for tests that call ``apply_seed_mutations``, since SEED
    exit runs ``validate_seed_output`` which includes ``_check_upstream_contract``
    (BRAINSTORM + DREAM validators).

    The base graph contains:
    - DREAM vision node
    - 1 character entity (``character::mentor``) with name/category/concept
    - 2 location entities (``location::archive``, ``location::tower``) — R-2.4
    - 1 dilemma (``dilemma::trust``) with question/why_it_matters/anchored_to/2 answers

    Args:
        extra_entities: Optional list of (node_id, data) pairs to add beyond the base set.
        extra_dilemmas: Optional list of (raw_id, question, [(answer_raw_id, description,
            is_canonical), ...], anchor_entity_id) tuples for additional dilemmas.

    Returns:
        A fully BRAINSTORM-contract-compliant ``Graph`` instance.
    """
    graph = Graph.empty()
    _create_compliant_vision(graph)
    # 1 character entity
    graph.create_node(
        "character::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "category": "character",
            "name": "Mentor",
            "concept": "Senior archivist with hidden motives",
        },
    )
    # 2 location entities (R-2.4: BRAINSTORM must produce ≥2 locations)
    graph.create_node(
        "location::archive",
        {
            "type": "entity",
            "raw_id": "archive",
            "category": "location",
            "name": "The Archive",
            "concept": "Ancient repository of forbidden texts",
        },
    )
    graph.create_node(
        "location::tower",
        {
            "type": "entity",
            "raw_id": "tower",
            "category": "location",
            "name": "The Tower",
            "concept": "Tall observatory on the hill",
        },
    )
    # 1 dilemma with 2 answers + anchored_to + why_it_matters
    graph.create_node(
        "dilemma::trust",
        {
            "type": "dilemma",
            "raw_id": "trust",
            "question": "Can the mentor be trusted?",
            "why_it_matters": "Trust defines whether the protagonist gains an ally or faces a foe.",
        },
    )
    graph.create_node(
        "dilemma::trust::alt::protector",
        {
            "type": "answer",
            "raw_id": "protector",
            "description": "The mentor genuinely protects the protagonist.",
            "is_canonical": True,
        },
    )
    graph.create_node(
        "dilemma::trust::alt::manipulator",
        {
            "type": "answer",
            "raw_id": "manipulator",
            "description": "The mentor is secretly manipulating for personal gain.",
            "is_canonical": False,
        },
    )
    graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::protector")
    graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::manipulator")
    graph.add_edge("anchored_to", "dilemma::trust", "character::mentor")

    # Add any caller-requested extra entities
    for node_id, data in extra_entities or []:
        graph.create_node(node_id, data)

    # Add any caller-requested extra dilemmas
    for raw_id, question, answers, anchor_entity_id in extra_dilemmas or []:
        d_id = f"dilemma::{raw_id}"
        graph.create_node(
            d_id,
            {
                "type": "dilemma",
                "raw_id": raw_id,
                "question": question,
                "why_it_matters": f"This choice is central to the story of {raw_id}.",
            },
        )
        for ans_raw, ans_desc, ans_canonical in answers:
            ans_id = f"{d_id}::alt::{ans_raw}"
            graph.create_node(
                ans_id,
                {
                    "type": "answer",
                    "raw_id": ans_raw,
                    "description": ans_desc,
                    "is_canonical": ans_canonical,
                },
            )
            graph.add_edge("has_answer", d_id, ans_id)
        graph.add_edge("anchored_to", d_id, anchor_entity_id)

    return graph


class TestHasMutationHandler:
    """Test the mutation handler check function."""

    def test_returns_true_for_dream(self) -> None:
        """Dream stage has a mutation handler."""
        assert has_mutation_handler("dream") is True

    def test_returns_true_for_brainstorm(self) -> None:
        """Brainstorm stage has a mutation handler."""
        assert has_mutation_handler("brainstorm") is True

    def test_returns_true_for_seed(self) -> None:
        """Seed stage has a mutation handler."""
        assert has_mutation_handler("seed") is True

    def test_returns_false_for_grow(self) -> None:
        """Grow stage handles its own mutations directly, not via apply_mutations."""
        assert has_mutation_handler("grow") is False

    def test_returns_false_for_unknown_stage(self) -> None:
        """Unknown stages don't have mutation handlers."""
        assert has_mutation_handler("fill") is False
        assert has_mutation_handler("ship") is False
        assert has_mutation_handler("mock") is False
        assert has_mutation_handler("nonexistent") is False


class TestPrefixId:
    """Test the _prefix_id helper function."""

    @pytest.mark.parametrize(
        ("node_type", "raw_id", "expected"),
        [
            pytest.param("entity", "the_detective", "entity::the_detective", id="raw_id-entity"),
            pytest.param(
                "dilemma", "host_motivation", "dilemma::host_motivation", id="raw_id-dilemma"
            ),
            pytest.param("path", "main_path", "path::main_path", id="raw_id-path"),
            pytest.param(
                "entity",
                "entity::the_detective",
                "entity::the_detective",
                id="correctly_prefixed-entity",
            ),
            pytest.param(
                "dilemma",
                "dilemma::host_motivation",
                "dilemma::host_motivation",
                id="correctly_prefixed-dilemma",
            ),
            pytest.param(
                "entity",
                "dilemma::the_detective",
                "entity::the_detective",
                id="wrong_prefix-entity",
            ),
            pytest.param(
                "dilemma",
                "entity::host_motivation",
                "dilemma::host_motivation",
                id="wrong_prefix-dilemma",
            ),
            pytest.param(
                "dilemma",
                "dilemma::dilemma::host_motivation",
                "dilemma::host_motivation",
                id="double_prefix",
            ),
        ],
    )
    def test_prefix_id(self, node_type: str, raw_id: str, expected: str) -> None:
        """Tests _prefix_id with various inputs including raw, prefixed, and double-prefixed IDs."""
        assert _prefix_id(node_type, raw_id) == expected


class TestApplyMutations:
    """Test the mutation router."""

    def test_routes_to_dream(self) -> None:
        """Routes dream stage to apply_dream_mutations."""
        graph = Graph.empty()
        output = {
            "genre": "noir",
            "themes": ["trust"],
            "tone": ["dark"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        }

        apply_mutations(graph, "dream", output)

        assert graph.has_node("vision")

    def test_routes_to_brainstorm(self) -> None:
        """Routes brainstorm stage to apply_brainstorm_mutations."""
        graph = Graph.empty()

        # DREAM stage must run first to create vision node
        dream_output = {
            "genre": "test",
            "tone": ["neutral"],
            "themes": ["test"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        }
        apply_mutations(graph, "dream", dream_output)
        graph.set_last_stage("dream")

        output = {
            "entities": [
                {
                    "entity_id": "char_001",
                    "entity_category": "character",
                    "name": "Character",
                    "concept": "Test",
                },
                {
                    "entity_id": "loc_001",
                    "entity_category": "location",
                    "name": "Place 1",
                    "concept": "Place 1",
                },
                {
                    "entity_id": "loc_002",
                    "entity_category": "location",
                    "name": "Place 2",
                    "concept": "Place 2",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "d1",
                    "question": "Q?",
                    "why_it_matters": "stakes",
                    "central_entity_ids": ["char_001"],
                    "answers": [
                        {"answer_id": "a", "description": "A", "is_canonical": True},
                        {"answer_id": "b", "description": "B", "is_canonical": False},
                    ],
                }
            ],
        }

        apply_mutations(graph, "brainstorm", output)

        # Entities now use category prefix (character::, location::, etc.)
        assert graph.has_node("character::char_001")

    def test_routes_to_seed(self) -> None:
        """Routes seed stage to apply_seed_mutations."""
        # Use a fully BRAINSTORM-compliant graph (required by validate_seed_output's
        # _check_upstream_contract, which now runs at SEED exit).
        graph = _create_compliant_brainstorm_graph()
        # Also mark the mentor entity as "proposed" so SEED can update its disposition.
        graph.get_node("character::mentor")["disposition"] = "proposed"

        output = {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["protector", "manipulator"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "name": "Protector Arc",
                    "description": "The mentor protects.",
                    "path_importance": "major",
                },
                {
                    "path_id": "trust__manipulator",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "name": "Manipulator Arc",
                    "description": "The mentor manipulates.",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "trust__protector",
                    "description": "Mentor's protection bears fruit.",
                    "narrative_effects": ["protagonist gains powerful ally"],
                },
                {
                    "consequence_id": "c_manipulator",
                    "path_id": "trust__manipulator",
                    "description": "Mentor's manipulation is exposed.",
                    "narrative_effects": ["protagonist must face danger alone"],
                },
            ],
            "human_approved_paths": True,
            "initial_beats": [
                {
                    "beat_id": "shared_pre",
                    "summary": "Both players encounter the mentor's cryptic hint.",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "commit_protector",
                    "summary": "Protagonist trusts the mentor.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_protector_1",
                    "summary": "Mentor reveals a vital secret.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_protector_2",
                    "summary": "Ally bond confirmed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "commit_manipulator",
                    "summary": "Protagonist distrusts the mentor.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_manipulator_1",
                    "summary": "Mentor's true motive surfaces.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_manipulator_2",
                    "summary": "Protagonist faces danger alone.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
            ],
        }

        apply_mutations(graph, "seed", output)

        assert graph.get_node("character::mentor")["disposition"] == "retained"

    def test_unknown_stage_raises(self) -> None:
        """Unknown stage raises ValueError."""
        graph = Graph.empty()

        with pytest.raises(ValueError, match="Unknown stage"):
            apply_mutations(graph, "unknown", {})


class TestDreamMutations:
    """Test DREAM stage mutations."""

    def test_creates_vision_node(self) -> None:
        """Creates vision node from dream output."""
        graph = Graph.empty()
        output = {
            "genre": "noir mystery",
            "subgenre": "hardboiled",
            "tone": ["dark", "atmospheric"],
            "themes": ["trust", "betrayal"],
            "audience": "adult",
            "style_notes": "First person narration",
            "scope": {"story_size": "short"},
            "human_approved": True,
        }

        apply_dream_mutations(graph, output)

        vision = graph.get_node("vision")
        assert vision is not None
        assert vision["type"] == "vision"
        assert vision["genre"] == "noir mystery"
        assert vision["subgenre"] == "hardboiled"
        assert vision["tone"] == ["dark", "atmospheric"]
        assert vision["themes"] == ["trust", "betrayal"]
        assert vision["audience"] == "adult"
        assert vision["style_notes"] == "First person narration"

    def test_replaces_existing_vision(self) -> None:
        """Replaces existing vision node."""
        graph = Graph.empty()
        graph.create_node("vision", {"type": "vision", "genre": "fantasy"})

        apply_dream_mutations(
            graph,
            {
                "genre": "noir",
                "themes": ["trust"],
                "tone": ["dark"],
                "audience": "adult",
                "scope": {"story_size": "short"},
                "human_approved": True,
            },
        )

        assert graph.get_node("vision")["genre"] == "noir"

    def test_handles_optional_fields(self) -> None:
        """Handles missing optional fields gracefully."""
        graph = Graph.empty()
        output = {
            "genre": "mystery",
            "themes": ["intrigue"],
            "tone": ["suspenseful"],
            "audience": "adult",
            "scope": {"story_size": "medium"},
            # No subgenre, style_notes, content_notes
            "human_approved": True,
        }

        apply_dream_mutations(graph, output)

        vision = graph.get_node("vision")
        assert vision is not None
        assert "subgenre" not in vision  # Not present, not None
        assert "style_notes" not in vision

    def test_includes_scope_if_present(self) -> None:
        """Includes scope data if present."""
        graph = Graph.empty()
        output = {
            "genre": "fantasy",
            "themes": ["adventure"],
            "tone": ["light"],
            "audience": "ya",
            "scope": {"story_size": "medium"},
            "human_approved": True,
        }

        apply_dream_mutations(graph, output)

        vision = graph.get_node("vision")
        assert vision is not None
        assert vision["scope"]["story_size"] == "medium"

    def test_includes_pov_fields_if_present(self) -> None:
        """Includes POV hint fields if present."""
        graph = Graph.empty()
        output = {
            "genre": "horror",
            "themes": ["fear"],
            "tone": ["tense"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "pov_style": "second_person",
            "protagonist_defined": True,
            "human_approved": True,
        }

        apply_dream_mutations(graph, output)

        vision = graph.get_node("vision")
        assert vision is not None
        assert vision["pov_style"] == "second_person"
        assert vision["protagonist_defined"] is True

    def test_pov_fields_default_correctly(self) -> None:
        """POV fields have correct defaults when not provided."""
        graph = Graph.empty()
        output = {
            "genre": "fantasy",
            "themes": ["adventure"],
            "tone": ["light"],
            "audience": "ya",
            "scope": {"story_size": "medium"},
            # No pov_style or protagonist_defined
            "human_approved": True,
        }

        apply_dream_mutations(graph, output)

        vision = graph.get_node("vision")
        assert vision is not None
        # pov_style should be absent (None values are cleaned)
        assert "pov_style" not in vision
        # protagonist_defined defaults to False
        assert vision["protagonist_defined"] is False


class TestBrainstormMutations:
    """Test BRAINSTORM stage mutations."""

    def test_entity_missing_id_raises(self) -> None:
        """Raises MutationError when entity missing entity_id."""
        graph = Graph.empty()
        output = {
            "entities": [{"entity_category": "character", "concept": "Test"}],  # Missing entity_id
            "dilemmas": [],
        }

        with pytest.raises(
            MutationError, match="Entity at index 0 missing required 'entity_id' field"
        ):
            apply_brainstorm_mutations(graph, output)

    def test_dilemma_missing_id_raises(self) -> None:
        """Raises MutationError when dilemma missing dilemma_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "dilemmas": [{"question": "Test?", "answers": []}],  # Missing dilemma_id
        }

        with pytest.raises(MutationError, match="Dilemma at index 0 missing dilemma_id"):
            apply_brainstorm_mutations(graph, output)

    def test_alternative_missing_id_raises(self) -> None:
        """Raises MutationError when answer missing answer_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "dilemma_001",
                    "question": "Test?",
                    "answers": [
                        {"description": "Option A", "is_canonical": True}
                    ],  # Missing answer_id
                }
            ],
        }

        with pytest.raises(
            MutationError,
            match="Answer at index 0 in dilemma 'dilemma_001' missing answer_id",
        ):
            apply_brainstorm_mutations(graph, output)

    def test_creates_entity_nodes(self) -> None:
        """Creates entity nodes from brainstorm output."""
        graph = Graph.empty()
        _create_compliant_vision(graph)

        output = {
            "entities": [
                {
                    "entity_id": "kay",
                    "entity_category": "character",
                    "name": "Kay",
                    "concept": "Young archivist",
                    "notes": "Curious and brave",
                },
                {
                    "entity_id": "archive",
                    "entity_category": "location",
                    "name": "Archive",
                    "concept": "Ancient repository",
                },
                {
                    "entity_id": "tower",
                    "entity_category": "location",
                    "name": "Tower",
                    "concept": "Tall structure",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "d1",
                    "question": "Q?",
                    "why_it_matters": "stakes",
                    "central_entity_ids": ["kay"],
                    "answers": [
                        {"answer_id": "a", "description": "A", "is_canonical": True},
                        {"answer_id": "b", "description": "B", "is_canonical": False},
                    ],
                }
            ],
        }

        apply_brainstorm_mutations(graph, output)

        # Entity IDs now use category prefix (character::, location::, etc.)
        kay = graph.get_node("character::kay")
        assert kay is not None
        assert kay["type"] == "entity"
        assert kay["raw_id"] == "kay"  # Original ID preserved
        assert kay["category"] == "character"  # Category field instead of entity_type
        assert kay["concept"] == "Young archivist"
        assert kay["notes"] == "Curious and brave"
        assert kay["disposition"] == "proposed"

        archive = graph.get_node("location::archive")
        assert archive is not None
        assert archive["category"] == "location"

    def test_stores_entity_name(self) -> None:
        """Entity.name is stored on graph node when provided (#1010)."""
        graph = Graph.empty()
        _create_compliant_vision(graph)

        output = {
            "entities": [
                {
                    "entity_id": "beatrice",
                    "entity_category": "character",
                    "name": "Lady Beatrice Ashford",
                    "concept": "A sharp-tongued dowager",
                },
                {
                    "entity_id": "manor",
                    "entity_category": "location",
                    "name": "Manor House",
                    "concept": "A crumbling estate",
                },
                {
                    "entity_id": "tower",
                    "entity_category": "location",
                    "name": "The Tower",
                    "concept": "A tall tower",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "d1",
                    "question": "Q?",
                    "why_it_matters": "stakes",
                    "central_entity_ids": ["beatrice"],
                    "answers": [
                        {"answer_id": "a", "description": "A", "is_canonical": True},
                        {"answer_id": "b", "description": "B", "is_canonical": False},
                    ],
                }
            ],
        }

        apply_brainstorm_mutations(graph, output)

        beatrice = graph.get_node("character::beatrice")
        assert beatrice is not None
        assert beatrice["name"] == "Lady Beatrice Ashford"

        manor = graph.get_node("location::manor")
        assert manor is not None
        assert manor["name"] == "Manor House"

    def test_strips_scope_prefixes_in_raw_ids(self) -> None:
        """Scoped IDs are stored unscoped in raw_id fields."""
        graph = Graph.empty()
        _create_compliant_vision(graph)

        output = {
            "entities": [
                {
                    "entity_id": "entity::kay",
                    "entity_category": "character",
                    "name": "Kay",
                    "concept": "Young archivist",
                },
                {
                    "entity_id": "entity::archive",
                    "entity_category": "location",
                    "name": "Archive",
                    "concept": "Archive",
                },
                {
                    "entity_id": "entity::tower",
                    "entity_category": "location",
                    "name": "Tower",
                    "concept": "Tower",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "question": "Can the mentor be trusted?",
                    "why_it_matters": "stakes",
                    "central_entity_ids": ["entity::kay"],
                    "answers": [
                        {
                            "answer_id": "answer::protector",
                            "description": "Mentor protects Kay",
                            "is_canonical": True,
                        },
                        {
                            "answer_id": "answer::manipulator",
                            "description": "Mentor manipulates Kay",
                            "is_canonical": False,
                        },
                    ],
                }
            ],
        }

        apply_brainstorm_mutations(graph, output)

        # Entity is created with category prefix
        entity = graph.get_node("character::kay")
        assert entity is not None
        assert entity["raw_id"] == "kay"

        dilemma = graph.get_node("dilemma::mentor_trust")
        assert dilemma is not None
        assert dilemma["raw_id"] == "mentor_trust"
        # central_entity_ids are stored as anchored_to edges, not node properties
        anchored = graph.get_edges(from_id="dilemma::mentor_trust", edge_type="anchored_to")
        assert {e["to"] for e in anchored} == {"character::kay"}

        protector = graph.get_node("dilemma::mentor_trust::alt::protector")
        assert protector is not None
        assert protector["raw_id"] == "protector"

        manipulator = graph.get_node("dilemma::mentor_trust::alt::manipulator")
        assert manipulator is not None
        assert manipulator["raw_id"] == "manipulator"

    def test_creates_dilemma_with_alternatives(self) -> None:
        """Creates dilemma nodes with linked answers."""
        graph = Graph.empty()
        _create_compliant_vision(graph)

        output = {
            "entities": [
                {
                    "entity_id": "kay",
                    "entity_category": "character",
                    "name": "Kay",
                    "concept": "Protagonist",
                },
                {
                    "entity_id": "mentor",
                    "entity_category": "character",
                    "name": "Mentor",
                    "concept": "Guide",
                },
                {
                    "entity_id": "archive",
                    "entity_category": "location",
                    "name": "Archive",
                    "concept": "Archive",
                },
                {
                    "entity_id": "tower",
                    "entity_category": "location",
                    "name": "Tower",
                    "concept": "Tower",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "mentor_trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "mentor"],  # Raw IDs from LLM
                    "why_it_matters": "Trust is key",
                    "answers": [
                        {
                            "answer_id": "protector",
                            "description": "Mentor protects Kay",
                            "is_canonical": True,
                        },
                        {
                            "answer_id": "manipulator",
                            "description": "Mentor manipulates Kay",
                            "is_canonical": False,
                        },
                    ],
                }
            ],
        }

        apply_brainstorm_mutations(graph, output)

        # Dilemma IDs are prefixed with "dilemma::"
        dilemma = graph.get_node("dilemma::mentor_trust")
        assert dilemma is not None
        assert dilemma["type"] == "dilemma"
        assert dilemma["raw_id"] == "mentor_trust"
        assert dilemma["question"] == "Can the mentor be trusted?"
        # central_entity_ids are stored as anchored_to edges, not node properties
        anchored = graph.get_edges(from_id="dilemma::mentor_trust", edge_type="anchored_to")
        assert {e["to"] for e in anchored} == {"character::kay", "character::mentor"}

        # Alternative IDs: dilemma::dilemma_id::alt::alt_id
        protector = graph.get_node("dilemma::mentor_trust::alt::protector")
        assert protector is not None
        assert protector["type"] == "answer"
        assert protector["raw_id"] == "protector"
        assert protector["is_canonical"] is True

        manipulator = graph.get_node("dilemma::mentor_trust::alt::manipulator")
        assert manipulator is not None
        assert manipulator["is_canonical"] is False

        # Check edges
        edges = graph.get_edges(from_id="dilemma::mentor_trust", edge_type="has_answer")
        assert len(edges) == 2

    def test_handles_empty_brainstorm(self) -> None:
        """Handles minimum required output with locations and one dilemma."""
        graph = Graph.empty()
        _create_compliant_vision(graph)

        output = {
            "entities": [
                {
                    "entity_id": "archive",
                    "entity_category": "location",
                    "name": "Archive",
                    "concept": "Archive",
                },
                {
                    "entity_id": "tower",
                    "entity_category": "location",
                    "name": "Tower",
                    "concept": "Tower",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "d1",
                    "question": "Q?",
                    "why_it_matters": "stakes",
                    "central_entity_ids": ["archive"],  # At least one entity must be anchored
                    "answers": [
                        {"answer_id": "a", "description": "A", "is_canonical": True},
                        {"answer_id": "b", "description": "B", "is_canonical": False},
                    ],
                }
            ],
        }

        apply_brainstorm_mutations(graph, output)

        # Should have vision node + 2 location nodes + 1 dilemma node + 2 answer nodes
        assert len(graph.to_dict()["nodes"]) == 6

    def test_apply_brainstorm_mutations_fails_on_unresolvable_entity(self) -> None:
        """R-3.6: dilemma referencing non-existent entity must raise, not silently drop."""
        graph = Graph.empty()
        output = {
            "entities": [
                {
                    "entity_id": "kay",
                    "entity_category": "character",
                    "name": "Kay",
                    "concept": "archivist",
                },
                {"entity_id": "a", "entity_category": "location", "name": "A", "concept": "x"},
                {"entity_id": "b", "entity_category": "location", "name": "B", "concept": "x"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "question": "Can we trust?",
                    "why_it_matters": "stakes",
                    "central_entity_ids": ["character::ghost"],  # does not exist
                    "answers": [
                        {"answer_id": "yes", "description": "d", "is_canonical": True},
                        {"answer_id": "no", "description": "d", "is_canonical": False},
                    ],
                }
            ],
        }

        with pytest.raises((MutationError, ValueError)) as exc_info:
            apply_brainstorm_mutations(graph, output)
        assert "ghost" in str(exc_info.value) or "anchored_to" in str(exc_info.value).lower()


class TestValidateBrainstormMutations:
    """Test BRAINSTORM semantic validation."""

    def test_valid_output_returns_empty(self) -> None:
        """Valid output with matching entity references returns no errors."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
                {"entity_id": "mentor", "entity_category": "character", "concept": "Mentor"},
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "mentor"],
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert errors == []

    def test_scoped_entity_and_dilemma_ids_are_accepted(self) -> None:
        """Scoped IDs are normalized during brainstorm validation."""
        output = {
            "entities": [
                {
                    "entity_id": "entity::kay",
                    "entity_category": "character",
                    "concept": "Archivist",
                },
                {
                    "entity_id": "entity::mentor",
                    "entity_category": "character",
                    "concept": "Mentor",
                },
                {
                    "entity_id": "entity::archive",
                    "entity_category": "location",
                    "concept": "Archive",
                },
                {
                    "entity_id": "entity::tower",
                    "entity_category": "location",
                    "concept": "Tower",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "dilemma::trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["entity::kay", "entity::mentor"],
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert errors == []

    def test_phantom_entity_reference_detected(self) -> None:
        """Detects when central_entity_ids references non-existent entity."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "phantom_entity"],  # phantom_entity doesn't exist
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        error = errors[0]
        assert "phantom_entity" in error.provided
        assert "kay" in error.available
        assert "central_entity_ids" in error.field_path

    def test_duplicate_alternative_ids_detected(self) -> None:
        """Detects duplicate answer IDs within a dilemma."""
        output = {
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["location::archive"],
                    "answers": [
                        {"answer_id": "option_a", "description": "A", "is_canonical": True},
                        {
                            "answer_id": "option_a",
                            "description": "B",
                            "is_canonical": False,
                        },
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        # Should find: duplicate ID (not default path issue since one is True, one is False)
        duplicate_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(duplicate_errors) == 1

    def test_empty_central_entity_ids_detected(self) -> None:
        """R-3.6: empty central_entity_ids fires the in-retry validator (#1524).

        Defense-in-depth alongside the Pydantic min_length=1 — the semantic
        validator catches None/coercion edge cases that bypass Pydantic and
        delivers the error with the concrete entity ID list (via
        BrainstormMutationError.to_feedback) rather than a generic message.
        """
        output = {
            "entities": [
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],  # R-3.6 violation — empty
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        # The empty central_entity_ids check fires BEFORE the per-ID existence loop,
        # so it produces exactly one error (not one per missing ID).
        assert len(errors) == 1
        assert "no central_entity_ids" in errors[0].issue
        assert "R-3.6" in errors[0].issue
        # Available entity IDs are echoed for the repair feedback path
        assert "archive" in errors[0].available
        assert "tower" in errors[0].available

    def test_missing_central_entity_ids_key_detected(self) -> None:
        """R-3.6: missing central_entity_ids key (None) is treated as empty (#1524)."""
        output = {
            "entities": [
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    # central_entity_ids key intentionally absent
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        assert "no central_entity_ids" in errors[0].issue

    def test_no_default_path_detected(self) -> None:
        """Detects when no answer has is_canonical=True."""
        output = {
            "entities": [
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["location::archive"],
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": False},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        assert "No answer has is_canonical=true" in errors[0].issue

    def test_multiple_default_paths_detected(self) -> None:
        """Detects when multiple answers have is_canonical=True."""
        output = {
            "entities": [
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["location::archive"],
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": True},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        assert "Multiple answers have is_canonical=true" in errors[0].issue

    def test_multiple_errors_collected(self) -> None:
        """Multiple errors across different validations are all collected."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["phantom1", "phantom2"],  # Both invalid
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": False},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        # Should find: 2 phantom entity errors + 1 no default error + 1 location count error
        assert len(errors) == 4

    def test_empty_dilemmas_valid(self) -> None:
        """Empty dilemmas list is valid."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [],
        }

        errors = validate_brainstorm_mutations(output)

        assert errors == []

    def test_empty_alternatives_detected(self) -> None:
        """Dilemma with no answers fails default path validation."""
        output = {
            "entities": [
                {"entity_id": "archive", "entity_category": "location", "concept": "Archive"},
                {"entity_id": "tower", "entity_category": "location", "concept": "Tower"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["location::archive"],
                    "answers": [],  # No answers at all
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        assert "No answer has is_canonical=true" in errors[0].issue

    def test_empty_entity_id_detected(self) -> None:
        """Detects empty or missing entity_id values."""
        output = {
            "entities": [
                {"entity_id": "valid_id", "entity_category": "character", "concept": "Test"},
                {"entity_id": "", "entity_category": "character", "concept": "Empty ID"},
                {"entity_id": None, "entity_category": "location", "concept": "None ID"},
                {"entity_category": "object", "concept": "Missing ID"},  # No entity_id key
            ],
            "dilemmas": [],
        }

        errors = validate_brainstorm_mutations(output)

        # Should find 3 errors for empty/missing entity_ids
        entity_errors = [e for e in errors if "entity_id" in e.field_path]
        assert len(entity_errors) == 3
        # Verify each problematic entity index is reported
        error_paths = {e.field_path for e in entity_errors}
        assert "entities.1.entity_id" in error_paths  # Empty string
        assert "entities.2.entity_id" in error_paths  # None
        assert "entities.3.entity_id" in error_paths  # Missing key

    def test_duplicate_entity_ids_detected(self) -> None:
        """Detects duplicate entity IDs within the same category."""
        output = {
            "entities": [
                {
                    "entity_id": "loc_boathouse_and_dock",
                    "entity_category": "location",
                    "concept": "A boathouse",
                },
                {
                    "entity_id": "loc_boathouse_and_dock",
                    "entity_category": "location",
                    "concept": "A boathouse (duplicate)",
                },
            ],
            "dilemmas": [],
        }

        errors = validate_brainstorm_mutations(output)

        duplicate_errors = [e for e in errors if "Duplicate entity" in e.issue]
        assert len(duplicate_errors) == 1
        assert "location::loc_boathouse_and_dock" in duplicate_errors[0].issue
        assert "index 0" in duplicate_errors[0].issue

    def test_same_raw_id_different_categories_valid(self) -> None:
        """Same raw ID in different categories is valid (different graph nodes)."""
        output = {
            "entities": [
                {
                    "entity_id": "the_archive",
                    "entity_category": "location",
                    "concept": "A building",
                },
                {
                    "entity_id": "the_archive",
                    "entity_category": "faction",
                    "concept": "A secret society",
                },
            ],
            "dilemmas": [],
        }

        errors = validate_brainstorm_mutations(output)

        duplicate_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(duplicate_errors) == 0

    def test_duplicate_dilemma_ids_detected(self) -> None:
        """Detects duplicate dilemma IDs."""
        output = {
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "trust_or_betray",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["location::archive"],
                    "answers": [
                        {"answer_id": "trust", "description": "Trust", "is_canonical": True},
                        {"answer_id": "betray", "description": "Betray", "is_canonical": False},
                    ],
                },
                {
                    "dilemma_id": "trust_or_betray",
                    "question": "Duplicate dilemma",
                    "central_entity_ids": ["location::archive"],
                    "answers": [
                        {"answer_id": "a", "description": "A", "is_canonical": True},
                        {"answer_id": "b", "description": "B", "is_canonical": False},
                    ],
                },
            ],
        }

        errors = validate_brainstorm_mutations(output)

        duplicate_errors = [e for e in errors if "Duplicate dilemma_id" in e.issue]
        assert len(duplicate_errors) == 1
        assert "trust_or_betray" in duplicate_errors[0].issue
        assert "index 0" in duplicate_errors[0].issue

    def test_validate_brainstorm_mutations_requires_two_locations(self) -> None:
        """R-2.4: BRAINSTORM output with <2 locations must fail validation."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "name": "Kay", "concept": "x"},
                {
                    "entity_id": "archive",
                    "entity_category": "location",
                    "name": "Archive",
                    "concept": "x",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "dilemma::x",
                    "question": "Q?",
                    "why_it_matters": "stakes",
                    "central_entity_ids": ["character::kay"],
                    "answers": [
                        {"answer_id": "y", "description": "d", "is_canonical": True},
                        {"answer_id": "n", "description": "d", "is_canonical": False},
                    ],
                }
            ],
        }
        errors = validate_brainstorm_mutations(output)
        assert any("location" in e.issue.lower() for e in errors)


class TestBrainstormMutationError:
    """Test BrainstormMutationError formatting."""

    def test_to_feedback_includes_all_error_info(self) -> None:
        """to_feedback() includes error details for LLM retry."""
        errors = [
            BrainstormValidationError(
                field_path="dilemmas.0.central_entity_ids",
                issue="Entity 'phantom' not in entities list",
                available=["kay", "mentor"],
                provided="phantom",
            )
        ]
        error = BrainstormMutationError(errors)

        feedback = error.to_feedback()

        assert "BRAINSTORM has invalid internal references" in feedback
        assert "dilemmas.0.central_entity_ids" in feedback
        assert "Entity 'phantom' not in entities list" in feedback
        assert "kay" in feedback
        assert "mentor" in feedback

    def test_error_limit_applied(self) -> None:
        """Only shows first 8 errors plus count of remaining."""
        errors = [
            BrainstormValidationError(
                field_path=f"dilemmas.{i}.central_entity_ids",
                issue=f"Entity 'phantom{i}' not in entities list",
                available=[],
                provided=f"phantom{i}",
            )
            for i in range(12)
        ]
        error = BrainstormMutationError(errors)

        feedback = error.to_feedback()

        # Should show first 8 plus "... and 4 more errors"
        assert "phantom7" in feedback  # 8th error (0-indexed)
        assert "phantom8" not in feedback  # 9th error hidden
        assert "... and 4 more errors" in feedback

    def test_to_feedback_inlines_valid_entity_ids_when_available(self) -> None:
        """When errors carry `available` IDs, retry feedback inlines them.

        Per @prompt-engineer Rule 5, the small model on retry has lost the system prompt
        context, so the bottom-line hint must list the authoritative entity IDs
        rather than the generic "use entity_id values from the entities list".
        """
        errors = [
            BrainstormValidationError(
                field_path="dilemmas.0.central_entity_ids",
                issue="Entity 'phantom' not in entities list",
                available=["character::kay", "character::mentor", "location::archive"],
                provided="phantom",
            ),
        ]
        error = BrainstormMutationError(errors)
        feedback = error.to_feedback()

        assert "Valid entity_id values:" in feedback
        assert "`character::kay`" in feedback
        assert "`character::mentor`" in feedback
        assert "`location::archive`" in feedback
        # Generic fallback hint must not appear when we have a real list.
        assert "Use entity_id values from the entities list." not in feedback

    def test_to_feedback_truncates_long_id_list_with_count(self) -> None:
        """When >15 valid IDs are available, feedback shows 15 with a count suffix."""
        long_available = [f"character::npc_{i:02d}" for i in range(20)]
        errors = [
            BrainstormValidationError(
                field_path="dilemmas.0.central_entity_ids",
                issue="Entity 'phantom' not in entities list",
                available=long_available,
                provided="phantom",
            ),
        ]
        error = BrainstormMutationError(errors)
        feedback = error.to_feedback()

        assert "showing 15 of 20" in feedback
        # First 15 (sorted alphabetically) should be present; last 5 should not.
        assert "`character::npc_00`" in feedback
        assert "`character::npc_14`" in feedback
        assert "`character::npc_19`" not in feedback

    def test_to_feedback_falls_back_when_no_available_ids(self) -> None:
        """No `available` IDs anywhere → keep the generic hint, don't emit empty list."""
        errors = [
            BrainstormValidationError(
                field_path="dilemmas.0.answers",
                issue="Duplicate answer_id 'yes' appears 2 times",
                available=[],
                provided="yes",
            ),
        ]
        error = BrainstormMutationError(errors)
        feedback = error.to_feedback()

        assert "Use entity_id values from the entities list." in feedback
        assert "Valid entity_id values:" not in feedback


class TestSeedMutations:
    """Test SEED stage mutations."""

    def test_entity_decision_missing_id_raises(self) -> None:
        """Raises MutationError when entity decision missing entity_id."""
        graph = Graph.empty()
        output = {
            "entities": [{"disposition": "retained"}],  # Missing entity_id
            "paths": [],
            "initial_beats": [],
        }

        with pytest.raises(
            MutationError, match="Entity decision at index 0 missing required 'entity_id' field"
        ):
            apply_seed_mutations(graph, output)

    def test_path_missing_id_raises(self) -> None:
        """Raises MutationError when path missing path_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "paths": [{"name": "Test Path"}],  # Missing path_id
            "initial_beats": [],
        }

        with pytest.raises(MutationError, match="Path at index 0 missing required 'path_id' field"):
            apply_seed_mutations(graph, output)

    def test_beat_missing_id_raises(self) -> None:
        """Raises MutationError when beat missing beat_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "paths": [],
            "initial_beats": [{"summary": "Test Beat"}],  # Missing beat_id
        }

        with pytest.raises(MutationError, match="Beat at index 0 missing required 'beat_id' field"):
            apply_seed_mutations(graph, output)

    def test_updates_entity_dispositions(self) -> None:
        """Updates entity dispositions from seed output."""
        # Use a fully BRAINSTORM-compliant graph (required by validate_seed_output's
        # _check_upstream_contract). The base graph gives us mentor/archive/tower plus
        # a dilemma. We add kay (character) and extra (character) for the triage test.
        graph = _create_compliant_brainstorm_graph(
            extra_entities=[
                (
                    "character::kay",
                    {
                        "type": "entity",
                        "raw_id": "kay",
                        "category": "character",
                        "name": "Kay",
                        "concept": "Young archivist",
                    },
                ),
                (
                    "character::extra",
                    {
                        "type": "entity",
                        "raw_id": "extra",
                        "category": "character",
                        "name": "Extra",
                        "concept": "Minor character",
                    },
                ),
            ]
        )

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},  # Raw IDs from LLM
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "extra", "disposition": "cut"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["protector"], "unexplored": ["manipulator"]},
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "name": "Protector Arc",
                    "description": "Trust path.",
                    "path_importance": "major",
                }
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "trust__protector",
                    "description": "Mentor's protection confirmed.",
                    "narrative_effects": ["protagonist gains ally"],
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "commit",
                    "summary": "Trust decided.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_1",
                    "summary": "Aftermath follows.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_2",
                    "summary": "Ally bond forms.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "human_approved_paths": True,
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "soft",
                    "payoff_budget": 2,
                    "reasoning": "Trust question.",
                    "ending_salience": "low",
                    "residue_weight": "light",
                },
            ],
        }

        apply_seed_mutations(graph, output)

        assert graph.get_node("character::kay")["disposition"] == "retained"
        assert graph.get_node("character::mentor")["disposition"] == "retained"
        assert graph.get_node("character::extra")["disposition"] == "cut"

    def test_creates_paths(self) -> None:
        """Creates path nodes from seed output."""
        # The base compliant graph has dilemma::trust with protector/manipulator answers.
        # We use a separate dilemma name (mentor_trust) to match the test's intent while
        # satisfying the full BRAINSTORM contract required by validate_seed_output.
        graph = _create_compliant_brainstorm_graph(
            extra_dilemmas=[
                (
                    "mentor_trust",
                    "Can the mentor be trusted?",
                    [
                        ("protector", "Mentor protects.", True),
                        ("deceiver", "Mentor deceives.", False),
                    ],
                    "character::mentor",
                ),
            ]
        )

        output = {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                # Base dilemma must also have a decision
                {"dilemma_id": "trust", "explored": ["protector"], "unexplored": ["manipulator"]},
                {"dilemma_id": "mentor_trust", "explored": ["protector"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "path_mentor_trust",
                    "name": "Mentor Trust Arc",
                    "dilemma_id": "mentor_trust",  # Raw dilemma ID from LLM
                    "answer_id": "protector",  # Local alt ID, not full path
                    "description": "Exploring mentor relationship",
                    "path_importance": "major",
                },
                {
                    "path_id": "trust__protector",
                    "name": "Trust Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "description": "Trust the mentor.",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_mentor",
                    "path_id": "path_mentor_trust",
                    "description": "Mentor revealed.",
                    "narrative_effects": ["trust established"],
                },
                {
                    "consequence_id": "c_trust",
                    "path_id": "trust__protector",
                    "description": "Trust confirmed.",
                    "narrative_effects": ["ally gained"],
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "resolution",
                    "summary": "Mentor's true nature revealed.",
                    "path_id": "path_mentor_trust",
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked in"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "resolution_post",
                    "summary": "Consequences of the revelation.",
                    "path_id": "path_mentor_trust",
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "advances", "note": "Fallout"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "resolution_post_2",
                    "summary": "Aftermath settles.",
                    "path_id": "path_mentor_trust",
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "advances", "note": "Settling"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_commit",
                    "summary": "Trust resolved.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_post_1",
                    "summary": "Trust aftermath.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_post_2",
                    "summary": "Trust settled.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "human_approved_paths": True,
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "soft",
                    "payoff_budget": 2,
                    "reasoning": "Trust.",
                    "ending_salience": "low",
                    "residue_weight": "light",
                },
                {
                    "dilemma_id": "mentor_trust",
                    "dilemma_role": "soft",
                    "payoff_budget": 2,
                    "reasoning": "Mentor trust.",
                    "ending_salience": "low",
                    "residue_weight": "light",
                },
            ],
        }

        apply_seed_mutations(graph, output)

        # Path ID is prefixed with "path::"
        path = graph.get_node("path::path_mentor_trust")
        assert path is not None
        assert path["type"] == "path"
        assert path["raw_id"] == "path_mentor_trust"
        assert path["name"] == "Mentor Trust Arc"

        # Check explores edge - links to full prefixed answer ID
        edges = graph.get_edges(from_id="path::path_mentor_trust", edge_type="explores")
        assert len(edges) == 1
        assert edges[0]["to"] == "dilemma::mentor_trust::alt::protector"

    def test_path_is_canonical_from_answer(self) -> None:
        """Path's is_canonical is set from answer's is_canonical.

        The base compliant graph already has dilemma::trust with canonical answer
        'protector' (is_canonical=True) and 'manipulator' (is_canonical=False).
        We explore both answers and verify that paths inherit is_canonical.
        """
        graph = _create_compliant_brainstorm_graph()

        output = {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "explored": ["protector", "manipulator"],
                    "unexplored": [],
                },
            ],
            "paths": [
                {
                    "path_id": "mentor_protects",
                    "name": "Mentor Protects Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",  # Canonical answer
                    "description": "The mentor genuinely protects",
                    "path_importance": "major",
                },
                {
                    "path_id": "mentor_manipulates",
                    "name": "Mentor Manipulates Arc",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",  # Non-canonical answer
                    "description": "The mentor is manipulating",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protects",
                    "path_id": "mentor_protects",
                    "description": "Protection confirmed.",
                    "narrative_effects": ["ally gained"],
                },
                {
                    "consequence_id": "c_manipulates",
                    "path_id": "mentor_manipulates",
                    "description": "Manipulation exposed.",
                    "narrative_effects": ["protagonist alone"],
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "shared_setup",
                    "summary": "Both paths share this setup beat.",
                    "path_id": "mentor_protects",
                    "also_belongs_to": "mentor_manipulates",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "protects_beat_01",
                    "summary": "Mentor reveals protection.",
                    "path_id": "mentor_protects",
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "protects_beat_02",
                    "summary": "Protection confirmed.",
                    "path_id": "mentor_protects",
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "protects_beat_03",
                    "summary": "Ally bond solidifies.",
                    "path_id": "mentor_protects",
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Growth"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "manipulates_beat_01",
                    "summary": "Mentor reveals manipulation.",
                    "path_id": "mentor_manipulates",
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "manipulates_beat_02",
                    "summary": "Manipulation confirmed.",
                    "path_id": "mentor_manipulates",
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                    ],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "manipulates_beat_03",
                    "summary": "Protagonist must stand alone.",
                    "path_id": "mentor_manipulates",
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Hardship"}
                    ],
                    "entities": ["mentor"],
                },
            ],
            "human_approved_paths": True,
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
            ],
        }

        apply_seed_mutations(graph, output)

        # Path from canonical answer should have is_canonical=True
        protects_path = graph.get_node("path::mentor_protects")
        assert protects_path is not None
        assert protects_path.get("is_canonical") is True

        # Path from non-canonical answer should have is_canonical=False
        manipulates_path = graph.get_node("path::mentor_manipulates")
        assert manipulates_path is not None
        assert manipulates_path.get("is_canonical") is False

    def test_creates_beats(self) -> None:
        """Creates beat nodes from seed output with correctly prefixed entity/location IDs."""
        # _create_compliant_brainstorm_graph provides:
        #   character::mentor, location::archive, location::tower, dilemma::trust
        # We add character::kay as an extra entity to reference in beats.
        graph = _create_compliant_brainstorm_graph(
            extra_entities=[
                (
                    "character::kay",
                    {
                        "type": "entity",
                        "raw_id": "kay",
                        "category": "character",
                        "name": "Kay",
                        "concept": "Young archivist",
                    },
                )
            ]
        )

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["protector", "manipulator"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "name": "Protector Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "description": "Mentor protects protagonist.",
                    "path_importance": "major",
                },
                {
                    "path_id": "trust__manipulator",
                    "name": "Manipulator Arc",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "description": "Mentor manipulates protagonist.",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "trust__protector",
                    "description": "Protagonist gains a powerful ally.",
                    "narrative_effects": ["Trust becomes an asset in the final confrontation"],
                },
                {
                    "consequence_id": "c_manipulator",
                    "path_id": "trust__manipulator",
                    "description": "Protagonist must face danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
            ],
            "initial_beats": [
                # Shared pre-commit beat (dual belongs_to via also_belongs_to)
                {
                    "beat_id": "opening_001",
                    "summary": "Kay meets the mentor for the first time",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["kay", "mentor"],
                    "location": "archive",
                },
                # Commit beat for protector path
                {
                    "beat_id": "commit_protector",
                    "summary": "Kay decides to trust the mentor.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["kay", "mentor"],
                },
                # Post-commit beats for protector path (need 2-4)
                {
                    "beat_id": "post_protector_1",
                    "summary": "Mentor reveals a vital secret.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_protector_2",
                    "summary": "Ally bond confirmed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for manipulator path
                {
                    "beat_id": "commit_manipulator",
                    "summary": "Kay distrusts the mentor.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["kay", "mentor"],
                },
                # Post-commit beats for manipulator path
                {
                    "beat_id": "post_manipulator_1",
                    "summary": "Mentor's true motive surfaces.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_manipulator_2",
                    "summary": "Protagonist faces danger alone.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["kay"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question that defines story spine.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
            ],
            "human_approved_paths": True,
        }

        apply_seed_mutations(graph, output)

        # Beat ID is prefixed with "beat::"
        beat = graph.get_node("beat::opening_001")
        assert beat is not None
        assert beat["type"] == "beat"
        assert beat["raw_id"] == "opening_001"
        assert beat["summary"] == "Kay meets the mentor for the first time"
        # Entities are stored with category-prefix (character::) not bare entity::
        assert beat["entities"] == ["character::kay", "character::mentor"]
        assert beat["location"] == "location::archive"

        # Check belongs_to edges — shared pre-commit beat gets dual edges
        edges = graph.get_edges(from_id="beat::opening_001", edge_type="belongs_to")
        assert len(edges) == 2
        to_paths = {e["to"] for e in edges}
        assert "path::trust__protector" in to_paths
        assert "path::trust__manipulator" in to_paths

    def test_temporal_hint_stored_on_beat(self) -> None:
        """Temporal hint is stored on beat node with prefixed dilemma ID.

        Uses the base trust dilemma (protector/manipulator) as the primary dilemma
        and adds fight_or_flee as a second dilemma for the temporal_hint reference.
        """
        graph = _create_compliant_brainstorm_graph(
            extra_dilemmas=[
                (
                    "fight_or_flee",
                    "Fight or flee?",
                    [
                        ("fight", "Character stands and fights.", True),
                        ("flee", "Character escapes the danger.", False),
                    ],
                    "character::mentor",
                ),
            ]
        )

        output = {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["protector", "manipulator"], "unexplored": []},
                {"dilemma_id": "fight_or_flee", "explored": ["fight", "flee"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "name": "Protector Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "description": "Mentor protects.",
                    "path_importance": "major",
                },
                {
                    "path_id": "trust__manipulator",
                    "name": "Manipulator Arc",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "description": "Mentor manipulates.",
                    "path_importance": "major",
                },
                {
                    "path_id": "fight_or_flee__fight",
                    "name": "Fight Arc",
                    "dilemma_id": "fight_or_flee",
                    "answer_id": "fight",
                    "description": "Character fights.",
                    "path_importance": "major",
                },
                {
                    "path_id": "fight_or_flee__flee",
                    "name": "Flee Arc",
                    "dilemma_id": "fight_or_flee",
                    "answer_id": "flee",
                    "description": "Character flees.",
                    "path_importance": "minor",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "trust__protector",
                    "description": "Protagonist gains an ally.",
                    "narrative_effects": ["Trust shapes the climax"],
                },
                {
                    "consequence_id": "c_manipulator",
                    "path_id": "trust__manipulator",
                    "description": "Protagonist faces danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
                {
                    "consequence_id": "c_fight",
                    "path_id": "fight_or_flee__fight",
                    "description": "Character is wounded but prevails.",
                    "narrative_effects": ["Battle scars define later choices"],
                },
                {
                    "consequence_id": "c_flee",
                    "path_id": "fight_or_flee__flee",
                    "description": "Character escapes but loses ground.",
                    "narrative_effects": ["Cowardice haunts later scenes"],
                },
            ],
            "initial_beats": [
                # Shared pre-commit beat for trust dilemma
                {
                    "beat_id": "trust_pre",
                    "summary": "Protagonist encounters the mentor",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for protector — with temporal_hint referencing fight_or_flee
                {
                    "beat_id": "opening_001",
                    "summary": "Kay decides to trust the mentor",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                    "temporal_hint": {
                        "relative_to": "fight_or_flee",
                        "position": "before_commit",
                    },
                },
                # Post-commit beats for protector path
                {
                    "beat_id": "post_protector_1",
                    "summary": "Mentor shares vital information.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_protector_2",
                    "summary": "Ally bond confirmed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for manipulator path
                {
                    "beat_id": "commit_manipulator",
                    "summary": "Kay distrusts the mentor.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats for manipulator path
                {
                    "beat_id": "post_manipulator_1",
                    "summary": "Mentor's true motive surfaces.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_manipulator_2",
                    "summary": "Protagonist faces danger alone.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Shared pre-commit beat for fight_or_flee
                {
                    "beat_id": "fight_pre",
                    "summary": "Danger approaches.",
                    "path_id": "fight_or_flee__fight",
                    "also_belongs_to": "fight_or_flee__flee",
                    "dilemma_impacts": [{"dilemma_id": "fight_or_flee", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beats for fight_or_flee paths
                {
                    "beat_id": "commit_fight",
                    "summary": "Character stands and fights.",
                    "path_id": "fight_or_flee__fight",
                    "dilemma_impacts": [{"dilemma_id": "fight_or_flee", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "commit_flee",
                    "summary": "Character flees the danger.",
                    "path_id": "fight_or_flee__flee",
                    "dilemma_impacts": [{"dilemma_id": "fight_or_flee", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats
                {
                    "beat_id": "post_fight_1",
                    "summary": "Wounds slow the journey.",
                    "path_id": "fight_or_flee__fight",
                    "dilemma_impacts": [{"dilemma_id": "fight_or_flee", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_fight_2",
                    "summary": "Victory is hollow.",
                    "path_id": "fight_or_flee__fight",
                    "dilemma_impacts": [{"dilemma_id": "fight_or_flee", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_flee_1",
                    "summary": "Cowardice haunts.",
                    "path_id": "fight_or_flee__flee",
                    "dilemma_impacts": [{"dilemma_id": "fight_or_flee", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_flee_2",
                    "summary": "Escape has a price.",
                    "path_id": "fight_or_flee__flee",
                    "dilemma_impacts": [{"dilemma_id": "fight_or_flee", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
                {
                    "dilemma_id": "fight_or_flee",
                    "dilemma_role": "soft",
                    "payoff_budget": 1,
                    "reasoning": "Action question with moderate impact.",
                    "ending_salience": "low",
                    "residue_weight": "light",
                },
            ],
            "human_approved_paths": True,
        }

        apply_seed_mutations(graph, output)

        beat = graph.get_node("beat::opening_001")
        assert beat["temporal_hint"] == {
            "relative_to": "dilemma::fight_or_flee",
            "position": "before_commit",
        }

    @staticmethod
    def _trust_compliant_output(commit_beat_extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a fully contract-compliant SEED output using only the trust dilemma.

        Provides two paths (trust__protector, trust__manipulator) with:
        - one shared pre-commit beat (dual belongs_to via also_belongs_to)
        - one commit beat per path
        - two post-commit beats per path
        - consequences with ripples
        - dilemma_analyses with all three required fields

        Args:
            commit_beat_extra: Extra fields merged into the protector commit beat.
        """
        commit_beat: dict[str, Any] = {
            "beat_id": "opening_001",
            "summary": "Kay decides to trust the mentor",
            "path_id": "trust__protector",
            "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
            "entities": ["mentor"],
        }
        if commit_beat_extra:
            commit_beat.update(commit_beat_extra)

        return {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["protector", "manipulator"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "name": "Protector Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "description": "Mentor protects protagonist.",
                    "path_importance": "major",
                },
                {
                    "path_id": "trust__manipulator",
                    "name": "Manipulator Arc",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "description": "Mentor manipulates protagonist.",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "trust__protector",
                    "description": "Protagonist gains an ally.",
                    "narrative_effects": ["Trust shapes the climax"],
                },
                {
                    "consequence_id": "c_manipulator",
                    "path_id": "trust__manipulator",
                    "description": "Protagonist faces danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
            ],
            "initial_beats": [
                # Shared pre-commit beat (dual belongs_to)
                {
                    "beat_id": "trust_pre",
                    "summary": "Protagonist encounters the mentor",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for protector (may have extra fields from caller)
                commit_beat,
                # Post-commit beats for protector
                {
                    "beat_id": "post_protector_1",
                    "summary": "Mentor shares vital information.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_protector_2",
                    "summary": "Ally bond confirmed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for manipulator
                {
                    "beat_id": "commit_manipulator",
                    "summary": "Kay distrusts the mentor.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats for manipulator
                {
                    "beat_id": "post_manipulator_1",
                    "summary": "Mentor's true motive surfaces.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_manipulator_2",
                    "summary": "Protagonist faces danger alone.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question that defines story spine.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
            ],
            "human_approved_paths": True,
        }

    def test_temporal_hint_absent_not_stored(self) -> None:
        """Beat without temporal_hint does not have the key in node data."""
        graph = _create_compliant_brainstorm_graph()
        output = self._trust_compliant_output()

        apply_seed_mutations(graph, output)

        beat = graph.get_node("beat::opening_001")
        # _clean_dict removes None values, so temporal_hint should not be present
        assert "temporal_hint" not in beat

    def test_temporal_hint_partial_not_stored(self) -> None:
        """Temporal hint with null position is discarded, not stored malformed."""
        graph = _create_compliant_brainstorm_graph()
        output = self._trust_compliant_output(
            commit_beat_extra={
                "temporal_hint": {
                    "relative_to": "fight_or_flee",
                    "position": None,
                }
            }
        )

        apply_seed_mutations(graph, output)

        beat = graph.get_node("beat::opening_001")
        # Partial hint (missing position) should be discarded
        assert "temporal_hint" not in beat

    def test_validates_missing_entities(self) -> None:
        """Raises SeedMutationError when referencing non-existent entities."""
        graph = Graph.empty()
        # Only kay exists (with prefixed ID and raw_id for validation)
        graph.create_node(
            "entity::kay",
            {"type": "entity", "raw_id": "kay", "disposition": "proposed"},
        )

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},  # Valid
                {"entity_id": "missing", "disposition": "retained"},  # Doesn't exist
            ],
            "paths": [],
            "initial_beats": [],
        }

        # Should raise SeedMutationError with validation feedback
        with pytest.raises(SeedMutationError) as exc_info:
            apply_seed_mutations(graph, output)

        # Verify error contains helpful feedback
        assert len(exc_info.value.errors) == 1
        error = exc_info.value.errors[0]
        assert "missing" in error.provided
        assert "kay" in error.available  # Shows valid options


class TestSeedCompletenessValidation:
    """Test SEED completeness validation (all entities/dilemmas have decisions)."""

    def test_complete_decisions_valid(self) -> None:
        """All entities and dilemmas have decisions passes validation."""
        graph = Graph.empty()
        # Add entities from BRAINSTORM
        graph.create_node("entity::kay", {"type": "entity", "raw_id": "kay"})
        graph.create_node("entity::mentor", {"type": "entity", "raw_id": "mentor"})
        # Add dilemma from BRAINSTORM
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "cut"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust_arc",
                    "name": "Trust Arc",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "discovery",
                    "summary": "Trust questioned",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Tension builds"}
                    ],
                },
                {
                    "beat_id": "resolution",
                    "summary": "Trust resolved",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "aftermath",
                    "summary": "Consequences unfold",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_missing_entity_decision_detected(self) -> None:
        """Detects when entity from BRAINSTORM has no decision in SEED."""
        graph = Graph.empty()
        # Add entities from BRAINSTORM with categories
        graph.create_node(
            "entity::kay", {"type": "entity", "raw_id": "kay", "entity_type": "character"}
        )
        graph.create_node(
            "entity::mentor", {"type": "entity", "raw_id": "mentor", "entity_type": "character"}
        )
        graph.create_node(
            "entity::archive",
            {"type": "entity", "raw_id": "archive", "entity_type": "location"},
        )

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                # Missing: mentor, archive
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find 2 missing entity decisions with category-specific messages
        entity_errors = [
            e for e in errors if "Missing decision for" in e.issue and "dilemma" not in e.issue
        ]
        assert len(entity_errors) == 2
        missing_ids = {e.issue.split("'")[1] for e in entity_errors}
        assert missing_ids == {"mentor", "archive"}
        # Verify category is included in error message
        assert any("character" in e.issue for e in entity_errors)
        assert any("location" in e.issue for e in entity_errors)

    def test_missing_dilemma_decision_detected(self) -> None:
        """Detects when dilemma from BRAINSTORM has no decision in SEED."""
        graph = Graph.empty()
        # Add dilemmas from BRAINSTORM
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::loyalty", {"type": "dilemma", "raw_id": "loyalty"})

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": [], "unexplored": []},
                # Missing: loyalty
            ],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find 1 missing dilemma decision
        dilemma_errors = [e for e in errors if "Missing decision for dilemma" in e.issue]
        assert len(dilemma_errors) == 1
        assert "loyalty" in dilemma_errors[0].issue

    def test_both_entity_and_dilemma_missing_detected(self) -> None:
        """Detects missing decisions for both entities and dilemmas."""
        graph = Graph.empty()
        # Add entity and dilemma from BRAINSTORM
        graph.create_node(
            "entity::kay", {"type": "entity", "raw_id": "kay", "entity_type": "character"}
        )
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})

        output = {
            "entities": [],  # Missing kay
            "dilemmas": [],  # Missing trust
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find both missing entity and missing dilemma
        # Entity errors use category-specific messages (e.g., "Missing decision for character")
        entity_errors = [
            e for e in errors if "Missing decision for" in e.issue and "dilemma" not in e.issue
        ]
        dilemma_errors = [e for e in errors if "Missing decision for dilemma" in e.issue]
        assert len(entity_errors) == 1
        assert len(dilemma_errors) == 1

    def test_missing_faction_entity_shows_category(self) -> None:
        """Missing faction entity decision shows 'faction' in error message."""
        graph = Graph.empty()
        # Add a faction entity from BRAINSTORM
        graph.create_node(
            "entity::the_family",
            {"type": "entity", "raw_id": "the_family", "entity_type": "faction"},
        )
        graph.create_node(
            "entity::the_detective",
            {"type": "entity", "raw_id": "the_detective", "entity_type": "character"},
        )

        output = {
            "entities": [
                {"entity_id": "the_detective", "disposition": "retained"},
                # Missing: the_family (faction)
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find 1 missing entity decision with "faction" in message
        assert len(errors) == 1
        assert "Missing decision for faction 'the_family'" in errors[0].issue

    def test_dilemma_without_path_detected(self) -> None:
        """Detects when a dilemma has no corresponding path."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::loyalty", {"type": "dilemma", "raw_id": "loyalty"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},
                {"dilemma_id": "loyalty", "explored": [], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust_arc",
                    "name": "Trust Arc",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                }
                # Missing: no path for loyalty
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        path_errors = [e for e in errors if "has no path" in e.issue]
        assert len(path_errors) == 1
        assert "loyalty" in path_errors[0].issue
        assert path_errors[0].category == SeedErrorCategory.COMPLETENESS

    def test_all_dilemmas_with_paths_valid(self) -> None:
        """All dilemmas having paths passes path completeness check."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::loyalty", {"type": "dilemma", "raw_id": "loyalty"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.create_node("dilemma::loyalty::alt::stand", {"type": "answer", "raw_id": "stand"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        graph.add_edge("has_answer", "dilemma::loyalty", "dilemma::loyalty::alt::stand")

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},
                {"dilemma_id": "loyalty", "explored": ["stand"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust_arc",
                    "name": "Trust Arc",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                },
                {
                    "path_id": "loyalty_arc",
                    "name": "Loyalty Arc",
                    "dilemma_id": "loyalty",
                    "answer_id": "stand",
                },
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        path_errors = [e for e in errors if "has no path" in e.issue]
        assert path_errors == []

    def test_scoped_dilemma_id_in_path_satisfies_completeness(self) -> None:
        """Scoped dilemma_id (dilemma::trust) in path satisfies completeness."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [],
            "dilemmas": [{"dilemma_id": "trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "trust_arc",
                    "name": "Trust Arc",
                    "dilemma_id": "dilemma::trust",  # Scoped
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        path_errors = [e for e in errors if "has no path" in e.issue]
        assert path_errors == []

    def test_missing_paths_for_explored_answers(self) -> None:
        """Each explored answer needs its own path - missing paths caught."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.create_node("dilemma::trust::alt::no", {"type": "answer", "raw_id": "no"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::no")

        output = {
            "entities": [],
            "dilemmas": [
                # Both answers explored, but only 1 path created
                {"dilemma_id": "trust", "explored": ["yes", "no"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust_yes",
                    "name": "Trust Yes",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                },
                # Missing path for 'no' answer!
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        missing_path_errors = [e for e in errors if "explored answers" in e.issue]
        assert len(missing_path_errors) == 1
        assert "2 explored answers" in missing_path_errors[0].issue
        assert "1 path" in missing_path_errors[0].issue
        assert missing_path_errors[0].category == SeedErrorCategory.COMPLETENESS

    def test_all_explored_alternatives_have_paths(self) -> None:
        """When each explored answer has a path, validation passes."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.create_node("dilemma::trust::alt::no", {"type": "answer", "raw_id": "no"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::no")

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes", "no"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust_yes",
                    "name": "Trust Yes",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                },
                {
                    "path_id": "trust_no",
                    "name": "Trust No",
                    "dilemma_id": "trust",
                    "answer_id": "no",
                },
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        missing_path_errors = [e for e in errors if "explored answers" in e.issue]
        assert missing_path_errors == []

    # NOTE: Arc count validation tests removed - now handled by runtime pruning
    # (over-generate-and-select pattern in seed_pruning.py)

    def test_empty_brainstorm_valid(self) -> None:
        """Empty BRAINSTORM data (no entities/dilemmas) is valid."""
        graph = Graph.empty()
        # No entities or dilemmas in graph

        output = {
            "entities": [],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_extra_decisions_invalid(self) -> None:
        """Extra decisions for non-existent entities/dilemmas are caught."""
        graph = Graph.empty()
        # Only kay exists
        graph.create_node("entity::kay", {"type": "entity", "raw_id": "kay"})

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "nonexistent", "disposition": "retained"},  # Doesn't exist
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find invalid entity reference (existing check 1)
        invalid_errors = [e for e in errors if "not in BRAINSTORM" in e.issue]
        assert len(invalid_errors) == 1
        assert "nonexistent" in invalid_errors[0].provided


class TestValidation11dDefaultAnswerInExplored:
    """Test SEED validation check 11d: default answer must be in explored bucket."""

    def test_default_answer_in_explored_passes(self) -> None:
        """No error when the default answer is in explored."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node(
            "dilemma::trust::alt::yes",
            {"type": "answer", "raw_id": "yes", "is_canonical": True},
        )
        graph.create_node(
            "dilemma::trust::alt::no",
            {"type": "answer", "raw_id": "no", "is_canonical": False},
        )
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::no")

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes", "no"], "unexplored": []},
            ],
            "paths": [
                {"path_id": "trust_yes", "name": "Yes", "dilemma_id": "trust", "answer_id": "yes"},
                {"path_id": "trust_no", "name": "No", "dilemma_id": "trust", "answer_id": "no"},
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)
        default_errors = [e for e in errors if "default answer" in e.issue]
        assert default_errors == []

    def test_default_answer_in_unexplored_detected(self) -> None:
        """Error when the default answer is in unexplored (inverted buckets)."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node(
            "dilemma::trust::alt::yes",
            {"type": "answer", "raw_id": "yes", "is_canonical": True},
        )
        graph.create_node(
            "dilemma::trust::alt::no",
            {"type": "answer", "raw_id": "no", "is_canonical": False},
        )
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::no")

        output = {
            "entities": [],
            "dilemmas": [
                # Inverted! Default "yes" is in unexplored
                {"dilemma_id": "trust", "explored": ["no"], "unexplored": ["yes"]},
            ],
            "paths": [
                {"path_id": "trust_no", "name": "No", "dilemma_id": "trust", "answer_id": "no"},
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)
        default_errors = [e for e in errors if "default answer" in e.issue]
        assert len(default_errors) == 1
        assert "yes" in default_errors[0].issue
        assert "unexplored" in default_errors[0].issue
        assert default_errors[0].category == SeedErrorCategory.CROSS_REFERENCE

    def test_no_unexplored_skips_check(self) -> None:
        """No error when unexplored is empty (nothing to invert)."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node(
            "dilemma::trust::alt::yes",
            {"type": "answer", "raw_id": "yes", "is_canonical": True},
        )
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},
            ],
            "paths": [
                {"path_id": "trust_yes", "name": "Yes", "dilemma_id": "trust", "answer_id": "yes"},
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)
        default_errors = [e for e in errors if "default answer" in e.issue]
        assert default_errors == []


class TestBeatDilemmaAlignment:
    """Test SEED validation checks 12 and 13: beat-path-dilemma alignment."""

    def _make_graph(self) -> Graph:
        """Create graph with one dilemma and one answer."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        # Second dilemma for cross-reference tests
        graph.create_node("dilemma::loyalty", {"type": "dilemma", "raw_id": "loyalty"})
        graph.create_node(
            "dilemma::loyalty::alt::faithful", {"type": "answer", "raw_id": "faithful"}
        )
        graph.add_edge("has_answer", "dilemma::loyalty", "dilemma::loyalty::alt::faithful")
        return graph

    def _base_output(self) -> dict:
        """Output with one path for dilemma::trust."""
        return {
            "entities": [{"entity_id": "hero", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},
                {"dilemma_id": "loyalty", "explored": ["faithful"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust_arc",
                    "name": "Trust Arc",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [],
        }

    def test_beat_wrong_dilemma_detected(self) -> None:
        """Beat referencing wrong dilemma triggers semantic error."""
        graph = self._make_graph()
        output = self._base_output()
        output["initial_beats"] = [
            {
                "beat_id": "opening",
                "summary": "Start",
                "paths": ["trust_arc"],
                "dilemma_impacts": [
                    # Wrong dilemma - should be 'trust' for path trust_arc
                    {"dilemma_id": "loyalty", "effect": "commits", "note": "Wrong"}
                ],
            }
        ]

        errors = validate_seed_mutations(graph, output)

        alignment_errors = [e for e in errors if "does not reference its parent dilemma" in e.issue]
        assert len(alignment_errors) == 1
        assert "trust_arc" in alignment_errors[0].issue
        assert "trust" in alignment_errors[0].available

    def test_beat_no_dilemma_impacts_detected(self) -> None:
        """Beat with no dilemma_impacts triggers error for missing parent dilemma."""
        graph = self._make_graph()
        output = self._base_output()
        output["initial_beats"] = [
            {
                "beat_id": "opening",
                "summary": "Start",
                "paths": ["trust_arc"],
                # No dilemma_impacts at all
            }
        ]

        errors = validate_seed_mutations(graph, output)

        alignment_errors = [e for e in errors if "does not reference its parent dilemma" in e.issue]
        assert len(alignment_errors) == 1

    def test_path_no_commits_beat_detected(self) -> None:
        """Path without commits beat triggers completeness error."""
        graph = self._make_graph()
        output = self._base_output()
        output["initial_beats"] = [
            {
                "beat_id": "opening",
                "summary": "Start",
                "paths": ["trust_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "trust", "effect": "advances", "note": "Begins"}
                ],
            }
        ]

        errors = validate_seed_mutations(graph, output)

        commits_errors = [e for e in errors if "no beat with effect='commits'" in e.issue]
        assert len(commits_errors) == 1
        assert "trust_arc" in commits_errors[0].issue
        assert commits_errors[0].category == SeedErrorCategory.COMPLETENESS

    def test_beat_with_additional_dilemmas_allowed(self) -> None:
        """Beat can reference additional dilemmas beyond its own path's dilemma."""
        graph = self._make_graph()
        output = self._base_output()
        # Add path for loyalty so check 11 (all dilemmas need paths) passes
        output["paths"].append(
            {
                "path_id": "loyalty_arc",
                "name": "Loyalty Arc",
                "dilemma_id": "loyalty",
                "answer_id": "faithful",
            }
        )
        output["initial_beats"] = [
            {
                "beat_id": "opening",
                "summary": "Start",
                "paths": ["trust_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "trust", "effect": "commits", "note": "Primary"},
                    {"dilemma_id": "loyalty", "effect": "advances", "note": "Secondary"},
                ],
            },
            {
                "beat_id": "opening_post",
                "summary": "Trust aftermath",
                "paths": ["trust_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"},
                ],
            },
            {
                "beat_id": "loyalty_commit",
                "summary": "Loyalty locked",
                "paths": ["loyalty_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "loyalty", "effect": "commits", "note": "Locked"}
                ],
            },
            {
                "beat_id": "loyalty_commit_post",
                "summary": "Loyalty aftermath",
                "paths": ["loyalty_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "loyalty", "effect": "advances", "note": "Fallout"}
                ],
            },
        ]

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []

    def test_multiple_paths_all_need_commits(self) -> None:
        """Each path independently needs a commits beat."""
        graph = self._make_graph()
        output = self._base_output()
        # Add second path for loyalty dilemma
        output["paths"].append(
            {
                "path_id": "loyalty_arc",
                "name": "Loyalty Arc",
                "dilemma_id": "loyalty",
                "answer_id": "faithful",
            }
        )
        output["initial_beats"] = [
            {
                "beat_id": "trust_commit",
                "summary": "Trust resolved",
                "paths": ["trust_arc"],
                "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits", "note": "Locked"}],
            },
            # loyalty_arc has no commits beat
            {
                "beat_id": "loyalty_advance",
                "summary": "Loyalty tested",
                "paths": ["loyalty_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "loyalty", "effect": "advances", "note": "Tested"}
                ],
            },
        ]

        errors = validate_seed_mutations(graph, output)

        commits_errors = [e for e in errors if "no beat with effect='commits'" in e.issue]
        assert len(commits_errors) == 1
        assert "loyalty_arc" in commits_errors[0].issue

    def test_all_paths_with_commits_passes(self) -> None:
        """All paths having commits beats produces no errors."""
        graph = self._make_graph()
        output = self._base_output()
        output["paths"].append(
            {
                "path_id": "loyalty_arc",
                "name": "Loyalty Arc",
                "dilemma_id": "loyalty",
                "answer_id": "faithful",
            }
        )
        output["initial_beats"] = [
            {
                "beat_id": "trust_commit",
                "summary": "Trust resolved",
                "paths": ["trust_arc"],
                "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits", "note": "Locked"}],
            },
            {
                "beat_id": "trust_commit_post",
                "summary": "Trust aftermath",
                "paths": ["trust_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                ],
            },
            {
                "beat_id": "loyalty_commit",
                "summary": "Loyalty locked",
                "paths": ["loyalty_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "loyalty", "effect": "commits", "note": "Locked"}
                ],
            },
            {
                "beat_id": "loyalty_commit_post",
                "summary": "Loyalty aftermath",
                "paths": ["loyalty_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "loyalty", "effect": "advances", "note": "Fallout"}
                ],
            },
        ]

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []


class TestSeedArcStructureValidation:
    """Test SEED validation checks 14-15: arc structure warnings.

    "How Branching Stories Work", Part 2: "This scaffold must be complete —
    the arc from beginning to end must be present." Checks:
    14. Each path has advances/reveals before its commit beat.
    15. Each path has at least one beat after its commit beat.

    These are non-blocking warnings (SeedErrorCategory.WARNING).
    """

    @staticmethod
    def _make_graph() -> Graph:
        """Create a minimal graph with one entity, dilemma, and answer."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        return graph

    @staticmethod
    def _make_output(initial_beats: list[dict[str, Any]]) -> dict[str, Any]:
        """Create minimal valid SEED output with given beats."""
        return {
            "entities": [{"entity_id": "hero", "disposition": "retained"}],
            "dilemmas": [{"dilemma_id": "trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "trust_arc",
                    "name": "Trust Arc",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": initial_beats,
            "human_approved_paths": True,
        }

    def test_complete_arc_no_warnings(self) -> None:
        """Advances → commits → consequence produces zero warnings."""
        graph = self._make_graph()
        output = self._make_output(
            [
                {
                    "beat_id": "develop",
                    "summary": "Trust tested",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Builds tension"}
                    ],
                },
                {
                    "beat_id": "commit",
                    "summary": "Trust decided",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "aftermath",
                    "summary": "Consequences",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ]
        )

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_reveals_before_commit_satisfies_check_14(self) -> None:
        """A 'reveals' beat before commit also satisfies the pre-commit check."""
        graph = self._make_graph()
        output = self._make_output(
            [
                {
                    "beat_id": "reveal",
                    "summary": "Secret uncovered",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "reveals", "note": "Truth emerges"}
                    ],
                },
                {
                    "beat_id": "commit",
                    "summary": "Trust decided",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "aftermath",
                    "summary": "Consequences",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ]
        )

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_missing_pre_commit_development_warns(self) -> None:
        """Commit without prior advances/reveals produces a warning."""
        graph = self._make_graph()
        output = self._make_output(
            [
                {
                    "beat_id": "commit",
                    "summary": "Trust decided",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "aftermath",
                    "summary": "Consequences",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ]
        )

        errors = validate_seed_mutations(graph, output)

        warnings = [e for e in errors if e.category == SeedErrorCategory.WARNING]
        blocking = _blocking_errors(errors)
        assert blocking == []
        assert len(warnings) == 1
        assert "advances" in warnings[0].issue
        assert "reveals" in warnings[0].issue
        assert "before" in warnings[0].issue
        assert warnings[0].field_path == "initial_beats.trust_arc.arc_structure"

    def test_missing_post_commit_beat_warns(self) -> None:
        """Commit without any following beat produces a COMPLETENESS error."""
        graph = self._make_graph()
        output = self._make_output(
            [
                {
                    "beat_id": "develop",
                    "summary": "Trust tested",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Builds tension"}
                    ],
                },
                {
                    "beat_id": "commit",
                    "summary": "Trust decided",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
            ]
        )

        errors = validate_seed_mutations(graph, output)

        blocking = _blocking_errors(errors)
        warnings = [e for e in errors if e.category == SeedErrorCategory.WARNING]
        assert len(blocking) == 1
        assert blocking[0].category == SeedErrorCategory.COMPLETENESS
        assert "after" in blocking[0].issue
        assert "consequence" in blocking[0].issue
        assert blocking[0].field_path == "initial_beats.trust_arc.arc_structure"
        assert warnings == []

    def test_single_commit_beat_produces_both_warnings(self) -> None:
        """A path with only a commit beat gets a pre-commit warning and a post-commit COMPLETENESS error."""
        graph = self._make_graph()
        output = self._make_output(
            [
                {
                    "beat_id": "commit",
                    "summary": "Trust decided",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
            ]
        )

        errors = validate_seed_mutations(graph, output)

        warnings = [e for e in errors if e.category == SeedErrorCategory.WARNING]
        blocking = _blocking_errors(errors)
        # Post-commit missing is now COMPLETENESS (blocking); pre-commit missing stays WARNING
        assert len(blocking) == 1
        assert blocking[0].category == SeedErrorCategory.COMPLETENESS
        assert "after" in blocking[0].issue
        assert len(warnings) == 1
        assert "before" in warnings[0].issue

    def test_complicates_before_commit_does_not_satisfy_check_14(self) -> None:
        """'complicates' is not advances/reveals, so doesn't satisfy the pre-commit check."""
        graph = self._make_graph()
        output = self._make_output(
            [
                {
                    "beat_id": "complication",
                    "summary": "Things get harder",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "complicates", "note": "Setback"}
                    ],
                },
                {
                    "beat_id": "commit",
                    "summary": "Trust decided",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "aftermath",
                    "summary": "Consequences",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ]
        )

        errors = validate_seed_mutations(graph, output)

        warnings = [e for e in errors if e.category == SeedErrorCategory.WARNING]
        assert not _blocking_errors(errors)
        assert len(warnings) == 1
        assert "advances" in warnings[0].issue

    def test_warnings_do_not_block_apply_seed_mutations(self) -> None:
        """apply_seed_mutations succeeds when only WARNING-category issues are present.

        The pre-commit development check (check 14) produces a WARNING when a path
        jumps straight to commits without a prior advances/reveals beat.  This test
        verifies that apply_seed_mutations does NOT raise even in that case.

        Uses _create_compliant_brainstorm_graph() to satisfy the upstream contract,
        then builds a two-path output (trust dilemma) where one path triggers the
        WARNING by having no advances beat before its commit.
        """
        graph = _create_compliant_brainstorm_graph()

        # Use trust__protector / trust__manipulator paths.
        # trust__manipulator has commits without prior advances → WARNING only.
        output = {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "explored": ["protector", "manipulator"],
                    "unexplored": [],
                },
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "name": "Protector Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "description": "Mentor protects.",
                    "path_importance": "major",
                },
                {
                    "path_id": "trust__manipulator",
                    "name": "Manipulator Arc",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "description": "Mentor manipulates.",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "trust__protector",
                    "description": "Protagonist gains an ally.",
                    "narrative_effects": ["Trust shapes the climax"],
                },
                {
                    "consequence_id": "c_manipulator",
                    "path_id": "trust__manipulator",
                    "description": "Protagonist faces danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
            ],
            "initial_beats": [
                # Shared pre-commit beat (only advances protector; manipulator has no advances)
                {
                    "beat_id": "trust_pre",
                    "summary": "Protagonist encounters the mentor",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for protector (has prior advances via shared beat)
                {
                    "beat_id": "commit_protector",
                    "summary": "Trust decided — protagonist sides with mentor.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats for protector
                {
                    "beat_id": "post_protector_1",
                    "summary": "Mentor reveals a vital secret.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_protector_2",
                    "summary": "Ally bond confirmed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for manipulator — no prior advances; triggers WARNING
                {
                    "beat_id": "commit_manipulator",
                    "summary": "Trust betrayed — protagonist distrusts mentor.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats for manipulator
                {
                    "beat_id": "post_manipulator_1",
                    "summary": "Mentor's true motive surfaces.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_manipulator_2",
                    "summary": "Protagonist faces danger alone.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
            ],
            "human_approved_paths": True,
        }

        # Should not raise: pre-commit development check (check 14) is WARNING only
        apply_seed_mutations(graph, output)

        # Verify mutation was applied (beat nodes exist)
        assert graph.get_node("beat::commit_protector") is not None
        assert graph.get_node("beat::post_protector_1") is not None

    def test_path_without_commits_skips_arc_checks(self) -> None:
        """Paths missing a commit beat get an error for check 13, not arc warnings."""
        graph = self._make_graph()
        output = self._make_output(
            [
                {
                    "beat_id": "develop",
                    "summary": "Trust tested",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "trust", "effect": "advances", "note": "Builds tension"}
                    ],
                },
            ]
        )

        errors = validate_seed_mutations(graph, output)

        blocking = _blocking_errors(errors)
        warnings = [e for e in errors if e.category == SeedErrorCategory.WARNING]
        # Should have a COMPLETENESS error for missing commit, no arc warnings
        assert len(blocking) == 1
        assert blocking[0].category == SeedErrorCategory.COMPLETENESS
        assert warnings == []


class TestSeedDuplicateValidation:
    """Test SEED validation detects duplicate entity/dilemma IDs.

    Fixes #239: LLM may output the same entity or dilemma multiple times,
    which should be caught and reported as validation errors.
    """

    def test_duplicate_entity_id_detected(self) -> None:
        """Detects when the same entity_id appears multiple times in output."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                {"entity_id": "hero", "disposition": "retained"},  # Duplicate!
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        dup_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(dup_errors) == 1
        assert "hero" in dup_errors[0].issue
        assert "2 times" in dup_errors[0].issue

    def test_duplicate_dilemma_id_detected(self) -> None:
        """Detects when the same dilemma_id appears multiple times in output."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},  # Duplicate!
                {"dilemma_id": "trust", "explored": ["yes"], "unexplored": []},  # Triple!
            ],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        dup_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(dup_errors) == 1
        assert "trust" in dup_errors[0].issue
        assert "3 times" in dup_errors[0].issue

    def test_duplicate_with_scoped_ids_detected(self) -> None:
        """Detects duplicates even when IDs use different scope prefixes."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                {
                    "entity_id": "entity::hero",
                    "disposition": "retained",
                },  # Same after normalization
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        dup_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(dup_errors) == 1
        assert "hero" in dup_errors[0].issue

    def test_no_duplicates_passes(self) -> None:
        """No errors when all IDs are unique."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("entity::mentor", {"type": "entity", "raw_id": "mentor"})

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "retained"},
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        dup_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(dup_errors) == 0


class TestMutationIntegration:
    """Integration tests for multi-stage mutation flow."""

    def test_dream_brainstorm_seed_flow(self) -> None:
        """Test complete mutation flow across stages."""
        graph = Graph.empty()

        # DREAM stage
        dream_output = {
            "genre": "dark fantasy mystery",
            "tone": ["atmospheric", "morally ambiguous"],
            "themes": ["forbidden knowledge", "trust"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        }
        apply_mutations(graph, "dream", dream_output)
        graph.set_last_stage("dream")

        # BRAINSTORM stage
        brainstorm_output = {
            "entities": [
                {
                    "entity_id": "kay",
                    "entity_category": "character",
                    "name": "Kay",
                    "concept": "Young archivist",
                },
                {
                    "entity_id": "mentor",
                    "entity_category": "character",
                    "name": "Mentor",
                    "concept": "Senior archivist",
                },
                {
                    "entity_id": "archive",
                    "entity_category": "location",
                    "name": "Archive",
                    "concept": "Ancient library",
                },
                {
                    "entity_id": "tower",
                    "entity_category": "location",
                    "name": "Tower",
                    "concept": "Tall observatory",
                },
            ],
            "dilemmas": [
                {
                    "dilemma_id": "mentor_trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "mentor"],  # Raw IDs from LLM
                    "why_it_matters": "Trust defines ally or foe",
                    "answers": [
                        {
                            "answer_id": "protector",
                            "description": "Mentor protects",
                            "is_canonical": True,
                        },
                        {
                            "answer_id": "manipulator",
                            "description": "Mentor manipulates",
                            "is_canonical": False,
                        },
                    ],
                }
            ],
        }
        apply_mutations(graph, "brainstorm", brainstorm_output)
        graph.set_last_stage("brainstorm")

        # SEED stage - uses raw IDs as LLM would produce
        # Explores both answers (protector + manipulator) to satisfy contract.
        seed_output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "mentor_trust",
                    "explored": ["protector", "manipulator"],
                    "unexplored": [],
                },
            ],
            "paths": [
                {
                    "path_id": "mentor_trust__protector",
                    "name": "Protector Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "protector",
                    "description": "The mentor genuinely protects.",
                    "path_importance": "major",
                },
                {
                    "path_id": "mentor_trust__manipulator",
                    "name": "Manipulator Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "manipulator",
                    "description": "The mentor secretly manipulates.",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "mentor_trust__protector",
                    "description": "Protagonist gains a loyal ally.",
                    "narrative_effects": ["Trust becomes a shield in the final confrontation"],
                },
                {
                    "consequence_id": "c_manipulator",
                    "path_id": "mentor_trust__manipulator",
                    "description": "Protagonist must face danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
            ],
            "initial_beats": [
                # Shared pre-commit beat (dual belongs_to)
                {
                    "beat_id": "opening",
                    "summary": "Kay meets the mentor for the first time",
                    "path_id": "mentor_trust__protector",
                    "also_belongs_to": "mentor_trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["kay", "mentor"],
                },
                # Commit beat for protector
                {
                    "beat_id": "commit_protector",
                    "summary": "Kay decides to trust the mentor",
                    "path_id": "mentor_trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats for protector
                {
                    "beat_id": "post_protector_1",
                    "summary": "The mentor reveals a vital secret.",
                    "path_id": "mentor_trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "post_protector_2",
                    "summary": "Ally bond confirmed.",
                    "path_id": "mentor_trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beat for manipulator
                {
                    "beat_id": "commit_manipulator",
                    "summary": "Kay distrusts the mentor.",
                    "path_id": "mentor_trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats for manipulator
                {
                    "beat_id": "opening_post",
                    "summary": "The mentor's nature becomes clear",
                    "path_id": "mentor_trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "manipulator_end",
                    "summary": "Protagonist faces danger alone.",
                    "path_id": "mentor_trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["kay"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "mentor_trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
            ],
            "human_approved_paths": True,
        }
        apply_mutations(graph, "seed", seed_output)
        graph.set_last_stage("seed")

        # Verify final state - entity IDs use category prefix, others use type prefix
        assert graph.get_last_stage() == "seed"
        assert graph.has_node("vision")
        assert graph.has_node("character::kay")  # Category-based entity ID
        assert graph.has_node("character::mentor")  # Category-based entity ID
        assert graph.has_node("dilemma::mentor_trust")
        assert graph.has_node("dilemma::mentor_trust::alt::protector")
        assert graph.has_node("path::mentor_trust__protector")
        assert graph.has_node("beat::opening")

        # Check entity dispositions
        assert graph.get_node("character::kay")["disposition"] == "retained"

        # Check edges — 2 paths now (protector + manipulator both explored)
        assert len(graph.get_edges(edge_type="has_answer")) == 2
        assert len(graph.get_edges(edge_type="explores")) == 2  # one per path
        # belongs_to: shared pre-commit beat has 2 edges; 6 per-path beats have 1 each = 8 total
        assert len(graph.get_edges(edge_type="belongs_to")) == 8

        # Check node counts by type
        assert len(graph.get_nodes_by_type("vision")) == 1
        assert len(graph.get_nodes_by_type("entity")) == 4  # kay, mentor, archive, tower
        assert len(graph.get_nodes_by_type("dilemma")) == 1
        assert len(graph.get_nodes_by_type("answer")) == 2
        assert len(graph.get_nodes_by_type("path")) == 2  # protector + manipulator
        assert len(graph.get_nodes_by_type("beat")) == 7  # 1 shared + 3 per path = 7


class TestSeedErrorCategory:
    """Tests for SeedErrorCategory enum."""

    def test_has_inner_category(self) -> None:
        """INNER category exists for schema errors."""
        assert SeedErrorCategory.INNER is not None

    def test_has_semantic_category(self) -> None:
        """SEMANTIC category exists for invalid ID references."""
        assert SeedErrorCategory.SEMANTIC is not None

    def test_has_completeness_category(self) -> None:
        """COMPLETENESS category exists for missing items."""
        assert SeedErrorCategory.COMPLETENESS is not None

    def test_has_fatal_category(self) -> None:
        """FATAL category exists for unrecoverable errors."""
        assert SeedErrorCategory.FATAL is not None


class TestCategorizeError:
    """Tests for categorize_error function."""

    def test_semantic_not_in_brainstorm(self) -> None:
        """'not in BRAINSTORM' errors are SEMANTIC."""
        error = SeedValidationError(
            field_path="paths.0.dilemma_id",
            issue="Dilemma 'phantom' not in BRAINSTORM",
            available=["real_dilemma"],
            provided="phantom",
        )
        assert categorize_error(error) == SeedErrorCategory.SEMANTIC

    def test_semantic_not_in_seed(self) -> None:
        """'not defined in SEED' errors are SEMANTIC."""
        error = SeedValidationError(
            field_path="initial_beats.0.path_id",
            issue="Path 'ghost' not defined in SEED paths",
            available=["real_path"],
            provided="ghost",
        )
        assert categorize_error(error) == SeedErrorCategory.SEMANTIC

    def test_completeness_missing_decision(self) -> None:
        """'Missing decision' errors are COMPLETENESS."""
        error = SeedValidationError(
            field_path="entities",
            issue="Missing decision for character 'hero'",
            available=[],
            provided="",
        )
        assert categorize_error(error) == SeedErrorCategory.COMPLETENESS

    def test_inner_for_other_errors(self) -> None:
        """Unrecognized errors default to INNER."""
        error = SeedValidationError(
            field_path="paths.0.name",
            issue="Name must not be empty",
            available=[],
            provided="",
        )
        assert categorize_error(error) == SeedErrorCategory.INNER

    def test_case_insensitive_matching(self) -> None:
        """Issue matching is case-insensitive."""
        error = SeedValidationError(
            field_path="entities.0.entity_id",
            issue="Entity 'hero' NOT IN BRAINSTORM",
            available=[],
            provided="hero",
        )
        assert categorize_error(error) == SeedErrorCategory.SEMANTIC


class TestCategorizeErrors:
    """Tests for categorize_errors function."""

    def test_groups_by_category(self) -> None:
        """Groups errors by their category."""
        errors = [
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma 'x' not in BRAINSTORM",
            ),
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for character 'hero'",
            ),
            SeedValidationError(
                field_path="paths.1.dilemma_id",
                issue="Dilemma 'y' not in BRAINSTORM",
            ),
        ]

        by_category = categorize_errors(errors)

        assert len(by_category[SeedErrorCategory.SEMANTIC]) == 2
        assert len(by_category[SeedErrorCategory.COMPLETENESS]) == 1
        assert SeedErrorCategory.INNER not in by_category

    def test_empty_list_returns_empty_dict(self) -> None:
        """Empty error list returns empty dict."""
        assert categorize_errors([]) == {}

    def test_all_same_category(self) -> None:
        """All errors of same category grouped together."""
        errors = [
            SeedValidationError(field_path="a", issue="Missing decision for x"),
            SeedValidationError(field_path="b", issue="Missing decision for y"),
        ]

        by_category = categorize_errors(errors)

        assert len(by_category) == 1
        assert len(by_category[SeedErrorCategory.COMPLETENESS]) == 2


class TestErrorPatternConsistency:
    """Tests that error patterns match actual validation output.

    These tests verify that the pattern constants used by categorize_error()
    match the error messages produced by validate_seed_mutations().
    See issue #216 for the plan to replace string matching with structured codes.
    """

    def test_semantic_errors_match_pattern(self) -> None:
        """Real semantic errors from validate_seed_mutations are categorized correctly."""
        graph = Graph.empty()
        # Add an entity from BRAINSTORM
        graph.create_node("entity::real", {"type": "entity", "raw_id": "real"})

        output = {
            "entities": [
                {"entity_id": "phantom", "disposition": "retained"},  # Not in BRAINSTORM
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Find the semantic error (invalid entity reference)
        semantic_errors = [e for e in errors if categorize_error(e) == SeedErrorCategory.SEMANTIC]
        assert len(semantic_errors) >= 1
        # Verify the error message contains expected pattern
        assert any("not in brainstorm" in e.issue.lower() for e in semantic_errors)

    def test_completeness_errors_match_pattern(self) -> None:
        """Real completeness errors from validate_seed_mutations are categorized correctly."""
        graph = Graph.empty()
        # Add entities that need decisions
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::villain",
            {"type": "entity", "raw_id": "villain", "entity_type": "character"},
        )

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                # Missing: villain
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Find the completeness error (missing decision)
        completeness_errors = [
            e for e in errors if categorize_error(e) == SeedErrorCategory.COMPLETENESS
        ]
        assert len(completeness_errors) == 1
        # Verify the error message contains expected pattern
        assert "missing decision" in completeness_errors[0].issue.lower()


class TestNormalizeId:
    """Tests for _normalize_id helper function."""

    def test_unscoped_id_returned_as_is(self) -> None:
        """Unscoped ID is returned unchanged."""
        normalized, error = _normalize_id("hero", "entity")
        assert normalized == "hero"
        assert error is None

    def test_correctly_scoped_id_strips_prefix(self) -> None:
        """Correctly scoped ID has prefix stripped."""
        normalized, error = _normalize_id("entity::hero", "entity")
        assert normalized == "hero"
        assert error is None

    def test_dilemma_scope_accepted(self) -> None:
        """Dilemma scope is handled correctly."""
        normalized, error = _normalize_id("dilemma::mentor_trust", "dilemma")
        assert normalized == "mentor_trust"
        assert error is None

    def test_path_scope_accepted(self) -> None:
        """Path scope is handled correctly."""
        normalized, error = _normalize_id("path::mentor_arc", "path")
        assert normalized == "mentor_arc"
        assert error is None

    def test_wrong_scope_returns_error(self) -> None:
        """Wrong scope prefix returns error message."""
        normalized, error = _normalize_id("dilemma::hero", "entity")
        # Returns original ID unchanged when scope is wrong
        assert normalized == "dilemma::hero"
        assert error is not None
        # Error should mention entity categories and the wrong scope
        assert "character/location/object/faction" in error
        assert "dilemma::" in error

    def test_entity_scope_rejected_when_path_expected(self) -> None:
        """Entity scope rejected when path is expected."""
        normalized, error = _normalize_id("entity::mentor_arc", "path")
        assert normalized == "entity::mentor_arc"
        assert error is not None
        assert "path::" in error
        assert "entity::" in error

    def test_category_prefix_accepted_for_entity(self) -> None:
        """Category prefixes (character::, location::, etc.) accepted for entities."""
        # character:: prefix
        normalized, error = _normalize_id("character::hero", "entity")
        assert normalized == "hero"
        assert error is None

        # location:: prefix
        normalized, error = _normalize_id("location::manor", "entity")
        assert normalized == "manor"
        assert error is None

        # object:: prefix
        normalized, error = _normalize_id("object::sword", "entity")
        assert normalized == "sword"
        assert error is None

        # faction:: prefix
        normalized, error = _normalize_id("faction::guild", "entity")
        assert normalized == "guild"
        assert error is None

    def test_legacy_entity_prefix_still_accepted(self) -> None:
        """Legacy entity:: prefix still works for backwards compatibility."""
        normalized, error = _normalize_id("entity::hero", "entity")
        assert normalized == "hero"
        assert error is None

    def test_id_with_colons_in_raw_id(self) -> None:
        """IDs with :: only split on first occurrence."""
        # Edge case: raw_id containing ::
        normalized, error = _normalize_id("entity::my::complex::id", "entity")
        assert normalized == "my::complex::id"
        assert error is None

    def test_empty_raw_id_after_scope(self) -> None:
        """Scoped ID with empty raw_id."""
        normalized, error = _normalize_id("entity::", "entity")
        assert normalized == ""
        assert error is None


class TestScopedIdValidation:
    """Tests for scoped ID acceptance in validate_seed_mutations."""

    def test_scoped_entity_id_accepted(self) -> None:
        """Scoped entity IDs (entity::hero) are accepted in entity decisions."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_scoped_dilemma_id_accepted(self) -> None:
        """Scoped dilemma IDs (dilemma::trust) are accepted in dilemma decisions."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [],
            "dilemmas": [{"dilemma_id": "dilemma::trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "trust_arc",
                    "name": "Trust Arc",
                    "dilemma_id": "dilemma::trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "resolution",
                    "summary": "Trust resolved",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "resolution_post",
                    "summary": "Trust aftermath",
                    "paths": ["trust_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []

    def test_scoped_path_id_accepted_in_beats(self) -> None:
        """Scoped path IDs (path::mentor) are accepted in beat path references."""
        graph = Graph.empty()
        # Set up BRAINSTORM data
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "dilemmas": [{"dilemma_id": "dilemma::trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "mentor",
                    "name": "Mentor Arc",
                    "dilemma_id": "dilemma::trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "The beginning",
                    "paths": ["path::mentor"],  # Scoped path ID
                    "entities": ["entity::hero"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "opening_post",
                    "summary": "Trust aftermath",
                    "paths": ["path::mentor"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []

    def test_wrong_scope_detected_for_entity(self) -> None:
        """Wrong scope (dilemma:: instead of entity category) is detected."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})

        output = {
            "entities": [{"entity_id": "dilemma::hero", "disposition": "retained"}],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert len(errors) >= 1
        scope_errors = [e for e in errors if "Wrong scope prefix" in e.issue]
        assert len(scope_errors) == 1
        # Error should mention entity categories and the wrong scope
        assert "character/location/object/faction" in scope_errors[0].issue
        assert "dilemma::" in scope_errors[0].issue

    def test_wrong_scope_detected_for_dilemma(self) -> None:
        """Wrong scope (entity:: instead of dilemma::) is detected."""
        graph = Graph.empty()
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})

        output = {
            "entities": [],
            "dilemmas": [{"dilemma_id": "entity::trust", "explored": [], "unexplored": []}],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert len(errors) >= 1
        scope_errors = [e for e in errors if "Wrong scope prefix" in e.issue]
        assert len(scope_errors) == 1
        assert "dilemma::" in scope_errors[0].issue
        assert "entity::" in scope_errors[0].issue

    def test_wrong_scope_detected_for_path_in_beat(self) -> None:
        """Wrong scope (entity:: instead of path::) in beat path references."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "dilemmas": [{"dilemma_id": "dilemma::trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "mentor",
                    "name": "Mentor Arc",
                    "dilemma_id": "dilemma::trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "The beginning",
                    "paths": ["entity::mentor"],  # Wrong scope - should be path::
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        scope_errors = [e for e in errors if "Wrong scope prefix" in e.issue]
        assert len(scope_errors) == 1
        assert "path::" in scope_errors[0].issue

    def test_mixed_scoped_and_unscoped_ids(self) -> None:
        """Both scoped and unscoped IDs work together."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("entity::villain", {"type": "entity", "raw_id": "villain"})

        output = {
            "entities": [
                {"entity_id": "entity::hero", "disposition": "retained"},  # Scoped
                {"entity_id": "villain", "disposition": "cut"},  # Unscoped
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_scoped_ids_in_completeness_check(self) -> None:
        """Scoped IDs are normalized for completeness checking."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("entity::villain", {"type": "entity", "raw_id": "villain"})

        output = {
            "entities": [
                {"entity_id": "entity::hero", "disposition": "retained"},
                {"entity_id": "entity::villain", "disposition": "cut"},
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # No missing decision errors since scoped IDs should be normalized
        completeness_errors = [e for e in errors if "Missing decision" in e.issue]
        assert completeness_errors == []

    def test_wrong_scope_not_counted_in_completeness(self) -> None:
        """Wrong-scope IDs should not satisfy completeness check."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero", {"type": "entity", "raw_id": "hero", "entity_type": "character"}
        )

        output = {
            "entities": [
                {"entity_id": "dilemma::hero", "disposition": "retained"}  # Wrong scope
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should have both: scope error AND missing decision error
        scope_errors = [e for e in errors if "Wrong scope prefix" in e.issue]
        completeness_errors = [e for e in errors if "Missing decision" in e.issue]
        assert len(scope_errors) == 1
        assert len(completeness_errors) == 1
        assert "hero" in completeness_errors[0].issue

    def test_scoped_entity_in_beat_entities(self) -> None:
        """Scoped entity IDs work in beat.entities array."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("entity::mentor", {"type": "entity", "raw_id": "mentor"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [
                {"entity_id": "entity::hero", "disposition": "retained"},
                {"entity_id": "entity::mentor", "disposition": "retained"},
            ],
            "dilemmas": [{"dilemma_id": "dilemma::trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "dilemma_id": "dilemma::trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Meet mentor",
                    "entities": ["entity::hero", "entity::mentor"],  # Scoped IDs
                    "location": "entity::hero",  # Scoped location
                    "paths": ["path::mentor_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "opening_post",
                    "summary": "Trust aftermath",
                    "paths": ["path::mentor_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # No blocking errors expected (arc structure warnings are non-blocking)
        assert _blocking_errors(errors) == []

    def test_scoped_dilemma_in_dilemma_impacts(self) -> None:
        """Scoped dilemma IDs work in beat.dilemma_impacts."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "dilemmas": [{"dilemma_id": "dilemma::trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "dilemma_id": "dilemma::trust",
                    "answer_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Build trust",
                    "paths": ["path::mentor_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "advances"}  # Scoped ID
                    ],
                },
                {
                    "beat_id": "resolution",
                    "summary": "Trust locked in",
                    "paths": ["path::mentor_arc"],
                    "dilemma_impacts": [{"dilemma_id": "dilemma::trust", "effect": "commits"}],
                },
                {
                    "beat_id": "resolution_post",
                    "summary": "Trust aftermath",
                    "paths": ["path::mentor_arc"],
                    "dilemma_impacts": [{"dilemma_id": "dilemma::trust", "effect": "advances"}],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []

    def test_scoped_path_in_consequences(self) -> None:
        """Scoped path IDs work in consequence.path_id."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "dilemmas": [{"dilemma_id": "dilemma::trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "dilemma_id": "dilemma::trust",
                    "answer_id": "yes",
                }
            ],
            "consequences": [
                {
                    "consequence_id": "trust_earned",
                    "path_id": "path::mentor_arc",  # Scoped path ID
                    "description": "Trust is earned",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "resolution",
                    "summary": "Trust resolved",
                    "paths": ["mentor_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "resolution_post",
                    "summary": "Trust aftermath",
                    "paths": ["mentor_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []

    def test_scoped_path_definitions_and_consequences(self) -> None:
        """Path definitions using scoped IDs work with scoped consequence references.

        Regression test for issue #230: When LLM outputs path definitions with
        scoped IDs (path::foo) and consequences reference them with scoped IDs,
        validation should pass since seed_path_ids is normalized.
        """
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("dilemma::trust", {"type": "dilemma", "raw_id": "trust"})
        graph.create_node("dilemma::trust::alt::yes", {"type": "answer", "raw_id": "yes"})
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "dilemmas": [{"dilemma_id": "dilemma::trust", "explored": ["yes"], "unexplored": []}],
            "paths": [
                {
                    "path_id": "path::mentor_arc",  # Scoped ID in definition
                    "name": "Mentor Arc",
                    "dilemma_id": "dilemma::trust",
                    "answer_id": "yes",
                }
            ],
            "consequences": [
                {
                    "consequence_id": "trust_earned",
                    "path_id": "path::mentor_arc",  # Scoped ID in reference
                    "description": "Trust is earned",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "resolution",
                    "summary": "Trust resolved",
                    "paths": ["path::mentor_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "commits", "note": "Locked in"}
                    ],
                },
                {
                    "beat_id": "resolution_post",
                    "summary": "Trust aftermath",
                    "paths": ["path::mentor_arc"],
                    "dilemma_impacts": [
                        {"dilemma_id": "dilemma::trust", "effect": "advances", "note": "Fallout"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []


class TestSortBySimilarity:
    """Tests for _sort_by_similarity helper function."""

    def test_sorts_by_similarity_descending(self) -> None:
        """Most similar ID appears first."""
        sorted_ids = _sort_by_similarity(
            "hollow_archive",
            ["echo_chamber", "the_hollow_archive", "keeper_of_depths"],
        )
        assert sorted_ids[0][0] == "the_hollow_archive"

    def test_returns_similarity_scores(self) -> None:
        """Each result includes a similarity score."""
        sorted_ids = _sort_by_similarity(
            "the_archive",
            ["the_hollow_archive", "keeper_of_depths"],
        )
        for sid, score in sorted_ids:
            assert isinstance(sid, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_handles_scoped_ids(self) -> None:
        """Strips scope prefixes before comparing."""
        sorted_ids = _sort_by_similarity(
            "the_archive",
            ["entity::the_hollow_archive", "entity::keeper_of_depths"],
        )
        # Should match on the raw ID part
        assert sorted_ids[0][0] == "entity::the_hollow_archive"

    def test_case_insensitive(self) -> None:
        """Comparison is case-insensitive."""
        sorted_ids = _sort_by_similarity(
            "THE_ARCHIVE",
            ["the_hollow_archive", "KEEPER_OF_DEPTHS"],
        )
        assert sorted_ids[0][0] == "the_hollow_archive"

    def test_empty_available_returns_empty(self) -> None:
        """Empty available list returns empty result."""
        sorted_ids = _sort_by_similarity("test", [])
        assert sorted_ids == []

    def test_exact_match_has_score_one(self) -> None:
        """Exact match has score of 1.0."""
        sorted_ids = _sort_by_similarity("hero", ["hero", "villain"])
        assert sorted_ids[0] == ("hero", 1.0)


class TestFormatAvailableWithSuggestions:
    """Tests for _format_available_with_suggestions helper function."""

    def test_high_confidence_prescriptive(self) -> None:
        """Score >= 0.85 gives single 'Use X instead' suggestion."""
        # "hollow_archive" vs "the_hollow_archive" = 87.5% similarity
        result = _format_available_with_suggestions(
            "hollow_archive",
            ["the_hollow_archive", "keeper_of_depths", "echo_chamber"],
        )
        assert "Use 'the_hollow_archive' instead" in result
        assert "Did you mean" not in result

    def test_medium_confidence_ranked(self) -> None:
        """Score 0.6-0.85 gives ranked list with percentages."""
        # "the_archive" vs "the_hollow_archive" = 75% similarity (medium)
        result = _format_available_with_suggestions(
            "the_archive",
            ["the_hollow_archive", "keeper_of_depths", "echo_chamber"],
        )
        assert "Did you mean one of these?" in result
        assert "%" in result
        # Most similar should be first
        assert "the_hollow_archive" in result

    def test_low_confidence_sorted_only(self) -> None:
        """Score < 0.6 gives sorted list without suggestions."""
        result = _format_available_with_suggestions(
            "xyz_unknown",
            ["the_hollow_archive", "keeper_of_depths", "echo_chamber"],
        )
        assert "most similar first" in result
        assert "Did you mean" not in result
        assert "Use '" not in result

    def test_empty_available_returns_empty(self) -> None:
        """Empty available list returns empty string."""
        result = _format_available_with_suggestions("test", [])
        assert result == ""

    def test_truncates_long_list(self) -> None:
        """Long available lists are truncated with ellipsis."""
        available = [f"item_{i}" for i in range(20)]
        result = _format_available_with_suggestions("xyz_unknown", available)
        # Should have truncation indicator
        assert "..." in result


class TestSimilarityFeedbackIntegration:
    """Integration tests for similarity-based feedback in error messages."""

    def test_seed_error_high_confidence(self) -> None:
        """SeedMutationError shows prescriptive suggestion for high confidence match."""
        # "hollow_archive" vs "the_hollow_archive" = 87.5% similarity
        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity 'hollow_archive' not in BRAINSTORM",
                available=["the_hollow_archive", "keeper_of_depths", "echo_chamber"],
                provided="hollow_archive",
            )
        ]
        error = SeedMutationError(errors)

        feedback = error.to_feedback()

        assert "Use 'the_hollow_archive' instead" in feedback

    def test_seed_error_medium_confidence(self) -> None:
        """SeedMutationError shows ranked list for medium confidence match."""
        # "the_archive" vs "the_hollow_archive" = 75% (medium confidence)
        errors = [
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma 'the_archive' not in BRAINSTORM",
                available=["the_hollow_archive", "archive_keeper", "echo_chamber"],
                provided="the_archive",
            )
        ]
        error = SeedMutationError(errors)

        feedback = error.to_feedback()

        assert "Did you mean one of these?" in feedback
        assert "the_hollow_archive" in feedback

    def test_seed_error_low_confidence(self) -> None:
        """SeedMutationError shows sorted list for low confidence."""
        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity 'xyz_unknown' not in BRAINSTORM",
                available=["the_hollow_archive", "keeper_of_depths", "echo_chamber"],
                provided="xyz_unknown",
            )
        ]
        error = SeedMutationError(errors)

        feedback = error.to_feedback()

        assert "most similar first" in feedback

    def test_brainstorm_error_high_confidence(self) -> None:
        """BrainstormMutationError shows prescriptive suggestion for high confidence match."""
        # "hollow_archive" vs "the_hollow_archive" = 87.5% similarity
        errors = [
            BrainstormValidationError(
                field_path="dilemmas.0.central_entity_ids",
                issue="Entity 'hollow_archive' not in entities list",
                available=["the_hollow_archive", "keeper_of_depths", "echo_chamber"],
                provided="hollow_archive",
            )
        ]
        error = BrainstormMutationError(errors)

        feedback = error.to_feedback()

        assert "Use 'the_hollow_archive' instead" in feedback

    def test_error_without_provided_uses_fallback(self) -> None:
        """Errors without provided value use fallback formatting."""
        errors = [
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for character 'hero'",
                available=["hero", "villain"],
                provided="",  # No provided value
            )
        ]
        error = SeedMutationError(errors)

        feedback = error.to_feedback()

        # Should show available list in fallback comma-separated format
        assert "Available: hero, villain" in feedback

    def test_seq3_scenario(self) -> None:
        """Real failure case from seq-3: 'the_archive' instead of 'the_hollow_archive'.

        Key improvement: Even though similarity (75%) is below high confidence threshold,
        the correct ID now appears FIRST in the suggestions, making recovery much easier.
        Previously this ID might have been truncated away in an arbitrary list.
        """
        errors = [
            SeedValidationError(
                field_path="entity_id",
                issue="not in BRAINSTORM entities",
                available=[
                    "the_hollow_archive",
                    "keeper_of_depths",
                    "fractured_lattice",
                    "chamber_of_stillness",
                    "garden_of_questions",
                    "weaver_of_echoes",
                ],
                provided="the_archive",
            )
        ]
        error = SeedMutationError(errors)

        feedback = error.to_feedback()

        # The correct ID should appear FIRST in suggestions (sorted by similarity)
        assert "the_hollow_archive" in feedback
        # Should show ranked suggestions with the correct ID prominently displayed
        assert "Did you mean one of these?" in feedback
        # Verify similarity score is shown
        assert "%" in feedback


class TestFormatSemanticErrorsAsContent:
    """Tests for format_semantic_errors_as_content function."""

    def test_returns_empty_string_for_no_errors(self) -> None:
        """Should return empty string when no errors provided."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        result = format_semantic_errors_as_content([])
        assert result == ""

    def test_formats_completeness_errors(self) -> None:
        """Should format completeness errors in Missing items section."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for entity 'hollow_key'",
                available=[],
                provided="",
            ),
            SeedValidationError(
                field_path="dilemmas",
                issue="Missing decision for dilemma 'ancient_scroll'",
                available=[],
                provided="",
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Missing items" in result
        assert "hollow_key" in result
        assert "ancient_scroll" in result

    def test_formats_path_completeness_errors(self) -> None:
        """Should format path completeness errors in Missing paths section."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="paths",
                issue="Dilemma 'loyalty' has no path. Create at least one path exploring this dilemma.",
                available=[],
                provided="",
                category=SeedErrorCategory.COMPLETENESS,
            ),
            SeedValidationError(
                field_path="paths",
                issue="Dilemma 'trust' has no path. Create at least one path exploring this dilemma.",
                available=[],
                provided="",
                category=SeedErrorCategory.COMPLETENESS,
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Missing paths" in result
        assert "loyalty" in result
        assert "trust" in result
        # Should NOT appear in decision-missing section
        assert "Missing items" not in result

    def test_formats_semantic_errors_with_suggestions(self) -> None:
        """Should format semantic errors with similarity-based suggestions."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity 'ghost' not in BRAINSTORM",
                available=["guest", "ghost_hunter", "host"],
                provided="ghost",
            )
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Invalid references" in result
        assert "entities.0.entity_id: 'ghost' is not valid" in result

    def test_formats_mixed_error_categories(self) -> None:
        """Should handle multiple error categories in one output."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            # Completeness error
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for character 'hero'",
                available=[],
                provided="",
            ),
            # Semantic error
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma 'unknown' not in BRAINSTORM",
                available=["main_dilemma"],
                provided="unknown",
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Missing items" in result
        assert "Invalid references" in result
        assert "Please reconsider the summary" in result

    def test_includes_closing_guidance(self) -> None:
        """Should include guidance about using BRAINSTORM IDs."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for entity 'test'",
                available=[],
                provided="",
            )
        ]

        result = format_semantic_errors_as_content(errors)

        assert "ensuring you only reference" in result
        assert "BRAINSTORM" in result

    def test_formats_cross_reference_errors(self) -> None:
        """Should format CROSS_REFERENCE errors in Bucket misplacement section."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="dilemmas",
                issue=(
                    "Dilemma 'loyalty': default answer 'trusts_council' is in "
                    "unexplored but MUST be in explored. Move it from unexplored "
                    "to explored."
                ),
                available=["betrays_council"],
                provided="trusts_council",
                category=SeedErrorCategory.CROSS_REFERENCE,
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Bucket misplacement" in result
        assert "trusts_council" in result
        assert "Move it from unexplored" in result

    def test_formats_mixed_with_cross_reference(self) -> None:
        """Should handle CROSS_REFERENCE alongside other error categories."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for entity 'hero'",
                available=[],
                provided="",
            ),
            SeedValidationError(
                field_path="dilemmas",
                issue="Default answer 'X' is in unexplored but MUST be in explored.",
                available=[],
                provided="X",
                category=SeedErrorCategory.CROSS_REFERENCE,
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Missing items" in result
        assert "Bucket misplacement" in result

    def test_formats_warning_errors_as_scaffold_notes(self) -> None:
        """Should format WARNING-category errors as non-blocking scaffold notes."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="beats.path_1",
                issue="No advances/reveals beat before commit",
                category=SeedErrorCategory.WARNING,
            ),
            SeedValidationError(
                field_path="beats.path_2",
                issue="No beat after commit",
                category=SeedErrorCategory.WARNING,
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Scaffold notes" in result
        assert "non-blocking" in result
        assert "No advances/reveals beat before commit" in result
        assert "No beat after commit" in result


class TestTypeAwareFeedback:
    """Tests for type-aware cross-type error messages in validate_seed_mutations.

    When an ID is used as the wrong type (e.g., entity used as dilemma_id),
    the error message should indicate what type it actually is, rather than
    the generic "not in BRAINSTORM" message.
    """

    def test_type_aware_feedback_entity_as_dilemma(self) -> None:
        """Entity faction name used as dilemma_id gives helpful message."""
        graph = Graph.empty()
        # isolation_protocol is a faction entity in brainstorm
        graph.create_node(
            "entity::isolation_protocol",
            {"type": "entity", "raw_id": "isolation_protocol", "entity_type": "faction"},
        )
        graph.create_node(
            "dilemma::trust_or_betray",
            {"type": "dilemma", "raw_id": "trust_or_betray"},
        )

        output = {
            "entities": [
                {"entity_id": "isolation_protocol", "disposition": "retained"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust_or_betray", "explored": [], "unexplored": []},
            ],
            "paths": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "dilemma_impacts": [
                        # Using entity name as dilemma_id (the seq-9 bug)
                        {"dilemma_id": "isolation_protocol", "effect": "advances"}
                    ],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # Should get a type-aware error, not generic "not in BRAINSTORM"
        dilemma_impact_errors = [e for e in errors if "dilemma_impacts" in e.field_path]
        assert len(dilemma_impact_errors) == 1
        assert "is an entity (faction), not a dilemma" in dilemma_impact_errors[0].issue
        assert "subject_X_or_Y" in dilemma_impact_errors[0].issue

    def test_type_aware_feedback_path_as_dilemma(self) -> None:
        """Path ID used as dilemma_id gives helpful message."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "dilemma::trust_or_betray",
            {"type": "dilemma", "raw_id": "trust_or_betray"},
        )
        graph.create_node(
            "dilemma::trust_or_betray::alt::trust",
            {"type": "answer", "raw_id": "trust"},
        )
        graph.add_edge(
            "has_answer",
            "dilemma::trust_or_betray",
            "dilemma::trust_or_betray::alt::trust",
        )

        output = {
            "entities": [{"entity_id": "hero", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "trust_or_betray", "explored": ["trust"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "dilemma_id": "trust_or_betray",
                    "answer_id": "trust",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "dilemma_impacts": [
                        # Using path ID as dilemma_id
                        {"dilemma_id": "mentor_arc", "effect": "advances"}
                    ],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        dilemma_impact_errors = [e for e in errors if "dilemma_impacts" in e.field_path]
        assert len(dilemma_impact_errors) == 1
        assert "is a path ID, not a dilemma" in dilemma_impact_errors[0].issue

    def test_type_aware_feedback_dilemma_as_entity(self) -> None:
        """Dilemma ID used as entity in beat gives helpful message."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "dilemma::trust_or_betray",
            {"type": "dilemma", "raw_id": "trust_or_betray"},
        )

        output = {
            "entities": [{"entity_id": "hero", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "trust_or_betray", "explored": [], "unexplored": []},
            ],
            "paths": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    # Using dilemma ID as entity reference
                    "entities": ["trust_or_betray"],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        entity_errors = [e for e in errors if "initial_beats.0.entities" in e.field_path]
        assert len(entity_errors) == 1
        assert "is a dilemma ID, not an entity" in entity_errors[0].issue


class TestCutEntityInBeats:
    """Tests for cut-entity-in-beats validation.

    Beats should not reference entities that have disposition 'cut'.
    This catches the seq-10 scenario where cut entities are still
    referenced in beats.
    """

    def test_cut_entity_in_beat_entities_detected(self) -> None:
        """Cut entity in beat entities[] raises error."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::chrono_lock",
            {"type": "entity", "raw_id": "chrono_lock", "entity_type": "object"},
        )

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                {"entity_id": "chrono_lock", "disposition": "cut"},
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "entities": ["hero", "chrono_lock"],  # chrono_lock is cut!
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        cut_errors = [e for e in errors if "disposition 'cut'" in e.issue]
        assert len(cut_errors) == 1
        assert "chrono_lock" in cut_errors[0].issue
        assert "initial_beats.0.entities" in cut_errors[0].field_path

    def test_cut_entity_in_beat_location_detected(self) -> None:
        """Cut entity as beat location raises error."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::watchers_guild",
            {"type": "entity", "raw_id": "watchers_guild", "entity_type": "location"},
        )

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                {"entity_id": "watchers_guild", "disposition": "cut"},
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "entities": ["hero"],
                    "location": "watchers_guild",  # watchers_guild is cut!
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        cut_errors = [e for e in errors if "disposition 'cut'" in e.issue]
        assert len(cut_errors) == 1
        assert "watchers_guild" in cut_errors[0].issue
        assert "initial_beats.0.location" in cut_errors[0].field_path

    def test_retained_entity_in_beat_no_error(self) -> None:
        """Retained entity in beat passes fine."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::tavern",
            {"type": "entity", "raw_id": "tavern", "entity_type": "location"},
        )

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                {"entity_id": "tavern", "disposition": "retained"},
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "entities": ["hero"],
                    "location": "tavern",
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        cut_errors = [e for e in errors if "disposition 'cut'" in e.issue]
        assert cut_errors == []

    def test_cut_entity_in_location_alternatives_detected(self) -> None:
        """Cut entity in beat location_alternatives raises error."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::tavern",
            {"type": "entity", "raw_id": "tavern", "entity_type": "location"},
        )
        graph.create_node(
            "entity::ruins",
            {"type": "entity", "raw_id": "ruins", "entity_type": "location"},
        )

        output = {
            "entities": [
                {"entity_id": "hero", "disposition": "retained"},
                {"entity_id": "tavern", "disposition": "retained"},
                {"entity_id": "ruins", "disposition": "cut"},
            ],
            "dilemmas": [],
            "paths": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "entities": ["hero"],
                    "location": "tavern",
                    "location_alternatives": ["ruins"],  # ruins is cut!
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        cut_errors = [e for e in errors if "disposition 'cut'" in e.issue]
        assert len(cut_errors) == 1
        assert "ruins" in cut_errors[0].issue
        assert "initial_beats.0.location_alternatives" in cut_errors[0].field_path


class TestBackfillExploredFromPaths:
    """Tests for _backfill_explored_from_paths migration function."""

    def test_backfills_empty_explored_from_paths(self) -> None:
        """Empty explored array is filled from path alternative_ids."""
        output = {
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "explored": []},
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_a",
                },
                {
                    "path_id": "path2",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_b",
                },
            ],
        }

        _backfill_explored_from_paths(output)

        assert output["dilemmas"][0]["explored"] == ["option_a", "option_b"]

    def test_preserves_existing_explored(self) -> None:
        """Non-empty explored array is not modified."""
        output = {
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "explored": ["existing_value"]},
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_a",
                },
            ],
        }

        _backfill_explored_from_paths(output)

        assert output["dilemmas"][0]["explored"] == ["existing_value"]

    def test_handles_scoped_dilemma_ids(self) -> None:
        """Handles dilemma IDs with scope prefix (dilemma::id)."""
        output = {
            "dilemmas": [
                {"dilemma_id": "dilemma::choice_a_or_b", "explored": []},
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "dilemma::choice_a_or_b",
                    "answer_id": "option_a",
                },
            ],
        }

        _backfill_explored_from_paths(output)

        assert output["dilemmas"][0]["explored"] == ["option_a"]

    def test_handles_mixed_scoped_and_unscoped(self) -> None:
        """Matches dilemma with scoped ID to path with unscoped ID."""
        output = {
            "dilemmas": [
                {"dilemma_id": "dilemma::choice_a_or_b", "explored": []},
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_a",
                },
            ],
        }

        _backfill_explored_from_paths(output)

        assert output["dilemmas"][0]["explored"] == ["option_a"]

    def test_backfills_when_only_old_considered_key_present(self) -> None:
        """Old 'considered' key is ignored; backfill uses 'explored' only.

        Pydantic migration handles considered→explored at the model layer.
        The backfill function only checks 'explored'.
        """
        output = {
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "considered": ["existing"]},
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_a",
                },
            ],
        }

        _backfill_explored_from_paths(output)

        # 'explored' key absent means backfill triggers (old 'considered' ignored)
        assert output["dilemmas"][0]["explored"] == ["option_a"]

    def test_no_paths_no_backfill(self) -> None:
        """Empty paths list does not modify dilemmas."""
        output = {
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "explored": []},
            ],
            "paths": [],
        }

        _backfill_explored_from_paths(output)

        assert output["dilemmas"][0]["explored"] == []

    def test_path_without_alternative_id_ignored(self) -> None:
        """Paths without answer_id are skipped."""
        output = {
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "explored": []},
            ],
            "paths": [
                {"path_id": "path1", "dilemma_id": "choice_a_or_b"},  # no answer_id
            ],
        }

        _backfill_explored_from_paths(output)

        assert output["dilemmas"][0]["explored"] == []

    def test_multiple_dilemmas_independently_backfilled(self) -> None:
        """Each dilemma is backfilled from its own paths."""
        output = {
            "dilemmas": [
                {"dilemma_id": "dilemma_one", "explored": []},
                {"dilemma_id": "dilemma_two", "explored": []},
            ],
            "paths": [
                {"path_id": "t1", "dilemma_id": "dilemma_one", "answer_id": "opt_a"},
                {"path_id": "t2", "dilemma_id": "dilemma_two", "answer_id": "opt_x"},
                {"path_id": "t3", "dilemma_id": "dilemma_two", "answer_id": "opt_y"},
            ],
        }

        _backfill_explored_from_paths(output)

        assert output["dilemmas"][0]["explored"] == ["opt_a"]
        assert output["dilemmas"][1]["explored"] == ["opt_x", "opt_y"]


class TestValidation11cPathAlternativeInExplored:
    """Tests for validation check 11c: path.answer_id IN dilemma.explored."""

    def test_path_alternative_in_explored_passes(self) -> None:
        """Path with answer_id matching explored passes validation."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::choice_a_or_b",
            {
                "type": "dilemma",
                "raw_id": "choice_a_or_b",
                "answers": [
                    {"answer_id": "option_a", "is_canonical": True},
                    {"answer_id": "option_b", "is_canonical": False},
                ],
            },
        )

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "explored": ["option_a", "option_b"]},
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_a",
                    "name": "Path One",
                },
                {
                    "path_id": "path2",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_b",
                    "name": "Path Two",
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "b1",
                    "summary": "Test",
                    "paths": ["path1"],
                    "dilemma_impacts": [
                        {"dilemma_id": "choice_a_or_b", "effect": "commits", "note": "Commits"}
                    ],
                },
                {
                    "beat_id": "b2",
                    "summary": "Test",
                    "paths": ["path2"],
                    "dilemma_impacts": [
                        {"dilemma_id": "choice_a_or_b", "effect": "commits", "note": "Commits"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # Filter for 11c errors specifically
        check_11c_errors = [
            e for e in errors if "is not in dilemma" in e.issue and "explored list" in e.issue
        ]
        assert check_11c_errors == []

    def test_path_alternative_not_in_explored_detected(self) -> None:
        """Path with answer_id NOT in explored fails validation."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::choice_a_or_b",
            {
                "type": "dilemma",
                "raw_id": "choice_a_or_b",
                "answers": [
                    {"answer_id": "option_a", "is_canonical": True},
                    {"answer_id": "option_b", "is_canonical": False},
                ],
            },
        )

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "explored": ["option_a"]},  # missing option_b!
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_a",
                    "name": "Path One",
                },
                {
                    "path_id": "path2",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_b",  # not in explored!
                    "name": "Path Two",
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "b1",
                    "summary": "Test",
                    "paths": ["path1"],
                    "dilemma_impacts": [
                        {"dilemma_id": "choice_a_or_b", "effect": "commits", "note": "Commits"}
                    ],
                },
                {
                    "beat_id": "b2",
                    "summary": "Test",
                    "paths": ["path2"],
                    "dilemma_impacts": [
                        {"dilemma_id": "choice_a_or_b", "effect": "commits", "note": "Commits"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # Filter for 11c errors specifically
        check_11c_errors = [
            e for e in errors if "is not in dilemma" in e.issue and "explored list" in e.issue
        ]
        assert len(check_11c_errors) == 1
        assert "option_b" in check_11c_errors[0].issue
        assert "choice_a_or_b" in check_11c_errors[0].issue
        assert check_11c_errors[0].field_path == "paths.1.answer_id"
        assert check_11c_errors[0].category == SeedErrorCategory.CROSS_REFERENCE

    def test_empty_explored_detected(self) -> None:
        """Path with answer_id but empty explored fails validation."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::choice_a_or_b",
            {
                "type": "dilemma",
                "raw_id": "choice_a_or_b",
                "answers": [
                    {"answer_id": "option_a", "is_canonical": True},
                ],
            },
        )

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "choice_a_or_b", "explored": []},  # empty!
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",
                    "answer_id": "option_a",
                    "name": "Path One",
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "b1",
                    "summary": "Test",
                    "paths": ["path1"],
                    "dilemma_impacts": [
                        {"dilemma_id": "choice_a_or_b", "effect": "commits", "note": "Commits"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # Filter for 11c errors
        check_11c_errors = [
            e for e in errors if "is not in dilemma" in e.issue and "explored list" in e.issue
        ]
        assert len(check_11c_errors) == 1
        assert "option_a" in check_11c_errors[0].issue

    def test_scoped_dilemma_ids_matched_correctly(self) -> None:
        """Scoped dilemma IDs are normalized for matching."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::choice_a_or_b",
            {
                "type": "dilemma",
                "raw_id": "choice_a_or_b",
                "answers": [
                    {"answer_id": "option_a", "is_canonical": True},
                ],
            },
        )

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "dilemma::choice_a_or_b", "explored": ["option_a"]},  # scoped
            ],
            "paths": [
                {
                    "path_id": "path1",
                    "dilemma_id": "choice_a_or_b",  # unscoped
                    "answer_id": "option_a",
                    "name": "Path One",
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "b1",
                    "summary": "Test",
                    "paths": ["path1"],
                    "dilemma_impacts": [
                        {"dilemma_id": "choice_a_or_b", "effect": "commits", "note": "Commits"}
                    ],
                },
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # Filter for 11c errors
        check_11c_errors = [
            e for e in errors if "is not in dilemma" in e.issue and "explored list" in e.issue
        ]
        assert check_11c_errors == []


class TestBackfillIntegrationWithApplySeedMutations:
    """Test that backfill runs before validation in apply_seed_mutations."""

    def test_backfill_runs_before_validation(self) -> None:
        """Backfill fixes data before validation, preventing 11c errors.

        Uses _create_compliant_brainstorm_graph() to satisfy the upstream contract.
        The trust dilemma (protector/manipulator) stands in for the legacy
        choice_a_or_b pattern.  Empty explored=[] gets backfilled from paths before
        validation runs, so no 11c error is raised.
        """
        graph = _create_compliant_brainstorm_graph()

        # Legacy data pattern: paths exist but explored is empty
        output = {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "explored": [],  # empty — backfill should fill from paths
                },
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "name": "Protector Arc",
                    "description": "Mentor protects.",
                    "path_importance": "major",
                },
                {
                    "path_id": "trust__manipulator",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "name": "Manipulator Arc",
                    "description": "Mentor manipulates.",
                    "path_importance": "major",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_protector",
                    "path_id": "trust__protector",
                    "description": "Protagonist gains an ally.",
                    "narrative_effects": ["Trust shapes the climax"],
                },
                {
                    "consequence_id": "c_manipulator",
                    "path_id": "trust__manipulator",
                    "description": "Protagonist faces danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
            ],
            "initial_beats": [
                # Shared pre-commit beat (dual belongs_to)
                {
                    "beat_id": "trust_pre",
                    "summary": "Protagonist encounters the mentor",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # Commit beats
                {
                    "beat_id": "b1",
                    "summary": "Kay trusts the mentor.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2",
                    "summary": "Kay distrusts the mentor.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                # Post-commit beats
                {
                    "beat_id": "b1_post",
                    "summary": "Protector path aftermath",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b1_post2",
                    "summary": "Protector path continues",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2_post",
                    "summary": "Manipulator path aftermath",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2_post2",
                    "summary": "Manipulator path continues",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
            ],
            "human_approved_paths": True,
        }

        # Should NOT raise because backfill fixes the data before validation
        apply_seed_mutations(graph, output)

        # Verify the dilemma node was updated with backfilled explored
        dilemma_node = graph.get_node("dilemma::trust")
        assert dilemma_node is not None
        assert sorted(dilemma_node.get("explored", [])) == ["manipulator", "protector"]


class TestValidateSeedPovCharacter:
    """Test POV character validation in SEED mutations."""

    @pytest.fixture
    def graph_with_entities(self) -> Graph:
        """Graph with BRAINSTORM entities for testing."""
        graph = Graph.empty()
        # Add entity nodes
        graph.create_node(
            "character::kay",
            {"type": "entity", "raw_id": "kay", "entity_type": "character", "concept": "Archivist"},
        )
        graph.create_node(
            "character::mentor",
            {"type": "entity", "raw_id": "mentor", "entity_type": "character", "concept": "Mentor"},
        )
        graph.create_node(
            "location::manor",
            {"type": "entity", "raw_id": "manor", "entity_type": "location", "concept": "Manor"},
        )
        # Add dilemma and answers
        graph.create_node(
            "dilemma::trust_mentor_or_not",
            {
                "type": "dilemma",
                "raw_id": "trust_mentor_or_not",
                "question": "Trust the mentor?",
            },
        )
        graph.create_node(
            "dilemma::trust_mentor_or_not::alt::trust",
            {"type": "answer", "raw_id": "trust", "is_canonical": True},
        )
        graph.create_node(
            "dilemma::trust_mentor_or_not::alt::distrust",
            {"type": "answer", "raw_id": "distrust", "is_canonical": False},
        )
        graph.add_edge(
            "has_answer",
            "dilemma::trust_mentor_or_not",
            "dilemma::trust_mentor_or_not::alt::trust",
        )
        graph.add_edge(
            "has_answer",
            "dilemma::trust_mentor_or_not",
            "dilemma::trust_mentor_or_not::alt::distrust",
        )
        return graph

    def test_valid_pov_character_accepted(self, graph_with_entities: Graph) -> None:
        """Valid pov_character reference is accepted."""
        output = {
            "entities": [{"entity_id": "kay", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "trust_mentor_or_not", "explored": ["trust"], "unexplored": []}
            ],
            "paths": [
                {
                    "path_id": "path::trust_mentor_or_not__trust",
                    "name": "Trust Path",
                    "dilemma_id": "trust_mentor_or_not",
                    "answer_id": "trust",
                    "path_importance": "major",
                    "description": "Trust the mentor",
                    "pov_character": "kay",  # Valid entity reference
                }
            ],
            "consequences": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph_with_entities, output)

        pov_errors = [e for e in errors if "pov_character" in e.field_path]
        assert pov_errors == []

    def test_invalid_pov_character_detected(self, graph_with_entities: Graph) -> None:
        """Invalid pov_character reference raises error."""
        output = {
            "entities": [{"entity_id": "kay", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "trust_mentor_or_not", "explored": ["trust"], "unexplored": []}
            ],
            "paths": [
                {
                    "path_id": "path::trust_mentor_or_not__trust",
                    "name": "Trust Path",
                    "dilemma_id": "trust_mentor_or_not",
                    "answer_id": "trust",
                    "path_importance": "major",
                    "description": "Trust the mentor",
                    "pov_character": "nonexistent_entity",  # Invalid
                }
            ],
            "consequences": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph_with_entities, output)

        pov_errors = [e for e in errors if "pov_character" in e.field_path]
        assert len(pov_errors) == 1
        assert "not found" in pov_errors[0].issue
        assert "kay" in pov_errors[0].available  # Should suggest valid entities

    def test_pov_character_none_valid(self, graph_with_entities: Graph) -> None:
        """Path without pov_character (None) is valid."""
        output = {
            "entities": [{"entity_id": "kay", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "trust_mentor_or_not", "explored": ["trust"], "unexplored": []}
            ],
            "paths": [
                {
                    "path_id": "path::trust_mentor_or_not__trust",
                    "name": "Trust Path",
                    "dilemma_id": "trust_mentor_or_not",
                    "answer_id": "trust",
                    "path_importance": "major",
                    "description": "Trust the mentor",
                    # No pov_character field
                }
            ],
            "consequences": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph_with_entities, output)

        pov_errors = [e for e in errors if "pov_character" in e.field_path]
        assert pov_errors == []

    def test_pov_character_non_character_rejected(self, graph_with_entities: Graph) -> None:
        """POV character must be a character entity, not location/object/faction."""
        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "manor", "disposition": "retained"},
            ],
            "dilemmas": [
                {"dilemma_id": "trust_mentor_or_not", "explored": ["trust"], "unexplored": []}
            ],
            "paths": [
                {
                    "path_id": "path::trust_mentor_or_not__trust",
                    "name": "Trust Path",
                    "dilemma_id": "trust_mentor_or_not",
                    "answer_id": "trust",
                    "path_importance": "major",
                    "description": "Trust the mentor",
                    "pov_character": "manor",  # Location, not character
                }
            ],
            "consequences": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph_with_entities, output)

        pov_errors = [e for e in errors if "pov_character" in e.field_path]
        assert len(pov_errors) == 1
        assert "character entity" in pov_errors[0].issue
        assert "location" in pov_errors[0].issue
        # Should suggest character entities
        assert "kay" in pov_errors[0].available
        assert "mentor" in pov_errors[0].available
        assert "manor" not in pov_errors[0].available


class TestApplySeedConvergenceAnalysis:
    """Test convergence analysis mutations (sections 7+8) in apply_seed_mutations."""

    def _graph_with_dilemmas(self) -> Graph:
        """Build a fully BRAINSTORM-compliant graph with two extra dilemmas.

        Uses _create_compliant_brainstorm_graph() as the base (providing vision,
        mentor/archive/tower entities, and the trust dilemma).  Adds two extra
        dilemmas (trust_or_not and stay_or_go) that the convergence-analysis
        tests exercise.  The base trust dilemma is also explored in _base_output
        to satisfy the completeness validator.
        """
        return _create_compliant_brainstorm_graph(
            extra_dilemmas=[
                (
                    "trust_or_not",
                    "Trust the mentor or not?",
                    [
                        ("trust", "Protagonist trusts the mentor.", True),
                        ("distrust", "Protagonist distrusts the mentor.", False),
                    ],
                    "character::mentor",
                ),
                (
                    "stay_or_go",
                    "Stay at the archive or leave?",
                    [
                        ("stay", "Protagonist stays at the archive.", True),
                        ("go", "Protagonist leaves for unknown territory.", False),
                    ],
                    "location::archive",
                ),
            ]
        )

    def _base_output(self) -> dict:
        """Fully contract-compliant SEED output exploring three dilemmas.

        Covers all graph dilemmas (trust, trust_or_not, stay_or_go).
        Consequences have ripples.  Each path has exactly one commit beat and
        two post-commit beats.  All beats have non-empty entities.
        dilemma_analyses covers all three dilemmas.
        """
        return {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "explored": ["protector", "manipulator"],
                    "unexplored": [],
                },
                {
                    "dilemma_id": "trust_or_not",
                    "explored": ["trust", "distrust"],
                    "unexplored": [],
                },
                {
                    "dilemma_id": "stay_or_go",
                    "explored": ["stay", "go"],
                    "unexplored": [],
                },
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "name": "Protector Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "path_importance": "major",
                    "description": "Mentor protects protagonist.",
                },
                {
                    "path_id": "trust__manipulator",
                    "name": "Manipulator Arc",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "path_importance": "major",
                    "description": "Mentor manipulates protagonist.",
                },
                {
                    "path_id": "trust_or_not__trust",
                    "name": "Trust Path",
                    "dilemma_id": "trust_or_not",
                    "answer_id": "trust",
                    "path_importance": "major",
                    "description": "Trust the mentor",
                },
                {
                    "path_id": "trust_or_not__distrust",
                    "name": "Distrust Path",
                    "dilemma_id": "trust_or_not",
                    "answer_id": "distrust",
                    "path_importance": "major",
                    "description": "Distrust the mentor",
                },
                {
                    "path_id": "stay_or_go__stay",
                    "name": "Stay Path",
                    "dilemma_id": "stay_or_go",
                    "answer_id": "stay",
                    "path_importance": "major",
                    "description": "Stay at the location",
                },
                {
                    "path_id": "stay_or_go__go",
                    "name": "Go Path",
                    "dilemma_id": "stay_or_go",
                    "answer_id": "go",
                    "path_importance": "minor",
                    "description": "Leave for unknown territory",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_trust_protector",
                    "path_id": "trust__protector",
                    "description": "Protagonist gains a loyal ally.",
                    "narrative_effects": ["Trust shapes the climax"],
                },
                {
                    "consequence_id": "c_trust_manipulator",
                    "path_id": "trust__manipulator",
                    "description": "Protagonist faces danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
                {
                    "consequence_id": "c_trust_or_not_trust",
                    "path_id": "trust_or_not__trust",
                    "description": "Protagonist gains an ally.",
                    "narrative_effects": ["Alliance unlocks new paths"],
                },
                {
                    "consequence_id": "c_trust_or_not_distrust",
                    "path_id": "trust_or_not__distrust",
                    "description": "Protagonist acts alone.",
                    "narrative_effects": ["Solo path limits options"],
                },
                {
                    "consequence_id": "c_stay",
                    "path_id": "stay_or_go__stay",
                    "description": "Safety of the archive.",
                    "narrative_effects": ["Security enables research"],
                },
                {
                    "consequence_id": "c_go",
                    "path_id": "stay_or_go__go",
                    "description": "Unknown territory awaits.",
                    "narrative_effects": ["Risk brings discovery"],
                },
            ],
            "initial_beats": [
                # trust dilemma — shared pre-commit + per-path beats
                {
                    "beat_id": "trust_pre",
                    "summary": "Mentor encounter",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_commit_protector",
                    "summary": "Kay trusts the mentor.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_post_p1",
                    "summary": "Mentor reveals vital secret.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_post_p2",
                    "summary": "Ally bond confirmed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_commit_manipulator",
                    "summary": "Kay distrusts the mentor.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_post_m1",
                    "summary": "Mentor's true motive surfaces.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_post_m2",
                    "summary": "Protagonist faces danger alone.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # trust_or_not dilemma — shared pre-commit + per-path beats
                {
                    "beat_id": "ton_pre",
                    "summary": "Trust question arises.",
                    "path_id": "trust_or_not__trust",
                    "also_belongs_to": "trust_or_not__distrust",
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_trust",
                    "summary": "Commit trust",
                    "path_id": "trust_or_not__trust",
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_trust_post",
                    "summary": "Trust aftermath",
                    "path_id": "trust_or_not__trust",
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_trust_post2",
                    "summary": "Trust confirmed.",
                    "path_id": "trust_or_not__trust",
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_distrust",
                    "summary": "Commit distrust",
                    "path_id": "trust_or_not__distrust",
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_distrust_post",
                    "summary": "Distrust aftermath",
                    "path_id": "trust_or_not__distrust",
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_distrust_post2",
                    "summary": "Distrust confirmed.",
                    "path_id": "trust_or_not__distrust",
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # stay_or_go dilemma — shared pre-commit + per-path beats
                {
                    "beat_id": "sog_pre",
                    "summary": "The choice to stay or leave.",
                    "path_id": "stay_or_go__stay",
                    "also_belongs_to": "stay_or_go__go",
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_stay",
                    "summary": "Commit stay",
                    "path_id": "stay_or_go__stay",
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_stay_post",
                    "summary": "Stay aftermath",
                    "path_id": "stay_or_go__stay",
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_stay_post2",
                    "summary": "Safety confirmed.",
                    "path_id": "stay_or_go__stay",
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_go",
                    "summary": "Commit go",
                    "path_id": "stay_or_go__go",
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_go_post",
                    "summary": "Go aftermath",
                    "path_id": "stay_or_go__go",
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b_go_post2",
                    "summary": "Discovery confirmed.",
                    "path_id": "stay_or_go__go",
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
                {
                    "dilemma_id": "trust_or_not",
                    "dilemma_role": "hard",
                    "payoff_budget": 4,
                    "reasoning": "Core trust choice.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
                {
                    "dilemma_id": "stay_or_go",
                    "dilemma_role": "soft",
                    "payoff_budget": 2,
                    "reasoning": "Spatial choice.",
                    "ending_salience": "low",
                    "residue_weight": "light",
                },
            ],
            "human_approved_paths": True,
        }

    def test_dilemma_role_stored_on_dilemma_node(self) -> None:
        """Convergence policy from analysis is stored on the dilemma graph node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "Mutually exclusive outcomes",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        assert node["dilemma_role"] == "hard"

    def test_payoff_budget_stored_on_dilemma_node(self) -> None:
        """Payoff budget from analysis is stored on the dilemma graph node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "stay_or_go",
                "dilemma_role": "soft",
                "payoff_budget": 3,
                "reasoning": "Shared beats after divergence",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::stay_or_go")
        assert node["payoff_budget"] == 3

    def test_unanalyzed_dilemma_gets_defaults(self) -> None:
        """Dilemma not in analyses gets default soft/2."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        # Only analyze trust_or_not; stay_or_go is unanalyzed
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "test",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::stay_or_go")
        assert node["dilemma_role"] == "soft"
        assert node["payoff_budget"] == 2

    def test_convergence_point_stored_on_dilemma_node(self) -> None:
        """convergence_point from analysis is stored on the dilemma graph node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "stay_or_go",
                "dilemma_role": "soft",
                "payoff_budget": 3,
                "reasoning": "Paths diverge then merge at the river crossing",
                "convergence_point": "The river crossing camp",
                "residue_note": "Trust levels differ based on choice",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::stay_or_go")
        assert node["convergence_point"] == "The river crossing camp"
        assert node["residue_note"] == "Trust levels differ based on choice"

    def test_null_convergence_fields_stored_explicitly(self) -> None:
        """Explicit null convergence_point/residue_note are stored on node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "Mutually exclusive world states",
                "convergence_point": None,
                "residue_note": None,
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        assert node.get("convergence_point") is None
        assert node.get("residue_note") is None

    def test_absent_convergence_fields_not_stored(self) -> None:
        """When convergence_point/residue_note are absent, they are not added."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "stay_or_go",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "reasoning": "test",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::stay_or_go")
        assert "convergence_point" not in node
        assert "residue_note" not in node

    def test_dilemma_ordering_edge_created(self) -> None:
        """Dilemma ordering relationship creates an edge between dilemma nodes."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_relationships"] = [
            {
                "dilemma_a": "stay_or_go",
                "dilemma_b": "trust_or_not",
                "ordering": "wraps",
                "description": "Stay/go wraps trust subplot",
                "reasoning": "test",
            },
        ]
        apply_seed_mutations(graph, output)

        # Edge type IS the ordering value (semantic edge, not generic dilemma_ordering)
        edges = graph.get_edges(
            from_id="dilemma::stay_or_go",
            edge_type="wraps",
        )
        assert len(edges) == 1
        assert edges[0]["to"] == "dilemma::trust_or_not"
        # ordering is encoded in the edge type, not stored as a separate attribute
        assert "ordering" not in edges[0]
        assert edges[0]["description"] == "Stay/go wraps trust subplot"

    def test_ordering_edge_skipped_if_node_missing(self) -> None:
        """Ordering edge is not created when a dilemma node is missing."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_relationships"] = [
            {
                "dilemma_a": "trust_or_not",
                "dilemma_b": "nonexistent_dilemma",
                "ordering": "serial",
                "description": "Should be skipped",
            },
        ]
        apply_seed_mutations(graph, output)

        edges = graph.get_edges(
            from_id="dilemma::trust_or_not",
            edge_type="dilemma_ordering",
        )
        assert len(edges) == 0

    def test_ending_salience_stored_on_dilemma_node(self) -> None:
        """ending_salience from analysis is stored on the dilemma graph node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "Core story choice",
                "ending_salience": "high",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        assert node["ending_salience"] == "high"

    def test_ending_salience_none_stored(self) -> None:
        """ending_salience: none is stored on the dilemma graph node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "stay_or_go",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "reasoning": "Cosmetic choice",
                "ending_salience": "none",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::stay_or_go")
        assert node["ending_salience"] == "none"

    def test_absent_ending_salience_uses_default(self) -> None:
        """When ending_salience is absent from analysis dict, the default 'none' is stored.

        R-7.3 requires every dilemma to have ending_salience on the graph node.
        The mutations code defaults to 'none' when not provided.
        """
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "test",
                "residue_weight": "heavy",
                # ending_salience intentionally absent
            },
            {
                "dilemma_id": "stay_or_go",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "reasoning": "minor choice",
                "ending_salience": "low",
                "residue_weight": "light",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        # Default "none" is applied when absent
        assert node["ending_salience"] == "none"

    def test_residue_weight_stored_on_dilemma_node(self) -> None:
        """residue_weight from analysis is stored on the dilemma graph node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "soft",
                "payoff_budget": 3,
                "reasoning": "Core story choice",
                "residue_weight": "heavy",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        assert node["residue_weight"] == "heavy"

    def test_residue_weight_cosmetic_stored(self) -> None:
        """residue_weight: cosmetic is stored on the dilemma graph node."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "stay_or_go",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "reasoning": "Cosmetic choice",
                "residue_weight": "cosmetic",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::stay_or_go")
        assert node["residue_weight"] == "cosmetic"

    def test_absent_residue_weight_uses_default(self) -> None:
        """When residue_weight is absent from analysis dict, the default 'cosmetic' is stored.

        R-7.2 requires every dilemma to have residue_weight on the graph node.
        The mutations code defaults to 'cosmetic' when not provided.
        """
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "test",
                "ending_salience": "high",
                # residue_weight intentionally absent
            },
            {
                "dilemma_id": "stay_or_go",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "reasoning": "minor choice",
                "ending_salience": "low",
                "residue_weight": "light",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        # Default "cosmetic" is applied when absent
        assert node["residue_weight"] == "cosmetic"


# ---------------------------------------------------------------------------
# Tasks 19 + 20 — concurrent normalization (R-8.3) and shared_entity rejection (R-8.4)
# ---------------------------------------------------------------------------


class TestApplySeedConcurrentOrdering:
    """R-8.3: concurrent ordering must be stored with lex-smaller dilemma_id as dilemma_a."""

    def _graph_with_two_dilemmas(self) -> Graph:
        """Build a fully BRAINSTORM-compliant graph with two extra dilemmas.

        Uses mentor_trust and z_later (lex order matters for R-8.3 tests).
        Each has two answers with descriptions and an anchored_to edge.
        """
        return _create_compliant_brainstorm_graph(
            extra_dilemmas=[
                (
                    "mentor_trust",
                    "Trust the mentor?",
                    [
                        ("yes", "Protagonist trusts the mentor.", True),
                        ("no", "Protagonist distrusts the mentor.", False),
                    ],
                    "character::mentor",
                ),
                (
                    "z_later",
                    "Go later?",
                    [
                        ("yes", "Protagonist goes later.", True),
                        ("no", "Protagonist goes now.", False),
                    ],
                    "location::archive",
                ),
            ]
        )

    def _base_output(self) -> dict:
        """Fully contract-compliant SEED output for concurrent ordering tests.

        Covers all three dilemmas (trust, mentor_trust, z_later).
        """
        return {
            "entities": [
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
                {"entity_id": "tower", "disposition": "retained"},
            ],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "explored": ["protector", "manipulator"],
                    "unexplored": [],
                },
                {
                    "dilemma_id": "mentor_trust",
                    "explored": ["yes", "no"],
                    "unexplored": [],
                },
                {
                    "dilemma_id": "z_later",
                    "explored": ["yes", "no"],
                    "unexplored": [],
                },
            ],
            "paths": [
                {
                    "path_id": "trust__protector",
                    "name": "Protector Arc",
                    "dilemma_id": "trust",
                    "answer_id": "protector",
                    "path_importance": "major",
                    "description": "Mentor protects.",
                },
                {
                    "path_id": "trust__manipulator",
                    "name": "Manipulator Arc",
                    "dilemma_id": "trust",
                    "answer_id": "manipulator",
                    "path_importance": "major",
                    "description": "Mentor manipulates.",
                },
                {
                    "path_id": "mentor_trust__yes",
                    "name": "Trust",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "yes",
                    "path_importance": "major",
                    "description": "Trust the mentor",
                },
                {
                    "path_id": "mentor_trust__no",
                    "name": "No Trust",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "no",
                    "path_importance": "major",
                    "description": "Distrust the mentor",
                },
                {
                    "path_id": "z_later__yes",
                    "name": "Later",
                    "dilemma_id": "z_later",
                    "answer_id": "yes",
                    "path_importance": "major",
                    "description": "Go later",
                },
                {
                    "path_id": "z_later__no",
                    "name": "Now",
                    "dilemma_id": "z_later",
                    "answer_id": "no",
                    "path_importance": "minor",
                    "description": "Go now",
                },
            ],
            "consequences": [
                {
                    "consequence_id": "c_trust_p",
                    "path_id": "trust__protector",
                    "description": "Ally gained.",
                    "narrative_effects": ["Trust shapes the climax"],
                },
                {
                    "consequence_id": "c_trust_m",
                    "path_id": "trust__manipulator",
                    "description": "Danger alone.",
                    "narrative_effects": ["Isolation defines the climax"],
                },
                {
                    "consequence_id": "c_mt_yes",
                    "path_id": "mentor_trust__yes",
                    "description": "Mentor trusted.",
                    "narrative_effects": ["Alliance unlocks new paths"],
                },
                {
                    "consequence_id": "c_mt_no",
                    "path_id": "mentor_trust__no",
                    "description": "Mentor distrusted.",
                    "narrative_effects": ["Solo path limits options"],
                },
                {
                    "consequence_id": "c_zl_yes",
                    "path_id": "z_later__yes",
                    "description": "Delayed action.",
                    "narrative_effects": ["Patience enables preparation"],
                },
                {
                    "consequence_id": "c_zl_no",
                    "path_id": "z_later__no",
                    "description": "Immediate action.",
                    "narrative_effects": ["Speed sacrifices planning"],
                },
            ],
            "initial_beats": [
                # trust dilemma
                {
                    "beat_id": "trust_pre",
                    "summary": "Mentor encounter",
                    "path_id": "trust__protector",
                    "also_belongs_to": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_cp",
                    "summary": "Kay trusts.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_pp1",
                    "summary": "Secret revealed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_pp2",
                    "summary": "Bond confirmed.",
                    "path_id": "trust__protector",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_cm",
                    "summary": "Kay distrusts.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_pm1",
                    "summary": "Motive exposed.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "trust_pm2",
                    "summary": "Danger alone.",
                    "path_id": "trust__manipulator",
                    "dilemma_impacts": [{"dilemma_id": "trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # mentor_trust dilemma
                {
                    "beat_id": "mt_pre",
                    "summary": "Trust question",
                    "path_id": "mentor_trust__yes",
                    "also_belongs_to": "mentor_trust__no",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b1",
                    "summary": "Commit mentor trust",
                    "path_id": "mentor_trust__yes",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b1_post",
                    "summary": "After mentor trust",
                    "path_id": "mentor_trust__yes",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b1_post2",
                    "summary": "Trust confirmed.",
                    "path_id": "mentor_trust__yes",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b1_no",
                    "summary": "Commit no trust",
                    "path_id": "mentor_trust__no",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b1_no_post",
                    "summary": "Distrust aftermath.",
                    "path_id": "mentor_trust__no",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b1_no_post2",
                    "summary": "Distrust confirmed.",
                    "path_id": "mentor_trust__no",
                    "dilemma_impacts": [{"dilemma_id": "mentor_trust", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                # z_later dilemma
                {
                    "beat_id": "zl_pre",
                    "summary": "Later question",
                    "path_id": "z_later__yes",
                    "also_belongs_to": "z_later__no",
                    "dilemma_impacts": [{"dilemma_id": "z_later", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2",
                    "summary": "Commit z later",
                    "path_id": "z_later__yes",
                    "dilemma_impacts": [{"dilemma_id": "z_later", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2_post",
                    "summary": "After z later",
                    "path_id": "z_later__yes",
                    "dilemma_impacts": [{"dilemma_id": "z_later", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2_post2",
                    "summary": "Later confirmed.",
                    "path_id": "z_later__yes",
                    "dilemma_impacts": [{"dilemma_id": "z_later", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2_no",
                    "summary": "Commit go now",
                    "path_id": "z_later__no",
                    "dilemma_impacts": [{"dilemma_id": "z_later", "effect": "commits"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2_no_post",
                    "summary": "Now aftermath.",
                    "path_id": "z_later__no",
                    "dilemma_impacts": [{"dilemma_id": "z_later", "effect": "advances"}],
                    "entities": ["mentor"],
                },
                {
                    "beat_id": "b2_no_post2",
                    "summary": "Now confirmed.",
                    "path_id": "z_later__no",
                    "dilemma_impacts": [{"dilemma_id": "z_later", "effect": "advances"}],
                    "entities": ["mentor"],
                },
            ],
            "dilemma_analyses": [
                {
                    "dilemma_id": "trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 2,
                    "reasoning": "Binary trust question.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
                {
                    "dilemma_id": "mentor_trust",
                    "dilemma_role": "hard",
                    "payoff_budget": 4,
                    "reasoning": "Core trust choice.",
                    "ending_salience": "high",
                    "residue_weight": "heavy",
                },
                {
                    "dilemma_id": "z_later",
                    "dilemma_role": "soft",
                    "payoff_budget": 2,
                    "reasoning": "Timing question.",
                    "ending_salience": "low",
                    "residue_weight": "light",
                },
            ],
            "human_approved_paths": True,
        }

    def test_concurrent_ordering_normalized_to_lex_smaller_a(self) -> None:
        """R-8.3: reversed concurrent pair is normalized so lex-smaller dilemma is dilemma_a."""
        graph = self._graph_with_two_dilemmas()
        output = self._base_output()
        output["dilemma_relationships"] = [
            {
                "dilemma_a": "z_later",  # lex-larger — should be swapped
                "dilemma_b": "mentor_trust",
                "ordering": "concurrent",
                "description": "Both run at the same time.",
                "reasoning": "They overlap narratively.",
            }
        ]
        apply_seed_mutations(graph, output)

        # Edge should exist FROM lex-smaller (mentor_trust) TO lex-larger (z_later)
        edges_from_mentor = graph.get_edges(from_id="dilemma::mentor_trust", edge_type="concurrent")
        assert len(edges_from_mentor) == 1
        assert edges_from_mentor[0]["to"] == "dilemma::z_later"

        # No edge in the reversed direction
        edges_from_z = graph.get_edges(from_id="dilemma::z_later", edge_type="concurrent")
        assert len(edges_from_z) == 0

    def test_concurrent_already_ordered_unchanged(self) -> None:
        """R-8.3: pair already in lex order is written as-is."""
        graph = self._graph_with_two_dilemmas()
        output = self._base_output()
        output["dilemma_relationships"] = [
            {
                "dilemma_a": "mentor_trust",  # lex-smaller — already correct
                "dilemma_b": "z_later",
                "ordering": "concurrent",
                "description": "Both run at the same time.",
                "reasoning": "They overlap.",
            }
        ]
        apply_seed_mutations(graph, output)
        edges = graph.get_edges(from_id="dilemma::mentor_trust", edge_type="concurrent")
        assert len(edges) == 1
        assert edges[0]["to"] == "dilemma::z_later"

    def test_shared_entity_relationship_raises_mutation_error(self) -> None:
        """R-8.4: shared_entity is derived, never declared; apply_seed_mutations must raise."""
        from questfoundry.graph.mutations import MutationError

        graph = self._graph_with_two_dilemmas()
        output = self._base_output()
        output["dilemma_relationships"] = [
            {
                "dilemma_a": "mentor_trust",
                "dilemma_b": "z_later",
                "ordering": "shared_entity",  # forbidden
                "description": "They share an entity.",
                "reasoning": "Same entity.",
            }
        ]
        with pytest.raises(MutationError, match="shared_entity"):
            apply_seed_mutations(graph, output)


# ---------------------------------------------------------------------------
# Phase 2 - Y-shape dual belongs_to tests (Tasks 2.1-2.7)
# ---------------------------------------------------------------------------


def test_get_path_ids_from_beat_post_commit_returns_one() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    beat = {"path_id": "path::a"}
    assert _get_path_ids_from_beat(beat) == ("path::a",)


def test_get_path_ids_from_beat_pre_commit_returns_both() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    beat = {"path_id": "path::a", "also_belongs_to": "path::b"}
    assert _get_path_ids_from_beat(beat) == ("path::a", "path::b")


def test_get_path_ids_from_beat_legacy_paths_list_returns_all() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    beat = {"paths": ["path::a", "path::b"]}
    assert _get_path_ids_from_beat(beat) == ("path::a", "path::b")


def test_get_path_ids_from_beat_empty_returns_empty_tuple() -> None:
    from questfoundry.graph.mutations import _get_path_ids_from_beat

    assert _get_path_ids_from_beat({}) == ()


# ---------------------------------------------------------------------------
# Shared helpers for Y-shape guard rail tests
# ---------------------------------------------------------------------------


def _trust_graph() -> Graph:
    """Return a BRAINSTORM-contract-compliant graph for the trust dilemma.

    Contains ONLY ``dilemma::trust_protector_or_manipulator`` (no base ``trust``
    dilemma) so that ``_trust_seed_output`` only needs decisions for one dilemma.
    Includes vision, proper category-prefixed entity IDs, and ≥2 location
    entities so the SEED exit validator passes.
    Matches the seed returned by :func:`_trust_seed_output`.
    """
    g = Graph.empty()
    _create_compliant_vision(g)
    g.create_node(
        "character::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "category": "character",
            "name": "Mentor",
            "concept": "Senior archivist with hidden motives",
        },
    )
    g.create_node(
        "location::archive",
        {
            "type": "entity",
            "raw_id": "archive",
            "category": "location",
            "name": "The Archive",
            "concept": "Ancient repository of forbidden texts",
        },
    )
    g.create_node(
        "location::tower",
        {
            "type": "entity",
            "raw_id": "tower",
            "category": "location",
            "name": "The Tower",
            "concept": "Tall observatory on the hill",
        },
    )
    g.create_node(
        "dilemma::trust_protector_or_manipulator",
        {
            "type": "dilemma",
            "raw_id": "trust_protector_or_manipulator",
            "question": "Should the protagonist trust the mentor as protector or see through manipulation?",
            "why_it_matters": "Trust defines whether the protagonist gains an ally or faces a foe.",
        },
    )
    g.create_node(
        "dilemma::trust_protector_or_manipulator::alt::protector",
        {
            "type": "answer",
            "raw_id": "protector",
            "description": "The mentor genuinely protects the protagonist.",
            "is_canonical": True,
        },
    )
    g.create_node(
        "dilemma::trust_protector_or_manipulator::alt::manipulator",
        {
            "type": "answer",
            "raw_id": "manipulator",
            "description": "The mentor is secretly manipulating for personal gain.",
            "is_canonical": False,
        },
    )
    g.add_edge(
        "has_answer",
        "dilemma::trust_protector_or_manipulator",
        "dilemma::trust_protector_or_manipulator::alt::protector",
    )
    g.add_edge(
        "has_answer",
        "dilemma::trust_protector_or_manipulator",
        "dilemma::trust_protector_or_manipulator::alt::manipulator",
    )
    g.add_edge("anchored_to", "dilemma::trust_protector_or_manipulator", "character::mentor")
    return g


def _trust_seed_output(initial_beats: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """SEED output for the trust_protector_or_manipulator dilemma with two paths.

    Callers supply ``initial_beats``; defaults to a minimal compliant beat
    structure (1 shared pre-commit + 2 commit beats + 2 post-commit beats).
    All beats include ``entities`` to satisfy R-3.13.

    The base graph also contains ``dilemma::trust`` (from
    ``_create_compliant_brainstorm_graph``); that dilemma is left unexplored
    here — this is allowed by the SEED contract.
    """
    if initial_beats is None:
        initial_beats = [
            {
                "beat_id": "shared_setup",
                "summary": "Protagonist observes the mentor's first move.",
                "path_id": "trust_protector_or_manipulator__protector",
                "also_belongs_to": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "sets up the choice",
                    }
                ],
            },
            {
                "beat_id": "commit_protector",
                "summary": "Protagonist decides to trust the mentor.",
                "path_id": "trust_protector_or_manipulator__protector",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "locked in trust",
                    }
                ],
            },
            {
                "beat_id": "post_protector_1",
                "summary": "Mentor reveals hidden allies.",
                "path_id": "trust_protector_or_manipulator__protector",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "fallout begins",
                    }
                ],
            },
            {
                "beat_id": "post_protector_2",
                "summary": "Protagonist gains mentor's full support.",
                "path_id": "trust_protector_or_manipulator__protector",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "resolution",
                    }
                ],
            },
            {
                "beat_id": "commit_manipulator",
                "summary": "Protagonist sees through the mentor's deception.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "locked in distrust",
                    }
                ],
            },
            {
                "beat_id": "post_manipulator_1",
                "summary": "Protagonist confronts the mentor.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "confrontation",
                    }
                ],
            },
            {
                "beat_id": "post_manipulator_2",
                "summary": "Mentor escapes with stolen knowledge.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "resolution",
                    }
                ],
            },
        ]
    return {
        "entities": [
            {"entity_id": "mentor", "disposition": "retained"},
            {"entity_id": "archive", "disposition": "retained"},
            {"entity_id": "tower", "disposition": "retained"},
        ],
        "dilemmas": [
            {
                "dilemma_id": "trust_protector_or_manipulator",
                "explored": ["protector", "manipulator"],
                "unexplored": [],
            }
        ],
        "paths": [
            {
                "path_id": "trust_protector_or_manipulator__protector",
                "dilemma_id": "trust_protector_or_manipulator",
                "answer_id": "protector",
                "name": "Protector",
                "description": "The mentor genuinely protects the protagonist throughout.",
            },
            {
                "path_id": "trust_protector_or_manipulator__manipulator",
                "dilemma_id": "trust_protector_or_manipulator",
                "answer_id": "manipulator",
                "name": "Manipulator",
                "description": "The mentor secretly manipulates the protagonist for personal gain.",
            },
        ],
        "consequences": [
            {
                "consequence_id": "mentor_trusted",
                "path_id": "trust_protector_or_manipulator__protector",
                "description": "Trust is established; the mentor becomes a genuine ally.",
                "narrative_effects": ["protagonist gains access to hidden archives"],
            },
            {
                "consequence_id": "mentor_distrusted",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "description": "Betrayal is exposed; the mentor becomes an antagonist.",
                "narrative_effects": ["protagonist must act without support"],
            },
        ],
        "dilemma_analyses": [
            {
                "dilemma_id": "trust_protector_or_manipulator",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "ending_salience": "none",
                "residue_weight": "cosmetic",
            }
        ],
        "initial_beats": initial_beats,
        "human_approved_paths": True,
    }


def _two_dilemma_graph() -> Graph:
    """Return a BRAINSTORM-contract-compliant graph with two unrelated dilemmas.

    Contains ONLY ``dilemma_a`` and ``dilemma_b`` (no base ``trust`` dilemma)
    so that ``_two_dilemma_seed_output`` only needs decisions for two dilemmas.
    Includes vision node, proper category-prefixed entity IDs, and ≥2 location
    entities (R-2.4).
    """
    g = Graph.empty()
    _create_compliant_vision(g)
    g.create_node(
        "character::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "category": "character",
            "name": "Mentor",
            "concept": "Senior archivist with hidden motives",
        },
    )
    g.create_node(
        "location::archive",
        {
            "type": "entity",
            "raw_id": "archive",
            "category": "location",
            "name": "The Archive",
            "concept": "Ancient repository of forbidden texts",
        },
    )
    g.create_node(
        "location::tower",
        {
            "type": "entity",
            "raw_id": "tower",
            "category": "location",
            "name": "The Tower",
            "concept": "Tall observatory on the hill",
        },
    )
    # Dilemma A
    g.create_node(
        "dilemma::dilemma_a",
        {
            "type": "dilemma",
            "raw_id": "dilemma_a",
            "question": "Should the protagonist choose answer A1 or A2?",
            "why_it_matters": "This choice defines the confrontation arc.",
        },
    )
    g.create_node(
        "dilemma::dilemma_a::alt::answer_a1",
        {
            "type": "answer",
            "raw_id": "answer_a1",
            "description": "Taking path A1 leads to confrontation.",
            "is_canonical": True,
        },
    )
    g.create_node(
        "dilemma::dilemma_a::alt::answer_a2",
        {
            "type": "answer",
            "raw_id": "answer_a2",
            "description": "Taking path A2 avoids open conflict.",
            "is_canonical": False,
        },
    )
    g.add_edge("has_answer", "dilemma::dilemma_a", "dilemma::dilemma_a::alt::answer_a1")
    g.add_edge("has_answer", "dilemma::dilemma_a", "dilemma::dilemma_a::alt::answer_a2")
    g.add_edge("anchored_to", "dilemma::dilemma_a", "character::mentor")
    # Dilemma B
    g.create_node(
        "dilemma::dilemma_b",
        {
            "type": "dilemma",
            "raw_id": "dilemma_b",
            "question": "Should the protagonist choose answer B1 or B2?",
            "why_it_matters": "This choice defines the alliance arc.",
        },
    )
    g.create_node(
        "dilemma::dilemma_b::alt::answer_b1",
        {
            "type": "answer",
            "raw_id": "answer_b1",
            "description": "Taking path B1 forges a new alliance.",
            "is_canonical": True,
        },
    )
    g.create_node(
        "dilemma::dilemma_b::alt::answer_b2",
        {
            "type": "answer",
            "raw_id": "answer_b2",
            "description": "Taking path B2 preserves independence.",
            "is_canonical": False,
        },
    )
    g.add_edge("has_answer", "dilemma::dilemma_b", "dilemma::dilemma_b::alt::answer_b1")
    g.add_edge("has_answer", "dilemma::dilemma_b", "dilemma::dilemma_b::alt::answer_b2")
    g.add_edge("anchored_to", "dilemma::dilemma_b", "character::mentor")
    return g


def _two_dilemma_seed_output(initial_beats: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """SEED output for two unrelated dilemmas with one explored path each.

    All tests using this helper pass custom ``initial_beats`` that trigger guard
    rail errors BEFORE the exit validator runs, so the default beats here are a
    minimal compliant fallback (not exercised by any current test).
    """
    if initial_beats is None:
        initial_beats = [
            {
                "beat_id": "commit_a",
                "summary": "Dilemma A commits.",
                "path_id": "dilemma_a__answer_a1",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_a_1",
                "summary": "Dilemma A aftermath.",
                "path_id": "dilemma_a__answer_a1",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"}],
            },
            {
                "beat_id": "post_a_2",
                "summary": "Dilemma A resolution.",
                "path_id": "dilemma_a__answer_a1",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"}],
            },
            {
                "beat_id": "commit_b",
                "summary": "Dilemma B commits.",
                "path_id": "dilemma_b__answer_b1",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_b_1",
                "summary": "Dilemma B aftermath.",
                "path_id": "dilemma_b__answer_b1",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "advances", "note": "x"}],
            },
            {
                "beat_id": "post_b_2",
                "summary": "Dilemma B resolution.",
                "path_id": "dilemma_b__answer_b1",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "advances", "note": "x"}],
            },
        ]
    return {
        "entities": [
            {"entity_id": "mentor", "disposition": "retained"},
            {"entity_id": "archive", "disposition": "retained"},
            {"entity_id": "tower", "disposition": "retained"},
        ],
        "dilemmas": [
            {"dilemma_id": "dilemma_a", "explored": ["answer_a1"], "unexplored": ["answer_a2"]},
            {"dilemma_id": "dilemma_b", "explored": ["answer_b1"], "unexplored": ["answer_b2"]},
        ],
        "paths": [
            {
                "path_id": "dilemma_a__answer_a1",
                "dilemma_id": "dilemma_a",
                "answer_id": "answer_a1",
                "name": "A1 Path",
                "description": "The confrontation path for dilemma A.",
            },
            {
                "path_id": "dilemma_b__answer_b1",
                "dilemma_id": "dilemma_b",
                "answer_id": "answer_b1",
                "name": "B1 Path",
                "description": "The alliance-forging path for dilemma B.",
            },
        ],
        "consequences": [
            {
                "consequence_id": "c_a",
                "path_id": "dilemma_a__answer_a1",
                "description": "Confrontation reshapes the protagonist's standing.",
                "narrative_effects": ["protagonist gains respect through conflict"],
            },
            {
                "consequence_id": "c_b",
                "path_id": "dilemma_b__answer_b1",
                "description": "Alliance shifts the balance of power.",
                "narrative_effects": ["protagonist gains a powerful ally"],
            },
        ],
        "dilemma_analyses": [
            {
                "dilemma_id": "dilemma_a",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "ending_salience": "none",
                "residue_weight": "cosmetic",
            },
            {
                "dilemma_id": "dilemma_b",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "ending_salience": "none",
                "residue_weight": "cosmetic",
            },
        ],
        "initial_beats": initial_beats,
        "human_approved_paths": True,
    }


# ---------------------------------------------------------------------------
# Task 2.3: dual belongs_to edge emission
# ---------------------------------------------------------------------------


def test_apply_seed_mutations_emits_dual_belongs_to_for_pre_commit_beat() -> None:
    """Pre-commit beats with ``also_belongs_to`` get two ``belongs_to`` edges."""
    from questfoundry.graph.mutations import apply_seed_mutations

    graph = _trust_graph()
    seed = _trust_seed_output(
        initial_beats=[
            {
                "beat_id": "shared_setup",
                "summary": "Both players see this setup.",
                "path_id": "trust_protector_or_manipulator__protector",
                "also_belongs_to": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "x",
                    },
                ],
            },
            # Each path needs a commit beat and ≥2 post-commit beats (R-3.12).
            {
                "beat_id": "commit_protector",
                "summary": "Protector path commits.",
                "path_id": "trust_protector_or_manipulator__protector",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "locked",
                    }
                ],
            },
            {
                "beat_id": "post_protector_1",
                "summary": "Protector aftermath begins.",
                "path_id": "trust_protector_or_manipulator__protector",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "fallout",
                    }
                ],
            },
            {
                "beat_id": "post_protector_2",
                "summary": "Protector path resolves.",
                "path_id": "trust_protector_or_manipulator__protector",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "resolution",
                    }
                ],
            },
            {
                "beat_id": "commit_manipulator",
                "summary": "Manipulator path commits.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "locked",
                    }
                ],
            },
            {
                "beat_id": "post_manipulator_1",
                "summary": "Manipulator aftermath begins.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "fallout",
                    }
                ],
            },
            {
                "beat_id": "post_manipulator_2",
                "summary": "Manipulator path resolves.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "entities": ["character::mentor"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "resolution",
                    }
                ],
            },
        ]
    )
    apply_seed_mutations(graph, seed)

    edges = [
        e for e in graph.get_edges(edge_type="belongs_to") if e["from"] == "beat::shared_setup"
    ]
    to_ids = {e["to"] for e in edges}
    assert to_ids == {
        "path::trust_protector_or_manipulator__protector",
        "path::trust_protector_or_manipulator__manipulator",
    }


# ---------------------------------------------------------------------------
# Task 2.4: guard rail 1 - cross-dilemma dual belongs_to is forbidden
# ---------------------------------------------------------------------------


def test_apply_seed_mutations_rejects_cross_dilemma_dual_belongs_to() -> None:
    """Guard rail 1: pre-commit beats must share a dilemma across both paths."""
    from questfoundry.graph.mutations import apply_seed_mutations

    graph = _two_dilemma_graph()
    seed = _two_dilemma_seed_output(
        initial_beats=[
            {
                "beat_id": "bad_dual",
                "summary": "Cross-dilemma.",
                "path_id": "dilemma_a__answer_a1",
                "also_belongs_to": "dilemma_b__answer_b1",
                "dilemma_impacts": [
                    {"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"},
                ],
            },
            # Still need commits beats for validation to pass other checks.
            {
                "beat_id": "commit_a",
                "summary": "Dilemma A commits.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_a",
                "summary": "Dilemma A aftermath.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"}],
            },
            {
                "beat_id": "commit_b",
                "summary": "Dilemma B commits.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_b",
                "summary": "Dilemma B aftermath.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "advances", "note": "x"}],
            },
        ]
    )

    with pytest.raises(ValueError, match="cross-dilemma dual belongs_to"):
        apply_seed_mutations(graph, seed)


def test_apply_seed_mutations_rejects_also_belongs_to_equal_path_id() -> None:
    """Raw-dict input with path_id == also_belongs_to must be rejected (defense-in-depth)."""
    # Bypass Pydantic by constructing the seed dict directly.
    from questfoundry.graph.mutations import apply_seed_mutations

    graph = _two_dilemma_graph()
    seed = _two_dilemma_seed_output(
        initial_beats=[
            {
                "beat_id": "bad_beat",
                "summary": "Same path twice.",
                "path_id": "dilemma_a__answer_a1",
                "also_belongs_to": "dilemma_a__answer_a1",
                "dilemma_impacts": [
                    {
                        "dilemma_id": "dilemma_a",
                        "effect": "advances",
                        "note": "x",
                    }
                ],
            },
            {
                "beat_id": "commit_a",
                "summary": "Dilemma A commits.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_a",
                "summary": "Dilemma A aftermath.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"}],
            },
            {
                "beat_id": "commit_b",
                "summary": "Dilemma B commits.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_b",
                "summary": "Dilemma B aftermath.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "advances", "note": "x"}],
            },
        ]
    )
    with pytest.raises(ValueError, match="must be distinct paths"):
        apply_seed_mutations(graph, seed)


# ---------------------------------------------------------------------------
# Task 2.5: guard rail 2 - commit beats must be single-membership
# ---------------------------------------------------------------------------


def test_apply_seed_mutations_rejects_dual_on_commit_beat() -> None:
    """Guard rail 2: a beat with ``effect=commits`` must have only one belongs_to."""
    from questfoundry.graph.mutations import apply_seed_mutations

    graph = _trust_graph()
    # Provide valid commit beats for each path (so validation passes),
    # then the bad_commit beat with dual membership + commits effect hits guard rail 2.
    seed = _trust_seed_output(
        initial_beats=[
            {
                "beat_id": "bad_commit",
                "summary": "A commit beat cannot be pre-commit.",
                "path_id": "trust_protector_or_manipulator__protector",
                "also_belongs_to": "trust_protector_or_manipulator__manipulator",
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "Bad: commit with dual.",
                    },
                ],
            },
            # Valid single-membership commit beats so the per-path validation passes.
            {
                "beat_id": "commit_protector",
                "summary": "Protector commits.",
                "path_id": "trust_protector_or_manipulator__protector",
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "locked",
                    }
                ],
            },
            {
                "beat_id": "post_protector",
                "summary": "Protector aftermath.",
                "path_id": "trust_protector_or_manipulator__protector",
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "fallout",
                    }
                ],
            },
            {
                "beat_id": "commit_manipulator",
                "summary": "Manipulator commits.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "locked",
                    }
                ],
            },
            {
                "beat_id": "post_manipulator",
                "summary": "Manipulator aftermath.",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "fallout",
                    }
                ],
            },
        ]
    )
    with pytest.raises(ValueError, match="guard rail 2"):
        apply_seed_mutations(graph, seed)


# ---------------------------------------------------------------------------
# Task 11 (#1282): write-time cross-dilemma rejection (R-3.6 / R-3.9)
# ---------------------------------------------------------------------------


def test_apply_seed_mutations_rejects_precommit_with_mismatched_dilemmas() -> None:
    """R-3.6 / R-3.9: pre-commit dual belongs_to must reference paths of the
    same dilemma. Mismatched dilemmas must raise at write time, not slip
    through to the exit validator."""
    from questfoundry.graph.mutations import apply_seed_mutations

    # Reuse the two-dilemma fixture: dilemma_a and dilemma_b are separate.
    graph = _two_dilemma_graph()
    seed = _two_dilemma_seed_output(
        initial_beats=[
            {
                "beat_id": "bad_precommit",
                "summary": "Cross-dilemma pre-commit beat.",
                "path_id": "dilemma_a__answer_a1",
                "also_belongs_to": "dilemma_b__answer_b1",
                "dilemma_impacts": [
                    {"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"},
                ],
            },
            # Commits for each path to satisfy other validation rules.
            {
                "beat_id": "commit_a",
                "summary": "Dilemma A commits.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_a",
                "summary": "Dilemma A aftermath.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"}],
            },
            {
                "beat_id": "commit_b",
                "summary": "Dilemma B commits.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_b",
                "summary": "Dilemma B aftermath.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "advances", "note": "x"}],
            },
        ]
    )

    # Must raise at write time (before the exit validator is reached).
    with pytest.raises((ValueError, Exception), match="cross-dilemma dual belongs_to"):
        apply_seed_mutations(graph, seed)


# ---------------------------------------------------------------------------
# Task 12 (#1283): post-apply property test — no beat has cross-dilemma
# belongs_to edges in the resulting graph (R-3.9).
#
# This test covers the *broader* prohibition: not just the primary path_id +
# also_belongs_to pair, but any beat in the graph after apply completes.
# If future code adds a second belongs_to write path this test catches it.
# ---------------------------------------------------------------------------


def test_apply_seed_mutations_never_produces_cross_dilemma_belongs_to() -> None:
    """R-3.9: after the beat-write phase of apply_seed_mutations, no beat in the
    graph has ``belongs_to`` edges to paths of more than one dilemma.

    The exit validator is patched so we can focus on the graph-state invariant
    independently of fixture completeness. This is a property test of the write
    path, not of the exit validator.

    Broader than Task 11 (#1282): that test guards the primary path_id +
    also_belongs_to pair; this test inspects the full resulting graph."""
    from unittest.mock import patch

    from questfoundry.graph.mutations import apply_seed_mutations

    graph = _two_dilemma_graph()
    seed = _two_dilemma_seed_output(
        initial_beats=[
            {
                "beat_id": "commit_a",
                "summary": "Dilemma A commits.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_a",
                "summary": "Dilemma A aftermath.",
                "path_id": "dilemma_a__answer_a1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_a", "effect": "advances", "note": "x"}],
            },
            {
                "beat_id": "commit_b",
                "summary": "Dilemma B commits.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "commits", "note": "x"}],
            },
            {
                "beat_id": "post_b",
                "summary": "Dilemma B aftermath.",
                "path_id": "dilemma_b__answer_b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma_b", "effect": "advances", "note": "x"}],
            },
        ]
    )

    # Patch exit validator so we can inspect the graph state without requiring
    # a fully-compliant fixture (R-7.1/R-7.2/R-7.3/R-6.4/etc. would fail).
    with patch("questfoundry.graph.mutations.validate_seed_output", return_value=[]):
        apply_seed_mutations(graph, seed)

    # Collect all belongs_to edges and group by source beat.
    beat_to_paths: dict[str, list[str]] = {}
    for edge in graph.get_edges(edge_type="belongs_to"):
        src = edge["from"]
        tgt = edge["to"]
        beat_to_paths.setdefault(src, []).append(tgt)

    # For every beat, all its belongs_to targets must share one dilemma_id.
    for beat_id, path_ids in beat_to_paths.items():
        dilemmas: set[str | None] = set()
        for pid in path_ids:
            node = graph.get_node(pid)
            if node:
                dilemmas.add(node.get("dilemma_id"))
        dilemmas.discard(None)
        assert len(dilemmas) <= 1, (
            f"Beat {beat_id!r} has cross-dilemma belongs_to edges: "
            f"paths {path_ids!r} span dilemmas {dilemmas!r}"
        )
