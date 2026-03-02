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
        output = {"genre": "noir", "themes": ["trust"], "tone": ["dark"]}

        apply_mutations(graph, "dream", output)

        assert graph.has_node("vision")

    def test_routes_to_brainstorm(self) -> None:
        """Routes brainstorm stage to apply_brainstorm_mutations."""
        graph = Graph.empty()
        output = {
            "entities": [
                {"entity_id": "char_001", "entity_category": "character", "concept": "Test"}
            ],
            "dilemmas": [],
        }

        apply_mutations(graph, "brainstorm", output)

        # Entities now use category prefix (character::, location::, etc.)
        assert graph.has_node("character::char_001")

    def test_routes_to_seed(self) -> None:
        """Routes seed stage to apply_seed_mutations."""
        graph = Graph.empty()
        # Pre-populate with entity from brainstorm
        graph.create_node(
            "entity::char_001",
            {"type": "entity", "raw_id": "char_001", "disposition": "proposed"},
        )
        # Add 2 dilemmas with both answers (required for minimum arc count)
        for i in range(2):
            tid = f"dilemma::t{i}"
            graph.create_node(tid, {"type": "dilemma", "raw_id": f"t{i}"})
            for alt in ["a", "b"]:
                alt_id = f"{tid}::alt::{alt}"
                graph.create_node(alt_id, {"type": "answer", "raw_id": alt})
                graph.add_edge("has_answer", tid, alt_id)

        output = {
            "entities": [{"entity_id": "char_001", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "t0", "explored": ["a", "b"], "unexplored": []},
                {"dilemma_id": "t1", "explored": ["a", "b"], "unexplored": []},
            ],
            "paths": [
                {"path_id": "path_0", "dilemma_id": "t0", "answer_id": "a"},
                {"path_id": "path_1", "dilemma_id": "t0", "answer_id": "b"},
                {"path_id": "path_2", "dilemma_id": "t1", "answer_id": "a"},
                {"path_id": "path_3", "dilemma_id": "t1", "answer_id": "b"},
            ],
            "initial_beats": [
                # Minimal beats with commits for each path
                {
                    "beat_id": "b0",
                    "paths": ["path_0"],
                    "dilemma_impacts": [{"dilemma_id": "t0", "effect": "commits"}],
                },
                {
                    "beat_id": "b1",
                    "paths": ["path_1"],
                    "dilemma_impacts": [{"dilemma_id": "t0", "effect": "commits"}],
                },
                {
                    "beat_id": "b2",
                    "paths": ["path_2"],
                    "dilemma_impacts": [{"dilemma_id": "t1", "effect": "commits"}],
                },
                {
                    "beat_id": "b3",
                    "paths": ["path_3"],
                    "dilemma_impacts": [{"dilemma_id": "t1", "effect": "commits"}],
                },
            ],
        }

        apply_mutations(graph, "seed", output)

        assert graph.get_node("entity::char_001")["disposition"] == "retained"

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

        apply_dream_mutations(graph, {"genre": "noir", "themes": [], "tone": []})

        assert graph.get_node("vision")["genre"] == "noir"

    def test_handles_optional_fields(self) -> None:
        """Handles missing optional fields gracefully."""
        graph = Graph.empty()
        output = {
            "genre": "mystery",
            "themes": ["intrigue"],
            "tone": ["suspenseful"],
            "audience": "adult",
            # No subgenre, style_notes, scope, content_notes
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
            "themes": [],
            "tone": [],
            "audience": "ya",
            "scope": {
                "estimated_passages": 50,
                "target_word_count": 25000,
                "branching_depth": "moderate",
            },
        }

        apply_dream_mutations(graph, output)

        vision = graph.get_node("vision")
        assert vision is not None
        assert vision["scope"]["estimated_passages"] == 50

    def test_includes_pov_fields_if_present(self) -> None:
        """Includes POV hint fields if present."""
        graph = Graph.empty()
        output = {
            "genre": "horror",
            "themes": ["fear"],
            "tone": ["tense"],
            "audience": "adult",
            "pov_style": "second",
            "protagonist_defined": True,
        }

        apply_dream_mutations(graph, output)

        vision = graph.get_node("vision")
        assert vision is not None
        assert vision["pov_style"] == "second"
        assert vision["protagonist_defined"] is True

    def test_pov_fields_default_correctly(self) -> None:
        """POV fields have correct defaults when not provided."""
        graph = Graph.empty()
        output = {
            "genre": "fantasy",
            "themes": ["adventure"],
            "tone": ["light"],
            "audience": "ya",
            # No pov_style or protagonist_defined
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
        output = {
            "entities": [
                {
                    "entity_id": "kay",
                    "entity_category": "character",
                    "concept": "Young archivist",
                    "notes": "Curious and brave",
                },
                {
                    "entity_id": "archive",
                    "entity_category": "location",
                    "concept": "Ancient repository",
                },
            ],
            "dilemmas": [],
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
                    "concept": "A crumbling estate",
                    # name omitted — should not appear on node
                },
            ],
            "dilemmas": [],
        }

        apply_brainstorm_mutations(graph, output)

        beatrice = graph.get_node("character::beatrice")
        assert beatrice is not None
        assert beatrice["name"] == "Lady Beatrice Ashford"

        manor = graph.get_node("location::manor")
        assert manor is not None
        assert "name" not in manor  # _clean_dict removes None values

    def test_strips_scope_prefixes_in_raw_ids(self) -> None:
        """Scoped IDs are stored unscoped in raw_id fields."""
        graph = Graph.empty()
        output = {
            "entities": [
                {
                    "entity_id": "entity::kay",
                    "entity_category": "character",
                    "concept": "Young archivist",
                }
            ],
            "dilemmas": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "question": "Can the mentor be trusted?",
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
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Protagonist"},
                {"entity_id": "mentor", "entity_category": "character", "concept": "Guide"},
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
        """Handles empty entities and dilemmas."""
        graph = Graph.empty()
        output = {"entities": [], "dilemmas": []}

        apply_brainstorm_mutations(graph, output)

        assert len(graph.to_dict()["nodes"]) == 0


class TestValidateBrainstormMutations:
    """Test BRAINSTORM semantic validation."""

    def test_valid_output_returns_empty(self) -> None:
        """Valid output with matching entity references returns no errors."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
                {"entity_id": "mentor", "entity_category": "character", "concept": "Mentor"},
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
                    "central_entity_ids": [],
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

    def test_no_default_path_detected(self) -> None:
        """Detects when no answer has is_canonical=True."""
        output = {
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],
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
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],
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

        # Should find: 2 phantom entity errors + 1 no default error
        assert len(errors) == 3

    def test_empty_dilemmas_valid(self) -> None:
        """Empty dilemmas list is valid."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
            ],
            "dilemmas": [],
        }

        errors = validate_brainstorm_mutations(output)

        assert errors == []

    def test_empty_alternatives_detected(self) -> None:
        """Dilemma with no answers fails default path validation."""
        output = {
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],
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
                    "central_entity_ids": [],
                    "answers": [
                        {"answer_id": "trust", "description": "Trust", "is_canonical": True},
                        {"answer_id": "betray", "description": "Betray", "is_canonical": False},
                    ],
                },
                {
                    "dilemma_id": "trust_or_betray",
                    "question": "Duplicate dilemma",
                    "central_entity_ids": [],
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
        graph = Graph.empty()
        # Pre-populate entities from brainstorm (with raw_id for validation)
        graph.create_node(
            "entity::kay",
            {"type": "entity", "raw_id": "kay", "disposition": "proposed"},
        )
        graph.create_node(
            "entity::mentor",
            {"type": "entity", "raw_id": "mentor", "disposition": "proposed"},
        )
        graph.create_node(
            "entity::extra",
            {"type": "entity", "raw_id": "extra", "disposition": "proposed"},
        )

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},  # Raw IDs from LLM
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "extra", "disposition": "cut"},
            ],
            "paths": [],
            "initial_beats": [],
        }

        apply_seed_mutations(graph, output)

        assert graph.get_node("entity::kay")["disposition"] == "retained"
        assert graph.get_node("entity::mentor")["disposition"] == "retained"
        assert graph.get_node("entity::extra")["disposition"] == "cut"

    def test_creates_paths(self) -> None:
        """Creates path nodes from seed output."""
        graph = Graph.empty()
        # Pre-populate dilemma and answer from brainstorm (with raw_id for validation)
        graph.create_node(
            "dilemma::mentor_trust",
            {"type": "dilemma", "raw_id": "mentor_trust", "question": "Can the mentor be trusted?"},
        )
        graph.create_node(
            "dilemma::mentor_trust::alt::protector",
            {"type": "answer", "raw_id": "protector", "description": "Mentor protects"},
        )
        # Link dilemma to answer (add_edge takes edge_type, from_id, to_id)
        graph.add_edge(
            "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::protector"
        )

        output = {
            "entities": [],
            "dilemmas": [
                {"dilemma_id": "mentor_trust", "explored": ["protector"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "path_mentor_trust",
                    "name": "Mentor Trust Arc",
                    "dilemma_id": "mentor_trust",  # Raw dilemma ID from LLM
                    "answer_id": "protector",  # Local alt ID, not full path
                    "description": "Exploring mentor relationship",
                    "consequence_ids": ["consequence_trust"],
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "resolution",
                    "summary": "Mentor's true nature revealed",
                    "paths": ["path_mentor_trust"],
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked in"}
                    ],
                }
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
        """Path's is_canonical is set from answer's is_canonical."""
        graph = Graph.empty()
        # Create dilemma with two answers - one canonical, one not
        graph.create_node(
            "dilemma::mentor_trust",
            {"type": "dilemma", "raw_id": "mentor_trust", "question": "Can the mentor be trusted?"},
        )
        graph.create_node(
            "dilemma::mentor_trust::alt::protector",
            {
                "type": "answer",
                "raw_id": "protector",
                "description": "Mentor protects",
                "is_canonical": True,  # This is the canonical path
            },
        )
        graph.create_node(
            "dilemma::mentor_trust::alt::manipulator",
            {
                "type": "answer",
                "raw_id": "manipulator",
                "description": "Mentor manipulates",
                "is_canonical": False,  # Branch path
            },
        )
        graph.add_edge(
            "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::protector"
        )
        graph.add_edge(
            "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::manipulator"
        )

        output = {
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "mentor_trust",
                    "explored": ["protector", "manipulator"],
                    "unexplored": [],
                },
            ],
            "paths": [
                {
                    "path_id": "mentor_protects",
                    "name": "Mentor Protects Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "protector",  # Canonical
                    "description": "The mentor genuinely protects",
                    "consequence_ids": [],
                },
                {
                    "path_id": "mentor_manipulates",
                    "name": "Mentor Manipulates Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "manipulator",  # Non-canonical
                    "description": "The mentor is manipulating",
                    "consequence_ids": [],
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "protects_beat_01",
                    "summary": "Mentor reveals protection",
                    "paths": ["mentor_protects"],
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked"}
                    ],
                },
                {
                    "beat_id": "manipulates_beat_01",
                    "summary": "Mentor reveals manipulation",
                    "paths": ["mentor_manipulates"],
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked"}
                    ],
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
        """Creates beat nodes from seed output."""
        graph = Graph.empty()
        # Pre-populate entities from brainstorm (with raw_id for validation)
        graph.create_node(
            "entity::kay", {"type": "entity", "raw_id": "kay", "concept": "Young archivist"}
        )
        graph.create_node(
            "entity::mentor", {"type": "entity", "raw_id": "mentor", "concept": "Senior archivist"}
        )
        graph.create_node(
            "entity::archive",
            {"type": "entity", "raw_id": "archive", "concept": "Ancient repository"},
        )
        # Pre-populate dilemma and answer from brainstorm (for path validation)
        graph.create_node(
            "dilemma::mentor_trust",
            {"type": "dilemma", "raw_id": "mentor_trust", "question": "Can the mentor be trusted?"},
        )
        graph.create_node(
            "dilemma::mentor_trust::alt::protector",
            {"type": "answer", "raw_id": "protector", "description": "Mentor protects"},
        )
        # Link dilemma to answer (add_edge takes edge_type, from_id, to_id)
        graph.add_edge(
            "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::protector"
        )

        output = {
            # Completeness: decisions for all entities
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
            ],
            # Completeness: decisions for all dilemmas
            "dilemmas": [
                {"dilemma_id": "mentor_trust", "explored": ["protector"], "unexplored": []},
            ],
            # Path must be in SEED output for beat path references to validate
            "paths": [
                {
                    "path_id": "path_mentor_trust",
                    "name": "Mentor Trust Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "protector",
                    "description": "The mentor trust path",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening_001",
                    "summary": "Kay meets the mentor for the first time",
                    "paths": ["path_mentor_trust"],  # Raw path IDs from LLM
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "advances", "note": "Trust begins"}
                    ],
                    "entities": ["kay", "mentor"],  # Raw entity IDs from LLM
                    "location": "archive",  # Raw location ID from LLM
                },
                {
                    "beat_id": "resolution_001",
                    "summary": "Mentor's loyalty confirmed",
                    "paths": ["path_mentor_trust"],
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked in"}
                    ],
                    "entities": ["kay", "mentor"],
                    "location": "archive",
                },
            ],
        }

        apply_seed_mutations(graph, output)

        # Beat ID is prefixed with "beat::"
        beat = graph.get_node("beat::opening_001")
        assert beat is not None
        assert beat["type"] == "beat"
        assert beat["raw_id"] == "opening_001"
        assert beat["summary"] == "Kay meets the mentor for the first time"
        # Entities and location are prefixed in storage
        assert beat["entities"] == ["entity::kay", "entity::mentor"]
        assert beat["location"] == "entity::archive"

        # Check belongs_to edge - links to prefixed path ID
        edges = graph.get_edges(from_id="beat::opening_001", edge_type="belongs_to")
        assert len(edges) == 1
        assert edges[0]["to"] == "path::path_mentor_trust"

    def test_temporal_hint_stored_on_beat(self) -> None:
        """Temporal hint is stored on beat node with prefixed dilemma ID."""
        graph = Graph.empty()
        graph.create_node(
            "entity::kay", {"type": "entity", "raw_id": "kay", "concept": "Archivist"}
        )
        graph.create_node(
            "dilemma::mentor_trust",
            {"type": "dilemma", "raw_id": "mentor_trust", "question": "Trust?"},
        )
        # fight_or_flee: referenced by temporal_hint, must also be a brainstorm dilemma
        graph.create_node(
            "dilemma::fight_or_flee",
            {"type": "dilemma", "raw_id": "fight_or_flee", "question": "Fight?"},
        )
        graph.create_node(
            "dilemma::mentor_trust::alt::protector",
            {"type": "answer", "raw_id": "protector", "description": "Protects"},
        )
        graph.add_edge(
            "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::protector"
        )
        graph.create_node(
            "dilemma::fight_or_flee::alt::fight",
            {"type": "answer", "raw_id": "fight", "description": "Fights"},
        )
        graph.add_edge("has_answer", "dilemma::fight_or_flee", "dilemma::fight_or_flee::alt::fight")

        output = {
            "entities": [{"entity_id": "kay", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "mentor_trust", "explored": ["protector"], "unexplored": []},
                {"dilemma_id": "fight_or_flee", "explored": ["fight"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "path_mentor_trust",
                    "name": "Trust Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "protector",
                    "description": "The mentor trust path",
                },
                {
                    "path_id": "path_fight",
                    "name": "Fight Arc",
                    "dilemma_id": "fight_or_flee",
                    "answer_id": "fight",
                    "description": "The fight path",
                },
            ],
            "initial_beats": [
                {
                    "beat_id": "opening_001",
                    "summary": "Kay meets the mentor",
                    "paths": ["path_mentor_trust"],
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked"}
                    ],
                    "entities": ["kay"],
                    "temporal_hint": {
                        "relative_to": "fight_or_flee",
                        "position": "before_commit",
                    },
                },
                {
                    "beat_id": "fight_001",
                    "summary": "Kay prepares for battle",
                    "paths": ["path_fight"],
                    "dilemma_impacts": [
                        {"dilemma_id": "fight_or_flee", "effect": "commits", "note": "Engaged"}
                    ],
                    "entities": ["kay"],
                },
            ],
        }

        apply_seed_mutations(graph, output)

        beat = graph.get_node("beat::opening_001")
        assert beat["temporal_hint"] == {
            "relative_to": "dilemma::fight_or_flee",
            "position": "before_commit",
        }

    def test_temporal_hint_absent_not_stored(self) -> None:
        """Beat without temporal_hint does not have the key in node data."""
        graph = Graph.empty()
        graph.create_node(
            "entity::kay", {"type": "entity", "raw_id": "kay", "concept": "Archivist"}
        )
        graph.create_node(
            "dilemma::mentor_trust",
            {"type": "dilemma", "raw_id": "mentor_trust", "question": "Trust?"},
        )
        graph.create_node(
            "dilemma::mentor_trust::alt::protector",
            {"type": "answer", "raw_id": "protector", "description": "Protects"},
        )
        graph.add_edge(
            "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::protector"
        )

        output = {
            "entities": [{"entity_id": "kay", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "mentor_trust", "explored": ["protector"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "path_mentor_trust",
                    "name": "Trust Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "protector",
                    "description": "The mentor trust path",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening_001",
                    "summary": "Kay meets the mentor",
                    "paths": ["path_mentor_trust"],
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked"}
                    ],
                    "entities": ["kay"],
                },
            ],
        }

        apply_seed_mutations(graph, output)

        beat = graph.get_node("beat::opening_001")
        # _clean_dict removes None values, so temporal_hint should not be present
        assert "temporal_hint" not in beat

    def test_temporal_hint_partial_not_stored(self) -> None:
        """Temporal hint with null position is discarded, not stored malformed."""
        graph = Graph.empty()
        graph.create_node(
            "entity::kay", {"type": "entity", "raw_id": "kay", "concept": "Archivist"}
        )
        graph.create_node(
            "dilemma::mentor_trust",
            {"type": "dilemma", "raw_id": "mentor_trust", "question": "Trust?"},
        )
        graph.create_node(
            "dilemma::mentor_trust::alt::protector",
            {"type": "answer", "raw_id": "protector", "description": "Protects"},
        )
        graph.add_edge(
            "has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::protector"
        )

        output = {
            "entities": [{"entity_id": "kay", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "mentor_trust", "explored": ["protector"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "path_mentor_trust",
                    "name": "Trust Arc",
                    "dilemma_id": "mentor_trust",
                    "answer_id": "protector",
                    "description": "The mentor trust path",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening_001",
                    "summary": "Kay meets the mentor",
                    "paths": ["path_mentor_trust"],
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked"}
                    ],
                    "entities": ["kay"],
                    "temporal_hint": {
                        "relative_to": "fight_or_flee",
                        "position": None,
                    },
                },
            ],
        }

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

    def test_handles_empty_seed(self) -> None:
        """Handles empty seed output."""
        graph = Graph.empty()
        output = {"entities": [], "paths": [], "initial_beats": []}

        apply_seed_mutations(graph, output)
        # No errors, no changes


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
                "beat_id": "loyalty_commit",
                "summary": "Loyalty locked",
                "paths": ["loyalty_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "loyalty", "effect": "commits", "note": "Locked"}
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
                "beat_id": "loyalty_commit",
                "summary": "Loyalty locked",
                "paths": ["loyalty_arc"],
                "dilemma_impacts": [
                    {"dilemma_id": "loyalty", "effect": "commits", "note": "Locked"}
                ],
            },
        ]

        errors = validate_seed_mutations(graph, output)

        assert _blocking_errors(errors) == []


class TestSeedArcStructureValidation:
    """Test SEED validation checks 14-15: arc structure warnings.

    Doc 1 Part 2: "This scaffold must be complete — the arc from beginning
    to end must be present." Checks:
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
        assert warnings[0].field_path == "paths.trust_arc.arc_structure"

    def test_missing_post_commit_beat_warns(self) -> None:
        """Commit without any following beat produces a warning."""
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

        warnings = [e for e in errors if e.category == SeedErrorCategory.WARNING]
        blocking = _blocking_errors(errors)
        assert blocking == []
        assert len(warnings) == 1
        assert "after" in warnings[0].issue
        assert "consequence" in warnings[0].issue
        assert warnings[0].field_path == "paths.trust_arc.arc_structure"

    def test_single_commit_beat_produces_both_warnings(self) -> None:
        """A path with only a commit beat gets both pre and post warnings."""
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
        assert blocking == []
        assert len(warnings) == 2
        pre_warn = [w for w in warnings if "before" in w.issue]
        post_warn = [w for w in warnings if "after" in w.issue]
        assert len(pre_warn) == 1
        assert len(post_warn) == 1

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
        """apply_seed_mutations succeeds with warnings (logs them, doesn't raise)."""
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

        # Should not raise despite both arc structure warnings
        apply_seed_mutations(graph, output)

        # Verify mutation was applied (beat node exists)
        beat_node = graph.get_node("beat::commit")
        assert beat_node is not None

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
        }
        apply_mutations(graph, "dream", dream_output)
        graph.set_last_stage("dream")

        # BRAINSTORM stage
        brainstorm_output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Young archivist"},
                {
                    "entity_id": "mentor",
                    "entity_category": "character",
                    "concept": "Senior archivist",
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
        seed_output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "retained"},
            ],
            # Completeness: decisions for all dilemmas
            "dilemmas": [
                {"dilemma_id": "mentor_trust", "explored": ["protector"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "path_mentor",
                    "name": "Mentor Arc",
                    "dilemma_id": "mentor_trust",  # Raw dilemma ID
                    "answer_id": "protector",  # Local alt ID
                    "description": "The mentor relationship path",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Kay meets the mentor",
                    "paths": ["path_mentor"],  # Raw path IDs
                    "dilemma_impacts": [
                        {"dilemma_id": "mentor_trust", "effect": "commits", "note": "Locked in"}
                    ],
                }
            ],
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
        assert graph.has_node("path::path_mentor")
        assert graph.has_node("beat::opening")

        # Check entity dispositions
        assert graph.get_node("character::kay")["disposition"] == "retained"

        # Check edges
        assert len(graph.get_edges(edge_type="has_answer")) == 2
        assert len(graph.get_edges(edge_type="explores")) == 1
        assert len(graph.get_edges(edge_type="belongs_to")) == 1

        # Check node counts by type
        assert len(graph.get_nodes_by_type("vision")) == 1
        assert len(graph.get_nodes_by_type("entity")) == 2
        assert len(graph.get_nodes_by_type("dilemma")) == 1
        assert len(graph.get_nodes_by_type("answer")) == 2
        assert len(graph.get_nodes_by_type("path")) == 1
        assert len(graph.get_nodes_by_type("beat")) == 1


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
                }
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
                }
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
                }
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
                }
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
                }
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
        """Backfill fixes data before validation, preventing 11c errors."""
        graph = Graph.empty()
        # Create dilemma node
        graph.create_node(
            "dilemma::choice_a_or_b",
            {
                "type": "dilemma",
                "raw_id": "choice_a_or_b",
            },
        )
        # Create answer nodes
        graph.create_node(
            "dilemma::choice_a_or_b::alt::option_a",
            {"type": "answer", "raw_id": "option_a", "is_canonical": True},
        )
        graph.create_node(
            "dilemma::choice_a_or_b::alt::option_b",
            {"type": "answer", "raw_id": "option_b", "is_canonical": False},
        )
        # Create has_answer edges (required for validation)
        graph.add_edge(
            "has_answer",
            "dilemma::choice_a_or_b",
            "dilemma::choice_a_or_b::alt::option_a",
        )
        graph.add_edge(
            "has_answer",
            "dilemma::choice_a_or_b",
            "dilemma::choice_a_or_b::alt::option_b",
        )

        # Legacy data pattern: paths exist but explored is empty
        output = {
            "entities": [],
            "dilemmas": [
                {
                    "dilemma_id": "choice_a_or_b",
                    "explored": [],
                },  # empty explored - should be backfilled
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
            "consequences": [],
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

        # Should NOT raise because backfill fixes the data before validation
        apply_seed_mutations(graph, output)

        # Verify the dilemma node was updated with backfilled explored
        dilemma_node = graph.get_node("dilemma::choice_a_or_b")
        assert dilemma_node is not None
        assert sorted(dilemma_node.get("explored", [])) == ["option_a", "option_b"]


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
        """Build a graph with two brainstorm dilemmas and entities."""
        graph = Graph.empty()
        graph.create_node(
            "entity::kay",
            {"type": "entity", "raw_id": "kay", "entity_type": "character"},
        )
        graph.create_node(
            "dilemma::trust_or_not",
            {
                "type": "dilemma",
                "raw_id": "trust_or_not",
                "question": "Trust the mentor?",
            },
        )
        graph.add_edge("anchored_to", "dilemma::trust_or_not", "entity::kay")
        graph.create_node(
            "dilemma::trust_or_not::alt::trust",
            {"type": "answer", "raw_id": "trust", "is_canonical": True},
        )
        graph.add_edge("has_answer", "dilemma::trust_or_not", "dilemma::trust_or_not::alt::trust")
        graph.create_node(
            "dilemma::stay_or_go",
            {
                "type": "dilemma",
                "raw_id": "stay_or_go",
                "question": "Stay or leave?",
            },
        )
        graph.add_edge("anchored_to", "dilemma::stay_or_go", "entity::kay")
        graph.create_node(
            "dilemma::stay_or_go::alt::stay",
            {"type": "answer", "raw_id": "stay", "is_canonical": True},
        )
        graph.add_edge("has_answer", "dilemma::stay_or_go", "dilemma::stay_or_go::alt::stay")
        return graph

    def _base_output(self) -> dict:
        """Minimal SEED output dict with two dilemmas, paths, and beats."""
        return {
            "entities": [{"entity_id": "kay", "disposition": "retained"}],
            "dilemmas": [
                {"dilemma_id": "trust_or_not", "explored": ["trust"], "unexplored": []},
                {"dilemma_id": "stay_or_go", "explored": ["stay"], "unexplored": []},
            ],
            "paths": [
                {
                    "path_id": "trust_or_not__trust",
                    "name": "Trust Path",
                    "dilemma_id": "trust_or_not",
                    "answer_id": "trust",
                    "path_importance": "major",
                    "description": "Trust the mentor",
                },
                {
                    "path_id": "stay_or_go__stay",
                    "name": "Stay Path",
                    "dilemma_id": "stay_or_go",
                    "answer_id": "stay",
                    "path_importance": "major",
                    "description": "Stay at the location",
                },
            ],
            "consequences": [],
            "initial_beats": [
                {
                    "beat_id": "b_trust",
                    "summary": "Commit trust",
                    "paths": ["trust_or_not__trust"],
                    "dilemma_impacts": [{"dilemma_id": "trust_or_not", "effect": "commits"}],
                },
                {
                    "beat_id": "b_stay",
                    "summary": "Commit stay",
                    "paths": ["stay_or_go__stay"],
                    "dilemma_impacts": [{"dilemma_id": "stay_or_go", "effect": "commits"}],
                },
            ],
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

    def test_absent_ending_salience_not_stored(self) -> None:
        """When ending_salience is absent from analysis dict, it is not added."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "test",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        assert "ending_salience" not in node

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

    def test_absent_residue_weight_not_stored(self) -> None:
        """When residue_weight is absent from analysis dict, it is not added."""
        graph = self._graph_with_dilemmas()
        output = self._base_output()
        output["dilemma_analyses"] = [
            {
                "dilemma_id": "trust_or_not",
                "dilemma_role": "hard",
                "payoff_budget": 4,
                "reasoning": "test",
            },
        ]
        apply_seed_mutations(graph, output)

        node = graph.get_node("dilemma::trust_or_not")
        assert "residue_weight" not in node
