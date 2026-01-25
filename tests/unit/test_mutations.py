"""Tests for stage mutation appliers."""

from __future__ import annotations

import pytest

from questfoundry.graph import Graph, MutationError, SeedMutationError
from questfoundry.graph.mutations import (
    BrainstormMutationError,
    BrainstormValidationError,
    SeedErrorCategory,
    SeedValidationError,
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
                "tension", "host_motivation", "tension::host_motivation", id="raw_id-tension"
            ),
            pytest.param("thread", "main_thread", "thread::main_thread", id="raw_id-thread"),
            pytest.param(
                "entity",
                "entity::the_detective",
                "entity::the_detective",
                id="correctly_prefixed-entity",
            ),
            pytest.param(
                "tension",
                "tension::host_motivation",
                "tension::host_motivation",
                id="correctly_prefixed-tension",
            ),
            pytest.param(
                "entity",
                "tension::the_detective",
                "entity::the_detective",
                id="wrong_prefix-entity",
            ),
            pytest.param(
                "tension",
                "entity::host_motivation",
                "tension::host_motivation",
                id="wrong_prefix-tension",
            ),
            pytest.param(
                "tension",
                "tension::tension::host_motivation",
                "tension::host_motivation",
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
            "tensions": [],
        }

        apply_mutations(graph, "brainstorm", output)

        assert graph.has_node("entity::char_001")

    def test_routes_to_seed(self) -> None:
        """Routes seed stage to apply_seed_mutations."""
        graph = Graph.empty()
        # Pre-populate with entity from brainstorm (using prefixed ID + raw_id for validation)
        graph.create_node(
            "entity::char_001",
            {"type": "entity", "raw_id": "char_001", "disposition": "proposed"},
        )

        output = {
            "entities": [{"entity_id": "char_001", "disposition": "retained"}],
            "threads": [],
            "initial_beats": [],
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


class TestBrainstormMutations:
    """Test BRAINSTORM stage mutations."""

    def test_entity_missing_id_raises(self) -> None:
        """Raises MutationError when entity missing entity_id."""
        graph = Graph.empty()
        output = {
            "entities": [{"entity_category": "character", "concept": "Test"}],  # Missing entity_id
            "tensions": [],
        }

        with pytest.raises(
            MutationError, match="Entity at index 0 missing required 'entity_id' field"
        ):
            apply_brainstorm_mutations(graph, output)

    def test_tension_missing_id_raises(self) -> None:
        """Raises MutationError when tension missing tension_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "tensions": [{"question": "Test?", "alternatives": []}],  # Missing tension_id
        }

        with pytest.raises(
            MutationError, match="Tension at index 0 missing required 'tension_id' field"
        ):
            apply_brainstorm_mutations(graph, output)

    def test_alternative_missing_id_raises(self) -> None:
        """Raises MutationError when alternative missing alternative_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "tensions": [
                {
                    "tension_id": "tension_001",
                    "question": "Test?",
                    "alternatives": [
                        {"description": "Option A", "is_default_path": True}
                    ],  # Missing alternative_id
                }
            ],
        }

        with pytest.raises(
            MutationError,
            match="Alternative at index 0 in tension 'tension_001' missing required 'alternative_id' field",
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
            "tensions": [],
        }

        apply_brainstorm_mutations(graph, output)

        # Entity IDs are prefixed with "entity::"
        kay = graph.get_node("entity::kay")
        assert kay is not None
        assert kay["type"] == "entity"
        assert kay["raw_id"] == "kay"  # Original ID preserved
        assert kay["entity_type"] == "character"
        assert kay["concept"] == "Young archivist"
        assert kay["notes"] == "Curious and brave"
        assert kay["disposition"] == "proposed"

        archive = graph.get_node("entity::archive")
        assert archive is not None
        assert archive["entity_type"] == "location"

    def test_creates_tension_with_alternatives(self) -> None:
        """Creates tension nodes with linked alternatives."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "tensions": [
                {
                    "tension_id": "mentor_trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "mentor"],  # Raw IDs from LLM
                    "why_it_matters": "Trust is key",
                    "alternatives": [
                        {
                            "alternative_id": "protector",
                            "description": "Mentor protects Kay",
                            "is_default_path": True,
                        },
                        {
                            "alternative_id": "manipulator",
                            "description": "Mentor manipulates Kay",
                            "is_default_path": False,
                        },
                    ],
                }
            ],
        }

        apply_brainstorm_mutations(graph, output)

        # Tension IDs are prefixed with "tension::"
        tension = graph.get_node("tension::mentor_trust")
        assert tension is not None
        assert tension["type"] == "tension"
        assert tension["raw_id"] == "mentor_trust"
        assert tension["question"] == "Can the mentor be trusted?"
        # central_entity_ids list is prefixed in storage
        assert tension["central_entity_ids"] == ["entity::kay", "entity::mentor"]

        # Alternative IDs: tension::tension_id::alt::alt_id
        protector = graph.get_node("tension::mentor_trust::alt::protector")
        assert protector is not None
        assert protector["type"] == "alternative"
        assert protector["raw_id"] == "protector"
        assert protector["is_default_path"] is True

        manipulator = graph.get_node("tension::mentor_trust::alt::manipulator")
        assert manipulator is not None
        assert manipulator["is_default_path"] is False

        # Check edges
        edges = graph.get_edges(from_id="tension::mentor_trust", edge_type="has_alternative")
        assert len(edges) == 2

    def test_handles_empty_brainstorm(self) -> None:
        """Handles empty entities and tensions."""
        graph = Graph.empty()
        output = {"entities": [], "tensions": []}

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
            "tensions": [
                {
                    "tension_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "mentor"],
                    "alternatives": [
                        {"alternative_id": "yes", "description": "Yes", "is_default_path": True},
                        {"alternative_id": "no", "description": "No", "is_default_path": False},
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
            "tensions": [
                {
                    "tension_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "phantom_entity"],  # phantom_entity doesn't exist
                    "alternatives": [
                        {"alternative_id": "yes", "description": "Yes", "is_default_path": True},
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
        """Detects duplicate alternative IDs within a tension."""
        output = {
            "entities": [],
            "tensions": [
                {
                    "tension_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],
                    "alternatives": [
                        {"alternative_id": "option_a", "description": "A", "is_default_path": True},
                        {
                            "alternative_id": "option_a",
                            "description": "B",
                            "is_default_path": False,
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
        """Detects when no alternative has is_default_path=True."""
        output = {
            "entities": [],
            "tensions": [
                {
                    "tension_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],
                    "alternatives": [
                        {"alternative_id": "yes", "description": "Yes", "is_default_path": False},
                        {"alternative_id": "no", "description": "No", "is_default_path": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        assert "No alternative has is_default_path=true" in errors[0].issue

    def test_multiple_default_paths_detected(self) -> None:
        """Detects when multiple alternatives have is_default_path=True."""
        output = {
            "entities": [],
            "tensions": [
                {
                    "tension_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],
                    "alternatives": [
                        {"alternative_id": "yes", "description": "Yes", "is_default_path": True},
                        {"alternative_id": "no", "description": "No", "is_default_path": True},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        assert "Multiple alternatives have is_default_path=true" in errors[0].issue

    def test_multiple_errors_collected(self) -> None:
        """Multiple errors across different validations are all collected."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
            ],
            "tensions": [
                {
                    "tension_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["phantom1", "phantom2"],  # Both invalid
                    "alternatives": [
                        {"alternative_id": "yes", "description": "Yes", "is_default_path": False},
                        {"alternative_id": "no", "description": "No", "is_default_path": False},
                    ],
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        # Should find: 2 phantom entity errors + 1 no default error
        assert len(errors) == 3

    def test_empty_tensions_valid(self) -> None:
        """Empty tensions list is valid."""
        output = {
            "entities": [
                {"entity_id": "kay", "entity_category": "character", "concept": "Archivist"},
            ],
            "tensions": [],
        }

        errors = validate_brainstorm_mutations(output)

        assert errors == []

    def test_empty_alternatives_detected(self) -> None:
        """Tension with no alternatives fails default path validation."""
        output = {
            "entities": [],
            "tensions": [
                {
                    "tension_id": "trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": [],
                    "alternatives": [],  # No alternatives at all
                }
            ],
        }

        errors = validate_brainstorm_mutations(output)

        assert len(errors) == 1
        assert "No alternative has is_default_path=true" in errors[0].issue

    def test_empty_entity_id_detected(self) -> None:
        """Detects empty or missing entity_id values."""
        output = {
            "entities": [
                {"entity_id": "valid_id", "entity_category": "character", "concept": "Test"},
                {"entity_id": "", "entity_category": "character", "concept": "Empty ID"},
                {"entity_id": None, "entity_category": "location", "concept": "None ID"},
                {"entity_category": "object", "concept": "Missing ID"},  # No entity_id key
            ],
            "tensions": [],
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


class TestBrainstormMutationError:
    """Test BrainstormMutationError formatting."""

    def test_to_feedback_includes_all_error_info(self) -> None:
        """to_feedback() includes error details for LLM retry."""
        errors = [
            BrainstormValidationError(
                field_path="tensions.0.central_entity_ids",
                issue="Entity 'phantom' not in entities list",
                available=["kay", "mentor"],
                provided="phantom",
            )
        ]
        error = BrainstormMutationError(errors)

        feedback = error.to_feedback()

        assert "BRAINSTORM has invalid internal references" in feedback
        assert "tensions.0.central_entity_ids" in feedback
        assert "Entity 'phantom' not in entities list" in feedback
        assert "kay" in feedback
        assert "mentor" in feedback

    def test_error_limit_applied(self) -> None:
        """Only shows first 8 errors plus count of remaining."""
        errors = [
            BrainstormValidationError(
                field_path=f"tensions.{i}.central_entity_ids",
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
            "threads": [],
            "initial_beats": [],
        }

        with pytest.raises(
            MutationError, match="Entity decision at index 0 missing required 'entity_id' field"
        ):
            apply_seed_mutations(graph, output)

    def test_thread_missing_id_raises(self) -> None:
        """Raises MutationError when thread missing thread_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "threads": [{"name": "Test Thread"}],  # Missing thread_id
            "initial_beats": [],
        }

        with pytest.raises(
            MutationError, match="Thread at index 0 missing required 'thread_id' field"
        ):
            apply_seed_mutations(graph, output)

    def test_beat_missing_id_raises(self) -> None:
        """Raises MutationError when beat missing beat_id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "threads": [],
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
            "threads": [],
            "initial_beats": [],
        }

        apply_seed_mutations(graph, output)

        assert graph.get_node("entity::kay")["disposition"] == "retained"
        assert graph.get_node("entity::mentor")["disposition"] == "retained"
        assert graph.get_node("entity::extra")["disposition"] == "cut"

    def test_creates_threads(self) -> None:
        """Creates thread nodes from seed output."""
        graph = Graph.empty()
        # Pre-populate tension and alternative from brainstorm (with raw_id for validation)
        graph.create_node(
            "tension::mentor_trust",
            {"type": "tension", "raw_id": "mentor_trust", "question": "Can the mentor be trusted?"},
        )
        graph.create_node(
            "tension::mentor_trust::alt::protector",
            {"type": "alternative", "raw_id": "protector", "description": "Mentor protects"},
        )
        # Link tension to alternative (add_edge takes edge_type, from_id, to_id)
        graph.add_edge(
            "has_alternative", "tension::mentor_trust", "tension::mentor_trust::alt::protector"
        )

        output = {
            "entities": [],
            "tensions": [
                {"tension_id": "mentor_trust", "explored": ["protector"], "implicit": []},
            ],
            "threads": [
                {
                    "thread_id": "thread_mentor_trust",
                    "name": "Mentor Trust Arc",
                    "tension_id": "mentor_trust",  # Raw tension ID from LLM
                    "alternative_id": "protector",  # Local alt ID, not full path
                    "description": "Exploring mentor relationship",
                    "consequence_ids": ["consequence_trust"],
                }
            ],
            "initial_beats": [],
        }

        apply_seed_mutations(graph, output)

        # Thread ID is prefixed with "thread::"
        thread = graph.get_node("thread::thread_mentor_trust")
        assert thread is not None
        assert thread["type"] == "thread"
        assert thread["raw_id"] == "thread_mentor_trust"
        assert thread["name"] == "Mentor Trust Arc"

        # Check explores edge - links to full prefixed alternative ID
        edges = graph.get_edges(from_id="thread::thread_mentor_trust", edge_type="explores")
        assert len(edges) == 1
        assert edges[0]["to"] == "tension::mentor_trust::alt::protector"

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
        # Pre-populate tension and alternative from brainstorm (for thread validation)
        graph.create_node(
            "tension::mentor_trust",
            {"type": "tension", "raw_id": "mentor_trust", "question": "Can the mentor be trusted?"},
        )
        graph.create_node(
            "tension::mentor_trust::alt::protector",
            {"type": "alternative", "raw_id": "protector", "description": "Mentor protects"},
        )
        # Link tension to alternative (add_edge takes edge_type, from_id, to_id)
        graph.add_edge(
            "has_alternative", "tension::mentor_trust", "tension::mentor_trust::alt::protector"
        )

        output = {
            # Completeness: decisions for all entities
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "retained"},
                {"entity_id": "archive", "disposition": "retained"},
            ],
            # Completeness: decisions for all tensions
            "tensions": [
                {"tension_id": "mentor_trust", "explored": ["protector"], "implicit": []},
            ],
            # Thread must be in SEED output for beat thread references to validate
            "threads": [
                {
                    "thread_id": "thread_mentor_trust",
                    "name": "Mentor Trust Arc",
                    "tension_id": "mentor_trust",
                    "alternative_id": "protector",
                    "description": "The mentor trust thread",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening_001",
                    "summary": "Kay meets the mentor for the first time",
                    "threads": ["thread_mentor_trust"],  # Raw thread IDs from LLM
                    "tension_impacts": [
                        {"tension_id": "mentor_trust", "effect": "advances", "note": "Trust begins"}
                    ],
                    "entities": ["kay", "mentor"],  # Raw entity IDs from LLM
                    "location": "archive",  # Raw location ID from LLM
                }
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

        # Check belongs_to edge - links to prefixed thread ID
        edges = graph.get_edges(from_id="beat::opening_001", edge_type="belongs_to")
        assert len(edges) == 1
        assert edges[0]["to"] == "thread::thread_mentor_trust"

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
            "threads": [],
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
        output = {"entities": [], "threads": [], "initial_beats": []}

        apply_seed_mutations(graph, output)
        # No errors, no changes


class TestSeedCompletenessValidation:
    """Test SEED completeness validation (all entities/tensions have decisions)."""

    def test_complete_decisions_valid(self) -> None:
        """All entities and tensions have decisions passes validation."""
        graph = Graph.empty()
        # Add entities from BRAINSTORM
        graph.create_node("entity::kay", {"type": "entity", "raw_id": "kay"})
        graph.create_node("entity::mentor", {"type": "entity", "raw_id": "mentor"})
        # Add tension from BRAINSTORM
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "mentor", "disposition": "cut"},
            ],
            "tensions": [
                {"tension_id": "trust", "explored": ["yes"], "implicit": []},
            ],
            "threads": [
                {
                    "thread_id": "trust_arc",
                    "name": "Trust Arc",
                    "tension_id": "trust",
                    "alternative_id": "yes",
                }
            ],
            "initial_beats": [],
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
            "tensions": [],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find 2 missing entity decisions with category-specific messages
        entity_errors = [
            e for e in errors if "Missing decision for" in e.issue and "tension" not in e.issue
        ]
        assert len(entity_errors) == 2
        missing_ids = {e.issue.split("'")[1] for e in entity_errors}
        assert missing_ids == {"mentor", "archive"}
        # Verify category is included in error message
        assert any("character" in e.issue for e in entity_errors)
        assert any("location" in e.issue for e in entity_errors)

    def test_missing_tension_decision_detected(self) -> None:
        """Detects when tension from BRAINSTORM has no decision in SEED."""
        graph = Graph.empty()
        # Add tensions from BRAINSTORM
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::loyalty", {"type": "tension", "raw_id": "loyalty"})

        output = {
            "entities": [],
            "tensions": [
                {"tension_id": "trust", "explored": [], "implicit": []},
                # Missing: loyalty
            ],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find 1 missing tension decision
        tension_errors = [e for e in errors if "Missing decision for tension" in e.issue]
        assert len(tension_errors) == 1
        assert "loyalty" in tension_errors[0].issue

    def test_both_entity_and_tension_missing_detected(self) -> None:
        """Detects missing decisions for both entities and tensions."""
        graph = Graph.empty()
        # Add entity and tension from BRAINSTORM
        graph.create_node(
            "entity::kay", {"type": "entity", "raw_id": "kay", "entity_type": "character"}
        )
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})

        output = {
            "entities": [],  # Missing kay
            "tensions": [],  # Missing trust
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find both missing entity and missing tension
        # Entity errors use category-specific messages (e.g., "Missing decision for character")
        entity_errors = [
            e for e in errors if "Missing decision for" in e.issue and "tension" not in e.issue
        ]
        tension_errors = [e for e in errors if "Missing decision for tension" in e.issue]
        assert len(entity_errors) == 1
        assert len(tension_errors) == 1

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
            "tensions": [],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find 1 missing entity decision with "faction" in message
        assert len(errors) == 1
        assert "Missing decision for faction 'the_family'" in errors[0].issue

    def test_tension_without_thread_detected(self) -> None:
        """Detects when a tension has no corresponding thread."""
        graph = Graph.empty()
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::loyalty", {"type": "tension", "raw_id": "loyalty"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [],
            "tensions": [
                {"tension_id": "trust", "explored": ["yes"], "implicit": []},
                {"tension_id": "loyalty", "explored": [], "implicit": []},
            ],
            "threads": [
                {
                    "thread_id": "trust_arc",
                    "name": "Trust Arc",
                    "tension_id": "trust",
                    "alternative_id": "yes",
                }
                # Missing: no thread for loyalty
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        thread_errors = [e for e in errors if "has no thread" in e.issue]
        assert len(thread_errors) == 1
        assert "loyalty" in thread_errors[0].issue
        assert thread_errors[0].category == SeedErrorCategory.COMPLETENESS

    def test_all_tensions_with_threads_valid(self) -> None:
        """All tensions having threads passes thread completeness check."""
        graph = Graph.empty()
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::loyalty", {"type": "tension", "raw_id": "loyalty"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.create_node(
            "tension::loyalty::alt::stand", {"type": "alternative", "raw_id": "stand"}
        )
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")
        graph.add_edge("has_alternative", "tension::loyalty", "tension::loyalty::alt::stand")

        output = {
            "entities": [],
            "tensions": [
                {"tension_id": "trust", "explored": ["yes"], "implicit": []},
                {"tension_id": "loyalty", "explored": ["stand"], "implicit": []},
            ],
            "threads": [
                {
                    "thread_id": "trust_arc",
                    "name": "Trust Arc",
                    "tension_id": "trust",
                    "alternative_id": "yes",
                },
                {
                    "thread_id": "loyalty_arc",
                    "name": "Loyalty Arc",
                    "tension_id": "loyalty",
                    "alternative_id": "stand",
                },
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        thread_errors = [e for e in errors if "has no thread" in e.issue]
        assert thread_errors == []

    def test_scoped_tension_id_in_thread_satisfies_completeness(self) -> None:
        """Scoped tension_id (tension::trust) in thread satisfies completeness."""
        graph = Graph.empty()
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [],
            "tensions": [{"tension_id": "trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "trust_arc",
                    "name": "Trust Arc",
                    "tension_id": "tension::trust",  # Scoped
                    "alternative_id": "yes",
                }
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        thread_errors = [e for e in errors if "has no thread" in e.issue]
        assert thread_errors == []

    def test_empty_brainstorm_valid(self) -> None:
        """Empty BRAINSTORM data (no entities/tensions) is valid."""
        graph = Graph.empty()
        # No entities or tensions in graph

        output = {
            "entities": [],
            "tensions": [],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_extra_decisions_invalid(self) -> None:
        """Extra decisions for non-existent entities/tensions are caught."""
        graph = Graph.empty()
        # Only kay exists
        graph.create_node("entity::kay", {"type": "entity", "raw_id": "kay"})

        output = {
            "entities": [
                {"entity_id": "kay", "disposition": "retained"},
                {"entity_id": "nonexistent", "disposition": "retained"},  # Doesn't exist
            ],
            "tensions": [],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        # Should find invalid entity reference (existing check 1)
        invalid_errors = [e for e in errors if "not in BRAINSTORM" in e.issue]
        assert len(invalid_errors) == 1
        assert "nonexistent" in invalid_errors[0].provided


class TestSeedDuplicateValidation:
    """Test SEED validation detects duplicate entity/tension IDs.

    Fixes #239: LLM may output the same entity or tension multiple times,
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
            "tensions": [],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        dup_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(dup_errors) == 1
        assert "hero" in dup_errors[0].issue
        assert "2 times" in dup_errors[0].issue

    def test_duplicate_tension_id_detected(self) -> None:
        """Detects when the same tension_id appears multiple times in output."""
        graph = Graph.empty()
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [],
            "tensions": [
                {"tension_id": "trust", "explored": ["yes"], "implicit": []},
                {"tension_id": "trust", "explored": ["yes"], "implicit": []},  # Duplicate!
                {"tension_id": "trust", "explored": ["yes"], "implicit": []},  # Triple!
            ],
            "threads": [],
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
            "tensions": [],
            "threads": [],
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
            "tensions": [],
            "threads": [],
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
            "tensions": [
                {
                    "tension_id": "mentor_trust",
                    "question": "Can the mentor be trusted?",
                    "central_entity_ids": ["kay", "mentor"],  # Raw IDs from LLM
                    "why_it_matters": "Trust defines ally or foe",
                    "alternatives": [
                        {
                            "alternative_id": "protector",
                            "description": "Mentor protects",
                            "is_default_path": True,
                        },
                        {
                            "alternative_id": "manipulator",
                            "description": "Mentor manipulates",
                            "is_default_path": False,
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
            # Completeness: decisions for all tensions
            "tensions": [
                {"tension_id": "mentor_trust", "explored": ["protector"], "implicit": []},
            ],
            "threads": [
                {
                    "thread_id": "thread_mentor",
                    "name": "Mentor Arc",
                    "tension_id": "mentor_trust",  # Raw tension ID
                    "alternative_id": "protector",  # Local alt ID
                    "description": "The mentor relationship thread",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Kay meets the mentor",
                    "threads": ["thread_mentor"],  # Raw thread IDs
                }
            ],
        }
        apply_mutations(graph, "seed", seed_output)
        graph.set_last_stage("seed")

        # Verify final state - all IDs are prefixed by type
        assert graph.get_last_stage() == "seed"
        assert graph.has_node("vision")
        assert graph.has_node("entity::kay")
        assert graph.has_node("entity::mentor")
        assert graph.has_node("tension::mentor_trust")
        assert graph.has_node("tension::mentor_trust::alt::protector")
        assert graph.has_node("thread::thread_mentor")
        assert graph.has_node("beat::opening")

        # Check entity dispositions
        assert graph.get_node("entity::kay")["disposition"] == "retained"

        # Check edges
        assert len(graph.get_edges(edge_type="has_alternative")) == 2
        assert len(graph.get_edges(edge_type="explores")) == 1
        assert len(graph.get_edges(edge_type="belongs_to")) == 1

        # Check node counts by type
        assert len(graph.get_nodes_by_type("vision")) == 1
        assert len(graph.get_nodes_by_type("entity")) == 2
        assert len(graph.get_nodes_by_type("tension")) == 1
        assert len(graph.get_nodes_by_type("alternative")) == 2
        assert len(graph.get_nodes_by_type("thread")) == 1
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
            field_path="threads.0.tension_id",
            issue="Tension 'phantom' not in BRAINSTORM",
            available=["real_tension"],
            provided="phantom",
        )
        assert categorize_error(error) == SeedErrorCategory.SEMANTIC

    def test_semantic_not_in_seed(self) -> None:
        """'not defined in SEED' errors are SEMANTIC."""
        error = SeedValidationError(
            field_path="initial_beats.0.threads",
            issue="Thread 'ghost' not defined in SEED threads",
            available=["real_thread"],
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
            field_path="threads.0.name",
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
                field_path="threads.0.tension_id",
                issue="Tension 'x' not in BRAINSTORM",
            ),
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for character 'hero'",
            ),
            SeedValidationError(
                field_path="threads.1.tension_id",
                issue="Tension 'y' not in BRAINSTORM",
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
            "tensions": [],
            "threads": [],
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
            "tensions": [],
            "threads": [],
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

    def test_tension_scope_accepted(self) -> None:
        """Tension scope is handled correctly."""
        normalized, error = _normalize_id("tension::mentor_trust", "tension")
        assert normalized == "mentor_trust"
        assert error is None

    def test_thread_scope_accepted(self) -> None:
        """Thread scope is handled correctly."""
        normalized, error = _normalize_id("thread::mentor_arc", "thread")
        assert normalized == "mentor_arc"
        assert error is None

    def test_wrong_scope_returns_error(self) -> None:
        """Wrong scope prefix returns error message."""
        normalized, error = _normalize_id("tension::hero", "entity")
        # Returns original ID unchanged when scope is wrong
        assert normalized == "tension::hero"
        assert error is not None
        assert "entity::" in error
        assert "tension::" in error

    def test_entity_scope_rejected_when_thread_expected(self) -> None:
        """Entity scope rejected when thread is expected."""
        normalized, error = _normalize_id("entity::mentor_arc", "thread")
        assert normalized == "entity::mentor_arc"
        assert error is not None
        assert "thread::" in error
        assert "entity::" in error

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
            "tensions": [],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_scoped_tension_id_accepted(self) -> None:
        """Scoped tension IDs (tension::trust) are accepted in tension decisions."""
        graph = Graph.empty()
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [],
            "tensions": [{"tension_id": "tension::trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "trust_arc",
                    "name": "Trust Arc",
                    "tension_id": "tension::trust",
                    "alternative_id": "yes",
                }
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_scoped_thread_id_accepted_in_beats(self) -> None:
        """Scoped thread IDs (thread::mentor) are accepted in beat thread references."""
        graph = Graph.empty()
        # Set up BRAINSTORM data
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "tensions": [{"tension_id": "tension::trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "mentor",
                    "name": "Mentor Arc",
                    "tension_id": "tension::trust",
                    "alternative_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "The beginning",
                    "threads": ["thread::mentor"],  # Scoped thread ID
                    "entities": ["entity::hero"],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_wrong_scope_detected_for_entity(self) -> None:
        """Wrong scope (tension:: instead of entity::) is detected."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})

        output = {
            "entities": [{"entity_id": "tension::hero", "disposition": "retained"}],
            "tensions": [],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert len(errors) >= 1
        scope_errors = [e for e in errors if "Wrong scope prefix" in e.issue]
        assert len(scope_errors) == 1
        assert "entity::" in scope_errors[0].issue
        assert "tension::" in scope_errors[0].issue

    def test_wrong_scope_detected_for_tension(self) -> None:
        """Wrong scope (entity:: instead of tension::) is detected."""
        graph = Graph.empty()
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})

        output = {
            "entities": [],
            "tensions": [{"tension_id": "entity::trust", "explored": [], "implicit": []}],
            "threads": [],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert len(errors) >= 1
        scope_errors = [e for e in errors if "Wrong scope prefix" in e.issue]
        assert len(scope_errors) == 1
        assert "tension::" in scope_errors[0].issue
        assert "entity::" in scope_errors[0].issue

    def test_wrong_scope_detected_for_thread_in_beat(self) -> None:
        """Wrong scope (entity:: instead of thread::) in beat thread references."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "tensions": [{"tension_id": "tension::trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "mentor",
                    "name": "Mentor Arc",
                    "tension_id": "tension::trust",
                    "alternative_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "The beginning",
                    "threads": ["entity::mentor"],  # Wrong scope - should be thread::
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        scope_errors = [e for e in errors if "Wrong scope prefix" in e.issue]
        assert len(scope_errors) == 1
        assert "thread::" in scope_errors[0].issue

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
            "tensions": [],
            "threads": [],
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
            "tensions": [],
            "threads": [],
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
                {"entity_id": "tension::hero", "disposition": "retained"}  # Wrong scope
            ],
            "tensions": [],
            "threads": [],
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
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [
                {"entity_id": "entity::hero", "disposition": "retained"},
                {"entity_id": "entity::mentor", "disposition": "retained"},
            ],
            "tensions": [{"tension_id": "tension::trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "tension_id": "tension::trust",
                    "alternative_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Meet mentor",
                    "entities": ["entity::hero", "entity::mentor"],  # Scoped IDs
                    "location": "entity::hero",  # Scoped location
                    "threads": ["thread::mentor_arc"],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # No errors expected
        assert errors == []

    def test_scoped_tension_in_tension_impacts(self) -> None:
        """Scoped tension IDs work in beat.tension_impacts."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "tensions": [{"tension_id": "tension::trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "tension_id": "tension::trust",
                    "alternative_id": "yes",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Build trust",
                    "threads": ["thread::mentor_arc"],
                    "tension_impacts": [
                        {"tension_id": "tension::trust", "effect": "advances"}  # Scoped ID
                    ],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_scoped_thread_in_consequences(self) -> None:
        """Scoped thread IDs work in consequence.thread_id."""
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "tensions": [{"tension_id": "tension::trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "tension_id": "tension::trust",
                    "alternative_id": "yes",
                }
            ],
            "consequences": [
                {
                    "consequence_id": "trust_earned",
                    "thread_id": "thread::mentor_arc",  # Scoped thread ID
                    "description": "Trust is earned",
                }
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []

    def test_scoped_thread_definitions_and_consequences(self) -> None:
        """Thread definitions using scoped IDs work with scoped consequence references.

        Regression test for issue #230: When LLM outputs thread definitions with
        scoped IDs (thread::foo) and consequences reference them with scoped IDs,
        validation should pass since seed_thread_ids is normalized.
        """
        graph = Graph.empty()
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero"})
        graph.create_node("tension::trust", {"type": "tension", "raw_id": "trust"})
        graph.create_node("tension::trust::alt::yes", {"type": "alternative", "raw_id": "yes"})
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")

        output = {
            "entities": [{"entity_id": "entity::hero", "disposition": "retained"}],
            "tensions": [{"tension_id": "tension::trust", "explored": ["yes"], "implicit": []}],
            "threads": [
                {
                    "thread_id": "thread::mentor_arc",  # Scoped ID in definition
                    "name": "Mentor Arc",
                    "tension_id": "tension::trust",
                    "alternative_id": "yes",
                }
            ],
            "consequences": [
                {
                    "consequence_id": "trust_earned",
                    "thread_id": "thread::mentor_arc",  # Scoped ID in reference
                    "description": "Trust is earned",
                }
            ],
            "initial_beats": [],
        }

        errors = validate_seed_mutations(graph, output)

        assert errors == []


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
                field_path="threads.0.tension_id",
                issue="Tension 'the_archive' not in BRAINSTORM",
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
                field_path="tensions.0.central_entity_ids",
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
                field_path="tensions",
                issue="Missing decision for tension 'ancient_scroll'",
                available=[],
                provided="",
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Missing items" in result
        assert "hollow_key" in result
        assert "ancient_scroll" in result

    def test_formats_thread_completeness_errors(self) -> None:
        """Should format thread completeness errors in Missing threads section."""
        from questfoundry.graph.mutations import format_semantic_errors_as_content

        errors = [
            SeedValidationError(
                field_path="threads",
                issue="Tension 'loyalty' has no thread. Create at least one thread exploring this tension.",
                available=[],
                provided="",
                category=SeedErrorCategory.COMPLETENESS,
            ),
            SeedValidationError(
                field_path="threads",
                issue="Tension 'trust' has no thread. Create at least one thread exploring this tension.",
                available=[],
                provided="",
                category=SeedErrorCategory.COMPLETENESS,
            ),
        ]

        result = format_semantic_errors_as_content(errors)

        assert "Missing threads" in result
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
                field_path="threads.0.tension_id",
                issue="Tension 'unknown' not in BRAINSTORM",
                available=["main_tension"],
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


class TestTypeAwareFeedback:
    """Tests for type-aware cross-type error messages in validate_seed_mutations.

    When an ID is used as the wrong type (e.g., entity used as tension_id),
    the error message should indicate what type it actually is, rather than
    the generic "not in BRAINSTORM" message.
    """

    def test_type_aware_feedback_entity_as_tension(self) -> None:
        """Entity faction name used as tension_id gives helpful message."""
        graph = Graph.empty()
        # isolation_protocol is a faction entity in brainstorm
        graph.create_node(
            "entity::isolation_protocol",
            {"type": "entity", "raw_id": "isolation_protocol", "entity_type": "faction"},
        )
        graph.create_node(
            "tension::trust_or_betray",
            {"type": "tension", "raw_id": "trust_or_betray"},
        )

        output = {
            "entities": [
                {"entity_id": "isolation_protocol", "disposition": "retained"},
            ],
            "tensions": [
                {"tension_id": "trust_or_betray", "explored": [], "implicit": []},
            ],
            "threads": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "tension_impacts": [
                        # Using entity name as tension_id (the seq-9 bug)
                        {"tension_id": "isolation_protocol", "effect": "advances"}
                    ],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        # Should get a type-aware error, not generic "not in BRAINSTORM"
        tension_errors = [e for e in errors if "tension_impacts" in e.field_path]
        assert len(tension_errors) == 1
        assert "is an entity (faction), not a tension" in tension_errors[0].issue
        assert "subject_X_or_Y" in tension_errors[0].issue

    def test_type_aware_feedback_thread_as_tension(self) -> None:
        """Thread ID used as tension_id gives helpful message."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "tension::trust_or_betray",
            {"type": "tension", "raw_id": "trust_or_betray"},
        )
        graph.create_node(
            "tension::trust_or_betray::alt::trust",
            {"type": "alternative", "raw_id": "trust"},
        )
        graph.add_edge(
            "has_alternative",
            "tension::trust_or_betray",
            "tension::trust_or_betray::alt::trust",
        )

        output = {
            "entities": [{"entity_id": "hero", "disposition": "retained"}],
            "tensions": [
                {"tension_id": "trust_or_betray", "explored": ["trust"], "implicit": []},
            ],
            "threads": [
                {
                    "thread_id": "mentor_arc",
                    "name": "Mentor Arc",
                    "tension_id": "trust_or_betray",
                    "alternative_id": "trust",
                }
            ],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    "tension_impacts": [
                        # Using thread ID as tension_id
                        {"tension_id": "mentor_arc", "effect": "advances"}
                    ],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        tension_errors = [e for e in errors if "tension_impacts" in e.field_path]
        assert len(tension_errors) == 1
        assert "is a thread ID, not a tension" in tension_errors[0].issue

    def test_type_aware_feedback_tension_as_entity(self) -> None:
        """Tension ID used as entity in beat gives helpful message."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "tension::trust_or_betray",
            {"type": "tension", "raw_id": "trust_or_betray"},
        )

        output = {
            "entities": [{"entity_id": "hero", "disposition": "retained"}],
            "tensions": [
                {"tension_id": "trust_or_betray", "explored": [], "implicit": []},
            ],
            "threads": [],
            "initial_beats": [
                {
                    "beat_id": "opening",
                    "summary": "Test",
                    # Using tension ID as entity reference
                    "entities": ["trust_or_betray"],
                }
            ],
        }

        errors = validate_seed_mutations(graph, output)

        entity_errors = [e for e in errors if "initial_beats.0.entities" in e.field_path]
        assert len(entity_errors) == 1
        assert "is a tension ID, not an entity" in entity_errors[0].issue


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
            "tensions": [],
            "threads": [],
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
            "tensions": [],
            "threads": [],
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
            "tensions": [],
            "threads": [],
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
            "tensions": [],
            "threads": [],
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
