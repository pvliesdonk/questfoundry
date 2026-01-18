"""Tests for stage mutation appliers."""

from __future__ import annotations

import pytest

from questfoundry.graph import Graph, MutationError, SeedMutationError
from questfoundry.graph.mutations import (
    BrainstormMutationError,
    BrainstormValidationError,
    SeedErrorCategory,
    SeedValidationError,
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

    def test_returns_false_for_unknown_stage(self) -> None:
        """Unknown stages don't have mutation handlers."""
        assert has_mutation_handler("grow") is False
        assert has_mutation_handler("fill") is False
        assert has_mutation_handler("ship") is False
        assert has_mutation_handler("mock") is False
        assert has_mutation_handler("nonexistent") is False


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
            "threads": [],
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
