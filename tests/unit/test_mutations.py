"""Tests for stage mutation appliers."""

from __future__ import annotations

import pytest

from questfoundry.graph import Graph, MutationError
from questfoundry.graph.mutations import (
    apply_brainstorm_mutations,
    apply_dream_mutations,
    apply_mutations,
    apply_seed_mutations,
    has_mutation_handler,
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
            "entities": [{"id": "char_001", "type": "character", "concept": "Test"}],
            "tensions": [],
        }

        apply_mutations(graph, "brainstorm", output)

        assert graph.has_node("entity::char_001")

    def test_routes_to_seed(self) -> None:
        """Routes seed stage to apply_seed_mutations."""
        graph = Graph.empty()
        # Pre-populate with entity from brainstorm (using prefixed ID)
        graph.add_node("entity::char_001", {"type": "entity", "disposition": "proposed"})

        output = {
            "entities": [{"id": "char_001", "disposition": "retained"}],
            "threads": [],
            "beats": [],
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
        graph.set_node("vision", {"type": "vision", "genre": "fantasy"})

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
        """Raises MutationError when entity missing id."""
        graph = Graph.empty()
        output = {
            "entities": [{"type": "character", "concept": "Test"}],  # Missing id
            "tensions": [],
        }

        with pytest.raises(MutationError, match="Entity at index 0 missing required 'id' field"):
            apply_brainstorm_mutations(graph, output)

    def test_tension_missing_id_raises(self) -> None:
        """Raises MutationError when tension missing id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "tensions": [{"question": "Test?", "alternatives": []}],  # Missing id
        }

        with pytest.raises(MutationError, match="Tension at index 0 missing required 'id' field"):
            apply_brainstorm_mutations(graph, output)

    def test_alternative_missing_id_raises(self) -> None:
        """Raises MutationError when alternative missing id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "tensions": [
                {
                    "id": "tension_001",
                    "question": "Test?",
                    "alternatives": [{"description": "Option A", "canonical": True}],  # Missing id
                }
            ],
        }

        with pytest.raises(
            MutationError,
            match="Alternative at index 0 in tension 'tension_001' missing required 'id' field",
        ):
            apply_brainstorm_mutations(graph, output)

    def test_creates_entity_nodes(self) -> None:
        """Creates entity nodes from brainstorm output."""
        graph = Graph.empty()
        output = {
            "entities": [
                {
                    "id": "kay",
                    "type": "character",
                    "concept": "Young archivist",
                    "notes": "Curious and brave",
                },
                {
                    "id": "archive",
                    "type": "location",
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
                    "id": "mentor_trust",
                    "question": "Can the mentor be trusted?",
                    "involves": ["kay", "mentor"],  # Raw IDs from LLM
                    "why_it_matters": "Trust is key",
                    "alternatives": [
                        {
                            "id": "protector",
                            "description": "Mentor protects Kay",
                            "canonical": True,
                        },
                        {
                            "id": "manipulator",
                            "description": "Mentor manipulates Kay",
                            "canonical": False,
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
        # Involves list is prefixed in storage
        assert tension["involves"] == ["entity::kay", "entity::mentor"]

        # Alternative IDs: tension::tension_id::alt::alt_id
        protector = graph.get_node("tension::mentor_trust::alt::protector")
        assert protector is not None
        assert protector["type"] == "alternative"
        assert protector["raw_id"] == "protector"
        assert protector["canonical"] is True

        manipulator = graph.get_node("tension::mentor_trust::alt::manipulator")
        assert manipulator is not None
        assert manipulator["canonical"] is False

        # Check edges
        edges = graph.get_edges(from_id="tension::mentor_trust", edge_type="has_alternative")
        assert len(edges) == 2

    def test_handles_empty_brainstorm(self) -> None:
        """Handles empty entities and tensions."""
        graph = Graph.empty()
        output = {"entities": [], "tensions": []}

        apply_brainstorm_mutations(graph, output)

        assert len(graph.to_dict()["nodes"]) == 0


class TestSeedMutations:
    """Test SEED stage mutations."""

    def test_entity_decision_missing_id_raises(self) -> None:
        """Raises MutationError when entity decision missing id."""
        graph = Graph.empty()
        output = {
            "entities": [{"disposition": "retained"}],  # Missing id
            "threads": [],
            "initial_beats": [],
        }

        with pytest.raises(
            MutationError, match="Entity decision at index 0 missing required 'id' field"
        ):
            apply_seed_mutations(graph, output)

    def test_thread_missing_id_raises(self) -> None:
        """Raises MutationError when thread missing id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "threads": [{"name": "Test Thread"}],  # Missing id
            "initial_beats": [],
        }

        with pytest.raises(MutationError, match="Thread at index 0 missing required 'id' field"):
            apply_seed_mutations(graph, output)

    def test_beat_missing_id_raises(self) -> None:
        """Raises MutationError when beat missing id."""
        graph = Graph.empty()
        output = {
            "entities": [],
            "threads": [],
            "initial_beats": [{"summary": "Test Beat"}],  # Missing id
        }

        with pytest.raises(MutationError, match="Beat at index 0 missing required 'id' field"):
            apply_seed_mutations(graph, output)

    def test_updates_entity_dispositions(self) -> None:
        """Updates entity dispositions from seed output."""
        graph = Graph.empty()
        # Pre-populate entities from brainstorm (using prefixed IDs)
        graph.add_node("entity::kay", {"type": "entity", "disposition": "proposed"})
        graph.add_node("entity::mentor", {"type": "entity", "disposition": "proposed"})
        graph.add_node("entity::extra", {"type": "entity", "disposition": "proposed"})

        output = {
            "entities": [
                {"id": "kay", "disposition": "retained"},  # Raw IDs from LLM
                {"id": "mentor", "disposition": "retained"},
                {"id": "extra", "disposition": "cut"},
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
        # Pre-populate alternative from brainstorm (with full prefixed ID)
        graph.add_node("tension::mentor_trust::alt::protector", {"type": "alternative"})

        output = {
            "entities": [],
            "threads": [
                {
                    "id": "thread_mentor_trust",
                    "name": "Mentor Trust Arc",
                    "tension_id": "mentor_trust",  # Raw tension ID from LLM
                    "alternative_id": "protector",  # Local alt ID, not full path
                    "description": "Exploring mentor relationship",
                    "consequences": ["consequence_trust"],
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
        # Pre-populate thread (with prefixed ID)
        graph.add_node("thread::thread_mentor_trust", {"type": "thread"})

        output = {
            "entities": [],
            "threads": [],
            "initial_beats": [
                {
                    "id": "opening_001",
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

    def test_skips_missing_entities(self) -> None:
        """Skips entity updates for entities not in graph."""
        graph = Graph.empty()
        # Only kay exists (with prefixed ID)
        graph.add_node("entity::kay", {"type": "entity", "disposition": "proposed"})

        output = {
            "entities": [
                {"id": "kay", "disposition": "retained"},  # Raw ID from LLM
                {"id": "missing", "disposition": "retained"},  # Doesn't exist
            ],
            "threads": [],
            "initial_beats": [],
        }

        # Should not raise, just skip missing
        apply_seed_mutations(graph, output)

        assert graph.get_node("entity::kay")["disposition"] == "retained"
        assert not graph.has_node("entity::missing")

    def test_handles_empty_seed(self) -> None:
        """Handles empty seed output."""
        graph = Graph.empty()
        output = {"entities": [], "threads": [], "initial_beats": []}

        apply_seed_mutations(graph, output)
        # No errors, no changes


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
                {"id": "kay", "type": "character", "concept": "Young archivist"},
                {"id": "mentor", "type": "character", "concept": "Senior archivist"},
            ],
            "tensions": [
                {
                    "id": "mentor_trust",
                    "question": "Can the mentor be trusted?",
                    "involves": ["kay", "mentor"],  # Raw IDs from LLM
                    "why_it_matters": "Trust defines ally or foe",
                    "alternatives": [
                        {
                            "id": "protector",
                            "description": "Mentor protects",
                            "canonical": True,
                        },
                        {
                            "id": "manipulator",
                            "description": "Mentor manipulates",
                            "canonical": False,
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
                {"id": "kay", "disposition": "retained"},
                {"id": "mentor", "disposition": "retained"},
            ],
            "threads": [
                {
                    "id": "thread_mentor",
                    "name": "Mentor Arc",
                    "tension_id": "mentor_trust",  # Raw tension ID
                    "alternative_id": "protector",  # Local alt ID
                    "description": "The mentor relationship thread",
                }
            ],
            "initial_beats": [
                {
                    "id": "opening",
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
