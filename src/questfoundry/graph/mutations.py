"""Stage mutation appliers.

Each stage produces structured output that the runtime applies as graph
mutations. This module contains the logic for each stage's mutations.

See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class MutationError(ValueError):
    """Error during mutation application."""

    pass


def _require_field(item: dict[str, Any], field: str, context: str) -> Any:
    """Require a field exists in a dict, raising clear error if missing.

    Args:
        item: Dictionary to check.
        field: Field name to require.
        context: Description for error message (e.g., "entity", "tension").

    Returns:
        The field value.

    Raises:
        MutationError: If field is missing.
    """
    if field not in item:
        raise MutationError(f"{context} missing required '{field}' field")
    return item[field]


def apply_mutations(graph: Graph, stage: str, output: dict[str, Any]) -> None:
    """Apply stage output as graph mutations.

    Routes to the appropriate stage-specific mutation function.

    Args:
        graph: Graph to mutate.
        stage: Stage name (dream, brainstorm, seed, etc.).
        output: Stage output data.

    Raises:
        ValueError: If stage is unknown.
        MutationError: If output is malformed.
    """
    mutation_funcs = {
        "dream": apply_dream_mutations,
        "brainstorm": apply_brainstorm_mutations,
        "seed": apply_seed_mutations,
    }

    if stage not in mutation_funcs:
        raise ValueError(f"Unknown stage: {stage}")

    mutation_funcs[stage](graph, output)


def apply_dream_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply DREAM stage output to graph.

    Creates or replaces the "vision" node with the dream artifact data.

    Args:
        graph: Graph to mutate.
        output: DREAM stage output (DreamArtifact fields).
    """
    # Extract fields, ensuring we have the right structure
    # The output from DREAM stage is a DreamArtifact-like dict
    vision_data = {
        "type": "vision",
        "genre": output.get("genre"),
        "subgenre": output.get("subgenre"),
        "tone": output.get("tone", []),
        "themes": output.get("themes", []),
        "audience": output.get("audience"),
        "style_notes": output.get("style_notes"),
        "scope": output.get("scope"),
        "content_notes": output.get("content_notes"),
    }

    # Remove None values for cleaner storage
    vision_data = {k: v for k, v in vision_data.items() if v is not None}

    graph.set_node("vision", vision_data)


def apply_brainstorm_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply BRAINSTORM stage output to graph.

    Creates entity nodes and tension nodes with their alternatives.
    Tensions are linked to their alternatives via has_alternative edges.

    Args:
        graph: Graph to mutate.
        output: BRAINSTORM stage output (entities, tensions).

    Raises:
        MutationError: If entities or tensions are missing required 'id' fields.
    """
    # Add entities
    for i, entity in enumerate(output.get("entities", [])):
        entity_id = _require_field(entity, "id", f"Entity at index {i}")
        node_data = {
            "type": "entity",
            "entity_type": entity.get("type"),  # character, location, object, faction
            "concept": entity.get("concept"),
            "notes": entity.get("notes"),
            "disposition": "proposed",  # All entities start as proposed
        }
        # Remove None values
        node_data = {k: v for k, v in node_data.items() if v is not None}
        graph.add_node(entity_id, node_data)

    # Add tensions with alternatives
    for i, tension in enumerate(output.get("tensions", [])):
        tension_id = _require_field(tension, "id", f"Tension at index {i}")

        # Create tension node
        tension_data = {
            "type": "tension",
            "question": tension.get("question"),
            "involves": tension.get("involves", []),
            "why_it_matters": tension.get("why_it_matters"),
        }
        tension_data = {k: v for k, v in tension_data.items() if v is not None}
        graph.add_node(tension_id, tension_data)

        # Create alternative nodes and edges
        # Use '::' separator to avoid collisions (e.g., t1::a1 vs t1_a1)
        for j, alt in enumerate(tension.get("alternatives", [])):
            alt_local_id = _require_field(
                alt, "id", f"Alternative at index {j} in tension '{tension_id}'"
            )
            alt_id = f"{tension_id}::{alt_local_id}"
            alt_data = {
                "type": "alternative",
                "description": alt.get("description"),
                "canonical": alt.get("canonical", False),
            }
            alt_data = {k: v for k, v in alt_data.items() if v is not None}
            graph.add_node(alt_id, alt_data)
            graph.add_edge("has_alternative", tension_id, alt_id)


def apply_seed_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply SEED stage output to graph.

    Updates entity dispositions, creates threads from explored tensions,
    and creates initial beats.

    Args:
        graph: Graph to mutate.
        output: SEED stage output (entities, threads, beats).

    Raises:
        MutationError: If required 'id' fields are missing.
    """
    # Update entity dispositions
    for i, entity_decision in enumerate(output.get("entities", [])):
        entity_id = _require_field(entity_decision, "id", f"Entity decision at index {i}")
        if graph.has_node(entity_id):
            graph.update_node(
                entity_id,
                {"disposition": entity_decision.get("disposition", "retained")},
            )

    # Create threads from explored tensions
    for i, thread in enumerate(output.get("threads", [])):
        thread_id = _require_field(thread, "id", f"Thread at index {i}")
        thread_data = {
            "type": "thread",
            "name": thread.get("name"),
            "description": thread.get("description"),
            "consequences": thread.get("consequences", []),
        }
        thread_data = {k: v for k, v in thread_data.items() if v is not None}
        graph.add_node(thread_id, thread_data)

        # Link thread to the alternative it explores
        if "alternative_id" in thread:
            graph.add_edge("explores", thread_id, thread["alternative_id"])

    # Create initial beats
    for i, beat in enumerate(output.get("beats", [])):
        beat_id = _require_field(beat, "id", f"Beat at index {i}")
        beat_data = {
            "type": "beat",
            "name": beat.get("name"),
            "description": beat.get("description"),
            "beat_type": beat.get("beat_type"),
            "tension_impacts": beat.get("tension_impacts", []),
        }
        beat_data = {k: v for k, v in beat_data.items() if v is not None}
        graph.add_node(beat_id, beat_data)

        # Link beat to threads it belongs to
        for thread_id in beat.get("threads", []):
            graph.add_edge("belongs_to", beat_id, thread_id)
