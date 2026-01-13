"""Stage mutation appliers.

Each stage produces structured output that the runtime applies as graph
mutations. This module contains the logic for each stage's mutations.

See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def apply_mutations(graph: Graph, stage: str, output: dict[str, Any]) -> None:
    """Apply stage output as graph mutations.

    Routes to the appropriate stage-specific mutation function.

    Args:
        graph: Graph to mutate.
        stage: Stage name (dream, brainstorm, seed, etc.).
        output: Stage output data.

    Raises:
        ValueError: If stage is unknown.
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
    """
    # Add entities
    for entity in output.get("entities", []):
        entity_id = entity["id"]
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
    for tension in output.get("tensions", []):
        tension_id = tension["id"]

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
        for alt in tension.get("alternatives", []):
            alt_id = f"{tension_id}_{alt['id']}"
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
    """
    # Update entity dispositions
    for entity_decision in output.get("entities", []):
        entity_id = entity_decision["id"]
        if graph.has_node(entity_id):
            graph.update_node(
                entity_id,
                {"disposition": entity_decision.get("disposition", "retained")},
            )

    # Create threads from explored tensions
    for thread in output.get("threads", []):
        thread_id = thread["id"]
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
    for beat in output.get("beats", []):
        beat_id = beat["id"]
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
