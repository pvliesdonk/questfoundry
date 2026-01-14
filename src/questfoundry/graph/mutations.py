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


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from a dictionary for cleaner storage.

    Args:
        data: Dictionary to clean.

    Returns:
        New dictionary with None values removed.
    """
    return {k: v for k, v in data.items() if v is not None}


# Registry of stages with mutation handlers
_MUTATION_STAGES = frozenset({"dream", "brainstorm", "seed"})


def has_mutation_handler(stage: str) -> bool:
    """Check if a stage has a mutation handler.

    Use this to determine whether to call apply_mutations() for a stage.
    This is the single source of truth for which stages support graph mutations.

    Args:
        stage: Stage name to check.

    Returns:
        True if the stage has a mutation handler, False otherwise.
    """
    return stage in _MUTATION_STAGES


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
    vision_data = _clean_dict(vision_data)

    graph.set_node("vision", vision_data)


def _prefix_id(node_type: str, raw_id: str) -> str:
    """Prefix a raw ID with its node type for namespace isolation.

    This allows entities and tensions to have the same raw ID without collision.
    E.g., both can use "cipher_journal" -> "entity::cipher_journal", "tension::cipher_journal"

    Args:
        node_type: Node type prefix (entity, tension, thread, etc.)
        raw_id: Raw ID from LLM output.

    Returns:
        Prefixed ID in format "type::raw_id".
    """
    return f"{node_type}::{raw_id}"


def apply_brainstorm_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply BRAINSTORM stage output to graph.

    Creates entity nodes and tension nodes with their alternatives.
    Tensions are linked to their alternatives via has_alternative edges.

    Node IDs are prefixed by type to avoid collisions:
    - entity::raw_id
    - tension::raw_id
    - tension::tension_id::alt::alt_id (for alternatives)

    Args:
        graph: Graph to mutate.
        output: BRAINSTORM stage output (entities, tensions).

    Raises:
        MutationError: If entities or tensions are missing required id fields.
    """
    # Add entities
    for i, entity in enumerate(output.get("entities", [])):
        raw_id = _require_field(entity, "entity_id", f"Entity at index {i}")
        entity_id = _prefix_id("entity", raw_id)
        node_data = {
            "type": "entity",
            "raw_id": raw_id,  # Store original ID for reference
            "entity_type": entity.get("entity_category"),  # character, location, object, faction
            "concept": entity.get("concept"),
            "notes": entity.get("notes"),
            "disposition": "proposed",  # All entities start as proposed
        }
        # Remove None values
        node_data = _clean_dict(node_data)
        graph.add_node(entity_id, node_data)

    # Add tensions with alternatives
    for i, tension in enumerate(output.get("tensions", [])):
        raw_id = _require_field(tension, "tension_id", f"Tension at index {i}")
        tension_id = _prefix_id("tension", raw_id)

        # Prefix entity references in central_entity_ids list
        raw_central_entities = tension.get("central_entity_ids", [])
        prefixed_central_entities = [_prefix_id("entity", eid) for eid in raw_central_entities]

        # Create tension node
        tension_data = {
            "type": "tension",
            "raw_id": raw_id,  # Store original ID for reference
            "question": tension.get("question"),
            "central_entity_ids": prefixed_central_entities,
            "why_it_matters": tension.get("why_it_matters"),
        }
        tension_data = _clean_dict(tension_data)
        graph.add_node(tension_id, tension_data)

        # Create alternative nodes and edges
        for j, alt in enumerate(tension.get("alternatives", [])):
            alt_local_id = _require_field(
                alt, "alternative_id", f"Alternative at index {j} in tension '{raw_id}'"
            )
            # Alternative ID format: tension::tension_raw_id::alt::alt_local_id
            alt_id = f"{tension_id}::alt::{alt_local_id}"
            alt_data = {
                "type": "alternative",
                "raw_id": alt_local_id,
                "description": alt.get("description"),
                "is_default_path": alt.get("is_default_path", False),
            }
            alt_data = _clean_dict(alt_data)
            graph.add_node(alt_id, alt_data)
            graph.add_edge("has_alternative", tension_id, alt_id)


def apply_seed_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply SEED stage output to graph.

    Updates entity dispositions, creates threads from explored tensions,
    creates consequences, and creates initial beats.

    Node IDs are prefixed by type to match BRAINSTORM's namespacing:
    - thread::raw_id
    - consequence::raw_id
    - beat::raw_id

    References to entities/tensions from LLM output are prefixed for lookup.

    Args:
        graph: Graph to mutate.
        output: SEED stage output (SeedOutput fields).

    Raises:
        MutationError: If required id fields are missing.
    """
    # Update entity dispositions
    for i, entity_decision in enumerate(output.get("entities", [])):
        raw_id = _require_field(entity_decision, "entity_id", f"Entity decision at index {i}")
        entity_id = _prefix_id("entity", raw_id)
        if graph.has_node(entity_id):
            graph.update_node(
                entity_id,
                {"disposition": entity_decision.get("disposition", "retained")},
            )

    # Update tension exploration decisions
    for i, tension_decision in enumerate(output.get("tensions", [])):
        raw_id = _require_field(tension_decision, "tension_id", f"Tension decision at index {i}")
        tension_id = _prefix_id("tension", raw_id)
        if graph.has_node(tension_id):
            graph.update_node(
                tension_id,
                {
                    "explored": tension_decision.get("explored", []),
                    "implicit": tension_decision.get("implicit", []),
                },
            )

    # Create threads from explored tensions (must be created before consequences)
    for i, thread in enumerate(output.get("threads", [])):
        raw_id = _require_field(thread, "thread_id", f"Thread at index {i}")
        thread_id = _prefix_id("thread", raw_id)

        # Store prefixed tension reference
        raw_tension_id = thread.get("tension_id")
        prefixed_tension_id = _prefix_id("tension", raw_tension_id) if raw_tension_id else None

        # Prefix unexplored alternatives from the same tension
        raw_unexplored = thread.get("unexplored_alternative_ids", [])
        prefixed_unexplored = []
        if prefixed_tension_id:
            for unexplored_alt_id in raw_unexplored:
                # Format: tension::tension_id::alt::alt_id
                full_unexplored_id = f"{prefixed_tension_id}::alt::{unexplored_alt_id}"
                prefixed_unexplored.append(full_unexplored_id)

        thread_data = {
            "type": "thread",
            "raw_id": raw_id,
            "name": thread.get("name"),
            "tension_id": prefixed_tension_id,
            "alternative_id": thread.get("alternative_id"),  # Local alt ID, not prefixed
            "unexplored_alternative_ids": prefixed_unexplored,
            "thread_importance": thread.get("thread_importance"),
            "description": thread.get("description"),
            "consequence_ids": thread.get("consequence_ids", []),
        }
        thread_data = _clean_dict(thread_data)
        graph.add_node(thread_id, thread_data)

        # Link thread to the alternative it explores
        if "alternative_id" in thread and prefixed_tension_id:
            alt_local_id = thread["alternative_id"]
            # Alternative ID format: tension::tension_id::alt::alt_id
            full_alt_id = f"{prefixed_tension_id}::alt::{alt_local_id}"
            graph.add_edge("explores", thread_id, full_alt_id)

    # Create consequences (after threads so edges can be created)
    for i, consequence in enumerate(output.get("consequences", [])):
        raw_id = _require_field(consequence, "consequence_id", f"Consequence at index {i}")
        consequence_id = _prefix_id("consequence", raw_id)

        # Prefix thread reference
        raw_thread_id = consequence.get("thread_id")
        prefixed_thread_id = _prefix_id("thread", raw_thread_id) if raw_thread_id else None

        consequence_data = {
            "type": "consequence",
            "raw_id": raw_id,
            "thread_id": prefixed_thread_id,
            "description": consequence.get("description"),
            "narrative_effects": consequence.get("narrative_effects", []),
        }
        consequence_data = _clean_dict(consequence_data)
        graph.add_node(consequence_id, consequence_data)

        # Link consequence to its thread (thread must exist)
        if prefixed_thread_id and graph.has_node(prefixed_thread_id):
            graph.add_edge("has_consequence", prefixed_thread_id, consequence_id)

    # Create initial beats
    for i, beat in enumerate(output.get("initial_beats", [])):
        raw_id = _require_field(beat, "beat_id", f"Beat at index {i}")
        beat_id = _prefix_id("beat", raw_id)

        # Prefix entity references
        raw_entities = beat.get("entities", [])
        prefixed_entities = [_prefix_id("entity", eid) for eid in raw_entities]

        # Prefix location reference (location is an entity)
        raw_location = beat.get("location")
        prefixed_location = _prefix_id("entity", raw_location) if raw_location else None

        # Prefix location_alternatives (also entity IDs)
        raw_location_alts = beat.get("location_alternatives", [])
        prefixed_location_alts = [_prefix_id("entity", eid) for eid in raw_location_alts]

        # Prefix tension_id in tension_impacts
        raw_impacts = beat.get("tension_impacts", [])
        prefixed_impacts = []
        for impact in raw_impacts:
            prefixed_impact = dict(impact)
            if "tension_id" in impact:
                prefixed_impact["tension_id"] = _prefix_id("tension", impact["tension_id"])
            prefixed_impacts.append(prefixed_impact)

        beat_data = {
            "type": "beat",
            "raw_id": raw_id,
            "summary": beat.get("summary"),
            "tension_impacts": prefixed_impacts,
            "entities": prefixed_entities,
            "location": prefixed_location,
            "location_alternatives": prefixed_location_alts,
        }
        beat_data = _clean_dict(beat_data)
        graph.add_node(beat_id, beat_data)

        # Link beat to threads it belongs to
        for raw_thread_id in beat.get("threads", []):
            prefixed_thread_id = _prefix_id("thread", raw_thread_id)
            graph.add_edge("belongs_to", beat_id, prefixed_thread_id)

    # Store convergence sketch as metadata
    if "convergence_sketch" in output:
        sketch = output["convergence_sketch"]
        graph.set_node(
            "convergence_sketch",
            {
                "type": "convergence_sketch",
                "convergence_points": sketch.get("convergence_points", []),
                "residue_notes": sketch.get("residue_notes", []),
            },
        )
