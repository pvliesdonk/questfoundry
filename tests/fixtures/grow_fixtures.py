"""Factory functions for GROW stage test graphs.

These build pre-populated Graph objects with the node/edge structures
that GROW expects from earlier stages (BRAINSTORM, SEED).

Graph structure conventions:
- Entity nodes: entity::{raw_id}
- Tension nodes: tension::{raw_id}
- Thread nodes: thread::{raw_id}
- Beat nodes: beat::{raw_id}
- Consequence nodes: consequence::{raw_id}
- Edges: belongs_to (beat→thread), explores (thread→tension),
         has_consequence (thread→consequence), requires (beat→beat)
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph


def make_single_tension_graph() -> Graph:
    """Create a minimal graph with 1 tension, 2 threads, 4 beats.

    Structure:
        tension: mentor_trust
        threads: mentor_trust_canonical (canonical), mentor_trust_alt (alternative)
        beats: opening, mentor_meet, mentor_commits_canonical, mentor_commits_alt

    Beat ordering (requires edges):
        opening → mentor_meet → mentor_commits_canonical
        opening → mentor_meet → mentor_commits_alt

    Consequences:
        mentor_trusted (thread: mentor_trust_canonical)
        mentor_distrusted (thread: mentor_trust_alt)

    Returns:
        Populated Graph instance.
    """
    graph = Graph.empty()

    # Entities
    graph.create_node(
        "entity::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "entity_category": "character",
            "concept": "A wise mentor",
        },
    )
    graph.create_node(
        "entity::hero",
        {
            "type": "entity",
            "raw_id": "hero",
            "entity_category": "character",
            "concept": "The protagonist",
        },
    )

    # Tension
    graph.create_node(
        "tension::mentor_trust",
        {
            "type": "tension",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
        },
    )

    # Threads
    graph.create_node(
        "thread::mentor_trust_canonical",
        {
            "type": "thread",
            "raw_id": "mentor_trust_canonical",
            "tension_id": "mentor_trust",
            "alternative_id": "trust_yes",
            "thread_importance": "major",
            "is_canonical": True,
        },
    )
    graph.create_node(
        "thread::mentor_trust_alt",
        {
            "type": "thread",
            "raw_id": "mentor_trust_alt",
            "tension_id": "mentor_trust",
            "alternative_id": "trust_no",
            "thread_importance": "major",
            "is_canonical": False,
        },
    )

    # Thread → Tension edges
    graph.add_edge("explores", "thread::mentor_trust_canonical", "tension::mentor_trust")
    graph.add_edge("explores", "thread::mentor_trust_alt", "tension::mentor_trust")

    # Beats
    graph.create_node(
        "beat::opening",
        {
            "type": "beat",
            "raw_id": "opening",
            "summary": "The story begins.",
            "threads": ["mentor_trust_canonical", "mentor_trust_alt"],
        },
    )
    graph.create_node(
        "beat::mentor_meet",
        {
            "type": "beat",
            "raw_id": "mentor_meet",
            "summary": "Hero meets the mentor.",
            "threads": ["mentor_trust_canonical", "mentor_trust_alt"],
        },
    )
    graph.create_node(
        "beat::mentor_commits_canonical",
        {
            "type": "beat",
            "raw_id": "mentor_commits_canonical",
            "summary": "Hero trusts the mentor fully.",
            "threads": ["mentor_trust_canonical"],
            "tension_impacts": [{"tension_id": "mentor_trust", "effect": "commits"}],
        },
    )
    graph.create_node(
        "beat::mentor_commits_alt",
        {
            "type": "beat",
            "raw_id": "mentor_commits_alt",
            "summary": "Hero rejects the mentor.",
            "threads": ["mentor_trust_alt"],
            "tension_impacts": [{"tension_id": "mentor_trust", "effect": "commits"}],
        },
    )

    # Beat → Thread edges (belongs_to)
    graph.add_edge("belongs_to", "beat::opening", "thread::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::opening", "thread::mentor_trust_alt")
    graph.add_edge("belongs_to", "beat::mentor_meet", "thread::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::mentor_meet", "thread::mentor_trust_alt")
    graph.add_edge("belongs_to", "beat::mentor_commits_canonical", "thread::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::mentor_commits_alt", "thread::mentor_trust_alt")

    # Beat ordering (requires edges): opening → mentor_meet → commits
    graph.add_edge("requires", "beat::mentor_meet", "beat::opening")
    graph.add_edge("requires", "beat::mentor_commits_canonical", "beat::mentor_meet")
    graph.add_edge("requires", "beat::mentor_commits_alt", "beat::mentor_meet")

    # Consequences
    graph.create_node(
        "consequence::mentor_trusted",
        {
            "type": "consequence",
            "raw_id": "mentor_trusted",
            "thread_id": "mentor_trust_canonical",
            "description": "The mentor becomes an ally.",
        },
    )
    graph.create_node(
        "consequence::mentor_distrusted",
        {
            "type": "consequence",
            "raw_id": "mentor_distrusted",
            "thread_id": "mentor_trust_alt",
            "description": "The mentor becomes an adversary.",
        },
    )
    graph.add_edge(
        "has_consequence", "thread::mentor_trust_canonical", "consequence::mentor_trusted"
    )
    graph.add_edge("has_consequence", "thread::mentor_trust_alt", "consequence::mentor_distrusted")

    return graph


def make_two_tension_graph() -> Graph:
    """Create a graph with 2 tensions, 4 threads, 8 beats.

    Structure:
        tension: mentor_trust (2 threads: canonical, alt)
        tension: artifact_quest (2 threads: canonical, alt)

    Threads:
        mentor_trust_canonical, mentor_trust_alt
        artifact_quest_canonical, artifact_quest_alt

    Beats per thread (with shared opening/closing beats):
        opening (all threads)
        mentor_meet (mentor threads)
        artifact_discover (artifact threads)
        mentor_commits_canonical (mentor_trust_canonical)
        mentor_commits_alt (mentor_trust_alt)
        artifact_commits_canonical (artifact_quest_canonical)
        artifact_commits_alt (artifact_quest_alt)
        finale (all threads)

    Beat ordering:
        opening → mentor_meet → mentor_commits_*
        opening → artifact_discover → artifact_commits_*
        mentor_commits_* → finale
        artifact_commits_* → finale

    Returns:
        Populated Graph instance.
    """
    graph = Graph.empty()

    # Entities
    graph.create_node(
        "entity::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "entity_category": "character",
            "concept": "A wise mentor",
        },
    )
    graph.create_node(
        "entity::hero",
        {
            "type": "entity",
            "raw_id": "hero",
            "entity_category": "character",
            "concept": "The protagonist",
        },
    )
    graph.create_node(
        "entity::artifact",
        {
            "type": "entity",
            "raw_id": "artifact",
            "entity_category": "object",
            "concept": "A powerful artifact",
        },
    )

    # Tensions
    graph.create_node(
        "tension::mentor_trust",
        {
            "type": "tension",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
        },
    )
    graph.create_node(
        "tension::artifact_quest",
        {
            "type": "tension",
            "raw_id": "artifact_quest",
            "question": "Does the hero use the artifact for good?",
        },
    )

    # Threads
    for thread_id, tension_id, alt_id, is_canon in [
        ("mentor_trust_canonical", "mentor_trust", "trust_yes", True),
        ("mentor_trust_alt", "mentor_trust", "trust_no", False),
        ("artifact_quest_canonical", "artifact_quest", "use_good", True),
        ("artifact_quest_alt", "artifact_quest", "use_selfish", False),
    ]:
        graph.create_node(
            f"thread::{thread_id}",
            {
                "type": "thread",
                "raw_id": thread_id,
                "tension_id": tension_id,
                "alternative_id": alt_id,
                "thread_importance": "major",
                "is_canonical": is_canon,
            },
        )
        graph.add_edge("explores", f"thread::{thread_id}", f"tension::{tension_id}")

    # Beats
    all_threads = [
        "mentor_trust_canonical",
        "mentor_trust_alt",
        "artifact_quest_canonical",
        "artifact_quest_alt",
    ]
    mentor_threads = ["mentor_trust_canonical", "mentor_trust_alt"]
    artifact_threads = ["artifact_quest_canonical", "artifact_quest_alt"]

    beats = [
        ("opening", "The story begins.", all_threads, []),
        ("mentor_meet", "Hero meets the mentor.", mentor_threads, []),
        ("artifact_discover", "Hero discovers the artifact.", artifact_threads, []),
        (
            "mentor_commits_canonical",
            "Hero trusts the mentor.",
            ["mentor_trust_canonical"],
            [{"tension_id": "mentor_trust", "effect": "commits"}],
        ),
        (
            "mentor_commits_alt",
            "Hero distrusts the mentor.",
            ["mentor_trust_alt"],
            [{"tension_id": "mentor_trust", "effect": "commits"}],
        ),
        (
            "artifact_commits_canonical",
            "Hero uses artifact for good.",
            ["artifact_quest_canonical"],
            [{"tension_id": "artifact_quest", "effect": "commits"}],
        ),
        (
            "artifact_commits_alt",
            "Hero uses artifact selfishly.",
            ["artifact_quest_alt"],
            [{"tension_id": "artifact_quest", "effect": "commits"}],
        ),
        ("finale", "The conclusion.", all_threads, []),
    ]

    for beat_id, summary, threads, impacts in beats:
        graph.create_node(
            f"beat::{beat_id}",
            {
                "type": "beat",
                "raw_id": beat_id,
                "summary": summary,
                "threads": threads,
                "tension_impacts": impacts,
            },
        )
        for thread_id in threads:
            graph.add_edge("belongs_to", f"beat::{beat_id}", f"thread::{thread_id}")

    # Beat ordering (requires edges)
    # opening → mentor_meet, artifact_discover
    graph.add_edge("requires", "beat::mentor_meet", "beat::opening")
    graph.add_edge("requires", "beat::artifact_discover", "beat::opening")
    # mentor_meet → mentor_commits_*
    graph.add_edge("requires", "beat::mentor_commits_canonical", "beat::mentor_meet")
    graph.add_edge("requires", "beat::mentor_commits_alt", "beat::mentor_meet")
    # artifact_discover → artifact_commits_*
    graph.add_edge("requires", "beat::artifact_commits_canonical", "beat::artifact_discover")
    graph.add_edge("requires", "beat::artifact_commits_alt", "beat::artifact_discover")
    # commits → finale
    graph.add_edge("requires", "beat::finale", "beat::mentor_commits_canonical")
    graph.add_edge("requires", "beat::finale", "beat::mentor_commits_alt")
    graph.add_edge("requires", "beat::finale", "beat::artifact_commits_canonical")
    graph.add_edge("requires", "beat::finale", "beat::artifact_commits_alt")

    # Consequences
    for cons_id, thread_id, desc in [
        ("mentor_trusted", "mentor_trust_canonical", "Mentor becomes ally."),
        ("mentor_distrusted", "mentor_trust_alt", "Mentor becomes adversary."),
        ("artifact_saved", "artifact_quest_canonical", "World is saved."),
        ("artifact_corrupted", "artifact_quest_alt", "World is corrupted."),
    ]:
        graph.create_node(
            f"consequence::{cons_id}",
            {
                "type": "consequence",
                "raw_id": cons_id,
                "thread_id": thread_id,
                "description": desc,
            },
        )
        graph.add_edge("has_consequence", f"thread::{thread_id}", f"consequence::{cons_id}")

    return graph
