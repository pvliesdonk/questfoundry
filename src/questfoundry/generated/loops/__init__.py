"""Generated loop definitions.

These LoopIR objects are generated from MyST domain files in domain/loops/.
They define workflow patterns for the orchestration system.

Usage:
    from questfoundry.generated.loops import ALL_LOOPS, STORY_SPARK

    # Get a specific loop
    loop = ALL_LOOPS["story_spark"]
"""

from questfoundry.compiler.models import (
    GraphEdgeIR,
    GraphNodeIR,
    LoopIR,
    QualityGateIR,
)

# =============================================================================
# story_spark - Initial content discovery loop
# =============================================================================

STORY_SPARK = LoopIR(
    id="story_spark",
    name="Story Spark",
    trigger="user_request",
    entry_point="showrunner",
    nodes=[
        GraphNodeIR(
            id="showrunner",
            role="showrunner",
            timeout=300,
            max_iterations=5,
        ),
        GraphNodeIR(
            id="plotwright",
            role="plotwright",
            timeout=600,
            max_iterations=10,
        ),
        GraphNodeIR(
            id="lorekeeper",
            role="lorekeeper",
            timeout=300,
            max_iterations=5,
        ),
        GraphNodeIR(
            id="gatekeeper",
            role="gatekeeper",
            timeout=300,
            max_iterations=3,
        ),
        GraphNodeIR(
            id="scene_smith",
            role="scene_smith",
            timeout=900,
            max_iterations=15,
        ),
    ],
    edges=[
        # From Showrunner
        GraphEdgeIR(
            source="showrunner",
            target="plotwright",
            condition="intent.status == 'brief_created'",
        ),
        GraphEdgeIR(
            source="showrunner",
            target="__end__",
            condition="intent.type == 'terminate'",
        ),
        # From Plotwright
        GraphEdgeIR(
            source="plotwright",
            target="lorekeeper",
            condition="intent.status == 'needs_lore'",
        ),
        GraphEdgeIR(
            source="plotwright",
            target="gatekeeper",
            condition="intent.status == 'topology_complete'",
        ),
        GraphEdgeIR(
            source="plotwright",
            target="showrunner",
            condition="intent.type == 'escalation'",
        ),
        # From Lorekeeper
        GraphEdgeIR(
            source="lorekeeper",
            target="plotwright",
            condition="intent.status == 'lore_verified'",
        ),
        GraphEdgeIR(
            source="lorekeeper",
            target="scene_smith",
            condition="intent.status == 'canon_provided'",
        ),
        GraphEdgeIR(
            source="lorekeeper",
            target="showrunner",
            condition="intent.status == 'promoted_to_canon'",
        ),
        GraphEdgeIR(
            source="lorekeeper",
            target="showrunner",
            condition="intent.type == 'escalation'",
        ),
        # From Gatekeeper
        GraphEdgeIR(
            source="gatekeeper",
            target="plotwright",
            condition="intent.status == 'topology_failed'",
        ),
        GraphEdgeIR(
            source="gatekeeper",
            target="scene_smith",
            condition="intent.status == 'topology_passed'",
        ),
        GraphEdgeIR(
            source="gatekeeper",
            target="scene_smith",
            condition="intent.status == 'prose_failed'",
        ),
        GraphEdgeIR(
            source="gatekeeper",
            target="lorekeeper",
            condition="intent.status == 'prose_passed'",
        ),
        GraphEdgeIR(
            source="gatekeeper",
            target="showrunner",
            condition="intent.status == 'waiver_requested'",
        ),
        # From Scene Smith
        GraphEdgeIR(
            source="scene_smith",
            target="gatekeeper",
            condition="intent.status == 'prose_complete'",
        ),
        GraphEdgeIR(
            source="scene_smith",
            target="lorekeeper",
            condition="intent.status == 'needs_canon'",
        ),
        GraphEdgeIR(
            source="scene_smith",
            target="showrunner",
            condition="intent.type == 'escalation'",
        ),
    ],
    quality_gates=[
        QualityGateIR(
            before="gatekeeper",
            role="gatekeeper",
            bars=["reachability", "nonlinearity", "gateways"],
            blocking=True,
        ),
    ],
)

# =============================================================================
# Registry
# =============================================================================

ALL_LOOPS: dict[str, LoopIR] = {
    "story_spark": STORY_SPARK,
}

__all__ = [
    "ALL_LOOPS",
    "STORY_SPARK",
]
