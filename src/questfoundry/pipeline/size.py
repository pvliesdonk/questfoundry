"""Story size profiles for coordinated pipeline parameters.

Defines preset size profiles that control story scale across all pipeline
stages. The LLM picks a preset during DREAM (via Scope.story_size), and
downstream stages use the resolved SizeProfile to coordinate entity counts,
arc limits, beat counts, and prompt guidance.

Presets:
    vignette: Tight single-thread story (5-15 passages)
    short: Modest branching (15-30 passages)
    standard: Full branching, current default (30-60 passages)
    long: Extensive branching (60-120 passages)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


@dataclass(frozen=True)
class SizeProfile:
    """Coordinated size parameters for a story preset.

    All min/max pairs represent ranges used in prompt templates
    (e.g., "aim for 5-10 characters").
    """

    preset: str

    # Arc / branching structure
    max_arcs: int
    fully_explored: int

    # Entity counts (for brainstorm/seed guidance)
    characters_min: int
    characters_max: int
    locations_min: int
    locations_max: int
    objects_min: int
    objects_max: int
    dilemmas_min: int
    dilemmas_max: int

    # Brainstorm over-generation targets
    entities_min: int
    entities_max: int

    # Beat structure
    beats_per_path_min: int
    beats_per_path_max: int
    convergence_points_min: int
    convergence_points_max: int

    # Output estimates
    est_passages_min: int
    est_passages_max: int
    est_words_min: int
    est_words_max: int

    # Voice / prose guidance
    tone_words_min: int
    tone_words_max: int

    def range_str(self, field_prefix: str) -> str:
        """Format a min/max pair as a 'min-max' string for prompt injection.

        Args:
            field_prefix: Field name prefix (e.g., "characters" for
                characters_min/characters_max).

        Returns:
            Formatted range string like "5-10".

        Raises:
            AttributeError: If the field prefix doesn't exist on this profile.
        """
        lo = getattr(self, f"{field_prefix}_min")
        hi = getattr(self, f"{field_prefix}_max")
        return f"{lo}-{hi}"


PRESETS: dict[str, SizeProfile] = {
    "vignette": SizeProfile(
        preset="vignette",
        max_arcs=2,
        fully_explored=1,
        characters_min=2,
        characters_max=4,
        locations_min=1,
        locations_max=2,
        objects_min=1,
        objects_max=2,
        dilemmas_min=2,
        dilemmas_max=3,
        entities_min=5,
        entities_max=10,
        beats_per_path_min=2,
        beats_per_path_max=3,
        convergence_points_min=0,
        convergence_points_max=1,
        est_passages_min=5,
        est_passages_max=15,
        est_words_min=2000,
        est_words_max=5000,
        tone_words_min=2,
        tone_words_max=4,
    ),
    "short": SizeProfile(
        preset="short",
        max_arcs=8,
        fully_explored=3,
        characters_min=4,
        characters_max=6,
        locations_min=2,
        locations_max=4,
        objects_min=2,
        objects_max=4,
        dilemmas_min=3,
        dilemmas_max=5,
        entities_min=10,
        entities_max=18,
        beats_per_path_min=2,
        beats_per_path_max=4,
        convergence_points_min=1,
        convergence_points_max=2,
        est_passages_min=15,
        est_passages_max=30,
        est_words_min=5000,
        est_words_max=15000,
        tone_words_min=3,
        tone_words_max=4,
    ),
    "standard": SizeProfile(
        preset="standard",
        max_arcs=16,
        fully_explored=4,
        characters_min=5,
        characters_max=10,
        locations_min=3,
        locations_max=6,
        objects_min=3,
        objects_max=5,
        dilemmas_min=4,
        dilemmas_max=8,
        entities_min=15,
        entities_max=25,
        beats_per_path_min=2,
        beats_per_path_max=4,
        convergence_points_min=1,
        convergence_points_max=2,
        est_passages_min=30,
        est_passages_max=60,
        est_words_min=15000,
        est_words_max=30000,
        tone_words_min=3,
        tone_words_max=5,
    ),
    "long": SizeProfile(
        preset="long",
        max_arcs=32,
        fully_explored=5,
        characters_min=8,
        characters_max=15,
        locations_min=4,
        locations_max=8,
        objects_min=4,
        objects_max=7,
        dilemmas_min=5,
        dilemmas_max=10,
        entities_min=20,
        entities_max=35,
        beats_per_path_min=3,
        beats_per_path_max=5,
        convergence_points_min=2,
        convergence_points_max=4,
        est_passages_min=60,
        est_passages_max=120,
        est_words_min=30000,
        est_words_max=60000,
        tone_words_min=4,
        tone_words_max=6,
    ),
}

VALID_PRESETS = frozenset(PRESETS.keys())


def get_size_profile(preset: str = "standard") -> SizeProfile:
    """Resolve a preset name to a SizeProfile.

    Args:
        preset: One of "vignette", "short", "standard", "long".

    Returns:
        The corresponding SizeProfile with all coordinated parameters.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown story_size preset: {preset!r}. Valid presets: {sorted(VALID_PRESETS)}"
        )
    return PRESETS[preset]


def resolve_size_from_graph(graph: Graph) -> SizeProfile:
    """Read story_size from the DREAM vision node and resolve to a SizeProfile.

    Looks up the vision node's ``scope.story_size`` field. Falls back to
    ``"standard"`` if the vision node, scope, or story_size field is missing
    (backward compatibility with projects created before size presets).

    Args:
        graph: Graph containing the DREAM vision node.

    Returns:
        Resolved SizeProfile.
    """
    vision = graph.get_node("vision")
    if vision is None:
        return get_size_profile("standard")

    scope: dict[str, Any] | None = vision.get("scope")
    if scope is None:
        return get_size_profile("standard")

    story_size = scope.get("story_size", "standard")
    if story_size not in PRESETS:
        return get_size_profile("standard")

    return get_size_profile(story_size)


def size_template_vars(profile: SizeProfile | None = None) -> dict[str, str]:
    """Build template variable dict from a size profile.

    Returns a dict mapping ``{size_*}`` template variable names to formatted
    range strings. Falls back to ``standard`` preset if no profile is given.

    Args:
        profile: Size profile to extract ranges from. Defaults to standard.

    Returns:
        Dict with keys like ``size_characters``, ``size_dilemmas``, etc.
    """
    p = profile or get_size_profile("standard")
    return {
        "size_characters": p.range_str("characters"),
        "size_locations": p.range_str("locations"),
        "size_objects": p.range_str("objects"),
        "size_dilemmas": p.range_str("dilemmas"),
        "size_entities": p.range_str("entities"),
        "size_beats_per_path": p.range_str("beats_per_path"),
        "size_convergence_points": p.range_str("convergence_points"),
        "size_est_passages": p.range_str("est_passages"),
        "size_est_words": p.range_str("est_words"),
        "size_tone_words": p.range_str("tone_words"),
        "size_preset": p.preset,
    }
