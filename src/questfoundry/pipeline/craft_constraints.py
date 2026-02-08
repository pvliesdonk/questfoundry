"""Curated craft constraint pool for FILL expand phase.

Provides ~30 constraints in 4 categories (structure, sensory, rhythm,
character) with narrative-function-weighted selection. The orchestrator
picks a constraint per passage so the LLM's creative choices are
steered without the model having to self-select.

~30% of passages probabilistically receive no constraint to preserve
natural variety.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import deque

# ---------------------------------------------------------------------------
# Constraint pool
# ---------------------------------------------------------------------------

CONSTRAINTS: dict[str, list[str]] = {
    "structure": [
        "Open in medias res — drop the reader into mid-action",
        "Frame the passage as a single unbroken moment in time",
        "End on an unanswered question or unresolved image",
        "Use a circular structure — echo the opening line at the close",
        "Withhold one key piece of information until the final sentence",
        "Structure around three escalating beats: setup, turn, payoff",
        "Begin with the consequence, then reveal the cause",
    ],
    "sensory": [
        "Anchor every paragraph in a different sense (sight, sound, smell, touch, taste)",
        "Use only sound and touch — no visual descriptions",
        "Build the atmosphere through smell alone before revealing the visual",
        "Ground each beat in a concrete physical texture",
        "Layer two senses per image — never rely on sight alone",
        "Let the environment react to the character's emotional state",
        "Describe the space through what is absent, not what is present",
        "Use temperature as the primary atmospheric thread",
    ],
    "rhythm": [
        "Alternate short declarative sentences with one long flowing sentence",
        "Use sentence fragments for tension, full sentences for resolution",
        "Build to a single long sentence that carries the emotional climax",
        "Keep all sentences under 15 words for urgency",
        "Vary paragraph length dramatically — one word, then a full paragraph",
        "Use repetition of a key phrase to create rhythm",
        "Write one paragraph of pure dialogue, one of pure description",
    ],
    "character": [
        "Reveal character through a single involuntary physical gesture",
        "Show the character noticing something no one else would",
        "Let the character contradict their own words with body language",
        "Give the character a moment of private vulnerability",
        "Show the character's relationship to an object through how they handle it",
        "Reveal backstory through a sensory trigger, not exposition",
        "Let silence or what goes unsaid carry the emotional weight",
        "Show the character making a small choice that reveals their values",
    ],
}

# Narrative-function weights: how likely each category is for each function.
# Values per function must sum to 1.0.
WEIGHTS: dict[str, dict[str, float]] = {
    "introduce": {"sensory": 0.4, "structure": 0.3, "character": 0.2, "rhythm": 0.1},
    "develop": {"character": 0.4, "sensory": 0.2, "structure": 0.2, "rhythm": 0.2},
    "complicate": {"structure": 0.4, "rhythm": 0.3, "character": 0.2, "sensory": 0.1},
    "confront": {"rhythm": 0.3, "character": 0.3, "structure": 0.3, "sensory": 0.1},
    "resolve": {"sensory": 0.3, "structure": 0.3, "character": 0.2, "rhythm": 0.2},
}

# Probability of skipping constraint entirely (preserves natural variety).
SKIP_PROBABILITY = 0.3


def select_constraint(
    narrative_function: str,
    recently_used: deque[str],
    rng: Random | None = None,
) -> str:
    """Select a craft constraint weighted by narrative function.

    Args:
        narrative_function: One of introduce/develop/complicate/confront/resolve.
        recently_used: Deque(maxlen=5) of recently selected constraints.
            Caller owns this; function appends to it when a constraint is selected.
        rng: Random instance for reproducibility. Uses module-level default if None.

    Returns:
        Constraint string, or empty string if probabilistically skipped.
    """
    if rng is None:
        rng = Random()

    # Probabilistic skip
    if rng.random() < SKIP_PROBABILITY:
        return ""

    weights = WEIGHTS.get(narrative_function, WEIGHTS["develop"])

    # Weighted category selection
    categories = list(weights.keys())
    category_weights = [weights[c] for c in categories]
    chosen_category = rng.choices(categories, weights=category_weights, k=1)[0]

    # Pick from that category, avoiding recently used
    pool = CONSTRAINTS[chosen_category]
    available = [c for c in pool if c not in recently_used]
    if not available:
        # All used recently — pick from full pool
        available = pool

    constraint = rng.choice(available)
    recently_used.append(constraint)
    return constraint
