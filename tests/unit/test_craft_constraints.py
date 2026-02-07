"""Tests for craft constraint pool and selection."""

from __future__ import annotations

from collections import deque
from random import Random

from questfoundry.pipeline.craft_constraints import (
    CONSTRAINTS,
    SKIP_PROBABILITY,
    WEIGHTS,
    select_constraint,
)


class TestConstraintPool:
    def test_all_categories_present(self) -> None:
        assert set(CONSTRAINTS.keys()) == {"structure", "sensory", "rhythm", "character"}

    def test_each_category_has_constraints(self) -> None:
        for category, constraints in CONSTRAINTS.items():
            assert len(constraints) >= 5, f"{category} has too few constraints"

    def test_no_duplicate_constraints(self) -> None:
        all_constraints: list[str] = []
        for constraints in CONSTRAINTS.values():
            all_constraints.extend(constraints)
        assert len(all_constraints) == len(set(all_constraints)), "duplicate constraints found"


class TestWeights:
    def test_all_narrative_functions_present(self) -> None:
        assert set(WEIGHTS.keys()) == {
            "introduce",
            "develop",
            "complicate",
            "confront",
            "resolve",
        }

    def test_weights_sum_to_one(self) -> None:
        for func, weights in WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"{func} weights sum to {total}"

    def test_all_categories_in_weights(self) -> None:
        for func, weights in WEIGHTS.items():
            assert set(weights.keys()) == set(CONSTRAINTS.keys()), f"{func} missing categories"


class TestSelectConstraint:
    def test_returns_string(self) -> None:
        result = select_constraint("introduce", deque(maxlen=5), rng=Random(42))
        assert isinstance(result, str)

    def test_returns_constraint_from_pool(self) -> None:
        all_constraints = [c for cat in CONSTRAINTS.values() for c in cat]
        rng = Random(42)
        results: list[str] = []
        for _ in range(50):
            result = select_constraint("develop", deque(maxlen=5), rng=rng)
            if result:
                results.append(result)
        assert len(results) > 0, "no constraints selected in 50 attempts"
        for result in results:
            assert result in all_constraints, f"unknown constraint: {result}"

    def test_skip_probability(self) -> None:
        rng = Random(42)
        recently_used: deque[str] = deque(maxlen=5)
        empty_count = 0
        total = 200
        for _ in range(total):
            result = select_constraint("introduce", recently_used, rng=rng)
            if result == "":
                empty_count += 1
        # With SKIP_PROBABILITY=0.3, expect ~30% empty (allow wide margin)
        ratio = empty_count / total
        assert 0.15 < ratio < 0.45, f"skip ratio {ratio} outside expected range"

    def test_recently_used_dedup(self) -> None:
        rng = Random(42)
        recently_used: deque[str] = deque(maxlen=5)
        # Fill recently_used with all constraints from one category
        for c in CONSTRAINTS["sensory"][:5]:
            recently_used.append(c)
        # Should still return a constraint (falls back to full pool if all used)
        non_empty = [select_constraint("introduce", recently_used, rng=rng) for _ in range(20)]
        assert any(r != "" for r in non_empty)

    def test_unknown_function_falls_back_to_develop(self) -> None:
        rng = Random(42)
        # Should not raise, falls back to develop weights
        result = select_constraint("unknown_function", deque(maxlen=5), rng=rng)
        assert isinstance(result, str)

    def test_deterministic_with_seed(self) -> None:
        results1 = [
            select_constraint("confront", deque(maxlen=5), rng=Random(123)) for _ in range(10)
        ]
        results2 = [
            select_constraint("confront", deque(maxlen=5), rng=Random(123)) for _ in range(10)
        ]
        assert results1 == results2

    def test_skip_probability_value(self) -> None:
        assert SKIP_PROBABILITY == 0.3
