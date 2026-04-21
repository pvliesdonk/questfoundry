"""POLISH Stage Output Contract validator tests.

Task 2 seeds this file with smoke tests for ``PolishContractError``.
Task 3 adds phase_validation contract tests.
Task 4 adds the layered DREAM + BRAINSTORM + SEED + GROW + POLISH
compliant baseline and the rule-by-rule contract tests that mirror
the pattern of ``tests/unit/test_grow_validation_contract.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_validation import (
    PolishContractError,
)


def test_polish_contract_error_is_value_error() -> None:
    """PolishContractError is a ValueError subclass (same convention as GrowContractError)."""
    assert issubclass(PolishContractError, ValueError)


def test_polish_contract_error_carries_message() -> None:
    """PolishContractError preserves the error message for callers."""
    err = PolishContractError("R-4a.4: intersection groups consumed")
    assert "R-4a.4" in str(err)


def test_phase_validation_raises_contract_error_on_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_validation raises PolishContractError (not PhaseResult) when
    validate_polish_output returns errors."""
    from unittest.mock import MagicMock

    from questfoundry.pipeline.stages.polish import deterministic

    graph = Graph.empty()

    def _mock_validate(g: Graph) -> list[str]:  # noqa: ARG001
        return ["R-4a.4: intersection groups consumed (test)"]

    monkeypatch.setattr(
        "questfoundry.graph.polish_validation.validate_polish_output",
        _mock_validate,
    )

    with pytest.raises(PolishContractError, match=r"R-4a\.4"):
        asyncio.run(deterministic.phase_validation(graph, MagicMock()))


def test_phase_validation_passes_on_clean_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_validation returns completed PhaseResult when no errors."""
    from unittest.mock import MagicMock

    from questfoundry.pipeline.stages.polish import deterministic

    graph = Graph.empty()
    monkeypatch.setattr(
        "questfoundry.graph.polish_validation.validate_polish_output",
        lambda g: [],  # noqa: ARG005
    )

    result = asyncio.run(deterministic.phase_validation(graph, MagicMock()))
    assert result.status == "completed"
