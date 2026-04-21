"""POLISH Stage Output Contract validator tests.

Task 2 seeds this file with smoke tests for ``PolishContractError``.
Task 4 adds the layered DREAM + BRAINSTORM + SEED + GROW + POLISH
compliant baseline and the rule-by-rule contract tests that mirror
the pattern of ``tests/unit/test_grow_validation_contract.py``.
"""

from __future__ import annotations

import pytest  # noqa: F401

from questfoundry.graph.graph import Graph  # noqa: F401
from questfoundry.graph.polish_validation import (
    PolishContractError,
    validate_polish_output,  # noqa: F401
)


def test_polish_contract_error_is_value_error() -> None:
    """PolishContractError is a ValueError subclass (same convention as GrowContractError)."""
    assert issubclass(PolishContractError, ValueError)


def test_polish_contract_error_carries_message() -> None:
    """PolishContractError preserves the error message for callers."""
    err = PolishContractError("R-4a.4: intersection groups consumed")
    assert "R-4a.4" in str(err)
