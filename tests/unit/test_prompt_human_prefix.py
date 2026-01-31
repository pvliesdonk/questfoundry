"""Regression tests: prompt functions must not inject role prefixes.

ChatPromptTemplate.from_template().format() prepends "Human: " to the
rendered string.  When that string is later wrapped in a SystemMessage,
the LLM sees conflicting role signals and degrades output quality.

PromptTemplate.from_template().format() does NOT add a prefix.
These tests guard against accidentally switching back.
"""

from __future__ import annotations

import pytest

from questfoundry.agents.prompts import (
    get_brainstorm_discuss_prompt,
    get_brainstorm_summarize_prompt,
    get_seed_serialize_prompt,
    get_seed_summarize_prompt,
)


class TestNoHumanPrefix:
    """Every prompt builder must return raw text, never 'Human: …'."""

    @pytest.mark.parametrize(
        "func,kwargs",
        [
            pytest.param(
                get_brainstorm_discuss_prompt,
                {"vision_context": "v", "research_tools_available": False, "interactive": False},
                id="brainstorm_discuss",
            ),
            pytest.param(get_brainstorm_summarize_prompt, {}, id="brainstorm_summarize"),
            pytest.param(
                get_seed_summarize_prompt, {"brainstorm_context": "c"}, id="seed_summarize"
            ),
            pytest.param(get_seed_serialize_prompt, {}, id="seed_serialize"),
        ],
    )
    def test_no_role_prefix_anywhere(self, func: object, kwargs: dict) -> None:
        """No prompt should contain any role prefix as a line start."""
        result = func(**kwargs)  # type: ignore[operator]
        for line in result.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("Human:"), (
                f"{func.__name__}() contains a line starting with 'Human:': {stripped[:80]}"
            )
