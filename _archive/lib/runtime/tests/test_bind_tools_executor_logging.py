from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from questfoundry.runtime.core.bind_tools_executor import BindToolsExecutor


class _DummyLLM:
    """Minimal LLM stub that supports bind_tools."""

    def bind_tools(self, tools: list[Any]) -> "_DummyLLM":
        # The executor never calls invoke() in these tests, so we can
        # safely return self without implementing full LLM behavior.
        return self


class _FailingTool:
    name = "failing_tool"

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def _run(self, **_: Any) -> dict[str, Any]:
        # Return the preconfigured payload directly so that
        # BindToolsExecutor._execute_tool can exercise its failure
        # logging and JSON wrapping logic.
        return self._payload


def _run_execute_tool(payload: dict[str, Any]) -> tuple[str, bool]:
    """Helper to invoke _execute_tool inside an event loop."""
    tool = _FailingTool(payload)
    executor = BindToolsExecutor(
        llm=_DummyLLM(),
        tools=[tool],
        role_id="test-role",
        system_prompt="system",
        state={},
    )

    async def _run() -> tuple[str, bool]:
        return await executor._execute_tool(tool.name, {})

    return asyncio.run(_run())


def test_execute_tool_logs_error_with_explicit_error(caplog) -> None:
    payload = {"success": False, "error": "explicit failure", "extra": "data"}

    with caplog.at_level(logging.ERROR, logger="questfoundry.runtime.core.bind_tools_executor"):
        observation, success = _run_execute_tool(payload)

    assert success is False
    # Observation should be JSON containing the payload
    data = json.loads(observation)
    assert data["error"] == "explicit failure"

    messages = [rec.getMessage() for rec in caplog.records]
    joined = "\n".join(messages)
    assert "Tool failing_tool failed: explicit failure" in joined
    assert "Unknown error" not in joined


def test_execute_tool_logs_error_with_hint_when_no_error_field(caplog) -> None:
    payload = {
        "success": False,
        "artifact_type": "tu_brief",
        "hint": "Add missing required fields: owner_a",
        "missing_fields": ["owner_a"],
    }

    with caplog.at_level(logging.ERROR, logger="questfoundry.runtime.core.bind_tools_executor"):
        observation, success = _run_execute_tool(payload)

    assert success is False
    data = json.loads(observation)
    assert data["hint"].startswith("Add missing required fields")

    messages = [rec.getMessage() for rec in caplog.records]
    joined = "\n".join(messages)
    assert "Tool failing_tool failed: Add missing required fields" in joined
    assert "Unknown error" not in joined
