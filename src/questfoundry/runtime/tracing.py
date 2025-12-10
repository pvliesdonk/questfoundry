"""LangSmith tracing integration for QuestFoundry.

Provides automatic tracing of orchestration runs and role executions.
Tracing is auto-enabled when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.

The trace hierarchy shows the complete workflow:
- Orchestrator Run (top-level, contains loop_id and request)
  - Role: showrunner (turn N)
    - LLM calls (auto-traced by LangChain)
    - Tool calls
  - Role: plotwright (task="...")
    - LLM calls
    - Tool calls
  - Role: showrunner (turn N+1)
  - ...

Usage:
    # Tracing is automatic when env vars are set:
    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY=<your-key>
    export LANGSMITH_PROJECT=questfoundry  # optional, defaults to "default"

    # Then just run your code normally
    result = await orchestrator.run("Create a story")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)

# Type vars for decorator
F = TypeVar("F", bound=Callable[..., Any])

# Cache tracing state
_tracing_enabled: bool | None = None


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled via environment variables."""
    global _tracing_enabled
    if _tracing_enabled is None:
        _tracing_enabled = (
            os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
            and bool(os.environ.get("LANGSMITH_API_KEY"))
        )
        if _tracing_enabled:
            logger.info("LangSmith tracing enabled")
    return _tracing_enabled


def trace_orchestrator_run(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorator to trace the orchestrator run as a parent span.

    Captures:
    - loop_id: The workflow loop identifier
    - request: The user's request
    - Tags: ["orchestrator", "questfoundry"]
    """
    @wraps(func)
    async def wrapper(
        self: Any,
        request: str,
        loop_id: str = "default",
        resume_run_id: str | None = None,
        resume_checkpoint_id: int | None = None,
    ) -> Any:
        if not is_tracing_enabled():
            return await func(
                self, request, loop_id, resume_run_id, resume_checkpoint_id
            )

        try:
            from langsmith import traceable

            @traceable(
                name="QuestFoundry Orchestration",
                run_type="chain",
                tags=["orchestrator", "questfoundry"],
                metadata={
                    "loop_id": loop_id,
                    "request": request[:500] if request else "(resuming)",
                    "resume_run_id": resume_run_id,
                    "resume_checkpoint_id": resume_checkpoint_id,
                },
            )
            async def traced_run(
                orchestrator: Any,
                req: str,
                lid: str,
                run_id: str | None,
                checkpoint_id: int | None,
            ) -> Any:
                return await func(orchestrator, req, lid, run_id, checkpoint_id)

            return await traced_run(
                self, request, loop_id, resume_run_id, resume_checkpoint_id
            )
        except ImportError:
            logger.debug("langsmith not available, skipping tracing")
            return await func(
                self, request, loop_id, resume_run_id, resume_checkpoint_id
            )

    return wrapper


def trace_role_execution(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorator to trace role execution as a child span.

    Captures:
    - role_id: The role being executed (e.g., "plotwright", "showrunner")
    - task: The task being performed (truncated)
    - Tags: ["role:{role_id}", "questfoundry"]
    """
    @wraps(func)
    async def wrapper(self: Any, task: str) -> Any:
        if not is_tracing_enabled():
            return await func(self, task)

        try:
            from langsmith import traceable

            role_obj = getattr(self, "role", None)
            role_id = role_obj.id if role_obj and hasattr(role_obj, "id") else "unknown"

            @traceable(
                name=f"Role: {role_id}",
                run_type="chain",
                tags=[f"role:{role_id}", "questfoundry"],
                metadata={
                    "role_id": role_id,
                    "task": task[:500],  # Truncate long tasks
                },
            )
            async def traced_execute(agent: Any, t: str) -> Any:
                return await func(agent, t)

            return await traced_execute(self, task)
        except ImportError:
            logger.debug("langsmith not available, skipping tracing")
            return await func(self, task)

    return wrapper


def trace_sr_turn(turn: int, delegation_count: int) -> Callable[[F], F]:
    """Context manager/decorator to trace SR turns.

    This is used within the orchestrator loop to trace each SR turn.

    Parameters
    ----------
    turn : int
        The turn number within this orchestration run.
    delegation_count : int
        Number of delegations so far.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            try:
                from langsmith import traceable

                @traceable(
                    name="Role: showrunner",
                    run_type="chain",
                    tags=["role:showrunner", "questfoundry"],
                    metadata={
                        "role_id": "showrunner",
                        "turn": turn,
                        "delegation_count": delegation_count,
                    },
                )
                async def traced_turn(*a: Any, **kw: Any) -> Any:
                    return await func(*a, **kw)

                return await traced_turn(*args, **kwargs)
            except ImportError:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]
    return decorator


class TracedSRTurn:
    """Context manager for tracing SR turns within the orchestrator loop.

    Usage:
        async with TracedSRTurn(turn=1, delegation_count=0):
            result = await sr_executor.run(prompt)
    """

    def __init__(self, turn: int, delegation_count: int, prompt: str = ""):
        self.turn = turn
        self.delegation_count = delegation_count
        self.prompt = prompt
        self._trace_context: Any = None

    async def __aenter__(self) -> TracedSRTurn:
        if not is_tracing_enabled():
            return self

        try:
            import langsmith as ls

            self._trace_context = ls.trace(
                name="Role: showrunner",
                run_type="chain",
                inputs={"prompt": self.prompt[:500]},
                tags=["role:showrunner", "questfoundry"],
                metadata={
                    "role_id": "showrunner",
                    "turn": self.turn,
                    "delegation_count": self.delegation_count,
                },
            )
            self._trace_context.__enter__()
        except ImportError:
            pass
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._trace_context is not None:
            try:
                if exc_type is not None:
                    # Record error
                    self._trace_context.end(error=str(exc_val))
                else:
                    self._trace_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug(f"Error ending trace: {e}")


def get_langsmith_project() -> str:
    """Get the LangSmith project name.

    Returns the LANGSMITH_PROJECT env var or 'questfoundry' as default.
    """
    return os.environ.get("LANGSMITH_PROJECT", "questfoundry")


def configure_tracing() -> None:
    """Configure LangSmith tracing with QuestFoundry defaults.

    Sets LANGSMITH_PROJECT to 'questfoundry' if not already set.
    Call this early in your application startup.
    """
    if "LANGSMITH_PROJECT" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = "questfoundry"

    if is_tracing_enabled():
        logger.info(
            f"LangSmith tracing configured: project={get_langsmith_project()}"
        )
