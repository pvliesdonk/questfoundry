"""
LangSmith Tracing integration for QuestFoundry runtime.

Provides hierarchical tracing with:
- Session as the top-level trace
- Turns as child runs within the session
- LLM calls as child runs within turns

Enabled when LANGSMITH_TRACING=true environment variable is set.

Uses langsmith.trace() context manager for proper parent-child linking.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")


class TracingManager:
    """
    Manages LangSmith tracing for a session.

    Creates a hierarchical trace structure:
    - Session (chain) - top level
      - Turn 1 (chain) - child of session
        - LLM Call (llm) - child of turn
      - Turn 2 (chain)
        - LLM Call (llm)
      ...

    Uses langsmith.trace() which properly handles parent-child linking
    and integrates with LangChain's automatic tracing.
    """

    def __init__(
        self,
        project_name: str = "questfoundry",
        *,
        enabled: bool | None = None,
    ):
        """
        Initialize tracing manager.

        Args:
            project_name: LangSmith project name (default: questfoundry)
            enabled: Override automatic detection (default: check env var)
        """
        self._project_name = project_name
        self._enabled = enabled if enabled is not None else is_tracing_enabled()
        self._session_context: Any = None
        self._turn_context: Any = None
        self._session_run_id: str | None = None
        self._turn_run_id: str | None = None

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    @contextmanager
    def session(
        self,
        session_id: str,
        agent_id: str,
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Create a session-level trace.

        Args:
            session_id: Unique session identifier
            agent_id: Entry agent ID
            project_id: Project identifier
            metadata: Additional metadata

        Yields:
            Trace context (or None if tracing disabled)
        """
        if not self._enabled:
            yield None
            return

        run = None
        try:
            from langsmith import trace

            run_name = f"Session: {session_id[:8]}"

            inputs = {
                "session_id": session_id,
                "agent_id": agent_id,
            }
            if project_id:
                inputs["project_id"] = project_id

            run_metadata = {
                "session_id": session_id,
                "agent_id": agent_id,
                **(metadata or {}),
            }

            # Use langsmith.trace() context manager
            with trace(
                name=run_name,
                run_type="chain",
                inputs=inputs,
                metadata=run_metadata,
                project_name=self._project_name,
            ) as run:
                self._session_run_id = str(run.id) if run else None
                logger.debug(f"Started LangSmith session trace: {self._session_run_id}")
                yield run

        except ImportError:
            logger.warning("langsmith package not installed - disabling tracing")
            self._enabled = False
            yield None

        except Exception:
            # If setup failed before yield, yield None so caller has a context
            # If exception was thrown INTO the generator (after yield), re-raise
            if run is None:
                # Setup failed - yield None to let caller proceed without tracing
                yield None
            else:
                # Exception thrown after yield - re-raise to propagate properly
                raise

        finally:
            self._session_run_id = None

    @contextmanager
    def turn(
        self,
        turn_id: int,
        user_input: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Create a turn-level trace as child of session.

        Args:
            turn_id: Turn number
            user_input: User's input text
            agent_id: Agent handling this turn
            metadata: Additional metadata

        Yields:
            Trace context (or None if tracing disabled)
        """
        if not self._enabled:
            yield None
            return

        run = None
        try:
            from langsmith import trace

            run_name = f"Turn {turn_id}"

            inputs = {
                "turn_id": turn_id,
                "user_input": user_input[:1000],  # Truncate for readability
            }
            if agent_id:
                inputs["agent_id"] = agent_id

            run_metadata = {
                "turn_id": turn_id,
                **(metadata or {}),
            }

            # Use langsmith.trace() - it automatically nests under parent
            with trace(
                name=run_name,
                run_type="chain",
                inputs=inputs,
                metadata=run_metadata,
                project_name=self._project_name,
            ) as run:
                self._turn_run_id = str(run.id) if run else None
                logger.debug(f"Started LangSmith turn trace: Turn {turn_id}")
                yield run

        except ImportError:
            logger.warning("langsmith package not installed - disabling tracing")
            self._enabled = False
            yield None

        except Exception:
            # If setup failed before yield, yield None so caller has a context
            # If exception was thrown INTO the generator (after yield), re-raise
            if run is None:
                # Setup failed - yield None to let caller proceed without tracing
                yield None
            else:
                # Exception thrown after yield - re-raise to propagate properly
                raise

        finally:
            self._turn_run_id = None

    def end_turn(
        self,
        output: str | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """
        End the current turn trace with outputs.

        Note: When using trace() context manager, outputs are captured automatically.
        This method is kept for compatibility but may be a no-op.
        """
        # With trace() context manager, ending is handled automatically
        pass

    def end_session(
        self,
        turn_count: int = 0,
        total_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        """
        End the session trace with summary.

        Note: When using trace() context manager, this is handled automatically.
        """
        # With trace() context manager, ending is handled automatically
        pass

    def get_langchain_callbacks(self) -> list[Any]:
        """
        Get LangChain callbacks for the current turn.

        Note: When using langsmith.trace(), LangChain calls are automatically
        nested under the current trace context. This returns an empty list
        because no explicit callbacks are needed.
        """
        # With langsmith.trace(), LangChain auto-tracing nests automatically
        return []
