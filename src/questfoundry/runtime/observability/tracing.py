"""
LangSmith Tracing integration for QuestFoundry runtime.

Provides hierarchical tracing with:
- Session as the top-level chain
- Turns as child runs within the session
- LLM calls as child runs within turns

Enabled when LANGSMITH_TRACING=true environment variable is set.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Generator

    from langsmith import Client
    from langsmith.run_trees import RunTree

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

    Usage:
        tracer = TracingManager(project_name="questfoundry")
        with tracer.session("session_123", agent_id="showrunner") as session_run:
            with tracer.turn(1, "Hello") as turn_run:
                # LLM calls automatically traced via langchain
                response = await llm.ainvoke(messages)
            turn_run.end(outputs={"response": response})
        session_run.end()
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
        self._client: Client | None = None
        self._session_run: RunTree | None = None
        self._current_turn_run: RunTree | None = None

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    def _get_client(self) -> Client | None:
        """Get or create LangSmith client."""
        if not self._enabled:
            return None

        if self._client is None:
            try:
                from langsmith import Client

                self._client = Client()
            except ImportError:
                logger.warning("langsmith package not installed, tracing disabled")
                self._enabled = False
                return None
            except Exception as e:
                logger.warning(f"Failed to create LangSmith client: {e}")
                self._enabled = False
                return None

        return self._client

    @contextmanager
    def session(
        self,
        session_id: str,
        agent_id: str,
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[RunTree | None, None, None]:
        """
        Create a session-level trace (chain).

        Args:
            session_id: Unique session identifier
            agent_id: Entry agent ID
            project_id: Project identifier
            metadata: Additional metadata

        Yields:
            RunTree for the session (or None if tracing disabled)
        """
        if not self._enabled:
            yield None
            return

        try:
            from langsmith.run_trees import RunTree

            run_id = uuid4()
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

            self._session_run = RunTree(
                name=run_name,
                run_type="chain",
                id=run_id,
                inputs=inputs,
                extra={"metadata": run_metadata},
                project_name=self._project_name,
            )

            logger.debug(f"Started LangSmith session trace: {run_id}")
            yield self._session_run

        except ImportError:
            logger.warning("langsmith package not installed")
            self._enabled = False
            yield None

        except Exception as e:
            logger.warning(f"Failed to create session trace: {e}")
            yield None

        finally:
            if self._session_run:
                try:
                    self._session_run.post()
                except Exception as e:
                    logger.warning(f"Failed to post session trace: {e}")
            self._session_run = None

    @contextmanager
    def turn(
        self,
        turn_id: int,
        user_input: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[RunTree | None, None, None]:
        """
        Create a turn-level trace (chain) as child of session.

        Args:
            turn_id: Turn number
            user_input: User's input text
            agent_id: Agent handling this turn
            metadata: Additional metadata

        Yields:
            RunTree for the turn (or None if tracing disabled)
        """
        if not self._enabled or not self._session_run:
            yield None
            return

        try:
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

            self._current_turn_run = self._session_run.create_child(
                name=run_name,
                run_type="chain",
                inputs=inputs,
                extra={"metadata": run_metadata},
            )

            logger.debug(f"Started LangSmith turn trace: Turn {turn_id}")
            yield self._current_turn_run

        except Exception as e:
            logger.warning(f"Failed to create turn trace: {e}")
            yield None

        finally:
            if self._current_turn_run:
                try:
                    self._current_turn_run.post()
                except Exception as e:
                    logger.warning(f"Failed to post turn trace: {e}")
            self._current_turn_run = None

    def end_turn(
        self,
        output: str | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """
        End the current turn trace with outputs.

        Args:
            output: Response output
            error: Error message if turn failed
            token_usage: Token usage stats
        """
        if not self._current_turn_run:
            return

        try:
            outputs: dict[str, Any] = {}
            if output:
                outputs["output"] = output[:2000]  # Truncate for readability
            if token_usage:
                outputs["token_usage"] = token_usage

            if error:
                self._current_turn_run.end(error=error)
            else:
                self._current_turn_run.end(outputs=outputs)

        except Exception as e:
            logger.warning(f"Failed to end turn trace: {e}")

    def end_session(
        self,
        turn_count: int = 0,
        total_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        """
        End the session trace with summary.

        Args:
            turn_count: Number of turns completed
            total_tokens: Total tokens used
            error: Error message if session failed
        """
        if not self._session_run:
            return

        try:
            outputs: dict[str, Any] = {
                "turn_count": turn_count,
            }
            if total_tokens:
                outputs["total_tokens"] = total_tokens

            if error:
                self._session_run.end(error=error)
            else:
                self._session_run.end(outputs=outputs)

        except Exception as e:
            logger.warning(f"Failed to end session trace: {e}")

    def get_langchain_callbacks(self) -> list[Any]:
        """
        Get LangChain callbacks for the current turn.

        Returns callbacks that will trace LLM calls as children of the current turn.
        """
        if not self._enabled or not self._current_turn_run:
            return []

        try:
            from langchain_core.tracers import LangChainTracer

            # Create a tracer that uses the current turn as parent
            tracer = LangChainTracer(
                project_name=self._project_name,
                client=self._get_client(),
            )
            # Set the parent run ID so LLM calls nest under the turn
            tracer.parent_run_id = self._current_turn_run.id  # type: ignore[attr-defined]

            return [tracer]

        except ImportError:
            return []
        except Exception as e:
            logger.warning(f"Failed to create LangChain callbacks: {e}")
            return []
