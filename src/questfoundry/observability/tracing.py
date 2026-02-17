"""LangSmith tracing integration for QuestFoundry.

Provides conditional @traceable decorator that gracefully degrades
if langsmith is not installed. This allows the tracing optional
dependency to remain truly optional.

Usage:
    from questfoundry.observability.tracing import traceable, get_current_run_tree

    @traceable(name="My Function", tags=["my-tag"])
    async def my_function():
        # Add dynamic metadata inside the decorated function
        if rt := get_current_run_tree():
            rt.metadata["key"] = "value"
        ...

Note:
    The @traceable decorator only supports async functions. Sync functions
    will raise a TypeError at runtime. The decorator automatically injects
    pipeline_run_id into metadata; add other dynamic metadata inside the
    function using get_current_run_tree().
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig

# Type for run_type parameter
RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]

# Try to import langsmith
try:
    import langsmith
    from langsmith import traceable as ls_traceable
    from langsmith.run_helpers import get_current_run_tree as ls_get_current_run_tree

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    langsmith = None  # type: ignore[assignment]
    ls_traceable = None  # type: ignore[assignment]
    ls_get_current_run_tree = None  # type: ignore[assignment]


# Context variable for pipeline run ID
# This allows all traces within a single pipeline invocation to share a correlation ID
_pipeline_run_id: ContextVar[str | None] = ContextVar("pipeline_run_id", default=None)

P = ParamSpec("P")
R = TypeVar("R")


def _prepare_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Create a copy of metadata and inject the pipeline run ID.

    Args:
        metadata: Optional metadata dict to copy.

    Returns:
        New dict with pipeline_run_id added if available.
    """
    meta = dict(metadata) if metadata else {}
    if run_id := get_pipeline_run_id():
        meta["pipeline_run_id"] = run_id
    return meta


def generate_run_id() -> str:
    """Generate a unique run ID for a pipeline invocation.

    Returns:
        A UUID string that can be used to correlate all traces
        from a single pipeline run.
    """
    return str(uuid.uuid4())


def set_pipeline_run_id(run_id: str) -> None:
    """Set the pipeline run ID for the current context.

    Args:
        run_id: The run ID to set. Use generate_run_id() to create one.
    """
    _pipeline_run_id.set(run_id)


def get_pipeline_run_id() -> str | None:
    """Get the current pipeline run ID.

    Returns:
        The run ID if set, None otherwise.
    """
    return _pipeline_run_id.get()


def get_current_run_tree() -> Any | None:
    """Get the current LangSmith run tree for adding dynamic metadata.

    Returns:
        The current RunTree if langsmith is available and tracing is active,
        None otherwise.

    Example:
        if rt := get_current_run_tree():
            rt.metadata["provider"] = provider_name
            rt.tags.append("custom-tag")
    """
    if not LANGSMITH_AVAILABLE or ls_get_current_run_tree is None:
        return None
    try:
        return ls_get_current_run_tree()
    except Exception:
        return None


def _safe_serialize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Strip non-serializable objects from traced function inputs.

    Langsmith serializes inputs in a background thread.  Objects that
    hold SQLite connections (stage instances, LangChain models, callback
    handlers) trigger ``ProgrammingError: SQLite objects created in a
    thread can only be used in that same thread``.

    This function replaces such objects with their ``repr()`` so the
    trace still records *what* was passed without touching thread-unsafe
    resources.
    """
    safe: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = str(value)[:500]
        elif isinstance(value, dict):
            safe[key] = {k: str(v)[:200] for k, v in value.items()}
        else:
            # Non-primitive: use type name + repr snippet to avoid
            # deep serialization of objects with SQLite connections.
            safe[key] = f"<{type(value).__name__}>"
    return safe


def traceable(
    name: str | None = None,
    *,
    run_type: RunType = "chain",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Conditional traceable decorator for async functions.

    If langsmith is installed and LANGSMITH_TRACING=true, decorates the
    function with @ls.traceable. Otherwise, returns the function unchanged.

    This decorator automatically adds the pipeline run ID to metadata
    if one is set in the current context. For other dynamic metadata,
    use get_current_run_tree() inside the decorated function.

    Note:
        Only async functions are supported. Decorating a sync function
        will raise a TypeError at runtime.

    Args:
        name: Name for the trace span. Defaults to function name.
        run_type: Type of run (chain, llm, tool, etc.). Defaults to "chain".
        tags: List of tags to attach to the trace (copied to avoid mutation).
        metadata: Dict of static metadata key-value pairs (copied to avoid mutation).

    Returns:
        Decorated function (with tracing if available) or original function.

    Example:
        @traceable(name="DREAM Stage", run_type="chain", tags=["stage:dream"])
        async def execute(self, model, user_prompt):
            # Static metadata is passed to decorator
            # Dynamic metadata is added inside the function:
            if rt := get_current_run_tree():
                rt.metadata["provider"] = model.provider_name
            ...
    """
    # Copy mutable arguments to prevent accidental modification
    effective_tags = list(tags) if tags else []
    effective_metadata = dict(metadata) if metadata else {}

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        if not LANGSMITH_AVAILABLE or ls_traceable is None:
            # No langsmith - return function unchanged
            return func

        # Build the traceable decorator with our parameters
        # We wrap the function to inject pipeline_run_id dynamically
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get current run tree to add dynamic metadata
            if rt := get_current_run_tree():
                # Add pipeline run ID if available
                run_id = get_pipeline_run_id()
                if run_id:
                    rt.metadata["pipeline_run_id"] = run_id

            return await func(*args, **kwargs)

        # Apply the langsmith traceable decorator
        # Note: ls_traceable expects run_type as first positional or keyword arg
        #
        # Type ignore rationale:
        # - langsmith.traceable() returns SupportsLangsmithExtra[P, Coroutine[...]]
        # - This is a Protocol that supports __call__ with the same signature
        # - At runtime, the decorated function is callable with the original signature
        # - mypy cannot verify this because SupportsLangsmithExtra is not a subtype
        #   of Callable in the type system, even though it's compatible at runtime
        # - See: https://docs.smith.langchain.com/reference/python/run_helpers
        traced_wrapper: Callable[P, Coroutine[Any, Any, R]] = ls_traceable(
            run_type,
            name=name or func.__name__,
            tags=effective_tags,
            metadata=effective_metadata,
            process_inputs=_safe_serialize_inputs,
        )(wrapper)  # type: ignore[assignment]

        return traced_wrapper

    return decorator


def trace_context(
    name: str,
    *,
    run_type: RunType = "chain",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> Any:
    """Context manager for tracing a code block.

    If langsmith is not available, returns a no-op context manager.

    Args:
        name: Name for the trace span.
        run_type: Type of run (chain, llm, tool, etc.).
        tags: List of tags to attach to the trace.
        metadata: Dict of metadata key-value pairs.
        inputs: Input values to log with the trace.

    Returns:
        LangSmith trace context manager or no-op context manager.

    Example:
        with trace_context("Serialize Attempt", metadata={"attempt": 1}):
            result = await model.ainvoke(messages)
    """
    effective_tags = list(tags) if tags else []
    effective_metadata = _prepare_metadata(metadata)

    if LANGSMITH_AVAILABLE and langsmith is not None:
        return langsmith.trace(
            name=name,
            run_type=run_type,
            tags=effective_tags,
            metadata=effective_metadata,
            inputs=inputs,
        )
    else:
        # Return a no-op context manager
        from contextlib import nullcontext

        return nullcontext()


def build_runnable_config(
    *,
    run_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    recursion_limit: int | None = None,
) -> RunnableConfig:
    """Build a RunnableConfig dict for LangChain model invocations.

    Automatically includes the pipeline run ID in metadata if set.
    This config can be passed to model.ainvoke() to propagate tracing
    metadata and callbacks to child LLM calls.

    Args:
        run_name: Name for this specific invocation (not inherited by sub-calls).
        tags: Tags that are inherited by all sub-calls.
        metadata: Metadata inherited by all sub-calls.
        callbacks: LangChain callback handlers passed to all sub-calls.
        recursion_limit: Maximum recursion depth for agents.

    Returns:
        RunnableConfig dict ready to pass to ainvoke().

    Example:
        config = build_runnable_config(
            run_name="Discuss Phase",
            tags=["dream", "discuss"],
            metadata={"stage": "dream", "phase": "discuss"},
            callbacks=logging_callbacks,
        )
        result = await agent.ainvoke({"messages": msgs}, config=config)
    """
    # Build the config dict - typed as RunnableConfig for compatibility
    config: RunnableConfig = {}

    if run_name:
        config["run_name"] = run_name

    # Build tags list (copy to avoid mutation)
    if tags:
        config["tags"] = list(tags)

    # Build metadata dict with pipeline run ID
    meta = _prepare_metadata(metadata)
    if meta:
        config["metadata"] = meta

    # Pass callbacks through to all sub-calls
    if callbacks:
        config["callbacks"] = callbacks

    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit

    return config


# Export convenience check
def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is available and enabled.

    Returns:
        True if langsmith is installed and LANGSMITH_TRACING=true.
    """
    if not LANGSMITH_AVAILABLE:
        return False

    import os

    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
