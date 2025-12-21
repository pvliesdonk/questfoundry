"""
Tool result caching for QuestFoundry runtime.

Provides caching abstractions to prevent repeated tool execution:
- Intra-turn (ACTIVATION scope): Prevent same tool called multiple times in one turn
- Inter-turn (SESSION scope): Cache static tools like consult_* across turns

This helps manage context growth by avoiding redundant tool result embedding.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CacheScope(Enum):
    """Scope of the cache entry."""

    ACTIVATION = "activation"  # Per activate() / activate_streaming() call
    SESSION = "session"  # Per session_id (for static tools)


class PresentationPolicy(Enum):
    """How to present cached results in the conversation."""

    FULL_CONTENT = "full"  # Re-emit the original content
    REFERENCE_ONLY = "reference"  # Just a short note referencing prior result
    SUMMARY = "summary"  # Summarized version of the result


@dataclass
class CachedToolResult:
    """A cached tool execution result."""

    tool_name: str
    args_json: str  # Canonical JSON-serialized arguments
    content: str | dict[str, Any]  # Raw result payload
    success: bool
    created_at: datetime = field(default_factory=datetime.now)
    approx_tokens: int | None = None  # Optional token estimate
    presentation_id: str = ""  # Opaque ID for referencing (e.g., "playbook:story_spark#1")


@dataclass
class ToolCachingPolicy:
    """Policy for how a tool participates in caching."""

    participate_in_session_cache: bool = False  # consult_* = True
    participate_in_activation_cache: bool = True  # Most tools = True
    presentation_on_hit: PresentationPolicy = PresentationPolicy.REFERENCE_ONLY


# Default policies for known tool types
DEFAULT_TOOL_POLICIES: dict[str, ToolCachingPolicy] = {
    # Static consult tools - cache across session
    "consult_playbook": ToolCachingPolicy(
        participate_in_session_cache=True,
        participate_in_activation_cache=True,
        presentation_on_hit=PresentationPolicy.REFERENCE_ONLY,
    ),
    "consult_schema": ToolCachingPolicy(
        participate_in_session_cache=True,
        participate_in_activation_cache=True,
        presentation_on_hit=PresentationPolicy.REFERENCE_ONLY,
    ),
    "consult_knowledge": ToolCachingPolicy(
        participate_in_session_cache=True,
        participate_in_activation_cache=True,
        presentation_on_hit=PresentationPolicy.REFERENCE_ONLY,
    ),
    "consult_corpus": ToolCachingPolicy(
        participate_in_session_cache=True,
        participate_in_activation_cache=True,
        presentation_on_hit=PresentationPolicy.REFERENCE_ONLY,
    ),
}

# Default policy for tools not in the map
DEFAULT_POLICY = ToolCachingPolicy(
    participate_in_session_cache=False,
    participate_in_activation_cache=True,
    presentation_on_hit=PresentationPolicy.REFERENCE_ONLY,
)


def _hash_args(args: dict[str, Any]) -> str:
    """Create a stable hash of tool arguments.

    Args are JSON-normalized (sorted keys, no whitespace) before hashing
    to ensure consistent cache keys regardless of argument order.
    """
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _make_cache_key(
    tool_name: str,
    tool_args: dict[str, Any],
    model_family: str | None = None,
) -> str:
    """Create a cache key from tool name and arguments.

    Args:
        tool_name: The tool being called
        tool_args: Arguments passed to the tool
        model_family: Optional model identifier (for model-sensitive caching)

    Returns:
        A string key for cache lookup
    """
    args_hash = _hash_args(tool_args)
    if model_family:
        return f"{tool_name}:{args_hash}:{model_family}"
    return f"{tool_name}:{args_hash}"


class ToolResultCache:
    """Cache for tool execution results.

    Supports two scopes:
    - ACTIVATION: Per-activation cache (cleared after each activate() call)
    - SESSION: Per-session cache (persists across turns for static tools)

    Usage:
        cache = ToolResultCache()

        # Before executing a tool
        hit = cache.lookup(
            session_id="sess_1",
            scope=CacheScope.ACTIVATION,
            tool_name="consult_playbook",
            tool_args={"playbook": "story_spark"},
        )
        if hit:
            # Use cached result
            return hit

        # Execute tool...
        result = await execute_tool(...)

        # Store in cache
        cache.record(
            session_id="sess_1",
            scope=CacheScope.ACTIVATION,
            tool_name="consult_playbook",
            tool_args={"playbook": "story_spark"},
            result=CachedToolResult(...)
        )
    """

    def __init__(self) -> None:
        # SESSION scope: {session_id: {cache_key: CachedToolResult}}
        self._session_cache: dict[str, dict[str, CachedToolResult]] = {}

        # ACTIVATION scope: {activation_id: {cache_key: CachedToolResult}}
        # activation_id is typically f"{session_id}:{turn_number}"
        self._activation_cache: dict[str, dict[str, CachedToolResult]] = {}

        # Counter for generating presentation IDs
        self._presentation_counter: int = 0

    def _get_session_store(self, session_id: str) -> dict[str, CachedToolResult]:
        """Get or create the session-level cache store."""
        if session_id not in self._session_cache:
            self._session_cache[session_id] = {}
        return self._session_cache[session_id]

    def _get_activation_store(self, activation_id: str) -> dict[str, CachedToolResult]:
        """Get or create the activation-level cache store."""
        if activation_id not in self._activation_cache:
            self._activation_cache[activation_id] = {}
        return self._activation_cache[activation_id]

    def lookup(
        self,
        *,
        session_id: str,
        scope: CacheScope,
        tool_name: str,
        tool_args: dict[str, Any],
        activation_id: str | None = None,
        model_family: str | None = None,
    ) -> CachedToolResult | None:
        """Look up a cached tool result.

        Args:
            session_id: The session ID
            scope: Which cache scope to check
            tool_name: The tool being called
            tool_args: Arguments passed to the tool
            activation_id: Required for ACTIVATION scope (e.g., "sess_1:turn_3")
            model_family: Optional model identifier for model-sensitive caching

        Returns:
            Cached result if found, None otherwise
        """
        cache_key = _make_cache_key(tool_name, tool_args, model_family)

        if scope == CacheScope.SESSION:
            store = self._get_session_store(session_id)
            return store.get(cache_key)
        elif scope == CacheScope.ACTIVATION:
            if not activation_id:
                return None
            store = self._get_activation_store(activation_id)
            return store.get(cache_key)

        return None

    def record(
        self,
        *,
        session_id: str,
        scope: CacheScope,
        tool_name: str,
        tool_args: dict[str, Any],
        result: CachedToolResult,
        activation_id: str | None = None,
        model_family: str | None = None,
    ) -> None:
        """Record a tool result in the cache.

        Args:
            session_id: The session ID
            scope: Which cache scope to store in
            tool_name: The tool that was called
            tool_args: Arguments passed to the tool
            result: The result to cache
            activation_id: Required for ACTIVATION scope
            model_family: Optional model identifier
        """
        cache_key = _make_cache_key(tool_name, tool_args, model_family)

        # Generate presentation ID if not set
        if not result.presentation_id:
            self._presentation_counter += 1
            args_summary = _hash_args(tool_args)[:8]
            result.presentation_id = f"{tool_name}:{args_summary}#{self._presentation_counter}"

        if scope == CacheScope.SESSION:
            store = self._get_session_store(session_id)
            store[cache_key] = result
        elif scope == CacheScope.ACTIVATION:
            if activation_id:
                store = self._get_activation_store(activation_id)
                store[cache_key] = result

    def invalidate(
        self,
        *,
        session_id: str,
        tool_name: str | None = None,
    ) -> None:
        """Invalidate cached entries.

        Args:
            session_id: The session to invalidate entries for
            tool_name: If provided, only invalidate entries for this tool.
                       If None, invalidate all entries for the session.
        """
        if session_id in self._session_cache:
            if tool_name:
                # Remove entries matching tool_name
                store = self._session_cache[session_id]
                keys_to_remove = [k for k in store if k.startswith(f"{tool_name}:")]
                for key in keys_to_remove:
                    del store[key]
            else:
                # Clear entire session cache
                del self._session_cache[session_id]

        # Also clear activation caches for this session
        activation_keys_to_remove = [
            k for k in self._activation_cache if k.startswith(f"{session_id}:")
        ]
        for key in activation_keys_to_remove:
            if tool_name:
                store = self._activation_cache[key]
                entry_keys_to_remove = [k for k in store if k.startswith(f"{tool_name}:")]
                for entry_key in entry_keys_to_remove:
                    del store[entry_key]
            else:
                del self._activation_cache[key]

    def clear_activation(self, activation_id: str) -> None:
        """Clear the activation cache for a specific activation.

        Call this at the end of each activate() to clean up.

        Args:
            activation_id: The activation to clear (e.g., "sess_1:turn_3")
        """
        if activation_id in self._activation_cache:
            del self._activation_cache[activation_id]

    def get_policy(self, tool_name: str) -> ToolCachingPolicy:
        """Get the caching policy for a tool.

        Args:
            tool_name: The tool name

        Returns:
            The caching policy for this tool
        """
        return DEFAULT_TOOL_POLICIES.get(tool_name, DEFAULT_POLICY)


def render_cached_hit_message(
    tool_name: str,
    cached_result: CachedToolResult,
    policy: ToolCachingPolicy,
) -> str:
    """Render the message content for a cache hit.

    Args:
        tool_name: The tool that was called
        cached_result: The cached result
        policy: The caching policy for the tool

    Returns:
        Content string to use in the tool result message
    """
    if policy.presentation_on_hit == PresentationPolicy.FULL_CONTENT:
        # Return the full original content
        if isinstance(cached_result.content, dict):
            return json.dumps(cached_result.content)
        return str(cached_result.content)

    elif policy.presentation_on_hit == PresentationPolicy.REFERENCE_ONLY:
        # Return a short reference
        return json.dumps(
            {
                "_cached": True,
                "_tool": tool_name,
                "_ref": cached_result.presentation_id,
                "_note": f"Identical to previous {tool_name} call; see earlier in conversation.",
            }
        )

    elif policy.presentation_on_hit == PresentationPolicy.SUMMARY:
        # Return a summary (for now, same as reference)
        # TODO: Implement actual summarization
        return json.dumps(
            {
                "_cached": True,
                "_tool": tool_name,
                "_ref": cached_result.presentation_id,
                "_note": f"Summarized from previous {tool_name} call.",
            }
        )

    return json.dumps({"_cached": True, "_tool": tool_name})
