"""HotStore - Ephemeral in-memory workspace for active creative work.

The HotStore is the mutable workspace where roles collaborate on drafts.
Content here is not player-safe and may contain spoilers.

Characteristics:
- Lifetime: Exists only during workflow execution
- Persistence: None by default; checkpointing optional for resume
- Contents: Artifacts in draft/proposed/in_progress states
- Spoilers: Allowed (not player-safe)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HotStore(BaseModel):
    """Ephemeral in-memory workspace for active creative work.

    Structure from stores.md:
    - artifacts: dict[str, Artifact] - All working artifacts by ID
    - hooks: list[HookCard] - Active hook cards
    - current_brief: Brief | None - Active work order
    - scratch: dict[str, Any] - Role working memory

    Examples
    --------
    Create and use a hot store::

        hot = HotStore()
        hot.put("scene_001", {"title": "Opening", "content": "..."})
        scene = hot.get("scene_001")

    Checkpoint and resume::

        hot.checkpoint(Path("project/.qf/checkpoint.json"))
        # ... crash and restart ...
        hot = HotStore.from_checkpoint(Path("project/.qf/checkpoint.json"))
    """

    artifacts: dict[str, Any] = Field(default_factory=dict)
    hooks: list[Any] = Field(default_factory=list)
    current_brief: Any | None = None
    scratch: dict[str, Any] = Field(default_factory=dict)

    # Internal tracking
    _dirty: bool = False

    model_config = {"extra": "allow"}

    # =========================================================================
    # Artifact Operations
    # =========================================================================

    def get(self, key: str) -> Any | None:
        """Get an artifact by key."""
        return self.artifacts.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store an artifact."""
        self.artifacts[key] = value
        self._dirty = True

    def delete(self, key: str) -> bool:
        """Delete an artifact. Returns True if existed."""
        if key in self.artifacts:
            del self.artifacts[key]
            self._dirty = True
            return True
        return False

    def keys(self) -> list[str]:
        """List all artifact keys."""
        return list(self.artifacts.keys())

    def has(self, key: str) -> bool:
        """Check if artifact exists."""
        return key in self.artifacts

    def clear(self) -> None:
        """Clear all artifacts."""
        self.artifacts.clear()
        self._dirty = True

    # =========================================================================
    # Hook Operations
    # =========================================================================

    def add_hook(self, hook: Any) -> None:
        """Add a hook card."""
        self.hooks.append(hook)
        self._dirty = True

    def get_hooks(self) -> list[Any]:
        """Get all hooks."""
        return list(self.hooks)

    def clear_hooks(self) -> None:
        """Clear all hooks."""
        self.hooks.clear()
        self._dirty = True

    # =========================================================================
    # Brief Operations
    # =========================================================================

    def set_brief(self, brief: Any) -> None:
        """Set the current brief."""
        self.current_brief = brief
        self._dirty = True

    def get_brief(self) -> Any | None:
        """Get the current brief."""
        return self.current_brief

    def clear_brief(self) -> None:
        """Clear the current brief."""
        self.current_brief = None
        self._dirty = True

    # =========================================================================
    # Scratch Operations (Role Working Memory)
    # =========================================================================

    def scratch_get(self, key: str) -> Any | None:
        """Get from scratch space."""
        return self.scratch.get(key)

    def scratch_put(self, key: str, value: Any) -> None:
        """Put to scratch space."""
        self.scratch[key] = value
        self._dirty = True

    def scratch_clear(self) -> None:
        """Clear scratch space."""
        self.scratch.clear()
        self._dirty = True

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def checkpoint(self, path: Path) -> None:
        """Save hot store state to a checkpoint file.

        Parameters
        ----------
        path : Path
            Path to checkpoint file (e.g., project/.qf/checkpoint.json)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize artifacts (handle Pydantic models)
        serialized_artifacts = {}
        for key, value in self.artifacts.items():
            if hasattr(value, "model_dump"):
                serialized_artifacts[key] = value.model_dump()
            else:
                serialized_artifacts[key] = value

        # Serialize hooks
        serialized_hooks = []
        for hook in self.hooks:
            if hasattr(hook, "model_dump"):
                serialized_hooks.append(hook.model_dump())
            else:
                serialized_hooks.append(hook)

        # Serialize brief
        serialized_brief = None
        if self.current_brief is not None:
            if hasattr(self.current_brief, "model_dump"):
                serialized_brief = self.current_brief.model_dump()
            else:
                serialized_brief = self.current_brief

        checkpoint_data = {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "artifacts": serialized_artifacts,
            "hooks": serialized_hooks,
            "current_brief": serialized_brief,
            "scratch": self.scratch,
        }

        with open(path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        self._dirty = False
        logger.info(f"Hot store checkpointed to {path}")

    @classmethod
    def from_checkpoint(cls, path: Path) -> HotStore:
        """Load hot store from a checkpoint file.

        Parameters
        ----------
        path : Path
            Path to checkpoint file.

        Returns
        -------
        HotStore
            Restored hot store.

        Raises
        ------
        FileNotFoundError
            If checkpoint file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path) as f:
            data = json.load(f)

        hot = cls(
            artifacts=data.get("artifacts", {}),
            hooks=data.get("hooks", []),
            current_brief=data.get("current_brief"),
            scratch=data.get("scratch", {}),
        )
        hot._dirty = False
        logger.info(f"Hot store restored from {path}")
        return hot

    @property
    def is_dirty(self) -> bool:
        """Check if hot store has unsaved changes."""
        return self._dirty

    # =========================================================================
    # Dict-like Access (for backwards compatibility with state["hot_store"])
    # =========================================================================

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to artifacts."""
        return self.artifacts[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment to artifacts."""
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        """Dict-like deletion from artifacts."""
        self.delete(key)

    def __contains__(self, key: str) -> bool:
        """Dict-like membership test."""
        return key in self.artifacts

    def __len__(self) -> int:
        """Number of artifacts."""
        return len(self.artifacts)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over artifact keys (overrides Pydantic's field iteration)."""
        return iter(self.artifacts)

    def items(self) -> Any:
        """Dict-like items()."""
        return self.artifacts.items()

    def values(self) -> Any:
        """Dict-like values()."""
        return self.artifacts.values()

    def get_dict(self, key: str, default: Any = None) -> Any:
        """Get with default (like dict.get)."""
        return self.artifacts.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value if key doesn't exist."""
        if key not in self.artifacts:
            self.artifacts[key] = default
            self._dirty = True
        return self.artifacts[key]
